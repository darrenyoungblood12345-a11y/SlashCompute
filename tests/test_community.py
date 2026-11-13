"""Accounts, 1:1 credits, grants, and the engine hooks that keep them honest."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from slashcompute.common import protocol as P
from slashcompute.common.config import EngineConfig
from slashcompute.community.auth import AuthError, google_client_id
from slashcompute.community.credits import POT_ID, CreditError
from slashcompute.community.grants import GrantError
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.partitioner import StagePlan
from slashcompute.coordinator.registry import Assignment
from slashcompute.coordinator.scheduler import EpochState
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture(autouse=True)
def fast_hash(monkeypatch):
    monkeypatch.setattr("slashcompute.community.auth.ITERATIONS", 1)


@pytest.fixture
def core(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    c = Coordinator(cfg)
    c._tiny = (tiny_model, tiny_dataset)
    return c


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    app = create_app(cfg)
    # A caller on the coordinator machine: only loopback may submit by path.
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _device(mem=8 << 30):
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=1.0, mem_bandwidth_gbps=100.0,
    )


def _usage(flops=1e9):
    return P.UsageSample(flops=flops, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def _account(auth, email="ada@lan.test", password="password1", name="Ada"):
    user = auth.register(email, password, name)
    user, token = auth.login(email, password)
    return user, token


# --------------------------------------------------------------------------- auth


def test_first_user_is_admin_second_is_not(core):
    a, _ = _account(core.auth, "a@lan.test", name="A")
    b, _ = _account(core.auth, "b@lan.test", name="B")
    assert a.admin is True
    assert b.admin is False


def test_admin_email_env(core, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_ADMIN_EMAIL", "ops@lan.test")
    _account(core.auth, "first@lan.test")
    ops, _ = _account(core.auth, "ops@lan.test")
    assert ops.admin is True


def test_register_rejects_bad_email_short_password_duplicate(core):
    with pytest.raises(AuthError, match="email"):
        core.auth.register("not-an-email", "password1", "x")
    with pytest.raises(AuthError, match="8"):
        core.auth.register("ok@lan.test", "short", "x")
    core.auth.register("ok@lan.test", "password1", "x")
    with pytest.raises(AuthError, match="already"):
        core.auth.register("OK@lan.test", "password1", "x")


def test_login_wrong_password_and_ban(core):
    user, token = _account(core.auth)
    with pytest.raises(AuthError) as e:
        core.auth.login("ada@lan.test", "wrong-password")
    assert e.value.status == 401
    core.auth.set_banned(user, True)
    with pytest.raises(AuthError) as e:
        core.auth.login("ada@lan.test", "password1")
    assert e.value.status == 403
    assert core.auth.user_from_token(token) is None
    banned = core.auth.session_user(token)
    assert banned is not None and banned.banned is True


def test_terms_and_grant_split(core):
    user, _ = _account(core.auth)
    assert user.accepted_terms_at is None
    user = core.auth.accept_terms(user)
    assert user.accepted_terms_at is not None
    user = core.auth.update_profile(user, grant_split=25)
    assert user.grant_split == 25
    with pytest.raises(AuthError):
        core.auth.update_profile(user, grant_split=101)
    with pytest.raises(AuthError):
        core.auth.update_profile(user, name="  ")


def test_google_disabled_without_client_id(core):
    assert google_client_id() == ""
    with pytest.raises(AuthError) as e:
        core.auth.login_google("id-token")
    assert e.value.status == 501


def test_google_login_creates_user(core, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    core.auth.verify_google = lambda tok, cid: {
        "email": "g@lan.test", "name": "Gia", "sub": "sub-1",
    }
    user, token = core.auth.login_google("id-token")
    assert user.email == "g@lan.test" and user.google_sub == "sub-1"
    assert core.auth.user_from_token(token) is not None
    with pytest.raises(AuthError):
        core.auth.login("g@lan.test", "password1")
    again, _ = core.auth.login_google("id-token")
    assert again.id == user.id


def test_logout_drops_session(core):
    _, token = _account(core.auth)
    assert core.auth.user_from_token(token) is not None
    core.auth.logout(token)
    assert core.auth.user_from_token(token) is None


# --------------------------------------------------------------------------- credits


def test_contribute_split_and_one_to_one_balance(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0, 25, node_id="n1")
    assert core.credits.lifetime_earned(user.id) == 100.0
    assert core.credits.balance(user.id) == 75.0
    assert core.credits.balance(POT_ID) == 25.0
    core.credits.contribute(user.id, 100.0, 0)
    assert core.credits.balance(user.id) == 175.0
    assert core.credits.lifetime_earned(user.id) == 200.0


def test_reserve_consume_settle(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 1000.0, 0)
    with pytest.raises(CreditError):
        core.credits.reserve_job(user.id, "job-x", 5000.0)
    core.credits.reserve_job(user.id, "job-1", 400.0)
    assert core.credits.balance(user.id) == 600.0
    assert not core.credits.job_exhausted("job-1")
    assert core.credits.consume_job("job-1", 150.0) == 150.0
    assert core.credits.lifetime_spent(user.id) == 150.0
    assert not core.credits.job_exhausted("job-1")
    # A charge is capped at what is left of the reserve.
    assert core.credits.consume_job("job-1", 300.0) == 250.0
    assert core.credits.job_exhausted("job-1")
    assert core.credits.consume_job("job-1", 1.0) == 0.0
    assert core.credits.consume_job("no-such-job", 1.0) == 0.0
    assert not core.credits.job_exhausted("no-such-job")
    assert core.credits.lifetime_spent(user.id) == 400.0
    core.credits.settle_job("job-1")
    assert core.credits.balance(user.id) == 600.0


def test_settle_releases_leftover(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0, 0)
    core.credits.reserve_job(user.id, "job-2", 80.0)
    core.credits.consume_job("job-2", 30.0)
    core.credits.settle_job("job-2")
    assert core.credits.balance(user.id) == 70.0
    assert core.credits.lifetime_spent(user.id) == 30.0


def test_donate_and_pot_allocate(core):
    donor, _ = _account(core.auth, "d@lan.test", name="Donor")
    recip, _ = _account(core.auth, "r@lan.test", name="Recip")
    core.credits.contribute(donor.id, 100.0, 50)
    assert core.credits.balance(donor.id) == 50.0
    assert core.credits.balance(POT_ID) == 50.0
    core.credits.donate(donor.id, recip.id, "g1", 20.0)
    assert core.credits.balance(donor.id) == 30.0
    assert core.credits.balance(recip.id) == 20.0
    core.credits.allocate_pot(recip.id, "g1", 50.0)
    assert core.credits.balance(POT_ID) == 0.0
    assert core.credits.balance(recip.id) == 70.0
    with pytest.raises(CreditError):
        core.credits.allocate_pot(recip.id, "g1", 1.0)


def test_leaderboard_ranks_generated(core):
    a, _ = _account(core.auth, "a@lan.test", name="Ann")
    b, _ = _account(core.auth, "b@lan.test", name="Bea")
    core.credits.contribute(a.id, 10.0, 0)
    core.credits.contribute(b.id, 50.0, 0)
    board = core.credits.leaderboard()
    assert [row["name"] for row in board] == ["Bea", "Ann"]
    assert board[0]["lifetime_earned"] == 50.0


# --------------------------------------------------------------------------- grants


def test_grant_lifecycle(core):
    admin, _ = _account(core.auth, "admin@lan.test", name="Admin")
    member, _ = _account(core.auth, "m@lan.test", name="Member")
    donor, _ = _account(core.auth, "d@lan.test", name="Donor")
    core.credits.contribute(donor.id, 200.0, 0)
    with pytest.raises(GrantError):
        core.grants.create(member, "hi", "too short", 1e3)
    g = core.grants.create(member, "Need compute", "A short paragraph about the need.", 100.0)
    assert g.status == "pending"
    assert core.grants.list(viewer=member)[0].id == g.id
    assert core.grants.list(viewer=None) == []
    with pytest.raises(GrantError, match="Admin"):
        core.grants.review(member, g.id, True)
    with pytest.raises(GrantError, match="approved"):
        core.grants.donate(donor, g.id, 10.0)
    core.grants.review(admin, g.id, True)
    with pytest.raises(GrantError, match="someone else"):
        core.grants.donate(member, g.id, 10.0)
    core.grants.donate(donor, g.id, 40.0)
    got = core.grants.get(g.id)
    assert got.status == "approved" and got.received_flops == 40.0
    assert core.credits.balance(donor.id) == 160.0
    assert core.credits.balance(member.id) == 40.0
    pub = core.grants.list(viewer=None)
    assert len(pub) == 1 and pub[0].id == g.id


def test_pot_allocate_admin_only(core):
    admin, _ = _account(core.auth, "admin@lan.test")
    member, _ = _account(core.auth, "m@lan.test")
    core.credits.contribute(member.id, 100.0, 100)
    g = core.grants.create(member, "Need compute", "A short paragraph about the need.", 50.0)
    core.grants.review(admin, g.id, True)
    with pytest.raises(GrantError, match="admin"):
        core.grants.donate(member, g.id, 10.0, from_pot=True)
    core.grants.donate(admin, g.id, 10.0, from_pot=True)
    assert core.grants.get(g.id).received_flops == 10.0
    assert core.credits.balance(member.id) == 10.0


def _race(n, fn):
    """Run fn from n threads released together; return how many calls were not
    refused. Anything other than a refusal is a real failure and re-raised."""
    gate = threading.Barrier(n)
    ok, crashed = [], []

    def run():
        gate.wait()
        try:
            fn()
            ok.append(1)
        except (CreditError, GrantError):
            pass
        except BaseException as e:  # noqa: BLE001 - surfaced on the main thread below
            crashed.append(e)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if crashed:
        raise crashed[0]
    return len(ok)


def _approved_grant(core):
    admin, _ = _account(core.auth, "admin@lan.test")
    member, _ = _account(core.auth, "m@lan.test")
    donor, _ = _account(core.auth, "d@lan.test")
    g = core.grants.create(member, "Need compute", "A short paragraph about the need.", 1e6)
    core.grants.review(admin, g.id, True)
    return admin, member, donor, g


def test_concurrent_spends_cannot_overdraw(core):
    """Sixteen simultaneous spends of the whole balance: exactly one may win.
    Repeated because a single round can pass by luck."""
    admin, member, donor, g = _approved_grant(core)
    rounds = 25
    for i in range(rounds):
        core.credits.contribute(donor.id, 10.0, 0)
        assert _race(16, lambda: core.grants.donate(donor, g.id, 10.0)) == 1, f"round {i}"
        assert core.credits.balance(donor.id) == 0.0
    assert core.grants.get(g.id).received_flops == 10.0 * rounds
    assert core.credits.balance(member.id) == 10.0 * rounds

    for i in range(rounds):
        core.credits.contribute(donor.id, 10.0, 100)  # all of it to the pot
        assert _race(16, lambda: core.grants.donate(admin, g.id, 10.0, from_pot=True)) == 1, f"round {i}"
        assert core.credits.balance(POT_ID) == 0.0
    assert core.grants.get(g.id).received_flops == 20.0 * rounds
    assert core.credits.balance(member.id) == 20.0 * rounds

    for i in range(rounds):
        jobs = iter(range(16))
        have = core.credits.balance(member.id)
        assert _race(16, lambda: core.credits.reserve_job(member.id, f"job-{i}-{next(jobs)}", have)) == 1
        assert core.credits.balance(member.id) == 0.0
        core.credits.contribute(member.id, 10.0, 0)


def test_concurrent_donations_keep_the_grant_total(core):
    """Every donation is affordable, so all must land on the grant's total."""
    _, member, donor, g = _approved_grant(core)
    rounds = 25
    core.credits.contribute(donor.id, 16.0 * rounds, 0)
    for i in range(rounds):
        assert _race(16, lambda: core.grants.donate(donor, g.id, 1.0)) == 16
        assert core.grants.get(g.id).received_flops == 16.0 * (i + 1), f"round {i}"
    assert core.credits.balance(donor.id) == 0.0
    assert core.credits.balance(member.id) == 16.0 * rounds


def test_concurrent_consume_and_settle_keep_job_books_straight(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 1000.0, 0)
    for i in range(10):
        job = f"job-{i}"
        core.credits.reserve_job(user.id, job, 80.0)
        # Every charge lands on the account: no lost updates to spent_flops.
        assert _race(16, lambda: core.credits.consume_job(job, 1.0)) == 16
        assert core.credits.job_account(job).spent_flops == 16.0
        assert core.credits.lifetime_spent(user.id) == 16.0 * (i + 1)
        # The leftover is released exactly once however many settles race.
        _race(16, lambda: core.credits.settle_job(job))
        assert core.credits.balance(user.id) == 1000.0 - 16.0 * (i + 1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -5.0])
def test_non_finite_or_non_positive_spends_are_refused(core, bad):
    admin, member, donor, g = _approved_grant(core)
    core.credits.contribute(donor.id, 100.0, 50)
    for spend in (lambda: core.grants.donate(donor, g.id, bad),
                  lambda: core.grants.donate(admin, g.id, bad, from_pot=True),
                  lambda: core.credits.reserve_job(donor.id, "job-bad", bad)):
        with pytest.raises(CreditError):
            spend()
    assert core.credits.balance(donor.id) == 50.0
    assert core.credits.balance(POT_ID) == 50.0
    assert core.credits.balance(member.id) == 0.0
    assert core.grants.get(g.id).received_flops == 0.0
    assert core.credits.job_account("job-bad") is None


def test_flag_and_comment(core):
    admin, _ = _account(core.auth, "admin@lan.test")
    member, _ = _account(core.auth, "m@lan.test", name="Member")
    g = core.grants.create(member, "Need compute", "A short paragraph about the need.", 10.0)
    core.grants.comment(member, g.id, "please")
    assert core.grants.comments(g.id)[0].body == "please"
    core.grants.review(admin, g.id, False)
    with pytest.raises(GrantError):
        core.grants.comment(member, g.id, "after decline")
    core.grants.flag_user(admin, member, "spam")
    assert core.auth.get(member.id).flagged is True


# --------------------------------------------------------------------------- http


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


def test_http_register_login_cookie_and_me(env):
    client, core, *_ = env
    r = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["admin"] is True
    assert body["user"]["accepted_terms"] is False
    assert r.cookies.get("slashcompute_session")
    me = client.get("/auth/me")
    assert me.json()["user"]["email"] == "ada@lan.test"
    client.post("/auth/logout")
    assert client.get("/auth/me").json()["user"] is None
    login = client.post("/auth/login", json={"email": "ada@lan.test", "password": "password1"})
    token = login.json()["token"]
    me = client.get("/auth/me", headers=_hdr(token))
    assert me.json()["credits"]["balance"] == 0.0
    patched = client.patch("/auth/me", json={"grant_split": 33}, headers=_hdr(token))
    assert patched.json()["user"]["grant_split"] == 33
    assert client.get("/auth/providers").json()["google"] is False
    assert client.post("/auth/google", json={"id_token": "x"}).status_code == 501


def test_http_google_when_configured(env, monkeypatch):
    client, core, *_ = env
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    core.auth.verify_google = lambda tok, cid: {
        "email": "g@lan.test", "name": "Gia", "sub": "sub-9",
    }
    r = client.post("/auth/google", json={"id_token": "jwt"})
    assert r.status_code == 200, r.text
    assert r.json()["user"]["email"] == "g@lan.test"
    assert client.get("/auth/providers").json()["google"] is True


def test_http_terms_required_to_take_and_failed_reserve_drops_job(env):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 1e12
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 403
    assert "terms" in r.json()["detail"].lower()
    assert core.jobs == {}
    client.post("/auth/accept-terms", headers=_hdr(token))
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert "Contribute" in r.json()["detail"]
    assert core.jobs == {}


def test_http_authenticated_take_reserves(env):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))
    user = core.auth.user_from_token(token)
    core.credits.contribute(user.id, 1e12, 0)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 2e11
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reserved_flops"] == 2e11
    assert body["user_id"] == user.id
    assert core.credits.balance(user.id) == pytest.approx(8e11)


def test_anonymous_submit_still_works(env):
    """No account is needed on a LAN pool: by path from the coordinator machine,
    by upload from anywhere else."""
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 200, r.text
    assert r.json()["reserved_flops"] is None
    assert r.json()["id"] in core.jobs

    remote = TestClient(client.app, client=("192.168.1.50", 50000))
    r = remote.post("/jobs", json=spec)
    assert r.status_code == 403, r.text
    assert len(core.jobs) == 1
    r = remote.post(
        "/jobs/upload",
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 1, "batch_size": 2,
              "microbatches": 1},
    )
    assert r.status_code == 200, r.text
    assert r.json()["reserved_flops"] is None and r.json()["id"] in core.jobs


def test_dataset_download_requires_the_owner_or_an_admin(env):
    client, core, tiny_model, tiny_dataset = env
    remote = TestClient(client.app, client=("192.168.1.50", 50000))
    tokens = {}
    for name in ("admin", "owner", "other"):   # the first account is the admin
        tokens[name] = client.post("/auth/register", json={
            "email": f"{name}@lan.test", "password": "password1", "name": name}).json()["token"]
        user = core.auth.session_user(tokens[name])
        core.auth.accept_terms(user)
        core.credits.contribute(user.id, 1e12, 0)
    upload = dict(
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 1, "batch_size": 2,
              "microbatches": 1, "max_flops": 1e9},
    )
    owned = remote.post("/jobs/upload", headers=_hdr(tokens["owner"]), **upload).json()["id"]
    upload["data"].pop("max_flops")
    anonymous = remote.post("/jobs/upload", **upload).json()["id"]

    def status(job_id, who=None, via=remote):
        return via.get(f"/jobs/{job_id}/dataset", headers=_hdr(tokens[who]) if who else {}).status_code

    assert status(owned) == 401
    assert status(owned, "other") == 403
    assert status(owned, "owner") == 200
    assert status(owned, "admin") == 200
    # Nobody owns an anonymous job: remotely only an admin (or its nodes) may read it.
    assert status(anonymous) == 401
    assert status(anonymous, "owner") == 403
    assert status(anonymous, "admin") == 200
    assert status(owned, via=client) == status(anonymous, via=client) == 200  # the operator
    core.auth.set_banned(core.auth.session_user(tokens["owner"]), True)
    assert status(owned, "owner") == 403


def test_http_grant_moderation_and_leaderboard(env):
    client, core, *_ = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    member_tok = client.post("/auth/register", json={
        "email": "m@lan.test", "password": "password1", "name": "Member",
    }).json()["token"]
    donor_tok = client.post("/auth/register", json={
        "email": "d@lan.test", "password": "password1", "name": "Donor",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    client.post("/auth/accept-terms", headers=_hdr(donor_tok))
    donor = core.auth.user_from_token(donor_tok)
    core.credits.contribute(donor.id, 80.0, 0)
    g = client.post("/grants", json={
        "title": "Need compute", "body": "A short paragraph about the need.",
        "goal_flops": 50,
    }, headers=_hdr(member_tok)).json()
    assert g["status"] == "pending"
    client.cookies.clear()
    assert client.get("/grants").json() == []
    hidden = client.get(f"/grants/{g['id']}")
    assert hidden.status_code == 404
    mine = client.get("/grants", headers=_hdr(member_tok)).json()
    assert mine[0]["id"] == g["id"]
    client.post(f"/admin/grants/{g['id']}/review", json={"approve": True},
                headers=_hdr(admin_tok))
    client.post(f"/grants/{g['id']}/donate", json={"flops": 20},
                headers=_hdr(donor_tok))
    pub = client.get("/grants").json()
    assert pub[0]["received_flops"] == 20
    board = client.get("/community/leaderboard").json()
    assert board[0]["name"] == "Donor"
    users = client.get("/admin/users", headers=_hdr(member_tok))
    assert users.status_code == 403


def test_http_ban_blocks_take(env):
    client, core, tiny_model, tiny_dataset = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    member_tok = client.post("/auth/register", json={
        "email": "m@lan.test", "password": "password1", "name": "Member",
    }).json()["token"]
    member = core.auth.user_from_token(member_tok)
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    core.credits.contribute(member.id, 1e12, 0)
    client.post(f"/admin/users/{member.id}/ban", json={"banned": True},
                headers=_hdr(admin_tok))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 1e6
    r = client.post("/jobs", json=spec, headers=_hdr(member_tok))
    assert r.status_code == 403, r.text
    assert core.jobs == {}


# --------------------------------------------------------------------------- engine hooks


def test_register_binds_only_after_terms(core):
    user, token = _account(core.auth)
    msg = P.Register(node_id="n1", name="mac", device=_device(),
                     data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                     session_token=token)

    async def send(_):
        return None

    asyncio.run(core.on_register(msg, send))
    assert core.registry.get("n1").user_id is None
    core.auth.accept_terms(user)
    asyncio.run(core.on_register(msg, send))
    assert core.registry.get("n1").user_id == user.id
    assert core.credits.owner_of("n1") == user.id


def test_banned_session_closes_agent(env):
    client, core, *_ = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    user = core.auth.session_user(token)
    core.auth.accept_terms(user)
    core.auth.set_banned(user, True)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="banned", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                session_token=token,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def _live_job(core, budget=None, *, split=0, nodes=("n1",)):
    """A running one-epoch job owned by a fresh account, with one registered
    node (owned by the same account) planned for each stage. `budget=None`
    leaves the job anonymous (no reserve)."""
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    if split:
        core.auth.update_profile(user, grant_split=split)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset, steps=20), allow_local_path=True)
    if budget is not None:
        core.credits.reserve_job(user.id, job.id, budget)

    async def send(_):
        return None

    plans = []
    for idx, node_id in enumerate(nodes):
        asyncio.run(core.on_register(P.Register(
            node_id=node_id, name="mac", device=_device(), data_host="127.0.0.1",
            data_port=9700 + idx, gpu_percent=50, session_token=token,
        ), send))
        core.registry.get(node_id).assignment = Assignment(job.id, 1, idx)
        plans.append(StagePlan(stage_idx=idx, node_id=node_id, layer_start=idx, layer_end=idx + 1,
                               est_bytes=0))
    job.current = EpochState(epoch=1, plans=plans)
    job.row.status = "running"
    return user, job


def _step(core, job, step=1, flops=1e9, *, node="n1", stage=0, epoch=1, usage=None):
    """Deliver step metrics the way the WebSocket handler does."""
    asyncio.run(core.handle(node, P.StepMetrics(
        job_id=job.id, epoch=epoch, stage_idx=stage, step=step, loss=1.0,
        in_digest="in", out_digest="out", usage=usage or _usage(flops),
    )))


def _books(core, user, job):
    acct = core.credits.job_account(job.id)
    return (core.credits.lifetime_earned(user.id) - 1e12, core.credits.lifetime_spent(user.id),
            acct.spent_flops if acct else None, len(core.ledger.job_records(job.id)))


def test_steps_earn_and_spend_one_to_one(core):
    user, job = _live_job(core, 5e9)
    _step(core, job, flops=1e9)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1e9)
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1e9)
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 5e9 + 1e9)


def test_anonymous_job_does_not_mint_credits(core):
    user, job = _live_job(core, None)
    _step(core, job, flops=1e9)
    assert core.credits.lifetime_earned(user.id) == 1e12
    assert core.credits.job_account(job.id) is None
    assert len(core.ledger.job_records(job.id)) == 1  # usage is still recorded


def test_budget_exhaust_cancels_job(core):
    user, job = _live_job(core, 1e9)
    _step(core, job, flops=1e9)
    assert job.row.status == "cancelled"
    assert job.row.error == "FLOP budget spent"
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 1e9 + 1e9)


def test_grant_split_on_step(core):
    user, job = _live_job(core, 1e10, split=40)
    _step(core, job, flops=100.0)
    assert core.credits.balance(POT_ID) == pytest.approx(40.0)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 100.0)


def test_step_credit_never_exceeds_what_the_job_was_charged(core):
    """The charge is capped at the reserve, so the credit must be too."""
    user, job = _live_job(core, 1e9, split=40)
    _step(core, job, flops=5e9)
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1e9)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1e9)
    assert core.credits.balance(POT_ID) == pytest.approx(0.4e9)
    assert job.row.status == "cancelled" and job.row.error == "FLOP budget spent"
    # Nothing more is minted once the reserve is gone.
    _step(core, job, step=2, flops=5e9)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1e9)


def test_duplicate_step_metrics_are_billed_once(core):
    user, job = _live_job(core, 1e12, nodes=("n1", "n2"))
    _step(core, job, step=1)
    _step(core, job, step=1, node="n2", stage=1)  # each stage reports the step once
    assert _books(core, user, job) == (2e9, 2e9, 2e9, 2)
    for _ in range(5):
        _step(core, job, step=1)
    assert _books(core, user, job) == (2e9, 2e9, 2e9, 2)
    _step(core, job, step=2)
    _step(core, job, step=1)  # replaying an earlier step after moving on
    _step(core, job, step=2)
    assert _books(core, user, job) == (3e9, 3e9, 3e9, 3)


@pytest.mark.parametrize("step", [0, -1, 21])
def test_steps_outside_the_job_are_not_billed(core, step):
    user, job = _live_job(core, 1e12)
    _step(core, job, step=step)
    assert _books(core, user, job) == (0, 0, 0, 0)


@pytest.mark.parametrize("state", ["closed", "recovering", "cancelled", "completed", "failed", "stale-epoch"])
def test_metrics_for_a_finished_epoch_are_not_billed(core, state):
    user, job = _live_job(core, 1e12)
    _step(core, job, step=1)
    if state == "closed":
        job.current.closed = True
    elif state == "stale-epoch":
        job.current = EpochState(epoch=2, plans=job.current.plans)
    else:
        job.current.closed = True
        job.row.status = state
    _step(core, job, step=2)
    assert _books(core, user, job) == (1e9, 1e9, 1e9, 1)
    if state not in ("closed", "stale-epoch"):
        # Belt and braces: the job's status alone is enough to refuse.
        job.current.closed = False
        _step(core, job, step=3)
        assert _books(core, user, job) == (1e9, 1e9, 1e9, 1)


def test_metrics_from_another_stage_or_node_are_not_billed(core):
    user, job = _live_job(core, 1e12, nodes=("n1", "n2"))
    _step(core, job, node="n1", stage=1)   # n1 holds stage 0, not 1
    _step(core, job, node="ghost", stage=0)
    assert _books(core, user, job) == (0, 0, 0, 0)


@pytest.mark.parametrize("field,bad", [
    ("flops", float("nan")), ("flops", float("inf")), ("flops", -1e9),
    ("wall_s", float("nan")), ("mem_byte_seconds", float("inf")), ("busy_s", -1.0),
])
def test_non_finite_or_negative_usage_is_not_billed(core, field, bad):
    user, job = _live_job(core, 1e12)
    _step(core, job, usage=_usage().model_copy(update={field: bad}))
    assert _books(core, user, job) == (0, 0, 0, 0)
    assert job.row.status == "running"
    # A bad report does not burn the step: the honest one still counts.
    _step(core, job)
    assert _books(core, user, job) == (1e9, 1e9, 1e9, 1)
