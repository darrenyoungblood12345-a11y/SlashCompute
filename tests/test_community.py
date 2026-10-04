"""Accounts, 1:1 credits, grants, and the engine hooks that keep them honest."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from starlette.websockets import WebSocketDisconnect

from slashcompute.common import protocol as P
from slashcompute.common.config import WELCOME_FLOPS, EngineConfig
from slashcompute.community.auth import AuthError, google_client_id
from slashcompute.community.credits import POT_ID, CreditError
from slashcompute.community.grants import GrantError
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import User
from slashcompute.coordinator.inference_accounting import CoreAccounting
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
    with TestClient(app) as client:
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
    user = core.auth.update_profile(user, grant_split=25, bio="  hello  ")
    assert user.grant_split == 25
    assert user.bio == "hello"
    assert core.auth.public_view(user)["bio"] == "hello"
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
    assert core.credits.summary(user.id)["reserved_in_flight"] == 400.0
    assert core.credits.consume_job("job-1", 150.0) == 150.0
    assert core.credits.lifetime_spent(user.id) == 150.0
    assert core.credits.job_exhausted("job-1") is False
    assert core.credits.summary(user.id)["reserved_in_flight"] == 250.0
    assert core.credits.consume_job("job-1", 300.0) == 250.0
    assert core.credits.lifetime_spent(user.id) == 400.0
    assert core.credits.job_exhausted("job-1") is True
    core.credits.settle_job("job-1")
    assert core.credits.balance(user.id) == 600.0
    assert core.credits.reserved_in_flight(user.id) == 0.0


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
    assert me.json()["credits"]["balance"] == WELCOME_FLOPS
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
    spec["max_flops"] = 2 * WELCOME_FLOPS
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
    assert core.credits.balance(user.id) == pytest.approx(WELCOME_FLOPS + 8e11)


def test_anonymous_submit_still_works(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 200, r.text
    assert r.json()["reserved_flops"] is None
    assert r.json()["id"] in core.jobs


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


# --------------------------------------------------------------------------- welcome credit


def _welcome_rows(client, token):
    items = client.get("/credits/transactions", headers=_hdr(token)).json()["items"]
    return [row for row in items if row["kind"] == "welcome"]


def test_welcome_credit_on_register_is_spendable_once(env):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    me = client.get("/credits/me", headers=_hdr(token)).json()
    assert me["balance"] == WELCOME_FLOPS == 1e15
    assert me["lifetime_earned"] == 0.0
    rows = _welcome_rows(client, token)
    assert len(rows) == 1 and rows[0]["note"] == "Welcome credit"
    for _ in range(2):
        client.post("/auth/login", json={"email": "ada@lan.test", "password": "password1"})
    assert core.credits.balance(core.auth.user_from_token(token).id) == WELCOME_FLOPS

    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = WELCOME_FLOPS
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 200, r.text
    assert core.credits.balance(r.json()["user_id"]) == 0.0


def test_welcome_credit_backfilled_on_login(env):
    client, core, *_ = env
    user = core.auth.register("old@lan.test", "password1", "Old")  # pre-feature account
    assert core.credits.balance(user.id) == 0.0
    token = client.post("/auth/login", json={
        "email": "old@lan.test", "password": "password1",
    }).json()["token"]
    client.post("/auth/login", json={"email": "old@lan.test", "password": "password1"})
    assert core.credits.balance(user.id) == WELCOME_FLOPS
    assert len(_welcome_rows(client, token)) == 1


def test_welcome_credit_backfilled_for_a_stored_session(env):
    # Upgraded while signed in: the app keeps polling /auth/me and never signs in again.
    client, core, *_ = env
    user, token = _account(core.auth, "kept@lan.test")
    assert core.credits.balance(user.id) == 0.0
    for _ in range(3):
        me = client.get("/auth/me", headers=_hdr(token)).json()
    assert me["credits"]["balance"] == WELCOME_FLOPS
    assert len(_welcome_rows(client, token)) == 1

    banned, banned_token = _account(core.auth, "ban@lan.test")
    core.auth.set_banned(banned, True)
    client.get("/auth/me", headers=_hdr(banned_token))
    assert core.credits.balance(banned.id) == 0.0


def test_welcome_credit_on_google_sign_in(env, monkeypatch):
    client, core, *_ = env
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    core.auth.verify_google = lambda tok, cid: {
        "email": "g@lan.test", "name": "Gia", "sub": "sub-9",
    }
    token = client.post("/auth/google", json={"id_token": "jwt"}).json()["token"]
    client.post("/auth/google", json={"id_token": "jwt"})
    user = core.auth.user_from_token(token)
    assert core.credits.balance(user.id) == WELCOME_FLOPS


def test_welcome_credit_granted_once_under_concurrency(core):
    import threading

    user = core.auth.register("ada@lan.test", "password1", "Ada")
    start = threading.Barrier(16)
    granted = []

    def sign_in():
        start.wait()
        granted.append(core.credits.grant_welcome(user.id))

    threads = [threading.Thread(target=sign_in) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(granted) == [0.0] * 15 + [WELCOME_FLOPS]
    assert core.credits.balance(user.id) == WELCOME_FLOPS


def test_welcome_credit_skips_banned_and_disabled(env, tmp_path, monkeypatch):
    client, core, *_ = env
    user = core.auth.register("ban@lan.test", "password1", "Ban")
    core.auth.set_banned(user, True)
    r = client.post("/auth/login", json={"email": "ban@lan.test", "password": "password1"})
    assert r.status_code == 403
    assert core.credits.grant_welcome(user.id) == 0.0
    assert core.credits.balance(user.id) == 0.0

    monkeypatch.setenv("SLASHCOMPUTE_WELCOME_FLOPS", "0")
    cfg = EngineConfig.from_env(home=tmp_path / "off", verify_rate=0.0)
    assert cfg.welcome_flops == 0.0
    with TestClient(create_app(cfg)) as off:
        token = off.post("/auth/register", json={
            "email": "ada@lan.test", "password": "password1", "name": "Ada",
        }).json()["token"]
        assert off.get("/credits/me", headers=_hdr(token)).json()["balance"] == 0.0


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


def test_steps_earn_and_spend_one_to_one(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 5e9)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))

    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1e9)
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1e9)
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 5e9 + 1e9)


def test_anonymous_job_does_not_mint_credits(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    job.current = EpochState(epoch=1, plans=[])

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert core.credits.lifetime_earned(user.id) == 0.0
    assert core.credits.job_account(job.id) is None


def test_budget_exhaust_cancels_job(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1e9)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert job.row.status == "cancelled"
    assert job.row.error == "FLOP budget spent"
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 1e9 + 1e9)


def test_grant_split_on_step(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.auth.update_profile(user, grant_split=40)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1e10)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(100.0),
    )))
    assert core.credits.balance(POT_ID) == pytest.approx(40.0)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 100.0)


def test_last_step_mints_only_charged_flops(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1.5e9)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(2e9),
    )))
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1.5e9)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1.5e9)
    assert job.row.status == "cancelled"
    assert job.row.error == "FLOP budget spent"


def _ledger_total(core):
    """Every FLOP in the book: all balances (pot included) plus reservations still in flight."""
    with core.db.session() as s:
        ids = [u.id for u in s.exec(select(User)).all()] + [POT_ID]
    return sum(core.credits.balance(i) + core.credits.reserved_in_flight(i) for i in ids)


def _host_and_chatter(core):
    host, host_tok = _account(core.auth, "host@lan.test", name="Host")
    chatter, _ = _account(core.auth, "bea@lan.test", name="Bea")
    core.auth.accept_terms(host)
    core.auth.accept_terms(chatter)
    core.credits.contribute(chatter.id, 1e12, 0)
    return host, host_tok, chatter


@pytest.mark.parametrize("banned", [False, True])
def test_inference_share_without_eligible_host_goes_to_pot(core, banned):
    host, host_tok, chatter = _host_and_chatter(core)
    acct = CoreAccounting(core)
    if banned:
        acct.bind_node("n1", host_tok)
        core.auth.set_banned(host, True)
    core.credits.reserve_job(chatter.id, "chat-1", 1e12)
    before = _ledger_total(core)

    acct.record("chat-1", {"n1": 4.64e11}, tokens=10, wall_s=1.0)

    assert core.credits.lifetime_spent(chatter.id) == pytest.approx(4.64e11)
    assert core.credits.lifetime_earned(host.id) == 0.0
    assert core.credits.balance(POT_ID) == pytest.approx(4.64e11)
    assert _ledger_total(core) == pytest.approx(before)


@pytest.mark.parametrize("banned", [False, True])
def test_training_share_without_eligible_host_goes_to_pot(core, banned):
    tiny_model, tiny_dataset = core._tiny
    host, host_tok, chatter = _host_and_chatter(core)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(chatter.id, job.id, 5e9)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=host_tok if banned else None,
    ), send))
    if banned:
        core.auth.set_banned(host, True)
    before = _ledger_total(core)
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert core.credits.lifetime_spent(chatter.id) == pytest.approx(1e9)
    assert core.credits.lifetime_earned(host.id) == 0.0
    assert core.credits.balance(POT_ID) == pytest.approx(1e9)
    assert _ledger_total(core) == pytest.approx(before)


def test_leaderboard_excludes_banned(core):
    a, _ = _account(core.auth, "a@lan.test", name="A")
    b, _ = _account(core.auth, "b@lan.test", name="B")
    core.credits.contribute(a.id, 10.0, 0)
    core.credits.contribute(b.id, 50.0, 0)
    core.auth.set_banned(b, True)
    assert [r["user_id"] for r in core.credits.leaderboard()] == [a.id]


def test_session_without_max_flops_abandons_job(env):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert "max_flops" in r.json()["detail"]
    assert core.jobs == {}
    client.cookies.clear()
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert core.jobs == {}
    login = client.post("/auth/login", json={"email": "ada@lan.test", "password": "password1"})
    assert login.status_code == 200
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert "max_flops" in r.json()["detail"]
    assert core.jobs == {}


def test_http_community_lists_live_and_admin(env):
    client, core, tiny_model, tiny_dataset = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    member_tok = client.post("/auth/register", json={
        "email": "m@lan.test", "password": "password1", "name": "Member",
    }).json()["token"]
    donor_tok = client.post("/auth/register", json={
        "email": "d@lan.test", "password": "password1", "name": "Donor",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(admin_tok))
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    client.post("/auth/accept-terms", headers=_hdr(donor_tok))
    donor = core.auth.user_from_token(donor_tok)
    member = core.auth.user_from_token(member_tok)
    core.credits.contribute(donor.id, 200.0, 50)
    core.credits.bind_node("n-mac", member.id)

    me = client.get("/credits/me", headers=_hdr(donor_tok)).json()
    assert me["balance"] == WELCOME_FLOPS + 100.0
    assert me["reserved_in_flight"] == 0.0
    assert me["pot"] == 100.0

    tx = client.get("/credits/transactions?limit=2", headers=_hdr(donor_tok)).json()
    assert len(tx["items"]) == 2
    assert {row["kind"] for row in tx["items"]} <= {"earn", "generated"}
    assert all("created_at" in row for row in tx["items"])
    page = client.get(
        f"/credits/transactions?limit=1&before_id={tx['items'][0]['id']}",
        headers=_hdr(donor_tok),
    ).json()
    assert len(page["items"]) == 1
    assert page["items"][0]["id"] < tx["items"][0]["id"]

    live = client.get("/credits/live?window_s=60", headers=_hdr(donor_tok)).json()
    assert live["window_s"] == 60.0
    assert live["you"]["flops"] == pytest.approx(200.0)
    assert live["community_flops"] == pytest.approx(200.0)
    assert live["reserved_in_flight"] == 0.0

    nodes = client.get("/auth/me/nodes", headers=_hdr(member_tok)).json()
    assert nodes == [{
        "node_id": "n-mac", "user_id": member.id, "online": False,
        "status": None, "gpu_percent": None,
    }]

    bio = client.patch("/auth/me", json={"bio": "writes grants"}, headers=_hdr(member_tok))
    assert bio.json()["user"]["bio"] == "writes grants"

    core.credits.contribute(member.id, 1e12, 0)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 2e11
    taken = client.post("/jobs", json=spec, headers=_hdr(member_tok))
    assert taken.status_code == 200, taken.text
    job_id = taken.json()["id"]
    mine = client.get("/jobs?mine=1", headers=_hdr(member_tok)).json()
    assert [j["id"] for j in mine] == [job_id]
    others = client.get("/jobs?mine=1", headers=_hdr(donor_tok)).json()
    assert others == []

    g = client.post("/grants", json={
        "title": "Need compute", "body": "A short paragraph about the need.",
        "goal_flops": 50,
    }, headers=_hdr(member_tok)).json()
    client.cookies.clear()
    comment = client.post(f"/grants/{g['id']}/comments", json={"body": "please fund"},
                          headers=_hdr(member_tok))
    assert comment.status_code == 200, comment.text
    assert comment.json()["id"] is not None

    declined = client.post(f"/admin/grants/{g['id']}/review",
                           json={"approve": False, "note": "too vague"},
                           headers=_hdr(admin_tok)).json()
    assert declined["status"] == "declined"
    assert declined["review_note"] == "too vague"
    assert declined["reviewed_by"]
    assert declined["reviewed_at"]

    g2 = client.post("/grants", json={
        "title": "Need more", "body": "Another short paragraph about the need.",
        "goal_flops": 80,
    }, headers=_hdr(member_tok)).json()
    client.post(f"/admin/grants/{g2['id']}/review", json={"approve": True},
                headers=_hdr(admin_tok))
    g3 = client.post("/grants", json={
        "title": "Need most", "body": "A third short paragraph about the need.",
        "goal_flops": 20,
    }, headers=_hdr(member_tok)).json()
    client.post(f"/admin/grants/{g3['id']}/review", json={"approve": True},
                headers=_hdr(admin_tok))
    d1 = client.post(f"/grants/{g2['id']}/donate", json={"flops": 10},
                     headers=_hdr(donor_tok))
    assert d1.status_code == 200, d1.text
    d2 = client.post(f"/grants/{g3['id']}/donate", json={"flops": 20, "from_pot": True},
                     headers=_hdr(admin_tok))
    assert d2.status_code == 200, d2.text
    assert d1.json()["received_flops"] == 10
    assert d2.json()["received_flops"] == 20

    least = client.get("/grants?sort=least").json()
    assert [row["id"] for row in least] == [g2["id"], g3["id"]]
    trending = client.get("/grants?sort=trending").json()
    assert [row["id"] for row in trending] == [g3["id"], g2["id"]]

    client.post(f"/admin/users/{member.id}/flag", json={"reason": "spam"},
                headers=_hdr(admin_tok))
    flags = client.get("/admin/flags", headers=_hdr(admin_tok)).json()
    assert flags[0]["user_id"] == member.id
    assert flags[0]["reason"] == "spam"
    assert client.get("/admin/flags", headers=_hdr(member_tok)).status_code == 403


def test_my_nodes_shows_live_inference_nodes_online(env):
    client, core, _, _ = env
    token = client.post("/auth/register", json={
        "email": "host@lan.test", "password": "password1", "name": "Host",
    }).json()["token"]
    user = core.auth.user_from_token(token)
    inference = client.app.state.inference
    for node_id, beat in [("inf-live", time.time()), ("inf-gone", time.time() - 3600)]:
        inference.conn.execute("INSERT INTO nodes (id, name, last_heartbeat, created_at) VALUES (?,?,?,?)",
                               (node_id, node_id, beat, beat))
        core.credits.bind_node(node_id, user.id)
    nodes = {n["node_id"]: n for n in client.get("/auth/me/nodes", headers=_hdr(token)).json()}
    assert nodes["inf-live"] == {"node_id": "inf-live", "user_id": user.id, "online": True,
                                 "status": "idle", "gpu_percent": None}
    assert nodes["inf-gone"]["online"] is False and nodes["inf-gone"]["status"] is None


def test_http_comment_only_on_grants_you_can_view(env):
    client, core, *_ = env
    tokens = {}
    for who in ("admin", "author", "other"):
        tokens[who] = client.post("/auth/register", json={
            "email": f"{who}@lan.test", "password": "password1", "name": who,
        }).json()["token"]
        client.post("/auth/accept-terms", headers=_hdr(tokens[who]))
    client.cookies.clear()
    g = client.post("/grants", json={
        "title": "Need compute", "body": "A short paragraph about the need.",
        "goal_flops": 50,
    }, headers=_hdr(tokens["author"])).json()
    path = f"/grants/{g['id']}"
    assert client.get(path, headers=_hdr(tokens["other"])).status_code == 404
    denied = client.post(f"{path}/comments", json={"body": "sneaky"},
                         headers=_hdr(tokens["other"]))
    assert denied.status_code == 404
    assert denied.json()["detail"] == "Grant not found."
    for who in ("author", "admin"):
        ok = client.post(f"{path}/comments", json={"body": "pending note"},
                         headers=_hdr(tokens[who]))
        assert ok.status_code == 200, ok.text
    client.post(f"/admin/grants/{g['id']}/review", json={"approve": True},
                headers=_hdr(tokens["admin"]))
    ok = client.post(f"{path}/comments", json={"body": "now public"},
                     headers=_hdr(tokens["other"]))
    assert ok.status_code == 200, ok.text
    bodies = [c["body"] for c in client.get(path).json()["comments"]]
    assert bodies == ["pending note", "pending note", "now public"]


def test_http_grants_reject_non_finite_goal_and_donation(env):
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
    core.credits.contribute(core.auth.user_from_token(donor_tok).id, 80.0, 0)
    grant = {"title": "Need compute", "body": "A short paragraph about the need."}
    for goal in ("1e309", "inf", "-inf", "nan", 1e31):
        r = client.post("/grants", json={**grant, "goal_flops": goal},
                        headers=_hdr(member_tok))
        assert r.status_code == 400, (goal, r.text)
    assert client.get("/grants", headers=_hdr(member_tok)).json() == []
    g = client.post("/grants", json={**grant, "goal_flops": 50},
                    headers=_hdr(member_tok)).json()
    client.post(f"/admin/grants/{g['id']}/review", json={"approve": True},
                headers=_hdr(admin_tok))
    for flops in ("nan", "inf", "-inf", "1e309"):
        r = client.post(f"/grants/{g['id']}/donate", json={"flops": flops},
                        headers=_hdr(donor_tok))
        assert r.status_code == 400, (flops, r.text)
    for sort in ("top", "least", "trending"):
        r = client.get("/grants", params={"sort": sort})
        assert r.status_code == 200 and r.json()[0]["received_flops"] == 0


def test_http_admin_cannot_ban_self_or_last_admin(env):
    client, core, *_ = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    admin = core.auth.user_from_token(admin_tok)
    r = client.post(f"/admin/users/{admin.id}/ban", json={}, headers=_hdr(admin_tok))
    assert r.status_code == 400, r.text
    assert client.get("/admin/users", headers=_hdr(admin_tok)).status_code == 200
    # A second admin may ban the first, but not then the one admin left.
    other_tok = client.post("/auth/register", json={
        "email": "other@lan.test", "password": "password1", "name": "Other",
    }).json()["token"]
    other = core.auth.user_from_token(other_tok)
    other.admin = True
    core.db.save(other)
    r = client.post(f"/admin/users/{admin.id}/ban", json={}, headers=_hdr(other_tok))
    assert r.status_code == 200, r.text
    r = client.post(f"/admin/users/{other.id}/ban", json={}, headers=_hdr(other_tok))
    assert r.status_code == 400, r.text
    assert client.get("/admin/users", headers=_hdr(other_tok)).status_code == 200


# --------------------------------------------------------------------------- engine hooks


def _race(n, fn):
    """Run fn(i) on n threads released together; return (results, errors)."""
    barrier = threading.Barrier(n)
    results, errors = [], []

    def run(i):
        barrier.wait()
        try:
            results.append(fn(i))
        except Exception as e:  # noqa: BLE001 - the test inspects what was raised
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def _slow_count(auth, monkeypatch):
    """Widen the check-then-insert window so a racy admin decision shows up every run."""
    count = auth.user_count

    def slow():
        n = count()
        time.sleep(0.05)
        return n

    monkeypatch.setattr(auth, "user_count", slow)


def test_concurrent_registrations_make_exactly_one_admin(core, monkeypatch):
    _slow_count(core.auth, monkeypatch)
    users, errors = _race(8, lambda i: core.auth.register(f"u{i}@lan.test", "password1", f"U{i}"))
    assert errors == []
    assert sum(u.admin for u in users) == 1
    assert sum(core.auth.get(u.id).admin for u in users) == 1


def test_concurrent_google_signups_make_exactly_one_admin(core, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    core.auth.verify_google = lambda tok, cid: {
        "email": f"{tok}@lan.test", "name": tok, "sub": f"sub-{tok}",
    }
    _slow_count(core.auth, monkeypatch)
    results, errors = _race(8, lambda i: core.auth.login_google(f"g{i}"))
    assert errors == []
    assert sum(u.admin for u, _ in results) == 1


def test_concurrent_same_email_registers_once(core):
    users, errors = _race(8, lambda i: core.auth.register("dup@lan.test", "password1", "Dup"))
    assert len(users) == 1
    assert len(errors) == 7
    assert all(isinstance(e, AuthError) and "already" in str(e) for e in errors)


def test_concurrent_same_google_account_signs_in_once(core, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    core.auth.verify_google = lambda tok, cid: {
        "email": "g@lan.test", "name": "Gia", "sub": "sub-1",
    }
    results, errors = _race(8, lambda i: core.auth.login_google("id-token"))
    assert errors == []
    assert len({u.id for u, _ in results}) == 1


@pytest.mark.parametrize("email", [
    "@.", "a b@c d.e", "@lan.test", "a@lan", "a@.test", "a@lan.", "a@ lan.test",
])
def test_register_rejects_malformed_email(core, email):
    with pytest.raises(AuthError, match="email") as e:
        core.auth.register(email, "password1", "x")
    assert e.value.status == 400
    assert core.auth.user_count() == 0


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/auth/register", {"email": 5, "password": "password1"}),
    ("post", "/auth/register", {"email": None, "password": "password1"}),
    ("post", "/auth/register", {"email": "x@lan.test", "password": 12345678}),
    ("post", "/auth/register", {"email": "x@lan.test", "password": "password1", "name": ["x"]}),
    ("post", "/auth/login", {"email": "admin@lan.test", "password": 12345678}),
    ("post", "/auth/login", {"email": ["admin@lan.test"], "password": "password1"}),
    ("patch", "/auth/me", {"name": 5}),
    ("patch", "/auth/me", {"bio": 5}),
    ("post", "/grants", {"title": 12345, "body": "A short paragraph about the need.",
                         "goal_flops": 50}),
    ("post", "/grants", {"title": "Need compute", "body": {"x": 1}, "goal_flops": 50}),
    ("post", "/grants/{approved}/comments", {"body": 123}),
    ("post", "/admin/grants/{pending}/review", {"approve": True, "note": 5}),
    ("post", "/admin/users/{member}/flag", {"reason": 5}),
])
def test_http_wrongly_typed_fields_are_400(env, method, path, payload):
    client, core, *_ = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(admin_tok))
    admin = core.auth.user_from_token(admin_tok)
    member = core.auth.register("m@lan.test", "password1", "Member")
    pending = core.grants.create(admin, "Pending one", "A short paragraph about the need.", 50)
    approved = core.grants.create(admin, "Approved one", "A short paragraph about the need.", 50)
    core.grants.review(admin, approved.id, True)
    client.cookies.clear()
    url = path.format(pending=pending.id, approved=approved.id, member=member.id)
    r = client.request(method, url, json=payload, headers=_hdr(admin_tok))
    assert r.status_code == 400, r.text
    assert core.auth.user_count() == 2
    assert core.grants.get(pending.id).status == "pending"


def test_non_finite_amounts_are_rejected(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0, 50)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(CreditError):
            core.credits.reserve_job(user.id, "job-nan", bad)
        with pytest.raises(CreditError):
            core.credits.allocate_pot(user.id, "g1", bad)
    assert core.credits.job_account("job-nan") is None
    assert core.credits.balance(user.id) == 50.0
    assert core.credits.balance(POT_ID) == 50.0


@pytest.mark.parametrize("max_flops", ["nan", "inf"])
def test_http_non_finite_max_flops_abandons_job(env, max_flops):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = max_flops
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert core.jobs == {}
    r = client.post("/jobs/upload", headers=_hdr(token),
                    files={"dataset": ("train.jsonl", tiny_dataset.read_bytes())},
                    data={"model": str(tiny_model), "steps": 2, "batch_size": 2,
                          "microbatches": 1, "min_stages": 1, "max_flops": max_flops})
    assert r.status_code == 400
    assert core.jobs == {}


def test_http_unexpected_reserve_error_abandons_job(env, monkeypatch):
    client, core, tiny_model, tiny_dataset = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))

    def boom(*a, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(core.credits, "reserve_job", boom)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 1e9
    with pytest.raises(RuntimeError):
        client.post("/jobs", json=spec, headers=_hdr(token))
    assert core.jobs == {}


def _race_count(n, fn):
    """Run fn from n threads released together; return how many calls succeeded."""
    gate = threading.Barrier(n)
    ok = []

    def run():
        gate.wait()
        try:
            fn()
            ok.append(1)
        except (CreditError, GrantError):
            pass

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(ok)


def test_concurrent_spends_cannot_overdraw(core):
    admin, _ = _account(core.auth, "admin@lan.test")
    member, _ = _account(core.auth, "m@lan.test")
    donor, _ = _account(core.auth, "d@lan.test")
    g = core.grants.create(member, "Need compute", "A short paragraph about the need.", 1e3)
    core.grants.review(admin, g.id, True)
    for _ in range(3):
        core.credits.contribute(donor.id, 10.0, 0)
        assert _race_count(16, lambda: core.grants.donate(donor, g.id, 10.0)) == 1
        assert core.credits.balance(donor.id) == 0.0
    assert core.grants.get(g.id).received_flops == 30.0
    assert core.credits.balance(member.id) == 30.0

    core.credits.contribute(member.id, 10.0, 100)
    assert _race_count(16, lambda: core.grants.donate(admin, g.id, 10.0, from_pot=True)) == 1
    assert core.credits.balance(POT_ID) == 0.0
    assert core.grants.get(g.id).received_flops == 40.0

    jobs = iter(range(16))
    assert _race_count(16, lambda: core.credits.reserve_job(member.id, f"job-{next(jobs)}", 40.0)) == 1
    assert core.credits.balance(member.id) == 0.0


def test_concurrent_consume_and_settle_keep_job_books_straight(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0, 0)
    core.credits.reserve_job(user.id, "job-r", 80.0)
    # Every charge lands on the account: no lost updates to spent_flops.
    assert _race_count(16, lambda: core.credits.consume_job("job-r", 1.0)) == 16
    assert core.credits.job_account("job-r").spent_flops == 16.0
    assert core.credits.lifetime_spent(user.id) == 16.0
    # The leftover is released exactly once however many settles race.
    _race_count(16, lambda: core.credits.settle_job("job-r"))
    assert core.credits.balance(user.id) == 84.0
    assert core.credits.reserved_in_flight(user.id) == 0.0
    assert core.credits.consume_job("job-r", 1.0) == 0.0


def test_http_admin_flags_reject_string_booleans(env):
    client, core, *_ = env
    admin_tok = client.post("/auth/register", json={
        "email": "admin@lan.test", "password": "password1", "name": "Admin",
    }).json()["token"]
    member_tok = client.post("/auth/register", json={
        "email": "m@lan.test", "password": "password1", "name": "Member",
    }).json()["token"]
    member = core.auth.user_from_token(member_tok)
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    g = client.post("/grants", json={
        "title": "Need compute", "body": "A short paragraph about the need.",
        "goal_flops": 50,
    }, headers=_hdr(member_tok)).json()
    # bool("false") is True: these used to approve and ban.
    for bad in ("yes", 0, 1, [], "False"):
        r = client.post(f"/admin/grants/{g['id']}/review", json={"approve": bad},
                        headers=_hdr(admin_tok))
        assert r.status_code == 400, (bad, r.text)
        r = client.post(f"/admin/users/{member.id}/ban", json={"banned": bad},
                        headers=_hdr(admin_tok))
        assert r.status_code == 400, (bad, r.text)
        r = client.post(f"/grants/{g['id']}/donate", json={"flops": 1, "from_pot": bad},
                        headers=_hdr(member_tok))
        assert r.status_code == 400, (bad, r.text)
    assert core.auth.get(member.id).banned is False
    declined = client.post(f"/admin/grants/{g['id']}/review", json={"approve": "false"},
                           headers=_hdr(admin_tok))
    assert declined.status_code == 200, declined.text
    assert declined.json()["status"] == "declined"
    r = client.post(f"/admin/users/{member.id}/ban", json={"banned": "false"},
                    headers=_hdr(admin_tok))
    assert r.status_code == 200, r.text
    assert core.auth.get(member.id).banned is False
    client.post(f"/admin/users/{member.id}/ban", json={}, headers=_hdr(admin_tok))
    assert core.auth.get(member.id).banned is True


# --------------------------------------------------------------------------- engine hooks


@pytest.mark.parametrize("bad", [
    {"flops": float("nan")}, {"flops": float("inf")}, {"flops": -1e9}, {"wall_s": float("nan")},
])
def test_step_with_insane_usage_is_ignored(core, bad):
    """NaN used to raise an IntegrityError (dropping the node); inf billed the whole reserve."""
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12, 0)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 5e9)
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    usage = _usage().model_copy(update=bad)
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=usage,
    )))
    assert core.credits.lifetime_spent(user.id) == 0.0
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12)
    assert core.ledger.job_records(job.id) == []
    assert job.row.status == "running"


def test_consume_job_ignores_non_finite_flops(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 1e12, 0)
    core.credits.reserve_job(user.id, "j", 5e9)
    assert core.credits.consume_job("j", float("inf")) == 0.0
    assert core.credits.consume_job("j", float("nan")) == 0.0
    assert core.credits.job_account("j").spent_flops == 0.0


def test_a_message_that_raises_does_not_escape_handle(core, monkeypatch):
    """An unexpected error handling one message used to drop the node's websocket."""
    def boom(*_):
        raise RuntimeError("bug")

    monkeypatch.setattr(core.registry, "heartbeat", boom)
    asyncio.run(core.handle("n1", P.Heartbeat(node_id="n1", status="idle")))
