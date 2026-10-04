"""1:1 FLOP credits. Earn what you compute; spend what you take."""

from __future__ import annotations

import math
import threading


from typing import Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from slashcompute.common.config import WELCOME_FLOPS
from slashcompute.coordinator.db import CreditTxn, Database, JobAccount, NodeOwner, User, now

POT_ID = "__pot__"
BALANCE_KINDS = ("earn", "reserve", "release", "donate", "receive", "allocate",
                 "welcome")


class CreditError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class InsufficientCredits(CreditError):
    """The balance does not cover a reservation (training keeps its 400; chat answers 402)."""


class Credits:
    def __init__(self, db: Database) -> None:
        self.db = db
        # Serialises check-balance-then-debit so concurrent spends cannot overdraw.
        self._spend_lock = threading.Lock()
        self._welcomed: set[str] = set()   # users known to hold their welcome credit: skip the insert

    def post(self, user_id: str, kind: str, amount: float, *, job_id: Optional[str] = None,
             grant_id: Optional[str] = None, node_id: Optional[str] = None,
             note: Optional[str] = None) -> CreditTxn:
        row = CreditTxn(user_id=user_id, kind=kind, amount=float(amount), job_id=job_id,
                        grant_id=grant_id, node_id=node_id, note=note)
        self.db.add(row)
        return row

    def balance(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind.in_(BALANCE_KINDS),
                )
            ).one()
        return float(total or 0.0)

    def lifetime_earned(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind == "generated",
                )
            ).one()
        return float(total or 0.0)

    def lifetime_spent(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind == "consumed",
                )
            ).one()
        return float(total or 0.0)

    def reserved_in_flight(self, user_id: str) -> float:
        with self.db.session() as s:
            rows = s.exec(select(JobAccount).where(JobAccount.user_id == user_id)).all()
        return float(sum(max(0.0, a.reserved_flops - a.spent_flops) for a in rows))

    def summary(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "balance": self.balance(user_id),
            "lifetime_earned": self.lifetime_earned(user_id),
            "lifetime_spent": self.lifetime_spent(user_id),
            "pot": self.balance(POT_ID),
            "reserved_in_flight": self.reserved_in_flight(user_id),
        }

    def list_txns(self, user_id: str, *, limit: int = 50,
                  before_id: Optional[int] = None) -> dict:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(200, limit))
        with self.db.session() as s:
            q = select(CreditTxn).where(CreditTxn.user_id == user_id)
            if before_id is not None:
                q = q.where(CreditTxn.id < int(before_id))
            q = q.order_by(CreditTxn.id.desc())
            rows = list(s.exec(q).all())[: limit + 1]
        items = rows[:limit]
        return {
            "items": [
                {
                    "id": r.id, "kind": r.kind, "amount": r.amount,
                    "job_id": r.job_id, "grant_id": r.grant_id, "node_id": r.node_id,
                    "note": r.note, "created_at": r.created_at,
                }
                for r in items
            ],
            "next_cursor": items[-1].id if len(rows) > limit else None,
        }

    def live(self, user_id: str, window_s: float = 60.0) -> dict:
        try:
            window_s = float(window_s)
        except (TypeError, ValueError):
            window_s = 60.0
        window_s = max(1.0, min(3600.0, window_s))
        cutoff = now() - window_s
        with self.db.session() as s:
            recent = list(s.exec(
                select(CreditTxn).where(
                    CreditTxn.kind == "generated", CreditTxn.created_at >= cutoff,
                )
            ).all())
        you_flops = sum(r.amount for r in recent if r.user_id == user_id)
        community_flops = sum(r.amount for r in recent if r.user_id != POT_ID)
        snap = self.summary(user_id)
        return {
            **snap,
            "window_s": window_s,
            "community_flops": community_flops,
            "you": {
                "flops": you_flops,
                "balance": snap["balance"],
                "lifetime_earned": snap["lifetime_earned"],
            },
        }

    def nodes_for(self, user_id: str) -> list[NodeOwner]:
        with self.db.session() as s:
            return list(s.exec(select(NodeOwner).where(NodeOwner.user_id == user_id)).all())

    def bind_node(self, node_id: str, user_id: str, *, take_over: bool = False) -> None:
        # Node ids are public: never hand one user's node (and its earnings) to another unless
        # the caller has established the new account may take it over (see Coordinator._check_claim).
        owner = self.owner_of(node_id)
        if owner is not None and owner != user_id and not take_over:
            raise PermissionError("node id belongs to another account")
        self.db.save(NodeOwner(node_id=node_id, user_id=user_id))

    def owner_of(self, node_id: str) -> Optional[str]:
        row = self.db.get(NodeOwner, node_id)
        return row.user_id if row else None

    def contribute(self, user_id: str, flops: float, grant_split: int, *,
                   node_id: Optional[str] = None, job_id: Optional[str] = None) -> None:
        if flops <= 0:
            return
        split = max(0, min(100, int(grant_split))) / 100.0
        pot = flops * split
        personal = flops - pot
        self.post(user_id, "generated", flops, node_id=node_id, job_id=job_id)
        if personal:
            self.post(user_id, "earn", personal, node_id=node_id, job_id=job_id)
        if pot:
            self.post(POT_ID, "earn", pot, node_id=node_id, job_id=job_id,
                      note=f"tithe from {user_id}")

    def credit_host(self, owner_id: Optional[str], flops: float, *,
                    node_id: Optional[str] = None, job_id: Optional[str] = None) -> None:
        """Pay out FLOPs a job was charged for a node's work. A share nobody may earn (unbound
        node, missing or banned owner) goes to the community pot so charged FLOPs never vanish."""
        if flops <= 0:
            return
        owner = self.db.get(User, owner_id) if owner_id else None
        if owner is None or owner.banned:
            self.post(POT_ID, "earn", flops, node_id=node_id, job_id=job_id,
                      note="unclaimed host share")
            return
        self.contribute(owner.id, flops, owner.grant_split, node_id=node_id, job_id=job_id)

    def grant_welcome(self, user_id: str, flops: float = WELCOME_FLOPS) -> float:
        """One-time sign-in credit. Returns FLOPs granted (0 if already given)."""
        if not math.isfinite(flops) or flops <= 0 or user_id in self._welcomed:
            return 0.0
        user = self.db.get(User, user_id)
        if user is None or user.banned:
            return 0.0
        try:
            self.post(user_id, "welcome", flops, note="Welcome credit")
        except IntegrityError:  # uq_credit_txns_welcome: someone else got there first
            self._welcomed.add(user_id)
            return 0.0
        self._welcomed.add(user_id)
        return float(flops)

    def reserve_job(self, user_id: str, job_id: str, flops: float) -> JobAccount:
        if not math.isfinite(flops) or flops <= 0:
            raise CreditError("Set a FLOP budget greater than zero.")
        with self._spend_lock:
            have = self.balance(user_id)
            if have < flops:
                raise InsufficientCredits(
                    f"Need {flops:.3e} FLOPs; you have {have:.3e}. Contribute first.",
                )
            acct = JobAccount(job_id=job_id, user_id=user_id, reserved_flops=flops, spent_flops=0.0)
            self.db.add(CreditTxn(user_id=user_id, kind="reserve", amount=-flops, job_id=job_id), acct)
        return acct

    def job_account(self, job_id: str) -> Optional[JobAccount]:
        return self.db.get(JobAccount, job_id)

    def consume_job(self, job_id: str, flops: float) -> float:
        """Charge a job's reserve. Returns FLOPs actually charged."""
        if not math.isfinite(flops) or flops <= 0:
            return 0.0
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return 0.0
            remaining = acct.reserved_flops - acct.spent_flops
            take = min(flops, remaining)
            if take:
                acct.spent_flops += take
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="consumed", amount=take, job_id=job_id))
                s.commit()
        return take

    def job_exhausted(self, job_id: str) -> bool:
        acct = self.job_account(job_id)
        if acct is None:
            return False
        return acct.spent_flops >= acct.reserved_flops - 1e-9

    def settle_job(self, job_id: str) -> None:
        # Settling shrinks the reserve to what was spent, so a settled account
        # has nothing left to release and repeat settles are no-ops.
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return
            leftover = max(0.0, acct.reserved_flops - acct.spent_flops)
            if leftover:
                acct.reserved_flops = acct.spent_flops
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="release", amount=leftover, job_id=job_id))
                s.commit()

    def donate(self, donor_id: str, recipient_id: str, grant_id: str, flops: float) -> None:
        if flops <= 0:
            raise CreditError("Donation must be positive.")
        self._transfer(donor_id, "donate", recipient_id, grant_id, flops,
                       "Not enough personal credits to donate.")

    def allocate_pot(self, recipient_id: str, grant_id: str, flops: float) -> None:
        if not math.isfinite(flops) or flops <= 0:
            raise CreditError("Allocation must be positive.")
        self._transfer(POT_ID, "allocate", recipient_id, grant_id, flops,
                       "Community pot does not have that many FLOPs.")

    def _transfer(self, source_id: str, kind: str, recipient_id: str, grant_id: str,
                  flops: float, short_msg: str) -> None:
        with self._spend_lock:
            if self.balance(source_id) < flops:
                raise CreditError(short_msg)
            self.db.add(
                CreditTxn(user_id=source_id, kind=kind, amount=-flops, grant_id=grant_id),
                CreditTxn(user_id=recipient_id, kind="receive", amount=flops, grant_id=grant_id),
            )

    def leaderboard(self, limit: int = 20) -> list[dict]:
        with self.db.session() as s:
            rows = s.exec(
                select(CreditTxn.user_id, func.sum(CreditTxn.amount)).where(
                    CreditTxn.kind == "generated",
                ).group_by(CreditTxn.user_id)
            ).all()
            users = {u.id: u for u in s.exec(select(User)).all()}
        ranked = sorted(
            ((uid, float(total or 0.0)) for uid, total in rows
             if uid != POT_ID and not (uid in users and users[uid].banned)),
            key=lambda x: x[1], reverse=True,
        )[:limit]
        out = []
        for uid, earned in ranked:
            u = users.get(uid)
            out.append({
                "user_id": uid,
                "name": u.name if u else uid[:8],
                "lifetime_earned": earned,
                "balance": self.balance(uid),
            })
        return out
