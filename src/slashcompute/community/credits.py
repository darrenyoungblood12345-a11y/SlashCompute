"""1:1 FLOP credits. Earn what you compute; spend what you take."""

from __future__ import annotations

import math
import threading
from typing import Optional

from sqlalchemy import func
from sqlmodel import select, update

from slashcompute.coordinator.db import CreditTxn, Database, Grant, JobAccount, NodeOwner, User

POT_ID = "__pot__"
BALANCE_KINDS = ("earn", "reserve", "release", "donate", "receive", "allocate")


class CreditError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Credits:
    def __init__(self, db: Database) -> None:
        self.db = db
        # HTTP handlers run on worker threads. Every check-then-debit holds this
        # lock and commits its rows in one transaction, so concurrent spends
        # cannot overdraw a balance or lose each other's updates.
        self._spend_lock = threading.Lock()

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

    def summary(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "balance": self.balance(user_id),
            "lifetime_earned": self.lifetime_earned(user_id),
            "lifetime_spent": self.lifetime_spent(user_id),
            "pot": self.balance(POT_ID),
        }

    def bind_node(self, node_id: str, user_id: str) -> None:
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

    def reserve_job(self, user_id: str, job_id: str, flops: float) -> JobAccount:
        if not math.isfinite(flops) or flops <= 0:
            raise CreditError("Set a FLOP budget greater than zero.")
        with self._spend_lock:
            have = self.balance(user_id)
            if have < flops:
                raise CreditError(
                    f"Need {flops:.3e} FLOPs; you have {have:.3e}. Contribute first.",
                )
            acct = JobAccount(job_id=job_id, user_id=user_id, reserved_flops=flops, spent_flops=0.0)
            self.db.add(CreditTxn(user_id=user_id, kind="reserve", amount=-flops, job_id=job_id), acct)
        return acct

    def job_account(self, job_id: str) -> Optional[JobAccount]:
        return self.db.get(JobAccount, job_id)

    def consume_job(self, job_id: str, flops: float) -> float:
        """Charge a job's reserve. Returns the FLOPs actually charged, which is
        less than asked once the reserve runs out (and 0 without a reserve)."""
        if not math.isfinite(flops) or flops <= 0:
            return 0.0
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return 0.0
            take = max(0.0, min(flops, acct.reserved_flops - acct.spent_flops))
            if take:
                acct.spent_flops += take
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="consumed", amount=take, job_id=job_id))
                s.commit()
            return take

    def job_exhausted(self, job_id: str) -> bool:
        acct = self.job_account(job_id)
        return acct is not None and acct.spent_flops >= acct.reserved_flops - 1e-9

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
        if not math.isfinite(flops) or flops <= 0:
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
        """Move credits to a grant's author and add them to the grant's total,
        all in one transaction."""
        with self._spend_lock:
            if self.balance(source_id) < flops:
                raise CreditError(short_msg)
            with self.db.session() as s:
                s.add(CreditTxn(user_id=source_id, kind=kind, amount=-flops, grant_id=grant_id))
                s.add(CreditTxn(user_id=recipient_id, kind="receive", amount=flops, grant_id=grant_id))
                # Incremented in SQL: saving a copy read earlier would drop
                # donations that landed in between.
                s.exec(update(Grant).where(Grant.id == grant_id)
                       .values(received_flops=Grant.received_flops + flops))
                s.commit()

    def leaderboard(self, limit: int = 20) -> list[dict]:
        with self.db.session() as s:
            rows = s.exec(
                select(CreditTxn.user_id, func.sum(CreditTxn.amount)).where(
                    CreditTxn.kind == "generated",
                ).group_by(CreditTxn.user_id)
            ).all()
            users = {u.id: u for u in s.exec(select(User)).all()}
        ranked = sorted(
            ((uid, float(total or 0.0)) for uid, total in rows if uid != POT_ID),
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
