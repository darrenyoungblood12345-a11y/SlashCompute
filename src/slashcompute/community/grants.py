"""Community grants: propose, moderate, fund, discuss."""

from __future__ import annotations

import secrets
from typing import Optional

from sqlmodel import select

from slashcompute.community.credits import Credits
from slashcompute.coordinator.db import Database, Grant, GrantComment, User, UserFlag


class GrantError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Grants:
    def __init__(self, db: Database, credits: Credits) -> None:
        self.db = db
        self.credits = credits

    def create(self, author: User, title: str, body: str, goal_flops: float) -> Grant:
        if author.banned:
            raise GrantError("Banned accounts cannot open grants.", 403)
        title = (title or "").strip()[:120]
        body = (body or "").strip()[:4000]
        if len(title) < 4 or len(body) < 20:
            raise GrantError("Describe the need: a title and at least a short paragraph.")
        try:
            goal = float(goal_flops)
        except (TypeError, ValueError) as e:
            raise GrantError("Goal must be a FLOP number.") from e
        if goal <= 0:
            raise GrantError("Goal must be greater than zero.")
        grant = Grant(
            id=secrets.token_hex(6), author_id=author.id, title=title, body=body,
            goal_flops=goal, status="pending",
        )
        self.db.add(grant)
        return grant

    def get(self, grant_id: str) -> Grant:
        g = self.db.get(Grant, grant_id)
        if g is None:
            raise GrantError("Grant not found.", 404)
        return g

    def review(self, admin: User, grant_id: str, approve: bool) -> Grant:
        if not admin.admin:
            raise GrantError("Admin only.", 403)
        g = self.get(grant_id)
        if g.status != "pending":
            raise GrantError("This grant was already reviewed.")
        g.status = "approved" if approve else "declined"
        self.db.save(g)
        return g

    def donate(self, donor: User, grant_id: str, flops: float, *, from_pot: bool = False) -> Grant:
        if donor.banned:
            raise GrantError("Banned accounts cannot donate.", 403)
        g = self.get(grant_id)
        if g.status != "approved":
            raise GrantError("Only approved grants can receive FLOPs.")
        if not from_pot and donor.id == g.author_id:
            raise GrantError("Donate to someone else's grant.")
        flops = float(flops)
        # The transfer also adds to the grant's total, atomically.
        if from_pot:
            if not donor.admin:
                raise GrantError("Only an admin can allocate the community pot.", 403)
            self.credits.allocate_pot(g.author_id, g.id, flops)
        else:
            self.credits.donate(donor.id, g.author_id, g.id, flops)
        return self.get(g.id)

    def comment(self, user: User, grant_id: str, body: str) -> GrantComment:
        g = self.get(grant_id)
        if g.status == "declined":
            raise GrantError("This grant is not public.")
        text = (body or "").strip()[:2000]
        if len(text) < 2:
            raise GrantError("Write a short comment.")
        row = GrantComment(grant_id=g.id, user_id=user.id, body=text)
        self.db.add(row)
        return row

    def comments(self, grant_id: str) -> list[GrantComment]:
        with self.db.session() as s:
            return list(s.exec(
                select(GrantComment).where(GrantComment.grant_id == grant_id)
                .order_by(GrantComment.created_at)
            ).all())

    def list(self, *, include_pending: bool = False, sort: str = "top",
             viewer: Optional[User] = None) -> list[Grant]:
        with self.db.session() as s:
            rows = list(s.exec(select(Grant)).all())
        visible = []
        for g in rows:
            if g.status == "approved":
                visible.append(g)
            elif include_pending or (viewer and (viewer.admin or viewer.id == g.author_id)):
                visible.append(g)
        rows = visible
        if sort == "least":
            rows.sort(key=lambda g: (g.received_flops / g.goal_flops) if g.goal_flops else 0)
        elif sort == "trending":
            rows.sort(key=lambda g: g.created_at, reverse=True)
        else:
            rows.sort(key=lambda g: g.received_flops, reverse=True)
        return rows

    def flag_user(self, admin: User, user: User, reason: str) -> UserFlag:
        if not admin.admin:
            raise GrantError("Admin only.", 403)
        text = (reason or "").strip()[:500] or "flagged"
        user.flagged = True
        self.db.save(user)
        row = UserFlag(user_id=user.id, admin_id=admin.id, reason=text)
        self.db.add(row)
        return row

    def view(self, g: Grant, names: Optional[dict[str, str]] = None) -> dict:
        names = names or {}
        return {
            "id": g.id, "author_id": g.author_id,
            "author": names.get(g.author_id, g.author_id[:8]),
            "title": g.title, "body": g.body, "goal_flops": g.goal_flops,
            "received_flops": g.received_flops, "status": g.status,
            "created_at": g.created_at, "progress": (
                g.received_flops / g.goal_flops if g.goal_flops else 0.0
            ),
        }
