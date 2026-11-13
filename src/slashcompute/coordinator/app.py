"""HTTP + WebSocket front end for the coordinator."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlmodel import select

from slashcompute.community.credits import CreditError
from slashcompute.community.http import _token, mount_community
from slashcompute.common.config import EngineConfig
from slashcompute.common.protocol import Register, dump, parse_agent_message
from slashcompute.coordinator.checkpoints import InvalidCheckpoint
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import Verification
from slashcompute.jobs import parse_spec

log = logging.getLogger(__name__)


def create_app(cfg: EngineConfig, advertise: bool = False) -> FastAPI:
    core = Coordinator(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        core.start()
        adv = None
        if advertise:
            try:
                from slashcompute.common.discovery import Advertiser

                adv = Advertiser(cfg.coordinator_port)
            except Exception as e:
                log.warning("mDNS advertising unavailable: %s", e)
        yield
        if adv:
            adv.close()
        await core.stop()

    app = FastAPI(title="/compute coordinator", lifespan=lifespan)
    app.state.core = core

    @app.middleware("http")
    async def session_cookie(request: Request, call_next):
        response = await call_next(request)
        token = getattr(request.state, "session_token", None)
        if token:
            response.set_cookie("slashcompute_session", token, httponly=True, samesite="lax")
        if request.url.path.rstrip("/").endswith("/auth/logout"):
            response.delete_cookie("slashcompute_session")
        return response

    mount_community(app, core)

    def _user(request: Request, authorization: Optional[str] = None):
        user = core.auth.session_user(_token(request, authorization))
        if user is not None and user.banned:
            raise HTTPException(403, "This account is banned.")
        return user

    def _reserve(user, job, max_flops: Optional[float]):
        if user is None:
            return
        def fail(status: int, msg: str):
            core.abandon_job(job, msg)
            raise HTTPException(status, msg)
        if user.accepted_terms_at is None:
            fail(403, "Accept the terms before taking from the pool.")
        if max_flops is None:
            fail(400, "Set a FLOP budget (max_flops) to take.")
        try:
            budget = float(max_flops)
        except (TypeError, ValueError):
            fail(400, "max_flops must be a number.")
        try:
            core.credits.reserve_job(user.id, job.id, budget)
        except CreditError as e:
            core.abandon_job(job, str(e))
            raise HTTPException(e.status, str(e)) from e

    # ------------------------------------------------------------ agents

    @app.websocket("/ws/agent")
    async def agent_ws(ws: WebSocket):
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send(msg):
            async with send_lock:
                await ws.send_text(dump(msg))

        node_id: Optional[str] = None
        try:
            first = parse_agent_message(await ws.receive_text())
            if not isinstance(first, Register):
                await ws.close(code=4000, reason="first message must be register")
                return
            node_id = first.node_id
            try:
                await core.on_register(first, send)
            except PermissionError:
                await ws.close(code=4003, reason="banned")
                return
            while True:
                raw = await ws.receive_text()
                try:
                    msg = parse_agent_message(raw)
                except ValidationError as e:
                    log.warning("bad message from %s: %s", node_id[:8], e)
                    continue
                await core.handle(node_id, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("agent session error")
        finally:
            node = core.registry.get(node_id) if node_id else None
            if node is not None and node.send is send:
                await core.on_disconnect(node_id)

    # ------------------------------------------------------------ jobs

    def _job(job_id: str):
        job = core.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/jobs")
    async def submit_job(body: dict, request: Request,
                         authorization: Optional[str] = Header(default=None)):
        user = _user(request, authorization)
        try:
            spec = parse_spec(body)
            job = core.submit(spec)
        except (ValidationError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        _reserve(user, job, body.get("max_flops"))
        return core.job_view(job)

    @app.post("/jobs/upload")
    async def upload_job(
        request: Request,
        dataset: UploadFile = File(...),
        model: str = Form("mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
        steps: int = Form(10),
        min_stages: int = Form(2),
        batch_size: int = Form(4),
        microbatches: int = Form(2),
        max_flops: Optional[float] = Form(None),
        authorization: Optional[str] = Header(default=None),
    ):
        uploads = cfg.home / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        raw = (dataset.filename or "train.jsonl").replace("\\", "/").split("/")[-1]
        dest = uploads / f"{uuid.uuid4().hex}_{raw}"
        dest.write_bytes(await dataset.read())
        user = _user(request, authorization)
        try:
            spec = parse_spec({
                "kind": "lora_finetune", "model": model, "dataset_path": str(dest),
                "steps": steps, "min_stages": min_stages,
                "batch_size": batch_size, "microbatches": microbatches,
            })
            job = core.submit(spec)
        except (ValidationError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        _reserve(user, job, max_flops)
        return core.job_view(job)

    @app.get("/jobs")
    async def list_jobs():
        return [core.job_view(j) for j in sorted(core.jobs.values(), key=lambda j: j.row.submitted_at)]

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return core.job_view(_job(job_id))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = _job(job_id)
        await core.cancel_job(job)
        return core.job_view(job)

    @app.get("/jobs/{job_id}/usage")
    async def job_usage(job_id: str):
        _job(job_id)
        return [r.model_dump() for r in core.ledger.job_records(job_id)]

    @app.get("/jobs/{job_id}/dataset")
    async def get_dataset(job_id: str):
        _job(job_id)
        return FileResponse(core.dataset_path(job_id))

    @app.get("/jobs/{job_id}/checkpoints/{step}")
    async def get_checkpoint(job_id: str, step: int):
        path = core.checkpoints.merged_path(job_id, step)
        if not path.exists():
            raise HTTPException(404, "checkpoint not found")
        return FileResponse(path)

    @app.post("/jobs/{job_id}/checkpoints/{step}")
    async def put_checkpoint(job_id: str, step: int, epoch: int, stage: int, request: Request,
                             x_node_token: Optional[str] = Header(default=None)):
        # Agents have no account; the token from their Welcome says which
        # registered node is calling.
        node = core.registry.by_token(x_node_token)
        if node is None:
            raise HTTPException(401, "a registered node's token is required (X-Node-Token)")
        data = await request.body()
        try:
            core.on_checkpoint_upload(node.node_id, job_id, epoch, stage, step, data)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except InvalidCheckpoint as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.get("/jobs/{job_id}/adapter/{name}")
    async def get_adapter(job_id: str, name: str):
        if name not in ("adapters.safetensors", "adapter_config.json"):
            raise HTTPException(404)
        path = core.checkpoints.job_dir(job_id) / "adapter" / name
        if not path.exists():
            raise HTTPException(404, "adapter not ready")
        return FileResponse(path)

    # ------------------------------------------------------------ nodes, ledger, verification

    @app.get("/nodes")
    async def nodes():
        return core.node_view()

    @app.get("/ledger")
    async def ledger():
        return core.ledger.summary()

    @app.get("/verifications")
    async def verifications():
        with core.db.session() as s:
            rows = s.exec(select(Verification).order_by(Verification.created_at)).all()
        return [r.model_dump() for r in rows]

    @app.post("/verify/{vid}/bundle")
    async def put_bundle(vid: str, request: Request):
        try:
            core.verification.store_bundle(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a bundle")
        return {"ok": True}

    @app.get("/verify/{vid}/bundle")
    async def get_bundle(vid: str):
        if core.db.get(Verification, vid) is None:
            raise HTTPException(404)
        path = core.verification.bundle_path(vid)
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/verify/{vid}/result")
    async def put_result(vid: str, request: Request):
        try:
            core.verification.store_result(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a result")
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True, "nodes": len(core.registry.nodes), "jobs": len(core.jobs)}

    return app
