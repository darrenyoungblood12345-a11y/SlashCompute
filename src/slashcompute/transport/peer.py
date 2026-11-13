"""Direct TCP links between neighbouring pipeline stages.

Each link is one TCP connection. A background task reads frames into a queue
so a stage can keep receiving while its own compute or sends are in flight.
The downstream stage listens on its data port and the upstream stage dials
it. A ``hello`` frame binds the connection to a (job, epoch) so stale stages
can't cross-talk.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from slashcompute.transport.serialization import (
    Frame, _HDR, decode_header, encode, tensor_from_bytes,
)

log = logging.getLogger(__name__)

_CLOSED = object()


class LinkClosed(ConnectionError):
    pass


class Link:
    """Bidirectional frame channel. Subclasses provide the byte transport."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.bytes_sent = 0
        self.bytes_received = 0

    async def send(self, frame: Frame) -> None:
        raise NotImplementedError

    async def recv(self, timeout: Optional[float] = None) -> Frame:
        item = await asyncio.wait_for(self._queue.get(), timeout)
        if item is _CLOSED:
            self._queue.put_nowait(_CLOSED)
            raise LinkClosed("peer link closed")
        if isinstance(item, BaseException):
            self._queue.put_nowait(item)
            raise LinkClosed(f"peer link failed: {item!r}") from item
        return item

    async def close(self) -> None:
        raise NotImplementedError


class TcpLink(Link):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer
        self._send_lock = asyncio.Lock()
        self._task = asyncio.create_task(self._read_loop())

    @property
    def peername(self):
        return self._writer.get_extra_info("peername")

    async def _read_frame(self) -> Frame:
        (hlen,) = _HDR.unpack(await self._reader.readexactly(_HDR.size))
        header = decode_header(await self._reader.readexactly(hlen))
        tensors = {}
        for spec in header["tensors"]:
            buf = await self._reader.readexactly(spec["nbytes"])
            tensors[spec["name"]] = tensor_from_bytes(spec, buf)
            self.bytes_received += spec["nbytes"]
        self.bytes_received += _HDR.size + hlen
        return Frame(header["kind"], header["meta"], tensors)

    async def _read_loop(self) -> None:
        try:
            while True:
                self._queue.put_nowait(await self._read_frame())
        except (asyncio.IncompleteReadError, ConnectionError):
            self._queue.put_nowait(_CLOSED)
        except asyncio.CancelledError:
            self._queue.put_nowait(_CLOSED)
        except Exception as e:  # malformed frame
            log.exception("peer link read failed")
            self._queue.put_nowait(e)

    async def send(self, frame: Frame) -> None:
        chunks = encode(frame)
        async with self._send_lock:
            if self._writer.is_closing():
                raise LinkClosed("peer link closed")
            try:
                for c in chunks:
                    self._writer.write(c)
                    self.bytes_sent += len(c)
                await self._writer.drain()
            except ConnectionError as e:
                raise LinkClosed(str(e)) from e

    def abort(self) -> None:
        """Drop the connection now (no flush); the read loop then reports it closed."""
        self._writer.transport.abort()

    async def close(self) -> None:
        self._task.cancel()
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except Exception:
            pass


class MemoryLink(Link):
    """In-process link, used by the single-process pipeline and tests."""

    def __init__(self) -> None:
        super().__init__()
        self.other: Optional[MemoryLink] = None

    @classmethod
    def pair(cls) -> tuple["MemoryLink", "MemoryLink"]:
        a, b = cls(), cls()
        a.other, b.other = b, a
        return a, b

    async def send(self, frame: Frame) -> None:
        if self.other is None:
            raise LinkClosed("peer link closed")
        self.other._queue.put_nowait(frame)

    async def close(self) -> None:
        if self.other is not None:
            self.other._queue.put_nowait(_CLOSED)
            self.other.other = None
        self.other = None


async def connect(host: str, port: int, hello: dict, timeout: float = 60.0,
                  retry_interval: float = 0.25) -> TcpLink:
    """Dial a downstream stage, retrying until it is listening."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            break
        except OSError:
            if loop.time() > deadline:
                raise
            await asyncio.sleep(retry_interval)
    link = TcpLink(reader, writer)
    try:
        await link.send(Frame("hello", hello))
        ack = await link.recv(timeout)
    except BaseException:
        # Cancelled or failed mid-hello: no caller holds this connection yet, so nobody else
        # would close it, and the peer may already have taken it as its upstream link.
        link.abort()
        raise
    if ack.kind != "hello_ack":
        await link.close()
        raise LinkClosed(f"peer rejected hello: {ack.meta}")
    return link


class LinkServer:
    """Accepts exactly one upstream link whose hello matches ``expect``."""

    def __init__(self, host: str, port: int, expect: dict) -> None:
        self.host, self.port, self.expect = host, port, expect
        self._accepted: asyncio.Future = asyncio.get_running_loop().create_future()
        self._claimed = False                 # accept() handed the link to the caller
        self._pending: set[TcpLink] = set()   # connections still in the hello exchange
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self) -> "LinkServer":
        # Bound first, served second: asyncio yields once after it starts serving, and a cancel
        # landing there would drop the only reference to a live listener (the port then stays
        # bound for the life of the process).
        self._server = await asyncio.start_server(self._on_conn, self.host, self.port,
                                                  reuse_address=True, start_serving=False)
        try:
            await self._server.start_serving()
        except BaseException:
            self._server.close()
            raise
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def _on_conn(self, reader, writer) -> None:
        link = TcpLink(reader, writer)
        self._pending.add(link)
        try:
            hello = await link.recv(10.0)
        except Exception:
            await link.close()
            return
        finally:
            self._pending.discard(link)
        ok = hello.kind == "hello" and all(hello.meta.get(k) == v for k, v in self.expect.items())
        if not ok or self._accepted.done():
            await link.send(Frame("hello_reject", {}))
            await link.close()
            return
        await link.send(Frame("hello_ack", {}))
        self._accepted.set_result(link)

    async def accept(self, timeout: Optional[float] = None) -> TcpLink:
        link = await asyncio.wait_for(asyncio.shield(self._accepted), timeout)
        self._claimed = True
        return link

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()                  # stops listening, so the port frees now
        # wait_closed() waits for every connection, so close the ones nobody else
        # will: half-done hellos and a link accepted after the caller gave up on it.
        links = list(self._pending)
        if self._accepted.done() and not self._accepted.cancelled() and not self._claimed:
            links.append(self._accepted.result())
        for link in links:
            await link.close()
        await self._server.wait_closed()
