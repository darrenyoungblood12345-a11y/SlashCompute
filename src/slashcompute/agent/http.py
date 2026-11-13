"""HTTP client for dataset / checkpoint / verification blobs on the coordinator."""

from __future__ import annotations

from pathlib import Path

import httpx


class CoordHTTP:
    def __init__(self, base: str, timeout: float = 120.0, node_token: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        # Issued by the coordinator in Welcome; identifies this node's requests.
        self.node_token = node_token

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return self.base + (path if path.startswith("/") else "/" + path)

    def _headers(self, url: str) -> dict[str, str]:
        base, target = httpx.URL(self.base), httpx.URL(url)
        # An assignment may point at an absolute URL; the token only ever goes
        # to the coordinator that issued it.
        if self.node_token and (base.scheme, base.host, base.port) == (
            target.scheme, target.host, target.port,
        ):
            return {"X-Node-Token": self.node_token}
        return {}

    def get_file(self, path: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self.url(path)
        with httpx.stream("GET", url, headers=self._headers(url), timeout=self.timeout) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    def put_bytes(self, path: str, data: bytes, params: dict | None = None) -> None:
        url = self.url(path)
        r = httpx.post(url, content=data, params=params, headers=self._headers(url),
                       timeout=self.timeout)
        r.raise_for_status()
