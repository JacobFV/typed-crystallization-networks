"""Caching rendered episodes, in a way that cannot go stale.

The corpus is infinite and `(lesson, seed, difficulty, presentation)` is a pure
function to bytes, so object storage here is a **cache and not the store of
record**. Nothing is ever only in the bucket; a miss costs a re-render, never a
lost episode. That is what removes the two problems a dataset bucket normally
has — there is no manifest to keep in sync, and there is nothing to reconcile
after a crash.

Two rules make it safe, and both are enforced here rather than left to the
caller.

**The renderer version is in the key.** :meth:`Address.cache_key` includes it, so
changing a renderer cannot silently serve bytes made by the old one. The failure
this prevents has already happened once in this repository, to the published
site, and went unnoticed for three commits.

**Cheap surfaces are never cached.** Re-rendering a text episode takes
microseconds and a round trip takes milliseconds, so caching text is a straight
loss — it costs a write, a read, and an opportunity to serve something stale.
Only what is expensive to make goes in: pictures, audio, frames.

:class:`LocalStore` is the same interface over a directory, which is what the
tests use and what a single machine wants. :class:`S3Store` speaks the S3 API
with SigV4 signed by hand, so Cloudflare R2 — or anything else S3-shaped — needs
no client library and the package keeps its promise of no runtime dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from .address import Address
from .surfaces import Content, transcode_example

__all__ = ["Store", "LocalStore", "S3Store", "CachedRenderer", "CACHEABLE",
           "store_from_env"]

#: Surfaces worth a round trip. Text is deliberately absent: regenerating it is
#: cheaper than asking whether it is cached.
CACHEABLE = frozenset({"raster", "video", "audio", "scene"})

_UNSIGNED = "UNSIGNED-PAYLOAD"


class Store(Protocol):
    """The three operations a cache needs."""

    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, data: bytes, *, content_type: str = "") -> None: ...
    def has(self, key: str) -> bool: ...


@dataclass
class LocalStore:
    """A cache in a directory. Keys become nested paths by digest.

    Sharded two characters deep, because a hundred thousand entries in one
    directory is slow on every filesystem that has ever existed.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _path(self, key: str) -> Path:
        d = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        return self.root / d[:2] / d[2:4] / d

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes, *, content_type: str = "") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)                                 # atomic: no torn reads

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def __repr__(self) -> str:
        return f"<LocalStore {self.root}>"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


@dataclass
class S3Store:
    """An S3-compatible bucket, signed by hand.

    Written out rather than pulled in because the package has no runtime
    dependencies and is keeping it that way. SigV4 is a hash of a canonical
    request and four nested HMACs; that is a page of code, and a page of code is
    cheaper than a dependency that would have to be installed everywhere this
    ever runs.

    For Cloudflare R2, ``endpoint`` is
    ``https://<account>.r2.cloudflarestorage.com`` and ``region`` is ``auto``.
    """

    bucket: str
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "auto"
    service: str = "s3"
    prefix: str = "episodes/"
    timeout: float = 20.0

    # ---- signing -----------------------------------------------------
    def _headers(self, method: str, key: str, payload_hash: str,
                 content_type: str = "") -> dict[str, str]:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        day = now.strftime("%Y%m%d")
        host = self.endpoint.split("://", 1)[-1].rstrip("/")
        path = f"/{self.bucket}/{self.prefix}{key}"

        headers = {"host": host, "x-amz-content-sha256": payload_hash,
                   "x-amz-date": stamp}
        if content_type:
            headers["content-type"] = content_type
        signed = ";".join(sorted(headers))
        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
        canonical = "\n".join([method, path, "", canonical_headers, signed, payload_hash])

        scope = f"{day}/{self.region}/{self.service}/aws4_request"
        to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope,
                             hashlib.sha256(canonical.encode()).hexdigest()])
        k = _sign(f"AWS4{self.secret_key}".encode(), day)
        k = _sign(k, self.region)
        k = _sign(k, self.service)
        k = _sign(k, "aws4_request")
        signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()

        headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}")
        return headers

    def _request(self, method: str, key: str, data: bytes | None = None,
                 content_type: str = "") -> bytes | None:
        payload_hash = hashlib.sha256(data).hexdigest() if data is not None \
            else hashlib.sha256(b"").hexdigest()
        headers = self._headers(method, key, payload_hash, content_type)
        url = f"{self.endpoint.rstrip('/')}/{self.bucket}/{self.prefix}{key}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):                   # absent, or absent to us
                return None
            raise

    def get(self, key: str) -> bytes | None:
        return self._request("GET", key)

    def put(self, key: str, data: bytes, *, content_type: str = "") -> None:
        self._request("PUT", key, data, content_type or "application/octet-stream")

    def has(self, key: str) -> bool:
        return self._request("HEAD", key) is not None

    def __repr__(self) -> str:
        return f"<S3Store {self.bucket} at {self.endpoint}>"


def store_from_env(env: dict[str, str] | None = None) -> Store | None:
    """A store built from the environment, or ``None`` if none is configured.

    ``LANGCURRICULUM_CACHE`` names a directory. Otherwise the R2 variables are
    read; missing credentials mean no cache rather than an error, because a cache
    is an optimization and a missing one must never stop a render.
    """
    env = dict(os.environ if env is None else env)
    local = env.get("LANGCURRICULUM_CACHE")
    if local:
        return LocalStore(Path(local))
    account = env.get("R2_ACCOUNT_ID")
    bucket = env.get("R2_BUCKET")
    key = env.get("R2_ACCESS_KEY_ID")
    secret = env.get("R2_SECRET_ACCESS_KEY")
    if not (bucket and key and secret and (account or env.get("R2_ENDPOINT"))):
        return None
    endpoint = env.get("R2_ENDPOINT") or f"https://{account}.r2.cloudflarestorage.com"
    return S3Store(bucket=bucket, endpoint=endpoint, access_key=key,
                   secret_key=secret, region=env.get("R2_REGION", "auto"),
                   prefix=env.get("R2_PREFIX", "episodes/"))


@dataclass
class CachedRenderer:
    """Render an address, going to the store first for the expensive surfaces.

    The cache holds one asset per address — the primary one, which is the whole
    artifact for a picture or a waveform. Frame sequences are cached as their
    APNG for the same reason. Everything else is rebuilt, because rebuilding is
    the cheap operation and this design leans on that being true.
    """

    store: Store | None = None
    hits: int = 0
    misses: int = 0
    skipped: int = 0

    def content(self, address: Address, **options: Any) -> Content:
        surface = address.presentation.surface
        example = address.example()
        if self.store is None or surface not in CACHEABLE:
            self.skipped += 1
            return transcode_example(example, surface, **options)

        key = address.digest()
        cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            from .surfaces import Asset, Fidelity
            mime = "audio/wav" if surface == "audio" else (
                "image/apng" if surface == "video" else "image/png")
            return Content(surface=surface, text=example.prompt,
                           target=example.target,
                           assets=(Asset(mime=mime, data=cached),),
                           fidelity=Fidelity(),
                           meta={"cached": True, "key": key})
        self.misses += 1
        content = transcode_example(example, surface, **options)
        if content.assets:
            primary = next((a for a in content.assets if a.role == "prompt"),
                           content.assets[0])
            self.store.put(key, primary.data, content_type=primary.mime)
        return content

    def warm(self, addresses: Iterable[Address], **options: Any) -> dict[str, int]:
        """Render and cache a batch. Returns the tallies."""
        for address in addresses:
            self.content(address, **options)
        return {"hits": self.hits, "misses": self.misses, "skipped": self.skipped}
