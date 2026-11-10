"""RustFS S3-over-NIXL client: move objects straight between RustFS and registered GPU or host memory.

RustFS serves a GET or PUT that carries `x-rustfs-nixl-agent`,
`x-rustfs-nixl-buffer` (`<vram|dram>:<device>:<addr>:<len>`) and
`x-rustfs-nixl-metadata` by RDMA: a GET writes the object (or its `Range`)
into that buffer, a PUT reads the object body from it. The HTTP body stays
empty either way. Buffers are registered once and reused, so concurrent
transfers into disjoint parts of them share one exported metadata blob.
Requests use per-thread keep-alive connections and SigV4 with
UNSIGNED-PAYLOAD (the payload never crosses HTTP).

On a bucket bound to the `kv-cache` engine, `contains_many` checks many keys
in one `prefix-query` request instead of a HEAD per key, and a write refused
because the owner's drive is full (503 "cache full") is not retried: the
caller skips it. `Placement` mirrors the engine's owner choice (xxh3-64
rendezvous over every node drive) so requests go straight to the owner
instead of being forwarded; a stale view only costs that forwarding hop.
"""
import base64
import collections
import datetime
import hashlib
import hmac
import http.client
import json
import os
import random
import threading
import time
import urllib.parse
import uuid

import xxhash
from nixl._api import nixl_agent, nixl_agent_config

UNSIGNED = "UNSIGNED-PAYLOAD"
RETRY_ATTEMPTS = int(os.environ.get("RUSTFS_NIXL_RETRY_ATTEMPTS", "10"))
# kv-cache engine limits: keys per query and request body bytes.
ENGINE_MAX_KEYS = 4096
ENGINE_MAX_BODY = 1 << 20


class Placement:
    """Client copy of a kv-cache bucket's placement view (`placement-view`)."""

    HASH = "xxh3-64-rendezvous-v1"
    _TARGET_SALT = xxhash.xxh3_64_intdigest(b"rustfs.bucket-engine.target")

    def __init__(self, view, endpoints):
        if view.get("hash") != self.HASH:
            raise RustfsError(f"unsupported placement hash {view.get('hash')!r}")
        self.version = view["version"]
        by_netloc = {urllib.parse.urlsplit(e).netloc: e.rstrip("/") for e in endpoints}
        # (seed, node, drive, endpoint) in node then drive order, like PlacementView::targets.
        self.targets = [
            (xxhash.xxh3_64_intdigest(n["node"].encode(), d ^ self._TARGET_SALT), n["node"], d, by_netloc.get(n["node"]))
            for n in view["nodes"]
            for d in range(n["drives"])
        ]

    def owner_endpoint(self, key: str):
        """Endpoint of the node owning `key`, or None if the client has no endpoint for it."""
        data = key.encode()
        best = None
        for seed, node, drive, endpoint in self.targets:
            rank = (-xxhash.xxh3_64_intdigest(data, seed), node, drive)
            if best is None or rank < best[0]:
                best = (rank, endpoint)
        return best[1] if best else None


def _cache_full(status, body: bytes) -> bool:
    return status == 503 and b"cache full" in body


def _retryable(status, body: bytes) -> bool:
    """Back-pressure answers worth retrying: SlowDown/503, a momentarily exhausted staging pool,
    or a request that timed out waiting for a disk permit."""
    if _cache_full(status, body):
        return False
    return status == 503 or (status == 500 and (b"exhausted" in body or b"waiting for disk permit" in body))


class RustfsError(IOError):
    def __init__(self, message, status=None, cache_full=False):
        super().__init__(message)
        self.status = status
        self.cache_full = cache_full


class RustfsNixlClient:
    def __init__(self, endpoints, access_key, secret_key, region="us-east-1", devices="", agent_name=None):
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.access_key, self.secret_key, self.region = access_key, secret_key, region
        self.name = agent_name or f"rustfs-nixl-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.agent = nixl_agent(self.name, nixl_agent_config(enable_prog_thread=True, backends=[]))
        params = {"device_list": devices} if devices else {}
        self.agent.create_backend("UCX", params)
        self._lock = threading.Lock()
        self._regions = []  # (mem, device, addr, size, handle)
        self._metadata = None
        self._next = 0
        self._local = threading.local()
        self._signing_key = (None, None)
        self._retry_lock = threading.Lock()
        self._retries = collections.Counter()  # method -> back-off retries taken

    def endpoint(self, hint=None):
        if hint is not None:
            return self.endpoints[hint % len(self.endpoints)]
        with self._lock:
            ep = self.endpoints[self._next % len(self.endpoints)]
            self._next += 1
        return ep

    # ---- registration --------------------------------------------------
    def register(self, addr, size, mem="VRAM", device=0):
        """Registers `size` bytes at `addr` (GPU memory on `device`, or host memory) for RDMA."""
        mem = mem.upper()
        with self._lock:
            handle = self.agent.register_memory([(addr, size, device, "")], mem)
            self._regions.append((mem, device, addr, size, handle))
            # Export after registering so servers see every buffer.
            self._metadata = base64.b64encode(self.agent.get_agent_metadata()).decode()

    def is_registered(self, addr, length):
        return any(base <= addr and addr + length <= base + size for _, _, base, size, _ in self._regions)

    def deregister_all(self):
        with self._lock:
            for *_, handle in self._regions:
                self.agent.deregister_memory(handle)
            self._regions.clear()
            self._metadata = None

    def _region_of(self, addr, length):
        for mem, device, base, size, _ in self._regions:
            if base <= addr and addr + length <= base + size:
                return mem, device
        raise ValueError(f"buffer {addr:#x}+{length} is not inside a registered region")

    # ---- HTTP + SigV4 --------------------------------------------------
    def _key(self, date):
        cached_date, key = self._signing_key
        if cached_date != date:
            k = hmac.new(("AWS4" + self.secret_key).encode(), date.encode(), hashlib.sha256).digest()
            for part in (self.region, "s3", "aws4_request"):
                k = hmac.new(k, part.encode(), hashlib.sha256).digest()
            self._signing_key = (date, k)
            key = k
        return key

    def _sign(self, method, host, path, query, headers):
        now = datetime.datetime.now(datetime.timezone.utc)
        amz_date, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        headers["x-amz-date"] = amz_date
        headers["x-amz-content-sha256"] = UNSIGNED
        canonical = (
            f"{method}\n{path}\n{query}\nhost:{host}\nx-amz-content-sha256:{UNSIGNED}\nx-amz-date:{amz_date}\n\n"
            f"host;x-amz-content-sha256;x-amz-date\n{UNSIGNED}"
        )
        scope = f"{date}/{self.region}/s3/aws4_request"
        to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
        signature = hmac.new(self._key(date), to_sign.encode(), hashlib.sha256).hexdigest()
        headers["Authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
            f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={signature}"
        )

    def _conn(self, endpoint, fresh=False):
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        conn = conns.get(endpoint)
        if conn is None or fresh:
            if conn is not None:
                conn.close()
            netloc = urllib.parse.urlsplit(endpoint).netloc
            conn = conns[endpoint] = http.client.HTTPConnection(netloc, timeout=1800)
        return conn

    def request(self, method, bucket, key, headers=None, endpoint=None, query="", body=b""):
        """Signed request; returns (status, headers, body). Retries once on a dropped keep-alive connection.

        `query` must already be canonical (sorted, encoded); `key` "" addresses the bucket."""
        endpoint = endpoint or self.endpoint()
        path = "/" + urllib.parse.quote(f"{bucket}/{key}" if key else bucket, safe="/-_.~")
        host = urllib.parse.urlsplit(endpoint).netloc
        target = f"{path}?{query}" if query else path
        for attempt in (0, 1):
            h = dict(headers or {})
            self._sign(method, host, path, query, h)
            h["Content-Length"] = str(len(body))
            conn = self._conn(endpoint, fresh=attempt > 0)
            try:
                conn.request(method, target, body=body or None, headers=h)
                r = conn.getresponse()
                body = r.read()
                return r.status, r.headers, body
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, ConnectionError, BrokenPipeError):
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def get_bytes(self, bucket, key, start=None, length=None, endpoint=None):
        """Plain HTTP GET (small objects and headers), optionally a byte range."""
        headers = {} if start is None else {"Range": f"bytes={start}-{start + length - 1}"}
        status, _, body = self.request("GET", bucket, key, headers, endpoint)
        if status not in (200, 206):
            raise RustfsError(f"{bucket}/{key}: GET {status}: {body[:300]!r}", status)
        return body

    def head_size(self, bucket, key, endpoint=None):
        """Object size, or None if it does not exist."""
        status, headers, _ = self._retrying_request("HEAD", bucket, key, None, endpoint)
        if status == 404:
            return None
        if status != 200:
            raise RustfsError(f"{bucket}/{key}: HEAD {status}", status)
        return int(headers["Content-Length"])

    # ---- RDMA transfers ------------------------------------------------
    def _nixl_headers(self, addr, length):
        mem, device = self._region_of(addr, length)
        return {
            "x-rustfs-nixl-agent": self.name,
            "x-rustfs-nixl-buffer": f"{mem.lower()}:{device}:{addr:#x}:{length}",
            "x-rustfs-nixl-metadata": self._metadata,
        }

    def _retrying_request(self, method, bucket, key, headers, endpoint, query="", body=b""):
        """Send a request, retrying back-pressure answers with jittered exponential backoff."""
        for attempt in range(RETRY_ATTEMPTS):
            status, h, reply = self.request(method, bucket, key, headers, endpoint, query, body)
            if not _retryable(status, reply) or attempt == RETRY_ATTEMPTS - 1:
                return status, h, reply
            with self._retry_lock:
                self._retries[method] += 1
            time.sleep(min(4.0, 0.05 * 2 ** attempt) * (0.5 + random.random()))
        return status, h, reply

    def take_retries(self):
        """Retries taken per method since the previous call."""
        with self._retry_lock:
            taken, self._retries = self._retries, collections.Counter()
        return taken

    def fetch_into(self, bucket, key, addr, length, start=0, endpoint=None):
        """RDMA-fetch bytes [start, start+length) of `bucket/key` to `addr`; returns seconds taken.

        Raises RustfsError with status 404 if the object does not exist."""
        started = time.time()
        headers = self._nixl_headers(addr, length)
        headers["Range"] = f"bytes={start}-{start + length - 1}"
        status, h, body = self._retrying_request("GET", bucket, key, headers, endpoint)
        if status not in (200, 206):
            raise RustfsError(f"{bucket}/{key}: GET {status}: {body[:300]!r}", status)
        reply, moved = h.get("x-amz-rdma-reply"), int(h.get("x-amz-rdma-bytes-transferred") or -1)
        if reply is None or moved != length:
            raise RustfsError(f"{bucket}/{key}: not served over RDMA (reply {reply!r}, moved {moved} of {length})")
        return time.time() - started

    def store_from(self, bucket, key, addr, length, endpoint=None):
        """RDMA-PUT `length` bytes at `addr` as `bucket/key`; returns seconds taken."""
        started = time.time()
        status, h, body = self._retrying_request("PUT", bucket, key, self._nixl_headers(addr, length), endpoint)
        reply, moved = h.get("x-amz-rdma-reply"), int(h.get("x-amz-rdma-bytes-transferred") or -1)
        if status != 200 or reply != "200" or moved != length:
            raise RustfsError(
                f"{bucket}/{key}: PUT {status} (rdma reply {reply!r}, moved {moved} of {length}): {body[:300]!r}",
                status, cache_full=_cache_full(status, body),
            )
        return time.time() - started

    # ---- kv-cache engine -----------------------------------------------
    def engine_op(self, bucket, operation, payload: bytes, endpoint=None):
        """POST a kv-cache engine custom operation; returns the reply body."""
        status, _, body = self._retrying_request(
            "POST", bucket, "", None, endpoint, f"x-rustfs-engine-op={operation}", payload
        )
        if status != 200:
            raise RustfsError(f"{bucket}: {operation} {status}: {body[:300]!r}", status)
        return body

    def placement(self, bucket, endpoint=None):
        """The bucket's current placement view."""
        return Placement(json.loads(self.engine_op(bucket, "placement-view", b"{}", endpoint)), self.endpoints)

    def contains_many(self, bucket, keys, endpoint=None):
        """Whether each key is stored, in order, from `prefix-query` requests of at most
        ENGINE_MAX_KEYS keys and ENGINE_MAX_BODY bytes each."""
        present = []
        start = 0
        while start < len(keys):
            end, size = start, 16
            while end < len(keys) and end - start < ENGINE_MAX_KEYS and size + len(keys[end]) + 8 < ENGINE_MAX_BODY:
                size += len(keys[end]) + 8
                end += 1
            reply = json.loads(self.engine_op(bucket, "prefix-query", json.dumps({"keys": keys[start:end]}).encode(), endpoint))
            if len(reply.get("present", ())) != end - start:
                raise RustfsError(f"{bucket}: prefix-query answered {len(reply.get('present', ()))} of {end - start} keys")
            present.extend(reply["present"])
            start = end
        return present
