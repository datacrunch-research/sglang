"""RustFS S3-over-NIXL client: GET objects straight into registered GPU or host memory.

RustFS answers a GET that carries `x-rustfs-nixl-agent`, `x-rustfs-nixl-buffer`
(`<vram|dram>:<device>:<addr>:<len>`) and `x-rustfs-nixl-metadata` by
RDMA-writing the object, or the requested `Range`, into that buffer; the HTTP
body stays empty. Buffers are registered once and reused, so concurrent
fetches into disjoint parts of them share one exported agent metadata blob.
"""
import base64
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from nixl._api import nixl_agent, nixl_agent_config


class RustfsNixlClient:
    def __init__(self, endpoints, access_key, secret_key, region="us-east-1", devices="", agent_name=None):
        self.endpoints = list(endpoints)
        self.creds = Credentials(access_key, secret_key)
        self.region = region
        self.name = agent_name or f"sglang-loader-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.agent = nixl_agent(self.name, nixl_agent_config(enable_prog_thread=True, backends=[]))
        params = {"device_list": devices} if devices else {}
        self.agent.create_backend("UCX", params)
        self._lock = threading.Lock()
        self._regions = []  # (mem, device, addr, size, handle)
        self._metadata = None
        self._next = 0

    def endpoint(self, hint=None):
        if hint is not None:
            return self.endpoints[hint % len(self.endpoints)]
        with self._lock:
            ep = self.endpoints[self._next % len(self.endpoints)]
            self._next += 1
        return ep

    # ---- registration --------------------------------------------------
    def register(self, addr, size, mem="VRAM", device=0):
        """Registers `size` bytes at `addr` (GPU memory on `device`, or host memory) for RDMA writes."""
        mem = mem.upper()
        with self._lock:
            handle = self.agent.register_memory([(addr, size, device, "")], mem)
            self._regions.append((mem, device, addr, size, handle))
            # Export after registering so servers see every buffer.
            self._metadata = base64.b64encode(self.agent.get_agent_metadata()).decode()

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

    # ---- requests ------------------------------------------------------
    def _signed(self, method, url, headers=None):
        req = AWSRequest(method=method, url=url, headers=dict(headers or {}))
        S3SigV4Auth(self.creds, "s3", self.region).add_auth(req)
        return dict(req.headers)

    def get_bytes(self, bucket, key, start=None, length=None, endpoint=None):
        """Plain HTTP GET (small objects and headers), optionally a byte range."""
        url = f"{endpoint or self.endpoint()}/{bucket}/{key}"
        headers = {} if start is None else {"Range": f"bytes={start}-{start + length - 1}"}
        req = urllib.request.Request(url, headers=self._signed("GET", url, headers))
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read()

    def head_size(self, bucket, key, endpoint=None):
        url = f"{endpoint or self.endpoint()}/{bucket}/{key}"
        req = urllib.request.Request(url, method="HEAD", headers=self._signed("HEAD", url))
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])

    def fetch_into(self, bucket, key, addr, length, start=0, endpoint=None):
        """RDMA-fetch bytes [start, start+length) of `bucket/key` to `addr`; returns seconds taken."""
        mem, device = self._region_of(addr, length)
        started = time.time()
        url = f"{endpoint or self.endpoint()}/{bucket}/{key}"
        headers = {
            "Range": f"bytes={start}-{start + length - 1}",
            "x-rustfs-nixl-agent": self.name,
            "x-rustfs-nixl-buffer": f"{mem.lower()}:{device}:{addr:#x}:{length}",
            "x-rustfs-nixl-metadata": self._metadata,
        }
        req = urllib.request.Request(url, headers=self._signed("GET", url, headers))
        try:
            r = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            raise IOError(f"{bucket}/{key}: GET {e.code}: {e.read()[:300]!r}") from None
        with r:
            r.read()
            reply = r.headers.get("x-amz-rdma-reply")
            moved = int(r.headers.get("x-amz-rdma-bytes-transferred") or -1)
        if reply is None or moved != length:
            raise IOError(f"{bucket}/{key}: not served over RDMA (reply {reply!r}, moved {moved} of {length})")
        return time.time() - started
