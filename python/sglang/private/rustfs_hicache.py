"""SGLang HiCache L3 storage backend on RustFS, moving pages over NIXL RDMA.

Enable with:

  --enable-hierarchical-cache --hicache-mem-layout page_first
  --hicache-storage-backend dynamic
  --hicache-storage-backend-extra-config '{"backend_name": "rustfs",
      "module_path": "sglang.private.rustfs_hicache", "class_name": "RustfsHiCacheStorage",
      "interface_v1": 1, "hicache_storage_pass_prefix_keys": true,
      "endpoints": "http://10.20.9.28:9000,...", "bucket": "kvcache",
      "rails": "mlx5_0,mlx5_0,mlx5_7,mlx5_7,mlx5_10,mlx5_10,mlx5_13,mlx5_13"}'

Objects (each written and read by RDMA straight from/to pinned host memory):

* KV: every full group of `group_pages` pages at an aligned absolute position
  (from `prefix_keys`) is one object `<prefix>/<last page hash><suffix>.g<N>`
  holding the pages back to back; page hashes chain from the sequence start,
  so the last hash names the whole prefix. Pages outside a full group in a
  write call are single-page objects `<prefix>/<page hash><suffix>`. Reads use
  ranged GETs, so a prefetch batch can take part of a group.
* Sidecar pools (Mamba state): one object per page snapshot
  `<prefix>/<page hash>.<pool><suffix>` holding the page's buffers back to back.

Writes are write-behind by default: a backup call copies each object into its
own slot of a registered pinned write-staging area, starts the PUT on a
background pool and returns, so the backup thread never waits on RustFS; when
every slot is in flight, writers wait. An object whose PUT is still running
counts as existing, and a GET for it first waits for that PUT. A failed
background PUT only turns into a later miss. Reads go straight into host pages
when they are contiguous, else through a separate read-staging area. Existence
checks consult a local cache of objects known to exist; the rest are one
`prefix-query` request per batch on a bucket bound to the `kv-cache` engine
(detected at start, or forced with batch_exists), else concurrent HEADs. A
write the engine refuses because its drive is full is skipped, not retried.
On such a bucket every request goes to the node owning its key, from the
bucket's placement view (refreshed every `placement_refresh_s`, 60), and
queries are split by owner.

Extra config keys: endpoints (required), bucket ("kvcache"), prefix ("hicache"),
access_key / secret_key (or RUSTFS_ACCESS_KEY / RUSTFS_SECRET_KEY), region,
rails (UCX device per local GPU index), group_pages (32), io_threads (32),
staging_slots (16, read staging), write_behind (true), write_slots (64),
write_threads (32), get_split_mb (8: ranged GETs per object read, issued in
parallel), mamba_probe_window (32: trailing-state existence is probed from the
end of the prefix in windows of this many pages), batch_exists ("auto": use
`prefix-query` when the bucket answers it; true or false to force).
"""

from __future__ import annotations

import collections
import concurrent.futures
import ctypes
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_BUFFER_ATTRS = ("kv_buffer", "k_buffer", "v_buffer", "index_k_buffer", "index_k_scale_buffer", "temporal_buffer")


def _split(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


class _Lru:
    """Bounded map (also used as a set) shared by I/O threads."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def put(self, key, value=True):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def discard(self, key):
        with self._lock:
            self._data.pop(key, None)

    def clear(self):
        with self._lock:
            self._data.clear()


class _Staging:
    """Fixed slots of registered pinned host memory for gathering scattered pages."""

    def __init__(self, client, slot_bytes: int, slots: int):
        self.slot_bytes = slot_bytes
        self.tensor = torch.empty(slot_bytes * slots, dtype=torch.uint8, pin_memory=True)
        client.register(self.tensor.data_ptr(), slot_bytes * slots, "DRAM", 0)
        self._free: queue.Queue = queue.Queue()
        for i in range(slots):
            self._free.put(i)

    def acquire(self) -> Tuple[int, int]:
        slot = self._free.get()
        return slot, self.tensor.data_ptr() + slot * self.slot_bytes

    def release(self, slot: int):
        self._free.put(slot)


class RustfsHiCacheStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        from sglang.private.rustfs_nixl import RustfsNixlClient

        if storage_config.unified_suffixes is not None:
            raise NotImplementedError("RustFS HiCache storage does not support the unified key scheme.")
        extra = dict(storage_config.extra_config or {})
        endpoints = _split(extra.get("endpoints") or os.environ.get("RUSTFS_HICACHE_ENDPOINTS"))
        if not endpoints:
            raise ValueError("RustFS HiCache storage needs 'endpoints' in --hicache-storage-backend-extra-config")
        self.bucket = extra.get("bucket", "kvcache")
        self.prefix = str(extra.get("prefix", "hicache")).strip("/")
        # Pages per KV object. A group forms only when all its pages arrive in one
        # backup call, i.e. within one prefill chunk: with DCP a storage page is
        # page_size * dcp_size tokens, so 32 pages x 64 x 8 = one 16384-token chunk.
        self.group_pages = int(extra.get("group_pages", os.environ.get("RUSTFS_HICACHE_GROUP_PAGES", 32)))
        if self.group_pages < 1:
            raise ValueError(f"group_pages must be >= 1, got {self.group_pages}")
        self.io_threads = int(extra.get("io_threads", 32))
        self.staging_slots = int(extra.get("staging_slots", 16))
        self.write_behind = str(extra.get("write_behind", True)).lower() not in ("0", "false", "no")
        self.write_slots = int(extra.get("write_slots", 64))
        self.write_threads = int(extra.get("write_threads", 32))
        self.get_split = int(extra.get("get_split_mb", 8)) << 20
        self.probe_window = max(1, int(extra.get("mamba_probe_window", 32)))
        batch_exists = str(extra.get("batch_exists", "auto")).lower()
        self.placement_refresh = float(extra.get("placement_refresh_s", 60))
        self._placement = None
        self._placement_at = 0.0
        self._placement_lock = threading.Lock()
        rails = _split(extra.get("rails"))
        device = torch.cuda.current_device() if torch.cuda.is_available() else storage_config.tp_rank
        self.storage_config = storage_config
        self.page_size_kv = (
            storage_config.logical_page_size if storage_config.dcp_size > 1 else None
        )  # resolved from the host pool when DCP is off
        self.client = RustfsNixlClient(
            endpoints,
            extra.get("access_key") or os.environ.get("RUSTFS_ACCESS_KEY", "rustfsadmin"),
            extra.get("secret_key") or os.environ.get("RUSTFS_SECRET_KEY", "rustfsadmin"),
            region=extra.get("region", "us-east-1"),
            devices=rails[device % len(rails)] if rails else "",
            agent_name=f"hicache-{os.uname().nodename}-{os.getpid()}-tp{storage_config.tp_rank}",
        )
        self.debug = os.environ.get("RUSTFS_HICACHE_DEBUG", "0") == "1"
        self._io = concurrent.futures.ThreadPoolExecutor(self.io_threads, thread_name_prefix="rustfs-hicache")
        # Write-side copies and background PUTs get their own pools so a write
        # burst waiting for staging slots never delays prefetch reads.
        self._wio = concurrent.futures.ThreadPoolExecutor(self.io_threads, thread_name_prefix="rustfs-hicache-wcopy")
        self._wb = concurrent.futures.ThreadPoolExecutor(self.write_threads, thread_name_prefix="rustfs-hicache-wb")
        # Sub-range GETs of one object read; separate from _io, whose tasks wait on them.
        self._sub = concurrent.futures.ThreadPoolExecutor(max(16, 2 * self.io_threads), thread_name_prefix="rustfs-hicache-sub")
        self._pending: Dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()
        self._wstaging: Optional[_Staging] = None
        self._init_suffix(storage_config)
        self._known = _Lru(1 << 20)       # object names known to exist
        self._page_loc = _Lru(1 << 21)    # KV page hash -> (group object, index in group)
        self._staging: Optional[_Staging] = None
        self._staging_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = collections.Counter()
        self._last_report = time.time()
        self.use_query = self._detect_query(batch_exists)
        logger.info(
            "[rustfs-hicache] tp%d: bucket=%s prefix=%s group_pages=%d endpoints=%d rail=%s suffix=%s prefix_query=%s",
            storage_config.tp_rank, self.bucket, self.prefix, self.group_pages, len(endpoints),
            rails[device % len(rails)] if rails else "all", self.config_suffix, self.use_query,
        )

    def _detect_query(self, mode: str) -> bool:
        """Whether existence checks use the kv-cache engine's `prefix-query`."""
        if mode in ("0", "false", "no"):
            return False
        try:
            self.client.contains_many(self.bucket, [f"{self.prefix}/.probe"])
            self._refresh_placement()
            return True
        except Exception as e:
            if mode != "auto":
                raise ValueError(f"batch_exists is on but {self.bucket} does not answer prefix-query: {e}") from e
            logger.info("[rustfs-hicache] %s does not answer prefix-query (%s); using HEADs", self.bucket, e)
            return False

    # ---- keys ------------------------------------------------------------
    def _init_suffix(self, cfg: HiCacheStorageConfig) -> None:
        """Same key families as HiCacheFile: shared MLA KV vs per-rank sharded state."""
        model = "-".join(cfg.model_name.split("/")) if cfg.model_name else ""
        shard = ""
        if cfg.pp_size > 1:
            shard += f"_{cfg.pp_size}_{cfg.pp_rank}"
        if cfg.attn_cp_size > 1:
            shard += f"_cp{cfg.attn_cp_rank}_{cfg.attn_cp_size}"
        self.tp_config_suffix = f"_{model}_{cfg.tp_rank}_{cfg.tp_size}{shard}"
        self.config_suffix = f"_{model}{shard}" if cfg.is_rank_replicated else self.tp_config_suffix
        if cfg.dcp_size > 1:
            dcp = f"_tp{cfg.tp_size}_dcp{cfg.dcp_rank}_{cfg.dcp_size}_page{cfg.logical_page_size}"
            self.config_suffix += dcp
            self.tp_config_suffix += dcp

    def _page_obj(self, key: str) -> str:
        return f"{self.prefix}/{key}{self.config_suffix}"

    def _group_obj(self, last_key: str) -> str:
        return f"{self.prefix}/{last_key}{self.config_suffix}.g{self.group_pages}"

    def _sidecar_obj(self, name, key: str) -> str:
        suffix = self.tp_config_suffix if name == PoolName.MAMBA else self.config_suffix
        return f"{self.prefix}/{key}.{name}{suffix}"

    # ---- pools and memory -----------------------------------------------
    def _register_pool(self, pool) -> None:
        tensors = [getattr(pool, a, None) for a in _BUFFER_ATTRS]
        tensors += list(getattr(pool, "conv_buffer", None) or [])
        if hasattr(pool, "get_hybrid_pool_buffer"):
            try:
                tensors += list(pool.get_hybrid_pool_buffer())
            except Exception:  # not every pool implements it fully
                pass
        for t in tensors:
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            addr, nbytes = t.data_ptr(), t.numel() * t.element_size()
            if not self.client.is_registered(addr, nbytes):
                self.client.register(addr, nbytes, "DRAM", 0)

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        if self.page_size_kv is None:
            self.page_size_kv = mem_pool_host.page_size
        self._register_pool(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._register_pool(host_pool)

    def on_host_pools_registered(self) -> None:
        self._ensure_staging()

    def _pool_page_size(self, name, pool) -> int:
        if name == PoolName.KV and self.page_size_kv:
            return self.page_size_kv
        return getattr(pool, "page_size", 1) or 1

    def _page_bytes(self, name, pool) -> int:
        """Bytes of one page (all of its buffers) in `pool`."""
        ps = self._pool_page_size(name, pool)
        _, sizes = pool.get_page_buffer_meta(torch.arange(ps, dtype=torch.int64))
        return sum(sizes)

    def _ensure_staging(self) -> None:
        with self._staging_lock:
            if self._staging is not None:
                return
            slot = 0
            pools = dict(getattr(self, "registered_pools", {}) or {})
            if getattr(self, "mem_pool_host", None) is not None:
                pools.setdefault(PoolName.KV, self.mem_pool_host)
            for name, pool in pools.items():
                try:
                    per_page = self._page_bytes(name, pool)
                except Exception as e:
                    logger.warning("[rustfs-hicache] no zero-copy page meta for pool %s: %s", name, e)
                    continue
                slot = max(slot, per_page * (self.group_pages if name == PoolName.KV else 1))
            if slot:
                self._staging = _Staging(self.client, slot, self.staging_slots)
                if self.write_behind:
                    self._wstaging = _Staging(self.client, slot, self.write_slots)
                logger.info(
                    "[rustfs-hicache] tp%d: read staging %d x %.1f MB, write-behind %s (%d slots, %d PUT threads)",
                    self.storage_config.tp_rank, self.staging_slots, slot / 1e6,
                    "on" if self.write_behind else "off", self.write_slots if self.write_behind else 0,
                    self.write_threads,
                )

    def _kv_pool(self):
        return getattr(self, "registered_pools", {}).get(PoolName.KV) or self.mem_pool_host

    # ---- transfers ---------------------------------------------------------
    @staticmethod
    def _contiguous(ptrs: List[int], sizes: List[int]) -> bool:
        return all(ptrs[i] + sizes[i] == ptrs[i + 1] for i in range(len(ptrs) - 1))

    def _put_pieces(self, obj: str, ptrs: List[int], sizes: List[int]) -> bool:
        """PUT the concatenation of (ptr, size) pieces as one object."""
        if self.write_behind:
            return self._put_behind(obj, ptrs, sizes)
        total = sum(sizes)
        try:
            if self._contiguous(ptrs, sizes):
                seconds = self.client.store_from(self.bucket, obj, ptrs[0], total, self._owner(obj))
            else:
                self._ensure_staging()
                slot, base = self._staging.acquire()
                try:
                    off = 0
                    for p, n in zip(ptrs, sizes):
                        ctypes.memmove(base + off, p, n)
                        off += n
                    seconds = self.client.store_from(self.bucket, obj, base, total, self._owner(obj))
                finally:
                    self._staging.release(slot)
            self._known.put(obj)
            self._count(put_objects=1, put_bytes=total, put_seconds=seconds)
            return True
        except Exception as e:
            self._put_failed(obj, e)
            return False

    def _put_behind(self, obj: str, ptrs: List[int], sizes: List[int]) -> bool:
        """Copy the pieces into a write-staging slot and PUT them in the background."""
        with self._pending_lock:
            if obj in self._pending or self._known.get(obj):
                return True
            done = self._pending[obj] = threading.Event()
        slot = None
        try:
            self._ensure_staging()
            total = sum(sizes)
            slot, base = self._wstaging.acquire()  # back-pressure when every slot is in flight
            off = 0
            for p, n in zip(ptrs, sizes):
                ctypes.memmove(base + off, p, n)
                off += n
        except Exception as e:
            if slot is not None:
                self._wstaging.release(slot)
            with self._pending_lock:
                self._pending.pop(obj, None)
            done.set()
            self._count(put_errors=1)
            logger.warning("[rustfs-hicache] staging %s for write-behind failed: %s", obj, e)
            return False

        def flush():
            try:
                seconds = self.client.store_from(self.bucket, obj, base, total, self._owner(obj))
                self._known.put(obj)
                self._count(put_objects=1, put_bytes=total, put_seconds=seconds)
            except Exception as e:
                self._put_failed(obj, e, "background ")
            finally:
                self._wstaging.release(slot)
                with self._pending_lock:
                    self._pending.pop(obj, None)
                done.set()

        self._wb.submit(flush)
        return True

    def _refresh_placement(self) -> None:
        try:
            placement = self.client.placement(self.bucket)
        except Exception as e:
            logger.warning("[rustfs-hicache] placement view refresh failed, keeping the old one: %s", e)
            placement = self._placement
        self._placement, self._placement_at = placement, time.time()

    def _owner(self, obj: str) -> Optional[str]:
        """Endpoint of the node owning `obj`; None (any node) without a placement view."""
        if not self.use_query:
            return None
        if time.time() - self._placement_at > self.placement_refresh and self._placement_lock.acquire(blocking=False):
            try:
                self._refresh_placement()
            finally:
                self._placement_lock.release()
        placement = self._placement
        return placement.owner_endpoint(obj) if placement is not None else None

    def _put_failed(self, obj: str, e: Exception, kind: str = "") -> None:
        if getattr(e, "cache_full", False):
            self._count(put_full=1)  # the owner's drive is full: skip the entry
            return
        self._count(put_errors=1)
        logger.warning("[rustfs-hicache] %sPUT %s failed: %s", kind, obj, e)

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until every background PUT has finished; False on timeout."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._pending_lock:
                events = list(self._pending.values())
            if not events:
                return True
            for done in events:
                left = None if deadline is None else max(0.0, deadline - time.time())
                if not done.wait(left):
                    return False

    def _wait_pending(self, obj: str) -> None:
        with self._pending_lock:
            done = self._pending.get(obj)
        if done is not None and not done.wait(300):
            logger.warning("[rustfs-hicache] still waiting on background PUT of %s", obj)

    def _fetch_range(self, obj: str, addr: int, total: int, start: int) -> float:
        """GET bytes [start, start+total) of `obj` to `addr`, split into parallel ranged GETs."""
        owner = self._owner(obj)
        if total <= self.get_split:
            return self.client.fetch_into(self.bucket, obj, addr, total, start, owner)
        parts = [(o, min(self.get_split, total - o)) for o in range(0, total, self.get_split)]
        t = time.time()
        list(self._sub.map(lambda p: self.client.fetch_into(self.bucket, obj, addr + p[0], p[1], start + p[0], owner), parts))
        return time.time() - t

    def _get_pieces(self, obj: str, start: int, ptrs: List[int], sizes: List[int]) -> bool:
        """GET bytes [start, start+sum(sizes)) of `obj` scattered into (ptr, size) pieces."""
        self._wait_pending(obj)
        total = sum(sizes)
        try:
            if self._contiguous(ptrs, sizes):
                seconds = self._fetch_range(obj, ptrs[0], total, start)
            else:
                self._ensure_staging()
                slot, base = self._staging.acquire()
                try:
                    seconds = self._fetch_range(obj, base, total, start)
                    off = 0
                    for p, n in zip(ptrs, sizes):
                        ctypes.memmove(p, base + off, n)
                        off += n
                finally:
                    self._staging.release(slot)
            self._known.put(obj)
            self._count(get_objects=1, get_bytes=total, get_seconds=seconds)
            return True
        except Exception as e:
            if getattr(e, "status", None) in (404, 416):
                self._known.discard(obj)
                self._count(get_misses=1)
            else:
                self._count(get_errors=1)
                logger.warning("[rustfs-hicache] GET %s failed: %s", obj, e)
            return False

    def _exists(self, obj: str) -> bool:
        if self._known.get(obj):
            return True
        with self._pending_lock:
            if obj in self._pending:
                return True
        started = time.time()
        try:
            found = self.client.head_size(self.bucket, obj) is not None
        except Exception as e:
            logger.warning("[rustfs-hicache] HEAD %s failed: %s", obj, e)
            self._count(head_errors=1)
            found = False
        self._count(heads=1, head_seconds=time.time() - started)
        if found:
            self._known.put(obj)
        return found

    def _exists_many(self, objs: List[str]) -> Dict[str, bool]:
        unique = list(dict.fromkeys(objs))
        if not self.use_query:
            return dict(zip(unique, self._io.map(self._exists, unique)))
        found = {}
        ask = []
        with self._pending_lock:
            for obj in unique:
                if self._known.get(obj) or obj in self._pending:
                    found[obj] = True
                else:
                    ask.append(obj)
        if not ask:
            return found
        started = time.time()
        by_owner: Dict[Optional[str], List[str]] = collections.defaultdict(list)
        for obj in ask:
            by_owner[self._owner(obj)].append(obj)
        groups = list(by_owner.items())
        try:
            answers = list(self._io.map(lambda g: self.client.contains_many(self.bucket, g[1], g[0]), groups))
        except Exception as e:
            logger.warning("[rustfs-hicache] prefix-query of %d keys failed, falling back to HEADs: %s", len(ask), e)
            self._count(query_errors=1)
            found.update(zip(ask, self._io.map(self._exists, ask)))
            return found
        self._count(queries=1, query_keys=len(ask), query_seconds=time.time() - started)
        for (_, objs), present in zip(groups, answers):
            for obj, hit in zip(objs, present):
                if hit:
                    self._known.put(obj)
                found[obj] = hit
        return found

    def _count(self, **values):
        with self._stats_lock:
            self._stats.update(values)
            now = time.time()
            if now - self._last_report < 30:
                return
            s, self._stats, elapsed = self._stats, collections.Counter(), now - self._last_report
            self._last_report = now
        retries = self.client.take_retries()
        logger.info(
            "[rustfs-hicache] tp%d last %.0fs: put %d obj %.2f GB (%.2f GB/s, %.1f ms avg), "
            "get %d obj %.2f GB (%.2f GB/s, %.1f ms avg), heads %d (%.1f ms avg), misses %d, "
            "retries put %d get %d head %d, failed put %d get %d head %d, write-behind pending %d, "
            "queries %d (%d keys, %.1f ms avg, failed %d), skipped full %d",
            self.storage_config.tp_rank, elapsed,
            s["put_objects"], s["put_bytes"] / 1e9, s["put_bytes"] / 1e9 / elapsed,
            1e3 * s["put_seconds"] / max(1, s["put_objects"]),
            s["get_objects"], s["get_bytes"] / 1e9, s["get_bytes"] / 1e9 / elapsed,
            1e3 * s["get_seconds"] / max(1, s["get_objects"]),
            s["heads"], 1e3 * s["head_seconds"] / max(1, s["heads"]), s["get_misses"], retries["PUT"], retries["GET"], retries["HEAD"],
            s["put_errors"], s["get_errors"], s["head_errors"], len(self._pending),
            s["queries"], s["query_keys"], 1e3 * s["query_seconds"] / max(1, s["queries"]), s["query_errors"], s["put_full"],
        )

    # ---- KV (v1 zero-copy path) ------------------------------------------
    def _abs_start(self, extra_info: Optional[HiCacheStorageExtraInfo]) -> Optional[int]:
        if extra_info is None or extra_info.prefix_keys is None:
            return None
        return len(extra_info.prefix_keys)

    def _kv_meta(self, keys: List[str], host_indices: torch.Tensor):
        pool = self._kv_pool()
        ptrs, sizes = pool.get_page_buffer_meta(host_indices)
        m = len(ptrs) // len(keys)
        return [(ptrs[i * m:(i + 1) * m], sizes[i * m:(i + 1) * m]) for i in range(len(keys))]

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        if not keys:
            return []
        pages = self._kv_meta(keys, host_indices)
        p0 = self._abs_start(extra_info)
        g = self.group_pages
        jobs = []  # (object, page indices)
        i = 0
        while i < len(keys):
            if p0 is not None and g > 1 and (p0 + i) % g == 0 and i + g <= len(keys):
                obj = self._group_obj(keys[i + g - 1])
                for j in range(g):
                    self._page_loc.put(keys[i + j], (obj, j))
                jobs.append((obj, list(range(i, i + g))))
                i += g
            else:
                jobs.append((self._page_obj(keys[i]), [i]))
                i += 1
        ok = [False] * len(keys)

        def run(job):
            obj, idx = job
            if self._known.get(obj):
                return job, True
            ptrs = [p for k in idx for p in pages[k][0]]
            sizes = [s for k in idx for s in pages[k][1]]
            return job, self._put_pieces(obj, ptrs, sizes)

        for (obj, idx), success in self._wio.map(run, jobs):
            for k in idx:
                ok[k] = success
        return ok

    def _candidates(self, keys: List[str], p0: Optional[int]) -> List[Tuple[str, int]]:
        """(object, index in object) to probe for each page: a recorded group, the
        aligned group whose last page is in `keys`, else the single-page object."""
        locs = []
        g = self.group_pages
        for i, key in enumerate(keys):
            loc = self._page_loc.get(key)
            if loc is None and p0 is not None and g > 1:
                last = i + (g - 1 - (p0 + i) % g)
                if last < len(keys):
                    loc = (self._group_obj(keys[last]), (p0 + i) % g)
            locs.append(loc or (self._page_obj(key), 0))
        return locs

    def _locate(self, keys: List[str], p0: Optional[int]) -> List[Tuple[str, int]]:
        """Where to read each page: a group only once it is known to exist."""
        return [
            (obj, j) if (obj.endswith(f".g{self.group_pages}") and self._known.get(obj)) else (self._page_obj(key), 0)
            for (obj, j), key in zip(self._candidates(keys, p0), keys)
        ]

    def _infer_start(self, keys: List[str]) -> Optional[int]:
        """Absolute start (mod group_pages) of `keys` when the caller passes no prefix_keys.

        SGLang drops an empty prefix_keys list to None, e.g. for a prefix at the
        radix root of a node that never saw it. A group object exists only under
        the hash of a page that ends a group, so the first existing candidate
        among the first `group_pages` pages fixes the alignment. A page's index in
        a group is its distance from that end, so a found object is never
        misread whatever the true absolute start is.
        """
        g = self.group_pages
        for w in range(0, len(keys), g):
            ends = list(range(w, min(w + g, len(keys))))
            found = self._exists_many([self._group_obj(keys[e]) for e in ends])
            for e in ends:
                if found[self._group_obj(keys[e])]:
                    return (g - 1 - e) % g
            # No group ends here: the prefix continues only if these pages are single-page objects.
            singles = self._exists_many([self._page_obj(keys[e]) for e in ends])
            if not all(singles.values()):
                return None
        return None

    def _kv_prefix_hits(self, keys: List[str], p0: Optional[int]) -> int:
        if not keys:
            return 0
        if p0 is None and self.group_pages > 1:
            # A recorded group location fixes the alignment (a page's index in its group is its absolute
            # position mod group_pages). Without an alignment, pages other nodes grouped after this node's
            # last lookup would be probed as single-page objects and the prefix would stop short.
            loc = self._page_loc.get(keys[0])
            p0 = loc[1] if loc else self._infer_start(keys)
        locs = self._candidates(keys, p0)
        groups = self._exists_many([obj for obj, _ in locs])
        # Pages whose group is missing may still be single-page objects.
        fallback = [self._page_obj(k) for (obj, _), k in zip(locs, keys) if not groups[obj] and obj.endswith(f".g{self.group_pages}")]
        singles = self._exists_many(fallback) if fallback else {}
        hits = 0
        for (obj, j), key in zip(locs, keys):
            if groups.get(obj):
                if obj.endswith(f".g{self.group_pages}"):
                    self._page_loc.put(key, (obj, j))
            elif singles.get(self._page_obj(key)):
                self._page_loc.discard(key)
            else:
                break
            hits += 1
        return hits

    def batch_exists(self, keys, extra_info=None) -> int:
        hits = self._kv_prefix_hits(keys, self._abs_start(extra_info))
        if self.debug:
            logger.info("[rustfs-hicache] tp%d exists: %d keys p0=%s first=%s -> kv %d", self.storage_config.tp_rank,
                        len(keys), self._abs_start(extra_info), keys[0][:16] if keys else None, hits)
        return hits

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        if not keys:
            return []
        pages = self._kv_meta(keys, host_indices)
        locs = self._locate(keys, self._abs_start(extra_info))
        page_bytes = sum(pages[0][1])
        # Coalesce consecutive pages of the same object into one ranged GET.
        runs = []
        for i, (obj, j) in enumerate(locs):
            if runs and runs[-1][0] == obj and runs[-1][2] + len(runs[-1][3]) == j:
                runs[-1][3].append(i)
            else:
                runs.append([obj, i, j, [i]])
        ok = [False] * len(keys)

        def run(r):
            obj, _, j0, idx = r
            ptrs = [p for k in idx for p in pages[k][0]]
            sizes = [s for k in idx for s in pages[k][1]]
            return idx, self._get_pieces(obj, j0 * page_bytes, ptrs, sizes)

        for idx, success in self._io.map(run, runs):
            for k in idx:
                ok[k] = success
        return ok

    # ---- sidecar pools (v2) -----------------------------------------------
    def _transfer_pages(self, transfer: PoolTransfer):
        pool = self.registered_pools[transfer.name]
        keys = transfer.keys or []
        if not keys:
            return pool, []
        ptrs, sizes = pool.get_page_buffer_meta(transfer.host_indices)
        m = len(ptrs) // len(keys)
        return pool, [(ptrs[i * m:(i + 1) * m], sizes[i * m:(i + 1) * m]) for i in range(len(keys))]

    def batch_set_v2(self, transfers: List[PoolTransfer], extra_info=None) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        for t in transfers:
            if t.name == PoolName.KV:
                results[t.name] = self.batch_set_v1(t.keys or [], t.host_indices, extra_info)
                continue
            _, pages = self._transfer_pages(t)
            objs = [self._sidecar_obj(t.name, k) for k in t.keys or []]
            results[t.name] = list(self._wio.map(
                lambda a: True if self._known.get(a[0]) else self._put_pieces(a[0], a[1][0], a[1][1]),
                zip(objs, pages)))
        return results

    def batch_get_v2(self, transfers: List[PoolTransfer], extra_info=None) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        for t in transfers:
            if t.name == PoolName.KV:
                results[t.name] = self.batch_get_v1(t.keys or [], t.host_indices, extra_info)
                continue
            _, pages = self._transfer_pages(t)
            objs = [self._sidecar_obj(t.name, k) for k in t.keys or []]
            results[t.name] = list(self._io.map(lambda a: self._get_pieces(a[0], 0, a[1][0], a[1][1]), zip(objs, pages)))
        return results

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None) -> PoolTransferResult:
        kv_pages = self._kv_prefix_hits(keys, self._abs_start(extra_info))
        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        for transfer in pool_transfers or []:
            if not restorable:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                # Only the longest restorable prefix matters: probe from the end, one window at a time.
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = self._trailing_boundary(name, keys, kv_pages, trailing)
                if boundary:
                    hit_count[name] = boundary
                restorable = [p for p in restorable if p == boundary]
                continue
            present = self._exists_many([self._sidecar_obj(name, keys[i]) for i in range(kv_pages)])
            has = [present[self._sidecar_obj(name, keys[i])] for i in range(kv_pages)]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next((i for i in range(kv_pages) if not has[i]), kv_pages)
                pool_restorable = set(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                pool_restorable = {
                    n for n in range(1, kv_pages + 1) if all(has[i] for i in range(max(0, n - trailing), n))
                }
                boundary = max(pool_restorable, default=0)
            else:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            if boundary:
                hit_count[name] = boundary
            restorable = [p for p in restorable if p in pool_restorable]
        if self.debug:
            logger.info(
                "[rustfs-hicache] tp%d exists_v2: %d keys p0=%s first=%s last=%s pools=%s -> kv %d final %d",
                self.storage_config.tp_rank, len(keys), self._abs_start(extra_info), keys[0][:16] if keys else None,
                keys[-1][:16] if keys else None, [(t.name.value, t.hit_policy.value, len(t.keys or [])) for t in pool_transfers or []],
                kv_pages, restorable[-1] if restorable else 0,
            )
        return PoolTransferResult(restorable[-1] if restorable else 0, hit_count, restorable)

    def _trailing_boundary(self, name, keys: List[str], kv_pages: int, trailing: int) -> int:
        """Largest n <= kv_pages whose last `trailing` pages all have `name` state stored (0 if none)."""
        hi = kv_pages
        while hi > 0:
            lo = max(0, hi - self.probe_window)
            idx = list(range(max(0, lo - trailing + 1), hi))
            found = self._exists_many([self._sidecar_obj(name, keys[i]) for i in idx])
            has = {i: found[self._sidecar_obj(name, keys[i])] for i in idx}
            for n in range(hi, lo, -1):
                if all(has.get(i, False) for i in range(max(0, n - trailing), n)):
                    return n
            hi = lo
        return 0

    # ---- legacy single-key surface (not used by the zero-copy paths) ---------
    def get(self, key, target_location=None, target_sizes=None):
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        return False

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return self._kv_prefix_hits([key], None) == 1

    def clear(self) -> bool:
        # Objects stay in RustFS (shared across processes); forget local state only.
        self._known.clear()
        self._page_loc.clear()
        return True

    def get_stats(self):
        return None

    def close(self):
        self._wio.shutdown(wait=True)
        self._wb.shutdown(wait=True)  # finish background PUTs before the memory goes away
        self._sub.shutdown(wait=True)
        self._io.shutdown(wait=True)
        self.client.deregister_all()
