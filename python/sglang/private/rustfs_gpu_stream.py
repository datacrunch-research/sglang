"""Stream a safetensors checkpoint from RustFS into GPU memory over NIXL, shared across ranks over NCCL.

No checkpoint byte touches disk, tmpfs or host memory:

* Rank 0 reads every shard's safetensors header (small HTTP range GETs) and
  splits the checkpoint into units of whole, contiguous tensors of about
  `unit_bytes` (a larger tensor is a unit of its own). Units are laid out in
  order in a per-GPU staging ring of `ring_bytes`; every rank gets the plan.
* Unit `u` is owned by rank `u % world`. The owner's GPU RDMA-fetches the
  unit's byte range from RustFS (GPUDirect, ranged NIXL GETs spread over the
  servers, on the NIC closest to that GPU) into the unit's ring region.
* Every rank consumes the units in order: an NCCL broadcast from the owner
  into the same ring region, then the tensors are yielded as views (cloned by
  default) into that region.
* A region is refilled only after this rank's reads of the units that last
  used it have completed (CUDA event), so staging stays at `ring_bytes`.
"""
import concurrent.futures
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
for _name, _attr in (("F8_E8M0", "float8_e8m0fnu"), ("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
    if hasattr(torch, _attr):
        DTYPES[_name] = getattr(torch, _attr)

ALIGN = 256


@dataclass
class Unit:
    key: str            # object key of the shard
    start: int          # absolute byte offset of the unit in the object
    length: int
    ring_off: int       # offset of the unit in every rank's staging ring
    reuse_after: int    # index of the last earlier unit whose ring region overlaps this one (-1: none)
    tensors: List[Tuple[str, str, List[int], int, int]] = field(default_factory=list)  # name, dtype, shape, rel_off, nbytes


HEADER_PROBE = 1 << 20


def _header(client, bucket, key, endpoint):
    head = client.get_bytes(bucket, key, 0, HEADER_PROBE, endpoint)
    (n,) = struct.unpack("<Q", head[:8])
    if 8 + n > len(head):
        head += client.get_bytes(bucket, key, len(head), 8 + n - len(head), endpoint)
    return 8 + n, json.loads(head[8:8 + n])


def build_plan(client, bucket, keys, unit_bytes, ring_bytes) -> List[Unit]:
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        headers = list(ex.map(lambda ik: _header(client, bucket, ik[1], client.endpoint(ik[0])), enumerate(keys)))
    units: List[Unit] = []
    for key, (data_start, header) in zip(keys, headers):
        entries = sorted(
            ((name, meta) for name, meta in header.items() if name != "__metadata__"),
            key=lambda item: item[1]["data_offsets"][0],
        )
        current: Optional[Unit] = None
        for name, meta in entries:
            begin, end = meta["data_offsets"]
            if meta["dtype"] not in DTYPES:
                raise ValueError(f"{key}: tensor {name} has unsupported dtype {meta['dtype']}")
            if current is not None and data_start + end - current.start > unit_bytes:
                units.append(current)
                current = None
            if current is None:
                current = Unit(key, data_start + begin, 0, 0, -1)
            current.tensors.append((name, meta["dtype"], meta["shape"], data_start + begin - current.start, end - begin))
            current.length = data_start + end - current.start
        if current is not None:
            units.append(current)
    # Lay the units out in the ring in order, wrapping when the tail is too short.
    head, placed = 0, []
    for index, unit in enumerate(units):
        if unit.length > ring_bytes:
            raise ValueError(f"{unit.key}: unit of {unit.length} bytes does not fit the {ring_bytes}-byte staging ring")
        if head + unit.length > ring_bytes:
            head = 0
        unit.ring_off = head
        lo, hi = head, head + unit.length
        for prev in reversed(placed):
            other = units[prev]
            if other.ring_off < hi and lo < other.ring_off + other.length:
                unit.reuse_after = prev
                break
        placed.append(index)
        placed = placed[-4096:]
        head = (hi + ALIGN - 1) // ALIGN * ALIGN
    return units


class GpuWeightStream:
    """Yields (name, tensor) on this rank's GPU for every tensor of `keys`, in plan order."""

    def __init__(self, client, bucket, keys, *, group, cpu_group, unit_bytes=512 << 20, ring_bytes=16 << 30,
                 chunk_bytes=128 << 20, fetch_units=4, streams_per_unit=8, clone=True):
        self.client, self.bucket = client, bucket
        self.group, self.cpu_group = group, cpu_group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.src_ranks = dist.get_process_group_ranks(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.chunk_bytes, self.fetch_units, self.streams = chunk_bytes, fetch_units, streams_per_unit
        self.clone = clone

        plan = [build_plan(client, bucket, keys, unit_bytes, ring_bytes)] if self.rank == 0 else [None]
        dist.broadcast_object_list(plan, src=self.src_ranks[0], group=cpu_group)
        self.units: List[Unit] = plan[0]
        ring_used = max(u.ring_off + u.length for u in self.units)
        self.ring = torch.empty(ring_used, dtype=torch.uint8, device=self.device)
        client.register(self.ring.data_ptr(), ring_used, "VRAM", self.device.index)

        self._cv = threading.Condition()
        self._consumed = -1                       # highest unit index this rank has consumed
        self._events = {}                         # unit index -> CUDA event after its reads
        self._ready = {}                          # owned unit index -> threading.Event
        self._error: Optional[BaseException] = None
        self.fetched_bytes = 0
        self.fetch_seconds = 0.0
        self.wait_seconds = 0.0
        self.total_bytes = sum(u.length for u in self.units)
        self.owned = [i for i in range(len(self.units)) if i % self.world == self.rank]
        for i in self.owned:
            self._ready[i] = threading.Event()
        self._chunks = concurrent.futures.ThreadPoolExecutor(fetch_units * streams_per_unit, thread_name_prefix="rustfs-chunk")
        self._fetchers = concurrent.futures.ThreadPoolExecutor(fetch_units, thread_name_prefix="rustfs-unit")
        self._queue = iter(self.owned)
        self._queue_lock = threading.Lock()
        for _ in range(fetch_units):
            self._fetchers.submit(self._fetch_loop)

    # ---- owner side -----------------------------------------------------
    def _wait_region_free(self, index):
        after = self.units[index].reuse_after
        if after < 0:
            return
        with self._cv:
            while self._consumed < after and self._error is None:
                self._cv.wait(0.1)
            event = self._events.get(after)
        if event is not None:
            event.synchronize()

    def _fetch_loop(self):
        try:
            while True:
                with self._queue_lock:
                    index = next(self._queue, None)
                if index is None:
                    return
                self._wait_region_free(index)
                if self._error is not None:
                    return
                unit = self.units[index]
                base = self.ring.data_ptr() + unit.ring_off
                parts = [(o, min(self.chunk_bytes, unit.length - o)) for o in range(0, unit.length, self.chunk_bytes)]
                started = time.time()
                list(self._chunks.map(
                    lambda p: self.client.fetch_into(
                        self.bucket, unit.key, base + p[0], p[1], unit.start + p[0],
                        self.client.endpoint(index * len(parts) + p[0] // self.chunk_bytes)),
                    parts))
                with self._cv:
                    self.fetch_seconds += time.time() - started
                    self.fetched_bytes += unit.length
                self._ready[index].set()
        except BaseException as e:  # surfaced on the consumer thread
            logger.exception("[rustfs-loader] rank %d fetch failed", self.rank)
            with self._cv:
                self._error = e
                self._cv.notify_all()
            for event in self._ready.values():
                event.set()

    # ---- consumer side --------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        started = time.time()
        try:
            for index, unit in enumerate(self.units):
                owner = index % self.world
                region = self.ring[unit.ring_off:unit.ring_off + unit.length]
                if owner == self.rank:
                    t = time.time()
                    self._ready[index].wait()
                    self.wait_seconds += time.time() - t
                    if self._error is not None:
                        raise RuntimeError(f"RustFS fetch failed on rank {self.rank}") from self._error
                dist.broadcast(region, src=self.src_ranks[owner], group=self.group)
                for name, dtype_name, shape, rel, nbytes in unit.tensors:
                    dtype = DTYPES[dtype_name]
                    raw = region[rel:rel + nbytes]
                    if rel % dtype.itemsize:
                        tensor = raw.clone().view(dtype).reshape(shape)
                    else:
                        tensor = raw.view(dtype).reshape(shape)
                        if self.clone:
                            tensor = tensor.clone()
                    yield name, tensor
                event = torch.cuda.Event()
                event.record()
                with self._cv:
                    self._events[index] = event
                    self._events.pop(index - 4096, None)
                    self._consumed = index
                    self._cv.notify_all()
            torch.cuda.synchronize()
        finally:
            self.close()
            self.wall_seconds = time.time() - started
            logger.info(
                "[rustfs-loader] rank %d: %d units, checkpoint %.1f GB streamed in %.1f s (%.1f GB/s); "
                "fetched %.1f GB in %.1f s of unit transfers; waited %.1f s for own units",
                self.rank, len(self.units), self.total_bytes / 1e9, self.wall_seconds,
                self.total_bytes / 1e9 / max(self.wall_seconds, 1e-9), self.fetched_bytes / 1e9,
                self.fetch_seconds, self.wait_seconds,
            )

    def close(self):
        with self._cv:
            if self._error is None:
                self._error = RuntimeError("stream closed")
            self._cv.notify_all()
        self._fetchers.shutdown(wait=True)
        self._chunks.shutdown(wait=True)
        torch.cuda.synchronize()
        self.client.deregister_all()
        self.ring = None
        torch.cuda.empty_cache()
