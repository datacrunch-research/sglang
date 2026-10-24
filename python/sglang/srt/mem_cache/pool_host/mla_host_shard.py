"""MLA HiCache host tier striped over the attention-TP ranks.

Every attention-TP rank computes and keeps the same MLA latent KV on the device
(the latent projection is replicated). The stock host tier therefore stores N
identical copies of every cached page, one per rank. With
``SGLANG_HICACHE_MLA_HOST_SHARD=1`` each rank keeps only 1/N of the cached pages
on the host instead, so the same per-rank host memory holds N times as many
tokens.

Placement (logical host slots, widened by N; one physical page = ``page_size``
rows): logical host page ``h`` is owned by rank ``h % N`` and is that rank's
local host page ``h // N``. Logical pages ``0..N-1`` (local page 0 of every
rank) are never allocated; transfers pad with them.

The device pool is untouched and stays a full copy on every rank, so attention
and the P/D sender read it as before. Per transfer, the controller asks the pool
for an owner-major plan built on the CPU from the host slots (no device sync):

- backup (device -> host): each rank copies only the pages it owns;
- load-back (host -> device): each rank copies its own pages to their device
  rows on the host-to-device stream (copies only, as stock HiCache). The other
  ranks' pages are filled by one all-gather per layer over a dedicated NCCL
  communicator, run on the forward stream when the consuming batch starts
  (``LayerDoneCounter.set_consumer``), after the whole load has landed. A
  collective on a side stream would deadlock against kernels that grid-sync
  across ranks (MegaMoE): they occupy every SM while waiting for peers whose
  side-stream collective cannot get SMs.

Every rank's block is padded to the largest owner's page count with page 0
(host local page 0 and device page 0, which no allocator hands out), so all
collectives have equal sizes and no rank needs to know another rank's counts at
run time.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.mem_cache.pool_host.base import synchronized

logger = logging.getLogger(__name__)

_SUPPORTED_LAYOUTS = ("layer_first", "page_first")


class MLAHostShard:
    """The shard group seen by the host pool: rank, size, NCCL communicator."""

    def __init__(self, rank: int, size: int, group_coordinator, chunk_tokens: int):
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.rank = rank
        self.size = size
        self.chunk_tokens = chunk_tokens
        self.comm = PyNcclCommunicator(
            group=group_coordinator.cpu_group, device=group_coordinator.device
        )
        logger.info(
            "MLA host shard: rank %d of %d, dedicated NCCL communicator for "
            "load-back all-gathers, chunk %d tokens",
            rank,
            size,
            chunk_tokens,
        )


class MLAHostShardPoolMixin:
    """Logical-size widening, owner-major transfer plans and the load gather."""

    def __init__(self, *args, host_shard: MLAHostShard, **kwargs):
        # Set before super().__init__: the base constructor calls clear(),
        # which sizes the free list from logical_size.
        self.host_shard = host_shard
        self.host_shard_rank = host_shard.rank
        self.host_shard_size = host_shard.size
        self._gather_send = None
        self._gather_recv = None
        self._pending_gathers: dict = {}
        super().__init__(*args, **kwargs)
        assert self.dcp_size == 1, "MLA host sharding does not compose with DCP"
        assert self.layout in _SUPPORTED_LAYOUTS, (
            f"MLA host sharding supports {_SUPPORTED_LAYOUTS}, got {self.layout!r}"
        )
        # The staged write-back kernel reads raw CPU host slots; this pool's
        # backups go through the per-rank plan with the indexed kernels instead.
        self.can_use_write_back_jit = False
        self.staging_buffer = None
        logger.info(
            "MLA host shard pool: %d rows per rank (%.2f GB), logical capacity "
            "%d tokens over %d ranks",
            self.size,
            self.size * self.size_per_token / 1e9,
            self.logical_size,
            self.host_shard_size,
        )

    # ---- capacity -----------------------------------------------------------

    @property
    def logical_size(self) -> int:
        return self.size * self.dcp_size * self.host_shard_size

    @synchronized
    def clear(self):
        super().clear()
        # Reserve logical pages 0..N-1 (local page 0 on every rank): the
        # padding target of every transfer plan.
        reserved = self.page_size * self.host_shard_size
        self.free_slots = self.free_slots[reserved:]

    # ---- per-transfer plan ----------------------------------------------------

    def host_shard_plan(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logical (host, device) slot pairs -> (this rank's local host rows,
        owner-major device rows of all ranks), both padded per rank to the
        largest owner's page count. Built from the CPU host slots."""
        ps = self.page_size
        n_ranks = self.host_shard_size
        rank = self.host_shard_rank
        h = host_indices.cpu() if host_indices.is_cuda else host_indices
        n = h.numel()
        if n == 0:
            return h, device_indices
        assert n % ps == 0, f"host transfer of {n} slots is not page-aligned ({ps})"
        assert device_indices.numel() == n, "host/device transfer lengths differ"
        h = h.to(torch.int64)
        page_first_slot = h[::ps]
        assert bool((page_first_slot % ps == 0).all()), "host pages not aligned"
        logical_page = page_first_slot // ps
        owner = logical_page % n_ranks
        counts = torch.bincount(owner, minlength=n_ranks)
        m = int(counts.max())
        order = torch.argsort(owner, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        # Owner-major page positions, -1 = padding.
        pos = torch.full((n_ranks, m), -1, dtype=torch.int64)
        for o in range(n_ranks):
            c = int(counts[o])
            if c:
                pos[o, :c] = order[int(starts[o]) : int(starts[o]) + c]
        offs = torch.arange(ps, dtype=torch.int64)
        valid = pos >= 0
        tok_pos = (pos.clamp(min=0) * ps).unsqueeze(-1) + offs  # [N, m, ps]
        pad_rows = offs.expand(n_ranks, m, ps)
        # This rank's local host rows.
        mine = pos[rank]
        mine_valid = valid[rank]
        local_page = logical_page[mine.clamp(min=0)] // n_ranks
        host_rows = (local_page * ps).unsqueeze(-1) + offs
        host_rows = torch.where(mine_valid.unsqueeze(-1), host_rows, offs.expand(m, ps))
        # Device rows of every rank's block (gathered on the device side).
        dev = device_indices.to(torch.int64)
        tok_pos_dev = tok_pos.reshape(-1).to(dev.device, non_blocking=True)
        valid_dev = (
            valid.unsqueeze(-1)
            .expand(n_ranks, m, ps)
            .reshape(-1)
            .to(dev.device, non_blocking=True)
        )
        pad_dev = pad_rows.reshape(-1).to(dev.device, non_blocking=True)
        device_all = torch.where(valid_dev, dev[tok_pos_dev], pad_dev)
        return host_rows.reshape(-1), device_all

    def _my_block(self, device_all: torch.Tensor) -> torch.Tensor:
        block = device_all.numel() // self.host_shard_size
        r = self.host_shard_rank
        return device_all[r * block : (r + 1) * block]

    # ---- transfers --------------------------------------------------------------

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        if host_indices.numel() == 0:
            return
        self.backup_from_device_all_layer(
            device_pool, host_indices, self._my_block(device_indices), io_backend
        )

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        if host_indices.numel() == 0:
            return
        self.load_to_device_per_layer(
            device_pool,
            host_indices,
            self._my_block(device_indices),
            layer_id,
            io_backend,
            is_draft=is_draft,
        )

    def register_pending_gather(self, producer_index: int, device_all: torch.Tensor):
        """Record the device rows a load submitted under ``producer_index``
        must fill from peers; replaces any unconsumed plan of that slot."""
        self._pending_gathers[producer_index] = device_all

    def run_pending_gather(self, consumer_index: int, finish_event) -> None:
        """On the forward stream, before the consuming batch: wait for the
        load, then fill the other ranks' pages layer by layer."""
        device_all = self._pending_gathers.pop(consumer_index, None)
        if device_all is None or device_all.numel() == 0:
            return
        stream = torch.cuda.current_stream()
        stream.wait_event(finish_event)
        device_all.record_stream(stream)
        mine = self._my_block(device_all)
        for layer_id in self._target_layer_indices:
            buf = self.device_kv_buffers[self._buffer_index_by_layer[layer_id]]
            self._gather_layer(buf, mine, device_all)

    def _gather_buffers(self, buf: torch.Tensor, rows: int):
        n_ranks = self.host_shard_size
        row_shape = tuple(buf.shape[1:])
        if (
            self._gather_send is None
            or self._gather_send.dtype != buf.dtype
            or tuple(self._gather_send.shape[1:]) != row_shape
            or self._gather_send.device != buf.device
        ):
            self._gather_send = torch.empty(
                (rows, *row_shape), dtype=buf.dtype, device=buf.device
            )
            self._gather_recv = torch.empty(
                (n_ranks * rows, *row_shape), dtype=buf.dtype, device=buf.device
            )
        return self._gather_send, self._gather_recv

    def _gather_layer(
        self, buf: torch.Tensor, mine: torch.Tensor, device_all: torch.Tensor
    ) -> None:
        n_ranks = self.host_shard_size
        block = mine.numel()
        if block == 0:
            return
        chunk = max(
            self.page_size,
            self.host_shard.chunk_tokens // self.page_size * self.page_size,
        )
        send_buf, recv_buf = self._gather_buffers(buf, chunk)
        dst_all = device_all.view(n_ranks, block)
        comm = self.host_shard.comm
        for s in range(0, block, chunk):
            c = min(chunk, block - s)
            send = send_buf[:c]
            torch.index_select(buf, 0, mine[s : s + c], out=send)
            recv = recv_buf[: n_ranks * c]
            with comm.change_state(enable=True):
                comm.all_gather(recv, send)
            dst = dst_all[:, s : s + c].reshape(-1)
            buf.index_copy_(0, dst, recv)


@functools.cache
def mla_host_shard_pool_cls(host_pool_cls: type) -> type:
    """``host_pool_cls`` with its host tier striped over the attention-TP ranks."""
    return type(
        f"MLAHostShard{host_pool_cls.__name__}",
        (MLAHostShardPoolMixin, host_pool_cls),
        {},
    )


def make_mla_host_shard(
    *, storage_backend: Optional[str], layout: str
) -> Optional[MLAHostShard]:
    """The shard group for the MLA host pool, or None when the env is off.

    Refuses configurations the striped host tier does not support."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    if not envs.SGLANG_HICACHE_MLA_HOST_SHARD.get():
        return None
    parallel = get_parallel()
    if parallel.dcp_enabled:
        raise ValueError(
            "SGLANG_HICACHE_MLA_HOST_SHARD does not support --dcp-size > 1"
        )
    if storage_backend is not None:
        raise ValueError(
            "SGLANG_HICACHE_MLA_HOST_SHARD supports the host tier (L2) only; "
            f"remove --hicache-storage-backend ({storage_backend!r})"
        )
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            f"SGLANG_HICACHE_MLA_HOST_SHARD needs --hicache-mem-layout in "
            f"{_SUPPORTED_LAYOUTS}, got {layout!r}"
        )
    group = parallel.attn_tp_group
    if group.world_size <= 1:
        logger.warning("SGLANG_HICACHE_MLA_HOST_SHARD ignored: attention TP size 1")
        return None
    return MLAHostShard(
        rank=group.rank_in_group,
        size=group.world_size,
        group_coordinator=group,
        chunk_tokens=envs.SGLANG_HICACHE_MLA_HOST_SHARD_CHUNK_TOKENS.get(),
    )
