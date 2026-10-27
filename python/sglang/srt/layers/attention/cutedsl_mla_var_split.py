"""Length-balanced split-KV for the flashinfer cute-dsl monolithic MLA decode under DCP.

The stock call (``trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl")``)
picks one split count per launch from occupancy only
(``num_sms // (2 * batch)``: 4 / 3 / 2 / 1 at 16 / 24 / 32 / 40+ rows on a
148-SM GPU) and cuts every request's KV into that many equal chunks. A batch
with one long request then runs as long as that request's chunk: at 32 rows a
640K-token request beside 116K requests makes the MLA decode about 3x longer,
and at 40+ rows (split 1) about 4.5x, whatever the batch's total KV.

This module launches the same compiled kernel family with
``is_var_split_kv=True``: the grid keeps a fixed split dimension ``smax``
(chosen per captured batch size), and ``block_split_kvs[b]`` gives request
``b`` a split count proportional to its length. The cluster budget stays the
stock one (``batch * base_split``), so a batch of equal lengths gets the stock
split and only the length distribution changes. Planning is a handful of torch
ops on device per step (CUDA-graph capturable, no host sync).

Gate: ``SGLANG_CUTEDSL_MLA_VAR_SPLIT=1``; ``SGLANG_CUTEDSL_MLA_VAR_SPLIT_MAX``
sets ``smax`` as ``xN`` (N times the stock split) or an absolute count
(default ``x4``), always capped so ``batch * smax <= 512``.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

_MAX_GRID_SPLITS_X_BATCH = 512  # B * smax above 1024 produced NaN in a microbench
_KV_TILE = 128  # mma_qk_tiler n of the cute-dsl MLA decode


def var_split_enabled() -> bool:
    return os.environ.get("SGLANG_CUTEDSL_MLA_VAR_SPLIT", "0") == "1"


def stock_split(batch_size: int, num_sms: int) -> int:
    """flashinfer's occupancy split for fixed-Q decode with one 128-row Q tile."""
    return max(1, min(32, num_sms // (2 * batch_size)))


def smax_for(batch_size: int, num_sms: int) -> int:
    base = stock_split(batch_size, num_sms)
    spec = os.environ.get("SGLANG_CUTEDSL_MLA_VAR_SPLIT_MAX", "x4")
    if spec.startswith("x"):
        smax = base * int(spec[1:])
    else:
        smax = int(spec)
    smax = min(smax, _MAX_GRID_SPLITS_X_BATCH // batch_size)
    return max(smax, base)


def plan_block_splits(
    local_seq_lens: torch.Tensor, out: torch.Tensor, base_split: int, smax: int
) -> None:
    """Per-request split counts, written into ``out`` (int32 [B]).

    target tiles per split = ceil(sum(tiles) / (B * base_split)); a request
    gets ceil(tiles / target) splits, clamped to [1, smax]. Equal lengths give
    every request ``base_split``.
    """
    bsz = local_seq_lens.shape[0]
    tiles = (local_seq_lens.to(torch.int32) + (_KV_TILE - 1)) // _KV_TILE
    budget = bsz * base_split
    target = torch.clamp((tiles.sum() + (budget - 1)) // budget, min=2)
    torch.clamp((tiles + target - 1) // target, 1, smax, out=out)


def _skip_empty_split_stores_enabled() -> bool:
    return os.environ.get("SGLANG_CUTEDSL_MLA_VAR_SPLIT_SKIP_EMPTY", "1") == "1"


_PATCHED_FP8_CLASS = None

# flashinfer 0.7.1 cute-dsl MLA decode, compute warps: a DCP split slot with no K
# tile stores O = 0 and LSE = -inf. The split reducer reads slot 0 of an empty
# request and never a slot past the request's nonempty split count, so only
# slot 0 needs the store; with var split-KV most grid slots are past it.
_EMPTY_STORE_OLD = """                elif cutlass.const_expr(self.enable_dcp):
                    self.store_empty_dcp_work(
                        mO,
                        mLSE,
                        mAccO,
                        mAccLSE,
                        blk_coord,
                        q_begin,
                        valid_q_rows,
                        tidx,
                    )
"""
_EMPTY_STORE_NEW = """                elif cutlass.const_expr(self.enable_dcp):
                    if blk_coord[3] == 0:
                        self.store_empty_dcp_work(
                            mO,
                            mLSE,
                            mAccO,
                            mAccLSE,
                            blk_coord,
                            q_begin,
                            valid_q_rows,
                            tidx,
                        )
"""


def _patched_fp8_class():
    """Load a copy of flashinfer's fp8 MLA decode module with the empty-slot store
    limited to slot 0 (written to a real file so the CuTe DSL can read its source)."""
    global _PATCHED_FP8_CLASS
    if _PATCHED_FP8_CLASS is not None:
        return _PATCHED_FP8_CLASS
    import importlib.util
    import inspect
    import tempfile

    from flashinfer.cute_dsl.attention.monolithic import mla_decode_fp8 as base

    src = inspect.getsource(base)
    if src.count(_EMPTY_STORE_OLD) != 1:
        raise RuntimeError(
            "flashinfer mla_decode_fp8 changed; empty-slot store patch does not apply"
        )
    src = src.replace(_EMPTY_STORE_OLD, _EMPTY_STORE_NEW)
    d = tempfile.mkdtemp(prefix="sglang_cutedsl_mla_vs_")
    path = os.path.join(d, "mla_decode_fp8_vs.py")
    with open(path, "w") as f:
        f.write(src)
    name = base.__name__ + "_vs"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = base.__package__
    spec.loader.exec_module(mod)
    _PATCHED_FP8_CLASS = mod.BlackwellMultiHeadLatentAttentionForwardFP8
    logger.info(
        "cute-dsl MLA var split-KV: empty split slots past slot 0 skip their store (%s)",
        path,
    )
    return _PATCHED_FP8_CLASS


class CuteDslMLAVarSplitDecode:
    """Direct launcher of the monolithic cute-dsl MLA decode with var split-KV."""

    def __init__(
        self,
        device: torch.device,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        cp_world: int,
    ):
        from flashinfer.cute_dsl.attention.monolithic import mla_decode as impl
        from flashinfer.cute_dsl.attention.monolithic.mla_helpers import MAX_SPLITS
        from flashinfer.utils import device_support_pdl, get_compute_capability

        self._impl = impl
        self._max_splits = MAX_SPLITS
        # flashinfer 0.7.2: the compile entry takes the target arch first. Its stock
        # launches size the split reducer to the split count, but a var split-KV
        # kernel must keep the generic reducer (MAX_SPLITS), as in 0.7.1.
        self._arch = None
        if "arch" in inspect.signature(impl._get_compiled_mla_kernel).parameters:
            from flashinfer.cute_dsl.utils import cute_dsl_compile_arch

            self._arch = cute_dsl_compile_arch(*get_compute_capability(device))
        self.num_sms = impl.get_num_sm(device)
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = page_size
        self.cp_world = cp_world
        self.enable_pdl = bool(device_support_pdl(device))
        self._kernels = {}

    def workspace_bytes(self, batch_size: int, smax: int) -> int:
        import cutlass

        return (
            self._impl.BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
                128, 1, self.kv_lora_rank, batch_size, smax, cutlass.Float32
            )
        )

    def _kernel(
        self,
        q_dtype: torch.dtype,
        out_dtype: torch.dtype,
        is_workspace_size_zero: bool = False,
    ):
        key = (q_dtype, out_dtype, is_workspace_size_zero)
        k = self._kernels.get(key)
        if k is None:
            k = self._compile(q_dtype, out_dtype, is_workspace_size_zero)
            self._kernels[key] = k
        return k

    def _compile(
        self,
        q_dtype: torch.dtype,
        out_dtype: torch.dtype,
        is_workspace_size_zero: bool = False,
    ):
        patched = _skip_empty_split_stores_enabled() and q_dtype == torch.float8_e4m3fn
        orig = self._impl.BlackwellMultiHeadLatentAttentionForwardFP8
        compile_fn = self._impl._get_compiled_mla_kernel
        if patched:
            # Compile from a patched copy of the fp8 kernel class; the stock path keeps its own.
            self._impl.BlackwellMultiHeadLatentAttentionForwardFP8 = (
                _patched_fp8_class()
            )
            compile_fn = compile_fn.__wrapped__
        try:
            extra = {} if self._arch is None else {"arch": self._arch}
            return compile_fn(
                **extra,
                torch_dtype=q_dtype,
                torch_out_dtype=out_dtype,
                page_size=self.page_size,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                num_heads=self.num_heads,
                seq_len_q=1,
                is_persistent=False,
                is_var_seq=True,
                is_var_q=False,
                is_var_split_kv=True,
                reducer_d_tiles=1,
                reducer_max_splits=self._max_splits,
                skip_correction_threshold=0.0,
                is_workspace_size_zero=is_workspace_size_zero,
                enable_pdl=self.enable_pdl,
                enable_dcp=True,
                cp_world=self.cp_world,
            )
        finally:
            self._impl.BlackwellMultiHeadLatentAttentionForwardFP8 = orig

    def run(
        self,
        query: torch.Tensor,  # [B, 1, H, L + R]
        kv_cache: torch.Tensor,  # [pages, 1, page, L + R] or [pages, page, L + R]
        workspace: torch.Tensor,  # int8, >= workspace_bytes(B, smax)
        block_tables: torch.Tensor,  # [B, max_pages] int32
        seq_lens: torch.Tensor,  # [B] rank-local int32
        causal_seqs: torch.Tensor,  # [B] global int32
        cp_rank: int,
        block_split_kvs: torch.Tensor,  # [B] int32
        smax: int,
        softmax_scale: float,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from cutlass import Float32, Int32

        bsz = query.shape[0]
        L = self.kv_lora_rank
        if kv_cache.dim() == 4:
            kv_cache = kv_cache.squeeze(1)
        ws_bytes = self.workspace_bytes(bsz, smax)
        # A single split has no partial-result workspace. Match the provider's
        # None specialization rather than passing a zero-length tensor to a
        # kernel compiled to read and write intermediate results.
        is_workspace_size_zero = ws_bytes == 0
        if workspace.numel() < ws_bytes:
            raise ValueError(
                f"var-split workspace too small: {workspace.numel()} < {ws_bytes}"
            )
        if workspace.dtype != torch.int8:
            workspace = workspace.view(torch.int8)
        out = torch.empty(
            (bsz, 1, self.num_heads, L), dtype=out_dtype, device=query.device
        )
        lse = torch.empty(
            (bsz, 1, self.num_heads), dtype=torch.float32, device=query.device
        )
        self._kernel(query.dtype, out_dtype, is_workspace_size_zero)(
            query[..., :L],
            query[..., L:],
            kv_cache[:, :, :L],
            kv_cache[:, :, L:],
            block_tables,
            out,
            lse,
            None if is_workspace_size_zero else workspace[:ws_bytes],
            Int32(smax),
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32),
            None,
            (
                causal_seqs
                if causal_seqs.dtype == torch.int32
                else causal_seqs.to(torch.int32)
            ),
            Int32(cp_rank),
            block_split_kvs,
            Float32(softmax_scale),
            Float32(1.0),
            Float32(
                1.0 / math.log2(math.e)
            ),  # natural-log LSE, as the stock DCP call returns
        )
        return out, lse
