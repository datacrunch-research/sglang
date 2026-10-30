"""Multimodal support for the embedded Rust server."""

from __future__ import annotations

import copy
import importlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple, Union

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_mm, get_parallel, get_serving

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RustMmSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved parameters of the Rust MM pipeline for one model,
    consumed by the Rust worker pool (as the typed extension ``MmSpec``, see
    :meth:`RustServer._build_mm_spec`), the ``_multimodal`` parity API
    (:meth:`rust_json`) and the drain adapter
    (:meth:`RustMmProcessor.wrap_encoded`)."""

    family: str
    feature_shm: bool
    image_token_id: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    image_mean: Tuple[float, ...]
    image_std: Tuple[float, ...]
    # Which HF processor the Rust resize must reproduce bit-exactly.
    resample: str
    vision_start_token_id: Optional[int]
    vision_end_token_id: Optional[int]
    video_token_id: Optional[int]

    # Used by the drain adapter only; every other field goes to Rust.
    DRAIN_ONLY = ("vision_start_token_id", "vision_end_token_id", "video_token_id")

    @property
    def feature_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    def rust_json(self) -> str:
        """The subset `sglang_mm::registry::pipeline_from_spec` parses — the
        JSON form the ``_multimodal`` parity API takes; the server itself is
        handed the typed ``MmSpec`` instead."""
        fields = (f for f in self.__struct_fields__ if f not in self.DRAIN_ONLY)
        return msgspec.json.encode({f: getattr(self, f) for f in fields}).decode()


class RustKimiK3MmSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved parameters of the Kimi K3 Rust MM pipeline, read back from
    the Python ``KimiK3ImageProcessor`` it replaces (see
    :meth:`RustMmProcessor._resolve_kimi_k3_spec`): the NaViT fields of its
    ``KimiK3PreprocessConfig``, the ``<|media_pad|>`` id and its processor
    fingerprint go to Rust (``MmSpec.kimi_k3``); the normalization and
    transparent-background settings stay here, for the drain adapter to
    rebuild each item's ``KimiK3DeferredPreprocessing``."""

    family: str
    feature_shm: bool
    image_token_id: int
    patch_size: int
    merge_kernel_size: int
    in_patch_limit: int
    patch_limit_on_one_side: int
    fixed_output_tokens: Optional[int]
    processor_fingerprint: str
    image_mean: Tuple[float, ...]
    image_std: Tuple[float, ...]
    transparent_bg_config: Optional[Dict[str, Any]]


# The sidecar key of a Kimi K3 item's NaViT resize config
# (`sglang_mm::kimi_k3::RESIZE_CONFIG_KEY`), as
# `[num_tokens, new_width, new_height, pad_width, pad_height]`.
KIMI_K3_RESIZE_CONFIG_KEY = "kimi_k3_resize_config"


class RustMmFamily(msgspec.Struct, frozen=True, kw_only=True):
    """The Python half of one Rust MM family (an arm of
    `sglang_mm::registry::pipeline_from_spec`): which models it serves.
    Supporting a new model family = one entry in :data:`RUST_MM_FAMILIES`
    plus its Rust arm — the launch gate is data-driven."""

    name: str
    # The registered Python MM processor the Rust pipeline replaces, as
    # "module:Class". Compared by identity, so an
    # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE override still disables the Rust path.
    mm_processor: str
    # Model types whose image-only M-RoPE matches the family's fast path.
    model_types: FrozenSet[str]
    # HF image processors the Rust resize reproduces bit-exactly, each mapped
    # to the `resample` the Rust pipeline must use (see `RustMmSpec.resample`).
    image_processors: Dict[str, str]

    def serves(self, mm_processor_cls: Any, model_type: Optional[str]) -> bool:
        # The model type first: it spares importing another family's
        # processor module (Kimi K3's pulls in its model code).
        if model_type not in self.model_types:
            return False
        module_name, _, class_name = self.mm_processor.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return mm_processor_cls is cls


RUST_MM_FAMILIES: Tuple[RustMmFamily, ...] = (
    RustMmFamily(
        name="qwen_vl",
        mm_processor="sglang.srt.multimodal.processors.qwen_vl:QwenVLImageProcessor",
        model_types=frozenset(
            (
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
            )
        ),
        image_processors={
            "Qwen2VLImageProcessor": "aten_u8",
            "Qwen2VLImageProcessorFast": "aten_u8",
            "Qwen2VLImageProcessorPil": "pil",
        },
    ),
    RustMmFamily(
        name="kimi_k3",
        mm_processor="sglang.srt.multimodal.processors.kimi_k3:KimiK3ImageProcessor",
        model_types=frozenset(("kimi_k3",)),
        # No HF resize to reproduce: K3 images reach the model as raw pixels
        # and are resized on the GPU, as the Python processor defers them.
        image_processors={},
    ),
)


def rust_mm_family_for(
    mm_processor_cls: Any, model_type: Optional[str]
) -> Optional[RustMmFamily]:
    """The declared family serving this model, or ``None`` — which
    :meth:`RustServer.launch` turns into a hard error (no Python fallback)."""
    return next(
        (f for f in RUST_MM_FAMILIES if f.serves(mm_processor_cls, model_type)), None
    )


class _ModalityTable(dict):
    """``mm.meta`` modality names -> scheduler ``Modality``, resolved lazily so
    this module keeps its light import footprint."""

    def __missing__(self, name):
        from sglang.srt.managers.schedule_batch import Modality

        self.update(
            {"image": Modality.IMAGE, "video": Modality.VIDEO, "audio": Modality.AUDIO}
        )
        # An unknown name must raise here, not recurse.
        if name not in self:
            raise KeyError(name)
        return dict.__getitem__(self, name)


_MODALITIES = _ModalityTable()


def settle_shm_buffers(buffers, *, admitted: bool) -> None:
    """Decide the unlink duty for every segment a drained request parked in
    POSIX shm (any buffer that is not an inline numpy array is the Rust
    extension's ``ShmBuffer``, whatever its name). Admitted: Rust hands the
    duty to the ``ShmPointerMMData`` stubs, which unlink after the
    post-broadcast open on every rank. Rejected: unlink now. Both are
    idempotent on the Rust side. This is the drain loop's job, never a
    wrapper's, so a package's own ``_wrap_mm_result`` needs no knowledge of it."""
    import numpy as np

    for value in buffers.values():
        if isinstance(value, np.ndarray):
            continue
        if admitted:
            value.release()
        else:
            value.discard()


class MmTransportStats:
    """Zero-copy accounting at the Rust-to-Python multimodal hand-off.

    A feature tensor crosses in one of two shapes: ``inline`` -- a numpy array
    that owns the Rust vector, viewed by ``torch.from_numpy`` -- or ``shm`` -- a
    POSIX segment the worker wrote, named in the request so every TP rank maps
    it after the broadcast."""

    __slots__ = (
        "inline_features",
        "inline_bytes",
        "shm_features",
        "shm_bytes",
        "copies",
    )

    def __init__(self) -> None:
        self.inline_features = 0
        self.inline_bytes = 0
        self.shm_features = 0
        self.shm_bytes = 0
        self.copies = 0

    def record_inline(self, array, tensor) -> None:
        self.inline_features += 1
        self.inline_bytes += array.nbytes
        if array.flags["OWNDATA"] or tensor.data_ptr() != array.ctypes.data:
            self.copies += 1

    def record_shm(self, stub) -> None:
        import numpy as np

        self.shm_features += 1
        self.shm_bytes += int(np.prod(stub.shape)) * np.dtype(stub.dtype).itemsize

    def snapshot(self) -> Dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


MM_TRANSPORT_STATS = MmTransportStats()


class RustMmProcessor:
    """Builds and validates the Rust MM pipeline for one model.

    Construction registers the same ``mm_processor`` mapping the Python
    TokenizerManager would build — not to process requests (the Rust worker pool
    does that, GIL-free) but as the source of truth
    :meth:`resolve_spec` resolves the pipeline parameters from. At drain
    time :meth:`wrap_encoded` wraps the Rust-produced buffers into the
    scheduler's ``MultimodalProcessorOutput``.

    There is no Python fallback: a model without a Rust MM spec fails at launch,
    and inputs outside the pipeline's scope are rejected per request.
    """

    # Rust mm-worker threads when --mm-processor-worker-num is 0. They are
    # GIL-free, so unlike the Python processor pool more than one always helps.
    AUTO_MM_WORKERS = 8

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        processor: Any = None,
    ):
        # Lazy: this class exists only for multimodal models under
        # SGLANG_RUST_SERVER.
        from sglang.srt.managers.multimodal_processor import import_processors
        from sglang.srt.managers.tokenizer_manager import get_processor_wrapper

        self.server_args = server_args
        self.model_config = model_config
        # Worker threads == max concurrently-processed mm requests.
        self.mm_workers = get_mm().mm_processor_worker_num or self.AUTO_MM_WORKERS

        # The mapping the Python TokenizerManager builds in
        # init_tokenizer_and_processor. The caller's already-loaded HF
        # AutoProcessor is reused when available (identical construction args).
        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        self._processor = processor or get_processor_wrapper()

    def resolve_spec(self) -> Optional[RustMmSpec]:
        """The :class:`RustMmSpec` for this model, or ``None`` when it has no
        Rust pipeline (the launch gate turns that into a hard error).

        Carries only resolved settings — patch geometry, pixel limits,
        normalization, token ids — never the HF config, and is conservative by
        design: an unrecognized knob disables the Rust path rather than being
        approximated."""
        from sglang.srt.managers.multimodal_processor import get_mm_processor_cls

        hf_config = self.model_config.hf_config
        mm_processor_cls = get_mm_processor_cls(
            hf_config, model_config=self.model_config
        )
        family = rust_mm_family_for(
            mm_processor_cls, getattr(hf_config, "model_type", None)
        )
        if family is None:
            return None
        if family.name == "kimi_k3":
            return self._resolve_kimi_k3_spec(mm_processor_cls)
        image_processor = getattr(self._processor, "image_processor", None)
        resample = family.image_processors.get(type(image_processor).__name__)
        if resample is None:
            return None
        # The Rust pipeline always resizes, rescales by 1/255 and normalizes;
        # Rust's fused normalize constants assume that factor. Anything else
        # would silently produce different features.
        stages = ("do_resize", "do_rescale", "do_normalize")
        if not all(getattr(image_processor, stage, True) for stage in stages):
            return None
        if getattr(image_processor, "rescale_factor", None) != 1 / 255:
            return None

        # `--mm-process-config {"image": {...}}`: only pixel-limit overrides are
        # mirrored by Rust; anything else disables the pipeline.
        image_overrides = dict((get_mm().mm_process_config or {}).get("image", {}))
        if not set(image_overrides) <= {"min_pixels", "max_pixels"}:
            return None

        size = getattr(image_processor, "size", None) or {}
        min_pixels = image_overrides.get(
            "min_pixels",
            getattr(image_processor, "min_pixels", None) or size.get("shortest_edge"),
        )
        max_pixels = image_overrides.get(
            "max_pixels",
            getattr(image_processor, "max_pixels", None) or size.get("longest_edge"),
        )
        try:
            spec = RustMmSpec(
                family=family.name,
                feature_shm=self._use_feature_shm(),
                image_token_id=hf_config.image_token_id,
                patch_size=image_processor.patch_size,
                merge_size=image_processor.merge_size,
                temporal_patch_size=image_processor.temporal_patch_size,
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
                image_mean=tuple(float(x) for x in image_processor.image_mean),
                image_std=tuple(float(x) for x in image_processor.image_std),
                resample=resample,
                vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
                vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
                video_token_id=getattr(hf_config, "video_token_id", None),
            )
        except (AttributeError, TypeError):  # missing/odd processor attrs
            return None
        logger.info("rust server: Rust MM pipeline enabled (family=%s)", family.name)
        return spec

    def _resolve_kimi_k3_spec(self, mm_processor_cls) -> Optional[RustKimiK3MmSpec]:
        """The Kimi K3 spec, read back from the Python processor itself: it is
        built here as the TokenizerManager builds it (same configuration, no
        worker pool), so its preprocess config, ``<|media_pad|>`` id and
        processor fingerprint -- which every image identity covers -- are the
        ones a Python server with these settings would use.

        Conservative like the Qwen gate: a setting the Rust pipeline does not
        reproduce disables it rather than being approximated."""
        from sglang.srt.multimodal.transport import determine_tensor_transport_mode

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            # Python then gives items throwaway hashes instead of identities.
            return None
        media_processor = getattr(self._processor, "media_processor", None)
        media_cfg = getattr(media_processor, "media_proc_cfg", None)
        if not isinstance(media_cfg, Mapping):
            return None
        # The GPU path composites the background after the resize; a checkpoint
        # asking for another stage is not what either path reproduces.
        if media_cfg.get("transparent_bg_fill_stage", "after_resize") != "after_resize":
            return None

        processor = mm_processor_cls(
            self.model_config.hf_config,
            self.server_args,
            self._processor,
            determine_tensor_transport_mode(),
            skip_mm_pool=True,
        )
        try:
            config = processor._processor.preprocess_config
            fingerprint = processor._kimi_config_fingerprint()
            image_token_id = processor.mm_tokens.image_token_id
        finally:
            processor.shutdown()
            processor.cpu_executor.shutdown(wait=False)
        spec = RustKimiK3MmSpec(
            family="kimi_k3",
            feature_shm=self._use_feature_shm(),
            image_token_id=int(image_token_id),
            patch_size=config.patch_size,
            merge_kernel_size=config.merge_kernel_size,
            in_patch_limit=config.in_patch_limit,
            patch_limit_on_one_side=config.patch_limit_on_one_side,
            fixed_output_tokens=config.fixed_output_tokens,
            processor_fingerprint=fingerprint,
            image_mean=tuple(config.image_mean),
            image_std=tuple(config.image_std),
            transparent_bg_config=copy.deepcopy(config.transparent_bg_config),
        )
        logger.info("rust server: Rust MM pipeline enabled (family=kimi_k3)")
        return spec

    def _use_feature_shm(self) -> bool:
        """Whether to park feature buffers in POSIX shm rather than inline.

        On exactly when the drained request is broadcast across TP ranks *and*
        the receiver's ``unwrap_shm_features`` will materialize the stubs (its
        gates: non-default tensor transport, no ``skip_tokenizer_init``).

        Inline, the whole ~20 MB/image buffer rides ``broadcast_pyobj`` serially
        on the scheduler loop, so ranks 1..n start the TP-sharded ViT ~30 ms
        after rank 0 and every rank then stalls that long at the first
        collective. With shm the broadcast carries a ~100-byte stub and all ranks
        map in parallel — the transport the Python TokenizerManager already uses.
        Single-rank serving stays inline, where shm would only add a copy.
        """
        from sglang.srt.multimodal.transport import (
            determine_tensor_transport_mode,
        )

        return (
            get_parallel().tp_size > 1
            and determine_tensor_transport_mode() != "default"
            and not get_serving().skip_tokenizer_init
        )

    @staticmethod
    def wrap_encoded(spec: Union[RustMmSpec, RustKimiK3MmSpec], buffers):
        """Drain-time adapter: wrap one request's ``mm.*`` buffers (a
        ``{name: numpy array | ShmBuffer}`` mapping from ``recv_requests``) into
        the scheduler's ``MultimodalProcessorOutput``. Wrapping only -- load,
        resize, patchify, token expansion and M-RoPE all ran in Rust.

        The buffers are ``mm.feature.{i}`` per item (shaped ``[rows,
        feature_dim]`` f32, inline or shm), ``mm.mrope`` (``[3, seq_len]``
        int64), and ``mm.meta``, a msgpack sidecar decoding to ``{"items":
        [{"modality", "hash", "offsets", "model_specific_data"}, ...],
        "token_ids", "mrope_delta"}`` with the items in buffer order. External
        model packages receive the same layout and wrap it their own way.

        Runs on the scheduler loop, so it must stay copy-free *and* hash-free:
        the inline numpy arrays own the Rust buffers, ``torch.from_numpy`` just
        views them, and each item's ``hash`` is worker-precomputed so
        ``set_pad_value`` skips ``hash_feature``. Any per-byte work here — memcpy,
        sha256, tens of MB per image-heavy request — measurably inflates every
        running request's inter-token latency."""
        if spec.family == "kimi_k3":
            return RustMmProcessor._wrap_kimi_k3(spec, buffers)

        import msgspec
        import torch

        from sglang.srt.managers.schedule_batch import (
            MultimodalProcessorOutput,
        )

        # Ownership of the segments is not decided here: the drain loop
        # releases or discards every `ShmBuffer` once it knows whether the
        # request is admitted (`settle_shm_buffers`). A failure part-way only
        # has to close the stubs built so far, which hold an open mapping each.
        stubs: list = []
        try:
            # Decoded straight from the numpy buffer (a memoryview over the
            # Rust-owned bytes), not from a `.tobytes()` copy of it.
            meta = msgspec.msgpack.decode(memoryview(buffers["mm.meta"]))
            items = [
                RustMmProcessor._wrap_item(
                    index=index, item=item, buffers=buffers, stubs=stubs
                )
                for index, item in enumerate(meta["items"])
            ]
            if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
                for item in items:
                    item.set_pad_value()
            output = MultimodalProcessorOutput(
                mm_items=items,
                im_token_id=spec.image_token_id,
                im_start_id=spec.vision_start_token_id,
                im_end_id=spec.vision_end_token_id,
                video_token_id=spec.video_token_id,
                mrope_positions=torch.from_numpy(buffers["mm.mrope"]),
                mrope_position_delta=torch.tensor(
                    [[meta["mrope_delta"]]], dtype=torch.long
                ),
            )
        except BaseException:
            for stub in stubs:
                stub.close_and_unlink()
            raise
        return output

    @staticmethod
    def _wrap_feature(feature, *, precomputed_hash: int, stubs: list):
        """One ``mm.feature.{i}`` buffer as the item's feature: an inline
        numpy array viewed by ``torch.from_numpy``, or a shm segment as its
        ``ShmPointerMMData`` stub, in the buffer's own dtype."""
        import numpy as np
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData

        if isinstance(feature, np.ndarray):
            array = feature
            feature = torch.from_numpy(array)
            MM_TRANSPORT_STATS.record_inline(array=array, tensor=feature)
            return feature
        MM_TRANSPORT_STATS.record_shm(stub=feature)
        # The worker placed this item's buffer in a named POSIX segment
        # (see `_use_feature_shm`). Build the stub in its
        # post-`__setstate__` form: rank 0 never pickle-roundtrips its
        # own copy, and `materialize()` needs the mapped view.
        stub = ShmPointerMMData.__new__(ShmPointerMMData)
        stub.__setstate__(
            {
                "shm_name": feature.name,
                "shape": tuple(feature.shape),
                "dtype": torch.from_numpy(np.empty(0, dtype=feature.dtype)).dtype,
                "precomputed_hash": precomputed_hash,
            }
        )
        stubs.append(stub)
        return stub

    @staticmethod
    def _wrap_kimi_k3(spec: RustKimiK3MmSpec, buffers):
        """Drain-time adapter for Kimi K3. Each ``mm.feature.{i}`` is the
        source image as ``[C, H, W]`` uint8 (alpha kept), and each sidecar item
        carries its grid, its NaViT resize config and its 32-byte identity.
        Builds the item the Python ``KimiK3ImageProcessor`` builds on its
        deferred path (``compose_request`` over a deferred artifact): the raw
        feature, ``image_grid_thw``, a ``KimiK3DeferredPreprocessing`` the
        model finishes on the GPU, and ``set_identity``, which derives the
        wide pad sentinels. No M-RoPE: K3 positions are 1-D."""
        import msgspec
        import torch

        from sglang.srt.managers.schedule_batch import (
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )
        from sglang.srt.multimodal.kimi_k3_image_processing import (
            DEFERRED_PREPROCESSING_KEY,
            KimiK3DeferredPreprocessing,
        )

        stubs: list = []
        try:
            meta = msgspec.msgpack.decode(memoryview(buffers["mm.meta"]))
            items = []
            for index, item in enumerate(meta["items"]):
                data = item["model_specific_data"]
                num_tokens, new_width, new_height, pad_width, pad_height = data[
                    KIMI_K3_RESIZE_CONFIG_KEY
                ]
                mm_item = MultimodalDataItem(
                    modality=_MODALITIES[item["modality"]],
                    feature=RustMmProcessor._wrap_feature(
                        buffers[f"mm.feature.{index}"],
                        precomputed_hash=item["hash"],
                        stubs=stubs,
                    ),
                    offsets=[tuple(span) for span in item["offsets"]],
                    model_specific_data={
                        "image_grid_thw": torch.tensor(
                            [data["image_grid_thw"]], dtype=torch.int64
                        ),
                        DEFERRED_PREPROCESSING_KEY: KimiK3DeferredPreprocessing(
                            backend="gpu",
                            image_mean=list(spec.image_mean),
                            image_std=list(spec.image_std),
                            transparent_bg_config=spec.transparent_bg_config,
                            resize_config={
                                "num_tokens": num_tokens,
                                "new_width": new_width,
                                "new_height": new_height,
                                "pad_width": pad_width,
                                "pad_height": pad_height,
                            },
                        ),
                    },
                )
                mm_item.set_identity(bytes(item["identity"]))
                items.append(mm_item)
            output = MultimodalProcessorOutput(
                mm_items=items, im_token_id=spec.image_token_id
            )
        except BaseException:
            for stub in stubs:
                stub.close_and_unlink()
            raise
        return output

    @staticmethod
    def _wrap_item(*, index: int, item: dict, buffers, stubs: list):
        import torch

        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        feature = RustMmProcessor._wrap_feature(
            buffers[f"mm.feature.{index}"], precomputed_hash=item["hash"], stubs=stubs
        )
        t, h, w = item["model_specific_data"]["image_grid_thw"]
        return MultimodalDataItem(
            modality=_MODALITIES[item["modality"]],
            feature=feature,
            hash=item["hash"],
            offsets=[tuple(span) for span in item["offsets"]],
            model_specific_data={
                "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.long)
            },
        )
