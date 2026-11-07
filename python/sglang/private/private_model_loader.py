"""SGLang private model loader: stream safetensors weights from RustFS straight into GPU memory.

Enabled with `--load-format private` (SGLang imports
`sglang.private.private_model_loader.PrivateModelLoader`). `--model-path` is a
local directory with the model's metadata only (config, tokenizer, remote code
and `model.safetensors.index.json`; see `rustfs_prepare.py`). The shards are
never written to disk, tmpfs or host memory: each GPU RDMA-fetches its share
of the checkpoint from RustFS over NIXL (GPUDirect) and the ranks exchange the
units over NCCL; see `rustfs_gpu_stream.py`.

Configuration (environment):
  RUSTFS_LOADER_ENDPOINTS    comma-separated http://host:port list (required)
  RUSTFS_LOADER_BUCKET       bucket (required)
  RUSTFS_LOADER_PREFIX       object prefix of the model directory (required)
  RUSTFS_LOADER_ACCESS_KEY / RUSTFS_LOADER_SECRET_KEY   credentials (required)
  RUSTFS_LOADER_RAILS        UCX device per local GPU index, comma-separated
                             (default: one shared list from RUSTFS_LOADER_DEVICES, else all)
  RUSTFS_LOADER_UNIT_MB      target unit size (default 512)
  RUSTFS_LOADER_RING_GB      per-GPU staging ring (default 16)
  RUSTFS_LOADER_CHUNK_MB     bytes per ranged NIXL GET (default 128)
  RUSTFS_LOADER_FETCH_UNITS  units each GPU fetches concurrently (default 4)
  RUSTFS_LOADER_STREAMS      ranged GETs per unit in flight (default 8)
  RUSTFS_LOADER_CLONE        "0" yields views into the ring instead of copies (default 1)
"""

import json
import logging
import os
import time
from typing import Generator, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.model_loader.loader import DefaultModelLoader

logger = logging.getLogger(__name__)

INDEX_FILE = "model.safetensors.index.json"


def _env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        raise ValueError(f"{name} must be set for --load-format private (RustFS loader)")
    return value


def _rail(device_index):
    rails = [r.strip() for r in _env("RUSTFS_LOADER_RAILS", "").split(",") if r.strip()]
    if rails:
        return rails[device_index % len(rails)]
    return _env("RUSTFS_LOADER_DEVICES", "")


class PrivateModelLoader(DefaultModelLoader):
    """DefaultModelLoader whose safetensors weights stream from RustFS into GPU memory."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config) -> None:
        pass  # the weights stream in during load

    def _prepare_weights(self, model_name_or_path, revision, fall_back_to_pt, allow_patterns_overrides=None):
        with open(os.path.join(model_name_or_path, INDEX_FILE)) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return model_name_or_path, [os.path.join(model_name_or_path, s) for s in shards], True

    def _get_weights_iterator(self, source, *, resolved_source=None, **_kwargs) -> Generator[Tuple[str, torch.Tensor], None, None]:
        from sglang.private.rustfs_gpu_stream import GpuWeightStream
        from sglang.private.rustfs_nixl import RustfsNixlClient

        folder, files, _ = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt, source.allow_patterns_overrides
        )
        prefix = _env("RUSTFS_LOADER_PREFIX", required=True).rstrip("/")
        keys = [f"{prefix}/{os.path.relpath(f, folder)}" for f in files]
        world = get_world_group()
        device = torch.cuda.current_device()
        client = RustfsNixlClient(
            [e.strip() for e in _env("RUSTFS_LOADER_ENDPOINTS", required=True).split(",") if e.strip()],
            _env("RUSTFS_LOADER_ACCESS_KEY", required=True),
            _env("RUSTFS_LOADER_SECRET_KEY", required=True),
            devices=_rail(device),
            agent_name=f"sglang-loader-{os.uname().nodename}-{os.getpid()}-g{device}",
        )
        mb = 1 << 20
        stream = GpuWeightStream(
            client, _env("RUSTFS_LOADER_BUCKET", required=True), keys,
            group=world.device_group, cpu_group=world.cpu_group,
            unit_bytes=int(_env("RUSTFS_LOADER_UNIT_MB", "512")) * mb,
            ring_bytes=int(_env("RUSTFS_LOADER_RING_GB", "16")) << 30,
            chunk_bytes=int(_env("RUSTFS_LOADER_CHUNK_MB", "128")) * mb,
            fetch_units=int(_env("RUSTFS_LOADER_FETCH_UNITS", "4")),
            streams_per_unit=int(_env("RUSTFS_LOADER_STREAMS", "8")),
            clone=_env("RUSTFS_LOADER_CLONE", "1") != "0",
        )
        weights = iter(stream)
        if self.load_config.draft_model_idx is not None:
            return self._filter_mtp_weights(weights, source.prefix, self.load_config.draft_model_idx)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, tensor) for (name, tensor) in weights)
