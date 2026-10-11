# Mooncake cut for the Kimi-K3 release of 12 October 2026

This branch (`verda/kimi-k3-2026-10-11-mooncake-pr4509`) is the 3 October Kimi-K3 overlay (tag
`verda/kimi-k3-2026-10-03`, v0.5.21 + 17 Python files) run against this Mooncake build:

| | |
|---|---|
| Mooncake source | Venkat2811/Mooncake, branch `venkat/store-offload-evict-accounting` (kvcache-ai/Mooncake PR 4509) |
| Mooncake commit | `aa0c8530f9ca97216e4110d11cbc8527d548dee6` (upstream main `692aa0a7` merged in) |
| Wheel | `mooncake_transfer_engine-0.3.13-cp312-cp312-manylinux_2_39_x86_64.whl` (389,631,548 bytes) |
| sha256 | `936b8d8240092064976de885acb068e58c3c1a8b52ae604e80975366517883e4` |
| Built | 2026-10-11 12:21 UTC in `lmsysorg/sglang:v0.5.21-cu130` (x86_64, Python 3.12) |
| CMake | `-DUSE_HTTP=ON -DUSE_CUDA=ON -DUSE_ETCD=OFF -DSTORE_USE_ETCD=OFF -DWITH_STORE=ON -DWITH_TE=ON -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF -DBUILD_UNIT_TESTS=OFF` |

The wheel is installed at pod start (`pip install --force-reinstall --no-deps`) over the image's
`mooncake-transfer-engine-cuda13` 0.3.13 in the engines, the store master and the store owners; the image's SGLang
checkout stays editable and the overlay is applied to it in place.

`USE_CUDA=ON` is required for the engines: without it the transfer engine has no DMA-BUF path and registers GPU
memory with `ibv_reg_mr`, which needs `nvidia_peermem`; on hosts without peermem the prefill-to-decode KV hand-off
fails. With it, `WITH_NVIDIA_PEERMEM=0` selects DMA-BUF registration.

No SGLang change is needed for this build. The Python bindings SGLang calls are the same as in 0.3.13:
`MooncakeDistributedStore.setup` gained one trailing keyword (`enable_embedded_master`, default false),
`TransferEngine.transfer_sync_write` and `batch_transfer_sync_write` gained a trailing `nic_hint` (default empty),
`batch_probe_key` is new and unused here; `setup_dummy`, `register_buffer`, `put`, `get`, `is_exist`, `remove`,
`remove_all`, `get_size`, `batch_put_from`, `batch_put_from_multi_buffers`, `batch_get_into`,
`batch_get_into_multi_buffers`, `batch_get_into_multi_buffer_ranges`, `batch_get_session_start`/`_end`,
`batch_is_exist`, `MooncakeHostMemAllocator.alloc`/`free` and `ReplicateConfig` are unchanged.
