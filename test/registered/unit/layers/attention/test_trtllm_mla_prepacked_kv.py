"""trtllm_mla ragged prefill: K/V that the backend packed to unit-scale FP8.

With an FP8 KV cache, ``TRTLLMMLABackend.pack_prefix_chunk_kv`` packs the
kv_b_proj output and k_pe into contiguous FP8 K/V at unit scale, and
``_quantize_fp8_qkv`` must leave those tensors alone. The scales it returns set
BMM1 (``q_scale * k_scale * softmax_scale``) and BMM2 (``v_scale``), so for
unit-scale data they have to be 1.0 whatever the checkpoint's k_scale/v_scale.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


def _layer(k_scale=0.5, v_scale=2.0):
    """A layer whose checkpoint carries calibrated (non-unit) KV scales."""
    return SimpleNamespace(
        k_scale_float=k_scale,
        v_scale_float=v_scale,
        k_scale=torch.tensor(k_scale),
        v_scale=torch.tensor(v_scale),
    )


def _qkv(kv_dtype):
    torch.manual_seed(0)
    q = torch.randn(6, 4, 192, dtype=torch.bfloat16)
    k = torch.randn(6, 4, 192, dtype=torch.bfloat16).to(kv_dtype)
    v = torch.randn(6, 4, 128, dtype=torch.bfloat16).to(kv_dtype)
    return q, k, v


def _fake_scaled_fp8_quant(x, scale):
    return (x.float() / scale).to(FP8), scale


class TestQuantizeFp8Qkv(CustomTestCase):
    def test_prepacked_kv_passes_through_with_unit_scales(self):
        """Packed K/V are unit-scale FP8: no second quantization, and the
        returned scales are 1.0 so the checkpoint's 0.5 / 2.0 never rescale
        BMM1 / BMM2 of data that was not divided by them."""
        q, k, v = _qkv(FP8)
        with patch.object(trtllm_mla_backend, "scaled_fp8_quant") as quant:
            q8, k8, v8, k_scale, v_scale = trtllm_mla_backend._quantize_fp8_qkv(
                q, k, v, _layer()
            )
        quant.assert_not_called()
        self.assertIs(k8, k)
        self.assertIs(v8, v)
        self.assertEqual(q8.dtype, FP8)
        self.assertEqual((k_scale, v_scale), (1.0, 1.0))

    def test_model_dtype_kv_keeps_the_checkpoint_scales(self):
        """Unpacked (bf16) K/V are still divided by the checkpoint scales, and
        those scales are returned for BMM1 / BMM2, as before."""
        q, k, v = _qkv(torch.bfloat16)
        with patch.object(
            trtllm_mla_backend,
            "scaled_fp8_quant",
            side_effect=_fake_scaled_fp8_quant,
        ) as quant:
            _, k8, v8, k_scale, v_scale = trtllm_mla_backend._quantize_fp8_qkv(
                q, k, v, _layer()
            )
        self.assertEqual(quant.call_count, 2)
        self.assertEqual((k8.dtype, k8.shape), (FP8, k.shape))
        self.assertEqual((v8.dtype, v8.shape), (FP8, v.shape))
        self.assertEqual((k_scale, v_scale), (0.5, 2.0))


if __name__ == "__main__":
    unittest.main()
