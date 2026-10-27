"""Rank-local packed ids give FlashInfer's routed MXFP4 MoE the same output.

Under EP the fused Kimi-K3 route+quant launch can write the rank's experts as
0..count-1 and any other expert as count + slot, and the runner then passes
num_experts = count + top_k with offset 0. That only changes which routing-kernel
tier FlashInfer runs; the (token, local expert, weight) entries reaching the
GEMMs, and so the output, must stay bitwise equal to the global-id call. Needs
SM100 for the trtllm-gen MoE; weights are random bytes, which is enough to
compare two routings of the same layer.

With a ``num_valid`` scalar (CUDA-graph padding) the rows at and past it are
masked before the rank-local remap: id -1 / weight 0, packed as a non-local id
on every rank. The real rows must keep the unmasked output bitwise, the padded
rows must contribute nothing, and the masked rank-local call must equal the
masked global-id call.
"""

import unittest

import torch

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.srt.layers.moe.route_quant_handoff import rank_local_num_experts
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584
INTERMEDIATE = 384


class TestRankLocalRoutedMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("trtllm-gen routed MoE requires SM100+")
        if not moe_route_quant_fused.available():
            raise unittest.SkipTest("fused route+quant kernel unavailable")

    def test_routed_moe_output_unchanged(self):
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import RoutingMethodType

        local_entries = 0
        for ep_size in (2, 4, 8, 16):
            count = NUM_EXPERTS // ep_size
            g = torch.Generator(device="cuda").manual_seed(ep_size)

            def rand_bytes(*shape, low=0, high=256):
                return torch.randint(
                    low, high, shape, device="cuda", dtype=torch.uint8, generator=g
                )

            # e2m1 weights are any byte; UE8M0 scales 2^-10..2^-6 keep it finite
            w13 = rand_bytes(count, 2 * INTERMEDIATE, HIDDEN // 2)
            w13_scale = rand_bytes(
                count, 2 * INTERMEDIATE, HIDDEN // 32, low=117, high=122
            )
            w2 = rand_bytes(count, HIDDEN, INTERMEDIATE // 2)
            w2_scale = rand_bytes(count, HIDDEN, INTERMEDIATE // 32, low=117, high=122)

            def moe(packed, x_q, x_s, num_experts, local_expert_offset):
                out = torch.empty(
                    packed.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16
                )
                trtllm_fp4_block_scale_routed_moe(
                    topk_ids=packed,
                    routing_bias=None,
                    hidden_states=x_q,
                    hidden_states_scale=x_s.view(torch.float8_e4m3fn),
                    gemm1_weights=w13,
                    gemm1_weights_scale=w13_scale.view(torch.float8_e4m3fn),
                    gemm1_bias=None,
                    gemm1_alpha=None,
                    gemm1_beta=None,
                    gemm1_clamp_limit=None,
                    gemm2_weights=w2,
                    gemm2_weights_scale=w2_scale.view(torch.float8_e4m3fn),
                    gemm2_bias=None,
                    output1_scale_scalar=None,
                    output1_scale_gate_scalar=None,
                    output2_scale_scalar=None,
                    num_experts=num_experts,
                    top_k=TOPK,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=INTERMEDIATE,
                    local_expert_offset=local_expert_offset,
                    local_num_experts=count,
                    routed_scaling_factor=None,
                    routing_method_type=RoutingMethodType.TopK.value,
                    output=out,
                )
                return out

            for rank in sorted({0, ep_size // 2, ep_size - 1}):
                offset = rank * count
                # FlashInfer routing: block kernel up to 4 tokens, dyn-block or
                # cluster up to 16, cluster above; 64 is the fused launch's cap
                for num_tokens in (1, 4, 5, 8, 16, 17, 24, 64):
                    torch.manual_seed(num_tokens + rank)
                    scores = torch.randn(
                        num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
                    )
                    bias = torch.randn(NUM_EXPERTS, device="cuda")
                    x = torch.randn(
                        num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16
                    )
                    args = (scores, bias, x, TOPK)
                    kwargs = dict(
                        renormalize=True, routed_scaling_factor=2.5, apply_scale=False
                    )
                    _, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
                        *args, **kwargs
                    )
                    local_packed = moe_route_quant_fused.route_quant_fused(
                        *args,
                        **kwargs,
                        local_expert_offset=offset,
                        num_local_experts=count,
                    )[2]
                    ref = moe(packed, x_q, x_s, NUM_EXPERTS, offset)
                    out = moe(
                        local_packed,
                        x_q,
                        x_s,
                        rank_local_num_experts(offset, count, NUM_EXPERTS, TOPK),
                        0,
                    )
                    tag = f"ep={ep_size} rank={rank} m={num_tokens}"
                    self.assertTrue(bool(torch.isfinite(ref.float()).all()), msg=tag)
                    self.assertTrue(torch.equal(out, ref), msg=tag)
                    local_entries += int(
                        ((ids >= offset) & (ids < offset + count)).sum()
                    )
        # not vacuous: rows did reach the rank's experts
        self.assertGreater(local_entries, 0)

    def test_padded_rows_masked_before_rank_local_remap(self):
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import RoutingMethodType

        checked = 0
        for ep_size in (8, 16):
            count = NUM_EXPERTS // ep_size
            compact = rank_local_num_experts(0, count, NUM_EXPERTS, TOPK)
            g = torch.Generator(device="cuda").manual_seed(100 + ep_size)

            def rand_bytes(*shape, low=0, high=256):
                return torch.randint(
                    low, high, shape, device="cuda", dtype=torch.uint8, generator=g
                )

            w13 = rand_bytes(count, 2 * INTERMEDIATE, HIDDEN // 2)
            w13_scale = rand_bytes(
                count, 2 * INTERMEDIATE, HIDDEN // 32, low=117, high=122
            )
            w2 = rand_bytes(count, HIDDEN, INTERMEDIATE // 2)
            w2_scale = rand_bytes(count, HIDDEN, INTERMEDIATE // 32, low=117, high=122)

            def moe(packed, x_q, x_s, num_experts, local_expert_offset):
                out = torch.empty(
                    packed.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16
                )
                trtllm_fp4_block_scale_routed_moe(
                    topk_ids=packed,
                    routing_bias=None,
                    hidden_states=x_q,
                    hidden_states_scale=x_s.view(torch.float8_e4m3fn),
                    gemm1_weights=w13,
                    gemm1_weights_scale=w13_scale.view(torch.float8_e4m3fn),
                    gemm1_bias=None,
                    gemm1_alpha=None,
                    gemm1_beta=None,
                    gemm1_clamp_limit=None,
                    gemm2_weights=w2,
                    gemm2_weights_scale=w2_scale.view(torch.float8_e4m3fn),
                    gemm2_bias=None,
                    output1_scale_scalar=None,
                    output1_scale_gate_scalar=None,
                    output2_scale_scalar=None,
                    num_experts=num_experts,
                    top_k=TOPK,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=INTERMEDIATE,
                    local_expert_offset=local_expert_offset,
                    local_num_experts=count,
                    routed_scaling_factor=None,
                    routing_method_type=RoutingMethodType.TopK.value,
                    output=out,
                )
                return out

            for rank in sorted({0, ep_size // 2, ep_size - 1}):
                offset = rank * count
                # graph sizes with 1 .. 7 padded rows, block / dyn-block /
                # cluster routing tiers, and the fused launch's 64-token cap
                for num_tokens, num_valid in (
                    (4, 3),
                    (8, 5),
                    (16, 9),
                    (24, 17),
                    (32, 31),
                    (64, 57),
                    (24, 24),
                ):
                    torch.manual_seed(1000 * ep_size + 10 * num_tokens + rank)
                    scores = torch.randn(
                        num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
                    )
                    bias = torch.randn(NUM_EXPERTS, device="cuda")
                    x = torch.randn(
                        num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16
                    )
                    kwargs = dict(
                        renormalize=True, routed_scaling_factor=2.5, apply_scale=False
                    )
                    # reference: every row real, global ids
                    _, ids_ref, packed_ref, x_q_ref, x_s_ref = (
                        moe_route_quant_fused.route_quant_fused(
                            scores, bias, x, TOPK, **kwargs
                        )
                    )
                    ref = moe(packed_ref, x_q_ref, x_s_ref, NUM_EXPERTS, offset)
                    # padded rows carry stale state: non-finite router scores
                    pad_scores = scores.clone()
                    pad_scores[num_valid:] = float("nan")
                    nv = torch.tensor([num_valid], device="cuda", dtype=torch.int32)
                    _, ids_g, packed_g, x_q, x_s = (
                        moe_route_quant_fused.route_quant_fused(
                            pad_scores, bias, x, TOPK, **kwargs, num_valid=nv
                        )
                    )
                    _, ids_l, packed_l, _, _ = moe_route_quant_fused.route_quant_fused(
                        pad_scores,
                        bias,
                        x,
                        TOPK,
                        **kwargs,
                        local_expert_offset=offset,
                        num_local_experts=count,
                        num_valid=nv,
                    )
                    tag = f"ep={ep_size} rank={rank} m={num_tokens} valid={num_valid}"
                    # real rows: routing untouched by the mask
                    self.assertTrue(
                        torch.equal(ids_g[:num_valid], ids_ref[:num_valid]), msg=tag
                    )
                    self.assertTrue(torch.equal(ids_l, ids_g), msg=tag)
                    self.assertTrue(
                        torch.equal(packed_g[:num_valid], packed_ref[:num_valid]),
                        msg=tag,
                    )
                    # padded rows: id -1, weight 0; global packing 0xFFFF0000,
                    # rank-local packing count + slot (non-local), below compact
                    pad = num_tokens - num_valid
                    if pad:
                        self.assertTrue(bool((ids_g[num_valid:] == -1).all()), msg=tag)
                        self.assertTrue(
                            bool((packed_g[num_valid:] == -65536).all()), msg=tag
                        )
                        slot = torch.arange(TOPK, device="cuda", dtype=torch.int32)
                        want = ((count + slot) << 16).expand(pad, TOPK)
                        self.assertTrue(
                            torch.equal(packed_l[num_valid:], want), msg=tag
                        )
                    self.assertTrue(
                        bool(((packed_l >> 16) & 0xFFFF).lt(compact).all()), msg=tag
                    )
                    out_g = moe(packed_g, x_q, x_s, NUM_EXPERTS, offset)
                    out_l = moe(packed_l, x_q, x_s, compact, 0)
                    self.assertTrue(bool(torch.isfinite(ref.float()).all()), msg=tag)
                    # rank-local and global packing of the masked rows agree
                    self.assertTrue(torch.equal(out_l, out_g), msg=tag)
                    # real rows bitwise equal to the all-real reference
                    self.assertTrue(
                        torch.equal(out_l[:num_valid], ref[:num_valid]), msg=tag
                    )
                    # padded rows contribute nothing
                    self.assertTrue(bool((out_l[num_valid:] == 0).all()), msg=tag)
                    checked += 1
        self.assertEqual(checked, 2 * 3 * 7)


if __name__ == "__main__":
    unittest.main()
