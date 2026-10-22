"""Kimi K3's own kernels for a layer's sums and gathers serve its
attention-residual bank only: the standard residual path keeps the
boundary's collectives, even where those kernels run.
With SGLANG_K3_SP_ATTN_RES the bank and the stream stay on each rank's shard
across SP-MoE layers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
from sglang.srt.layers.layer_boundary import BatchVariant, layer_stack
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.ops import attn_tp_gather_input, keep_output
from sglang.srt.layers.layer_boundary.residual.attn_bank import (
    AttnBank,
    AttnBankOutputRead,
)
from sglang.srt.models import kimi_k3
from sglang.test.boundary_fixtures import parallel_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A latent MoE on every layer after the first.
CONFIG = SimpleNamespace(
    is_moe=True,
    num_experts=8,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    routed_expert_hidden_size=2,
)


def moe_layer(*, bank, sp_moe, all_reduce_fusion):
    """A K3 latent MoE layer's attributes that its stage declarations read."""
    layer = kimi_k3.KimiK3DecoderLayer.__new__(kimi_k3.KimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_residuals = bank
    layer.is_block_write_layer = False
    layer._sp_moe = sp_moe
    layer.all_reduce_fusion = all_reduce_fusion
    layer.input_layernorm = fixture.Norm()
    layer.post_attention_layernorm = fixture.Norm()
    for name in (
        "self_attention_res_proj",
        "self_attention_res_norm",
        "mlp_res_proj",
        "mlp_res_norm",
    ):
        setattr(layer, name, None)
    return layer


def declare(layers, *, sp_moe, carries=False, config=CONFIG):
    """Declare ``layers`` (index -> layer) in one layer stack, as if the
    stack carried the bank's shards (``carries``). A bank stack ends on the
    model's final read of the bank, which gathers an SP-MoE layer's rows in
    K3's tuned all-gather."""
    bank = AttnBank()
    has_bank = any(layer.use_attn_residuals for layer in layers.values())
    # Every layer of a model with a bank block size carries the bank.
    config = SimpleNamespace(
        **{**vars(config), "attn_res_block_size": 4 if has_bank else None}
    )
    final_read = (
        AttnBankOutputRead(
            bank,
            None,
            None,
            fixture.Norm(),
            attn_tp_gather=k3_sp_collective.all_gather if sp_moe else None,
            reads_attn_tp_slices=carries,
        )
        if has_bank
        else None
    )
    with (
        patch.object(k3_sp_collective, "enabled", return_value=True),
        patch.object(k3_ar_fusion, "enabled", return_value=True),
        patch.object(kimi_k3, "_shards_moe_rows", return_value=sp_moe),
        patch.object(kimi_k3, "_carries_bank_slices", return_value=carries),
        envs.SGLANG_K3_SP_ATTN_RES.override(carries),
        fixture.planning(
            fixture.parallel_of(attn_dp=1, attn_tp=2),
            a2a=sp_moe,
            boundary_reduction="ar",
        ),
        layer_stack(final_read=final_read),
    ):
        for idx, layer in layers.items():
            layer._declare_stages(config, idx, bank)


class TestKimiK3StageKernels(CustomTestCase):
    def ffn_path(self, *, bank, sp_moe, all_reduce_fusion):
        layer = moe_layer(bank=bank, sp_moe=sp_moe, all_reduce_fusion=all_reduce_fusion)
        declare({1: layer}, sp_moe=sp_moe)
        return layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY]

    @staticmethod
    def kernels(path):
        step = path.entry.prepare.keywords["step"]
        gather = getattr(path.output_move, "keywords", {}).get("gather")
        fusions = tuple(f.run for f in step.keywords["read_fusions"])
        return step.func, fusions, gather

    def test_the_standard_path_keeps_the_boundary_collectives(self):
        for sp_moe in (True, False):
            with self.subTest(sp_moe=sp_moe):
                entry, fusions, gather = self.kernels(
                    self.ffn_path(bank=False, sp_moe=sp_moe, all_reduce_fusion=True)
                )
                self.assertIs(
                    entry,
                    comm_ops._attn_tp_reduce_scatter_update_read
                    if sp_moe
                    else comm_ops._reduce_update_read,
                )
                self.assertEqual(fusions, ())
                self.assertIsNone(gather)

    def test_the_bank_takes_the_tuned_reduce_scatter_and_gather(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=True, all_reduce_fusion=False)
        )
        self.assertIs(entry, comm_ops._attn_tp_reduce_scatter_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_reduce_scatter_add,))
        self.assertIs(gather, k3_sp_collective.all_gather)

    def test_the_bank_takes_the_fused_all_reduce(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=False, all_reduce_fusion=True)
        )
        self.assertIs(entry, comm_ops._reduce_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_all_reduce_add,))
        self.assertIsNone(gather)


class TestKimiK3BankOnShards(CustomTestCase):
    def stack(self, *, carries):
        layers = {
            idx: moe_layer(bank=True, sp_moe=True, all_reduce_fusion=False)
            for idx in (1, 2, 3)
        }
        declare(layers, sp_moe=True, carries=carries)
        return [
            (
                layer.attn_boundary.plan.paths[BatchVariant.ORDINARY],
                layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY],
            )
            for layer in layers.values()
        ]

    def test_consecutive_moe_layers_stay_on_their_shards(self):
        for idx, (attn, ffn) in enumerate(self.stack(carries=True)):
            with self.subTest(layer=idx + 1):
                # The stack's last FFN too: the final read takes its shard.
                self.assertIs(ffn.output_move, keep_output)
                step = ffn.entry.prepare.keywords["step"]
                self.assertIs(step.func, comm_ops._attn_tp_reduce_scatter_update_read)
                self.assertEqual(step.keywords["scatters_residual"], idx == 0)
                # The bank's fused reduce-scatter and read first, then K3's
                # reduce-scatter with the residual add.
                fusions = step.keywords["read_fusions"]
                self.assertEqual([f.reads for f in fusions], [True, False])
                self.assertIs(fusions[1].run, kimi_k3._k3_reduce_scatter_add)
                if idx == 0:
                    self.assertIsNone(attn.entry.input_move)
                    continue
                # Read on the shard, then gathered: by the bank's kernel that
                # does both, else K3's tuned all-gather, else the boundary's.
                self.assertIs(attn.entry.input_move.func, attn_tp_gather_input)
                self.assertIs(
                    attn.entry.input_move.keywords["gather"],
                    k3_sp_collective.all_gather,
                )
                read = attn.entry.prepare.keywords["step"]
                self.assertEqual(len(read.keywords["read_gathers"]), 1)

    def test_the_shard_mode_is_decided_at_construction(self):
        # A rank holds only its own rows of the bank: no pipeline rank after
        # it, no draft model capturing the target's hidden states, and no
        # dense layer after an MoE layer.
        cases = {
            (None, 1, 1): True,
            ("EAGLE", 1, 1): True,
            (None, 2, 1): False,
            ("EAGLE3", 1, 1): False,
            ("DFLASH", 1, 1): False,
            ("DSPARK", 1, 1): False,
            (None, 1, 2): False,
        }
        for (algorithm, pp_size, moe_layer_freq), carries in cases.items():
            config = SimpleNamespace(
                **{
                    **vars(CONFIG),
                    "attn_res_block_size": 4,
                    "num_hidden_layers": 6,
                    "moe_layer_freq": moe_layer_freq,
                }
            )
            with (
                self.subTest(
                    algorithm=algorithm,
                    pp_size=pp_size,
                    moe_layer_freq=moe_layer_freq,
                ),
                patch.object(kimi_k3, "_shards_moe_rows", return_value=True),
                envs.SGLANG_K3_SP_ATTN_RES.override(True),
                patch.object(
                    kimi_k3, "get_parallel", lambda: SimpleNamespace(pp_size=pp_size)
                ),
                patch.object(
                    kimi_k3,
                    "get_spec",
                    lambda: SimpleNamespace(speculative_algorithm=algorithm),
                ),
            ):
                self.assertEqual(
                    kimi_k3._carries_bank_slices(config, sp_collective=True), carries
                )


class TestKimiK3DenseLayer(CustomTestCase):
    def test_dense_tp_size_one_runs_the_dense_layer_on_local_rows(self):
        """With --moe-dense-tp-size 1 each DP rank may keep only its own token
        count, so the dense layer must not gather DP rows over TP: a TP-sharded
        dense layer sized that gather from local counts and hung."""
        config = SimpleNamespace(**{**vars(CONFIG), "attn_res_block_size": None})
        for dense_tp_size, group, width in ((1, "replicated", 1), (None, "tp", 4)):
            parallel = parallel_of(
                attn_dp=2,
                attn_tp=2,
                moe_dense_tp_size=dense_tp_size,
                enable_dense_mlp_attn_tp=False,
            )
            with (
                self.subTest(moe_dense_tp_size=dense_tp_size),
                patch_communicator("get_parallel", lambda: parallel),
                patch.object(kimi_k3, "get_parallel", lambda: parallel),
                patch.object(kimi_k3, "is_dp_attention_enabled", return_value=True),
                patch.object(
                    kimi_k3,
                    "resolve_linear_parallel_group",
                    lambda name: (0, {"replicated": 1, "tp": 4}[name]),
                ),
            ):
                self.assertEqual(kimi_k3._dense_mlp_parallel_group(), group)
                _, ffn = kimi_k3._stage_facts(config, 0, sp_collective=False)
                self.assertEqual(ffn.dense_tp_size, width)


if __name__ == "__main__":
    unittest.main()
