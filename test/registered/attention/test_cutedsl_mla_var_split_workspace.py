"""CPU regression checks for the CuTe MLA workspace specialization contract."""

import ast
import importlib.metadata
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_launcher():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/layers/attention/cutedsl_mla_var_split.py"
    )
    spec = importlib.util.spec_from_file_location("var_split_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed_workspace_size():
    # Read the installed provider's actual sizing function, without importing
    # its CUDA compilation stack into this CPU regression test.
    try:
        dist = importlib.metadata.distribution("flashinfer-python")
    except importlib.metadata.PackageNotFoundError as exc:
        raise unittest.SkipTest("FlashInfer workspace contract is required") from exc
    path = dist.locate_file(
        "flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "BlackwellMultiHeadLatentAttentionForwardFP16"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_workspace_size"
    )
    fn.decorator_list = []
    header = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = ast.fix_missing_locations(ast.Module(body=[header, fn], type_ignores=[]))
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["get_workspace_size"]


class Float32:
    width = 32

    def __new__(cls, value):
        return float(value)


class TestVarSplitWorkspace(unittest.TestCase):
    def setUp(self):
        self.module = load_launcher()
        self.compiles, self.launches = [], []
        sizing = installed_workspace_size()

        def compile_kernel(**kwargs):
            self.compiles.append(kwargs)

            def launch(*args):
                self.launches.append(
                    {
                        "workspace": args[7],
                        "compiled_zero": kwargs["is_workspace_size_zero"],
                    }
                )

            return launch

        fake = SimpleNamespace(
            BlackwellMultiHeadLatentAttentionForwardFP16=SimpleNamespace(
                get_workspace_size=sizing
            ),
            BlackwellMultiHeadLatentAttentionForwardFP8=object(),
            _get_compiled_mla_kernel=compile_kernel,
        )
        self.launcher = self.module.CuteDslMLAVarSplitDecode.__new__(
            self.module.CuteDslMLAVarSplitDecode
        )
        self.launcher.__dict__.update(
            _impl=fake,
            _max_splits=32,
            _arch=None,
            num_sms=148,
            num_heads=96,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=64,
            cp_world=8,
            enable_pdl=False,
            _kernels={},
        )
        self.addCleanup(patch.stopall)
        patch.dict(
            sys.modules, {"cutlass": SimpleNamespace(Float32=Float32, Int32=int)}
        ).start()
        patch.object(
            self.module, "_skip_empty_split_stores_enabled", return_value=False
        ).start()
        patch.dict("os.environ", {"SGLANG_CUTEDSL_MLA_VAR_SPLIT_MAX": "x4"}).start()

    def run_shape(self, batch):
        smax = self.module.smax_for(batch, 148)
        size = self.launcher.workspace_bytes(batch, smax)
        self.launcher.run(
            query=torch.empty((batch, 1, 96, 576), dtype=torch.float8_e4m3fn),
            kv_cache=torch.empty((1, 1, 64, 576), dtype=torch.float8_e4m3fn),
            workspace=torch.empty(size, dtype=torch.int8),
            block_tables=torch.zeros((batch, 1), dtype=torch.int32),
            seq_lens=torch.ones(batch, dtype=torch.int32),
            causal_seqs=torch.ones(batch, dtype=torch.int32),
            cp_rank=0,
            block_split_kvs=torch.ones(batch, dtype=torch.int32),
            smax=smax,
            softmax_scale=1.0,
        )
        return size

    def test_single_split_compiles_zero_workspace_variant(self):
        self.assertEqual(self.run_shape(512), 0)
        self.assertTrue(self.compiles[-1]["is_workspace_size_zero"])

    def test_single_split_passes_none_instead_of_empty_tensor(self):
        self.run_shape(512)
        self.assertIsNone(self.launches[-1]["workspace"])

    def test_cache_separates_zero_and_nonzero_workspace(self):
        for batch in (64, 512, 128, 512):
            self.run_shape(batch)
        self.assertEqual(
            [c["is_workspace_size_zero"] for c in self.compiles], [False, True]
        )
        self.assertEqual(len(self.launches), 4)

    def test_64_and_128_retain_workspace_variant(self):
        for batch in (64, 128):
            with self.subTest(batch=batch):
                size = self.run_shape(batch)
                self.assertGreater(size, 0)
                self.assertFalse(self.launches[-1]["compiled_zero"])
                self.assertEqual(self.launches[-1]["workspace"].numel(), size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
