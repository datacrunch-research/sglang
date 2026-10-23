"""The barrier after weight loading waits SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S
for the slowest TP rank: a multi-node load whose slowest rank finishes 600 s after
this one fails at the default 480 s and succeeds once the variable is raised."""

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SLOWEST_RANK_LAG = datetime.timedelta(seconds=600)


def _barrier_with_slow_rank(group, timeout, wait_all_ranks):
    # gloo's monitored_barrier raises once the timeout passes without every rank.
    if timeout < SLOWEST_RANK_LAG:
        raise RuntimeError("Rank 1 failed to pass monitoredBarrier")


class TestLoadBarrierTimeout(CustomTestCase):
    def _barrier(self):
        parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=object()))
        with (
            patch.object(load_model_utils, "get_parallel", return_value=parallel),
            patch.object(
                load_model_utils.dist, "monitored_barrier", _barrier_with_slow_rank
            ),
        ):
            load_model_utils.dist_barrier_after_load(elastic_ep_backend=None, tp_rank=0)

    def test_default_timeout_fails_a_slow_rank(self):
        # Unset inside the override, which restores the caller's value on exit.
        with envs.SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S.override(None):
            envs.SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S.clear()
            with self.assertRaisesRegex(ValueError, "now 480 s"):
                self._barrier()

    def test_raised_timeout_waits_for_a_slow_rank(self):
        with envs.SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S.override(2400):
            self._barrier()


if __name__ == "__main__":
    unittest.main()
