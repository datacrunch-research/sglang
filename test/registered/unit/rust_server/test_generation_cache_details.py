"""Cache-source metadata must survive the Python-to-Rust positional wire."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGenerationCacheDetails(CustomTestCase):
    @staticmethod
    def _request(rid, sources=(2, 3, 3)):
        tokens = array("q", [42])
        req = SimpleNamespace(
            rid=rid,
            beam_group=None,
            stream=True,
            sampling_params=SimpleNamespace(stream_interval=None),
            check_match_stop_str_prefix=lambda: False,
            output_ids=tokens,
            output_ids_through_stop=tokens,
            send_token_offset=0,
            send_output_token_logprobs_offset=0,
            finished_reason=None,
            finished_output=False,
            finished_len=None,
            origin_input_ids=list(range(10)),
            reasoning_tokens=0,
            cached_tokens=sum(sources),
            cached_tokens_device=sources[0],
            cached_tokens_host=sources[1],
            cached_tokens_storage=sources[2],
            get_mm_token_counts=lambda: (0, 0, 0),
            retraction_count=0,
            time_stats=None,
            customized_info=None,
            weight_version_events=[],
        )
        req.finished = lambda: req.finished_reason is not None
        return req

    @staticmethod
    def _details(req):
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
        )

        context = SimpleNamespace(enable_hicache_storage=lambda: False)
        return SchedulerOutputStreamer.get_cached_tokens_details(context, req)

    @staticmethod
    def _serialize(reqs, getter):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler_components.output_streamer import (
            _GenerationStreamAccumulator,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        accumulator = _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=getter,
            is_request_level_customized_info_key=lambda key: False,
            current_weight_version="default",
            rust_server_mode=True,
        )
        for req in reqs:
            accumulator.accept(req=req)
        captured = []
        bridge = RustServer.__new__(RustServer)
        bridge.server = SimpleNamespace(
            push_decode_result_batch=lambda header, columns: (
                captured.append(msgspec.msgpack.decode(header)) or True
            )
        )
        RustServer.push_generation(
            bridge, accumulator.to_payload(dp_rank=0, is_idle_batch=False)
        )
        return captured[0]

    def test_rust_details_are_produced_only_at_first_output_and_finish(self):
        """Steady decode must skip both metadata construction and its wire column."""
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        req = self._request("cached")
        getter = Mock(side_effect=self._details)
        expected = {"device": 2, "host": 3, "storage": 3}
        self.assertEqual(self._serialize([req], getter)[18], [expected])
        self.assertEqual(getter.call_count, 1)
        req.output_ids.append(43)
        middle = self._serialize([req], getter)
        self.assertEqual(getter.call_count, 1)
        self.assertEqual(len(middle), 6)
        req.output_ids.append(44)
        req.finished_reason = FINISH_LENGTH(3)
        self.assertEqual(self._serialize([req], getter)[18], [expected])
        self.assertEqual(getter.call_count, 2)

    def test_mixed_first_middle_and_finished_rows_keep_wire_alignment(self):
        """Omitted middle metadata must not shift first/terminal rows in a batch."""
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        first, middle, last = [
            self._request(name) for name in ["first", "middle", "last"]
        ]
        middle.output_ids.append(43)
        middle.send_token_offset = 1
        last.finished_reason = FINISH_LENGTH(1)
        getter = Mock(side_effect=self._details)
        header = self._serialize([first, middle, last], getter)
        expected = {"device": 2, "host": 3, "storage": 3}
        self.assertEqual(header[0], ["first", "middle", "last"])
        self.assertEqual(header[18], [expected, None, expected])
        self.assertEqual(getter.call_count, 2)

    def test_all_cold_rust_batches_omit_the_empty_cache_detail_column(self):
        """An all-None column must retain the six-column decode fast path."""
        header = self._serialize([self._request("cold", (0, 0, 0))], self._details)
        self.assertEqual(len(header), 6)

    def test_rust_scheduler_batch_keeps_cache_details_aligned(self):
        """A real Rust-mode batch must not emit an empty cache-detail column."""
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
            _GenerationStreamAccumulator,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        cache_context = SimpleNamespace(enable_hicache_storage=lambda: False)
        accumulator = _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: (
                SchedulerOutputStreamer.get_cached_tokens_details(cache_context, req)
            ),
            is_request_level_customized_info_key=lambda key: False,
            current_weight_version=None,
            rust_server_mode=True,
        )
        for rid, sources in [("cold", (0, 0, 0)), ("warm", (2, 3, 3))]:
            tokens = array("q", [42])
            req = SimpleNamespace(
                rid=rid,
                beam_group=None,
                finished=lambda: False,
                stream=False,
                output_ids=tokens,
                output_ids_through_stop=tokens,
                send_token_offset=0,
                send_output_token_logprobs_offset=0,
                finished_reason=None,
                origin_input_ids=list(range(10)),
                reasoning_tokens=0,
                cached_tokens=sum(sources),
                cached_tokens_device=sources[0],
                cached_tokens_host=sources[1],
                cached_tokens_storage=sources[2],
                get_mm_token_counts=lambda: (0, 0, 0),
                retraction_count=0,
                time_stats=None,
                customized_info=None,
            )
            accumulator.accept(req=req)
        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
        captured = []
        bridge = RustServer.__new__(RustServer)
        bridge.server = SimpleNamespace(
            push_decode_result_batch=lambda header, columns: (
                captured.append(msgspec.msgpack.decode(header)) or True
            )
        )
        RustServer.push_generation(bridge, payload)
        self.assertEqual(captured[0][0], ["cold", "warm"])
        self.assertEqual(
            captured[0][18], [None, {"device": 2, "host": 3, "storage": 3}]
        )

    def test_details_follow_existing_columns_with_and_without_logprobs(self):
        """Details cannot occupy an old logprob slot or disappear on the fast path."""
        details = [
            None,
            {"device": 2, "host": 3, "storage": 3, "storage_backend": "test-store"},
        ]
        for include_details, logprobs in [(False, False), (True, False), (True, True)]:
            with self.subTest(include_details=include_details, logprobs=logprobs):
                captured = []

                def push(header, columns, frames=captured):
                    frames.append((msgspec.msgpack.decode(header), columns))
                    return True

                bridge = RustServer.__new__(RustServer)
                bridge.server = SimpleNamespace(push_decode_result_batch=push)
                fields = {
                    "rids": ["r0", "r1"],
                    "finished_reasons": [None, None],
                    "output_ids": [array("q", [1]), array("q", [2])],
                    "prompt_tokens": [10, 10],
                    "cached_tokens": [0, 8],
                    "reasoning_tokens": [0, 0],
                    "cached_tokens_details": details if include_details else None,
                    "output_hidden_states": None,
                }
                for family in [
                    "output_token_logprobs",
                    "input_token_logprobs",
                    "output_top_logprobs",
                    "input_top_logprobs",
                    "output_token_ids_logprobs",
                    "input_token_ids_logprobs",
                ]:
                    fields[family + "_val"] = None
                    fields[family + "_idx"] = None
                if logprobs:
                    fields["output_token_logprobs_val"] = [[-0.5], None]
                    fields["output_token_logprobs_idx"] = [[1], None]
                RustServer.push_generation(bridge, SimpleNamespace(**fields))
                header, columns = captured[0]
                self.assertEqual(
                    header[:6],
                    [["r0", "r1"], [None, None], [10, 10], [1, 1], [0, 8], [0, 0]],
                )
                self.assertEqual(columns[0], array("q", [1, 2]).tobytes())
                if include_details:
                    self.assertEqual(len(header), 19)
                    self.assertEqual(header[18], details)
                    self.assertEqual(header[6], [1, 0] if logprobs else [])
                else:
                    self.assertEqual(len(header), 6)


if __name__ == "__main__":
    unittest.main()
