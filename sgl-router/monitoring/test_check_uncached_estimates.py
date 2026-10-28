# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for check_uncached_estimates.py: python3 -m unittest discover -s monitoring"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_uncached_estimates as checker  # noqa: E402

BUCKETS = {
    "buckets": [
        {"id": "L", "min_uncached_tokens": 256001},
        {"id": "S", "max_uncached_tokens": 32000},
        {"id": "M", "min_uncached_tokens": 32001, "max_uncached_tokens": 256000},
    ]
}


def record(bucket, uncached, estimate=None, prompt=500_000):
    """A record as a load client writes it: router headers plus engine usage."""
    return {
        "x-sgl-bucket": bucket,
        "x-sgl-uncached-estimate": uncached if estimate is None else estimate,
        "usage": {
            "prompt_tokens": prompt,
            "prompt_tokens_details": {"cached_tokens": prompt - uncached},
        },
    }


def correct(count):
    return [record("S", 2_000) for _ in range(count)]


class CheckerTest(unittest.TestCase):
    def run_checker(self, records, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            buckets = os.path.join(tmp, "buckets.json")
            rows = os.path.join(tmp, "records.jsonl")
            with open(buckets, "w") as handle:
                json.dump(BUCKETS, handle)
            with open(rows, "w") as handle:
                handle.write("\n".join(json.dumps(row) for row in records) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = checker.main(["--buckets", buckets, rows, "--json", *extra])
            return code, json.loads(out.getvalue())

    def verdicts(self, result):
        return {rule["rule"]: rule["pass"] for rule in result["rules"]}

    def test_lanes_order_short_to_long(self):
        self.assertEqual([lane[0] for lane in checker.lanes(BUCKETS)], ["S", "M", "L"])

    def test_estimates_in_the_right_lane_pass(self):
        code, result = self.run_checker(correct(1_000))
        self.assertEqual(code, 0)
        self.assertTrue(all(self.verdicts(result).values()))
        self.assertEqual(result["counts"]["usable"], 1_000)

    def test_one_percent_of_long_requests_in_a_short_lane_fails_the_unsafe_rule(self):
        rows = correct(990) + [record("S", 100_000, estimate=2_000) for _ in range(10)]
        code, result = self.run_checker(rows, "--tolerance", "100")
        self.assertEqual(code, 1)
        self.assertEqual(
            self.verdicts(result),
            {"unsafe": False, "long_as_short": True, "safe": True, "estimate_error_p90": True},
        )

    def test_a_giant_in_the_short_lane_fails_long_as_short_first(self):
        rows = correct(998) + [record("S", 600_000, estimate=2_000, prompt=600_000) for _ in range(2)]
        code, result = self.run_checker(rows)
        self.assertEqual(code, 1)
        verdicts = self.verdicts(result)
        self.assertFalse(verdicts["long_as_short"])
        self.assertTrue(verdicts["unsafe"], "0.2 % unsafe is under 0.5 %")

    def test_short_requests_sent_long_count_against_the_safe_rule(self):
        rows = correct(940) + [record("M", 2_000, estimate=40_000) for _ in range(60)]
        code, result = self.run_checker(rows, "--tolerance", "100")
        self.assertEqual(code, 1)
        self.assertEqual(self.verdicts(result)["safe"], False)
        self.assertEqual(self.verdicts(result)["unsafe"], True)

    def test_the_estimate_must_be_within_tolerance_at_p90(self):
        rows = correct(850) + [record("S", 2_000, estimate=2_400) for _ in range(150)]
        code, result = self.run_checker(rows)
        self.assertEqual(code, 1)
        self.assertFalse(self.verdicts(result)["estimate_error_p90"])
        rows = correct(950) + [record("S", 2_000, estimate=2_400) for _ in range(50)]
        self.assertEqual(self.run_checker(rows)[0], 0)

    def test_flat_fields_are_read_and_incomplete_records_skipped(self):
        flat = {"bucket": "M", "uncached_router": 50_000, "prompt_tokens": 90_000, "cached_tokens": 40_000}
        rows = [flat, {"bucket": "S"}, record("X", 10)]
        code, result = self.run_checker(rows)
        self.assertEqual(code, 0)
        self.assertEqual(result["counts"]["usable"], 1)
        self.assertEqual(result["counts"]["skipped"], 2)

    def test_no_usable_record_exits_2(self):
        self.assertEqual(self.run_checker([{"bucket": "S"}])[0], 2)


if __name__ == "__main__":
    unittest.main()
