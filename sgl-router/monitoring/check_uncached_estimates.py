#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the router's uncached-token estimates against what the engines reported.

Reads per-request records (JSON lines) that carry the router's view, from the
`x-sgl-bucket` and `x-sgl-uncached-estimate` response headers, and the engine's
counts, from the response's `usage.prompt_tokens` and
`usage.prompt_tokens_details.cached_tokens`. The engine's uncached tokens
(prompt - cached) name the bucket the request belonged in, by the bucket file's
`min/max_uncached_tokens`; buckets are ordered short to long by their upper bound.
Under `thresholds: prefill_seconds` the bounds are `min/max_prefill_seconds` and
the tokens are divided by the record's `x-sgl-prefill-rate` (or --prefill-rate).

Gate, each a share of the usable records:
  unsafe     the engine's bucket is longer than the router's     (< --max-unsafe, 0.5 %)
  long_as_short  the longest bucket's request routed to the shortest  (< --max-long-as-short, 0.1 %)
  safe       the engine's bucket is shorter than the router's    (< --max-safe, 5 %)
  estimate   |estimate - engine uncached| / engine uncached at the --quantile (0.9)
             is at most --tolerance (10 %)

Exit status: 0 when every rule passes, 1 when one fails, 2 when no record is usable.

usage: check_uncached_estimates.py --buckets buckets.json records.jsonl [more.jsonl ...] [--json]
"""

import argparse
import json
import math
import sys

BUCKET_FIELDS = ("bucket", "x-sgl-bucket")
ESTIMATE_FIELDS = ("uncached_estimate", "uncached_router", "x-sgl-uncached-estimate")
PROMPT_FIELDS = ("prompt_tokens", "usage.prompt_tokens")
CACHED_FIELDS = ("cached_tokens", "usage.prompt_tokens_details.cached_tokens")
RATE_FIELDS = ("prefill_rate", "x-sgl-prefill-rate")


def field(record, names):
    """The first of `names` present in `record`; a dotted name also reads nested keys."""
    for name in names:
        value = record.get(name)
        if value is None and "." in name:
            value = record
            for part in name.split("."):
                value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def seconds_bounds(bucket_file):
    """Whether the file bounds buckets in prefill seconds rather than tokens."""
    return bucket_file.get("thresholds") == "prefill_seconds"


def lanes(bucket_file):
    """Bucket ids short to long, with their inclusive bounds (tokens or seconds)."""
    unit = "prefill_seconds" if seconds_bounds(bucket_file) else "uncached_tokens"
    buckets = []
    for spec in bucket_file.get("buckets", []):
        low, high = spec.get(f"min_{unit}"), spec.get(f"max_{unit}")
        if low is None and high is None:
            continue
        buckets.append((spec["id"], low or 0, math.inf if high is None else high))
    buckets.sort(key=lambda bucket: (bucket[2], bucket[1]))
    return buckets


def lane_of(buckets, uncached):
    for index, (_, low, high) in enumerate(buckets):
        if low <= uncached <= high:
            return index
    return None


def quantile(values, q):
    """Nearest-rank quantile of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def check(records, buckets, limits, seconds=False, rate=None):
    index = {bucket_id: i for i, (bucket_id, _, _) in enumerate(buckets)}
    counts = {"records": 0, "usable": 0, "skipped": 0, "unsafe": 0, "long_as_short": 0, "safe": 0}
    errors = []
    for record in records:
        counts["records"] += 1
        bucket = field(record, BUCKET_FIELDS)
        estimate = field(record, ESTIMATE_FIELDS)
        prompt = field(record, PROMPT_FIELDS)
        cached = field(record, CACHED_FIELDS)
        if bucket not in index or None in (estimate, prompt, cached):
            counts["skipped"] += 1
            continue
        actual = max(0, int(prompt) - int(cached))
        measure = actual
        if seconds:
            record_rate = float(field(record, RATE_FIELDS) or rate or 0)
            if record_rate <= 0:
                counts["skipped"] += 1
                continue
            measure = actual / record_rate
        engine_lane = lane_of(buckets, measure)
        if engine_lane is None:
            counts["skipped"] += 1
            continue
        counts["usable"] += 1
        router_lane = index[bucket]
        if engine_lane > router_lane:
            counts["unsafe"] += 1
            if engine_lane == len(buckets) - 1 and router_lane == 0 and len(buckets) > 1:
                counts["long_as_short"] += 1
        elif engine_lane < router_lane:
            counts["safe"] += 1
        errors.append(abs(int(estimate) - actual) / max(actual, 1))
    if not counts["usable"]:
        return counts, None
    usable = counts["usable"]
    error_q = quantile(errors, limits["quantile"])
    rules = [
        ("unsafe", counts["unsafe"] / usable, limits["max_unsafe"]),
        ("long_as_short", counts["long_as_short"] / usable, limits["max_long_as_short"]),
        ("safe", counts["safe"] / usable, limits["max_safe"]),
        (f"estimate_error_p{round(limits['quantile'] * 100)}", error_q, limits["tolerance"]),
    ]
    # Shares must stay under their limit; the estimate error may reach it.
    return counts, [
        {
            "rule": name,
            "value": value,
            "limit": limit,
            "pass": value <= limit if name.startswith("estimate") else value < limit,
        }
        for name, value, limit in rules
    ]


def read_records(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", help="JSON-lines files, one request per line")
    parser.add_argument("--buckets", required=True, help="the router's --bucket-config file")
    parser.add_argument("--max-unsafe", type=float, default=0.005)
    parser.add_argument("--max-long-as-short", type=float, default=0.001)
    parser.add_argument("--max-safe", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument(
        "--prefill-rate",
        type=float,
        help="tokens per second for prefill-second bounds when a record carries no x-sgl-prefill-rate",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    with open(args.buckets) as handle:
        bucket_file = json.load(handle)
    buckets = lanes(bucket_file)
    if not buckets:
        parser.error("the bucket file has no bucket with uncached-token or prefill-second bounds")
    limits = {
        "max_unsafe": args.max_unsafe,
        "max_long_as_short": args.max_long_as_short,
        "max_safe": args.max_safe,
        "tolerance": args.tolerance,
        "quantile": args.quantile,
    }
    counts, rules = check(
        records=read_records(args.records),
        buckets=buckets,
        limits=limits,
        seconds=seconds_bounds(bucket_file),
        rate=args.prefill_rate,
    )
    if args.json:
        print(json.dumps({"counts": counts, "rules": rules}, indent=2))
    else:
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        for rule in rules or []:
            verdict = "PASS" if rule["pass"] else "FAIL"
            print(f"{verdict} {rule['rule']}: {rule['value']:.4%} (limit {rule['limit']:.2%})")
    if rules is None:
        print("no usable record", file=sys.stderr)
        return 2
    return 0 if all(rule["pass"] for rule in rules) else 1


if __name__ == "__main__":
    sys.exit(main())
