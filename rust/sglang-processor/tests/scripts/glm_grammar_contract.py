"""Contract check for a GLM engine image: `/generate` with the glm47 tool
grammar as plain `ebnf` plus `custom_params.__sglang_reasoning_in_grammar`
(what `lower_chat` sends) must decode exactly as the engine's own
`/v1/chat/completions` with the same tools, which flags the grammar
`ebnf_full_assistant` in-process.

If the engine drops or renames the key, the grammar applies after `</think>`
and asks for a second thinking block; the token ids then differ here.

Run inside a GLM engine pod (its own SGLang builds the grammar), greedy and
one request at a time:
    kubectl exec <engine pod> -- python3 - --url http://127.0.0.1:30000 < glm_grammar_contract.py
Exit status 0 unless a prompt differs where the engine route is repeatable;
a prompt where two engine-route runs already differ is reported as "skip".
"""

import argparse
import json
import sys
import urllib.request

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.glm4_moe_detector import (
    GlmSpecialTokenConfig,
    generate_glm_grammar,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                    "env": {"type": "object"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "estimate_hours": {"type": "number"},
                },
                "required": ["title", "priority"],
            },
        },
    },
]
PROMPTS = [
    "What is the weather in Paris and in Tokyo today? Use Fahrenheit for Tokyo.",
    "Run `cargo test` with RUST_LOG=debug set, and kill it after 600 seconds.",
    'Open a high ticket titled "Engine OOM" with the labels gpu and oncall, estimated at 2.5 hours.',
    "Hello! Who are you?",
]
MAX_TOKENS = 2048


def post(url, path, body, headers=None):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    url = parser.parse_args().url
    grammar = generate_glm_grammar(
        enable_thinking=True,
        functions=[Tool(**t).function for t in TOOLS],
        special_tokens=GlmSpecialTokenConfig(),
        chat_template_version="glm47",
        accommodate_chat_template=True,
        allow_multiple_assistant_turns=False,
    )
    failed = 0
    for prompt in PROMPTS:
        chat = {
            "model": "x",
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
        }
        # Warm the prefix cache first, so every compared request prefills alike.
        post(url, "/v1/chat/completions", {**chat, "max_tokens": 1})
        ids = {"x-sglext-return-input-ids": "1", "x-sglext-return-output-ids": "1"}
        engine = post(url, "/v1/chat/completions", chat, ids)
        want = engine["sglext"]["output_ids"][0]
        again = post(url, "/v1/chat/completions", chat, ids)["sglext"]["output_ids"][0]
        routed = post(
            url,
            "/generate",
            {
                "input_ids": engine["sglext"]["input_ids"],
                "require_reasoning": True,
                "sampling_params": {
                    "max_new_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "skip_special_tokens": False,
                    "ebnf": grammar,
                    "custom_params": {"__sglang_reasoning_in_grammar": True},
                },
            },
        )
        got = routed["output_ids"]
        same = got == want
        # Greedy decoding with MTP is not always repeatable on one engine; a
        # difference counts only where the engine route agrees with itself.
        stable = again == want
        failed += stable and not same
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        label = "ok  " if same else ("DIFF" if stable else "skip")
        print(
            f"{label} engine {len(want):5} again_same={stable!s:5} key {len(got):5} first_diff={first}  {prompt[:50]}"
        )
    print("PASS" if not failed else f"FAIL: {failed} of {len(PROMPTS)} prompts differ")
    sys.exit(1 if failed else 0)


main()
