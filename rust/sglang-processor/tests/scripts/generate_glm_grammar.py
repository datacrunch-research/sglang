"""Fill tests/fixtures/glm_grammar.json with SGLang's GLM tool grammar.

Each case names the request's tools, `enable_thinking` and
`parallel_tool_calls`; this records the `full_assistant_ebnf` grammar that
`FunctionCallParser.get_structure_constraint` builds for `tool_choice` "auto"
with the glm47 parser, or null where SGLang raises.

Run from rust/sglang-processor in a SGLang Python environment; set
SGLANG_PYTHON to the `python/` directory of the SGLang to record (the engine's):
    SGLANG_PYTHON=.../python python tests/scripts/generate_glm_grammar.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.environ.get("SGLANG_PYTHON", str(Path(__file__).resolve().parents[4] / "python"))
)

from fixture_json import dump

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/glm_grammar.json"


def grammar(tools, enable_thinking, parallel_tool_calls):
    parser = FunctionCallParser([Tool(**tool) for tool in tools], "glm47")
    try:
        constraint = parser.get_structure_constraint(
            "auto", parallel_tool_calls=parallel_tool_calls, thinking_mode=enable_thinking
        )
    except ValueError:
        return None
    assert constraint[0] == "full_assistant_ebnf", constraint
    return constraint[1]


fixture = json.loads(FIXTURE.read_text())
for case in fixture["cases"]:
    case["grammar"] = grammar(
        case["tools"], case["enable_thinking"], case["parallel_tool_calls"]
    )
FIXTURE.write_text(dump(fixture))
