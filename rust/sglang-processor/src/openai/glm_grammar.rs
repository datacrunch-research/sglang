//! The full-assistant EBNF that SGLang constrains a `glm47` request with tools
//! by (`generate_glm_grammar` in `function_call/glm4_moe_detector.py`, as
//! `FunctionCallParser.get_structure_constraint` calls it for `tool_choice`
//! `auto`): the thinking block, the text and the XML tool calls of one
//! assistant turn. The engine takes it on `/generate` with
//! `ebnf_full_assistant`, which keeps its reasoner wrapper off the grammar.
//!
//! The output is byte-identical to Python's, so the engine's grammar cache
//! keys match those of its own OpenAI layer.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::py::python_json;

/// `GlmSpecialTokenConfig`, in its field order.
const BEGIN_OF_THINKING: &str = "<think>";
const END_OF_THINKING: &str = "</think>";
const BEGIN_OF_TOOL_CALL: &str = "<tool_call>";
const END_OF_TOOL_CALL: &str = "</tool_call>";
const BEGIN_OF_KEY: &str = "<arg_key>";
const END_OF_KEY: &str = "</arg_key>";
const BEGIN_OF_VALUE: &str = "<arg_value>";
const END_OF_VALUE: &str = "</arg_value>";
const ASSISTANT_TOKEN: &str = "<|assistant|>";
const SPECIAL_TOKENS: [&str; 9] = [
    BEGIN_OF_THINKING,
    END_OF_THINKING,
    BEGIN_OF_TOOL_CALL,
    END_OF_TOOL_CALL,
    BEGIN_OF_KEY,
    END_OF_KEY,
    BEGIN_OF_VALUE,
    END_OF_VALUE,
    ASSISTANT_TOKEN,
];

/// `_GLM_XML_GRAMMAR_RULES`.
const XML_GRAMMAR_RULES: [&str; 11] = [
    r#"basic_string ::= (([\"] basic_string_1 [\"]))"#,
    r#"basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1"#,
    r#"escape ::= ["\\//bfnrt] | "u" [A-Fa-f0-9]{4}"#,
    r#"basic_integer ::= "-"? ("0" | [1-9] [0-9]*) ".0"?"#,
    r#"basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?"#,
    r#"basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]""#,
    r#"basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}""#,
    r#"ws ::= [ \n\t]*"#,
    "basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object",
    r#"basic_boolean ::= "true" | "false""#,
    r#"basic_null ::= "null""#,
];

const TEXT: &str = "text_without_special_tokens";

/// `_GLM_TYPE_MAPPING`.
fn type_rule(kind: &str) -> &'static str {
    match kind {
        "string" => TEXT,
        "number" | "integer" => "basic_number",
        "boolean" => "basic_boolean",
        "null" => "basic_null",
        "array" => "basic_array",
        "object" => "basic_object",
        _ => TEXT,
    }
}

/// `generate_glm_grammar(enable_thinking, functions, GlmSpecialTokenConfig(),
/// "glm47", accommodate_chat_template=True, allow_multiple_assistant_turns=False,
/// required=False, parallel_tool_calls)` over the request's tool definitions
/// (`{"type": "function", "function": {...}}`). `None` where Python raises
/// (a duplicate non-terminal).
pub(super) fn glm47_grammar(
    tools: &[Value],
    enable_thinking: bool,
    parallel_tool_calls: bool,
) -> Option<String> {
    let mut lines = vec![
        "root ::= assistant_turn".to_owned(),
        "assistant_turn ::= thinking_block text_block tool_call_blocks".to_owned(),
    ];
    if enable_thinking {
        lines.push(format!(
            r#"thinking_block ::= thinking_block_content "{END_OF_THINKING}""#
        ));
        let exclusions = [
            BEGIN_OF_TOOL_CALL,
            END_OF_TOOL_CALL,
            BEGIN_OF_KEY,
            END_OF_KEY,
            BEGIN_OF_VALUE,
            END_OF_VALUE,
            END_OF_THINKING,
        ];
        lines.extend(string_exclude("thinking_block_content", &exclusions));
    } else {
        lines.push(r#"thinking_block ::= """#.to_owned());
    }
    lines.extend(string_exclude(TEXT, &SPECIAL_TOKENS));
    // glm47's separator is empty, which leaves the double spaces Python prints.
    lines.push(format!("text_block ::= (  {TEXT} )?"));
    if tools.is_empty() {
        lines.push(r#"tool_call_blocks ::= """#.to_owned());
    } else {
        lines.extend(tool_call_rules(tools, parallel_tool_calls));
    }
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut deduped = Vec::with_capacity(lines.len());
    for line in lines {
        let lhs = line
            .split("::=")
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
        match seen.get(&lhs) {
            Some(previous) if *previous == line => continue,
            Some(_) => return None,
            None => {
                seen.insert(lhs, line.clone());
                deduped.push(line);
            }
        }
    }
    Some(deduped.join("\n"))
}

/// `_glm_build_tool_call_rules` for glm47, `required=False`.
fn tool_call_rules(tools: &[Value], parallel_tool_calls: bool) -> Vec<String> {
    let repetition = if parallel_tool_calls { "*" } else { "?" };
    let functions: Vec<&Value> = tools.iter().map(|tool| &tool["function"]).collect();
    let name = |function: &Value| function["name"].as_str().unwrap_or_default().to_owned();
    let hashes: Vec<String> = functions
        .iter()
        .enumerate()
        .map(|(index, function)| hash_name(&format!("{}{index}", name(function))))
        .collect();
    let mut rules = vec![
        format!("tool_call_blocks ::= (  tool_call_unit ){repetition}"),
        format!(
            r#"tool_call_unit ::= "{BEGIN_OF_TOOL_CALL}" single_tool_call "{END_OF_TOOL_CALL}""#
        ),
        format!(
            "single_tool_call ::= {}",
            hashes
                .iter()
                .map(|hash| format!("call_{hash}"))
                .collect::<Vec<_>>()
                .join(" | ")
        ),
    ];
    let empty = Value::Object(Map::new());
    for (function, hash) in functions.iter().zip(&hashes) {
        // `func.parameters or {}`.
        let params = match &function["parameters"] {
            Value::Null => &empty,
            Value::Object(map) if map.is_empty() => &empty,
            params => params,
        };
        let properties = match has_complete_properties(params) {
            true => schema_properties(params),
            false => Vec::new(),
        };
        let composed = params.as_object().is_some_and(|p| {
            ["allOf", "anyOf", "oneOf"]
                .iter()
                .any(|k| p.contains_key(*k))
        });
        let pairs: Vec<String> = properties
            .iter()
            .map(|(key, schema)| {
                let rule = match composed {
                    true => TEXT.to_owned(),
                    false => value_rule(schema),
                };
                format!(
                    r#""{BEGIN_OF_KEY}{}{END_OF_KEY}"  "{BEGIN_OF_VALUE}" ({rule}) "{END_OF_VALUE}""#,
                    escape_ebnf_string(key)
                )
            })
            .collect();
        let arguments = if pairs.is_empty() {
            format!(
                r#"( "{BEGIN_OF_KEY}" {TEXT} "{END_OF_KEY}"  "{BEGIN_OF_VALUE}" {TEXT} "{END_OF_VALUE}"  )*"#
            )
        } else {
            let choices = pairs.join(" | ");
            format!("( ( {choices} ) (  ( {choices} ) )* )?")
        };
        rules.push(format!(
            r#"call_{hash} ::= "{}"  ( arguments_{hash}  )?"#,
            escape_ebnf_string(&name(function))
        ));
        rules.push(format!("arguments_{hash} ::= {arguments}"));
    }
    rules.extend(XML_GRAMMAR_RULES.iter().map(|rule| (*rule).to_owned()));
    rules
}

/// `_glm_get_value_rule`.
fn value_rule(schema: &Value) -> String {
    let Value::Object(schema) = schema else {
        return TEXT.to_owned();
    };
    if let Some(values) = schema.get("enum") {
        return enum_rule(values);
    }
    match schema.get("type") {
        Some(Value::String(kind)) => type_rule(kind).to_owned(),
        Some(Value::Array(kinds)) if !kinds.is_empty() => kinds
            .iter()
            .map(|kind| kind.as_str().map_or(TEXT, type_rule))
            .collect::<Vec<_>>()
            .join(" | "),
        _ => TEXT.to_owned(),
    }
}

/// `_glm_handle_enum`.
fn enum_rule(values: &Value) -> String {
    let values: Vec<String> = values
        .as_array()
        .into_iter()
        .flatten()
        .map(|value| {
            let text = match value {
                Value::String(text) => text.clone(),
                other => python_json(other, false),
            };
            format!("\"{}\"", escape_ebnf_string(&text))
        })
        .collect();
    match values.len() {
        0 => TEXT.to_owned(),
        1 => values.into_iter().next().expect("one value"),
        _ => format!("({})", values.join(" | ")),
    }
}

/// `_glm_has_complete_properties`.
fn has_complete_properties(schema: &Value) -> bool {
    let Value::Object(map) = schema else {
        return *schema == Value::Bool(false);
    };
    let open = |key: &str| map.get(key).is_some_and(|v| *v != Value::Bool(false));
    if [
        "$ref",
        "$dynamicRef",
        "patternProperties",
        "dependentSchemas",
        "if",
        "then",
        "else",
    ]
    .iter()
    .any(|key| map.contains_key(*key))
        || open("additionalProperties")
        || open("unevaluatedProperties")
    {
        return false;
    }
    let closed = map.get("additionalProperties") == Some(&Value::Bool(false));
    let branches: Vec<Value> = ["allOf", "anyOf", "oneOf"]
        .iter()
        .flat_map(|key| python_iter(map.get(*key)))
        .collect();
    if let Some(Value::Object(properties)) = map.get("properties") {
        return branches.is_empty() && (!properties.is_empty() || closed);
    }
    if !branches.is_empty() {
        return branches.iter().all(has_complete_properties);
    }
    closed
}

/// The items Python's `for` takes from a schema keyword: a list's items, a
/// dict's keys, a string's characters.
fn python_iter(value: Option<&Value>) -> Vec<Value> {
    match value {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::Object(map)) => map.keys().cloned().map(Value::String).collect(),
        Some(Value::String(text)) => text.chars().map(|c| Value::String(c.into())).collect(),
        _ => Vec::new(),
    }
}

/// `get_schema_properties` (`function_call/utils.py`): the top-level
/// `properties`, else those of the `anyOf`/`oneOf`/`allOf` branches, first wins.
pub(crate) fn schema_properties(schema: &Value) -> Vec<(String, Value)> {
    let Value::Object(map) = schema else {
        return Vec::new();
    };
    if let Some(Value::Object(properties)) = map.get("properties") {
        return properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
    }
    let mut merged: Vec<(String, Value)> = Vec::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(branches)) = map.get(keyword) {
            for branch in branches {
                for (key, value) in schema_properties(branch) {
                    if !merged.iter().any(|(k, _)| *k == key) {
                        merged.push((key, value));
                    }
                }
            }
        }
    }
    merged
}

/// `_glm_hash_name`.
fn hash_name(name: &str) -> String {
    let digest = Sha256::digest(name.as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// `_glm_escape_ebnf_string`: `json.dumps(s, ensure_ascii=False)[1:-1]`.
fn escape_ebnf_string(text: &str) -> String {
    let dumped = python_json(&Value::String(text.to_owned()), false);
    dumped[1..dumped.len() - 1].to_owned()
}

/// `_glm_escape_char_class`.
fn escape_char_class(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '\\' | ']' | '^' | '-' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 32 || (c as u32) > 126 => {
                out.push_str(&format!("\\x{:02X}", c as u32))
            }
            c => out.push(c),
        }
    }
    out
}

/// `_glm_escape_string`.
fn escape_string(c: char) -> String {
    match c {
        '"' => "\\\"".into(),
        '\\' => "\\\\".into(),
        '\n' => "\\n".into(),
        '\t' => "\\t".into(),
        '\r' => "\\r".into(),
        c if (c as u32) < 32 || (c as u32) > 126 => format!("\\x{:02X}", c as u32),
        c => c.into(),
    }
}

/// `_GlmTrieNode`: children in insertion order, as Python's dict keeps them.
struct Node {
    children: Vec<(char, usize)>,
    is_end: bool,
    fail: usize,
}

impl Node {
    fn child(&self, c: char) -> Option<usize> {
        self.children
            .iter()
            .find_map(|&(k, id)| (k == c).then_some(id))
    }
}

/// `_glm_build_trie_with_failure_links`; node 0 is the root.
fn trie(patterns: &[&str]) -> Vec<Node> {
    let mut nodes = vec![Node {
        children: Vec::new(),
        is_end: false,
        fail: 0,
    }];
    for pattern in patterns {
        let mut node = 0;
        for c in pattern.chars() {
            node = match nodes[node].child(c) {
                Some(child) => child,
                None => {
                    let id = nodes.len();
                    nodes.push(Node {
                        children: Vec::new(),
                        is_end: false,
                        fail: 0,
                    });
                    nodes[node].children.push((c, id));
                    id
                }
            };
        }
        nodes[node].is_end = true;
    }
    let mut queue: VecDeque<usize> = nodes[0].children.iter().map(|&(_, id)| id).collect();
    while let Some(node) = queue.pop_front() {
        for (c, child) in nodes[node].children.clone() {
            queue.push_back(child);
            let mut fail = nodes[node].fail;
            while fail != 0 && nodes[fail].child(c).is_none() {
                fail = nodes[fail].fail;
            }
            nodes[child].fail = match nodes[fail].child(c) {
                Some(target) if target != child => target,
                _ => 0,
            };
            // A suffix match also completes a forbidden pattern.
            if nodes[nodes[child].fail].is_end {
                nodes[child].is_end = true;
            }
        }
    }
    nodes
}

/// `_glm_get_transition`.
fn transition(nodes: &[Node], node: usize, c: char) -> usize {
    let mut current = node;
    loop {
        if let Some(child) = nodes[current].child(c) {
            return child;
        }
        if current == 0 {
            return 0;
        }
        current = nodes[current].fail;
    }
}

/// `_glm_cached_string_exclude`: EBNF for any string without the given
/// substrings, as Aho-Corasick states.
fn string_exclude(rule_name: &str, negative: &[&str]) -> Vec<String> {
    let sorted: Vec<&str> = negative
        .iter()
        .copied()
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if sorted.is_empty() {
        return vec![format!("{rule_name} ::= [^]*")];
    }
    let prefix = {
        let digest = Sha256::digest(sorted.join("\0").as_bytes());
        digest
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let nodes = trie(&sorted);
    let chars: BTreeSet<char> = sorted.iter().flat_map(|p| p.chars()).collect();
    let state = |id: usize| format!("s_{prefix}_{id}");
    let mut rules = vec![format!("{rule_name} ::= {}", state(0))];
    for (id, node) in nodes.iter().enumerate() {
        if node.is_end {
            continue;
        }
        let mut explicit: BTreeSet<char> = BTreeSet::new();
        let mut by_target: BTreeMap<usize, BTreeSet<char>> = BTreeMap::new();
        for &c in &chars {
            explicit.insert(c);
            let target = transition(&nodes, id, c);
            if !nodes[target].is_end {
                by_target.entry(target).or_default().insert(c);
            }
        }
        let mut alternatives = Vec::new();
        match explicit.is_empty() {
            true => alternatives.push(format!("[^] {}", state(0))),
            false => alternatives.push(format!(
                "[^{}] {}",
                escape_char_class(&explicit.iter().collect::<String>()),
                state(0)
            )),
        }
        for (target, chars) in &by_target {
            for &c in chars {
                alternatives.push(format!("\"{}\" {}", escape_string(c), state(*target)));
            }
        }
        alternatives.push(r#""""#.to_owned());
        rules.push(format!("{} ::= {}", state(id), alternatives.join(" | ")));
    }
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `tests/fixtures/glm_grammar.json`, from `tests/scripts/generate_glm_grammar.py`.
    #[test]
    fn grammars_match_sglang() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm_grammar.json");
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let got = glm47_grammar(
                case["tools"].as_array().unwrap(),
                case["enable_thinking"].as_bool().unwrap(),
                case["parallel_tool_calls"].as_bool().unwrap(),
            );
            let want = case["grammar"].as_str().map(str::to_owned);
            assert_eq!(got, want, "{}", case["name"]);
        }
    }

    #[test]
    fn a_suffix_of_a_pattern_is_excluded_too() {
        let rules = string_exclude("t", &["ab", "b"]);
        // "b" alone ends a pattern, so no state moves on "b".
        assert!(rules.iter().all(|r| !r.contains(r#""b" s_"#)), "{rules:?}");
        assert_eq!(hash_name("x").len(), 16);
        assert!(glm47_grammar(&[json!({"function": {"name": "f"}})], true, true).is_some());
    }
}
