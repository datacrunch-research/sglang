//! Kimi-K3's `tool_choice: "auto"` structural tag
//! (`get_kimik3_auto_tool_call_structural_tag` in
//! `function_call/kimik3_structural_tag.py`), dumped as xgrammar's
//! `StructuralTag.model_dump(by_alias=True)`.

use serde_json::{Map, Value, json};

use super::Unsupported;

use crate::kimi_k3_format::{
    CALL_CLOSE, CALL_OPEN, THINK_CLOSE, THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};

const CLOSE_TOKEN: &str = "<|close|>";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";

const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "array", "object", "null",
];
/// The schema keywords `_schema_types` reads a type from, in its order.
const TYPE_KEYWORDS: [(&str, &str); 4] = [
    (
        "object",
        "additionalProperties dependentRequired dependentSchemas maxProperties minProperties \
         patternProperties properties propertyNames required",
    ),
    (
        "array",
        "contains items maxContains maxItems minContains minItems prefixItems uniqueItems",
    ),
    ("string", "format maxLength minLength pattern"),
    (
        "number",
        "exclusiveMaximum exclusiveMinimum maximum minimum multipleOf",
    ),
];

/// How deep and how long `_schema_types` may run on one argument. Python
/// recurses until its recursion limit and then drops the tag; a schema past
/// these bounds is left to the engine, which does exactly that.
const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_SCHEMA_STEPS: usize = 10_000;

/// The auto tag over `tools` (request tool definitions), without the `strip`
/// tokens `strip_structural_tag_excludes` removes; `None` without tools. Only
/// the trigger's excludes hold think tokens.
pub(super) fn auto_tool_call_tag(
    tools: &[Value],
    parallel_tool_calls: bool,
    strip: &[&str],
) -> Result<Option<Value>, Unsupported> {
    if tools.is_empty() {
        return Ok(None);
    }
    let call_tags = tools
        .iter()
        .map(|tool| tool_call_tag(&tool["function"]))
        .collect::<Option<_>>()
        .ok_or(Unsupported("tool_definition"))?;
    let tools_tag = tool_calls_tag(call_tags, parallel_tool_calls);
    // `_auto_suffix`: a retriggered second tools section would evade the single-call limit.
    let excludes: Vec<&str> = [THINK_OPEN, THINK_CLOSE, CALL_OPEN]
        .into_iter()
        .filter(|token| !strip.contains(token))
        .collect();
    let format = json!({
        "type": "triggered_tags",
        "triggers": [TOOLS_OPEN],
        "tags": [tools_tag],
        "at_least_one": false,
        "stop_after_first": !parallel_tool_calls,
        "excludes": excludes,
    });
    Ok(Some(json!({"type": "structural_tag", "format": format})))
}

/// `get_kimik3_structural_tag`: the tag Python builds for `tool_choice`
/// `"required"` or a named function (`at_least_one`: free text, then one tools
/// section with at least one call) and for `"auto"` with a strict tool (the
/// auto trigger over the same calls). A named choice keeps only its tool. The
/// reasoning prefix is left to the reasoning parser (`thinking_mode=False`, as
/// with `--reasoning-parser kimi_k3`); `strip` as in [`auto_tool_call_tag`].
/// `Ok(None)` where Python's builder raises and the request goes unconstrained.
pub(super) fn tool_call_tag_for_choice(
    tools: &[Value],
    named: Option<&str>,
    at_least_one: bool,
    parallel_tool_calls: bool,
    strip: &[&str],
) -> Result<Option<Value>, Unsupported> {
    // `_select_tools`: a named choice keeps only its tool.
    let selected: Vec<&Value> = tools
        .iter()
        .filter(|tool| named.is_none_or(|name| tool["function"]["name"] == name))
        .collect();
    if selected.is_empty() {
        return Err(Unsupported("tool_choice"));
    }
    let Ok(call_tags) = selected
        .iter()
        .map(|tool| choice_call_tag(&tool["function"]))
        .collect::<Result<Vec<Value>, ()>>()
    else {
        return Ok(None);
    };
    let tools_tag = tool_calls_tag(call_tags, parallel_tool_calls);
    let keep = |tokens: &[&'static str]| -> Vec<&'static str> {
        tokens
            .iter()
            .copied()
            .filter(|t| !strip.contains(t))
            .collect()
    };
    let format = if at_least_one {
        let excludes = keep(&[TOOLS_OPEN, THINK_OPEN, THINK_CLOSE, CALL_OPEN]);
        sequence(vec![any_text(&excludes), tools_tag])
    } else {
        json!({
            "type": "triggered_tags",
            "triggers": [TOOLS_OPEN],
            "tags": [tools_tag],
            "at_least_one": false,
            "stop_after_first": !parallel_tool_calls,
            "excludes": keep(&[THINK_OPEN, THINK_CLOSE, CALL_OPEN]),
        })
    };
    Ok(Some(json!({"type": "structural_tag", "format": format})))
}

/// `_tool_call_tag(tool)` with `_tool_arguments_format`: a non-strict tool takes
/// any number of loose arguments, a strict one its schema's.
fn choice_call_tag(function: &Value) -> Result<Value, ()> {
    let arguments = if function["strict"] == true {
        strict_tool_arguments(&function["parameters"])?
    } else {
        json!({"type": "star", "content": dynamic_argument_format(&Value::Bool(true), &json!({}), true)?})
    };
    let name = escape_attr(function["name"].as_str().unwrap_or_default());
    Ok(json!({
        "type": "tag",
        "begin": format!("{CALL_OPEN} tool=\"{name}\" index=\""),
        "content": sequence(vec![
            json!({"type": "regex", "pattern": "[1-9][0-9]*"}),
            json!({"type": "const_string", "value": "\"<|sep|>"}),
            arguments,
        ]),
        "end": CALL_CLOSE,
    }))
}

fn xtml_type(json_type: &str) -> &str {
    if json_type == "integer" {
        "number"
    } else {
        json_type
    }
}

fn types_of(schema: &Value, root: &Value) -> Result<Vec<&'static str>, ()> {
    let mut steps = 0;
    schema_types(schema, root, &[], &mut steps).ok_or(())
}

/// `_matches_json_type`.
fn matches_json_type(value: &Value, json_type: &str) -> bool {
    let value_type = self::json_type(value);
    value_type == json_type || (json_type == "number" && value_type == "integer")
}

/// `_with_root_definitions`.
fn with_root_definitions(schema: Value, root: &Value) -> Value {
    let Value::Object(mut result) = schema else {
        return schema;
    };
    for key in ["$defs", "definitions"] {
        if let Some(definitions) = root.get(key)
            && !result.contains_key(key)
        {
            result.insert(key.into(), definitions.clone());
        }
    }
    Value::Object(result)
}

/// `_restrict_schema_type`; `Ok(None)` is Python's `False`.
fn restrict_schema_type(
    schema: &Value,
    json_type: &str,
    root: &Value,
    depth: usize,
) -> Result<Option<Value>, ()> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(());
    }
    let Value::Object(map) = schema else {
        return Ok((schema != &Value::Bool(false)).then(|| json!({"type": json_type})));
    };
    if let Some(Value::String(reference)) = map.get("$ref")
        && let Some(target) = resolve_local_ref(reference, root)
    {
        return Ok(restrict_schema_type(target, json_type, root, depth + 1)?
            .map(|restricted| with_root_definitions(restricted, root)));
    }
    let mut result = map.clone();
    match result.get("type") {
        Some(Value::Array(types)) => {
            if !types.iter().any(|t| t == json_type) {
                return Ok(None);
            }
            result.insert("type".into(), json_type.into());
        }
        Some(Value::String(t)) => {
            if t != json_type {
                return Ok(None);
            }
        }
        _ => {
            result.insert("type".into(), json_type.into());
        }
    }
    for keyword in ["anyOf", "oneOf"] {
        let Some(Value::Array(options)) = result.get(keyword).cloned() else {
            continue;
        };
        let mut restricted = Vec::new();
        for option in options.iter().filter(|o| o.is_boolean() || o.is_object()) {
            if types_of(option, root)?.contains(&json_type) {
                restricted.push(
                    restrict_schema_type(option, json_type, root, depth + 1)?
                        .unwrap_or(Value::Bool(false)),
                );
            }
        }
        if restricted.is_empty() {
            return Ok(None);
        }
        match restricted.as_slice() {
            [Value::Object(only)] => {
                result.shift_remove(keyword);
                for (key, value) in only {
                    result.insert(key.clone(), value.clone());
                }
            }
            _ => {
                result.insert(keyword.into(), Value::Array(restricted));
            }
        }
    }
    if let Some(Value::Array(values)) = result.get("enum") {
        let kept: Vec<Value> = values
            .iter()
            .filter(|v| matches_json_type(v, json_type))
            .cloned()
            .collect();
        if kept.is_empty() {
            return Ok(None);
        }
        result.insert("enum".into(), Value::Array(kept));
    }
    if result
        .get("const")
        .is_some_and(|value| !matches_json_type(value, json_type))
    {
        return Ok(None);
    }
    Ok(Some(with_root_definitions(Value::Object(result), root)))
}

/// `_value_format`.
fn value_format(schema: Value, json_type: &str, loose_string: bool) -> Value {
    if loose_string && json_type == "string" {
        return any_text(&[]);
    }
    let mut schema = schema;
    // XGrammar 0.2.1 miscompiles a one-sided negative integer lower bound; the
    // range is split at zero, as Python does.
    if json_type == "integer"
        && let Value::Object(map) = &schema
        && !["maximum", "exclusiveMaximum", "multipleOf"]
            .iter()
            .any(|k| map.contains_key(*k))
    {
        let int = |key: &str| {
            map.get(key)
                .filter(|v| v.is_i64() || v.is_u64())
                .and_then(Value::as_i64)
        };
        let bounds: Vec<i64> = [int("minimum"), int("exclusiveMinimum").map(|v| v + 1)]
            .into_iter()
            .flatten()
            .collect();
        if bounds.iter().max().is_some_and(|max| *max < 0) {
            let mut negative = map.clone();
            negative.insert("maximum".into(), json!(-1));
            let mut nonnegative = map.clone();
            nonnegative.shift_remove("exclusiveMinimum");
            nonnegative.insert("minimum".into(), json!(0));
            schema = json!({"anyOf": [negative, nonnegative]});
        }
    }
    let style = if json_type == "string" {
        "qwen_xml"
    } else {
        "json"
    };
    json!({"type": "json_schema", "json_schema": schema, "style": style, "any_order": false, "max_whitespace_cnt": null})
}

/// `_argument_value_variants`.
fn argument_value_variants(
    schema: &Value,
    root: &Value,
    loose_strings: bool,
) -> Result<Vec<(&'static str, Value)>, ()> {
    let mut variants = Vec::new();
    for json_type in types_of(schema, root)? {
        if let Some(restricted) = restrict_schema_type(schema, json_type, root, 0)? {
            variants.push((
                json_type,
                value_format(restricted, json_type, loose_strings),
            ));
        }
    }
    Ok(variants)
}

/// `_dynamic_argument_format`: any key, typed values.
fn dynamic_argument_format(schema: &Value, root: &Value, loose_strings: bool) -> Result<Value, ()> {
    let variants: Vec<Value> = argument_value_variants(schema, root, loose_strings)?
        .into_iter()
        .map(|(json_type, value)| {
            sequence(vec![
                json!({"type": "regex", "pattern": r#"[^"& \t\r\n\f\v=<>]+"#}),
                json!({"type": "const_string", "value": format!("\" type=\"{}\"<|sep|>", xtml_type(json_type))}),
                value,
            ])
        })
        .collect();
    let content = match variants.len() {
        0 => return Err(()),
        1 => variants.into_iter().next().expect("one variant"),
        _ => json!({"type": "or", "elements": variants}),
    };
    Ok(
        json!({"type": "tag", "begin": "<|open|>argument key=\"", "content": content, "end": ARGUMENT_CLOSE}),
    )
}

/// `_known_argument_format`.
fn known_argument_format(key: &str, schema: &Value, root: &Value) -> Result<Option<Value>, ()> {
    let key = escape_attr(key);
    let mut variants: Vec<Value> = argument_value_variants(schema, root, false)?
        .into_iter()
        .map(|(json_type, value)| {
            json!({
                "type": "tag",
                "begin": format!("<|open|>argument key=\"{key}\" type=\"{}\"<|sep|>", xtml_type(json_type)),
                "content": value,
                "end": ARGUMENT_CLOSE,
            })
        })
        .collect();
    Ok(match variants.len() {
        0 => None,
        1 => variants.pop(),
        _ => Some(json!({"type": "or", "elements": variants})),
    })
}

/// `_strict_arguments_format`.
fn strict_arguments_format(parameters: &Value) -> Result<Value, ()> {
    let empty = json!({});
    let properties = parameters
        .get("properties")
        .unwrap_or(&empty)
        .as_object()
        .ok_or(())?;
    let no_required = json!([]);
    let required: Vec<&str> = parameters
        .get("required")
        .unwrap_or(&no_required)
        .as_array()
        .ok_or(())?
        .iter()
        .map(|r| r.as_str().ok_or(()))
        .collect::<Result<_, _>>()?;
    if required.iter().any(|key| !properties.contains_key(*key)) {
        return Err(());
    }
    let mut elements = Vec::new();
    for (key, schema) in properties {
        if !(schema.is_boolean() || schema.is_object()) {
            return Err(());
        }
        match known_argument_format(key, schema, parameters)? {
            Some(argument) if required.contains(&key.as_str()) => elements.push(argument),
            Some(argument) => elements.push(json!({"type": "optional", "content": argument})),
            None if required.contains(&key.as_str()) => return Err(()),
            None => {}
        }
    }
    match parameters.get("additionalProperties").unwrap_or(&Value::Bool(true)) {
        Value::Bool(true) => elements.push(
            json!({"type": "star", "content": dynamic_argument_format(&Value::Bool(true), parameters, false)?}),
        ),
        additional @ Value::Object(_) => elements.push(
            json!({"type": "star", "content": dynamic_argument_format(additional, parameters, false)?}),
        ),
        Value::Bool(false) => {}
        _ => return Err(()),
    }
    Ok(if elements.is_empty() {
        json!({"type": "const_string", "value": ""})
    } else {
        sequence(elements)
    })
}

/// `_tool_arguments_format` for a strict tool.
fn strict_tool_arguments(parameters: &Value) -> Result<Value, ()> {
    match parameters {
        // Strict tools without parameters take no arguments.
        Value::Null => Ok(json!({"type": "const_string", "value": ""})),
        Value::Object(_) if types_of(parameters, parameters)?.contains(&"object") => {
            strict_arguments_format(parameters)
        }
        _ => Err(()),
    }
}

fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn any_text(excludes: &[&str]) -> Value {
    json!({"type": "any_text", "excludes": excludes, "max_tokens": null, "max_chars": null})
}

fn sequence(elements: Vec<Value>) -> Value {
    json!({"type": "sequence", "elements": elements})
}

/// `_tool_call_tag` with `_auto_tool_arguments_format`; `None` past the schema bounds.
fn tool_call_tag(function: &Value) -> Option<Value> {
    let name = escape_attr(function["name"].as_str().unwrap_or_default());
    Some(json!({
        "type": "tag",
        "begin": format!("{CALL_OPEN} tool=\"{name}\" index=\""),
        "content": sequence(vec![
            json!({"type": "regex", "pattern": "[1-9][0-9]*"}),
            json!({"type": "const_string", "value": "\"<|sep|>"}),
            auto_arguments_format(&function["parameters"])?,
        ]),
        "end": CALL_CLOSE,
    }))
}

/// `_tool_calls_tag`.
fn tool_calls_tag(call_tags: Vec<Value>, parallel_tool_calls: bool) -> Value {
    let content = if parallel_tool_calls {
        json!({"type": "tags_with_separator", "tags": call_tags, "separator": "", "at_least_one": true, "stop_after_first": false})
    } else if call_tags.len() == 1 {
        call_tags.into_iter().next().expect("one tag")
    } else {
        json!({"type": "or", "elements": call_tags})
    };
    json!({"type": "tag", "begin": TOOLS_OPEN, "content": content, "end": TOOLS_CLOSE})
}

/// `_nonempty_argument_format`: a token-based end keeps the first close
/// token out of both content formats.
fn nonempty_argument_format(key: &str, xtml_type: &str) -> Value {
    let argument = json!({
        "type": "tag",
        "begin": format!("<|open|>argument key=\"{}\" type=\"{xtml_type}\"<|sep|>", escape_attr(key)),
        "content": sequence(vec![
            json!({"type": "exclude_token", "exclude_tokens": []}),
            json!({"type": "any_tokens", "exclude_tokens": [], "max_tokens": null}),
        ]),
        "end": {"type": "token", "token": CLOSE_TOKEN},
    });
    sequence(vec![
        argument,
        json!({"type": "const_string", "value": "argument<|sep|>"}),
    ])
}

/// `_auto_tool_arguments_format`: required arguments present and non-empty, then any text.
fn auto_arguments_format(parameters: &Value) -> Option<Value> {
    let loose = || Some(any_text(&[]));
    let Value::Object(schema) = parameters else {
        return loose();
    };
    let empty = Value::Object(Map::new());
    let properties = schema.get("properties").unwrap_or(&empty);
    let no_required = Value::Array(Vec::new());
    let required = schema.get("required").unwrap_or(&no_required);
    let (Value::Object(properties), Value::Array(required)) = (properties, required) else {
        return loose();
    };
    let mut elements = Vec::new();
    for key in required {
        let Some(key) = key.as_str() else {
            return loose();
        };
        let Some(property) = properties
            .get(key)
            .filter(|s| s.is_boolean() || s.is_object())
        else {
            return loose();
        };
        let Some(xtml_type) = single_xtml_type(property, parameters)? else {
            return loose();
        };
        elements.push(nonempty_argument_format(key, xtml_type));
    }
    if elements.is_empty() {
        let mut alternatives = Vec::new();
        for (key, property) in properties {
            if (property.is_boolean() || property.is_object())
                && let Some(xtml_type) = single_xtml_type(property, parameters)?
            {
                alternatives.push(nonempty_argument_format(key, xtml_type));
            }
        }
        match alternatives.len() {
            0 => return loose(),
            1 => elements.push(alternatives.remove(0)),
            _ => elements.push(json!({"type": "or", "elements": alternatives})),
        }
    }
    elements.push(any_text(&[CALL_OPEN, CALL_CLOSE]));
    Some(sequence(elements))
}

/// `_single_xtml_type`; the outer `None` is past the schema bounds.
fn single_xtml_type(schema: &Value, root: &Value) -> Option<Option<&'static str>> {
    let mut steps = 0;
    Some(
        match schema_types(schema, root, &[], &mut steps)?.as_slice() {
            ["integer"] => Some("number"),
            [only] => Some(*only),
            _ => None,
        },
    )
}

/// `_json_type`.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `_resolve_local_ref`.
fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    let path = reference.strip_prefix("#/")?;
    let mut value = root;
    for part in path.split('/') {
        let key = part.replace("~1", "/").replace("~0", "~");
        value = value.as_object()?.get(&key)?;
    }
    (value.is_boolean() || value.is_object()).then_some(value)
}

/// The JSON types in `JSON_TYPES` order that are members of `set`.
fn ordered(set: impl Fn(&str) -> bool) -> Vec<&'static str> {
    JSON_TYPES.into_iter().filter(|t| set(t)).collect()
}

/// `_schema_types`; `seen` holds the `$ref`s already followed on this path.
/// `None` past `MAX_SCHEMA_DEPTH` or `MAX_SCHEMA_STEPS`.
fn schema_types(
    schema: &Value,
    root: &Value,
    seen: &[&str],
    steps: &mut usize,
) -> Option<Vec<&'static str>> {
    *steps += 1;
    if seen.len() > MAX_SCHEMA_DEPTH || *steps > MAX_SCHEMA_STEPS {
        return None;
    }
    let schema = match schema {
        Value::Bool(false) => return Some(Vec::new()),
        Value::Object(schema) => schema,
        _ => return Some(JSON_TYPES.to_vec()),
    };
    if let Some(Value::String(reference)) = schema.get("$ref")
        && !seen.contains(&reference.as_str())
        && let Some(target) = resolve_local_ref(reference, root)
    {
        let seen = [seen, &[reference.as_str()]].concat();
        return schema_types(target, root, &seen, steps);
    }
    // Nesting without `$ref`s also counts toward the depth.
    let nested = [seen, &[""]].concat();
    let mut branches = |options: &Vec<Value>| -> Option<Vec<Vec<&'static str>>> {
        options
            .iter()
            .filter(|o| o.is_boolean() || o.is_object())
            .map(|o| schema_types(o, root, &nested, steps))
            .collect()
    };
    match schema.get("type") {
        Some(Value::String(t)) => {
            return Some(match JSON_TYPES.iter().find(|j| *j == t) {
                Some(t) => vec![*t],
                None => JSON_TYPES.to_vec(),
            });
        }
        Some(Value::Array(types)) => return Some(ordered(|t| types.iter().any(|v| v == t))),
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = schema.get(keyword) {
            let types = branches(options)?.concat();
            return Some(ordered(|t| types.contains(&t)));
        }
    }
    if let Some(Value::Array(options)) = schema.get("allOf") {
        let widen = |types: Vec<&'static str>| -> Vec<&'static str> {
            let mut types = types;
            if types.contains(&"number") && !types.contains(&"integer") {
                types.push("integer");
            }
            types
        };
        let constrained: Vec<Vec<&'static str>> = branches(options)?
            .into_iter()
            .filter(|types| ordered(|t| types.contains(&t)) != JSON_TYPES.to_vec())
            .collect();
        if let Some((first, rest)) = constrained.split_first() {
            let mut result = widen(first.clone());
            for types in rest {
                let widened = widen(types.clone());
                result.retain(|t| widened.contains(t));
            }
            // number survives only if every branch allows it, making integer redundant.
            if result.contains(&"number") {
                result.retain(|t| *t != "integer");
            }
            return Some(ordered(|t| result.contains(&t)));
        }
    }
    if let Some(value) = schema.get("const") {
        return Some(vec![json_type(value)]);
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        let types: Vec<&str> = values.iter().map(json_type).collect();
        return Some(ordered(|t| types.contains(&t)));
    }
    Some(
        TYPE_KEYWORDS
            .iter()
            .find(|(_, keywords)| keywords.split_whitespace().any(|k| schema.contains_key(k)))
            .map_or_else(|| JSON_TYPES.to_vec(), |(json_type, _)| vec![*json_type]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool whose required argument is `hops + 1` `$ref`s from its type, or
    /// with `fanout`, `hops` levels that each refer twice to the next.
    fn ref_tool(hops: usize, fanout: bool) -> Value {
        let mut defs: Map<String, Value> = (0..hops)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                let def = match fanout {
                    true => json!({"anyOf": [next, next]}),
                    false => next,
                };
                (format!("D{i}"), def)
            })
            .collect();
        defs.insert(format!("D{hops}"), json!({"type": "integer"}));
        json!({"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"$ref": "#/$defs/D0"}},
            "required": ["x"], "$defs": defs,
        }}})
    }

    /// `$ref` chains and fan-outs past the bounds go to the engine, on a small stack.
    #[test]
    fn schema_types_are_bounded() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let tag = |tool| auto_tool_call_tag(&[tool], true, &[]);
                assert!(matches!(
                    tag(ref_tool(MAX_SCHEMA_DEPTH - 1, false)),
                    Ok(Some(_))
                ));
                assert!(tag(ref_tool(MAX_SCHEMA_DEPTH, false)).is_err());
                assert!(tag(ref_tool(10_000, false)).is_err());
                // Within the depth bound, 2^30 paths exhaust the steps.
                assert!(tag(ref_tool(30, true)).is_err());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
