//! GLM-4.7 and GLM-5 (`Glm47MoeDetector` in `function_call/glm47_moe_detector.py`
//! of the engine GLM-5.3 serves on), with `FunctionCallParser`'s wrapping.
//!
//! Calls are `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
//! Values are typed by the request's tool schemas, unions narrowed by the
//! sibling arguments, as `_parse_argument_pairs` does. Differences from Python:
//! `ast.literal_eval` reads only JSON-like literals (`qwen3_coder::literal_eval`),
//! so a set or bytes literal stays a string where Python fails the whole call;
//! regex `$` matches only at the very end, where Python's also matches before a
//! final newline, which every match strips; `\s` misses U+001C..U+001F.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use super::qwen3_coder::{literal_eval, py_float, py_int};
use crate::py::{py_strip, python_dict, python_float, python_json, python_json_reencode};
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector};

const BOT: &str = "<tool_call>";
const EOT: &str = "</tool_call>";
const KEY_OPEN: &str = "<arg_key>";
const KEY_CLOSE: &str = "</arg_key>";
const VALUE_OPEN: &str = "<arg_value>";
const VALUE_CLOSE: &str = "</arg_value>";

/// `func_call_regex`.
static CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)(<tool_call>.*?</tool_call>)|<tool_call>.*").expect("valid regex")
});
/// `func_detail_regex`.
static DETAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\A<tool_call>(.*?)(<arg_key>.*?)?</tool_call>").expect("valid regex")
});
/// `func_arg_regex`.
static ARG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>")
        .expect("valid regex")
});
/// `_STREAMING_PARTIAL_PATTERN`.
static PARTIAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>(.*?)(?:(<arg_key.*?))?(?:(</tool_call>)|$)").expect("valid regex")
});

/// `StreamState`.
#[derive(Clone, Copy, PartialEq)]
enum State {
    Init,
    Between,
    InKey,
    WaitingValue,
    InValue,
}

pub(super) struct Glm47Detector {
    tools: Vec<Value>,
    buffer: String,
    current_tool_id: i64,
    current_tool_name_sent: bool,
    streamed_raw_length: usize,
    open_call: Option<(String, usize)>,
    end_scan_pos: usize,
    state: State,
    current_key: String,
    current_value: String,
    xml_tag_buffer: String,
    is_first_param: bool,
    value_started: bool,
    cached_value_type: Option<String>,
    tool_call_completed: bool,
    sent_empty_object: bool,
    buffer_arguments: Option<bool>,
}

impl Glm47Detector {
    pub(super) fn new(tools: Vec<Value>) -> Self {
        Self {
            tools,
            buffer: String::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_raw_length: 0,
            open_call: None,
            end_scan_pos: 0,
            state: State::Init,
            current_key: String::new(),
            current_value: String::new(),
            xml_tag_buffer: String::new(),
            is_first_param: true,
            value_started: false,
            cached_value_type: None,
            tool_call_completed: false,
            sent_empty_object: false,
            buffer_arguments: None,
        }
    }

    /// `_reset_streaming_state`.
    fn reset_streaming_state(&mut self) {
        self.state = State::Init;
        self.current_key.clear();
        self.current_value.clear();
        self.xml_tag_buffer.clear();
        self.is_first_param = true;
        self.value_started = false;
        self.cached_value_type = None;
        self.tool_call_completed = false;
        self.sent_empty_object = false;
        self.buffer_arguments = None;
    }

    fn schema(&self, func_name: &str) -> Option<&Value> {
        self.tools
            .iter()
            .find(|tool| tool["function"]["name"].as_str() == Some(func_name))
            .map(|tool| &tool["function"]["parameters"])
    }

    /// `detect_and_parse`, once the text holds a `<tool_call>`.
    fn detect_and_parse(&self, text: &str) -> Parsed {
        let mut normal = String::new();
        let mut last_end = 0;
        let mut bodies = Vec::new();
        for m in CALL_RE.captures_iter(text) {
            let Some(call) = m.get(1) else { continue };
            normal.push_str(&text[last_end..call.start()]);
            last_end = call.end();
            bodies.push(call.as_str());
        }
        normal.push_str(&text[last_end..]);
        // `_get_tool_indices`: a repeated name keeps its last index.
        let index = |name: &str| {
            self.tools
                .iter()
                .rposition(|tool| tool["function"]["name"].as_str() == Some(name))
        };
        let mut calls = Vec::new();
        for body in bodies {
            let Some(detail) = DETAIL_RE.captures(body) else {
                continue;
            };
            let name = py_strip(detail.get(1).map_or("", |m| m.as_str()));
            let args = detail.get(2).map_or("", |m| m.as_str());
            let arguments = match args.is_empty() {
                true => Vec::new(),
                false => self.parse_argument_pairs(&arg_pairs(args), name),
            };
            // `parse_base_json` drops calls to tools the request does not list.
            let Some(tool_index) = index(name).filter(|_| !name.is_empty()) else {
                continue;
            };
            calls.push(ToolCallItem {
                tool_index: tool_index as i64,
                name: name.to_owned(),
                parameters: dump_arguments(arguments),
            });
        }
        (py_strip(&normal).to_owned(), calls)
    }

    /// `_parse_argument_pairs`.
    fn parse_argument_pairs(
        &self,
        pairs: &[(String, String)],
        func_name: &str,
    ) -> Vec<(String, Arg)> {
        let schema = self.schema(func_name);
        let mut known: Vec<(String, Arg)> = Vec::new();
        if pairs.len() > 1 && schema.is_some_and(needs_argument_context) {
            for (key, raw) in pairs {
                let key = py_strip(key).to_owned();
                let parsed = self.parse_argument_pairs(&[(key.clone(), raw.clone())], func_name);
                let parsed = parsed.into_iter().next().expect("one pair").1;
                let native =
                    argument_types_of(schema, &key, &[], &Probe::Value(Value::String(raw.clone())));
                if parsed.is_string() || native.is_none_or(|types| !types.contains("string")) {
                    set(&mut known, key, parsed);
                }
            }
        }
        let mut arguments: Vec<(String, Arg)> = Vec::new();
        for (key, raw) in pairs {
            let key = py_strip(key).to_owned();
            let arg_type = argument_type(argument_types_of(schema, &key, &known, &Probe::Unset));
            let (parsed, good) = parse_arguments(raw, arg_type.as_deref().unwrap_or("string"));
            let value = match arg_type.as_deref() {
                Some("string") if parsed.is_string() => parsed,
                Some("string") => Arg::string(raw),
                None => {
                    let allowed = argument_types_of(schema, &key, &known, &parsed.probe());
                    let native = argument_types_of(
                        schema,
                        &key,
                        &known,
                        &Probe::Value(Value::String(raw.clone())),
                    );
                    let keep_raw = allowed.is_some_and(|a| !a.contains(parsed.json_type()))
                        && native.is_some_and(|n| n.contains("string"));
                    match keep_raw || !good {
                        true => Arg::string(raw),
                        false => parsed,
                    }
                }
                Some(_) if good => parsed,
                Some(_) => Arg::string(raw),
            };
            set(&mut known, key.clone(), value.clone());
            set(&mut arguments, key, value);
        }
        arguments
    }

    /// `_process_xml_to_json_streaming`.
    fn xml_to_json(&mut self, increment: &str, func_name: &str) -> String {
        let mut out = String::new();
        for c in increment.chars() {
            self.xml_tag_buffer.push(c);
            match self.state {
                State::Init | State::Between => {
                    if self.xml_tag_buffer.ends_with(KEY_OPEN) {
                        self.state = State::InKey;
                        self.current_key.clear();
                        self.xml_tag_buffer.clear();
                        out.push_str(if self.is_first_param { "{" } else { ", " });
                        self.is_first_param = false;
                    }
                }
                State::InKey => {
                    if self.xml_tag_buffer.ends_with(KEY_CLOSE) {
                        let key =
                            &self.xml_tag_buffer[..self.xml_tag_buffer.len() - KEY_CLOSE.len()];
                        self.current_key = py_strip(key).to_owned();
                        self.xml_tag_buffer.clear();
                        self.state = State::WaitingValue;
                        out.push_str(&python_json(
                            &Value::String(self.current_key.clone()),
                            false,
                        ));
                        out.push_str(": ");
                    }
                }
                State::WaitingValue => {
                    if self.xml_tag_buffer.ends_with(VALUE_OPEN) {
                        self.state = State::InValue;
                        self.current_value.clear();
                        self.xml_tag_buffer.clear();
                        self.value_started = false;
                        let schema = self.schema(func_name);
                        let kind = argument_type(argument_types_of(
                            schema,
                            &self.current_key,
                            &[],
                            &Probe::Unset,
                        ));
                        self.cached_value_type = Some(kind.unwrap_or_else(|| "auto".into()));
                    }
                }
                State::InValue => self.in_value(&mut out, func_name),
            }
        }
        out
    }

    /// The `IN_VALUE` step of `_process_xml_to_json_streaming`.
    fn in_value(&mut self, out: &mut String, func_name: &str) {
        let kind = self
            .cached_value_type
            .clone()
            .unwrap_or_else(|| "string".into());
        let parsed_kind = matches!(kind.as_str(), "auto" | "number" | "integer");
        if self.xml_tag_buffer.ends_with(VALUE_CLOSE) {
            let last =
                self.xml_tag_buffer[..self.xml_tag_buffer.len() - VALUE_CLOSE.len()].to_owned();
            self.current_value.push_str(&last);
            if parsed_kind {
                let pair = [(self.current_key.clone(), self.current_value.clone())];
                let parsed = self.parse_argument_pairs(&pair, func_name);
                let value = parsed
                    .into_iter()
                    .find(|(k, _)| *k == self.current_key)
                    .expect("parsed");
                out.push_str(&value.1.dumped);
            } else if self.value_started {
                if !last.is_empty() {
                    match kind.as_str() {
                        "string" => out.push_str(&json_string_body(&last)),
                        _ => out.push_str(&last),
                    }
                }
                if kind == "string" {
                    out.push('"');
                }
            } else if kind == "string" {
                out.push_str(&python_json(
                    &Value::String(self.current_value.clone()),
                    false,
                ));
            } else {
                out.push_str(&self.current_value);
            }
            self.xml_tag_buffer.clear();
            self.state = State::Between;
            self.current_value.clear();
            self.value_started = false;
            self.cached_value_type = None;
            return;
        }
        let potential_close = self.xml_tag_buffer.len() <= VALUE_CLOSE.len()
            && VALUE_CLOSE.starts_with(self.xml_tag_buffer.as_str());
        if potential_close {
            return;
        }
        let content = std::mem::take(&mut self.xml_tag_buffer);
        if parsed_kind {
            self.current_value.push_str(&content);
        } else if kind == "string" {
            if !self.value_started {
                out.push('"');
                self.value_started = true;
            }
            out.push_str(&json_string_body(&content));
            self.current_value.push_str(&content);
        } else {
            self.value_started = true;
            out.push_str(&content);
            self.current_value.push_str(&content);
        }
    }

    /// `_match_tool_call`: (name, raw arguments, the closer or "", end of the match).
    fn match_tool_call(&mut self, text: &str) -> Option<(String, String, bool, usize)> {
        if self.open_call.is_none() {
            let m = PARTIAL_RE.captures(text)?;
            let (args, closed) = (m.get(2), m.get(3).is_some());
            let end = m.get(0).expect("whole match").end();
            let name = py_strip(m.get(1).map_or("", |g| g.as_str())).to_owned();
            match args {
                Some(args) if !closed => {
                    self.open_call = Some((name, args.start()));
                    self.end_scan_pos = args.start();
                }
                args => {
                    let args = py_strip(args.map_or("", |g| g.as_str())).to_owned();
                    return Some((name, args, closed, end));
                }
            }
        }
        let (name, args_start) = self.open_call.clone().expect("an open call");
        let (end_start, end_end, closed) = match text[self.end_scan_pos..].find(EOT) {
            Some(at) => {
                let start = self.end_scan_pos + at;
                (start, start + EOT.len(), true)
            }
            None => (text.len(), text.len(), false),
        };
        if closed {
            self.open_call = None;
        } else {
            let mut pos = (text.len() + 1).saturating_sub(EOT.len()).max(args_start);
            while !text.is_char_boundary(pos) {
                pos -= 1;
            }
            self.end_scan_pos = pos;
        }
        let args = py_strip(&text[args_start..end_start]).to_owned();
        Some((name, args, closed, end_end))
    }

    /// `_process_arguments_streaming`.
    fn arguments_streaming(&mut self, func_name: &str, args: &str) -> Option<ToolCallItem> {
        if self.buffer_arguments.is_none() {
            self.buffer_arguments =
                Some(self.schema(func_name).is_some_and(needs_argument_context));
        }
        if self.buffer_arguments == Some(true) || args.len() <= self.streamed_raw_length {
            return None;
        }
        let increment = args[self.streamed_raw_length..].to_owned();
        let json = self.xml_to_json(&increment, func_name);
        self.streamed_raw_length = args.len();
        (!json.is_empty()).then(|| ToolCallItem {
            tool_index: self.current_tool_id,
            name: String::new(),
            parameters: json,
        })
    }

    /// `_finalize_tool_call`.
    fn finalize(
        &mut self,
        func_name: &str,
        args: &str,
        match_end: usize,
        text: &str,
    ) -> Vec<ToolCallItem> {
        let item = |id, parameters: String| ToolCallItem {
            tool_index: id,
            name: String::new(),
            parameters,
        };
        let mut calls = Vec::new();
        if self.buffer_arguments == Some(true) {
            let arguments = self.parse_argument_pairs(&arg_pairs(args), func_name);
            calls.push(item(self.current_tool_id, dump_arguments(arguments)));
            self.sent_empty_object = true;
        }
        if !self.sent_empty_object {
            let close = if self.is_first_param { "{}" } else { "}" };
            calls.push(item(self.current_tool_id, close.into()));
            self.sent_empty_object = true;
        }
        self.buffer = text[match_end..].to_owned();
        self.tool_call_completed = true;
        self.current_tool_id += 1;
        self.current_tool_name_sent = false;
        self.streamed_raw_length = 0;
        self.reset_streaming_state();
        calls
    }

    /// `parse_streaming_increment`.
    fn parse_streaming_increment(&mut self, new_text: &str) -> Parsed {
        self.buffer.push_str(new_text);
        let mut current = self.buffer.clone();
        let Some(first) = current.find(BOT) else {
            if partial_start(&current) {
                return (String::new(), Vec::new());
            }
            self.buffer.clear();
            return (current.replace(EOT, ""), Vec::new());
        };
        let mut normal = String::new();
        if first > 0 {
            normal = current[..first].to_owned();
            current = current[first..].to_owned();
            self.buffer = current.clone();
        }
        let Some((name, args, closed, match_end)) = self.match_tool_call(&current) else {
            return (normal, Vec::new());
        };
        if self.current_tool_id == -1 {
            self.current_tool_id = 0;
            self.streamed_raw_length = 0;
            self.current_tool_name_sent = false;
            self.reset_streaming_state();
        }
        let mut calls = Vec::new();
        // `_send_tool_name_if_needed`.
        if !self.current_tool_name_sent
            && (current.contains("<arg_key") || closed)
            && !name.is_empty()
        {
            self.current_tool_name_sent = true;
            self.streamed_raw_length = 0;
            self.reset_streaming_state();
            calls.push(ToolCallItem {
                tool_index: self.current_tool_id,
                name: name.clone(),
                parameters: String::new(),
            });
        }
        if self.current_tool_name_sent {
            calls.extend(self.arguments_streaming(&name, &args));
            if closed && !self.tool_call_completed {
                calls.extend(self.finalize(&name, &args, match_end, &current));
            }
        }
        (normal, calls)
    }
}

impl ToolDetector for Glm47Detector {
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains(BOT)
    }

    fn parse_non_stream(&self, text: &str) -> Parsed {
        match self.has_tool_call(text) {
            true => self.detect_and_parse(text),
            false => (text.to_owned(), Vec::new()),
        }
    }

    /// The base `finish` flushes nothing.
    fn parse_stream(&mut self, text: &str, _flush: bool) -> Parsed {
        self.parse_streaming_increment(text)
    }
}

/// Whether the text ends with a proper or whole prefix of `<tool_call>`.
fn partial_start(text: &str) -> bool {
    (1..=BOT.len().min(text.len()))
        .any(|i| text.is_char_boundary(text.len() - i) && BOT.starts_with(&text[text.len() - i..]))
}

/// `func_arg_regex.findall`.
fn arg_pairs(args: &str) -> Vec<(String, String)> {
    ARG_RE
        .captures_iter(args)
        .map(|c| (c[1].to_owned(), c[2].to_owned()))
        .collect()
}

/// `json.dumps(s, ensure_ascii=False)[1:-1]`.
fn json_string_body(text: &str) -> String {
    let dumped = python_json(&Value::String(text.to_owned()), false);
    dumped[1..dumped.len() - 1].to_owned()
}

/// `json.dumps(arguments, ensure_ascii=False)`.
fn dump_arguments(arguments: Vec<(String, Arg)>) -> String {
    python_dict(arguments.into_iter().map(|(k, v)| (k, v.dumped)))
}

/// `dict[key] = value`: a repeated key keeps its first position.
fn set(map: &mut Vec<(String, Arg)>, key: String, value: Arg) {
    match map.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = value,
        None => map.push((key, value)),
    }
}

/// A parsed argument: its `json.dumps` text, and its value where JSON holds it
/// (`None` for NaN and the infinities).
#[derive(Clone)]
struct Arg {
    dumped: String,
    value: Option<Value>,
}

impl Arg {
    fn string(text: &str) -> Self {
        let value = Value::String(text.to_owned());
        Self {
            dumped: python_json(&value, false),
            value: Some(value),
        }
    }

    fn from_dumped(dumped: String) -> Self {
        let value = serde_json::from_str(&dumped).ok();
        Self { dumped, value }
    }

    fn is_string(&self) -> bool {
        matches!(self.value, Some(Value::String(_)))
    }

    fn json_type(&self) -> &'static str {
        self.value.as_ref().map_or("number", json_type)
    }

    fn probe(&self) -> Probe {
        match &self.value {
            Some(value) => Probe::Value(value.clone()),
            None => Probe::NonFinite,
        }
    }
}

/// `parse_arguments`: the value, and whether it parsed.
fn parse_arguments(raw: &str, arg_type: &str) -> (Arg, bool) {
    // Strategy 1: the value is JSON.
    if let Some(dumped) = python_json_reencode(raw) {
        return (
            coerce_numeric_string(Arg::from_dumped(dumped), arg_type),
            true,
        );
    }
    // Strategy 2: the value is the body of a JSON string holding JSON.
    let wrapped = format!("{{\"tmp\": \"{raw}\"}}");
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&wrapped)
        && let Some(Value::String(inner)) = map.get("tmp")
        && let Some(dumped) = python_json_reencode(inner)
    {
        return (
            coerce_numeric_string(Arg::from_dumped(dumped), arg_type),
            true,
        );
    }
    // Strategy 2.5: a string keeps its bytes, without wrapping quotes.
    if arg_type == "string" {
        let quoted = raw.len() >= 2
            && (raw.starts_with('"') && raw.ends_with('"')
                || raw.starts_with('\'') && raw.ends_with('\''));
        return match quoted {
            true => (Arg::string(&raw[1..raw.len() - 1]), true),
            false => (Arg::string(raw), true),
        };
    }
    // Strategy 3: a Python literal.
    if let Some(dumped) = literal_eval(raw) {
        return (Arg::from_dumped(dumped), true);
    }
    // Strategy 4: the string itself.
    (Arg::string(raw), true)
}

/// `_coerce_numeric_string`.
fn coerce_numeric_string(arg: Arg, arg_type: &str) -> Arg {
    let Some(Value::String(text)) = &arg.value else {
        return arg;
    };
    let as_int = || py_int(text).map(Arg::from_dumped);
    let as_float = || py_float(text).map(|f| Arg::from_dumped(python_float(f)));
    match arg_type {
        "integer" => as_int().unwrap_or(arg),
        // `_convert_to_number`.
        "number" if text.contains('.') || text.to_lowercase().contains('e') => {
            as_float().unwrap_or(arg)
        }
        "number" => as_int().unwrap_or(arg),
        _ => arg,
    }
}

/// `_json_type`: an integral float is an integer.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(n) => match n.as_f64().is_some_and(|f| f.fract() == 0.0) {
            true => "integer",
            false => "number",
        },
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `_json_equal`.
fn json_equal(left: &Value, right: &Value) -> bool {
    if json_type(left) != json_type(right) {
        return false;
    }
    match (left, right) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| json_equal(v, w)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(v, w)| json_equal(v, w))
        }
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (a, b) => a == b,
    }
}

/// The `value` argument of `_argument_types`.
enum Probe {
    Unset,
    Value(Value),
    NonFinite,
}

impl Probe {
    fn equals(&self, allowed: &Value) -> bool {
        match self {
            Probe::Unset => true,
            Probe::Value(value) => json_equal(value, allowed),
            Probe::NonFinite => false,
        }
    }

    fn json_type(&self) -> &'static str {
        match self {
            Probe::Value(value) => json_type(value),
            _ => "number",
        }
    }
}

type Types = BTreeSet<&'static str>;

fn type_name(kind: &str) -> &'static str {
    match kind {
        "null" => "null",
        "boolean" => "boolean",
        "integer" => "integer",
        "number" => "number",
        "string" => "string",
        "array" => "array",
        "object" => "object",
        // Python keeps any name; none of the others can match a value's type.
        _ => "other",
    }
}

/// `_get_argument_types` over the function's `parameters`.
fn argument_types_of(
    schema: Option<&Value>,
    key: &str,
    known: &[(String, Arg)],
    value: &Probe,
) -> Option<Types> {
    let schema = schema?;
    argument_types(schema, schema, Some(key), &mut Vec::new(), known, value)
}

/// `get_argument_type`'s reading of the types.
fn argument_type(types: Option<Types>) -> Option<String> {
    let types = types?;
    if types.len() == 2 && types.contains("number") && types.contains("integer") {
        return Some("number".into());
    }
    (types.len() == 1).then(|| (*types.iter().next().expect("one")).to_owned())
}

/// A local `$ref` (`#` or `#/...`) resolved against `root`.
fn resolve_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    if reference != "#" && !reference.starts_with("#/") {
        return None;
    }
    let mut target = root;
    if reference != "#" {
        for part in reference[2..].split('/') {
            let part = part.replace("~1", "/").replace("~0", "~");
            target = match target {
                Value::Array(items) => items.get(py_index(&part, items.len())?)?,
                Value::Object(map) => map.get(&part)?,
                _ => return None,
            };
        }
    }
    Some(target)
}

/// Python's `list[int(part)]`, negative indices included.
fn py_index(part: &str, len: usize) -> Option<usize> {
    let index: i64 = py_int(part)?.parse().ok()?;
    match index < 0 {
        true => len.checked_sub(index.unsigned_abs() as usize),
        false => Some(index as usize),
    }
}

/// `_argument_types`: the JSON types an argument (`key`) or a value (no key)
/// may take under `schema`; `None` where the schema does not say.
fn argument_types(
    schema: &Value,
    root: &Value,
    key: Option<&str>,
    seen: &mut Vec<*const Value>,
    known: &[(String, Arg)],
    value: &Probe,
) -> Option<Types> {
    let Value::Object(map) = schema else {
        return None;
    };
    if seen.contains(&(schema as *const Value)) {
        return None;
    }
    let depth = seen.len();
    seen.push(schema);
    let mut constraints: Vec<Types> = Vec::new();
    if let Some(Value::String(reference)) = map.get("$ref")
        && let Some(target) = resolve_ref(reference, root)
        && let Some(types) = argument_types(target, root, key, seen, known, value)
    {
        constraints.push(types);
    }
    match key {
        Some(key) => {
            let field = map.get("properties").and_then(|p| p.get(key));
            if let Some(field) = field
                && let Some(types) = argument_types(field, root, None, seen, &[], value)
            {
                constraints.push(types);
            }
        }
        None => {
            let widen = |mut types: Types| {
                if types.contains("number") {
                    types.insert("integer");
                }
                types
            };
            match map.get("type") {
                Some(Value::String(kind)) => {
                    constraints.push(widen(Types::from([type_name(kind)])))
                }
                Some(Value::Array(kinds)) => constraints.push(widen(
                    kinds
                        .iter()
                        .filter_map(Value::as_str)
                        .map(type_name)
                        .collect(),
                )),
                _ => {}
            }
            let values = match map.get("const") {
                Some(constant) => Some(std::slice::from_ref(constant)),
                None => map.get("enum").and_then(Value::as_array).map(Vec::as_slice),
            };
            if let Some(values) = values {
                constraints.push(
                    values
                        .iter()
                        .filter(|v| value.equals(v))
                        .map(json_type)
                        .collect(),
                );
            }
        }
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = map.get(keyword) {
            let alternatives: Vec<Option<Types>> = branches
                .iter()
                .filter(|branch| key.is_none() || matches_discriminators(branch, root, known))
                .map(|branch| argument_types(branch, root, key, seen, known, value))
                .collect();
            if !alternatives.is_empty() && alternatives.iter().all(Option::is_some) {
                constraints.push(alternatives.into_iter().flatten().flatten().collect());
            }
        }
    }
    if let Some(Value::Array(branches)) = map.get("allOf") {
        for branch in branches {
            if let Some(types) = argument_types(branch, root, key, seen, known, value) {
                constraints.push(types);
            }
        }
    }
    seen.truncate(depth);
    let mut constraints = constraints.into_iter();
    let first = constraints.next()?;
    Some(constraints.fold(first, |acc, types| {
        acc.intersection(&types).copied().collect()
    }))
}

/// `_matches_discriminators`: every known argument's type fits the branch.
fn matches_discriminators(branch: &Value, root: &Value, known: &[(String, Arg)]) -> bool {
    known.iter().all(|(name, arg)| {
        let probe = arg.probe();
        match argument_types(branch, root, Some(name), &mut Vec::new(), &[], &probe) {
            Some(types) => types.contains(probe.json_type()),
            None => true,
        }
    })
}

/// `_needs_argument_context`: some property's type is a union only the
/// sibling arguments can narrow.
fn needs_argument_context(schema: &Value) -> bool {
    let mut keys: Vec<String> = Vec::new();
    let mut has_union = false;
    let mut seen: Vec<*const Value> = Vec::new();
    let mut stack = vec![schema];
    while let Some(node) = stack.pop() {
        let Value::Object(map) = node else { continue };
        if seen.contains(&(node as *const Value)) {
            continue;
        }
        seen.push(node);
        if let Some(Value::Object(properties)) = map.get("properties") {
            for key in properties.keys() {
                if !keys.contains(key) {
                    keys.push(key.clone());
                }
            }
        }
        if let Some(Value::String(reference)) = map.get("$ref")
            && let Some(target) = resolve_ref(reference, schema)
        {
            stack.push(target);
        }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(branches)) = map.get(keyword) {
                has_union |= keyword != "allOf";
                stack.extend(branches.iter());
            }
        }
    }
    has_union
        && keys.iter().any(|key| {
            match argument_types(
                schema,
                schema,
                Some(key),
                &mut Vec::new(),
                &[],
                &Probe::Unset,
            ) {
                None => true,
                Some(types) => {
                    types.len() > 1
                        && !(types.len() == 2
                            && types.contains("number")
                            && types.contains("integer"))
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}),
        ]
    }

    #[test]
    fn a_whole_call_parses_with_typed_values() {
        let text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                    <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        let (normal, calls) = Glm47Detector::new(weather()).parse_non_stream(text);
        assert_eq!(normal, "");
        assert_eq!(calls[0].parameters, r#"{"city": "Paris", "days": 3}"#);
    }
}
