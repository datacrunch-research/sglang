//! The OpenAI Responses API lowered to chat, so one implementation serves every
//! host (sglang-server, sgl-router, an upstream router). [`lower`] turns a
//! `/v1/responses` body into a chat completion request the host lowers with
//! [`super::lower_chat`]: `input` items become messages, `instructions` a
//! system turn, flat function tools the nested chat form, `text.format` a
//! response format. [`ResponseShaper`] then rewrites the chat output, SSE
//! chunks or a completion, into Responses events and the `response` object.
//! State (`previous_response_id`, `store: true`, `conversation`,
//! `background`) is refused: the hosts keep no response store.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

/// A request the Responses API cannot lower; the message is the client's 400.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ResponsesError(pub String);

/// `{"error": {message, type, param, code}}` from a chat-shaped error body,
/// the Responses API's error shape.
pub fn error_body(status: u16, chat_body: &str) -> String {
    let parsed: Value = serde_json::from_str(chat_body).unwrap_or(Value::Null);
    let inner = parsed.get("error").cloned().unwrap_or(parsed);
    let message = inner
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("request failed")
        .to_owned();
    json!({"error": {"message": message, "type": error_type(status), "param": null, "code": null}})
        .to_string()
}

pub fn error_type(status: u16) -> &'static str {
    match status {
        400 | 422 => "invalid_request_error",
        401 | 403 => "authentication_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        _ => "server_error",
    }
}

/// The chat request and what the response object echoes.
pub struct Lowered {
    pub chat: Value,
    pub context: ResponseContext,
}

/// What a `response` object repeats from its request.
#[derive(Debug, Clone)]
pub struct ResponseContext {
    pub id: String,
    pub created_at: u64,
    pub model: String,
    pub stream: bool,
    echo: Map<String, Value>,
}

impl ResponseContext {
    /// The `response` object with the given status, output and usage.
    fn response(
        &self,
        status: &str,
        output: Vec<Value>,
        usage: Option<Value>,
        incomplete: Option<Value>,
    ) -> Value {
        let mut response = json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "background": false,
            "error": null,
            "incomplete_details": incomplete,
            "instructions": null,
            "max_output_tokens": null,
            "max_tool_calls": null,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "prompt_cache_key": null,
            "reasoning": {"effort": null, "summary": null},
            "safety_identifier": null,
            "service_tier": "default",
            "store": false,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_logprobs": 0,
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": usage,
            "user": null,
            "metadata": {},
        });
        for (key, value) in &self.echo {
            response[key] = value.clone();
        }
        response
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn fresh_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn bad(message: impl Into<String>) -> ResponsesError {
    ResponsesError(message.into())
}

/// Lowers a Responses request to a chat completion request.
pub fn lower(body: &Value) -> Result<Lowered, ResponsesError> {
    let object = body
        .as_object()
        .ok_or_else(|| bad("body must be a JSON object"))?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("`model` is required"))?
        .to_owned();
    let stream = object.get("stream") == Some(&Value::Bool(true));

    // State the router does not keep.
    if object
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null())
    {
        return Err(bad(
            "`previous_response_id` is not supported: send the full `input` each turn",
        ));
    }
    if object.get("conversation").is_some_and(|v| !v.is_null()) {
        return Err(bad("`conversation` is not supported"));
    }
    if object.get("background") == Some(&Value::Bool(true)) {
        return Err(bad("`background` responses are not supported"));
    }
    if object.get("store") == Some(&Value::Bool(true)) {
        return Err(bad(
            "`store: true` is not supported: responses are not stored, send `store: false`",
        ));
    }

    let mut messages: Vec<Value> = Vec::new();
    match object.get("instructions") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => messages.push(json!({"role": "system", "content": text})),
        Some(_) => return Err(bad("`instructions` must be a string")),
    }
    match object.get("input") {
        Some(Value::String(text)) => messages.push(json!({"role": "user", "content": text})),
        Some(Value::Array(items)) => lower_items(items, &mut messages)?,
        None | Some(Value::Null) => return Err(bad("`input` is required")),
        Some(_) => return Err(bad("`input` must be a string or a list of items")),
    }
    if messages.is_empty() {
        return Err(bad("`input` has no messages"));
    }

    let mut chat = json!({"model": model, "messages": messages, "stream": stream});
    if stream {
        chat["stream_options"] = json!({"include_usage": true});
    }
    let mut echo = Map::new();

    let copy = |chat: &mut Value, echo: &mut Map<String, Value>, from: &str, to: &str| {
        if let Some(v) = object.get(from).filter(|v| !v.is_null()) {
            chat[to] = v.clone();
            echo.insert(from.to_owned(), v.clone());
        }
    };
    copy(&mut chat, &mut echo, "temperature", "temperature");
    copy(&mut chat, &mut echo, "top_p", "top_p");
    copy(&mut chat, &mut echo, "max_output_tokens", "max_tokens");
    copy(&mut chat, &mut echo, "user", "user");
    copy(
        &mut chat,
        &mut echo,
        "parallel_tool_calls",
        "parallel_tool_calls",
    );
    copy(&mut chat, &mut echo, "prompt_cache_key", "prompt_cache_key");
    copy(
        &mut chat,
        &mut echo,
        "safety_identifier",
        "safety_identifier",
    );
    // SGLang extensions clients use on chat, carried through unchanged.
    for key in [
        "chat_template_kwargs",
        "separate_reasoning",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
        "seed",
    ] {
        if let Some(v) = object.get(key).filter(|v| !v.is_null()) {
            chat[key] = v.clone();
        }
    }
    for key in [
        "instructions",
        "metadata",
        "max_tool_calls",
        "top_logprobs",
        "truncation",
        "service_tier",
    ] {
        if let Some(v) = object.get(key).filter(|v| !v.is_null()) {
            echo.insert(key.to_owned(), v.clone());
        }
    }
    if let Some(v) = object
        .get("top_logprobs")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
    {
        chat["logprobs"] = Value::Bool(true);
        chat["top_logprobs"] = Value::from(v);
    }
    if let Some(reasoning) = object.get("reasoning").filter(|v| v.is_object()) {
        if let Some(effort) = reasoning.get("effort").filter(|v| !v.is_null()) {
            chat["reasoning_effort"] = effort.clone();
        }
        echo.insert("reasoning".to_owned(), reasoning.clone());
    }
    if let Some(text) = object.get("text").filter(|v| v.is_object()) {
        if let Some(format) = text.get("format").filter(|v| v.is_object())
            && let Some(response_format) = lower_text_format(format)?
        {
            chat["response_format"] = response_format;
        }
        echo.insert("text".to_owned(), text.clone());
    }
    if let Some(tools) = object
        .get("tools")
        .and_then(Value::as_array)
        .filter(|t| !t.is_empty())
    {
        chat["tools"] = Value::Array(tools.iter().map(lower_tool).collect::<Result<_, _>>()?);
        echo.insert("tools".to_owned(), Value::Array(tools.clone()));
    }
    if let Some(choice) = object.get("tool_choice").filter(|v| !v.is_null()) {
        chat["tool_choice"] = lower_tool_choice(choice)?;
        echo.insert("tool_choice".to_owned(), choice.clone());
    }
    echo.insert("store".to_owned(), Value::Bool(false));

    let context = ResponseContext {
        id: fresh_id("resp"),
        created_at: now_secs(),
        model,
        stream,
        echo,
    };
    Ok(Lowered { chat, context })
}

/// Responses input items as chat messages. Function calls become the
/// assistant's `tool_calls`, their outputs `tool` messages, reasoning items
/// the next assistant message's `reasoning_content`.
fn lower_items(items: &[Value], messages: &mut Vec<Value>) -> Result<(), ResponsesError> {
    let mut pending_reasoning: Option<String> = None;
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| bad("`input` items must be objects"))?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match kind {
            "message" => {
                let role = match object.get("role").and_then(Value::as_str) {
                    Some("developer") => "system",
                    Some(role @ ("user" | "assistant" | "system")) => role,
                    Some(other) => return Err(bad(format!("unsupported message role `{other}`"))),
                    None => return Err(bad("`input` message without a role")),
                };
                let content = lower_content(object.get("content"), role)?;
                let mut message = json!({"role": role, "content": content});
                if role == "assistant"
                    && let Some(reasoning) = pending_reasoning.take()
                {
                    message["reasoning_content"] = Value::String(reasoning);
                }
                messages.push(message);
            }
            "function_call" => {
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad("function_call without a name"))?;
                let call_id = object.get("call_id").and_then(Value::as_str).unwrap_or("");
                let arguments = match object.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(other @ Value::Object(_)) => other.to_string(),
                    _ => "{}".to_owned(),
                };
                let call = json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                // Consecutive calls ride on one assistant message, as chat expresses them.
                match messages.last_mut() {
                    Some(last)
                        if last["role"] == "assistant"
                            && last.get("tool_calls").is_some_and(Value::is_array)
                            && pending_reasoning.is_none() =>
                    {
                        last["tool_calls"].as_array_mut().expect("array").push(call);
                    }
                    _ => {
                        let mut message =
                            json!({"role": "assistant", "content": null, "tool_calls": [call]});
                        if let Some(reasoning) = pending_reasoning.take() {
                            message["reasoning_content"] = Value::String(reasoning);
                        }
                        messages.push(message);
                    }
                }
            }
            "function_call_output" => {
                let call_id = object.get("call_id").and_then(Value::as_str).unwrap_or("");
                let output = match object.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join(""),
                    Some(other) if !other.is_null() => other.to_string(),
                    _ => String::new(),
                };
                messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
            }
            "reasoning" => {
                // Replayed reasoning: its text, when the client has it; encrypted
                // content cannot be replayed through chat and is dropped.
                let text = reasoning_text(object);
                if !text.is_empty() {
                    pending_reasoning = Some(text);
                }
            }
            "item_reference" => {
                return Err(bad(
                    "`item_reference` items need a response store, which this router has none",
                ));
            }
            other => return Err(bad(format!("unsupported input item type `{other}`"))),
        }
    }
    Ok(())
}

fn reasoning_text(item: &Map<String, Value>) -> String {
    let from = |key: &str, part_type: &str| -> String {
        item.get(key)
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter(|p| p.get("type").and_then(Value::as_str) == Some(part_type))
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default()
    };
    let content = from("content", "reasoning_text");
    if !content.is_empty() {
        content
    } else {
        from("summary", "summary_text")
    }
}

/// A message's content as chat content: a string stays one; parts map by type.
fn lower_content(content: Option<&Value>, role: &str) -> Result<Value, ResponsesError> {
    match content {
        Some(Value::String(text)) => Ok(Value::String(text.clone())),
        Some(Value::Array(parts)) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                let object = part
                    .as_object()
                    .ok_or_else(|| bad("content parts must be objects"))?;
                let kind = object.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "input_text" | "output_text" | "text" | "refusal" => {
                        let text = object
                            .get("text")
                            .or_else(|| object.get("refusal"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        out.push(json!({"type": "text", "text": text}));
                    }
                    "input_image" => {
                        if object.get("file_id").is_some_and(|v| !v.is_null()) {
                            return Err(bad(
                                "`input_image.file_id` is not supported: send `image_url`",
                            ));
                        }
                        let url = object
                            .get("image_url")
                            .and_then(Value::as_str)
                            .ok_or_else(|| bad("input_image without image_url"))?;
                        let mut image = json!({"url": url});
                        if let Some(detail) = object.get("detail").filter(|v| !v.is_null()) {
                            image["detail"] = detail.clone();
                        }
                        out.push(json!({"type": "image_url", "image_url": image}));
                    }
                    "input_audio" => {
                        let audio = object.get("input_audio").cloned().unwrap_or_else(|| {
                            json!({"data": object.get("data").cloned().unwrap_or(Value::Null),
                                   "format": object.get("format").cloned().unwrap_or(Value::Null)})
                        });
                        out.push(json!({"type": "input_audio", "input_audio": audio}));
                    }
                    "input_file" => return Err(bad("`input_file` parts are not supported")),
                    other => {
                        return Err(bad(format!(
                            "unsupported content part type `{other}` in a {role} message"
                        )));
                    }
                }
            }
            // A single text part is what chat templates expect from an assistant.
            if role != "user" && out.len() == 1 && out[0]["type"] == "text" {
                return Ok(out[0]["text"].clone());
            }
            Ok(Value::Array(out))
        }
        None | Some(Value::Null) => Ok(Value::String(String::new())),
        Some(_) => Err(bad("message content must be a string or a list of parts")),
    }
}

fn lower_text_format(format: &Value) -> Result<Option<Value>, ResponsesError> {
    match format.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type": "json_object"}))),
        Some("json_schema") => {
            let mut schema = json!({
                "name": format.get("name").cloned().unwrap_or_else(|| Value::String("response".into())),
                "schema": format.get("schema").cloned().unwrap_or(Value::Null),
            });
            for key in ["strict", "description"] {
                if let Some(v) = format.get(key).filter(|v| !v.is_null()) {
                    schema[key] = v.clone();
                }
            }
            Ok(Some(json!({"type": "json_schema", "json_schema": schema})))
        }
        Some(other) => Err(bad(format!("unsupported text format `{other}`"))),
    }
}

/// A flat Responses function tool as a nested chat tool.
fn lower_tool(tool: &Value) -> Result<Value, ResponsesError> {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("function tool without a name"))?;
            let mut function = json!({"name": name});
            for key in ["description", "parameters", "strict"] {
                if let Some(v) = tool.get(key).filter(|v| !v.is_null()) {
                    function[key] = v.clone();
                }
            }
            Ok(json!({"type": "function", "function": function}))
        }
        Some("custom") => Err(bad(
            "`custom` tools with freeform arguments are not supported; use a function tool",
        )),
        Some(other) => Err(bad(format!("hosted tool type `{other}` is not supported"))),
        None => Err(bad("tool without a type")),
    }
}

fn lower_tool_choice(choice: &Value) -> Result<Value, ResponsesError> {
    match choice {
        Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
            Ok(choice.clone())
        }
        Value::String(other) => Err(bad(format!("unsupported tool_choice `{other}`"))),
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad("tool_choice.function without a name"))?;
                Ok(json!({"type": "function", "function": {"name": name}}))
            }
            Some("allowed_tools") => match object.get("mode").and_then(Value::as_str) {
                Some("required") => Ok(Value::String("required".into())),
                _ => Ok(Value::String("auto".into())),
            },
            Some(other) => Err(bad(format!("unsupported tool_choice type `{other}`"))),
            None => Err(bad("tool_choice object without a type")),
        },
        _ => Err(bad("tool_choice must be a string or an object")),
    }
}

// --- Output: chat chunks and completions as Responses events and objects. ---

#[derive(Debug)]
enum ItemKind {
    Message {
        text: String,
    },
    Reasoning {
        text: String,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug)]
struct Item {
    id: String,
    kind: ItemKind,
    done: bool,
}

impl Item {
    fn value(&self, done: bool) -> Value {
        let status = if done { "completed" } else { "in_progress" };
        match &self.kind {
            ItemKind::Message { text } => json!({
                "id": self.id, "type": "message", "status": status, "role": "assistant",
                "content": if text.is_empty() && !done { json!([]) } else {
                    json!([{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]) },
            }),
            ItemKind::Reasoning { text } => json!({
                "id": self.id, "type": "reasoning", "status": status, "summary": [],
                "content": if text.is_empty() && !done { json!([]) } else { json!([{"type": "reasoning_text", "text": text}]) },
            }),
            ItemKind::FunctionCall {
                call_id,
                name,
                arguments,
            } => json!({
                "id": self.id, "type": "function_call", "status": status,
                "call_id": call_id, "name": name, "arguments": arguments,
            }),
        }
    }
}

/// Rewrites one chat stream, or one chat completion, as the Responses API.
pub struct ResponseShaper {
    ctx: ResponseContext,
    seq: u64,
    items: Vec<Item>,
    /// The item taking text right now (message or reasoning), by index.
    open: Option<usize>,
    /// Function-call items by the chat tool-call index.
    calls: Vec<(u64, usize)>,
    usage: Option<Value>,
    finish_reason: Option<String>,
    failed: bool,
    completed: bool,
}

impl ResponseShaper {
    pub fn new(ctx: ResponseContext) -> Self {
        Self {
            ctx,
            seq: 0,
            items: Vec::new(),
            open: None,
            calls: Vec::new(),
            usage: None,
            finish_reason: None,
            failed: false,
            completed: false,
        }
    }

    fn emit(&mut self, kind: &str, mut body: Value) -> String {
        body["type"] = Value::String(kind.to_owned());
        body["sequence_number"] = Value::from(self.seq);
        self.seq += 1;
        format!("event: {kind}\ndata: {body}\n\n")
    }

    fn output(&self) -> Vec<Value> {
        self.items.iter().map(|item| item.value(true)).collect()
    }

    /// Closes the item at `index`: its text or arguments done, then the item.
    fn close(&mut self, index: usize) -> Vec<String> {
        if self.items[index].done {
            return Vec::new();
        }
        self.items[index].done = true;
        let id = self.items[index].id.clone();
        let mut out = Vec::new();
        match &self.items[index].kind {
            ItemKind::Message { text } => {
                let text = text.clone();
                out.push(self.emit(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text, "logprobs": []}),
                ));
                out.push(self.emit(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []}}),
                ));
            }
            ItemKind::Reasoning { text } => {
                let text = text.clone();
                out.push(self.emit(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                ));
            }
            ItemKind::FunctionCall { arguments, .. } => {
                let arguments = arguments.clone();
                out.push(self.emit(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": arguments}),
                ));
            }
        }
        let item = self.items[index].value(true);
        out.push(self.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        ));
        if self.open == Some(index) {
            self.open = None;
        }
        out
    }

    fn close_open(&mut self) -> Vec<String> {
        match self.open.take() {
            Some(index) => self.close(index),
            None => Vec::new(),
        }
    }

    /// Opens a message or reasoning item and returns its index.
    fn open_text_item(&mut self, reasoning: bool) -> (usize, Vec<String>) {
        if let Some(index) = self.open
            && matches!(
                (&self.items[index].kind, reasoning),
                (ItemKind::Message { .. }, false) | (ItemKind::Reasoning { .. }, true)
            )
        {
            return (index, Vec::new());
        }
        let mut out = self.close_open();
        let kind = if reasoning {
            ItemKind::Reasoning {
                text: String::new(),
            }
        } else {
            ItemKind::Message {
                text: String::new(),
            }
        };
        let item = Item {
            id: fresh_id(if reasoning { "rs" } else { "msg" }),
            kind,
            done: false,
        };
        let index = self.items.len();
        let value = item.value(false);
        self.items.push(item);
        self.open = Some(index);
        out.push(self.emit(
            "response.output_item.added",
            json!({"output_index": index, "item": value}),
        ));
        if !reasoning {
            let id = self.items[index].id.clone();
            out.push(self.emit(
                "response.content_part.added",
                json!({"item_id": id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
            ));
        }
        (index, out)
    }

    fn text_delta(&mut self, delta: &str, reasoning: bool) -> Vec<String> {
        let (index, mut out) = self.open_text_item(reasoning);
        let id = self.items[index].id.clone();
        match &mut self.items[index].kind {
            ItemKind::Message { text } | ItemKind::Reasoning { text } => text.push_str(delta),
            ItemKind::FunctionCall { .. } => unreachable!("text items only"),
        }
        let kind = if reasoning {
            "response.reasoning_text.delta"
        } else {
            "response.output_text.delta"
        };
        let mut body =
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta});
        if !reasoning {
            body["logprobs"] = json!([]);
        }
        out.push(self.emit(kind, body));
        out
    }

    fn tool_call_delta(&mut self, call: &Value) -> Vec<String> {
        let tool_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let name = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let arguments = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        let call_id = call.get("id").and_then(Value::as_str).unwrap_or("");
        let mut out = Vec::new();
        let index = match self.calls.iter().find(|(i, _)| *i == tool_index) {
            Some((_, index)) => *index,
            None => {
                out.extend(self.close_open());
                let call_id = if call_id.is_empty() {
                    fresh_id("call")
                } else {
                    call_id.to_owned()
                };
                let item = Item {
                    id: fresh_id("fc"),
                    kind: ItemKind::FunctionCall {
                        call_id,
                        name: name.to_owned(),
                        arguments: String::new(),
                    },
                    done: false,
                };
                let index = self.items.len();
                let value = item.value(false);
                self.items.push(item);
                self.calls.push((tool_index, index));
                out.push(self.emit(
                    "response.output_item.added",
                    json!({"output_index": index, "item": value}),
                ));
                index
            }
        };
        if let ItemKind::FunctionCall {
            name: item_name,
            arguments: item_args,
            ..
        } = &mut self.items[index].kind
        {
            if item_name.is_empty() && !name.is_empty() {
                *item_name = name.to_owned();
            }
            if !arguments.is_empty() {
                item_args.push_str(arguments);
                let id = self.items[index].id.clone();
                out.push(self.emit(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": index, "delta": arguments}),
                ));
            }
        }
        out
    }

    fn remember_usage(&mut self, chunk: &Value) {
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(usage_value(usage));
        }
    }

    fn fail(&mut self, status: u16, chat_body: &Value) -> Vec<String> {
        self.failed = true;
        let inner = chat_body
            .get("error")
            .cloned()
            .unwrap_or_else(|| chat_body.clone());
        let message = inner
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("request failed")
            .to_owned();
        let code = inner
            .get("code")
            .and_then(Value::as_u64)
            .map(|c| c as u16)
            .unwrap_or(status);
        let mut response = self
            .ctx
            .response("failed", self.output(), self.usage.clone(), None);
        response["error"] = json!({"code": error_type(code), "message": message});
        vec![self.emit("response.failed", json!({"response": response}))]
    }

    /// The terminal event: completed, or incomplete when the model hit its
    /// token budget.
    fn finish(&mut self) -> Vec<String> {
        if self.completed || self.failed {
            return Vec::new();
        }
        self.completed = true;
        let mut out = self.close_open();
        let open: Vec<usize> = (0..self.items.len())
            .filter(|i| !self.items[*i].done)
            .collect();
        for index in open {
            out.extend(self.close(index));
        }
        let (status, incomplete) = match self.finish_reason.as_deref() {
            Some("length") => ("incomplete", Some(json!({"reason": "max_output_tokens"}))),
            Some("content_filter") => ("incomplete", Some(json!({"reason": "content_filter"}))),
            _ => ("completed", None),
        };
        let response = self
            .ctx
            .response(status, self.output(), self.usage.clone(), incomplete);
        let kind = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        out.push(self.emit(kind, json!({"response": response})));
        out
    }
}

/// Chat usage in the Responses shape.
fn usage_value(usage: &Value) -> Value {
    let n = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input = n("prompt_tokens");
    let output = n("completion_tokens");
    let cached = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(input + output),
    })
}

impl ResponseShaper {
    // --- The host-facing surface: one chat stream or completion in, Responses out. ---

    /// Events sent before the first mapped event of a stream.
    pub fn start(&mut self) -> Vec<String> {
        let response = self.ctx.response("in_progress", Vec::new(), None, None);
        vec![
            self.emit("response.created", json!({"response": response})),
            self.emit("response.in_progress", json!({"response": response})),
        ]
    }

    /// Maps one chat SSE event (`data: ...\n\n`), including error events and `[DONE]`.
    pub fn event(&mut self, event: &str) -> Vec<String> {
        let Some(data) = event.lines().find_map(|l| l.strip_prefix("data:")) else {
            return Vec::new();
        };
        let data = data.trim();
        if data == "[DONE]" {
            // The Responses stream ends on its terminal event.
            return Vec::new();
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if chunk.get("error").is_some() {
            return self.fail(500, &chunk);
        }
        self.remember_usage(&chunk);
        let mut out = Vec::new();
        let Some(choice) = chunk.pointer("/choices/0") else {
            return out;
        };
        if let Some(delta) = choice.get("delta") {
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                out.extend(self.text_delta(reasoning, true));
            }
            if let Some(content) = delta
                .get("content")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                out.extend(self.text_delta(content, false));
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    out.extend(self.tool_call_delta(call));
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_owned());
        }
        out
    }

    /// Events after the chat stream's last event: the terminal event.
    pub fn end(&mut self) -> Vec<String> {
        self.finish()
    }

    /// Maps a unary chat reply, status and JSON body, to the Responses reply.
    pub fn unary(&mut self, status: u16, body: &str) -> (u16, String) {
        let parsed: Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(_) => return (status, error_body(status, body)),
        };
        if status != 200 || parsed.get("error").is_some() {
            return (status, error_body(status, body));
        }
        let mut items: Vec<Value> = Vec::new();
        if let Some(message) = parsed.pointer("/choices/0/message") {
            if let Some(reasoning) = message
                .get("reasoning_content")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                items.push(json!({"id": fresh_id("rs"), "type": "reasoning", "status": "completed", "summary": [],
                    "content": [{"type": "reasoning_text", "text": reasoning}]}));
            }
            if let Some(content) = message
                .get("content")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                items
                    .push(json!({"id": fresh_id("msg"), "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": [], "logprobs": []}]}));
            }
            for call in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let call_id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned);
                items.push(json!({"id": fresh_id("fc"), "type": "function_call", "status": "completed",
                    "call_id": call_id.unwrap_or_else(|| fresh_id("call")),
                    "name": call.pointer("/function/name").and_then(Value::as_str).unwrap_or(""),
                    "arguments": call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("")}));
            }
        }
        let finish = parsed
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str);
        let (state, incomplete) = match finish {
            Some("length") => ("incomplete", Some(json!({"reason": "max_output_tokens"}))),
            Some("content_filter") => ("incomplete", Some(json!({"reason": "content_filter"}))),
            _ => ("completed", None),
        };
        let usage = parsed
            .get("usage")
            .filter(|u| u.is_object())
            .map(usage_value);
        (
            200,
            self.ctx
                .response(state, items, usage, incomplete)
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(out: &[String]) -> Vec<(String, Value)> {
        out.iter()
            .map(|e| {
                let kind = e
                    .lines()
                    .next()
                    .unwrap()
                    .strip_prefix("event: ")
                    .unwrap()
                    .to_owned();
                let data = e.lines().nth(1).unwrap().strip_prefix("data: ").unwrap();
                (kind, serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[test]
    fn lowers_input_items_tools_and_format() {
        let body = json!({
            "model": "k3", "stream": true, "store": false, "instructions": "be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}], "encrypted_content": "zzz"},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "12C"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It is 12C."}]},
                {"type": "message", "role": "developer", "content": "and polite"}
            ],
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}, "strict": true}],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "text": {"format": {"type": "json_schema", "name": "w", "schema": {"type": "object"}, "strict": true}},
            "reasoning": {"effort": "low"}, "max_output_tokens": 50, "temperature": 0.2, "metadata": {"k": "v"},
            "prompt_cache_key": "sess-1"
        });
        let Lowered { chat, context } = lower(&body).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "be brief"})
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["reasoning_content"], "think");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            messages[3],
            json!({"role": "tool", "tool_call_id": "call_1", "content": "12C"})
        );
        assert_eq!(
            messages[4],
            json!({"role": "assistant", "content": "It is 12C."})
        );
        assert_eq!(messages[5]["role"], "system");
        assert_eq!(chat["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(chat["tools"][0]["function"]["strict"], true);
        assert_eq!(
            chat["tool_choice"],
            json!({"type": "function", "function": {"name": "get_weather"}})
        );
        assert_eq!(chat["response_format"]["type"], "json_schema");
        assert_eq!(chat["response_format"]["json_schema"]["name"], "w");
        assert_eq!(chat["reasoning_effort"], "low");
        assert_eq!(chat["max_tokens"], 50);
        assert_eq!(chat["stream_options"]["include_usage"], true);
        assert_eq!(chat["prompt_cache_key"], "sess-1");
        assert!(chat.get("metadata").is_none());
        assert!(context.stream);
        let response = context.response("completed", vec![], None, None);
        assert_eq!(response["metadata"], json!({"k": "v"}));
        assert_eq!(response["store"], false);
        assert_eq!(response["reasoning"]["effort"], "low");
        assert_eq!(response["tools"][0]["name"], "get_weather");
    }

    #[test]
    fn rejects_state_and_unsupported_shapes() {
        let base = json!({"model": "k3", "input": "hi"});
        let with = |k: &str, v: Value| {
            let mut b = base.clone();
            b[k] = v;
            b
        };
        assert!(lower(&base).is_ok());
        assert!(lower(&with("previous_response_id", json!("resp_1"))).is_err());
        assert!(lower(&with("conversation", json!("conv_1"))).is_err());
        assert!(lower(&with("background", json!(true))).is_err());
        assert!(lower(&with("store", json!(true))).is_err());
        assert!(lower(&with("store", json!(false))).is_ok());
        assert!(lower(&with("tools", json!([{"type": "custom", "name": "x"}]))).is_err());
        assert!(lower(&with("tools", json!([{"type": "web_search"}]))).is_err());
        assert!(
            lower(&with(
                "input",
                json!([{"type": "item_reference", "id": "x"}])
            ))
            .is_err()
        );
        assert!(
            lower(&with(
                "input",
                json!([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}])
            ))
            .is_err()
        );
        assert!(lower(&json!({"model": "k3"})).is_err());
        let image = with(
            "input",
            json!([{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"}]}]),
        );
        let lowered = lower(&image).unwrap();
        assert_eq!(
            lowered.chat["messages"][0]["content"][0]["type"],
            "image_url"
        );
        assert_eq!(
            lowered.chat["messages"][0]["content"][0]["image_url"]["detail"],
            "low"
        );
    }

    fn chunk(delta: Value, finish: Option<&str>, usage: Option<Value>) -> String {
        let mut c = json!({"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "k3",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        if let Some(usage) = usage {
            c["usage"] = usage;
        }
        format!("data: {c}\n\n")
    }

    #[test]
    fn streams_reasoning_text_and_tool_calls_as_events() {
        let ctx = lower(&json!({"model": "k3", "input": "hi", "stream": true}))
            .unwrap()
            .context;
        let mut shaper = ResponseShaper::new(ctx);
        let mut out = shaper.start();
        out.extend(shaper.event(&chunk(
            json!({"role": "assistant", "content": ""}),
            None,
            None,
        )));
        out.extend(shaper.event(&chunk(json!({"reasoning_content": "hmm"}), None, None)));
        out.extend(shaper.event(&chunk(json!({"content": "Hello"}), None, None)));
        out.extend(shaper.event(&chunk(json!({"content": " world"}), None, None)));
        out.extend(shaper.event(&chunk(
            json!({"tool_calls": [{"index": 0, "id": "call_9", "type": "function", "function": {"name": "f", "arguments": "{\"a\""}}]}),
            None,
            None,
        )));
        out.extend(shaper.event(&chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}),
            Some("tool_calls"),
            None,
        )));
        out.extend(shaper.event(&chunk(
            json!({}),
            None,
            Some(json!({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})),
        )));
        out.extend(shaper.event("data: [DONE]\n\n"));
        out.extend(shaper.end());
        let parsed = events(&out);
        let kinds: Vec<&str> = parsed.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        // Sequence numbers count up from zero.
        for (i, (_, body)) in parsed.iter().enumerate() {
            assert_eq!(body["sequence_number"], i as u64);
        }
        let completed = &parsed.last().unwrap().1["response"];
        assert_eq!(completed["status"], "completed");
        let output = completed["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "hmm");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "Hello world");
        assert_eq!(output[2]["type"], "function_call");
        assert_eq!(output[2]["call_id"], "call_9");
        assert_eq!(output[2]["name"], "f");
        assert_eq!(output[2]["arguments"], "{\"a\":1}");
        assert_eq!(completed["usage"]["input_tokens"], 3);
        assert_eq!(completed["usage"]["output_tokens"], 4);
        assert_eq!(completed["usage"]["total_tokens"], 7);
        // Nothing after the terminal event.
        assert!(shaper.end().is_empty());
    }

    #[test]
    fn length_stops_are_incomplete_and_errors_fail_the_response() {
        let ctx = lower(&json!({"model": "k3", "input": "hi", "stream": true}))
            .unwrap()
            .context;
        let mut shaper = ResponseShaper::new(ctx.clone());
        shaper.start();
        shaper.event(&chunk(json!({"content": "x"}), Some("length"), None));
        let end = events(&shaper.end());
        assert_eq!(end.last().unwrap().0, "response.incomplete");
        assert_eq!(
            end.last().unwrap().1["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );

        let mut shaper = ResponseShaper::new(ctx);
        shaper.start();
        let failed = events(
            &shaper
                .event("data: {\"error\": {\"message\": \"boom\", \"type\": \"BadRequestError\", \"code\": 400}}\n\n"),
        );
        assert_eq!(failed[0].0, "response.failed");
        assert_eq!(failed[0].1["response"]["status"], "failed");
        assert_eq!(failed[0].1["response"]["error"]["message"], "boom");
        assert!(shaper.end().is_empty());
    }

    #[test]
    fn unary_completion_becomes_a_response_object() {
        let ctx = lower(&json!({"model": "k3", "input": "hi"}))
            .unwrap()
            .context;
        let mut shaper = ResponseShaper::new(ctx);
        let completion = json!({"id": "c", "object": "chat.completion", "model": "k3",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "Sure",
                "reasoning_content": "plan", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 2}, "prompt_tokens_details": {"cached_tokens": 8}}});
        let (status, body) = shaper.unary(200, &completion.to_string());
        assert_eq!(status, 200);
        let response: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(response["object"], "response");
        assert_eq!(response["status"], "completed");
        let output = response["output"].as_array().unwrap();
        assert_eq!(
            output
                .iter()
                .map(|i| i["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["reasoning", "message", "function_call"]
        );
        assert_eq!(output[2]["call_id"], "call_1");
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            8
        );
        assert_eq!(
            response["usage"]["output_tokens_details"]["reasoning_tokens"],
            2
        );
        let (status, body) = shaper.unary(
            400,
            r#"{"object":"error","message":"bad","type":"BadRequestError","code":400}"#,
        );
        assert_eq!(status, 400);
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(error["error"]["message"], "bad");
    }
}
