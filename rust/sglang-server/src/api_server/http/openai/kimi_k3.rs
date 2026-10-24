//! Kimi-K3 chat completions, served as SGLang's Python chat layer serves them.
//!
//! K3 ships no Jinja template: Python renders it with `encoding_k3.py` and
//! tokenizes with `tokenization_kimi.py`. The pieces exist in `sglang-processor`
//! (the K3 renderer, `tiktoken.model` loading, the K3 reasoning and tool
//! parsers, and the OpenAI-over-`/generate` layer the experimental router
//! uses), so this module only wires them in: the body is lowered to a
//! `/generate` body whose `input_ids` are the rendered prompt encoded with
//! Kimi's cuts, submitted through this server's own `/generate`, and its output
//! is raised back to the OpenAI shape by the processor's responder. Request
//! shapes the processor does not reproduce (tool_choice required or named,
//! strict tools, system-turn tools, media, logprobs) are answered 400.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::Response;
use bytes::BytesMut;
use futures::{Stream, StreamExt, stream};
use serde_json::Value;
use sglang_processor::dynamo_tokenizers::{EncodeSegment, Tokenizer};
use sglang_processor::openai::{
    ChatModel, OpenAiHeaders, OpenAiSettings, Reply, Responder, lower_chat,
};
use sglang_processor::{ChatFormatter as Renderer, ChatFormatterOptions, select_chat_formatter};

use super::super::app::AppState;
use super::super::native_api;
use super::openai_error;
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::tokenizer::kimi_encode;

pub struct KimiK3Chat {
    renderer: Renderer,
    tokenizer: Tokenizer,
    settings: OpenAiSettings,
    model: ChatModel,
    /// `--default-chat-template-kwargs`.
    defaults: HashMap<String, Value>,
    /// SGLang drops a leading BOS from the separately encoded continuation prefix.
    bos: Option<u32>,
}

impl KimiK3Chat {
    /// The K3 chat support, or `None` for another model (or without `tiktoken.model`).
    pub(super) fn load(server_args: &ServerArgs) -> Option<Self> {
        if server_args.chat_template.is_some() {
            return None;
        }
        let options = ChatFormatterOptions {
            tokenizer_path: server_args.tokenizer_path.clone(),
            model_path: server_args.model_path.clone(),
            revision: server_args.revision.clone(),
            chat_template: None,
        };
        let (Some(renderer @ Renderer::KimiK3), _) = select_chat_formatter(&options) else {
            return None;
        };
        let path = server_args.tokenizer_path.as_str();
        let revision = server_args.revision.as_deref();
        let file = sglang_processor::resolve_tokenizer_file(path, revision)?;
        // A tokenizer.json cannot keep the special-token boundaries of the rendered segments.
        if !(file.ends_with("tiktoken.model") || file.ends_with(".tiktoken")) {
            tracing::warn!(%file, "Kimi-K3 chat needs tiktoken.model; OpenAI chat completions disabled");
            return None;
        }
        let tokenizer = match sglang_processor::load_tokenizer(Some(path), revision, false) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                tracing::warn!(%error, "Kimi-K3 chat tokenizer failed to load; chat disabled");
                return None;
            }
        };
        // The resolved server args (`/server_info`) carry what Python's chat layer reads.
        let info = Value::Object((*server_args.launch_server_info.0).clone());
        let settings: OpenAiSettings = match serde_json::from_value(info) {
            Ok(settings) => settings,
            Err(error) => {
                tracing::warn!(%error, "Kimi-K3 chat: server args unreadable; chat disabled");
                return None;
            }
        };
        let read_json = |name: &str| {
            sglang_processor::resolve_model_file(path, revision, name)
                .and_then(|file| std::fs::read_to_string(file).ok())
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        };
        let model = ChatModel {
            context_length: Some(server_args.model_config.context_len),
            generation_config: read_json("generation_config.json")
                .and_then(|config| config.as_object().cloned())
                .unwrap_or_default(),
            always_thinks: false,
            reasoning_toggle: None,
            chat_encoding_spec: Some("kimi_k3".into()),
        };
        let bos = read_json("tokenizer_config.json")
            .and_then(|config| match &config["bos_token"] {
                Value::String(token) => Some(token.clone()),
                Value::Object(token) => token.get("content")?.as_str().map(str::to_owned),
                _ => None,
            })
            .and_then(|token| {
                let ids = kimi_encode(&tokenizer, &[EncodeSegment::control(&token)]).ok()?;
                (ids.len() == 1).then(|| ids[0])
            });
        let defaults = settings
            .default_chat_template_kwargs
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        tracing::info!(%file, "Kimi-K3 chat: sglang-processor renderer, tiktoken encoding with Kimi's cuts");
        Some(Self {
            renderer,
            tokenizer,
            settings,
            model,
            defaults,
            bos,
        })
    }

    /// The prompt ids of a chat body, as Python's `_process_messages` makes them.
    pub(super) fn encode(&self, request: &Value) -> Result<Vec<u32>, String> {
        let (prompt, prefix) = self
            .renderer
            .render_request(request.clone(), &self.defaults)
            .map_err(|error| error.to_string())?;
        let mut ids = match prompt.encode_segments() {
            Some(segments) => kimi_encode(&self.tokenizer, &segments)?,
            None => kimi_encode(&self.tokenizer, &[EncodeSegment::control(prompt.as_str())])?,
        };
        if !prefix.is_empty() {
            let mut suffix = kimi_encode(&self.tokenizer, &[EncodeSegment::control(&prefix)])?;
            if self.bos.is_some() && suffix.first() == self.bos.as_ref() {
                suffix.remove(0);
            }
            ids.extend(suffix);
        }
        Ok(ids)
    }

    /// `POST /v1/chat/completions` for K3.
    pub(super) async fn chat(
        self: Arc<Self>,
        state: Arc<AppState>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Response {
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        };
        let routing_key = header("x-smg-routing-key");
        let custom_labels = (self
            .settings
            .tokenizer_metrics_custom_labels_header
            .as_deref())
        .and_then(&header);
        let sglext_ids = ["x-sglext-return-input-ids", "x-sglext-return-output-ids"]
            .iter()
            .any(|name| header(name).as_deref() == Some("1"));
        // Parsing, rendering and encoding are CPU work (a 1M-token prompt takes a fraction of a second).
        let this = self.clone();
        let lowered = tokio::task::spawn_blocking(move || {
            let headers = OpenAiHeaders {
                routing_key: routing_key.as_deref(),
                custom_labels: custom_labels.as_deref(),
                sglext_ids,
            };
            let mut render_error = None;
            let render = |request: &Value| match this.encode(request) {
                Ok(ids) => Some(ids),
                Err(error) => {
                    render_error = Some(error);
                    None
                }
            };
            let lowered = lower_chat(&body, &headers, &this.settings, &this.model, render, None);
            lowered.map_err(|unsupported| match render_error {
                Some(error) => format!("Kimi-K3 chat render failed: {error}"),
                None => format!(
                    "Kimi-K3 chat request not served by the Rust server: {}",
                    unsupported.0
                ),
            })
        })
        .await;
        let (generate_body, responder) = match lowered {
            Ok(Ok(lowered)) => lowered,
            Ok(Err(message)) => return openai_error(StatusCode::BAD_REQUEST, message, false),
            Err(error) => {
                return openai_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string(), false);
            }
        };
        let streaming = generate_body["stream"] == true;
        let response = native_api::generate(State(state), headers, Ok(Json(generate_body))).await;
        let mut responder = Some(Responder::Chat(Box::new(responder)));
        respond(&mut responder, response, streaming).await
    }
}

/// The OpenAI response to a `/generate` response (`experimental/sgl-router`'s
/// `routes/chat/openai.rs`, the same responder protocol).
async fn respond(
    responder: &mut Option<Responder>,
    response: Response<Body>,
    streaming: bool,
) -> Response<Body> {
    let (mut parts, body) = response.into_parts();
    let status = parts.status.as_u16();
    if !streaming || status != 200 {
        let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await else {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            return response;
        };
        let reply = if status != 200 {
            Responder::rejected(&bytes)
        } else {
            responder.take().expect("OpenAI responder").unary(&bytes)
        };
        return match reply {
            Some(reply) => json_response(reply),
            None => Response::from_parts(parts, Body::from(bytes)),
        };
    }

    let mut responder = responder.take().expect("OpenAI responder");
    // Python answers only once the first OpenAI chunk exists, so an early
    // error is still an error status.
    let mut frames = Box::pin(sse_data(body).fuse());
    let mut first = Vec::new();
    let mut failed = None;
    let mut last = Bytes::new();
    while first.is_empty() && failed.is_none() {
        match frames.next().await {
            None => break,
            Some(Err(error)) => failed = Some(error),
            Some(Ok(data)) => match responder.stream_data(&data) {
                Ok(events) => (first, last) = (events, data),
                Err(reply) => return json_response(reply),
            },
        }
    }
    let pending = after(frames, responder, &last);
    let rest = stream::unfold(pending, |pending| async move {
        let (mut frames, responder) = pending?;
        let Some(mut responder) = responder else {
            while frames.next().await.is_some() {}
            return None;
        };
        let (events, data) = match frames.next().await? {
            Ok(data) => (
                responder
                    .stream_data(&data)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|event| Ok(Bytes::from(event)))
                    .collect(),
                data,
            ),
            Err(error) => (vec![Err(error)], Bytes::new()),
        };
        Some((stream::iter(events), after(frames, responder, &data)))
    })
    .flatten();
    let events = stream::iter(first.into_iter().map(|event| Ok(Bytes::from(event))))
        .chain(stream::iter(failed.map(Err)))
        .chain(rest);
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from_stream(events))
}

/// What is left to read after `data`: more frames, the rest of a finished
/// stream (no responder), or nothing. Dropping the `/generate` stream aborts
/// the request, as Python does at an abort or error but not after `[DONE]`.
fn after<F>(frames: F, responder: Responder, data: &[u8]) -> Option<(F, Option<Responder>)> {
    if !responder.done() {
        Some((frames, Some(responder)))
    } else if data == b"[DONE]" {
        Some((frames, None))
    } else {
        None
    }
}

fn json_response(reply: Reply) -> Response<Body> {
    let mut response = Response::new(Body::from(reply.body));
    *response.status_mut() = reply.status.try_into().unwrap_or_default();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// The `data:` payloads of an SSE body, then its error if it broke.
fn sse_data(body: Body) -> impl Stream<Item = Result<Bytes, axum::Error>> {
    let chunks = body.into_data_stream();
    stream::unfold(
        (chunks, BytesMut::new(), 0),
        |(mut chunks, mut buffer, mut scanned)| async move {
            loop {
                if let Some(end) = buffer[scanned..].windows(2).position(|w| w == b"\n\n") {
                    let event = buffer.split_to(scanned + end + 2).freeze();
                    scanned = 0;
                    if let Some(data) = event.strip_prefix(b"data: ") {
                        let data = event.slice_ref(data.trim_ascii_end());
                        return Some((Ok(data), (chunks, buffer, scanned)));
                    }
                    continue;
                }
                // A delimiter may straddle the chunk boundary.
                scanned = buffer.len().saturating_sub(1);
                match chunks.next().await? {
                    Ok(chunk) => buffer.extend_from_slice(&chunk),
                    Err(error) => return Some((Err(error), (chunks, BytesMut::new(), 0))),
                }
            }
        },
    )
}
