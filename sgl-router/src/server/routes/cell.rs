// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cell-level signals for an upstream router. A cell is one router and the
//! engines it serves; a deployment that spreads one model over several cells
//! can put a global router in front of them that chooses the cell by prefix
//! affinity and load, the way this router chooses an engine.
//!
//! `POST /internal/prefix_match` answers for one tokenized prompt;
//! `GET /internal/cell_load` is the load summary alone; `GET /internal/cell_info`
//! says which model this cell serves and how its engines render it, so the
//! upstream can load the matching tokenizer and parsers without being told.
//! All three read state the router already keeps for its own routing (the KV
//! tree or the external Indexer, the engine load table, the admission limits)
//! and never touch an engine. Every field is advisory: an upstream that gets
//! no answer, or a late one, is expected to route by load instead.

use crate::buckets_reorg::BucketGroups;
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::PrefixLookupResult;
use crate::policies_reorg::admission::partition_admitted;
use crate::policies_reorg::{PickDistribution, PickError};
use crate::server::app_context::{AppContext, ChatRouting};
use crate::server::error::ApiError;
use crate::server::routes::chat::lookup_prefix_for_tokens;
use crate::state::kv_events::{
    CacheNamespace, CheckpointQuery, HashScheme, RadixTreePrefixProvider,
};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::state::load_monitor::ttft_quote::{engine_ttft_quote, TtftQuote};
use crate::workers::Worker;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use sgl_kv_indexer::PrefixOutcome;
use sglang_api_types::runtime::v1 as proto;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const PREFIX_MATCH_PATH: &str = "/internal/prefix_match";
pub const CELL_LOAD_PATH: &str = "/internal/cell_load";
pub const CELL_INFO_PATH: &str = "/internal/cell_info";
pub const CELL_FILES_PATH: &str = "/internal/cell_files/{name}";

/// Body of `POST /internal/prefix_match`. The caller sends the prompt's
/// leading tokens (a few thousand are enough: an owner is decided by the
/// start of the prompt) and the full length separately.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct PrefixMatchRequest {
    /// The model the tokens belong to; must be the one this router serves.
    pub model: Option<String>,
    /// 64-bit, as the engine caches them: a multimodal prompt's pad sentinels
    /// (above 2^32) match the blocks the engine published.
    pub token_ids: Vec<u64>,
    /// The whole prompt's length; defaults to `token_ids.len()`, or to the
    /// full blocks a checkpoint query names.
    pub input_tokens: Option<u64>,
    /// KV namespace components, as the chat request would carry them.
    pub cache_salt: Option<String>,
    pub lora_name: Option<String>,
    /// The caller's remaining budget. A lookup that cannot finish inside it
    /// answers 504 rather than late, so the caller counts a miss and moves on.
    /// Only a remote indexer's lookup waits and can be cut short: one in the
    /// local KV tree, by ids or by checkpoints, never yields and runs to the
    /// end.
    pub deadline_ms: Option<u32>,
    /// The prompt as a sample of its chained block hashes instead of its
    /// token ids: the whole prompt, at a few hundred hashes whatever its
    /// length. When present it decides the depth and `token_ids` is unused.
    /// A cell whose engines hash otherwise cannot use it and answers 409, so
    /// the caller resends the token ids.
    pub checkpoints: Option<CheckpointQuery>,
}

#[derive(Debug, Serialize)]
pub struct PrefixMatchResponse {
    /// Deepest contiguous prefix any engine of this cell holds, in tokens,
    /// capped at the query length. For a checkpoint query, the depth of the
    /// deepest sample held: a lower bound.
    pub matched_prefix_tokens: u64,
    pub matched_prefix_blocks: u32,
    /// For a checkpoint query, the most an engine of this cell holds as far
    /// as its KV tree saw: one block short of the next sample, which none
    /// holds there. A chain stored before this router subscribed can run past
    /// it. Equal to `matched_prefix_tokens` where the samples are dense;
    /// absent for a token query, whose depth is exact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_prefix_tokens_upper: Option<u64>,
    /// The cell's block size; 0 until an engine reports one.
    pub page_size: u32,
    /// The depth came from a pending stake (a prompt just routed here whose KV
    /// events have not arrived), not from a confirmed KV event.
    pub pending: bool,
    pub load: CellLoad,
    /// Whether some engine of this cell would currently admit a request of
    /// `input_tokens`, under the admission limits its routing applies. True
    /// when the routing has no admission to consult.
    pub admits: bool,
    /// Milliseconds until this request's first token were it sent to this
    /// cell now: at the engine this cell's routing would pick (the
    /// expectation over its draw when the routing draws), the prefill queue
    /// ahead of it, then its uncached prefill, at that engine's measured
    /// rate. The cached prefix is what this query showed. No network time or
    /// decode. Absent when the cell cannot stand behind a number: legacy or
    /// session routing, no admitted engine, or an engine it may pick with no
    /// fresh report or no measured prefill rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_ttft_ms: Option<f64>,
    /// The part of `expected_ttft_ms` that is this request's own uncached
    /// prefill, at the same engine and rate: the work the request would cost
    /// the cell, as opposed to its wait in the queue. Present exactly when
    /// `expected_ttft_ms` is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_prefill_ms: Option<f64>,
}

/// The load of this cell's prefill-side engines (plain engines, or the
/// prefill group in P/D), summed from the engines' own load reports.
#[derive(Debug, Default, Serialize, PartialEq)]
pub struct CellLoad {
    /// Sum of waiting uncached tokens over the engines that report them.
    pub waiting_uncached_tokens: u64,
    /// The smallest estimate any engine reports: the queue a new request
    /// would see at the cell's best engine.
    pub estimated_prefill_queue_ms: Option<f64>,
    pub waiting_requests: u64,
    pub running_requests: u64,
    /// The smallest waiting count on any reporting engine; 0 when an engine
    /// is idle, and 0 when nothing reports (see `reporting_engines`).
    pub min_worker_queue: u64,
    /// Age of the newest report; `u32::MAX` when no engine has a fresh one.
    pub snapshot_age_ms: u32,
    /// Engines the summary covers, and how many had a fresh report.
    pub engines: u32,
    pub reporting_engines: u32,
    /// Requests this router has in flight to those engines right now, which
    /// is known even when no engine reports.
    pub router_inflight_requests: u64,
}

/// Why a prefix match produced no answer.
#[derive(Debug)]
pub enum PrefixMatchFailure {
    Api(ApiError),
    /// The lookup did not finish within `deadline_ms`.
    Deadline,
    /// The checkpoints were hashed in a way this cell's engines do not hash
    /// (block size, bigram mode, a LoRA namespace, or a remote indexer that
    /// only takes full chains), or the cell does not know yet how they hash.
    /// The caller resends the token ids.
    CheckpointsUnusable(String),
}

impl From<ApiError> for PrefixMatchFailure {
    fn from(error: ApiError) -> Self {
        PrefixMatchFailure::Api(error)
    }
}

/// `POST /internal/prefix_match`.
pub async fn prefix_match(
    State(ctx): State<Arc<AppContext>>,
    Json(request): Json<PrefixMatchRequest>,
) -> Result<Response, ApiError> {
    match prefix_match_answer(&ctx, request).await {
        Ok(answer) => Ok(Json(answer).into_response()),
        Err(PrefixMatchFailure::Deadline) => Ok((
            StatusCode::GATEWAY_TIMEOUT,
            "prefix lookup did not finish within deadline_ms",
        )
            .into_response()),
        Err(PrefixMatchFailure::CheckpointsUnusable(why)) => {
            Ok((StatusCode::CONFLICT, why).into_response())
        }
        Err(PrefixMatchFailure::Api(error)) => Err(error),
    }
}

/// The prefix-match answer for one query, shared by the HTTP route and the
/// gRPC `PrefixMatch` RPC.
pub async fn prefix_match_answer(
    ctx: &AppContext,
    request: PrefixMatchRequest,
) -> Result<PrefixMatchResponse, PrefixMatchFailure> {
    let model = served_model(ctx, request.model.as_deref())?;
    let namespace = CacheNamespace {
        cache_salt: request.cache_salt.filter(|s| !s.is_empty()),
        lora_name: request.lora_name.filter(|s| !s.is_empty()),
        ambiguous: false,
    };
    // What the query names: the tokens it carries, or the whole prompt its
    // checkpoints sample. Depths are capped at it.
    let query_tokens = match &request.checkpoints {
        Some(query) => u64::from(query.samples.total_blocks) * u64::from(query.scheme.page_size),
        None => request.token_ids.len() as u64,
    };
    let signal = match &request.checkpoints {
        // A lookup by hash takes one shard's read lock at a time and does no
        // hashing. Like the ids lookup in the local tree it never yields, so
        // racing it against the deadline could not cut it short.
        Some(query) => Some(lookup_prefix_for_checkpoints(ctx, query, &namespace)?),
        None => {
            let lookup = lookup_prefix_for_tokens(ctx, &request.token_ids, &namespace, &model.0);
            match request.deadline_ms {
                Some(ms) if ms > 0 => {
                    match tokio::time::timeout(Duration::from_millis(u64::from(ms)), lookup).await {
                        Ok(signal) => signal?,
                        Err(_) => return Err(PrefixMatchFailure::Deadline),
                    }
                }
                _ => lookup.await?,
            }
        }
    };

    let now = Instant::now();
    let engines = prefill_engines(ctx, &model);
    let urls: HashSet<&str> = engines.iter().map(|w| w.url.as_str()).collect();
    let (matched_prefix_blocks, pending) = cell_depth(signal.as_ref(), &urls);
    let page_size = ctx.block_size_oracle.get().unwrap_or(0);
    let to_tokens = |blocks: u32| (u64::from(blocks) * u64::from(page_size)).min(query_tokens);
    let matched_prefix_tokens = to_tokens(matched_prefix_blocks);
    let matched_prefix_tokens_upper = request
        .checkpoints
        .as_ref()
        .map(|query| to_tokens(query.samples.upper_bound(matched_prefix_blocks)));
    let snapshot = ctx.engine_reported_load.capture_snapshot(now);
    let load = summarize_load(&engines, &snapshot, now);
    let input_tokens = request.input_tokens.unwrap_or(query_tokens);
    let admits = admits(ctx, &model, input_tokens, &snapshot);
    let uncached = quoted_uncached_by_url(signal.as_ref(), to_tokens, input_tokens);
    let quote = cell_ttft_quote(ctx, &model, input_tokens, &uncached, &snapshot)
        .inspect_err(|reason| tracing::debug!(reason, "prefix match without a TTFT quote"))
        .ok();
    Ok(PrefixMatchResponse {
        matched_prefix_tokens,
        matched_prefix_blocks,
        matched_prefix_tokens_upper,
        page_size,
        pending,
        load,
        admits,
        expected_ttft_ms: quote.map(|quote| quote.ttft_ms()),
        expected_prefill_ms: quote.map(|quote| quote.prefill_ms),
    })
}

/// The lookup for a checkpoint query, from the in-process KV tree. Refused
/// (the caller falls back to token ids) unless the samples were hashed the
/// way this cell's engines publish: same block size and bigram mode, and no
/// LoRA namespace, whose published keys are per-worker ids the caller cannot
/// compute. A cache salt is fine: it seeds the chain the caller hashed.
fn lookup_prefix_for_checkpoints(
    ctx: &AppContext,
    query: &CheckpointQuery,
    namespace: &CacheNamespace,
) -> Result<PrefixLookupResult, PrefixMatchFailure> {
    let unusable = |why: String| PrefixMatchFailure::CheckpointsUnusable(why);
    query
        .samples
        .validate()
        .map_err(|e| PrefixMatchFailure::Api(ApiError::BadRequest(e.to_string())))?;
    if namespace.lora_name.is_some() {
        return Err(unusable("checkpoints do not cover a LoRA namespace".into()));
    }
    let (provider, scheme) = checkpoint_scheme(ctx).map_err(|why| unusable(why.into()))?;
    if scheme != query.scheme {
        return Err(unusable(format!(
            "this cell hashes {}-token pages (bigram {}), not {} (bigram {})",
            scheme.page_size, scheme.bigram, query.scheme.page_size, query.scheme.bigram
        )));
    }
    Ok(provider.match_checkpoints(&query.samples))
}

/// The hashing this cell answers checkpoint queries in, and the tree it
/// answers them from: its engines' block size and bigram mode, once known,
/// over the in-process KV tree. A cell matching through a remote indexer
/// takes only token queries.
fn checkpoint_scheme(
    ctx: &AppContext,
) -> Result<(&RadixTreePrefixProvider, HashScheme), &'static str> {
    let provider = ctx
        .radix_tree_prefix_provider
        .as_ref()
        .filter(|_| ctx.prefix_index.is_none())
        .ok_or("this cell matches prefixes without a local KV tree")?;
    let (page_size, bigram) = ctx
        .block_size_oracle
        .hash_config()
        .ok_or("this cell does not know its engines' hashing yet")?;
    Ok((provider, HashScheme::new(page_size, bigram)))
}

/// `GET /internal/cell_load`.
pub async fn cell_load(State(ctx): State<Arc<AppContext>>) -> Json<CellLoad> {
    Json(cell_load_answer(&ctx))
}

/// The cell's load now, shared by the HTTP route and the gRPC `GetCellLoad` RPC.
pub fn cell_load_answer(ctx: &AppContext) -> CellLoad {
    let now = Instant::now();
    let model = ModelId(ctx.config.model.id.clone());
    let engines = prefill_engines(ctx, &model);
    let snapshot = ctx.engine_reported_load.capture_snapshot(now);
    summarize_load(&engines, &snapshot, now)
}

/// `GET /internal/cell_info`: the model this cell serves and the facts an
/// upstream needs to render and tokenize for it exactly as the engines do.
#[derive(Debug, Serialize)]
pub struct CellInfo {
    /// This router's `--model-id`: the `model` prefix-match queries must carry.
    pub model: String,
    /// This router's `--tokenizer-path`, when set: a path on this router's
    /// filesystem or an HF repo id. An upstream resolves it locally; the
    /// model id itself is an HF repo id to fall back to.
    pub tokenizer_path: Option<String>,
    /// The cell's block size; 0 until an engine reports one.
    pub page_size: u32,
    /// How the engines hash KV blocks, when this cell answers checkpoint
    /// prefix queries (`PrefixMatchRequest::checkpoints`): an upstream that reads it
    /// may name a prompt by a sample of its block hashes. Absent from a
    /// router that does not take them, or before the engines report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_hash: Option<HashScheme>,
    /// This router's native gRPC port (`--grpc-port`), when it serves one:
    /// `sglang.runtime.v1.SglangService` on the same host as the HTTP port.
    pub grpc_port: Option<u16>,
    /// The engines' OpenAI-layer launch arguments that shape rendering and
    /// parsing, from `/server_info`; `None` when no engine reported them.
    pub chat: Option<CellChatSettings>,
    /// Whether every engine that reported agrees on `chat`. An upstream that
    /// renders for a cell whose engines disagree cannot match all of them.
    pub consistent: bool,
    /// What media this cell takes; `None` when no engine reports multimodal
    /// support, or when any reporting engine lacks it (the router may send a
    /// request to any of them).
    pub mm: Option<CellMm>,
    /// This router's own `--default-chat-template-kwargs`, which it applies
    /// when it renders; the engines' are in `chat`.
    pub router_default_chat_template_kwargs: serde_json::Map<String, serde_json::Value>,
    pub engines: u32,
    pub engines_reporting_chat: u32,
}

/// The cell's media capability: the intersection of its engines' `--mm`
/// launch args. An upstream router sends media only to cells that report
/// the modality, within the smallest `max_items`, and derives cache
/// identities with `hash`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CellMm {
    /// Modalities the engines take. Engines report no per-modality list, so
    /// a multimodal engine is taken to accept images; video and audio are
    /// listed when `--limit-mm-data-per-request` names them.
    pub modalities: Vec<String>,
    /// The smallest per-request item limit any engine sets, per modality.
    pub max_items: std::collections::BTreeMap<String, u32>,
    /// The smallest remote-media byte cap any engine sets.
    pub max_request_bytes: Option<u64>,
    /// The model's family with its processor spec, read from the model files
    /// beside the tokenizer (`config.json`'s `model_type`;
    /// `preprocessor_config.json`'s `media_proc_cfg`, the whole file for a
    /// model without one, or `config.json`'s `vision_config` plus
    /// `image_token_id` for a checkpoint with no preprocessor config, as
    /// DeepSeek V4.1): `[{name, modalities, spec}]`. Empty when the files
    /// are not there; an upstream without a planner for the family matches on
    /// the text prefix only.
    pub families: Vec<serde_json::Value>,
    /// The identity algorithm the engines key their media caches on:
    /// `sha256:<64 hex>` of the raw bytes (`mm_content_hashes`).
    pub hash: &'static str,
    /// Whether the engines adopt caller content hashes without re-hashing.
    pub trust_content_hashes: bool,
    pub engines_multimodal: u32,
    pub engines_reporting_multimodal: u32,
}

impl CellMm {
    /// The intersection over the engines that reported; `None` unless every
    /// reporting engine is multimodal. `tokenizer_path` names the model files
    /// the family spec is read from.
    fn from_engines(engines: &[Arc<Worker>], tokenizer_path: Option<&str>) -> Option<Self> {
        let reported: Vec<&crate::workers::introspect::EngineMm> = engines
            .iter()
            .filter_map(|w| w.mm_settings())
            .map(|m| m.as_ref())
            .collect();
        if reported.is_empty() || reported.iter().any(|m| !m.enabled) {
            return None;
        }
        let mut max_items: std::collections::BTreeMap<String, u32> = Default::default();
        for mm in &reported {
            for (modality, limit) in &mm.max_items {
                max_items
                    .entry(modality.clone())
                    .and_modify(|current| *current = (*current).min(*limit))
                    .or_insert(*limit);
            }
        }
        let mut modalities: Vec<String> = vec!["image".to_owned()];
        for modality in max_items.keys() {
            if !modalities.contains(modality) && max_items[modality] > 0 {
                modalities.push(modality.clone());
            }
        }
        modalities.retain(|m| max_items.get(m).is_none_or(|limit| *limit > 0));
        Some(Self {
            modalities,
            max_items,
            max_request_bytes: reported.iter().filter_map(|m| m.max_media_bytes).min(),
            families: tokenizer_path.map(model_families).unwrap_or_default(),
            hash: "sha256",
            trust_content_hashes: reported.iter().all(|m| m.trust_content_hashes),
            engines_multimodal: reported.iter().filter(|m| m.enabled).count() as u32,
            engines_reporting_multimodal: reported.len() as u32,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CellChatSettings {
    pub served_model_name: Option<String>,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    pub default_chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    pub context_length: Option<u64>,
}

impl CellChatSettings {
    fn from_engine(settings: &sglang_processor::openai::OpenAiSettings) -> Self {
        Self {
            served_model_name: settings.served_model_name.clone(),
            reasoning_parser: settings.reasoning_parser.clone(),
            tool_call_parser: settings.tool_call_parser.clone(),
            default_chat_template_kwargs: settings.default_chat_template_kwargs.clone(),
            context_length: settings.context_length,
        }
    }
}

pub async fn cell_info(State(ctx): State<Arc<AppContext>>) -> Json<CellInfo> {
    Json(cell_info_answer(&ctx))
}

/// The model files a Global Router may fetch from this cell: what it needs
/// to tokenize and render as the engines do, from the directory holding the
/// tokenizer (`--tokenizer-path`, a file or a directory). Nothing else under
/// that directory is served: the weights are not on this list.
/// `tiktoken.model` is the tokenizer of models that ship no `tokenizer.json`
/// (Kimi-K3).
pub const CELL_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "tiktoken.model",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
];

/// `GET /internal/cell_files/{name}`: one of [`CELL_FILES`] beside the
/// tokenizer, for a Global Router whose own host has no copy of the model
/// (it learns the model from `cell_info` and loads the tokenizer from
/// here). A name off the list, no tokenizer path, or a missing file is a 404.
pub async fn cell_file(
    State(ctx): State<Arc<AppContext>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    if !CELL_FILES.contains(&name.as_str()) {
        return (StatusCode::NOT_FOUND, "not a cell file").into_response();
    }
    let Some(tokenizer_path) = ctx.config.model.tokenizer_path.as_deref() else {
        return (StatusCode::NOT_FOUND, "no tokenizer path").into_response();
    };
    let dir = {
        let path = std::path::Path::new(tokenizer_path);
        if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().map(|p| p.to_path_buf()).unwrap_or_default()
        }
    };
    match tokio::fs::read(dir.join(&name)).await {
        Ok(bytes) => {
            let content_type = if name.ends_with(".json") {
                "application/json"
            } else {
                "application/octet-stream"
            };
            ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "no such file beside the tokenizer").into_response(),
    }
}

/// What this cell serves, shared by the HTTP route and the gRPC `GetCellInfo` RPC.
pub fn cell_info_answer(ctx: &AppContext) -> CellInfo {
    let model = ModelId(ctx.config.model.id.clone());
    let engines = prefill_engines(ctx, &model);
    let reported: Vec<CellChatSettings> = engines
        .iter()
        .filter_map(|w| w.openai_settings())
        .map(|s| CellChatSettings::from_engine(s))
        .collect();
    let consistent = reported.windows(2).all(|pair| pair[0] == pair[1]);
    let router_kwargs = ctx
        .config
        .model
        .default_chat_template_kwargs
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    CellInfo {
        model: model.0,
        tokenizer_path: ctx.config.model.tokenizer_path.clone(),
        page_size: ctx.block_size_oracle.get().unwrap_or(0),
        kv_hash: checkpoint_scheme(ctx).ok().map(|(_, scheme)| scheme),
        grpc_port: ctx.config.server.grpc_port,
        engines: engines.len() as u32,
        engines_reporting_chat: reported.len() as u32,
        chat: reported.into_iter().next(),
        consistent,
        mm: CellMm::from_engines(&engines, ctx.config.model.tokenizer_path.as_deref()),
        router_default_chat_template_kwargs: router_kwargs,
    }
}

/// The model's family and processor spec from the files beside the tokenizer,
/// read once per path (an HF repo id downloads the two files on first use, as
/// the tokenizer's own files are). An upstream router builds its planner from
/// `spec`, so it plans the exact geometry the engines apply.
fn model_families(tokenizer_path: &str) -> Vec<serde_json::Value> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, Vec<serde_json::Value>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(families) = cache.lock().unwrap().get(tokenizer_path) {
        return families.clone();
    }
    let families = read_model_families(tokenizer_path);
    cache
        .lock()
        .unwrap()
        .insert(tokenizer_path.to_owned(), families.clone());
    families
}

fn read_model_families(tokenizer_path: &str) -> Vec<serde_json::Value> {
    let files = crate::tokenizer::adapter::ModelFiles::open(tokenizer_path);
    let config = files.json("config.json").ok().flatten();
    let preprocessor = files.json("preprocessor_config.json").ok().flatten();
    let Some(spec) = family_spec(config.as_ref(), preprocessor) else {
        return Vec::new();
    };
    let name = config
        .as_ref()
        .and_then(|c| c.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    vec![serde_json::json!({
        "name": name,
        "modalities": ["image"],
        "spec": spec,
    })]
}

/// The processor spec of the model's image family. A preprocessor config is
/// it: Kimi's processors nest their geometry under `media_proc_cfg`, others
/// spell it at the top level. A checkpoint that ships none may configure its
/// vision tower in `config.json` instead (DeepSeek V4.1: `vision_config` with
/// the geometry, `image_token_id` beside it), which is the spec then.
fn family_spec(
    config: Option<&serde_json::Value>,
    preprocessor: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    if let Some(preprocessor) = preprocessor {
        return Some(
            preprocessor
                .get("media_proc_cfg")
                .cloned()
                .unwrap_or(preprocessor),
        );
    }
    let config = config?;
    let mut vision = config.get("vision_config")?.as_object()?.clone();
    if let Some(id) = config.get("image_token_id") {
        vision.insert("image_token_id".to_owned(), id.clone());
    }
    Some(serde_json::Value::Object(vision))
}

// --- The same signals as `sglang.runtime.v1` messages, for the gRPC ingress. ---

impl From<proto::PrefixMatchRequest> for PrefixMatchRequest {
    fn from(request: proto::PrefixMatchRequest) -> Self {
        Self {
            model: request.model,
            token_ids: request.token_ids,
            input_tokens: request.input_tokens,
            cache_salt: request.cache_salt,
            lora_name: request.lora_name,
            deadline_ms: request.deadline_ms,
            checkpoints: None,
        }
    }
}

impl From<CellLoad> for proto::CellLoad {
    fn from(load: CellLoad) -> Self {
        Self {
            waiting_uncached_tokens: load.waiting_uncached_tokens,
            estimated_prefill_queue_ms: load.estimated_prefill_queue_ms,
            waiting_requests: load.waiting_requests,
            running_requests: load.running_requests,
            min_worker_queue: load.min_worker_queue,
            snapshot_age_ms: load.snapshot_age_ms,
            engines: load.engines,
            reporting_engines: load.reporting_engines,
            router_inflight_requests: load.router_inflight_requests,
        }
    }
}

impl From<PrefixMatchResponse> for proto::PrefixMatchResponse {
    fn from(answer: PrefixMatchResponse) -> Self {
        Self {
            matched_prefix_tokens: answer.matched_prefix_tokens,
            matched_prefix_blocks: answer.matched_prefix_blocks,
            page_size: answer.page_size,
            pending: answer.pending,
            load: Some(answer.load.into()),
            admits: answer.admits,
            expected_ttft_ms: answer.expected_ttft_ms,
            expected_prefill_ms: answer.expected_prefill_ms,
        }
    }
}

impl From<CellInfo> for proto::CellInfo {
    fn from(info: CellInfo) -> Self {
        // The nested objects travel as the JSON the HTTP route serves, so both
        // forms read the same.
        Self {
            model: info.model,
            tokenizer_path: info.tokenizer_path,
            page_size: info.page_size,
            grpc_port: info.grpc_port.map(u32::from),
            chat_json: info
                .chat
                .as_ref()
                .and_then(|chat| serde_json::to_string(chat).ok())
                .unwrap_or_default(),
            consistent: info.consistent,
            router_default_chat_template_kwargs_json: serde_json::to_string(
                &info.router_default_chat_template_kwargs,
            )
            .unwrap_or_default(),
            engines: info.engines,
            engines_reporting_chat: info.engines_reporting_chat,
            mm_json: info
                .mm
                .as_ref()
                .and_then(|mm| serde_json::to_string(mm).ok())
                .unwrap_or_default(),
        }
    }
}

/// The one model this router serves, or 404 when the caller names another.
fn served_model(ctx: &AppContext, requested: Option<&str>) -> Result<ModelId, ApiError> {
    let served = &ctx.config.model.id;
    match requested {
        Some(name) if !name.is_empty() && name != served => {
            Err(ApiError::ModelNotFound(name.to_owned()))
        }
        _ => Ok(ModelId(served.clone())),
    }
}

/// The engines a prompt would be prefilled on: plain engines and prefill
/// engines, healthy and routable right now.
fn prefill_engines(ctx: &AppContext, model: &ModelId) -> Vec<Arc<Worker>> {
    ctx.registry
        .healthy_workers_for(model)
        .into_iter()
        .filter(|w| matches!(w.mode(), WorkerMode::Plain | WorkerMode::Prefill))
        .collect()
}

/// The deepest match held by one of `urls`, in blocks, and whether that
/// depth is a pending stake rather than a confirmed placement.
fn cell_depth(signal: Option<&PrefixLookupResult>, urls: &HashSet<&str>) -> (u32, bool) {
    let Some(signal) = signal else {
        return (0, false);
    };
    let PrefixOutcome::Matched { matches, .. } = &signal.outcome else {
        return (0, false);
    };
    let best = matches
        .iter()
        .filter(|m| urls.contains(m.address.as_str()))
        .max_by_key(|m| m.matched_prefix_blocks);
    match best {
        Some(best) => (
            best.matched_prefix_blocks,
            signal.pending_winners.contains(&best.address),
        ),
        None => (0, false),
    }
}

/// Sums the engines' fresh load reports. Running and waiting counts come from
/// the basic report, uncached tokens and the queue estimate from the native
/// cache-aware report, each engine counted once.
fn summarize_load(
    engines: &[Arc<Worker>],
    snapshot: &EngineReportedLoadSnapshot,
    now: Instant,
) -> CellLoad {
    let mut load = CellLoad {
        engines: engines.len() as u32,
        min_worker_queue: u64::MAX,
        ..CellLoad::default()
    };
    let mut newest: Option<Instant> = None;
    for engine in engines {
        load.router_inflight_requests += engine.router_inflight_load() as u64;
        let basic = snapshot.fresh_load_for_url(&engine.url);
        let native = snapshot.fresh_native_cache_load_for_url(&engine.url);
        if basic.is_none() && native.is_none() {
            continue;
        }
        load.reporting_engines += 1;
        let (running, waiting, captured) = match (basic, native) {
            (Some(b), _) => (b.num_running_reqs, b.num_waiting_reqs, b.captured_at),
            (None, Some(n)) => (n.num_running_reqs, n.num_waiting_reqs, n.captured_at),
            (None, None) => unreachable!("one report is present"),
        };
        load.running_requests += running;
        load.waiting_requests += waiting;
        load.min_worker_queue = load.min_worker_queue.min(waiting);
        newest = Some(newest.map_or(captured, |t: Instant| t.max(captured)));
        if let Some(n) = native {
            load.waiting_uncached_tokens += n.num_waiting_uncached_tokens;
            if let Some(ms) = n.estimated_prefill_queue_ms {
                load.estimated_prefill_queue_ms = Some(
                    load.estimated_prefill_queue_ms
                        .map_or(ms, |best| best.min(ms)),
                );
            }
            newest = Some(newest.map_or(n.captured_at, |t| t.max(n.captured_at)));
        }
    }
    if load.min_worker_queue == u64::MAX {
        load.min_worker_queue = 0;
    }
    load.snapshot_age_ms = newest
        .map(|t| u32::try_from(now.saturating_duration_since(t).as_millis()).unwrap_or(u32::MAX))
        .unwrap_or(u32::MAX);
    load
}

/// Whether some engine would admit a prompt of `input_tokens` now. Under
/// reorg routing this walks the buckets that fit the length and applies each
/// prefill-side group's admission to its members; a policy without admission
/// admits. Legacy routing has no admission, so it admits.
fn admits(
    ctx: &AppContext,
    model: &ModelId,
    input_tokens: u64,
    snapshot: &EngineReportedLoadSnapshot,
) -> bool {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        return true;
    };
    let Some(resolver) = resolvers.get(model) else {
        return true;
    };
    let Ok(buckets) = resolver.resolve(input_tokens, None, None, None) else {
        return true;
    };
    if buckets.is_empty() {
        // No bucket takes a prompt this long: the router would answer 400.
        return false;
    }
    for bucket in buckets {
        let (group, stage) = match &bucket.groups {
            BucketGroups::Plain(group) => (group, WorkerMode::Plain),
            BucketGroups::Pd { prefill, .. } => (prefill, WorkerMode::Prefill),
        };
        let members = group.members(&ctx.registry, model, stage);
        if members.is_empty() {
            continue;
        }
        let Some(admission) = group.policy.admission() else {
            return true;
        };
        match partition_admitted(admission, &members, snapshot) {
            Ok((admitted, _)) if !admitted.is_empty() => return true,
            Ok(_) => continue,
            // An admission that cannot be evaluated does not veto the cell.
            Err(_) => return true,
        }
    }
    false
}

/// Each engine's uncached share of a prompt of `input_tokens`, by URL, for
/// the engines holding part of it. The held part is what the query showed
/// (`to_tokens` caps blocks at it), never extrapolated: a query of the first
/// few thousand tokens that an engine holds says nothing of the rest.
fn quoted_uncached_by_url(
    signal: Option<&PrefixLookupResult>,
    to_tokens: impl Fn(u32) -> u64,
    input_tokens: u64,
) -> HashMap<&str, u64> {
    let mut uncached = HashMap::new();
    let Some(PrefixLookupResult {
        outcome: PrefixOutcome::Matched { matches, .. },
        ..
    }) = signal
    else {
        return uncached;
    };
    for entry in matches {
        let held = to_tokens(entry.matched_prefix_blocks).min(input_tokens);
        if held > 0 {
            let left = uncached
                .entry(entry.address.as_str())
                .or_insert(input_tokens);
            *left = (*left).min(input_tokens - held);
        }
    }
    uncached
}

/// The time to first token this cell quotes for a prompt of `input_tokens`,
/// in its two parts: each engine's [`engine_ttft_quote`], weighted by the
/// chance its routing picks that engine, in the first bucket that would take
/// the request. An error says why the cell gives none.
fn cell_ttft_quote(
    ctx: &AppContext,
    model: &ModelId,
    input_tokens: u64,
    uncached: &HashMap<&str, u64>,
    snapshot: &EngineReportedLoadSnapshot,
) -> Result<TtftQuote, &'static str> {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        return Err("legacy routing picks no engine to quote");
    };
    let resolver = resolvers.get(model).ok_or("no routing for the model")?;
    let buckets = resolver
        .resolve(input_tokens, None, None, None)
        .map_err(|_| "no bucket takes the request")?;
    for bucket in buckets {
        let chances = match bucket.prefill_distribution(
            &ctx.registry,
            model,
            input_tokens,
            uncached,
            snapshot,
        ) {
            Ok(PickDistribution::Engines(chances)) => chances,
            Ok(PickDistribution::Unsupported) => {
                return Err("the routing's pick depends on more than the prompt")
            }
            // The routing would move to the next bucket.
            Err((
                _,
                PickError::NoCandidates
                | PickError::NoAdmissibleEngine(_)
                | PickError::AdmissionRejected(_),
            )) => continue,
            Err(_) => return Err("the routing cannot pick an engine"),
        };
        let mut quote = TtftQuote::default();
        for (engine, chance) in chances {
            let left = uncached
                .get(engine.url.as_str())
                .copied()
                .unwrap_or(input_tokens);
            let at_engine = engine_ttft_quote(&engine, left, snapshot)?;
            quote.queue_ms += chance * at_engine.queue_ms;
            quote.prefill_ms += chance * at_engine.prefill_ms;
        }
        let finite = |ms: f64| ms.is_finite() && ms >= 0.0;
        return Some(quote)
            .filter(|quote| finite(quote.queue_ms) && finite(quote.prefill_ms))
            .ok_or("the quote is not a finite duration");
    }
    Err("no engine would admit the request")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
    use crate::state::kv_events::prefix_provider::RadixTreePrefixProvider;
    use crate::state::kv_events::tree::{HashTree, KvWorkerId};
    use crate::state::kv_events::wire::BlockStored;
    use crate::state::kv_events::BlockCheckpoints;
    use crate::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    const BLOCK: u32 = 4;

    fn add_worker(ctx: &AppContext, id: &str, url: &str, mode: WorkerMode) {
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: url.into(),
                mode,
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                // A prefill engine is routable only once its bootstrap port is known.
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
                ..Default::default()
            })
            .expect("worker accepted");
    }

    fn load_stat(running: u64, waiting: u64, uncached: Option<u64>) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: 100,
            max_total_num_tokens: 1000,
            native_cache: uncached.map(|tokens| NativeCacheRankLoad {
                num_waiting_uncached_tokens: tokens,
                num_total_tokens: 100,
                max_running_requests: 64,
                total_prefill_uncached_tokens: 0,
                total_prefill_busy_us: 0,
            }),
        }
    }

    /// A context with a local KV tree where `url_a` holds the first `blocks`
    /// blocks of `prompt`.
    fn ctx_with_tree(prompt: &[u64], blocks: usize) -> Arc<AppContext> {
        ctx_and_tree(prompt, blocks).0
    }

    /// [`ctx_with_tree`], and the tree, for a test that publishes more events.
    fn ctx_and_tree(prompt: &[u64], blocks: usize) -> (Arc<AppContext>, Arc<HashTree>) {
        let mut ctx = AppContext::stub();
        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK).expect("block size");
        let tree = Arc::new(HashTree::new());
        let provider = RadixTreePrefixProvider::new(Arc::clone(&tree), Arc::clone(&oracle));
        let hashes = oracle
            .block_hashes(prompt, &CacheNamespace::default())
            .expect("hashes");
        tree.store_event(
            &KvWorkerId::new("http://a:30000".into(), 0),
            &BlockStored {
                block_hashes: hashes[..blocks].to_vec(),
                parent_block_hash: None,
                token_ids: prompt[..blocks * BLOCK as usize].to_vec(),
                block_size: BLOCK,
                lora_id: None,
                lora_name: None,
                cache_salt: None,
                is_bigram: false,
                medium: None,
            },
        );
        ctx.block_size_oracle = oracle;
        ctx.radix_tree_prefix_provider = Some(provider);
        add_worker(&ctx, "a", "http://a:30000", WorkerMode::Plain);
        add_worker(&ctx, "b", "http://b:30000", WorkerMode::Plain);
        (Arc::new(ctx), tree)
    }

    async fn post_json(ctx: Arc<AppContext>, path: &str, body: Value) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn get_json(ctx: Arc<AppContext>, path: &str) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    /// The checkpoint query an upstream would send for `prompt`: a sample
    /// every `stride` blocks and the last full block.
    fn checkpoint_query(prompt: &[u64], stride: usize) -> Value {
        checkpoint_query_hashed(prompt, stride, false)
    }

    /// [`checkpoint_query`], hashed over bigrams or not. A bigram page is
    /// `BLOCK` bigrams, of which a prompt has one fewer than tokens.
    fn checkpoint_query_hashed(prompt: &[u64], stride: usize, bigram: bool) -> Value {
        let hashes = CacheNamespace::default().block_hashes(prompt, BLOCK as usize, bigram);
        let full = (prompt.len() - usize::from(bigram)) / BLOCK as usize;
        let samples = BlockCheckpoints::grid(&hashes, full, stride);
        json!({"page_size": BLOCK, "bigram": bigram, "blocks": samples.blocks,
               "hashes": samples.hashes, "total_blocks": samples.total_blocks})
    }

    #[tokio::test]
    async fn a_checkpoint_query_sees_the_whole_prompt_past_any_token_cap() {
        // 400 tokens = 100 blocks; engine a holds the first 70.
        let prompt: Vec<u64> = (1..=400).collect();
        let ctx = ctx_with_tree(&prompt, 70);
        ctx.block_size_oracle.set_bigram(false);
        let model = ctx.config.model.id.clone();
        // A token query capped at 16 tokens learns only that the start is held.
        let (_, capped) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": model, "token_ids": prompt[..16], "input_tokens": 400}),
        )
        .await;
        assert_eq!(capped["matched_prefix_tokens"], 16);
        // A sample every 12 blocks: 11, 23, …, 59, 71, …, 95, and 99.
        let (status, body) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": checkpoint_query(&prompt, 12)}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // Block 59 is the deepest held, 71 the next.
        assert_eq!(body["matched_prefix_blocks"], 60);
        assert_eq!(body["matched_prefix_tokens"], 240);
        assert_eq!(body["matched_prefix_tokens_upper"], 284);
        let truth = 70 * BLOCK as u64;
        assert!(body["matched_prefix_tokens"].as_u64().unwrap() <= truth);
        assert!(body["matched_prefix_tokens_upper"].as_u64().unwrap() >= truth);
    }

    #[tokio::test]
    async fn a_checkpoint_query_finds_a_chain_whose_start_the_tree_missed() {
        // The engine stored blocks 0..30 before this router subscribed, then
        // blocks 30..60 as a continuation of block 29: the tree holds those as
        // an orphan, which a walk from the root never reaches.
        let prompt: Vec<u64> = (1..=240).collect();
        let (ctx, tree) = ctx_and_tree(&prompt, 0);
        ctx.block_size_oracle.set_bigram(false);
        let hashes = ctx
            .block_size_oracle
            .block_hashes(&prompt, &CacheNamespace::default())
            .unwrap();
        tree.store_event(
            &KvWorkerId::new("http://a:30000".into(), 0),
            &BlockStored {
                block_hashes: hashes[30..60].to_vec(),
                parent_block_hash: Some(hashes[29]),
                token_ids: prompt[120..240].to_vec(),
                block_size: BLOCK,
                lora_id: None,
                lora_name: None,
                cache_salt: None,
                is_bigram: false,
                medium: None,
            },
        );
        let model = ctx.config.model.id.clone();
        let (_, walked) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": model, "token_ids": prompt}),
        )
        .await;
        assert_eq!(
            walked["matched_prefix_tokens"], 0,
            "the walk stops at the missing block 0"
        );
        let (status, body) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": checkpoint_query(&prompt, 16)}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["matched_prefix_tokens"], 240,
            "the last block names the whole prompt"
        );
    }

    #[tokio::test]
    async fn a_bigram_cell_answers_bigram_checkpoints_and_refuses_unigram_ones() {
        // 41 tokens: 40 bigrams, 10 full pages; engine a holds 7 of them.
        let prompt: Vec<u64> = (1..=41).collect();
        let (ctx, tree) = ctx_and_tree(&prompt, 0);
        ctx.block_size_oracle.set_bigram(true);
        let hashes = ctx
            .block_size_oracle
            .block_hashes(&prompt, &CacheNamespace::default())
            .unwrap();
        assert_eq!(hashes.len(), 10);
        tree.insert(
            &KvWorkerId::new("http://a:30000".into(), 0),
            None,
            &hashes[..7],
        );
        let model = ctx.config.model.id.clone();
        let (status, body) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": checkpoint_query_hashed(&prompt, 2, true)}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // Samples at blocks 1, 3, 5, 7 and 9: block 5 is the deepest held.
        assert_eq!(body["matched_prefix_blocks"], 6);
        assert_eq!(body["matched_prefix_tokens"], 6 * BLOCK);
        assert_eq!(body["matched_prefix_tokens_upper"], 7 * BLOCK);
        let (status, _) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": checkpoint_query(&prompt, 2)}),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CONFLICT,
            "unigram samples of a bigram fleet"
        );
    }

    #[tokio::test]
    async fn a_checkpoint_query_hashed_another_way_is_refused_with_409() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 6);
        let model = ctx.config.model.id.clone();
        // The cell does not know its engines' bigram mode yet.
        let (status, _) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": checkpoint_query(&prompt, 4)}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        ctx.block_size_oracle.set_bigram(false);
        let mut query = checkpoint_query(&prompt, 4);
        query["page_size"] = json!(BLOCK * 2);
        let (status, _) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": query}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        let mut query = checkpoint_query(&prompt, 4);
        query["hashes"].as_array_mut().unwrap().pop();
        let (status, _) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"model": model, "checkpoints": query}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn prefix_match_reports_the_cell_depth_in_tokens_and_blocks() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 6);
        let model = ctx.config.model.id.clone();
        let (status, body) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"model": model, "token_ids": prompt, "input_tokens": 5000, "deadline_ms": 50}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["matched_prefix_blocks"], 6);
        assert_eq!(body["matched_prefix_tokens"], 24);
        assert_eq!(body["page_size"], BLOCK);
        assert_eq!(body["pending"], false);
        assert_eq!(body["admits"], true, "legacy routing has no admission");
        assert_eq!(body["load"]["engines"], 2);
        assert_eq!(body["load"]["reporting_engines"], 0);
        assert_eq!(body["load"]["snapshot_age_ms"], u32::MAX);
    }

    #[tokio::test]
    async fn prefix_match_is_capped_at_the_query_and_ignores_unknown_engines() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 10);
        // Only the first two blocks are sent: the answer cannot exceed them.
        let (status, body) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"token_ids": prompt[..8].to_vec()}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["matched_prefix_blocks"], 2);
        assert_eq!(body["matched_prefix_tokens"], 8);
        // The holder leaves the registry: its depth no longer counts for the cell.
        ctx.registry.remove(&WorkerId("a".into()));
        let (_, body) = post_json(ctx, PREFIX_MATCH_PATH, json!({"token_ids": prompt})).await;
        assert_eq!(body["matched_prefix_blocks"], 0);
        assert_eq!(body["matched_prefix_tokens"], 0);
    }

    #[tokio::test]
    async fn prefix_match_rejects_another_model_and_a_different_namespace_misses() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 6);
        let (status, _) = post_json(
            Arc::clone(&ctx),
            PREFIX_MATCH_PATH,
            json!({"model": "someone-else", "token_ids": prompt}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, body) = post_json(
            ctx,
            PREFIX_MATCH_PATH,
            json!({"token_ids": prompt, "cache_salt": "tenant-b"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["matched_prefix_blocks"], 0);
    }

    /// A Global Router with no copy of the model fetches the files beside the
    /// tokenizer from the cell: the listed names only, nothing else under
    /// that directory, and nothing without a tokenizer path.
    #[tokio::test]
    async fn cell_files_serves_the_model_files_beside_the_tokenizer() {
        let dir =
            std::env::temp_dir().join(format!("sgl-router-cell-files-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"version": "1.0"}"#).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type": "test"}"#).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path =
            Some(dir.join("tokenizer.json").to_string_lossy().into_owned());
        let ctx = Arc::new(ctx);
        let (status, body) = get_json(ctx.clone(), "/internal/cell_files/tokenizer.json").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["version"], "1.0");
        let (status, body) = get_json(ctx.clone(), "/internal/cell_files/config.json").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model_type"], "test");
        // Listed but absent, and present but not listed: both 404.
        let (status, _) =
            get_json(ctx.clone(), "/internal/cell_files/generation_config.json").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = get_json(ctx.clone(), "/internal/cell_files/model.safetensors").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        // A directory as the tokenizer path serves the same files.
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path = Some(dir.to_string_lossy().into_owned());
        let (status, _) = get_json(Arc::new(ctx), "/internal/cell_files/tokenizer.json").await;
        assert_eq!(status, StatusCode::OK);
        // No tokenizer path: nothing to serve.
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path = None;
        let (status, _) = get_json(Arc::new(ctx), "/internal/cell_files/tokenizer.json").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A model whose tokenizer is a tiktoken vocabulary (Kimi-K3: no
    /// `tokenizer.json`) is loadable from the cell: `tiktoken.model` is
    /// served byte for byte.
    #[tokio::test]
    async fn cell_files_serves_a_tiktoken_vocabulary() {
        let dir =
            std::env::temp_dir().join(format!("sgl-router-cell-tiktoken-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab: &[u8] = b"IQ== 0\nIg== 1\nIw== 2\n";
        std::fs::write(dir.join("tiktoken.model"), vocab).unwrap();
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path = Some(dir.to_string_lossy().into_owned());
        let app = crate::server::app::build_router(Arc::new(ctx));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/internal/cell_files/tiktoken.model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()[axum::http::header::CONTENT_TYPE],
            "application/octet-stream"
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], vocab);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn cell_load_sums_fresh_reports_over_prefill_side_engines() {
        let ctx = AppContext::stub();
        add_worker(&ctx, "a", "http://a:30000", WorkerMode::Prefill);
        add_worker(&ctx, "b", "http://b:30000", WorkerMode::Prefill);
        add_worker(&ctx, "d", "http://d:30000", WorkerMode::Decode);
        let now = Instant::now();
        ctx.engine_reported_load
            .set("http://a:30000", 0, load_stat(3, 2, Some(4000)), now);
        ctx.engine_reported_load
            .set("http://b:30000", 0, load_stat(1, 0, Some(500)), now);
        // Decode load is not prefill-side load.
        ctx.engine_reported_load
            .set("http://d:30000", 0, load_stat(50, 9, Some(99_999)), now);
        let (status, body) = get_json(Arc::new(ctx), CELL_LOAD_PATH).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["engines"], 2);
        assert_eq!(body["reporting_engines"], 2);
        assert_eq!(body["running_requests"], 4);
        assert_eq!(body["waiting_requests"], 2);
        assert_eq!(body["min_worker_queue"], 0);
        assert_eq!(body["waiting_uncached_tokens"], 4500);
        assert!(body["snapshot_age_ms"].as_u64().unwrap() < 5_000, "{body}");
    }

    #[tokio::test]
    async fn cell_info_names_the_model_and_the_block_size() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 6);
        let model = ctx.config.model.id.clone();
        let (status, body) = get_json(ctx, CELL_INFO_PATH).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], model);
        assert_eq!(body["page_size"], BLOCK);
        assert!(body["grpc_port"].is_null(), "the stub serves no gRPC port");
        assert!(
            body["mm"].is_null(),
            "no engine reported multimodal support"
        );
        assert_eq!(body["engines"], 2);
        assert_eq!(body["engines_reporting_chat"], 0);
        assert!(body["chat"].is_null());
        assert_eq!(body["consistent"], true);
        assert!(
            body.get("kv_hash").is_none(),
            "no checkpoint queries before the engines' bigram mode is known"
        );
    }

    #[tokio::test]
    async fn cell_info_advertises_checkpoint_queries_once_the_hashing_is_known() {
        let ctx = ctx_with_tree(&(1..=40).collect::<Vec<u64>>(), 6);
        ctx.block_size_oracle.set_bigram(false);
        let (_, body) = get_json(ctx, CELL_INFO_PATH).await;
        assert_eq!(
            body["kv_hash"],
            json!({"page_size": BLOCK, "bigram": false})
        );
    }

    #[tokio::test]
    async fn cell_signals_convert_to_their_grpc_messages() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = ctx_with_tree(&prompt, 6);
        let info: proto::CellInfo = cell_info_answer(&ctx).into();
        assert_eq!(info.model, ctx.config.model.id);
        assert_eq!(info.engines, 2);
        assert_eq!(info.page_size, BLOCK);
        assert!(
            info.chat_json.is_empty(),
            "no engine reported chat settings"
        );
        assert!(
            info.mm_json.is_empty(),
            "no engine reported multimodal support"
        );
        let kwargs: Value =
            serde_json::from_str(&info.router_default_chat_template_kwargs_json).unwrap();
        assert!(kwargs.is_object());

        let load: proto::CellLoad = cell_load_answer(&ctx).into();
        assert_eq!(load.engines, 2);
        assert_eq!(load.reporting_engines, 0);
        assert_eq!(load.snapshot_age_ms, u32::MAX);

        let request = proto::PrefixMatchRequest {
            model: Some(ctx.config.model.id.clone()),
            token_ids: prompt.clone(),
            input_tokens: Some(40),
            cache_salt: None,
            lora_name: None,
            deadline_ms: Some(1000),
        };
        let answer: proto::PrefixMatchResponse = prefix_match_answer(&ctx, request.into())
            .await
            .expect("the cell answers")
            .into();
        assert_eq!(answer.matched_prefix_blocks, 6);
        assert_eq!(answer.matched_prefix_tokens, 24);
        assert_eq!(answer.page_size, BLOCK);
        assert!(answer.admits);
        assert_eq!(answer.load.expect("load rides along").engines, 2);
        let other = proto::PrefixMatchRequest {
            model: Some("someone-else".into()),
            ..Default::default()
        };
        assert!(matches!(
            prefix_match_answer(&ctx, other.into()).await,
            Err(PrefixMatchFailure::Api(_))
        ));
    }

    #[test]
    fn multimodal_cells_report_the_models_processor_spec() {
        let dir = std::env::temp_dir().join(format!("sgl-router-cell-mm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "kimi_k3", "media_placeholder_token_id": 163605}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("preprocessor_config.json"),
            r#"{"media_proc_cfg": {"patch_size": 14, "merge_kernel_size": 2, "in_patch_limit": 65536,
                "patch_limit_on_one_side": 512, "image_mean": [0.5, 0.5, 0.5]}, "auto_map": {}}"#,
        )
        .unwrap();
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path =
            Some(dir.join("tokenizer.json").to_string_lossy().into_owned());
        let profile = crate::workers::EngineProfile {
            answered: true,
            protocol: crate::workers::WireProtocol::default(),
            dp_ranks: 1,
            openai: None,
            grpc_port: None,
            mm: Some(Arc::new(crate::workers::introspect::EngineMm {
                enabled: true,
                max_items: [("image".to_owned(), 4)].into_iter().collect(),
                max_media_bytes: Some(64 << 20),
                trust_content_hashes: false,
            })),
        };
        ctx.registry
            .add_with_cb(
                WorkerSpec {
                    id: WorkerId("mm".into()),
                    url: "http://mm:30000".into(),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId(ctx.config.model.id.clone())],
                    ..Default::default()
                },
                None,
                profile,
            )
            .expect("worker accepted");
        let info = cell_info_answer(&ctx);
        let mm = info
            .mm
            .expect("a multimodal engine makes the cell multimodal");
        assert_eq!(mm.modalities, vec!["image"]);
        assert_eq!(mm.max_items.get("image"), Some(&4));
        assert_eq!(mm.families.len(), 1);
        assert_eq!(mm.families[0]["name"], "kimi_k3");
        assert_eq!(mm.families[0]["spec"]["patch_size"], 14);
        assert_eq!(mm.families[0]["spec"]["patch_limit_on_one_side"], 512);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// DeepSeek V4.1 ships no preprocessor config: its family spec is the
    /// vision tower's geometry from `config.json`, with the image token id.
    #[test]
    fn a_vision_config_without_a_preprocessor_config_is_the_family_spec() {
        let config = serde_json::json!({
            "model_type": "deepseek_v41", "image_token_id": 129264,
            "vision_config": {"model_type": "deepseek_v41_vision", "patch_size": 14, "downsample_ratio": 3,
                "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null}});
        let spec = family_spec(Some(&config), None).expect("a vision config is a spec");
        assert_eq!(spec["patch_size"], 14);
        assert_eq!(spec["downsample_ratio"], 3);
        assert_eq!(spec["image_token_id"], 129264);
        // A preprocessor config wins when both are there; a text checkpoint has neither.
        let pre = serde_json::json!({"media_proc_cfg": {"patch_size": 16}});
        assert_eq!(
            family_spec(Some(&config), Some(pre)).unwrap()["patch_size"],
            16
        );
        assert!(family_spec(Some(&serde_json::json!({"model_type": "llama"})), None).is_none());
        assert!(family_spec(None, None).is_none());
    }

    /// [`ctx_with_tree`] under reorg routing: one plain bucket of every
    /// engine, picked by `policy`.
    fn reorg_ctx_with_tree(
        prompt: &[u64],
        blocks: usize,
        policy: impl FnOnce(&AppContext) -> Arc<dyn crate::policies_reorg::Policy>,
    ) -> Arc<AppContext> {
        use crate::buckets_reorg::{Bucket, BucketResolver, EngineGroup};
        let (ctx, _) = ctx_and_tree(prompt, blocks);
        let mut ctx = Arc::into_inner(ctx).expect("not shared yet");
        let bucket = Bucket::new("all", BucketGroups::Plain(EngineGroup::new(policy(&ctx))));
        let resolver = BucketResolver::new(vec![bucket]).expect("a valid bucket");
        let model = ModelId(ctx.config.model.id.clone());
        ctx.chat_routing = ChatRouting::Reorg([(model, resolver)].into());
        Arc::new(ctx)
    }

    fn power_of_two(ctx: &AppContext) -> Arc<dyn crate::policies_reorg::Policy> {
        Arc::new(crate::policies_reorg::power_of_two::PowerOfTwoPolicy::new(
            Arc::clone(&ctx.engine_reported_load),
        ))
    }

    /// Two native frames from `url`: 10k tokens prefilled in 1 s of busy
    /// time, and `waiting` uncached tokens queued now.
    fn report_prefill(ctx: &AppContext, url: &str, waiting: u64) {
        let frame = |waiting: u64, prefilled: u64, busy_us: u64| {
            let mut stat = load_stat(0, u64::from(waiting > 0), Some(waiting));
            let native = stat.native_cache.as_mut().unwrap();
            native.total_prefill_uncached_tokens = prefilled;
            native.total_prefill_busy_us = busy_us;
            stat
        };
        let now = Instant::now();
        ctx.engine_reported_load.set(url, 0, frame(0, 0, 0), now);
        ctx.engine_reported_load
            .set(url, 0, frame(waiting, 10_000, 1_000_000), now);
    }

    fn quote(body: &Value) -> f64 {
        body["expected_ttft_ms"]
            .as_f64()
            .unwrap_or_else(|| panic!("no quote in {body}"))
    }

    fn prefill(body: &Value) -> f64 {
        body["expected_prefill_ms"]
            .as_f64()
            .unwrap_or_else(|| panic!("no prefill quote in {body}"))
    }

    /// The prefix owner's queue, then the part of the prompt it lacks, at its
    /// rate: a holds 24 of 5000 tokens behind 20k waiting, at 10k tokens/s.
    /// The prefill alone is that part: 4976 tokens.
    #[tokio::test]
    async fn prefix_match_quotes_the_ttft_at_the_engine_the_cell_would_pick() {
        use crate::config::{AffinityConfig, AffinityMode};
        use crate::policies_reorg::cache_aware::{CacheAwarePolicy, CacheSource};
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = reorg_ctx_with_tree(&prompt, 6, |ctx| {
            let provider = RadixTreePrefixProvider::new(
                Arc::new(HashTree::new()),
                Arc::clone(&ctx.block_size_oracle),
            );
            let config = AffinityConfig {
                mode: AffinityMode::Prefer,
                cache_affinity_min_matched_tokens: None,
                ..Default::default()
            };
            let policy = CacheAwarePolicy::new(
                Arc::new(CacheSource::Local(provider)),
                Arc::clone(&ctx.engine_reported_load),
                config,
            );
            Arc::new(policy.expect("valid settings"))
        });
        report_prefill(&ctx, "http://a:30000", 20_000);
        report_prefill(&ctx, "http://b:30000", 0);
        let query = json!({"token_ids": prompt, "input_tokens": 5000});
        let (status, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["matched_prefix_tokens"], 24);
        assert!((quote(&body) - 2_497.6).abs() < 1e-6, "{body}");
        assert!((prefill(&body) - 497.6).abs() < 1e-6, "{body}");

        let request = proto::PrefixMatchRequest {
            token_ids: prompt,
            input_tokens: Some(5000),
            ..Default::default()
        };
        let answer: proto::PrefixMatchResponse = prefix_match_answer(&ctx, request.into())
            .await
            .expect("the cell answers")
            .into();
        let grpc_quote = answer.expected_ttft_ms.expect("the quote rides along");
        assert!((grpc_quote - 2_497.6).abs() < 1e-6);
        let grpc_prefill = answer.expected_prefill_ms.expect("the prefill rides along");
        assert!((grpc_prefill - 497.6).abs() < 1e-6);
    }

    /// Power of two over a (20k waiting) and b (idle) always lands on b;
    /// work dispatched to b since its report joins its queue until it ends.
    /// It lengthens the wait, not the prompt's own prefill.
    #[tokio::test]
    async fn prefix_match_counts_work_dispatched_since_the_report() {
        let prompt: Vec<u64> = (1..=40).collect();
        let ctx = reorg_ctx_with_tree(&prompt, 6, power_of_two);
        report_prefill(&ctx, "http://a:30000", 20_000);
        report_prefill(&ctx, "http://b:30000", 0);
        let query = json!({"token_ids": prompt, "input_tokens": 5000});
        let (_, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query.clone()).await;
        assert!((quote(&body) - 500.0).abs() < 1e-6, "{body}");
        assert!((prefill(&body) - 500.0).abs() < 1e-6, "{body}");

        let b = ctx.registry.get(&WorkerId("b".into())).unwrap();
        let dispatched = b.prefill_load_guard(2_000);
        let (_, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query.clone()).await;
        assert!((quote(&body) - 700.0).abs() < 1e-6, "{body}");
        assert!((prefill(&body) - 500.0).abs() < 1e-6, "{body}");
        drop(dispatched);
        let (_, body) = post_json(ctx, PREFIX_MATCH_PATH, query).await;
        assert!((quote(&body) - 500.0).abs() < 1e-6, "{body}");
    }

    #[tokio::test]
    async fn prefix_match_omits_the_quote_it_cannot_stand_behind() {
        let prompt: Vec<u64> = (1..=40).collect();
        let query = json!({"token_ids": prompt, "input_tokens": 5000});
        // The two parts come and go together.
        let omitted = |body: &Value| {
            let none = body.get("expected_ttft_ms").is_none();
            assert_eq!(none, body.get("expected_prefill_ms").is_none(), "{body}");
            none
        };

        // Legacy routing picks no engine to quote, reports or not.
        let ctx = ctx_with_tree(&prompt, 6);
        report_prefill(&ctx, "http://a:30000", 0);
        report_prefill(&ctx, "http://b:30000", 0);
        let (_, body) = post_json(ctx, PREFIX_MATCH_PATH, query.clone()).await;
        assert!(omitted(&body), "{body}");

        // No engine reports.
        let ctx = reorg_ctx_with_tree(&prompt, 6, power_of_two);
        let (_, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query.clone()).await;
        assert!(omitted(&body), "{body}");

        // One engine the draw may land on has reported, but never prefilled.
        report_prefill(&ctx, "http://a:30000", 0);
        let now = Instant::now();
        ctx.engine_reported_load
            .set("http://b:30000", 0, load_stat(0, 0, Some(0)), now);
        let (_, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query.clone()).await;
        assert!(omitted(&body), "{body}");
        report_prefill(&ctx, "http://b:30000", 0);
        let (_, body) = post_json(Arc::clone(&ctx), PREFIX_MATCH_PATH, query.clone()).await;
        assert!(!omitted(&body), "{body}");

        // No engine admits the request.
        let ctx = reorg_ctx_with_tree(&prompt, 6, |ctx| {
            let mut policy = crate::policies_reorg::power_of_two::PowerOfTwoPolicy::new(
                Arc::clone(&ctx.engine_reported_load),
            );
            policy.admission = Arc::new(crate::policies_reorg::admission::AdmissionLimits {
                max_waiting_requests: Some(1),
                ..Default::default()
            });
            Arc::new(policy)
        });
        report_prefill(&ctx, "http://a:30000", 100);
        report_prefill(&ctx, "http://b:30000", 100);
        let (_, body) = post_json(ctx, PREFIX_MATCH_PATH, query).await;
        assert_eq!(body["admits"], false, "{body}");
        assert!(omitted(&body), "{body}");
    }

    #[test]
    fn cell_load_without_reports_is_marked_stale() {
        let ctx = AppContext::stub();
        add_worker(&ctx, "a", "http://a:30000", WorkerMode::Plain);
        let model = ModelId(ctx.config.model.id.clone());
        let engines = prefill_engines(&ctx, &model);
        let now = Instant::now();
        let load = summarize_load(
            &engines,
            &ctx.engine_reported_load.capture_snapshot(now),
            now,
        );
        assert_eq!(load.engines, 1);
        assert_eq!(load.reporting_engines, 0);
        assert_eq!(load.min_worker_queue, 0);
        assert_eq!(load.snapshot_age_ms, u32::MAX);
    }
}
