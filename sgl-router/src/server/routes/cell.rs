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
use crate::server::app_context::{AppContext, ChatRouting};
use crate::server::error::ApiError;
use crate::server::routes::chat::lookup_prefix_for_tokens;
use crate::state::kv_events::CacheNamespace;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use sgl_kv_indexer::PrefixOutcome;
use sglang_api_types::runtime::v1 as proto;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const PREFIX_MATCH_PATH: &str = "/internal/prefix_match";
pub const CELL_LOAD_PATH: &str = "/internal/cell_load";
pub const CELL_INFO_PATH: &str = "/internal/cell_info";

/// Body of `POST /internal/prefix_match`. The caller sends the prompt's
/// leading tokens (a few thousand are enough: an owner is decided by the
/// start of the prompt) and the full length separately.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct PrefixMatchRequest {
    /// The model the tokens belong to; must be the one this router serves.
    pub model: Option<String>,
    pub token_ids: Vec<u32>,
    /// The whole prompt's length; defaults to `token_ids.len()`.
    pub input_tokens: Option<u64>,
    /// KV namespace components, as the chat request would carry them.
    pub cache_salt: Option<String>,
    pub lora_name: Option<String>,
    /// The caller's remaining budget. A lookup that cannot finish inside it
    /// answers 504 rather than late, so the caller counts a miss and moves on.
    pub deadline_ms: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct PrefixMatchResponse {
    /// Deepest contiguous prefix any engine of this cell holds, in tokens,
    /// capped at the query length.
    pub matched_prefix_tokens: u64,
    pub matched_prefix_blocks: u32,
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

/// Why a prefix match produced no answer: the budget ran out.
#[derive(Debug)]
pub enum PrefixMatchFailure {
    Api(ApiError),
    /// The lookup did not finish within `deadline_ms`.
    Deadline,
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
    let lookup = lookup_prefix_for_tokens(ctx, &request.token_ids, &namespace, &model.0);
    let signal = match request.deadline_ms {
        Some(ms) if ms > 0 => {
            match tokio::time::timeout(Duration::from_millis(u64::from(ms)), lookup).await {
                Ok(signal) => signal?,
                Err(_) => return Err(PrefixMatchFailure::Deadline),
            }
        }
        _ => lookup.await?,
    };

    let now = Instant::now();
    let engines = prefill_engines(ctx, &model);
    let urls: HashSet<&str> = engines.iter().map(|w| w.url.as_str()).collect();
    let (matched_prefix_blocks, pending) = cell_depth(signal.as_ref(), &urls);
    let page_size = ctx.block_size_oracle.get().unwrap_or(0);
    let matched_prefix_tokens = (u64::from(matched_prefix_blocks) * u64::from(page_size))
        .min(request.token_ids.len() as u64);
    let snapshot = ctx.engine_reported_load.capture_snapshot(now);
    let load = summarize_load(&engines, &snapshot, now);
    let input_tokens = request
        .input_tokens
        .unwrap_or(request.token_ids.len() as u64);
    let admits = admits(ctx, &model, input_tokens, &snapshot);
    Ok(PrefixMatchResponse {
        matched_prefix_tokens,
        matched_prefix_blocks,
        page_size,
        pending,
        load,
        admits,
    })
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
    /// Model families with their processor spec. Empty until engines report
    /// their processor configuration; an upstream without a planner for the
    /// family matches on the text prefix only.
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
    /// reporting engine is multimodal.
    fn from_engines(engines: &[Arc<Worker>]) -> Option<Self> {
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
            families: Vec::new(),
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
        grpc_port: ctx.config.server.grpc_port,
        engines: engines.len() as u32,
        engines_reporting_chat: reported.len() as u32,
        chat: reported.into_iter().next(),
        consistent,
        mm: CellMm::from_engines(&engines),
        router_default_chat_template_kwargs: router_kwargs,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
    use crate::state::kv_events::prefix_provider::RadixTreePrefixProvider;
    use crate::state::kv_events::tree::{HashTree, KvWorkerId};
    use crate::state::kv_events::wire::BlockStored;
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
    fn ctx_with_tree(prompt: &[u32], blocks: usize) -> Arc<AppContext> {
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
        Arc::new(ctx)
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

    #[tokio::test]
    async fn prefix_match_reports_the_cell_depth_in_tokens_and_blocks() {
        let prompt: Vec<u32> = (1..=40).collect();
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
        let prompt: Vec<u32> = (1..=40).collect();
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
        let prompt: Vec<u32> = (1..=40).collect();
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
        let prompt: Vec<u32> = (1..=40).collect();
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
    }

    #[tokio::test]
    async fn cell_signals_convert_to_their_grpc_messages() {
        let prompt: Vec<u32> = (1..=40).collect();
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
