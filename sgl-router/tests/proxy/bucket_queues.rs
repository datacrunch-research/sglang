// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Reorg buckets on estimated uncached tokens, with bounded queues, 429 +
//! `Retry-After`, depth from membership and one-way idle spill, end to end
//! through the chat route.

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use bytes::Bytes;
use http_body_util::BodyExt;
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::factory::build_resolver;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::KvEventIndex;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tower::ServiceExt;

/// An engine that answers chat and `/generate` after `delay`, reporting the
/// prompt as its body size / 4 and `cached` cached tokens.
struct Engine {
    url: String,
    hits: Arc<AtomicUsize>,
    cached: Arc<AtomicU64>,
    _shutdown: oneshot::Sender<()>,
}

#[derive(Clone)]
struct EngineState {
    delay: Duration,
    hits: Arc<AtomicUsize>,
    cached: Arc<AtomicU64>,
}

async fn answer(State(state): State<EngineState>, body: Bytes) -> axum::Json<serde_json::Value> {
    state.hits.fetch_add(1, Ordering::SeqCst);
    tokio::time::sleep(state.delay).await;
    let prompt = (body.len() / 4) as u64;
    let cached = state.cached.load(Ordering::SeqCst).min(prompt);
    axum::Json(serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": 1,
            "total_tokens": prompt + 1,
            "prompt_tokens_details": {"cached_tokens": cached}
        },
        "meta_info": {"prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": 1}
    }))
}

impl Engine {
    async fn start(delay: Duration) -> Self {
        let hits = Arc::new(AtomicUsize::new(0));
        let cached = Arc::new(AtomicU64::new(0));
        let state = EngineState {
            delay,
            hits: Arc::clone(&hits),
            cached: Arc::clone(&cached),
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(answer))
            .route("/generate", post(answer))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            hits,
            cached,
            _shutdown: tx,
        }
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::PowerOfTwo,
            decode_policy: Default::default(),
            dp_aware: false,
            dp_rank_policy: Default::default(),
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            roles: Default::default(),
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
    }
}

/// A reorg context over `workers` (id, mode, engine) and the bucket file `buckets`.
fn context(workers: &[(&str, WorkerMode, &Engine)], buckets: serde_json::Value) -> Arc<AppContext> {
    let mut config = config();
    config.model.reorg_buckets = Some(serde_json::from_value(buckets).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for &(id, mode, engine) in workers {
        registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: engine.url.clone(),
                mode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
                ..Default::default()
            })
            .unwrap();
    }
    let state = KvEventIndex::new();
    let (resolver, _) = build_resolver(&config.model, &state, None).unwrap();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let mut ctx = AppContext::new(
        config,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        Arc::new(PolicyRegistry::default()),
    );
    ctx.engine_reported_load = state.engine_reported_load();
    ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
    Arc::new(ctx)
}

fn chat(session: Option<&str>, messages: &[(&str, &str)]) -> Request<Body> {
    let messages: Vec<_> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(session) = session {
        builder = builder.header("x-session-id", session);
    }
    builder
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({"model": "tiny", "messages": messages}))
                .unwrap(),
        ))
        .unwrap()
}

/// Send `request`, read the whole response; returns status, headers and body.
async fn send(
    app: &axum::Router,
    request: Request<Body>,
) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let (parts, body) = response.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (parts.status, parts.headers, body)
}

fn estimator_sessions(ctx: &AppContext) -> usize {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        unreachable!()
    };
    resolvers[&ModelId("tiny".into())]
        .estimator
        .as_ref()
        .map_or(0, |estimator| estimator.len())
}

/// The estimator learns a turn when the request's lifecycle ends, just after
/// its body is read.
async fn wait_for_sessions(ctx: &AppContext, sessions: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while estimator_sessions(ctx) < sessions {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the completed turn reaches the estimator");
}

async fn wait_for_hits(engine: &Engine, hits: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while engine.hits() < hits {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("the engine receives the request");
}

/// Two plain lanes split at 200 uncached tokens.
fn short_and_long_lanes() -> serde_json::Value {
    serde_json::json!({"buckets": [
        {"id": "S", "max_uncached_tokens": 200, "plain": {"worker_ids": ["s"]}},
        {"id": "L", "min_uncached_tokens": 201, "plain": {"worker_ids": ["l"]}}
    ]})
}

#[tokio::test]
async fn a_long_conversation_appending_a_short_turn_goes_to_the_short_lane() {
    let (short, long) = (
        Engine::start(Duration::ZERO).await,
        Engine::start(Duration::ZERO).await,
    );
    let ctx = context(
        &[
            ("s", WorkerMode::Plain, &short),
            ("l", WorkerMode::Plain, &long),
        ],
        short_and_long_lanes(),
    );
    let app = build_router(Arc::clone(&ctx));
    let task = "x".repeat(2_000);
    let first = [("system", "agent"), ("user", task.as_str())];
    // First turn: no history, the whole ~500-token prompt is uncached.
    assert_eq!(send(&app, chat(Some("a"), &first)).await.0, StatusCode::OK);
    assert_eq!((short.hits(), long.hits()), (0, 1));
    wait_for_sessions(&ctx, 1).await;
    // Second turn appends a few tokens to the same conversation.
    long.cached.store(u64::MAX, Ordering::SeqCst);
    let second = [
        ("system", "agent"),
        ("user", task.as_str()),
        ("assistant", "ok"),
        ("user", "go on"),
    ];
    assert_eq!(send(&app, chat(Some("a"), &second)).await.0, StatusCode::OK);
    assert_eq!((short.hits(), long.hits()), (1, 1));
    // Another session sending the same prompt is cold.
    assert_eq!(send(&app, chat(Some("b"), &second)).await.0, StatusCode::OK);
    assert_eq!((short.hits(), long.hits()), (1, 2));
}

#[tokio::test]
async fn without_a_session_header_the_first_messages_key_the_conversation() {
    let (short, long) = (
        Engine::start(Duration::ZERO).await,
        Engine::start(Duration::ZERO).await,
    );
    let ctx = context(
        &[
            ("s", WorkerMode::Plain, &short),
            ("l", WorkerMode::Plain, &long),
        ],
        short_and_long_lanes(),
    );
    let app = build_router(Arc::clone(&ctx));
    let task = "y".repeat(2_000);
    let first = [("system", "agent"), ("user", task.as_str())];
    assert_eq!(send(&app, chat(None, &first)).await.0, StatusCode::OK);
    wait_for_sessions(&ctx, 1).await;
    let second = [
        ("system", "agent"),
        ("user", task.as_str()),
        ("assistant", "ok"),
        ("user", "next"),
    ];
    assert_eq!(send(&app, chat(None, &second)).await.0, StatusCode::OK);
    assert_eq!((short.hits(), long.hits()), (1, 1));
}

#[tokio::test]
async fn a_full_bucket_answers_429_at_once_with_retry_after() {
    let engine = Engine::start(Duration::from_millis(800)).await;
    // One member, a one-token depth: one request runs, none queue.
    let ctx = context(
        &[("s", WorkerMode::Plain, &engine)],
        serde_json::json!({"buckets": [{
            "id": "S", "max_uncached_tokens": 1000000,
            "queue": {"mean_wait_s": 1, "rate_tokens_per_s": 1, "rate_requests_per_s": 1},
            "plain": {"worker_ids": ["s"]}
        }]}),
    );
    let app = build_router(Arc::clone(&ctx));
    let running = tokio::spawn({
        let app = app.clone();
        async move { send(&app, chat(Some("a"), &[("user", "first")])).await.0 }
    });
    wait_for_hits(&engine, 1).await;
    let started = Instant::now();
    let (status, headers, body) = send(&app, chat(Some("b"), &[("user", "second")])).await;
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "refused at once"
    );
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers["x-router-error-code"], "bucket_full");
    assert_eq!(headers["x-sgl-bucket"], "S");
    let retry_after: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=30).contains(&retry_after));
    assert_eq!(body["error"]["type"], "bucket_full");
    assert_eq!(body["error"]["depth_tokens"], 1);
    assert_eq!(
        engine.hits(),
        1,
        "a refused request never reaches an engine"
    );
    assert_eq!(running.await.unwrap(), StatusCode::OK);
    // The finished request's place is free again.
    assert_eq!(
        send(&app, chat(Some("c"), &[("user", "third")])).await.0,
        StatusCode::OK
    );
}

/// One engine admitting one request at a time behind a deep queue.
fn one_at_a_time(mean_wait_s: f64) -> serde_json::Value {
    serde_json::json!({"buckets": [{
        "id": "S", "max_uncached_tokens": 1000000,
        "queue": {"mean_wait_s": mean_wait_s, "rate_tokens_per_s": 1e9, "rate_requests_per_s": 100},
        "plain": {"worker_ids": ["s"], "admission": {"max_inflight_requests": 1}}
    }]})
}

#[tokio::test]
async fn an_admitted_request_waits_for_its_engine() {
    let engine = Engine::start(Duration::from_millis(300)).await;
    let ctx = context(&[("s", WorkerMode::Plain, &engine)], one_at_a_time(1.0));
    let app = build_router(Arc::clone(&ctx));
    let running = tokio::spawn({
        let app = app.clone();
        async move { send(&app, chat(Some("a"), &[("user", "first")])).await.0 }
    });
    wait_for_hits(&engine, 1).await;
    let started = Instant::now();
    let (status, _, _) = send(&app, chat(Some("b"), &[("user", "second")])).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        started.elapsed() >= Duration::from_millis(200),
        "waited for the running request"
    );
    assert_eq!(running.await.unwrap(), StatusCode::OK);
    assert_eq!(engine.hits(), 2);
}

#[tokio::test]
async fn a_wait_past_the_deadline_answers_429_not_5xx() {
    let engine = Engine::start(Duration::from_millis(1_500)).await;
    // Deadline 0.2 s x 2.3 = 0.46 s, well before the running request ends.
    let ctx = context(&[("s", WorkerMode::Plain, &engine)], one_at_a_time(0.2));
    let app = build_router(Arc::clone(&ctx));
    let _running = tokio::spawn({
        let app = app.clone();
        async move { send(&app, chat(Some("a"), &[("user", "first")])).await.0 }
    });
    wait_for_hits(&engine, 1).await;
    let started = Instant::now();
    let (status, headers, body) = send(&app, chat(Some("b"), &[("user", "second")])).await;
    let waited = started.elapsed();
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert!(waited >= Duration::from_millis(440), "{waited:?}");
    assert!(waited < Duration::from_millis(1_400), "{waited:?}");
    assert_eq!(headers["x-router-error-code"], "bucket_deadline");
    assert_eq!(headers["retry-after"], "2");
    assert_eq!(body["error"]["type"], "bucket_deadline");
    assert_eq!(engine.hits(), 1);
}

#[tokio::test]
async fn a_pd_request_holds_its_place_until_prefill_finishes() {
    let prefill = Engine::start(Duration::from_millis(600)).await;
    let decode = Engine::start(Duration::ZERO).await;
    // A zero-depth bucket: one request per prefill member, none queued.
    let ctx = context(
        &[
            ("p", WorkerMode::Prefill, &prefill),
            ("d", WorkerMode::Decode, &decode),
        ],
        serde_json::json!({"buckets": [{
            "id": "L", "min_uncached_tokens": 1,
            "queue": {"mean_wait_s": 0, "rate_tokens_per_s": 10000, "rate_requests_per_s": 0.05},
            "prefill": {"worker_ids": ["p"]}, "decode": {"worker_ids": ["d"]}
        }]}),
    );
    let app = build_router(Arc::clone(&ctx));
    // Decode answers at once; the detached prefill is still running.
    assert_eq!(
        send(&app, chat(Some("a"), &[("user", "first")])).await.0,
        StatusCode::OK
    );
    let (status, headers, _) = send(&app, chat(Some("b"), &[("user", "second")])).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers["x-router-error-code"], "bucket_full");
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(
        send(&app, chat(Some("c"), &[("user", "third")])).await.0,
        StatusCode::OK
    );
    assert_eq!((prefill.hits(), decode.hits()), (2, 2));
}

fn queue_members(ctx: &AppContext, bucket: usize) -> u64 {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        unreachable!()
    };
    resolvers[&ModelId("tiny".into())].buckets[bucket]
        .queue
        .as_ref()
        .unwrap()
        .members()
}

#[tokio::test]
async fn bucket_depth_follows_membership_events_without_traffic() {
    let (a, b) = (
        Engine::start(Duration::ZERO).await,
        Engine::start(Duration::ZERO).await,
    );
    let ctx = context(
        &[("a", WorkerMode::Plain, &a), ("b", WorkerMode::Plain, &b)],
        serde_json::json!({"buckets": [{
            "id": "S", "max_uncached_tokens": 32000,
            "queue": {"mean_wait_s": 4, "rate_tokens_per_s": 20000, "rate_requests_per_s": 3},
            "plain": {"worker_ids": ["a", "b"]}
        }]}),
    );
    let watch = sgl_router::server::bucket_admission::spawn_membership_watch(Arc::clone(&ctx))
        .expect("a queued bucket is watched");
    let members = |expected: u64| {
        let ctx = Arc::clone(&ctx);
        async move {
            tokio::time::timeout(Duration::from_millis(500), async {
                while queue_members(&ctx, 0) != expected {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .is_ok()
        }
    };
    assert!(members(2).await);
    // Well inside the watch's periodic recheck: the event itself recomputes.
    tokio::time::sleep(Duration::from_millis(50)).await;
    ctx.registry.remove(&WorkerId("b".into()));
    ctx.registry
        .membership
        .record(sgl_router::workers::MembershipEvent::Removed);
    assert!(members(1).await, "a Removed lowers the depth at once");
    watch.abort();
}

/// Report `engine` as running `running` requests with `waiting` uncached tokens queued.
fn report_load(ctx: &AppContext, engine: &Engine, running: u64, waiting: u64) {
    use sgl_router::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};
    ctx.engine_reported_load.set(
        &engine.url,
        0,
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 1_000_000,
            native_cache: Some(NativeCacheRankLoad {
                num_waiting_uncached_tokens: waiting,
                num_total_tokens: 0,
                max_running_requests: 64,
                total_prefill_uncached_tokens: 0,
                total_prefill_busy_us: 0,
            }),
        },
        Instant::now(),
    );
}

/// S (one engine, one request at a time) spills to L; L never spills to S.
fn spilling_lanes() -> serde_json::Value {
    serde_json::json!({"buckets": [
        {"id": "S", "max_uncached_tokens": 200, "spill_to": ["L"],
         "queue": {"mean_wait_s": 0.2, "rate_tokens_per_s": 1e9, "rate_requests_per_s": 100},
         "plain": {"worker_ids": ["s"], "admission": {"max_inflight_requests": 1}}},
        {"id": "L", "min_uncached_tokens": 201,
         "queue": {"mean_wait_s": 0.2, "rate_tokens_per_s": 1e9, "rate_requests_per_s": 100},
         "plain": {"worker_ids": ["l"], "admission": {"max_inflight_requests": 1}}}
    ]})
}

#[tokio::test]
async fn a_short_request_spills_to_an_idle_long_engine_and_never_back() {
    let short = Engine::start(Duration::from_millis(3_000)).await;
    let long = Engine::start(Duration::from_millis(1_000)).await;
    let ctx = context(
        &[
            ("s", WorkerMode::Plain, &short),
            ("l", WorkerMode::Plain, &long),
        ],
        spilling_lanes(),
    );
    let app = build_router(Arc::clone(&ctx));
    report_load(&ctx, &long, 0, 0);
    // The short engine is busy with one request.
    let busy = tokio::spawn({
        let app = app.clone();
        async move { send(&app, chat(None, &[("user", "a")])).await.0 }
    });
    wait_for_hits(&short, 1).await;
    // The next short request takes the idle long engine instead of waiting.
    let started = Instant::now();
    assert_eq!(
        send(&app, chat(None, &[("user", "b")])).await.0,
        StatusCode::OK
    );
    assert!(
        started.elapsed() < Duration::from_millis(2_000),
        "served before the short engine frees up"
    );
    assert_eq!((short.hits(), long.hits()), (1, 1));
    // With the long engine reported busy, a short request waits and is refused.
    report_load(&ctx, &long, 5, 0);
    let (status, headers, _) = send(&app, chat(None, &[("user", "c")])).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers["x-router-error-code"], "bucket_deadline");
    assert_eq!(busy.await.unwrap(), StatusCode::OK);
    // A long request never spills onto the idle short engine.
    report_load(&ctx, &short, 0, 0);
    let filler = "z".repeat(2_000);
    let blocker = tokio::spawn({
        let app = app.clone();
        let filler = filler.clone();
        async move { send(&app, chat(None, &[("user", filler.as_str())])).await.0 }
    });
    wait_for_hits(&long, 2).await;
    let (status, _, _) = send(&app, chat(None, &[("user", filler.as_str())])).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        short.hits(),
        1,
        "nothing moved from the long lane to the short one"
    );
    assert_eq!(blocker.await.unwrap(), StatusCode::OK);
}

#[tokio::test]
async fn responses_name_the_bucket_estimate_wait_and_engines() {
    let (short, long) = (
        Engine::start(Duration::ZERO).await,
        Engine::start(Duration::ZERO).await,
    );
    let ctx = context(
        &[
            ("s", WorkerMode::Plain, &short),
            ("l", WorkerMode::Plain, &long),
        ],
        short_and_long_lanes(),
    );
    let app = build_router(Arc::clone(&ctx));
    let task = "w".repeat(2_000);
    let (status, headers, _) = send(&app, chat(Some("a"), &[("user", task.as_str())])).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-sgl-bucket"], "L");
    assert_eq!(headers["x-sgl-prefill-worker"], "l");
    let estimate: u64 = headers["x-sgl-uncached-estimate"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        estimate > 200,
        "a cold first turn counts its whole prompt: {estimate}"
    );
    assert!(headers.get("x-sgl-decode-worker").is_none(), "plain");
    assert!(headers.get("x-sgl-queue-wait-ms").is_none(), "not queued");

    let prefill = Engine::start(Duration::ZERO).await;
    let decode = Engine::start(Duration::ZERO).await;
    let ctx = context(
        &[
            ("p", WorkerMode::Prefill, &prefill),
            ("d", WorkerMode::Decode, &decode),
        ],
        serde_json::json!({"buckets": [{
            "id": "S", "max_uncached_tokens": 1000000,
            "queue": {"mean_wait_s": 4, "rate_tokens_per_s": 20000, "rate_requests_per_s": 3},
            "prefill": {"worker_ids": ["p"]}, "decode": {"worker_ids": ["d"]}
        }]}),
    );
    let app = build_router(Arc::clone(&ctx));
    let (status, headers, _) = send(&app, chat(Some("a"), &[("user", "hi")])).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-sgl-bucket"], "S");
    assert_eq!(headers["x-sgl-prefill-worker"], "p");
    assert_eq!(headers["x-sgl-decode-worker"], "d");
    let waited: u64 = headers["x-sgl-queue-wait-ms"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(waited < 1_000);
}

async fn scrape(app: &axum::Router) -> String {
    let response = app
        .clone()
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

#[tokio::test]
async fn metrics_export_the_bucket_series_under_their_fixed_names() {
    let prefill = Engine::start(Duration::from_millis(600)).await;
    let decode = Engine::start(Duration::ZERO).await;
    let ctx = context(
        &[
            ("p", WorkerMode::Prefill, &prefill),
            ("d", WorkerMode::Decode, &decode),
        ],
        serde_json::json!({"buckets": [{
            "id": "L", "min_uncached_tokens": 1,
            "queue": {"mean_wait_s": 0, "rate_tokens_per_s": 10000, "rate_requests_per_s": 0.05},
            "prefill": {"worker_ids": ["p"]},
            "decode": {"worker_ids": ["d"], "admission": {"max_inflight_requests": 40}}
        }]}),
    );
    let app = build_router(Arc::clone(&ctx));
    assert_eq!(
        send(&app, chat(Some("a"), &[("user", "first")])).await.0,
        StatusCode::OK
    );
    let (status, _, _) = send(&app, chat(Some("b"), &[("user", "second")])).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    let body = scrape(&app).await;
    for line in [
        "sgl_router_bucket_members{bucket=\"L\"} 1\n",
        "sgl_router_bucket_depth_tokens{bucket=\"L\"} 0\n",
        "sgl_router_bucket_depth_requests{bucket=\"L\"} 0\n",
        "sgl_router_bucket_queue_requests{bucket=\"L\"} 1\n",
        "sgl_router_bucket_admitted_total{bucket=\"L\"} 1\n",
        "sgl_router_bucket_rejected_total{bucket=\"L\",reason=\"bucket_full\"} 1\n",
        "sgl_router_bucket_rejected_total{bucket=\"L\",reason=\"deadline\"} 0\n",
        "sgl_router_bucket_rejected_total{bucket=\"L\",reason=\"decode_permits\"} 0\n",
        "sgl_router_bucket_queue_wait_seconds_count{bucket=\"L\"} 1\n",
        "sgl_router_retry_after_seconds_count{bucket=\"L\"} 1\n",
        "sgl_router_decode_permits 40\n",
        "sgl_router_decode_running{worker=\"d\"} 0\n",
        "sgl_router_membership_events_total{kind=\"added\"} 0\n",
        "# TYPE sgl_router_bucket_queue_tokens gauge\n",
        "# TYPE sgl_router_bucket_spill_total counter\n",
        "# TYPE sgl_router_session_uncached_estimate_error histogram\n",
    ] {
        assert!(body.contains(line), "missing {line:?} in\n{body}");
    }
    // Once the prefill ends, its place is free and the gauges say so.
    tokio::time::sleep(Duration::from_millis(800)).await;
    let body = scrape(&app).await;
    assert!(
        body.contains("sgl_router_bucket_queue_requests{bucket=\"L\"} 0\n"),
        "{body}"
    );
    assert!(
        body.contains("sgl_router_bucket_queue_tokens{bucket=\"L\"} 0\n"),
        "{body}"
    );
}
