// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::{body::Body, http::Request, http::StatusCode};
use clap::Parser;
use dynamo_tokenizers::{traits, Encoding};
use std::sync::{atomic::AtomicUsize, mpsc, Condvar, Mutex};
use std::time::Duration;
use tower::ServiceExt;

struct GatedEncoder {
    inner: Arc<Tokenizer>,
    entered: Mutex<Option<mpsc::Sender<()>>>,
    release: Arc<(Mutex<bool>, Condvar)>,
    calls: Arc<AtomicUsize>,
}

impl GatedEncoder {
    fn wait(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(entered) = self.entered.lock().unwrap().take() {
            entered.send(()).unwrap();
            let (lock, changed) = &*self.release;
            let (_guard, timeout) = changed
                .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |done| !*done)
                .unwrap();
            assert!(
                !timeout.timed_out(),
                "test controller did not release encoder"
            );
        }
    }
}

impl traits::Encoder for GatedEncoder {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.wait();
        self.inner.encode(input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl traits::Decoder for GatedEncoder {
    fn decode(&self, ids: &[u32], skip_special: bool) -> Result<traits::DecodeResult> {
        self.wait();
        self.inner.decode(ids, skip_special)
    }
}

impl traits::Tokenizer for GatedEncoder {}

fn generation() -> Request<Body> {
    Request::post("/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"text":"hello","sampling_params":{"max_new_tokens":1}}"#,
        ))
        .unwrap()
}

fn context(encoder: Arc<dyn traits::Tokenizer>) -> Arc<crate::server::app_context::AppContext> {
    use crate::{
        config::Cli, policies::factory::build_registry_with_defaults, proxy::Proxy,
        server::app_context::AppContext, workers::WorkerRegistry,
    };
    let config = Cli::try_parse_from([
        "router",
        "--model-id",
        "tiny",
        "--no-tokenizer",
        "--policy",
        "round_robin",
        "--worker-urls",
        "http://127.0.0.1:9",
        "--max-concurrent-requests",
        "1",
        "--queue-size",
        "0",
        "--rate-limit-tokens-per-second",
        "1",
    ])
    .unwrap()
    .into_config()
    .unwrap();
    let tokenizers = TokenizerRegistry {
        prompt_affixes: Some(Default::default()),
        ..Default::default()
    };
    tokenizers
        .inner
        .insert("tiny".into(), Arc::new(Tokenizer::from(encoder)));
    let policies = Arc::new(build_registry_with_defaults(&config).unwrap());
    Arc::new(AppContext::new(
        config,
        Arc::new(tokenizers),
        Arc::new(Proxy::new(Duration::from_secs(1)).unwrap()),
        Arc::new(WorkerRegistry::default()),
        policies,
    ))
}

async fn assert_control_routes_usable(input: Request<Body>) {
    use crate::server::{admin, app::build_router};
    let (entered_tx, entered_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let encoder: Arc<dyn traits::Tokenizer> = Arc::new(GatedEncoder {
        inner: adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap(),
        entered: Mutex::new(Some(entered_tx)),
        release: release.clone(),
        calls: calls.clone(),
    });
    let ctx = context(encoder);
    let app = build_router(ctx.clone());
    let task = tokio::spawn(app.clone().oneshot(input));
    let probe = Arc::new(tokio::sync::Notify::new());
    let probe_ready = probe.clone();
    let (responsive_tx, responsive_rx) = mpsc::channel();
    let (checked_tx, checked_rx) = mpsc::channel();
    let controller = std::thread::spawn(move || {
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        probe_ready.notify_one();
        let responsive = responsive_rx
            .recv_timeout(Duration::from_millis(500))
            .is_ok();
        if responsive {
            checked_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        let (lock, changed) = &*release;
        *lock.lock().unwrap() = true;
        changed.notify_all();
        responsive
    });
    probe.notified().await;
    for path in ["/liveness", "/metrics"] {
        let reply = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(reply.status(), StatusCode::OK);
    }
    let reply = admin::router(ctx.clone())
        .oneshot(Request::get("/admin/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(reply.status(), StatusCode::OK);
    let while_encoding = responsive_tx.send(()).is_ok();
    task.abort();
    let _ = task.await;
    let active_during_cancel = ctx.admission.active();
    let rejected = app.clone().oneshot(generation()).await.unwrap().status();
    let _ = checked_tx.send(());
    assert!(
        controller.join().unwrap(),
        "encoding blocked control routes on the async executor"
    );
    assert!(while_encoding);
    assert_eq!(
        active_during_cancel, 1,
        "cancelled CPU work lost its admission permit"
    );
    assert_eq!(rejected, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "rejected work was tokenized"
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        while ctx.admission.active() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("completed encoder must return admission");
}

/// Control routes stay responsive during encoding; cancellation retains admission until CPU work ends.
#[tokio::test(flavor = "current_thread")]
async fn encoding_keeps_control_routes_usable_and_cancelled_work_admitted() {
    assert_control_routes_usable(generation()).await;
}

#[tokio::test(flavor = "current_thread")]
async fn raw_tokenization_keeps_control_routes_usable_and_work_admitted() {
    let input = Request::post("/v1/tokenize")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"tiny","prompt":"hello"}"#))
        .unwrap();
    assert_control_routes_usable(input).await;
}

#[tokio::test(flavor = "current_thread")]
async fn raw_detokenization_keeps_control_routes_usable_and_work_admitted() {
    let input = Request::post("/v1/detokenize")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"tiny","tokens":[104,101]}"#))
        .unwrap();
    assert_control_routes_usable(input).await;
}

/// Native text RPCs rejected at the global cap must never run the encoder.
#[tokio::test]
async fn native_text_rpcs_take_admission_before_encoding() {
    use crate::server::{admission::RequestScope, grpc};
    use sglang_api_types::runtime::v1::{
        self as proto, sglang_service_client::SglangServiceClient,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let encoder: Arc<dyn traits::Tokenizer> = Arc::new(GatedEncoder {
        inner: adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap(),
        entered: Mutex::new(None),
        release: Arc::new((Mutex::new(true), Condvar::new())),
        calls: calls.clone(),
    });
    let ctx = context(encoder);
    let _held = RequestScope::enter(&ctx.admission).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let shutdown = tokio_util::sync::CancellationToken::new();
    let server = tokio::spawn(grpc::serve(
        listener,
        ctx,
        shutdown.clone().cancelled_owned(),
    ));
    let mut client = SglangServiceClient::connect(format!("http://{address}"))
        .await
        .unwrap();
    let generate = client
        .text_generate(proto::TextGenerateRequest {
            text: "hello".into(),
            ..Default::default()
        })
        .await
        .expect_err("full cap must reject TextGenerate");
    let embed = client
        .text_embed(proto::TextEmbedRequest {
            text: "hello".into(),
            ..Default::default()
        })
        .await
        .expect_err("full cap must reject TextEmbed");
    let classify = client
        .classify(proto::ClassifyRequest {
            text: "hello".into(),
            ..Default::default()
        })
        .await
        .expect_err("full cap must reject Classify");
    shutdown.cancel();
    server.await.unwrap().unwrap();
    for status in [generate, embed, classify] {
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "rejected native text RPCs ran the encoder"
    );
}
