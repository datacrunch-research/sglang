// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::common::mock_worker::MockWorker;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use clap::Parser;
use futures::future::join_all;
use http_body_util::BodyExt;
use sgl_router::{
    config::Cli,
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    policies::factory::build_registry_with_defaults,
    proxy::Proxy,
    server::{app::build_router, app_context::AppContext},
    tokenizer::TokenizerRegistry,
    workers::WorkerRegistry,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tower::ServiceExt;

fn app(worker: &MockWorker, cap: usize) -> axum::Router {
    app_url(&worker.url, cap)
}

fn app_url(url: &str, cap: usize) -> axum::Router {
    let cfg = Cli::try_parse_from([
        "router",
        "--model-id",
        "tiny",
        "--no-tokenizer",
        "--policy",
        "round_robin",
        "--worker-urls",
        url,
        "--disable-circuit-breaker",
        "--max-concurrent-requests",
        &cap.to_string(),
        "--queue-size",
        "0",
        "--rate-limit-tokens-per-second",
        "1",
    ])
    .expect("accept the gateway admission contract")
    .into_config()
    .unwrap();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("worker".into()),
            url: url.to_owned(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            ..Default::default()
        })
        .unwrap();
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(10)).unwrap());
    build_router(Arc::new(AppContext::new(
        cfg, tokenizers, proxy, registry, policies,
    )))
}

fn request(stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model":"tiny", "messages":[{"role":"user","content":"hello"}], "stream":stream,
            })
            .to_string(),
        ))
        .unwrap()
}

#[tokio::test]
async fn synchronized_bursts_hold_exactly_48_streams_and_reject_without_queueing() {
    let worker = MockWorker::start_slow_stream(
        vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_secs(10),
    )
    .await;
    for burst in [56, 64, 80] {
        let app = app(&worker, 48);
        let replies = join_all((0..burst).map(|_| {
            let app = app.clone();
            async move {
                let start = Instant::now();
                let response = app.oneshot(request(true)).await.unwrap();
                (response, start.elapsed())
            }
        }))
        .await;
        assert_eq!(
            replies
                .iter()
                .filter(|(r, _)| r.status() == StatusCode::OK)
                .count(),
            48
        );
        assert_eq!(
            replies
                .iter()
                .filter(|(r, _)| r.status() == StatusCode::TOO_MANY_REQUESTS)
                .count(),
            burst - 48
        );
        for (_, latency) in replies
            .iter()
            .filter(|(r, _)| r.status() == StatusCode::TOO_MANY_REQUESTS)
        {
            assert!(
                *latency < Duration::from_millis(10),
                "queue=0 rejection took {latency:?}"
            );
        }
        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        // Refill must never let a held stream bypass the strict concurrent limit.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert_eq!(
            app.clone().oneshot(request(true)).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(replies);
        // Cleanup is asynchronous when an engine abort is needed.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let response = app.clone().oneshot(request(true)).await.unwrap();
                if response.status() == StatusCode::OK {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped streams must release admission after cleanup");
    }
}

#[tokio::test]
async fn admission_releases_on_validation_error_and_completed_nonstreaming_response() {
    let worker = MockWorker::start(vec![]).await;
    let app = app(&worker, 1);
    let invalid = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    assert_eq!(
        app.clone().oneshot(invalid).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    for _ in 0..3 {
        let response = app.clone().oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.into_body().collect().await.unwrap();
    }
}

#[tokio::test]
async fn admission_stays_held_until_the_upstream_abort_completes() {
    use axum::routing::post;
    let (sent_tx, mut sent) = tokio::sync::mpsc::unbounded_channel();
    let (abort_tx, mut aborted) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(tokio::sync::Notify::new());
    let done = Arc::clone(&release);
    let engine = axum::Router::new()
        .route(
            "/v1/chat/completions",
            post(move || {
                let tx = sent_tx.clone();
                async move {
                    tx.send(()).unwrap();
                    std::future::pending::<Body>().await
                }
            }),
        )
        .route(
            "/abort_request",
            post(move || {
                let tx = abort_tx.clone();
                let done = Arc::clone(&done);
                async move {
                    tx.send(()).unwrap();
                    done.notified().await;
                    StatusCode::OK
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let app = app_url(&format!("http://{}", listener.local_addr().unwrap()), 1);
    let server = tokio::spawn(async {
        axum::serve(listener, engine).await.unwrap();
    });
    let task = tokio::spawn(app.clone().oneshot(request(false)));
    sent.recv().await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), aborted.recv())
        .await
        .unwrap()
        .unwrap();
    let invalid = || {
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap()
    };
    assert_eq!(
        app.clone().oneshot(invalid()).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS,
        "the engine may still be computing until abort acknowledges cleanup"
    );
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if app.clone().oneshot(invalid()).await.unwrap().status() == StatusCode::BAD_REQUEST {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    server.abort();
}
