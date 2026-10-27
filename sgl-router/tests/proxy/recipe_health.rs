// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json,
};
use serde_json::{json, Value};
use std::{
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::process::Command;

#[derive(Default)]
struct Engine {
    works: AtomicBool,
    delay_ms: AtomicU64,
    probe_delay_ms: AtomicU64,
    health_calls: AtomicU64,
    generation_calls: AtomicU64,
}

async fn generate(
    State(engine): State<Arc<Engine>>,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    engine.generation_calls.fetch_add(1, Ordering::Relaxed);
    let delay = engine.delay_ms.load(Ordering::Relaxed)
        + if headers.contains_key("x-sgl-router-probe") {
            engine.probe_delay_ms.load(Ordering::Relaxed)
        } else {
            0
        };
    tokio::time::sleep(Duration::from_millis(delay)).await;
    if engine.works.load(Ordering::Relaxed) && uri.path() == "/v1/chat/completions" {
        Json(json!({"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}})).into_response()
    } else if engine.works.load(Ordering::Relaxed) {
        Json(json!({"text":"OK","output_ids":[7],"meta_info":{"completion_tokens":1,"finish_reason":{"type":"length"}}})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":"generation unavailable"})),
        )
            .into_response()
    }
}
fn port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
async fn status(client: &reqwest::Client, base: &str, path: &str) -> Option<StatusCode> {
    client
        .get(format!("{base}{path}"))
        .send()
        .await
        .ok()
        .map(|r| r.status())
}
async fn wait_status(client: &reqwest::Client, base: &str, path: &str, expected: StatusCode) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if status(client, base, path).await == Some(expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{path} never returned {expected}"));
}

#[tokio::test]
async fn recipe_health_requires_generation_and_recovers_while_infra_routes_remain_live() {
    let engine = Arc::new(Engine::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new()
        .route("/health", get(|| async { StatusCode::OK })) // Deliberately lies about generation.
        .route(
            "/server_info",
            get(|| async { Json(json!({"served_model_name":"tiny","model_path":"fixture"})) }),
        )
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(generate))
        .route("/abort_request", post(|| async { StatusCode::OK }))
        .with_state(Arc::clone(&engine));
    let server = tokio::spawn(async {
        axum::serve(listener, app).await.unwrap();
    });
    let (http, metrics) = (port(), port());
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--worker-urls",
            &worker,
            "--policy",
            "round_robin",
            "--host",
            "127.0.0.1",
            "--port",
            &http.to_string(),
            "--prometheus-port",
            &metrics.to_string(),
            "--disable-retries",
            "--disable-circuit-breaker",
            "--health-check-mode",
            "generation",
            "--health-check-interval-secs",
            "1",
            "--health-check-timeout-secs",
            "1",
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "2",
            "--max-payload-size",
            "1024",
            "--shutdown-drain-secs",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let base = format!("http://127.0.0.1:{http}");
    let client = reqwest::Client::new();
    wait_status(&client, &base, "/liveness", StatusCode::OK).await;
    wait_status(
        &client,
        &base,
        "/readiness",
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;
    // Several successful /health polls must not invent a full-path success.
    tokio::time::sleep(Duration::from_millis(2200)).await;
    assert_eq!(
        status(&client, &base, "/readiness").await,
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    let workers: Value = client
        .get(format!("{base}/workers"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(workers["workers"][0]["is_healthy"], false);
    engine.works.store(true, Ordering::Relaxed);
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    wait_status(&client, &base, "/health", StatusCode::OK).await;
    let workers: Value = client
        .get(format!("{base}/workers"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(workers["workers"][0]["url"], worker);
    assert_eq!(workers["workers"][0]["worker_type"], "regular");
    assert_eq!(workers["workers"][0]["is_healthy"], true);
    assert_eq!(
        status(&client, &base, "/server_info").await,
        Some(StatusCode::OK)
    );
    let body = json!({"model":"tiny","messages":[{"role":"user","content":"hello"}]});
    for supplied in [Some("caller-id"), None, None] {
        let mut request = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&body);
        if let Some(id) = supplied {
            request = request.header("x-request-id", id);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let id = response
            .headers()
            .get("x-request-id")
            .expect("mint or echo request correlation")
            .to_str()
            .unwrap();
        if let Some(expected) = supplied {
            assert_eq!(id, expected);
        } else {
            uuid::Uuid::parse_str(id).unwrap();
        }
    }
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("a".repeat(2048))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let scrape = client
        .get(format!("http://127.0.0.1:{metrics}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(scrape.status(), StatusCode::OK);
    assert!(scrape
        .text()
        .await
        .unwrap()
        .contains("sgl_router_requests_total"));
    // Slower than a cheap GET, but within the configured generation-probe budget.
    engine.delay_ms.store(700, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(2800)).await;
    assert_eq!(
        status(&client, &base, "/readiness").await,
        Some(StatusCode::OK)
    );
    engine.works.store(false, Ordering::Relaxed);
    wait_status(
        &client,
        &base,
        "/readiness",
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;
    assert_eq!(
        status(&client, &base, "/liveness").await,
        Some(StatusCode::OK)
    );
    engine.delay_ms.store(0, Ordering::Relaxed);
    engine.works.store(true, Ordering::Relaxed);
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    process.kill().await.unwrap();
    process.wait().await.unwrap();
    server.abort();
}

async fn fake_engine(
    works: bool,
    delay_ms: u64,
) -> (Arc<Engine>, String, tokio::task::JoinHandle<()>) {
    let engine = Arc::new(Engine {
        works: AtomicBool::new(works),
        delay_ms: AtomicU64::new(delay_ms),
        probe_delay_ms: AtomicU64::new(0),
        ..Default::default()
    });
    let app = axum::Router::new()
        .route(
            "/health",
            get(|State(engine): State<Arc<Engine>>| async move {
                engine.health_calls.fetch_add(1, Ordering::Relaxed);
                StatusCode::OK
            }),
        )
        .route(
            "/server_info",
            get(|| async { Json(json!({"served_model_name":"tiny","model_path":"fixture"})) }),
        )
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(generate))
        .route("/abort_request", post(|| async { StatusCode::OK }))
        .with_state(Arc::clone(&engine));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async {
        axum::serve(listener, app).await.unwrap();
    });
    (engine, url, task)
}

#[tokio::test]
async fn default_http_and_off_modes_do_not_probe_generation() {
    for mode in [None, Some("off")] {
        let (engine, worker, server) = fake_engine(false, 0).await;
        let http = port();
        let mut command = Command::new(env!("CARGO_BIN_EXE_sgl-router"));
        command.args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--worker-urls",
            &worker,
            "--policy",
            "round_robin",
            "--host",
            "127.0.0.1",
            "--port",
            &http.to_string(),
            "--prometheus-port",
            "0",
            "--disable-circuit-breaker",
            "--health-success-threshold",
            "1",
            "--health-check-interval-secs",
            "1",
            "--shutdown-drain-secs",
            "0",
        ]);
        if let Some(mode) = mode {
            command.args(["--health-check-mode", mode]);
        }
        let mut process = command
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://127.0.0.1:{http}");
        wait_status(&client, &base, "/readiness", StatusCode::OK).await;
        assert_eq!(
            engine.generation_calls.load(Ordering::Relaxed),
            0,
            "mode={mode:?}"
        );
        if mode.is_some() {
            assert_eq!(
                engine.health_calls.load(Ordering::Relaxed),
                0,
                "off disables active HTTP probes too"
            );
        } else {
            assert!(
                engine.health_calls.load(Ordering::Relaxed) > 0,
                "HTTP mode checks status"
            );
        }
        process.kill().await.unwrap();
        process.wait().await.unwrap();
        server.abort();
    }
}

#[tokio::test]
async fn pd_health_keeps_a_slow_prefill_and_excludes_a_decode_that_cannot_generate() {
    let (_p, p_url, p_task) = fake_engine(true, 700).await;
    let (d0, d0_url, d0_task) = fake_engine(true, 0).await;
    let (_d1, d1_url, d1_task) = fake_engine(false, 0).await;
    let http = port();
    let base = format!("http://127.0.0.1:{http}");
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--pd-disaggregation",
            "--prefill",
            &p_url,
            "30000",
            "--decode",
            &d0_url,
            "--decode",
            &d1_url,
            "--prefill-policy",
            "round_robin",
            "--decode-policy",
            "round_robin",
            "--host",
            "127.0.0.1",
            "--port",
            &http.to_string(),
            "--prometheus-port",
            "0",
            "--disable-retries",
            "--disable-circuit-breaker",
            "--health-check-mode",
            "generation",
            "--health-check-interval-secs",
            "1",
            "--health-check-timeout-secs",
            "1",
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "2",
            "--shutdown-drain-secs",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let client = reqwest::Client::new();
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    let workers: Value = client
        .get(format!("{base}/workers"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for (url, healthy) in [(&p_url, true), (&d0_url, true), (&d1_url, false)] {
        let worker = workers["workers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w["url"] == *url)
            .unwrap();
        assert_eq!(worker["is_healthy"], healthy, "{worker}");
    }
    for _ in 0..4 {
        let response = client
            .post(format!("{base}/generate"))
            .json(&json!({"input_ids":[1],"sampling_params":{"max_new_tokens":1}}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "a dead decode must be excluded from round robin"
        );
        assert_eq!(response.headers()["x-sgl-decode-url"], d0_url);
    }
    d0.works.store(false, Ordering::Relaxed);
    wait_status(
        &client,
        &base,
        "/readiness",
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;
    assert_eq!(
        status(&client, &base, "/liveness").await,
        Some(StatusCode::OK)
    );
    d0.works.store(true, Ordering::Relaxed);
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    process.kill().await.unwrap();
    process.wait().await.unwrap();
    p_task.abort();
    d0_task.abort();
    d1_task.abort();
}

#[tokio::test]
async fn startup_liveness_is_available_while_worker_model_resolution_is_pending() {
    let unavailable = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", unavailable.local_addr().unwrap());
    let http = port();
    let base = format!("http://127.0.0.1:{http}");
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-path",
            "/models/no-tokenizer-needed",
            "--no-tokenizer",
            "--worker-urls",
            &worker,
            "--worker-startup-timeout-secs",
            "3",
            "--host",
            "127.0.0.1",
            "--port",
            &http.to_string(),
            "--prometheus-port",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let client = reqwest::Client::new();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if status(&client, &base, "/liveness").await == Some(StatusCode::OK) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("worker startup must not delay router liveness");
    assert_eq!(
        status(&client, &base, "/readiness").await,
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    process.kill().await.unwrap();
    process.wait().await.unwrap();
}

#[tokio::test]
async fn recent_successful_generation_keeps_busy_workers_ready_without_masking_stalls() {
    let engine = Arc::new(Engine::default());
    engine.works.store(true, Ordering::Relaxed);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route(
            "/server_info",
            get(|| async { Json(json!({"served_model_name":"tiny","model_path":"fixture"})) }),
        )
        .route("/generate", post(generate))
        .route("/abort_request", post(|| async { StatusCode::OK }))
        .with_state(Arc::clone(&engine));
    let server = tokio::spawn(async {
        axum::serve(listener, app).await.unwrap();
    });
    let http = port();
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--worker-urls",
            &worker,
            "--policy",
            "round_robin",
            "--host",
            "127.0.0.1",
            "--port",
            &http.to_string(),
            "--prometheus-port",
            &port().to_string(),
            "--disable-retries",
            "--disable-circuit-breaker",
            "--health-check-mode",
            "generation",
            "--health-check-interval-secs",
            "1",
            "--health-check-timeout-secs",
            "1",
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "1",
            "--shutdown-drain-secs",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let base = format!("http://127.0.0.1:{http}");
    let client = reqwest::Client::new();
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    engine.probe_delay_ms.store(2_000, Ordering::Relaxed);
    for _ in 0..65 {
        let response = client
            .post(format!("{base}/generate"))
            .json(&json!({"input_ids":[7],"sampling_params":{"max_new_tokens":1}}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "completed generation must retain full-path proof under load"
        );
        response.bytes().await.unwrap();
        assert_eq!(
            status(&client, &base, "/readiness").await,
            Some(StatusCode::OK)
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Old traffic must expire; an always-successful health GET cannot hide a stalled generator.
    wait_status(
        &client,
        &base,
        "/readiness",
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;
    engine.probe_delay_ms.store(0, Ordering::Relaxed);
    wait_status(&client, &base, "/readiness", StatusCode::OK).await;
    process.kill().await.unwrap();
    process.wait().await.unwrap();
    server.abort();
}
