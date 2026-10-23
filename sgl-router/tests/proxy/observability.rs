// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{json, Value};
use std::{
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::process::Command;

#[derive(Clone, Default)]
struct Engine {
    headers: Arc<Mutex<Vec<HeaderMap>>>,
    aborts: Arc<Mutex<Vec<Value>>>,
}
async fn chat(
    State(engine): State<Engine>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> axum::response::Response {
    engine.headers.lock().unwrap().push(headers);
    match body["test_case"].as_str() {
        Some("hold") => ([("content-type","text/event-stream")],Body::from_stream(futures::stream::pending::<Result<bytes::Bytes,std::io::Error>>())).into_response(),
        Some("late") => ([("content-type","text/event-stream")],"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\nevent: error\ndata: {\"error\":{\"message\":\"late failure\"}}\n\ndata: [DONE]\n\n").into_response(),
        _ => Json(json!({"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":7}}})).into_response(),
    }
}
async fn abort(State(engine): State<Engine>, Json(body): Json<Value>) -> StatusCode {
    engine.aborts.lock().unwrap().push(body);
    StatusCode::OK
}
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
async fn wait_ready(client: &reqwest::Client, base: &str) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if client
                .get(format!("{base}/readiness"))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
}
fn payload(kind: &str) -> Value {
    json!({"model":"tiny","messages":[{"role":"user","content":"hello"}],"stream":kind!="ok","test_case":kind})
}
fn lifecycle(log: &str, id: &str) -> Vec<Value> {
    log.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|line| {
            let fields = &line["fields"];
            (fields["request_id"] == id && fields["event"].is_string()).then(|| fields.clone())
        })
        .collect()
}

#[tokio::test]
async fn lifecycle_records_late_errors_and_admission_cleanup_with_bounded_metrics() {
    let engine = Engine::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route(
            "/server_info",
            get(|| async { Json(json!({"served_model_name":"tiny"})) }),
        )
        .route(
            "/generate",
            post(|| async { Json(json!({"text":"OK","meta_info":{"completion_tokens":1}})) }),
        )
        .route("/v1/chat/completions", post(chat))
        .route("/abort_request", post(abort))
        .with_state(engine.clone());
    let server = tokio::spawn(async {
        axum::serve(listener, app).await.unwrap();
    });
    let (port, metrics, admin) = (free_port(), free_port(), free_port());
    let log = tempfile::NamedTempFile::new().unwrap();
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--policy",
            "round_robin",
            "--worker-urls",
            &worker,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--prometheus-port",
            &metrics.to_string(),
            "--health-check-interval-secs",
            "1",
            "--health-success-threshold",
            "1",
            "--disable-retries",
            "--disable-circuit-breaker",
            "--max-concurrent-requests",
            "1",
            "--queue-size",
            "0",
            "--rate-limit-tokens-per-second",
            "1",
            "--log-level",
            "debug",
            "--log-format",
            "json",
            "--shutdown-drain-secs",
            "0",
            "--admin-port",
            &admin.to_string(),
            "--worker-api-key",
            "test-secret-value",
            "--forward-request-header",
            "x-client-context",
        ])
        .stdout(Stdio::from(log.as_file().try_clone().unwrap()))
        .stderr(Stdio::from(log.as_file().try_clone().unwrap()))
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    wait_ready(&client, &base).await;
    let trace = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
    let late = client
        .post(format!("{base}/v1/chat/completions"))
        .header("x-request-id", "lifecycle-late")
        .header("traceparent", trace)
        .header("x-attempt-id", "attempt-unchanged")
        .header("x-dynamo-session-id", "session-unchanged")
        .header("x-client-context", "context-unchanged")
        .header("x-private-context", "must-not-forward")
        .json(&payload("late"))
        .send()
        .await
        .unwrap();
    assert_eq!(late.status(), StatusCode::OK);
    assert!(late.text().await.unwrap().contains("late failure"));
    let ok = client
        .post(format!("{base}/v1/chat/completions"))
        .header("x-request-id", "lifecycle-ok")
        .json(&payload("ok"))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), StatusCode::OK);
    ok.text().await.unwrap();
    let held = client
        .post(format!("{base}/v1/chat/completions"))
        .header("x-request-id", "lifecycle-abort")
        .json(&payload("hold"))
        .send()
        .await
        .unwrap();
    assert_eq!(held.status(), StatusCode::OK);
    let rejected = client
        .post(format!("{base}/v1/chat/completions"))
        .header("x-request-id", "lifecycle-reject")
        .json(&payload("ok"))
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    rejected.text().await.unwrap();
    drop(held);
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let text = std::fs::read_to_string(log.path()).unwrap();
            if lifecycle(&text, "lifecycle-abort")
                .iter()
                .any(|e| e["event"] == "abort")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("abort lifecycle must be recorded after worker cleanup");
    let text = std::fs::read_to_string(log.path()).unwrap();
    for (id, terminal) in [
        ("lifecycle-late", "error"),
        ("lifecycle-ok", "finish"),
        ("lifecycle-abort", "abort"),
    ] {
        let events = lifecycle(&text, id);
        assert!(
            events.iter().any(|e| e["event"] == "admit"),
            "{id}: {events:?}"
        );
        let route = events
            .iter()
            .find(|e| e["event"] == "route")
            .expect("route decision missing");
        assert_eq!(route["prefill_url"], worker);
        assert!(route["candidates"].is_array());
        assert!(!route["candidates"].as_array().unwrap().is_empty());
        assert!(
            events.iter().any(|e| e["event"] == terminal),
            "{id}: missing {terminal}: {events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| ["finish", "abort", "error"].contains(&e["event"].as_str().unwrap()))
                .count(),
            1,
            "terminal log must be unique"
        );
        if terminal == "error" {
            assert!(events.iter().all(|e| e["event"] != "finish"));
            assert!(events.iter().any(|e| e["event"] == "first_progress"));
        }
    }
    let finished = lifecycle(&text, "lifecycle-ok")
        .into_iter()
        .find(|e| e["event"] == "finish")
        .unwrap();
    assert_eq!(finished["prompt_tokens"], 10);
    assert_eq!(finished["completion_tokens"], 1);
    assert!(finished["first_progress_ms"].is_number());
    assert!(lifecycle(&text, "lifecycle-reject")
        .iter()
        .any(|e| e["event"] == "reject"));
    let scrape = client
        .get(format!("http://127.0.0.1:{metrics}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    for metric in [
        "sgl_router_admission_inflight",
        "sgl_router_admission_capacity",
        "sgl_router_admission_total",
        "sgl_router_cache_tokens_total",
        "sgl_router_worker_health",
        "sgl_router_worker_cb_state",
    ] {
        assert!(scrape.contains(metric), "missing metric {metric}");
    }
    assert!(scrape
        .lines()
        .any(|l| l == "sgl_router_admission_inflight 0"));
    assert!(
        !scrape.contains("lifecycle-"),
        "request ids must never become metric labels"
    );
    {
        let seen = engine.headers.lock().unwrap();
        assert!(seen
            .iter()
            .any(|h| h.get("traceparent").is_some_and(|v| v == trace)
                && h.get("x-attempt-id")
                    .is_some_and(|v| v == "attempt-unchanged")
                && h.get("x-dynamo-session-id")
                    .is_some_and(|v| v == "session-unchanged")
                && h.get("x-client-context")
                    .is_some_and(|v| v == "context-unchanged")
                && !h.contains_key("x-private-context")));
    }
    let admin_base = format!("http://127.0.0.1:{admin}");
    assert_eq!(
        client
            .get(format!("{base}/admin/config"))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND,
        "admin must be on its own listener"
    );
    let config = client
        .get(format!("{admin_base}/admin/config"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!config.contains("test-secret-value"));
    assert_eq!(
        serde_json::from_str::<Value>(&config).unwrap()["worker_auth_configured"],
        true
    );
    let workers: Value = client
        .get(format!("{admin_base}/admin/workers"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(workers["workers"].as_array().unwrap().len(), 1);
    let index: Value = client
        .get(format!("{admin_base}/admin/index?limit=999999"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(index["limit"].as_u64().unwrap() <= 256);
    assert!(index["nodes"].as_array().unwrap().len() <= 256);
    let level = client
        .post(format!("{admin_base}/admin/log-level"))
        .json(&json!({"filter":"info,sgl_router=debug"}))
        .send()
        .await
        .unwrap();
    assert_eq!(level.status(), StatusCode::OK);
    let invalid = client
        .post(format!("{admin_base}/admin/log-level"))
        .json(&json!({"filter":"bad["}))
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    let adjusted = client
        .put(format!("{admin_base}/admin/cap"))
        .json(&json!({"max_concurrent_requests":2}))
        .send()
        .await
        .unwrap();
    assert_eq!(adjusted.status(), StatusCode::OK);
    let config: Value = client
        .get(format!("{admin_base}/admin/config"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(config["max_concurrent_requests"], 2);
    let mut held = Vec::new();
    for _ in 0..2 {
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&payload("hold"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        held.push(response);
    }
    let adjusted = client
        .put(format!("{admin_base}/admin/cap"))
        .json(&json!({"max_concurrent_requests":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(adjusted.status(), StatusCode::OK);
    assert_eq!(
        client
            .post(format!("{base}/v1/chat/completions"))
            .json(&payload("ok"))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    drop(held);
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let metrics = client
                .get(format!("http://127.0.0.1:{metrics}/metrics"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            if metrics
                .lines()
                .any(|l| l == "sgl_router_admission_inflight 0")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&payload("ok"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    response.text().await.unwrap();
    process.kill().await.unwrap();
    process.wait().await.unwrap();
    server.abort();
}
