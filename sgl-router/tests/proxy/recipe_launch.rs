// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::common::mock_worker::MockWorker;
use serde_json::{json, Value};
use std::{process::Stdio, time::Duration};
use tokio::process::Command;

#[tokio::test]
async fn gateway_cli_discovers_served_model_and_explicit_pd_workers() {
    let prefill = MockWorker::start(vec![]).await;
    let d0 = MockWorker::start(vec![]).await;
    let d1 = MockWorker::start(vec![]).await;
    let client = reqwest::Client::new();
    for pd in [false, true] {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut command = Command::new(env!("CARGO_BIN_EXE_sgl-router"));
        command.args([
            "--model-path",
            "/models/tokenizer-not-needed",
            "--no-tokenizer",
            "--worker-startup-timeout-secs",
            "5",
            "--health-check-interval-secs",
            "1",
            "--health-success-threshold",
            "1",
            "--prometheus-port",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--shutdown-drain-secs",
            "0",
            "--disable-circuit-breaker",
        ]);
        if pd {
            command.args([
                "--pd-disaggregation",
                "--prefill",
                &prefill.url,
                "30000",
                "--decode",
                &d0.url,
                "--decode",
                &d1.url,
                "--prefill-policy",
                "round_robin",
                "--decode-policy",
                "round_robin",
            ]);
        } else {
            command.args(["--worker-urls", &d0.url]);
        }
        let mut process = command
            .kill_on_drop(true)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let base = format!("http://127.0.0.1:{port}");
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                assert!(
                    process.try_wait().unwrap().is_none(),
                    "gateway-compatible command exited, pd={pd}"
                );
                if client
                    .get(format!("{base}/readyz"))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let models: Value = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            models["data"][0]["id"], "tiny",
            "the local model path is not the served model name"
        );
        for _ in 0..2 {
            let response = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&json!({"model":"tiny","messages":[{"role":"user","content":"hello"}]}))
                .send()
                .await
                .unwrap();
            assert!(
                response.status().is_success(),
                "{}",
                response.text().await.unwrap()
            );
        }
        if pd {
            prefill.wait_for_request().await;
            for worker in [&prefill, &d0, &d1] {
                let body: Value = serde_json::from_slice(&worker.wait_for_request().await).unwrap();
                assert_eq!(body["bootstrap_port"], 30000);
            }
        }
        process.kill().await.unwrap();
        process.wait().await.unwrap();
    }
}
