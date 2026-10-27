use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

#[tokio::test]
async fn omitted_prometheus_port_does_not_bind_the_default_listener() {
    // Holding the old default port makes an implicit listener fail startup.
    let _occupied = match std::net::TcpListener::bind("127.0.0.1:29000") {
        Ok(listener) => Some(listener),
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => None,
        Err(error) => panic!("bind test port: {error}"),
    };
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
        .args([
            "--model-id",
            "tiny",
            "--no-tokenizer",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--worker-urls",
            "http://127.0.0.1:9",
            "--health-check-mode",
            "off",
            "--shutdown-drain-secs",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let client = reqwest::Client::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            assert!(
                process.try_wait().unwrap().is_none(),
                "router exited while the default metrics port was occupied"
            );
            if client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    process.kill().await.unwrap();
    process.wait().await.unwrap();
}
