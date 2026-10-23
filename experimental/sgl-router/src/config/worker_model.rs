// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{Config, DiscoveryBackend};
use crate::workers::introspect::{worker_client, WorkerIntrospector};
use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use std::time::Duration;

/// Resolve gateway-style --model-path without publishing a local path as a served model name.
pub async fn resolve_worker_model(config: &mut Config) -> Result<()> {
    if !config.server.infer_model_from_workers {
        return Ok(());
    }
    let DiscoveryBackend::StaticUrls(workers) = &config.discovery else {
        anyhow::bail!("automatic served-model discovery requires static worker URLs");
    };
    let client = worker_client(Duration::from_secs(2), config.server.worker_auth.clone());
    let introspector = WorkerIntrospector::with_client(client);
    let name = tokio::time::timeout(
        Duration::from_secs(config.server.worker_startup_timeout_secs),
        async {
            loop {
                let mut pending: FuturesUnordered<_> = workers
                    .urls
                    .iter()
                    .map(|url| introspector.fetch(url))
                    .collect();
                while let Some(info) = pending.next().await {
                    if let Some(name) = info
                        .served_model_name
                        .filter(|name| !name.trim().is_empty())
                    {
                        return name;
                    }
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        },
    )
    .await
    .context("worker startup deadline expired before a served model name was available")?;
    tracing::info!(model = %name, "resolved served model from worker metadata");
    config.model.id = name;
    config.server.infer_model_from_workers = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Cli;
    use clap::Parser;

    #[tokio::test]
    async fn omitted_model_name_honors_worker_startup_deadline() {
        let mut config = Cli::try_parse_from([
            "sgl-router",
            "--model-path",
            "/models/local",
            "--no-tokenizer",
            "--worker-urls",
            "http://127.0.0.1:9",
            "--worker-startup-timeout-secs",
            "1",
        ])
        .unwrap()
        .into_config()
        .unwrap();
        let result =
            tokio::time::timeout(Duration::from_secs(3), resolve_worker_model(&mut config))
                .await
                .expect("the one-second startup budget must bound metadata retries");
        assert!(result.unwrap_err().to_string().contains("startup deadline"));
    }

    #[test]
    fn gateway_model_alias_and_invalid_pd_arguments_are_validated() {
        let config = Cli::try_parse_from([
            "sgl-router",
            "--served-model-name",
            "served",
            "--model-path",
            "/models/local",
            "--worker-urls",
            "http://worker:30000",
        ])
        .unwrap()
        .into_config()
        .unwrap();
        assert_eq!(config.model.id, "served");
        assert_eq!(
            config.model.tokenizer_path.as_deref(),
            Some("/models/local")
        );
        assert!(!config.server.infer_model_from_workers);

        for tail in [
            "--pd-disaggregation --prefill http://p:30000 30000",
            "--prefill http://p:30000 30000 --decode http://d:30000",
            "--pd-disaggregation --prefill http://p:30000 0 --decode http://d:30000",
            "--pd-disaggregation --prefill http://p:30000 30000 --decode http://p:30000",
            "--pd-disaggregation --prefill http://p:30000 30000 --decode http://d:30000 --worker-urls http://w:30000",
        ] {
            let args: Vec<_> = ["sgl-router", "--model-id", "served"].into_iter()
                .chain(tail.split_whitespace()).collect();
            let result = Cli::try_parse_from(args).map_err(anyhow::Error::from)
                .and_then(Cli::into_config);
            assert!(result.is_err(), "invalid static P/D configuration was accepted: {tail}");
        }
    }
}
