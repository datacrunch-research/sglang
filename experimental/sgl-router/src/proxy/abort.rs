// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{header::AUTHORIZATION, HeaderMap};
use reqwest::{Client, RequestBuilder, Url};
use std::time::Duration;

/// Cancels unfinished engine work without delaying request cleanup.
#[derive(Default)]
pub struct AbortOnDrop {
    request: Option<RequestBuilder>,
    guards: Option<Box<dyn Send + 'static>>,
}

impl AbortOnDrop {
    // Only router-minted IDs are safe: the engine aborts by prefix.
    pub(super) fn new(
        client: &Client,
        worker: &Url,
        headers: &HeaderMap,
        rid: Option<&str>,
    ) -> Self {
        Self {
            request: rid.filter(|rid| !rid.is_empty()).map(|rid| {
                let mut request = client
                    .post(worker.join("/abort_request").expect("validated worker URL"))
                    .json(&serde_json::json!({"rid": rid, "abort_all": false}))
                    .timeout(Duration::from_secs(5));
                if let Some(auth) = headers.get(AUTHORIZATION) {
                    request = request.header(AUTHORIZATION, auth);
                }
                request
            }),
            guards: None,
        }
    }

    /// Keep admission and worker occupancy until bounded abort cleanup finishes.
    pub fn with_guards(mut self, guards: Option<Box<dyn Send + 'static>>) -> Self {
        self.guards = guards;
        self
    }

    pub fn disarm(&mut self) {
        self.request = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let guards = self.guards.take();
            runtime.spawn(async move {
                let _guards = guards;
                if let Err(error) = request.send().await.and_then(|r| r.error_for_status()) {
                    tracing::warn!(%error, "engine abort failed");
                }
            });
        }
    }
}
