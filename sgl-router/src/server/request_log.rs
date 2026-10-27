// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request lifecycle observed through the last response/worker/abort owner.

use crate::server::metrics::MetricsRegistry;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

struct RequestContext {
    id: String,
    model: String,
    endpoint: &'static str,
    metrics: Arc<MetricsRegistry>,
}
#[derive(Default, Clone)]
struct Route {
    engine_rid: Option<String>,
    prefill_url: String,
    decode_url: Option<String>,
    candidates: Vec<Value>,
    predicted: Option<u64>,
}
pub(crate) struct Lifecycle {
    context: OnceLock<RequestContext>,
    started: Instant,
    terminal: AtomicU8,
    first: AtomicBool,
    first_ms: OnceLock<u64>,
    reason: OnceLock<&'static str>,
    pub selection: crate::policies::selection::SelectionDiagnostics,
    route: Mutex<Route>,
    observation: Mutex<Observation>,
    prefill_observation: Mutex<Option<u64>>,
    generation_workers: Mutex<Vec<Arc<crate::workers::Worker>>>,
    session_turn: OnceLock<crate::state::session_estimator::PendingTurn>,
}
impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            context: OnceLock::new(),
            started: Instant::now(),
            terminal: AtomicU8::new(0),
            first: AtomicBool::new(false),
            first_ms: OnceLock::new(),
            reason: OnceLock::new(),
            selection: Default::default(),
            route: Mutex::new(Route::default()),
            observation: Mutex::new(Observation::default()),
            prefill_observation: Mutex::new(None),
            generation_workers: Mutex::new(Vec::new()),
            session_turn: OnceLock::new(),
        }
    }
}
impl Lifecycle {
    pub fn configure(
        &self,
        id: String,
        model: String,
        endpoint: &'static str,
        metrics: Arc<MetricsRegistry>,
        active: usize,
        capacity: Option<usize>,
    ) {
        let _ = self.context.set(RequestContext {
            id,
            model,
            endpoint,
            metrics,
        });
        self.emit(
            "admit",
            || json!({"admission_inflight":active,"admission_capacity":capacity,"queue_length":0}),
        );
    }
    pub fn route(
        &self,
        engine_rid: Option<&str>,
        prefill: &str,
        decode: Option<&str>,
        candidates: Vec<Value>,
        streaming: bool,
        bootstrap_room: Option<u64>,
    ) {
        if self.selection.is_empty() {
            self.candidates(candidates.clone(), None);
        }
        let mut route = self.route.lock().unwrap();
        let (selected_candidates, predicted) = self
            .selection
            .take()
            .expect("route diagnostics initialized");
        route.candidates = selected_candidates;
        route.predicted = predicted;
        route.engine_rid = engine_rid.map(str::to_owned);
        route.prefill_url = prefill.into();
        route.decode_url = decode.map(str::to_owned);
        drop(route);
        self.observation.lock().unwrap().streaming = streaming;
        self.emit("route", || {
            let route = self.route.lock().unwrap();
            json!({"engine_rid":engine_rid,"prefill_url":prefill,"decode_url":decode,"bootstrap_room":bootstrap_room,"candidates":route.candidates})
        });
    }
    pub fn candidates(&self, candidates: Vec<Value>, predicted: Option<u64>) {
        self.selection.record(candidates, predicted);
    }
    pub fn progress(&self, source: &'static str) {
        if !self.first.swap(true, Ordering::Relaxed) {
            let _ = self.first_ms.set(self.started.elapsed().as_millis() as u64);
            self.emit("first_progress", || json!({"source":source}));
        }
    }
    pub fn fail(&self, reason: &'static str) {
        let _ = self.reason.set(reason);
        let _ = self
            .terminal
            .compare_exchange(0, 3, Ordering::Relaxed, Ordering::Relaxed);
    }
    pub fn abort(&self) {
        let _ = self
            .terminal
            .compare_exchange(0, 2, Ordering::Relaxed, Ordering::Relaxed);
    }
    /// Feed the engine's counts for this turn back to the uncached estimate
    /// once the request has finished; the first routed attempt's turn counts.
    pub fn track_session_turn(&self, turn: crate::state::session_estimator::PendingTurn) {
        let _ = self.session_turn.set(turn);
    }
    pub fn bind_generation_workers(&self, workers: Vec<Arc<crate::workers::Worker>>) {
        *self.generation_workers.lock().unwrap() = workers;
    }

    pub fn complete(&self) {
        let mut observation = self.observation.lock().unwrap();
        observation.finish();
        let error = observation.error;
        let generated =
            observation.generated || observation.total(|t| t.completion).is_some_and(|n| n > 0);
        drop(observation);
        if error {
            self.fail("response_error_event");
        } else if self
            .terminal
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
            && generated
        {
            for worker in self.generation_workers.lock().unwrap().iter() {
                worker.record_generation_success();
            }
        }
    }
    pub fn observe_prefill(&self, data: &[u8]) {
        let mut observation = Observation {
            streaming: data.starts_with(b"data:") || data.starts_with(b"event:"),
            ..Default::default()
        };
        observation.feed(data);
        observation.finish();
        *self.prefill_observation.lock().unwrap() = observation.actual();
    }
    pub fn observe(&self, data: &[u8]) {
        let mut o = self.observation.lock().unwrap();
        o.feed(data);
        if o.error {
            self.fail("response_error_event");
        }
    }
    pub fn observe_generation(&self, evidence: crate::proxy::grpc::GenerationEvidence) {
        let mut observation = self.observation.lock().unwrap();
        observation.generated |= evidence.generated;
        observation.error |= evidence.failed;
        if observation.error {
            self.fail("response_error_event");
        }
    }
    fn emit(&self, event: &str, extra: impl FnOnce() -> Value) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let Some(ctx) = self.context.get() else {
            return;
        };
        let mut fields = json!({"event":event,"request_id":ctx.id,"model_id":ctx.model,"endpoint":ctx.endpoint,"elapsed_ms":self.started.elapsed().as_millis() as u64});
        if let Value::Object(extra) = extra() {
            fields.as_object_mut().unwrap().extend(extra);
        }
        tracing::debug!(lifecycle_json=%fields,"request_lifecycle");
    }
}
impl Drop for Lifecycle {
    fn drop(&mut self) {
        let Some(ctx) = self.context.get() else {
            return;
        };
        let mut observation = self.observation.lock().unwrap();
        observation.finish();
        let client_actual = observation.actual();
        let error = observation.error;
        let prompt_tokens = observation.total(|t| t.prompt);
        let completion_tokens = observation.total(|t| t.completion);
        let finish_reason = observation.finish_reason.clone();
        drop(observation);
        let route = self.route.lock().unwrap().clone();
        let actual = if route.decode_url.is_some() {
            *self.prefill_observation.lock().unwrap()
        } else {
            client_actual
        };
        ctx.metrics
            .record_cache_observation(&ctx.model, route.predicted, actual);
        // Only a completed turn is in the engine's cache for the next one to extend.
        if let Some(turn) = self.session_turn.get() {
            if self.terminal.load(Ordering::Relaxed) == 1 && !error {
                turn.complete(prompt_tokens, actual);
            }
        }
        let event = match self.terminal.load(Ordering::Relaxed) {
            1 if !error => "finish",
            3 => "error",
            _ if error => "error",
            _ => "abort",
        };
        self.emit(event,|| json!({"engine_rid":route.engine_rid,"prefill_url":route.prefill_url,"decode_url":route.decode_url,
            "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"finish_reason":finish_reason,"first_progress_ms":self.first_ms.get(),"duration_ms":self.started.elapsed().as_millis() as u64,"reason":self.reason.get().copied().unwrap_or(""),"predicted_cached_tokens":route.predicted.unwrap_or(0),"prediction_known":route.predicted.is_some(),
            "actual_cached_tokens":actual.unwrap_or(0),"actual_cache_known":actual.is_some(),"cache_actual_source":if route.decode_url.is_some() {"prefill_response"}else {"client_response"},"client_reported_cached_tokens":client_actual.unwrap_or(0),"client_cache_known":client_actual.is_some()}));
    }
}

const MAX_OBSERVATION_BYTES: usize = 1 << 20;
#[derive(Default)]
struct Tokens {
    cached: Option<u64>,
    prompt: Option<u64>,
    completion: Option<u64>,
}
impl Tokens {
    fn update(&mut self, value: &Value) {
        self.cached = cached(value).or(self.cached);
        self.prompt = value["usage"]["prompt_tokens"]
            .as_u64()
            .or_else(|| value["meta_info"]["prompt_tokens"].as_u64())
            .or(self.prompt);
        self.completion = value["usage"]["completion_tokens"]
            .as_u64()
            .or_else(|| value["meta_info"]["completion_tokens"].as_u64())
            .or(self.completion);
    }
}
#[derive(Default)]
struct Observation {
    streaming: bool,
    pending: Vec<u8>,
    single: Tokens,
    items: HashMap<u64, Tokens>,
    finish_reason: Option<Value>,
    error: bool,
    overflow: bool,
    finished: bool,
    generated: bool,
}
impl Observation {
    fn feed(&mut self, data: &[u8]) {
        if self.overflow {
            return;
        }
        if self.pending.len().saturating_add(data.len()) > MAX_OBSERVATION_BYTES {
            self.pending.clear();
            self.overflow = true;
            return;
        }
        self.pending.extend_from_slice(data);
        if self.streaming {
            loop {
                let lf = self
                    .pending
                    .windows(2)
                    .position(|w| w == b"\n\n")
                    .map(|p| (p, 2));
                let crlf = self
                    .pending
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|p| (p, 4));
                let Some((end, size)) = lf.into_iter().chain(crlf).min_by_key(|p| p.0) else {
                    break;
                };
                let event: Vec<_> = self.pending.drain(..end + size).collect();
                let mut data = Vec::new();
                for line in event.split(|b| *b == b'\n') {
                    if let Some(line) = line.strip_prefix(b"data:") {
                        data.extend_from_slice(line);
                        data.push(b'\n');
                    }
                }
                if let Ok(value) = serde_json::from_slice(&data) {
                    self.value(&value);
                }
            }
        }
    }
    fn value(&mut self, value: &Value) {
        self.generated |= generated_output(value);
        if let Some(items) = value.as_array() {
            for (index, item) in items.iter().enumerate().take(4096) {
                self.items.entry(index as u64).or_default().update(item);
                self.error |= is_error(item);
            }
            return;
        }
        self.error |= is_error(value);
        let reason = &value["meta_info"]["finish_reason"];
        if !reason.is_null() {
            self.finish_reason = Some(reason.clone());
        }
        if let Some(choices) = value["choices"].as_array() {
            if let Some(reason) = choices
                .iter()
                .map(|c| &c["finish_reason"])
                .find(|v| !v.is_null())
            {
                self.finish_reason = Some(reason.clone());
            }
        }
        match value["index"].as_u64().filter(|i| *i < 4096) {
            Some(index) => self.items.entry(index).or_default().update(value),
            None => self.single.update(value),
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.streaming && !self.overflow {
            if let Ok(value) = serde_json::from_slice::<Value>(&self.pending) {
                self.value(&value);
            }
        }
        self.pending.clear();
    }
    fn total(&self, field: impl Fn(&Tokens) -> Option<u64>) -> Option<u64> {
        if self.overflow {
            return None;
        }
        if self.items.is_empty() {
            field(&self.single)
        } else {
            self.items
                .values()
                .try_fold(0u64, |total, t| Some(total.saturating_add(field(t)?)))
        }
    }
    fn actual(&self) -> Option<u64> {
        self.total(|t| t.cached)
    }
}
fn cached(value: &Value) -> Option<u64> {
    value["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .or_else(|| value["meta_info"]["cached_tokens"].as_u64())
}

pub(crate) fn is_error(value: &Value) -> bool {
    !value["error"].is_null() || value["meta_info"]["finish_reason"]["type"] == "abort"
}

pub(crate) fn generated_output(value: &Value) -> bool {
    let nonempty = |v: &Value| v.as_str().is_some_and(|s| !s.is_empty());
    value["usage"]["completion_tokens"]
        .as_u64()
        .is_some_and(|n| n > 0)
        || value["meta_info"]["completion_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0)
        || value["output_ids"]
            .as_array()
            .is_some_and(|ids| !ids.is_empty())
        || nonempty(&value["text"])
        || value["choices"].as_array().is_some_and(|choices| {
            choices.iter().any(|choice| {
                [&choice["delta"], &choice["message"]]
                    .iter()
                    .any(|message| {
                        nonempty(&message["content"])
                            || nonempty(&message["reasoning_content"])
                            || message["tool_calls"].as_array().is_some_and(|calls| {
                                calls.iter().any(|call| {
                                    nonempty(&call["function"]["name"])
                                        || nonempty(&call["function"]["arguments"])
                                })
                            })
                    })
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_observation_accepts_split_crlf_and_cumulative_usage_without_double_counting() {
        let mut o = Observation {
            streaming: true,
            ..Default::default()
        };
        let frame = b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":7}}}\r\n\r\n";
        for chunk in frame.chunks(3) {
            o.feed(chunk);
        }
        o.feed(frame);
        o.finish();
        assert_eq!(o.actual(), Some(7));
        assert!(!o.error);
    }
    #[test]
    fn native_batch_sums_latest_values_per_index_and_overflow_is_unknown() {
        let mut o = Observation::default();
        o.feed(br#"[{"meta_info":{"cached_tokens":3}},{"meta_info":{"cached_tokens":5}}]"#);
        o.finish();
        assert_eq!(o.actual(), Some(8));
        let mut o = Observation::default();
        o.feed(&vec![b' '; MAX_OBSERVATION_BYTES + 1]);
        o.finish();
        assert_eq!(o.actual(), None);
    }
    #[test]
    fn streaming_native_batch_reports_error_and_replaces_cumulative_counts() {
        let mut o = Observation {
            streaming: true,
            ..Default::default()
        };
        for (index, tokens) in [(0, 3), (1, 5), (0, 4)] {
            o.feed(
                format!(
                    "data: {}\n\n",
                    json!({"index":index,"meta_info":{"cached_tokens":tokens}})
                )
                .as_bytes(),
            );
        }
        o.feed(b"data: {\"meta_info\":{\"finish_reason\":{\"type\":\"abort\"}}}\n\n");
        o.finish();
        assert_eq!(o.actual(), Some(9));
        assert!(o.error);
    }
}

#[cfg(test)]
mod generation_proof_tests {
    use super::*;

    #[test]
    fn streaming_health_proof_uses_output_without_usage_but_excludes_errors_and_role_only() {
        for (delta, error, expected) in [
            (json!({"role":"assistant"}), false, false),
            (json!({"content":""}), false, false),
            (json!({"content":"hello"}), false, true),
            (json!({"reasoning_content":"thinking"}), false, true),
            (
                json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}),
                false,
                true,
            ),
            (json!({"content":"hello"}), true, false),
        ] {
            let worker = Arc::new(crate::workers::Worker::new(Default::default()));
            let life = Lifecycle::default();
            life.bind_generation_workers(vec![Arc::clone(&worker)]);
            life.route(None, "engine", None, vec![], true, None);
            let frame = format!(
                "data: {}\n\n",
                json!({"choices":[{"index":0,"delta":delta}]})
            );
            for bytes in frame.as_bytes().chunks(3) {
                life.observe(bytes);
            }
            if error {
                life.observe(b"data: {\"error\":{\"message\":\"failed\"}}\n\n");
            } else {
                life.observe(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
            }
            assert!(!worker.recently_generated(std::time::Duration::from_secs(1)));
            life.complete();
            assert_eq!(
                worker.recently_generated(std::time::Duration::from_secs(1)),
                expected,
                "delta={delta}, error={error}"
            );
        }
    }

    #[test]
    fn generation_proof_requires_positive_tokens_and_successful_completion() {
        for (body, abort, expected) in [
            (r#"{"meta_info":{"completion_tokens":1}}"#, false, true),
            (r#"{"meta_info":{"completion_tokens":0}}"#, false, false),
            (
                r#"{"error":{"message":"failed"},"meta_info":{"completion_tokens":1}}"#,
                false,
                false,
            ),
            (r#"{"meta_info":{"completion_tokens":1}}"#, true, false),
        ] {
            let worker = Arc::new(crate::workers::Worker::new(Default::default()));
            let life = Lifecycle::default();
            life.bind_generation_workers(vec![Arc::clone(&worker)]);
            life.observe(body.as_bytes());
            assert!(
                !worker.recently_generated(std::time::Duration::from_secs(1)),
                "body data alone is not completed proof"
            );
            if abort {
                life.abort();
            }
            life.complete();
            assert_eq!(
                worker.recently_generated(std::time::Duration::from_secs(1)),
                expected
            );
        }
    }
}
