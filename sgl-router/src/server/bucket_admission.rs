// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-bucket admission: bounded prefill work with a wait deadline.
//!
//! A bucket admits a request while its outstanding prefill work (admitted, not
//! yet through prefill) stays within the bucket's depth. By Little's law the
//! work a queue holds over the rate it drains at is the mean wait, so
//! `depth_tokens = members x rate_tokens_per_s x mean_wait_s` and
//! `depth_requests = members x rate_requests_per_s x mean_wait_s`, with
//! `members` the bucket's ready prefill (or plain) engines. Each member may
//! also run one request whatever its size, so a zero-depth bucket runs one
//! request per member and queues none.
//!
//! Over the depth the answer is 429 at once, with `Retry-After` = the excess
//! work over the bucket's drain rate. An admitted request that finds no engine
//! admitting it waits up to `deadline_factor x mean_wait_s`, then gets 429.
//! Nothing here answers 5xx.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::Notify;
use tokio::time::Instant;

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::buckets_reorg::Bucket;
use crate::discovery::{ModelId, WorkerMode};
use crate::server::app_context::{AppContext, ChatRouting};
use crate::server::histogram::Histogram;
use crate::server::metrics::escape_label;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::workers::{MembershipEvent, WorkerRegistry};
use dashmap::DashMap;

/// How often a waiting request re-checks engine admission: engine load
/// reports change without notifying the queue. Arbitrary; well under a band.
const ENGINE_POLL: Duration = Duration::from_millis(20);

/// Per-node constants of a bucket's queue, from `--bucket-config`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueSpec {
    /// Mean wait allowed in the queue, seconds.
    pub mean_wait_s: f64,
    /// One member's uncached-token rate for this bucket's request shape.
    pub rate_tokens_per_s: f64,
    /// One member's completion rate for this bucket's request shape.
    pub rate_requests_per_s: f64,
    /// A waiting request is refused after `deadline_factor x mean_wait_s`
    /// (about the ratio of a queue wait's p90 to its mean).
    #[serde(default = "default_deadline_factor")]
    pub deadline_factor: f64,
    /// Upper bound of `Retry-After` on a full bucket.
    #[serde(default = "default_retry_after_max_s")]
    pub retry_after_max_s: u64,
    /// `Retry-After` once a waiting request passes its deadline.
    #[serde(default = "default_deadline_retry_after_s")]
    pub deadline_retry_after_s: u64,
    /// `Retry-After` when the wait ended on the decode side.
    #[serde(default = "default_decode_retry_after_s")]
    pub decode_retry_after_s: u64,
    /// Where a member's uncached-token rate comes from: the engine's reported
    /// prefill throughput (`measured`, the default; `rate_tokens_per_s` until a
    /// member reports) or always `rate_tokens_per_s` (`static`).
    #[serde(default)]
    pub rate_source: RateSource,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateSource {
    #[default]
    Measured,
    Static,
}

/// A measured rate counts within this factor of the configured one, so one
/// odd sample cannot empty or flood a bucket. Arbitrary bound.
const MEASURED_RATE_BOUNDS: (f64, f64) = (0.25, 4.0);
/// Weight of the newest throughput sample in a member's smoothed rate.
const RATE_SMOOTHING: f64 = 0.3;

fn default_deadline_factor() -> f64 {
    2.3
}
fn default_retry_after_max_s() -> u64 {
    30
}
fn default_deadline_retry_after_s() -> u64 {
    2
}
fn default_decode_retry_after_s() -> u64 {
    5
}

impl QueueSpec {
    pub fn validate(&self) -> Result<(), String> {
        let rate = |value: f64| value.is_finite() && value >= 0.0;
        if !(rate(self.mean_wait_s)
            && rate(self.rate_tokens_per_s)
            && rate(self.rate_requests_per_s))
        {
            return Err("queue mean_wait_s and rates must be finite and non-negative".into());
        }
        if !(self.deadline_factor.is_finite() && self.deadline_factor > 0.0) {
            return Err("queue deadline_factor must be finite and positive".into());
        }
        if self.retry_after_max_s == 0
            || self.deadline_retry_after_s == 0
            || self.decode_retry_after_s == 0
        {
            return Err("queue Retry-After values must be at least 1 s".into());
        }
        Ok(())
    }
}

/// Why a bucket refused a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The bucket's outstanding work would pass its depth.
    Full,
    /// Admitted, but no prefill engine took the request before the deadline.
    Deadline,
    /// Admitted, but no decode engine took the request before the deadline.
    DecodePermits,
}

impl RejectReason {
    /// The `reason` label of the rejected-requests counter.
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "bucket_full",
            Self::Deadline => "deadline",
            Self::DecodePermits => "decode_permits",
        }
    }

    /// The error `type` and `x-router-error-code` of the 429.
    pub fn error_type(self) -> &'static str {
        match self {
            Self::Full => "bucket_full",
            Self::Deadline => "bucket_deadline",
            Self::DecodePermits => "decode_permits",
        }
    }
}

/// A refusal, with what the client needs to back off by the bucket's arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketRejection {
    pub bucket: String,
    pub reason: RejectReason,
    pub retry_after_s: u64,
    pub depth_tokens: u64,
    pub queued_tokens: u64,
    pub uncached_tokens: u64,
    pub waited: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Depth {
    pub tokens: u64,
    pub requests: u64,
}

#[derive(Debug, Default)]
struct QueueState {
    members: u64,
    /// Sum of the members' uncached-token rates the depth scales with.
    rate_tokens_per_s: f64,
    tokens: u64,
    requests: u64,
}

/// Seconds from arrival to an engine pick.
static WAIT_BOUNDS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 30.0,
];
static RETRY_AFTER_BOUNDS: [f64; 8] = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0];

#[derive(Debug)]
pub struct QueueStats {
    pub admitted: AtomicU64,
    /// Indexed by [`RejectReason`] in declaration order.
    pub rejected: [AtomicU64; 3],
    pub wait_seconds: Histogram,
    pub retry_after_seconds: Histogram,
}

impl Default for QueueStats {
    fn default() -> Self {
        Self {
            admitted: AtomicU64::new(0),
            rejected: Default::default(),
            wait_seconds: Histogram::new(&WAIT_BOUNDS),
            retry_after_seconds: Histogram::new(&RETRY_AFTER_BOUNDS),
        }
    }
}

#[derive(Debug)]
pub struct BucketQueue {
    pub id: String,
    pub spec: QueueSpec,
    pub stats: QueueStats,
    state: Mutex<QueueState>,
    changed: Notify,
}

impl BucketQueue {
    pub fn new(id: impl Into<String>, spec: QueueSpec) -> Self {
        Self {
            id: id.into(),
            spec,
            stats: QueueStats::default(),
            state: Mutex::new(QueueState::default()),
            changed: Notify::new(),
        }
    }

    /// The members' request rate: the configured per-member rate, scaled by
    /// how their summed token rate compares with the configured one.
    fn request_rate(&self, state: &QueueState) -> f64 {
        let configured = state.members as f64 * self.spec.rate_tokens_per_s;
        if configured > 0.0 {
            state.members as f64
                * self.spec.rate_requests_per_s
                * (state.rate_tokens_per_s / configured)
        } else {
            state.members as f64 * self.spec.rate_requests_per_s
        }
    }

    fn depth_for(&self, state: &QueueState) -> Depth {
        let scaled = |rate: f64| (rate * self.spec.mean_wait_s).floor() as u64;
        Depth {
            tokens: scaled(state.rate_tokens_per_s),
            requests: scaled(self.request_rate(state)),
        }
    }

    pub fn depth(&self) -> Depth {
        self.depth_for(&self.state.lock())
    }

    /// The summed member rate the depth scales with, uncached tokens per second.
    pub fn rate_tokens_per_s(&self) -> f64 {
        self.state.lock().rate_tokens_per_s
    }

    pub fn members(&self) -> u64 {
        self.state.lock().members
    }

    /// Outstanding work as (uncached tokens, requests).
    pub fn outstanding(&self) -> (u64, u64) {
        let state = self.state.lock();
        (state.tokens, state.requests)
    }

    /// Set `members` ready engines at the configured per-member rate.
    pub fn set_members(&self, members: u64) -> bool {
        self.set_capacity(members, members as f64 * self.spec.rate_tokens_per_s)
    }

    /// Set the ready-member count and their summed uncached-token rate the
    /// depth scales with. Work already admitted keeps its place; only new
    /// arrivals see the new depth. True when the depth or members changed.
    pub fn set_capacity(&self, members: u64, rate_tokens_per_s: f64) -> bool {
        let rate_tokens_per_s = if rate_tokens_per_s.is_finite() {
            rate_tokens_per_s.max(0.0)
        } else {
            0.0
        };
        let grew = {
            let mut state = self.state.lock();
            let before = (state.members, self.depth_for(&state));
            state.members = members;
            state.rate_tokens_per_s = rate_tokens_per_s;
            let after = (members, self.depth_for(&state));
            if before == after {
                return false;
            }
            after.0 > before.0 || after.1.tokens > before.1.tokens
        };
        if grew {
            self.changed.notify_waiters();
        }
        true
    }

    /// One member's uncached-token rate for the depth: its smoothed measured
    /// rate within bounds of the configured one, or the configured one.
    pub fn member_rate(&self, measured: Option<f64>) -> f64 {
        let configured = self.spec.rate_tokens_per_s;
        match (self.spec.rate_source, measured) {
            (RateSource::Measured, Some(rate)) if configured > 0.0 => rate.clamp(
                configured * MEASURED_RATE_BOUNDS.0,
                configured * MEASURED_RATE_BOUNDS.1,
            ),
            (RateSource::Measured, Some(rate)) => rate,
            _ => configured,
        }
    }

    /// How long an admitted request may wait for an engine.
    pub fn deadline(&self) -> Duration {
        Duration::try_from_secs_f64(self.spec.mean_wait_s * self.spec.deadline_factor)
            .unwrap_or(Duration::MAX)
    }

    /// Admit `tokens` of uncached prefill work, or refuse at once with the
    /// time the bucket needs to drain the excess.
    pub fn try_enter(self: &Arc<Self>, tokens: u64) -> Result<BucketSlot, BucketRejection> {
        let mut state = self.state.lock();
        let depth = self.depth_for(&state);
        let runner_free = state.requests < state.members;
        let within_depth = state.requests < depth.requests.max(state.members)
            && state.tokens.saturating_add(tokens) <= depth.tokens;
        if runner_free || within_depth {
            state.tokens = state.tokens.saturating_add(tokens);
            state.requests += 1;
            self.stats.admitted.fetch_add(1, Ordering::Relaxed);
            return Ok(BucketSlot {
                queue: Arc::clone(self),
                tokens,
            });
        }
        let retry_after_s = self.drain_seconds(&state, depth, tokens);
        self.count_rejection(RejectReason::Full, retry_after_s);
        Err(BucketRejection {
            bucket: self.id.clone(),
            reason: RejectReason::Full,
            retry_after_s,
            depth_tokens: depth.tokens,
            queued_tokens: state.tokens,
            uncached_tokens: tokens,
            waited: Duration::ZERO,
        })
    }

    /// Seconds until the excess over the depth drains at the members' rate,
    /// in tokens or requests, whichever is longer; at least 1, at most the cap.
    fn drain_seconds(&self, state: &QueueState, depth: Depth, tokens: u64) -> u64 {
        let max = self.spec.retry_after_max_s;
        let over = |excess: u64, rate: f64| match excess {
            0 => 0.0,
            excess if rate > 0.0 => excess as f64 / rate,
            _ => f64::INFINITY,
        };
        let tokens_over = state
            .tokens
            .saturating_add(tokens)
            .saturating_sub(depth.tokens);
        let requests_over = (state.requests + 1).saturating_sub(depth.requests.max(state.members));
        let seconds = over(tokens_over, state.rate_tokens_per_s)
            .max(over(requests_over, self.request_rate(state)))
            .ceil();
        if seconds.is_finite() {
            (seconds as u64).clamp(1, max)
        } else {
            max
        }
    }

    /// A refusal of an admitted request whose wait for an engine ended.
    pub fn expired(&self, reason: RejectReason, tokens: u64, waited: Duration) -> BucketRejection {
        let state = self.state.lock();
        let retry_after_s = match reason {
            RejectReason::DecodePermits => self.spec.decode_retry_after_s,
            _ => self.spec.deadline_retry_after_s,
        };
        self.count_rejection(reason, retry_after_s);
        BucketRejection {
            bucket: self.id.clone(),
            reason,
            retry_after_s,
            depth_tokens: self.depth_for(&state).tokens,
            queued_tokens: state.tokens,
            uncached_tokens: tokens,
            waited,
        }
    }

    fn count_rejection(&self, reason: RejectReason, retry_after_s: u64) {
        self.stats.rejected[reason as usize].fetch_add(1, Ordering::Relaxed);
        self.stats.retry_after_seconds.observe(retry_after_s as f64);
    }

    /// Record an admitted request's wait from arrival to its engine pick.
    pub fn record_wait(&self, waited: Duration) {
        self.stats.wait_seconds.observe(waited.as_secs_f64());
    }

    /// Wait until work leaves the bucket, members join, or one engine poll
    /// interval passes, never past `until`.
    pub async fn wait_for_change(&self, until: Instant) {
        let notified = self.changed.notified();
        let tick = until.min(Instant::now() + ENGINE_POLL);
        tokio::select! {
            () = notified => {}
            () = tokio::time::sleep_until(tick) => {}
        }
    }

    fn release(&self, tokens: u64) {
        {
            let mut state = self.state.lock();
            state.tokens = state.tokens.saturating_sub(tokens);
            state.requests = state.requests.saturating_sub(1);
        }
        self.changed.notify_waiters();
    }
}

/// Smoothed uncached-token rate per engine URL, from each engine's reported
/// prefill throughput (uncached tokens per busy second between two reports).
#[derive(Debug, Default)]
pub struct PrefillRates {
    by_url: DashMap<String, f64>,
}

impl PrefillRates {
    /// Fold a throughput sample into the engine's rate; `None` (no prefill
    /// between its last two reports) keeps the rate it had.
    pub fn observe(&self, url: &str, sample: Option<f64>) {
        let Some(sample) = sample.filter(|rate| rate.is_finite() && *rate > 0.0) else {
            return;
        };
        self.by_url
            .entry(url.to_owned())
            .and_modify(|rate| *rate += RATE_SMOOTHING * (sample - *rate))
            .or_insert(sample);
    }

    pub fn rate(&self, url: &str) -> Option<f64> {
        self.by_url.get(url).map(|rate| *rate)
    }

    /// Engine URLs and rates, sorted, for metrics.
    pub fn rates(&self) -> Vec<(String, f64)> {
        let mut rates: Vec<_> = (self.by_url.iter())
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();
        rates.sort_by(|a, b| a.0.cmp(&b.0));
        rates
    }

    /// Drop engines that left the registry.
    pub fn retain(&self, live: &std::collections::HashSet<String>) {
        self.by_url.retain(|url, _| live.contains(url));
    }
}

/// Size `queue` from `bucket`'s ready engines and their rates; true if its
/// depth or members changed.
pub fn size_queue(
    bucket: &Bucket,
    queue: &BucketQueue,
    registry: &WorkerRegistry,
    model: &ModelId,
    rates: &PrefillRates,
) -> bool {
    let engines = bucket.ready_prefill_engines(registry, model);
    let rate = engines
        .iter()
        .map(|engine| queue.member_rate(rates.rate(&engine.url)))
        .sum();
    queue.set_capacity(engines.len() as u64, rate)
}

/// Fold the engines' latest prefill throughput into the resolvers' rates (when
/// `load` is given), then size every queued bucket; true if any changed.
pub fn refresh_members(
    routing: &ChatRouting,
    registry: &WorkerRegistry,
    load: Option<&EngineReportedLoadTable>,
) -> bool {
    let ChatRouting::Reorg(resolvers) = routing else {
        return false;
    };
    let snapshot = load.map(|load| load.capture_snapshot(std::time::Instant::now()));
    let mut changed = false;
    for (model, resolver) in resolvers {
        if let Some(snapshot) = &snapshot {
            let engines = registry.workers_for(model);
            for engine in &engines {
                let sample = snapshot
                    .fresh_native_cache_load_for_url(&engine.url)
                    .and_then(|load| load.prefill_throughput_tokens_per_s);
                resolver.prefill_rates.observe(&engine.url, sample);
            }
            let live = engines.iter().map(|engine| engine.url.clone()).collect();
            resolver.prefill_rates.retain(&live);
        }
        for bucket in &resolver.buckets {
            if let Some(queue) = &bucket.queue {
                if size_queue(bucket, queue, registry, model, &resolver.prefill_rates) {
                    let depth = queue.depth();
                    tracing::info!(
                        bucket = %bucket.id,
                        members = queue.members(),
                        rate_tokens_per_s = queue.rate_tokens_per_s(),
                        depth_tokens = depth.tokens,
                        depth_requests = depth.requests,
                        "bucket depth recomputed"
                    );
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Readiness flips without a discovery event (health checks, breakers), so
/// the watch also re-reads membership on this period.
const MEMBERSHIP_RECHECK: Duration = Duration::from_secs(1);

/// Recompute bucket depths whenever discovery adds, removes, relabels or
/// re-roles a worker, and every [`MEMBERSHIP_RECHECK`]. `None` when no bucket
/// has a queue.
pub fn spawn_membership_watch(ctx: Arc<AppContext>) -> Option<tokio::task::JoinHandle<()>> {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        return None;
    };
    let queued = resolvers
        .values()
        .any(|resolver| resolver.buckets.iter().any(|bucket| bucket.queue.is_some()));
    queued.then(|| {
        tokio::spawn(async move {
            loop {
                let changed = ctx.registry.membership.changed();
                refresh_members(
                    &ctx.chat_routing,
                    &ctx.registry,
                    Some(&ctx.engine_reported_load),
                );
                tokio::select! {
                    () = changed => {}
                    () = tokio::time::sleep(MEMBERSHIP_RECHECK) => {}
                }
            }
        })
    })
}

const REASONS: [RejectReason; 3] = [
    RejectReason::Full,
    RejectReason::Deadline,
    RejectReason::DecodePermits,
];

/// The bucket, decode-permit, session-estimate and membership series, pulled
/// on scrape. Empty under legacy routing.
pub fn render_metrics(ctx: &AppContext) -> String {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        return String::new();
    };
    let mut out = String::new();
    let queues: Vec<_> = resolvers
        .values()
        .flat_map(|resolver| resolver.buckets.iter())
        .filter_map(|bucket| bucket.queue.as_ref())
        .collect();
    let mut family =
        |name: &str, kind: &str, help: &str, rows: &mut dyn Iterator<Item = String>| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
            for row in rows {
                out.push_str(&row);
            }
        };
    type Gauge = fn(&BucketQueue) -> u64;
    let gauges: [(&str, &str, Gauge); 5] = [
        ("sgl_router_bucket_queue_tokens", "Uncached prompt tokens admitted to the bucket and not yet through prefill.", |q| q.outstanding().0),
        ("sgl_router_bucket_queue_requests", "Requests admitted to the bucket and not yet through prefill.", |q| q.outstanding().1),
        ("sgl_router_bucket_depth_tokens", "Most uncached tokens the bucket holds before answering 429: members x rate x mean wait.", |q| q.depth().tokens),
        ("sgl_router_bucket_depth_requests", "Most requests the bucket holds before answering 429 (each member may also run one).", |q| q.depth().requests),
        ("sgl_router_bucket_members", "Ready engines the bucket's depth scales with.", BucketQueue::members),
    ];
    for (name, help, value) in gauges {
        family(
            name,
            "gauge",
            help,
            &mut queues.iter().map(|q| {
                format!(
                    "{name}{{bucket=\"{}\"}} {}\n",
                    escape_label(&q.id),
                    value(q)
                )
            }),
        );
    }
    family(
        "sgl_router_bucket_admitted_total",
        "counter",
        "Requests admitted to the bucket's queue.",
        &mut queues.iter().map(|q| {
            format!(
                "sgl_router_bucket_admitted_total{{bucket=\"{}\"}} {}\n",
                escape_label(&q.id),
                q.stats.admitted.load(Ordering::Relaxed)
            )
        }),
    );
    family(
        "sgl_router_bucket_rejected_total",
        "counter",
        "Requests the bucket answered 429, by reason.",
        &mut queues.iter().flat_map(|q| {
            REASONS.iter().map(move |reason| {
                format!(
                    "sgl_router_bucket_rejected_total{{bucket=\"{}\",reason=\"{}\"}} {}\n",
                    escape_label(&q.id),
                    reason.label(),
                    q.stats.rejected[*reason as usize].load(Ordering::Relaxed)
                )
            })
        }),
    );
    let spills: Vec<_> = resolvers
        .values()
        .flat_map(|resolver| resolver.buckets.iter())
        .flat_map(|bucket| bucket.spill_to.iter().map(move |target| (bucket, target)))
        .collect();
    family(
        "sgl_router_bucket_spill_total",
        "counter",
        "Requests a bucket placed on an idle engine of a longer bucket.",
        &mut spills.iter().map(|(bucket, target)| {
            format!(
                "sgl_router_bucket_spill_total{{from=\"{}\",to=\"{}\"}} {}\n",
                escape_label(&bucket.id),
                escape_label(&target.bucket),
                target.spilled.load(Ordering::Relaxed)
            )
        }),
    );
    family(
        "sgl_router_bucket_rate_tokens_per_second",
        "gauge",
        "Summed uncached-token rate of the bucket's ready engines that its depth scales with.",
        &mut queues.iter().map(|q| {
            format!(
                "sgl_router_bucket_rate_tokens_per_second{{bucket=\"{}\"}} {}\n",
                escape_label(&q.id),
                q.rate_tokens_per_s()
            )
        }),
    );
    family(
        "sgl_router_prefill_rate_tokens_per_second",
        "gauge",
        "Smoothed uncached-token rate each engine reports for its prefill (tokens per busy second).",
        &mut resolvers.values().flat_map(|resolver| resolver.prefill_rates.rates()).map(|(url, rate)| {
            format!(
                "sgl_router_prefill_rate_tokens_per_second{{worker_url=\"{}\"}} {rate}\n",
                escape_label(&url)
            )
        }),
    );
    let (permits, running) = decode_permits(ctx);
    family(
        "sgl_router_decode_permits",
        "gauge",
        "Sum over ready decode engines of their bucket group's max_inflight_requests; -1 when a group sets none.",
        &mut std::iter::once(format!("sgl_router_decode_permits {permits}\n")),
    );
    family(
        "sgl_router_decode_running",
        "gauge",
        "Requests this router has in flight on each decode engine.",
        &mut running.into_iter().map(|(worker, count)| {
            format!(
                "sgl_router_decode_running{{worker=\"{}\"}} {count}\n",
                escape_label(&worker)
            )
        }),
    );
    family(
        "sgl_router_membership_events_total",
        "counter",
        "Discovery events applied to the worker registry, by kind.",
        &mut MembershipEvent::ALL.iter().map(|event| {
            format!(
                "sgl_router_membership_events_total{{kind=\"{}\"}} {}\n",
                event.label(),
                ctx.registry.membership.count(*event)
            )
        }),
    );
    for (name, help, histogram) in [
        (
            "sgl_router_bucket_queue_wait_seconds",
            "Seconds from arrival to an engine pick, for requests a bucket admitted.",
            (|q: &BucketQueue| &q.stats.wait_seconds) as fn(&BucketQueue) -> &Histogram,
        ),
        (
            "sgl_router_retry_after_seconds",
            "Retry-After seconds sent with the bucket's 429s.",
            |q: &BucketQueue| &q.stats.retry_after_seconds,
        ),
    ] {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        for q in &queues {
            histogram(q).render(
                &mut out,
                name,
                &format!("bucket=\"{}\"", escape_label(&q.id)),
            );
        }
    }
    for resolver in resolvers.values() {
        if let Some(estimator) = &resolver.estimator {
            let name = "sgl_router_session_uncached_estimate_error";
            let _ = writeln!(
                out,
                "# HELP {name} Router uncached estimate minus the engine's reported uncached tokens (prompt minus cached), per completed turn.\n# TYPE {name} histogram"
            );
            estimator.estimate_error.render(&mut out, name, "");
        }
    }
    out
}

/// Decode permits over ready decode engines of PD buckets, each engine
/// counted once with the first bucket's cap, and the router's in-flight count
/// per decode engine.
fn decode_permits(ctx: &AppContext) -> (i64, Vec<(String, usize)>) {
    let ChatRouting::Reorg(resolvers) = &ctx.chat_routing else {
        return (0, Vec::new());
    };
    let mut seen = std::collections::HashSet::new();
    let mut permits = 0i64;
    for (model, resolver) in resolvers {
        for bucket in &resolver.buckets {
            for engine in bucket.ready_decode_members(&ctx.registry, model) {
                if seen.insert(engine.id.clone()) {
                    permits = match (permits, bucket.decode_permits_per_engine) {
                        (-1, _) | (_, None) => -1,
                        (sum, Some(cap)) => sum.saturating_add(cap as i64),
                    };
                }
            }
        }
    }
    let mut running: Vec<_> = ctx
        .registry
        .all()
        .into_iter()
        .filter(|worker| worker.mode() == WorkerMode::Decode)
        .map(|worker| (worker.id.0.clone(), worker.router_inflight_load()))
        .collect();
    running.sort();
    (permits, running)
}

/// Admitted work in a bucket; released when dropped, which the router does
/// once the request's prefill has finished (or the request ended earlier).
#[derive(Debug)]
pub struct BucketSlot {
    queue: Arc<BucketQueue>,
    tokens: u64,
}

impl BucketSlot {
    pub fn queue(&self) -> &Arc<BucketQueue> {
        &self.queue
    }
}

impl Drop for BucketSlot {
    fn drop(&mut self) {
        self.queue.release(self.tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(mean_wait_s: f64, rate_tokens_per_s: f64, rate_requests_per_s: f64) -> QueueSpec {
        serde_json::from_value(serde_json::json!({
            "mean_wait_s": mean_wait_s,
            "rate_tokens_per_s": rate_tokens_per_s,
            "rate_requests_per_s": rate_requests_per_s,
        }))
        .unwrap()
    }

    fn queue(members: u64, spec: QueueSpec) -> Arc<BucketQueue> {
        let queue = Arc::new(BucketQueue::new("S", spec));
        queue.set_members(members);
        queue
    }

    #[test]
    fn depth_is_members_times_rate_times_mean_wait() {
        let queue = queue(2, spec(4.0, 20_000.0, 3.0));
        assert_eq!(
            queue.depth(),
            Depth {
                tokens: 160_000,
                requests: 24
            }
        );
        assert_eq!(queue.deadline(), Duration::from_secs_f64(4.0 * 2.3));
    }

    #[test]
    fn work_over_the_token_depth_is_refused_with_its_drain_time() {
        // 2 members x 1,000 tokens/s x 10 s = 20,000 tokens, 20 requests.
        let queue = queue(2, spec(10.0, 1_000.0, 1.0));
        let held: Vec<_> = (0..3).map(|_| queue.try_enter(6_000).unwrap()).collect();
        assert_eq!(queue.outstanding(), (18_000, 3));
        let refused = queue.try_enter(7_000).unwrap_err();
        // 25,000 - 20,000 = 5,000 tokens over, drained at 2,000 tokens/s.
        assert_eq!(
            (
                refused.reason,
                refused.retry_after_s,
                refused.depth_tokens,
                refused.queued_tokens
            ),
            (RejectReason::Full, 3, 20_000, 18_000)
        );
        // A smaller request still fits.
        let fits = queue.try_enter(2_000).unwrap();
        drop(held);
        assert_eq!(queue.outstanding(), (2_000, 1));
        drop(fits);
        assert_eq!(queue.outstanding(), (0, 0));
    }

    #[test]
    fn work_over_the_request_depth_is_refused() {
        // 1 member x 2 requests/s x 1 s = 2 requests.
        let queue = queue(1, spec(1.0, 1e9, 2.0));
        let _held: Vec<_> = (0..2).map(|_| queue.try_enter(1).unwrap()).collect();
        let refused = queue.try_enter(1).unwrap_err();
        assert_eq!(
            (refused.reason, refused.retry_after_s),
            (RejectReason::Full, 1)
        );
    }

    #[test]
    fn a_zero_depth_bucket_runs_one_request_per_member_and_queues_none() {
        let queue = queue(2, spec(0.0, 10_000.0, 0.05));
        assert_eq!(queue.depth(), Depth::default());
        let first = queue.try_enter(900_000).unwrap();
        let _second = queue.try_enter(900_000).unwrap();
        let refused = queue.try_enter(300_000).unwrap_err();
        assert_eq!(refused.reason, RejectReason::Full);
        assert_eq!(refused.retry_after_s, 30, "capped");
        drop(first);
        assert!(queue.try_enter(300_000).is_ok());
    }

    #[test]
    fn a_bucket_without_members_refuses_with_the_longest_retry_after() {
        let queue = queue(0, spec(4.0, 20_000.0, 3.0));
        let refused = queue.try_enter(1).unwrap_err();
        assert_eq!(
            (refused.reason, refused.retry_after_s),
            (RejectReason::Full, 30)
        );
    }

    #[test]
    fn fewer_members_lower_the_depth_for_new_arrivals_only() {
        let queue = queue(2, spec(10.0, 1_000.0, 1.0));
        let held = queue.try_enter(15_000).unwrap();
        assert!(queue.set_members(1));
        assert!(!queue.set_members(1), "no change");
        assert_eq!(queue.depth().tokens, 10_000);
        assert_eq!(
            queue.outstanding(),
            (15_000, 1),
            "admitted work keeps its place"
        );
        assert!(queue.try_enter(1).is_err());
        drop(held);
        assert!(queue.try_enter(1).is_ok());
    }

    #[test]
    fn an_expired_wait_names_its_side_and_retry_after() {
        let queue = queue(1, spec(1.0, 1.0, 1.0));
        let deadline = queue.expired(RejectReason::Deadline, 5, Duration::from_secs(2));
        assert_eq!((deadline.retry_after_s, deadline.uncached_tokens), (2, 5));
        let decode = queue.expired(RejectReason::DecodePermits, 5, Duration::from_secs(2));
        assert_eq!(decode.retry_after_s, 5);
        assert_eq!(RejectReason::Deadline.label(), "deadline");
        assert_eq!(RejectReason::Deadline.error_type(), "bucket_deadline");
    }

    #[tokio::test(start_paused = true)]
    async fn a_release_wakes_a_waiting_request_before_its_poll() {
        let queue = queue(1, spec(1.0, 1.0, 1.0));
        let held = queue.try_enter(1).unwrap();
        let waiter = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move {
                let start = Instant::now();
                queue.wait_for_change(start + Duration::from_secs(10)).await;
                start.elapsed()
            })
        };
        tokio::task::yield_now().await;
        drop(held);
        assert!(waiter.await.unwrap() < ENGINE_POLL);
    }

    #[test]
    fn refreshing_members_follows_the_registry_per_bucket() {
        use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        use crate::policies_reorg::power_of_two::PowerOfTwoPolicy;
        use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

        let registry = WorkerRegistry::default();
        let add = |id: &str| {
            registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:1"),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("m".into())],
                    ..Default::default()
                })
                .unwrap()
        };
        add("a");
        add("b");
        let group = |ids: &[&str]| {
            let mut group = EngineGroup::new(Arc::new(PowerOfTwoPolicy::new(
                EngineReportedLoadTable::new(),
            )));
            group.worker_ids = Some(ids.iter().map(|id| WorkerId((*id).into())).collect());
            group
        };
        let mut short = Bucket::new("S", BucketGroups::Plain(group(&["a", "b"])));
        short.queue = Some(Arc::new(BucketQueue::new("S", spec(1.0, 100.0, 1.0))));
        let mut long = Bucket::new("L", BucketGroups::Plain(group(&["c"])));
        long.queue = Some(Arc::new(BucketQueue::new("L", spec(1.0, 100.0, 1.0))));
        let routing = ChatRouting::Reorg(
            [(
                ModelId("m".into()),
                BucketResolver::new(vec![short, long]).unwrap(),
            )]
            .into(),
        );
        let queue = |index: usize| {
            let ChatRouting::Reorg(resolvers) = &routing else {
                unreachable!()
            };
            Arc::clone(
                resolvers[&ModelId("m".into())].buckets[index]
                    .queue
                    .as_ref()
                    .unwrap(),
            )
        };
        assert!(refresh_members(&routing, &registry, None));
        assert_eq!((queue(0).members(), queue(1).members()), (2, 0));
        assert_eq!(queue(0).depth().tokens, 200);
        assert!(!refresh_members(&routing, &registry, None), "nothing moved");
        registry.remove(&WorkerId("b".into()));
        add("c");
        assert!(refresh_members(&routing, &registry, None));
        assert_eq!((queue(0).members(), queue(1).members()), (1, 1));
        assert_eq!(queue(0).depth().tokens, 100);
        assert!(!refresh_members(&ChatRouting::Legacy, &registry, None));
    }

    #[test]
    fn depth_scales_with_the_members_summed_rate() {
        // Configured: 1,000 tokens/s and 1 request/s per member, 10 s mean wait.
        let queue = queue(0, spec(10.0, 1_000.0, 1.0));
        assert!(queue.set_capacity(2, 5_000.0));
        assert_eq!(
            queue.depth(),
            Depth {
                tokens: 50_000,
                requests: 50
            },
            "2.5x the configured token rate carries the request rate with it"
        );
        let _held = queue.try_enter(50_000).unwrap();
        let _held2 = queue.try_enter(1).unwrap();
        // 50,001 + 6,000 - 50,000 tokens over the measured 5,000 tokens/s.
        assert_eq!(queue.try_enter(6_000).unwrap_err().retry_after_s, 2);
        assert!(!queue.set_capacity(2, 5_000.0), "no change");
    }

    #[test]
    fn a_member_rate_is_measured_within_bounds_or_configured() {
        let measured = queue(1, spec(1.0, 1_000.0, 1.0));
        assert_eq!(measured.member_rate(Some(3_000.0)), 3_000.0);
        assert_eq!(measured.member_rate(Some(90_000.0)), 4_000.0);
        assert_eq!(measured.member_rate(Some(10.0)), 250.0);
        assert_eq!(measured.member_rate(None), 1_000.0);
        let mut fixed = spec(1.0, 1_000.0, 1.0);
        fixed.rate_source = RateSource::Static;
        assert_eq!(queue(1, fixed).member_rate(Some(3_000.0)), 1_000.0);
        let parsed: QueueSpec = serde_json::from_value(serde_json::json!({
            "mean_wait_s": 1, "rate_tokens_per_s": 1, "rate_requests_per_s": 1, "rate_source": "static"
        }))
        .unwrap();
        assert_eq!(parsed.rate_source, RateSource::Static);
    }

    #[test]
    fn reported_prefill_throughput_sets_the_bucket_depth() {
        use crate::buckets_reorg::{BucketGroups, BucketResolver, EngineGroup};
        use crate::discovery::{WorkerId, WorkerSpec};
        use crate::policies_reorg::power_of_two::PowerOfTwoPolicy;
        use crate::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};

        let registry = WorkerRegistry::default();
        for id in ["a", "b"] {
            registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:1"),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("m".into())],
                    ..Default::default()
                })
                .unwrap();
        }
        let load = EngineReportedLoadTable::new();
        let report = |url: &str, tokens: u64, busy_us: u64| {
            load.set(
                url,
                0,
                LoadStat {
                    num_running_reqs: 0,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 1_000,
                    native_cache: Some(NativeCacheRankLoad {
                        num_waiting_uncached_tokens: 0,
                        num_total_tokens: 0,
                        max_running_requests: 64,
                        total_prefill_uncached_tokens: tokens,
                        total_prefill_busy_us: busy_us,
                    }),
                },
                std::time::Instant::now(),
            )
        };
        let mut group = EngineGroup::new(Arc::new(PowerOfTwoPolicy::new(Arc::clone(&load))));
        group.worker_ids = Some([WorkerId("a".into()), WorkerId("b".into())].into());
        let mut bucket = Bucket::new("S", BucketGroups::Plain(group));
        bucket.queue = Some(Arc::new(BucketQueue::new("S", spec(2.0, 10_000.0, 1.0))));
        let routing = ChatRouting::Reorg(
            [(
                ModelId("m".into()),
                BucketResolver::new(vec![bucket]).unwrap(),
            )]
            .into(),
        );
        let ChatRouting::Reorg(resolvers) = &routing else {
            unreachable!()
        };
        let queue = Arc::clone(
            resolvers[&ModelId("m".into())].buckets[0]
                .queue
                .as_ref()
                .unwrap(),
        );
        // Nothing reported yet: the configured rate per member.
        refresh_members(&routing, &registry, Some(&load));
        assert_eq!(queue.depth().tokens, 40_000);
        // a prefilled 15,000 tokens in one busy second; b reports nothing.
        report("http://a:1", 100_000, 10_000_000);
        report("http://a:1", 115_000, 11_000_000);
        assert!(refresh_members(&routing, &registry, Some(&load)));
        assert_eq!(queue.rate_tokens_per_s(), 25_000.0);
        assert_eq!(queue.depth().tokens, 50_000);
        // A rate is remembered when the engine idles between reports.
        report("http://a:1", 115_000, 11_000_000);
        refresh_members(&routing, &registry, Some(&load));
        assert_eq!(queue.depth().tokens, 50_000);
        let rates = resolvers[&ModelId("m".into())].prefill_rates.rates();
        assert_eq!(rates, [("http://a:1".to_string(), 15_000.0)]);
    }

    #[test]
    fn invalid_specs_are_refused() {
        assert!(spec(4.0, 20_000.0, 3.0).validate().is_ok());
        let mut not_a_number = spec(1.0, 1.0, 1.0);
        not_a_number.mean_wait_s = f64::NAN;
        assert!(not_a_number.validate().is_err());
        assert!(spec(1.0, -1.0, 1.0).validate().is_err());
        let mut zero_factor = spec(1.0, 1.0, 1.0);
        zero_factor.deadline_factor = 0.0;
        assert!(zero_factor.validate().is_err());
        assert!(serde_json::from_value::<QueueSpec>(
            serde_json::json!({"mean_wait_s": 1, "rate_tokens_per_s": 1, "rate_requests_per_s": 1, "depth": 3})
        )
        .is_err());
    }
}
