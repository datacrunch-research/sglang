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
}

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
    tokens: u64,
    requests: u64,
}

#[derive(Debug)]
pub struct BucketQueue {
    pub id: String,
    pub spec: QueueSpec,
    state: Mutex<QueueState>,
    changed: Notify,
}

impl BucketQueue {
    pub fn new(id: impl Into<String>, spec: QueueSpec) -> Self {
        Self {
            id: id.into(),
            spec,
            state: Mutex::new(QueueState::default()),
            changed: Notify::new(),
        }
    }

    fn depth_for(&self, members: u64) -> Depth {
        let scaled = |rate: f64| (members as f64 * rate * self.spec.mean_wait_s).floor() as u64;
        Depth {
            tokens: scaled(self.spec.rate_tokens_per_s),
            requests: scaled(self.spec.rate_requests_per_s),
        }
    }

    pub fn depth(&self) -> Depth {
        self.depth_for(self.state.lock().members)
    }

    pub fn members(&self) -> u64 {
        self.state.lock().members
    }

    /// Outstanding work as (uncached tokens, requests).
    pub fn outstanding(&self) -> (u64, u64) {
        let state = self.state.lock();
        (state.tokens, state.requests)
    }

    /// Set the ready-member count the depth scales with. Work already admitted
    /// keeps its place; only new arrivals see the new depth.
    pub fn set_members(&self, members: u64) -> bool {
        let grew = {
            let mut state = self.state.lock();
            if state.members == members {
                return false;
            }
            let grew = members > state.members;
            state.members = members;
            grew
        };
        if grew {
            self.changed.notify_waiters();
        }
        true
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
        let depth = self.depth_for(state.members);
        let runner_free = state.requests < state.members;
        let within_depth = state.requests < depth.requests.max(state.members)
            && state.tokens.saturating_add(tokens) <= depth.tokens;
        if runner_free || within_depth {
            state.tokens = state.tokens.saturating_add(tokens);
            state.requests += 1;
            return Ok(BucketSlot {
                queue: Arc::clone(self),
                tokens,
            });
        }
        let retry_after_s = self.drain_seconds(&state, depth, tokens);
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
        let members = state.members as f64;
        let over = |excess: u64, rate: f64| match excess {
            0 => 0.0,
            excess if members * rate > 0.0 => excess as f64 / (members * rate),
            _ => f64::INFINITY,
        };
        let tokens_over = state
            .tokens
            .saturating_add(tokens)
            .saturating_sub(depth.tokens);
        let requests_over = (state.requests + 1).saturating_sub(depth.requests.max(state.members));
        let seconds = over(tokens_over, self.spec.rate_tokens_per_s)
            .max(over(requests_over, self.spec.rate_requests_per_s))
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
        BucketRejection {
            bucket: self.id.clone(),
            reason,
            retry_after_s,
            depth_tokens: self.depth_for(state.members).tokens,
            queued_tokens: state.tokens,
            uncached_tokens: tokens,
            waited,
        }
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
