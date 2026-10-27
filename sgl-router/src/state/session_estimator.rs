// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Uncached prompt tokens per request, estimated before dispatch from the
//! session's previous turn.
//!
//! Agent conversations are append-only: a turn's prompt is the previous
//! prompt plus the previous answer plus the new message, so the tokens a
//! prefill engine must compute are about `prompt(now) - prompt(previous
//! turn)`. The estimator keeps, per session key, the last completed turn's
//! prompt length in the router's own count and corrects itself from what the
//! engine reported for that turn (`prompt_tokens`, `cached_tokens`):
//!
//! - an unknown, expired or shrunk session is cold: uncached = the prompt;
//! - a turn whose reported cached count came back under half of what the
//!   estimate expected marks the next turn cold (the cache lost the session's
//!   prefix: an eviction, or a move to an engine that never held it);
//! - the ratio of the engine's prompt count to the router's converts the
//!   router's count (exact when it tokenizes, a byte estimate otherwise) into
//!   engine tokens.
//!
//! Errors fall on the safe side: a short turn estimated long lands in a longer
//! bucket. A long turn is estimated short only when two conversations share a
//! key, which a header or the derived key below makes unlikely.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Keeps the map bounded when every request carries a fresh key.
const MAX_SESSIONS: usize = 1 << 20;
const SWEEP_EVERY: Duration = Duration::from_secs(60);
/// Engine tokens per router token, clamped so one odd report cannot run away.
const SCALE_BOUNDS: (f64, f64) = (0.25, 4.0);
/// Records older than this are cold; long enough to cover agent think times.
pub const DEFAULT_SESSION_IDLE: Duration = Duration::from_secs(7_200);

/// Where an estimate came from; exported as a label and in the request log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateSource {
    /// No session key: every request counts as cold.
    NoSession,
    /// First turn seen for the key, or its record expired.
    Cold,
    /// Prompt growth since the previous completed turn.
    Delta,
    /// The previous turn found the cache cold, so this one counts as cold.
    Reset,
    /// The prompt is shorter than the previous turn's: not an append.
    Shrunk,
}

impl EstimateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoSession => "no_session",
            Self::Cold => "cold",
            Self::Delta => "delta",
            Self::Reset => "reset",
            Self::Shrunk => "shrunk",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UncachedEstimate {
    /// Prompt tokens the prefill engine is expected to compute.
    pub uncached_tokens: u64,
    pub source: EstimateSource,
}

#[derive(Debug, Clone, Copy)]
struct Record {
    /// The router's prompt count for the last completed turn.
    prompt_tokens: u64,
    /// Engine prompt tokens per router prompt token.
    scale: f64,
    cold_next: bool,
    at: Instant,
}

#[derive(Debug)]
pub struct SessionEstimator {
    sessions: DashMap<String, Record>,
    idle: Duration,
    last_sweep: Mutex<Instant>,
}

impl SessionEstimator {
    pub fn new(idle: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            idle,
            last_sweep: Mutex::new(Instant::now()),
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// The uncached estimate for a turn of `key` whose prompt counts
    /// `prompt_tokens` in the router's count. Reads only; a turn is recorded
    /// when it completes ([`Self::observe`]), so a rejected or failed attempt
    /// never makes its retry look warm.
    pub fn estimate(&self, key: Option<&str>, prompt_tokens: u64) -> UncachedEstimate {
        let cold = |source, scale| UncachedEstimate {
            uncached_tokens: scaled(prompt_tokens, scale),
            source,
        };
        let Some(key) = key else {
            return cold(EstimateSource::NoSession, 1.0);
        };
        let Some(record) = self.sessions.get(key).map(|record| *record) else {
            return cold(EstimateSource::Cold, 1.0);
        };
        if record.at.elapsed() > self.idle {
            return cold(EstimateSource::Cold, record.scale);
        }
        if record.cold_next {
            return cold(EstimateSource::Reset, record.scale);
        }
        match prompt_tokens.checked_sub(record.prompt_tokens) {
            Some(delta) => UncachedEstimate {
                uncached_tokens: scaled(delta, record.scale),
                source: EstimateSource::Delta,
            },
            None => cold(EstimateSource::Shrunk, record.scale),
        }
    }

    /// Record a completed turn of `key`: the router's prompt count, the
    /// estimate it was routed on, and what the engine reported, when it did.
    pub fn observe(
        &self,
        key: &str,
        prompt_tokens: u64,
        estimate: UncachedEstimate,
        engine_prompt_tokens: Option<u64>,
        engine_cached_tokens: Option<u64>,
    ) {
        let previous = self.sessions.get(key).map(|record| record.scale);
        let scale = match engine_prompt_tokens {
            Some(engine) if engine > 0 && prompt_tokens > 0 => {
                (engine as f64 / prompt_tokens as f64).clamp(SCALE_BOUNDS.0, SCALE_BOUNDS.1)
            }
            _ => previous.unwrap_or(1.0),
        };
        // Only a delta estimate expects a hit; a cold one expects none.
        let cold_next = estimate.source == EstimateSource::Delta
            && engine_cached_tokens.is_some_and(|cached| {
                let prompt = engine_prompt_tokens.unwrap_or_else(|| scaled(prompt_tokens, scale));
                let expected = prompt.saturating_sub(estimate.uncached_tokens);
                expected > 0 && cached.saturating_mul(2) < expected
            });
        if previous.is_none() && self.sessions.len() >= MAX_SESSIONS {
            self.sweep(true);
            if self.sessions.len() >= MAX_SESSIONS {
                return;
            }
        }
        self.sessions.insert(
            key.to_owned(),
            Record {
                prompt_tokens,
                scale,
                cold_next,
                at: Instant::now(),
            },
        );
        self.sweep(false);
    }

    fn sweep(&self, force: bool) {
        {
            let mut last = self.last_sweep.lock();
            if !force && last.elapsed() < SWEEP_EVERY {
                return;
            }
            *last = Instant::now();
        }
        let idle = self.idle;
        self.sessions
            .retain(|_, record| record.at.elapsed() <= idle);
    }
}

/// A turn routed on an estimate, fed back to the estimator once the request
/// has finished and the engine's counts are known.
#[derive(Debug)]
pub struct PendingTurn {
    pub estimator: Arc<SessionEstimator>,
    pub key: String,
    pub prompt_tokens: u64,
    pub estimate: UncachedEstimate,
}

impl PendingTurn {
    pub fn complete(&self, engine_prompt_tokens: Option<u64>, engine_cached_tokens: Option<u64>) {
        self.estimator.observe(
            &self.key,
            self.prompt_tokens,
            self.estimate,
            engine_prompt_tokens,
            engine_cached_tokens,
        );
    }
}

fn scaled(tokens: u64, scale: f64) -> u64 {
    if scale == 1.0 {
        return tokens;
    }
    // f64 holds every realistic prompt length exactly; saturate the rest.
    let value = (tokens as f64 * scale).round();
    if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

/// The estimator key for a request: the session header's value when present,
/// else the key derived from the chat body. The two never collide.
pub fn session_key(header: Option<&str>, derived: Option<&str>) -> Option<String> {
    match (header, derived) {
        (Some(id), _) => Some(format!("h:{id}")),
        (None, Some(derived)) => Some(format!("d:{derived}")),
        (None, None) => None,
    }
}

/// A key for a chat without a session header: a hash of its first system (or
/// developer) message and its first user message, which every turn of one
/// conversation repeats. `None` when the body has no user message.
pub fn derived_session_key(body: &[u8]) -> Option<String> {
    #[derive(Deserialize)]
    struct Chat {
        #[serde(default)]
        messages: FirstMessages,
    }
    let chat: Chat = serde_json::from_slice(body).ok()?;
    let FirstMessages { system, user } = chat.messages;
    let user = user?;
    let mut hash = Sha256::new();
    hash.update(b"system\0");
    if let Some(system) = system {
        hash.update(system.to_string().as_bytes());
    }
    hash.update(b"\0user\0");
    hash.update(user.to_string().as_bytes());
    let digest = hash.finalize();
    Some(
        digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}

/// The first system and user contents of `messages`, skipping the rest
/// without building them.
#[derive(Default)]
struct FirstMessages {
    system: Option<Value>,
    user: Option<Value>,
}

impl<'de> Deserialize<'de> for FirstMessages {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Messages;
        #[derive(Deserialize)]
        struct Message {
            #[serde(default)]
            role: Option<String>,
            #[serde(default)]
            content: Value,
        }
        impl<'de> Visitor<'de> for Messages {
            type Value = FirstMessages;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a list of chat messages")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<FirstMessages, A::Error> {
                let mut first = FirstMessages::default();
                while first.user.is_none() {
                    let Some(message) = seq.next_element::<Message>()? else {
                        return Ok(first);
                    };
                    match message.role.as_deref() {
                        Some("system" | "developer") if first.system.is_none() => {
                            first.system = Some(message.content);
                        }
                        Some("user") => first.user = Some(message.content),
                        _ => {}
                    }
                }
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(first)
            }
        }
        deserializer.deserialize_seq(Messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimator() -> SessionEstimator {
        SessionEstimator::new(DEFAULT_SESSION_IDLE)
    }

    fn cold(tokens: u64) -> UncachedEstimate {
        UncachedEstimate {
            uncached_tokens: tokens,
            source: EstimateSource::Cold,
        }
    }

    fn delta(tokens: u64) -> UncachedEstimate {
        UncachedEstimate {
            uncached_tokens: tokens,
            source: EstimateSource::Delta,
        }
    }

    #[test]
    fn unknown_sessions_and_requests_without_a_key_are_cold() {
        let estimator = estimator();
        assert_eq!(
            estimator.estimate(None, 900),
            UncachedEstimate {
                uncached_tokens: 900,
                source: EstimateSource::NoSession
            }
        );
        assert_eq!(
            estimator.estimate(Some("h:a"), 900),
            UncachedEstimate {
                uncached_tokens: 900,
                source: EstimateSource::Cold
            }
        );
    }

    #[test]
    fn a_completed_turn_makes_the_next_one_its_prompt_growth() {
        let estimator = estimator();
        estimator.observe("h:a", 100_000, cold(100_000), Some(100_000), Some(0));
        assert_eq!(
            estimator.estimate(Some("h:a"), 103_000),
            UncachedEstimate {
                uncached_tokens: 3_000,
                source: EstimateSource::Delta
            }
        );
        // Another session is untouched.
        assert_eq!(
            estimator.estimate(Some("h:b"), 103_000).uncached_tokens,
            103_000
        );
    }

    #[test]
    fn a_shrunk_prompt_is_not_an_append_and_counts_as_cold() {
        let estimator = estimator();
        estimator.observe("h:a", 100_000, cold(100_000), None, None);
        let estimate = estimator.estimate(Some("h:a"), 40_000);
        assert_eq!(
            (estimate.uncached_tokens, estimate.source),
            (40_000, EstimateSource::Shrunk)
        );
    }

    #[test]
    fn a_turn_that_found_the_cache_cold_makes_the_next_turn_cold() {
        let estimator = estimator();
        estimator.observe("h:a", 100_000, cold(100_000), Some(100_000), Some(0));
        // Routed as a 2K turn, but the engine had only 10K of the 100K prefix.
        estimator.observe("h:a", 102_000, delta(2_000), Some(102_000), Some(10_000));
        let estimate = estimator.estimate(Some("h:a"), 104_000);
        assert_eq!(
            (estimate.uncached_tokens, estimate.source),
            (104_000, EstimateSource::Reset)
        );
        // A cold turn expects no hit, so it never flags the next one.
        estimator.observe("h:a", 104_000, cold(104_000), Some(104_000), Some(0));
        assert_eq!(
            estimator.estimate(Some("h:a"), 105_000).uncached_tokens,
            1_000
        );
        // A hit at or above half of the expected prefix keeps the delta.
        estimator.observe("h:a", 105_000, delta(1_000), Some(105_000), Some(52_000));
        assert_eq!(
            estimator.estimate(Some("h:a"), 106_000).uncached_tokens,
            1_000
        );
    }

    #[test]
    fn engine_counts_convert_the_router_count_into_engine_tokens() {
        let estimator = estimator();
        // A byte estimate of 80K for a prompt the engine counts as 100K.
        estimator.observe("h:a", 80_000, cold(80_000), Some(100_000), Some(0));
        assert_eq!(
            estimator.estimate(Some("h:a"), 84_000).uncached_tokens,
            5_000
        );
        // Without an engine count the previous ratio stays.
        estimator.observe("h:a", 84_000, delta(5_000), None, None);
        assert_eq!(
            estimator.estimate(Some("h:a"), 88_000).uncached_tokens,
            5_000
        );
        // One absurd report is clamped.
        estimator.observe("h:a", 88_000, delta(5_000), Some(10_000_000), None);
        assert_eq!(estimator.estimate(Some("h:a"), 88_001).uncached_tokens, 4);
    }

    #[test]
    fn records_expire_after_the_idle_time() {
        let estimator = SessionEstimator::new(Duration::ZERO);
        estimator.observe("h:a", 100_000, cold(100_000), None, None);
        std::thread::sleep(Duration::from_millis(2));
        let estimate = estimator.estimate(Some("h:a"), 101_000);
        assert_eq!(
            (estimate.uncached_tokens, estimate.source),
            (101_000, EstimateSource::Cold)
        );
    }

    #[test]
    fn header_and_derived_keys_live_apart() {
        assert_eq!(session_key(Some("x"), Some("y")).as_deref(), Some("h:x"));
        assert_eq!(session_key(None, Some("y")).as_deref(), Some("d:y"));
        assert_eq!(session_key(None, None), None);
    }

    #[test]
    fn derived_key_follows_the_first_system_and_user_messages() {
        let turn = |extra: &[Value]| {
            let mut messages = vec![
                serde_json::json!({"role": "system", "content": "You are a coding agent."}),
                serde_json::json!({"role": "user", "content": "Fix the failing test."}),
            ];
            messages.extend_from_slice(extra);
            serde_json::to_vec(&serde_json::json!({"model": "m", "messages": messages})).unwrap()
        };
        let first = derived_session_key(&turn(&[])).unwrap();
        let later = derived_session_key(&turn(&[
            serde_json::json!({"role": "assistant", "content": "Looking."}),
            serde_json::json!({"role": "user", "content": "Go on."}),
        ]))
        .unwrap();
        assert_eq!(first, later, "every turn of a conversation shares the key");
        assert_eq!(first.len(), 32);
        let other = serde_json::to_vec(&serde_json::json!({"model": "m", "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Write a parser."}
        ]}))
        .unwrap();
        assert_ne!(derived_session_key(&other).unwrap(), first);
        assert_eq!(derived_session_key(br#"{"messages": []}"#), None);
        assert_eq!(derived_session_key(b"not json"), None);
    }
}
