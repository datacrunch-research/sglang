// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::{sync::Arc, time::Duration};

use anyhow::{bail, ensure, Result};
use serde::Deserialize;

use crate::buckets_reorg::{
    Bucket, BucketGroups, BucketResolver, EngineGroup, IdleSpec, SecondsLimits, SloPreference,
    SpillTarget, TokenLimits,
};
use crate::config::{
    AffinityConfig, AffinityMode, BalancedBy, DecodePolicyKind, FilterKind, ModelConfig,
    PolicyKind, SessionAffinityMode,
};
use crate::discovery::WorkerId;
use crate::server::bucket_admission::{BucketQueue, QueueSpec, TpmLimitSpec, TpmLimiter};
use crate::state::kv_events::RadixTreePrefixProvider;
use crate::state::session_estimator::{SessionEstimator, DEFAULT_SESSION_IDLE};
use crate::state::{
    kv_events::KvEventIndex, load_monitor::router_inflight_load::JanitorHandle, AffinityStore,
};

use super::{
    admission::AdmissionLimits,
    cache_aware::{CacheAwarePolicy, CacheSource},
    power_of_two::PowerOfTwoPolicy,
    session_aware::SessionAwarePolicy,
    Policy, Stage,
};

/// Reorg `--bucket-config`: plain or P/D buckets whose groups each own their
/// membership, policy and admission.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketsConfig {
    pub buckets: Vec<BucketSpec>,
    pub ttft_slo: SloPreference,
    pub tps_slo: SloPreference,
    /// Seconds a session's last turn stays usable for the uncached estimate.
    pub session_idle_secs: Option<u64>,
    /// When an engine of a spill target counts as idle.
    pub spill_idle: IdleSpec,
    /// The pool's agreed tokens per minute; above it requests get 429.
    pub tpm_limit: Option<TpmLimitSpec>,
    /// What bucket bounds measure: `uncached_tokens` (`min/max_uncached_tokens`)
    /// or `prefill_seconds` (`min/max_prefill_seconds`, uncached tokens over the
    /// engines' measured prefill rate).
    pub thresholds: Thresholds,
    /// Under `prefill_seconds`, the rate used until the engines report one.
    pub prefill_reference_rate_tokens_per_s: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Thresholds {
    #[default]
    UncachedTokens,
    PrefillSeconds,
}

/// Set either `plain` or both `prefill` and `decode`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketSpec {
    pub id: String,
    pub rank: u32,
    pub min_input_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    /// Bounds on the request's estimated uncached prompt tokens.
    pub min_uncached_tokens: Option<u64>,
    pub max_uncached_tokens: Option<u64>,
    /// Bounds on its estimated prefill seconds (`thresholds: prefill_seconds`).
    pub min_prefill_seconds: Option<f64>,
    pub max_prefill_seconds: Option<f64>,
    pub max_context_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    /// Bounded wait queue: the request waits here for an engine of this
    /// bucket instead of moving to the next fitting bucket.
    pub queue: Option<QueueSpec>,
    /// Longer buckets, in order, whose idle engines may take this bucket's
    /// requests when none of its own admits one.
    pub spill_to: Vec<String>,
    pub plain: Option<GroupSpec>,
    pub prefill: Option<GroupSpec>,
    pub decode: Option<GroupSpec>,
}

/// Omitted fields mean every engine of the role and `--policy` (power-of-two on
/// decode); admission limits left unset take `--max-inflight-requests` / `--max-kv-usage`,
/// and affinity settings left unset take `--affinity-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupSpec {
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Any listed Kubernetes Service, as namespace/name; exclusive with worker_ids.
    pub worker_services: Option<HashSet<String>>,
    pub policy: Option<PolicyKind>,
    pub admission: Option<AdmissionLimits>,
    pub affinity: Option<AffinitySpec>,
}

/// Per-group `--affinity-mode`, `--affinity-balanced-by` and `--affinity-load-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AffinitySpec {
    pub mode: Option<AffinityMode>,
    pub balanced_by: Option<BalancedBy>,
    pub load_factor: Option<f64>,
    pub load_gap: Option<u64>,
}

impl AffinitySpec {
    /// Balanced-only fields need balanced `mode`; a set load factor must be at least 1.
    pub(crate) fn validate(&self, mode: AffinityMode) -> Result<()> {
        ensure!(
            mode == AffinityMode::Balanced
                || (self.balanced_by.is_none()
                    && self.load_factor.is_none()
                    && self.load_gap.is_none()),
            "--affinity-balanced-by and --affinity-load-* (group balanced_by and load_*) require balanced mode"
        );
        ensure!(
            (self.load_factor).is_none_or(|factor| factor.is_finite() && factor >= 1.0),
            "--affinity-load-factor (group load_factor) must be finite and at least 1"
        );
        Ok(())
    }

    /// `defaults` with each set field replaced, validated like the CLI flags.
    fn or(&self, defaults: &AffinityConfig) -> Result<AffinityConfig> {
        let balanced_by = self.balanced_by.unwrap_or(defaults.balanced_by);
        let config = AffinityConfig {
            mode: self.mode.unwrap_or(defaults.mode),
            balanced_by,
            load_factor: self.load_factor.unwrap_or(defaults.load_factor),
            // A CLI gap is in the CLI metric's unit; another metric uses its own default.
            load_gap: (self.load_gap).or(defaults
                .load_gap
                .filter(|_| balanced_by == defaults.balanced_by)),
            ..defaults.clone()
        };
        ensure!(
            matches!(config.mode, AffinityMode::Prefer | AffinityMode::Balanced),
            "group affinity mode must be prefer or balanced"
        );
        self.validate(config.mode)?;
        Ok(config)
    }
}

impl BucketsConfig {
    /// Discovery decides which bucket has candidates. Plain ranks first so plain
    /// deployments do not scan for prefill engines on every request.
    fn default_buckets() -> Self {
        let all = || Some(GroupSpec::default());
        Self {
            buckets: vec![
                BucketSpec {
                    id: "plain".into(),
                    plain: all(),
                    ..Default::default()
                },
                BucketSpec {
                    id: "pd".into(),
                    rank: 1,
                    prefill: all(),
                    decode: all(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }
}

pub fn validate(model: &ModelConfig) -> Result<()> {
    ensure!(
        matches!(
            model.policy,
            PolicyKind::PowerOfTwo | PolicyKind::CacheAware | PolicyKind::SessionAware
        ),
        "reorg routing supports power_of_two, cache_aware, and session_aware"
    );
    ensure!(
        model.bucket_config.is_none(),
        "reorg routing reads reorg_buckets, not the legacy bucket_config"
    );
    ensure!(
        model.decode_policy == DecodePolicyKind::PowerOfTwo,
        "reorg routing requires --decode-policy power_of_two"
    );
    if let Some(filters) = &model.eligibility {
        ensure!(
            filters
                .filters
                .iter()
                .all(|filter| *filter == FilterKind::Overloaded),
            "reorg routing only supports --filter overloaded"
        );
    }
    if let Some(affinity) = &model.affinity {
        ensure!(
            !affinity.stable_pair && affinity.session_affinity_mode == SessionAffinityMode::Bucket,
            "reorg routing uses bucket-scoped sessions without --stable-pair"
        );
        ensure!(
            affinity.min_load_choices == 2,
            "reorg cache fallback requires --min-load-choices 2"
        );
    }
    Ok(())
}

/// Build `model.reorg_buckets`, or the default plain and P/D buckets. Callers
/// run [`validate`] first; `Cli::into_config` does so before startup reaches this point.
pub fn build_resolver(
    model: &ModelConfig,
    state: &KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
) -> Result<(BucketResolver, Option<JanitorHandle>)> {
    let mut groups = Groups {
        model,
        state,
        external_index,
        affinity: model.affinity.clone().unwrap_or_default(),
        store: None,
        cleanup: None,
        source: None,
    };
    let default_buckets;
    let config = match &model.reorg_buckets {
        Some(config) => config,
        None => {
            default_buckets = BucketsConfig::default_buckets();
            &default_buckets
        }
    };
    ensure!(!config.buckets.is_empty(), "--bucket-config has no buckets");
    let mut ids = HashSet::new();
    let mut buckets = Vec::new();
    for spec in &config.buckets {
        let id = &spec.id;
        ensure!(
            !id.is_empty() && ids.insert(id),
            "bucket id {id:?} must be non-empty and unique"
        );
        ensure!(
            spec.min_input_tokens
                .zip(spec.max_input_tokens)
                .is_none_or(|(min, max)| min <= max)
                && spec.max_context_tokens != Some(0)
                && (spec.min_input_tokens)
                    .zip(spec.max_context_tokens)
                    .is_none_or(|(min, context)| min <= context)
                && spec.ttft_ms != Some(0)
                && spec
                    .tokens_per_second
                    .is_none_or(|t| t.is_finite() && t > 0.0),
            "bucket {id:?} needs min input tokens <= max input and context tokens, \
             and positive capacity and SLO estimates"
        );
        ensure!(
            spec.min_uncached_tokens
                .zip(spec.max_uncached_tokens)
                .is_none_or(|(min, max)| min <= max)
                && spec.max_uncached_tokens != Some(0),
            "bucket {id:?} needs min uncached tokens <= a positive max uncached tokens"
        );
        let bucket_groups = match (&spec.plain, &spec.prefill, &spec.decode) {
            (Some(plain), None, None) => {
                BucketGroups::Plain(groups.build(id, Stage::Plain, plain)?)
            }
            (None, Some(prefill), Some(decode)) => BucketGroups::Pd {
                prefill: groups.build(id, Stage::Prefill, prefill)?,
                decode: groups.build(id, Stage::Decode, decode)?,
            },
            _ => bail!("bucket {id:?} needs either plain or both prefill and decode"),
        };
        let mut bucket = Bucket::new(id.clone(), bucket_groups);
        bucket.rank = spec.rank;
        bucket.limits = TokenLimits {
            min: spec.min_input_tokens,
            max: spec.max_input_tokens,
        };
        bucket.uncached = TokenLimits {
            min: spec.min_uncached_tokens,
            max: spec.max_uncached_tokens,
        };
        validate_thresholds(config, spec)?;
        bucket.prefill_seconds = SecondsLimits {
            min: spec.min_prefill_seconds,
            max: spec.max_prefill_seconds,
        };
        if let Some(queue) = &spec.queue {
            queue
                .validate()
                .map_err(|error| anyhow::anyhow!("bucket {id:?}: {error}"))?;
            bucket.queue = Some(Arc::new(BucketQueue::new(id.clone(), queue.clone())));
        }
        bucket.spill_to = spec.spill_to.iter().map(SpillTarget::new).collect();
        if let Some(decode) = &spec.decode {
            let defaults = &model.reorg_admission;
            bucket.decode_permits_per_engine = decode
                .admission
                .as_ref()
                .map_or_else(|| defaults.clone(), |limits| limits.or(defaults))
                .max_inflight_requests;
        }
        bucket.max_context_tokens = spec.max_context_tokens;
        bucket.ttft_ms = spec.ttft_ms;
        bucket.tokens_per_second = spec.tokens_per_second;
        buckets.push(bucket);
    }
    validate_spill(&buckets)?;
    let mut resolver = BucketResolver::new(buckets)?;
    resolver.ttft_slo = config.ttft_slo;
    resolver.tps_slo = config.tps_slo;
    ensure!(
        config.spill_idle.max_running_requests > 0,
        "--bucket-config spill_idle.max_running_requests must be positive"
    );
    resolver.idle = config.spill_idle;
    if config.thresholds == Thresholds::PrefillSeconds {
        resolver.prefill_seconds_reference = config.prefill_reference_rate_tokens_per_s;
    }
    if let Some(spec) = &config.tpm_limit {
        spec.validate()
            .map_err(|error| anyhow::anyhow!("--bucket-config {error}"))?;
        resolver.tpm = Some(Arc::new(TpmLimiter::new(spec.clone())));
    }
    ensure!(
        config.session_idle_secs != Some(0),
        "--bucket-config session_idle_secs must be positive"
    );
    if resolver.buckets.iter().any(Bucket::bounds_uncached_tokens) {
        let idle = config
            .session_idle_secs
            .map_or(DEFAULT_SESSION_IDLE, Duration::from_secs);
        resolver.estimator = Some(Arc::new(SessionEstimator::new(idle)));
    }
    Ok((resolver, groups.cleanup))
}

/// A file bounds its buckets in one unit: uncached tokens, or prefill seconds
/// with a positive reference rate.
fn validate_thresholds(config: &BucketsConfig, spec: &BucketSpec) -> Result<()> {
    let id = &spec.id;
    let tokens = spec.min_uncached_tokens.is_some() || spec.max_uncached_tokens.is_some();
    let seconds = spec.min_prefill_seconds.is_some() || spec.max_prefill_seconds.is_some();
    match config.thresholds {
        Thresholds::UncachedTokens => ensure!(
            !seconds,
            "bucket {id:?} sets prefill seconds; they need thresholds: prefill_seconds"
        ),
        Thresholds::PrefillSeconds => {
            ensure!(
                !tokens,
                "bucket {id:?} sets uncached tokens under thresholds: prefill_seconds"
            );
            ensure!(
                config
                    .prefill_reference_rate_tokens_per_s
                    .is_some_and(|rate| rate.is_finite() && rate > 0.0),
                "thresholds: prefill_seconds needs a positive prefill_reference_rate_tokens_per_s"
            );
        }
    }
    let valid = |value: Option<f64>| value.is_none_or(|v| v.is_finite() && v >= 0.0);
    ensure!(
        valid(spec.min_prefill_seconds)
            && valid(spec.max_prefill_seconds)
            && spec.max_prefill_seconds != Some(0.0)
            && spec
                .min_prefill_seconds
                .zip(spec.max_prefill_seconds)
                .is_none_or(|(min, max)| min <= max),
        "bucket {id:?} needs min prefill seconds <= a positive max prefill seconds"
    );
    Ok(())
}

/// Spill runs one way, from shorter lanes to longer ones: a target must take
/// longer requests than its source (uncached bound first, then input bound).
fn validate_spill(buckets: &[Bucket]) -> Result<()> {
    let reach = |bucket: &Bucket| {
        (
            bucket.uncached_capacity(),
            bucket.limits.max.unwrap_or(u64::MAX),
        )
    };
    for bucket in buckets {
        for target in &bucket.spill_to {
            let Some(found) = buckets.iter().find(|b| b.id == target.bucket) else {
                bail!(
                    "bucket {:?} spills to unknown bucket {:?}",
                    bucket.id,
                    target.bucket
                );
            };
            ensure!(
                reach(found) > reach(bucket),
                "bucket {:?} may spill only to a longer bucket, not {:?}",
                bucket.id,
                target.bucket
            );
        }
    }
    Ok(())
}

/// Builds group policies; session stores and cache sources are shared by all groups.
struct Groups<'a> {
    model: &'a ModelConfig,
    state: &'a KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    affinity: AffinityConfig,
    store: Option<Arc<AffinityStore>>,
    cleanup: Option<JanitorHandle>,
    source: Option<Arc<CacheSource>>,
}

impl Groups<'_> {
    fn build(&mut self, bucket: &str, stage: Stage, spec: &GroupSpec) -> Result<EngineGroup> {
        let default = match stage {
            Stage::Decode => PolicyKind::PowerOfTwo,
            _ => self.model.policy,
        };
        let kind = spec.policy.unwrap_or(default);
        // Only --policy has its affinity, cache and tokenizer settings resolved.
        ensure!(
            kind == PolicyKind::PowerOfTwo || kind == self.model.policy,
            "bucket groups use power_of_two or --policy, not {kind:?}"
        );
        ensure!(
            spec.worker_ids.as_ref().is_none_or(|ids| {
                !ids.is_empty() && ids.iter().all(|id| !id.0.trim().is_empty())
            }),
            "bucket {bucket:?} {stage:?} worker_ids and their values must not be empty; omit it for every engine"
        );
        ensure!(
            spec.worker_ids.is_none() || spec.worker_services.is_none(),
            "bucket {bucket:?} {stage:?} must use either worker_ids or worker_services"
        );
        ensure!(
            spec.worker_services.as_ref().is_none_or(|services| {
                !services.is_empty() && services.iter().all(|service| {
                    service.split_once('/').is_some_and(|(namespace, name)| {
                        !namespace.trim().is_empty() && !name.trim().is_empty()
                            && !name.contains('/') && !service.chars().any(char::is_whitespace)
                    })
                })
            }),
            "bucket {bucket:?} {stage:?} worker_services must be a nonempty list of namespace/service names"
        );
        let defaults = &self.model.reorg_admission;
        let admission = spec
            .admission
            .as_ref()
            .map_or(defaults.clone(), |a| a.or(defaults));
        admission.validate()?;
        let admission = Arc::new(admission);
        ensure!(
            spec.affinity.is_none() || kind != PolicyKind::PowerOfTwo,
            "bucket {bucket:?} {stage:?} sets affinity on a power_of_two group"
        );
        let affinity = match &spec.affinity {
            Some(spec) => spec.or(&self.affinity)?,
            None => self.affinity.clone(),
        };
        let load = self.state.engine_reported_load();
        let policy: Arc<dyn Policy> = match kind {
            PolicyKind::PowerOfTwo => {
                let mut policy = PowerOfTwoPolicy::new(load);
                policy.admission = admission;
                Arc::new(policy)
            }
            PolicyKind::SessionAware => {
                let mut policy = SessionAwarePolicy::new(self.store(), load);
                policy.admission = admission;
                policy.config = affinity;
                Arc::new(policy)
            }
            PolicyKind::CacheAware => {
                let mut policy = CacheAwarePolicy::new(self.source(), load, affinity)?;
                policy.admission = admission;
                Arc::new(policy)
            }
            other => bail!("reorg routing does not implement --policy {other:?}"),
        };
        Ok(EngineGroup {
            worker_ids: spec.worker_ids.clone(),
            worker_services: spec.worker_services.clone(),
            policy,
        })
    }

    fn store(&mut self) -> Arc<AffinityStore> {
        let affinity = &self.affinity;
        let cleanup = &mut self.cleanup;
        Arc::clone(self.store.get_or_insert_with(|| {
            let store = AffinityStore::new(Duration::from_secs(affinity.session_idle_secs));
            *cleanup =
                store.spawn_sweeper(Duration::from_secs(affinity.session_eviction_interval_secs));
            store
        }))
    }

    fn source(&mut self) -> Arc<CacheSource> {
        let (state, index) = (self.state, self.external_index.clone());
        Arc::clone(self.source.get_or_insert_with(|| {
            Arc::new(match index {
                Some(index) => CacheSource::Remote {
                    index,
                    block_size: state.block_size_oracle(),
                },
                None => CacheSource::Local(RadixTreePrefixProvider::new(
                    state.tree(),
                    state.block_size_oracle(),
                )),
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model(buckets: serde_json::Value) -> ModelConfig {
        ModelConfig {
            id: "m".into(),
            tokenizer_path: None,
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::PowerOfTwo,
            decode_policy: DecodePolicyKind::PowerOfTwo,
            dp_aware: false,
            dp_rank_policy: Default::default(),
            bucket_config: None,
            reorg_buckets: Some(serde_json::from_value(buckets).unwrap()),
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        }
    }

    #[tokio::test]
    async fn uncached_bounds_build_an_estimator_and_are_validated() {
        let model = test_model;
        let state = KvEventIndex::new();
        let (resolver, _) = build_resolver(
            &model(serde_json::json!({"buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "plain": {}},
                {"id": "L", "min_uncached_tokens": 32001, "plain": {}}
            ]})),
            &state,
            None,
        )
        .unwrap();
        assert!(resolver.estimator.is_some());
        let found = resolver
            .resolve_uncached(900_000, Some(2_000), None, None, None)
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "S");
        let (plain, _) = build_resolver(
            &model(serde_json::json!({"buckets": [{"id": "all", "plain": {}}]})),
            &state,
            None,
        )
        .unwrap();
        assert!(
            plain.estimator.is_none(),
            "length-only buckets keep no sessions"
        );
        assert!(plain.buckets[0].queue.is_none());
        for invalid in [
            serde_json::json!({"buckets": [{"id": "x", "min_uncached_tokens": 9, "max_uncached_tokens": 8, "plain": {}}]}),
            serde_json::json!({"buckets": [{"id": "x", "max_uncached_tokens": 0, "plain": {}}]}),
            serde_json::json!({"session_idle_secs": 0, "buckets": [{"id": "x", "max_uncached_tokens": 8, "plain": {}}]}),
        ] {
            assert!(build_resolver(&model(invalid), &state, None).is_err());
        }
    }

    #[tokio::test]
    async fn a_bucket_queue_is_built_from_its_spec_and_validated() {
        let state = KvEventIndex::new();
        let build = |queue: serde_json::Value| {
            let mut model = test_model(serde_json::json!({"buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "queue": queue, "plain": {}}
            ]}));
            model.policy = PolicyKind::PowerOfTwo;
            build_resolver(&model, &state, None).map(|(resolver, _)| resolver)
        };
        let resolver = build(serde_json::json!(
            {"mean_wait_s": 4.0, "rate_tokens_per_s": 20000, "rate_requests_per_s": 3.0}
        ))
        .unwrap();
        let queue = resolver.buckets[0].queue.as_ref().unwrap();
        assert_eq!((queue.id.as_str(), queue.spec.deadline_factor), ("S", 2.3));
        assert!(build(serde_json::json!(
            {"mean_wait_s": -1, "rate_tokens_per_s": 1, "rate_requests_per_s": 1}
        ))
        .is_err());
    }

    #[tokio::test]
    async fn spill_runs_only_from_shorter_to_longer_buckets() {
        let state = KvEventIndex::new();
        let build = |buckets: serde_json::Value| {
            build_resolver(&test_model(buckets), &state, None).map(|(resolver, _)| resolver)
        };
        let resolver = build(serde_json::json!({
            "spill_idle": {"max_running_requests": 2},
            "buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "spill_to": ["M", "L"], "plain": {}},
                {"id": "M", "min_uncached_tokens": 32001, "max_uncached_tokens": 256000, "spill_to": ["L"], "plain": {}},
                {"id": "L", "min_uncached_tokens": 256001, "plain": {}}
            ]
        }))
        .unwrap();
        let targets: Vec<_> = (resolver.bucket("S").unwrap().spill_to.iter())
            .map(|target| target.bucket.as_str())
            .collect();
        assert_eq!(targets, ["M", "L"]);
        assert_eq!(resolver.idle.max_running_requests, 2);
        assert_eq!(resolver.idle.max_waiting_uncached_tokens, 0);
        for invalid in [
            serde_json::json!({"buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "plain": {}},
                {"id": "L", "min_uncached_tokens": 32001, "spill_to": ["S"], "plain": {}}
            ]}),
            serde_json::json!({"buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "spill_to": ["X"], "plain": {}}
            ]}),
            serde_json::json!({"buckets": [
                {"id": "S", "max_uncached_tokens": 32000, "spill_to": ["S"], "plain": {}}
            ]}),
            serde_json::json!({"spill_idle": {"max_running_requests": 0}, "buckets": [
                {"id": "S", "plain": {}}
            ]}),
        ] {
            assert!(build(invalid).is_err());
        }
    }

    #[tokio::test]
    async fn prefill_second_bounds_need_their_flag_and_a_reference_rate() {
        let state = KvEventIndex::new();
        let build = |buckets: serde_json::Value| {
            build_resolver(&test_model(buckets), &state, None).map(|(resolver, _)| resolver)
        };
        let resolver = build(serde_json::json!({
            "thresholds": "prefill_seconds",
            "prefill_reference_rate_tokens_per_s": 20000,
            "buckets": [
                {"id": "S", "max_prefill_seconds": 1.5, "spill_to": ["L"], "plain": {}},
                {"id": "L", "min_prefill_seconds": 1.5, "plain": {}}
            ]
        }))
        .unwrap();
        assert_eq!(resolver.prefill_seconds_reference, Some(20_000.0));
        assert!(
            resolver.estimator.is_some(),
            "seconds bounds read the estimate"
        );
        assert_eq!(resolver.bucket("S").unwrap().prefill_seconds.max, Some(1.5));
        for invalid in [
            serde_json::json!({"buckets": [{"id": "S", "max_prefill_seconds": 1, "plain": {}}]}),
            serde_json::json!({"thresholds": "prefill_seconds", "buckets": [
                {"id": "S", "max_prefill_seconds": 1, "plain": {}}]}),
            serde_json::json!({"thresholds": "prefill_seconds", "prefill_reference_rate_tokens_per_s": 1000,
                "buckets": [{"id": "S", "max_uncached_tokens": 1000, "plain": {}}]}),
            serde_json::json!({"thresholds": "prefill_seconds", "prefill_reference_rate_tokens_per_s": 1000,
                "buckets": [{"id": "S", "min_prefill_seconds": 3, "max_prefill_seconds": 2, "plain": {}}]}),
        ] {
            assert!(build(invalid).is_err());
        }
    }

    #[test]
    fn group_metric_change_drops_the_cli_gap() {
        let cli = AffinityConfig {
            mode: AffinityMode::Balanced,
            load_gap: Some(2_048),
            ..Default::default()
        };
        let by_requests: AffinitySpec =
            serde_json::from_str(r#"{"balanced_by": "running_requests"}"#).unwrap();
        let group = by_requests.or(&cli).unwrap();
        assert_eq!(
            (group.mode, group.balanced_by, group.load_gap()),
            (AffinityMode::Balanced, BalancedBy::RunningRequests, 4)
        );
        assert_eq!(AffinitySpec::default().or(&cli).unwrap().load_gap(), 2_048);
    }
}
