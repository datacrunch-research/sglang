// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Order length-compatible buckets by optional SLO preferences, then capacity and rank.
//!
//! ```text
//! BucketResolver (one model's buckets)
//!   -> Bucket (token limits, context capacity, rank, SLO estimates)
//!        -> Plain: one EngineGroup
//!        -> PD: prefill + decode EngineGroups
//!             -> each EngineGroup: worker membership + its own Policy
//! ```
//!
//! - [`BucketResolver::resolve`] returns length-compatible buckets in preference order.
//! - [`EngineGroup::pick`] filters live workers by model, health, stage, and membership,
//!   then calls [`Policy::pick`] and validates the returned engine.
//!
//! The handler tries buckets in order, advancing on missing candidates or admission
//! rejection. Both P/D picks must succeed in the same bucket before dispatch.
//! [`WorkerRegistry`] owns live workers; groups select by ID or Service. Policies own
//! their load/KV/affinity dependencies and pass selected observations to admission.

use std::collections::HashSet;
use std::sync::Arc;

use crate::discovery::{ModelId, WorkerId};
use crate::policies_reorg::admission::EngineMetrics;
use crate::policies_reorg::{Pick, PickError, PickRequest, Policy, Stage};
use crate::server::bucket_admission::{BucketQueue, PrefillRates, TpmLimiter};
use crate::state::kv_events::CacheNamespace;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::state::session_estimator::SessionEstimator;
use crate::workers::{paired_prefills, Worker, WorkerRegistry};

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenLimits {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl TokenLimits {
    fn fits(&self, tokens: u64) -> bool {
        self.min.is_none_or(|min| tokens >= min) && self.max.is_none_or(|max| tokens <= max)
    }
}

/// Inclusive range of estimated prefill seconds: uncached tokens over the
/// pool's prefill rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SecondsLimits {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl SecondsLimits {
    fn is_set(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }

    /// Unbounded limits fit anything; bounded ones need a positive rate.
    fn fits(&self, tokens: u64, rate: Option<f64>) -> bool {
        if !self.is_set() {
            return true;
        }
        let Some(rate) = rate.filter(|rate| rate.is_finite() && *rate > 0.0) else {
            return false;
        };
        let seconds = tokens as f64 / rate;
        self.min.is_none_or(|min| seconds >= min) && self.max.is_none_or(|max| seconds <= max)
    }
}

#[derive(Debug)]
pub struct EngineGroup {
    /// Exact identities. Omit both membership fields to include all engines of the role.
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Match any named Kubernetes Service (namespace/name), across pod replacements.
    pub worker_services: Option<HashSet<String>>,
    pub policy: Arc<dyn Policy>,
}

impl EngineGroup {
    pub fn new(policy: Arc<dyn Policy>) -> Self {
        Self {
            worker_ids: None,
            worker_services: None,
            policy,
        }
    }

    /// Resolve live members and invoke this group's policy. Never changes buckets.
    pub async fn pick(
        &self,
        workers: &WorkerRegistry,
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        self.pick_from(self.members(workers, request.model, request.stage), request)
            .await
    }

    fn members(&self, workers: &WorkerRegistry, model: &ModelId, stage: Stage) -> Vec<Arc<Worker>> {
        workers
            .healthy_workers_for(model)
            .into_iter()
            .filter(|engine| engine.mode() == stage)
            .filter(|engine| {
                self.worker_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&engine.id))
            })
            .filter(|engine| {
                self.worker_services
                    .as_ref()
                    .is_none_or(|services| engine.matches_services(services))
            })
            .collect()
    }

    /// Live members minus the engines this request already failed on.
    fn candidates(
        &self,
        workers: &WorkerRegistry,
        request: &BucketRequest<'_>,
        stage: Stage,
    ) -> Vec<Arc<Worker>> {
        let mut engines = self.members(workers, request.model, stage);
        engines.retain(|engine| !request.excluded.contains(&engine.id));
        engines
    }

    async fn pick_from(
        &self,
        mut engines: Vec<Arc<Worker>>,
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        // Stable order so cursor-based policies see a consistent candidate list.
        engines.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let pick = self.policy.pick(&engines, request).await?;
        if !engines
            .iter()
            .any(|engine| Arc::ptr_eq(engine, &pick.engine))
        {
            return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
        }
        Ok(pick)
    }
}

/// A bucket serves a request on one plain engine or on its own P/D groups.
#[derive(Debug)]
pub enum BucketGroups {
    Plain(EngineGroup),
    Pd {
        prefill: EngineGroup,
        decode: EngineGroup,
    },
}

/// Prepared request facts shared by all bucket attempts. The bucket supplies
/// its ID and each group's stage when calling policies.
#[derive(Debug)]
pub struct BucketRequest<'a> {
    pub model: &'a ModelId,
    pub input_tokens: u64,
    pub total_input_tokens: u64,
    pub expected_peak_tokens: Option<u64>,
    pub prefix: Option<&'a crate::policies_reorg::cache_aware::PrefixMemo>,
    pub token_ids: Option<&'a [u32]>,
    pub cache_namespace: &'a CacheNamespace,
    pub session_key: Option<&'a str>,
    pub routing_key: Option<&'a str>,
    /// Engines this request already failed on; no group offers them again.
    pub excluded: &'a [WorkerId],
}

/// A complete selection from one bucket. For plain serving, `prefill` is the
/// plain engine and `decode` is absent; PD supplies both picks.
#[derive(Debug)]
pub struct BucketPick {
    pub prefill: Pick,
    pub decode: Option<Pick>,
}

#[derive(Debug)]
pub struct Bucket {
    pub id: String,
    /// Break ties within an SLO tier between equally sized buckets; lower ranks win.
    pub rank: u32,
    /// Inclusive input-token range used to choose the bucket.
    pub limits: TokenLimits,
    /// Inclusive range of estimated uncached prompt tokens; the whole prompt
    /// counts when no estimate is supplied.
    pub uncached: TokenLimits,
    /// The same range in estimated prefill seconds, when the bucket file
    /// sets `thresholds: prefill_seconds`.
    pub prefill_seconds: SecondsLimits,
    /// Full sequence capacity, checked against the expected peak when known.
    pub max_context_tokens: Option<u64>,
    /// Optional service estimates used only for bucket ordering.
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub groups: BucketGroups,
    /// Bounded prefill work with a wait deadline; without it a bucket that
    /// cannot place a request passes it to the next fitting bucket.
    pub queue: Option<Arc<BucketQueue>>,
    /// Longer buckets whose idle engines take this bucket's requests when
    /// none of its own admits one; never the other way.
    pub spill_to: Vec<SpillTarget>,
    /// The decode group's per-engine in-flight cap, for the permits gauge.
    pub decode_permits_per_engine: Option<u64>,
}

#[derive(Debug)]
pub struct SpillTarget {
    pub bucket: String,
    /// Requests this bucket placed on the target's engines.
    pub spilled: std::sync::atomic::AtomicU64,
}

impl SpillTarget {
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            spilled: Default::default(),
        }
    }
}

/// When an engine counts as idle for a spill: its reported waiting uncached
/// tokens and running requests, and this router's own in-flight count, are
/// all low. Unreported load is never idle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IdleSpec {
    pub max_waiting_uncached_tokens: u64,
    /// Strictly fewer running (and router in-flight) requests than this.
    pub max_running_requests: u64,
}

impl Default for IdleSpec {
    fn default() -> Self {
        Self {
            max_waiting_uncached_tokens: 0,
            max_running_requests: 4,
        }
    }
}

impl IdleSpec {
    pub fn is_idle(&self, engine: &Worker, load: &EngineReportedLoadSnapshot) -> bool {
        let metrics = EngineMetrics::observe(engine, load);
        metrics
            .pending_prefill_tokens
            .is_some_and(|waiting| waiting <= self.max_waiting_uncached_tokens)
            && metrics
                .running_requests
                .is_some_and(|running| running < self.max_running_requests)
            && metrics.inflight_requests < self.max_running_requests
    }
}

impl Bucket {
    pub fn new(id: impl Into<String>, groups: BucketGroups) -> Self {
        Self {
            id: id.into(),
            rank: 0,
            limits: TokenLimits::default(),
            uncached: TokenLimits::default(),
            prefill_seconds: SecondsLimits::default(),
            max_context_tokens: None,
            ttft_ms: None,
            tokens_per_second: None,
            groups,
            queue: None,
            spill_to: Vec::new(),
            decode_permits_per_engine: None,
        }
    }

    /// Reject a policy installed on a stage it cannot serve before any request reaches it.
    pub fn validate(&self) -> Result<(), PickError> {
        let groups: &[(Stage, &EngineGroup)] = match &self.groups {
            BucketGroups::Plain(group) => &[(Stage::Plain, group)],
            BucketGroups::Pd { prefill, decode } => {
                &[(Stage::Prefill, prefill), (Stage::Decode, decode)]
            }
        };
        for (stage, group) in groups {
            if !group.policy.supports(*stage) {
                return Err(PickError::InvalidConfiguration(format!(
                    "bucket {} installs a policy that cannot serve the {stage:?} stage",
                    self.id
                )));
            }
        }
        Ok(())
    }

    /// Whether choosing or serving this bucket reads request tokens: a length
    /// bound, or a group policy that matches prefixes.
    pub fn needs_request_tokens(&self) -> bool {
        let groups: &[&EngineGroup] = match &self.groups {
            BucketGroups::Plain(group) => &[group],
            BucketGroups::Pd { prefill, decode } => &[prefill, decode],
        };
        self.limits.min.is_some()
            || self.limits.max.is_some()
            || self.max_context_tokens.is_some()
            || groups
                .iter()
                .any(|group| group.policy.needs_request_tokens())
    }

    /// Readiness uses this bucket's actual membership and pairing constraints.
    pub fn has_ready_workers(&self, workers: &WorkerRegistry, model: &ModelId) -> bool {
        match &self.groups {
            BucketGroups::Plain(group) => !group.members(workers, model, Stage::Plain).is_empty(),
            BucketGroups::Pd { prefill, decode } => !paired_prefills(
                prefill.members(workers, model, Stage::Prefill),
                &decode.members(workers, model, Stage::Decode),
            )
            .is_empty(),
        }
    }

    /// Engines that can serve this bucket now: plain members, or prefill
    /// members with a decode in their version group. Its queue depth scales with them.
    pub fn ready_members(&self, workers: &WorkerRegistry, model: &ModelId) -> usize {
        self.ready_prefill_engines(workers, model).len()
    }

    /// The engines [`Self::ready_members`] counts.
    pub fn ready_prefill_engines(
        &self,
        workers: &WorkerRegistry,
        model: &ModelId,
    ) -> Vec<Arc<Worker>> {
        match &self.groups {
            BucketGroups::Plain(group) => group.members(workers, model, Stage::Plain),
            BucketGroups::Pd { prefill, decode } => paired_prefills(
                prefill.members(workers, model, Stage::Prefill),
                &decode.members(workers, model, Stage::Decode),
            ),
        }
    }

    /// Ready decode engines of a P/D bucket; none for a plain one.
    pub fn ready_decode_members(
        &self,
        workers: &WorkerRegistry,
        model: &ModelId,
    ) -> Vec<Arc<Worker>> {
        match &self.groups {
            BucketGroups::Plain(_) => Vec::new(),
            BucketGroups::Pd { decode, .. } => decode.members(workers, model, Stage::Decode),
        }
    }

    /// Select this bucket's plain engine or complete P/D pair, without dispatching.
    /// A failed group reports its stage; the caller may then try another bucket.
    pub async fn pick_engines(
        &self,
        workers: &WorkerRegistry,
        request: &BucketRequest<'_>,
    ) -> Result<BucketPick, (Stage, PickError)> {
        self.pick_engines_where(workers, request, &|_| true).await
    }

    /// [`Self::pick_engines`] among the plain or prefill engines `keep` accepts;
    /// decode candidates are not filtered.
    pub async fn pick_engines_where(
        &self,
        workers: &WorkerRegistry,
        request: &BucketRequest<'_>,
        keep: &(dyn Fn(&Worker) -> bool + Sync),
    ) -> Result<BucketPick, (Stage, PickError)> {
        let (prefill, decode) = match &self.groups {
            BucketGroups::Plain(group) => {
                let mut engines = group.candidates(workers, request, Stage::Plain);
                engines.retain(|engine| keep(engine));
                let plain = self.pick_from_group(group, Stage::Plain, engines, request);
                (plain.await?, None)
            }
            BucketGroups::Pd { prefill, decode } => {
                let mut prefills = prefill.candidates(workers, request, Stage::Prefill);
                prefills.retain(|engine| keep(engine));
                let decoders = decode.candidates(workers, request, Stage::Decode);
                let stage = if prefills.is_empty() {
                    Stage::Prefill
                } else {
                    Stage::Decode
                };
                // A prefill hands its KV only to a decode in its own version group.
                let mut prefills = paired_prefills(prefills, &decoders);
                if prefills.is_empty() {
                    return Err((stage, PickError::NoCandidates));
                }
                loop {
                    let prefill = self
                        .pick_from_group(prefill, Stage::Prefill, prefills.clone(), request)
                        .await?;
                    let group = prefill.engine.version_group();
                    let peers = decoders
                        .iter()
                        .filter(|d| d.version_group() == group)
                        .cloned()
                        .collect();
                    match self
                        .pick_from_group(decode, Stage::Decode, peers, request)
                        .await
                    {
                        Ok(decode) => break (prefill, Some(decode)),
                        // A full group leaves the other version groups eligible.
                        Err((
                            _,
                            PickError::NoCandidates
                            | PickError::NoAdmissibleEngine(_)
                            | PickError::AdmissionRejected(_),
                        )) if prefills.iter().any(|p| p.version_group() != group) => {
                            prefills.retain(|p| p.version_group() != group);
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
        };
        Ok(BucketPick { prefill, decode })
    }

    /// Scope the request to this bucket and role, then ask the group for one engine.
    async fn pick_from_group(
        &self,
        group: &EngineGroup,
        stage: Stage,
        engines: Vec<Arc<Worker>>,
        request: &BucketRequest<'_>,
    ) -> Result<Pick, (Stage, PickError)> {
        let request = PickRequest {
            model: request.model,
            stage,
            bucket: &self.id,
            input_tokens: request.input_tokens,
            total_input_tokens: request.total_input_tokens,
            expected_peak_tokens: request.expected_peak_tokens,
            prefix: request.prefix,
            token_ids: request.token_ids,
            cache_namespace: request.cache_namespace,
            session_key: request.session_key,
            routing_key: request.routing_key,
        };
        group
            .pick_from(engines, &request)
            .await
            .map_err(|error| (stage, error))
    }

    fn fits(
        &self,
        input_tokens: u64,
        uncached_tokens: Option<u64>,
        prefill_rate: Option<f64>,
        expected_peak_tokens: Option<u64>,
    ) -> bool {
        let uncached = uncached_tokens.unwrap_or(input_tokens);
        self.limits.fits(input_tokens)
            && self.uncached.fits(uncached)
            && self.prefill_seconds.fits(uncached, prefill_rate)
            && self
                .max_context_tokens
                .is_none_or(|max| expected_peak_tokens.unwrap_or(input_tokens) <= max)
    }

    /// Whether the bucket is chosen on the uncached estimate.
    pub fn bounds_uncached_tokens(&self) -> bool {
        self.uncached.min.is_some() || self.uncached.max.is_some() || self.prefill_seconds.is_set()
    }

    /// The upper bound short-to-long ordering uses: uncached tokens, or
    /// prefill microseconds (a file uses one kind for all its buckets).
    pub fn uncached_capacity(&self) -> u64 {
        match (self.uncached.max, self.prefill_seconds.max) {
            (Some(tokens), _) => tokens,
            (None, Some(seconds)) => (seconds * 1e6).min(u64::MAX as f64) as u64,
            (None, None) => u64::MAX,
        }
    }

    fn input_capacity(&self) -> u64 {
        self.limits
            .max
            .unwrap_or(u64::MAX)
            .min(self.max_context_tokens.unwrap_or(u64::MAX))
    }
}

/// Soft preference; nonpreferred buckets remain available for fallback.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloPreference {
    #[default]
    Disabled,
    SloFirst,
    BestEffort,
}

impl SloPreference {
    fn penalty(self, matches: Option<bool>) -> u8 {
        match (self, matches) {
            (Self::SloFirst, Some(false)) | (Self::BestEffort, Some(true)) => 1,
            _ => 0,
        }
    }
}

/// Model-specific bucket configuration. Selection does not inspect engine state.
#[derive(Debug, Default)]
pub struct BucketResolver {
    pub buckets: Vec<Bucket>,
    pub ttft_slo: SloPreference,
    pub tps_slo: SloPreference,
    /// What makes an engine of a spill target idle.
    pub idle: IdleSpec,
    /// Each engine's smoothed prefill rate, for queue depths.
    pub prefill_rates: PrefillRates,
    /// The pool's agreed tokens per minute, checked before any bucket.
    pub tpm: Option<Arc<TpmLimiter>>,
    /// Set under `thresholds: prefill_seconds`: the prefill rate used until
    /// the engines report one.
    pub prefill_seconds_reference: Option<f64>,
    /// Set when a bucket bounds uncached tokens; estimates them per session.
    pub estimator: Option<Arc<SessionEstimator>>,
}

impl BucketResolver {
    /// Fails if any bucket installs a policy on a stage it cannot serve.
    pub fn new(buckets: Vec<Bucket>) -> Result<Self, PickError> {
        for bucket in &buckets {
            bucket.validate()?;
        }
        Ok(Self {
            buckets,
            ..Self::default()
        })
    }

    /// The rate that converts uncached tokens into prefill seconds: the mean
    /// measured rate of the ready prefill engines, or the reference rate
    /// before any reports. `None` unless buckets are bounded in seconds.
    pub fn prefill_rate_for_seconds(
        &self,
        workers: &WorkerRegistry,
        model: &ModelId,
    ) -> Option<f64> {
        let reference = self.prefill_seconds_reference?;
        let mut seen = HashSet::new();
        let measured: Vec<f64> = (self.buckets.iter())
            .flat_map(|bucket| bucket.ready_prefill_engines(workers, model))
            .filter(|engine| seen.insert(engine.url.clone()))
            .filter_map(|engine| self.prefill_rates.rate(&engine.url))
            .collect();
        if measured.is_empty() {
            Some(reference)
        } else {
            Some(measured.iter().sum::<f64>() / measured.len() as f64)
        }
    }

    pub fn bucket(&self, id: &str) -> Option<&Bucket> {
        self.buckets.iter().find(|bucket| bucket.id == id)
    }

    /// Place a request `home` cannot on an idle engine of one of its spill
    /// targets, in their listed order; the target bucket's id with the pick.
    pub async fn spill(
        &self,
        home: &Bucket,
        workers: &WorkerRegistry,
        load: &EngineReportedLoadSnapshot,
        request: &BucketRequest<'_>,
    ) -> Option<(BucketPick, &Bucket)> {
        let idle = |engine: &Worker| self.idle.is_idle(engine, load);
        for target in &home.spill_to {
            let Some(bucket) = self.bucket(&target.bucket) else {
                continue;
            };
            if let Ok(pick) = bucket.pick_engines_where(workers, request, &idle).await {
                target
                    .spilled
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some((pick, bucket));
            }
        }
        None
    }

    /// Whether any bucket reads request tokens; startup rejects this under `--no-tokenizer`.
    pub fn needs_request_tokens(&self) -> bool {
        self.buckets.iter().any(Bucket::needs_request_tokens)
    }

    /// Return all length-compatible buckets, ordered by unmet SLO preferences,
    /// then input capacity, rank, and ID. Both preferences have equal weight.
    /// The caller tries their groups in order until a complete engine selection succeeds.
    pub fn resolve(
        &self,
        input_tokens: u64,
        expected_peak_tokens: Option<u64>,
        ttft_ms: Option<u64>,
        tokens_per_second: Option<f64>,
    ) -> Result<Vec<&Bucket>, PickError> {
        self.resolve_uncached(
            input_tokens,
            None,
            expected_peak_tokens,
            ttft_ms,
            tokens_per_second,
        )
    }

    /// [`Self::resolve`] with the request's estimated uncached tokens, which
    /// buckets bounding them are matched on. Buckets with a smaller uncached
    /// bound order first, so the shortest fitting lane is the home bucket.
    pub fn resolve_uncached(
        &self,
        input_tokens: u64,
        uncached_tokens: Option<u64>,
        expected_peak_tokens: Option<u64>,
        ttft_ms: Option<u64>,
        tokens_per_second: Option<f64>,
    ) -> Result<Vec<&Bucket>, PickError> {
        self.resolve_estimate(
            input_tokens,
            uncached_tokens,
            None,
            expected_peak_tokens,
            ttft_ms,
            tokens_per_second,
        )
    }

    /// [`Self::resolve_uncached`] with the prefill rate that converts the
    /// estimate into seconds for buckets bounded in prefill seconds.
    pub fn resolve_estimate(
        &self,
        input_tokens: u64,
        uncached_tokens: Option<u64>,
        prefill_rate: Option<f64>,
        expected_peak_tokens: Option<u64>,
        ttft_ms: Option<u64>,
        tokens_per_second: Option<f64>,
    ) -> Result<Vec<&Bucket>, PickError> {
        if expected_peak_tokens.is_some_and(|tokens| tokens < input_tokens) {
            return Err(PickError::InvalidSignal(
                "expected peak tokens are below input length".into(),
            ));
        }
        if self.ttft_slo != SloPreference::Disabled && ttft_ms == Some(0) {
            return Err(PickError::InvalidSignal(
                "requested TTFT must be positive".into(),
            ));
        }
        if self.tps_slo != SloPreference::Disabled
            && tokens_per_second.is_some_and(|tps| !tps.is_finite() || tps <= 0.0)
        {
            return Err(PickError::InvalidSignal(
                "requested tokens per second must be finite and positive".into(),
            ));
        }
        let mut buckets: Vec<_> = self
            .buckets
            .iter()
            .filter(|bucket| {
                bucket.fits(
                    input_tokens,
                    uncached_tokens,
                    prefill_rate,
                    expected_peak_tokens,
                )
            })
            .collect();
        buckets.sort_by_key(|bucket| {
            let ttft_matches = ttft_ms.map(|target| {
                bucket
                    .ttft_ms
                    .is_some_and(|estimate| estimate > 0 && estimate <= target)
            });
            let tps_matches = tokens_per_second.map(|target| {
                bucket.tokens_per_second.is_some_and(|estimate| {
                    estimate.is_finite() && estimate > 0.0 && estimate >= target
                })
            });
            let penalty = self.ttft_slo.penalty(ttft_matches) + self.tps_slo.penalty(tps_matches);
            (
                penalty,
                bucket.uncached_capacity(),
                bucket.input_capacity(),
                bucket.rank,
                &bucket.id,
            )
        });
        Ok(buckets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies_reorg::power_of_two::PowerOfTwoPolicy;
    use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

    /// A bucket whose group policy reads no tokens, so only the bucket's own
    /// length fields can make it token-hungry.
    fn load_only_bucket(id: &str) -> Bucket {
        let policy: Arc<dyn Policy> =
            Arc::new(PowerOfTwoPolicy::new(EngineReportedLoadTable::new()));
        Bucket::new(id, BucketGroups::Plain(EngineGroup::new(policy)))
    }

    /// `reorg::chat_completions` gates request tokenization on this predicate, so
    /// every length field [`Bucket::fits`] reads has to keep it true. Drop one and
    /// a length-bounded bucket would silently select on the request-size estimate
    /// that [`crate::server::routes::chat`] falls back to without tokens.
    #[test]
    fn a_bucket_needs_tokens_for_every_length_field_it_selects_on() {
        type Mutate = fn(&mut Bucket);
        assert!(
            !load_only_bucket("b").needs_request_tokens(),
            "a load-only bucket selects on nothing that needs tokens"
        );
        for (field, mutate) in [
            (
                "limits.min",
                (|b: &mut Bucket| b.limits.min = Some(1)) as Mutate,
            ),
            ("limits.max", |b: &mut Bucket| b.limits.max = Some(1)),
            ("max_context_tokens", |b: &mut Bucket| {
                b.max_context_tokens = Some(1)
            }),
        ] {
            let mut bucket = load_only_bucket("b");
            mutate(&mut bucket);
            assert!(bucket.needs_request_tokens(), "{field}");
        }
    }

    fn uncached_bucket(id: &str, min: Option<u64>, max: Option<u64>) -> Bucket {
        let mut bucket = load_only_bucket(id);
        bucket.uncached = TokenLimits { min, max };
        bucket
    }

    /// The bucket is chosen on the uncached estimate, not the prompt: a long
    /// resume with a short new tail goes to the short lane.
    #[test]
    fn uncached_bounds_choose_the_bucket_from_the_estimate() {
        let resolver = BucketResolver::new(vec![
            uncached_bucket("long", Some(256_001), None),
            uncached_bucket("short", None, Some(32_000)),
            uncached_bucket("mid", Some(32_001), Some(256_000)),
        ])
        .unwrap();
        let ids = |input, uncached| {
            resolver
                .resolve_uncached(input, uncached, None, None, None)
                .unwrap()
                .iter()
                .map(|bucket| bucket.id.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(900_000, Some(3_000)), ["short"]);
        assert_eq!(ids(900_000, Some(100_000)), ["mid"]);
        assert_eq!(ids(900_000, Some(400_000)), ["long"]);
        // Without an estimate the whole prompt counts.
        assert_eq!(ids(900_000, None), ["long"]);
        assert_eq!(ids(20_000, None), ["short"]);
    }

    /// Under prefill-second bounds the same estimate lands in a different
    /// bucket as the prefill rate changes.
    #[test]
    fn prefill_second_bounds_convert_the_estimate_at_the_given_rate() {
        let seconds = |id: &str, min: Option<f64>, max: Option<f64>| {
            let mut bucket = load_only_bucket(id);
            bucket.prefill_seconds = SecondsLimits { min, max };
            bucket
        };
        let resolver = BucketResolver::new(vec![
            seconds("L", Some(1.5), None),
            seconds("S", None, Some(1.5)),
        ])
        .unwrap();
        assert!(resolver.buckets.iter().all(Bucket::bounds_uncached_tokens));
        let ids = |uncached, rate| {
            resolver
                .resolve_estimate(900_000, Some(uncached), rate, None, None, None)
                .unwrap()
                .iter()
                .map(|bucket| bucket.id.as_str())
                .collect::<Vec<_>>()
        };
        // 30,000 tokens: 1.5 s at 20,000 tokens/s, 3 s at 10,000.
        assert_eq!(ids(30_000, Some(20_000.0)), ["S", "L"]);
        assert_eq!(ids(20_000, Some(20_000.0)), ["S"]);
        assert_eq!(ids(30_000, Some(10_000.0)), ["L"]);
        assert!(ids(30_000, None).is_empty(), "bounded seconds need a rate");
    }

    #[test]
    fn the_seconds_rate_is_the_measured_mean_or_the_reference() {
        use crate::discovery::WorkerSpec;
        let workers = WorkerRegistry::default();
        for id in ["a", "b"] {
            workers
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:1"),
                    mode: Stage::Plain,
                    model_ids: vec![ModelId("m".into())],
                    ..Default::default()
                })
                .unwrap();
        }
        let mut resolver = BucketResolver::new(vec![load_only_bucket("all")]).unwrap();
        let model = ModelId("m".into());
        assert_eq!(resolver.prefill_rate_for_seconds(&workers, &model), None);
        resolver.prefill_seconds_reference = Some(5_000.0);
        assert_eq!(
            resolver.prefill_rate_for_seconds(&workers, &model),
            Some(5_000.0)
        );
        resolver.prefill_rates.observe("http://a:1", Some(30_000.0));
        resolver.prefill_rates.observe("http://b:1", Some(10_000.0));
        assert_eq!(
            resolver.prefill_rate_for_seconds(&workers, &model),
            Some(20_000.0)
        );
    }

    /// Overlapping uncached ranges order by the smaller uncached bound first.
    #[test]
    fn the_smallest_uncached_bound_is_tried_first() {
        let resolver = BucketResolver::new(vec![
            uncached_bucket("wide", None, Some(1_048_576)),
            uncached_bucket("narrow", None, Some(32_000)),
        ])
        .unwrap();
        let found = resolver
            .resolve_uncached(500_000, Some(1_000), None, None, None)
            .unwrap();
        assert_eq!(found[0].id, "narrow");
        assert!(found[0].bounds_uncached_tokens());
        assert!(!load_only_bucket("plain").bounds_uncached_tokens());
    }

    /// Spill lands only on a target engine that reports itself idle, and
    /// counts the move on the source bucket.
    #[tokio::test]
    async fn spill_takes_only_idle_engines_of_the_listed_targets() {
        use crate::discovery::WorkerSpec;
        use crate::state::load_monitor::engine_reported_load::{LoadStat, NativeCacheRankLoad};

        let workers = WorkerRegistry::default();
        for id in ["s", "l"] {
            workers
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:1"),
                    mode: Stage::Plain,
                    model_ids: vec![ModelId("m".into())],
                    ..Default::default()
                })
                .unwrap();
        }
        let load = EngineReportedLoadTable::new();
        let member = |id: &str| {
            let mut group = EngineGroup::new(Arc::new(PowerOfTwoPolicy::new(Arc::clone(&load))));
            group.worker_ids = Some([WorkerId(id.into())].into());
            group
        };
        let mut short = Bucket::new("S", BucketGroups::Plain(member("s")));
        short.uncached.max = Some(32_000);
        short.spill_to = vec![SpillTarget::new("L")];
        let mut long = Bucket::new("L", BucketGroups::Plain(member("l")));
        long.uncached.min = Some(32_001);
        let resolver = BucketResolver::new(vec![short, long]).unwrap();
        let report = |running: u64, waiting: u64| {
            load.set(
                "http://l:1",
                0,
                LoadStat {
                    num_running_reqs: running,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 1_000,
                    native_cache: Some(NativeCacheRankLoad {
                        num_waiting_uncached_tokens: waiting,
                        num_total_tokens: 0,
                        max_running_requests: 64,
                        total_prefill_uncached_tokens: 0,
                        total_prefill_busy_us: 0,
                    }),
                },
                std::time::Instant::now(),
            )
        };
        let model = ModelId("m".into());
        let request = BucketRequest {
            model: &model,
            input_tokens: 10,
            total_input_tokens: 10,
            expected_peak_tokens: None,
            prefix: None,
            token_ids: None,
            cache_namespace: CacheNamespace::NONE,
            session_key: None,
            routing_key: None,
            excluded: &[],
        };
        let home = resolver.bucket("S").unwrap();
        let spill = |home| spill_target(&resolver, home, &workers, &load, &request);
        assert_eq!(spill(home).await, None, "unreported load is not idle");
        report(5, 0);
        assert_eq!(spill(home).await, None, "too many running");
        report(0, 4_096);
        assert_eq!(spill(home).await, None, "uncached tokens waiting");
        report(1, 0);
        assert_eq!(spill(home).await, Some(("l".into(), "L".into())));
        let spilled = &home.spill_to[0].spilled;
        assert_eq!(spilled.load(std::sync::atomic::Ordering::Relaxed), 1);
        // The long bucket lists no target, so nothing ever spills from it.
        assert_eq!(spill(resolver.bucket("L").unwrap()).await, None);
    }

    async fn spill_target(
        resolver: &BucketResolver,
        home: &Bucket,
        workers: &WorkerRegistry,
        load: &EngineReportedLoadTable,
        request: &BucketRequest<'_>,
    ) -> Option<(String, String)> {
        let snapshot = load.capture_snapshot(std::time::Instant::now());
        let (pick, bucket) = resolver.spill(home, workers, &snapshot, request).await?;
        Some((pick.prefill.engine.id.0.clone(), bucket.id.clone()))
    }

    /// One bounded bucket makes every request to the model tokenize: the bucket is
    /// chosen from the token count, so it cannot be known to be irrelevant first.
    #[test]
    fn a_resolver_needs_tokens_when_any_bucket_does() {
        let mut bounded = load_only_bucket("bounded");
        bounded.limits.max = Some(1);
        assert!(!BucketResolver::new(vec![load_only_bucket("a")])
            .unwrap()
            .needs_request_tokens());
        assert!(BucketResolver::new(vec![load_only_bucket("a"), bounded])
            .unwrap()
            .needs_request_tokens());
    }
}
