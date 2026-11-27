// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefer admitted prefix owners, optionally balancing affinity against load.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};
use tokio::sync::OnceCell;

use crate::config::{AffinityConfig, AffinityMode};
use crate::state::kv_events::{
    BlockSizeOracle, CacheNamespace, PrefixLookupResult, RadixTreePrefixProvider,
};
use crate::state::load_monitor::engine_ranking::CandidateLoads;
use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::workers::Worker;

use super::admission::{
    partition_admitted, AdmissionLimits, Decision, EngineAdmission, EngineMetrics,
};
use super::affinity;
use super::power_of_two::PowerOfTwoPolicy;
use super::{Pick, PickDistribution, PickError, PickRequest, Policy, Rejection, Stage};

type Signal = Option<Arc<PrefixLookupResult>>;
type Lookup = Arc<OnceCell<Signal>>;

/// Local radix tree or remote indexer. Groups sharing an index namespace share
/// one `Arc<CacheSource>`.
pub enum CacheSource {
    Local(RadixTreePrefixProvider),
    Remote {
        index: Arc<dyn PrefixIndex>,
        block_size: Arc<BlockSizeOracle>,
    },
}

impl fmt::Debug for CacheSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Local(_) => "CacheSource::Local",
            Self::Remote { .. } => "CacheSource::Remote",
        })
    }
}

impl CacheSource {
    async fn lookup(
        &self,
        tokens: Option<&[u32]>,
        namespace: &CacheNamespace,
    ) -> Result<Signal, PickError> {
        let Some(tokens) = tokens else {
            return Ok(None);
        };
        // The KV path is 64-bit (multimodal pad sentinels); tokenizer ids widen.
        let tokens: Vec<u64> = tokens.iter().map(|&id| u64::from(id)).collect();
        let (index, oracle) = match self {
            Self::Local(provider) => {
                return Ok(provider
                    .match_request_tokens(&tokens, namespace)
                    .map(Arc::new))
            }
            Self::Remote { index, block_size } => (index, block_size),
        };
        let Some(hashes) = oracle.block_hashes(&tokens, namespace) else {
            return Ok(None);
        };
        let query_blocks = hashes.len();
        if query_blocks == 0 {
            return Ok(None);
        }
        match index.match_prefix(hashes).await {
            Ok(outcome) => Ok(Some(Arc::new(PrefixLookupResult {
                outcome,
                query_blocks,
                pending_winners: Vec::new(),
                block_hashes: None,
            }))),
            Err(PrefixIndexError::Rejected(code)) => Err(PickError::InvalidSignal(format!(
                "KV Indexer rejected the query: {code}"
            ))),
            Err(error) => {
                tracing::warn!(%error, "KV Indexer unavailable; using cache policy fallback");
                Ok(None)
            }
        }
    }
}

/// One per prepared request. Keyed by source identity so distinct index
/// namespaces never share an answer.
#[derive(Default)]
pub struct PrefixMemo {
    cells: Mutex<Vec<(Arc<CacheSource>, Lookup)>>,
}

impl fmt::Debug for PrefixMemo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PrefixMemo")
    }
}

impl PrefixMemo {
    /// The first answer any source gave, once a policy has looked it up.
    pub fn signal(&self) -> Option<Arc<PrefixLookupResult>> {
        let cells = self.cells.lock().unwrap_or_else(|e| e.into_inner());
        cells
            .iter()
            .find_map(|(_, cell)| cell.get().cloned().flatten())
    }

    /// The local tree's answer, once a policy has looked it up.
    pub fn local_signal(&self) -> Option<Arc<PrefixLookupResult>> {
        let cells = self.cells.lock().unwrap_or_else(|e| e.into_inner());
        cells
            .iter()
            .filter(|(source, _)| matches!(**source, CacheSource::Local(_)))
            .find_map(|(_, cell)| cell.get().cloned().flatten())
    }

    fn cell(&self, source: &Arc<CacheSource>) -> Lookup {
        let mut cells = self.cells.lock().unwrap_or_else(|e| e.into_inner());
        match cells.iter().find(|(s, _)| Arc::ptr_eq(s, source)) {
            Some((_, cell)) => Arc::clone(cell),
            None => {
                let cell = Arc::default();
                cells.push((Arc::clone(source), Arc::clone(&cell)));
                cell
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    engine: &'a Arc<Worker>,
    uncached_tokens: u64,
}

/// Less uncached work first, then lower prefill pressure, then worker id.
fn rank(loads: &CandidateLoads<'_>, left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_engines(left.engine, right.engine))
        .then_with(|| left.engine.id.0.cmp(&right.engine.id.0))
}

/// Each prefix holder's uncached share of `input`, by URL, before hit thresholds.
pub(crate) fn uncached_by_url(
    signal: Option<&PrefixLookupResult>,
    input: u64,
) -> HashMap<&str, u64> {
    let Some(PrefixLookupResult {
        outcome: PrefixOutcome::Matched { matches, .. },
        query_blocks,
        ..
    }) = signal.filter(|signal| signal.query_blocks > 0)
    else {
        return HashMap::new();
    };
    let query_blocks = *query_blocks as u64;
    let mut depths = HashMap::<&str, u64>::new();
    for entry in matches {
        let depth = depths.entry(entry.address.as_str()).or_default();
        *depth = (*depth).max(u64::from(entry.matched_prefix_blocks));
    }
    depths
        .into_iter()
        .map(|(url, blocks)| {
            (
                url,
                input - input.saturating_mul(blocks.min(query_blocks)) / query_blocks,
            )
        })
        .collect()
}

/// The prompt tokens of `request` that `engine` would prefill: the prefix hit
/// covers the longest prompt; the rest of the batch is uncached.
fn batch_uncached_tokens(
    request: &PickRequest<'_>,
    uncached: &HashMap<&str, u64>,
    engine: &Worker,
) -> u64 {
    let cached =
        (uncached.get(engine.url.as_str())).map_or(0, |&tokens| request.input_tokens - tokens);
    request.total_input_tokens.saturating_sub(cached)
}

/// A pick's decision before its fallback draws.
enum Plan {
    /// An admitted prefix owner, kept without asking the fallback.
    Affinity(Pick),
    /// The engines the fallback samples, and the owner the draw is balanced
    /// against, if any.
    Fallback {
        affinity: Option<Pick>,
        pool: Vec<Arc<Worker>>,
    },
}

#[derive(Debug)]
pub struct CacheAwarePolicy {
    source: Arc<CacheSource>,
    engine_load: Arc<EngineReportedLoadTable>,
    config: AffinityConfig,
    pub admission: Arc<dyn EngineAdmission>,
    /// Picks among the engines this policy's admission accepts.
    pub fallback: Arc<dyn Policy>,
}

impl CacheAwarePolicy {
    pub fn new(
        source: Arc<CacheSource>,
        engine_load: Arc<EngineReportedLoadTable>,
        config: AffinityConfig,
    ) -> Result<Self, PickError> {
        let unit = |ratio: f64| ratio.is_finite() && (0.0..=1.0).contains(&ratio);
        let valid = (1..=config.cache_candidate_max_workers)
            .contains(&config.cache_candidate_min_workers)
            && unit(config.cache_candidate_ratio)
            && config.cache_affinity_min_match_ratio.is_none_or(unit)
            && config.load_factor.is_finite()
            && config.load_factor >= 1.0
            && config.worker_queue_limit.is_none()
            && config.saturation_queue_floor.is_none();
        if !valid {
            return Err(PickError::InvalidConfiguration(
                "invalid cache candidate bounds or affinity settings".into(),
            ));
        }
        Ok(Self {
            source,
            fallback: Arc::new(PowerOfTwoPolicy::new(Arc::clone(&engine_load))),
            engine_load,
            config,
            admission: Arc::new(AdmissionLimits::default()),
        })
    }

    /// Prefix holders in `engines` (matched by exact URL) that pass the hit
    /// thresholds, ranked and bounded to the configured candidate count.
    fn candidates<'e>(
        &self,
        engines: &'e [Arc<Worker>],
        request: &PickRequest<'_>,
        uncached: &HashMap<&str, u64>,
        load: &EngineReportedLoadSnapshot,
    ) -> Vec<Candidate<'e>> {
        let config = &self.config;
        let input = request.input_tokens;
        let mut candidates: Vec<_> = engines
            .iter()
            .filter_map(|engine| {
                let uncached_tokens = *uncached.get(engine.url.as_str())?;
                let matched = input - uncached_tokens;
                let ratio = matched as f64 / input.max(1) as f64;
                let hit = matched > 0
                    && config
                        .cache_affinity_min_matched_tokens
                        .is_none_or(|min| matched >= min)
                    && config
                        .cache_affinity_min_match_ratio
                        .is_none_or(|min| ratio >= min);
                hit.then_some(Candidate {
                    engine,
                    uncached_tokens,
                })
            })
            .collect();
        let proportional = (config.cache_candidate_ratio * engines.len() as f64).ceil() as usize;
        let limit = engines
            .len()
            .min(config.cache_candidate_max_workers)
            .min(config.cache_candidate_min_workers.max(proportional));
        let loads = CandidateLoads::new(Some(load), candidates.iter().map(|c| c.engine));
        candidates.sort_by(|left, right| rank(&loads, left, right));
        candidates.truncate(limit);
        candidates
    }

    /// `None` when admitted.
    fn check(
        &self,
        engine: &Worker,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<Option<Rejection>, PickError> {
        let metrics = EngineMetrics::observe(engine, load);
        Ok(match self.admission.check(engine, &metrics)? {
            Decision::Allow => None,
            Decision::Reject(reason) => Some(Rejection {
                engine: engine.id.clone(),
                reason,
            }),
        })
    }

    /// Everything [`Policy::pick`] decides before the fallback's draw.
    fn plan(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        uncached: &HashMap<&str, u64>,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<Plan, PickError> {
        let candidates = self.candidates(engines, request, uncached, load);
        let mut rejections = Vec::new();
        let admitted = self.admit(&candidates, load, &mut rejections)?;
        let affinity = admitted.first().map(|c| Pick {
            engine: Arc::clone(c.engine),
            reason: "cache_candidate",
        });
        if self.config.mode != AffinityMode::Balanced {
            if let Some(pick) = affinity {
                return Ok(Plan::Affinity(pick));
            }
        }
        let pool: Vec<_> = engines
            .iter()
            .filter(|e| {
                !rejections.iter().any(|r| r.engine == e.id)
                    && affinity.as_ref().is_none_or(|p| p.engine.id != e.id)
            })
            .cloned()
            .collect();
        // The fallback samples only engines that pass admission.
        let (pool, pool_rejections) = partition_admitted(self.admission.as_ref(), &pool, load)?;
        rejections.extend(pool_rejections);
        if pool.is_empty() && affinity.is_none() && !rejections.is_empty() {
            return Err(PickError::NoAdmissibleEngine(rejections));
        }
        Ok(Plan::Fallback { affinity, pool })
    }

    fn admit<'e>(
        &self,
        candidates: &[Candidate<'e>],
        load: &EngineReportedLoadSnapshot,
        rejections: &mut Vec<Rejection>,
    ) -> Result<Vec<Candidate<'e>>, PickError> {
        let mut admitted = Vec::new();
        for &candidate in candidates {
            match self.check(candidate.engine, load)? {
                None => admitted.push(candidate),
                Some(rejection) => rejections.push(rejection),
            }
        }
        Ok(admitted)
    }
}

impl Policy for CacheAwarePolicy {
    fn admission(&self) -> Option<&dyn EngineAdmission> {
        Some(self.admission.as_ref())
    }

    fn supports(&self, stage: Stage) -> bool {
        stage != Stage::Decode
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            if request.stage == Stage::Decode {
                return Err(PickError::InvalidConfiguration(
                    "cache-aware selection requires a plain or prefill group".into(),
                ));
            }
            let lookup = || {
                self.source
                    .lookup(request.token_ids, request.cache_namespace)
            };
            let signal = match request.prefix {
                Some(memo) => memo
                    .cell(&self.source)
                    .get_or_try_init(lookup)
                    .await?
                    .clone(),
                None => lookup().await?,
            };
            // Capture load after remote I/O; selection and admission share it.
            let load = self.engine_load.capture_snapshot(Instant::now());
            let uncached = uncached_by_url(signal.as_deref(), request.input_tokens);
            let (affinity, pool) = match self.plan(engines, request, &uncached, &load)? {
                Plan::Affinity(pick) => return Ok(pick),
                Plan::Fallback { affinity, pool } => (affinity, pool),
            };
            let fallback = async {
                let mut pick = self.pick_fallback(&pool, request).await?;
                if !pool.iter().any(|e| Arc::ptr_eq(e, &pick.engine)) {
                    return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
                }
                pick.reason = if affinity.is_some() {
                    "affinity_load"
                } else {
                    "no_cache_candidate"
                };
                Ok(pick)
            }
            .await;
            let uncached_tokens =
                |engine: &Worker| batch_uncached_tokens(request, &uncached, engine);
            affinity::choose(&self.config, affinity, fallback, &load, uncached_tokens)
        })
    }

    fn pick_distribution(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        uncached: &HashMap<&str, u64>,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<PickDistribution, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        if request.stage == Stage::Decode {
            return Err(PickError::InvalidConfiguration(
                "cache-aware selection requires a plain or prefill group".into(),
            ));
        }
        let (affinity, pool) = match self.plan(engines, request, uncached, load)? {
            Plan::Affinity(pick) => {
                return Ok(PickDistribution::Engines(vec![(pick.engine, 1.0)]));
            }
            Plan::Fallback { affinity, pool } => (affinity, pool),
        };
        let uncached_tokens = |engine: &Worker| batch_uncached_tokens(request, uncached, engine);
        // Each outcome of the fallback's draw, balanced against the owner as `pick` does.
        let draws = match self
            .fallback
            .pick_distribution(&pool, request, uncached, load)
        {
            Ok(PickDistribution::Unsupported) => return Ok(PickDistribution::Unsupported),
            Ok(PickDistribution::Engines(draws)) => draws,
            Err(error) => {
                let pick =
                    affinity::choose(&self.config, affinity, Err(error), load, uncached_tokens)?;
                return Ok(PickDistribution::Engines(vec![(pick.engine, 1.0)]));
            }
        };
        let mut chances = Vec::with_capacity(draws.len() + 1);
        for (engine, chance) in draws {
            let alternative = Ok(Pick {
                engine,
                reason: "affinity_load",
            });
            let pick = affinity::choose(
                &self.config,
                affinity.clone(),
                alternative,
                load,
                uncached_tokens,
            )?;
            PickDistribution::add(&mut chances, pick.engine, chance);
        }
        Ok(PickDistribution::Engines(chances))
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        Some(self.fallback.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};
    use crate::state::kv_events::tree::HashTree;
    use crate::state::load_monitor::engine_reported_load::EngineReportedSchedulingLoad;

    const INPUT: u64 = 10_000;

    fn policy(mode: AffinityMode) -> CacheAwarePolicy {
        let provider =
            RadixTreePrefixProvider::new(Arc::new(HashTree::new()), BlockSizeOracle::new());
        let config = AffinityConfig {
            mode,
            load_gap: Some(1_000),
            ..Default::default()
        };
        CacheAwarePolicy::new(
            Arc::new(CacheSource::Local(provider)),
            EngineReportedLoadTable::new(),
            config,
        )
        .unwrap()
    }

    /// Engine `a` holds the whole prompt behind 30k waiting tokens; `b`, `c`
    /// and `d` hold nothing behind 0, 15k and 40k.
    fn fleet() -> (Vec<Arc<Worker>>, EngineReportedLoadSnapshot) {
        let now = Instant::now();
        let mut loads = HashMap::new();
        let engines = [("a", 30_000), ("b", 0), ("c", 15_000), ("d", 40_000)]
            .into_iter()
            .map(|(id, waiting)| {
                let engine = Arc::new(Worker::new(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    ..Default::default()
                }));
                let load = EngineReportedSchedulingLoad {
                    num_running_reqs: 0,
                    num_waiting_reqs: u64::from(waiting > 0),
                    num_waiting_uncached_tokens: waiting,
                    num_used_tokens: 0,
                    num_total_tokens: 0,
                    max_total_num_tokens: 1_000_000,
                    max_running_requests: 64,
                    prefill_throughput_tokens_per_s: Some(10_000.0),
                    estimated_prefill_queue_ms: Some(waiting as f64 / 10.0),
                    captured_at: now,
                };
                loads.insert(engine.url.clone(), load);
                engine
            })
            .collect();
        (
            engines,
            EngineReportedLoadSnapshot::from_native_cache_workers(1, loads),
        )
    }

    fn chances(policy: &CacheAwarePolicy, uncached: &HashMap<&str, u64>) -> Vec<(String, f64)> {
        let (engines, load) = fleet();
        let model = ModelId("m".into());
        let request = PickRequest::new(&model, Stage::Plain, INPUT);
        let PickDistribution::Engines(chances) = policy
            .pick_distribution(&engines, &request, uncached, &load)
            .unwrap()
        else {
            panic!("cache-aware selection has a distribution");
        };
        let mut chances: Vec<_> = chances
            .into_iter()
            .map(|(engine, chance)| (engine.id.0.clone(), chance))
            .collect();
        chances.sort_by(|left, right| left.0.cmp(&right.0));
        chances
    }

    #[test]
    fn an_admitted_owner_takes_the_request_outright() {
        let holder = HashMap::from([("http://a:30000", 0)]);
        assert_eq!(
            chances(&policy(AffinityMode::Prefer), &holder),
            vec![("a".into(), 1.0)]
        );
    }

    /// Balanced: the draw over b, c and d lands on b two times in three,
    /// which relieves the owner; c, drawn otherwise, is not light enough.
    #[test]
    fn a_balanced_owner_keeps_the_draws_that_do_not_relieve_it() {
        let holder = HashMap::from([("http://a:30000", 0)]);
        assert_eq!(
            chances(&policy(AffinityMode::Balanced), &holder),
            vec![("a".into(), 1.0 / 3.0), ("b".into(), 2.0 / 3.0)]
        );
    }

    #[test]
    fn without_an_owner_the_fallback_draws_over_every_engine() {
        assert_eq!(
            chances(&policy(AffinityMode::Prefer), &HashMap::new()),
            vec![
                ("a".into(), 2.0 / 12.0),
                ("b".into(), 6.0 / 12.0),
                ("c".into(), 4.0 / 12.0),
            ]
        );
    }
}
