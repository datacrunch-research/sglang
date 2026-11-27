// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::state::load_monitor::engine_ranking::{compare_decode_engines, compare_prefill_engines};
use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::workers::Worker;

use super::admission::{partition_admitted, AdmissionLimits, EngineAdmission};
use super::{Pick, PickDistribution, PickError, PickRequest, Policy, Stage};

/// Whether `right` wins a draw against `left` (`left` wins ties).
fn right_wins(
    stage: Stage,
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    load: &EngineReportedLoadSnapshot,
) -> bool {
    let ordering = match stage {
        Stage::Plain | Stage::Prefill => compare_prefill_engines(left, right, Some(load)),
        Stage::Decode => compare_decode_engines(left, right, Some(load)),
    };
    ordering.is_gt()
}

/// The chance each of `admitted` wins a power-of-two draw: every ordered pair
/// of distinct engines is equally likely, and the better-ranked one wins.
pub(super) fn draw_distribution(
    admitted: &[Arc<Worker>],
    stage: Stage,
    load: &EngineReportedLoadSnapshot,
) -> Vec<(Arc<Worker>, f64)> {
    if let [only] = admitted {
        return vec![(Arc::clone(only), 1.0)];
    }
    let n = admitted.len();
    let mut wins = vec![0u64; n];
    for i in 0..n {
        for j in (0..n).filter(|&j| j != i) {
            let winner = if right_wins(stage, &admitted[i], &admitted[j], load) {
                j
            } else {
                i
            };
            wins[winner] += 1;
        }
    }
    let pairs = (n * n.saturating_sub(1)) as f64;
    admitted
        .iter()
        .zip(wins)
        .filter(|(_, wins)| *wins > 0)
        .map(|(engine, wins)| (Arc::clone(engine), wins as f64 / pairs))
        .collect()
}

/// Samples two distinct admitted engines and selects the better-ranked one for
/// its stage. Engines that fail admission are never sampled, so the pick fails
/// only when no engine is admitted.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Shared application state; snapshots are local to each pick.
    engine_load: Arc<EngineReportedLoadTable>,
    pub admission: Arc<dyn EngineAdmission>,
}

impl PowerOfTwoPolicy {
    pub fn new(engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            engine_load,
            admission: Arc::new(AdmissionLimits::default()),
        }
    }
}

impl Policy for PowerOfTwoPolicy {
    fn admission(&self) -> Option<&dyn EngineAdmission> {
        Some(self.admission.as_ref())
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
            // Selection and admission use the same load observation.
            let load = self.engine_load.capture_snapshot(Instant::now());
            let (admitted, rejections) =
                partition_admitted(self.admission.as_ref(), engines, &load)?;
            let engine = match admitted.as_slice() {
                [] => return Err(PickError::NoAdmissibleEngine(rejections)),
                [engine] => Arc::clone(engine),
                _ => {
                    let mut rng = rand::thread_rng();
                    let i = rng.gen_range(0..admitted.len());
                    let mut j = rng.gen_range(0..admitted.len() - 1);
                    if j >= i {
                        j += 1;
                    }
                    let (left, right) = (&admitted[i], &admitted[j]);
                    let right_wins = right_wins(request.stage, left, right, &load);
                    Arc::clone(if right_wins { right } else { left })
                }
            };
            Ok(Pick {
                engine,
                reason: "power_of_two",
            })
        })
    }

    fn pick_distribution(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        _uncached: &HashMap<&str, u64>,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<PickDistribution, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let (admitted, rejections) = partition_admitted(self.admission.as_ref(), engines, load)?;
        if admitted.is_empty() {
            return Err(PickError::NoAdmissibleEngine(rejections));
        }
        Ok(PickDistribution::Engines(draw_distribution(
            &admitted,
            request.stage,
            load,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};
    use crate::state::load_monitor::engine_reported_load::EngineReportedSchedulingLoad;

    /// Engines named by id, each with a waiting queue of that many tokens.
    fn fleet(queues: &[(&str, u64)]) -> (Vec<Arc<Worker>>, EngineReportedLoadSnapshot) {
        let now = Instant::now();
        let engines = queues
            .iter()
            .map(|(id, _)| {
                Arc::new(Worker::new(WorkerSpec {
                    id: WorkerId((*id).into()),
                    url: format!("http://{id}:30000"),
                    ..Default::default()
                }))
            })
            .collect::<Vec<_>>();
        let loads = engines
            .iter()
            .zip(queues)
            .map(|(engine, (_, waiting))| {
                let load = EngineReportedSchedulingLoad {
                    num_running_reqs: 0,
                    num_waiting_reqs: u64::from(*waiting > 0),
                    num_waiting_uncached_tokens: *waiting,
                    num_used_tokens: 0,
                    num_total_tokens: 0,
                    max_total_num_tokens: 1_000_000,
                    max_running_requests: 64,
                    prefill_throughput_tokens_per_s: Some(10_000.0),
                    estimated_prefill_queue_ms: Some(*waiting as f64 / 10.0),
                    captured_at: now,
                };
                (engine.url.clone(), load)
            })
            .collect();
        (
            engines,
            EngineReportedLoadSnapshot::from_native_cache_workers(1, loads),
        )
    }

    fn chances(distribution: PickDistribution) -> Vec<(String, f64)> {
        let PickDistribution::Engines(chances) = distribution else {
            panic!("power of two always has a distribution");
        };
        let mut chances: Vec<_> = chances
            .into_iter()
            .map(|(engine, chance)| (engine.id.0.clone(), chance))
            .collect();
        chances.sort_by(|left, right| left.0.cmp(&right.0));
        chances
    }

    /// Over three ranked engines the best wins four of the six ordered
    /// pairs and the middle one two; the worst never wins.
    #[test]
    fn the_distribution_is_the_draw_over_ordered_pairs() {
        let (engines, load) = fleet(&[("a", 30_000), ("b", 0), ("c", 10_000)]);
        let policy = PowerOfTwoPolicy::new(EngineReportedLoadTable::new());
        let model = ModelId("m".into());
        let request = PickRequest::new(&model, Stage::Plain, 1_000);
        let distribution = policy
            .pick_distribution(&engines, &request, &HashMap::new(), &load)
            .unwrap();
        assert_eq!(
            chances(distribution),
            vec![("b".into(), 4.0 / 6.0), ("c".into(), 2.0 / 6.0)]
        );
        let alone = policy
            .pick_distribution(&engines[..1], &request, &HashMap::new(), &load)
            .unwrap();
        assert_eq!(chances(alone), vec![("a".into(), 1.0)]);
    }

    #[test]
    fn engines_that_fail_admission_are_never_drawn() {
        let (engines, load) = fleet(&[("a", 30_000), ("b", 5_000), ("c", 0)]);
        let mut policy = PowerOfTwoPolicy::new(EngineReportedLoadTable::new());
        policy.admission = Arc::new(AdmissionLimits {
            max_waiting_requests: Some(1),
            ..Default::default()
        });
        let model = ModelId("m".into());
        let request = PickRequest::new(&model, Stage::Plain, 1_000);
        let distribution = policy
            .pick_distribution(&engines, &request, &HashMap::new(), &load)
            .unwrap();
        assert_eq!(chances(distribution), vec![("c".into(), 1.0)]);
        assert!(matches!(
            policy.pick_distribution(&engines[..2], &request, &HashMap::new(), &load),
            Err(PickError::NoAdmissibleEngine(_))
        ));
        assert!(matches!(
            policy.pick_distribution(&[], &request, &HashMap::new(), &load),
            Err(PickError::NoCandidates)
        ));
    }
}
