// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Observable contract for decode policies.
//!
//! Load ranking requires complete, fresh native monitor samples. Decode
//! capacity and queue admission belong to the engine, not a second policy.

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::admission::{resolve_decode, CandidateDomain, DecisionReason};
use sgl_router::policies::decode::{
    resolve_decode_with_capacity_fallback, DecodePolicy, DecodePowerOfTwoPolicy,
    DecodeSelectionContext, LegacyHostAffinityDecodePolicy,
};
use sgl_router::policies::SelectionProposal;
use sgl_router::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedSchedulingLoad,
};
use sgl_router::workers::Worker;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[tokio::test]
async fn published_kv_loads_do_not_override_or_refuse_decode_policy() {
    use sgl_router::policies::admission::{resolve_prefill_admitted, CandidateRange};
    use sgl_router::policies::decode::DecodeRoundRobinPolicy;
    use sgl_router::state::kv_events::discovery::EventConfig;
    use sgl_router::state::kv_events::KvEventIndex;
    use zeromq::SocketSend;

    let index = KvEventIndex::new();
    let workers = [worker("d0"), worker("d1")];
    let mut publishers = Vec::new();
    for worker in &workers {
        let (publisher, port) = super::zmq_helpers::make_pub_bound().await;
        index
            .add_worker(
                &worker.url,
                Some(EventConfig {
                    host: "127.0.0.1".into(),
                    port_base: port,
                    topic: String::new(),
                    block_size: 4,
                    dp_size: 1,
                    load_port_base: Some(port),
                    load_topic: Some(String::new()),
                    replay_port_base: None,
                    is_bigram: false,
                }),
            )
            .await;
        publishers.push(publisher);
    }
    let domain = CandidateDomain::global_decode(&workers);
    let policy = DecodeRoundRobinPolicy::default();
    for both_full in [false, true] {
        let started = Instant::now();
        let loads = loop {
            for (i, publisher) in publishers.iter_mut().enumerate() {
                let full = i == 0 || both_full;
                let mut payload = Vec::new();
                rmp::encode::write_array_len(&mut payload, 11).unwrap();
                rmp::encode::write_str(&mut payload, "LoadStat").unwrap();
                for value in [if i == 0 { 64 } else { 2 }, 0, 0, 1000] {
                    rmp::encode::write_u64(&mut payload, value).unwrap();
                }
                rmp::encode::write_nil(&mut payload).unwrap();
                for value in [0, if full { 1000 } else { 0 }, 64, 0, 0] {
                    rmp::encode::write_u64(&mut payload, value).unwrap();
                }
                publisher
                    .send(super::zmq_helpers::build_multipart(1, payload))
                    .await
                    .unwrap();
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let loads = index
                .engine_reported_load()
                .capture_snapshot(Instant::now());
            if workers.iter().enumerate().all(|(i, worker)| {
                loads
                    .fresh_native_cache_load_for_url(&worker.url)
                    .is_some_and(|load| {
                        load.num_total_tokens == if i == 0 || both_full { 1000 } else { 0 }
                    })
            }) {
                break loads;
            }
            assert!(
                started.elapsed().as_secs() < 5,
                "published loads did not arrive"
            );
        };
        for expected in &workers {
            let proposal = policy
                .propose(&domain, &DecodeSelectionContext::new())
                .unwrap();
            assert_eq!(proposal.primary.id, expected.id);
            let decision = resolve_decode(&domain, &proposal, 64, &loads)
                .expect("the engine queues requests even when projected KV is full");
            assert_eq!(decision.selected.id, expected.id);
            assert_eq!(decision.reason, DecisionReason::Primary);
        }
        let p2 = DecodePowerOfTwoPolicy::new()
            .propose(
                &domain,
                &DecodeSelectionContext::new().with_load_snapshot(&loads),
            )
            .unwrap();
        assert_eq!(
            p2.primary.id, workers[1].id,
            "load ranking still belongs to P2"
        );
        assert_eq!(
            resolve_decode(&domain, &p2, u64::MAX, &loads)
                .unwrap()
                .selected
                .id,
            workers[1].id
        );
        if both_full {
            assert!(
                resolve_prefill_admitted(
                    &CandidateRange::global(&workers),
                    &SelectionProposal::primary(workers[0].clone()),
                    1,
                    &loads,
                    None,
                )
                .is_none(),
                "prefill capacity admission must remain enforced"
            );
        }
    }
    index.shutdown().await;
}

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("m".into())],
        ..Default::default()
    }))
}

#[test]
fn recipe_round_robin_decode_cycles_independently_of_load_and_iteration_order() {
    use clap::Parser;
    use sgl_router::config::Cli;
    use sgl_router::policies::decode::build_decode_policy;

    let config = Cli::try_parse_from([
        "sgl-router",
        "--model-id",
        "m",
        "--no-tokenizer",
        "--worker-urls",
        "http://worker:30000",
        "--decode-policy",
        "round_robin",
    ])
    .expect("the recipe's decode policy must be accepted")
    .into_config()
    .unwrap();
    let policy = build_decode_policy(config.model.decode_policy);
    let first = worker("d0");
    let second = worker("d1");
    first.active_requests.store(100, Ordering::Relaxed);
    let context = DecodeSelectionContext::new();
    assert!(policy
        .propose(&CandidateDomain::global_decode(&[]), &context)
        .is_none());
    for i in 0..8 {
        let context = DecodeSelectionContext::new();
        let candidates = if i % 2 == 0 {
            vec![first.clone(), second.clone()]
        } else {
            vec![second.clone(), first.clone()]
        };
        let proposal = policy
            .propose(&CandidateDomain::global_decode(&candidates), &context)
            .unwrap();
        assert_eq!(proposal.primary.id.0, if i % 2 == 0 { "d0" } else { "d1" });
    }
    assert_eq!(
        policy
            .propose(
                &CandidateDomain::global_decode(std::slice::from_ref(&second)),
                &context,
            )
            .unwrap()
            .primary
            .id,
        second.id
    );
}

fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineReportedLoadSnapshot {
    EngineReportedLoadSnapshot::from_native_cache_workers(
        7,
        entries
            .iter()
            .map(|(worker, running, waiting, used, capacity)| {
                (
                    worker.url.clone(),
                    EngineReportedSchedulingLoad {
                        num_running_reqs: *running,
                        num_waiting_reqs: *waiting,
                        num_waiting_uncached_tokens: *waiting,
                        num_used_tokens: *used,
                        num_total_tokens: *used,
                        max_total_num_tokens: *capacity,
                        max_running_requests: 64,
                        prefill_throughput_tokens_per_s: None,
                        estimated_prefill_queue_ms: None,
                        captured_at: Instant::now(),
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    )
}

#[test]
fn decode_p2_proposes_a_distinct_lower_pressure_primary_and_backup() {
    let busy = worker("busy");
    let idle = worker("idle");
    busy.active_requests.store(8, Ordering::Relaxed);
    idle.active_requests.store(1, Ordering::Relaxed);
    let domain = CandidateDomain::global_decode(&[Arc::clone(&busy), Arc::clone(&idle)]);
    let ctx = DecodeSelectionContext::new();

    let proposal = DecodePowerOfTwoPolicy::new()
        .propose(&domain, &ctx)
        .expect("two decode candidates must produce a proposal");

    assert_eq!(proposal.primary.id, idle.id);
    assert_eq!(
        proposal.backup.expect("P2 keeps the other sample").id,
        busy.id
    );
}

/// With three or more decode workers, every pick is the least-loaded worker and
/// the backup the next; a random pair would miss the least-loaded one.
#[test]
fn decode_p2_ranks_every_worker_from_three_up() {
    let workers: Vec<_> = ["d0", "d1", "d2", "d3"].map(worker).into();
    for (worker, active) in workers.iter().zip([9, 7, 1, 5]) {
        worker.active_requests.store(active, Ordering::Relaxed);
    }
    let domain = CandidateDomain::global_decode(&workers);
    let policy = DecodePowerOfTwoPolicy::new();
    for _ in 0..64 {
        let proposal = policy
            .propose(&domain, &DecodeSelectionContext::new())
            .unwrap();
        assert_eq!(proposal.primary.id.0, "d2");
        assert_eq!(proposal.backup.expect("a backup").id.0, "d3");
    }

    // Published load ranks only when every worker has a fresh sample.
    let all = snapshot(&[
        (&workers[0], 1, 0, 10, 1000),
        (&workers[1], 9, 0, 10, 1000),
        (&workers[2], 8, 0, 10, 1000),
        (&workers[3], 7, 0, 10, 1000),
    ]);
    let ctx = DecodeSelectionContext::new().with_load_snapshot(&all);
    assert_eq!(policy.propose(&domain, &ctx).unwrap().primary.id.0, "d0");
    let partial = snapshot(&[(&workers[0], 1, 0, 10, 1000), (&workers[2], 8, 0, 10, 1000)]);
    let ctx = DecodeSelectionContext::new().with_load_snapshot(&partial);
    assert_eq!(policy.propose(&domain, &ctx).unwrap().primary.id.0, "d2");

    // Ties spread instead of always landing on the first worker.
    for worker in &workers {
        worker.active_requests.store(0, Ordering::Relaxed);
    }
    let mut seen = std::collections::HashSet::new();
    for _ in 0..256 {
        let proposal = policy
            .propose(&domain, &DecodeSelectionContext::new())
            .unwrap();
        seen.insert(proposal.primary.id.0.clone());
    }
    assert_eq!(seen.len(), workers.len(), "ties must not pin one worker");
}

#[test]
fn legacy_host_affinity_remains_an_explicit_single_primary_compatibility_policy() {
    let same_host = worker("host-a");
    let other_host = worker("host-b");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&same_host), other_host]);
    let ctx = DecodeSelectionContext::new().with_prefill_url("http://host-a:9999");

    let proposal = LegacyHostAffinityDecodePolicy
        .propose(&domain, &ctx)
        .expect("legacy policy selects one compatible decode worker");

    assert_eq!(proposal.primary.id, same_host.id);
    assert!(
        proposal.backup.is_none(),
        "legacy semantics do not invent a backup"
    );
}

#[test]
fn decode_admission_preserves_primary_when_projected_kv_is_full() {
    let primary = worker("primary");
    let backup = worker("backup");
    let fallback = worker("fallback");
    let domain = CandidateDomain::global_decode(&[
        Arc::clone(&primary),
        Arc::clone(&backup),
        Arc::clone(&fallback),
    ]);
    let loads = snapshot(&[
        (&primary, 4, 0, 950, 1_000),
        (&backup, 0, 0, 0, 1_000),
        (&fallback, 0, 0, 0, 1_000),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision = resolve_decode(&domain, &proposal, 64, &loads)
        .expect("decode admission belongs to the engine");

    assert_eq!(decision.selected.id, primary.id);
    assert_eq!(decision.reason, DecisionReason::Primary);
}

#[test]
fn decode_admission_does_not_replace_the_policy_pressure_ranking() {
    let primary = worker("primary");
    let backup = worker("backup");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&primary), Arc::clone(&backup)]);
    let loads = snapshot(&[(&primary, 3, 2, 900, 2_000), (&backup, 1, 0, 100, 2_000)]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision =
        resolve_decode(&domain, &proposal, 64, &loads).expect("both candidates are admitted");

    assert_eq!(decision.selected.id, primary.id);
    assert_eq!(decision.reason, DecisionReason::Primary);
}

#[test]
fn decode_all_capacity_full_preserves_primary_within_domain() {
    let primary = worker("primary");
    let backup = worker("backup");
    let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
    let domain = CandidateDomain::global_decode(&workers);
    let loads = snapshot(&[
        (&primary, 0, 0, 1_000, 1_000),
        (&backup, 0, 10, 1_000, 1_000),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision = resolve_decode_with_capacity_fallback(&domain, &proposal, 64, &loads)
        .expect("capacity exhaustion must preserve the decode policy");

    assert_eq!(decision.selected.id, primary.id);
    assert_eq!(decision.reason, DecisionReason::Primary);
}

#[test]
fn decode_rung_proposals_advance_once_per_request() {
    use sgl_router::policies::decode::DecodeRoundRobinPolicy;
    let policy = DecodeRoundRobinPolicy::default();
    let workers = vec![worker("d0"), worker("d1")];
    let domain = CandidateDomain::global_decode(&workers);
    let request = DecodeSelectionContext::new();
    for _ in 0..3 {
        assert_eq!(
            policy.propose(&domain, &request).unwrap().primary.id.0,
            "d0"
        );
    }
    assert_eq!(
        policy
            .propose(&domain, &DecodeSelectionContext::new())
            .unwrap()
            .primary
            .id
            .0,
        "d1"
    );
}
