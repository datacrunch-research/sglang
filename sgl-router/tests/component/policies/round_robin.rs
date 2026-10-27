// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::round_robin::RoundRobinPolicy;
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::workers::Worker;
use std::sync::Arc;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("m".into())],
        ..Default::default()
    }))
}

#[test]
fn cycles_through_workers() {
    let p = RoundRobinPolicy::new();
    let ws = vec![worker("a"), worker("b"), worker("c")];
    let model_id = ModelId("m".into());
    let picks: Vec<_> = (0..6)
        .filter_map(|_| p.select(&ws, &SelectionContext::new(&model_id, None)))
        .map(|w| w.id.0.clone())
        .collect();
    assert_eq!(picks, vec!["a", "b", "c", "a", "b", "c"]);
}

#[test]
fn empty_pool_returns_none() {
    let p = RoundRobinPolicy::new();
    let ws: Vec<Arc<Worker>> = vec![];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    assert!(p.select(&ws, &ctx).is_none());
}

#[test]
fn distribution_across_100_calls() {
    let p = RoundRobinPolicy::new();
    let ws = vec![worker("a"), worker("b"), worker("c")];
    let model_id = ModelId("m".into());
    let mut counts = std::collections::HashMap::new();
    for _ in 0..99 {
        let w = p
            .select(&ws, &SelectionContext::new(&model_id, None))
            .unwrap();
        *counts.entry(w.id.0.clone()).or_insert(0) += 1;
    }
    assert_eq!(counts["a"], 33);
    assert_eq!(counts["b"], 33);
    assert_eq!(counts["c"], 33);
}

#[test]
fn repeated_rung_proposals_share_one_request_rotation() {
    let policy = RoundRobinPolicy::new();
    let workers = vec![worker("a"), worker("b"), worker("c")];
    let model = ModelId("m".into());
    let request = SelectionContext::new(&model, None);
    for context in [request.clone(), request.clone(), request] {
        assert_eq!(policy.select(&workers, &context).unwrap().id.0, "a");
    }
    assert_eq!(
        policy
            .select(&workers, &SelectionContext::new(&model, None))
            .unwrap()
            .id
            .0,
        "b"
    );
}

#[test]
fn distinct_policies_and_empty_rungs_keep_independent_rotations() {
    let first = RoundRobinPolicy::new();
    let second = RoundRobinPolicy::new();
    let workers = vec![worker("a"), worker("b")];
    let model = ModelId("m".into());
    let request = SelectionContext::new(&model, None);
    assert!(first.select(&[], &request).is_none());
    assert_eq!(first.select(&workers, &request).unwrap().id.0, "a");
    assert_eq!(second.select(&workers, &request).unwrap().id.0, "a");
    assert_eq!(first.select(&workers, &request.clone()).unwrap().id.0, "a");
    let next = SelectionContext::new(&model, None);
    assert_eq!(first.select(&workers, &next).unwrap().id.0, "b");
    assert_eq!(second.select(&workers, &next).unwrap().id.0, "b");
}
