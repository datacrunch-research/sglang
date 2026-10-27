// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use sgl_kv_indexer::PrefixOutcome;
use sgl_router::state::kv_events::{
    compute_block_hashes, BlockSizeOracle, CacheNamespace, HashTree, KvWorkerId,
    RadixTreePrefixProvider,
};

#[test]
fn radix_tree_reports_contiguous_prefix_depth_per_worker() {
    let tokens = [11_u32, 12, 13, 14];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();

    tree.insert(&KvWorkerId::new("http://deep".into(), 0), None, &hashes);
    tree.insert(
        &KvWorkerId::new("http://deep".into(), 1),
        None,
        &hashes[..3],
    );
    tree.insert(
        &KvWorkerId::new("http://shallow".into(), 0),
        None,
        &hashes[..2],
    );

    let signal = RadixTreePrefixProvider::new(tree, oracle)
        .match_request_tokens(&tokens, CacheNamespace::NONE)
        .expect("established local tree must produce a prefix signal");
    let PrefixOutcome::Matched {
        matches,
        best_prefix_blocks,
    } = signal.outcome
    else {
        panic!("local radix-tree hit must be normalized as a match");
    };
    let depth_by_url: HashMap<_, _> = matches
        .into_iter()
        .map(|entry| (entry.address, entry.matched_prefix_blocks))
        .collect();

    assert_eq!(signal.query_blocks, 4);
    assert_eq!(best_prefix_blocks, 4);
    assert_eq!(depth_by_url.get("http://deep"), Some(&4));
    assert_eq!(depth_by_url.get("http://shallow"), Some(&2));
}

#[tokio::test]
async fn mixed_pages_match_and_price_each_workers_native_blocks() {
    use sgl_router::config::AffinityConfig;
    use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
    use sgl_router::policies::{
        cache_aware::CacheAwarePolicy, Policy, PrefillProposal, SelectionContext,
    };
    use sgl_router::state::kv_events::{EventConfig, KvEventIndex};
    use sgl_router::workers::Worker;

    for reverse in [false, true] {
        let index = KvEventIndex::new();
        let mut registrations = [("http://small", 64), ("http://large", 512)];
        if reverse {
            registrations.reverse();
        }
        let tokens: Vec<u32> = (0..1024).collect();
        for (url, size) in registrations {
            index
                .add_worker(
                    url,
                    Some(EventConfig {
                        host: "127.0.0.1".into(),
                        port_base: 59990,
                        topic: String::new(),
                        block_size: size,
                        dp_size: 1,
                        load_port_base: None,
                        load_topic: None,
                        replay_port_base: None,
                        is_bigram: false,
                    }),
                )
                .await;
            let stored = if size == 64 {
                &tokens[..256]
            } else {
                &tokens[..]
            };
            index.tree().insert(
                &KvWorkerId::new(url.into(), 0),
                None,
                &compute_block_hashes(stored, size as usize),
            );
        }
        let workers: Vec<_> = ["http://small", "http://large"]
            .into_iter()
            .map(|url| {
                Arc::new(Worker::new(WorkerSpec {
                    id: WorkerId(url.into()),
                    url: url.into(),
                    ..Default::default()
                }))
            })
            .collect();
        let provider = RadixTreePrefixProvider::new(index.tree(), index.block_size_oracle());
        let model = ModelId("tiny".into());
        for (length, expected) in [
            (192, vec![("http://small", 192)]),
            (600, vec![("http://large", 512), ("http://small", 256)]),
        ] {
            let signal = provider
                .match_request_tokens(&tokens[..length], CacheNamespace::NONE)
                .expect("worker metadata exists");
            let ctx = SelectionContext::new(&model, None)
                .with_input_tokens(length as u64)
                .with_external_prefix(Some(&signal));
            let affinity = AffinityConfig {
                cache_affinity_min_matched_tokens: None,
                ..Default::default()
            };
            let PrefillProposal::CacheCandidates(proposal) = CacheAwarePolicy::new(affinity)
                .propose_prefill(&workers, &ctx)
                .expect("cache holder")
            else {
                panic!("must use real prefix evidence");
            };
            let actual: Vec<_> = proposal
                .candidates
                .iter()
                .map(|c| (c.worker.url.as_str(), c.matched_prefix_tokens))
                .collect();
            assert_eq!(
                actual, expected,
                "discovery order {reverse}, prompt {length}"
            );
            for candidate in proposal.candidates {
                assert_eq!(
                    candidate.uncached_tokens,
                    length as u64 - candidate.matched_prefix_tokens
                );
            }
        }
        assert_eq!(
            provider.rank_depths(&tokens[..600], CacheNamespace::NONE, "http://large"),
            vec![(0, 1)]
        );
        index
            .tree()
            .clear_worker(&KvWorkerId::new("http://large".into(), 0));
        index
            .tree()
            .pending()
            .enable(std::time::Duration::from_secs(5));
        let signal = provider
            .match_request_tokens(&tokens[..600], CacheNamespace::NONE)
            .unwrap();
        provider.record_route(&signal, "http://large");
        let predicted = provider
            .match_request_tokens(&tokens[..600], CacheNamespace::NONE)
            .unwrap();
        let PrefixOutcome::Matched { matches, .. } = &predicted.outcome else {
            panic!("pending placement");
        };
        assert!(matches.iter().any(|entry| entry.address == "http://large"
            && predicted.matched_tokens(600, &entry.address, entry.matched_prefix_blocks) == 512));
        let confirmed = provider
            .match_confirmed_tokens(&tokens[..600], CacheNamespace::NONE)
            .unwrap();
        let PrefixOutcome::Matched { matches, .. } = confirmed.outcome else {
            panic!("small worker still holds blocks");
        };
        assert!(!matches.iter().any(|entry| entry.address == "http://large"));
        index.remove_worker("http://large").await;
        assert!(provider
            .rank_depths(&tokens, CacheNamespace::NONE, "http://large")
            .is_empty());
        index.shutdown().await;
    }
}
