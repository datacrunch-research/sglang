// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{BlockSizeOracle, CacheNamespace, HashTree};
use crate::policies::cache_aware::{cap_matched_prefix_blocks, estimate_matched_prefix_tokens};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix lookup result from the local radix tree or remote KV indexer,
/// consumed by cache-aware routing.
pub struct PrefixLookupResult {
    pub outcome: PrefixOutcome,
    /// Remote query length; for local mixed pages this is only the largest native query.
    pub query_blocks: usize,
    /// Workers whose reported depth came from a pending stake rather than
    /// from KV events (empty when the confirmed tree decided the lookup).
    /// Routing to one of them is a "follow" of that stake.
    pub pending_winners: Vec<String>,
    /// The query's block hashes when the local tree answered, so routing can
    /// record the placement without rehashing.
    pub block_hashes: Option<Arc<[i64]>>,
    /// Native query units and optional pending placement hashes, keyed by worker URL.
    pub local_queries: BTreeMap<String, LocalPrefixQuery>,
}

#[derive(Clone, Debug)]
pub struct LocalPrefixQuery {
    block_size: u32,
    query_blocks: usize,
    hashes: Option<Arc<[i64]>>,
}

impl PrefixLookupResult {
    pub fn matched_blocks(&self, url: &str, blocks: u32) -> u32 {
        let limit = self
            .local_queries
            .get(url)
            .map_or(self.query_blocks, |query| query.query_blocks);
        cap_matched_prefix_blocks(limit, blocks)
    }

    pub fn matched_tokens(&self, input: u64, url: &str, blocks: u32) -> u64 {
        let blocks = u64::from(self.matched_blocks(url, blocks));
        match self.local_queries.get(url) {
            Some(query) => blocks
                .saturating_mul(u64::from(query.block_size))
                .min(input),
            None => estimate_matched_prefix_tokens(input, self.query_blocks, blocks as u32),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RadixTreePrefixProvider {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Confirmed-first mode, on when set: a pending stake is consulted only
    /// when it is strictly deeper than the best confirmed KV-event match, and
    /// only while the confirmed match covers less than this share of the
    /// prompt. Above it the confirmed tree already routes the prompt and its
    /// successor turns, so the pending walk and the stake write are skipped:
    /// a warm request pays nothing for a tier it cannot benefit from. `None`
    /// keeps the original behavior: pending depths are merged with the
    /// confirmed ones by maximum on every request, and every request stakes.
    pending_consult_below: Option<f64>,
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
            pending_consult_below: None,
        }
    }

    /// Switch on confirmed-first mode: consult and stake pending prefixes
    /// only when the confirmed match covers less than `ratio` of the prompt
    /// and only where a stake is strictly deeper (see `pending_consult_below`).
    pub fn with_pending_consult_below(mut self, ratio: f64) -> Self {
        self.pending_consult_below = Some(ratio);
        self
    }

    pub fn match_request_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
    ) -> Option<PrefixLookupResult> {
        self.match_tokens(tokens, namespace, true)
    }

    /// The KV-event tree alone: no pending stake is consulted or written.
    pub fn match_confirmed_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
    ) -> Option<PrefixLookupResult> {
        self.match_tokens(tokens, namespace, false)
    }

    fn match_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        include_pending: bool,
    ) -> Option<PrefixLookupResult> {
        let configs = self.block_size_oracle.worker_configs();
        if !configs.is_empty() {
            return self.match_worker_configs(tokens, namespace, configs, include_pending);
        }
        let hashes = self.block_hashes(tokens, namespace)?;

        let mut depth_by_url = BTreeMap::<String, u32>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            merge_depth(&mut depth_by_url, worker.url, depth);
        }
        let confirmed = depth_by_url.values().copied().max();
        let confirmed_by_url = depth_by_url.clone();
        // Stakes at their follower cap are already filtered out by
        // `PendingPrefixes::depths`.
        let (consult_pending, merge_pending) = self.pending_mode(
            include_pending,
            confirmed.unwrap_or(0) as f64 / hashes.len() as f64,
        );
        let pending = if consult_pending {
            self.tree.pending().depths(&hashes)
        } else {
            Vec::new()
        };
        // Confirmed-first merges a stake only where it is strictly deeper than
        // the best confirmed match, each on its own: a stake no deeper than
        // that match never becomes a candidate, even beside one that is.
        for (url, depth) in pending {
            let deeper = Some(u32::try_from(depth).unwrap_or(u32::MAX)) > confirmed;
            if merge_pending || deeper {
                merge_depth(&mut depth_by_url, url.to_string(), depth);
            }
        }
        let pending_winners: Vec<String> = depth_by_url
            .iter()
            .filter(|(url, depth)| confirmed_by_url.get(*url) < Some(depth))
            .map(|(url, _)| url.clone())
            .collect();
        let outcome = match depth_by_url.values().copied().max() {
            None => PrefixOutcome::Empty,
            Some(best_prefix_blocks) => {
                if confirmed < Some(best_prefix_blocks) {
                    self.tree.pending().record_hit();
                }
                let matches = depth_by_url
                    .into_iter()
                    .map(|(address, matched_prefix_blocks)| PrefixMatch {
                        worker_id: address.clone(),
                        address,
                        matched_prefix_blocks,
                    })
                    .collect();
                PrefixOutcome::Matched {
                    matches,
                    best_prefix_blocks,
                }
            }
        };
        // Empty is still returned so routing can record where this prompt went.
        Some(PrefixLookupResult {
            outcome,
            query_blocks: hashes.len(),
            pending_winners,
            block_hashes: consult_pending.then(|| hashes.into()),
            local_queries: BTreeMap::new(),
        })
    }

    /// Whether to walk the pending store, and whether its depths merge by
    /// maximum (`true`) or only where strictly deeper than the best confirmed
    /// match (`false`), given the confirmed match's share of the prompt.
    fn pending_mode(&self, include_pending: bool, covered: f64) -> (bool, bool) {
        let enabled = include_pending && self.tree.pending().is_enabled();
        match self.pending_consult_below {
            // Original behavior: walk the pending store on every request and
            // merge its depths with the confirmed ones by maximum.
            None => (enabled, true),
            // Confirmed-first: pay for the pending tier only when the confirmed
            // tree has not already placed this prompt (a first turn, or a turn
            // racing its predecessor's KV events), and let a stake count only
            // where it is strictly deeper than the best confirmed match, so a
            // real match that is at least as good always decides.
            Some(below) => (enabled && covered < below, false),
        }
    }

    /// Remember that `signal`'s prompt was just routed to `url`. When the
    /// placement followed `url`'s pending stake, charge that stake a follower
    /// first, so the cap sees every request the stake attracted.
    pub fn record_route(&self, signal: &PrefixLookupResult, url: &str) {
        let hashes = signal
            .local_queries
            .get(url)
            .and_then(|query| query.hashes.as_ref())
            .or(signal.block_hashes.as_ref());
        if let Some(hashes) = hashes {
            if signal.pending_winners.iter().any(|w| w == url) {
                self.tree.pending().record_follow(url, hashes);
            }
            self.tree.pending().record(url, hashes);
        }
    }

    /// `(dp_rank, cached prefix blocks)` for each rank of `worker_url`.
    pub fn rank_depths(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        worker_url: &str,
    ) -> Vec<(u32, usize)> {
        let configs = self.block_size_oracle.worker_configs();
        let hashes = if configs.is_empty() {
            self.block_hashes(tokens, namespace)
        } else {
            configs
                .get(worker_url)
                .map(|&(size, bigram)| full_block_hashes(tokens, namespace, size, bigram))
        };
        let Some(hashes) = hashes else {
            return Vec::new();
        };
        self.tree
            .prefix_depths(None, &hashes)
            .into_iter()
            .filter(|(worker, _)| worker.url == worker_url)
            .map(|(worker, depth)| (worker.dp_rank, depth))
            .collect()
    }

    /// Mixed native page sizes: each worker group is queried in its own block
    /// units, and depths are compared across groups in tokens.
    fn match_worker_configs(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        configs: BTreeMap<String, (u32, bool)>,
        include_pending: bool,
    ) -> Option<PrefixLookupResult> {
        let mut groups = BTreeMap::<(u32, bool), Vec<String>>::new();
        for (url, config) in configs {
            groups.entry(config).or_default().push(url);
        }
        let groups: Vec<_> = groups
            .into_iter()
            .map(|((size, bigram), urls)| {
                let hashes = full_block_hashes(tokens, namespace, size, bigram);
                (size, urls, hashes)
            })
            .collect();
        let query_blocks = groups.iter().map(|(_, _, h)| h.len()).max().unwrap_or(0);
        if query_blocks == 0 {
            return None;
        }
        let query_tokens = groups
            .iter()
            .map(|(size, _, h)| (h.len() as u64).saturating_mul(u64::from(*size)))
            .max()
            .unwrap_or(0)
            .max(1);

        let mut depths = BTreeMap::new();
        let mut confirmed_tokens = 0u64;
        for (size, urls, hashes) in &groups {
            for (worker, depth) in self.tree.prefix_depths(None, hashes) {
                if urls.contains(&worker.url) {
                    confirmed_tokens =
                        confirmed_tokens.max((depth as u64).saturating_mul(u64::from(*size)));
                    merge_depth(&mut depths, worker.url, depth);
                }
            }
        }
        let confirmed_by_url = depths.clone();
        let (consult_pending, merge_pending) = self.pending_mode(
            include_pending,
            confirmed_tokens as f64 / query_tokens as f64,
        );

        let mut pending_tokens = 0u64;
        let mut local_queries = BTreeMap::new();
        for (size, urls, hashes) in groups {
            if consult_pending {
                for (url, depth) in self.tree.pending().depths(&hashes) {
                    let staked = (depth as u64).saturating_mul(u64::from(size));
                    if urls.iter().any(|worker| worker == url.as_ref())
                        && (merge_pending || staked > confirmed_tokens)
                    {
                        pending_tokens = pending_tokens.max(staked);
                        merge_depth(&mut depths, url.to_string(), depth);
                    }
                }
            }
            let query = LocalPrefixQuery {
                block_size: size,
                query_blocks: hashes.len(),
                hashes: consult_pending.then(|| Arc::<[i64]>::from(hashes)),
            };
            for url in urls {
                local_queries.insert(url, query.clone());
            }
        }
        if pending_tokens > confirmed_tokens {
            self.tree.pending().record_hit();
        }
        let pending_winners: Vec<String> = depths
            .iter()
            .filter(|(url, depth)| confirmed_by_url.get(*url) < Some(depth))
            .map(|(url, _)| url.clone())
            .collect();
        let outcome = match depths.values().copied().max() {
            None => PrefixOutcome::Empty,
            Some(best_prefix_blocks) => PrefixOutcome::Matched {
                matches: depths
                    .into_iter()
                    .map(|(address, matched_prefix_blocks)| PrefixMatch {
                        worker_id: address.clone(),
                        address,
                        matched_prefix_blocks,
                    })
                    .collect(),
                best_prefix_blocks,
            },
        };
        Some(PrefixLookupResult {
            outcome,
            query_blocks,
            pending_winners,
            block_hashes: None,
            local_queries,
        })
    }

    fn block_hashes(&self, tokens: &[u32], namespace: &CacheNamespace) -> Option<Vec<i64>> {
        let hashes = self.block_size_oracle.block_hashes(tokens, namespace)?;
        (!hashes.is_empty()).then_some(hashes)
    }
}

fn merge_depth(depth_by_url: &mut BTreeMap<String, u32>, url: String, depth: usize) {
    let depth = u32::try_from(depth).unwrap_or(u32::MAX);
    depth_by_url
        .entry(url)
        .and_modify(|current| *current = (*current).max(depth))
        .or_insert(depth);
}

fn full_block_hashes(
    tokens: &[u32],
    namespace: &CacheNamespace,
    size: u32,
    bigram: bool,
) -> Vec<i64> {
    let logical = tokens.len().saturating_sub(usize::from(bigram));
    let full = logical / size as usize * size as usize;
    if full == 0 {
        return Vec::new();
    }
    namespace.block_hashes(&tokens[..full + usize::from(bigram)], size as usize, bigram)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::tree::KvWorkerId;
    use crate::state::kv_events::wire::BlockStored;
    use std::time::Duration;

    const BLOCK: u32 = 2;

    fn provider(ttl_ms: u64, max_followers: u64) -> RadixTreePrefixProvider {
        let tree = Arc::new(HashTree::new());
        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK).expect("block size");
        tree.pending().enable(Duration::from_millis(ttl_ms));
        tree.pending().set_max_followers(max_followers);
        RadixTreePrefixProvider::new(tree, oracle)
    }

    fn tokens(blocks: usize) -> Vec<u32> {
        (1..=(blocks * BLOCK as usize) as u32).collect()
    }

    /// Publish the first `blocks` blocks of `tokens` for `url` as KV events.
    fn confirm(provider: &RadixTreePrefixProvider, url: &str, tokens: &[u32], blocks: usize) {
        let hashes = provider
            .block_hashes(tokens, &CacheNamespace::default())
            .expect("hashes");
        let event = BlockStored {
            block_hashes: hashes[..blocks].to_vec(),
            parent_block_hash: None,
            token_ids: tokens[..blocks * BLOCK as usize].to_vec(),
            block_size: BLOCK,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            is_bigram: false,
            medium: None,
        };
        provider
            .tree
            .store_event(&KvWorkerId::new(url.to_owned(), 0), &event);
    }

    fn lookup(provider: &RadixTreePrefixProvider, tokens: &[u32]) -> PrefixLookupResult {
        provider
            .match_request_tokens(tokens, &CacheNamespace::default())
            .expect("lookup")
    }

    fn depths(signal: &PrefixLookupResult) -> Vec<(String, u32)> {
        match &signal.outcome {
            PrefixOutcome::Matched { matches, .. } => {
                let mut v: Vec<_> = matches
                    .iter()
                    .map(|m| (m.address.clone(), m.matched_prefix_blocks))
                    .collect();
                v.sort();
                v
            }
            _ => Vec::new(),
        }
    }

    #[test]
    fn confirmed_match_wins_over_a_shallower_or_equal_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // A stake on b covering 4 blocks: shallower than a's confirmed 6.
        let probe = lookup(&provider, &prompt[..4 * BLOCK as usize]);
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().hits(), 0);
        // Equal depth on another worker: still the confirmed tree decides.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        provider.record_route(&probe, "http://c");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn without_the_switch_pending_merges_by_maximum_as_before() {
        let provider = provider(60_000, 0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // An equal-depth stake on another worker stays a candidate, as it did
        // before confirmed-first mode existed, and every request stakes.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        assert!(probe.block_hashes.is_some());
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 6), ("http://b".into(), 6)]
        );
        assert!(signal.block_hashes.is_some(), "every request stakes");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "not deeper than confirmed"
        );
    }

    #[test]
    fn deeper_stake_is_consulted_and_marked_as_pending_winner() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let signal = lookup(&provider, &prompt);
        provider.record_route(&signal, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn only_strictly_deeper_stakes_join_the_candidates() {
        let first = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&first, "http://a", &prompt, 2);
        // c stakes 3 blocks (deeper than a's 2), d stakes 2 (equal) and e 1.
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&first, &prompt[..blocks * BLOCK as usize]);
            first.record_route(&probe, url);
        }
        let signal = lookup(&first, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 2), ("http://c".into(), 3)],
            "the stakes no deeper than a's match are not candidates"
        );
        assert_eq!(signal.pending_winners, vec!["http://c".to_string()]);
        // Without the switch, every stake merges by maximum as before.
        let plain = provider(60_000, 0);
        confirm(&plain, "http://a", &prompt, 2);
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&plain, &prompt[..blocks * BLOCK as usize]);
            plain.record_route(&probe, url);
        }
        let signal = lookup(&plain, &prompt);
        assert_eq!(
            depths(&signal),
            [
                ("http://a".into(), 2),
                ("http://c".into(), 3),
                ("http://d".into(), 2),
                ("http://e".into(), 1)
            ]
        );
    }

    #[test]
    fn follower_cap_falls_back_to_the_confirmed_tree() {
        let provider = provider(60_000, 2).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let first = lookup(&provider, &prompt);
        provider.record_route(&first, "http://b");
        // Two followers may ride the stake.
        for _ in 0..2 {
            let signal = lookup(&provider, &prompt);
            assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
            provider.record_route(&signal, "http://b");
        }
        assert_eq!(provider.tree.pending().followed(), 2);
        // The third sees only the confirmed tree: a at 3 blocks, b gone.
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 3)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().capped(), 1);
        // Once b's events arrive, b is a confirmed match again.
        confirm(&provider, "http://b", &prompt, 10);
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn warm_prompt_skips_the_pending_tier_and_its_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // A cold prompt (nothing confirmed) consults the tier and is staked.
        let cold = lookup(&provider, &prompt);
        assert!(cold.block_hashes.is_some(), "a cold prompt is staked");
        provider.record_route(&cold, "http://b");
        // Once a confirmed match covers at least half the prompt, the deeper
        // stake on b is not even consulted, and the prompt writes no stake.
        confirm(&provider, "http://a", &prompt, 6);
        let warm = lookup(&provider, &prompt);
        assert_eq!(depths(&warm), [("http://a".into(), 6)]);
        assert!(warm.pending_winners.is_empty());
        assert!(warm.block_hashes.is_none(), "a warm prompt pays no stake");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "the tier was never walked"
        );
    }

    #[test]
    fn cold_prompt_below_the_gate_still_uses_the_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // Only the shared head is confirmed: 2 of 10 blocks, below the gate.
        confirm(&provider, "http://a", &prompt, 2);
        let first = lookup(&provider, &prompt);
        assert!(first.block_hashes.is_some());
        provider.record_route(&first, "http://b");
        let next = lookup(&provider, &prompt);
        assert_eq!(next.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(
            depths(&next),
            [("http://a".into(), 2), ("http://b".into(), 10)]
        );
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn routing_to_a_confirmed_worker_is_not_a_follow() {
        let provider = provider(60_000, 1);
        let prompt = tokens(4);
        confirm(&provider, "http://a", &prompt, 4);
        for _ in 0..3 {
            let signal = lookup(&provider, &prompt);
            assert!(signal.pending_winners.is_empty());
            provider.record_route(&signal, "http://a");
        }
        assert_eq!(provider.tree.pending().followed(), 0);
        assert_eq!(provider.tree.pending().capped(), 0);
    }

    /// Mixed native page sizes in confirmed-first mode: the coverage gate and
    /// the strictly-deeper rule compare tokens, since block counts of
    /// different page sizes are not comparable.
    #[test]
    fn mixed_pages_confirmed_first_compares_tokens() {
        let mixed = |below: f64| {
            let tree = Arc::new(HashTree::new());
            let oracle = BlockSizeOracle::new();
            oracle.register_worker("http://small", 2, false).unwrap();
            oracle.register_worker("http://large", 4, false).unwrap();
            tree.pending().enable(Duration::from_secs(60));
            RadixTreePrefixProvider::new(tree, oracle).with_pending_consult_below(below)
        };
        let ns = CacheNamespace::default();
        let prompt: Vec<u32> = (1..=16).collect();
        let small_hashes = full_block_hashes(&prompt, &ns, 2, false);
        let large_hashes = full_block_hashes(&prompt, &ns, 4, false);
        let confirm_small = |provider: &RadixTreePrefixProvider, blocks: usize| {
            provider.tree.insert(
                &KvWorkerId::new("http://small".into(), 0),
                None,
                &small_hashes[..blocks],
            );
        };
        // A cold lookup of the first `tokens` tokens, routed to large: large's stake.
        let stake_large = |provider: &RadixTreePrefixProvider, tokens: usize| {
            let probe = provider
                .match_request_tokens(&prompt[..tokens], &ns)
                .unwrap();
            provider.record_route(&probe, "http://large");
        };

        // large stakes 2 blocks of 4 (8 tokens); small holds 3 blocks of 2 (6 tokens).
        let provider = mixed(1.0);
        stake_large(&provider, 8);
        confirm_small(&provider, 3);
        let signal = provider.match_request_tokens(&prompt, &ns).unwrap();
        // 2 blocks are fewer than 3, but 8 tokens are more than 6: the stake counts.
        assert_eq!(
            depths(&signal),
            [("http://large".into(), 2), ("http://small".into(), 3)]
        );
        assert_eq!(signal.pending_winners, vec!["http://large".to_string()]);
        assert_eq!(provider.tree.pending().hits(), 1);
        assert_eq!(signal.matched_tokens(16, "http://large", 2), 8);
        assert_eq!(signal.matched_tokens(16, "http://small", 3), 6);
        // The confirmed-only lookup ignores the stake.
        let confirmed = provider.match_confirmed_tokens(&prompt, &ns).unwrap();
        assert_eq!(depths(&confirmed), [("http://small".into(), 3)]);
        assert!(confirmed.pending_winners.is_empty());

        // An equal-token stake (1 block of 4 against 2 blocks of 2) does not count.
        let provider = mixed(1.0);
        stake_large(&provider, 4);
        confirm_small(&provider, 2);
        let signal = provider.match_request_tokens(&prompt, &ns).unwrap();
        assert_eq!(depths(&signal), [("http://small".into(), 2)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().hits(), 0);

        // 10 of 16 tokens confirmed is past a 0.5 gate: no walk and no stake written.
        let provider = mixed(0.5);
        confirm_small(&provider, 5);
        let warm = provider.match_request_tokens(&prompt, &ns).unwrap();
        assert!(warm.local_queries.values().all(|q| q.hashes.is_none()));
        provider.record_route(&warm, "http://large");
        assert!(provider.tree.pending().depths(&large_hashes).is_empty());
        assert_eq!(provider.tree.pending().hits(), 0);
    }
}
