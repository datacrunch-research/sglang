// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{BlockSizeOracle, CacheNamespace, HashTree};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix lookup result from the local radix tree or remote KV indexer,
/// consumed by cache-aware routing.
pub struct PrefixLookupResult {
    pub outcome: PrefixOutcome,
    /// Remote query length; for local mixed pages this is only the largest native query.
    pub query_blocks: usize,
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
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
        }
    }

    pub fn match_request_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
    ) -> Option<PrefixLookupResult> {
        self.match_tokens(tokens, namespace, true)
    }

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
        if include_pending {
            for (url, depth) in self.tree.pending().depths(&hashes) {
                merge_depth(&mut depth_by_url, url.to_string(), depth);
            }
        }
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
            block_hashes: (include_pending && self.tree.pending().is_enabled())
                .then(|| hashes.into()),
            local_queries: BTreeMap::new(),
        })
    }

    /// Remember that `signal`'s prompt was just routed to `url`.
    pub fn record_route(&self, signal: &PrefixLookupResult, url: &str) {
        let hashes = signal
            .local_queries
            .get(url)
            .and_then(|query| query.hashes.as_ref())
            .or(signal.block_hashes.as_ref());
        if let Some(hashes) = hashes {
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
        let mut depths = BTreeMap::new();
        let mut local_queries = BTreeMap::new();
        let mut query_blocks = 0;
        let mut confirmed_tokens = 0u64;
        let mut pending_tokens = 0u64;
        for ((size, bigram), urls) in groups {
            let hashes = full_block_hashes(tokens, namespace, size, bigram);
            query_blocks = query_blocks.max(hashes.len());
            for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
                if urls.contains(&worker.url) {
                    confirmed_tokens =
                        confirmed_tokens.max((depth as u64).saturating_mul(u64::from(size)));
                    merge_depth(&mut depths, worker.url, depth);
                }
            }
            if include_pending {
                for (url, depth) in self.tree.pending().depths(&hashes) {
                    if urls.iter().any(|worker| worker == url.as_ref()) {
                        pending_tokens =
                            pending_tokens.max((depth as u64).saturating_mul(u64::from(size)));
                        merge_depth(&mut depths, url.to_string(), depth);
                    }
                }
            }
            let query = LocalPrefixQuery {
                block_size: size,
                query_blocks: hashes.len(),
                hashes: (include_pending && self.tree.pending().is_enabled())
                    .then(|| Arc::<[i64]>::from(hashes)),
            };
            for url in urls {
                local_queries.insert(url, query.clone());
            }
        }
        if query_blocks == 0 {
            return None;
        }
        if pending_tokens > confirmed_tokens {
            self.tree.pending().record_hit();
        }
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

/// Caps an indexer-supplied matched-block count at the blocks the query
/// actually asked about: a query cannot match more blocks than it contains.
/// Both the token estimate and the diverted-overlap histogram read the capped
/// value, so the clamp lives here rather than at each use.
pub(crate) fn cap_matched_prefix_blocks(query_blocks: usize, matched_prefix_blocks: u32) -> u32 {
    matched_prefix_blocks.min(u32::try_from(query_blocks).unwrap_or(u32::MAX))
}

pub(crate) fn estimate_matched_prefix_tokens(
    input_tokens: u64,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> u64 {
    let matched_prefix_blocks = u64::from(cap_matched_prefix_blocks(
        query_blocks,
        matched_prefix_blocks,
    ));
    let query_blocks = u64::try_from(query_blocks).unwrap_or(u64::MAX).max(1);
    input_tokens.saturating_mul(matched_prefix_blocks) / query_blocks
}
