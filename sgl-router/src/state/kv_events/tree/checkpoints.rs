// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint lookups on the tree: how deep each worker holds a prompt named
//! by a sample of its block hashes (see [`super::super::checkpoints`]).

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use super::super::BlockCheckpoints;
use super::{now_millis, HashTree, KvWorkerId, TreeState};

impl TreeState {
    /// Raise each worker's entry in `depths` to the deepest checkpoint it
    /// holds in this shard, on any tier (the same presence rule as
    /// [`Self::prefix_depths`]). Looked up by hash, not walked from the root,
    /// so a chain this shard holds without its first blocks still answers.
    fn checkpoint_depths(
        &self,
        checkpoints: &BlockCheckpoints,
        depths: &mut HashMap<KvWorkerId, u32>,
    ) {
        let now = now_millis();
        for (index, hash) in checkpoints.deepest_first() {
            let Some(ids) = self.by_hash.get(&hash) else {
                continue;
            };
            let depth = checkpoints.depth_at(index);
            for node in ids.iter().filter_map(|id| self.nodes.get(id)) {
                node.last_used.store(now, Ordering::Relaxed);
                for worker in node.workers.keys() {
                    match depths.get_mut(worker) {
                        Some(d) => *d = (*d).max(depth),
                        None => {
                            depths.insert(worker.clone(), depth);
                        }
                    }
                }
            }
        }
    }
}

impl HashTree {
    /// How many blocks of a prompt each worker holds, from a sample of the
    /// prompt's chained hashes (see [`super::super::checkpoints`]): the depth of the
    /// deepest sample the worker holds. Absent = no sample held.
    ///
    /// Unlike [`Self::prefix_depths`] this does not need the chain to be
    /// reachable from the root: a block hash names its whole prefix, so a
    /// chain the tree holds as an orphan (its first blocks were stored before
    /// this router subscribed) answers to its own depth, and so does a worker
    /// whose later blocks hang under another worker's copy of its prefix.
    /// It also reads through a block the tree lists the worker without, which
    /// only a removal that is not leaf-first could leave: see "What the depth
    /// rests on" in [`super::super::checkpoints`]. Every shard is consulted,
    /// under its own read lock, because an orphan chain lives in the shard of
    /// its first KNOWN block, not of the prompt's block 0.
    pub fn checkpoint_depths(&self, checkpoints: &BlockCheckpoints) -> HashMap<KvWorkerId, u32> {
        let mut depths = HashMap::new();
        if checkpoints.is_empty() {
            return depths;
        }
        for shard in &self.shards {
            shard.read().checkpoint_depths(checkpoints, &mut depths);
        }
        depths
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::worker;
    use super::*;

    /// Prompts that share prefixes of random length, hashed as the engine
    /// does, each stored by a random worker to a random depth, sometimes
    /// without its first blocks (an orphan). For every prompt and worker the
    /// checkpoint depth brackets the depth the engine really holds, and where
    /// the walk from the root reaches the chain it is never above the walk.
    #[test]
    fn checkpoint_depths_bracket_the_held_depth_including_orphans() {
        use crate::state::kv_events::hash::compute_block_hashes;
        use std::collections::HashMap as Map;

        const PAGE: usize = 4;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut below = |n: u64| {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
        };
        let ws: Vec<KvWorkerId> = (0..3).map(|i| worker(&format!("http://w{i}"), 0)).collect();
        let tree = HashTree::new();
        let base: Vec<u64> = (0..4000).collect();
        let mut prompts: Vec<Vec<u64>> = Vec::new();
        // `(skip, depth)` of each (prompt, worker) store: the engine holds
        // blocks `0..depth`, the tree was told `skip..depth`.
        let mut held: Map<(usize, usize), (usize, usize)> = Map::new();
        for p in 0..40 {
            let shared = below(3000) as usize;
            let len = shared + 4 + below(1000) as usize;
            let mut prompt = base[..shared].to_vec();
            prompt.extend((0..len - shared).map(|i| 100_000 + p * 10_000 + i as u64));
            let hashes = compute_block_hashes(&prompt, PAGE);
            let full = prompt.len() / PAGE;
            let w = below(ws.len() as u64) as usize;
            let depth = 1 + below(full as u64) as usize;
            // A quarter of the stores arrive without their first blocks.
            let skip = if below(4) == 0 {
                below(depth as u64) as usize
            } else {
                0
            };
            let parent = (skip > 0).then(|| hashes[skip - 1]);
            tree.insert(&ws[w], parent, &hashes[skip..depth]);
            held.insert((p as usize, w), (skip, depth));
            prompts.push(prompt);
        }
        for (p, prompt) in prompts.iter().enumerate() {
            let hashes = compute_block_hashes(prompt, PAGE);
            let full = prompt.len() / PAGE;
            let samples = BlockCheckpoints::grid(&hashes, full, 8);
            let by_checkpoint = tree.checkpoint_depths(&samples);
            let by_walk = tree.prefix_depths(None, &hashes[..full]);
            for (w, worker) in ws.iter().enumerate() {
                // Each store of this worker, as the blocks of THIS prompt it
                // covers: `[skip, depth)` of its own prompt, cut at the blocks
                // the two prompts share. The engine holds `0..depth`; the tree
                // knows `skip..depth`.
                let covered: Vec<(usize, usize)> = (0..prompts.len())
                    .filter_map(|q| held.get(&(q, w)).map(|&sd| (q, sd)))
                    .map(|(q, (skip, depth))| {
                        let common = prompt
                            .chunks(PAGE)
                            .zip(prompts[q].chunks(PAGE))
                            .take_while(|(a, b)| a.len() == PAGE && a == b)
                            .count();
                        (skip, common.min(depth))
                    })
                    .collect();
                let truth = covered.iter().map(|&(_, end)| end).max().unwrap_or(0) as u32;
                // The deepest sample inside a range the tree knows.
                let expected = samples
                    .blocks
                    .iter()
                    .rev()
                    .find(|&&b| {
                        covered
                            .iter()
                            .any(|&(s, e)| s <= b as usize && (b as usize) < e)
                    })
                    .map_or(0, |&b| b + 1);
                let lower = by_checkpoint.get(worker).copied().unwrap_or(0);
                assert_eq!(lower, expected, "prompt {p} worker {w}");
                assert!(
                    lower <= truth,
                    "a sample never overstates: {lower} > {truth}"
                );
                // A tree that knows every block of the prompt is bracketed.
                if covered.iter().all(|&(skip, _)| skip == 0) {
                    assert!(truth <= samples.upper_bound(lower), "prompt {p} worker {w}");
                }
                let walked = by_walk.get(worker).copied().unwrap_or(0) as u32;
                assert!(walked <= truth, "the walk never overstates");
                // An orphan the walk cannot reach, the checkpoints still see.
                if walked == 0 && expected > 0 {
                    assert!(lower > 0);
                }
            }
        }
    }

    /// The router subscribed after worker b stored a prompt's first 10
    /// blocks; b's next store names block 9 as its parent, which the tree
    /// holds as worker a's, so b's blocks hang under a's without b on the
    /// blocks before them. b holds the whole prefix (its engine could not
    /// have stored a continuation otherwise): the walk reads it as holding
    /// nothing, the checkpoints at its depth.
    #[test]
    fn a_worker_hung_under_another_workers_prefix_is_read_at_its_depth() {
        let (a, b) = (worker("http://a", 0), worker("http://b", 0));
        let chain: Vec<i64> = (1..=20).collect();
        let tree = HashTree::new();
        tree.insert(&a, None, &chain[..10]);
        tree.insert(&b, Some(chain[9]), &chain[10..]);
        let walked = tree.prefix_depths(None, &chain);
        assert_eq!(walked.get(&a), Some(&10));
        assert_eq!(walked.get(&b), None, "the walk stops where b is unlisted");
        let samples = BlockCheckpoints::grid(&chain, chain.len(), 4);
        let depths = tree.checkpoint_depths(&samples);
        assert_eq!(depths.get(&a), Some(&8), "block 7 is a's deepest sample");
        assert_eq!(depths.get(&b), Some(&20));
    }

    /// A block removed from the middle of a worker's chain: the tree lists
    /// the worker past it, and the checkpoints read through it. The tree
    /// cannot tell this from the case above, and an engine that evicts leaves
    /// first never publishes it; pinned so that changing it is deliberate.
    #[test]
    fn a_block_removed_mid_chain_is_read_through() {
        let w = worker("http://a", 0);
        let chain = [1i64, 2, 3, 4];
        let tree = HashTree::new();
        tree.insert(&w, None, &chain);
        tree.remove(&w, &chain[1..2]);
        assert_eq!(tree.prefix_depths(None, &chain).get(&w), Some(&1));
        let samples = BlockCheckpoints::grid(&chain, chain.len(), 4);
        assert_eq!(samples.blocks, [3]);
        assert_eq!(tree.checkpoint_depths(&samples).get(&w), Some(&4));
    }
}
