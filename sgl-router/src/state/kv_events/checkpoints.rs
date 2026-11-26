// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Sparse prefix queries: a prompt named by a sample of its block hashes.
//!
//! Block hashes chain ([`super::hash`]): block `k`'s hash digests block
//! `k - 1`'s full digest and block `k`'s tokens, so it names the whole prefix
//! `0..=k`, not one page. And an engine's radix cache only ever evicts
//! leaves, so a worker that holds block `k` holds every block before it. The
//! depth a worker holds a prompt to is therefore the deepest block whose hash
//! it holds, and finding it needs no walk from the root: a sample of the
//! prompt's hashes, each looked up on its own, bounds it to the gap between
//! two samples.
//!
//! That lets an upstream (the Global Router) ask how much of a 500K-token
//! prompt a cell holds by sending a few hundred hashes instead of the token
//! ids, and lets the answer survive a tree that missed the start of a chain
//! (a router that subscribed after its engines stored the prompt's first
//! blocks): the walk from the root stops at the missing block, a lookup by
//! hash does not.
//!
//! # What the depth rests on
//!
//! Two things the tree cannot check: an engine removes a block from every
//! tier only once it holds no block after it (`RadixCache` and the unified
//! tree both evict leaves; a block demoted to host is still held), and a
//! worker's events arrive in order. So a worker the tree lists on a block is
//! read as holding every block before it, listed or not. A worker can be
//! unlisted on its own prefix without having lost it: its first blocks were
//! stored before this router subscribed, and its later ones hang under
//! another worker's copy of them. The walk from the root reads that worker as
//! holding nothing; a lookup by hash reads it at its depth. The same state
//! left by a removal from the middle of a chain would be read through too,
//! and overstated, but the tree cannot tell the two apart, and an engine that
//! removes leaves first never leaves the second.
//!
//! The upper bound a sample puts on a depth
//! ([`BlockCheckpoints::upper_bound`]) rests on the tree having seen the
//! blocks stored: a worker can hold past it where the tree never saw the
//! next sample stored.

/// How a fleet's engines hash KV blocks: what a caller's sample must be
/// hashed in for its hashes to name the blocks the engines published. A cell
/// answers checkpoint queries in its engines' scheme and refuses others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct HashScheme {
    pub page_size: u32,
    pub bigram: bool,
}

impl HashScheme {
    pub fn new(page_size: u32, bigram: bool) -> Self {
        Self { page_size, bigram }
    }
}

/// A sample of one prompt's full-block hashes, ascending by block index.
///
/// On the wire it is two parallel arrays, which is the compact JSON form:
/// `{"blocks": [15, 31, 47, …], "hashes": [h15, h31, h47, …]}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockCheckpoints {
    /// Zero-based block indices, strictly ascending.
    pub blocks: Vec<u32>,
    /// `hashes[i]` is the chained hash of block `blocks[i]`.
    pub hashes: Vec<i64>,
    /// Full blocks in the prompt: the depth the last sample stands for when
    /// it is the prompt's last full block, and the upper bound past it.
    pub total_blocks: u32,
}

/// The `checkpoints` of a prefix-match query: a prompt's samples and the
/// hashing they were computed with, which the cell must share to use them.
/// On the wire: `{"page_size", "bigram", "blocks", "hashes", "total_blocks"}`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CheckpointQuery {
    #[serde(flatten)]
    pub scheme: HashScheme,
    #[serde(flatten)]
    pub samples: BlockCheckpoints,
}

/// Why a [`BlockCheckpoints`] from the wire cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointError {
    #[error("blocks and hashes differ in length ({blocks} vs {hashes})")]
    LengthMismatch { blocks: usize, hashes: usize },
    #[error("blocks are not strictly ascending at index {at}")]
    NotAscending { at: usize },
    #[error("block {block} is past total_blocks {total}")]
    PastEnd { block: u32, total: u32 },
    #[error("{count} checkpoints exceed the limit of {limit}")]
    TooMany { count: usize, limit: usize },
}

/// The most checkpoints one query may carry. A sample every 1024 tokens
/// ([`BlockCheckpoints::grid`]) reaches it at about 1M tokens, past which the
/// stride doubles; the limit keeps one request from making a cell do
/// unbounded lookups.
pub const MAX_CHECKPOINTS: usize = 1024;

impl BlockCheckpoints {
    /// Samples a prompt's chained `hashes` (one per block, as
    /// [`super::CacheNamespace::block_hashes`] returns them) whose prompt has
    /// `full_blocks` complete pages: the last block of every `stride` blocks,
    /// and the last full block. The partial last page is never sampled: an
    /// engine caches whole pages only.
    ///
    /// The Global Router builds the same grid in its own code; tests and
    /// benches here build it with this. The points are set by the
    /// stride, not by the prompt's length: a conversation's next turn samples
    /// the same blocks and more, and the lower bound is short of the truth by
    /// under one stride at any length. Past [`MAX_CHECKPOINTS`] the stride
    /// doubles until the sample fits, and a multiple of twice the stride is a
    /// multiple of the stride. An upstream that caches the chain may add the
    /// last block of the conversation's previous turn: a cell whose router
    /// lost the start of the chain still holds the chain the engine stored
    /// for that turn.
    pub fn grid(hashes: &[i64], full_blocks: usize, stride: usize) -> Self {
        let full_blocks = full_blocks.min(hashes.len());
        let mut stride = stride.max(1);
        // The grid's points, the last full block among them:
        // ceil(full_blocks / stride).
        while full_blocks.div_ceil(stride) > MAX_CHECKPOINTS {
            stride *= 2;
        }
        let mut blocks: Vec<u32> = (stride - 1..full_blocks)
            .step_by(stride)
            .map(|b| b as u32)
            .collect();
        if full_blocks > 0 && blocks.last() != Some(&(full_blocks as u32 - 1)) {
            blocks.push(full_blocks as u32 - 1);
        }
        let hashes = blocks.iter().map(|&b| hashes[b as usize]).collect();
        Self {
            blocks,
            hashes,
            total_blocks: full_blocks as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Checks what arrived from the wire.
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.blocks.len() != self.hashes.len() {
            return Err(CheckpointError::LengthMismatch {
                blocks: self.blocks.len(),
                hashes: self.hashes.len(),
            });
        }
        if self.blocks.len() > MAX_CHECKPOINTS {
            return Err(CheckpointError::TooMany {
                count: self.blocks.len(),
                limit: MAX_CHECKPOINTS,
            });
        }
        if let Some(at) = self.blocks.windows(2).position(|w| w[0] >= w[1]) {
            return Err(CheckpointError::NotAscending { at: at + 1 });
        }
        if let Some(&block) = self.blocks.last().filter(|&&b| b >= self.total_blocks) {
            return Err(CheckpointError::PastEnd {
                block,
                total: self.total_blocks,
            });
        }
        Ok(())
    }

    /// `(index, hash)` pairs, deepest first, so a lookup that keeps each
    /// worker's first hit (as the pending stakes do) has its depth.
    pub fn deepest_first(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        self.hashes.iter().copied().enumerate().rev()
    }

    /// The depth, in blocks, that holding sample `index` proves.
    pub fn depth_at(&self, index: usize) -> u32 {
        self.blocks[index] + 1
    }

    /// The deepest a worker whose deepest held sample is `depth` blocks can
    /// hold: one block short of the next sample, or the whole prompt past
    /// the last one. `depth == 0` (no sample held) is bounded by the first.
    /// A bound on what the lookup saw: a worker that holds the next sample
    /// where the tree never saw it stored (a chain stored before the router
    /// subscribed) holds more.
    pub fn upper_bound(&self, depth: u32) -> u32 {
        self.blocks
            .iter()
            .find(|&&b| b + 1 > depth)
            .map_or(self.total_blocks, |&next| next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| 1000 + i).collect()
    }

    #[test]
    fn a_stride_of_one_samples_every_block_exactly() {
        let h = hashes(10);
        let c = BlockCheckpoints::grid(&h, 10, 1);
        assert_eq!(c.blocks, (0..10).collect::<Vec<u32>>());
        assert_eq!(c.hashes, h);
        assert_eq!(c.total_blocks, 10);
        // Exact: holding block 6 bounds the depth to 7 from both sides.
        assert_eq!(c.depth_at(6), 7);
        assert_eq!(c.upper_bound(7), 7);
    }

    #[test]
    fn a_prompt_is_sampled_every_stride_and_at_its_last_block() {
        // 500K tokens at 64 per page: 7812 full blocks, a sample every 16.
        let h = hashes(7813);
        let c = BlockCheckpoints::grid(&h, 7812, 16);
        assert!(c.validate().is_ok());
        assert_eq!(&c.blocks[..3], [15, 31, 47]);
        assert_eq!(c.len(), 7812 / 16 + 1);
        assert_eq!(*c.blocks.last().unwrap(), 7811);
        // The partial last page (block 7812) is never sampled.
        assert!(!c.blocks.contains(&7812));
        for (i, &b) in c.blocks.iter().enumerate() {
            assert_eq!(c.hashes[i], h[b as usize]);
        }
        // A longer prompt, the next turn, samples the same blocks and more.
        let longer = BlockCheckpoints::grid(&hashes(9000), 9000, 16);
        let grid = &c.blocks[..c.len() - 1];
        assert!(grid.iter().all(|b| longer.blocks.contains(b)));
    }

    #[test]
    fn a_grid_past_the_limit_doubles_its_stride() {
        // 3000 points at stride 16, 1500 at 32, 750 at 64: the first to fit.
        let full = 16 * 3000 + 5;
        let c = BlockCheckpoints::grid(&hashes(full), full, 16);
        assert!(c.validate().is_ok());
        assert_eq!(c.len(), 750 + 1);
        assert!(c.blocks[..750].iter().all(|b| (b + 1) % 64 == 0));
        assert_eq!(*c.blocks.last().unwrap() as usize, full - 1);
        assert!(BlockCheckpoints::grid(&hashes(0), 0, 16).is_empty());
    }

    #[test]
    fn a_grid_of_exactly_the_limit_keeps_its_stride() {
        // ceil(full / 16) points: 1024 for 16 * 1024 - 1 and for 16 * 1024
        // blocks, 1025 one block later, which doubles the stride.
        for (full, stride, len) in [
            (16 * MAX_CHECKPOINTS - 1, 16, MAX_CHECKPOINTS),
            (16 * MAX_CHECKPOINTS, 16, MAX_CHECKPOINTS),
            (16 * MAX_CHECKPOINTS + 1, 32, MAX_CHECKPOINTS / 2 + 1),
        ] {
            let c = BlockCheckpoints::grid(&hashes(full), full, 16);
            assert!(c.validate().is_ok());
            assert_eq!(c.len(), len, "{full} blocks");
            assert_eq!(c.blocks[0] as usize + 1, stride, "{full} blocks");
        }
    }

    #[test]
    fn bounds_bracket_the_true_depth() {
        let h = hashes(1000);
        let c = BlockCheckpoints::grid(&h, 1000, 16);
        for true_depth in 0..=1000u32 {
            // The deepest sample a worker holding `true_depth` blocks holds.
            let lower = c
                .blocks
                .iter()
                .rev()
                .find(|&&b| b < true_depth)
                .map_or(0, |&b| b + 1);
            let upper = c.upper_bound(lower);
            assert!(
                lower <= true_depth && true_depth <= upper,
                "{lower} {true_depth} {upper}"
            );
        }
    }

    #[test]
    fn validate_rejects_malformed_queries() {
        let ok = BlockCheckpoints {
            blocks: vec![0, 3],
            hashes: vec![1, 2],
            total_blocks: 4,
        };
        assert!(ok.validate().is_ok());
        let mut bad = ok.clone();
        bad.hashes.pop();
        assert!(matches!(
            bad.validate(),
            Err(CheckpointError::LengthMismatch { .. })
        ));
        let mut bad = ok.clone();
        bad.blocks = vec![3, 3];
        assert!(matches!(
            bad.validate(),
            Err(CheckpointError::NotAscending { at: 1 })
        ));
        let mut bad = ok.clone();
        bad.total_blocks = 3;
        assert!(matches!(
            bad.validate(),
            Err(CheckpointError::PastEnd { .. })
        ));
    }

    /// The contract with the engines and with the routers that send these
    /// queries (the Global Router carries the same files): every block hash
    /// in `tests/fixtures/kv-hash.json`, which the engine's own code computed
    /// (narrow and wide pages, bigram or not, salted or not), is what
    /// [`super::super::CacheNamespace::block_hashes`] gives.
    #[test]
    fn the_engines_hashes() {
        use super::super::CacheNamespace;
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/kv-hash.json")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 16);
        for case in cases {
            let namespace = CacheNamespace {
                cache_salt: case["cache_salt"].as_str().map(String::from),
                ..CacheNamespace::default()
            };
            let tokens: Vec<u64> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap())
                .collect();
            let page = case["page_size"].as_u64().unwrap() as usize;
            let expected: Vec<i64> = case["hashes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h.as_i64().unwrap())
                .collect();
            assert_eq!(
                namespace.block_hashes(&tokens, page, case["bigram"] == true),
                expected,
                "{}",
                case["name"]
            );
        }
    }

    /// The query of `tests/fixtures/kv-hash-checkpoint-query.json`, which
    /// the Global Router sends for its prompt, is read as the router writes
    /// it, and answered as the file says: the deepest sample a worker holds,
    /// and one block short of the next sample.
    #[test]
    fn the_fixture_query_is_answered_as_its_file_says() {
        use super::super::tree::KvWorkerId;
        use super::super::{CacheNamespace, HashTree};
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/kv-hash-checkpoint-query.json"
        ))
        .unwrap();
        let query: CheckpointQuery = serde_json::from_value(fixture["query"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&query).unwrap(), fixture["query"]);
        query.samples.validate().unwrap();
        let prompt = &fixture["prompt"];
        assert_eq!(
            query.scheme,
            HashScheme::new(prompt["page_size"].as_u64().unwrap() as u32, false)
        );
        let tokens: Vec<u64> = prompt["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap())
            .collect();
        let hashes =
            CacheNamespace::default().block_hashes(&tokens, query.scheme.page_size as usize, false);
        for cell in fixture["cells"].as_array().unwrap() {
            let held = cell["held_blocks"].as_u64().unwrap() as usize;
            let tree = HashTree::new();
            let w = KvWorkerId::new("http://engine".into(), 0);
            if held > 0 {
                tree.insert(&w, None, &hashes[..held]);
            }
            let depth = tree
                .checkpoint_depths(&query.samples)
                .get(&w)
                .copied()
                .unwrap_or(0);
            assert_eq!(
                u64::from(depth),
                cell["matched_prefix_blocks"],
                "holding {held}"
            );
            assert_eq!(
                u64::from(query.samples.upper_bound(depth)),
                cell["matched_prefix_blocks_upper"],
                "holding {held}"
            );
        }
    }
}
