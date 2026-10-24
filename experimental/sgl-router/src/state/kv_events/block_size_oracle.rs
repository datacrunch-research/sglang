// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Worker-reported KV hashing configurations. Local lookups hash separately for
//! each page-size/bigram group. Single-configuration consumers (remote indexer
//! and peer snapshots) receive no global configuration for a heterogeneous fleet.

use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use super::CacheNamespace;

/// Tri-state for the bigram flag: distinguishes "not yet reported" from an
/// established `false`, so [`BlockSizeOracle::set_bigram`] can be first-wins
/// (matching `try_set`) rather than last-writer-wins.
const BIGRAM_UNKNOWN: u8 = 0;
const BIGRAM_UNIGRAM: u8 = 1;
const BIGRAM_BIGRAM: u8 = 2;

/// Worker hashing metadata; legacy setters support callers without discovery.
#[derive(Debug, Default)]
pub struct BlockSizeOracle {
    value: AtomicU32,
    bigram: AtomicU8,
    workers: RwLock<BTreeMap<String, (u32, bool)>>,
}

/// Returned by [`BlockSizeOracle::try_set`] when the candidate disagrees
/// with the already-established value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizeMismatch {
    pub established: u32,
    pub candidate: u32,
}

impl BlockSizeOracle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Uniform fleet block size, or `None` for unknown or mixed page sizes.
    pub fn get(&self) -> Option<u32> {
        let workers = self.workers.read();
        if let Some(&(size, _)) = workers.values().next() {
            return workers
                .values()
                .all(|(other, _)| *other == size)
                .then_some(size);
        }
        let v = self.value.load(Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// Publish whether the cluster's workers use bigram (EAGLE-family) KV-block
    /// hashing. Called from `KvEventIndex::add_worker` alongside `try_set`.
    /// First-wins: the first worker establishes the mode; a later worker that
    /// disagrees is logged (not silently honored), since the query-hashing mode
    /// is process-wide and one mismatched worker would zero out cache-aware
    /// routing for the cluster.
    pub fn set_bigram(&self, is_bigram: bool) {
        let candidate = if is_bigram {
            BIGRAM_BIGRAM
        } else {
            BIGRAM_UNIGRAM
        };
        match self.bigram.compare_exchange(
            BIGRAM_UNKNOWN,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(existing) if existing == candidate => {}
            Err(existing) => {
                tracing::warn!(
                    established_bigram = existing == BIGRAM_BIGRAM,
                    worker_bigram = is_bigram,
                    "kv-events: worker hashing mode (bigram/EAGLE) disagrees with the \
                     established cluster value; keeping the first. A heterogeneous \
                     EAGLE/non-EAGLE cluster will silently never match cache for the \
                     minority workers — check that all workers run the same model.",
                );
            }
        }
    }

    /// Whether query hashing should use the bigram variant
    /// ([`super::hash::compute_block_hashes_bigram`]). Defaults to `false`
    /// until a worker reports an EAGLE-family `speculative_algorithm`.
    pub fn is_bigram(&self) -> bool {
        let workers = self.workers.read();
        if let Some(&(_, bigram)) = workers.values().next() {
            return bigram && workers.values().all(|(_, other)| *other);
        }
        self.bigram.load(Ordering::Relaxed) == BIGRAM_BIGRAM
    }

    /// The prompt's block hashes as workers publish them; `None` until the
    /// block size is known.
    pub fn block_hashes(&self, token_ids: &[u32], namespace: &CacheNamespace) -> Option<Vec<i64>> {
        let workers = self.workers.read();
        if let Some(&config) = workers.values().next() {
            if !workers.values().all(|other| *other == config) {
                return None;
            }
            return Some(namespace.block_hashes(token_ids, config.0 as usize, config.1));
        }
        drop(workers);
        let block_size = self.get()? as usize;
        Some(namespace.block_hashes(token_ids, block_size, self.is_bigram()))
    }

    /// The block size and hashing mode together. `None` until both the size
    /// and the mode are reported. `add_worker` stores them separately, and in
    /// between `is_bigram()` defaults to `false`, which would misreport a
    /// bigram fleet as unigram.
    pub fn hash_config(&self) -> Option<(u32, bool)> {
        let workers = self.workers.read();
        if let Some(&config) = workers.values().next() {
            return workers
                .values()
                .all(|other| *other == config)
                .then_some(config);
        }
        drop(workers);
        let size = self.get()?;
        match self.bigram.load(Ordering::Relaxed) {
            BIGRAM_UNKNOWN => None,
            mode => Some((size, mode == BIGRAM_BIGRAM)),
        }
    }

    pub fn register_worker(
        &self,
        url: &str,
        size: u32,
        bigram: bool,
    ) -> Result<(), BlockSizeMismatch> {
        if size == 0 {
            return Err(BlockSizeMismatch {
                established: self.get().unwrap_or(0),
                candidate: 0,
            });
        }
        let mut workers = self.workers.write();
        if let Some(&(old_size, old_bigram)) = workers.get(url) {
            if (old_size, old_bigram) != (size, bigram) {
                return Err(BlockSizeMismatch {
                    established: old_size,
                    candidate: size,
                });
            }
        }
        workers.insert(url.to_string(), (size, bigram));
        Ok(())
    }

    pub fn remove_worker(&self, url: &str) {
        self.workers.write().remove(url);
    }

    pub fn worker_configs(&self) -> BTreeMap<String, (u32, bool)> {
        self.workers.read().clone()
    }

    /// Publish a candidate block size. Returns the established value on
    /// success (idempotent: same candidate as already set is `Ok`);
    /// returns `Err(BlockSizeMismatch)` when the candidate disagrees.
    ///
    /// `candidate == 0` is rejected because 0 is reserved as the "not
    /// yet known" sentinel.
    pub fn try_set(&self, candidate: u32) -> Result<u32, BlockSizeMismatch> {
        if candidate == 0 {
            return Err(BlockSizeMismatch {
                established: self.value.load(Ordering::Relaxed),
                candidate,
            });
        }
        match self
            .value
            .compare_exchange(0, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => Ok(candidate),
            Err(existing) if existing == candidate => Ok(existing),
            Err(existing) => Err(BlockSizeMismatch {
                established: existing,
                candidate,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_workers_have_no_global_hash_configuration() {
        let oracle = BlockSizeOracle::new();
        oracle.register_worker("small", 64, false).unwrap();
        oracle.register_worker("large", 512, false).unwrap();
        assert_eq!(oracle.get(), None);
        assert_eq!(oracle.hash_config(), None);
        assert_eq!(oracle.block_hashes(&[1, 2], CacheNamespace::NONE), None);
        assert!(oracle.register_worker("small", 512, false).is_err());
        oracle.remove_worker("small");
        assert_eq!(oracle.hash_config(), Some((512, false)));
        assert!(oracle.register_worker("invalid", 0, false).is_err());
        oracle.remove_worker("large");
        assert_eq!(oracle.get(), None);
    }

    #[test]
    fn fresh_oracle_returns_none() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.get(), None);
        assert_eq!(oracle.hash_config(), None);
    }

    #[test]
    fn hash_config_requires_both_worker_properties() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.hash_config(), None);
        oracle.try_set(64).unwrap();
        assert_eq!(
            oracle.hash_config(),
            None,
            "a block size alone must not transiently imply unigram hashing",
        );
        oracle.set_bigram(true);
        assert_eq!(oracle.hash_config(), Some((64, true)));
    }

    /// An established `false` is a reported mode, not an absent one.
    #[test]
    fn hash_config_reports_an_established_unigram_fleet() {
        let oracle = BlockSizeOracle::new();
        oracle.set_bigram(false);
        assert_eq!(oracle.hash_config(), None, "no block size yet");
        oracle.try_set(32).unwrap();
        assert_eq!(oracle.hash_config(), Some((32, false)));
    }

    #[test]
    fn first_set_establishes_the_value() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn matching_set_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn bigram_flag_defaults_false_first_wins_and_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert!(
            !oracle.is_bigram(),
            "unknown (no worker reported yet) reads as non-bigram"
        );
        oracle.set_bigram(true);
        assert!(oracle.is_bigram(), "first worker establishes the mode");
        oracle.set_bigram(true); // idempotent agreement
        assert!(oracle.is_bigram());
        // Independent of block_size establishment.
        assert_eq!(oracle.get(), None);
        // First-wins: a conflicting later worker is logged, not honored.
        oracle.set_bigram(false);
        assert!(
            oracle.is_bigram(),
            "a disagreeing worker must not flip the established mode"
        );
    }

    #[test]
    fn bigram_flag_establishes_false_first_wins() {
        let oracle = BlockSizeOracle::new();
        oracle.set_bigram(false);
        assert!(!oracle.is_bigram(), "established as unigram");
        oracle.set_bigram(true); // conflicting; first (unigram) wins
        assert!(!oracle.is_bigram());
    }

    #[test]
    fn mismatching_set_fails_without_changing_state() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).unwrap();
        assert_eq!(
            oracle.try_set(128),
            Err(BlockSizeMismatch {
                established: 64,
                candidate: 128
            })
        );
        assert_eq!(
            oracle.get(),
            Some(64),
            "mismatched candidate must not overwrite established value"
        );
    }

    #[test]
    fn zero_candidate_is_rejected() {
        let oracle = BlockSizeOracle::new();
        assert!(oracle.try_set(0).is_err());
        assert_eq!(oracle.get(), None);
    }
}
