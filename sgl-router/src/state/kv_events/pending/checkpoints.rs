// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Pending stakes seen through a checkpoint query (see
//! [`super::super::checkpoints`]).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::super::BlockCheckpoints;
use super::{Owner, PendingPrefixes};

impl PendingPrefixes {
    /// Pending depth per worker URL from a sample of the prompt's hashes: the
    /// deepest live sample each worker has a stake on (hashes chain, so a
    /// stake on a block is a stake on its prefix).
    ///
    /// The follower cap is read on that sample, the deepest block of the
    /// prompt this lookup can name, and a lookup the cap leaves without a
    /// candidate is counted as [`Self::depths`] counts it.
    /// [`Self::record_follow`] charges the deepest block of the stake a
    /// routed prompt matched, so the two meet only where a sample falls on
    /// that block. A repeat of the staked prompt does (its last full block is
    /// always sampled) and is hidden at the cap as by the full chain. A
    /// prompt that leaves the stake between two samples (a fan-out sharing a
    /// prefix, a longer next turn) reads a counter no route charges, so the
    /// stake stays visible past its cap for as long as it lives. The cell's
    /// own routing looks the prompt up by its full chain and applies the cap
    /// there.
    pub fn checkpoint_depths(&self, checkpoints: &BlockCheckpoints) -> Vec<(Arc<str>, usize)> {
        if !self.is_enabled() || checkpoints.is_empty() {
            return Vec::new();
        }
        let now_ms = self.millis(Instant::now());
        let max_followers = self.max_followers();
        let capped =
            |o: &Owner| max_followers > 0 && o.followers.load(Ordering::Relaxed) >= max_followers;
        let state = self.state.read();
        // Deepest first, so the first live sample a worker owns is its depth;
        // a capped one hides the stake (depth 0) rather than falling back.
        let mut depths: FxHashMap<Arc<str>, usize> = FxHashMap::default();
        for (index, hash) in checkpoints.deepest_first() {
            let owners = state.by_hash.get(&hash).map_or(&[][..], Vec::as_slice);
            for owner in owners
                .iter()
                .filter(|o| o.expiry_ms.load(Ordering::Relaxed) > now_ms)
            {
                depths.entry(Arc::clone(&owner.url)).or_insert_with(|| {
                    if capped(owner) {
                        0
                    } else {
                        checkpoints.depth_at(index) as usize
                    }
                });
            }
        }
        let staked = depths.len();
        let live: Vec<_> = depths.into_iter().filter(|(_, depth)| *depth > 0).collect();
        if staked > 0 && live.is_empty() {
            self.capped.fetch_add(1, Ordering::Relaxed);
        }
        live
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn depths(pending: &PendingPrefixes, samples: &BlockCheckpoints) -> Vec<(String, usize)> {
        let mut depths: Vec<_> = pending
            .checkpoint_depths(samples)
            .into_iter()
            .map(|(url, depth)| (url.to_string(), depth))
            .collect();
        depths.sort();
        depths
    }

    fn capped_at_one() -> PendingPrefixes {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.set_max_followers(1);
        pending
    }

    /// A repeat of the staked prompt: the sample on its last block is the
    /// block the follow charged, so the cap hides the stake on both paths.
    #[test]
    fn a_repeat_of_the_staked_prompt_is_capped_as_by_the_full_chain() {
        let pending = capped_at_one();
        let prompt: Vec<i64> = (1..=40).collect();
        pending.record("a", &prompt);
        let samples = BlockCheckpoints::grid(&prompt, prompt.len(), 8);
        assert_eq!(depths(&pending, &samples), [("a".into(), 40)]);
        pending.record_follow("a", &prompt);
        assert!(pending.depths(&prompt).is_empty());
        assert!(pending.checkpoint_depths(&samples).is_empty());
        assert_eq!(
            pending.capped(),
            2,
            "each lookup the cap emptied is counted"
        );
    }

    /// A prompt that shares the stake's first 20 blocks: the follow charges
    /// block 19, which no sample of a stride of 8 names, so the deepest live
    /// sample (block 15) reads a counter nothing charged. Pinned because it
    /// is what the doc says: such a stake outlives its cap here.
    #[test]
    fn a_prompt_leaving_the_stake_between_samples_sees_it_past_the_cap() {
        let pending = capped_at_one();
        let staked: Vec<i64> = (1..=40).collect();
        pending.record("a", &staked);
        let prompt: Vec<i64> = (1..=20).chain(1001..=1020).collect();
        pending.record_follow("a", &prompt);
        assert!(
            pending.depths(&prompt).is_empty(),
            "capped on the full chain"
        );
        let samples = BlockCheckpoints::grid(&prompt, prompt.len(), 8);
        assert_eq!(depths(&pending, &samples), [("a".into(), 16)]);
        assert_eq!(
            pending.capped(),
            1,
            "only the full-chain lookup was emptied"
        );
    }

    /// The deepest live sample decides; an expired stake is not one, and
    /// another worker's stake still answers beside a capped one without the
    /// lookup counting as capped.
    #[test]
    fn the_deepest_live_sample_decides_and_a_capped_stake_hides_only_itself() {
        let pending = capped_at_one();
        let prompt: Vec<i64> = (1..=32).collect();
        pending.record("a", &prompt);
        pending.record("b", &prompt[..12]);
        let samples = BlockCheckpoints::grid(&prompt, prompt.len(), 4);
        assert_eq!(
            depths(&pending, &samples),
            [("a".into(), 32), ("b".into(), 12)]
        );
        pending.record_follow("a", &prompt);
        assert_eq!(depths(&pending, &samples), [("b".into(), 12)]);
        assert_eq!(pending.capped(), 0);

        let expired = PendingPrefixes::default();
        expired.enable(Duration::from_millis(1));
        expired.record("a", &prompt);
        std::thread::sleep(Duration::from_millis(5));
        assert!(expired.checkpoint_depths(&samples).is_empty());
    }
}
