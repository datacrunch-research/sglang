// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode policy extension point, independent of prefill affinity.

use crate::config::DecodePolicyKind;
use crate::policies::admission::{
    compare_decode_engines, resolve_decode, CandidateDomain, FinalDecision,
};
use crate::policies::registry::select_decode_with_affinity;
use crate::policies::{ProposalKind, SelectionProposal};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use rand::Rng;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct DecodeSelectionContext<'a> {
    load_snapshot: Option<&'a EngineReportedLoadSnapshot>,
    prefill_url: Option<&'a str>,
    rotation: super::round_robin::RequestRotation,
}

impl<'a> DecodeSelectionContext<'a> {
    pub fn new() -> Self {
        Self {
            load_snapshot: None,
            prefill_url: None,
            rotation: Default::default(),
        }
    }

    /// Engine load snapshot captured at request ingress.
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineReportedLoadSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    pub fn load_snapshot(&self) -> Option<&EngineReportedLoadSnapshot> {
        self.load_snapshot
    }

    /// Prefill URL used by `legacy_host_affinity`.
    pub fn with_prefill_url(mut self, prefill_url: &'a str) -> Self {
        self.prefill_url = Some(prefill_url);
        self
    }

    pub fn prefill_url(&self) -> Option<&str> {
        self.prefill_url
    }
}

pub trait DecodePolicy: Send + Sync + std::fmt::Debug {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal>;
}

#[derive(Debug, Default)]
pub struct DecodeRoundRobinPolicy {
    counter: AtomicUsize,
}

impl DecodePolicy for DecodeRoundRobinPolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        if domain.workers.is_empty() {
            return None;
        }
        // Registry iteration order must not change the rotation between requests.
        let mut workers: Vec<_> = domain.workers.iter().collect();
        workers.sort_unstable_by(|a, b| a.id.0.cmp(&b.id.0));
        let index = ctx.rotation.ticket(&self.counter) % workers.len();
        Some(SelectionProposal::primary(Arc::clone(workers[index])))
    }
}

/// Compatibility entry point: decode capacity is scheduled by the engine.
pub fn resolve_decode_with_capacity_fallback(
    domain: &CandidateDomain,
    proposal: &SelectionProposal,
    request_kv_tokens: u64,
    snapshot: &EngineReportedLoadSnapshot,
) -> Option<FinalDecision> {
    resolve_decode(domain, proposal, request_kv_tokens, snapshot)
}

/// Samples two workers from a decode domain and orders them by decode pressure.
#[derive(Debug, Default)]
pub struct DecodePowerOfTwoPolicy;

impl DecodePowerOfTwoPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl DecodePolicy for DecodePowerOfTwoPolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match domain.workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(Arc::clone(&domain.workers[0]))
                    .with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                let left = &domain.workers[i];
                let right = &domain.workers[j];
                let (primary, backup) =
                    if compare_decode_engines(left, right, ctx.load_snapshot()).is_gt() {
                        (Arc::clone(right), Arc::clone(left))
                    } else {
                        (Arc::clone(left), Arc::clone(right))
                    };
                Some(
                    SelectionProposal::with_backup(primary, backup)
                        .with_kind(ProposalKind::PowerOfTwo),
                )
            }
        }
    }
}

/// Compatibility policy for legacy same-host PD decode selection.
#[derive(Debug, Default)]
pub struct LegacyHostAffinityDecodePolicy;

impl DecodePolicy for LegacyHostAffinityDecodePolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        let prefill_url = ctx.prefill_url()?;
        select_decode_with_affinity(prefill_url, &domain.workers).map(SelectionProposal::primary)
    }
}

/// Builds a decode policy scoped to one role.
pub fn build_decode_policy(kind: DecodePolicyKind) -> Box<dyn DecodePolicy> {
    match kind {
        DecodePolicyKind::RoundRobin => Box::new(DecodeRoundRobinPolicy::default()),
        DecodePolicyKind::PowerOfTwo => Box::new(DecodePowerOfTwoPolicy::new()),
        DecodePolicyKind::LegacyHostAffinity => Box::new(LegacyHostAffinityDecodePolicy),
    }
}
