// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::forward::{BucketRouting, SelectedWorkers};
use super::preparation::{session_id_header, PreparedRequest};
use super::{
    nonempty_header, parse_optional_positive_f64_header, parse_optional_positive_u64_header,
    record_prefill_route, X_SGL_TPS_SLO, X_SGL_TTFT_SLO_MS,
};
use crate::buckets_reorg::{Bucket, BucketPick, BucketRequest, BucketResolver, SloPreference};
use crate::discovery::{ModelId, WorkerId};
use crate::policies_reorg::{PickError, Stage};
use crate::server::app_context::AppContext;
use crate::server::bucket_admission::{
    size_queue, BucketQueue, BucketRejection, BucketSlot, RejectReason, TpmLimiter,
};
use crate::server::error::ApiError;
use crate::state::session_estimator::{session_key, PendingTurn};
use axum::http::HeaderMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::Instant;

/// Bucket-first selection used when `AppContext::chat_routing` is reorg.
pub(super) async fn select_workers(
    ctx: &AppContext,
    resolver: &BucketResolver,
    request: &PreparedRequest,
    headers: &HeaderMap,
    excluded: &[WorkerId],
) -> Result<SelectedWorkers, ApiError> {
    let input_tokens = request.sequence_token_count as u64;
    if request
        .output_tokens
        .is_some_and(|output| input_tokens.checked_add(output).is_none())
    {
        return Err(ApiError::BadRequest(
            "input and output token counts overflow".into(),
        ));
    }
    let expected_peak_tokens = request.expected_peak_sequence_tokens;

    let ttft_ms = if resolver.ttft_slo != SloPreference::Disabled {
        parse_optional_positive_u64_header(headers, &X_SGL_TTFT_SLO_MS, "TTFT SLO")?
    } else {
        None
    };
    let tokens_per_second = if resolver.tps_slo != SloPreference::Disabled {
        parse_optional_positive_f64_header(headers, &X_SGL_TPS_SLO, "TPS SLO")?
    } else {
        None
    };
    // Buckets bounding uncached tokens read the session's estimate.
    let session = resolver.estimator.as_ref().map(|estimator| {
        let header = nonempty_header(headers, session_id_header(ctx));
        let key = session_key(header, request.derived_session_key.as_deref());
        let estimate = estimator.estimate(key.as_deref(), input_tokens);
        (estimator, key, estimate)
    });
    let uncached_tokens = session
        .as_ref()
        .map(|(_, _, estimate)| estimate.uncached_tokens);
    let buckets = resolver
        .resolve_uncached(
            input_tokens,
            uncached_tokens,
            expected_peak_tokens,
            ttft_ms,
            tokens_per_second,
        )
        .map_err(|error| selection_error(error, &request.model, None))?;
    if buckets.is_empty() {
        return Err(selection_error(
            PickError::NoMatchingBucket,
            &request.model,
            None,
        ));
    }
    let prefix = crate::policies_reorg::cache_aware::PrefixMemo::default();
    let bucket_request = BucketRequest {
        prefix: Some(&prefix),
        model: &request.model,
        input_tokens,
        total_input_tokens: request.input_token_count as u64,
        expected_peak_tokens,
        token_ids: request.tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
        cache_namespace: &request.cache_namespace,
        session_key: ctx
            .config
            .model
            .affinity
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.session_id_header)),
        routing_key: ctx
            .config
            .model
            .sticky
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.header_name)),
        excluded,
    };
    let home = buckets[0];
    // Above the agreed tokens per minute the pool refuses before any bucket.
    if let Some(limiter) = &resolver.tpm {
        if let Err(retry_after_s) = limiter.try_take(request.input_token_count as u64) {
            return Err(ApiError::BucketRejected(Box::new(BucketRejection {
                bucket: home.id.clone(),
                reason: RejectReason::TpmLimit,
                retry_after_s,
                depth_tokens: limiter.capacity() as u64,
                queued_tokens: 0,
                uncached_tokens: uncached_tokens.unwrap_or(input_tokens),
                waited: std::time::Duration::ZERO,
            })));
        }
        if let Some(scope) = &request.scope {
            scope.life.charge_tpm_output(Arc::clone(limiter));
        }
    }
    // Input taken for a request no engine ends up serving goes back.
    let tpm_refund = TpmRefund {
        limiter: resolver.tpm.as_deref(),
        tokens: request.input_token_count as u64,
        armed: AtomicBool::new(true),
    };
    let selected = |picks: BucketPick,
                    bucket_slot: Option<BucketSlot>,
                    bucket: &str,
                    queue_wait: Option<std::time::Duration>| {
        if let (Some(scope), Some((estimator, Some(key), estimate))) = (&request.scope, &session) {
            scope.life.track_session_turn(PendingTurn {
                estimator: Arc::clone(estimator),
                key: key.clone(),
                prompt_tokens: input_tokens,
                estimate: *estimate,
            });
        }
        record_prefill_route(
            ctx,
            prefix.local_signal().as_deref(),
            &picks.prefill.engine.url,
        );
        tpm_refund.armed.store(false, Ordering::Relaxed);
        // Dispatch only after this bucket supplies the entire plain or PD selection.
        SelectedWorkers {
            prefill: picks.prefill.engine,
            decode: picks.decode.map(|pick| pick.engine),
            track_dispatch_timestamps: false,
            bucket_slot,
            routing: Some(BucketRouting {
                bucket: bucket.to_owned(),
                uncached_tokens,
                queue_wait,
            }),
        }
    };
    // A queued home bucket keeps the request: it waits there for an engine.
    if let Some(queue) = &home.queue {
        let tokens = uncached_tokens.unwrap_or(input_tokens);
        let (picks, slot, waited) = pick_queued(
            ctx,
            resolver,
            home,
            queue,
            &bucket_request,
            tokens,
            &request.model,
        )
        .await?;
        return Ok(selected(picks, Some(slot), &home.id, Some(waited)));
    }
    let mut rejections: Option<Vec<_>> = None;
    let mut missing_stage = None;
    for bucket in buckets {
        match bucket.pick_engines(&ctx.registry, &bucket_request).await {
            Ok(picks) => return Ok(selected(picks, None, &bucket.id, None)),
            Err((stage, error)) => {
                tracing::debug!(bucket = %bucket.id, ?stage, %error, "bucket selection failed");
                match error {
                    PickError::NoCandidates => missing_stage = Some(stage),
                    PickError::NoAdmissibleEngine(reasons) => {
                        rejections.get_or_insert_with(Vec::new).extend(reasons);
                    }
                    PickError::AdmissionRejected(reason) => {
                        rejections.get_or_insert_with(Vec::new).push(reason);
                    }
                    error => return Err(selection_error(error, &request.model, Some(stage))),
                }
            }
        }
    }
    if let Some((picks, _)) = spill(ctx, resolver, home, &bucket_request).await {
        return Ok(selected(picks, None, &home.id, None));
    }
    // Preserve admission exhaustion even if a later bucket has no candidates.
    let error = match rejections {
        Some(reasons) => PickError::NoAdmissibleEngine(reasons),
        None => PickError::NoCandidates,
    };
    Err(selection_error(error, &request.model, missing_stage))
}

/// Admit `tokens` into `home`'s queue (429 at once over its depth), then wait
/// for its engines up to the queue's deadline (429 after it). A request whose
/// client leaves while it waits is dropped here and frees its place.
async fn pick_queued(
    ctx: &AppContext,
    resolver: &BucketResolver,
    home: &Bucket,
    queue: &Arc<BucketQueue>,
    request: &BucketRequest<'_>,
    tokens: u64,
    model: &ModelId,
) -> Result<(BucketPick, BucketSlot, std::time::Duration), ApiError> {
    size_queue(home, queue, &ctx.registry, model, &resolver.prefill_rates);
    let arrived = Instant::now();
    let slot = queue
        .try_enter(tokens)
        .map_err(|rejection| ApiError::BucketRejected(Box::new(rejection)))?;
    let deadline = arrived + queue.deadline();
    loop {
        match home.pick_engines(&ctx.registry, request).await {
            Ok(picks) => {
                queue.record_wait(arrived.elapsed());
                return Ok((picks, slot, arrived.elapsed()));
            }
            Err((
                stage,
                PickError::NoCandidates
                | PickError::NoAdmissibleEngine(_)
                | PickError::AdmissionRejected(_),
            )) => {
                // Before waiting, an idle engine of a longer bucket may take it.
                if let Some((picks, _)) = spill(ctx, resolver, home, request).await {
                    queue.record_wait(arrived.elapsed());
                    return Ok((picks, slot, arrived.elapsed()));
                }
                if Instant::now() >= deadline {
                    let reason = match stage {
                        Stage::Decode => RejectReason::DecodePermits,
                        _ => RejectReason::Deadline,
                    };
                    let rejection = queue.expired(reason, tokens, arrived.elapsed());
                    return Err(ApiError::BucketRejected(Box::new(rejection)));
                }
                queue.wait_for_change(deadline).await;
            }
            Err((stage, error)) => return Err(selection_error(error, model, Some(stage))),
        }
    }
}

struct TpmRefund<'a> {
    limiter: Option<&'a TpmLimiter>,
    tokens: u64,
    armed: AtomicBool,
}

impl Drop for TpmRefund<'_> {
    fn drop(&mut self) {
        if let (Some(limiter), true) = (self.limiter, self.armed.load(Ordering::Relaxed)) {
            limiter.refund(self.tokens);
        }
    }
}

async fn spill<'r>(
    ctx: &AppContext,
    resolver: &'r BucketResolver,
    home: &Bucket,
    request: &BucketRequest<'_>,
) -> Option<(BucketPick, &'r Bucket)> {
    if home.spill_to.is_empty() {
        return None;
    }
    let load = ctx
        .engine_reported_load
        .capture_snapshot(std::time::Instant::now());
    let (picks, target) = resolver.spill(home, &ctx.registry, &load, request).await?;
    tracing::debug!(from = %home.id, to = %target.id, "spilled to an idle engine of a longer bucket");
    Some((picks, target))
}

fn selection_error(error: PickError, model: &ModelId, stage: Option<Stage>) -> ApiError {
    tracing::warn!(%model, ?stage, %error, "reorg selection failed");
    match error {
        PickError::NoMatchingBucket => {
            ApiError::BadRequest("no bucket supports the requested token length".into())
        }
        PickError::NoCandidates => match stage {
            Some(Stage::Prefill) => ApiError::NoPrefillWorkersAvailable {
                model: model.0.clone(),
            },
            Some(Stage::Decode) => ApiError::NoDecodeWorkersAvailable {
                model: model.0.clone(),
            },
            _ => ApiError::NoHealthyWorkers {
                model: model.0.clone(),
            },
        },
        PickError::NoAdmissibleEngine(_) | PickError::AdmissionRejected(_) => {
            ApiError::PolicySelectionFailed {
                model: model.0.clone(),
            }
        }
        error => ApiError::Internal(error.into()),
    }
}
