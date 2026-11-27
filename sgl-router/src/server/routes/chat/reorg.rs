// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::forward::SelectedWorkers;
use super::preparation::PreparedRequest;
use super::{
    nonempty_header, parse_optional_positive_f64_header, parse_optional_positive_u64_header,
    record_prefill_route, X_SGL_TPS_SLO, X_SGL_TTFT_SLO_MS,
};
use crate::buckets_reorg::{BucketRequest, BucketResolver, SloPreference};
use crate::discovery::{ModelId, WorkerId};
use crate::policies_reorg::cache_aware::{uncached_by_url, PrefixMemo};
use crate::policies_reorg::{PickError, Stage};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::http::HeaderMap;

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
    let buckets = resolver
        .resolve(
            input_tokens,
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
    let prefix = PrefixMemo::default();
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
    let mut rejections: Option<Vec<_>> = None;
    let mut missing_stage = None;
    for bucket in buckets {
        match bucket.pick_engines(&ctx.registry, &bucket_request).await {
            Ok(picks) => {
                record_prefill_route(
                    ctx,
                    prefix.local_signal().as_deref(),
                    &picks.prefill.engine.url,
                );
                let uncached_prefill_tokens =
                    uncached_prefill_tokens(&bucket_request, &prefix, &picks.prefill.engine.url);
                // Dispatch only after this bucket supplies the entire plain or PD selection.
                return Ok(SelectedWorkers {
                    prefill: picks.prefill.engine,
                    decode: picks.decode.map(|pick| pick.engine),
                    track_dispatch_timestamps: false,
                    uncached_prefill_tokens: Some(uncached_prefill_tokens),
                });
            }
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
    // Preserve admission exhaustion even if a later bucket has no candidates.
    let error = match rejections {
        Some(reasons) => PickError::NoAdmissibleEngine(reasons),
        None => PickError::NoCandidates,
    };
    Err(selection_error(error, &request.model, missing_stage))
}

/// The prompt tokens `prefill_url` has to prefill for `request`, from the
/// prefix lookup the pick made, as the cache-aware policy reckons them: the
/// hit covers the longest prompt, the rest of a batch is uncached. All of it
/// when no lookup was made.
fn uncached_prefill_tokens(
    request: &BucketRequest<'_>,
    prefix: &PrefixMemo,
    prefill_url: &str,
) -> u64 {
    let signal = prefix.signal();
    let uncached = uncached_by_url(signal.as_deref(), request.input_tokens);
    let cached = uncached
        .get(prefill_url)
        .map_or(0, |&left| request.input_tokens - left);
    request.total_input_tokens.saturating_sub(cached)
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
