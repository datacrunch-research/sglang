// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The time to first token a prompt can expect at one prefill engine: the
//! prefill queue ahead of it, then its own prefill, from the engine's own
//! load reports and the work this router sent it that no report shows yet.
//! Network time and decode are not in it.

use super::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

/// A prompt's time to first token at one engine, in its two parts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TtftQuote {
    /// Milliseconds for the prefill queue ahead of the prompt to drain.
    pub queue_ms: f64,
    /// Milliseconds of the prompt's own uncached prefill: the work it costs
    /// the engine, as opposed to its wait.
    pub prefill_ms: f64,
}

impl TtftQuote {
    /// The time to first token: the queue, then the prompt's own prefill.
    pub fn ttft_ms(&self) -> f64 {
        self.queue_ms + self.prefill_ms
    }
}

/// The time to first token of a prompt that leaves `uncached_tokens` to
/// prefill at `engine`, were it dispatched there now. The queue is the
/// engine's reported waiting uncached tokens plus what this router
/// dispatched since that report's cut-off. An error says why the engine has
/// no quote: no fresh, complete report, or no prefill measured.
pub fn engine_ttft_quote(
    engine: &Worker,
    uncached_tokens: u64,
    load: &EngineReportedLoadSnapshot,
) -> Result<TtftQuote, &'static str> {
    let (Some(report), Some(pace)) = (
        load.fresh_native_cache_load_for_url(&engine.url),
        load.fresh_prefill_pace_for_url(&engine.url),
    ) else {
        return Err("an engine has no fresh load report");
    };
    let rate = pace
        .rank_tokens_per_s
        .ok_or("an engine has not measured its prefill rate")?;
    let dispatched = engine.uncached_tokens_dispatched_since(pace.reflects_dispatches_before);
    Ok(ttft_quote(
        report
            .num_waiting_uncached_tokens
            .saturating_add(dispatched),
        uncached_tokens,
        pace.ranks,
        rate,
    ))
}

/// `queued_tokens` drain over all `ranks` at once; the prompt's own
/// `uncached_tokens` prefill on the one rank it lands on. Every rank runs at
/// `rank_tokens_per_s`.
pub fn ttft_quote(
    queued_tokens: u64,
    uncached_tokens: u64,
    ranks: u32,
    rank_tokens_per_s: f64,
) -> TtftQuote {
    let ranks = f64::from(ranks.max(1));
    TtftQuote {
        queue_ms: 1_000.0 * (queued_tokens as f64 / ranks) / rank_tokens_per_s,
        prefill_ms: 1_000.0 * uncached_tokens as f64 / rank_tokens_per_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    use crate::state::load_monitor::engine_reported_load::{
        EnginePrefillPace, EngineReportedSchedulingLoad,
    };
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    const URL: &str = "http://e:30000";

    fn engine() -> Worker {
        Worker::new(WorkerSpec {
            id: WorkerId("e".into()),
            url: URL.into(),
            ..Default::default()
        })
    }

    fn snapshot(waiting: u64, pace: EnginePrefillPace) -> EngineReportedLoadSnapshot {
        let report = EngineReportedSchedulingLoad {
            num_running_reqs: 1,
            num_waiting_reqs: 1,
            num_waiting_uncached_tokens: waiting,
            num_used_tokens: 0,
            num_total_tokens: 0,
            max_total_num_tokens: 1_000_000,
            max_running_requests: 64,
            prefill_throughput_tokens_per_s: None,
            estimated_prefill_queue_ms: None,
            captured_at: pace.reflects_dispatches_before,
        };
        EngineReportedLoadSnapshot::from_native_cache_workers(
            1,
            HashMap::from([(URL.into(), report)]),
        )
        .with_prefill_pace(URL, pace)
    }

    #[test]
    fn one_rank_prefills_the_queue_then_the_prompt() {
        // 30k queued and 10k of its own at 20k tokens/s: 1.5 s, then 0.5 s.
        let quote = ttft_quote(30_000, 10_000, 1, 20_000.0);
        assert_eq!(
            quote,
            TtftQuote {
                queue_ms: 1_500.0,
                prefill_ms: 500.0
            }
        );
        assert_eq!(quote.ttft_ms(), 2_000.0);
        assert_eq!(ttft_quote(0, 0, 1, 20_000.0).ttft_ms(), 0.0);
    }

    #[test]
    fn ranks_share_the_queue_but_not_the_prompt() {
        // Four ranks drain 80k in 1 s; the prompt's 10k take one rank 0.5 s.
        assert_eq!(
            ttft_quote(80_000, 10_000, 4, 20_000.0),
            TtftQuote {
                queue_ms: 1_000.0,
                prefill_ms: 500.0
            }
        );
    }

    #[test]
    fn work_dispatched_after_the_report_cut_off_joins_the_queue() {
        let engine = engine();
        let _counted_in_the_report = engine.prefill_load_guard(50_000);
        let cut_off = Instant::now();
        let _since = engine.prefill_load_guard(20_000);
        let pace = EnginePrefillPace {
            ranks: 1,
            rank_tokens_per_s: Some(10_000.0),
            reflects_dispatches_before: cut_off,
        };
        // 10k reported + 20k dispatched since, then 5k of its own, at 10k/s.
        assert_eq!(
            engine_ttft_quote(&engine, 5_000, &snapshot(10_000, pace)),
            Ok(TtftQuote {
                queue_ms: 3_000.0,
                prefill_ms: 500.0
            })
        );
    }

    #[test]
    fn no_quote_without_a_rate_or_a_report() {
        let engine = engine();
        let pace = EnginePrefillPace {
            ranks: 1,
            rank_tokens_per_s: None,
            reflects_dispatches_before: Instant::now() - Duration::from_secs(1),
        };
        assert!(engine_ttft_quote(&engine, 1, &snapshot(0, pace)).is_err());
        assert!(engine_ttft_quote(&engine, 1, &EngineReportedLoadSnapshot::default()).is_err());
    }
}
