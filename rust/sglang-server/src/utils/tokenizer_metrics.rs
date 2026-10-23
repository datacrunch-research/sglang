//! Python's `TokenizerMetricsCollector` (`observability/metrics_collector.py`)
//! for the Rust front end: the request metrics the Python TokenizerManager
//! records, with the same names, label names and bucket boundaries, kept in
//! this process and appended to the merged multiprocess families on each
//! `/metrics` scrape. The scheduler-side families keep coming from the Python
//! processes' files; the startup gauges are written by Python
//! (`rust_server/server.py`), which knows the startup timings.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::message::config::{DisaggregationMode, ServerArgs};
use crate::message::finish_reason::{FinishKind, FinishReason};
use crate::utils::prometheus::{Family, Sample, float_to_go_string};

type Labels = Vec<(String, String)>;

/// `default_bucket_prompt_tokens`, also the generation-token buckets.
const TOKEN_BUCKETS: [f64; 35] = [
    100.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0,
    8000.0, 9000.0, 10000.0, 12500.0, 15000.0, 17500.0, 20000.0, 22500.0, 25000.0, 27500.0,
    30000.0, 35000.0, 40000.0, 60000.0, 80000.0, 100000.0, 200000.0, 300000.0, 400000.0, 600000.0,
    800000.0, 1000000.0, 1100000.0,
];
const TTFT_BUCKETS: [f64; 18] = [
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0,
];
const E2E_BUCKETS: [f64; 22] = [
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0, 600.0, 1200.0, 1800.0, 2400.0,
];
const ITL_BUCKETS: [f64; 23] = [
    0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080,
    0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000,
];

/// Thread-name prefixes of the Rust front end's threads (the process is a
/// scheduler's): the `component="tokenizer"` CPU counter sums these only.
const FRONT_END_THREADS: [&str; 7] = [
    "tokenizer",
    "detokenizer",
    "mm-worker",
    "from-scheduler",
    "to-scheduler",
    "api-runtime",
    "tokio-runtime",
];

struct Histogram {
    bounds: &'static [f64],
    /// Per label set: per-bound counts (not cumulative; the last is `+Inf`) and the sum.
    series: HashMap<Labels, (Vec<f64>, f64)>,
    order: Vec<Labels>,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            series: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// `observe(value)` `count` times with `sum += total` (Python's fast
    /// `observe_inter_token_latency` path; plain observe is `count = 1`).
    fn observe_n(&mut self, labels: Labels, value: f64, count: f64, total: f64) {
        let n = self.bounds.len();
        let entry = self.series.entry(labels.clone()).or_insert_with(|| {
            self.order.push(labels);
            (vec![0.0; n + 1], 0.0)
        });
        let i = self.bounds.iter().position(|b| value <= *b).unwrap_or(n);
        entry.0[i] += count;
        entry.1 += total;
    }

    fn family(&self, name: &str, help: &str) -> Family {
        let mut samples = Vec::new();
        for labels in &self.order {
            let (counts, sum) = &self.series[labels];
            let mut acc = 0.0;
            for (i, count) in counts.iter().enumerate() {
                acc += count;
                let le = self.bounds.get(i).copied().unwrap_or(f64::INFINITY);
                let mut bucket_labels = labels.clone();
                bucket_labels.push(("le".into(), float_to_go_string(le)));
                samples.push(Sample {
                    name: format!("{name}_bucket"),
                    labels: bucket_labels,
                    value: acc,
                });
            }
            samples.push(Sample {
                name: format!("{name}_count"),
                labels: labels.clone(),
                value: acc,
            });
            samples.push(Sample {
                name: format!("{name}_sum"),
                labels: labels.clone(),
                value: *sum,
            });
        }
        Family {
            name: name.into(),
            help: help.into(),
            kind: "histogram".into(),
            samples,
        }
    }
}

#[derive(Default)]
struct Counter {
    series: HashMap<Labels, f64>,
    order: Vec<Labels>,
}

impl Counter {
    fn inc(&mut self, labels: Labels, by: f64) {
        let entry = self.series.entry(labels.clone()).or_insert_with(|| {
            self.order.push(labels);
            0.0
        });
        *entry += by;
    }

    /// `name` without `_total`, as prometheus_client stores counters.
    fn family(&self, name: &str, help: &str) -> Family {
        Family {
            name: name.into(),
            help: help.into(),
            kind: "counter".into(),
            samples: self
                .order
                .iter()
                .map(|labels| Sample {
                    name: format!("{name}_total"),
                    labels: labels.clone(),
                    value: self.series[labels],
                })
                .collect(),
        }
    }
}

struct Registry {
    prompt_tokens: Counter,
    generation_tokens: Counter,
    cached_tokens: Counter,
    num_requests: Counter,
    num_so_requests: Counter,
    num_aborted_requests: Counter,
    process_cpu: Counter,
    prompt_tokens_histogram: Histogram,
    uncached_prompt_tokens_histogram: Histogram,
    generation_tokens_histogram: Histogram,
    time_to_first_token: Histogram,
    inter_token_latency: Histogram,
    request_time_per_output_token: Histogram,
    e2e_request_latency: Histogram,
    /// `routing_keys_active`: active requests per `x-smg-routing-key`.
    routing_keys: HashMap<String, u64>,
}

pub struct TokenizerMetrics {
    /// `model_name` and `engine_type`, as the TokenizerManager labels.
    labels: Labels,
    prefill: bool,
    registry: Mutex<Registry>,
}

static METRICS: OnceLock<TokenizerMetrics> = OnceLock::new();

/// The collector, once `init` ran (`--enable-metrics`).
pub fn get() -> Option<&'static TokenizerMetrics> {
    METRICS.get()
}

/// Create the collector and start the front end's CPU counter.
pub fn init(server_args: &ServerArgs) {
    let engine_type = match server_args.disaggregation_mode {
        DisaggregationMode::Prefill => "prefill",
        DisaggregationMode::Decode => "decode",
        DisaggregationMode::Null => "unified",
    };
    let metrics = TokenizerMetrics {
        labels: vec![
            ("model_name".into(), server_args.served_model_name.clone()),
            ("engine_type".into(), engine_type.into()),
        ],
        prefill: server_args.disaggregation_mode == DisaggregationMode::Prefill,
        registry: Mutex::new(Registry {
            prompt_tokens: Counter::default(),
            generation_tokens: Counter::default(),
            cached_tokens: Counter::default(),
            num_requests: Counter::default(),
            num_so_requests: Counter::default(),
            num_aborted_requests: Counter::default(),
            process_cpu: Counter::default(),
            prompt_tokens_histogram: Histogram::new(&TOKEN_BUCKETS),
            uncached_prompt_tokens_histogram: Histogram::new(&TOKEN_BUCKETS),
            generation_tokens_histogram: Histogram::new(&TOKEN_BUCKETS),
            time_to_first_token: Histogram::new(&TTFT_BUCKETS),
            inter_token_latency: Histogram::new(&ITL_BUCKETS),
            request_time_per_output_token: Histogram::new(&ITL_BUCKETS),
            e2e_request_latency: Histogram::new(&E2E_BUCKETS),
            routing_keys: HashMap::new(),
        }),
    };
    if METRICS.set(metrics).is_ok() {
        let _ = std::thread::Builder::new()
            .name("cpu-monitor".into())
            .spawn(cpu_monitor);
    }
}

/// Per-request timing the detokenizer shard keeps (Python's `ReqState` time stats).
#[derive(Debug)]
pub struct RequestTimes {
    created_at: Instant,
    stream: bool,
    has_grammar: bool,
    first_token_at: Option<Instant>,
    last_time: Instant,
    last_completion_tokens: u64,
    completion_tokens: u64,
    /// The latest chunk's prompt and cached counts (the prompt's length before any).
    prompt_tokens: u32,
    cached_tokens: u32,
}

impl RequestTimes {
    pub fn new(created_at: Instant, stream: bool, has_grammar: bool, prompt_tokens: u32) -> Self {
        Self {
            created_at,
            stream,
            has_grammar,
            first_token_at: None,
            last_time: created_at,
            last_completion_tokens: 0,
            completion_tokens: 0,
            prompt_tokens,
            cached_tokens: 0,
        }
    }
}

fn bool_label(value: bool) -> String {
    if value { "true" } else { "false" }.into()
}

impl TokenizerMetrics {
    fn with(&self, extra: &[(&str, String)]) -> Labels {
        let mut labels = self.labels.clone();
        labels.extend(extra.iter().map(|(k, v)| ((*k).to_owned(), v.clone())));
        labels
    }

    /// `TokenizerManager.collect_metrics` for one output chunk: `new_tokens`
    /// generated in this chunk, the request's prompt and cached totals, and
    /// the finish reason on its last chunk.
    pub fn on_chunk(
        &self,
        times: &mut RequestTimes,
        new_tokens: u64,
        prompt_tokens: u32,
        cached_tokens: u32,
        finish: Option<&FinishReason>,
    ) {
        let now = Instant::now();
        times.completion_tokens += new_tokens;
        times.prompt_tokens = prompt_tokens;
        times.cached_tokens = cached_tokens;
        let kind = match finish {
            Some(FinishReason::Known(FinishKind::Abort(_))) => Some("abort"),
            Some(FinishReason::Known(FinishKind::Stop { .. })) => Some("stop"),
            Some(FinishReason::Known(FinishKind::Length { .. })) => Some("length"),
            Some(FinishReason::Unknown(_)) => Some("unknown"),
            None => None,
        };
        let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let stream_label = [("is_streaming", bool_label(times.stream))];
        if times.first_token_at.is_none() {
            // PD prefill workers never observe TTFT, so they never reach ITL.
            if kind != Some("abort") && !self.prefill {
                times.first_token_at = Some(now);
                times.last_time = now;
                times.last_completion_tokens = times.completion_tokens;
                let ttft = now.duration_since(times.created_at).as_secs_f64();
                registry
                    .time_to_first_token
                    .observe_n(self.with(&stream_label), ttft, 1.0, ttft);
            }
        } else {
            let new = times
                .completion_tokens
                .saturating_sub(times.last_completion_tokens);
            if new > 0 {
                let interval = now.duration_since(times.last_time).as_secs_f64();
                registry.inter_token_latency.observe_n(
                    self.labels.clone(),
                    interval / new as f64,
                    new as f64,
                    interval,
                );
                times.last_time = now;
                times.last_completion_tokens = times.completion_tokens;
            }
        }
        if kind.is_none() {
            return;
        }
        let completion = times.completion_tokens;
        let stream_labels = self.with(&stream_label);
        registry
            .prompt_tokens
            .inc(stream_labels.clone(), f64::from(prompt_tokens));
        registry
            .generation_tokens
            .inc(stream_labels.clone(), completion as f64);
        if cached_tokens > 0 {
            // The scheduler's per-source breakdown does not reach the Rust
            // front end; Python reports the same `total` source without it.
            registry.cached_tokens.inc(
                self.with(&[("cache_source", "total".into())]),
                f64::from(cached_tokens),
            );
        }
        registry.num_requests.inc(stream_labels.clone(), 1.0);
        if times.has_grammar {
            registry.num_so_requests.inc(self.labels.clone(), 1.0);
        }
        let e2e = now.duration_since(times.created_at).as_secs_f64();
        registry
            .e2e_request_latency
            .observe_n(stream_labels.clone(), e2e, 1.0, e2e);
        // Aborts have a truncated decode period; PD prefill workers have none.
        if !self.prefill
            && matches!(kind, Some("stop" | "length"))
            && let Some(first) = times.first_token_at
            && completion > 1
        {
            let decode = now.duration_since(first).as_secs_f64();
            if decode > 0.0 {
                let tpot = decode / (completion - 1) as f64;
                registry
                    .request_time_per_output_token
                    .observe_n(stream_labels, tpot, 1.0, tpot);
            }
        }
        let prompt = f64::from(prompt_tokens);
        registry
            .prompt_tokens_histogram
            .observe_n(self.labels.clone(), prompt, 1.0, prompt);
        let uncached = f64::from(prompt_tokens) - f64::from(cached_tokens);
        registry.uncached_prompt_tokens_histogram.observe_n(
            self.labels.clone(),
            uncached,
            1.0,
            uncached,
        );
        registry.generation_tokens_histogram.observe_n(
            self.labels.clone(),
            completion as f64,
            1.0,
            completion as f64,
        );
    }

    /// A request dropped before its finish chunk (client gone, abort, failure):
    /// Python observes it on the abort finish the scheduler then sends, which
    /// the Rust front end no longer routes.
    pub fn on_dropped(&self, times: &mut RequestTimes) {
        let abort = FinishReason::Known(FinishKind::Abort(Box::new(
            crate::message::finish_reason::AbortReason {
                message: None,
                status_code: None,
                err_type: None,
            },
        )));
        let (prompt, cached) = (times.prompt_tokens, times.cached_tokens);
        self.on_chunk(times, 0, prompt, cached, Some(&abort));
    }

    /// `observe_one_aborted_request`: an abort sent to the scheduler.
    pub fn on_abort(&self) {
        let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        registry.num_aborted_requests.inc(self.labels.clone(), 1.0);
    }

    /// `RefCountedGauge.inc` / `dec` on `sglang:routing_keys_active`.
    pub fn routing_key(&self, key: &str, active: bool) {
        let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        if active {
            *registry.routing_keys.entry(key.to_owned()).or_insert(0) += 1;
        } else if let Some(count) = registry.routing_keys.get_mut(key) {
            *count -= 1;
            if *count == 0 {
                registry.routing_keys.remove(key);
            }
        }
    }

    /// The families, in Python's `TokenizerMetricsCollector` order.
    pub fn families(&self) -> Vec<Family> {
        let registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let mut families = vec![
            registry.prompt_tokens.family(
                "sglang:prompt_tokens",
                "Number of prefill tokens processed.",
            ),
            registry.generation_tokens.family(
                "sglang:generation_tokens",
                "Number of generation tokens processed.",
            ),
            registry.prompt_tokens_histogram.family(
                "sglang:prompt_tokens_histogram",
                "Histogram of prompt token length.",
            ),
            registry.uncached_prompt_tokens_histogram.family(
                "sglang:uncached_prompt_tokens_histogram",
                "Histogram of uncached (compute) prompt token length.",
            ),
            registry.generation_tokens_histogram.family(
                "sglang:generation_tokens_histogram",
                "Histogram of generation token length.",
            ),
            registry.cached_tokens.family(
                "sglang:cached_tokens",
                "Number of cached prompt tokens by source (device/host/storage).",
            ),
            registry
                .num_requests
                .family("sglang:num_requests", "Number of requests processed."),
            registry.num_so_requests.family(
                "sglang:num_so_requests",
                "Number of structured output requests processed.",
            ),
            registry
                .num_aborted_requests
                .family("sglang:num_aborted_requests", "Number of requests aborted."),
            registry.time_to_first_token.family(
                "sglang:time_to_first_token_seconds",
                "Histogram of time to first token in seconds.",
            ),
            registry.inter_token_latency.family(
                "sglang:inter_token_latency_seconds",
                "Histogram of inter-token latency in seconds.",
            ),
            registry.request_time_per_output_token.family(
                "sglang:request_time_per_output_token_seconds",
                "Per-request TPOT in seconds: (finished_time - first_token_time) / (completion_tokens - 1).",
            ),
            registry.e2e_request_latency.family(
                "sglang:e2e_request_latency_seconds",
                "Histogram of End-to-end request latency in seconds",
            ),
            registry.process_cpu.family(
                "sglang:process_cpu_seconds",
                "Total CPU time consumed by this process (user + system)",
            ),
        ];
        families.push(Family {
            name: "sglang:routing_keys_active".into(),
            help: "Number of unique routing keys with active requests".into(),
            kind: "gauge".into(),
            samples: vec![Sample {
                name: "sglang:routing_keys_active".into(),
                labels: Vec::new(),
                value: registry.routing_keys.len() as f64,
            }],
        });
        families
    }
}

/// `start_cpu_monitor_thread`: every 5 s, add the CPU seconds the front end's
/// threads used since the last tick, as Python's TokenizerManager process
/// (`component="tokenizer"`) and DetokenizerManager process
/// (`component="detokenizer"`) report theirs. Counted per thread, so a thread
/// that exits loses at most its last 5 s.
fn cpu_monitor() {
    let ticks = 100.0; // USER_HZ on Linux.
    let sample = || -> HashMap<String, (&'static str, f64)> {
        let mut out = HashMap::new();
        let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
            return out;
        };
        for task in tasks.flatten() {
            let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
                continue;
            };
            let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
                continue;
            };
            let comm = &stat[open + 1..close];
            if !FRONT_END_THREADS
                .iter()
                .any(|prefix| comm.starts_with(prefix))
            {
                continue;
            }
            let component = if comm.starts_with("detokenizer") {
                "detokenizer"
            } else {
                "tokenizer"
            };
            let fields: Vec<&str> = stat[close + 2..].split_whitespace().collect();
            let (Some(Ok(utime)), Some(Ok(stime))) = (
                fields.get(11).map(|f| f.parse::<f64>()),
                fields.get(12).map(|f| f.parse::<f64>()),
            ) else {
                continue;
            };
            out.insert(
                task.file_name().to_string_lossy().into_owned(),
                (component, (utime + stime) / ticks),
            );
        }
        out
    };
    let mut last = sample();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let now = sample();
        let mut delta: HashMap<&'static str, f64> = HashMap::new();
        for (tid, (component, cpu)) in &now {
            let before = last.get(tid).map_or(0.0, |(_, cpu)| *cpu);
            *delta.entry(component).or_insert(0.0) += (cpu - before).max(0.0);
        }
        last = now;
        if let Some(metrics) = get() {
            let mut registry = metrics.registry.lock().unwrap_or_else(|e| e.into_inner());
            for component in ["tokenizer", "detokenizer"] {
                let value = delta.get(component).copied().unwrap_or(0.0);
                registry
                    .process_cpu
                    .inc(vec![("component".into(), component.into())], value);
            }
        }
    }
}

/// Append `ours` to `families`, extending a family the files already hold.
pub fn merge_into(families: &mut Vec<Family>, ours: Vec<Family>) {
    for family in ours.into_iter().filter(|f| !f.samples.is_empty()) {
        match families.iter_mut().find(|f| f.name == family.name) {
            Some(existing) => existing.samples.extend(family.samples),
            None => families.push(family),
        }
    }
}
