// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A fixed-bucket Prometheus histogram for the hand-rendered registry.

use std::fmt::Write;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[derive(Debug)]
pub struct Histogram {
    /// Upper bounds, ascending; `+Inf` is implied.
    bounds: &'static [f64],
    counts: Vec<AtomicU64>,
    /// Sum of observations in millionths, so it stays one atomic add.
    sum_micros: AtomicI64,
    count: AtomicU64,
}

impl Histogram {
    pub fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: bounds.iter().map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicI64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if let Some(index) = self.bounds.iter().position(|bound| value <= *bound) {
            self.counts[index].fetch_add(1, Ordering::Relaxed);
        }
        self.sum_micros
            .fetch_add((value * 1e6).round() as i64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Exposition lines for `name` with `labels` (`key="value",...` or empty);
    /// the caller writes `# HELP` / `# TYPE` once per family.
    pub fn render(&self, out: &mut String, name: &str, labels: &str) {
        let sep = if labels.is_empty() { "" } else { "," };
        let mut cumulative = 0;
        for (bound, count) in self.bounds.iter().zip(&self.counts) {
            cumulative += count.load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "{name}_bucket{{{labels}{sep}le=\"{bound}\"}} {cumulative}"
            );
        }
        let count = self.count();
        let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"+Inf\"}} {count}");
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        let braces = |labels: &str| {
            if labels.is_empty() {
                String::new()
            } else {
                format!("{{{labels}}}")
            }
        };
        let _ = writeln!(out, "{name}_sum{} {sum}", braces(labels));
        let _ = writeln!(out, "{name}_count{} {count}", braces(labels));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_cumulative_buckets_sum_and_count() {
        static BOUNDS: [f64; 3] = [-1.0, 0.0, 2.5];
        let histogram = Histogram::new(&BOUNDS);
        for value in [-5.0, 0.0, 1.0, 9.0, f64::NAN] {
            histogram.observe(value);
        }
        let mut out = String::new();
        histogram.render(&mut out, "x", "b=\"S\"");
        assert_eq!(
            out,
            "x_bucket{b=\"S\",le=\"-1\"} 1\nx_bucket{b=\"S\",le=\"0\"} 2\n\
             x_bucket{b=\"S\",le=\"2.5\"} 3\nx_bucket{b=\"S\",le=\"+Inf\"} 4\n\
             x_sum{b=\"S\"} 5\nx_count{b=\"S\"} 4\n"
        );
        let mut bare = String::new();
        Histogram::new(&BOUNDS).render(&mut bare, "y", "");
        assert!(bare.contains("y_bucket{le=\"+Inf\"} 0\n") && bare.ends_with("y_count 0\n"));
    }
}
