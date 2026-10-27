// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Round-robin tickets retained for one request across ladder proposals.
#[derive(Debug, Default)]
pub(crate) struct RequestRotation {
    tickets: Mutex<Vec<(usize, usize)>>,
}
impl RequestRotation {
    pub(crate) fn ticket(&self, counter: &AtomicUsize) -> usize {
        let key = std::ptr::from_ref(counter).addr();
        let mut tickets = self.tickets.lock().unwrap();
        if let Some((_, value)) = tickets.iter().find(|(policy, _)| *policy == key) {
            return *value;
        }
        let value = counter.fetch_add(1, Ordering::Relaxed);
        tickets.push((key, value));
        value
    }
}

#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Policy for RoundRobinPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }
        let i = ctx.round_robin.ticket(&self.counter) % workers.len();
        Some(workers[i].clone())
    }
}
