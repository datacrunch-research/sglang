// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Thresholded health for individual worker instances and compatible role pairs.

use crate::{
    config::{HealthCheckConfig, HealthCheckMode},
    discovery::{ModelId, WorkerMode},
    server::app_context::AppContext,
    workers::Worker,
};
use futures::{stream, StreamExt};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

/// Spacing of the join probes after a discovery event. Arbitrary; short next
/// to any health interval.
const JOIN_RETRY: Duration = Duration::from_millis(250);

pub struct ActiveHealth {
    config: HealthCheckConfig,
    state: Mutex<HealthState>,
}
#[derive(Default)]
struct HealthState {
    streaks: HashMap<u64, Streak>,
}
#[derive(Default)]
struct Streak {
    successes: u32,
    failures: u32,
    proven: bool,
}

impl ActiveHealth {
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            state: Mutex::new(HealthState::default()),
        }
    }

    pub fn ready(&self, ctx: &AppContext) -> bool {
        let workers = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()));
        let state = self.state.lock().unwrap();
        let proven = |worker: &Arc<Worker>| {
            worker.is_healthy()
                && !worker.awaiting_bootstrap_port()
                && state
                    .streaks
                    .get(&worker.health_instance())
                    .is_some_and(|s| s.proven)
        };
        workers.iter().any(|worker| {
            proven(worker)
                && match worker.mode() {
                    WorkerMode::Plain => true,
                    WorkerMode::Prefill => workers.iter().any(|decode| {
                        decode.mode() == WorkerMode::Decode
                            && decode.version_group() == worker.version_group()
                            && proven(decode)
                    }),
                    WorkerMode::Decode => false,
                }
        })
    }

    /// Check every worker each interval; after a discovery event (Added,
    /// ServicesChanged, ModeChanged, Removed), probe the workers not yet proven
    /// at once, so a ready engine joins in about a probe round trip rather
    /// than at the next interval.
    pub async fn run(ctx: Arc<AppContext>) {
        let health = Arc::clone(
            ctx.active_health
                .as_ref()
                .expect("active health configured"),
        );
        let mut interval = tokio::time::interval(Duration::from_secs(health.config.interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Kept across checks, so an event during one is not missed.
        let mut changed = std::pin::pin!(ctx.registry.membership.changed());
        loop {
            tokio::select! {
                _ = interval.tick() => health.check(&ctx).await,
                () = &mut changed => {
                    changed.set(ctx.registry.membership.changed());
                    health.join(&ctx).await;
                }
            }
        }
    }

    /// Probe the unproven workers up to `success_threshold` times,
    /// [`JOIN_RETRY`] apart, stopping once none is left. Only successes count
    /// here; failures are left to the interval check.
    async fn join(&self, ctx: &AppContext) {
        for attempt in 0..self.config.success_threshold.max(1) {
            if attempt > 0 {
                tokio::time::sleep(JOIN_RETRY).await;
            }
            if !self.prove_unproven(ctx).await {
                break;
            }
        }
    }

    /// One probe of the workers whose health is not proven; true while some remain.
    async fn prove_unproven(&self, ctx: &AppContext) -> bool {
        let workers = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()));
        let unproven: HashSet<u64> = {
            let state = self.state.lock().unwrap();
            (workers.iter())
                .map(|worker| worker.health_instance())
                .filter(|id| !state.streaks.get(id).is_some_and(|s| s.proven))
                .collect()
        };
        if unproven.is_empty() {
            return false;
        }
        let proven = self.probe(ctx, &workers, Some(&unproven)).await;
        let mut flipped = false;
        let mut remaining = false;
        {
            let mut state = self.state.lock().unwrap();
            for worker in &workers {
                let id = worker.health_instance();
                if !unproven.contains(&id) {
                    continue;
                }
                let streak = state.streaks.entry(id).or_default();
                if proven.contains(&id) {
                    streak.failures = 0;
                    streak.successes = streak.successes.saturating_add(1);
                    if streak.successes >= self.config.success_threshold {
                        streak.proven = true;
                        worker.set_active_healthy(true);
                        flipped = true;
                    }
                }
                remaining |= !streak.proven;
            }
        }
        if flipped {
            ctx.registry.membership.readiness_changed();
        }
        remaining
    }

    async fn check(&self, ctx: &AppContext) {
        let workers = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()));
        let proven = self.probe(ctx, &workers, None).await;
        let current: HashSet<_> = ctx
            .registry
            .workers_for(&ModelId(ctx.config.model.id.clone()))
            .iter()
            .map(|worker| worker.health_instance())
            .collect();
        let mut flipped = false;
        let mut state = self.state.lock().unwrap();
        state.streaks.retain(|id, _| current.contains(id));
        for worker in workers {
            let id = worker.health_instance();
            if !current.contains(&id) {
                continue;
            }
            let streak = state.streaks.entry(id).or_default();
            if proven.contains(&id) {
                streak.failures = 0;
                streak.successes = streak.successes.saturating_add(1);
                if streak.successes >= self.config.success_threshold {
                    flipped |= !streak.proven;
                    streak.proven = true;
                    worker.set_active_healthy(true);
                }
            } else {
                streak.successes = 0;
                streak.failures = streak.failures.saturating_add(1);
                if streak.failures >= self.config.failure_threshold {
                    flipped |= streak.proven;
                    streak.proven = false;
                    worker.set_active_healthy(false);
                }
            }
        }
        drop(state);
        if flipped {
            ctx.registry.membership.readiness_changed();
        }
    }

    /// The workers proven this round: live on `/health`, and under generation
    /// mode serving a one-token generation (per prefill/decode pair). With
    /// `focus`, generation probes only pairs with a focused side.
    async fn probe(
        &self,
        ctx: &AppContext,
        workers: &[Arc<Worker>],
        focus: Option<&HashSet<u64>>,
    ) -> HashSet<u64> {
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let freshness = Duration::from_secs(
            self.config
                .interval_secs
                .saturating_add(self.config.timeout_secs),
        );
        let live: HashSet<_> = stream::iter(workers.iter().cloned())
            .map(|worker| async move {
                if worker.recently_generated(freshness) {
                    return Some(worker.health_instance());
                }
                let mut request = ctx
                    .proxy
                    .client_for(worker.protocol())
                    .get(format!("{}/health", worker.url.trim_end_matches('/')))
                    .timeout(timeout);
                if let Some(auth) = &ctx.config.server.worker_auth {
                    request = request.header(reqwest::header::AUTHORIZATION, auth);
                }
                let ok = request.send().await.is_ok_and(|r| r.status().is_success());
                ok.then_some(worker.health_instance())
            })
            .buffer_unordered(16)
            .filter_map(std::future::ready)
            .collect()
            .await;
        if self.config.mode == HealthCheckMode::Http {
            live
        } else {
            self.generation_checks(ctx, workers, &live, timeout, freshness, focus)
                .await
        }
    }

    async fn generation_checks(
        &self,
        ctx: &AppContext,
        workers: &[Arc<Worker>],
        live: &HashSet<u64>,
        timeout: Duration,
        freshness: Duration,
        focus: Option<&HashSet<u64>>,
    ) -> HashSet<u64> {
        let focused =
            |worker: &Worker| focus.is_none_or(|ids| ids.contains(&worker.health_instance()));
        let mut pairs = Vec::new();
        for worker in workers {
            if !live.contains(&worker.health_instance()) || worker.awaiting_bootstrap_port() {
                continue;
            }
            match worker.mode() {
                WorkerMode::Plain if focused(worker) => pairs.push((Arc::clone(worker), None)),
                WorkerMode::Plain => {}
                WorkerMode::Prefill => {
                    for decode in workers {
                        if decode.mode() == WorkerMode::Decode
                            && live.contains(&decode.health_instance())
                            && decode.version_group() == worker.version_group()
                            && (focused(worker) || focused(decode))
                        {
                            pairs.push((Arc::clone(worker), Some(Arc::clone(decode))));
                        }
                    }
                }
                WorkerMode::Decode => {}
            }
        }
        let mut proven = HashSet::new();
        let mut probes = stream::iter(pairs)
            .map(|(prefill, decode)| async move {
                // Completed positive-token client responses prove service without queuing another probe behind them.
                let recent = prefill.recently_generated(freshness)
                    && decode
                        .as_ref()
                        .is_none_or(|worker| worker.recently_generated(freshness));
                let ok = recent
                    || tokio::time::timeout(
                        timeout,
                        crate::server::routes::chat::generation_probe(
                            ctx,
                            Arc::clone(&prefill),
                            decode.clone(),
                        ),
                    )
                    .await
                    .unwrap_or(false);
                (prefill, decode, ok)
            })
            .buffer_unordered(8);
        while let Some((prefill, decode, ok)) = probes.next().await {
            if ok {
                proven.insert(prefill.health_instance());
                if let Some(decode) = decode {
                    proven.insert(decode.health_instance());
                }
            }
        }
        proven
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    use axum::{
        extract::State,
        http::StatusCode,
        routing::{get, post},
        Json,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Default)]
    struct ProbeState {
        hang: AtomicBool,
        health_calls: AtomicUsize,
    }

    struct Fixture {
        ctx: Arc<AppContext>,
        health: Arc<ActiveHealth>,
        worker: Arc<Worker>,
        probe: Arc<ProbeState>,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn fixture() -> Fixture {
        let probe = Arc::new(ProbeState::default());
        let app = axum::Router::new()
            .route(
                "/health",
                get(|State(probe): State<Arc<ProbeState>>| async move {
                    probe.health_calls.fetch_add(1, Ordering::Relaxed);
                    if probe.hang.load(Ordering::Relaxed) {
                        std::future::pending::<()>().await;
                    }
                    StatusCode::OK
                }),
            )
            .route(
                "/generate",
                post(|| async {
                    Json(serde_json::json!({"text":"OK","meta_info":{"completion_tokens":1}}))
                }),
            )
            .with_state(Arc::clone(&probe));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async {
            axum::serve(listener, app).await.unwrap();
        });
        let mut ctx = AppContext::stub();
        ctx.policies =
            Arc::new(crate::policies::factory::build_registry_with_defaults(&ctx.config).unwrap());
        ctx.registry.require_active_health();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("original".into()),
                url,
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                ..Default::default()
            })
            .unwrap();
        let health = Arc::new(ActiveHealth::new(HealthCheckConfig {
            mode: HealthCheckMode::Generation,
            interval_secs: 1,
            timeout_secs: 1,
            failure_threshold: 3,
            success_threshold: 1,
        }));
        ctx.active_health = Some(Arc::clone(&health));
        ctx.mark_ready();
        let ctx = Arc::new(ctx);
        health.check(&ctx).await;
        assert_eq!(
            crate::server::routes::health::readyz(State(Arc::clone(&ctx))).await,
            StatusCode::OK
        );
        let worker = Arc::clone(&ctx.registry.all()[0]);
        Fixture {
            ctx,
            health,
            worker,
            probe,
            server,
        }
    }

    async fn expire_generation_proof() {
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(3)).await;
        tokio::time::resume();
    }

    fn add_unproved_worker(ctx: &AppContext) {
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("new".into()),
                url: "http://127.0.0.1:1".into(),
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                ..Default::default()
            })
            .unwrap();
    }

    #[tokio::test]
    async fn all_probe_timeouts_preserve_readiness_until_failure_threshold() {
        let f = fixture().await;
        f.probe.hang.store(true, Ordering::Relaxed);
        expire_generation_proof().await;
        for cycle in 1..=3 {
            f.health.check(&f.ctx).await;
            let status = crate::server::routes::health::readyz(State(Arc::clone(&f.ctx))).await;
            assert_eq!(
                status,
                if cycle < 3 {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                },
                "cycle {cycle}"
            );
        }
        f.probe.hang.store(false, Ordering::Relaxed);
        f.health.check(&f.ctx).await;
        assert!(
            f.health.ready(&f.ctx),
            "successful recovery restores readiness"
        );
    }

    #[tokio::test]
    async fn adding_unproved_worker_preserves_existing_readiness() {
        let f = fixture().await;
        add_unproved_worker(&f.ctx);
        assert_eq!(
            crate::server::routes::health::readyz(State(Arc::clone(&f.ctx))).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn topology_change_preserves_surviving_workers_failure_streak() {
        let f = fixture().await;
        f.probe.hang.store(true, Ordering::Relaxed);
        expire_generation_proof().await;
        f.health.check(&f.ctx).await;
        f.health.check(&f.ctx).await;
        assert!(f.worker.active_health_passes());
        add_unproved_worker(&f.ctx);
        f.health.check(&f.ctx).await;
        assert!(
            !f.worker.active_health_passes(),
            "adding an endpoint must not postpone exclusion of a failing survivor"
        );
    }

    #[tokio::test]
    async fn recent_generation_proof_skips_http_health_gate() {
        let f = fixture().await;
        f.worker.record_generation_success();
        f.probe.hang.store(true, Ordering::Relaxed);
        let calls = f.probe.health_calls.load(Ordering::Relaxed);
        f.health.check(&f.ctx).await;
        assert_eq!(f.probe.health_calls.load(Ordering::Relaxed), calls);
        assert!(f.health.ready(&f.ctx));
    }

    /// A worker discovered between intervals is probed at once and joins after
    /// `success_threshold` probes `JOIN_RETRY` apart, not at the next interval.
    #[tokio::test]
    async fn an_added_worker_joins_within_the_join_window_not_the_interval() {
        let app = axum::Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/generate",
                post(|| async {
                    Json(serde_json::json!({"text":"OK","meta_info":{"completion_tokens":1}}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async {
            axum::serve(listener, app).await.unwrap();
        });
        let mut ctx = AppContext::stub();
        ctx.policies =
            Arc::new(crate::policies::factory::build_registry_with_defaults(&ctx.config).unwrap());
        ctx.registry.require_active_health();
        ctx.active_health = Some(Arc::new(ActiveHealth::new(HealthCheckConfig {
            mode: HealthCheckMode::Generation,
            interval_secs: 60,
            timeout_secs: 1,
            failure_threshold: 3,
            success_threshold: 2,
        })));
        let ctx = Arc::new(ctx);
        let health = tokio::spawn(ActiveHealth::run(Arc::clone(&ctx)));
        // Let the first interval check pass with no workers.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let ready = ctx.registry.membership.ready();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("late".into()),
                url,
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                ..Default::default()
            })
            .unwrap();
        let worker = Arc::clone(&ctx.registry.all()[0]);
        assert!(!worker.active_health_passes(), "unproven until probed");
        let added = std::time::Instant::now();
        ctx.registry
            .membership
            .record(crate::workers::MembershipEvent::Added);
        tokio::time::timeout(Duration::from_secs(2), ready)
            .await
            .expect("proving the worker wakes readiness watchers");
        assert!(worker.active_health_passes());
        assert!(ctx.active_health.as_ref().unwrap().ready(&ctx));
        assert!(
            added.elapsed() >= JOIN_RETRY,
            "two probes for a threshold of two"
        );
        health.abort();
        server.abort();
    }

    #[test]
    fn topology_proof_does_not_survive_removal_and_readdition_of_the_same_endpoint() {
        let ctx = AppContext::stub();
        let spec = || WorkerSpec {
            id: WorkerId("worker".into()),
            url: "http://127.0.0.1:1".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(ctx.config.model.id.clone())],
            ..Default::default()
        };
        ctx.registry.add(spec()).unwrap();
        ctx.registry.all()[0].record_generation_success();
        let health = ActiveHealth::new(HealthCheckConfig::default());
        {
            let mut state = health.state.lock().unwrap();
            state.streaks.insert(
                ctx.registry.all()[0].health_instance(),
                Streak {
                    successes: 2,
                    failures: 0,
                    proven: true,
                },
            );
        }
        assert!(health.ready(&ctx));
        ctx.registry.add(spec()).unwrap();
        assert!(
            health.ready(&ctx),
            "unchanged metadata refresh preserves live proof"
        );
        assert!(ctx.registry.all()[0].recently_generated(Duration::from_secs(1)));
        ctx.registry.remove(&WorkerId("worker".into()));
        assert!(!health.ready(&ctx));
        ctx.registry.add(spec()).unwrap();
        assert!(
            !health.ready(&ctx),
            "recreated endpoints need fresh generation evidence"
        );
        assert!(!ctx.registry.all()[0].recently_generated(Duration::from_secs(1)));
    }
}
