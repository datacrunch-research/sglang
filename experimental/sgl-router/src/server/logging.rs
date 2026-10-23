// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! JSON tracing with bounded structured diagnostic fields, preserving normal filtering.

use serde_json::{Map, Value};
use std::fmt;
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::{
    fmt::{
        format::Writer,
        time::{FormatTime, SystemTime},
        FmtContext, FormatEvent, FormatFields, FormattedFields,
    },
    registry::LookupSpan,
};

pub struct StructuredJson;
#[derive(Default)]
struct Fields(Map<String, Value>);
impl Fields {
    fn insert(&mut self, field: &Field, value: Value) {
        let name = field.name();
        if name == "lifecycle_json" {
            if let Value::String(text) = &value {
                if let Ok(Value::Object(fields)) = serde_json::from_str(text) {
                    self.0.extend(fields);
                    return;
                }
            }
        }
        if let (Some(key), Value::String(text)) = (name.strip_suffix("_json"), &value) {
            if let Ok(structured) = serde_json::from_str(text) {
                self.0.insert(key.to_owned(), structured);
                return;
            }
        }
        self.0.insert(name.to_owned(), value);
    }
}
impl Visit for Fields {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.insert(
            field,
            serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number),
        );
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, value.into());
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, value.into());
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, value.into());
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, value.into());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert(field, format!("{value:?}").into());
    }
}
impl<S, N> FormatEvent<S, N> for StructuredJson
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let mut timestamp = String::new();
        SystemTime.format_time(&mut Writer::new(&mut timestamp))?;
        let meta = event.metadata();
        let mut value = serde_json::json!({"timestamp":timestamp,"level":meta.level().as_str(),"target":meta.target(),"fields":fields.0});
        if let Some(scope) = ctx.event_scope() {
            let mut spans = Vec::new();
            for span in scope.from_root() {
                let ext = span.extensions();
                let mut attrs = ext
                    .get::<FormattedFields<N>>()
                    .and_then(|f| serde_json::from_str::<Map<String, Value>>(&f.to_string()).ok())
                    .unwrap_or_default();
                attrs.insert("name".into(), span.name().into());
                spans.push(Value::Object(attrs));
            }
            if let Some(last) = spans.last() {
                value["span"] = last.clone();
            }
            value["spans"] = spans.into();
        }
        let line = serde_json::to_string(&value).map_err(|_| fmt::Error)?;
        writeln!(writer, "{line}")
    }
}

static RELOAD: std::sync::OnceLock<
    tracing_subscriber::reload::Handle<tracing_subscriber::EnvFilter, tracing_subscriber::Registry>,
> = std::sync::OnceLock::new();
pub fn set_reload_handle(
    handle: tracing_subscriber::reload::Handle<
        tracing_subscriber::EnvFilter,
        tracing_subscriber::Registry,
    >,
) {
    let _ = RELOAD.set(handle);
}
pub fn reload_filter(filter: &str) -> Result<(), String> {
    if filter.len() > 1024 {
        return Err("filter exceeds 1024 bytes".into());
    }
    let filter = tracing_subscriber::EnvFilter::try_new(filter).map_err(|e| e.to_string())?;
    RELOAD
        .get()
        .ok_or_else(|| "log-level reload unavailable".to_owned())?
        .reload(filter)
        .map_err(|e| e.to_string())
}
