//! `sglang.runtime.v1.SglangService` served beside `api.v1`, so a client that
//! speaks the engine bridge's protocol (sgl-router's engine side, or an
//! upstream router that forwards token ids) reaches this server over gRPC
//! without a proto change on its side. Each served RPC is translated to its
//! `api.v1` twin and run through [`GrpcService`]; `meta_info` goes back as
//! the bridge encodes it, one JSON-encoded string per field, and an item's
//! failure rides the stream as an abort with the error's status code. RPCs
//! the engine bridge serves and this server has no twin for (profiling,
//! weight updates, engine-state watches) stay `Unimplemented`.

use std::collections::HashMap;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService as ApiService;
use sglang_api_types::runtime::v1 as rt;
use sglang_api_types::runtime::v1::sglang_service_server::SglangService as RuntimeService;
use tonic::{Request, Response, Status};

use super::service::GrpcService;

type RuntimeStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// The `runtime.v1` face of the `api.v1` adapter.
pub(crate) struct RuntimeFacade {
    api: GrpcService,
}

impl RuntimeFacade {
    pub(crate) fn new(api: GrpcService) -> Self {
        Self { api }
    }
}

fn one_string(value: String) -> api::StringOrList {
    api::StringOrList {
        value: Some(api::string_or_list::Value::One(value)),
    }
}

fn one_bool(value: bool) -> api::BoolOrList {
    api::BoolOrList {
        value: Some(api::bool_or_list::Value::One(value)),
    }
}

fn one_i64(value: i64) -> api::Int64OrList {
    api::Int64OrList {
        value: Some(api::int64_or_list::Value::One(value)),
    }
}

fn one_optional_string(value: Option<String>) -> api::OptionalStringOrList {
    api::OptionalStringOrList {
        value: Some(api::optional_string_or_list::Value::One(
            api::OptionalString { value },
        )),
    }
}

fn one_optional_i64(value: Option<i64>) -> api::OptionalInt64OrList {
    api::OptionalInt64OrList {
        value: Some(api::optional_int64_or_list::Value::One(
            api::OptionalInt64 { value },
        )),
    }
}

/// Media sources as the bridge carries them: one list of URLs or data URLs.
fn media(sources: Vec<String>) -> Option<api::MediaInput> {
    if sources.is_empty() {
        return None;
    }
    let items = sources
        .into_iter()
        .map(|source| api::OptionalMediaItem {
            item: Some(api::MediaItem {
                value: Some(api::media_item::Value::Source(source)),
            }),
        })
        .collect();
    Some(api::MediaInput {
        value: Some(api::media_input::Value::Many(api::MediaItems { items })),
    })
}

/// The bridge's sampling parameters as `api.v1` sampling parameters. The
/// deprecated flat `json_schema`/`regex` fields and `guided_decoding` both
/// land on the `api.v1` constraint fields; `guided_decoding` wins.
#[allow(deprecated)]
pub(super) fn sampling_params(sp: rt::SamplingParams) -> api::SamplingParams {
    let mut out = api::SamplingParams {
        temperature: sp.temperature.map(f64::from),
        top_p: sp.top_p.map(f64::from),
        top_k: sp.top_k.map(i64::from),
        min_p: sp.min_p.map(f64::from),
        frequency_penalty: sp.frequency_penalty.map(f64::from),
        presence_penalty: sp.presence_penalty.map(f64::from),
        repetition_penalty: sp.repetition_penalty.map(f64::from),
        max_new_tokens: sp.max_new_tokens.map(i64::from),
        min_new_tokens: sp.min_new_tokens.map(i64::from),
        stop: (!sp.stop.is_empty()).then_some(api::StringOrList {
            value: Some(api::string_or_list::Value::Many(api::StringList {
                items: sp.stop,
            })),
        }),
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then(|| api::Int64List {
            items: sp.stop_token_ids.iter().map(|&id| i64::from(id)).collect(),
        }),
        ignore_eos: sp.ignore_eos,
        n: sp.n.map(i64::from),
        json_schema: sp.json_schema,
        regex: sp.regex,
        sampling_seed: sp.seed,
        ..Default::default()
    };
    if let Some(constraint) = sp.guided_decoding.and_then(|g| g.constraint) {
        use rt::guided_decoding::Constraint;
        match constraint {
            Constraint::JsonSchema(schema) => out.json_schema = Some(schema),
            Constraint::Regex(regex) => out.regex = Some(regex),
            Constraint::Ebnf(ebnf) => out.ebnf = Some(ebnf),
            Constraint::StructuralTag(tag) => out.structural_tag = Some(tag),
            Constraint::Choice(choice) => out.choice = choice.values,
        }
    }
    out
}

/// A bridge `Generate` request as the `api.v1` request it means: one prompt
/// of token ids with the same sampling, logprob, routing and media fields.
pub(super) fn to_api_request(request: rt::GenerateRequest) -> api::GenerateRequest {
    let disaggregated = request.disaggregated_params;
    api::GenerateRequest {
        rid: request.rid.map(one_string),
        input_ids: Some(api::TokenIdsOrList {
            value: Some(api::token_ids_or_list::Value::One(api::TokenIds {
                ids: request.input_ids.iter().map(|&id| i64::from(id)).collect(),
            })),
        }),
        stream: request.stream,
        sampling_params: request.sampling_params.map(|sp| api::SamplingParamsOrList {
            value: Some(api::sampling_params_or_list::Value::One(sampling_params(
                sp,
            ))),
        }),
        return_logprob: request.return_logprob.map(one_bool),
        logprob_start_len: request.logprob_start_len.map(|n| one_i64(i64::from(n))),
        top_logprobs_num: request.top_logprobs_num.map(|n| one_i64(i64::from(n))),
        bootstrap_host: disaggregated
            .as_ref()
            .map(|d| one_optional_string(Some(d.bootstrap_host.clone()))),
        bootstrap_port: disaggregated
            .as_ref()
            .map(|d| one_optional_i64(Some(i64::from(d.bootstrap_port)))),
        bootstrap_room: disaggregated
            .as_ref()
            .map(|d| one_optional_i64(Some(d.bootstrap_room))),
        routed_dp_rank: request.routed_dp_rank.map(i64::from),
        image_data: media(request.image_data),
        mm_hashes: (!request.mm_hashes.is_empty()).then_some(api::StringListOrList {
            value: Some(api::string_list_or_list::Value::One(api::StringList {
                items: request.mm_hashes,
            })),
        }),
        video_data: media(request.video_data),
        audio_data: media(request.audio_data),
        priority: request.priority.map(i64::from),
        require_reasoning: request.require_reasoning,
        lora_path: request
            .lora_path
            .map(|path| one_optional_string(Some(path))),
        routing_key: request.routing_key,
        max_thinking_tokens: request.max_thinking_tokens.map(u64::from),
        session_id: request.session_id,
        ..Default::default()
    }
}

/// `meta_info` as the bridge encodes it: every field a JSON-encoded string.
fn encode_meta_info(meta: &api::GenerateMetaInfo) -> HashMap<String, String> {
    let value = serde_json::to_value(meta).unwrap_or_default();
    match value {
        serde_json::Value::Object(map) => {
            map.into_iter().map(|(k, v)| (k, v.to_string())).collect()
        }
        _ => HashMap::new(),
    }
}

/// Which output ids a bridge frame carries: the bridge streams the new ids of
/// each frame, so a cumulative `api.v1` frame is cut after what was sent.
pub(super) struct Progress {
    sent: HashMap<u32, usize>,
}

impl Progress {
    fn new_ids(&mut self, index: u32, ids: &[i64], incremental: bool) -> Vec<i32> {
        let start = if incremental {
            0
        } else {
            (*self.sent.get(&index).unwrap_or(&0)).min(ids.len())
        };
        let new = &ids[start..];
        if !incremental {
            self.sent.insert(index, ids.len());
        }
        new.iter()
            .map(|&id| i32::try_from(id).unwrap_or(i32::MAX))
            .collect()
    }
}

/// An `api.v1` stream item as the bridge frame the client expects.
pub(super) fn to_runtime_frame(
    item: api::GenerateStreamItem,
    progress: &mut Progress,
    incremental: bool,
    return_text: bool,
) -> rt::GenerateResponse {
    match item.item {
        Some(api::generate_stream_item::Item::Frame(frame)) => {
            let index = frame.index.unwrap_or(0);
            let ids = frame
                .output_ids
                .as_ref()
                .map(|t| t.ids.as_slice())
                .unwrap_or(&[]);
            let output_ids = progress.new_ids(index, ids, incremental);
            let finished = frame
                .meta_info
                .as_ref()
                .is_some_and(|m| m.finish_reason.is_some());
            rt::GenerateResponse {
                output_ids,
                meta_info: frame
                    .meta_info
                    .as_ref()
                    .map(encode_meta_info)
                    .unwrap_or_default(),
                finished,
                text: (return_text && !frame.text.is_empty()).then_some(frame.text),
                index: frame.index,
            }
        }
        Some(api::generate_stream_item::Item::Error(error)) => {
            let body = error.error.unwrap_or_default();
            let finish_reason = serde_json::json!({
                "type": "abort",
                "status_code": body.code,
                "message": body.message,
            });
            rt::GenerateResponse {
                output_ids: Vec::new(),
                meta_info: HashMap::from([("finish_reason".to_owned(), finish_reason.to_string())]),
                finished: true,
                text: None,
                index: error.index,
            }
        }
        None => rt::GenerateResponse {
            finished: true,
            ..Default::default()
        },
    }
}

#[tonic::async_trait]
impl RuntimeService for RuntimeFacade {
    async fn generate(
        &self,
        request: Request<rt::GenerateRequest>,
    ) -> Result<Response<RuntimeStream<rt::GenerateResponse>>, Status> {
        let (metadata, extensions, request) = request.into_parts();
        let return_text = request.return_text.unwrap_or(false);
        let api_request = Request::from_parts(metadata, extensions, to_api_request(request));
        let incremental = self.api.incremental_streaming_output();
        let stream = ApiService::generate(&self.api, api_request)
            .await?
            .into_inner();
        let mut progress = Progress {
            sent: HashMap::new(),
        };
        let mapped = stream.map(move |item| {
            item.map(|item| to_runtime_frame(item, &mut progress, incremental, return_text))
        });
        Ok(Response::new(Box::pin(mapped)))
    }

    async fn abort(
        &self,
        request: Request<rt::AbortRequest>,
    ) -> Result<Response<rt::AbortResponse>, Status> {
        let request = request.into_inner();
        let reply = ApiService::abort(
            &self.api,
            Request::new(api::AbortRequest {
                rid: request.rid,
                abort_all: request.abort_all,
            }),
        )
        .await?
        .into_inner();
        Ok(Response::new(rt::AbortResponse {
            success: reply.success,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<rt::HealthCheckRequest>,
    ) -> Result<Response<rt::HealthCheckResponse>, Status> {
        let reply =
            ApiService::health_check(&self.api, Request::new(api::HealthCheckRequest::default()))
                .await?
                .into_inner();
        Ok(Response::new(rt::HealthCheckResponse {
            healthy: reply.healthy,
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<rt::GetModelInfoRequest>,
    ) -> Result<Response<rt::GetModelInfoResponse>, Status> {
        let info = ApiService::get_model_info(
            &self.api,
            Request::new(api::GetModelInfoRequest::default()),
        )
        .await?
        .into_inner();
        Ok(Response::new(rt::GetModelInfoResponse {
            model_path: info.model_path.clone(),
            json_info: serde_json::to_string(&info).unwrap_or_default(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<rt::GetServerInfoRequest>,
    ) -> Result<Response<rt::GetServerInfoResponse>, Status> {
        let info = ApiService::get_server_info(
            &self.api,
            Request::new(api::GetServerInfoRequest::default()),
        )
        .await?
        .into_inner();
        Ok(Response::new(rt::GetServerInfoResponse {
            json_info: serde_json::to_string(&info).unwrap_or_default(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_requests_become_api_requests() {
        let request = rt::GenerateRequest {
            input_ids: vec![1, 2, 3],
            sampling_params: Some(rt::SamplingParams {
                temperature: Some(0.5),
                max_new_tokens: Some(8),
                stop: vec!["x".into()],
                stop_token_ids: vec![7],
                n: Some(1),
                seed: Some(9),
                guided_decoding: Some(rt::GuidedDecoding {
                    constraint: Some(rt::guided_decoding::Constraint::Ebnf(
                        "root ::= \"a\"".into(),
                    )),
                }),
                ..Default::default()
            }),
            stream: Some(true),
            rid: Some("r1".into()),
            routing_key: Some("k".into()),
            image_data: vec!["data:image/png;base64,AA".into()],
            mm_hashes: vec!["sha256:00".into()],
            return_text: Some(false),
            ..Default::default()
        };
        let api = to_api_request(request);
        assert_eq!(
            api.input_ids.unwrap().value,
            Some(api::token_ids_or_list::Value::One(api::TokenIds {
                ids: vec![1, 2, 3]
            }))
        );
        assert_eq!(api.rid, Some(one_string("r1".into())));
        assert_eq!(api.stream, Some(true));
        assert_eq!(api.routing_key.as_deref(), Some("k"));
        let Some(api::sampling_params_or_list::Value::One(sp)) = api.sampling_params.unwrap().value
        else {
            panic!("one sampling params")
        };
        assert_eq!(sp.temperature, Some(0.5));
        assert_eq!(sp.max_new_tokens, Some(8));
        assert_eq!(sp.stop_token_ids.unwrap().items, vec![7]);
        assert_eq!(sp.sampling_seed, Some(9));
        assert_eq!(sp.ebnf.as_deref(), Some("root ::= \"a\""));
        assert!(matches!(
            api.image_data.unwrap().value,
            Some(api::media_input::Value::Many(items)) if items.items.len() == 1
        ));
        assert!(matches!(
            api.mm_hashes.unwrap().value,
            Some(api::string_list_or_list::Value::One(list)) if list.items == vec!["sha256:00"]
        ));
    }

    #[test]
    fn api_frames_become_bridge_frames_with_json_meta_info() {
        let frame = |ids: &[i64], finished: bool| api::GenerateStreamItem {
            item: Some(api::generate_stream_item::Item::Frame(
                api::GenerateResponse {
                    text: "hi".into(),
                    meta_info: Some(api::GenerateMetaInfo {
                        id: "r1".into(),
                        prompt_tokens: 3,
                        completion_tokens: ids.len() as u64,
                        finish_reason: finished.then(|| api::FinishReason::stop(None)),
                        ..Default::default()
                    }),
                    output_ids: Some(api::TokenIds { ids: ids.to_vec() }),
                    index: None,
                    prompt_token_ids: None,
                },
            )),
        };
        // Cumulative api frames are cut to their new ids.
        let mut progress = Progress {
            sent: HashMap::new(),
        };
        let first = to_runtime_frame(frame(&[5], false), &mut progress, false, false);
        assert_eq!(first.output_ids, vec![5]);
        assert!(!first.finished);
        assert!(first.text.is_none(), "no text unless asked");
        assert_eq!(first.meta_info["id"], "\"r1\"");
        assert_eq!(first.meta_info["prompt_tokens"], "3");
        let second = to_runtime_frame(frame(&[5, 6], true), &mut progress, false, true);
        assert_eq!(second.output_ids, vec![6]);
        assert!(second.finished);
        assert_eq!(second.text.as_deref(), Some("hi"));
        let reason: serde_json::Value =
            serde_json::from_str(&second.meta_info["finish_reason"]).unwrap();
        assert_eq!(reason["type"], "stop");
        // Incremental api frames pass through as they are.
        let mut progress = Progress {
            sent: HashMap::new(),
        };
        let delta = to_runtime_frame(frame(&[9], false), &mut progress, true, false);
        assert_eq!(delta.output_ids, vec![9]);
        // An item's failure is an abort with the error's status code.
        let error = api::GenerateStreamItem {
            item: Some(api::generate_stream_item::Item::Error(
                api::GenerateStreamError {
                    error: Some(api::ErrorBody {
                        message: "boom".into(),
                        code: 429,
                    }),
                    index: Some(0),
                },
            )),
        };
        let aborted = to_runtime_frame(error, &mut progress, true, false);
        assert!(aborted.finished);
        let reason: serde_json::Value =
            serde_json::from_str(&aborted.meta_info["finish_reason"]).unwrap();
        assert_eq!(reason["type"], "abort");
        assert_eq!(reason["status_code"], 429);
        assert_eq!(aborted.index, Some(0));
    }
}
