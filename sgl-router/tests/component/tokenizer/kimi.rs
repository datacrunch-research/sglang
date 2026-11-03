// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter};
use sha2::{Digest, Sha256};

#[path = "../../fixtures/kimi_k3.rs"]
mod fixture;

#[test]
fn kimi_tokens_match_sglang() {
    let fixture = fixture::tokenizer();
    let path = fixture.path().join("tiktoken.model");
    let path = path.to_str().unwrap();
    let tokenizer = adapter::load(path).unwrap();
    let formatter = ChatFormatter::load("served-alias", path).unwrap().unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../../fixtures/kimi_k3/prompts.json")).unwrap();
    for case in cases {
        let mut request = case["request"].clone();
        if let Some(repeat) = case["repeat"].as_u64() {
            request["messages"][0]["content"] = request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .repeat(repeat as usize)
                .into();
        }
        let ids = formatter.encode(&tokenizer, &request).unwrap();
        let mut hash = Sha256::new();
        for id in &ids {
            hash.update(id.to_le_bytes());
        }
        assert_eq!(json!(ids.len()), case["token_count"], "{}", case["name"]);
        assert_eq!(
            format!("{:x}", hash.finalize()),
            case["sha256"],
            "{}",
            case["name"]
        );
    }
    let request = json!({"messages":[{"role":"user","content":"hi"}], "chat_template_kwargs":{"thinking_effort":null}});
    assert!(formatter.encode(&tokenizer, &request).is_err());
    // SGLang's `_validate_request` rejects a required tool_choice without tools.
    let request = json!({"messages":[{"role":"user","content":"hi"}], "tool_choice":"required"});
    assert!(formatter.encode(&tokenizer, &request).is_err());
    // K3 image parts render to their placeholders in the processor
    // (1482d201a8, "images on the token path: Kimi K3 renders them").
    let image = json!({"type":"image_url","image_url":{"url":"https://example.com/a.png"}});
    let request = json!({"messages":[{"role":"user","content":[image]}]});
    assert!(formatter.encode(&tokenizer, &request).is_ok());
}

/// A multimodal wrapper names its own `model_type` in `config.json`, which dynamo-tokenizers
/// does not list for tiktoken, while its `text_config.model_type` is one it lists. The router
/// encodes and decodes the wrapper's vocabulary as the text model's. The checkpoint's own
/// `config.json` still selects the chat formatter: none for the wrapper type, so its chat
/// stays with the engines.
#[test]
fn a_wrapper_loads_its_tiktoken_vocabulary_as_its_text_model() {
    let text_model = fixture::tokenizer();
    let wrapper = fixture::tokenizer();
    let tiktoken = |dir: &tempfile::TempDir| {
        dir.path()
            .join("tiktoken.model")
            .to_str()
            .unwrap()
            .to_owned()
    };
    let relabel = |dir: &tempfile::TempDir, config: Value| {
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    };
    relabel(
        &wrapper,
        json!({"model_type": "test_wrapper", "text_config": {"model_type": "kimi_linear"}}),
    );
    let (text_path, wrapper_path) = (tiktoken(&text_model), tiktoken(&wrapper));
    let text_tokenizer = adapter::load(&text_path).unwrap();
    let wrapper_tokenizer = adapter::load(&wrapper_path).unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../../fixtures/kimi_k3/prompts.json")).unwrap();
    let mut texts: Vec<String> = cases
        .iter()
        .filter_map(|case| case["request"]["messages"].as_array())
        .flatten()
        .filter_map(|message| message["content"].as_str().map(str::to_owned))
        .collect();
    texts.push("[BOS]system<|end_of_msg|>plain text, 1234 and ünïcödé".to_owned());
    assert!(texts.len() > 1);
    for text in &texts {
        let ids = adapter::encode(&text_tokenizer, text).unwrap();
        assert_eq!(
            adapter::encode(&wrapper_tokenizer, text).unwrap(),
            ids,
            "{text}"
        );
        for skip_special in [false, true] {
            assert_eq!(
                adapter::decode_complete(&wrapper_tokenizer, &ids, skip_special).unwrap(),
                adapter::decode_complete(&text_tokenizer, &ids, skip_special).unwrap(),
                "{text}"
            );
        }
    }
    assert!(ChatFormatter::load("served-alias", &wrapper_path)
        .unwrap()
        .is_none());
}

/// A wrapper whose text model the tokenizer does not list either, or one with no
/// `text_config`, is refused as before, with the error for its own type.
#[test]
fn an_unknown_wrapper_is_still_refused() {
    let wrapper = fixture::tokenizer();
    let path = wrapper.path().join("tiktoken.model");
    let path = path.to_str().unwrap();
    for config in [
        json!({"model_type": "test_wrapper", "text_config": {"model_type": "test_text"}}),
        json!({"model_type": "test_wrapper"}),
        json!({"model_type": "test_wrapper", "text_config": {"model_type": "test_wrapper"}}),
    ] {
        std::fs::write(wrapper.path().join("config.json"), config.to_string()).unwrap();
        let error = format!("{:#}", adapter::load(path).err().unwrap());
        assert!(error.contains("'test_wrapper'"), "{config}: {error}");
    }
}
