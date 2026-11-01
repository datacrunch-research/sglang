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
    // Media renders on the engine; the router never forwards ids for it.
    let image = json!({"type":"image_url","image_url":{"url":"https://example.com/a.png"}});
    let request = json!({"messages":[{"role":"user","content":[image]}]});
    assert!(formatter.encode(&tokenizer, &request).is_err());
}

/// Kimi-K3.1 ships Kimi-K3's `tiktoken.model` and `tokenization_kimi.py` under `config.json`
/// model_type `kimi_k31`, which dynamo-tokenizers does not list: the router loads it as
/// Kimi-K3 and every text encodes and decodes exactly as on the Kimi-K3 checkpoint. The
/// checkpoint's own `config.json` still selects the chat formatter, and there is none for
/// `kimi_k31` (its "medium" thinking effort is not Kimi-K3's), so its chat stays with the
/// engines.
#[test]
fn kimi_k31_tokenizes_as_kimi_k3() {
    let k3 = fixture::tokenizer();
    let k31 = fixture::tokenizer();
    std::fs::write(
        k31.path().join("config.json"),
        r#"{"model_type":"kimi_k31"}"#,
    )
    .unwrap();
    let path = |dir: &tempfile::TempDir| {
        dir.path()
            .join("tiktoken.model")
            .to_str()
            .unwrap()
            .to_owned()
    };
    let (k3_tokenizer, k31_tokenizer) = (
        adapter::load(&path(&k3)).unwrap(),
        adapter::load(&path(&k31)).unwrap(),
    );
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../../fixtures/kimi_k3/prompts.json")).unwrap();
    let mut texts: Vec<String> = cases
        .iter()
        .flat_map(|case| {
            case["request"]["messages"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|message| message["content"].as_str().map(str::to_owned))
        .collect();
    texts.extend([
        "[BOS]hi<|end_of_msg|>".to_owned(),
        "plain text, 1234 ünïcödé".to_owned(),
    ]);
    assert!(texts.len() > 2);
    for text in &texts {
        let ids = adapter::encode(&k3_tokenizer, text).unwrap();
        assert_eq!(
            adapter::encode(&k31_tokenizer, text).unwrap(),
            ids,
            "{text}"
        );
        for skip in [false, true] {
            assert_eq!(
                adapter::decode_complete(&k31_tokenizer, &ids, skip).unwrap(),
                adapter::decode_complete(&k3_tokenizer, &ids, skip).unwrap(),
                "{text}"
            );
        }
    }
    assert!(ChatFormatter::load("served-alias", &path(&k31))
        .unwrap()
        .is_none());
    // An unlisted model type is still refused.
    std::fs::write(
        k31.path().join("config.json"),
        r#"{"model_type":"kimi_k99"}"#,
    )
    .unwrap();
    assert!(adapter::load(&path(&k31)).is_err());
}
