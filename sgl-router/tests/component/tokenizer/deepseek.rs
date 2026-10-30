// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use sgl_router::tokenizer::chat_formatter::ChatFormatter;

/// DeepSeek V4.1 image parts render to their placeholders in the processor
/// (7efb4242dc, "DeepSeek V4.1 image input through the processor and the
/// cell"); whether media then takes the token path is the cell's decision.
#[test]
fn v41_renders_media() {
    let formatter = ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").unwrap();
    let request = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]});
    assert!(formatter.render(&request).is_ok());
}
