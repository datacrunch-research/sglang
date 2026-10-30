//! The model-family seam of the server MM pipeline.
//!
//! Design rule: **families produce data, the driver owns control flow.** A
//! family never sees the request loop, the thread pool, or the failure
//! protocol — it implements [`MmFamilyProcessor`], turning decoded media into
//! named tensors and describing its prompt geometry as a [`TokenLayout`]
//! value. `driver::process` applies the layout mechanically, so expansion,
//! per-item offsets, and position inputs all derive from one declarative
//! structure and every family gets identical failure semantics for free.

/// Typed tensor payload. Grows a variant per dtype actually produced by a
/// family — not speculatively.
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// Raw BF16 bits, exposed to numpy as u16 without copying the allocation.
    Bf16(Vec<u16>),
    /// Raw u8 pixels: Kimi K3's `[C, H, W]` source image, which the model
    /// resizes and normalizes on the GPU (Python's deferred preprocessing).
    U8(Vec<u8>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::Bf16(data) => data.len(),
            Self::U8(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Named auxiliary tensors that reach the model runner as kwargs — the Rust
/// analogue of Python's `MultimodalDataItem.model_specific_data`, e.g. qwen's
/// `image_grid_thw`.
pub type NamedTensors = Vec<(String, Tensor)>;

/// One decoded media item handed to [`MmFamilyProcessor::process_item`].
/// Grows a variant per modality as families that need it are ported.
pub enum DecodedMedia {
    /// HWC u8 RGB.
    Image {
        rgb: Vec<u8>,
        height: usize,
        width: usize,
    },
}

/// Family-internal geometry of one processed item, consumed by
/// [`MmFamilyProcessor::layout`] / [`MmFamilyProcessor::positions`]. Grows a
/// variant per family style; the driver never interprets it.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// `[t, h, w]` patch grid (`t` = 1 for still images).
    Grid([u32; 3]),
    /// A `[t, h, w]` patch grid plus the source image's size and the item's
    /// token count, for families whose prompt framing names the source size
    /// (Kimi K3: `<|media_begin|>image {width}x{height}<|media_content|>`).
    SizedGrid {
        grid: [u32; 3],
        width: u32,
        height: u32,
        tokens: usize,
    },
}

/// One processed media item, mirroring Python's `MultimodalDataItem`: the
/// primary feature tensor, named auxiliary tensors, and the geometry the
/// family's own `layout`/`positions` hooks need.
pub struct ProcessedItem {
    /// The model's feature tensor for this item (qwen: `pixel_values`).
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Geometry,
}

/// The tokens one media item occupies in the expanded prompt.
pub enum TokenPattern {
    /// N copies of one placeholder id (qwen-style).
    Repeat { id: i64, n: usize },
    /// An explicit id sequence — tile markers, row separators, wrapper
    /// tokens (minicpm/internvl-style structured expansions).
    Explicit(Vec<i64>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy `src` (a range into the original ids) verbatim.
    Text(std::ops::Range<usize>),
    /// Ids the family inserts that are neither source text nor media: framing
    /// tokens around a media span (Kimi K3's `<|media_begin|>image WxH
    /// <|media_content|>` and `<|media_end|>`). They consume no source token
    /// and belong to no item's span.
    Literal(Vec<i64>),
    /// Media item `item`'s token span.
    Media { item: usize, pattern: TokenPattern },
}

/// Prompt geometry as data: the family *describes* the expansion, the driver
/// *applies* it (`common::token_layout::apply_layout`) — deriving final input ids
/// and per-item offsets, and validating that every item is placed exactly
/// once.
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// The cache identity of one processed media item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemIdentity {
    /// The 64-bit item hash (`MultimodalDataItem.hash`).
    pub hash: u64,
    /// A full 32-byte identity (`MultimodalDataItem.set_identity`): the
    /// scheduler derives the item's four wide pad sentinels from it. `None`
    /// leaves the sentinels to `hash`.
    pub identity: Option<[u8; 32]>,
}

/// Text-to-ids access for a family whose expansion inserts text around media
/// (Kimi K3's framing). Specials in `text` are encoded as special tokens, as
/// Python's `tokenizer.encode(text, allowed_special="all")` does; no BOS is
/// added. The server backs it with its own tokenizer.
pub trait TextEncoder: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i64>, String>;
}

/// Modalities a family accepts; the server's message layer rejects anything
/// a family does not declare.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub video: bool,
    pub audio: bool,
}

/// Position scheme of the expanded prompt.
pub enum PositionOutput {
    /// Plain sequential positions — the scheduler needs nothing extra.
    Rope1D,
    /// M-RoPE: flattened row-major `[3, input_len]` positions + the position
    /// delta (`max + 1 - input_len`).
    MRope { positions: Vec<i64>, delta: i64 },
}

/// The per-model-family hooks of the server pipeline. Adding a family =
/// implementing this in `src/<model>/mod.rs` and adding its `family` arm to
/// [`crate::registry::pipeline_from_spec`]. All parameters come from the
/// runtime spec JSON (resolved from the HF config on the Python side);
/// nothing is hardcoded per model.
pub trait MmFamilyProcessor: Send + Sync {
    /// Modalities beyond images this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Preprocess one decoded media item: the model's HF processor
    /// equivalent (resize/tile/normalize/patchify → named tensors) plus the
    /// geometry `layout`/`positions` will need.
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;

    /// Turn one item's encoded source bytes into the processed item and its
    /// cache identity. The default decodes to RGB ([`crate::common::decode_rgb`];
    /// inputs PIL accepts but it refuses, e.g. 16-bit PNG, reject the request),
    /// runs [`Self::process_item`], and identifies the item by
    /// [`crate::common::content_hash_u64`] of the bytes. A family whose decode
    /// or identity differs (Kimi K3 keeps alpha and derives a full identity
    /// from content, config and grid) overrides this.
    fn process_encoded(&self, bytes: &[u8]) -> Result<(ProcessedItem, ItemIdentity), String> {
        let hash = crate::common::content_hash_u64(bytes);
        let (rgb, height, width) = crate::common::decode_rgb(bytes)?;
        let item = self.process_item(&DecodedMedia::Image { rgb, height, width })?;
        Ok((
            item,
            ItemIdentity {
                hash,
                identity: None,
            },
        ))
    }

    /// Describe how the prompt expands around the processed items (in
    /// prompt order). Sees the full original prompt and all items, so
    /// structured schemes (tile markers, separators) are expressible.
    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String>;

    /// Positions for the expanded prompt. Families without a custom scheme
    /// keep the default.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}
