//! The worker pool: drain requests in `Encoding`, run the selected processor,
//! and hand each request back with the expanded ids and the named feature
//! buffers that ride the ring with it.

use std::sync::Arc;

use sglang_mm::driver::{self, Output};
use sglang_mm::pipeline::{MmFamilyProcessor, PositionOutput, Tensor, TensorData, TextEncoder};
use sglang_mm::registry::{PipelineSpec, build_pipeline_with_encoder};
use sglang_processor::TextTokenizer;

use super::encoded::{MRope, MmEncodedEntry, MmEncodedItem, MmMeta, MmMetaValue, MmModality};
use crate::message::buffers::{Buffer, BufferData, BufferStore};
use crate::message::config::MmSpec;
use crate::message::request::{MmData, Request, RequestKind};
use crate::message::types::TokenIds;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::error::Error;
use crate::utils::fsm::Event;
use crate::utils::runtime::Runnable;

/// One parsed `mm_hashes` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallerHash {
    /// A value below 2^64: the item hash, as before.
    Narrow(u64),
    /// A wider value (a full SHA-256, say): a 32-byte identity, so the wide pad
    /// sentinels that key the prefix cache keep all of its bits.
    Wide([u8; 32]),
}

/// Domain tag for folding a caller value wider than 256 bits into an identity
/// (Python `MultimodalDataItem.set_caller_hash`).
const CALLER_HASH_TAG: &[u8] = b"sglang-mm-caller-hash-v1";

/// Caller hashes override the computed ones; a length mismatch or malformed
/// entry warns and keeps the computed hash — never blocks the request. A
/// family that owns its identities (Kimi K3) never gets here with caller
/// hashes: `FamilyMmProcessor::check_media` refuses them at admission.
fn apply_caller_hashes(items: &mut [MmEncodedItem], caller: &[String]) {
    if caller.is_empty() {
        return;
    }
    if caller.len() != items.len() {
        tracing::warn!(
            caller = caller.len(),
            items = items.len(),
            "mm_hashes length != mm item count; ignoring caller hashes"
        );
        return;
    }
    for (item, entry) in items.iter_mut().zip(caller) {
        match parse_caller_hash(entry) {
            Some(CallerHash::Narrow(hash)) => {
                item.hash = hash;
                item.identity = None;
            }
            Some(CallerHash::Wide(identity)) => {
                // Python `set_identity`: the hash keeps the leading 64 bits for
                // scalar consumers; the pad sentinels use the whole identity.
                item.hash = u64::from_be_bytes(identity[..8].try_into().expect("8 bytes"));
                item.identity = Some(identity);
            }
            None => tracing::warn!(%entry, "malformed mm_hashes entry; keeping computed hash"),
        }
    }
}

/// Hex of any width, read as Python's `int(hex_hash, 16)` reads it. A value
/// below 2^64 is the item hash. A wider value used to keep only its low 64
/// bits, so two hashes sharing those bits shared prefix-cache keys; it is now
/// a 32-byte identity (the value itself up to 256 bits, a SHA-256 fold above),
/// as Python's `MultimodalDataItem.set_caller_hash` derives it.
fn parse_caller_hash(entry: &str) -> Option<CallerHash> {
    let hex = entry.strip_prefix("0x").unwrap_or(entry);
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let digits = hex.trim_start_matches('0');
    if digits.len() <= 16 {
        return u64::from_str_radix(if digits.is_empty() { "0" } else { digits }, 16)
            .ok()
            .map(CallerHash::Narrow);
    }
    // Big-endian bytes of the value without leading zero bytes (Python
    // `value.to_bytes((value.bit_length() + 7) // 8, "big")`).
    let padded = if digits.len() % 2 == 1 {
        format!("0{digits}")
    } else {
        digits.to_owned()
    };
    let bytes: Vec<u8> = (0..padded.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&padded[i..i + 2], 16))
        .collect::<Result<_, _>>()
        .ok()?;
    let mut identity = [0u8; 32];
    if bytes.len() <= identity.len() {
        identity[32 - bytes.len()..].copy_from_slice(&bytes);
    } else {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(CALLER_HASH_TAG);
        hasher.update(&bytes);
        identity = hasher.finalize().into();
    }
    Some(CallerHash::Wide(identity))
}

/// Complete result of one multimodal processor invocation.
pub struct MmProcessOutput {
    /// The final placeholder-expanded prompt ids.
    pub input_ids: TokenIds,
    pub result: MmEncodedEntry,
}

/// Client/server class of a media preparation failure, mirroring Python's
/// loader contract: `ValueError` is a 400, any other exception a 500.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MediaError {
    /// Bad input from the client (400).
    #[error("{0}")]
    Client(String),
    /// A fault on the server side, e.g. a media backend that is down (500).
    #[error("{0}")]
    Server(String),
}

/// Multimodal processor shared by built-in and external implementations.
/// Implementations run on the fixed Rust worker pool and must not retain
/// request-scoped Python objects. `input_ids` arrive tokenized (the FSM runs
/// `Tokenizing` before `Encoding` for a text prompt), so a processor never
/// tokenizes: it expands placeholders in ids. `mm` is the request's media,
/// owned.
///
/// The media hooks run earlier, during admission on the API runtime, before
/// any media is fetched or the request reaches the worker pool.
pub trait MmProcessor: Send + Sync {
    fn process(&self, input_ids: TokenIds, mm: MmData) -> Result<MmProcessOutput, String>;

    /// Reject a request's media before any of it is fetched, e.g. a modality
    /// the served model cannot encode. `Err` is a client error (400). Runs on
    /// an async worker, so it must not block.
    fn check_media(&self, _mm: &MmData) -> Result<(), String> {
        Ok(())
    }

    /// Fetch one I/O-backed source, charging its bytes to `budget`; runs on
    /// the API runtime's blocking pool. `Ok(None)` leaves the source to the
    /// built-in fetcher (`http(s)://`, `file://` and absolute paths), which
    /// rejects every other URI scheme.
    fn fetch_media(
        &self,
        _source: &str,
        _modality: MmModality,
        _budget: &crate::MediaByteBudget,
    ) -> Result<Option<bytes::Bytes>, MediaError> {
        Ok(None)
    }

    /// Finish item layout after caller hash overrides, before feature publication.
    fn finalize(&self, _entry: &mut MmEncodedEntry) -> Result<(), String> {
        Ok(())
    }
}

/// The server tokenizer as the family pipeline's [`TextEncoder`]: specials in
/// the text encode as special tokens and no BOS is added, as Python's
/// `tokenizer.encode(text, allowed_special="all")` does for Kimi K3's framing.
struct TokenizerEncoder(Arc<dyn TextTokenizer>);

impl TextEncoder for TokenizerEncoder {
    fn encode(&self, text: &str) -> Result<Vec<i64>, String> {
        let segment = sglang_processor::dynamo_tokenizers::EncodeSegment::new(text, true);
        self.0
            .encode_segments(&[segment], false)
            .map(|ids| ids.into_iter().map(i64::from).collect())
            .map_err(|e| format!("mm framing encode: {e}"))
    }
}

/// A built-in `sglang_mm` family behind the worker contract.
struct FamilyMmProcessor {
    family: Box<dyn MmFamilyProcessor>,
    /// Kimi K3: image identities come from the processor, never from callers
    /// (Python's `KimiGridMMDataMixin.reject_caller_image_identity`).
    processor_owned_identity: bool,
}

impl FamilyMmProcessor {
    fn new(spec: MmSpec, tokenizer: Option<Arc<dyn TextTokenizer>>) -> Result<Self, String> {
        let processor_owned_identity = matches!(spec.pipeline, PipelineSpec::KimiK3(_));
        let encoder = tokenizer.map(|t| Arc::new(TokenizerEncoder(t)) as Arc<dyn TextEncoder>);
        Ok(Self {
            family: build_pipeline_with_encoder(spec.pipeline, encoder)?,
            processor_owned_identity,
        })
    }
}

/// Python's message for a Kimi request carrying `mm_hashes` (`kimi_common.py`).
const KIMI_CALLER_HASHES: &str = "Caller-supplied mm_hashes are not supported for Kimi.";

impl MmProcessor for FamilyMmProcessor {
    fn process(&self, input_ids: TokenIds, mm: MmData) -> Result<MmProcessOutput, String> {
        let input = super::payload::to_mm_input(input_ids, mm)?;
        let output = driver::process(self.family.as_ref(), input)?;
        encode_driver_output(output)
    }

    /// A family that owns its identities refuses caller `mm_hashes` with a
    /// 400 before any media is fetched, as Python's Kimi processor raises
    /// `ValueError` for them.
    fn check_media(&self, mm: &MmData) -> Result<(), String> {
        if self.processor_owned_identity && !mm.mm_hashes.is_empty() {
            return Err(KIMI_CALLER_HASHES.into());
        }
        Ok(())
    }

    /// Defence in depth for the same rule: whatever reached the item hashes,
    /// an item with a processor identity keeps that identity's leading 64
    /// bits, which is what Python's `set_identity` assigns.
    fn finalize(&self, entry: &mut MmEncodedEntry) -> Result<(), String> {
        if self.processor_owned_identity {
            for item in &mut entry.items {
                let identity = item
                    .identity
                    .ok_or("mm: a processor-identity item has no identity")?;
                item.hash = u64::from_be_bytes(identity[..8].try_into().expect("8 bytes"));
            }
        }
        Ok(())
    }
}

/// Map the driver's output straight onto the common encoded result: each
/// feature tensor is moved with its own shape and dtype, the per-item span
/// and hash ride along, and every auxiliary tensor is translated once into
/// the sidecar's metadata form. Nothing is reconstructed from lengths.
fn encode_driver_output(output: Output) -> Result<MmProcessOutput, String> {
    let Output {
        input_ids,
        items,
        offsets,
        positions,
    } = output;
    if items.len() != offsets.len() {
        return Err(format!(
            "mm: {} items but {} spans",
            items.len(),
            offsets.len()
        ));
    }
    let items = items
        .into_iter()
        .zip(offsets)
        .map(|(item, span)| {
            Ok(MmEncodedItem {
                modality: MmModality::Image,
                feature: item.feature,
                hash: item.hash,
                identity: item.identity,
                offsets: vec![span],
                model_specific_data: item
                    .aux
                    .into_iter()
                    .map(|(name, tensor)| Ok((name.clone(), aux_meta(&name, tensor)?)))
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mrope = match positions {
        PositionOutput::Rope1D => None,
        PositionOutput::MRope { positions, delta } => Some(MRope { positions, delta }),
    };
    Ok(MmProcessOutput {
        input_ids,
        result: MmEncodedEntry {
            items,
            token_ids: None,
            mrope,
        },
    })
}

/// An auxiliary tensor as sidecar metadata. Only integer tensors have a
/// metadata form (a grid, a count); anything else is a processor bug, not
/// something to approximate.
fn aux_meta(name: &str, tensor: Tensor) -> Result<MmMetaValue, String> {
    match tensor.data {
        TensorData::I64(values) if values.len() == 1 && tensor.shape.is_empty() => {
            Ok(MmMetaValue::Int(values[0]))
        }
        TensorData::I64(values) => Ok(MmMetaValue::Ints(values)),
        _ => Err(format!("mm: aux tensor {name:?} has no metadata form")),
    }
}

/// Shared state of the multimodal path, built once at worker startup.
pub struct MmContext {
    pub processor: Arc<dyn MmProcessor>,
    /// Place feature tensors in POSIX shm. Set by the Python launcher
    /// (`RustMmProcessor._use_feature_shm`) exactly when the scheduler broadcasts
    /// across TP ranks and will unwrap `ShmPointerMMData`.
    pub feature_shm: bool,
}

impl MmContext {
    /// `tokenizer` is the server's own, which a family that frames media with
    /// text needs (Kimi K3); `None` only without one (`skip_tokenizer_init`).
    pub fn new(spec: MmSpec, tokenizer: Option<Arc<dyn TextTokenizer>>) -> Result<Self, String> {
        let feature_shm = spec.feature_shm;
        Ok(Self {
            processor: Arc::new(FamilyMmProcessor::new(spec, tokenizer)?),
            feature_shm,
        })
    }

    pub fn with_processor(processor: Arc<dyn MmProcessor>, feature_shm: bool) -> Self {
        Self {
            processor,
            feature_shm,
        }
    }
}

fn tensor_data(data: TensorData) -> BufferData {
    match data {
        TensorData::F32(v) => BufferData::F32(v),
        TensorData::I64(v) => BufferData::I64(v),
        TensorData::Bf16(v) => BufferData::U16(v),
        TensorData::U8(v) => BufferData::U8(v),
    }
}

/// Lay each item's feature tensor out as `mm.feature.{i}`: in its own shm
/// segment when `shm` is set -- the unit Python's `ShmPointerMMData` maps --
/// else inline. An empty tensor stays inline even under `shm`: a zero-length
/// segment cannot be mapped on either side, and there is nothing to share.
/// Any shm failure (`/dev/shm` full) falls the whole request back to inline,
/// as Python's `_wrap_shm_or_inline` does: degrade to the slow path, never
/// fail the request. `segment_name` names each item's segment.
fn place_features(
    features: Vec<Tensor>,
    shm: bool,
    mut segment_name: impl FnMut(usize) -> String,
) -> Vec<Buffer> {
    let name = |i: usize| format!("mm.feature.{i}");
    let features: Vec<(Vec<usize>, BufferData)> = features
        .into_iter()
        .map(|t| (t.shape, tensor_data(t.data)))
        .collect();
    if shm {
        let parked: Result<Vec<Buffer>, String> = features
            .iter()
            .enumerate()
            .map(|(i, (shape, data))| match data.len() {
                0 => Buffer::inline_shaped(name(i), shape.clone(), BufferData::empty(data.dtype())),
                _ => Buffer::shm(name(i), segment_name(i), shape.clone(), data),
            })
            .collect();
        match parked {
            Ok(buffers) => return buffers,
            Err(error) => {
                tracing::warn!(%error, "mm: shm feature transport failed; falling back to inline");
            }
        }
    }
    features
        .into_iter()
        .enumerate()
        .map(|(i, (shape, data))| Buffer {
            name: name(i),
            shape,
            store: BufferStore::Inline(data),
        })
        .collect()
}

/// The ring's named buffers for one result: the per-item features (see
/// [`place_features`]), the M-RoPE positions, and the `mm.meta` sidecar --
/// always last, so a reader that finds it knows the rest is present.
fn make_buffers(entry: MmEncodedEntry, feature_shm: bool) -> Result<Vec<Buffer>, String> {
    let meta = MmMeta::of(&entry).encode()?;
    let MmEncodedEntry { items, mrope, .. } = entry;
    let features = items.into_iter().map(|item| item.feature).collect();
    let mut buffers = place_features(features, feature_shm, |_| {
        crate::utils::shm::unique_name("mm")
    });
    if let Some(mrope) = mrope {
        let len = mrope.positions.len() / 3;
        buffers.push(Buffer::inline_shaped(
            "mm.mrope",
            vec![3, len],
            mrope.positions,
        )?);
    }
    buffers.push(Buffer::inline("mm.meta", meta));
    Ok(buffers)
}

/// Run the processor for one request. `Ok` returns the final expanded ids and
/// the buffers to ride the ring; `Err` rejects the request back to the client.
pub(crate) fn process(
    ctx: &MmContext,
    input_ids: TokenIds,
    mut mm: MmData,
) -> Result<(TokenIds, Vec<Buffer>), String> {
    let caller_hashes = std::mem::take(&mut mm.mm_hashes);
    let mut output = ctx.processor.process(input_ids, mm)?;
    output.result.validate(output.input_ids.len())?;
    apply_caller_hashes(&mut output.result.items, &caller_hashes);
    ctx.processor.finalize(&mut output.result)?;
    output.result.validate(output.input_ids.len())?;
    let buffers = make_buffers(output.result, ctx.feature_shm)?;
    Ok((output.input_ids, buffers))
}

/// Boot-time wiring of the MM path, held privately by the `Runtime` for the
/// late pool spawn (`Runtime::start_mm_workers`, once Python has resolved
/// the spec).
pub struct MmWiring {
    /// Requests in `Encoding`, drained by the worker pool. Stays empty for
    /// non-multimodal models -- nothing routes to it.
    pub mm_rx: flume::Receiver<Request>,
    /// Back-channel for the workers' `Encoded` into the to-scheduler loop.
    pub tm_tx: flume::Sender<TmEvent>,
}

/// One MM worker, spawned via `Runtime::start_mm_workers` (which owns the
/// pinning policy for this pool — see its docs).
pub struct MmWorker {
    mm_rx: flume::Receiver<Request>,
    tm_tx: flume::Sender<TmEvent>,
    ctx: Arc<MmContext>,
}

impl MmWorker {
    pub fn new(
        mm_rx: flume::Receiver<Request>,
        tm_tx: flume::Sender<TmEvent>,
        ctx: Arc<MmContext>,
    ) -> Self {
        Self { mm_rx, tm_tx, ctx }
    }
}

impl Runnable for MmWorker {
    /// Drain until the mm channel closes (to-scheduler drops its sender on
    /// shutdown). One request at a time, so the pool size bounds MM
    /// concurrency. Mirrors `TokenizerWorker`: carve the work out of the
    /// request, process, write the result back, advance the FSM
    /// (`EncodeDone` -> PreSendValidating, or `Error` -> Failed, which intake
    /// rejects to the client as Python turns a per-request exception into a
    /// 400), and return the request as `Encoded`.
    fn run(self) {
        while let Ok(mut req) = self.mm_rx.recv() {
            // A request is never dropped here: a kind this pool cannot serve
            // goes back as `Failed`, so intake rejects it and releases its
            // tracking entry instead of leaving the client hung.
            let event = if let RequestKind::Generate(g) = &mut req.kind {
                // Move the processor's inputs out: the unexpanded ids (always
                // present by `Encoding`; the expanded ids replace them) and the
                // media, so the processor owns the bytes without a copy. `text`
                // stays for the scheduler header; nothing reads `mm` after this.
                let input_ids = g.input_ids.take().unwrap_or_default();
                let mm = g.mm.take().map(|m| *m).unwrap_or_default();
                match process(&self.ctx, input_ids, mm) {
                    Ok((input_ids, buffers)) => {
                        tracing::debug!(rid = %req.rid, tokens = input_ids.len(), "mm: processed");
                        g.input_ids = Some(input_ids);
                        g.mm_buffers = buffers;
                        Event::EncodeDone
                    }
                    Err(message) => {
                        tracing::warn!(rid = %req.rid, %message, "mm processing rejected");
                        Event::Error(Error::Encode(message))
                    }
                }
            } else {
                tracing::error!(rid = %req.rid, "mm pool received a non-generate request");
                Event::Error(Error::Internal(
                    "non-generate request in the mm pool".into(),
                ))
            };
            let _ = req.state.apply(event);
            if self.tm_tx.send(TmEvent::Encoded(req)).is_err() {
                return; // to-scheduler gone: shutdown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use sglang_mm::driver::OutputItem;
    use std::collections::BTreeMap;

    use super::*;
    use crate::message::buffers::find;
    use crate::utils::shm::{shm_path, unique_name};

    /// An external-style processor: one image item with a caller-shaped
    /// feature and spans, no M-RoPE.
    struct ExternalProcessor {
        shape: Vec<usize>,
        offsets: Vec<(u32, u32)>,
    }

    impl MmProcessor for ExternalProcessor {
        fn process(&self, input_ids: TokenIds, _mm: MmData) -> Result<MmProcessOutput, String> {
            Ok(MmProcessOutput {
                input_ids,
                result: MmEncodedEntry {
                    items: vec![MmEncodedItem {
                        modality: MmModality::Image,
                        feature: Tensor {
                            shape: self.shape.clone(),
                            data: TensorData::F32(vec![1.0]),
                        },
                        hash: 7,
                        identity: None,
                        offsets: self.offsets.clone(),
                        model_specific_data: BTreeMap::from([(
                            "clip_index".to_owned(),
                            MmMetaValue::Int(3),
                        )]),
                    }],
                    token_ids: None,
                    mrope: None,
                },
            })
        }
    }

    /// The driver's output maps onto the encoded result without any
    /// reconstruction: feature tensors keep their shape and dtype, spans and
    /// hashes ride per item, integer aux tensors become metadata, and M-RoPE
    /// carries over; 1-D positions leave `mrope` empty.
    #[test]
    fn driver_output_maps_directly() {
        let item = |shape: Vec<usize>, hash| OutputItem {
            feature: Tensor {
                shape,
                data: TensorData::F32(vec![0.5; 12]),
            },
            aux: vec![
                (
                    "image_grid_thw".into(),
                    Tensor {
                        shape: vec![3],
                        data: TensorData::I64(vec![1, 2, 3]),
                    },
                ),
                (
                    "clip_index".into(),
                    Tensor {
                        shape: vec![],
                        data: TensorData::I64(vec![7]),
                    },
                ),
            ],
            hash,
            identity: None,
        };
        let output = Output {
            input_ids: vec![1, 2, 2, 3],
            items: vec![item(vec![6, 2], 11), item(vec![2, 6], 22)],
            offsets: vec![(1, 2), (3, 3)],
            positions: PositionOutput::MRope {
                positions: vec![0; 12],
                delta: -2,
            },
        };
        let encoded = encode_driver_output(output).unwrap();
        assert_eq!(encoded.input_ids, [1, 2, 2, 3]);
        let items = &encoded.result.items;
        assert_eq!(items[0].feature.shape, [6, 2], "shape kept, not rebuilt");
        assert_eq!(items[1].feature.shape, [2, 6]);
        assert_eq!((items[0].hash, items[1].hash), (11, 22));
        assert_eq!(items[0].offsets, [(1, 2)]);
        assert_eq!(items[1].offsets, [(3, 3)]);
        assert_eq!(
            items[0].model_specific_data["image_grid_thw"],
            MmMetaValue::Ints(vec![1, 2, 3])
        );
        assert_eq!(
            items[0].model_specific_data["clip_index"],
            MmMetaValue::Int(7)
        );
        let mrope = encoded.result.mrope.as_ref().unwrap();
        assert_eq!((mrope.positions.len(), mrope.delta), (12, -2));

        let plain = Output {
            input_ids: vec![1],
            items: vec![],
            offsets: vec![],
            positions: PositionOutput::Rope1D,
        };
        assert!(encode_driver_output(plain).unwrap().result.mrope.is_none());

        let bad = Output {
            input_ids: vec![1],
            items: vec![OutputItem {
                feature: Tensor {
                    shape: vec![1],
                    data: TensorData::F32(vec![1.0]),
                },
                aux: vec![(
                    "scale".into(),
                    Tensor {
                        shape: vec![1],
                        data: TensorData::F32(vec![1.0]),
                    },
                )],
                hash: 1,
                identity: None,
            }],
            offsets: vec![(0, 0)],
            positions: PositionOutput::Rope1D,
        };
        let Err(err) = encode_driver_output(bad) else {
            panic!("an f32 aux tensor must be rejected");
        };
        assert!(err.contains("no metadata form"), "{err}");
    }

    /// The built-in Qwen shape: two items with grids, spans and M-RoPE.
    fn qwen_entry() -> MmEncodedEntry {
        let item = |rows: usize, feature: Vec<f32>, grid: [i64; 3], hash, span| MmEncodedItem {
            modality: MmModality::Image,
            feature: Tensor {
                shape: vec![rows, 2],
                data: TensorData::F32(feature),
            },
            hash,
            identity: None,
            offsets: vec![span],
            model_specific_data: BTreeMap::from([(
                "image_grid_thw".to_owned(),
                MmMetaValue::Ints(grid.to_vec()),
            )]),
        };
        MmEncodedEntry {
            items: vec![
                item(2, vec![1.0, 2.0, 3.0, 4.0], [1, 2, 1], 11, (1, 2)),
                item(1, vec![5.0, 6.0], [1, 1, 1], 22, (3, 3)),
            ],
            token_ids: None,
            mrope: Some(MRope {
                positions: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
                delta: -1,
            }),
        }
    }

    fn decoded_meta(buffers: &[Buffer]) -> rmpv::Value {
        let BufferStore::Inline(BufferData::U8(bytes)) = &find(buffers, "mm.meta").unwrap().store
        else {
            panic!("mm.meta must be an inline byte buffer")
        };
        rmpv::decode::read_value(&mut bytes.as_slice()).unwrap()
    }

    /// Inline: one shaped `mm.feature.{i}` per item owning its own tensor, the
    /// M-RoPE positions as `[3, len]`, and the sidecar last, decoding to named
    /// maps with the item metadata in order.
    #[test]
    fn buffers_are_shaped_features_mrope_and_meta_sidecar() {
        let buffers = make_buffers(qwen_entry(), false).unwrap();
        let names: Vec<&str> = buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            ["mm.feature.0", "mm.feature.1", "mm.mrope", "mm.meta"]
        );
        let feature1 = find(&buffers, "mm.feature.1").unwrap();
        assert_eq!(feature1.shape, [1, 2]);
        assert!(
            matches!(&feature1.store, BufferStore::Inline(BufferData::F32(v)) if v == &[5.0, 6.0])
        );
        assert_eq!(find(&buffers, "mm.mrope").unwrap().shape, [3, 3]);

        let meta = decoded_meta(&buffers);
        let get = |m: &rmpv::Value, key: &str| {
            m.as_map()
                .unwrap()
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        let items = get(&meta, "items");
        let items = items.as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(get(&items[1], "hash").as_u64(), Some(22));
        assert_eq!(get(&items[1], "modality").as_str(), Some("image"));
        assert_eq!(
            get(&items[1], "offsets"),
            rmpv::Value::Array(vec![rmpv::Value::Array(vec![3.into(), 3.into()])])
        );
        assert_eq!(
            get(&get(&items[0], "model_specific_data"), "image_grid_thw"),
            rmpv::Value::Array(vec![1.into(), 2.into(), 1.into()])
        );
        assert_eq!(get(&meta, "mrope_delta").as_i64(), Some(-1));
        assert!(get(&meta, "token_ids").is_nil());
    }

    /// Shm: each item's tensor lands in its own segment holding exactly its
    /// bytes, shaped as the tensor; the segment lives as long as the buffer.
    #[test]
    fn shm_places_each_item_in_its_own_segment() {
        let names: Vec<String> = (0..2).map(|_| unique_name("test")).collect();
        let namer = names.clone();
        let features: Vec<Tensor> = qwen_entry().items.into_iter().map(|i| i.feature).collect();
        let expected: Vec<Vec<f32>> = features
            .iter()
            .map(|t| match &t.data {
                TensorData::F32(v) => v.clone(),
                _ => unreachable!(),
            })
            .collect();
        let buffers = place_features(features, true, move |i| namer[i].clone());
        for (i, name) in names.iter().enumerate() {
            assert!(matches!(buffers[i].store, BufferStore::Shm { .. }));
            assert_eq!(buffers[i].shape, [[2, 2], [1, 2]][i]);
            let bytes = std::fs::read(shm_path(name)).unwrap();
            assert_eq!(bytes, bytemuck::cast_slice::<f32, u8>(&expected[i]));
        }
        drop(buffers);
        assert!(names.iter().all(|n| !shm_path(n).exists()), "drop unlinks");
    }

    /// An empty tensor has no segment to make: it stays inline while its
    /// siblings still go to shm, with no request-wide fallback.
    #[test]
    fn shm_keeps_empty_items_inline() {
        let names: Vec<String> = (0..2).map(|_| unique_name("test")).collect();
        let namer = names.clone();
        let features = vec![
            Tensor {
                shape: vec![0, 2],
                data: TensorData::F32(vec![]),
            },
            Tensor {
                shape: vec![1, 2],
                data: TensorData::F32(vec![5.0, 6.0]),
            },
        ];
        let buffers = place_features(features, true, move |i| namer[i].clone());
        assert!(
            matches!(&buffers[0].store, BufferStore::Inline(BufferData::F32(v)) if v.is_empty())
        );
        assert_eq!(buffers[0].shape, [0, 2]);
        assert!(
            !shm_path(&names[0]).exists(),
            "no segment for the empty item"
        );
        assert!(matches!(buffers[1].store, BufferStore::Shm { .. }));
        assert!(
            shm_path(&names[1]).exists(),
            "the sibling still goes to shm"
        );
    }

    /// A full `/dev/shm` (page reservation refused with `ENOSPC`) reaches the
    /// same inline fallback as any other segment failure: the request keeps
    /// every feature value, and the segments made before the failure are gone.
    #[test]
    fn shm_exhaustion_falls_back_to_inline_with_values_intact() {
        use crate::utils::shm::{ShmSegment, shm_path};
        use std::os::fd::BorrowedFd;

        // Item 0's segment is created for real; item 1's reservation fails as
        // an exhausted tmpfs would. `place_features` sees one Err and must
        // fall the whole request back, dropping item 0's segment with it.
        fn no_space(_: BorrowedFd<'_>, _: u64) -> rustix::io::Result<()> {
            Err(rustix::io::Errno::NOSPC)
        }
        let names: Vec<String> = (0..2).map(|_| unique_name("test")).collect();
        let first = names[0].clone();
        let expected: Vec<Vec<f32>> = qwen_entry()
            .items
            .iter()
            .map(|i| match &i.feature.data {
                TensorData::F32(v) => v.clone(),
                _ => unreachable!(),
            })
            .collect();
        let bytes0 = bytemuck::cast_slice::<f32, u8>(&expected[0]).to_vec();
        let parked: Result<Vec<Buffer>, String> = (0..2)
            .map(|i| {
                let segment = if i == 0 {
                    ShmSegment::create(first.clone(), &bytes0)
                } else {
                    ShmSegment::create_with(names[1].clone(), &[1], no_space)
                }?;
                Ok(Buffer {
                    name: format!("mm.feature.{i}"),
                    shape: vec![1],
                    store: BufferStore::Shm {
                        segment,
                        dtype: crate::message::buffers::DType::F32,
                    },
                })
            })
            .collect();
        let err = parked.unwrap_err();
        assert!(err.contains("fallocate("), "{err}");
        assert!(err.contains("No space left"), "{err}");
        assert!(!shm_path(&first).exists(), "item 0's segment released");

        // And through `place_features` itself, with a namer whose second
        // segment cannot be reserved: values survive, inline.
        let features: Vec<Tensor> = qwen_entry().items.into_iter().map(|i| i.feature).collect();
        let buffers = place_features(features, true, |_| "bad\0name".into());
        for (i, buffer) in buffers.iter().enumerate() {
            let BufferStore::Inline(BufferData::F32(v)) = &buffer.store else {
                panic!("expected inline f32, got {:?}", buffer.store);
            };
            assert_eq!(v, &expected[i]);
        }
    }

    /// A segment that cannot be created degrades the whole request to inline
    /// rather than rejecting it (Python's `_wrap_shm_or_inline` parity).
    #[test]
    fn shm_failure_falls_back_to_inline() {
        let features = qwen_entry().items.into_iter().map(|i| i.feature).collect();
        let buffers = place_features(features, true, |_| "bad\0name".into());
        assert_eq!(buffers.len(), 2);
        assert!(
            buffers
                .iter()
                .all(|b| matches!(b.store, BufferStore::Inline(_)))
        );
    }

    /// An external processor's result rides the same buffers: its tensor's
    /// dtype and shape are kept, the caller hash override applies, and the
    /// sidecar carries its scalar metadata.
    #[test]
    fn external_processor_result_becomes_buffers() {
        let ctx = MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 1)],
            }),
            false,
        );
        let mm = MmData {
            mm_hashes: vec!["2a".to_owned()],
            ..Default::default()
        };
        let (input_ids, buffers) = process(&ctx, vec![1, 2], mm).unwrap();
        assert_eq!(input_ids, [1, 2]);
        assert_eq!(find(&buffers, "mm.feature.0").unwrap().shape, [1]);
        assert!(find(&buffers, "mm.mrope").is_none());
        let meta = decoded_meta(&buffers);
        let text = format!("{meta}");
        assert!(text.contains("42"), "caller hash 0x2a applied: {text}");
        assert!(text.contains("clip_index"), "{text}");
    }

    #[test]
    fn finalization_runs_after_caller_hashes_and_is_validated() {
        struct FinalizingProcessor {
            final_end: u32,
        }

        impl MmProcessor for FinalizingProcessor {
            fn process(&self, ids: TokenIds, mm: MmData) -> Result<MmProcessOutput, String> {
                ExternalProcessor {
                    shape: vec![1],
                    offsets: vec![(0, 0)],
                }
                .process(ids, mm)
            }

            fn finalize(&self, entry: &mut MmEncodedEntry) -> Result<(), String> {
                let parent_hash = entry.items[0].hash;
                assert_eq!(parent_hash, 42);
                entry.items[0].hash = parent_hash + 1;
                entry.items.push(MmEncodedItem {
                    modality: MmModality::Image,
                    feature: Tensor {
                        shape: vec![1],
                        data: TensorData::F32(vec![2.0]),
                    },
                    hash: parent_hash + 2,
                    identity: None,
                    offsets: vec![(1, self.final_end)],
                    model_specific_data: BTreeMap::new(),
                });
                Ok(())
            }
        }

        for final_end in [1, 2] {
            let ctx = MmContext::with_processor(Arc::new(FinalizingProcessor { final_end }), false);
            let result = process(
                &ctx,
                vec![1, 2],
                MmData {
                    mm_hashes: vec!["2a".to_owned()],
                    ..Default::default()
                },
            );
            if final_end == 2 {
                assert!(result.is_err(), "finalized spans must be validated");
                continue;
            }
            let (ids, buffers) = result.expect("valid finalized items");
            assert_eq!(ids, [1, 2]);
            assert!(find(&buffers, "mm.feature.1").is_some());
            let meta: serde_json::Value = rmpv::ext::from_value(decoded_meta(&buffers)).unwrap();
            assert_eq!(meta["items"].as_array().unwrap().len(), 2);
            assert_eq!(meta["items"][0]["hash"].as_u64(), Some(43));
            assert_eq!(meta["items"][1]["hash"].as_u64(), Some(44));
        }
    }

    /// Malformed results are rejected before any buffer exists, so nothing
    /// reaches the ring for them.
    #[test]
    fn malformed_processor_results_are_rejected() {
        for (shape, offsets) in [
            (vec![2], vec![(1, 1)]),
            (vec![usize::MAX, 2], vec![(1, 1)]),
            (vec![1], vec![(2, 1)]),
            (vec![1], vec![(1, 2)]),
        ] {
            let ctx =
                MmContext::with_processor(Arc::new(ExternalProcessor { shape, offsets }), false);
            assert!(process(&ctx, vec![1, 2], MmData::default()).is_err());
        }
    }

    /// A request in `Encoding` for the worker loop, with the given ids.
    fn encoding_req(rid: &str, input_ids: Vec<i64>) -> Request {
        use crate::message::request::GenerateRequest;
        use crate::message::response::ResponseSink;
        use crate::utils::fsm::RequestState;
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        Request {
            rid: rid.to_string().into(),
            state: RequestState::Encoding,
            sink: ResponseSink::Local(tx),
            kind: RequestKind::Generate(Box::new(GenerateRequest {
                rid: rid.to_string().into(),
                input_ids: Some(input_ids),
                ..Default::default()
            })),
        }
    }

    /// The worker loop takes the whole request, like the tokenizer pool: on
    /// success it writes the expanded ids and buffers back and advances to
    /// `PreSendValidating`; on a processor error it marks the request
    /// `Failed(Encode)`. Either way the request comes back as `Encoded`.
    #[test]
    fn worker_returns_request_with_fsm_advanced() {
        use crate::utils::fsm::RequestState;
        let (mm_tx, mm_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let ok_ctx = Arc::new(MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 1)],
            }),
            false,
        ));
        let worker = std::thread::spawn({
            let mm_rx = mm_rx.clone();
            move || MmWorker::new(mm_rx, tm_tx, ok_ctx).run()
        });
        mm_tx.send(encoding_req("ok", vec![1, 2])).unwrap();
        drop(mm_tx); // closes the pool edge -> the loop exits after draining
        worker.join().unwrap();

        let TmEvent::Encoded(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Encoded");
        };
        assert_eq!(req.rid.as_str(), "ok");
        assert!(
            matches!(req.state, RequestState::PreSendValidating),
            "{:?}",
            req.state
        );
        let RequestKind::Generate(g) = &req.kind else {
            panic!("generate")
        };
        assert_eq!(g.input_ids.as_deref(), Some(&[1, 2][..]));
        assert!(find(&g.mm_buffers, "mm.feature.0").is_some());
        assert!(find(&g.mm_buffers, "mm.meta").is_some());

        // A malformed result (span past the prompt) fails the request in place.
        let (mm_tx, mm_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let bad_ctx = Arc::new(MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 2)],
            }),
            false,
        ));
        let worker = std::thread::spawn(move || MmWorker::new(mm_rx, tm_tx, bad_ctx).run());
        mm_tx.send(encoding_req("bad", vec![1, 2])).unwrap();
        drop(mm_tx);
        worker.join().unwrap();
        let TmEvent::Encoded(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Encoded");
        };
        assert!(
            matches!(req.state, RequestState::Failed(Error::Encode(_))),
            "{:?}",
            req.state
        );
    }

    /// A kind the pool cannot serve is returned `Failed`, never dropped:
    /// intake still holds its sink registration and pool-tracking entry.
    #[test]
    fn non_generate_request_is_returned_failed_not_dropped() {
        use crate::message::response::ResponseSink;
        use crate::utils::fsm::RequestState;
        let (mm_tx, mm_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let ctx = Arc::new(MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 1)],
            }),
            false,
        ));
        let (sink_tx, _sink_rx) = tokio::sync::mpsc::channel(4);
        mm_tx
            .send(Request {
                rid: "detok".to_string().into(),
                state: RequestState::Encoding,
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Detokenize {
                    token_ids: vec![1, 2],
                    skip_special_tokens: true,
                },
            })
            .unwrap();
        drop(mm_tx);
        MmWorker::new(mm_rx, tm_tx, ctx).run();

        let TmEvent::Encoded(req) = tm_rx.try_recv().expect("returned, not dropped") else {
            panic!("expected Encoded");
        };
        assert_eq!(req.rid.as_str(), "detok");
        assert!(
            matches!(req.state, RequestState::Failed(Error::Internal(_))),
            "{:?}",
            req.state
        );
    }

    fn hashed_items(hashes: &[u64]) -> Vec<MmEncodedItem> {
        hashes
            .iter()
            .map(|&hash| MmEncodedItem {
                modality: MmModality::Image,
                feature: Tensor {
                    shape: vec![1],
                    data: TensorData::F32(vec![1.0]),
                },
                hash,
                identity: None,
                offsets: vec![(0, 0)],
                model_specific_data: BTreeMap::new(),
            })
            .collect()
    }

    fn hashes_of(items: &[MmEncodedItem]) -> Vec<u64> {
        items.iter().map(|item| item.hash).collect()
    }

    /// The first sidecar item's hash and identity (msgpack `bin`, omitted
    /// when the item has none).
    fn first_item_identity(buffers: &[Buffer]) -> (Option<u64>, Option<Vec<u8>>) {
        let meta = decoded_meta(buffers);
        let field = |map: &rmpv::Value, key: &str| {
            map.as_map()
                .unwrap()
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone())
        };
        let items = field(&meta, "items").unwrap();
        let item = &items.as_array().unwrap()[0];
        (
            field(item, "hash").and_then(|hash| hash.as_u64()),
            field(item, "identity").map(|identity| identity.as_slice().unwrap().to_vec()),
        )
    }

    /// Caller hashes override computed ones; mismatched lengths and malformed
    /// entries fall back per item, never reject (Python parity).
    #[test]
    fn caller_hashes_override_with_fallback() {
        let mut items = hashed_items(&[1, 2, 3]);
        apply_caller_hashes(&mut items, &[]);
        assert_eq!(hashes_of(&items), [1, 2, 3]);

        apply_caller_hashes(&mut items, &["ff".into()]); // length mismatch
        assert_eq!(hashes_of(&items), [1, 2, 3]);

        apply_caller_hashes(&mut items, &["ff".into(), "not-hex".into(), "0x10".into()]);
        assert_eq!(hashes_of(&items), [0xff, 2, 0x10]);
        assert!(items.iter().all(|item| item.identity.is_none()));
    }

    /// A caller value below 2^64 stays the item hash, leading zeros or not. A
    /// wider one (a full SHA-256, what routers send) becomes a 32-byte
    /// identity instead of its low 64 bits, so two hashes that share those
    /// bits no longer share prefix-cache keys.
    #[test]
    fn wide_caller_hashes_keep_every_bit() {
        assert_eq!(parse_caller_hash("0x2a"), Some(CallerHash::Narrow(42)));
        assert_eq!(
            parse_caller_hash(&format!("{}2a", "0".repeat(62))),
            Some(CallerHash::Narrow(42))
        );
        assert_eq!(
            parse_caller_hash(&"f".repeat(16)),
            Some(CallerHash::Narrow(u64::MAX))
        );
        assert_eq!(parse_caller_hash("0x"), None);
        assert_eq!(parse_caller_hash(""), None);
        assert_eq!(parse_caller_hash("sha256:ab"), None);

        let sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let mut same_low_bits = "0".repeat(48);
        same_low_bits.push_str(&sha256[48..]);
        same_low_bits.replace_range(0..1, "1");
        let mut items = hashed_items(&[1, 2]);
        apply_caller_hashes(&mut items, &[sha256.into(), same_low_bits.clone()]);
        let Some(CallerHash::Wide(expected)) = parse_caller_hash(sha256) else {
            panic!("a 256-bit value is wide");
        };
        let mut want = [0u8; 32];
        for (i, byte) in want.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&sha256[2 * i..2 * i + 2], 16).unwrap();
        }
        assert_eq!(expected, want, "up to 256 bits, the identity is the value");
        assert_eq!(items[0].identity, Some(want));
        assert_eq!(items[0].hash, 0xe3b0c44298fc1c14, "the leading 64 bits");
        assert_ne!(
            items[0].identity, items[1].identity,
            "the same low 64 bits no longer collide"
        );

        // Wider than 256 bits (a SHA-512): folded with SHA-256, still distinct.
        // The fold matches Python's `set_caller_hash` for 2**512 - 1.
        let sha512 = "f".repeat(128);
        let mut other = sha512.clone();
        other.replace_range(0..1, "e");
        let (Some(CallerHash::Wide(a)), Some(CallerHash::Wide(b))) =
            (parse_caller_hash(&sha512), parse_caller_hash(&other))
        else {
            panic!("a 512-bit value is wide");
        };
        assert_ne!(a, b);
        let hex: String = a.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            hex,
            "430b77ed09f1b8422b83a4e390dee6efedf6c1308740b746cf8ae7cd4363e50e"
        );
    }

    /// A wide caller identity reaches Python through the sidecar beside the
    /// hash it implies; a narrow one sends none.
    #[test]
    fn wide_caller_identity_rides_the_meta_sidecar() {
        let ctx = MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 1)],
            }),
            false,
        );
        let sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let sha256_bytes: Vec<u8> = (0..32)
            .map(|i| u8::from_str_radix(&sha256[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        for (entry, identity, hash) in [
            (sha256, Some(sha256_bytes), 0xe3b0c44298fc1c14_u64),
            ("2a", None, 42),
        ] {
            let mm = MmData {
                mm_hashes: vec![entry.to_owned()],
                ..Default::default()
            };
            let (_, buffers) = process(&ctx, vec![1, 2], mm).unwrap();
            assert_eq!(first_item_identity(&buffers), (Some(hash), identity));
        }
    }
}
