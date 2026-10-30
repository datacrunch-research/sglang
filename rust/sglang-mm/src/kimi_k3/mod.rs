//! Kimi K3 family server-pipeline image processor.
//!
//! Reproduces what the Python `KimiK3ImageProcessor`
//! (`python/sglang/srt/multimodal/processors/kimi_k3.py`) hands the scheduler
//! for an image on a CUDA node with CPU feature transport (the default on
//! HGX B300): the *deferred* item, whose feature is the decoded source image
//! as `[C, H, W]` uint8 (alpha kept, `to_chw_uint8`) and whose resize,
//! transparent-background fill, normalization and patchify run on the GPU
//! inside the model (`KimiK3DeferredPreprocessing`, backend `"gpu"`, the same
//! `_gpu_preprocess_images` the eager path calls). So the encoder input is
//! computed by the same code from the same bytes on both servers.
//!
//! What this module computes on the CPU, each mirroring one Python function:
//! * [`navit_resize_config`] — `kimi_k25.navit_resize_config`, and the grid
//!   from `_grid_thw_from_resize_config`;
//! * [`decode_chw`] — PIL decode plus `to_chw_uint8` (which images keep alpha);
//! * [`kimi_image_identity`] over [`artifact_key`] — the full identity
//!   `compose_request` sets (`kimi_common.kimi_image_identity` of
//!   `MediaArtifactCacheMixin._artifact_key`), whose four wide pad sentinels
//!   key the radix cache;
//! * the layout — `_expand_k3_image_prompt_token_ids`: each `<|media_pad|>`
//!   becomes `<|media_begin|>image {W}x{H}<|media_content|>`, the pad repeated
//!   `num_tokens` times, then `<|media_end|>`, the framing encoded by the
//!   server's tokenizer with specials allowed.
//!
//! Parameters come from the runtime spec, resolved on the Python side from the
//! checkpoint's `media_proc_cfg` and the Python processor's fingerprint.

use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::pipeline::{
    DecodedMedia, Geometry, ItemIdentity, MmFamilyProcessor, ProcessedItem, Segment, Tensor,
    TensorData, TextEncoder, TokenLayout, TokenPattern,
};

/// `CONTENT_HASH_PREFIX` (`multimodal/cache/identity.py`).
pub const CONTENT_HASH_PREFIX: &str = "sha256:";
/// `_KIMI_IMAGE_IDENTITY_TAG` (`kimi_common.py`).
const KIMI_IMAGE_IDENTITY_TAG: &[u8] = b"sglang-kimi-image-identity-v1";
/// The aux tensor carrying `navit_resize_config` as
/// `[num_tokens, new_width, new_height, pad_width, pad_height]`; the drain
/// adapter rebuilds the deferred-preprocessing `resize_config` dict from it.
pub const RESIZE_CONFIG_KEY: &str = "kimi_k3_resize_config";

/// Resolved processor params, deserialized from the Python-side spec
/// (`KimiK3PreprocessConfig` plus the processor's token id and fingerprint).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct KimiK3Spec {
    /// `<|media_pad|>` (`hf_config.media_placeholder_token_id`).
    pub image_token_id: i64,
    pub patch_size: u32,
    pub merge_kernel_size: u32,
    pub in_patch_limit: u32,
    pub patch_limit_on_one_side: u32,
    pub fixed_output_tokens: Option<u32>,
    /// `build_processor_fingerprint` of the Python processor this server
    /// replaces (`sha256:<64 hex>`): part of every image identity, so the
    /// identities equal a Python server's with the same settings.
    pub processor_fingerprint: String,
}

/// `navit_resize_config`'s result for one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizeConfig {
    pub num_tokens: usize,
    pub new_width: u32,
    pub new_height: u32,
    pub pad_width: u32,
    pub pad_height: u32,
}

impl ResizeConfig {
    /// `_grid_thw_from_resize_config`: `[1, padded_h / patch, padded_w / patch]`.
    pub fn grid_thw(&self, patch_size: u32) -> [u32; 3] {
        [
            1,
            (self.new_height + self.pad_height) / patch_size,
            (self.new_width + self.pad_width) / patch_size,
        ]
    }
}

/// `kimi_k25.navit_resize_config`, in the same f64 arithmetic (Python floats
/// are IEEE doubles, `int()` truncates as `as u64` does for positive values).
pub fn navit_resize_config(width: u32, height: u32, spec: &KimiK3Spec) -> ResizeConfig {
    let (w, h) = (f64::from(width), f64::from(height));
    let patch = spec.patch_size;
    let patches_w = f64::from(width / patch).max(1.0);
    let patches_h = f64::from(height / patch).max(1.0);
    let s1 = (f64::from(spec.in_patch_limit) / (patches_w * patches_h)).sqrt();
    let side = u64::from(spec.patch_limit_on_one_side) * u64::from(patch);
    let s2 = side as f64 / w;
    let s3 = side as f64 / h;
    let scale = 1.0_f64.min(s1).min(s2).min(s3);
    let new_w = ((w * scale) as u64).max(1).min(side) as u32;
    let new_h = ((h * scale) as u64).max(1).min(side) as u32;
    let factor = spec.merge_kernel_size * patch;
    let pad_height = (factor - new_h % factor) % factor;
    let pad_width = (factor - new_w % factor) % factor;
    let num_tokens = match spec.fixed_output_tokens {
        Some(fixed) => fixed as usize,
        None => ((new_h + pad_height) / factor) as usize * ((new_w + pad_width) / factor) as usize,
    };
    ResizeConfig {
        num_tokens,
        new_width: new_w,
        new_height: new_h,
        pad_width,
        pad_height,
    }
}

/// A decoded source image as `to_chw_uint8` lays it out.
pub struct ChwImage {
    /// `[channels, height, width]` row-major.
    pub data: Vec<u8>,
    /// 3 (RGB) or 4 (RGBA).
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

/// Decode as PIL then `to_chw_uint8` does: alpha is kept exactly when PIL
/// reports it (`mode != "RGB"` and an `A` band or a `transparency` entry), so
/// RGBA, LA, and palette or grayscale PNGs with `tRNS` decode to four
/// channels and everything else to three (grayscale replicated). Samples
/// deeper than 8 bits are refused, as [`crate::common::decode_rgb`] refuses
/// them. EXIF orientation is not applied, as the K3 Python path does not
/// apply it.
pub fn decode_chw(bytes: &[u8]) -> Result<ChwImage, String> {
    use image::ColorType;

    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode: {e}"))?;
    if !matches!(
        img.color(),
        ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
    ) {
        return Err(format!("image decode: unsupported color {:?}", img.color()));
    }
    let (width, height) = (img.width() as usize, img.height() as usize);
    let alpha = img.color().has_alpha() && pil_keeps_alpha(bytes);
    let (hwc, channels) = if alpha {
        (img.into_rgba8().into_raw(), 4)
    } else {
        (img.into_rgb8().into_raw(), 3)
    };
    let plane = height * width;
    let mut data = vec![0u8; channels * plane];
    for (pixel, values) in hwc.chunks_exact(channels).enumerate() {
        for (c, &value) in values.iter().enumerate() {
            data[c * plane + pixel] = value;
        }
    }
    Ok(ChwImage {
        data,
        channels,
        height,
        width,
    })
}

/// Whether PIL would keep the alpha the decoder produced. Two containers
/// differ from the decoded color type: a truecolor PNG with `tRNS` stays PIL
/// mode `RGB` (alpha dropped), and a GIF's first frame has alpha only when its
/// graphic control extension names a transparent index.
fn pil_keeps_alpha(bytes: &[u8]) -> bool {
    const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() > 25 && bytes.starts_with(PNG_SIGNATURE) && &bytes[12..16] == b"IHDR" {
        // IHDR color type 2: truecolor.
        return bytes[25] != 2;
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return gif_first_frame_transparent(bytes).unwrap_or(false);
    }
    true
}

/// The transparent-color flag of the graphic control extension before a GIF's
/// first image descriptor; `None` on a truncated stream.
fn gif_first_frame_transparent(bytes: &[u8]) -> Option<bool> {
    let packed = *bytes.get(10)?;
    let mut i = 13;
    if packed & 0x80 != 0 {
        i += 3 * (1usize << ((packed & 0x07) + 1));
    }
    loop {
        match *bytes.get(i)? {
            // Image descriptor: no transparency declared before the first frame.
            0x2c => return Some(false),
            0x21 => {
                let label = *bytes.get(i + 1)?;
                if label == 0xf9 {
                    // Block size 4, then the packed field; bit 0 = transparent index.
                    return Some(*bytes.get(i + 3)? & 0x01 != 0);
                }
                // Skip the extension's sub-blocks.
                i += 2;
                loop {
                    let len = usize::from(*bytes.get(i)?);
                    i += 1 + len;
                    if len == 0 {
                        break;
                    }
                }
            }
            _ => return None,
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 0x0f)] as char);
    }
    out
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    let digit = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let text = text.as_bytes();
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.chunks_exact(2)
        .map(|pair| Some((digit(pair[0])? << 4) | digit(pair[1])?))
        .collect()
}

/// `parse_content_hash`: `sha256:` and 64 hex digits, lowercased.
fn parse_content_hash(value: &str) -> Result<String, String> {
    let digest = value
        .strip_prefix(CONTENT_HASH_PREFIX)
        .filter(|digest| digest.len() == 64 && unhex(digest).is_some())
        .ok_or_else(|| format!("content hash {value:?} is not 'sha256:<64 hex digits>'"))?;
    Ok(format!(
        "{CONTENT_HASH_PREFIX}{}",
        digest.to_ascii_lowercase()
    ))
}

/// `snapshot_media(bytes).content_digest`: SHA-256 of the exact source bytes.
pub fn content_digest(bytes: &[u8]) -> String {
    format!("{CONTENT_HASH_PREFIX}{}", hex(&Sha256::digest(bytes)))
}

/// `build_artifact_key(content_digest, modality="image", processor_fingerprint,
/// preprocess_kwargs={})`: SHA-256 of the canonical JSON `_canonical_json`
/// writes for that payload (keys sorted, compact separators, ASCII). The
/// request carries no per-image preprocess options (`detail` and the like),
/// so `preprocess_kwargs` is always the empty mapping, as for a Python request
/// with the defaults.
pub fn artifact_key(content_digest: &str, processor_fingerprint: &str) -> Result<String, String> {
    let content_digest = parse_content_hash(content_digest)?;
    let string = |value: &str| format!(r#"{{"type":"str","value":"{value}"}}"#);
    let payload = format!(
        r#"{{"items":[[{},{}],[{},{}],[{},{{"items":[],"type":"mapping"}}],[{},{}]],"type":"mapping"}}"#,
        string("content_digest"),
        string(&content_digest),
        string("modality"),
        string("image"),
        string("preprocess_kwargs"),
        string("processor_fingerprint"),
        json_string_body(processor_fingerprint),
    );
    Ok(content_digest_of(payload.as_bytes()))
}

fn content_digest_of(payload: &[u8]) -> String {
    format!("{CONTENT_HASH_PREFIX}{}", hex(&Sha256::digest(payload)))
}

/// `{"type":"str","value":<value>}` with `value` escaped as `json.dumps`
/// escapes it (`ensure_ascii`): the fingerprint is `sha256:<hex>` in practice,
/// but escape anyway so a malformed one cannot change the payload's shape.
fn json_string_body(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    escaped.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => escaped.push(c),
        }
    }
    format!(r#"{{"type":"str","value":"{escaped}"}}"#)
}

/// `kimi_common.kimi_image_identity`: SHA-256 over the tag, the 32 digest
/// bytes of `content_config_digest`, then each grid dimension as an 8-byte
/// big-endian signed integer.
pub fn kimi_image_identity(
    content_config_digest: &str,
    grid_thw: [u32; 3],
) -> Result<[u8; 32], String> {
    let digest = parse_content_hash(content_config_digest)?;
    let bytes = unhex(&digest[CONTENT_HASH_PREFIX.len()..]).expect("validated hex");
    let mut hasher = Sha256::new();
    hasher.update(KIMI_IMAGE_IDENTITY_TAG);
    hasher.update(&bytes);
    for dim in grid_thw {
        hasher.update(i64::from(dim).to_be_bytes());
    }
    Ok(hasher.finalize().into())
}

pub struct KimiK3Processor {
    spec: KimiK3Spec,
    encoder: Arc<dyn TextEncoder>,
    /// `<|media_end|>`, encoded once.
    media_end: Vec<i64>,
}

impl KimiK3Processor {
    pub fn new(spec: KimiK3Spec, encoder: Arc<dyn TextEncoder>) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_kernel_size == 0 || spec.patch_limit_on_one_side == 0
        {
            return Err("kimi_k3 spec: sizes must be positive".into());
        }
        parse_content_hash(&spec.processor_fingerprint)
            .map_err(|e| format!("kimi_k3 spec: processor_fingerprint: {e}"))?;
        let media_end = encoder.encode("<|media_end|>")?;
        Ok(Self {
            spec,
            encoder,
            media_end,
        })
    }

    pub fn spec(&self) -> &KimiK3Spec {
        &self.spec
    }

    /// The item for one decoded image: its `[C, H, W]` pixels, the grid, the
    /// resize config the GPU will apply, and the framing geometry.
    fn item(&self, image: ChwImage) -> Result<(ProcessedItem, [u32; 3]), String> {
        let (width, height) = (
            u32::try_from(image.width).map_err(|_| "kimi_k3: image too wide")?,
            u32::try_from(image.height).map_err(|_| "kimi_k3: image too tall")?,
        );
        if width == 0 || height == 0 {
            return Err("kimi_k3: empty image".into());
        }
        let resize = navit_resize_config(width, height, &self.spec);
        let grid = resize.grid_thw(self.spec.patch_size);
        let item = ProcessedItem {
            feature: Tensor {
                shape: vec![image.channels, image.height, image.width],
                data: TensorData::U8(image.data),
            },
            aux: vec![
                (
                    "image_grid_thw".to_string(),
                    Tensor {
                        shape: vec![3],
                        data: TensorData::I64(grid.iter().map(|&d| i64::from(d)).collect()),
                    },
                ),
                (
                    RESIZE_CONFIG_KEY.to_string(),
                    Tensor {
                        shape: vec![5],
                        data: TensorData::I64(vec![
                            resize.num_tokens as i64,
                            i64::from(resize.new_width),
                            i64::from(resize.new_height),
                            i64::from(resize.pad_width),
                            i64::from(resize.pad_height),
                        ]),
                    },
                ),
            ],
            geometry: Geometry::SizedGrid {
                grid,
                width,
                height,
                tokens: resize.num_tokens,
            },
        };
        Ok((item, grid))
    }
}

impl MmFamilyProcessor for KimiK3Processor {
    /// An already-decoded RGB image (no alpha, no source bytes, so no
    /// identity); the server path goes through [`Self::process_encoded`].
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let plane = height * width;
        if rgb.len() != 3 * plane {
            return Err("kimi_k3: RGB buffer does not match its size".into());
        }
        let mut data = vec![0u8; 3 * plane];
        for (pixel, values) in rgb.chunks_exact(3).enumerate() {
            for (c, &value) in values.iter().enumerate() {
                data[c * plane + pixel] = value;
            }
        }
        let image = ChwImage {
            data,
            channels: 3,
            height: *height,
            width: *width,
        };
        self.item(image).map(|(item, _)| item)
    }

    fn process_encoded(&self, bytes: &[u8]) -> Result<(ProcessedItem, ItemIdentity), String> {
        let digest = content_digest(bytes);
        let (item, grid) = self.item(decode_chw(bytes)?)?;
        let key = artifact_key(&digest, &self.spec.processor_fingerprint)?;
        let identity = kimi_image_identity(&key, grid)?;
        let hash = u64::from_be_bytes(identity[..8].try_into().expect("8 bytes"));
        Ok((
            item,
            ItemIdentity {
                hash,
                identity: Some(identity),
            },
        ))
    }

    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String> {
        let pad = self.spec.image_token_id;
        let found = input_ids.iter().filter(|&&id| id == pad).count();
        if found != items.len() {
            return Err(format!(
                "Expected {} image placeholder token(s), found {found}.",
                items.len()
            ));
        }
        let mut segments = Vec::with_capacity(4 * items.len() + 1);
        let mut text_start = 0;
        let mut next = items.iter().enumerate();
        for (pos, &id) in input_ids.iter().enumerate() {
            if id != pad {
                continue;
            }
            let (index, geometry) = next.next().expect("counted above");
            let Geometry::SizedGrid {
                width,
                height,
                tokens,
                ..
            } = geometry
            else {
                return Err(format!("kimi_k3: unexpected item geometry {geometry:?}"));
            };
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            segments.push(Segment::Literal(self.encoder.encode(&format!(
                "<|media_begin|>image {width}x{height}<|media_content|>"
            ))?));
            segments.push(Segment::Media {
                item: index,
                pattern: TokenPattern::Repeat {
                    id: pad,
                    n: *tokens,
                },
            });
            segments.push(Segment::Literal(self.media_end.clone()));
            text_start = pos + 1;
        }
        if text_start < input_ids.len() {
            segments.push(Segment::Text(text_start..input_ids.len()));
        }
        Ok(TokenLayout { segments })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::token_layout::apply_layout;

    fn spec() -> KimiK3Spec {
        // Kimi K3's released media_proc_cfg.
        KimiK3Spec {
            image_token_id: 7,
            patch_size: 14,
            merge_kernel_size: 2,
            in_patch_limit: 65_536,
            patch_limit_on_one_side: 512,
            fixed_output_tokens: None,
            processor_fingerprint: format!("sha256:{}", "ab".repeat(32)),
        }
    }

    /// A stand-in tokenizer: one id per framing string, fixed by its size.
    struct FakeEncoder;
    impl TextEncoder for FakeEncoder {
        fn encode(&self, text: &str) -> Result<Vec<i64>, String> {
            Ok(match text {
                "<|media_end|>" => vec![102],
                t if t.starts_with("<|media_begin|>image ") => {
                    let size =
                        &t["<|media_begin|>image ".len()..t.len() - "<|media_content|>".len()];
                    let (w, h) = size.split_once('x').unwrap();
                    vec![100, w.parse().unwrap(), h.parse().unwrap(), 101]
                }
                other => return Err(format!("unexpected text {other:?}")),
            })
        }
    }

    fn processor() -> KimiK3Processor {
        KimiK3Processor::new(spec(), Arc::new(FakeEncoder)).unwrap()
    }

    fn png(img: image::DynamicImage) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Values from `kimi_k25.navit_resize_config` with the released config
    /// (the same table the Global Router's planner pins).
    #[test]
    fn resize_config_matches_python() {
        let spec = spec();
        let r = navit_resize_config(640, 480, &spec);
        assert_eq!(
            r,
            ResizeConfig {
                num_tokens: 414,
                new_width: 640,
                new_height: 480,
                pad_width: 4,
                pad_height: 24
            }
        );
        assert_eq!(r.grid_thw(14), [1, 36, 46]);
        let r = navit_resize_config(4000, 3000, &spec);
        assert_eq!((r.num_tokens, r.grid_thw(14)), (15444, [1, 216, 286]));
        let r = navit_resize_config(20000, 100, &spec);
        assert_eq!((r.num_tokens, r.grid_thw(14)), (512, [1, 4, 512]));
        assert_eq!((r.new_width, r.new_height), (7168, 35));
        let r = navit_resize_config(10000, 10000, &spec);
        assert_eq!((r.num_tokens, r.grid_thw(14)), (16641, [1, 258, 258]));
        // Smaller than one patch: still one merged cell.
        let r = navit_resize_config(3, 5, &spec);
        assert_eq!((r.num_tokens, r.grid_thw(14)), (1, [1, 2, 2]));
        let fixed = KimiK3Spec {
            fixed_output_tokens: Some(64),
            ..spec
        };
        assert_eq!(navit_resize_config(640, 480, &fixed).num_tokens, 64);
    }

    /// Reference digests from Python (`hashlib`, SGLang's identity helpers).
    #[test]
    fn identity_matches_python_helpers() {
        let digest = content_digest(b"hello image bytes");
        assert_eq!(
            digest,
            "sha256:aa110ddcc4a3b385a0ec95f69e509337226c4042b8ec0cdffba9ac6bb75d7c17"
        );
        // kimi_image_identity(DIGEST, [1, 24, 36]) (the Global Router's pin).
        let identity = kimi_image_identity(&digest, [1, 24, 36]).unwrap();
        assert_eq!(
            hex(&identity),
            "f16e823fd0f6b212f5d0582350d201d0e554a409e8928fe52ca59903243159b2"
        );
        assert!(kimi_image_identity("sha256:zz", [1, 1, 1]).is_err());
        assert!(artifact_key("md5:00", &spec().processor_fingerprint).is_err());
        // Upper-case digests are normalized, as parse_content_hash does.
        assert_eq!(
            artifact_key(&digest.to_uppercase().replace("SHA256", "sha256"), "f").unwrap(),
            artifact_key(&digest, "f").unwrap()
        );
    }

    /// RGB stays three channels, RGBA keeps alpha, grayscale replicates, and a
    /// truecolor PNG's tRNS is dropped as PIL drops it; the layout is CHW.
    #[test]
    fn decode_keeps_alpha_as_pil_does() {
        let rgb = image::RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8, y as u8, 9]));
        let out = decode_chw(&png(rgb.into())).unwrap();
        assert_eq!((out.channels, out.height, out.width), (3, 2, 3));
        // Plane 0 is R = x, plane 1 is G = y, plane 2 is B = 9.
        assert_eq!(out.data[..6], [0, 1, 2, 0, 1, 2]);
        assert_eq!(out.data[6..12], [0, 0, 0, 1, 1, 1]);
        assert!(out.data[12..].iter().all(|&v| v == 9));

        let rgba = image::RgbaImage::from_fn(2, 2, |x, _| image::Rgba([1, 2, 3, x as u8 * 200]));
        let out = decode_chw(&png(rgba.into())).unwrap();
        assert_eq!(out.channels, 4);
        assert_eq!(out.data[12..], [0, 200, 0, 200]);

        let gray = image::GrayImage::from_fn(2, 1, |x, _| image::Luma([x as u8 * 50]));
        let out = decode_chw(&png(gray.into())).unwrap();
        assert_eq!((out.channels, out.data), (3, vec![0, 50, 0, 50, 0, 50]));

        // Truecolor + tRNS: patch a tRNS chunk in before IDAT.
        let mut bytes = png(image::RgbImage::from_pixel(2, 2, image::Rgb([5, 6, 7])).into());
        let idat = bytes.windows(4).position(|w| w == b"IDAT").unwrap() - 4;
        let mut trns = vec![0, 0, 0, 6];
        trns.extend_from_slice(b"tRNS");
        trns.extend_from_slice(&[0, 5, 0, 6, 0, 7]);
        let mut crc = crc32(&trns[4..]).to_be_bytes().to_vec();
        trns.append(&mut crc);
        bytes.splice(idat..idat, trns);
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert!(decoded.color().has_alpha(), "the decoder honours tRNS");
        assert_eq!(decode_chw(&bytes).unwrap().channels, 3, "PIL keeps RGB");
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn gif_transparency_follows_the_first_graphic_control_extension() {
        // Header, 1x1 screen with a 2-color global table, GCE, image.
        let mut gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00".to_vec();
        gif.extend_from_slice(&[0, 0, 0, 255, 255, 255]);
        let with = |flags: u8| {
            let mut g = gif.clone();
            g.extend_from_slice(&[0x21, 0xf9, 4, flags, 0, 0, 0, 0]);
            g.extend_from_slice(&[0x2c, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 0x44, 0x01, 0, 0x3b]);
            g
        };
        assert_eq!(gif_first_frame_transparent(&with(0x01)), Some(true));
        assert_eq!(gif_first_frame_transparent(&with(0x00)), Some(false));
        assert!(!pil_keeps_alpha(&with(0x00)));
        assert_eq!(decode_chw(&with(0x00)).unwrap().channels, 3);
        assert_eq!(decode_chw(&with(0x01)).unwrap().channels, 4);
    }

    /// Each pad becomes begin framing, the pad run, and the end framing; the
    /// item spans cover the pads only, as `get_mm_items_offset` finds them.
    #[test]
    fn layout_frames_each_image() {
        let proc = processor();
        let items = [
            Geometry::SizedGrid {
                grid: [1, 2, 2],
                width: 28,
                height: 28,
                tokens: 1,
            },
            Geometry::SizedGrid {
                grid: [1, 2, 4],
                width: 56,
                height: 20,
                tokens: 2,
            },
        ];
        let ids = [1, 7, 2, 7, 3];
        let layout = proc.layout(&ids, &items).unwrap();
        let expanded = apply_layout(&ids, &layout, 2).unwrap();
        assert_eq!(
            expanded.input_ids,
            [
                1, 100, 28, 28, 101, 7, 102, 2, 100, 56, 20, 101, 7, 7, 102, 3
            ]
        );
        assert_eq!(expanded.offsets, [(5, 5), (12, 13)]);
        let err = proc.layout(&[1, 7], &items).err().unwrap();
        assert!(err.contains("Expected 2 image placeholder"), "{err}");
    }

    /// The encoded path: CHW u8 feature, grid and resize config as aux, and a
    /// full identity whose leading 64 bits are the hash; the identity changes
    /// with the bytes and the fingerprint.
    #[test]
    fn process_encoded_identifies_and_keeps_pixels() {
        let proc = processor();
        let bytes =
            png(image::RgbImage::from_fn(30, 20, |x, y| image::Rgb([x as u8, y as u8, 1])).into());
        let (item, identity) = proc.process_encoded(&bytes).unwrap();
        assert_eq!(item.feature.shape, [3, 20, 30]);
        assert!(matches!(item.feature.data, TensorData::U8(ref v) if v.len() == 1800));
        let full = identity.identity.unwrap();
        assert_eq!(
            identity.hash,
            u64::from_be_bytes(full[..8].try_into().unwrap())
        );
        let key = artifact_key(&content_digest(&bytes), &spec().processor_fingerprint).unwrap();
        assert_eq!(full, kimi_image_identity(&key, [1, 2, 4]).unwrap());
        assert_eq!(item.aux[0].0, "image_grid_thw");
        assert!(matches!(&item.aux[1].1.data, TensorData::I64(v) if v == &[2, 30, 20, 26, 8]));

        let other = KimiK3Processor::new(
            KimiK3Spec {
                processor_fingerprint: format!("sha256:{}", "cd".repeat(32)),
                ..spec()
            },
            Arc::new(FakeEncoder),
        )
        .unwrap();
        assert_ne!(
            other.process_encoded(&bytes).unwrap().1.identity,
            Some(full)
        );
        assert!(
            KimiK3Processor::new(
                KimiK3Spec {
                    processor_fingerprint: "nope".into(),
                    ..spec()
                },
                Arc::new(FakeEncoder)
            )
            .is_err()
        );
    }

    /// Through the shared driver: the request's ids are framed and expanded,
    /// every item carries its identity, and no M-RoPE is produced.
    #[test]
    fn driver_runs_the_kimi_family() {
        let proc = processor();
        let bytes = png(image::RgbImage::from_pixel(28, 28, image::Rgb([1, 2, 3])).into());
        let out = crate::driver::process(
            &proc,
            crate::driver::MmInput {
                input_ids: vec![5, 7, 6],
                images: vec![crate::driver::ImageSource::Bytes(bytes)],
            },
        )
        .unwrap();
        assert_eq!(out.input_ids, [5, 100, 28, 28, 101, 7, 102, 6]);
        assert_eq!(out.offsets, [(5, 5)]);
        assert!(out.items[0].identity.is_some());
        assert!(matches!(
            out.positions,
            crate::pipeline::PositionOutput::Rope1D
        ));
    }
}
