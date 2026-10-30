"""Kimi K3: the Rust MM pipeline against the Python processor, at the
scheduler-input boundary.

The Rust side is the server's own path: `_server.mm_process_for_parity` builds
the Kimi K3 family with the server tokenizer (loaded from the checkpoint as the
server loads it), runs the admission check and the MM worker's `process`, and
`RustMmProcessor.wrap_encoded` turns the buffers into the scheduler's items.
The Python side is `KimiK3ImageProcessor.process_mm_data_async` with its
production defaults (preprocess cache on, CPU feature transport), run as on a
CUDA node -- `is_cuda()` true, so the deferral decision is the GPU one -- with
the CUDA tensors kept on the CPU.

Compared exactly: the expanded prompt ids, each item's offsets, grid,
32-byte identity, hash and wide pad sentinels, the deferred-preprocessing
config, and the pixels. For an image Python defers, the scheduler tensor is
the source image as `[C, H, W]` uint8 on both sides, byte for byte. Both
sides' encoder input is then materialized the way the model does it
(`_gpu_preprocess_images`, here on the CPU): equal bytes, since it is the same
code on the same pixels. Python preprocesses an image eagerly when its raw
pixels outweigh the float32 patches; the Rust path still hands over the raw
image, and the encoder inputs still match bit for bit.

JPEG is the one input that is not exact: the Rust decoder (zune-jpeg) and PIL
(libjpeg-turbo) are different IDCT and chroma-upsampling implementations, and
on a CUDA node Python itself decodes with nvJPEG instead. That case pins a
stated bound instead.

Needs a Kimi K3 checkpoint directory with its processor and tokenizer files
(no weights are read): set SGLANG_TEST_KIMI_K3_PATH.
"""

import asyncio
import base64
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sglang.srt.multimodal.kimi_k3_image_processing import (  # noqa: E402
    DEFERRED_PREPROCESSING_KEY,
    fill_transparent_bg,
    normalization_tensors,
    to_chw_uint8,
)
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmProcessor,
    settle_shm_buffers,
)

register_cpu_ci(
    est_time=120,
    suite="base-a-test-cpu",
    disabled="needs a Kimi K3 checkpoint's processor files",
)

K3_PATH = os.environ.get("SGLANG_TEST_KIMI_K3_PATH")

# A multi-turn K3 conversation; each `<|media_pad|>` is one image, as the K3
# chat encoding renders it.
MULTI_TURN = (
    "<|im_system|>system<|im_middle|>You are a careful visual assistant.<|im_end|>"
    "<|im_user|>user<|im_middle|>What is in this picture?<|media_pad|><|im_end|>"
    "<|im_assistant|>assistant<|im_middle|>A gradient with some noise.<|im_end|>"
    "<|im_user|>user<|im_middle|>Compare it with this one: <|media_pad|> "
    "and say which is brighter.<|im_end|><|im_assistant|>assistant<|im_middle|>"
)
ONE_IMAGE = "<|im_user|>user<|im_middle|>Describe <|media_pad|> briefly.<|im_end|>"


def natural_image(width, height, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        (
            x * 255 / max(width - 1, 1),
            y * 255 / max(height - 1, 1),
            (x // 7 + y // 5) % 2 * 255,
        ),
        axis=-1,
    )
    rgb = np.clip(base + rng.integers(0, 24, base.shape), 0, 255).astype(np.uint8)
    if mode == "RGBA":
        alpha = ((x * 3 + y * 5) % 256).astype(np.uint8)[..., None]
        return Image.fromarray(np.concatenate([rgb, alpha], axis=-1), mode="RGBA")
    return Image.fromarray(rgb)


def encoded(image, fmt="PNG", **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def data_url(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def encoder_input(item):
    """The tensor the vision tower receives for one item: a deferred item is
    finished as `KimiK3ForConditionalGeneration` finishes it (backend "gpu",
    here on the CPU); an eager one already is."""
    from sglang.srt.multimodal.processors.kimi_k25 import _gpu_preprocess_images

    feature = item.feature
    if hasattr(feature, "materialize"):
        feature = feature.materialize()
    config = item.model_specific_data.get(DEFERRED_PREPROCESSING_KEY)
    if config is None:
        return feature
    scale, bias = normalization_tensors(config.image_mean, config.image_std, "cpu")
    pixels, grids = _gpu_preprocess_images(
        [feature],
        [config.resize_config],
        scale,
        bias,
        14,
        to_chw=lambda image: to_chw_uint8(image, device="cpu"),
        post_resize=lambda x: fill_transparent_bg(x, config.transparent_bg_config),
    )
    assert grids.tolist() == item.model_specific_data["image_grid_thw"].tolist()
    return pixels


@unittest.skipUnless(K3_PATH, "set SGLANG_TEST_KIMI_K3_PATH to a Kimi K3 checkpoint")
class TestKimiK3E2eParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoProcessor

        from sglang.srt.managers.multimodal_processor import import_processors
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        import_processors("sglang.srt.multimodal.processors")
        cls.hf_config = AutoConfig.from_pretrained(K3_PATH, trust_remote_code=True)
        cls.hf_processor = AutoProcessor.from_pretrained(
            K3_PATH, trust_remote_code=True
        )
        cls.tokenizer = cls.hf_processor.tokenizer
        # Production defaults on HGX B300: CPU feature transport (what an
        # unset --mm-feature-transport resolves to there), the preprocess cache
        # on (auto size), image processor backend auto, no --mm-process-config.
        publish(
            ServerArgs(
                model_path="dummy",
                model_impl="sglang",
                tp_size=1,
                mm_feature_transport="cpu",
                mm_process_config={},
            ),
            role="tokenizer",
        )
        cls.server_args = SimpleNamespace(model_impl="sglang", tp_size=1)

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.runtime_context import reset_context

        reset_context()

    def setUp(self):
        from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor

        self.processor = KimiK3ImageProcessor(
            self.hf_config, self.server_args, self.hf_processor, None, skip_mm_pool=True
        )
        self.spec = self.native_spec()

    def tearDown(self):
        self.processor.shutdown()
        self.processor.cpu_executor.shutdown()

    # --- the two paths under comparison ---

    def native_spec(self):
        """Through the production gate, so a gate that stops serving K3 (or
        reads a different fingerprint) fails here."""
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(hf_config=self.hf_config)
        host._processor = self.hf_processor
        host.server_args = self.server_args
        spec = host.resolve_spec()
        self.assertIsNotNone(spec, "the Rust MM gate does not serve kimi_k3")
        return spec

    def encode(self, text):
        return list(self.tokenizer.encode(text, allowed_special="all"))

    def run_native(self, ids, sources, spec=None, mm_hashes=()):
        from sglang.srt.rust_extensions import _server
        from sglang.srt.rust_server.server import RustServer

        spec = spec or self.spec
        expanded, buffers = _server.mm_process_for_parity(
            RustServer._build_mm_spec(spec), K3_PATH, ids, sources, list(mm_hashes)
        )
        buffers = dict(buffers)
        try:
            output = RustMmProcessor.wrap_encoded(spec, buffers)
        except BaseException:
            settle_shm_buffers(buffers, admitted=False)
            raise
        settle_shm_buffers(buffers, admitted=True)
        return expanded, output

    def run_python(self, ids, sources):
        """`process_mm_data_async` as on a CUDA node with CPU transport: the
        GPU deferral rule and the GPU preprocessing code, on CPU tensors."""
        from sglang.srt.multimodal.processors import kimi_k3, kimi_k25

        get_norm = kimi_k25.KimiGPUProcessorWrapper._get_gpu_norm_tensors
        with (
            mock.patch.object(kimi_k3, "is_cuda", return_value=True),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                kimi_k3,
                "_k3_to_cuda_chw",
                side_effect=lambda image: to_chw_uint8(image),
            ),
            mock.patch.object(
                kimi_k25.KimiGPUProcessorWrapper,
                "_get_gpu_norm_tensors",
                lambda self, device="cuda": get_norm(self, device="cpu"),
            ),
        ):
            output = asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=list(sources),
                    input_text=list(ids),
                    request_obj=SimpleNamespace(
                        video_data=None,
                        audio_data=None,
                        mm_hashes=None,
                        mm_content_hashes=None,
                        rid="parity",
                    ),
                )
            )
        return output.input_ids, output

    def assert_parity(self, text, sources, *, exact_pixels=True, spec=None):
        ids = self.encode(text)
        rust_ids, rust = self.run_native(ids, sources, spec=spec)
        python_ids, python = self.run_python(ids, sources)
        self.assertEqual(list(rust_ids), list(python_ids), "expanded prompt ids")
        self.assertEqual(rust.im_token_id, python.im_token_id)
        self.assertEqual(len(rust.mm_items), len(sources))
        self.assertEqual(len(python.mm_items), len(sources))
        report = []
        for index, (r, p) in enumerate(zip(rust.mm_items, python.mm_items)):
            with self.subTest(item=index):
                self.assertEqual(r.offsets, p.offsets, "offsets")
                self.assertEqual(
                    r.model_specific_data["image_grid_thw"].tolist(),
                    p.model_specific_data["image_grid_thw"].tolist(),
                    "image_grid_thw",
                )
                self.assertIsNotNone(p.identity, "Python sets a full identity")
                self.assertEqual(r.identity, p.identity, "identity")
                self.assertEqual(r.hash, p.hash, "hash")
                self.assertEqual(r.pad_values, p.pad_values, "pad sentinels")
                self.assertEqual(r.pad_value, p.pad_value)
                r_cfg = r.model_specific_data[DEFERRED_PREPROCESSING_KEY]
                p_cfg = p.model_specific_data.get(DEFERRED_PREPROCESSING_KEY)
                r_feature = r.feature
                if hasattr(r_feature, "materialize"):
                    r_feature = r_feature.materialize()
                    r = SimpleNamespace(
                        feature=r_feature, model_specific_data=r.model_specific_data
                    )
                self.assertEqual(r_feature.dtype, torch.uint8)
                if p_cfg is not None:
                    self.assertEqual(r_cfg, p_cfg, "deferred preprocessing")
                    self.assertEqual(r_feature.dtype, p.feature.dtype)
                    self.assertEqual(tuple(r_feature.shape), tuple(p.feature.shape))
                r_pixels, p_pixels = encoder_input(r), encoder_input(p)
                self.assertEqual(r_pixels.dtype, p_pixels.dtype)
                self.assertEqual(tuple(r_pixels.shape), tuple(p_pixels.shape))
                if exact_pixels:
                    if p_cfg is not None:
                        self.assertTrue(torch.equal(r_feature, p.feature), "raw pixels")
                    self.assertEqual(
                        r_pixels.numpy().tobytes(),
                        p_pixels.numpy().tobytes(),
                        "encoder input",
                    )
                report.append(
                    {
                        "python_path": "deferred" if p_cfg is not None else "eager",
                        "grid": p.model_specific_data["image_grid_thw"].tolist()[0],
                        "tokens": p.offsets[0][1] - p.offsets[0][0] + 1,
                        "raw_shape": list(r_feature.shape),
                        "encoder_shape": list(p_pixels.shape),
                        "max_abs_encoder_diff": float(
                            (r_pixels - p_pixels).abs().max()
                        ),
                        "identity": p.identity.hex()[:16],
                    }
                )
        print(f"\n{self._testMethodName}: {len(python_ids)} ids, items {report}")
        return report

    # --- cases ---

    def test_small_png(self):
        image = encoded(natural_image(64, 48))
        report = self.assert_parity(ONE_IMAGE, [data_url(image)])
        self.assertEqual(report[0]["python_path"], "deferred")

    def test_large_png_needing_resize(self):
        # 5000x4000: 101745 patches > in_patch_limit 65536, so NaViT scales it.
        from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config

        image = encoded(natural_image(5000, 4000, seed=1))
        report = self.assert_parity(ONE_IMAGE, [data_url(image)])
        resize = navit_resize_config(5000, 4000, 14, 2, 65536, 512)
        self.assertLess(resize["new_width"], 5000)
        self.assertEqual(
            report[0]["grid"],
            [
                1,
                (resize["new_height"] + resize["pad_height"]) // 14,
                (resize["new_width"] + resize["pad_width"]) // 14,
            ],
        )

    def test_non_square_pngs(self):
        for width, height in ((900, 300), (200, 700)):
            with self.subTest(size=(width, height)):
                image = encoded(natural_image(width, height, seed=width))
                self.assert_parity(ONE_IMAGE, [data_url(image)])

    def test_two_images_in_one_message(self):
        first = encoded(natural_image(320, 240, seed=2))
        second = encoded(natural_image(128, 512, seed=3))
        text = "<|im_user|>user<|im_middle|><|media_pad|><|media_pad|>Same scene?<|im_end|>"
        self.assert_parity(text, [data_url(first), data_url(second)])

    def test_multi_turn_chat_with_images(self):
        first = encoded(natural_image(480, 360, seed=4))
        second = encoded(natural_image(333, 517, seed=5))
        self.assert_parity(MULTI_TURN, [data_url(first), data_url(second)])

    def test_rgba_png_keeps_alpha_for_the_gpu_composite(self):
        image = encoded(natural_image(300, 200, seed=6, mode="RGBA"))
        rust_ids, rust = self.run_native(self.encode(ONE_IMAGE), [data_url(image)])
        self.assertEqual(tuple(rust.mm_items[0].feature.shape), (4, 200, 300))
        self.assert_parity(ONE_IMAGE, [data_url(image)])

    def test_python_eager_image_matches_at_the_encoder(self):
        # 30000x600: the patches (7168x168 padded) weigh less than the raw
        # pixels, so Python preprocesses eagerly; Rust still defers.
        image = encoded(natural_image(30000, 600, seed=7))
        report = self.assert_parity(ONE_IMAGE, [data_url(image)])
        self.assertEqual(report[0]["python_path"], "eager")

    def test_shm_transport_carries_uint8_pixels(self):
        import msgspec

        image = encoded(natural_image(96, 80, seed=8))
        spec = msgspec.structs.replace(self.spec, feature_shm=True)
        self.assert_parity(ONE_IMAGE, [data_url(image)], spec=spec)

    def test_jpeg_within_decoder_tolerance(self):
        # Different JPEG decoders (zune-jpeg vs libjpeg-turbo) round the IDCT
        # and the 4:2:0 chroma upsampling differently: ids, grid and identity
        # stay exact (they never look at pixels); pixels may differ by a few
        # 8-bit levels.
        image = encoded(natural_image(640, 480, seed=9), "JPEG", quality=90)
        ids = self.encode(ONE_IMAGE)
        sources = [data_url(image, "image/jpeg")]
        self.assert_parity(ONE_IMAGE, sources, exact_pixels=False)
        _, rust = self.run_native(ids, sources)
        _, python = self.run_python(ids, sources)
        diff = (
            rust.mm_items[0].feature.to(torch.int16)
            - python.mm_items[0].feature.to(torch.int16)
        ).abs()
        print(
            f"jpeg raw-pixel diff: max {int(diff.max())} levels, "
            f"{float((diff == 0).float().mean()):.4f} exact"
        )
        self.assertLessEqual(int(diff.max()), JPEG_MAX_LEVELS)

    def test_caller_hashes_are_refused_as_python_refuses_them(self):
        image = data_url(encoded(natural_image(32, 32)))
        with self.assertRaisesRegex(ValueError, "mm_hashes are not supported for Kimi"):
            self.run_native(self.encode(ONE_IMAGE), [image], mm_hashes=["ab" * 32])

    def test_placeholder_count_must_match(self):
        image = data_url(encoded(natural_image(32, 32)))
        with self.assertRaisesRegex(ValueError, "image placeholder"):
            self.run_native(self.encode(ONE_IMAGE), [image, image])


# The largest raw-pixel difference the JPEG case tolerates (see its comment).
JPEG_MAX_LEVELS = 8


if __name__ == "__main__":
    unittest.main()
