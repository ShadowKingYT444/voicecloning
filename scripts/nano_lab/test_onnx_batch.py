"""Pure factory and manifest checks for the persistent ONNX benchmark."""

import sys
import types
import unittest
import json
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import onnx_batch as batch
import onnx_pipeline as pipeline


class _FakeRuntime:
    created = 0
    closed = 0

    def __init__(self, *args, **kwargs):
        type(self).created += 1

    def close(self):
        type(self).closed += 1


class BatchFactoryTests(unittest.TestCase):
    def test_factory_reuses_session_and_closes_once(self):
        fake = types.SimpleNamespace(
            T3UnifiedOrtRuntime=_FakeRuntime,
            FlowEncoderOrtRuntime=_FakeRuntime,
            MeanflowEstimatorOrtRuntime=_FakeRuntime,
            VocoderOrtRuntime=_FakeRuntime,
        )
        _FakeRuntime.created = 0
        _FakeRuntime.closed = 0
        with patch.dict(sys.modules, {"onnx_staged_runtime": fake}):
            factory = batch.PersistentRuntimeFactory()
            first = factory("t3", Path("/tmp/model"), ort_profile=False)
            second = factory("t3", Path("/tmp/model"), ort_profile=False)
            self.assertIs(first, second)
            self.assertTrue(getattr(first, "_nano_persistent"))
            self.assertEqual(factory.creation["t3"]["case_index"], 0)
            factory.set_case(2)
            self.assertIs(first, factory("t3", Path("/tmp/model"), ort_profile=False))
            factory.close()
        self.assertEqual(_FakeRuntime.created, 1)
        self.assertEqual(_FakeRuntime.closed, 1)

    def test_pipeline_factory_is_injectable_and_resettable(self):
        seen = []

        def factory(stage, model_dir, **kwargs):
            seen.append((stage, model_dir, kwargs))
            return object()

        args = types.SimpleNamespace(model_dir=Path("/tmp/model"), ort_provider="cpu", ort_threads=1)
        pipeline.set_runtime_factory(factory)
        try:
            runtime = pipeline._make_runtime("vocoder", args)
        finally:
            pipeline.set_runtime_factory(None)
        self.assertIsNotNone(runtime)
        self.assertEqual(seen[0][0], "vocoder")
        self.assertEqual(seen[0][1], Path("/tmp/model"))
        self.assertFalse(seen[0][2]["ort_profile"])

    def test_manifest_requires_explicit_voice_text_and_seed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('[{"voice":"asmr_soft","text":"Hello.","seed":3}]')
            cases = batch._load_manifest(path)
        self.assertEqual(cases[0].case_id, "case_000")
        self.assertEqual(cases[0].seed, 3)
        self.assertEqual(cases[0].steps, 2)

    def test_worker_finish_option_reuses_one_marker_and_reports_full_request(self):
        class Marker:
            created = 0

            def __init__(self):
                type(self).created += 1

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [batch.BatchCase("a", "voice", "Hello.", 1), batch.BatchCase("b", "voice", "World.", 2)]
            options = types.SimpleNamespace(
                ort_profile=False,
                finish_in_worker=True,
                batch_dir=root,
                model_dir=root,
                tokenizer_dir=root,
                ort_provider="cpu",
                ort_threads=1,
                cuda_device_id=0,
                gpu_mem_limit_mib=2048,
                arena_extend_strategy="kSameAsRequested",
                cudnn_conv_algo_search="HEURISTIC",
                do_copy_in_default_stream=True,
                cuda_kv_resident=False,
                manifest=root / "manifest.json",
            )
            options.manifest.write_text(json.dumps({"cases": []}))

            def fake_generation(options, case, run_dir, factory, case_index):
                return {"case_id": case.case_id, "generation_seconds": 0.01}

            def fake_finish(options, case, run_dir, output, watermarker):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"wav")
                # The integrated finish is the only place this fake worker
                # imports Torch, matching the production assertion.
                sys.modules["torch"] = types.SimpleNamespace()
                return {"audio_seconds": 1.0}, 0.02

            perth = types.SimpleNamespace(PerthImplicitWatermarker=Marker)
            report_path = root / "worker.json"
            with patch.dict(sys.modules, {"perth": perth, "torch": None}, clear=False), patch.object(
                batch, "_run_worker_case", side_effect=fake_generation
            ), patch.object(batch, "_finish_worker_case", side_effect=fake_finish):
                rc = batch._worker(options, cases, report_path)
            report = json.loads(report_path.read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(Marker.created, 1)
        self.assertTrue(report["finish_in_worker"])
        self.assertTrue(report["torch_imported"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertGreater(report["cases"][0]["full_worker_request_seconds"], 0.0)
        self.assertEqual(report["cases"][0]["watermarker_init_seconds"], report["watermarker_init_seconds"])

    def test_stage_finish_wraps_watermark_in_inference_mode(self):
        class Context:
            entered = False

            def __enter__(self):
                type(self).entered = True

            def __exit__(self, *args):
                return False

        class Torch:
            @staticmethod
            def inference_mode():
                return Context()

        class Marker:
            calls = 0

            def apply_watermark(self, audio, sample_rate):
                type(self).calls += 1
                return audio

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out.wav"
            args = types.SimpleNamespace(run_dir=root, output=output)
            fake_sf = types.SimpleNamespace(write=lambda *args, **kwargs: None)
            with patch.dict(sys.modules, {"torch": Torch, "soundfile": fake_sf}, clear=False), patch.object(
                pipeline, "_load_npz", return_value={"audio": np.zeros(240, dtype=np.float32)}
            ), patch.object(pipeline, "_master_light", side_effect=lambda value: (value, {})), patch.object(
                pipeline, "_write_json"
            ), patch.object(pipeline, "_stage_dir", return_value=root), patch.object(pipeline.os, "replace"):
                result = pipeline.stage_finish(args, watermarker=Marker())
        self.assertTrue(Context.entered)
        self.assertEqual(Marker.calls, 1)
        self.assertTrue(result["watermark_applied"])


if __name__ == "__main__":
    unittest.main()
