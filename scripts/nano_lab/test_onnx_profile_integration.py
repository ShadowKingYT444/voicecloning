"""Fail-closed fitted-profile and mel-calibration tests without model loads."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import onnx_pipeline as pipeline


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_root(root: Path, *, patched: bool, adapter: Path | None = None, scale: float = 1.0) -> Path:
    model = root / ("patched" if patched else "base")
    t3 = model / "t3"
    t3.mkdir(parents=True)
    graph = t3 / "nano_t3_kv.onnx"
    graph.write_bytes(b"patched graph" if patched else b"base graph")
    manifest = {
        "stage": "t3",
        "status": "exported",
        "graph": {"path": graph.name, "sha256": _sha(graph)},
    }
    if patched:
        assert adapter is not None
        manifest["adapter_patch"] = {
            "format": "nano_t3_lora_merged_v1",
            "path": str(adapter),
            "sha256": _sha(adapter),
            "adapter_sha256": _sha(adapter),
            "scale": scale,
            "checkpoint_scale": scale,
            "patched_graph_sha256": _sha(graph),
            "base_graph_sha256": "0" * 64,
        }
    (t3 / "manifest.json").write_text(json.dumps(manifest))
    return model


class ProfileValidationTests(unittest.TestCase):
    def test_matching_adapter_provenance_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"adapter")
            model = _model_root(root, patched=True, adapter=adapter, scale=0.75)
            value = pipeline._validate_profile_for_model(
                root / "voice.json",
                {"adapter": str(adapter), "adapter_scale": 0.75},
                model,
            )
        self.assertTrue(value["adapter"]["enabled"])
        self.assertEqual(value["adapter"]["profile_scale"], 0.75)

    def test_adapter_missing_patch_or_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"adapter")
            base = _model_root(root, patched=False)
            with self.assertRaisesRegex(ValueError, "requires a staged T3 adapter_patch"):
                pipeline._validate_profile_for_model(root / "voice.json", {"adapter": str(adapter), "adapter_scale": 1.0}, base)

            patched = _model_root(root, patched=True, adapter=adapter, scale=0.75)
            with self.assertRaisesRegex(ValueError, "does not match staged adapter scale"):
                pipeline._validate_profile_for_model(root / "voice.json", {"adapter": str(adapter), "adapter_scale": 1.0}, patched)

            with self.assertRaisesRegex(ValueError, "patched T3 graph"):
                pipeline._validate_profile_for_model(root / "voice.json", {}, patched)

    def test_mel_delta_is_applied_only_at_vocoder_and_metadata_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _model_root(root, patched=False)
            run = root / "run"
            (run / "estimator").mkdir(parents=True)
            (run / "prepare").mkdir(parents=True)
            mel = np.zeros((1, 80, 3), dtype=np.float32)
            np.savez(run / "estimator" / "mel.npz", mel=mel, noise_speech=np.zeros((1, 80, 6), np.float32), noise_full=mel)
            delta = np.linspace(-0.1, 0.1, 80, dtype=np.float32)
            delta_path = root / "delta.npy"
            np.save(delta_path, delta)
            report = root / "report.json"
            report.write_text(json.dumps({
                "format": "nano_mel_calibration_fit_v1",
                "delta_path": str(delta_path),
                "delta": delta.tolist(),
            }))
            report_sha = _sha(report)
            delta_sha = _sha(delta_path)
            (run / "prepare" / "profile.json").write_text(json.dumps({
                "mel_calibration": str(report),
                "mel_calibration_strength": 0.5,
            }))

            class Runtime:
                seen = None

                def info(self):
                    return {}

                def synthesize(self, speech_feat, *, phase_noise, sine_noise):
                    type(self).seen = np.array(speech_feat, copy=True)
                    return np.zeros((1, speech_feat.shape[2] * 480), dtype=np.float32)

                def close(self):
                    return None

            args = SimpleNamespace(
                run_dir=run,
                model_dir=model,
                seed=31,
                ort_threads=1,
                ort_provider="cpu",
                cuda_device_id=0,
                gpu_mem_limit_mib=2048,
                arena_extend_strategy="kSameAsRequested",
                cudnn_conv_algo_search="HEURISTIC",
                do_copy_in_default_stream=True,
                ort_profile=False,
                cuda_kv_resident=False,
            )
            pipeline.set_runtime_factory(lambda stage, model_dir, **kwargs: Runtime())
            try:
                result = pipeline.stage_vocoder(args)
            finally:
                pipeline.set_runtime_factory(None)
            with np.load(run / "estimator" / "mel.npz") as values:
                np.testing.assert_array_equal(values["mel"], mel)
        np.testing.assert_allclose(Runtime.seen[0, :, 0], delta * 0.5, atol=1e-7)
        self.assertTrue(result["mel_calibration"]["applied"])
        self.assertEqual(result["mel_calibration"]["strength"], 0.5)
        self.assertEqual(result["mel_calibration"]["report_sha256"], report_sha)
        self.assertEqual(result["mel_calibration"]["delta_sha256"], delta_sha)


if __name__ == "__main__":
    unittest.main()
