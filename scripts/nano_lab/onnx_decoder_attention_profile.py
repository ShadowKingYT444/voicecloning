"""Bind a requested acoustic adapter to the exact folded ONNX stage."""
from pathlib import Path
import json
import math

from file_fingerprint import fingerprint_file
from onnx_staged_runtime import _read_manifest


def validate_decoder_attention_profile(profile, model_dir, root):
    manifest_path = Path(model_dir) / 'meanflow_estimator/manifest.json'
    value = profile.get('decoder_attention')
    raw_strength = profile.get('decoder_attention_strength', 1.0 if value else 0.0)
    if isinstance(raw_strength, bool):
        raise ValueError('Invalid decoder attention strength')
    strength = float(raw_strength)
    if not math.isfinite(strength) or strength < 0 or (strength and not value):
        raise ValueError('Invalid decoder attention strength or missing adapter')
    # Legacy unit fixtures can omit unused acoustic stages. Actual session
    # creation still requires the ordinary stage manifest.
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    patch = manifest.get('adapter_patch')
    if not strength and not patch:
        return dict(enabled=False)
    if not isinstance(patch, dict):
        raise ValueError('Profile requires a folded acoustic ONNX stage')
    _read_manifest(Path(model_dir), 'meanflow_estimator')
    if patch.get('strength') != strength:
        raise ValueError('Profile and ONNX acoustic adapter strengths differ')
    if not strength:
        return dict(enabled=False)
    adapter = Path(root) / value
    if not adapter.is_file() or fingerprint_file(adapter) != patch['adapter_sha256']:
        raise ValueError('Profile and ONNX acoustic adapter hashes differ')
    cache_value = profile.get('conditioning_cache')
    if not cache_value:
        raise ValueError('Acoustic ONNX adapter requires the fitted conditioning cache')
    cache = Path(root) / cache_value
    if not cache.is_file() or fingerprint_file(cache) != patch['initial_conditionals_sha256']:
        raise ValueError('Acoustic ONNX conditioning cache differs from the fitted cache')
    return dict(enabled=True, strength=strength, adapter_sha256=patch['adapter_sha256'],
                initial_conditionals_sha256=patch['initial_conditionals_sha256'],
                patched_weights_sha256=patch['patched_weights_sha256'],
                graph_sha256=patch['graph_sha256'], manifest=str(manifest_path.resolve()))
