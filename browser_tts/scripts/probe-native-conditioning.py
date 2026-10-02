#!/usr/bin/env python3
"""Compare the public Nano prefix with native Nano T3 conditioning.

Run through scripts/nano_lab/bounded_job.py. This loads the 5.7 MB voice
encoder and only the selected rows from the Nano T3 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "browser_tts/public/voice/asmr_t3_seed47_fit.wav"
STATE_MANIFEST = ROOT / "browser_tts/public/voice/asmr-state.json"
STATE_DATA = ROOT / "browser_tts/public/voice/asmr-state.bin"
CHECKPOINT = ROOT / "models/chatterbox-nano"
NANO_REVISION = "71ccd1d0081b430592cea481f4307e764e07bc64"
REFERENCE_SHA256 = "67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0"
STATE_SHA256 = "6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e"
VE_SHA256 = "f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c"
T3_SHA256 = "72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9"
FEATURE_SHAPE = (1, 89, 768)
TOKEN_SHAPE = (1, 88)
SPEECH_VOCAB = 6561


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_reference(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1
                or wav.getframerate() != 24_000 or wav.getsampwidth() != 3):
            raise ValueError("Reference must be the pinned mono 24 kHz PCM-24 WAV.")
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if len(raw) != frames * 3:
        raise ValueError("Reference frame count does not match its PCM payload.")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    signed = (values ^ 0x800000) - 0x800000
    return signed.astype(np.float32).reshape(-1) / np.float32(8_388_608.0)


def read_public_state(manifest_path: Path, data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "chatterbox-nano-reference-state-v1"
            or manifest.get("model", {}).get("revision") != "4a66d7dab72a9e98f24b515d49a1d7a81632df2e"
            or manifest.get("reference", {}).get("sha256") != REFERENCE_SHA256):
        raise ValueError("Fixed voice-state manifest is not the pinned public state.")
    if manifest.get("data", {}).get("sha256") != STATE_SHA256:
        raise ValueError("Fixed voice-state manifest has an unexpected binary hash.")
    raw = data_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != STATE_SHA256 or len(raw) != 331_200:
        raise ValueError("Fixed voice-state binary failed its pinned size/SHA-256 check.")
    features = np.frombuffer(raw, dtype="<f4", count=89 * 768).reshape(FEATURE_SHAPE).copy()
    tokens = np.frombuffer(raw, dtype="<i8", count=88, offset=89 * 768 * 4).reshape(TOKEN_SHAPE).copy()
    if not np.isfinite(features).all() or np.any(tokens < 0) or np.any(tokens >= SPEECH_VOCAB):
        raise ValueError("Public state has non-finite features or invalid S3 prompt tokens.")
    return features, tokens


def selected_speech_rows(checkpoint, token_ids: list[int]) -> np.ndarray:
    entry = checkpoint.header.get("speech_emb.weight")
    if (entry is None or entry.get("dtype") != "F32"
            or entry.get("shape") != [SPEECH_VOCAB + 2, 768]):
        raise ValueError("Unexpected Nano speech embedding tensor contract.")
    rows: dict[int, np.ndarray] = {}
    data_offset = entry["data_offsets"][0]
    row_bytes = 768 * np.dtype("<f4").itemsize
    for token_id in set(token_ids):
        checkpoint.file.seek(checkpoint.start + data_offset + token_id * row_bytes)
        raw = checkpoint.file.read(row_bytes)
        if len(raw) != row_bytes:
            raise ValueError(f"Could not read Nano speech embedding row {token_id}.")
        rows[token_id] = np.frombuffer(raw, dtype="<f4").copy()
    return np.stack([rows[token_id] for token_id in token_ids]).astype(np.float32, copy=False)


def compare(public: np.ndarray, native: np.ndarray) -> dict[str, float]:
    left = np.asarray(public, dtype=np.float64).reshape(-1)
    right = np.asarray(native, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Comparison arrays have different shapes or non-finite values.")
    delta = left - right
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denominator = left_norm * right_norm
    return {
        "elements": int(left.size),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": float(np.dot(left, right) / denominator) if denominator else 0.0,
        "public_l2_norm": left_norm,
        "native_l2_norm": right_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--state-manifest", type=Path, default=STATE_MANIFEST)
    parser.add_argument("--state-data", type=Path, default=STATE_DATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.out_dir / "native-prefix.npy").exists() or (args.out_dir / "probe.json").exists():
        raise FileExistsError("Probe output already exists; choose a fresh --out-dir.")

    reference_hash = sha256_file(args.reference)
    if reference_hash != REFERENCE_SHA256:
        raise ValueError(f"Reference SHA-256 mismatch: {reference_hash}.")
    public_features, public_tokens = read_public_state(args.state_manifest, args.state_data)
    ve_path = args.checkpoint / "ve.safetensors"
    t3_path = args.checkpoint / "t3_nano_v1.safetensors"
    hashes = {"ve.safetensors": sha256_file(ve_path), "t3_nano_v1.safetensors": sha256_file(t3_path)}
    if hashes["ve.safetensors"] != VE_SHA256 or hashes["t3_nano_v1.safetensors"] != T3_SHA256:
        raise ValueError(f"Native Nano checkpoint hashes do not match revision {NANO_REVISION}: {hashes}.")

    sys.path.insert(0, str(ROOT / "vendor/chatterbox/src"))
    sys.path.insert(0, str(ROOT / "scripts/nano_lab"))
    import librosa
    import torch
    import torch.nn.functional as functional
    from checkpoint_reader import CheckpointReader
    from safetensors.torch import load_file
    from chatterbox.models.voice_encoder import VoiceEncoder

    torch.set_num_threads(2)
    voice_encoder = VoiceEncoder().eval()
    voice_encoder.load_state_dict(load_file(str(ve_path), device="cpu"), strict=True)
    audio_24k = read_reference(args.reference)
    audio_16k = librosa.resample(audio_24k, orig_sr=24_000, target_sr=16_000)
    with torch.inference_mode():
        speaker = voice_encoder.embeds_from_wavs([audio_16k], sample_rate=16_000)
    speaker = np.asarray(speaker, dtype=np.float32)
    if speaker.shape != (1, 256) or not np.isfinite(speaker).all():
        raise ValueError(f"Native VoiceEncoder returned unexpected speaker embedding {speaker.shape}.")

    token_ids = [int(value) for value in public_tokens[0]]
    with CheckpointReader(t3_path) as checkpoint:
        projection_weight = checkpoint.get_tensor("cond_enc.spkr_enc.weight").float().numpy().copy()
        projection_bias = checkpoint.get_tensor("cond_enc.spkr_enc.bias").float().numpy().copy()
        prompt_rows = selected_speech_rows(checkpoint, token_ids)
    if projection_weight.shape != (768, 256) or projection_bias.shape != (768,):
        raise ValueError("Nano speaker projection has an unexpected shape.")
    with torch.inference_mode():
        speaker_row = functional.linear(
            torch.from_numpy(speaker),
            torch.from_numpy(projection_weight),
            torch.from_numpy(projection_bias),
        ).numpy()
    native_prefix = np.concatenate((speaker_row[:, None, :], prompt_rows[None, :, :]), axis=1).astype(np.float32)
    if native_prefix.shape != FEATURE_SHAPE:
        raise ValueError(f"Native prefix has unexpected shape {native_prefix.shape}.")

    duration_seconds = len(audio_24k) / 24_000
    report = {
        "schema": "browser_tts.native-conditioning-probe/v1",
        "reference": {"sha256": reference_hash, "sample_rate": 24_000, "duration_seconds": duration_seconds,
                      "native_prepare_conditionals_duration_supported": duration_seconds > 5.0,
                      "norm_loudness": False, "resample_target_hz": 16_000},
        "public_state_sha256": STATE_SHA256,
        "native_checkpoint": {"repo_id": "ResembleAI/chatterbox-nano", "revision": NANO_REVISION, "sha256": hashes},
        "public_feature_shape": list(public_features.shape),
        "native_prefix_shape": list(native_prefix.shape),
        "prompt_token_count": len(token_ids),
        "speaker_row": compare(public_features[0, 0], native_prefix[0, 0]),
        "prompt_rows": compare(public_features[0, 1:], native_prefix[0, 1:]),
        "comparison": "descriptive_only; this script does not synthesize speech or apply an acceptance threshold",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = args.out_dir / "native-prefix.npy"
    report_path = args.out_dir / "probe.json"
    np.save(prefix_path, native_prefix, allow_pickle=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
