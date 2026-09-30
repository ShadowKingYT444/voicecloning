"""Measure how reference conditioning changes a fixed source-token decode.

This is a source reconstruction diagnostic.  It is not a zero-shot clone
candidate and it does not test text generation.  The target speech-token
sequence is extracted once from the source recording and reused for every
conditioning variant.  Each variant receives the same explicit random seed,
flow decoder, native HiFT vocoder, Perth watermark, and delivery master.

The four conditioning variants are:

``baseline``
    The saved production conditionals.
``self_all``
    Conditionals prepared from the target source itself.
``baseline_self_embedding``
    Baseline prompt tokens/features with only the source S3Gen speaker embedding.
``baseline_self_prompt``
    Baseline speaker embeddings with the source S3Gen prompt token/feature
    pair and their lengths.

The self variants are intentionally labelled ``SELFLEAKDIAGNOSTIC``.  They
measure codec and conditioning effects only.  They are not evidence of new
text voice cloning quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
SAMPLE_RATE = 24_000
S3GEN_SIL = 4299
DEFAULT_MAX_TARGET_TOKENS = 500
DEFAULT_SEEDS = (10031, 10097)


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
    generated = temporary if temporary.exists() else Path(str(temporary) + ".npz")
    os.replace(generated, path)


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _clone_value(value: Any) -> Any:
    """Clone tensors without retaining an autograd graph; copy other values."""

    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return clone().detach()
    return copy.deepcopy(value)


def clone_conditionals(conditionals: Any) -> Any:
    """Deep-copy a ``Conditionals`` object without changing tensor values."""

    return copy.deepcopy(conditionals)


def mix_conditionals(baseline: Any, source: Any, variant: str) -> Any:
    """Construct one ablation while preserving all unselected fields.

    The function is intentionally duck-typed.  Pure tests can use light fake
    objects, while the command uses the vendor ``Conditionals`` class.
    """

    if variant not in {"baseline", "self_all", "baseline_self_embedding", "baseline_self_prompt"}:
        raise ValueError(f"unknown conditioning variant: {variant}")
    if variant == "baseline":
        return clone_conditionals(baseline)
    if variant == "self_all":
        return clone_conditionals(source)

    result = clone_conditionals(baseline)
    if variant == "baseline_self_embedding":
        if "embedding" not in result.gen or "embedding" not in source.gen:
            raise ValueError("embedding-only variant requires gen['embedding'] in both conditionals")
        result.gen["embedding"] = _clone_value(source.gen["embedding"])
        return result

    required = ("prompt_token", "prompt_token_len", "prompt_feat", "prompt_feat_len")
    missing = [key for key in required if key not in source.gen]
    if missing:
        raise ValueError(f"source conditionals are missing prompt fields: {missing}")
    for key in required:
        result.gen[key] = _clone_value(source.gen[key])
    return result


def _array_for_hash(value: Any) -> np.ndarray | None:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.dtype == object:
        return None
    return np.ascontiguousarray(array)


def _value_record(value: Any) -> dict[str, Any]:
    array = _array_for_hash(value)
    if array is None:
        encoded = repr(value).encode("utf-8")
        return {"kind": "repr", "sha256": _sha256_bytes(encoded), "repr": repr(value)}
    return {
        "kind": "tensor_or_array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def conditionals_record(conditionals: Any) -> dict[str, Any]:
    """Record all vendor T3 and S3Gen conditional tensor values."""

    t3 = getattr(conditionals, "t3", None)
    t3_record: dict[str, Any] = {}
    for name in ("speaker_emb", "clap_emb", "cond_prompt_speech_tokens", "cond_prompt_speech_emb", "emotion_adv"):
        if t3 is not None and hasattr(t3, name):
            value = getattr(t3, name)
            if value is not None:
                t3_record[name] = _value_record(value)
    gen = getattr(conditionals, "gen", None)
    gen_record: dict[str, Any] = {}
    if isinstance(gen, Mapping):
        for key in sorted(gen):
            gen_record[str(key)] = _value_record(gen[key])
    return {"t3": t3_record, "gen": gen_record}


def target_token_record(tokens: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tokens, dtype=np.int64)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "count": int(values.size),
        "sha256": _sha256_bytes(np.ascontiguousarray(values).tobytes(order="C")),
    }


def _seed_all(torch: Any, seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sync(torch: Any, model: Any) -> None:
    if str(getattr(model, "device", "")).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_source_audio(path: Path, *, sample_rate: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"source audio is empty or non-finite: {path}")
    return audio


def _apply_watermark_and_master(model: Any, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from quality_sweep import master

    raw = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("HiFT waveform contains NaN or infinity")
    marked = model.watermarker.apply_watermark(raw, sample_rate=SAMPLE_RATE)
    marked = np.asarray(marked, dtype=np.float32).reshape(-1)
    mastered, processing = master(marked, SAMPLE_RATE)
    return marked, np.asarray(mastered, dtype=np.float32), processing


def _write_audio(path: Path, waveform: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(waveform, dtype=np.float32), SAMPLE_RATE, subtype="PCM_24")


def _flow_decode(model: Any, conditionals: Any, tokens: np.ndarray, *, torch: Any, seed: int, steps: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _seed_all(torch, seed)
    speech = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).to(device=model.device, dtype=torch.long).view(1, -1)
    started = time.perf_counter()
    with torch.inference_mode():
        mel = model.s3gen.flow_inference(
            speech_tokens=speech,
            ref_dict=conditionals.gen,
            n_cfm_timesteps=int(steps),
            finalize=True,
        )
        hift = model.s3gen.hift_inference(mel, None)
    _sync(torch, model)
    waveform = hift[0] if isinstance(hift, (tuple, list)) else hift
    waveform = waveform.detach().float().cpu().numpy().reshape(-1)
    mel_array = mel.detach().float().cpu().numpy()
    if mel_array.ndim != 3 or mel_array.shape[0] != 1 or mel_array.shape[1] != 80:
        raise RuntimeError(f"flow returned unexpected mel shape {mel_array.shape}")
    return mel_array, waveform, {"elapsed_seconds": time.perf_counter() - started, "mel_shape": list(mel_array.shape), "token_count": int(tokens.size)}


def _source_mel(model: Any, source_path: Path, *, torch: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import librosa
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram

    audio24, _ = librosa.load(str(source_path), sr=SAMPLE_RATE, mono=True)
    audio24 = np.asarray(audio24, dtype=np.float32)
    normalized = np.asarray(model.norm_loudness(audio24, SAMPLE_RATE), dtype=np.float32)
    with torch.inference_mode():
        mel = mel_spectrogram(torch.from_numpy(normalized)[None, :]).detach().float().cpu().numpy()
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
        raise RuntimeError(f"vendor mel extractor returned unexpected shape {mel.shape}")
    return mel, normalized, {
        "mel_extractor": "vendor/chatterbox/src/chatterbox/models/s3gen/utils/mel.py::mel_spectrogram",
        "normalization": "model.norm_loudness(..., target_lufs=-27) via prepare_conditionals path",
        "sample_rate": SAMPLE_RATE,
        "audio_samples": int(normalized.size),
        "mel_shape": list(mel.shape),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source WAV/MP3 used for target tokens and self conditioning")
    parser.add_argument("--conditionals", type=Path, required=True, help="saved baseline Conditionals cache")
    parser.add_argument("--text", required=True, help="audited transcript label for the source excerpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-target-tokens", type=int, default=DEFAULT_MAX_TARGET_TOKENS)
    parser.add_argument("--steps", type=int, default=2)
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_target_tokens < 2:
        raise ValueError("--max-target-tokens must be >= 2")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    source = _resolve(args.source)
    conditionals_path = _resolve(args.conditionals)
    model_dir = _resolve(args.model_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    if not conditionals_path.exists():
        raise FileNotFoundError(conditionals_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    # Model-backed imports and loading stay inside the command.  The module
    # remains safe to import for pure tests and manifest inspection.
    import torch
    from adaptation import configure_runtime, extract_target_tokens, load_audio, load_nano
    from chatterbox.models.s3gen.const import S3GEN_SIL as VENDOR_S3GEN_SIL
    from chatterbox.tts_turbo import Conditionals

    if int(VENDOR_S3GEN_SIL) != S3GEN_SIL:
        raise RuntimeError(f"unexpected vendor S3GEN_SIL={VENDOR_S3GEN_SIL}; expected {S3GEN_SIL}")
    configure_runtime(2)
    started = time.perf_counter()
    rss_before = _rss_bytes()
    model = load_nano(model_dir, args.device)
    model.t3.eval()
    model.s3gen.eval()
    raw16 = load_audio(source, sample_rate=16_000)
    with torch.inference_mode():
        source_ids = np.asarray(extract_target_tokens(model, raw16, int(args.max_target_tokens)), dtype=np.int64)
    target_tokens = np.concatenate((source_ids, np.full((3,), S3GEN_SIL, dtype=np.int64)))
    np.save(output_dir / "target_speech_tokens.npy", target_tokens)

    baseline = Conditionals.load(conditionals_path, map_location="cpu").to(model.device)
    with torch.inference_mode():
        model.prepare_conditionals(str(source), norm_loudness=True)
    source_conditionals = clone_conditionals(model.conds)
    variants = {
        "baseline": mix_conditionals(baseline, source_conditionals, "baseline"),
        "self_all": mix_conditionals(baseline, source_conditionals, "self_all"),
        "baseline_self_embedding": mix_conditionals(baseline, source_conditionals, "baseline_self_embedding"),
        "baseline_self_prompt": mix_conditionals(baseline, source_conditionals, "baseline_self_prompt"),
    }
    target_record = target_token_record(target_tokens)
    target_record.update({"source_ids_count": int(source_ids.size), "silence_id": S3GEN_SIL, "silence_count": 3})
    _write_json(output_dir / "target_speech_tokens.json", {
        "kind": "genuine_source_target_tokens",
        "synthetic_tts": False,
        "source_audio": str(source),
        "source_audio_sha256": _sha256(source),
        "target_tokens_path": str((output_dir / "target_speech_tokens.npy").resolve()),
        "target_tokens": target_record,
        "text_label": args.text,
        "disclosure": "These IDs are copied from source speech for reconstruction diagnostics. They are not text-generated TTS tokens.",
    })

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "nano_acoustic_conditioning_probe",
        "research_only": True,
        "synthetic_tts": False,
        "synthetic_audio": True,
        "self_leak_diagnostic": True,
        "source_audio": str(source),
        "source_audio_sha256": _sha256(source),
        "conditionals": str(conditionals_path),
        "conditionals_sha256": _sha256(conditionals_path),
        "model_dir": str(model_dir),
        "model_loader": "adaptation.load_nano -> runtime.NanoEngine.from_pretrained(optimized=True, fp32, cpu_threads=2)",
        "device": str(args.device),
        "cpu_threads": 2,
        "text": args.text,
        "target_tokens": target_record,
        "target_tokens_path": str((output_dir / "target_speech_tokens.npy").resolve()),
        "flow_steps": int(args.steps),
        "seeds": [int(value) for value in args.seeds],
        "conditioning_variants": {},
        "rows": [],
        "noise_control": {
            "seed_reset_per_variant": True,
            "seed_reset_before_flow": True,
            "same_target_tokens": True,
            "prompt_length_caveat": "Different prompt lengths consume different portions of the flow noise stream; embedding-only has identical prompt shape and is the clean ablation.",
        },
        "disclosure": "SELFLEAKDIAGNOSTIC source-token reconstruction. Self-conditioned rows use the target recording as a reference and are not zero-shot or new-text clone evidence.",
    }
    for name, conds in variants.items():
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        cond_record = conditionals_record(conds)
        cond_payload = {
            "variant": name,
            "conditioning": cond_record,
            "target_tokens_sha256": target_record["sha256"],
            "baseline_conditionals_sha256": _sha256(conditionals_path),
            "source_conditionals_audio_sha256": _sha256(source),
            "self_leak_diagnostic": name != "baseline",
        }
        _write_json(variant_dir / "conditioning.json", cond_payload)
        manifest["conditioning_variants"][name] = {
            "conditioning_path": str((variant_dir / "conditioning.json").resolve()),
            "conditioning": cond_record,
            "self_leak_diagnostic": name != "baseline",
        }
        for seed in [int(value) for value in args.seeds]:
            run_dir = variant_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            row_started = time.perf_counter()
            mel, waveform, stats = _flow_decode(model, conds, target_tokens, torch=torch, seed=seed, steps=args.steps)
            raw, mastered, processing = _apply_watermark_and_master(model, waveform)
            mel_path = run_dir / "mel.npz"
            _save_npz(mel_path, {"mel": mel, "speech_tokens": target_tokens})
            raw_path = run_dir / "audio.wav"
            master_path = run_dir / "audio.master.wav"
            _write_audio(raw_path, raw)
            _write_audio(master_path, mastered)
            row = {
                "variant": name,
                "seed": seed,
                "text": args.text,
                "target_tokens_sha256": target_record["sha256"],
                "target_token_count": int(target_tokens.size),
                "mel_path": str(mel_path.resolve()),
                "mel_sha256": _sha256(mel_path),
                "audio_path": str(raw_path.resolve()),
                "audio_sha256": _sha256(raw_path),
                "master_path": str(master_path.resolve()),
                "master_sha256": _sha256(master_path),
                "seconds": float(raw.size / SAMPLE_RATE),
                "flow": stats,
                "master_processing": processing,
                "rss_bytes": _rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(),
                "elapsed_seconds": time.perf_counter() - row_started,
                "self_leak_diagnostic": name != "baseline",
            }
            manifest["rows"].append(row)
            _write_json(output_dir / "manifest.json", manifest)
            print(json.dumps({"variant": name, "seed": seed, "peak_rss_mib": _peak_rss_bytes() / 2**20}), flush=True)

    # One source-mel HiFT upper bound.  This keeps the exact target waveform
    # envelope while retaining the native vocoder, Perth, and master stages.
    source_mel, normalized_source, source_mel_meta = _source_mel(model, source, torch=torch)
    upper_dir = output_dir / "source_mel_upperbound" / f"seed_{int(args.seeds[0])}"
    upper_dir.mkdir(parents=True, exist_ok=True)
    started_upper = time.perf_counter()
    _seed_all(torch, int(args.seeds[0]))
    with torch.inference_mode():
        upper_hift = model.s3gen.hift_inference(torch.from_numpy(source_mel).to(model.device), None)
    upper_wave = upper_hift[0] if isinstance(upper_hift, (tuple, list)) else upper_hift
    upper_wave = upper_wave.detach().float().cpu().numpy().reshape(-1)
    upper_raw, upper_master, upper_processing = _apply_watermark_and_master(model, upper_wave)
    upper_mel_path = upper_dir / "mel.npz"
    _save_npz(upper_mel_path, {"mel": source_mel})
    upper_raw_path = upper_dir / "audio.wav"
    upper_master_path = upper_dir / "audio.master.wav"
    _write_audio(upper_raw_path, upper_raw)
    _write_audio(upper_master_path, upper_master)
    manifest["source_mel_upperbound"] = {
        "variant": "source_mel_upperbound",
        "seed": int(args.seeds[0]),
        "mel_path": str(upper_mel_path.resolve()),
        "mel_sha256": _sha256(upper_mel_path),
        "audio_path": str(upper_raw_path.resolve()),
        "audio_sha256": _sha256(upper_raw_path),
        "master_path": str(upper_master_path.resolve()),
        "master_sha256": _sha256(upper_master_path),
        "source_mel": source_mel_meta,
        "normalized_source_sha256": _sha256_bytes(np.ascontiguousarray(normalized_source).tobytes()),
        "master_processing": upper_processing,
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": time.perf_counter() - started_upper,
        "disclosure": "Vocoder upper bound uses vendor mel_spectrogram of normalized source audio. It is not a TTS clone.",
    }
    manifest["peak_rss_bytes"] = _peak_rss_bytes()
    manifest["rss_after_run_bytes"] = _rss_bytes()
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["rss_before_load_bytes"] = rss_before
    manifest["command"] = list(sys.argv)
    _write_json(output_dir / "manifest.json", manifest)
    return {"output_dir": str(output_dir), "rows": len(manifest["rows"]), "peak_rss_bytes": manifest["peak_rss_bytes"], "research_only": True}


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

