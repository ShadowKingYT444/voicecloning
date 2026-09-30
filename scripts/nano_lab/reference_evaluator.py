#!/usr/bin/env python3
"""Evaluate generated voice clips against the prepared Nano references.

The evaluator is intentionally conservative.  It reports three independent
signals:

* text fidelity: optional Faster-Whisper tiny.en transcription and word error
  rate (WER) against the supplied expected text;
* speaker similarity: optional Resemblyzer cosine to *held-out* source clips
  from the manifest, never to the prompt clip itself;
* waveform diagnostics: clipping and noise-floor/spectral proxies.

Noise fields are labelled proxies.  They cannot prove that a clip is free of
background sound without a clean reference or a human listening pass.  The
script keeps all CPU work at two threads and writes a JSON report suitable for
model sweeps.

Examples::

    python scripts/nano_lab/reference_evaluator.py \
      --manifest artifacts/nano_lab/references/manifest.json \
      --audio artifacts/outputs/sample.wav \
      --expected-text "The meeting starts at nine." \
      --family asmr7 --output artifacts/nano_lab/references/sample_eval.json

For a batch, pass ``--inputs-json`` with rows containing ``audio_path``,
``expected_text`` (optional), ``family`` (optional), and ``label`` (optional).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.signal
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "artifacts" / "nano_lab" / "references" / "manifest.json"
DEFAULT_WHISPER_MODEL = ROOT / "Whisper_fast_package" / "models" / "tiny.en"
TARGET_SR = 16_000


def _rss_mb() -> float | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                try:
                    return round(float(line.split()[1]) / 1024.0, 2)
                except (ValueError, IndexError):
                    pass
    try:
        return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2)
    except (AttributeError, ValueError):
        return None


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, copy=False)
    if sample_rate is not None and int(sr) != int(sample_rate):
        gcd = math.gcd(int(sr), int(sample_rate))
        audio = scipy.signal.resample_poly(
            audio,
            int(sample_rate) // gcd,
            int(sr) // gcd,
        ).astype(np.float32, copy=False)
        sr = int(sample_rate)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32), int(sr)


def _safe_db(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(value), floor)))


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_ms: float = 20.0) -> np.ndarray:
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    count = max(1, int(math.ceil(len(audio) / frame)))
    padded = np.pad(audio, (0, count * frame - len(audio)))
    return np.sqrt(np.mean(padded.reshape(count, frame) ** 2, axis=1) + 1e-20)


def audio_diagnostics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return waveform and clearly labelled noise/clipping diagnostics."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-20)) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    frames = _frame_rms(audio, sample_rate)
    sorted_frames = np.sort(frames)
    low_count = max(1, int(math.ceil(len(sorted_frames) * 0.10)))
    upper_count = max(1, int(math.ceil(len(sorted_frames) * 0.50)))
    noise_rms = float(np.median(sorted_frames[:low_count]))
    speech_rms = float(np.median(sorted_frames[-upper_count:]))
    # A 30 ms frame proxy catches isolated codec/static bursts better than a
    # whole-file RMS while remaining independent of ASR.
    burst_frames = _frame_rms(audio, sample_rate, frame_ms=30.0)
    burst_p99 = float(np.percentile(burst_frames, 99.0))
    nperseg = min(2048, max(64, len(audio)))
    freqs, power = scipy.signal.welch(audio, fs=sample_rate, nperseg=nperseg)
    power = np.maximum(power.astype(np.float64), 1e-20)
    pnorm = power / power.sum()
    high_mask = freqs >= min(8000.0, sample_rate / 3.0)
    high_fraction = float(power[high_mask].sum() / power.sum()) if np.any(high_mask) else 0.0

    # Estimate a high-frequency static proxy only on the lowest-energy frames.
    # A large value is a warning signal, not a perceptual judgement.
    frame_len = max(1, int(round(sample_rate * 0.02)))
    count = max(1, int(math.ceil(len(audio) / frame_len)))
    padded = np.pad(audio, (0, count * frame_len - len(audio))).reshape(count, frame_len)
    low_ids = np.argsort(np.sqrt(np.mean(padded**2, axis=1) + 1e-20))[:low_count]
    high_sos = scipy.signal.butter(4, min(8000.0, sample_rate / 2.0 - 100.0), btype="highpass", fs=sample_rate, output="sos")
    high_frames = scipy.signal.sosfiltfilt(high_sos, padded.reshape(-1)).reshape(count, frame_len)
    stationary_high_rms = float(np.median(np.sqrt(np.mean(high_frames[low_ids] ** 2, axis=1) + 1e-20)))

    near_clip_threshold = 10 ** (-0.1 / 20.0)
    clip_threshold = 10 ** (-0.01 / 20.0)
    return {
        "duration_s": round(len(audio) / sample_rate, 6),
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "rms_dbfs": round(_safe_db(rms), 3),
        "peak_dbfs": round(_safe_db(peak), 3),
        "crest_db": round(_safe_db(peak / max(rms, 1e-12)), 3),
        "dc_offset": round(float(np.mean(audio)) if len(audio) else 0.0, 8),
        "clipping_fraction_abs_ge_minus_0.01dbfs": round(float(np.mean(np.abs(audio) >= clip_threshold)), 8),
        "near_clip_fraction_abs_ge_minus_0.1dbfs": round(float(np.mean(np.abs(audio) >= near_clip_threshold)), 8),
        "burst_p99_dbfs_proxy_30ms": round(_safe_db(burst_p99), 3),
        "noise_floor_dbfs_proxy_low10pct_frames": round(_safe_db(noise_rms), 3),
        "speech_level_dbfs_proxy_high50pct_frames": round(_safe_db(speech_rms), 3),
        "snr_db_proxy_high50_minus_low10": round(_safe_db(speech_rms / max(noise_rms, 1e-12)), 3),
        "stationary_high_band_dbfs_proxy_low10pct_frames": round(_safe_db(stationary_high_rms), 3),
        "spectral_centroid_hz": round(float(np.sum(freqs * pnorm)), 3),
        "spectral_flatness_proxy": round(float(np.exp(np.mean(np.log(power))) / np.mean(power)), 6),
        "high_band_fraction_proxy_ge8khz": round(high_fraction, 6),
        "noise_fields_are_heuristic_proxies": True,
    }


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

_CONTRACTION_EXPANSIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def _expand_contractions(text: str) -> str:
    """Expand common orthographic contractions for a secondary WER view."""

    expanded = text.lower().replace("’", "'")
    # Longest strings first avoids replacing a prefix of a longer contraction.
    for contraction, replacement in sorted(_CONTRACTION_EXPANSIONS.items(), key=lambda item: -len(item[0])):
        expanded = re.sub(rf"(?<![a-z0-9]){re.escape(contraction)}(?![a-z0-9])", replacement, expanded)
    return expanded


def _words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    return _WORD_RE.findall(text)


def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Compute WER and S/I/D counts with a standard Levenshtein DP."""

    ref = _words(reference)
    hyp = _words(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    distance = np.zeros((rows, cols), dtype=np.int32)
    ops: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        distance[i, 0] = i
        ops[i][0] = "D"
    for j in range(1, cols):
        distance[0, j] = j
        ops[0][j] = "I"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                distance[i, j] = distance[i - 1, j - 1]
                ops[i][j] = "C"
                continue
            choices = (
                (distance[i - 1, j - 1] + 1, "S"),
                (distance[i - 1, j] + 1, "D"),
                (distance[i, j - 1] + 1, "I"),
            )
            distance[i, j], ops[i][j] = min(choices, key=lambda x: (x[0], {"S": 0, "D": 1, "I": 2}[x[1]]))
    i, j = len(ref), len(hyp)
    substitutions = deletions = insertions = 0
    while i or j:
        op = ops[i][j]
        if op == "C":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            raise RuntimeError(f"invalid WER operation at {i},{j}")
    errors = substitutions + deletions + insertions
    return {
        "wer": round(errors / max(1, len(ref)), 6),
        "reference_word_count": len(ref),
        "hypothesis_word_count": len(hyp),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": ref,
        "hypothesis_words": hyp,
    }


class TinyEnglishASR:
    def __init__(self, model_path: Path) -> None:
        import torch
        from faster_whisper import WhisperModel

        torch.set_num_threads(2)
        self.model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if sample_rate != TARGET_SR:
            gcd = math.gcd(int(sample_rate), TARGET_SR)
            audio = scipy.signal.resample_poly(audio, TARGET_SR // gcd, int(sample_rate) // gcd).astype(np.float32)
            sample_rate = TARGET_SR
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            texts.append(text)
            rows.append(
                {
                    "start_s": round(float(segment.start), 4),
                    "end_s": round(float(segment.end), 4),
                    "text": text,
                    "avg_logprob": round(float(segment.avg_logprob), 6),
                    "no_speech_prob": round(float(segment.no_speech_prob), 6),
                }
            )
        elapsed = time.perf_counter() - started
        return {
            "text": " ".join(texts).strip(),
            "segments": rows,
            "language": getattr(info, "language", "en"),
            "elapsed_s": round(elapsed, 6),
            "real_time_factor": round(elapsed / max(1e-9, len(audio) / sample_rate), 6),
            "model": "bundled faster-whisper tiny.en",
            "compute_type": "int8",
            "cpu_threads": 2,
        }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


class SpeakerScorer:
    def __init__(self) -> None:
        import torch
        from resemblyzer import VoiceEncoder

        torch.set_num_threads(2)
        self.encoder = VoiceEncoder("cpu", verbose=False)

    def embedding(self, path: Path) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(str(path))
        return np.asarray(self.encoder.embed_utterance(wav), dtype=np.float32)

    def score(self, generated: Path, heldout: Sequence[Path]) -> dict[str, Any]:
        generated_embedding = self.embedding(generated)
        rows: list[dict[str, Any]] = []
        for path in heldout:
            try:
                score = _cosine(generated_embedding, self.embedding(path))
                rows.append({"path": str(path), "cosine": round(score, 6)})
            except Exception as exc:
                rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        values = [float(row["cosine"]) for row in rows if "cosine" in row and np.isfinite(row["cosine"])]
        return {
            "metric": "resemblyzer cosine to independent held-out same-voice refs",
            "heldout_count": len(heldout),
            "successful_count": len(values),
            "mean_cosine": round(float(np.mean(values)), 6) if values else None,
            "min_cosine": round(float(np.min(values)), 6) if values else None,
            "max_cosine": round(float(np.max(values)), 6) if values else None,
            "rows": rows,
            "interpretation": "heuristic speaker-similarity signal; ASMR whisper embeddings can be unstable",
        }


def _manifest_heldout(
    manifest: dict[str, Any],
    family: str | None,
    variant: str,
    excluded_ids: set[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    excluded_ids = excluded_ids or set()
    for entry in manifest.get("heldout", []):
        if entry.get("id") in excluded_ids:
            continue
        if family and entry.get("family") != family:
            continue
        path_value = entry.get("variants", {}).get(variant, {}).get("path")
        if path_value:
            path = _resolve(path_value)
            if path.exists():
                paths.append(path)
    return paths


def heldout_source_baseline(
    manifest: dict[str, Any],
    *,
    family: str | None,
    variant: str,
    excluded_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Measure source-vs-source cosine to establish an attainable range.

    This uses only independent held-out source windows.  It does not compare a
    generated clip to itself or to the prompt window.  Families with one
    held-out clip (the current Harvey source) report ``pair_count=0``.
    """

    excluded_ids = excluded_ids or set()
    scorer = SpeakerScorer()
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for entry in manifest.get("heldout", []):
        if entry.get("id") in excluded_ids:
            continue
        if family and entry.get("family") != family:
            continue
        path_value = entry.get("variants", {}).get(variant, {}).get("path")
        path = _resolve(path_value) if path_value else None
        if path and path.exists():
            grouped.setdefault(str(entry.get("family")), []).append((str(entry.get("id")), path))
    result: dict[str, Any] = {
        "metric": "resemblyzer source-vs-source cosine between independent held-out windows",
        "variant": variant,
        "excluded_ids": sorted(excluded_ids),
        "interpretation": "attainable source consistency range; whisper/style changes can lower cosine",
        "families": {},
        "rss_after_model_mb": _rss_mb(),
    }
    for family_name, rows in grouped.items():
        embeddings: dict[str, np.ndarray] = {}
        errors: list[dict[str, str]] = []
        for item_id, path in rows:
            try:
                embeddings[item_id] = scorer.embedding(path)
            except Exception as exc:
                errors.append({"id": item_id, "error": f"{type(exc).__name__}: {exc}"})
        pairs: list[dict[str, Any]] = []
        ids = sorted(embeddings)
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1 :]:
                pairs.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "cosine": round(_cosine(embeddings[left_id], embeddings[right_id]), 6),
                    }
                )
        values = [float(row["cosine"]) for row in pairs]
        result["families"][family_name] = {
            "clip_count": len(rows),
            "embedding_success_count": len(embeddings),
            "pair_count": len(pairs),
            "mean_cosine": round(float(np.mean(values)), 6) if values else None,
            "min_cosine": round(float(np.min(values)), 6) if values else None,
            "max_cosine": round(float(np.max(values)), 6) if values else None,
            "pairs": pairs,
            "errors": errors,
        }
    return result


def _load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.inputs_json:
        rows = json.loads(Path(args.inputs_json).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("--inputs-json must contain a JSON list")
        result = []
        for row in rows:
            if not isinstance(row, dict) or "audio_path" not in row:
                raise ValueError("each --inputs-json row needs audio_path")
            result.append(dict(row))
        return result
    if not args.audio:
        raise ValueError("pass --audio at least once or use --inputs-json")
    expected = args.expected_text or []
    if len(expected) not in (0, 1, len(args.audio)):
        raise ValueError("--expected-text count must be 0, 1, or match --audio count")
    family = args.family
    labels = args.label or []
    result = []
    for index, audio in enumerate(args.audio):
        result.append(
            {
                "audio_path": audio,
                "expected_text": expected[0] if len(expected) == 1 else (expected[index] if expected else None),
                "family": family,
                "label": labels[index] if index < len(labels) else Path(audio).stem,
            }
        )
    return result


def evaluate(
    manifest: dict[str, Any],
    inputs: Sequence[dict[str, Any]],
    *,
    whisper_model: Path | None,
    transcribe: bool,
    speaker: bool,
    heldout_variant: str,
    excluded_heldout_ids: set[str] | None = None,
) -> dict[str, Any]:
    asr: TinyEnglishASR | None = None
    speaker_scorer: SpeakerScorer | None = None
    load_events: dict[str, Any] = {"rss_before_models_mb": _rss_mb()}
    if transcribe:
        if whisper_model is None or not whisper_model.exists():
            raise FileNotFoundError(f"Whisper model not found: {whisper_model}")
        asr = TinyEnglishASR(whisper_model)
        load_events["rss_after_whisper_mb"] = _rss_mb()
    rows: list[dict[str, Any]] = []
    for item in inputs:
        path = _resolve(item["audio_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        audio, sr = _load_audio(path)
        row: dict[str, Any] = {
            "label": item.get("label") or path.stem,
            "audio_path": str(path),
            "family": item.get("family"),
            "expected_text": item.get("expected_text"),
            "measurements": audio_diagnostics(audio, sr),
            "rss_after_audio_load_mb": _rss_mb(),
        }
        if asr is not None:
            row["asr"] = asr.transcribe(audio, sr)
            if item.get("expected_text"):
                row["wer"] = word_error_rate(item["expected_text"], row["asr"]["text"])
                row["wer_contraction_normalized"] = word_error_rate(
                    _expand_contractions(item["expected_text"]),
                    _expand_contractions(row["asr"]["text"]),
                )
        if speaker:
            if speaker_scorer is None:
                speaker_scorer = SpeakerScorer()
                load_events["rss_after_speaker_model_mb"] = _rss_mb()
            heldout = _manifest_heldout(
                manifest,
                item.get("family"),
                heldout_variant,
                excluded_ids=excluded_heldout_ids,
            )
            if heldout:
                row["speaker_similarity"] = speaker_scorer.score(path, heldout)
            else:
                row["speaker_similarity"] = {
                    "metric": "resemblyzer cosine to independent held-out same-voice refs",
                    "heldout_count": 0,
                    "successful_count": 0,
                    "mean_cosine": None,
                    "rows": [],
                    "error": "no heldout refs for requested family",
                }
        rows.append(row)
    load_events["rss_after_evaluation_mb"] = _rss_mb()
    return {
        "schema_version": 1,
        "generated_by": "scripts/nano_lab/reference_evaluator.py",
        "manifest": str(DEFAULT_MANIFEST),
        "runtime": {
            "cpu_threads": 2,
            "whisper_enabled": transcribe,
            "speaker_enabled": speaker,
            "whisper_model": str(whisper_model) if whisper_model else None,
            "heldout_variant": heldout_variant,
            "excluded_heldout_ids": sorted(excluded_heldout_ids or set()),
            "rss_mb": load_events,
        },
        "inputs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio", action="append", help="Generated WAV path; repeat for a batch.")
    parser.add_argument("--expected-text", action="append", help="Expected text; repeat or pass one for all audios.")
    parser.add_argument("--family", choices=("asmr7", "harvey"))
    parser.add_argument("--label", action="append")
    parser.add_argument("--inputs-json", type=Path, help="JSON list of audio_path/expected_text/family/label rows.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--no-transcribe", action="store_true", help="Skip ASR/WER and report waveform/speaker metrics only.")
    parser.add_argument("--no-speaker", action="store_true", help="Skip Resemblyzer held-out speaker cosine.")
    parser.add_argument("--heldout-variant", choices=("raw", "natural", "clean"), default="natural")
    parser.add_argument(
        "--heldout-baseline",
        action="store_true",
        help="Report source-vs-source cosine among manifest heldouts, without evaluating generated audio.",
    )
    parser.add_argument(
        "--exclude-heldout-id",
        action="append",
        default=[],
        help="Exclude a manifest heldout id (repeatable; useful when a baseline prompt overlaps it).",
    )
    args = parser.parse_args()
    manifest_path = _resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.heldout_baseline:
        report = heldout_source_baseline(
            manifest,
            family=args.family,
            variant=args.heldout_variant,
            excluded_ids=set(args.exclude_heldout_id),
        )
        report["manifest"] = str(manifest_path)
        encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return
    inputs = _load_inputs(args)
    report = evaluate(
        manifest,
        inputs,
        whisper_model=args.whisper_model,
        transcribe=not args.no_transcribe,
        speaker=not args.no_speaker,
        heldout_variant=args.heldout_variant,
        excluded_heldout_ids=set(args.exclude_heldout_id),
    )
    report["manifest"] = str(manifest_path)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
