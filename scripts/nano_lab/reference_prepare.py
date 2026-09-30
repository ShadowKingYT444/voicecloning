#!/usr/bin/env python3
"""Prepare reproducible Chatterbox Nano voice-reference clips.

The input recordings stay untouched.  This script decodes the two source MP3s,
extracts phrase-aligned mono 24 kHz windows, and writes three deliberately
different variants for each window:

``raw``
    The selected source window after mono downmix/resampling only.
``natural``
    A gentle 55 Hz high-pass and 10.8 kHz low-pass, with safe peak scaling.
``clean``
    ``natural`` followed by a restrained soft spectral gate.  The gate has a
    high gain floor and is intended to reduce a stationary MP3/room bed while
    retaining breath texture.  It is an A/B candidate, never the only copy.

The manifest records exact source timing, source hashes, processing, text, and
measurements.  It also writes short, non-overlapping training rows and held-out
same-speaker rows for later evaluation.  CPU transcription is optional and is
limited to two threads when requested.

This module intentionally uses only dependencies already present in the
voicebox backend environment: numpy, scipy, soundfile, and (optionally)
faster-whisper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.signal
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 24_000
DEFAULT_OUT = ROOT / "artifacts" / "nano_lab" / "references"

ASMR_SOURCE = ROOT / (
    "ASMR  7 Minutes in Heaven... With Your Bully_  [Enemies to Lovers] "
    "[Tsundere] [Confession] [Kiss].mp3"
)
HARVEY_SOURCE = ROOT / "when Harvey speaks, success listens #Suits #HarveySpecter #Shorts.mp3"


@dataclass(frozen=True)
class ClipSpec:
    """One source interval and its text label."""

    clip_id: str
    family: str
    source: Path
    start_s: float
    end_s: float
    transcript: str
    style: str
    split: str = "reference"
    transcript_source: str = "faster-whisper-tiny.en (auto, unverified)"


# Phrase boundaries are intentionally conservative.  Each reference begins
# close to a word boundary, remains between 5 and 15 seconds, and avoids the
# kiss/sound-effect section of the ASMR recording.  ASMR text is copied from
# the bundled tiny.en decode and marked unverified below.  This prevents us
# from claiming a human transcript review that did not occur.
ASMR_REFERENCES: tuple[ClipSpec, ...] = (
    ClipSpec(
        "asmr7_bully_01",
        "asmr7",
        ASMR_SOURCE,
        117.7,
        129.0,
        "I do know you. Shit. What did I do to end up here with you? I know I spun the bottle. Fucking dumbass.",
        "energetic bully / close speech",
    ),
    ClipSpec(
        "asmr7_bully_02",
        "asmr7",
        ASMR_SOURCE,
        138.7,
        151.2,
        "No, it is that bad. You are the biggest nerd I know, and no, I either have to kiss you or miss out on my turn of seven minutes in heaven. All right, leave us.",
        "taunting to conversational",
    ),
    ClipSpec(
        "asmr7_conversational_01",
        "asmr7",
        ASMR_SOURCE,
        181.3,
        194.6,
        "For some reason, you're here now, and stuck in a closet with me. I'm not sure if this is more torture for me or for you, but we are stuck here for 7 minutes. Possibly more considering they might be kind enough to offer me a few extra minutes.",
        "low conversational ASMR",
    ),
    ClipSpec(
        "asmr7_soft_praise_01",
        "asmr7",
        ASMR_SOURCE,
        354.2,
        368.1,
        "The fact is, when I see you, I see this cute little perfect being who lives in a bubble of perfect and eats perfect little bowls of cereal for breakfast.",
        "soft praise / breathy",
    ),
    ClipSpec(
        "asmr7_emotional_01",
        "asmr7",
        ASMR_SOURCE,
        372.6,
        386.0,
        "Okay, what I'm trying to say is that you made me jealous, that you live in this perfect world and you're always so nice. No matter how bad of a day you have or how many times I mean to you.",
        "quiet confession",
    ),
    ClipSpec(
        "asmr7_emotional_02",
        "asmr7",
        ASMR_SOURCE,
        390.7,
        404.7,
        "Mean to you because you're perfect and I want to be the one bad thing to make your world less perfect. You seriously only got from all that that I call you cute.",
        "intimate confession",
    ),
    ClipSpec(
        "asmr7_party_01",
        "asmr7",
        ASMR_SOURCE,
        35.1,
        48.2,
        "So as the one throwing the party, and it also being my party, I would like to announce that I'm turning this game of spin the bottle into a front 4, 7 minutes in heaven.",
        "clearer spoken / energetic",
    ),
)


HARVEY_REFERENCES: tuple[ClipSpec, ...] = (
    ClipSpec(
        "harvey_quote_01",
        "harvey",
        HARVEY_SOURCE,
        0.0,
        6.3,
        "I don't take meetings. I set them. And my respect isn't demanded. It's earned. Excuses don't win championships.",
        "firm declarative",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_quote_02",
        "harvey",
        HARVEY_SOURCE,
        9.0,
        16.1,
        "Never destroy anyone in public when you can accomplish the same result in private. I don't play the odds. I play the man.",
        "measured authority",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_quote_03",
        "harvey",
        HARVEY_SOURCE,
        18.6,
        24.7,
        "Winners don't make excuses when the other side plays the game. They don't have dreams. I have goals. Get it through your head.",
        "assertive rhythm",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_quote_04",
        "harvey",
        HARVEY_SOURCE,
        26.7,
        33.2,
        "I don't pave the way for people. People pave the way for me. Life is this, I like this.",
        "short clipped phrasing",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
)


# Adaptation rows are short and phrase aligned.  They deliberately do not use
# the held-out windows below.  Rows with uncertain words from the ASMR decode
# are omitted; a smaller clean set is more useful than noisy pseudo-labels.
TRAINING_ROWS: tuple[ClipSpec, ...] = (
    ClipSpec(
        "asmr7_train_01",
        "asmr7",
        ASMR_SOURCE,
        35.2,
        41.2,
        "So as the one throwing the party, and it also being my party, I would like to announce",
        "training / energetic",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_02",
        "asmr7",
        ASMR_SOURCE,
        52.1,
        58.0,
        "There's a closet in pretty much every room in this house, so we might as well put them to use.",
        "training / conversational",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_03",
        "asmr7",
        ASMR_SOURCE,
        67.1,
        72.0,
        "I'll be nice enough to let whoever started the game up to go on first.",
        "training / playful",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_04",
        "asmr7",
        ASMR_SOURCE,
        82.8,
        89.0,
        "It's my birthday, so they better make these seven minutes worth it. Please be cute.",
        "training / playful",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_05",
        "asmr7",
        ASMR_SOURCE,
        121.6,
        128.8,
        "What did I do to end up here with you? I know I spun the bottle. Fucking dumbass.",
        "training / bully",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_06",
        "asmr7",
        ASMR_SOURCE,
        181.4,
        187.7,
        "For some reason, you're here now, and stuck in a closet with me.",
        "training / low conversational",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_07",
        "asmr7",
        ASMR_SOURCE,
        329.8,
        335.9,
        "And I hate you, you're too perfect, like untouchable by misfortune.",
        "training / soft confession",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_08",
        "asmr7",
        ASMR_SOURCE,
        354.3,
        360.6,
        "The fact is, when I see you, I see this cute little perfect being who lives in a bubble.",
        "training / soft praise",
        split="train",
    ),
    ClipSpec(
        "asmr7_train_09",
        "asmr7",
        ASMR_SOURCE,
        372.8,
        378.8,
        "Okay, what I'm trying to say is that you made me jealous, that you live in this perfect world",
        "training / intimate confession",
        split="train",
    ),
    ClipSpec(
        "harvey_train_01",
        "harvey",
        HARVEY_SOURCE,
        0.0,
        5.8,
        "I don't take meetings. I set them. And my respect isn't demanded. It's earned.",
        "training / firm declarative",
        split="train",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_train_02",
        "harvey",
        HARVEY_SOURCE,
        9.1,
        13.7,
        "Never destroy anyone in public when you can accomplish the same result in private.",
        "training / measured authority",
        split="train",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_train_03",
        "harvey",
        HARVEY_SOURCE,
        13.7,
        18.3,
        "I don't play the odds. I play the man. Winners don't make excuses when the other side plays the game.",
        "training / assertive rhythm",
        split="train",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
    ClipSpec(
        "harvey_train_04",
        "harvey",
        HARVEY_SOURCE,
        26.8,
        32.9,
        "People pave the way for me. Life is this, I like this.",
        "training / clipped phrasing",
        split="train",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
)


# These windows never appear in the adaptation rows or normal reference set.
# They provide independent same-speaker material for the evaluator.  The first
# four ASMR windows avoid the kiss section; the final window is a quiet outro.
HELDOUT_ROWS: tuple[ClipSpec, ...] = (
    ClipSpec(
        "asmr7_heldout_01",
        "asmr7",
        ASMR_SOURCE,
        92.1,
        102.1,
        "Let's see who's the lucky person who gets to kiss me. Wait a minute. Do I know you?",
        "heldout / playful",
        split="heldout",
    ),
    ClipSpec(
        "asmr7_heldout_02",
        "asmr7",
        ASMR_SOURCE,
        239.0,
        251.1,
        "Not even friends, enemies even. In fact, when I see you, I audibly groan. I feel so disgusted I almost throw up.",
        "heldout / aggressive",
        split="heldout",
    ),
    ClipSpec(
        "asmr7_heldout_03",
        "asmr7",
        ASMR_SOURCE,
        303.8,
        315.9,
        "That's not what I hate you, no it isn't, no don't cry please, if it makes you feel any better.",
        "heldout / emotional",
        split="heldout",
    ),
    ClipSpec(
        "asmr7_heldout_04",
        "asmr7",
        ASMR_SOURCE,
        475.7,
        486.8,
        "Thanks. Wait, that better not be implying I'm also a nerd because I will take back all the nice things I said about you.",
        "heldout / playful",
        split="heldout",
    ),
    ClipSpec(
        "asmr7_heldout_05",
        "asmr7",
        ASMR_SOURCE,
        585.9,
        599.8,
        "Think of this as me making up for all the times I called you something I didn't mean. And well, I guess I've been mean a lot.",
        "heldout / quiet outro",
        split="heldout",
    ),
    ClipSpec(
        "harvey_heldout_01",
        "harvey",
        HARVEY_SOURCE,
        34.0,
        40.8,
        "I don't get lucky. I make my own luck. You know what? That's the difference between us, Allison. You want to lose small. I want to win big.",
        "heldout / closing quote",
        split="heldout",
        transcript_source="faster-whisper-tiny.en (auto, high confidence)",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(path: Path) -> dict[str, Any]:
    """Read stable source metadata without requiring libsndfile MP3 support."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,sample_rate,channels,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        probe = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe is required to inspect {path}: {exc}") from exc
    stream = (probe.get("streams") or [{}])[0]
    fmt = probe.get("format") or {}
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "duration_s": float(fmt.get("duration", 0.0)),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": int(stream["channels"]) if stream.get("channels") else None,
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
    }


def _decode_mono_24k(path: Path) -> np.ndarray:
    """Decode any FFmpeg-supported source as contiguous mono float32 at 24 kHz."""

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"FFmpeg could not decode {path}: {exc}") from exc
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise ValueError(f"Decoded no samples from {path}")
    return np.nan_to_num(np.clip(audio, -1.0, 1.0), copy=False)


def _safe_db(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(value), floor)))


def _frame_rms(audio: np.ndarray, sample_rate: int = TARGET_SR, frame_ms: float = 20.0) -> np.ndarray:
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    count = max(1, int(math.ceil(len(audio) / frame)))
    padded = np.pad(audio, (0, count * frame - len(audio)))
    return np.sqrt(np.mean(padded.reshape(count, frame) ** 2, axis=1) + 1e-20)


def measure_audio(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict[str, float]:
    """Return diagnostics; noise fields are explicitly heuristic proxies."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-20)) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    frames = _frame_rms(audio, sample_rate)
    sorted_frames = np.sort(frames)
    low_count = max(1, int(math.ceil(0.10 * len(sorted_frames))))
    low_rms = float(np.median(sorted_frames[:low_count]))
    high_count = max(1, int(math.ceil(0.50 * len(sorted_frames))))
    high_rms = float(np.median(sorted_frames[-high_count:]))

    # Welch is robust for short clips, and only used for descriptive metrics.
    nperseg = min(2048, max(64, len(audio)))
    freqs, power = scipy.signal.welch(audio, fs=sample_rate, nperseg=nperseg)
    power = np.maximum(power.astype(np.float64), 1e-20)
    pnorm = power / np.sum(power)
    centroid = float(np.sum(freqs * pnorm))
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    high_mask = freqs >= min(8000.0, sample_rate / 3.0)
    high_fraction = float(np.sum(power[high_mask]) / np.sum(power)) if np.any(high_mask) else 0.0
    clip_threshold = 10 ** (-0.1 / 20.0)
    clip_fraction = float(np.mean(np.abs(audio) >= clip_threshold)) if audio.size else 0.0
    return {
        "duration_s": round(len(audio) / sample_rate, 6),
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "rms_dbfs": round(_safe_db(rms), 3),
        "peak_dbfs": round(_safe_db(peak), 3),
        "crest_db": round(_safe_db(peak / max(rms, 1e-12)), 3),
        "dc_offset": round(float(np.mean(audio)) if audio.size else 0.0, 8),
        "clipping_fraction_proxy": round(clip_fraction, 8),
        # These are deliberately named proxies, not claims of perceptual SNR.
        "noise_floor_dbfs_proxy_low10pct_frames": round(_safe_db(low_rms), 3),
        "speech_level_dbfs_proxy_high50pct_frames": round(_safe_db(high_rms), 3),
        "snr_db_proxy_high50_minus_low10": round(_safe_db(high_rms / max(low_rms, 1e-12)), 3),
        "spectral_centroid_hz": round(centroid, 3),
        "spectral_flatness_proxy": round(flatness, 6),
        "high_band_fraction_proxy_ge8khz": round(high_fraction, 6),
    }


def _sos_filter(audio: np.ndarray, sample_rate: int, *, highpass_hz: float | None, lowpass_hz: float | None) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32)
    if highpass_hz is not None:
        sos = scipy.signal.butter(2, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        result = scipy.signal.sosfiltfilt(sos, result).astype(np.float32, copy=False)
    if lowpass_hz is not None:
        sos = scipy.signal.butter(2, lowpass_hz, btype="lowpass", fs=sample_rate, output="sos")
        result = scipy.signal.sosfiltfilt(sos, result).astype(np.float32, copy=False)
    return np.asarray(result, dtype=np.float32)


def _safe_peak_scale(audio: np.ndarray, target_peak_dbfs: float = -2.0, max_gain_db: float = 8.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-8:
        return np.asarray(audio, dtype=np.float32)
    desired = 10 ** (target_peak_dbfs / 20.0)
    gain = desired / peak
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = min(gain, max_gain)
    return np.asarray(np.clip(audio * gain, -0.98, 0.98), dtype=np.float32)


def _gentle_spectral_gate(audio: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Reduce stationary bed with a high floor to preserve breath texture."""

    if len(audio) < 256:
        return np.asarray(audio, dtype=np.float32)
    n_fft = 1024
    hop = 240  # 10 ms at 24 kHz
    win = scipy.signal.windows.hann(n_fft, sym=False)
    freqs, times, stft = scipy.signal.stft(
        audio,
        fs=sample_rate,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(stft)
    frame_energy = np.sqrt(np.mean(magnitude**2, axis=0) + 1e-20)
    noise_count = max(2, int(np.ceil(0.20 * len(frame_energy))))
    noise_indices = np.argsort(frame_energy)[:noise_count]
    noise = np.percentile(magnitude[:, noise_indices], 30.0, axis=1, keepdims=True)
    # At stationary noise magnitude ~= noise; strong speech approaches gain 1.
    ratio = magnitude / (noise + 1e-7)
    attenuation = np.clip((ratio - 1.0) / 2.5, 0.0, 1.0)
    gain = 0.55 + 0.45 * attenuation
    # Smooth gain over nearby bins/frames, avoiding musical noise.
    gain = scipy.ndimage.gaussian_filter(gain, sigma=(1.0, 1.0), mode="nearest")
    clean_stft = stft * gain
    _, clean = scipy.signal.istft(
        clean_stft,
        fs=sample_rate,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    clean = np.asarray(clean[: len(audio)], dtype=np.float32)
    if len(clean) < len(audio):
        clean = np.pad(clean, (0, len(audio) - len(clean)))
    return clean


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32), sample_rate, subtype="PCM_16")


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _variant_audio(raw: np.ndarray, variant: str) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "raw":
        return raw.astype(np.float32, copy=True), {"operations": ["mono_downmix", "resample_24khz"]}
    filtered = _sos_filter(raw - np.mean(raw), TARGET_SR, highpass_hz=55.0, lowpass_hz=10_800.0)
    operations: list[str] = ["remove_dc", "highpass_55hz", "lowpass_10.8khz"]
    if variant == "clean":
        filtered = _gentle_spectral_gate(filtered, TARGET_SR)
        operations.append("soft_spectral_gate_strength_0.35_floor_0.55")
    elif variant != "natural":
        raise ValueError(f"Unknown variant {variant!r}")
    filtered = _safe_peak_scale(filtered)
    operations.append("safe_peak_scale_target_-2dbfs_max_gain_8db")
    return filtered, {"operations": operations}


def _source_level(audio: np.ndarray) -> dict[str, float]:
    return measure_audio(audio, TARGET_SR)


def _iter_specs() -> Iterable[ClipSpec]:
    yield from ASMR_REFERENCES
    yield from HARVEY_REFERENCES
    yield from TRAINING_ROWS
    yield from HELDOUT_ROWS


def _check_specs(specs: Sequence[ClipSpec], source_lengths: dict[Path, float]) -> None:
    for spec in specs:
        duration = spec.end_s - spec.start_s
        if not 2.0 <= duration <= 15.0:
            raise ValueError(f"{spec.clip_id} duration {duration:.3f}s is outside 2-15s")
        if spec.split == "reference" and duration < 5.0:
            raise ValueError(f"{spec.clip_id} is too short for Nano: {duration:.3f}s")
        if spec.start_s < 0 or spec.end_s > source_lengths[spec.source] + 0.05:
            raise ValueError(f"{spec.clip_id} is outside source duration")
    # Guard against accidental held-out leakage.  The intervals may touch a
    # training/ref boundary only if there is no sample overlap.
    for held in (x for x in specs if x.split == "heldout"):
        for other in (x for x in specs if x.split != "heldout" and x.family == held.family):
            if max(held.start_s, other.start_s) < min(held.end_s, other.end_s):
                raise ValueError(f"held-out interval overlaps {other.clip_id}: {held.clip_id}")


def _prepare_spec(
    spec: ClipSpec,
    source_audio: dict[Path, np.ndarray],
    source_meta: dict[Path, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    source = source_audio[spec.source]
    start = max(0, int(round(spec.start_s * TARGET_SR)))
    end = min(len(source), int(round(spec.end_s * TARGET_SR)))
    raw = source[start:end].astype(np.float32, copy=True)
    if len(raw) < TARGET_SR * 2:
        raise ValueError(f"{spec.clip_id} decoded to too few samples")
    family_dir = output_root / spec.family
    clip_dir = family_dir / spec.clip_id
    files: dict[str, str] = {}
    variants: dict[str, Any] = {}
    for variant in ("raw", "natural", "clean"):
        audio, processing = _variant_audio(raw, variant)
        filename = f"{spec.clip_id}__{variant}.wav"
        path = clip_dir / filename
        _write_wav(path, audio)
        files[variant] = _relative_or_absolute(path)
        variants[variant] = {
            "path": _relative_or_absolute(path),
            "processing": processing,
            "measurements": measure_audio(audio, TARGET_SR),
        }
    return {
        "id": spec.clip_id,
        "family": spec.family,
        "split": spec.split,
        "style": spec.style,
        "source": {
            **source_meta[spec.source],
            "start_s": spec.start_s,
            "end_s": spec.end_s,
            "duration_s": round(spec.end_s - spec.start_s, 6),
            "decoded_sample_rate_hz": TARGET_SR,
            "decoded_channel_layout": "mono",
            "window_measurements": _source_level(raw),
        },
        "transcript": spec.transcript,
        "transcript_source": spec.transcript_source,
        "transcript_review": "auto_tiny_en_unverified",
        "files": files,
        "variants": variants,
    }


def _manifest_rows(entries: Sequence[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry["split"] != "train":
            continue
        # Adaptation gets the natural reference by default; clean/raw remain
        # available for ablations and are not silently mixed into the dataset.
        rows.append(
            {
                "audio_path": entry["variants"]["natural"]["path"],
                "text": entry["transcript"],
                "split": "train",
                "speaker_id": entry["family"],
                "source_id": entry["id"],
                "start_s": entry["source"]["start_s"],
                "end_s": entry["source"]["end_s"],
                "transcript_source": entry["transcript_source"],
            }
        )
    return rows


def _write_source_acoustics(
    source_audio: dict[Path, np.ndarray],
    source_meta: dict[Path, dict[str, Any]],
    entries: Sequence[dict[str, Any]],
    output_root: Path,
) -> None:
    report: dict[str, Any] = {"generated_by": "reference_prepare.py", "sources": {}}
    for path, audio in source_audio.items():
        metadata = dict(source_meta[path])
        metadata["decoded_sample_rate_hz"] = TARGET_SR
        metadata["decoded_measurements"] = measure_audio(audio, TARGET_SR)
        # 10-second acoustic survey rows make source choice auditable without
        # shipping another large copy of the MP3.
        windows: list[dict[str, Any]] = []
        window = 10 * TARGET_SR
        for start in range(0, len(audio), window):
            clip = audio[start : min(len(audio), start + window)]
            if len(clip) < TARGET_SR:
                break
            windows.append(
                {
                    "start_s": round(start / TARGET_SR, 3),
                    "end_s": round((start + len(clip)) / TARGET_SR, 3),
                    "measurements": measure_audio(clip, TARGET_SR),
                }
            )
        report["sources"][path.stem] = {**metadata, "ten_second_windows": windows}
    report["reference_ids"] = [x["id"] for x in entries if x["split"] == "reference"]
    report["heldout_ids"] = [x["id"] for x in entries if x["split"] == "heldout"]
    (output_root / "source_acoustics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def prepare(output_root: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    specs = tuple(_iter_specs())
    sources = sorted({spec.source for spec in specs})
    source_meta = {path: _source_metadata(path) for path in sources}
    source_audio = {path: _decode_mono_24k(path) for path in sources}
    source_lengths = {path: len(audio) / TARGET_SR for path, audio in source_audio.items()}
    _check_specs(specs, source_lengths)

    entries = [_prepare_spec(spec, source_audio, source_meta, output_root) for spec in specs]
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/nano_lab/reference_prepare.py",
        "target": {
            "sample_rate_hz": TARGET_SR,
            "channels": 1,
            "format": "PCM_16 WAV",
            "nano_reference_window_s": "5-15 (reference clips)",
        },
        "processing_policy": {
            "default_variant": "natural",
            "clean_variant_role": "A/B only; restrained gate keeps a 0.55 gain floor",
            "raw_variant_role": "source audit and fallback",
            "normalisation": "peak target -2 dBFS, maximum gain +8 dB for processed variants",
            "noise_metrics": "heuristic proxies; no clean-room reference is available",
            "original_sources_preserved": True,
        },
        "sources": [source_meta[path] for path in sources],
        "references": [x for x in entries if x["split"] == "reference"],
        "heldout": [x for x in entries if x["split"] == "heldout"],
        "training": [x for x in entries if x["split"] == "train"],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = _manifest_rows(entries, output_root)
    (output_root / "adaptation_dataset.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_root / "adaptation_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_source_acoustics(source_audio, source_meta, entries, output_root)
    return manifest


def _maybe_transcribe(path: Path, model_path: Path) -> str:
    """Transcribe one file with the bundled tiny.en model, if requested."""

    from faster_whisper import WhisperModel

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        gcd = math.gcd(int(sr), 16_000)
        audio = scipy.signal.resample_poly(audio, 16_000 // gcd, int(sr) // gcd).astype(np.float32)
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    segments, _ = model.transcribe(audio, language="en", beam_size=3, condition_on_previous_text=False)
    return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Decode prepared WAVs with a local faster-whisper model and add ASR text to manifest.",
    )
    parser.add_argument(
        "--whisper-model",
        type=Path,
        default=ROOT / "Whisper_fast_package" / "models" / "tiny.en",
    )
    args = parser.parse_args()
    manifest = prepare(args.output)
    if args.transcribe:
        for entry in manifest["references"] + manifest["heldout"] + manifest["training"]:
            path = ROOT / entry["variants"]["natural"]["path"]
            try:
                entry["asr_check_tiny_en"] = _maybe_transcribe(path, args.whisper_model)
            except Exception as exc:  # pragma: no cover - optional runtime dependency
                entry["asr_check_error"] = f"{type(exc).__name__}: {exc}"
        args.output.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output / 'manifest.json'), "references": len(manifest["references"]), "heldout": len(manifest["heldout"]), "training": len(manifest["training"])}, indent=2))


if __name__ == "__main__":
    main()
