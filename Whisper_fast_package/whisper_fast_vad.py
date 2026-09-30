"""Low-latency Whisper tiny + WebRTC-VAD runtime.

The package deliberately keeps the realtime surface small: feed a file/array
for batch transcription or feed PCM chunks to ``stream_transcribe``. Silence is
removed before Whisper runs, and CTranslate2's tiny.en INT8 model is reused for
every segment.
"""

from __future__ import annotations

import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import scipy.signal
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "models" / "tiny.en"
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SpeechSegment:
    """A VAD-selected sample range at 16 kHz."""

    start_sample: int
    end_sample: int
    audio: np.ndarray

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: tuple[dict[str, object], ...]
    language: str
    audio_seconds: float
    speech_seconds: float
    vad_segments: int
    elapsed_seconds: float

    @property
    def real_time_factor(self) -> float:
        return self.elapsed_seconds / self.audio_seconds if self.audio_seconds else 0.0


def load_audio(audio_path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load any soundfile-readable audio as mono float32 at ``sample_rate``."""
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                src_rate = handle.getframerate()
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                raw = handle.readframes(handle.getnframes())
            if sample_width == 2:
                audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                audio = audio.reshape(-1, channels).mean(axis=1)
            else:
                audio, src_rate = sf.read(str(path), always_2d=False, dtype="float32")
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
        except (wave.Error, OSError):
            audio, src_rate = sf.read(str(path), always_2d=False, dtype="float32")
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
    else:
        audio, src_rate = sf.read(str(path), always_2d=False, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    return resample_audio(audio, int(src_rate), sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = np.nan_to_num(audio, copy=False)
    if source_rate != target_rate:
        gcd = math.gcd(int(source_rate), int(target_rate))
        audio = scipy.signal.resample_poly(
            audio,
            int(target_rate) // gcd,
            int(source_rate) // gcd,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


class WebRTCVAD:
    """Small, CPU-only VAD with padding and silence-gap merging."""

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_ms: int = 20,
        padding_ms: int = 240,
        max_silence_ms: int = 260,
        min_speech_ms: int = 120,
        fallback_full_audio: bool = True,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be 0..3")
        self.frame_ms = frame_ms
        self.frame_samples = TARGET_SAMPLE_RATE * frame_ms // 1000
        self.padding_frames = max(0, padding_ms // frame_ms)
        self.max_silence_frames = max(0, max_silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.fallback_full_audio = fallback_full_audio
        self._vad = webrtcvad.Vad(aggressiveness)

    def _flags(self, audio: np.ndarray) -> list[bool]:
        frame = self.frame_samples
        padded = np.pad(audio, (0, (-len(audio)) % frame))
        pcm = (np.clip(padded, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        return [
            self._vad.is_speech(
                pcm[start : start + frame].tobytes(),
                TARGET_SAMPLE_RATE,
            )
            for start in range(0, len(pcm), frame)
        ]

    def ranges(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Return padded speech ranges in samples."""
        if len(audio) == 0:
            return []
        flags = self._flags(audio)
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        last_voice = -10**9
        for index, voiced in enumerate(flags):
            if voiced:
                if run_start is None:
                    run_start = index
                last_voice = index
            elif run_start is not None and index - last_voice > self.max_silence_frames:
                if last_voice - run_start + 1 >= self.min_speech_frames:
                    runs.append((run_start, last_voice + 1))
                run_start = None
        if run_start is not None and last_voice - run_start + 1 >= self.min_speech_frames:
            runs.append((run_start, last_voice + 1))

        if not runs:
            return [(0, len(audio))] if self.fallback_full_audio else []

        result: list[tuple[int, int]] = []
        for start_frame, end_frame in runs:
            start = max(0, (start_frame - self.padding_frames) * self.frame_samples)
            end = min(len(audio), (end_frame + self.padding_frames) * self.frame_samples)
            if result and start <= result[-1][1] + self.max_silence_frames * self.frame_samples:
                result[-1] = (result[-1][0], max(result[-1][1], end))
            else:
                result.append((start, end))
        return result

    def segment(self, audio: np.ndarray) -> list[SpeechSegment]:
        return [
            SpeechSegment(start, end, np.ascontiguousarray(audio[start:end], dtype=np.float32))
            for start, end in self.ranges(audio)
        ]


class StreamingVAD:
    """Incremental VAD for 10/20/30-ms PCM or float chunks."""

    def __init__(self, detector: WebRTCVAD | None = None) -> None:
        self.detector = detector or WebRTCVAD()
        self._audio = np.empty(0, dtype=np.float32)
        self._processed_frames = 0
        self._active_start: int | None = None
        self._last_voice_frame = -1

    def reset(self) -> None:
        self._audio = np.empty(0, dtype=np.float32)
        self._processed_frames = 0
        self._active_start = None
        self._last_voice_frame = -1

    def push(self, audio: np.ndarray, final: bool = False) -> list[SpeechSegment]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio):
            self._audio = np.concatenate((self._audio, audio))
        frame = self.detector.frame_samples
        complete = len(self._audio) // frame
        emitted: list[SpeechSegment] = []
        for frame_index in range(self._processed_frames, complete):
            start = frame_index * frame
            pcm = (np.clip(self._audio[start : start + frame], -1.0, 1.0) * 32767).astype("<i2")
            voiced = self.detector._vad.is_speech(pcm.tobytes(), TARGET_SAMPLE_RATE)
            if voiced:
                if self._active_start is None:
                    self._active_start = max(0, start - self.detector.padding_frames * frame)
                self._last_voice_frame = frame_index
            elif self._active_start is not None and frame_index - self._last_voice_frame > self.detector.max_silence_frames:
                end = min(len(self._audio), (frame_index + 1) * frame)
                emitted.append(SpeechSegment(self._active_start, end, self._audio[self._active_start:end].copy()))
                self._active_start = None
        self._processed_frames = complete
        if final:
            if self._active_start is not None:
                end = len(self._audio)
                emitted.append(SpeechSegment(self._active_start, end, self._audio[self._active_start:end].copy()))
            self.reset()
        return emitted


class WhisperFastVAD:
    """Faster-Whisper tiny.en with a reusable VAD-gated transcription API."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 1,
        num_workers: int = 1,
        vad: WebRTCVAD | None = None,
        max_window_seconds: float = 30.0,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_DIR
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing Faster-Whisper model directory: {self.model_path}")
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = max(1, int(cpu_threads))
        self.num_workers = max(1, int(num_workers))
        self.max_window_samples = max(1, int(max_window_seconds * TARGET_SAMPLE_RATE))
        self.vad = vad or WebRTCVAD()
        self.model = WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
        )

    def _windows(self, segment: SpeechSegment) -> Iterator[SpeechSegment]:
        for start in range(0, len(segment.audio), self.max_window_samples):
            audio = segment.audio[start : start + self.max_window_samples]
            if len(audio):
                yield SpeechSegment(segment.start_sample + start, segment.start_sample + start + len(audio), audio)

    def _decode(self, audio: np.ndarray, language: str | None, with_timestamps: bool) -> tuple[str, list[dict[str, object]]]:
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=not with_timestamps,
            word_timestamps=False,
        )
        rows: list[dict[str, object]] = []
        text_parts: list[str] = []
        for item in segments:
            text = item.text.strip()
            if text:
                text_parts.append(text)
            rows.append(
                {
                    "start": float(item.start),
                    "end": float(item.end),
                    "text": text,
                    "avg_logprob": float(getattr(item, "avg_logprob", 0.0)),
                    "no_speech_prob": float(getattr(item, "no_speech_prob", 0.0)),
                }
            )
        return " ".join(text_parts).strip(), rows

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = "en",
        return_result: bool = False,
        with_timestamps: bool = False,
    ) -> str | TranscriptionResult:
        started = time.perf_counter()
        normalized = resample_audio(audio, sample_rate)
        selected = self.vad.segment(normalized)
        text_parts: list[str] = []
        rows: list[dict[str, object]] = []
        speech_seconds = 0.0
        for selected_segment in selected:
            speech_seconds += selected_segment.duration_seconds
            for window in self._windows(selected_segment):
                text, decoded_rows = self._decode(window.audio, language, with_timestamps)
                if text:
                    text_parts.append(text)
                for row in decoded_rows:
                    row["start"] = float(row["start"]) + window.start_sample / TARGET_SAMPLE_RATE
                    row["end"] = float(row["end"]) + window.start_sample / TARGET_SAMPLE_RATE
                    rows.append(row)
        result = TranscriptionResult(
            text=" ".join(text_parts).strip(),
            segments=tuple(rows),
            language=language or "auto",
            audio_seconds=len(normalized) / TARGET_SAMPLE_RATE,
            speech_seconds=speech_seconds,
            vad_segments=len(selected),
            elapsed_seconds=time.perf_counter() - started,
        )
        return result if return_result else result.text

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str | None = "en",
        return_result: bool = False,
        with_timestamps: bool = False,
    ) -> str | TranscriptionResult:
        audio = load_audio(audio_path)
        return self.transcribe_array(
            audio,
            TARGET_SAMPLE_RATE,
            language=language,
            return_result=return_result,
            with_timestamps=with_timestamps,
        )

    def stream_transcribe(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int = TARGET_SAMPLE_RATE,
        language: str | None = "en",
    ) -> Iterator[TranscriptionResult]:
        """Yield a result whenever streaming VAD closes a speech segment."""
        detector = StreamingVAD(self.vad)
        for chunk in chunks:
            normalized = resample_audio(chunk, sample_rate)
            for speech in detector.push(normalized):
                started = time.perf_counter()
                text, rows = self._decode(speech.audio, language, False)
                yield TranscriptionResult(
                    text=text,
                    segments=tuple(rows),
                    language=language or "auto",
                    audio_seconds=speech.duration_seconds,
                    speech_seconds=speech.duration_seconds,
                    vad_segments=1,
                    elapsed_seconds=time.perf_counter() - started,
                )
        for speech in detector.push(np.empty(0, dtype=np.float32), final=True):
            started = time.perf_counter()
            text, rows = self._decode(speech.audio, language, False)
            yield TranscriptionResult(
                text=text,
                segments=tuple(rows),
                language=language or "auto",
                audio_seconds=speech.duration_seconds,
                speech_seconds=speech.duration_seconds,
                vad_segments=1,
                elapsed_seconds=time.perf_counter() - started,
            )

    def close(self) -> None:
        self.model = None

    def __repr__(self) -> str:
        return (
            f"WhisperFastVAD(model={self.model_path.name!r}, device={self.device!r}, "
            f"compute_type={self.compute_type!r}, threads={self.cpu_threads}, "
            f"sample_rate={TARGET_SAMPLE_RATE})"
        )

