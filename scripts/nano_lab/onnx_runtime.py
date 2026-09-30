"""Small ONNX Runtime wrapper for the exported Chatterbox-Nano T3 stage.

This module intentionally has no dependency on the full Nano model.  It loads
the two ONNX graphs produced by :mod:`onnx_export`, keeps the KV cache in NumPy
arrays, and performs sampling in Python.  The caller supplies the T3 reference
speaker embedding, 375 reference speech tokens, and already-tokenized text.

The output is speech-token IDs.  Feed those IDs to the existing PyTorch S3Gen
runtime until the acoustic decoder and HiFT vocoder have their own numerical
parity-checked exports.  Any final audio should still pass through
``NanoEngine``'s Perth watermark path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "nano_lab" / "onnx"
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SPEECH_VOCAB = 6563


def _top_k_top_p(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """Return filtered logits without changing the input array."""

    result = logits.astype(np.float32, copy=True)
    if top_k > 0 and top_k < result.shape[-1]:
        threshold = np.partition(result, -top_k)[-top_k]
        result[result < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(result)[::-1]
        values = result[order]
        finite = np.isfinite(values)
        if finite.any():
            safe = np.where(finite, values, -1e30)
            safe = safe - np.max(safe)
            probs = np.exp(safe)
            probs[~finite] = 0.0
            probs /= max(float(probs.sum()), 1e-12)
            cumulative = np.cumsum(probs)
            remove = cumulative > float(top_p)
            # Keep the first token that crosses the threshold.  This is the
            # same right-shift used by Transformers' TopPLogitsWarper.
            remove[1:] = remove[:-1].copy()
            remove[0] = False
            result[order[remove]] = -np.inf
    return result


def sampling_selfcheck() -> dict[str, Any]:
    """Check the top-p boundary rule with a tiny deterministic distribution."""

    logits = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
    filtered = _top_k_top_p(logits, top_k=0, top_p=0.7)
    kept = np.flatnonzero(np.isfinite(filtered)).tolist()
    expected = [0, 1]
    if kept != expected:
        raise AssertionError(f"top-p selfcheck expected {expected}, got {kept}")
    return {"status": "passed", "kept_indices": kept, "top_p": 0.7}


def _sample(
    logits: np.ndarray,
    generated: Sequence[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rng: np.random.Generator,
) -> int:
    values = logits.astype(np.float32, copy=True)
    # Match inference_turbo's processor order. Repetition penalties come
    # after temperature/top-k/top-p; moving them first changes the support.
    if temperature > 0 and temperature != 1.0:
        values = values / float(temperature)
    values = _top_k_top_p(values, int(top_k), float(top_p))
    if repetition_penalty != 1.0:
        for token in set(int(value) for value in generated):
            if token < 0 or token >= values.shape[-1]:
                continue
            values[token] = values[token] * repetition_penalty if values[token] < 0 else values[token] / repetition_penalty
    finite = np.isfinite(values)
    if not finite.any():
        return int(np.argmax(logits))
    shifted = values - np.max(values[finite])
    probs = np.exp(np.where(finite, shifted, -np.inf))
    probs[~finite] = 0.0
    probs /= max(float(probs.sum()), 1e-12)
    return int(rng.choice(values.shape[-1], p=probs))


@dataclass
class OnnxGeneration:
    """Speech-token result from :class:`NanoT3OnnxRuntime`."""

    speech_tokens: np.ndarray
    elapsed_seconds: float
    prefill_seconds: float
    decode_seconds: float
    provider: str


class NanoT3OnnxRuntime:
    """CPU ONNX Runtime for Nano's exact T3 prefill/decode graphs."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
        providers: Sequence[str] | None = None,
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir).expanduser().resolve()
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Nano ONNX manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("status") != "partial_t3_exported":
            raise RuntimeError(f"Unsupported Nano ONNX manifest status: {self.manifest.get('status')!r}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(intra_op_num_threads))
        options.inter_op_num_threads = max(1, int(inter_op_num_threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        requested = list(providers or ["CPUExecutionProvider"])
        self.prefill_session = ort.InferenceSession(
            str(self.model_dir / "nano_t3_prefill.onnx"), options, providers=requested
        )
        self.decode_session = ort.InferenceSession(
            str(self.model_dir / "nano_t3_decode.onnx"), options, providers=requested
        )
        self.providers = tuple(self.prefill_session.get_providers())
        if self.providers != tuple(self.decode_session.get_providers()):
            raise RuntimeError("Prefill and decode ONNX providers differ")

    @staticmethod
    def _validate_inputs(
        speaker_emb: np.ndarray,
        cond_prompt_speech_tokens: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        speaker = np.asarray(speaker_emb, dtype=np.float32)
        prompt = np.asarray(cond_prompt_speech_tokens, dtype=np.int64)
        text = np.asarray(text_tokens, dtype=np.int64)
        if speaker.ndim == 1:
            speaker = speaker[None, :]
        if prompt.ndim == 1:
            prompt = prompt[None, :]
        if text.ndim == 1:
            text = text[None, :]
        if speaker.shape != (1, 256):
            raise ValueError(f"speaker_emb must have shape (1,256), got {speaker.shape}")
        if prompt.shape[0] != 1 or prompt.shape[1] < 1:
            raise ValueError(f"cond_prompt_speech_tokens must have shape (1,N), got {prompt.shape}")
        if text.shape[0] != 1 or text.shape[1] < 1:
            raise ValueError(f"text_tokens must have shape (1,N), got {text.shape}")
        return speaker, prompt, text

    def prefill(
        self,
        speaker_emb: np.ndarray,
        cond_prompt_speech_tokens: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
        speaker, prompt, text = self._validate_inputs(speaker_emb, cond_prompt_speech_tokens, text_tokens)
        started = time.perf_counter()
        outputs = self.prefill_session.run(
            None,
            {
                "speaker_emb": speaker,
                "cond_prompt_speech_tokens": prompt,
                "text_tokens": text,
            },
        )
        elapsed = time.perf_counter() - started
        logits = np.asarray(outputs[0])[0]
        cache = tuple(np.asarray(value) for value in outputs[1:])
        if len(cache) != 24:
            raise RuntimeError(f"Expected 24 prefill cache tensors, got {len(cache)}")
        return logits, cache, elapsed

    def decode(
        self,
        speech_token: int,
        cache: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
        if len(cache) != 24:
            raise ValueError(f"Expected 24 cache tensors, got {len(cache)}")
        started = time.perf_counter()
        outputs = self.decode_session.run(
            None,
            {
                "speech_token": np.asarray([[int(speech_token)]], dtype=np.int64),
                "cache_position": np.asarray([cache[0].shape[2]], dtype=np.int64),
            }
            | {f"past_{index}": value for index, value in enumerate(cache)},
        )
        elapsed = time.perf_counter() - started
        return np.asarray(outputs[0])[0], tuple(np.asarray(value) for value in outputs[1:]), elapsed

    def generate(
        self,
        speaker_emb: np.ndarray,
        cond_prompt_speech_tokens: np.ndarray,
        text_tokens: np.ndarray,
        *,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        seed: int | None = 0,
        stop_on_eos: bool = True,
    ) -> OnnxGeneration:
        """Generate Nano speech tokens while keeping the KV cache in NumPy."""

        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        logits, cache, prefill_seconds = self.prefill(speaker_emb, cond_prompt_speech_tokens, text_tokens)
        generated: list[int] = []
        decode_seconds = 0.0
        for _ in range(max(1, int(max_new_tokens))):
            token = _sample(
                logits,
                generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                rng=rng,
            )
            if token == STOP_SPEECH_TOKEN and stop_on_eos:
                break
            generated.append(token)
            logits, cache, elapsed = self.decode(token, cache)
            decode_seconds += elapsed
        return OnnxGeneration(
            speech_tokens=np.asarray(generated, dtype=np.int64),
            elapsed_seconds=time.perf_counter() - started,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            provider=self.providers[0] if self.providers else "unknown",
        )

    def benchmark(
        self,
        speaker_emb: np.ndarray,
        cond_prompt_speech_tokens: np.ndarray,
        text_tokens: np.ndarray,
        *,
        tokens: int = 16,
        seed: int = 0,
    ) -> dict[str, Any]:
        result = self.generate(
            speaker_emb,
            cond_prompt_speech_tokens,
            text_tokens,
            max_new_tokens=tokens,
            temperature=0.0,
            seed=seed,
        )
        seconds_per_token = result.decode_seconds / max(1, len(result.speech_tokens))
        return {
            "tokens": int(len(result.speech_tokens)),
            "elapsed_seconds": result.elapsed_seconds,
            "prefill_seconds": result.prefill_seconds,
            "decode_seconds": result.decode_seconds,
            "decode_ms_per_token": seconds_per_token * 1000.0,
            "provider": result.provider,
            "rss_note": "Measure parent process RSS externally; ORT session load is intentionally isolated from PyTorch Nano.",
        }


__all__ = ["NanoT3OnnxRuntime", "OnnxGeneration", "sampling_selfcheck"]
