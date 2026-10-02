"""Low-resident-memory runtime for the Chatterbox Nano checkpoint.

This module provides a small wrapper around the same ``T3`` and ``S3Gen`` classes used by
``ChatterboxTurboTTS`` while adding two measured loading paths:

* ``optimized=True`` constructs modules on the meta device and streams each
  safetensors tensor into its final device.  This avoids a complete CPU state
  dictionary and the second device copy used by the stock loader.
* ``optimized=False`` follows the stock CPU state-dictionary loader.  This is
  useful as a matched control in memory and quality experiments.

The wrapper also drops the unused T3 GPT-2 token embedding and text head after
loading, caches reference conditionals, and can release the two encoders used
only while preparing a reference.  The generated waveform still receives the
upstream Perth watermark.

The public entry point is :class:`NanoEngine`.  ``engine.model`` is the
upstream-compatible model object, so callers can inspect ``model.conds`` and
the individual T3/S3Gen modules when running ablations.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

# Keep the runtime conservative on shared CPU hosts.  These are defaults only;
# a caller may set the variables before importing this module.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder import VoiceEncoder
from s3tokenizer.model_v2 import precompute_freqs_cis
from chatterbox.tts_turbo import (
    ChatterboxTurboTTS,
    Conditionals,
    NANO_REPO_ID,
    punc_norm,
)
from chatterbox.models.s3gen.const import S3GEN_SIL

try:
    import perth
except Exception:  # pragma: no cover - surfaced when construction is attempted
    perth = None

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "chatterbox-nano"
DEFAULT_REFERENCE = ROOT / "artifacts" / "references" / "chunks" / "asmr7_chunk_36-46.wav"


def _set_cpu_threads(limit: int = 4) -> None:
    """Apply the lab's CPU limit without changing a caller's explicit limit."""

    limit = max(1, min(int(limit), 4))
    try:
        torch.set_num_threads(limit)
    except RuntimeError:
        # PyTorch does not allow changing the thread count after parallel work
        # starts.  The environment variables above still protect subprocesses.
        pass


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for NanoEngine, but torch.cuda.is_available() is false")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for NanoEngine, but MPS is unavailable")
    return resolved


def _resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if dtype is None or dtype == "auto":
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    value = str(dtype).lower().replace("-", "")
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dtype {dtype!r}; use auto, fp32, fp16, or bf16")
    resolved = aliases[value]
    if device.type == "cpu" and resolved is not torch.float32:
        LOGGER.warning("CPU Nano inference with %s is experimental; fp32 is the supported CPU dtype", resolved)
    return resolved


def _nano_hp() -> T3Config:
    """Return the exact Nano/Turbo-compatible T3 configuration."""

    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_small"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    return hp


def _module_state_targets(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Map state-dict names to live parameter/buffer tensors."""

    # ``named_buffers`` also includes non-persistent runtime buffers (for
    # example GPT-2's causal attention mask).  ``state_dict`` is the authority
    # for what a safetensors checkpoint is expected to contain.
    persistent_names = set(module.state_dict().keys())
    targets: dict[str, torch.Tensor] = {}
    targets.update({name: value for name, value in module.named_parameters() if name in persistent_names})
    targets.update({name: value for name, value in module.named_buffers() if name in persistent_names})
    return targets


def _stream_load_safetensors(
    module: torch.nn.Module,
    checkpoint: Path,
    *,
    device: torch.device,
    strict: bool = True,
    skip_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Load a safetensors file one tensor at a time.

    The default official mmap reader keeps tensors on the destination device.
    ``NANO_SAFETENSORS_BACKEND=streamed`` selects a CPU reader without a
    whole-file mapping. Only one checkpoint tensor is materialised at a time,
    in contrast to ``load_file`` which creates a full state dictionary.
    """

    targets = _module_state_targets(module)
    loaded: list[str] = []
    skipped: list[str] = []
    unexpected: list[str] = []
    source_device = str(device)
    # A whole-checkpoint reservation can exceed a strict address-space cap.
    # Keep the official mmap default; explicitly select streamed CPU reads
    # where mapping the unused checkpoint tensors cannot fit.
    file_backend = os.environ.get('NANO_SAFETENSORS_BACKEND', 'mmap')
    if file_backend not in ('mmap', 'pread', 'streamed'):
        raise ValueError('NANO_SAFETENSORS_BACKEND must be mmap, pread, or streamed')
    if file_backend in ('pread', 'streamed') and device.type != 'cpu':
        raise ValueError('The lab streamed checkpoint paths are CPU-only')
    if file_backend == 'streamed':
        from checkpoint_reader import CheckpointReader
        reader = CheckpointReader(checkpoint)
    else:
        reader = safe_open(str(checkpoint), framework="pt", device=source_device, backend=file_backend)
    with reader as handle:
        for name in handle.keys():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes):
                skipped.append(name)
                continue
            target = targets.get(name)
            if target is None:
                unexpected.append(name)
                continue
            source = handle.get_tensor(name)
            if source.shape != target.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: checkpoint {tuple(source.shape)} vs module {tuple(target.shape)}"
                )
            # ``copy_`` performs the only conversion required for fp16/bf16
            # optimized loads.  ``source`` is freed at the next loop turn.
            with torch.no_grad():
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded.append(name)
            # Release the prior buffer before allocating the next tensor.
            del source

    def is_skipped(name: str) -> bool:
        return any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes)

    missing = sorted(set(targets) - set(loaded))
    # A small number of deterministic tokenizer/runtime buffers are registered
    # in the module state dict but intentionally omitted from the checkpoint.
    # Upstream's ``load_state_dict`` handles these through
    # ``ignore_state_dict_missing``; the tensor-at-a-time loader must apply the
    # same contract before repairing them below.
    ignored_missing = {"trim_fade"}
    for module_name, child in module.named_modules():
        for ignored_name in getattr(child, "ignore_state_dict_missing", ()):
            full_name = f"{module_name}.{ignored_name}" if module_name else str(ignored_name)
            ignored_missing.add(full_name)
    missing_required = [
        name for name in missing if name not in ignored_missing and not is_skipped(name)
    ]
    if strict and (missing_required or unexpected):
        details = []
        if missing_required:
            details.append("missing=" + ", ".join(missing_required[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        raise RuntimeError(f"Could not stream {checkpoint.name}: " + "; ".join(details))
    return {
        "checkpoint_io_backend": file_backend,
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "unexpected_tensors": unexpected,
        "missing_tensors": missing_required,
        "ignored_missing_tensors": [name for name in missing if name in ignored_missing and not is_skipped(name)],
    }


def _load_state_dict_stock(module: torch.nn.Module, checkpoint: Path, *, device: torch.device, dtype: torch.dtype,
                           strict: bool = True) -> dict[str, Any]:
    """Stock control path: materialise the full CPU state dictionary first."""

    state = load_file(str(checkpoint), device="cpu", backend="mmap")
    try:
        incompat = module.load_state_dict(state, strict=strict)
        return {
            "loaded_tensors": len(state),
            "missing_tensors": list(incompat.missing_keys),
            "unexpected_tensors": list(incompat.unexpected_keys),
        }
    finally:
        del state


def _repair_runtime_buffers(s3gen: torch.nn.Module, device: torch.device, dtype: torch.dtype, missing_buffers: Sequence[str] = ()) -> None:
    """Recreate non-persistent buffers after a meta-device construction."""

    n_trim = S3GEN_SR // 50
    trim_fade = torch.zeros(2 * n_trim, device=device, dtype=dtype)
    trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim, device=device, dtype=dtype)) + 1) / 2
    # Keep the registered-buffer name used by upstream S3Token2Wav.
    s3gen.trim_fade = trim_fade

    # S3Gen's ESPnet relative positional encoding keeps ``pe`` as a plain
    # tensor rather than a registered buffer.  A meta-device constructor leaves
    # that tensor on ``meta`` forever unless it is rebuilt explicitly.
    for module_name, module in s3gen.named_modules():
        # S3TokenizerV2 stores the rotary frequencies as a plain tensor.  A
        # meta-device constructor leaves it on ``meta`` and it is not part of
        # the safetensors state dict.  Rebuild the same deterministic table
        # used by the upstream constructor before reference conditioning.
        freqs_cis = getattr(module, "freqs_cis", None)
        if torch.is_tensor(freqs_cis) and freqs_cis.is_meta:
            module.freqs_cis = precompute_freqs_cis(
                int(freqs_cis.shape[-1]), int(freqs_cis.shape[0])
            ).to(device=device)

        # S3Tokenizer's mel filter bank and Hann window are deterministic
        # buffers omitted by the checkpoint.  ``to_empty`` materialises a
        # random window from the meta tensor, so regenerate it explicitly.
        mel_filters = getattr(module, "_mel_filters", None)
        if torch.is_tensor(mel_filters) and (mel_filters.is_meta or f"{module_name}._mel_filters" in missing_buffers):
            n_mels, n_fft_plus_one = map(int, mel_filters.shape)
            n_fft = (n_fft_plus_one - 1) * 2
            module._mel_filters = torch.from_numpy(
                librosa.filters.mel(sr=S3_SR, n_fft=n_fft, n_mels=n_mels)
            ).to(device=device, dtype=torch.float32)
        window = getattr(module, "window", None)
        if torch.is_tensor(window) and (window.is_meta or f"{module_name}.window" in missing_buffers):
            module.window = torch.hann_window(int(window.shape[0]), device="cpu", dtype=torch.float32).to(device=device,dtype=window.dtype)

        if module.__class__.__name__ != "EspnetRelPositionalEncoding":
            continue
        old_pe = getattr(module, "pe", None)
        if torch.is_tensor(old_pe):
            # Constructor shape is (1, 2 * max_len - 1, d_model).
            max_len = (int(old_pe.shape[1]) + 1) // 2
        else:
            max_len = 5000
        module.pe = None
        module.extend_pe(torch.zeros(1, max_len, device=device, dtype=dtype))


def _defer_t3_causal_masks(t3: torch.nn.Module) -> None:
    """Avoid allocating one large deterministic mask per layer on to_empty."""
    for module in t3.modules():
        bias = getattr(module, '_buffers', {}).get('bias')
        if torch.is_tensor(bias) and bias.ndim == 4 and bias.dtype == torch.bool:
            module._nano_causal_mask_size = int(bias.shape[-1])
            module.bias = None


def _repair_t3_runtime_buffers(t3: torch.nn.Module, device: torch.device) -> None:
    """Restore shared read-only GPT-2 causal masks omitted by checkpoints.

    GPT-2 only slices these masks during attention. All layers of the same
    size can share the exact same values instead of holding separate copies.
    """

    causal_masks = {}
    for module in t3.modules():
        buffers = getattr(module, "_buffers", {})
        bias = buffers.get("bias")
        deferred_size = getattr(module, '_nano_causal_mask_size', None)
        if deferred_size is not None or (torch.is_tensor(bias) and bias.ndim == 4 and bias.dtype == torch.bool):
            size = deferred_size if deferred_size is not None else int(bias.shape[-1])
            if size not in causal_masks:
                causal_masks[size] = torch.tril(torch.ones((size, size), dtype=torch.bool, device=device)).view(1, 1, size, size)
            module.bias = causal_masks[size]
            if deferred_size is not None:
                del module._nano_causal_mask_size
        masked_bias = buffers.get("masked_bias")
        if torch.is_tensor(masked_bias) and masked_bias.ndim == 0:
            module.masked_bias = torch.tensor(-1e4, dtype=torch.float32, device=device)


def _delete_unused_t3_weights(t3: T3) -> list[str]:
    """Drop parameters never touched by ``inference_turbo``.

    ``tfmr.wte`` is the GPT-2 token embedding.  Nano supplies custom text and
    speech embeddings, so the GPT-2 table is dead at inference.  ``text_head``
    is a training/evaluation head and is also dead in the turbo loop.
    """

    removed: list[str] = []
    tfmr = getattr(t3, "tfmr", None)
    if tfmr is not None and hasattr(tfmr, "wte"):
        del tfmr.wte
        removed.append("tfmr.wte")
    if hasattr(t3, "text_head"):
        del t3.text_head
        removed.append("text_head")
    return removed


class _DeviceSentinel(torch.nn.Module):
    """Tiny module preserving ``S3Gen.device`` after tokenizer release."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.register_parameter(
            "device_parameter",
            torch.nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False),
        )


def _set_eval(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    return module


def _prepare_model(
    ckpt_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    optimized: bool,
    decoder: str,
    conditionals_path: Path | None = None,
) -> tuple[ChatterboxTurboTTS, dict[str, Any]]:
    """Construct the upstream model and return load diagnostics."""

    decoder = decoder.lower()
    if decoder not in {"meanflow", "original"}:
        raise ValueError("decoder must be 'meanflow' or 'original'")

    hp = _nano_hp()
    load_report: dict[str, Any] = {
        "optimized": bool(optimized),
        "device": str(device),
        "dtype": str(dtype),
        "decoder": decoder,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if conditionals_path is not None and not optimized:
        raise ValueError("conditionals_path requires optimized=True so reference tensors can be skipped while loading")

    load_reference_encoders = conditionals_path is None
    # Keep the VE in fp32.  It is normally used once and released after the
    # reference conditionals are prepared; fp32 avoids backend-specific LSTM
    # half-precision failures during that short preparation step.  A frozen
    # conditionals file does not need either reference encoder at all.
    if load_reference_encoders:
        if optimized:
            with torch.device("meta"):
                ve = VoiceEncoder()
            ve.to_empty(device=device)
            ve_report = _stream_load_safetensors(ve, ckpt_dir / "ve.safetensors", device=device, strict=True)
        else:
            ve = VoiceEncoder()
            ve_report = _load_state_dict_stock(ve, ckpt_dir / "ve.safetensors", device=device, dtype=torch.float32)
            ve.to(device=device, dtype=torch.float32)
        ve_report["dtype"] = str(torch.float32)
        _set_eval(ve)
    else:
        ve = None
        ve_report = {"skipped": True, "reason": "conditionals_path supplied"}
    load_report["ve"] = ve_report

    if optimized:
        with torch.device("meta"):
            t3 = T3(hp)
        # Remove dead modules before allocating destination storage.  The
        # skipped checkpoint keys are explicitly permitted below.
        removed = _delete_unused_t3_weights(t3)
        _defer_t3_causal_masks(t3)
        t3.to(dtype=dtype)
        t3.to_empty(device=device)
        t3_report = _stream_load_safetensors(
            t3,
            ckpt_dir / "t3_nano_v1.safetensors",
            device=device,
            strict=True,
            skip_prefixes=("tfmr.wte", "text_head"),
        )
        _repair_t3_runtime_buffers(t3, device)
    else:
        t3 = T3(hp)
        t3_report = _load_state_dict_stock(t3, ckpt_dir / "t3_nano_v1.safetensors", device=device, dtype=dtype)
        removed = _delete_unused_t3_weights(t3)
        t3.to(device=device, dtype=dtype)
    t3_report["removed_parameters"] = removed
    _set_eval(t3)
    load_report["t3"] = t3_report

    meanflow = decoder == "meanflow"
    # HiFT deliberately creates float32 harmonic signals. More importantly,
    # upstream S3Token2Mel casts reference token IDs through its model dtype;
    # fp16 cannot exactly represent every speech ID up to 6560. Keep the
    # complete acoustic stage in fp32 and apply reduced precision only to T3.
    acoustic_dtype = torch.float32
    load_report["precision_policy"] = {"t3":str(dtype),"s3gen":str(acoustic_dtype)}
    s3_ckpt = ckpt_dir / ("s3gen_meanflow.safetensors" if meanflow else "s3gen.safetensors")
    if optimized:
        with torch.device("meta"):
            s3gen = S3Gen(meanflow=meanflow)
        if conditionals_path is not None:
            # Remove reference-only modules before allocating storage, rather
            # than merely skipping their weights after to_empty.
            s3gen.speaker_encoder = None
            s3gen.tokenizer = None
        s3gen.to(dtype=acoustic_dtype)
        s3gen.to_empty(device=device)
        s3_report = _stream_load_safetensors(
            s3gen,
            s3_ckpt,
            device=device,
            strict=True,
            skip_prefixes=("speaker_encoder", "tokenizer") if conditionals_path is not None else (),
        )
        _repair_runtime_buffers(s3gen, device, acoustic_dtype, s3_report["ignored_missing_tensors"])
        if conditionals_path is not None:
            # A frozen conditionals file already contains both the T3 speaker
            # vector and the S3Gen x-vector.  Remove the encoders and keep a
            # zero-sized device sentinel for ``S3Gen.device``.
            s3gen.speaker_encoder = None
            s3gen.tokenizer = _DeviceSentinel(device, acoustic_dtype)
    else:
        s3gen = S3Gen(meanflow=meanflow)
        s3_report = _load_state_dict_stock(s3gen, s3_ckpt, device=device, dtype=acoustic_dtype, strict=True)
        s3gen.to(device=device, dtype=acoustic_dtype)
    _set_eval(s3gen)
    load_report["s3gen"] = s3_report

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != 50276:
        LOGGER.warning("Nano tokenizer length is %s, expected 50276", len(tokenizer))

    # Built-in conditions are optional and are loaded to the same device as the
    # model.  We cast T3 conditioning tensors later when fp16/bf16 is selected.
    conds = None
    if conditionals_path is not None:
        conditionals_path = conditionals_path.expanduser().resolve()
        if not conditionals_path.exists():
            raise FileNotFoundError(f"Conditionals file does not exist: {conditionals_path}")
        conds = Conditionals.load(conditionals_path, map_location="cpu").to(device)
        load_report["conditionals_path"] = str(conditionals_path)
    else:
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location="cpu").to(device)

    model = ChatterboxTurboTTS(
        t3=t3,
        s3gen=s3gen,
        ve=ve,
        tokenizer=tokenizer,
        device=str(device),
        conds=conds,
        model_label="Nano",
    )
    # Ensure Perth remains part of the output contract even when a local
    # environment imported the dependency lazily.
    if perth is None:
        raise RuntimeError("The perth package is required to preserve Chatterbox's audio watermark")
    load_report["parameters"] = {
        name: (sum(parameter.numel() for parameter in getattr(model, name).parameters()) if getattr(model, name) is not None else 0)
        for name in ("t3", "s3gen", "ve")
    }
    return model, load_report


def _cast_t3_conditionals(conds: Conditionals, dtype: torch.dtype, device: torch.device) -> Conditionals:
    """Move/cast T3 inputs to match a half/bfloat16 T3 module."""

    conds.to(device)
    t3 = conds.t3
    for name in ("speaker_emb", "cond_prompt_speech_tokens", "cond_prompt_speech_emb", "clap_emb", "emotion_adv"):
        value = getattr(t3, name, None)
        if torch.is_tensor(value):
            # Token ids must remain integral; all other T3 condition values are
            # activations and need the model dtype for matmul compatibility.
            target_dtype = value.dtype if not value.is_floating_point() else dtype
            setattr(t3, name, value.to(device=device, dtype=target_dtype))
    return conds


def _normalise_embedding(embedding: Any) -> torch.Tensor:
    """Convert a supplied speaker embedding or embedding list to (1, 256)."""

    if torch.is_tensor(embedding):
        result = embedding.detach().float().cpu()
    else:
        result = torch.as_tensor(np.asarray(embedding), dtype=torch.float32)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.ndim != 2 or result.shape[-1] != 256:
        raise ValueError(f"voice_embeddings must have shape (N,256), got {tuple(result.shape)}")
    result = result.mean(dim=0, keepdim=True)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _seed_everywhere(seed: int) -> None:
    """Reset the random sources consumed by T3 and S3Gen."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reference_cache_key(path: str | os.PathLike[str], exaggeration: float, norm_loudness: bool) -> tuple[Any, ...]:
    ref = Path(path).expanduser().resolve()
    try:
        stat = ref.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        fingerprint = None
    return (str(ref), fingerprint, float(exaggeration), bool(norm_loudness))


@dataclass
class GenerationResult:
    """Optional structured result used by benchmark callers."""

    audio: torch.Tensor
    speech_tokens: torch.Tensor
    elapsed_seconds: float


class GenerationLimitError(RuntimeError):
    """Autoregressive speech did not finish within the configured token budget."""


class NanoEngine:
    """Memory-aware wrapper around Chatterbox's Nano model.

    Parameters mirror the controls needed by the lab. The default is fp32.
    Explicit fp16/bf16 applies only to T3. The acoustic decoder stays fp32 to
    preserve reference token IDs and the vocoder's harmonic-source arithmetic.
    """

    def __init__(
        self,
        model: ChatterboxTurboTTS,
        *,
        device: torch.device,
        dtype: torch.dtype,
        optimized: bool,
        decoder: str,
        load_report: Mapping[str, Any] | None = None,
        cache_conditionals: bool = True,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.optimized = bool(optimized)
        self.decoder = decoder
        self.cache_conditionals = bool(cache_conditionals)
        self._conditional_cache: dict[tuple[Any, ...], Conditionals] = {}
        self.load_report = dict(load_report or {})
        self._release_voice_encoder_after_prepare = False
        self._release_decoder_encoder_after_prepare = False
        self._release_tokenizer_after_prepare = False
        self.load_report["engine"] = {
            "optimized": self.optimized,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "decoder": self.decoder,
        }

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
        *,
        device: str | torch.device = "cuda",
        optimized: bool = True,
        dtype: str | torch.dtype | None = "auto",
        decoder: str = "meanflow",
        unload_voice_encoder: bool = False,
        unload_decoder_encoder: bool = False,
        unload_tokenizer: bool = False,
        cache_conditionals: bool = True,
        conditionals_path: str | os.PathLike[str] | None = None,
        quantize: str | None = None,
        cpu_threads: int = 4,
    ) -> "NanoEngine":
        _set_cpu_threads(cpu_threads)
        target_device = _resolve_device(device)
        target_dtype = _resolve_dtype(dtype, target_device)
        ckpt = Path(ckpt_dir).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Nano checkpoint directory does not exist: {ckpt}")
        if quantize not in (None, "none", "dynamic-int8"):
            raise ValueError("quantize must be None, 'none', or 'dynamic-int8'")
        if quantize == "dynamic-int8" and target_device.type != "cpu":
            raise ValueError("dynamic-int8 is CPU-only; keep CUDA weights in fp16/bf16/fp32")

        started = time.perf_counter()
        model, report = _prepare_model(
            ckpt,
            device=target_device,
            dtype=target_dtype,
            optimized=optimized,
            decoder=decoder,
            conditionals_path=(Path(conditionals_path) if conditionals_path is not None else None),
        )
        report["load_seconds"] = time.perf_counter() - started
        engine = cls(
            model,
            device=target_device,
            dtype=target_dtype,
            optimized=optimized,
            decoder=decoder,
            load_report=report,
            cache_conditionals=cache_conditionals,
        )
        if quantize == "dynamic-int8":
            engine.quantize_dynamic_int8()
        if unload_tokenizer and conditionals_path is not None:
            engine.unload_tokenizer()
        # The reference encoders cannot be released until a reference has been
        # prepared.  Remember the requested policy for the next preparation.
        engine._release_voice_encoder_after_prepare = bool(unload_voice_encoder)
        engine._release_decoder_encoder_after_prepare = bool(unload_decoder_encoder)
        engine._release_tokenizer_after_prepare = bool(unload_tokenizer and conditionals_path is None)
        return engine

    # Compatibility alias for callers that naturally use ``from_local``.
    from_local = from_pretrained

    @property
    def sr(self) -> int:
        return int(self.model.sr)

    @property
    def conds(self) -> Conditionals | None:
        return self.model.conds

    @property
    def has_voice_encoder(self) -> bool:
        return getattr(self.model, "ve", None) is not None

    @property
    def has_decoder_reference_encoder(self) -> bool:
        return getattr(self.model.s3gen, "speaker_encoder", None) is not None

    def _cast_current_conditionals(self) -> None:
        if self.model.conds is not None:
            _cast_t3_conditionals(self.model.conds, self.dtype, self.device)

    def prepare_conditionals(
        self,
        wav_fpath: str | os.PathLike[str],
        *,
        exaggeration: float = 0.0,
        norm_loudness: bool = True,
        voice_embeddings: Any | None = None,
        force: bool = False,
    ) -> Conditionals:
        """Prepare and cache T3/S3Gen reference conditions.

        ``voice_embeddings`` can be one or more precomputed 256-dimensional
        voice-encoder vectors.  When supplied, the mean normalized vector
        replaces the T3 speaker embedding while the S3Gen decoder still uses
        the acoustic reference embedding computed from ``wav_fpath``.
        """

        if not self.has_voice_encoder:
            raise RuntimeError("The voice encoder was unloaded; prepare a reference before releasing it")
        if not self.has_decoder_reference_encoder:
            raise RuntimeError("The S3Gen reference encoder was unloaded; prepare a new engine")
        key = _reference_cache_key(wav_fpath, exaggeration, norm_loudness)
        cached = self._conditional_cache.get(key) if self.cache_conditionals and not force else None
        if cached is not None:
            self.model.conds = cached
            self._cast_current_conditionals()
        else:
            # Upstream's helper does not itself enter inference mode.  Keep
            # the cached conditionals free of autograd graphs; otherwise the
            # reference encoder's intermediate tensors remain reachable from
            # ``model.conds`` and inflate both RSS and CUDA peaks.
            with torch.inference_mode():
                self.model.prepare_conditionals(
                    str(wav_fpath),
                    exaggeration=float(exaggeration),
                    norm_loudness=bool(norm_loudness),
                )
            if self.model.conds is None:
                raise RuntimeError("Chatterbox did not produce conditionals")
            _cast_t3_conditionals(self.model.conds, self.dtype, self.device)
            if voice_embeddings is not None:
                self.model.conds.t3.speaker_emb = _normalise_embedding(voice_embeddings).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            if self.cache_conditionals:
                self._conditional_cache[key] = self.model.conds

        if self._release_voice_encoder_after_prepare:
            self.unload_voice_encoder()
        if self._release_decoder_encoder_after_prepare:
            self.unload_decoder_reference_encoder()
        if self._release_tokenizer_after_prepare:
            self.unload_tokenizer()
        return self.model.conds

    def set_conditionals(self, conds: Conditionals) -> None:
        """Install caller-supplied conditions, preserving dtype/device rules."""

        self.model.conds = conds
        self._cast_current_conditionals()

    def save_conditionals(self, path: str | os.PathLike[str]) -> Path:
        """Persist prepared reference conditions for encoderless loading."""

        if self.model.conds is None:
            raise RuntimeError("No conditionals are prepared")
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Save a CPU copy so the file is portable across CUDA/CPU runs and does
        # not pin a device allocation after the caller releases the engine.
        conds = self.model.conds
        cpu_t3 = T3Cond(
            speaker_emb=conds.t3.speaker_emb.detach().float().cpu() if torch.is_tensor(conds.t3.speaker_emb) else conds.t3.speaker_emb,
            clap_emb=conds.t3.clap_emb.detach().float().cpu() if torch.is_tensor(conds.t3.clap_emb) else conds.t3.clap_emb,
            cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens.detach().cpu()
            if torch.is_tensor(conds.t3.cond_prompt_speech_tokens)
            else conds.t3.cond_prompt_speech_tokens,
            cond_prompt_speech_emb=conds.t3.cond_prompt_speech_emb.detach().float().cpu()
            if torch.is_tensor(conds.t3.cond_prompt_speech_emb)
            else conds.t3.cond_prompt_speech_emb,
            emotion_adv=conds.t3.emotion_adv.detach().float().cpu() if torch.is_tensor(conds.t3.emotion_adv) else conds.t3.emotion_adv,
        )
        cpu_gen: dict[str, Any] = {}
        for key, value in conds.gen.items():
            if torch.is_tensor(value):
                cpu_gen[key] = value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
            else:
                cpu_gen[key] = value
        Conditionals(cpu_t3, cpu_gen).save(destination)
        return destination

    def clear_condition_cache(self) -> None:
        self._conditional_cache.clear()
        gc.collect()

    def unload_voice_encoder(self) -> None:
        """Release the T3 voice encoder after reference preparation."""

        if getattr(self.model, "ve", None) is not None:
            self.model.ve.to("cpu")
            self.model.ve = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def unload_decoder_reference_encoder(self) -> None:
        """Release S3Gen's CAMPPlus reference encoder after preparation."""

        encoder = getattr(self.model.s3gen, "speaker_encoder", None)
        if encoder is not None:
            encoder.to("cpu")
            self.model.s3gen.speaker_encoder = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def unload_tokenizer(self) -> None:
        """Release S3Tokenizer after conditions are prepared.

        A one-parameter sentinel preserves upstream ``S3Gen.device``.  Calls
        to ``prepare_conditionals`` after this point correctly fail with an
        explicit encoder/tokenizer error instead of an opaque ``next()`` error.
        """

        tokenizer = getattr(self.model.s3gen, "tokenizer", None)
        if tokenizer is not None and not isinstance(tokenizer, _DeviceSentinel):
            self.model.s3gen.tokenizer = _DeviceSentinel(self.device, self.dtype)
            del tokenizer
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def quantize_dynamic_int8(self) -> dict[str, Any]:
        """Apply supported CPU dynamic-int8 Linear layers.

        GPT-2's custom Conv1D blocks and S3Gen convolutions stay fp32.  The
        method reports the actual number of replaced modules so callers do not
        mistake partial dynamic quantization for a full int8 model.
        """

        if self.device.type != "cpu":
            raise RuntimeError("dynamic int8 quantization is CPU-only")
        try:
            from torch.ao.quantization import quantize_dynamic
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch dynamic quantization is unavailable") from exc
        before = sum(1 for module in self.model.modules() if isinstance(module, torch.nn.Linear))
        # quantize_dynamic recursively replaces supported Linear modules.  It
        # leaves embeddings, GPT2 Conv1D, convolutions, and LSTMs unchanged.
        self.model.t3 = quantize_dynamic(self.model.t3, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        self.model.s3gen = quantize_dynamic(self.model.s3gen, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        after = sum(1 for module in self.model.modules() if isinstance(module, torch.nn.quantized.dynamic.Linear))
        report = {"requested": "dynamic-int8", "linear_modules_before": before, "dynamic_int8_linear_modules": after}
        self.load_report["quantization"] = report
        return report

    def generate(
        self,
        text: str,
        *,
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        n_cfm_steps: int | None = None,
        audio_prompt_path: str | os.PathLike[str] | None = None,
        norm_loudness: bool = True,
        return_tokens: bool = False,
        return_result: bool = False,
        seed: int | None = None,
        acoustic_seed: int | None = None,
        max_gen_len: int = 700,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | GenerationResult:
        """Generate watermarked audio with a configurable S3Gen CFM step count.

        ``seed`` controls T3 speech-token sampling.  ``acoustic_seed`` is an
        optional second seed reset immediately before S3Gen inference, which
        makes acoustic comparisons independent of the number of random draws
        consumed by T3.  When it is omitted, the historical single-seed flow
        is preserved.
        """

        if audio_prompt_path is not None:
            self.prepare_conditionals(
                audio_prompt_path,
                exaggeration=exaggeration,
                norm_loudness=norm_loudness,
            )
        elif self.model.conds is None:
            raise AssertionError("Please prepare_conditionals first or provide audio_prompt_path")
        if seed is not None:
            _seed_everywhere(seed)

        self._cast_current_conditionals()
        if n_cfm_steps is None:
            n_cfm_steps = 2 if self.decoder == "meanflow" else 10
        n_cfm_steps = int(n_cfm_steps)
        if n_cfm_steps < 1:
            raise ValueError("n_cfm_steps must be >= 1")
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            LOGGER.warning("Nano/Turbo does not implement cfg_weight, exaggeration, or min_p in inference_turbo")

        text_tokens = self.model.tokenizer(
            punc_norm(text), return_tensors="pt", padding=True, truncation=False
        ).input_ids.to(self.device)
        if text_tokens.shape[-1]>350:
            raise ValueError("Text exceeds 350 tokens; split it into shorter sentences")
        if not 1<=int(max_gen_len)<=700:
            raise ValueError("max_gen_len must be between 1 and 700")
        started = time.perf_counter()
        with torch.inference_mode():
            speech_tokens = self.model.t3.inference_turbo(
                t3_cond=self.model.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_gen_len=int(max_gen_len),
            )
            if speech_tokens.numel()>=int(max_gen_len):
                raise GenerationLimitError("Speech generation reached its length limit; output was not decoded")
            speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
            if speech_tokens.numel()==0:
                raise RuntimeError("Speech generation returned no audio tokens")
            silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long, device=self.device)
            speech_tokens = torch.cat([speech_tokens, silence])
            if acoustic_seed is not None:
                _seed_everywhere(acoustic_seed)
            wav, _ = self.model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.model.conds.gen,
                n_cfm_timesteps=n_cfm_steps,
            )
            wav = wav.squeeze(0).detach().float().cpu().numpy()
            watermarked = self.model.watermarker.apply_watermark(wav, sample_rate=self.model.sr)
            audio = torch.from_numpy(np.asarray(watermarked)).unsqueeze(0)
        elapsed = time.perf_counter() - started
        if return_result:
            return GenerationResult(audio=audio, speech_tokens=speech_tokens.detach().cpu(), elapsed_seconds=elapsed)
        if return_tokens:
            return audio, speech_tokens.detach().cpu()
        return audio

    def memory_snapshot(self) -> dict[str, float | int | None]:
        """Return process RSS and CUDA allocator counters for reports."""

        import resource
        import psutil

        process = psutil.Process()
        result: dict[str, float | int | None] = {
            "rss_mib": process.memory_info().rss / 2**20,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_allocated_mib": None,
            "cuda_reserved_mib": None,
            "cuda_peak_allocated_mib": None,
            "cuda_peak_reserved_mib": None,
        }
        if torch.cuda.is_available():
            result.update(
                cuda_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        return result

    def unload(self) -> None:
        """Release model references and allocator blocks in a long-lived host."""

        self.clear_condition_cache()
        self.model.conds = None
        for name in ("ve", "t3", "s3gen", "tokenizer"):
            if hasattr(self.model, name):
                setattr(self.model, name, None)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def onnx_feasibility() -> dict[str, Any]:
    """Explain why this runtime does not claim an ONNX conversion.

    Exporting the complete pipeline requires a custom autoregressive T3 cache,
    dynamic token length, S3Gen flow loops, and the HiFT vocoder.  A successful
    ``torch.onnx.export`` of one submodule would not be a usable Nano runtime,
    so the benchmark records this as an explicit future task.
    """

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        available = True
    except Exception:
        available = False
    return {
        "onnx_packages_available": available,
        "status": "not_attempted",
        "reason": (
            "Complete Nano export is not a drop-in optimization: T3 uses a Python sampling loop with KV cache, "
            "S3Gen uses dynamic-length flow matching, and HiFT performs a separate vocoder pass. "
            "No partial export is represented as an end-to-end ONNX claim."
        ),
    }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_REFERENCE",
    "GenerationResult",
    "GenerationLimitError",
    "NanoEngine",
    "onnx_feasibility",
]
