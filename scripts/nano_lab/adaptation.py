"""Optional per-voice Nano T3 adapter experiment.

This module deliberately trains only a small LoRA adapter on the Nano GPT-2
backbone.  It does not fine-tune the speech tokenizer, S3Gen vocoder, or voice
encoder.  The training objective follows ``T3.inference_turbo`` exactly:

``conditioning + text + <speech-bos> -> speech[0]``
``conditioning + text + <speech-bos> + speech[0] -> speech[1]``

The stock ``T3.loss`` method is not used.  It feeds each speech target at the
same position whose logits are scored, which leaks the target through its
embedding for the Turbo GPT-2 model.  This script uses shifted teacher forcing
and therefore keeps prompt/target accounting explicit.

The script is intentionally conservative.  It defaults to CPU and two torch
threads, requires an explicit validation split, rejects the same audio file in
multiple target splits, and records all parameter counts and configuration in
the checkpoint metadata.  GPU execution is opt-in with ``--device cuda``.

Manifest rows are JSON objects with at least ``audio_path``, ``text``, and
``split``.  ``reference_audio_path`` (or ``conditioning_audio_path``) is
recommended for every target row.  If it is omitted, a distinct row with
``split=reference`` is selected, then a distinct row in the manifest.  A
self-reference is refused unless ``--allow-self-reference`` is supplied.

Example:

    .venv-nano/bin/python scripts/nano_lab/adaptation.py prepare \
        --manifest references.json \
        --cache artifacts/nano_lab/adaptation_cache.json

    .venv-nano/bin/python scripts/nano_lab/adaptation.py train \
        --cache artifacts/nano_lab/adaptation_cache.json \
        --checkpoint artifacts/nano_lab/adaptation_mommy.pt

The implementation is also importable.  ``attach_lora`` and
``load_adapter_checkpoint`` are useful to inference code that already owns a
loaded ``ChatterboxTurboTTS`` instance.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

# Keep tokenizer/OpenMP fan-out bounded before importing torch or transformers.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
if VENDOR_SRC.exists() and str(VENDOR_SRC) not in sys.path:
    # Prefer the checked-out vendor implementation over a different globally
    # installed Chatterbox revision.  This makes the causal alignment and
    # checkpoint loader reproducible.
    sys.path.insert(0, str(VENDOR_SRC))

DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_CACHE = ROOT / "artifacts" / "nano_lab" / "adaptation_cache.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "nano_lab" / "adaptation_best.pt"
SPEECH_VOCAB_SIZE = 6561


def configure_runtime(threads: int = 2) -> None:
    """Set bounded CPU threading without touching caller-owned CUDA state."""

    if threads < 1:
        raise ValueError("threads must be >= 1")
    torch.set_num_threads(int(threads))
    # Inter-op pools are process-global and can only be set before work starts.
    try:
        torch.set_num_interop_threads(max(1, min(int(threads), 2)))
    except RuntimeError:
        # A caller may already have initialized the pool.  In that case the
        # intra-op setting above still gives a deterministic bound for this
        # experiment.
        pass


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list/object or JSONL manifest."""

    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text()
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("rows"), list):
            rows = data["rows"]
        else:
            raise ValueError("manifest JSON must be a list or an object with a 'rows' list")
    if not rows:
        raise ValueError(f"manifest has no rows: {path}")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} is not an object")
        missing = [key for key in ("audio_path", "text", "split") if key not in row]
        if missing:
            raise ValueError(f"manifest row {index} is missing {', '.join(missing)}")
        text = str(row["text"]).strip()
        split = str(row["split"]).strip().lower()
        if not text:
            raise ValueError(f"manifest row {index} has empty text")
        if split in {"val", "dev", "validation"}:
            split = "valid"
        if split not in {"train", "valid", "test", "reference", "ref"}:
            raise ValueError(
                f"manifest row {index} has unsupported split {split!r}; "
                "use train, valid, test, or reference"
            )
        normalized = dict(row)
        normalized["text"] = text
        normalized["split"] = "reference" if split == "ref" else split
        normalized["speaker_id"] = str(normalized.get("speaker_id", "default"))
        normalized["_manifest_index"] = index
        out.append(normalized)
    return out


def resolve_path(raw: str | os.PathLike[str], *, manifest_path: Path) -> Path:
    value = Path(raw)
    if value.is_absolute():
        return value.resolve()
    candidates = [manifest_path.parent / value, ROOT / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the manifest-relative path for a useful error below.
    return candidates[0].resolve()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or os.path.samefile(a, b)
    except (FileNotFoundError, OSError):
        return a.resolve() == b.resolve()


def validate_target_splits(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject target leakage across train/valid/test by path or source interval."""

    paths: dict[str, set[str]] = {}
    for row in rows:
        split = str(row["split"])
        if split == "reference":
            continue
        path = str(Path(row["audio_path"]).resolve())
        paths.setdefault(path, set()).add(split)
    leaks = {path: sorted(splits) for path, splits in paths.items() if len(splits) > 1}
    if leaks:
        details = "; ".join(f"{path}: {splits}" for path, splits in list(leaks.items())[:5])
        raise ValueError("the same target audio appears in multiple splits: " + details)

    # Reference preparation records source_id/start_s/end_s.  Use those fields
    # when available to catch distinct WAV clips cut from overlapping source
    # intervals, which an exact-path check cannot detect.
    interval_rows = [
        row
        for row in rows
        if str(row.get("split")) != "reference"
        and row.get("source_id") is not None
        and row.get("start_s") is not None
        and row.get("end_s") is not None
    ]
    for index, left in enumerate(interval_rows):
        left_source = str(left["source_id"])
        left_start, left_end = float(left["start_s"]), float(left["end_s"])
        if left_end <= left_start:
            raise ValueError(f"invalid source interval in manifest row {left.get('_manifest_index', '?')}")
        for right in interval_rows[index + 1 :]:
            if left_source != str(right["source_id"]):
                continue
            if str(left["split"]) == str(right["split"]):
                continue
            right_start, right_end = float(right["start_s"]), float(right["end_s"])
            if max(left_start, right_start) < min(left_end, right_end):
                raise ValueError(
                    "target source intervals overlap across splits: "
                    f"{left_source} rows {left.get('_manifest_index', '?')} and "
                    f"{right.get('_manifest_index', '?')}"
                )


def validate_reference_interval(
    row: Mapping[str, Any],
    reference_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_self_reference: bool,
) -> None:
    """Reject a reference row that overlaps the target source interval."""

    if allow_self_reference:
        return
    if row.get("source_id") is None or row.get("start_s") is None or row.get("end_s") is None:
        return
    target_source = str(row["source_id"])
    target_start, target_end = float(row["start_s"]), float(row["end_s"])
    for candidate in rows:
        if str(candidate.get("split")) != "reference":
            continue
        candidate_path = Path(candidate.get("audio_path", "")).resolve()
        if candidate_path != reference_path.resolve():
            continue
        if candidate.get("source_id") is None or candidate.get("start_s") is None or candidate.get("end_s") is None:
            continue
        if target_source != str(candidate["source_id"]):
            continue
        reference_start, reference_end = float(candidate["start_s"]), float(candidate["end_s"])
        if max(target_start, reference_start) < min(target_end, reference_end):
            raise ValueError(
                f"row {row.get('_manifest_index', '?')} target interval overlaps its conditioning reference "
                f"on source {target_source}; choose a disjoint reference excerpt"
            )


def choose_reference_path(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    global_reference: Path | None,
    allow_self_reference: bool,
) -> tuple[Path, str]:
    """Resolve a conditioning excerpt and report its provenance."""

    target = Path(row["audio_path"]).resolve()
    explicit = row.get("reference_audio_path") or row.get("conditioning_audio_path")
    if explicit:
        candidate = resolve_path(str(explicit), manifest_path=manifest_path)
        source = "row.reference_audio_path" if row.get("reference_audio_path") else "row.conditioning_audio_path"
    elif global_reference is not None:
        candidate = global_reference.resolve()
        source = "--reference-audio"
    else:
        # Prefer a dedicated same-speaker reference.  If the manifest has no
        # reference row, only same-speaker TRAIN rows are eligible.  VALID and
        # TEST excerpts are never silently used as conditioning prompts.
        speaker_id = str(row.get("speaker_id", "default"))
        same_speaker = lambda other: str(other.get("speaker_id", "default")) == speaker_id
        refs = [
            other
            for other in rows
            if str(other.get("split")) == "reference" and same_speaker(other)
        ]
        train_rows = [
            other
            for other in rows
            if str(other.get("split")) == "train"
            and other is not row
            and same_speaker(other)
        ]
        candidates = refs + train_rows
        candidate = None
        for other in candidates:
            other_path = Path(other["audio_path"]).resolve()
            if other_path.exists() and not _same_file(target, other_path):
                candidate = other_path
                source = "manifest-distinct-row"
                break
        if candidate is None:
            if not allow_self_reference:
                raise ValueError(
                    f"row {row.get('_manifest_index', '?')} has no distinct conditioning excerpt. "
                    "Add a same-speaker reference_audio_path, a reference row, or --reference-audio. "
                    "Use --allow-self-reference only for a deliberately leaky diagnostic."
                )
            candidate = target
            source = "self-reference-explicit"

    if not candidate.exists():
        raise FileNotFoundError(f"conditioning audio does not exist: {candidate}")
    if _same_file(target, candidate) and not allow_self_reference:
        raise ValueError(
            f"row {row.get('_manifest_index', '?')} uses its target as conditioning audio. "
            "This leaks prompt content into the target; provide a distinct excerpt or "
            "pass --allow-self-reference for a diagnostic run."
        )
    return candidate, source


def load_nano(model_dir: Path, device: str):
    """Load the checked-out Nano model through the vendor API."""

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    required = [
        model_dir / "ve.safetensors",
        model_dir / "t3_nano_v1.safetensors",
        model_dir / "s3gen_meanflow.safetensors",
        model_dir / "vocab.json",
        model_dir / "merges.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Nano model files missing: " + ", ".join(missing))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    from runtime import NanoEngine
    return NanoEngine.from_pretrained(model_dir,device=device,dtype="fp32",optimized=True,cpu_threads=2).model


def load_audio(path: Path, *, sample_rate: int = 16_000) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1 or wav.size == 0:
        raise ValueError(f"audio is empty or not mono after loading: {path}")
    if not np.isfinite(wav).all():
        raise ValueError(f"audio contains NaN/Inf values: {path}")
    return wav


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for cache provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_conditioning(
    model: Any,
    reference_path: Path,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Extract Turbo conditionals through the vendor helper without alteration.

    ``ChatterboxTurboTTS.prepare_conditionals`` performs the reference
    resampling, loudness normalization, S3Gen speaker extraction, and T3
    prompt-token extraction that production inference uses.  Calling those
    lower-level encoders here independently previously changed the speaker
    vector (``as_spk=True`` averages partial embeddings) and could produce a
    different conditioning sequence.  Keep the exact tensors produced by the
    helper.  The returned metadata is intentionally explicit so cache readers
    can distinguish a production-compatible reference from a hand-built one.
    """

    reference_path = Path(reference_path).resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"conditioning audio does not exist: {reference_path}")
    # Do not pass a waveform here.  The upstream method owns the 24 kHz load,
    # -27 LUFS normalization, 16 kHz resampling, and reference-window limits.
    model.prepare_conditionals(str(reference_path), norm_loudness=True)
    conds = getattr(model, "conds", None)
    t3_cond = getattr(conds, "t3", None)
    if t3_cond is None:
        raise RuntimeError("prepare_conditionals did not produce T3 conditionals")
    speaker_tensor = getattr(t3_cond, "speaker_emb", None)
    prompt_tensor = getattr(t3_cond, "cond_prompt_speech_tokens", None)
    if not torch.is_tensor(speaker_tensor) or not torch.is_tensor(prompt_tensor):
        raise RuntimeError("T3 conditionals are missing speaker_emb or cond_prompt_speech_tokens")
    speaker_tensor = speaker_tensor.detach().cpu()
    prompt_tensor = prompt_tensor.detach().cpu().long()
    speaker = speaker_tensor.reshape(-1).numpy().astype(np.float32, copy=True)
    prompt = [int(value) for value in prompt_tensor.reshape(-1).tolist()]
    if speaker.size != 256 or not np.isfinite(speaker).all():
        raise ValueError(f"invalid speaker embedding shape/value: {tuple(speaker_tensor.shape)}")
    if not prompt:
        raise ValueError("conditioning excerpt produced zero T3 prompt tokens")
    provenance = {
        "path": str(reference_path),
        "sha256": file_sha256(reference_path),
        "preprocessing": {
            "entrypoint": "ChatterboxTurboTTS.prepare_conditionals",
            "norm_loudness": True,
            "exaggeration": 0.5,
            "speaker_embedding_source": "model.conds.t3.speaker_emb",
            "prompt_token_source": "model.conds.t3.cond_prompt_speech_tokens",
            "prompt_tokens_unmodified": True,
            "speaker_embedding_unmodified": True,
        },
    }
    return speaker, prompt, provenance


def extract_target_tokens(model: Any, wav16: np.ndarray, max_tokens: int) -> list[int]:
    """Extract the 25 Hz target stream without silent transcript truncation."""

    # Do not pass max_len to the tokenizer.  Cropping here would leave the full
    # transcript paired with a partial target and corrupt the causal objective.
    tokens, lengths = model.s3gen.tokenizer.forward([wav16], max_len=None)
    length = int(lengths[0].item())
    values = tokens[0, :length].detach().cpu().long()
    values = values[values < SPEECH_VOCAB_SIZE]
    if values.numel() > max_tokens:
        duration_s = len(wav16) / 16_000.0
        raise ValueError(
            f"target excerpt is {values.numel()} tokens ({duration_s:.2f}s), above "
            f"--max-target-tokens {max_tokens}; increase the cap or shorten the excerpt"
        )
    if values.numel() < 2:
        raise ValueError("target excerpt produced fewer than two valid speech tokens")
    return [int(x) for x in values.tolist()]


@torch.inference_mode()
def prepare_cache(
    *,
    manifest_path: Path,
    cache_path: Path,
    model_dir: Path,
    device: str,
    max_target_tokens: int,
    min_target_tokens: int,
    global_reference: Path | None,
    allow_self_reference: bool,
) -> dict[str, Any]:
    """Extract immutable features/tokens and write a portable JSON cache."""

    rows = read_rows(manifest_path)
    for row in rows:
        row["audio_path"] = str(resolve_path(row["audio_path"], manifest_path=manifest_path))
    validate_target_splits(rows)
    target_rows = [row for row in rows if row["split"] != "reference"]
    if not any(row["split"] == "train" for row in target_rows):
        raise ValueError("manifest needs at least one train row")
    if not any(row["split"] == "valid" for row in target_rows):
        raise ValueError("manifest needs at least one valid row for early stopping")
    for row in rows:
        if not Path(row["audio_path"]).exists():
            raise FileNotFoundError(f"target audio does not exist: {row['audio_path']}")

    from dataset_contract import require_audited_rows
    require_audited_rows(rows)

    model = load_nano(model_dir, device)
    # Match ChatterboxTurboTTS.generate's punc_norm before tokenization.  This
    # keeps adaptation inputs aligned with the text path used at inference.
    from chatterbox.tts_turbo import punc_norm

    tokenizer = model.tokenizer
    cache_rows: list[dict[str, Any]] = []
    conditioning_cache: dict[str, tuple[np.ndarray, list[int], dict[str, Any]]] = {}
    started = time.perf_counter()
    for index, row in enumerate(rows):
        target_path = Path(row["audio_path"]).resolve()
        if row["split"] == "reference":
            # Reference-only rows are eligible conditioning sources but have no
            # transcript target and never enter the adaptation loss.
            cache_rows.append(
                {
                    "id": str(row.get("id", f"reference-{index}")),
                    "audio_path": str(target_path),
                    "split": "reference",
                    "text": str(row.get("text", "")),
                    "speaker_id": str(row.get("speaker_id", "default")),
                }
            )
            continue
        reference_path, reference_source = choose_reference_path(
            row,
            rows,
            manifest_path=manifest_path,
            global_reference=global_reference,
            allow_self_reference=allow_self_reference,
        )
        validate_reference_interval(
            row,
            reference_path,
            rows,
            allow_self_reference=allow_self_reference,
        )
        target_audio = load_audio(target_path)
        target_tokens = extract_target_tokens(model, target_audio, max_target_tokens)
        if len(target_tokens) < min_target_tokens:
            raise ValueError(
                f"row {index} has {len(target_tokens)} target tokens, below --min-target-tokens "
                f"{min_target_tokens}; use a longer excerpt"
            )
        reference_key = str(reference_path.resolve())
        if reference_key not in conditioning_cache:
            conditioning_cache[reference_key] = extract_conditioning(model, reference_path)
        speaker, prompt, reference_provenance = conditioning_cache[reference_key]
        source_text = str(row["text"]).strip()
        text = punc_norm(source_text)
        encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        text_ids = encoded.input_ids[0].detach().cpu().long().tolist()
        if not text_ids:
            raise ValueError(f"row {index} text tokenized to zero tokens")
        cache_rows.append(
            {
                "id": str(row.get("id", f"row-{index}")),
                "audio_path": str(target_path),
                "reference_audio_path": str(reference_path),
                "reference_source": reference_source,
                "reference_sha256": reference_provenance["sha256"],
                "conditioning_provenance": reference_provenance,
                "text": text,
                "source_text": source_text,
                "transcript_audit": dict(row["transcript_audit"]),
                "text_tokens": [int(x) for x in text_ids],
                "speech_tokens": target_tokens,
                "split": row["split"],
                "speaker_id": str(row.get("speaker_id", "default")),
                "speaker_emb": [float(x) for x in speaker.tolist()],
                "cond_prompt_speech_tokens": prompt,
            }
        )
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "index": index,
                    "id": cache_rows[-1]["id"],
                    "split": row["split"],
                    "target_tokens": len(target_tokens),
                    "prompt_tokens": len(prompt),
                    "reference_source": reference_source,
                }
            ),
            flush=True,
        )

    config = {
        "format": "nano_t3_adaptation_cache_v1",
        "created_at_unix": time.time(),
        "source_manifest": str(manifest_path.resolve()),
        "model_dir": str(model_dir.resolve()),
        "device_at_prepare": device,
        "target_sample_rate": 16_000,
        "target_token_rate_hz": 25,
        "max_target_tokens": max_target_tokens,
        "conditioning_references": {
            path: provenance for path, (_, _, provenance) in conditioning_cache.items()
        },
        "rows": cache_rows,
        "counts": {
            "all": len(cache_rows),
            "train": sum(row["split"] == "train" for row in cache_rows),
            "valid": sum(row["split"] == "valid" for row in cache_rows),
            "test": sum(row["split"] == "test" for row in cache_rows),
            "reference": sum(row["split"] == "reference" for row in cache_rows),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "prompt_target_leakage_policy": {
            "target_and_reference_must_be_distinct": not allow_self_reference,
            "self_reference_allowed_for_diagnostic": allow_self_reference,
            "target_audio_is_never_used_as_reference_by_default": True,
        },
    }
    write_json(cache_path, config)
    write_json(cache_path.with_suffix(".manifest.json"), {k: v for k, v in config.items() if k != "rows"})
    return config


class LoRAConv1D(nn.Module):
    """LoRA residual around a frozen GPT-2 ``Conv1D`` projection.

    Hugging Face GPT-2 ``Conv1D`` stores its weight as ``[in_features,
    out_features]`` and accepts ``[..., in_features]``.  Two ordinary Linear
    layers therefore implement the same low-rank residual without changing the
    base module or its checkpoint format.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("LoRAConv1D requires a 2-D-weight GPT-2 Conv1D-like module")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout_p = float(dropout)
        in_features, out_features = map(int, base.weight.shape)
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # Zero B means a freshly attached adapter is exactly the zero-shot base.
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return int(self.base.weight.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.base.weight.shape[1])

    def forward(self, hidden_states: Tensor) -> Tensor:
        delta = self.lora_B(
            F.dropout(hidden_states, p=self.dropout_p, training=self.training)
            @ self.lora_A.weight.t()
        )
        return self.base(hidden_states) + delta * self.scaling

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_A.weight
        yield self.lora_B.weight


def parse_target_modules(spec: str) -> tuple[str, ...]:
    aliases = {
        "attn": ("attn.c_attn", "attn.c_proj"),
        "attention": ("attn.c_attn", "attn.c_proj"),
        "mlp": ("mlp.c_fc", "mlp.c_proj"),
        "all": ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"),
    }
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("--target-modules cannot be empty")
    expanded: list[str] = []
    for part in parts:
        expanded.extend(aliases.get(part, (part,)))
    valid = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}
    unknown = sorted(set(expanded) - valid)
    if unknown:
        raise ValueError(f"unsupported target module(s): {unknown}; use attn, mlp, all, or explicit names")
    return tuple(dict.fromkeys(expanded))


def parse_layer_indices(spec: str, n_layers: int) -> tuple[int, ...]:
    spec = spec.strip().lower()
    if spec == "all":
        return tuple(range(n_layers))
    if spec.startswith("last"):
        count = int(spec[4:])
        if count < 1:
            raise ValueError("lastN layer selection requires N >= 1")
        return tuple(range(max(0, n_layers - count), n_layers))
    values: list[int] = []
    for piece in spec.split(","):
        index = int(piece.strip())
        if index < 0:
            index += n_layers
        if not 0 <= index < n_layers:
            raise ValueError(f"layer index {index} is outside [0, {n_layers})")
        values.append(index)
    return tuple(sorted(set(values)))


def _replace_child(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _get_child(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str, nn.Module]:
    bits = dotted_name.split(".")
    parent = root
    for bit in bits[:-1]:
        parent = getattr(parent, bit) if not bit.isdigit() else parent[int(bit)]
    child_name = bits[-1]
    child = getattr(parent, child_name) if not child_name.isdigit() else parent[int(child_name)]
    return parent, child_name, child


def attach_lora(
    tfmr: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
    layers: str = "last4",
    target_modules: str | Sequence[str] = "attn",
) -> dict[str, LoRAConv1D]:
    """Attach and return named LoRA modules on GPT-2 transformer blocks."""

    if not hasattr(tfmr, "h"):
        raise TypeError("Nano T3 adapter expects a GPT-2 transformer with .h blocks")
    n_layers = len(tfmr.h)
    indices = parse_layer_indices(layers, n_layers)
    targets = parse_target_modules(target_modules) if isinstance(target_modules, str) else tuple(target_modules)
    adapters: dict[str, LoRAConv1D] = {}
    for layer_index in indices:
        block = tfmr.h[layer_index]
        for target in targets:
            dotted = f"h.{layer_index}.{target}"
            parent, child_name, child = _get_child(tfmr, dotted)
            if isinstance(child, LoRAConv1D):
                raise ValueError(f"adapter already attached at {dotted}")
            adapter = LoRAConv1D(child, rank=rank, alpha=alpha, dropout=dropout)
            # The base T3 is commonly loaded directly on CUDA/MPS.  New
            # adapter parameters start on CPU, so inherit the projection's
            # device and dtype before the first forward pass.
            adapter.to(device=child.weight.device, dtype=child.weight.dtype)
            _replace_child(parent, child_name, adapter)
            adapters[dotted] = adapter
    if not adapters:
        raise ValueError("no GPT-2 modules selected for LoRA")
    return adapters


def adapter_parameters(adapters: Mapping[str, LoRAConv1D]) -> list[nn.Parameter]:
    return [parameter for module in adapters.values() for parameter in module.adapter_parameters()]


def count_parameters(module: nn.Module) -> int:
    if module is None:return 0
    return sum(parameter.numel() for parameter in module.parameters())


def adapter_parameter_count(adapters: Mapping[str, LoRAConv1D]) -> int:
    return sum(parameter.numel() for parameter in adapter_parameters(adapters))


def adapter_config(adapters: Mapping[str, LoRAConv1D]) -> dict[str, Any]:
    if not adapters:
        raise ValueError("adapter set is empty")
    first = next(iter(adapters.values()))
    return {
        "rank": first.rank,
        "alpha": first.alpha,
        "dropout": first.dropout_p,
        "modules": list(adapters),
        "module_shapes": {
            name: {
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
            for name, module in adapters.items()
        },
    }


def adapter_state_dict(adapters: Mapping[str, LoRAConv1D]) -> dict[str, dict[str, Tensor]]:
    return {
        name: {
            "lora_A": module.lora_A.weight.detach().cpu(),
            "lora_B": module.lora_B.weight.detach().cpu(),
        }
        for name, module in adapters.items()
    }


def _load_state_into_adapters(
    adapters: Mapping[str, LoRAConv1D],
    state: Mapping[str, Mapping[str, Tensor]],
) -> None:
    if set(adapters) != set(state):
        missing = sorted(set(adapters) - set(state))
        extra = sorted(set(state) - set(adapters))
        raise ValueError(f"adapter module mismatch; missing={missing}, extra={extra}")
    for name, module in adapters.items():
        values = state[name]
        for key, parameter in (("lora_A", module.lora_A.weight), ("lora_B", module.lora_B.weight)):
            value = values[key]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"checkpoint shape mismatch for {name}.{key}: {tuple(value.shape)} != {tuple(parameter.shape)}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def save_adapter_checkpoint(
    path: Path,
    *,
    adapters: Mapping[str, LoRAConv1D],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nano_t3_lora_checkpoint_v1",
        "config": adapter_config(adapters),
        "metadata": dict(metadata),
        "adapter_state": adapter_state_dict(adapters),
    }
    torch.save(payload, path)
    sidecar = path.with_suffix(".json")
    write_json(sidecar, {"format": payload["format"], "config": payload["config"], "metadata": payload["metadata"]})


def load_adapter_checkpoint(
    tfmr: nn.Module,
    path: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, LoRAConv1D], dict[str, Any]]:
    """Attach a checkpoint's adapter configuration and load its weights."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "nano_t3_lora_checkpoint_v1":
        raise ValueError(f"unsupported adapter checkpoint format: {payload.get('format')!r}")
    config = payload["config"]
    module_names = list(config["modules"])
    # Derive a layer/module selection from the explicit module names.  Passing
    # explicit names avoids accidental attachment to a changed block count.
    adapters: dict[str, LoRAConv1D] = {}
    for dotted in module_names:
        parent, child_name, child = _get_child(tfmr, dotted)
        if isinstance(child, LoRAConv1D):
            raise ValueError(f"adapter already attached at {dotted}")
        shape = config["module_shapes"][dotted]
        if tuple(child.weight.shape) != (shape["in_features"], shape["out_features"]):
            raise ValueError(f"base projection shape changed at {dotted}")
        adapter = LoRAConv1D(
            child,
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            dropout=float(config.get("dropout", 0.0)),
        )
        adapter.to(device=child.weight.device, dtype=child.weight.dtype)
        _replace_child(parent, child_name, adapter)
        adapters[dotted] = adapter
    _load_state_into_adapters(adapters, payload["adapter_state"])
    if strict and set(module_names) != set(adapters):
        raise ValueError("checkpoint adapter module names are not unique")
    return adapters, dict(payload.get("metadata", {}))


def load_adapter(model_or_t3: Any, checkpoint: Path | str) -> dict[str, LoRAConv1D]:
    """Convenience loader for inference sweeps.

    Callers may pass either a loaded ``ChatterboxTurboTTS.t3`` object or its
    underlying ``tfmr`` module.  The return value is the attached adapter map;
    checkpoint metadata is available through the checkpoint sidecar or
    ``load_adapter_checkpoint`` when it is needed by the caller.
    """

    tfmr = getattr(model_or_t3, "tfmr", model_or_t3)
    adapters, _ = load_adapter_checkpoint(tfmr, Path(checkpoint))
    for parameter in tfmr.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter_parameters(adapters):
        parameter.requires_grad_(False)
    tfmr.eval()
    set_adapter_mode(adapters, False)
    return adapters


def make_condition(model: Any, row: Mapping[str, Any], device: torch.device):
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    speaker = torch.tensor(row["speaker_emb"], dtype=torch.float32, device=device).view(1, 256)
    prompt = torch.tensor(row["cond_prompt_speech_tokens"], dtype=torch.long, device=device).view(1, -1)
    return T3Cond(
        speaker_emb=speaker,
        cond_prompt_speech_tokens=prompt,
        emotion_adv=None,
    )


def teacher_forced_logits(
    model: Any,
    row: Mapping[str, Any],
    *,
    device: torch.device,
    max_target_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return shifted Turbo logits and targets for one cache row.

    The returned logits have shape ``[1, target_tokens, speech_vocab]`` and
    predict ``speech + EOS`` from ``BOS + speech[:-1]``.  Keeping this forward
    pass separate from the cross-entropy calculation lets the optional KL
    regularizer compare an adapted pass with the exact frozen-base pass.
    """

    text_tokens = torch.tensor(row["text_tokens"], dtype=torch.long, device=device).view(1, -1)
    speech = torch.tensor(row["speech_tokens"], dtype=torch.long, device=device)
    if max_target_tokens is not None and speech.numel() > max_target_tokens:
        raise ValueError(
            f"cache row {row.get('id')} has {speech.numel()} target tokens, above "
            f"--max-target-tokens {max_target_tokens}; refusing silent target truncation"
        )
    if speech.numel() < 2:
        raise ValueError(f"row {row.get('id')} has fewer than two target tokens")
    hp = model.t3.hp
    target = torch.cat((speech, torch.tensor([hp.stop_speech_token], dtype=torch.long, device=device)))
    decoder_input = torch.cat((torch.tensor([hp.start_speech_token], dtype=torch.long, device=device), target[:-1]))
    decoder_input = decoder_input.view(1, -1)
    cond = make_condition(model, row, device)
    embeds, _ = model.t3.prepare_input_embeds(
        t3_cond=cond,
        text_tokens=text_tokens,
        speech_tokens=decoder_input,
        cfg_weight=0.0,
    )
    outputs = model.t3.tfmr(
        inputs_embeds=embeds,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state[:, -decoder_input.shape[1] :, :]
    logits = model.t3.speech_head(hidden)
    return logits, target


def teacher_forced_loss(
    model: Any,
    row: Mapping[str, Any],
    *,
    device: torch.device,
    max_target_tokens: int | None = None,
) -> tuple[Tensor, int, int]:
    """Compute exact shifted Turbo causal loss for one cache row.

    Returns ``(cross_entropy, token_count, correct_count)``.  The returned
    token count includes the stop token, which lets reports expose true token
    perplexity and accuracy.
    """

    logits, target = teacher_forced_logits(
        model,
        row,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.view(-1))
    predicted = logits.argmax(dim=-1).view(-1)
    correct = int((predicted == target).sum().item())
    return loss, int(target.numel()), correct


def teacher_forced_loss_with_base_kl(
    model: Any,
    row: Mapping[str, Any],
    adapters: Mapping[str, LoRAConv1D],
    *,
    device: torch.device,
    max_target_tokens: int | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, int, int, Tensor]:
    """Compute adapted CE and KL to the exact frozen base on one cache row.

    The base logits use the same condition, text, and shifted decoder input as
    the adapted pass.  Every adapter residual is disabled by setting its
    runtime scale to zero, and the base forward is wrapped in ``no_grad``.
    Adapter scales and training/evaluation modes are restored before returning.
    The final tensor is a per-token temperature-scaled KL divergence suitable
    for adding to the CE objective.
    """

    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("KL temperature must be finite and > 0")
    saved_scalings = {name: float(module.scaling) for name, module in adapters.items()}
    saved_modes = {name: bool(module.training) for name, module in adapters.items()}
    try:
        # Running the wrappers in eval mode makes this reference pass
        # independent of the adapter dropout setting.  The frozen GPT-2 base
        # remains the same module and receives no gradient.
        set_adapter_mode(adapters, False)
        for module in adapters.values():
            module.scaling = 0.0
        with torch.no_grad():
            base_logits, base_target = teacher_forced_logits(
                model,
                row,
                device=device,
                max_target_tokens=max_target_tokens,
            )
    finally:
        for name, module in adapters.items():
            module.scaling = saved_scalings[name]
            module.train(saved_modes[name])

    adapted_logits, target = teacher_forced_logits(
        model,
        row,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    if not torch.equal(base_target, target):
        raise RuntimeError("base and adapted teacher-forcing targets differ")
    loss = F.cross_entropy(adapted_logits.reshape(-1, adapted_logits.shape[-1]), target.view(-1))
    predicted = adapted_logits.argmax(dim=-1).view(-1)
    correct = int((predicted == target).sum().item())
    vocab = adapted_logits.shape[-1]
    adapted_log_probs = F.log_softmax(
        adapted_logits.reshape(-1, vocab) / float(temperature),
        dim=-1,
    )
    base_probs = F.softmax(
        base_logits.detach().reshape(-1, vocab) / float(temperature),
        dim=-1,
    )
    # Floating-point roundoff can make an identical-distribution KL a tiny
    # negative number.  Clamp that numerical artifact while preserving the
    # true non-negative penalty.
    kl = (
        F.kl_div(adapted_log_probs, base_probs, reduction="batchmean")
        * float(temperature) ** 2
    ).clamp_min(0.0)
    return loss, int(target.numel()), correct, kl


@dataclasses.dataclass
class EvalStats:
    loss: float
    tokens: int
    correct: int

    @property
    def token_accuracy(self) -> float:
        return self.correct / self.tokens if self.tokens else float("nan")

    @property
    def perplexity(self) -> float:
        return math.exp(min(20.0, self.loss))

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "tokens": self.tokens,
            "correct": self.correct,
            "token_accuracy": self.token_accuracy,
            "perplexity": self.perplexity,
        }


def evaluate(model: Any, rows: Sequence[Mapping[str, Any]], *, device: torch.device, max_target_tokens: int | None) -> EvalStats:
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for row in rows:
            loss, tokens, correct = teacher_forced_loss(
                model,
                row,
                device=device,
                max_target_tokens=max_target_tokens,
            )
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            total_correct += correct
    return EvalStats(
        loss=total_loss / max(total_tokens, 1),
        tokens=total_tokens,
        correct=total_correct,
    )


def peak_rss_mib() -> float:
    # Linux reports KiB; macOS reports bytes.  The runner is Linux today, but
    # handling both keeps the report portable.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def load_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "nano_t3_adaptation_cache_v1":
        raise ValueError(f"unsupported cache format: {payload.get('format')!r}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("cache rows must be a list")
    train = [row for row in rows if row.get("split") == "train"]
    valid = [row for row in rows if row.get("split") == "valid"]
    if not train or not valid:
        raise ValueError("cache requires non-empty train and valid splits")
    from dataset_contract import require_audited_rows
    require_audited_rows(train + valid)
    from dataset_contract import require_inference_conditioning
    require_inference_conditioning(train + valid)
    for row in train + valid:
        for key in ("text_tokens", "speech_tokens", "speaker_emb", "cond_prompt_speech_tokens"):
            if key not in row:
                raise ValueError(f"cache row {row.get('id')} missing {key}")
    return payload


def freeze_for_adapter(model: Any) -> None:
    for module in (model.t3,model.s3gen,model.ve):
        if module is not None:
            for parameter in module.parameters():parameter.requires_grad_(False)
            module.eval()


def set_adapter_mode(adapters: Mapping[str, LoRAConv1D], training: bool) -> None:
    for module in adapters.values():
        module.train(training)


def select_speaker_rows(
    rows: Sequence[Mapping[str, Any]],
    speaker_id: str | None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    """Select one voice for an adapter and reject mixed-speaker training."""

    target_rows = [row for row in rows if row.get("split") in {"train", "valid", "test"}]
    speakers = sorted({str(row.get("speaker_id", "default")) for row in target_rows})
    if speaker_id is None:
        if len(speakers) != 1:
            raise ValueError(
                "cache contains multiple speakers "
                f"{speakers}; pass --speaker-id to train one per-voice adapter"
            )
        selected = speakers[0]
    else:
        selected = str(speaker_id)
        if selected not in speakers:
            raise ValueError(f"--speaker-id {selected!r} is not present; available speakers: {speakers}")
    train_rows = [row for row in target_rows if row.get("split") == "train" and str(row.get("speaker_id", "default")) == selected]
    valid_rows = [row for row in target_rows if row.get("split") == "valid" and str(row.get("speaker_id", "default")) == selected]
    if not train_rows or not valid_rows:
        raise ValueError(
            f"speaker {selected!r} needs non-empty train and valid splits "
            f"(train={len(train_rows)}, valid={len(valid_rows)})"
        )
    return train_rows, valid_rows, selected


def train_adapter(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime(args.threads)
    cache = load_cache(Path(args.cache))
    model_dir = Path(args.model_dir)
    device = torch.device(args.device)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    from t3_training import load_cached_t3
    model = load_cached_t3(model_dir, args.device)
    model.t3.eval()
    freeze_for_adapter(model)

    resume_path = Path(args.resume) if args.resume else None
    if resume_path:
        adapters, _ = load_adapter_checkpoint(model.t3.tfmr, resume_path)
        config = adapter_config(adapters)
    else:
        adapters = attach_lora(
            model.t3.tfmr,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            layers=args.layers,
            target_modules=args.target_modules,
        )
        config = adapter_config(adapters)
    # Re-freeze after wrapping, then enable only LoRA matrices.
    freeze_for_adapter(model)
    trainable = adapter_parameters(adapters)
    for parameter in trainable:
        parameter.requires_grad_(True)
    set_adapter_mode(adapters, True)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    train_rows, valid_rows, selected_speaker = select_speaker_rows(cache["rows"], args.speaker_id)
    max_target_tokens = args.max_target_tokens or cache.get("max_target_tokens")
    kl_coef = float(getattr(args, "kl_coef", 0.0))
    kl_temperature = float(getattr(args, "kl_temperature", 1.0))
    if kl_coef < 0.0 or not math.isfinite(kl_coef):
        raise ValueError("--kl-coef must be finite and >= 0")
    # Measure the heldout objective before any optimizer step.  A per-voice
    # adapter is adopted only if it beats this initial value by min_delta.
    set_adapter_mode(adapters, False)
    baseline_valid_stats = evaluate(
        model,
        valid_rows,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    if not math.isfinite(baseline_valid_stats.loss):
        raise RuntimeError("initial heldout loss is not finite")
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "valid": baseline_valid_stats.as_dict(),
            "objective": "initial heldout loss before optimizer steps",
        }
    ]
    print(json.dumps({"event": "baseline", "epoch": 0, "valid": baseline_valid_stats.as_dict()}), flush=True)
    best_valid = baseline_valid_stats.loss
    best_epoch = 0
    best_checkpoint_saved = False
    epochs_without_improvement = 0
    started = time.perf_counter()
    checkpoint = Path(args.checkpoint)

    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(train_rows)))
        random.Random((args.seed or 0) + epoch).shuffle(order)
        set_adapter_mode(adapters, True)
        train_loss_sum = 0.0
        train_kl_sum = 0.0
        train_tokens = 0
        train_correct = 0
        for step_index, row_index in enumerate(order, start=1):
            row = train_rows[row_index]
            optimizer.zero_grad(set_to_none=True)
            if kl_coef > 0.0:
                loss, tokens, correct, kl = teacher_forced_loss_with_base_kl(
                    model,
                    row,
                    adapters,
                    device=device,
                    max_target_tokens=max_target_tokens,
                    temperature=kl_temperature,
                )
            else:
                loss, tokens, correct = teacher_forced_loss(
                    model,
                    row,
                    device=device,
                    max_target_tokens=max_target_tokens,
                )
                kl = torch.zeros((), device=device)
            l2 = torch.zeros((), device=device)
            if args.adapter_l2:
                l2 = torch.stack([parameter.float().pow(2).mean() for parameter in trainable]).mean()
            objective = loss + float(args.adapter_l2) * l2 + kl_coef * kl
            objective.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            train_loss_sum += float(loss.detach().item()) * tokens
            train_kl_sum += float(kl.detach().item()) * tokens
            train_tokens += tokens
            train_correct += correct
            if args.max_steps and (epoch - 1) * len(order) + step_index >= args.max_steps:
                break
        train_stats = EvalStats(
            loss=train_loss_sum / max(train_tokens, 1),
            tokens=train_tokens,
            correct=train_correct,
        )
        set_adapter_mode(adapters, False)
        valid_stats = evaluate(
            model,
            valid_rows,
            device=device,
            max_target_tokens=max_target_tokens,
        )
        record = {
            "epoch": epoch,
            "train": train_stats.as_dict(),
            "valid": valid_stats.as_dict(),
            "objective": "shifted speech-token cross-entropy + adapter L2 + base-logit KL"
            if kl_coef > 0.0
            else "shifted speech-token cross-entropy + adapter L2",
            "adapter_l2": float(args.adapter_l2),
            "kl_coef": kl_coef,
            "kl_temperature": kl_temperature,
            "train_kl": train_kl_sum / max(train_tokens, 1),
        }
        history.append(record)
        print(json.dumps({"event": "epoch", **record}), flush=True)
        improved = valid_stats.loss < best_valid - float(args.min_delta)
        if improved:
            best_valid = valid_stats.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            metadata = {
                "experiment": "nano_t3_per_voice_lora",
                "mode": "adapted",
                "training_loader": "t3_only_cached_features; acoustic/reference encoders not loaded",
                "zero_shot_label": "base_nano_without_adapter",
                "adapted_label": "nano_lora_adapter",
                "objective": "exact Turbo causal shifted speech-token cross-entropy",
                "causal_alignment": {
                    "input": "conditioning + text + BOS + speech[:-1]",
                    "target": "speech + EOS",
                    "stock_t3_loss_used": False,
                    "reason": "stock loss scores logits at positions whose speech embedding contains the same target",
                },
                "model_dir": str(model_dir.resolve()),
                "device": str(device),
                "cpu_threads": int(args.threads),
                "adapter": config,
                "parameter_counts": {
                    "adapter_trainable": adapter_parameter_count(adapters),
                    "t3_total": count_parameters(model.t3),
                    "s3gen_frozen": count_parameters(model.s3gen),
                    "voice_encoder_frozen": count_parameters(model.ve),
                },
                "data": {
                    "cache": str(Path(args.cache).resolve()),
                    "speaker_id": selected_speaker,
                    "train_rows": len(train_rows),
                    "valid_rows": len(valid_rows),
                    "max_target_tokens": max_target_tokens,
                    "prompt_target_leakage_policy": cache.get("prompt_target_leakage_policy", {}),
                },
                "regularization": {
                    "adapter_l2": float(args.adapter_l2),
                    "weight_decay": float(args.weight_decay),
                    "grad_clip": float(args.grad_clip),
                    "base_logit_kl": {
                        "coefficient": kl_coef,
                        "temperature": kl_temperature,
                        "reference": "same frozen base T3 pass with every LoRA residual scale set to zero",
                        "reference_gradients": False,
                        "reduction": "per-token batchmean, multiplied by temperature squared",
                    },
                },
                "training": {
                    "epoch": epoch,
                    "initial_valid": baseline_valid_stats.as_dict(),
                    "best_valid_loss": valid_stats.loss,
                    "early_stop_patience": int(args.patience),
                    "history": history,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_mib": peak_rss_mib(),
                },
                "adoption_gate": {
                    "baseline_valid_loss": baseline_valid_stats.loss,
                    "required_improvement": float(args.min_delta),
                    "accepted_for_audio_evaluation": True,
                    "release_status": "unreviewed",
                },
                "tradeoffs": {
                    "adapter_scope": "GPT-2 attention projections only unless --target-modules changes it",
                    "expected_effect": "small timbre/prosody correction; no new speaker data or vocoder denoising capacity",
                    "8gb_gpu_feasibility": "training is likely feasible with Nano plus LoRA and short excerpts; full model load and optimizer/vocoder RSS must be measured on the target GPU",
                    "quality_claim": "not established until heldout audio and speaker-similarity metrics are evaluated",
                },
            }
            save_adapter_checkpoint(checkpoint, adapters=adapters, metadata=metadata)
            best_checkpoint_saved = True
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(args.patience):
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
        if args.max_steps and (epoch - 1) * len(order) + len(order) >= args.max_steps:
            break

    report = {
        "format": "nano_t3_adaptation_report_v1",
        "training_loader": "t3_only_cached_features; acoustic/reference encoders not loaded",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_written": best_checkpoint_saved,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "initial_valid": baseline_valid_stats.as_dict(),
        "adoption": {
            "accepted_for_audio_evaluation": best_checkpoint_saved,
            "release_status": "unreviewed" if best_checkpoint_saved else "rejected",
            "reason": "heldout loss improved beyond min_delta" if best_checkpoint_saved else "no heldout improvement beyond min_delta; adapter rejected",
            "speaker_id": selected_speaker,
        },
        "history": history,
        "parameter_counts": {
            "adapter_trainable": adapter_parameter_count(adapters),
            "t3_total": count_parameters(model.t3),
            "s3gen_frozen": count_parameters(model.s3gen),
            "voice_encoder_frozen": count_parameters(model.ve),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": peak_rss_mib(),
        "resume": str(resume_path.resolve()) if resume_path else None,
        "config": config,
    }
    report_path = checkpoint.with_name(checkpoint.stem + "_report.json")
    write_json(report_path, report)
    return report


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime(args.threads)
    if args.checkpoint and not args.load_model:
        payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "format": payload.get("format"),
            "config": payload.get("config"),
            "metadata": payload.get("metadata"),
        }
        print(json.dumps(result, indent=2, default=_json_default))
        return result
    model = load_nano(Path(args.model_dir), args.device)
    result = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "device": args.device,
        "t3_parameters": count_parameters(model.t3),
        "s3gen_parameters": count_parameters(model.s3gen),
        "voice_encoder_parameters": count_parameters(model.ve),
        "t3_transformer": type(model.t3.tfmr).__name__,
        "t3_layers": len(model.t3.tfmr.h) if hasattr(model.t3.tfmr, "h") else None,
        "hidden_size": int(model.t3.dim),
        "turbo_causal_alignment": "conditioning + text + BOS predicts speech[0]; prior speech predicts next speech",
    }
    if args.checkpoint:
        adapters, metadata = load_adapter_checkpoint(model.t3.tfmr, Path(args.checkpoint))
        result["adapter_parameters"] = adapter_parameter_count(adapters)
        result["adapter_config"] = adapter_config(adapters)
        result["checkpoint_metadata"] = metadata
    print(json.dumps(result, indent=2, default=_json_default))
    return result


def generate_with_adapter(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime(args.threads)
    model = load_nano(Path(args.model_dir), args.device)
    adapters, metadata = load_adapter_checkpoint(model.t3.tfmr, Path(args.adapter))
    for parameter in model.t3.parameters():
        parameter.requires_grad_(False)
    model.t3.eval()
    set_adapter_mode(adapters, False)
    model.prepare_conditionals(str(Path(args.reference_audio)), exaggeration=0.0)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        audio = model.generate(args.text, temperature=args.temperature, top_p=args.top_p)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    wav = audio.squeeze(0).detach().cpu().numpy()
    sf.write(output, wav, model.sr, subtype="PCM_24")
    result = {
        "format": "nano_t3_adapted_sample_v1",
        "path": str(output.resolve()),
        "text": args.text,
        "reference_audio": str(Path(args.reference_audio).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "label": "adapted",
        "model_label": "Nano",
        "seconds": len(wav) / model.sr,
        "metadata": metadata,
        "adapter_parameters": adapter_parameter_count(adapters),
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, default=_json_default))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="extract target tokens and distinct conditioning features")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prepare.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    prepare.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    prepare.add_argument("--threads", type=int, default=2)
    prepare.add_argument("--max-target-tokens", type=int, default=400)
    prepare.add_argument("--min-target-tokens", type=int, default=2)
    prepare.add_argument("--reference-audio", type=Path)
    prepare.add_argument("--allow-self-reference", action="store_true")

    train = sub.add_parser("train", help="fit a frozen-base LoRA adapter with heldout early stopping")
    train.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train.add_argument("--resume", type=Path)
    train.add_argument("--speaker-id", help="voice to adapt when cache contains more than one speaker")
    train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--rank", type=int, default=4)
    train.add_argument("--alpha", type=float, default=8.0)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--layers", default="last4", help="all, lastN, or comma-separated GPT-2 layer indices")
    train.add_argument("--target-modules", default="attn", help="attn, mlp, all, or comma-separated projections")
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--adapter-l2", type=float, default=1e-5)
    train.add_argument(
        "--kl-coef",
        type=float,
        default=0.0,
        help="optional coefficient for per-token KL to the frozen base logits (default: 0)",
    )
    train.add_argument(
        "--kl-temperature",
        type=float,
        default=1.0,
        help="temperature for the optional base-logit KL reference (default: 1)",
    )
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--patience", type=int, default=2)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--max-target-tokens", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--seed", type=int, default=31)

    inspect = sub.add_parser("inspect", help="report Nano and adapter parameter counts")
    inspect.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    inspect.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    inspect.add_argument("--threads", type=int, default=2)
    inspect.add_argument("--checkpoint", type=Path)
    inspect.add_argument("--load-model", action="store_true", help="load model even when checkpoint metadata is enough")

    generate = sub.add_parser("generate", help="load an adapter checkpoint and write one labeled sample")
    generate.add_argument("--adapter", type=Path, required=True)
    generate.add_argument("--reference-audio", type=Path, required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    generate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    generate.add_argument("--threads", type=int, default=2)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=31)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        configure_runtime(args.threads)
        prepare_cache(
            manifest_path=args.manifest,
            cache_path=args.cache,
            model_dir=args.model_dir,
            device=args.device,
            max_target_tokens=args.max_target_tokens,
            min_target_tokens=args.min_target_tokens,
            global_reference=args.reference_audio,
            allow_self_reference=args.allow_self_reference,
        )
        return 0
    if args.command == "train":
        train_adapter(args)
        return 0
    if args.command == "inspect":
        inspect_model(args)
        return 0
    if args.command == "generate":
        generate_with_adapter(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
