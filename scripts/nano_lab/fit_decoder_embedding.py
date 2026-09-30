"""Fit a small S3Gen decoder speaker embedding on aligned mel targets.

This is a bounded research experiment.  It loads only the S3Gen flow encoder
and meanflow estimator.  T3, the speech tokenizer, the voice encoder, and HiFT
are not constructed.  All loaded weights stay frozen.  Only the 192-value
``gen.embedding`` vector from an audited ``Conditionals`` cache is optimized.

The input is a :mod:`mel_calibration` prepare directory.  Its rows contain
the normalized source mel and the native flow reconstruction.  Before any
optimizer step this script reproduces the native reconstruction with the
streamed flow modules, the same prompt conditionals, exact two-step
meanflow Euler integration, and a fixed per-row Torch noise seed.  A frame
geometry gate requires source mel length ``2*N`` or ``2*N-1`` for ``N`` source
speech tokens.  The three appended ``S3GEN_SIL`` IDs account for six output
frames. The measured reference mel/token offset (zero or one frame) and
source framing tail determine the exact source crop. There is no time warp,
resampling, or DTW.

The output is an ordinary ``{t3, gen}`` conditionals cache with only
``gen.embedding`` changed.  The report keeps baseline parity, epoch-0
validation, best validation, cosine/norm constraints, provenance, and the
experimental status visible.  A parity-only command is available for a
guarded smoke check.  This module performs no model work when imported.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREPARE_DIR = ROOT / "artifacts" / "nano_lab" / "mel_calibration_asmr"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "nano_lab" / "decoder_embedding_fit"
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
SPEECH_VOCAB = 6561
S3GEN_SIL = 4299
MEL_BANDS = 80
EMBEDDING_DIM = 192
DEFAULT_STEPS = 2
DEFAULT_MAX_STEPS = 40
DEFAULT_PATIENCE = 3
DEFAULT_LR = 1e-2
DEFAULT_REG = 1e-2
DEFAULT_ENVELOPE_WEIGHT = 0.25
DEFAULT_PARITY_ATOL = 1e-4
DEFAULT_MIN_COSINE = 0.98


def _resolve(value: str | os.PathLike[str], *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _normalise_tokens(values: Any, *, row_id: str) -> np.ndarray:
    tokens = np.asarray(values)
    if tokens.ndim != 1 or tokens.size < 2:
        raise ValueError(f"row {row_id} speech_tokens must be a vector with >=2 IDs")
    if not np.issubdtype(tokens.dtype, np.integer):
        if not np.isfinite(tokens).all() or not np.equal(tokens, np.floor(tokens)).all():
            raise ValueError(f"row {row_id} speech_tokens are not integer IDs")
    tokens = tokens.astype(np.int64, copy=False)
    if np.any(tokens < 0) or np.any(tokens >= SPEECH_VOCAB):
        raise ValueError(f"row {row_id} contains out-of-range or special speech IDs")
    return tokens


def _append_silence_tokens(tokens: np.ndarray) -> np.ndarray:
    """Append the three native S3Gen silence IDs exactly once."""

    values = np.asarray(tokens, dtype=np.int64).reshape(-1)
    return np.concatenate((values, np.full((3,), S3GEN_SIL, dtype=np.int64)))


def _load_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping) and payload.get("format") == "nano_acoustic_cache_v1":
        # Acoustic supervision uses recorded audio and actual source tokens.
        # Its separate validator enforces source/split/interval provenance;
        # it must never manufacture an accepted text transcript audit.
        from prepare_acoustic_data import validate_acoustic_cache

        return validate_acoustic_cache(path)
    if not isinstance(payload, Mapping) or payload.get("format") != "nano_t3_adaptation_cache_v1":
        raise ValueError(f"unsupported aligned adaptation cache: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("aligned adaptation cache has no rows")
    return dict(payload)


def _validate_prepare_inputs(prepare_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path, dict[str, Any], Path]:
    """Join prepare rows to the aligned token cache and reject drift."""

    manifest_path = prepare_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "nano_mel_calibration_prepare_v1":
        raise ValueError("fit expects nano_mel_calibration_prepare_v1")
    cache_path = _resolve(manifest.get("cache"))
    cache = _load_cache(cache_path)
    if not isinstance(manifest.get("cache_sha256"), str) or len(manifest["cache_sha256"]) != 64:
        raise ValueError("prepare manifest is missing cache SHA-256")
    if manifest["cache_sha256"] != _sha256(cache_path):
        raise ValueError("prepare manifest cache SHA-256 does not match current cache")
    cache_rows = {str(row.get("id")): dict(row) for row in cache["rows"]}
    prepare_rows = manifest.get("rows")
    if not isinstance(prepare_rows, list) or not prepare_rows:
        raise ValueError("prepare manifest has no rows")
    acoustic_only = manifest.get("acoustic_only") is True
    if cache.get("format") == "nano_acoustic_cache_v1" and not acoustic_only:
        raise ValueError("acoustic cache requires an explicitly acoustic-only preparation")
    if acoustic_only:
        if (manifest.get("status") != "ready"
                or manifest.get("completed_rows") != len(prepare_rows)
                or manifest.get("reference_native_parity", {}).get("status") != "passed"
                or manifest.get("reference_source_overlap", {}).get("status") != "passed"):
            raise ValueError("acoustic preparation is incomplete or lacks reference parity")
        if {str(row.get("id")) for row in prepare_rows} != set(cache_rows):
            raise ValueError("acoustic preparation does not cover the complete source cache")
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in prepare_rows:
        row = dict(item)
        row_id = str(row.get("id"))
        if row_id in seen:
            raise ValueError(f"duplicate prepare row {row_id}")
        seen.add(row_id)
        source = cache_rows.get(row_id)
        if source is None:
            raise ValueError(f"prepare row {row_id} is missing from aligned cache")
        if str(row.get("split")) not in {"train", "valid"} or source.get("split") != row.get("split"):
            raise ValueError(f"split mismatch for row {row_id}")
        if str(row.get("audio_path")) != str(source.get("audio_path")):
            raise ValueError(f"audio path mismatch for row {row_id}")
        source_path = _resolve(str(source["audio_path"]))
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_sha = _sha256(source_path)
        if not isinstance(row.get("source_sha256"), str) or len(row["source_sha256"]) != 64:
            raise ValueError(f"prepare row {row_id} has no source SHA-256")
        if row["source_sha256"] != source_sha:
            raise ValueError(f"source SHA-256 mismatch for row {row_id}")
        tokens = _normalise_tokens(source.get("speech_tokens"), row_id=row_id)
        if int(row.get("speech_token_count", -1)) != int(tokens.size):
            raise ValueError(f"speech token count mismatch for row {row_id}")
        try:
            row_seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"prepare row {row_id} has no integer native seed") from exc
        mel_path = _resolve(row["mel_path"])
        if acoustic_only and row.get("mel_sha256") != _sha256(mel_path):
            raise ValueError(f"prepared mel SHA-256 mismatch for {row_id}")
        arrays = _load_npz(mel_path)
        source_mel = np.asarray(arrays.get("source_mel"), dtype=np.float32)
        if acoustic_only:
            cached_source_mel = _load_npz(_resolve(source["source_mel_path"]))["source_mel"]
            if not np.array_equal(source_mel, cached_source_mel):
                raise ValueError(f"prepared source mel differs from acoustic cache for {row_id}")
        reconstructed_mel = np.asarray(arrays.get("reconstructed_mel"), dtype=np.float32)
        if source_mel.ndim != 2 or source_mel.shape[0] != MEL_BANDS:
            raise ValueError(f"source mel shape for {row_id} is {source_mel.shape}, expected (80,T)")
        if reconstructed_mel.ndim != 2 or reconstructed_mel.shape[0] != MEL_BANDS:
            raise ValueError(f"reconstructed mel shape for {row_id} is {reconstructed_mel.shape}, expected (80,T)")
        if not np.isfinite(source_mel).all() or not np.isfinite(reconstructed_mel).all():
            raise ValueError(f"source/reconstructed mel for {row_id} contains NaN or infinity")
        source_frames = int(source_mel.shape[1])
        expected_source_frames = {2 * int(tokens.size), 2 * int(tokens.size) - 1}
        if source_frames not in expected_source_frames:
            raise ValueError(
                f"source mel frame count for {row_id} is {source_frames}; expected 2N or 2N-1 for N={tokens.size}"
            )
        expected_reconstructed_frames = 2 * (int(tokens.size) + 3)
        if reconstructed_mel.shape[1] not in {expected_reconstructed_frames, expected_reconstructed_frames - 1}:
            raise ValueError(
                f"prepared reconstructed mel frame count for {row_id} is {reconstructed_mel.shape[1]}; "
                f"expected {expected_reconstructed_frames} or one fewer for an odd reference mel length"
            )
        joined.append(
            {
                "id": row_id,
                "split": str(row["split"]),
                "audio_path": str(source_path),
                "source_sha256": source_sha,
                "tokens": tokens,
                "seed": row_seed,
                "source_mel": source_mel,
                "prepared_reconstructed_mel": reconstructed_mel,
                "prepared_mel_path": str(_resolve(row["mel_path"])),
                "source_frames": source_frames,
                "conditioning_reference_path": source.get("reference_audio_path"),
                "conditioning_reference_sha256": source.get("reference_sha256"),
            }
        )
    train_ids = {row["id"] for row in joined if row["split"] == "train"}
    valid_ids = {row["id"] for row in joined if row["split"] == "valid"}
    if not train_ids or not valid_ids or train_ids & valid_ids:
        raise ValueError("joined dataset needs disjoint non-empty train and valid splits")
    conditionals_path = _resolve(manifest.get("conditionals_path"))
    if not conditionals_path.exists():
        raise FileNotFoundError(conditionals_path)
    if not isinstance(manifest.get("conditioning_sha256"), str) or len(manifest["conditioning_sha256"]) != 64:
        raise ValueError("prepare manifest is missing conditioning SHA-256")
    if manifest["conditioning_sha256"] != _sha256(conditionals_path):
        raise ValueError("prepare manifest conditionals SHA-256 does not match current cache")
    return manifest, joined, conditionals_path, cache, manifest_path


def _load_conditionals_payload(torch: Any, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("t3"), Mapping) or not isinstance(payload.get("gen"), Mapping):
        raise ValueError("conditionals cache must contain {t3, gen} dictionaries")
    payload = copy.deepcopy(dict(payload))
    gen = payload["gen"]
    embedding = gen.get("embedding")
    if not torch.is_tensor(embedding) or tuple(embedding.shape) != (1, EMBEDDING_DIM):
        raise ValueError(f"gen.embedding must have shape (1,{EMBEDDING_DIM})")
    if not torch.isfinite(embedding).all():
        raise ValueError("gen.embedding contains NaN or infinity")
    for key in ("prompt_token", "prompt_token_len", "prompt_feat"):
        if key not in gen or not torch.is_tensor(gen[key]):
            raise ValueError(f"conditionals cache is missing tensor gen[{key!r}]")
    if tuple(gen["prompt_token"].shape)[0] != 1 or tuple(gen["prompt_feat"].shape)[0] != 1 or tuple(gen["prompt_feat"].shape[-1:]) != (MEL_BANDS,):
        raise ValueError("conditionals prompt tensors have invalid shapes")
    prompt_tokens = int(gen["prompt_token"].shape[1])
    prompt_feats = int(gen["prompt_feat"].shape[1])
    prompt_token_len = int(gen["prompt_token_len"].reshape(-1)[0].item())
    # Native Conditionals can store None here. Flow inference uses the actual
    # feature shape; it does not require an explicit feature-length tensor.
    feature_length = gen.get("prompt_feat_len")
    if feature_length is not None and not torch.is_tensor(feature_length):
        raise ValueError("gen.prompt_feat_len must be a tensor or None")
    prompt_feat_len = prompt_feats if feature_length is None else int(feature_length.reshape(-1)[0].item())
    if prompt_tokens != prompt_token_len or prompt_feats != prompt_feat_len or prompt_feats - 2 * prompt_tokens not in {0, 1}:
        raise ValueError(
            f"prompt token/feature lengths are not paired: token={prompt_tokens}/{prompt_token_len}, feat={prompt_feats}/{prompt_feat_len}"
        )
    return payload


def _stream_s3gen_modules(torch: Any, model_dir: Path, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Load only the flow encoder and meanflow estimator from S3Gen."""

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from onnx_staged import (
        _build_flow_encoder,
        _build_meanflow_estimator,
        _repair_encoder_meta_buffers,
        _stream_load,
    )

    checkpoint = model_dir / "s3gen_meanflow.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    flow_encoder = _build_flow_encoder()
    flow_encoder.to_empty(device=device)
    flow_report = _stream_load(
        flow_encoder,
        checkpoint,
        prefixes=(("flow.input_embedding", "input_embedding"), ("flow.encoder", "encoder"), ("flow.encoder_proj", "encoder_proj")),
    )
    _repair_encoder_meta_buffers(flow_encoder, device, torch.float32)
    estimator = _build_meanflow_estimator()
    estimator.to_empty(device=device)
    estimator_report = _stream_load(
        estimator,
        checkpoint,
        prefixes=(("flow.spk_embed_affine_layer", "spk_embed_affine_layer"), ("flow.decoder.estimator", "estimator")),
    )
    flow_encoder.eval()
    estimator.eval()
    for module in (flow_encoder, estimator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "flow_encoder": flow_report,
        "meanflow_estimator": estimator_report,
        "loaded_components": ["flow.input_embedding", "flow.encoder", "flow.encoder_proj", "flow.spk_embed_affine_layer", "flow.decoder.estimator"],
        "omitted_components": ["T3", "S3Gen tokenizer", "S3Gen speaker encoder", "HiFT vocoder"],
        "frozen": True,
    }
    return flow_encoder, estimator, report


def _prepare_flow_row(torch: Any, flow_encoder: Any, payload: Mapping[str, Any], row: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Build frozen ``mu``, ``mask``, ``cond``, and fixed noise for one row."""

    from chatterbox.models.s3gen.utils.mask import make_pad_mask

    gen = payload["gen"]
    prompt_token = gen["prompt_token"].to(device=device, dtype=torch.long)
    prompt_token_len = gen["prompt_token_len"].to(device=device, dtype=torch.long).reshape(1)
    prompt_feat = gen["prompt_feat"].to(device=device, dtype=torch.float32)
    target_tokens_np = _append_silence_tokens(np.asarray(row["tokens"], dtype=np.int64))
    tokens = torch.from_numpy(target_tokens_np).to(device=device, dtype=torch.long).view(1, -1)
    token_len = torch.tensor([tokens.shape[1]], device=device, dtype=torch.long)
    full_tokens = torch.cat((prompt_token, tokens), dim=1)
    full_len = prompt_token_len + token_len
    with torch.no_grad():
        hidden, hidden_masks = flow_encoder(full_tokens, full_len)
        mu = hidden.transpose(1, 2).contiguous()
        hidden_lengths = hidden_masks.sum(dim=-1).squeeze(dim=-1)
        mask = (~make_pad_mask(hidden_lengths)).unsqueeze(1).to(hidden)
        prompt_len = int(prompt_feat.shape[1])
        if mu.shape[2] < prompt_len:
            raise ValueError(f"flow output is shorter than prompt features for row {row['id']}")
        cond = torch.zeros((1, mu.shape[2], MEL_BANDS), device=device, dtype=mu.dtype)
        cond[:, :prompt_len, :] = prompt_feat
        cond = cond.transpose(1, 2).contiguous()
    generated_frames = int(mu.shape[2] - prompt_len)
    prompt_frame_offset = prompt_len - 2 * int(prompt_token_len.item())
    if prompt_frame_offset not in {0, 1}:
        raise ValueError(f"unsupported reference mel/token frame offset: {prompt_frame_offset}")
    expected_generated = 2 * int(target_tokens_np.size) - prompt_frame_offset
    if generated_frames != expected_generated:
        raise ValueError(
            f"flow generated frame geometry for row {row['id']} is {generated_frames}; expected {expected_generated}"
        )
    return {
        "id": row["id"],
        "mu": mu.detach(),
        "mask": mask.detach(),
        "cond": cond.detach(),
        "prompt_len": prompt_len,
        "prompt_frame_offset": prompt_frame_offset,
        "target_token_count": int(target_tokens_np.size),
        "source_token_count": int(np.asarray(row["tokens"]).size),
        "seed": int(row["seed"]),
        "source_frames": int(row["source_frames"]),
        "target": torch.from_numpy(np.asarray(row["source_mel"], dtype=np.float32)).to(device=device).unsqueeze(0),
    }


def _fixed_noise(torch: Any, row_state: Mapping[str, Any], *, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # S3Token2Wav.flow_inference draws target noise BEFORE the CFM's full
    # prompt+target draw. CFM then replaces the target suffix with that first
    # draw. Both draws and their order are required for native seed parity.
    mu = row_state["mu"]
    prompt_len = int(row_state["prompt_len"])
    # Native draws 2*speech_tokens even when an odd reference mel length
    # makes the returned target suffix one frame shorter. CFM splices that
    # draw into the end of mu, including the last prompt frame in that case.
    target_len = 2 * int(row_state["target_token_count"]) if "target_token_count" in row_state else int(mu.shape[2]) - prompt_len
    if target_len <= 0:
        raise ValueError("flow noise needs a non-empty target suffix")
    target_noise = torch.randn(mu.shape[0], mu.shape[1], target_len,
                               device=mu.device, dtype=mu.dtype)
    noise = torch.randn_like(mu)
    noise[:, :, -target_len:] = target_noise
    return noise


def _basic_euler(torch: Any, estimator: Any, row_state: Mapping[str, Any], embedding: Any, noise: Any, *, steps: int) -> torch.Tensor:
    """Differentiable equivalent of CausalConditionalCFM.basic_euler."""

    import torch.nn.functional as F

    if int(steps) != 2:
        raise ValueError("decoder embedding fit is fixed to two meanflow Euler steps")
    mu = row_state["mu"]
    mask = row_state["mask"]
    cond = row_state["cond"]
    x = noise
    t_span = torch.linspace(0.0, 1.0, int(steps) + 1, device=mu.device, dtype=mu.dtype)
    # Match CausalConditionalCFM.basic_euler exactly: normalize the 192-D
    # embedding, project to the estimator's 80-D speaker vector, and retain
    # gradients only through that path.
    spks = estimator.spk_embed_affine_layer(F.normalize(embedding, dim=1))
    for t, r in zip(t_span[:-1], t_span[1:]):
        t_one = t[None]
        r_one = r[None]
        dxdt = estimator.estimator.forward(
            x,
            mask=mask,
            mu=mu,
            t=t_one,
            spks=spks,
            cond=cond,
            r=r_one,
        )
        x = x + (r_one - t_one) * dxdt
    return x


def _project_embedding_to_cap(torch: Any, embedding: Any, baseline_unit: Any, baseline_norm: float, min_cosine: float) -> tuple[float, bool]:
    """Project a 192-D embedding onto a norm-preserving spherical cap."""

    if not 0.0 < float(min_cosine) <= 1.0:
        raise ValueError("min_cosine must be in (0,1]")
    if not torch.isfinite(embedding).all() or not torch.isfinite(baseline_unit).all():
        raise ValueError("embedding projection received non-finite values")
    with torch.no_grad():
        unit = torch.nn.functional.normalize(embedding, dim=1)
        cosine = float(torch.nn.functional.cosine_similarity(unit, baseline_unit, dim=1).detach().cpu())
        projected = cosine < float(min_cosine)
        if projected:
            orthogonal = unit - cosine * baseline_unit
            orth_norm = float(torch.linalg.vector_norm(orthogonal).detach().cpu())
            if orth_norm <= 1e-12:
                unit = baseline_unit
            else:
                tangent = orthogonal / orth_norm
                unit = float(min_cosine) * baseline_unit + math.sqrt(max(0.0, 1.0 - float(min_cosine) ** 2)) * tangent
            unit = torch.nn.functional.normalize(unit, dim=1)
            embedding.copy_(unit * float(baseline_norm))
            cosine = float(torch.nn.functional.cosine_similarity(unit, baseline_unit, dim=1).detach().cpu())
        else:
            embedding.copy_(unit * float(baseline_norm))
    return cosine, projected


def _aligned_prediction(predicted: Any, row_state: Mapping[str, Any]) -> Any:
    generated_frames = int(predicted.shape[2])
    source_frames = int(row_state["source_frames"])
    extra = generated_frames - source_frames
    offset = int(row_state.get("prompt_frame_offset", 0))
    if offset not in {0, 1} or extra not in {6 - offset, 7 - offset}:
        raise ValueError(f"generated/source frame trim is {extra}; inconsistent with reference offset {offset}")
    if "source_token_count" in row_state:
        expected = 2 * (int(row_state["source_token_count"]) + 3) - offset
        if generated_frames != expected:
            raise ValueError(f"generated frame count {generated_frames} differs from exact native geometry {expected}")
    return predicted[:, :, :source_frames]


def _loss(torch: Any, predicted: Any, target: Any, embedding: Any, baseline_embedding: Any, *, envelope_weight: float, regularizer: float) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    raw = F.mse_loss(predicted, target)
    pred_env = predicted.mean(dim=2)
    target_env = target.mean(dim=2)
    envelope = F.mse_loss(pred_env, target_env)
    current_unit = F.normalize(embedding, dim=1)
    base_unit = F.normalize(baseline_embedding, dim=1)
    cosine = F.cosine_similarity(current_unit, base_unit, dim=1).mean()
    penalty = regularizer * (1.0 - cosine)
    total = raw + float(envelope_weight) * envelope + penalty
    return total, {
        "total": float(total.detach().cpu()),
        "raw_mse": float(raw.detach().cpu()),
        "envelope_mse": float(envelope.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "regularization": float(penalty.detach().cpu()),
    }


def _evaluate(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], embedding: Any, baseline_embedding: Any, noises: Sequence[Any], *, steps: int, envelope_weight: float, regularizer: float) -> dict[str, Any]:
    rows = []
    for state, noise in zip(states, noises):
        with torch.no_grad():
            generated = _basic_euler(torch, estimator, state, embedding, noise, steps=steps)
            predicted = _aligned_prediction(generated[:, :, state["prompt_len"]:], state)
            _, metrics = _loss(torch, predicted, state["target"], embedding, baseline_embedding, envelope_weight=envelope_weight, regularizer=regularizer)
        metrics["id"] = state["id"]
        rows.append(metrics)
    keys = ("total", "raw_mse", "envelope_mse", "cosine", "regularization")
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return {"mean": mean, "rows": rows}


def _parity(torch: Any, estimator: Any, states: Sequence[Mapping[str, Any]], baseline_embedding: Any, noises: Sequence[Any], prepared: Mapping[str, Mapping[str, Any]], *, steps: int, atol: float) -> dict[str, Any]:
    rows = []
    failures = []
    for state, noise in zip(states, noises):
        with torch.no_grad():
            generated = _basic_euler(torch, estimator, state, baseline_embedding, noise, steps=steps)
            generated = generated[:, :, state["prompt_len"]:]
        got = generated.detach().float().cpu().numpy()
        expected = np.asarray(prepared[state["id"]], dtype=np.float32)
        if got.shape != expected.shape:
            failures.append({"id": state["id"], "reason": "shape", "actual": list(got.shape), "expected": list(expected.shape)})
            continue
        delta = got.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        row = {"id": state["id"], "max_abs": max_abs, "rmse": rmse, "shape": list(got.shape), "passed": bool(max_abs <= atol)}
        rows.append(row)
        if not row["passed"]:
            failures.append({"id": state["id"], "reason": "numeric", "max_abs": max_abs, "rmse": rmse, "atol": atol})
    return {"status": "passed" if not failures else "failed", "atol": float(atol), "rows": rows, "failures": failures, "max_abs": max((row["max_abs"] for row in rows), default=None), "rmse_mean": float(np.mean([row["rmse"] for row in rows])) if rows else None}


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    fit_started = time.perf_counter()
    if args.steps != DEFAULT_STEPS:
        raise ValueError("--steps must equal 2 for the native meanflow alignment")
    if not 1 <= args.max_steps <= 200:
        raise ValueError("--max-steps must be in 1..200")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    if args.lr <= 0 or args.regularizer < 0 or args.envelope_weight < 0:
        raise ValueError("learning rate must be positive; regularizers must be non-negative")
    if not 0.0 < args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in (0,1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    prepare_dir = _resolve(args.prepare_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows, conditionals_path, cache, manifest_path = _validate_prepare_inputs(prepare_dir)
    if args.seed is not None:
        for index, row in enumerate(rows):
            expected_seed = int(args.seed) + index * 1009
            if int(row["seed"]) != expected_seed:
                raise ValueError(
                    f"requested --seed {args.seed} does not match prepared native seed for {row['id']}: "
                    f"{row['seed']} != {expected_seed}"
                )
    payload = _load_conditionals_payload(torch, conditionals_path)
    baseline_embedding = payload["gen"]["embedding"].detach().to(device=device, dtype=torch.float32)
    baseline_norm = float(torch.linalg.vector_norm(baseline_embedding).detach().cpu())
    if not math.isfinite(baseline_norm) or baseline_norm <= 1e-8:
        raise ValueError("baseline embedding norm is zero or non-finite")
    baseline_unit = torch.nn.functional.normalize(baseline_embedding, dim=1)
    model_dir = _resolve(args.model_dir)
    rss_before_load = _rss_bytes()
    flow_encoder, estimator, loader_report = _stream_s3gen_modules(torch, model_dir, device)
    torch.set_grad_enabled(True)
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    all_states: list[dict[str, Any]] = []
    prepared_recon: dict[str, np.ndarray] = {}
    for row in rows:
        state = _prepare_flow_row(torch, flow_encoder, payload, row, device)
        all_states.append(state)
        prepared_recon[row["id"]] = np.asarray(row["prepared_reconstructed_mel"], dtype=np.float32)[None, ...]
    states_by_id = {state["id"]: state for state in all_states}
    train_states = [states_by_id[row["id"]] for row in train_rows]
    valid_states = [states_by_id[row["id"]] for row in valid_rows]
    row_seeds = {row["id"]: int(row["seed"]) for row in rows}
    noises = {row_id: _fixed_noise(torch, state, seed=seed) for row_id, state, seed in ((state["id"], state, row_seeds[state["id"]]) for state in all_states)}
    parity = _parity(torch, estimator, all_states, baseline_embedding, [noises[state["id"]] for state in all_states], prepared_recon, steps=args.steps, atol=args.parity_atol)
    report_base: dict[str, Any] = {
        "format": "nano_decoder_embedding_fit_v1",
        "status": "parity_failed" if parity["status"] != "passed" else "parity_passed",
        "diagnostic_only": True,
        "experimental": True,
        "prepare_dir": str(prepare_dir),
        "prepare_manifest": str(manifest_path),
        "prepare_manifest_sha256": _sha256(manifest_path),
        "cache": str(_resolve(manifest["cache"])),
        "cache_sha256": _sha256(_resolve(manifest["cache"])),
        "conditionals_path": str(conditionals_path),
        "conditionals_sha256": _sha256(conditionals_path),
        "model_dir": str(model_dir),
        "model_checkpoint_sha256": _sha256(model_dir / "s3gen_meanflow.safetensors"),
        "device": str(device),
        "loader": loader_report,
        "train_rows": [row["id"] for row in train_rows],
        "valid_rows": [row["id"] for row in valid_rows],
        "target_alignment": {
            "speech_tokens_plus_s3gen_sil": 3,
            "source_frames": "2*N or 2*N-1",
            "generated_frames": "2*(N+3) minus reference mel/token offset (0 or 1)",
            "trim_policy": "crop native output to source frames; account for three silence tokens, source framing tail, and measured odd reference frame; no time warp/resampling/DTW",
        },
        "baseline_embedding": {"shape": list(baseline_embedding.shape), "norm": baseline_norm, "cosine_to_itself": 1.0},
        "parity": parity,
        "hyperparameters": {
            "steps": int(args.steps),
            "max_steps": int(args.max_steps),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "regularizer": float(args.regularizer),
            "envelope_weight": float(args.envelope_weight),
            "min_cosine": float(args.min_cosine),
            "requested_seed": args.seed,
        },
        "native_row_seeds": row_seeds,
        "command": list(sys.argv),
        "elapsed_seconds": time.perf_counter() - fit_started,
        "rss_before_load_bytes": rss_before_load,
        "peak_rss_bytes": _peak_rss_bytes(),
        "history": [],
        "disclosure": "Decoder embedding fit uses source-token reconstruction targets. It is not text generation and does not establish zero-shot voice cloning quality.",
    }
    _write_json(output_dir / "fit_report.json", report_base)
    if parity["status"] != "passed":
        return report_base
    if args.parity_only:
        report_base["status"] = "parity_passed"
        report_base["parity_only"] = True
        report_base["elapsed_seconds"] = time.perf_counter() - fit_started
        _write_json(output_dir / "fit_report.json", report_base)
        return report_base

    embedding = torch.nn.Parameter(baseline_embedding.detach().clone())
    optimizer = torch.optim.AdamW([embedding], lr=float(args.lr), weight_decay=0.0)
    initial_valid = _evaluate(torch, estimator, valid_states, embedding, baseline_embedding, [noises[state["id"]] for state in valid_states], steps=args.steps, envelope_weight=args.envelope_weight, regularizer=args.regularizer)
    initial_train = _evaluate(torch, estimator, train_states, embedding, baseline_embedding, [noises[state["id"]] for state in train_states], steps=args.steps, envelope_weight=args.envelope_weight, regularizer=args.regularizer)
    report_base["epoch0"] = {"train": initial_train, "valid": initial_valid}
    report_base["elapsed_seconds"] = time.perf_counter() - fit_started
    _write_json(output_dir / "fit_report.json", report_base)
    best_valid = float(initial_valid["mean"]["total"])
    best_embedding = embedding.detach().clone()
    best_step = 0
    _atomic_torch_save(
        torch,
        {"format": "nano_decoder_embedding_checkpoint_v1", "embedding": best_embedding.detach().cpu(), "step": 0, "valid": initial_valid["mean"]},
        output_dir / "best_embedding.pt",
    )
    stale = 0
    nonzero_gradient = False
    projection_events = 0
    gradient_probe: dict[str, Any] | None = None
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_rows_metrics = []
        for state in train_states:
            generated = _basic_euler(torch, estimator, state, embedding, noises[state["id"]], steps=args.steps)
            predicted = _aligned_prediction(generated[:, :, state["prompt_len"]:], state)
            loss, metrics = _loss(torch, predicted, state["target"], embedding, baseline_embedding, envelope_weight=args.envelope_weight, regularizer=args.regularizer)
            (loss / len(train_states)).backward()
            train_rows_metrics.append(metrics)
        grad_norm_value = float(torch.linalg.vector_norm(embedding.grad).detach().cpu()) if embedding.grad is not None else 0.0
        nonzero_gradient = nonzero_gradient or grad_norm_value > 1e-10
        if embedding.grad is None or not torch.isfinite(embedding.grad).all():
            raise RuntimeError("embedding gradient is missing or non-finite")
        frozen_grads = [parameter for module in (flow_encoder, estimator) for parameter in module.parameters() if parameter.grad is not None]
        if frozen_grads:
            raise RuntimeError("a frozen flow parameter received a gradient")
        if gradient_probe is None:
            gradient_probe = {
                "row_count": len(train_states),
                "gradient_norm": grad_norm_value,
                "finite": True,
                "nonzero": bool(grad_norm_value > 1e-10),
            }
            report_base["gradient_probe"] = gradient_probe
            report_base["frozen_params_no_grads"] = True
        torch.nn.utils.clip_grad_norm_([embedding], 1.0)
        optimizer.step()
        _, projected = _project_embedding_to_cap(torch, embedding, baseline_unit, baseline_norm, float(args.min_cosine))
        projection_events += int(projected)
        valid = _evaluate(torch, estimator, valid_states, embedding, baseline_embedding, [noises[state["id"]] for state in valid_states], steps=args.steps, envelope_weight=args.envelope_weight, regularizer=args.regularizer)
        train_mean = {key: float(np.mean([row[key] for row in train_rows_metrics])) for key in ("total", "raw_mse", "envelope_mse", "cosine", "regularization")}
        record = {"step": step, "train": {"mean": train_mean, "rows": train_rows_metrics}, "valid": valid, "gradient_norm": grad_norm_value}
        report_base["history"].append(record)
        report_base["elapsed_seconds"] = time.perf_counter() - fit_started
        report_base["status"] = "running"
        _write_json(output_dir / "fit_report.json", report_base)
        score = float(valid["mean"]["total"])
        if score < best_valid - 1e-8:
            best_valid = score
            best_step = step
            best_embedding = embedding.detach().clone()
            stale = 0
            _atomic_torch_save(torch, {"format": "nano_decoder_embedding_checkpoint_v1", "embedding": best_embedding.detach().cpu(), "step": step, "valid": valid["mean"]}, output_dir / "best_embedding.pt")
        else:
            stale += 1
        if stale >= int(args.patience):
            break

    fitted_unit = torch.nn.functional.normalize(best_embedding, dim=1)
    fitted_norm = float(torch.linalg.vector_norm(best_embedding).detach().cpu())
    fitted_cosine = float(torch.nn.functional.cosine_similarity(fitted_unit, baseline_unit, dim=1).detach().cpu())
    if not math.isfinite(fitted_norm) or not math.isfinite(fitted_cosine) or abs(fitted_norm - baseline_norm) > 1e-4 or fitted_cosine < float(args.min_cosine) - 1e-6:
        raise RuntimeError(f"fitted embedding violated norm/cosine constraints: norm={fitted_norm}, cosine={fitted_cosine}")
    final_valid = _evaluate(torch, estimator, valid_states, best_embedding, baseline_embedding, [noises[state["id"]] for state in valid_states], steps=args.steps, envelope_weight=args.envelope_weight, regularizer=args.regularizer)
    report_base.update(
        {
            "status": "fit_complete",
            "best_step": best_step,
            "best_valid_total": best_valid,
            "stopped_after_steps": len(report_base["history"]),
            "nonzero_finite_gradient_observed": nonzero_gradient,
            "gradient_probe": gradient_probe,
            "projection_events": projection_events,
            "fitted_embedding": {"shape": list(best_embedding.shape), "norm": fitted_norm, "cosine_to_baseline": fitted_cosine},
            "final_valid": final_valid,
            "adoption_gate": {"improved_over_epoch0": bool(best_valid < float(initial_valid["mean"]["total"])), "min_cosine": float(args.min_cosine), "status": "experimental_not_promoted"},
            "output_conditionals": str((output_dir / "conditionals.pt").resolve()),
            "checkpoint": str((output_dir / "best_embedding.pt").resolve()),
        }
    )
    output_payload = copy.deepcopy(payload)
    output_payload["gen"]["embedding"] = best_embedding.detach().cpu()
    _atomic_torch_save(torch, output_payload, output_dir / "conditionals.pt")
    report_base["output_conditionals_sha256"] = _sha256(output_dir / "conditionals.pt")
    report_base["peak_rss_bytes"] = _peak_rss_bytes()
    report_base["rss_after_fit_bytes"] = _rss_bytes()
    report_base["elapsed_seconds"] = time.perf_counter() - fit_started
    _write_json(output_dir / "fit_report.json", report_base)
    return report_base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--regularizer", type=float, default=DEFAULT_REG)
    parser.add_argument("--envelope-weight", type=float, default=DEFAULT_ENVELOPE_WEIGHT)
    parser.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument("--parity-atol", type=float, default=DEFAULT_PARITY_ATOL)
    parser.add_argument("--seed", type=int, default=None, help="optional expected prepare seed base; native row seeds are always used")
    parser.add_argument("--parity-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = _fit(args)
    except Exception as exc:
        report_path = output_dir / "fit_report.json"
        try:
            failure = json.loads(report_path.read_text()) if report_path.exists() else {}
        except Exception:
            failure = {}
        failure.update(
            {
                "format": "nano_decoder_embedding_fit_v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_dir": str(output_dir),
                "command": list(sys.argv),
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )
        _write_json(output_dir / "fit_report.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": report.get("status"), "output_dir": str(output_dir), "parity": report.get("parity")}, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"parity_passed", "fit_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
