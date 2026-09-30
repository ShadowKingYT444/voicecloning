"""Bounded, staged ONNX export for the exact Chatterbox-Nano pipeline.

This tool exports one component per process invocation.  A stage owns only the
checkpoint tensors needed by that component, writes a small deterministic
example and its PyTorch output, then releases the model.  ``verify`` is a
separate pure-ONNX-Runtime command, so it does not keep PyTorch weights and an
ORT copy resident at the same time.

The loaders construct modules on ``meta`` and use ``to_empty`` followed by a
tensor-at-a-time safetensors stream.  Dead Nano T3 GPT-2 tables are removed
before destination allocation.  This is required for the desktop's hard
3-GiB model-job limit.

The stages are intentionally explicit:

``t3``
    One unified prefill/decode graph with a dynamic legacy KV cache.  The
    caller assembles Nano's speaker/prompt/text embeddings and performs token
    sampling in Python.
``flow_encoder``
    S3Gen token embedding, upsample-Conformer encoder, and 80-channel
    projection.  It accepts the already concatenated prompt and generated
    token IDs used by S3Gen.
``meanflow_estimator``
    One exact meanflow ConditionalDecoder call.  Noise/state ``x`` and times
    ``t``/``r`` are inputs; the Euler loop and classifier-free guidance stay in
    Python.
``vocoder``
    An attempted HiFT graph.  Current PyTorch ONNX support may reject the
    complex STFT/ISTFT path.  A failed attempt is recorded as unavailable and
    never presented as a usable audio export.
``watermark``
    A manifest-only stage.  Perth is an extension-backed watermark operation,
    so it remains after ONNX vocoding in the PyTorch release path.

No stage substitutes Chatterbox-Turbo weights.  Audio remains incomplete until
the flow, vocoder, and Perth paths have independent numerical evidence.

Examples (run through ``bounded_job.py`` on the shared desktop)::

    .venv-nano/bin/python scripts/nano_lab/onnx_staged.py export \
      --stage t3 --output-dir artifacts/nano_lab/onnx_staged
    .venv-nano/bin/python scripts/nano_lab/onnx_staged.py verify \
      --stage t3 --output-dir artifacts/nano_lab/onnx_staged
    .venv-nano/bin/python scripts/nano_lab/onnx_staged.py status \
      --output-dir artifacts/nano_lab/onnx_staged
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
DEFAULT_CHECKPOINT = ROOT / "models" / "chatterbox-nano"
DEFAULT_OUTPUT = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
STAGES = ("t3", "flow_encoder", "meanflow_estimator", "vocoder", "watermark")
S3GEN_SR = 24000
S3_TOKEN_VOCAB = 6561
S3_INPUT_SIZE = 512
S3_OUTPUT_SIZE = 80
T3_LAYERS = 12
T3_HEADS = 12
T3_HEAD_DIM = 64
T3_HIDDEN = 768


def _ensure_vendor_path() -> None:
    if str(VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(VENDOR_SRC))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    return {"path": str(path.resolve()), "exists": True, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _stage_dir(output_dir: Path, stage: str) -> Path:
    return output_dir.resolve() / stage


def _update_root_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    stages: dict[str, Any] = {}
    for stage in STAGES:
        manifest_path = _stage_dir(output_dir, stage) / "manifest.json"
        if not manifest_path.exists():
            stages[stage] = {"status": "not_attempted"}
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # pragma: no cover - diagnostic path
            stages[stage] = {"status": "invalid_manifest", "error": str(exc)}
        else:
            stages[stage] = {
                "status": manifest.get("status"),
                "graph": manifest.get("graph"),
                "reason": manifest.get("reason"),
                "verification": manifest.get("verification"),
            }
    root_manifest = {
        "schema_version": 1,
        "model": "ResembleAI/chatterbox-nano",
        "source": "local Nano checkpoints; no Turbo substitution",
        "output_dir": str(output_dir),
        "stages": stages,
        "watermark_policy": "Perth remains in the PyTorch post-vocoder path unless a separately verified replacement is added.",
        "updated_unix": time.time(),
    }
    _write_json(output_dir / "manifest.json", root_manifest)
    return root_manifest


def _save_examples(stage_dir: Path, inputs: Mapping[str, Any], outputs: Sequence[Any]) -> dict[str, Any]:
    """Save NumPy examples without serialising Python or torch objects."""

    path = stage_dir / "examples.npz"
    arrays: dict[str, np.ndarray] = {}
    input_names: list[str] = []
    for name, value in inputs.items():
        input_names.append(str(name))
        arrays[f"input.{name}"] = np.asarray(value)
    output_names: list[str] = []
    for index, value in enumerate(outputs):
        name = f"output_{index}"
        output_names.append(name)
        arrays[f"output.{name}"] = np.asarray(value)
    np.savez(path, **arrays)
    return {
        "path": str(path.name),
        "input_names": input_names,
        "output_names": output_names,
        "bytes": path.stat().st_size,
    }


def _save_case_examples(stage_dir: Path, cases: Mapping[str, tuple[Mapping[str, Any], Sequence[Any]]]) -> dict[str, Any]:
    """Save several input/output cases in one NPZ file.

    The unified T3 graph is exercised twice: once with an empty cache and once
    with the cache returned by that prefill.  Keeping the cases explicit avoids
    accidentally comparing one ORT invocation with two concatenated output
    sets.
    """

    path = stage_dir / "examples.npz"
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"path": str(path.name), "cases": {}}
    for case_name, (inputs, outputs) in cases.items():
        input_names: list[str] = []
        for name, value in inputs.items():
            input_names.append(str(name))
            arrays[f"case.{case_name}.input.{name}"] = np.asarray(value)
        output_names: list[str] = []
        for index, value in enumerate(outputs):
            name = f"output_{index}"
            output_names.append(name)
            arrays[f"case.{case_name}.output.{name}"] = np.asarray(value)
        metadata["cases"][case_name] = {"input_names": input_names, "output_names": output_names}
    np.savez(path, **arrays)
    metadata["bytes"] = path.stat().st_size
    return metadata


def _torch_export(
    module: Any,
    args: tuple[Any, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    external_parameters: bool = False,
) -> dict[str, Any]:
    """Export and check one graph, without opening an ORT session."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    module.requires_grad_(False)
    started = time.perf_counter()
    external_report = None
    try:
        if external_parameters:
            from onnx_external_export import export_external_parameters
            external_report = export_external_parameters(
                module, args, path, runtime_input_names=input_names,
                output_names=output_names, dynamic_axes=dynamic_axes, opset=opset,
            )
        else:
            torch.onnx.export(
                module,
                args,
                str(path),
                opset_version=int(opset),
                dynamo=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                # ORT can fold constants after the PyTorch process has exited.
                # Folding here duplicates large CPU parameter tensors during export.
                do_constant_folding=False,
                external_data=False,
            )
        # Do not load a second full protobuf beside the live PyTorch model.
        # The separate verification process validates the graph with ORT.
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path.name),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "opset": int(opset),
        "requested_inputs": input_names,
        "requested_outputs": output_names,
        "validation": "pending separate ORT verification",
        "export_constant_folding": False,
        "external_parameters": external_report,
    }


def _stream_load(
    module: Any,
    checkpoint: Path,
    *,
    prefixes: Sequence[tuple[str, str]],
    strict: bool = True,
) -> dict[str, Any]:
    """Stream selected safetensors tensors into a small live module.

    ``prefixes`` maps checkpoint prefixes to local module prefixes.  The loader
    does not materialise the complete S3Gen state dictionary.
    """

    from safetensors import safe_open

    persistent = set(module.state_dict().keys())
    targets: dict[str, Any] = {}
    targets.update({name: value for name, value in module.named_parameters() if name in persistent})
    targets.update({name: value for name, value in module.named_buffers() if name in persistent})
    loaded: list[str] = []
    selected: list[str] = []
    skipped: list[str] = []
    unexpected: list[str] = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            local_name: str | None = None
            for checkpoint_prefix, local_prefix in prefixes:
                if name == checkpoint_prefix:
                    local_name = local_prefix
                    break
                marker = checkpoint_prefix.rstrip(".") + "."
                if name.startswith(marker):
                    suffix = name[len(marker):]
                    local_name = local_prefix.rstrip(".") + ("." if local_prefix else "") + suffix
                    break
            if local_name is None:
                skipped.append(name)
                continue
            selected.append(name)
            target = targets.get(local_name)
            if target is None:
                unexpected.append(f"{name} -> {local_name}")
                continue
            source = handle.get_tensor(name)
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"shape mismatch for {name}: checkpoint {tuple(source.shape)} vs module {tuple(target.shape)}"
                )
            with _no_grad():
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded.append(name)
    missing = sorted(set(targets) - {local for checkpoint_name in loaded for local in [_map_checkpoint(checkpoint_name, prefixes)]})
    # The expression above maps only loaded names.  Keep the diagnostic simple
    # and reliable by recomputing local targets from the selected checkpoint
    # names.
    loaded_local = {_map_checkpoint(name, prefixes) for name in loaded}
    missing = sorted(set(targets) - loaded_local)
    if strict and (missing or unexpected):
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected[:8]))
        raise RuntimeError(f"could not stream {checkpoint.name}: " + "; ".join(details))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "selected_tensors": len(selected),
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
    }


def _map_checkpoint(name: str, prefixes: Sequence[tuple[str, str]]) -> str:
    for checkpoint_prefix, local_prefix in prefixes:
        marker = checkpoint_prefix.rstrip(".") + "."
        if name == checkpoint_prefix:
            return local_prefix
        if name.startswith(marker):
            suffix = name[len(marker):]
            return local_prefix.rstrip(".") + ("." if local_prefix else "") + suffix
    return name


class _no_grad:
    """Tiny lazy context to avoid importing torch at module import time."""

    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def _module_device(module: Any, device: Any) -> None:
    """Move parameters and known plain tensors used by S3Gen."""

    module.to(device)
    for child in module.modules():
        plain_pe = getattr(child, "pe", None)
        if plain_pe is not None and hasattr(plain_pe, "to"):
            child.pe = plain_pe.to(device)
        freqs = getattr(child, "freqs_cis", None)
        if freqs is not None and hasattr(freqs, "to"):
            child.freqs_cis = freqs.to(device)
        window = getattr(child, "stft_window", None)
        if window is not None and hasattr(window, "to"):
            child.stft_window = window.to(device)


def _repair_encoder_meta_buffers(module: Any, device: Any, dtype: Any) -> None:
    """Repair deterministic plain tensors after ``to_empty``.

    The S3Gen Conformer keeps ESPnet relative positional encodings as a plain
    tensor, so ``to_empty`` cannot materialise it.  This mirrors the repair in
    ``runtime._repair_runtime_buffers`` without constructing the complete
    S3Gen pipeline for this small staged component.
    """

    import torch

    for child in module.modules():
        if child.__class__.__name__ != "EspnetRelPositionalEncoding":
            continue
        old_pe = getattr(child, "pe", None)
        if torch.is_tensor(old_pe) and old_pe.ndim >= 2:
            max_len = (int(old_pe.shape[1]) + 1) // 2
        else:
            max_len = 5000
        child.pe = None
        child.extend_pe(torch.zeros(1, max_len, device=device, dtype=dtype))


def _save_t3_external_weights(stage_dir: Path, model: Any) -> dict[str, Any]:
    """Save mmap-friendly NumPy tables needed to assemble Nano embeddings.

    The unified transformer graph intentionally accepts ``inputs_embeds``.  A
    separate process can therefore mmap these tables, perform ID lookup and
    speaker projection, and keep the large transformer graph as its only ORT
    session allocation.
    """

    arrays = {
        "speech_embedding": model.speech_emb.weight,
        "text_embedding": model.text_emb.weight,
        "speaker_projection_weight": model.cond_enc.spkr_enc.weight,
        "speaker_projection_bias": model.cond_enc.spkr_enc.bias,
    }
    report: dict[str, Any] = {}
    for name, tensor in arrays.items():
        path = stage_dir / f"{name}.npy"
        values = tensor.detach().to(device="cpu", dtype=getattr(tensor, "dtype", None)).numpy()
        np.save(path, values)
        del values
        report[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return report


def _device_and_dtype(device_name: str) -> tuple[Any, Any]:
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for staged export but torch.cuda.is_available() is false")
    # Staged exports are fp32 by default.  This matches the Nano acoustic
    # runtime and avoids silently changing IDs/embeddings through fp16 casts.
    return device, torch.float32


def _load_t3_stage(checkpoint_dir: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    """Load Nano T3 with meta construction and a tensor-at-a-time stream."""

    import torch

    _ensure_vendor_path()
    # These helpers are the same ones used by the measured optimized runtime:
    # they delete dead GPT-2 tables before destination allocation, stream only
    # matching tensors, and restore non-persistent causal masks.
    from runtime import (
        T3,
        _delete_unused_t3_weights,
        _nano_hp,
        _repair_t3_runtime_buffers,
        _stream_load_safetensors,
    )
    from onnx_t3_core import NanoT3KV

    with torch.device("meta"):
        model = T3(_nano_hp())
    removed = _delete_unused_t3_weights(model)
    model.to(dtype=torch.float32)
    model.to_empty(device=device)
    report = _stream_load_safetensors(
        model,
        checkpoint_dir / "t3_nano_v1.safetensors",
        device=device,
        strict=True,
        skip_prefixes=("tfmr.wte", "text_head"),
    )
    _repair_t3_runtime_buffers(model, device)
    report["removed_parameters"] = removed
    report["optimized_loader"] = "runtime._stream_load_safetensors after meta/to_empty"
    wrapper = NanoT3KV(model).eval()
    return wrapper, report


def _build_flow_encoder() -> Any:
    import torch
    import torch.nn as nn

    _ensure_vendor_path()
    from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
    from chatterbox.models.s3gen.utils.mask import make_pad_mask

    class FlowEncoderStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embedding = nn.Embedding(S3_TOKEN_VOCAB, S3_INPUT_SIZE)
            self.encoder = UpsampleConformerEncoder(
                output_size=512,
                attention_heads=8,
                linear_units=2048,
                num_blocks=6,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer="linear",
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                input_size=512,
                use_cnn_module=False,
                macaron_style=False,
            )
            self.encoder_proj = nn.Linear(512, S3_OUTPUT_SIZE)

        def forward(self, speech_tokens, token_lengths):
            mask = (~make_pad_mask(token_lengths, speech_tokens.size(1))).unsqueeze(-1).to(self.input_embedding.weight)
            embedded = self.input_embedding(torch.clamp(speech_tokens, min=0)) * mask
            hidden, hidden_masks = self.encoder(embedded, token_lengths)
            return self.encoder_proj(hidden), hidden_masks

    with torch.device("meta"):
        return FlowEncoderStage().eval()


def _load_flow_encoder_stage(checkpoint_dir: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    model = _build_flow_encoder()
    import torch

    model.to_empty(device=device)
    report = _stream_load(
        model,
        checkpoint_dir / "s3gen_meanflow.safetensors",
        prefixes=(
            ("flow.input_embedding", "input_embedding"),
            ("flow.encoder", "encoder"),
            ("flow.encoder_proj", "encoder_proj"),
        ),
    )
    _repair_encoder_meta_buffers(model, device, torch.float32)
    return model, report


def _build_meanflow_estimator() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _ensure_vendor_path()
    from chatterbox.models.s3gen.configs import CFM_PARAMS
    from chatterbox.models.s3gen.decoder import ConditionalDecoder

    class MeanflowEstimatorStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.spk_embed_affine_layer = nn.Linear(192, 80)
            self.estimator = ConditionalDecoder(
                in_channels=320,
                out_channels=80,
                causal=True,
                channels=[256],
                dropout=0.0,
                attention_head_dim=64,
                n_blocks=4,
                num_mid_blocks=12,
                num_heads=8,
                act_fn="gelu",
                meanflow=True,
            )

        def forward(self, x, mask, mu, t, speaker_embedding, cond, r):
            spks = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))
            return self.estimator(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)

    with torch.device("meta"):
        return MeanflowEstimatorStage().eval()


def _load_meanflow_estimator(checkpoint_dir: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    model = _build_meanflow_estimator()
    model.to_empty(device=device)
    report = _stream_load(
        model,
        checkpoint_dir / "s3gen_meanflow.safetensors",
        prefixes=(
            ("flow.spk_embed_affine_layer", "spk_embed_affine_layer"),
            ("flow.decoder.estimator", "estimator"),
        ),
    )
    return model, report


def _build_vocoder() -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.nn as nn

    _ensure_vendor_path()
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
    from chatterbox.models.s3gen.hifigan import HiFTGenerator

    class VocoderStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel2wav = HiFTGenerator(
                sampling_rate=S3GEN_SR,
                upsample_rates=[8, 5, 3],
                upsample_kernel_sizes=[16, 11, 7],
                source_resblock_kernel_sizes=[7, 7, 11],
                source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                f0_predictor=ConvRNNF0Predictor(),
            )

        def forward(self, speech_feat, phase_noise, sine_noise):
            # This is the same path as S3Token2Wav.hift_inference with an empty
            # source cache.  Upstream draws phase and source noise internally.
            # They are explicit graph inputs here so ORT and PyTorch can be
            # compared bit-for-bit for a fixed example and seed.
            f0 = self.mel2wav.f0_predictor(speech_feat)
            f0_upsampled = self.mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)
            # Inline SourceModuleHnNSF/SineGen while replacing their random
            # draws with caller-provided tensors.  Keep the original harmonic
            # count and thresholds from the constructed module.
            sine_gen = self.mel2wav.m_source.l_sin_gen
            harmonic_count = sine_gen.harmonic_num + 1
            harmonics = torch.arange(
                1, harmonic_count + 1, device=f0_upsampled.device, dtype=f0_upsampled.dtype
            ).view(1, harmonic_count, 1)
            f_mat = f0_upsampled.transpose(1, 2) * harmonics / sine_gen.sampling_rate
            theta = 2 * float(np.pi) * (torch.cumsum(f_mat, dim=-1) % 1)
            phase = torch.cat([torch.zeros_like(phase_noise[:, :1, :]), phase_noise[:, 1:harmonic_count, :]], dim=1)
            sine_waves = sine_gen.sine_amp * torch.sin(theta + phase)
            uv = (f0_upsampled.transpose(1, 2) > sine_gen.voiced_threshold).to(sine_waves.dtype)
            noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
            sine_waves = sine_waves * uv + noise_amp * sine_noise
            sine_waves = sine_waves.transpose(1, 2)
            sine_merge = self.mel2wav.m_source.l_tanh(self.mel2wav.m_source.l_linear(sine_waves))
            source = sine_merge.transpose(1, 2)
            # Upstream also draws a separate unused noise branch. It has no
            # effect on this waveform and is not a graph input.
            return self.mel2wav.decode(speech_feat, source)

    with torch.device("meta"):
        return VocoderStage().eval()


def _load_vocoder(checkpoint_dir: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    model = _build_vocoder()
    model.to_empty(device=device)
    report = _stream_load(
        model,
        checkpoint_dir / "s3gen_meanflow.safetensors",
        prefixes=(("mel2wav", "mel2wav"),),
    )
    _module_device(model, device)
    return model, report


def _base_manifest(stage: str, checkpoint: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "model": "ResembleAI/chatterbox-nano",
        "checkpoint": _checkpoint_report(checkpoint),
        "source": "local Nano checkpoint; no Turbo substitution",
        "status": "failed",
        "created_unix": time.time(),
        "verification": {"status": "not_run", "command": f"onnx_staged.py verify --stage {stage}"},
    }


def _release(*objects: Any) -> None:
    for value in objects:
        try:
            del value
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def export_t3(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    device_name: str,
    opset: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    stage = "t3"
    stage_dir = _stage_dir(output_dir, stage)
    checkpoint = checkpoint_dir / "t3_nano_v1.safetensors"
    manifest = _base_manifest(stage, checkpoint)
    wrapper = model = None
    try:
        torch.set_num_threads(max(1, min(int(os.environ.get("OMP_NUM_THREADS", "2")), 2)))
        torch.manual_seed(seed)
        device, dtype = _device_and_dtype(device_name)
        wrapper, load_report = _load_t3_stage(checkpoint_dir, device)
        model = wrapper.t3
        inputs_embeds = torch.randn((1, 4, T3_HIDDEN), device=device, dtype=dtype)
        cache_position = torch.arange(4, device=device, dtype=torch.long)
        empty_cache = tuple(torch.zeros((1, T3_HEADS, 0, T3_HEAD_DIM), device=device, dtype=dtype) for _ in range(T3_LAYERS * 2))
        with torch.inference_mode():
            prefill_out = wrapper(inputs_embeds, cache_position, *empty_cache)
        decode_embeds = torch.randn((1, 1, T3_HIDDEN), device=device, dtype=dtype)
        decode_position = torch.tensor([4], device=device, dtype=torch.long)
        with torch.inference_mode():
            decode_out = wrapper(decode_embeds, decode_position, *prefill_out[1:])

        graph = _torch_export(
            wrapper,
            (inputs_embeds, cache_position, *empty_cache),
            stage_dir / "nano_t3_kv.onnx",
            input_names=["inputs_embeds", "cache_position"] + [f"past_{i}" for i in range(T3_LAYERS * 2)],
            output_names=["logits"] + [f"present_{i}" for i in range(T3_LAYERS * 2)],
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "sequence"},
                "cache_position": {0: "position_length"},
                **{f"past_{i}": {0: "batch", 2: "past_length"} for i in range(T3_LAYERS * 2)},
                "logits": {0: "batch"},
                **{f"present_{i}": {0: "batch", 2: "present_length"} for i in range(T3_LAYERS * 2)},
            },
            opset=opset,
        )
        external_weights = _save_t3_external_weights(stage_dir, model)
        examples = _save_case_examples(
            stage_dir,
            {
                "prefill": (
                    {
                        "inputs_embeds": inputs_embeds.detach().cpu().numpy(),
                        "cache_position": cache_position.detach().cpu().numpy(),
                        **{f"past_{i}": value.detach().cpu().numpy() for i, value in enumerate(empty_cache)},
                    },
                    [value.detach().cpu().numpy() for value in prefill_out],
                ),
                "decode": (
                    {
                        "inputs_embeds": decode_embeds.detach().cpu().numpy(),
                        "cache_position": decode_position.detach().cpu().numpy(),
                        **{f"past_{i}": value.detach().cpu().numpy() for i, value in enumerate(prefill_out[1:])},
                    },
                    [value.detach().cpu().numpy() for value in decode_out],
                ),
            },
        )
        manifest.update(
            {
                "status": "exported",
                "device": str(device),
                "dtype": str(dtype),
                "load": load_report,
                "graph": graph,
                "external_weights": external_weights,
                "examples": examples,
                "torch_output_layout": {
                    "prefill": ["output_%d" % i for i in range(1 + T3_LAYERS * 2)],
                    "decode": ["output_%d" % i for i in range(1 + T3_LAYERS * 2)],
                },
                "runtime_contract": "Caller mmap-loads the external embedding/projection .npy tables, assembles exact Nano embeddings, and performs Python sampling; zero-length past selects prefill.",
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        _release(wrapper, model)
    _write_json(stage_dir / "manifest.json", manifest)
    _update_root_manifest(output_dir)
    return manifest


def export_flow_encoder(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    device_name: str,
    opset: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    stage = "flow_encoder"
    stage_dir = _stage_dir(output_dir, stage)
    checkpoint = checkpoint_dir / "s3gen_meanflow.safetensors"
    manifest = _base_manifest(stage, checkpoint)
    model = None
    try:
        torch.set_num_threads(max(1, min(int(os.environ.get("OMP_NUM_THREADS", "2")), 2)))
        torch.manual_seed(seed)
        device, dtype = _device_and_dtype(device_name)
        model, load_report = _load_flow_encoder_stage(checkpoint_dir, device)
        tokens = torch.randint(0, S3_TOKEN_VOCAB, (1, 12), device=device, dtype=torch.long)
        lengths = torch.tensor([12], device=device, dtype=torch.long)
        with torch.inference_mode():
            torch_out = model(tokens, lengths)
        graph = _torch_export(
            model,
            (tokens, lengths),
            stage_dir / "nano_flow_encoder.onnx",
            input_names=["speech_tokens", "token_lengths"],
            output_names=["mu", "mask"],
            external_parameters=True,
            dynamic_axes={
                "speech_tokens": {0: "batch", 1: "token_length"},
                "token_lengths": {0: "batch"},
                "mu": {0: "batch", 1: "mel_length"},
                "mask": {0: "batch", 2: "mel_length"},
            },
            opset=opset,
        )
        cases = {"tokens12": (
            {"speech_tokens": tokens.detach().cpu().numpy(), "token_lengths": lengths.detach().cpu().numpy()},
            [value.detach().cpu().numpy() for value in torch_out],
        )}
        for token_length in (120, 600):
            sample_tokens = torch.randint(0, S3_TOKEN_VOCAB, (1, token_length), device=device, dtype=torch.long)
            sample_lengths = torch.tensor([token_length-3 if token_length==120 else token_length],device=device,dtype=torch.long)
            with torch.inference_mode():
                expected = model(sample_tokens,sample_lengths)
            cases[f"tokens{token_length}"] = (
                {"speech_tokens":sample_tokens.cpu().numpy(),"token_lengths":sample_lengths.cpu().numpy()},
                [value.cpu().numpy() for value in expected],
            )
        examples = _save_case_examples(stage_dir,cases)
        manifest.update(
            {
                "status": "exported",
                "device": str(device),
                "dtype": str(dtype),
                "load": load_report,
                "graph": graph,
                "examples": examples,
                "runtime_contract": "Input tokens are the concatenated S3Gen prompt and generated tokens; output mu is (B,mel_length,80), mask is (B,1,mel_length).",
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        _release(model)
    _write_json(stage_dir / "manifest.json", manifest)
    _update_root_manifest(output_dir)
    return manifest


def export_meanflow_estimator(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    device_name: str,
    opset: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    stage = "meanflow_estimator"
    stage_dir = _stage_dir(output_dir, stage)
    checkpoint = checkpoint_dir / "s3gen_meanflow.safetensors"
    manifest = _base_manifest(stage, checkpoint)
    model = None
    try:
        torch.set_num_threads(max(1, min(int(os.environ.get("OMP_NUM_THREADS", "2")), 2)))
        torch.manual_seed(seed)
        device, dtype = _device_and_dtype(device_name)
        model, load_report = _load_meanflow_estimator(checkpoint_dir, device)
        length = 8
        x = torch.randn((1, 80, length), device=device, dtype=dtype)
        mask = torch.ones((1, 1, length), device=device, dtype=dtype)
        mu = torch.randn((1, 80, length), device=device, dtype=dtype)
        t = torch.tensor([0.25], device=device, dtype=dtype)
        speaker = torch.randn((1, 192), device=device, dtype=dtype)
        cond = torch.randn((1, 80, length), device=device, dtype=dtype)
        r = torch.tensor([0.75], device=device, dtype=dtype)
        args = (x, mask, mu, t, speaker, cond, r)
        with torch.inference_mode():
            torch_out = (model(*args),)
        graph = _torch_export(
            model,
            args,
            stage_dir / "nano_meanflow_estimator.onnx",
            input_names=["x", "mask", "mu", "t", "speaker_embedding", "cond", "r"],
            output_names=["dxdt"],
            external_parameters=True,
            dynamic_axes={
                "x": {0: "batch", 2: "mel_length"},
                "mask": {0: "batch", 2: "mel_length"},
                "mu": {0: "batch", 2: "mel_length"},
                "t": {0: "batch"},
                "speaker_embedding": {0: "batch"},
                "cond": {0: "batch", 2: "mel_length"},
                "r": {0: "batch"},
                "dxdt": {0: "batch", 2: "mel_length"},
            },
            opset=opset,
        )
        input_keys = ["x", "mask", "mu", "t", "speaker_embedding", "cond", "r"]
        cases = {"mel8": (
            {name: value.detach().cpu().numpy() for name, value in zip(input_keys, args)},
            [torch_out[0].detach().cpu().numpy()],
        )}
        for length in (64, 256):
            values = (
                torch.randn((1,80,length),device=device,dtype=dtype),
                torch.ones((1,1,length),device=device,dtype=dtype),
                torch.randn((1,80,length),device=device,dtype=dtype),
                t, speaker,
                torch.randn((1,80,length),device=device,dtype=dtype), r,
            )
            with torch.inference_mode():
                expected = model(*values)
            cases[f"mel{length}"] = (
                {name:value.cpu().numpy() for name,value in zip(input_keys,values)},
                [expected.cpu().numpy()],
            )
        examples = _save_case_examples(stage_dir, cases)
        manifest.update(
            {
                "status": "exported",
                "device": str(device),
                "dtype": str(dtype),
                "load": load_report,
                "graph": graph,
                "examples": examples,
                "runtime_contract": "One deterministic meanflow estimator call. x is an explicit state/noise input; Python owns Euler steps and CFG.",
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        _release(model)
    _write_json(stage_dir / "manifest.json", manifest)
    _update_root_manifest(output_dir)
    return manifest


def export_vocoder(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    device_name: str,
    opset: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    stage = "vocoder"
    stage_dir = _stage_dir(output_dir, stage)
    checkpoint = checkpoint_dir / "s3gen_meanflow.safetensors"
    manifest = _base_manifest(stage, checkpoint)
    model = None
    try:
        torch.set_num_threads(max(1, min(int(os.environ.get("OMP_NUM_THREADS", "2")), 2)))
        torch.manual_seed(seed)
        device, dtype = _device_and_dtype(device_name)
        model, load_report = _load_vocoder(checkpoint_dir, device)
        # A short but non-degenerate frame sequence keeps the export example
        # bounded.  Dynamic mel length is declared in the graph contract.
        mel_length = 12
        source_length = mel_length * 120 * 4
        speech_feat = torch.randn((1, 80, mel_length), device=device, dtype=dtype)
        phase_noise = torch.randn((1, 9, 1), device=device, dtype=dtype)
        sine_noise = torch.randn((1, 9, source_length), device=device, dtype=dtype)
        with torch.inference_mode():
            native_out = model(speech_feat, phase_noise, sine_noise)
            from onnx_spectral import build_hift_spectral_stage
            from types import MethodType
            model.mel2wav.real_analysis = build_hift_spectral_stage("stft").to(device)
            model.mel2wav.real_synthesis = build_hift_spectral_stage("istft").to(device)
            model.mel2wav._stft = MethodType(lambda module, x: module.real_analysis(x), model.mel2wav)
            model.mel2wav._istft = MethodType(lambda module, magnitude, phase: module.real_synthesis(magnitude, phase), model.mel2wav)
            replacement_out = model(speech_feat, phase_noise, sine_noise)
            spectral_max_error = float((native_out-replacement_out).abs().max())
            torch.testing.assert_close(replacement_out,native_out,atol=3e-4,rtol=3e-4)
            torch_out = (native_out,)
        graph = _torch_export(
            model,
            (speech_feat, phase_noise, sine_noise),
            stage_dir / "nano_hift_vocoder.onnx",
            input_names=["speech_feat", "phase_noise", "sine_noise"],
            output_names=["audio"],
            dynamic_axes={
                "speech_feat": {0: "batch", 2: "mel_length"},
                "phase_noise": {0: "batch"},
                "sine_noise": {0: "batch", 2: "source_length"},
                "audio": {0: "batch", 1: "audio_length"},
            },
            opset=opset,
        )
        examples = _save_examples(
            stage_dir,
            {
                "speech_feat": speech_feat.detach().cpu().numpy(),
                "phase_noise": phase_noise.detach().cpu().numpy(),
                "sine_noise": sine_noise.detach().cpu().numpy(),
            },
            [torch_out[0].detach().cpu().numpy()],
        )
        manifest.update(
            {
                "status": "exported",
                "device": str(device),
                "dtype": str(dtype),
                "load": load_report,
                "graph": graph,
                "examples": examples,
                "spectral_replacement_vs_native_max_abs": spectral_max_error,
                "runtime_contract": "Inputs are S3Gen mel features plus explicit phase/source noise; output is unwatermarked waveform. Apply Perth in the release path.",
            }
        )
    except Exception as exc:
        # STFT/ISTFT exporter gaps are an expected, honest outcome for this
        # optional stage.  Keep the exact exception for a future implementation.
        manifest.update(
            {
                "status": "not_exported",
                "reason": f"HiFT ONNX export unavailable: {type(exc).__name__}: {exc}",
                "blocker": "The vocoder contains complex torch.stft/torch.istft and a source-filter path; no graph is released without parity evidence.",
            }
        )
    finally:
        _release(model)
    _write_json(stage_dir / "manifest.json", manifest)
    _update_root_manifest(output_dir)
    return manifest


def export_watermark(output_dir: Path) -> dict[str, Any]:
    stage = "watermark"
    stage_dir = _stage_dir(output_dir, stage)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "model": "ResembleAI/chatterbox-nano",
        "status": "not_exported",
        "reason": "Perth watermark is an extension-backed post-processing operation, not a neural ONNX graph.",
        "runtime_contract": "Keep Perth watermarking in the PyTorch post-vocoder path; do not ship unwatermarked audio as a final clone.",
        "created_unix": time.time(),
        "verification": {"status": "not_applicable"},
    }
    _write_json(stage_dir / "manifest.json", manifest)
    _update_root_manifest(output_dir)
    return manifest


def _load_examples(stage_dir: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    examples = manifest.get("examples") or {}
    path = stage_dir / str(examples.get("path", "examples.npz"))
    if not path.exists():
        raise FileNotFoundError(f"staged examples not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        inputs = {name[len("input."):]: np.asarray(data[name]) for name in data.files if name.startswith("input.")}
        outputs = [np.asarray(data[name]) for name in examples.get("output_names", []) for name in [f"output.{name}"]]
        if not outputs:
            outputs = [np.asarray(data[name]) for name in sorted(data.files) if name.startswith("output.")]
    return inputs, outputs


def _load_case_examples(
    stage_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, tuple[dict[str, np.ndarray], list[np.ndarray]]]:
    """Load case-separated examples, such as unified T3 prefill/decode."""

    examples = manifest.get("examples") or {}
    path = stage_dir / str(examples.get("path", "examples.npz"))
    cases = examples.get("cases")
    if not cases:
        raise ValueError("manifest does not contain case-separated examples")
    if not path.exists():
        raise FileNotFoundError(f"staged examples not found: {path}")
    result: dict[str, tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}
    with np.load(path, allow_pickle=False) as data:
        for case_name, metadata in cases.items():
            prefix = f"case.{case_name}."
            inputs = {
                name[len(prefix) + len("input."):]: np.asarray(data[name])
                for name in data.files
                if name.startswith(prefix + "input.")
            }
            outputs: list[np.ndarray] = []
            for output_name in metadata.get("output_names", []):
                key = f"{prefix}output.{output_name}"
                if key not in data:
                    raise KeyError(f"missing saved output {key}")
                outputs.append(np.asarray(data[key]))
            result[str(case_name)] = (inputs, outputs)
    return result


def verify_stage(output_dir: Path, stage: str, *, intra_op_num_threads: int = 1) -> dict[str, Any]:
    """Compare a saved PyTorch example against an ORT CPU session."""

    if stage == "watermark":
        result = {"stage": stage, "status": "not_applicable", "reason": "Perth is intentionally kept in PyTorch."}
        _write_json(_stage_dir(output_dir, stage) / "verification.json", result)
        return result
    stage_dir = _stage_dir(output_dir, stage)
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"staged manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "exported":
        result = {"stage": stage, "status": "not_applicable", "reason": manifest.get("reason", "stage is not exported")}
        _write_json(stage_dir / "verification.json", result)
        return result
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(int(intra_op_num_threads), 2))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    graph_path = Path(manifest["graph"]["path"])
    if not graph_path.is_absolute():
        graph_path = stage_dir / graph_path
    session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
    feed_names = [item.name for item in session.get_inputs()]
    errors: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    examples = manifest.get("examples") or {}
    case_reports: dict[str, Any] = {}
    if examples.get("cases"):
        case_iter = _load_case_examples(stage_dir, manifest).items()
    else:
        case_iter = [("default", _load_examples(stage_dir, manifest))]
    total_outputs = 0
    for case_name, (inputs, expected) in case_iter:
        missing_inputs = sorted(set(feed_names) - set(inputs))
        if missing_inputs:
            errors.append({"case": case_name, "kind": "missing_inputs", "inputs": missing_inputs})
            continue
        actual = session.run(None, {name: inputs[name] for name in feed_names})
        total_outputs += len(actual)
        case_errors_before = len(errors)
        if len(actual) != len(expected):
            errors.append({"case": case_name, "kind": "output_count", "actual": len(actual), "expected": len(expected)})
            continue
        for index, (got, want) in enumerate(zip(actual, expected)):
            got_array = np.asarray(got)
            want_array = np.asarray(want)
            if got_array.shape != want_array.shape:
                errors.append({"case": case_name, "index": index, "kind": "shape", "actual": list(got_array.shape), "expected": list(want_array.shape)})
                continue
            if np.issubdtype(want_array.dtype, np.floating):
                abs_error = float(np.max(np.abs(got_array.astype(np.float64) - want_array.astype(np.float64))))
                rel_error = float(np.max(np.abs(got_array.astype(np.float64) - want_array.astype(np.float64)) / np.maximum(np.abs(want_array.astype(np.float64)), 1e-5)))
                max_abs = max(max_abs, abs_error)
                max_rel = max(max_rel, rel_error)
                if not np.allclose(got_array, want_array, atol=3e-3, rtol=3e-3):
                    errors.append({"case": case_name, "index": index, "kind": "numeric", "max_abs": abs_error, "max_rel": rel_error})
            elif not np.array_equal(got_array, want_array):
                errors.append({"case": case_name, "index": index, "kind": "exact"})
        case_reports[case_name] = {"outputs": len(actual), "errors": len(errors) - case_errors_before}
    result = {
        "stage": stage,
        "status": "verified" if not errors else "mismatch",
        "provider": "CPUExecutionProvider",
        "inputs": feed_names,
        "outputs_per_case": case_reports,
        "outputs": total_outputs,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "errors": errors[:16],
        "method": "separate pure-ORT process against saved PyTorch reference arrays",
    }
    _write_json(stage_dir / "verification.json", result)
    manifest["verification"] = result
    _write_json(manifest_path, manifest)
    _update_root_manifest(output_dir)
    return result


def _export_stage(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    common = {
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "device_name": args.device,
        "opset": args.opset,
        "seed": args.seed,
    }
    if args.stage == "t3":
        return export_t3(**common)
    if args.stage == "flow_encoder":
        return export_flow_encoder(**common)
    if args.stage == "meanflow_estimator":
        return export_meanflow_estimator(**common)
    if args.stage == "vocoder":
        return export_vocoder(**common)
    return export_watermark(output_dir)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="export one bounded stage; run stages in separate bounded jobs")
    exp.add_argument("--stage", choices=STAGES, required=True)
    exp.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    exp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    exp.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    exp.add_argument("--opset", type=int, default=18)
    exp.add_argument("--seed", type=int, default=1234)
    ver = sub.add_parser("verify", help="verify one exported stage in pure ORT CPU")
    ver.add_argument("--stage", choices=STAGES, required=True)
    ver.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ver.add_argument("--intra-op-num-threads", type=int, default=1)
    status = sub.add_parser("status", help="read stage manifests without loading a model")
    status.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "export":
        result = _export_stage(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status")=="exported" or args.stage=="watermark" else 1
    elif args.command == "verify":
        result = verify_stage(args.output_dir, args.stage, intra_op_num_threads=args.intra_op_num_threads)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status")=="verified" or args.stage=="watermark" else 1
    else:
        print(json.dumps(_update_root_manifest(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
