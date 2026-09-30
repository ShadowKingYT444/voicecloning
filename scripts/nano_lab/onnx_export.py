"""Export and verify the exact Chatterbox-Nano T3 decoder to ONNX.

Nano's autoregressive T3 stage is the useful ONNX target for CPU inference.  The
export contains the Nano GPT2-small weights, Nano's speaker and speech
condition projections, the text/speech embeddings, and a legacy per-layer KV
cache.  Python still owns token sampling because that makes top-k/top-p and
seed handling stable across ONNX Runtime versions.

S3Gen is intentionally not silently substituted with Chatterbox-Turbo.  Its
flow matcher and HiFT vocoder are reported as separate follow-up work by this
tool.  The generated speech tokens are therefore a real partial export, not an
end-to-end audio claim.

Examples::

    .venv-nano/bin/python scripts/nano_lab/onnx_export.py export \
      --output-dir artifacts/nano_lab/onnx --opset 18 --device cpu
    .venv-nano/bin/python scripts/nano_lab/onnx_export.py verify \
      --output-dir artifacts/nano_lab/onnx
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = ROOT / "vendor" / "chatterbox" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from chatterbox.models.t3 import T3  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "models" / "chatterbox-nano"
DEFAULT_OUTPUT = ROOT / "artifacts" / "nano_lab" / "onnx"
T3_LAYERS = 12
T3_HEADS = 12
T3_HEAD_DIM = 64
T3_HIDDEN = 768
T3_PROMPT_LEN = 375
T3_SPEECH_VOCAB = 6563
T3_START_SPEECH = 6561


def nano_hp() -> T3Config:
    """Build the same T3 configuration used by ``runtime.NanoEngine``."""

    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_small"
    hp.speech_tokens_dict_size = T3_SPEECH_VOCAB
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = T3_PROMPT_LEN
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    return hp


def _stream_load(module: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    """Load only T3 weights, one safetensors tensor at a time."""

    from safetensors import safe_open

    targets = dict(module.named_parameters())
    # GPT2 causal masks are non-persistent buffers and are absent in the file.
    loaded: list[str] = []
    unexpected: list[str] = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            target = targets.get(name)
            if target is None:
                # The full checkpoint contains two inference-dead modules.  Do
                # not allocate them just to satisfy an export.
                if name.startswith("tfmr.wte.") or name.startswith("text_head."):
                    continue
                unexpected.append(name)
                continue
            source = handle.get_tensor(name)
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(f"shape mismatch for {name}: {tuple(source.shape)} vs {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(source)
            loaded.append(name)
    if unexpected:
        raise RuntimeError("unexpected Nano T3 tensors: " + ", ".join(unexpected[:8]))
    missing = sorted(set(targets) - set(loaded))
    # These are the dead modules, removed below, if they were included in the
    # target map.  Runtime buffers are not parameters and do not appear here.
    return {"loaded_tensors": len(loaded), "missing_parameters": missing[:16]}


def load_t3(checkpoint_dir: Path = DEFAULT_CHECKPOINT) -> tuple[T3, dict[str, Any]]:
    """Load exact Nano T3 on CPU without loading S3Gen or the voice encoder."""

    hp = nano_hp()
    model = T3(hp).eval()
    # The 500 MB text head and GPT2 token embedding are not touched by
    # inference_turbo.  Remove them before loading to reduce RSS and graph
    # size while retaining every inference parameter.
    dead: list[str] = []
    if hasattr(model, "tfmr") and hasattr(model.tfmr, "wte"):
        del model.tfmr.wte
        dead.append("tfmr.wte")
    if hasattr(model, "text_head"):
        del model.text_head
        dead.append("text_head")
    report = _stream_load(model, checkpoint_dir / "t3_nano_v1.safetensors")
    report["removed_parameters"] = dead
    report["checkpoint"] = str((checkpoint_dir / "t3_nano_v1.safetensors").resolve())
    return model, report


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Convert Transformers 4.57 DynamicCache or old tuples to legacy pairs."""

    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    return tuple(cache)


class NanoT3Prefill(torch.nn.Module):
    """Nano T3 prefill: condition IDs and text IDs to logits plus KV cache."""

    def __init__(self, t3: T3):
        super().__init__()
        self.t3 = t3

    def forward(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # This is the exact inference_turbo embedding order for Nano:
        # speaker projection, 375 reference speech embeddings, text, BOS.
        cond_spkr = self.t3.cond_enc.spkr_enc(speaker_emb.reshape(-1, 256))[:, None]
        prompt_emb = self.t3.speech_emb(cond_prompt_speech_tokens.to(dtype=torch.long))
        text_emb = self.t3.text_emb(text_tokens.to(dtype=torch.long))
        bos = torch.full(
            (text_tokens.shape[0], 1),
            T3_START_SPEECH,
            dtype=torch.long,
            device=text_tokens.device,
        )
        bos_emb = self.t3.speech_emb(bos)
        inputs_embeds = torch.cat((cond_spkr, prompt_emb, text_emb, bos_emb), dim=1)
        out = self.t3.tfmr(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
        )
        logits = self.t3.speech_head(out.last_hidden_state[:, -1, :])
        past = _legacy_cache(out.past_key_values)
        return (logits, *[value for pair in past for value in pair])


class NanoT3Decode(torch.nn.Module):
    """Nano T3 one-token decode with explicit legacy KV tensors."""

    def __init__(self, t3: T3):
        super().__init__()
        self.t3 = t3

    def forward(
        self,
        speech_token: torch.Tensor,
        cache_position: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        token_emb = self.t3.speech_emb(speech_token.to(dtype=torch.long))
        # Transformers 4.57's GPT2 DynamicCache causal-mask helper traces the
        # first cache length into an Expand node.  A direct one-query GPT2
        # block loop keeps the exact weights and KV layout while making the
        # cache sequence dimension genuinely dynamic in ONNX Runtime.
        hidden = token_emb + self.t3.tfmr.wpe(cache_position.to(dtype=torch.long)).to(token_emb.device)
        hidden = self.t3.tfmr.drop(hidden)
        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(self.t3.tfmr.h):
            key_past = past[layer_index * 2]
            value_past = past[layer_index * 2 + 1]
            residual = hidden
            hidden_norm = block.ln_1(hidden)
            query, key, value = block.attn.c_attn(hidden_norm).split(block.attn.split_size, dim=2)
            shape = (*query.shape[:-1], -1, block.attn.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            value = value.view(shape).transpose(1, 2)
            key = torch.cat((key_past, key), dim=2)
            value = torch.cat((value_past, value), dim=2)
            scale = float(block.attn.head_dim) ** -0.5 if block.attn.scale_attn_weights else 1.0
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
            weights = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
            attn = torch.matmul(weights, value)
            # [B, heads, query_len, head_dim] -> [B, query_len, hidden].
            # Capture dimensions after the transpose; using the pre-transpose
            # shape here would accidentally produce [B, heads, head_dim].
            attn = attn.transpose(1, 2).contiguous().reshape(attn.shape[0], attn.shape[2], -1)
            attn = block.attn.c_proj(attn)
            hidden = residual + attn
            residual = hidden
            hidden = block.ln_2(hidden)
            hidden = block.mlp(hidden)
            hidden = residual + hidden
            present.append((key, value))
        hidden = self.t3.tfmr.ln_f(hidden)
        logits = self.t3.speech_head(hidden[:, -1, :])
        return (logits, *[value for pair in present for value in pair])


from onnx_t3_core import NanoT3KV


def _output_names(prefix: str = "present") -> list[str]:
    return ["logits"] + [f"{prefix}_{index}" for index in range(T3_LAYERS * 2)]


def _export_one(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Legacy exporter is deliberate here.  It emits standard Gather/MatMul/
    # Attention graphs and does not serialize Transformers' DynamicCache class.
    torch.onnx.export(
        module,
        args,
        str(path),
        opset_version=int(opset),
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "opset": int(opset),
        "inputs": [value.name for value in graph.graph.input],
        "outputs": [value.name for value in graph.graph.output],
        "nodes": len(graph.graph.node),
    }


def export_t3(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT,
    opset: int = 18,
    seed: int = 1234,
) -> dict[str, Any]:
    """Export prefill/decode ONNX graphs and write a reproducible manifest."""

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, load_report = load_t3(checkpoint_dir)
    prefill = NanoT3Prefill(model).eval()
    decode = NanoT3Decode(model).eval()

    # Keep examples small.  Both text length and cache length are dynamic in
    # the graph; these only determine the trace's representative shape.
    speaker = torch.randn(1, 256)
    prompt = torch.randint(0, T3_SPEECH_VOCAB, (1, T3_PROMPT_LEN), dtype=torch.long)
    text = torch.tensor([[255, 17, 42, 0]], dtype=torch.long)
    with torch.inference_mode():
        pre_out = prefill(speaker, prompt, text)
    past = tuple(pre_out[1:])
    export_report = {
        "format": "onnx",
        "status": "partial_t3_exported",
        "model": "ResembleAI/chatterbox-nano",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "source": "vendor/chatterbox @ commit 5de7a54 (local checkout)",
        "architecture": {
            "backbone": "GPT2_small",
            "layers": T3_LAYERS,
            "hidden_size": T3_HIDDEN,
            "speech_vocab": T3_SPEECH_VOCAB,
            "prompt_tokens": T3_PROMPT_LEN,
            "cache_layout": "12 layers x key,value; [batch,heads,sequence,head_dim]",
        },
        "load": load_report,
        "prefill": _export_one(
            prefill,
            (speaker, prompt, text),
            output_dir / "nano_t3_prefill.onnx",
            input_names=["speaker_emb", "cond_prompt_speech_tokens", "text_tokens"],
            output_names=_output_names("present"),
            dynamic_axes={
                "speaker_emb": {0: "batch"},
                "cond_prompt_speech_tokens": {0: "batch", 1: "prompt_len"},
                "text_tokens": {0: "batch", 1: "text_len"},
                **{f"present_{i}": {0: "batch", 2: "cache_len"} for i in range(T3_LAYERS * 2)},
                "logits": {0: "batch"},
            },
            opset=opset,
        ),
        "decode": _export_one(
            decode,
            (
                torch.tensor([[T3_START_SPEECH]], dtype=torch.long),
                torch.tensor([381], dtype=torch.long),
                *past,
            ),
            output_dir / "nano_t3_decode.onnx",
            input_names=["speech_token", "cache_position"] + [f"past_{i}" for i in range(T3_LAYERS * 2)],
            output_names=_output_names("present"),
            dynamic_axes={
                "speech_token": {0: "batch"},
                "cache_position": {0: "one"},
                **{f"past_{i}": {0: "batch", 2: "cache_len"} for i in range(T3_LAYERS * 2)},
                **{f"present_{i}": {0: "batch", 2: "present_len"} for i in range(T3_LAYERS * 2)},
                "logits": {0: "batch"},
            },
            opset=opset,
        ),
        "acoustic_decoder": {
            "status": "not_exported",
            "reason": "S3Gen CausalConditionalCFM flow loop and dynamic mel length require a dedicated graph; no Turbo graph substituted.",
        },
        "vocoder": {
            "status": "not_exported",
            "reason": "HiFTGenerator has a separate f0/source path and stateful cache; preserve PyTorch path until a numerical parity wrapper exists.",
        },
        "watermark": {
            "status": "pytorch_only",
            "reason": "Perth watermark is applied by NanoEngine after vocoding and remains in released audio.",
        },
        "created_unix": time.time(),
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(export_report, indent=2, sort_keys=True) + "\n")
    del model, prefill, decode, pre_out, past
    gc.collect()
    return export_report


def _numpy_inputs_from_example(output_dir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    return {
        "speaker_emb": rng.normal(size=(1, 256)).astype(np.float32),
        "cond_prompt_speech_tokens": rng.integers(0, T3_SPEECH_VOCAB, size=(1, T3_PROMPT_LEN), dtype=np.int64),
        "text_tokens": np.asarray([[255, 17, 42, 0]], dtype=np.int64),
    }


def verify_t3(output_dir: Path = DEFAULT_OUTPUT, *, seed: int = 1234) -> dict[str, Any]:
    """Verify ONNX Runtime against the exact PyTorch T3 for one deterministic step."""

    import onnxruntime as ort

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"export manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_dir = Path(manifest["checkpoint_dir"])
    model, _ = load_t3(checkpoint_dir)
    prefill = NanoT3Prefill(model).eval()
    decode = NanoT3Decode(model).eval()
    data = _numpy_inputs_from_example(output_dir)
    torch_inputs = tuple(torch.from_numpy(data[name]) for name in ("speaker_emb", "cond_prompt_speech_tokens", "text_tokens"))
    with torch.inference_mode():
        torch_pre = prefill(*torch_inputs)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    pre_session = ort.InferenceSession(str(output_dir / "nano_t3_prefill.onnx"), opts, providers=["CPUExecutionProvider"])
    ort_pre = pre_session.run(None, data)
    pre_err = float(np.max(np.abs(ort_pre[0] - torch_pre[0].numpy())))
    pre_rel = float(np.max(np.abs(ort_pre[0] - torch_pre[0].numpy()) / np.maximum(np.abs(torch_pre[0].numpy()), 1e-5)))

    # Compare a decode call using the exact PyTorch prefill cache.
    speech_token = np.asarray([[123]], dtype=np.int64)
    torch_past = tuple(torch_pre[1:])
    cache_position = np.asarray([torch_past[0].shape[2]], dtype=np.int64)
    with torch.inference_mode():
        torch_dec = decode(torch.from_numpy(speech_token), torch.from_numpy(cache_position), *torch_past)
    decode_feed = {"speech_token": speech_token, "cache_position": cache_position}
    for index, value in enumerate(torch_past):
        decode_feed[f"past_{index}"] = value.numpy()
    dec_session = ort.InferenceSession(str(output_dir / "nano_t3_decode.onnx"), opts, providers=["CPUExecutionProvider"])
    ort_dec = dec_session.run(None, decode_feed)
    dec_err = float(np.max(np.abs(ort_dec[0] - torch_dec[0].numpy())))
    dec_rel = float(np.max(np.abs(ort_dec[0] - torch_dec[0].numpy()) / np.maximum(np.abs(torch_dec[0].numpy()), 1e-5)))
    report = {
        "status": "verified" if max(pre_err, dec_err) < 2e-3 else "numerical_mismatch",
        "seed": seed,
        "providers": ["CPUExecutionProvider"],
        "prefill": {"max_abs_error": pre_err, "max_relative_error": pre_rel, "cache_outputs": len(ort_pre) - 1},
        "decode": {"max_abs_error": dec_err, "max_relative_error": dec_rel, "cache_outputs": len(ort_dec) - 1},
        "audio": "not_verified; S3Gen/HiFT remain PyTorch",
    }
    (output_dir / "numerical_parity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    del model, prefill, decode, torch_pre, torch_dec
    gc.collect()
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    exp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    exp.add_argument("--opset", type=int, default=18)
    exp.add_argument("--seed", type=int, default=1234)
    ver = sub.add_parser("verify")
    ver.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ver.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "export":
        print(json.dumps(export_t3(args.output_dir, checkpoint_dir=args.checkpoint_dir, opset=args.opset, seed=args.seed), indent=2))
    else:
        print(json.dumps(verify_t3(args.output_dir, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
