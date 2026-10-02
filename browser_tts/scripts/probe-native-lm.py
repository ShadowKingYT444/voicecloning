#!/usr/bin/env python3
"""Capture a short CPU-only native Nano language-model control trace.

The script loads the minimal upstream GPT-2 and speech head, then feeds the
public voice state's audio_features, selected original text_emb rows, and one
original speech_emb BOS row. It saves prefill logits and up to four decode
logits, plus sampled speech-token IDs. It does not load or decode audio. Run it
through scripts/nano_lab/bounded_job.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NANO_LAB = ROOT / "scripts" / "nano_lab"
BROWSER_SCRIPTS = ROOT / "browser_tts" / "scripts"
DEFAULT_CHECKPOINT = ROOT / "models" / "chatterbox-nano"
DEFAULT_VOICE_MANIFEST = ROOT / "browser_tts" / "public" / "voice" / "asmr-state.json"
DEFAULT_TEXT = "Take a slow breath in, and let your shoulders relax."
TEXT_EMB = "text_emb.weight"
SPEECH_EMB = "speech_emb.weight"
SPEECH_BOS = 6561
SPEECH_EOS = 6562
VOCAB_SIZE = 6563
HIDDEN = 768
RAW_LOGIT_PROBES = 5  # Prefill plus the next four generation positions.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_worker_sampler():
    """Import the shared exact LCG/top-p sampler. This creates no ORT session."""
    source = BROWSER_SCRIPTS / "probe-cpu-pipeline.py"
    spec = importlib.util.spec_from_file_location("browser_cpu_pipeline_probe", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the browser CPU probe helper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorkerSampler, module.ranked_ids


def read_selected_rows(reader, name: str, indices: list[int]) -> np.ndarray:
    """Read only requested embedding rows from CheckpointReader's header/file."""
    entry = reader.header[name]
    shape = entry["shape"]
    if len(shape) != 2 or shape[1] != HIDDEN:
        raise ValueError(f"{name} has unexpected shape {shape}")
    dtype = entry["dtype"]
    if dtype not in {"F32", "F16", "BF16"}:
        raise ValueError(f"{name} has unsupported dtype {dtype}")
    item_size = 4 if dtype == "F32" else 2
    row_bytes = HIDDEN * item_size
    first, last = entry["data_offsets"]
    if last - first != shape[0] * row_bytes:
        raise ValueError(f"{name} has an invalid safetensors data range")
    rows = []
    for index in indices:
        if not 0 <= index < shape[0]:
            raise ValueError(f"Row {index} is outside {name} with {shape[0]} rows")
        reader.file.seek(reader.start + first + index * row_bytes)
        raw = reader.file.read(row_bytes)
        if len(raw) != row_bytes:
            raise ValueError(f"Could not read {name}[{index}]")
        if dtype == "F32":
            row = np.frombuffer(raw, dtype="<f4").copy()
        elif dtype == "F16":
            row = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            row = np.left_shift(words, 16).view("<f4")
        if not np.isfinite(row).all():
            raise ValueError(f"{name}[{index}] contains NaN or infinity")
        rows.append(row)
    return np.stack(rows).astype(np.float32, copy=False)


def load_audio_features(manifest_path: Path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("format") != "chatterbox-nano-reference-state-v1":
        raise ValueError("Unsupported public voice-state manifest")
    data = manifest["data"]
    filename = data["file"]
    if Path(filename).name != filename:
        raise ValueError("Voice-state data file must be a local sibling of its manifest")
    data_path = manifest_path.parent / filename
    raw = data_path.read_bytes()
    state_hash = hashlib.sha256(raw).hexdigest()
    if len(raw) != data["size_bytes"] or state_hash != data["sha256"]:
        raise ValueError("Public voice-state binary failed its manifest size/SHA-256 check")
    item = next(row for row in manifest["tensors"] if row["name"] == "audio_features")
    dims = item["dims"]
    offset = item["offset"]
    count = math.prod(dims)
    if (item["type"] != "float32" or dims != [1, 89, HIDDEN]
            or item["byte_length"] != count * 4 or offset < 0 or offset + count * 4 > len(raw)):
        raise ValueError("Public audio_features differs from the expected [1, 89, 768] FP32 contract")
    features = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy().reshape(dims)
    if not np.isfinite(features).all():
        raise ValueError("Public audio_features contains NaN or infinity")
    return features, {
        "manifestPath": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "binaryPath": str(data_path.resolve()),
        "binarySha256": state_hash,
        "binaryBytes": len(raw),
        "referenceSha256": manifest.get("reference", {}).get("sha256"),
        "revision": manifest.get("model", {}).get("revision"),
        "audioFeaturesShape": dims,
    }


def save_logits(out: Path, step: int, logits: np.ndarray, ranked_ids) -> dict:
    filename = out / f"native-lm-logits-step-{step:03d}.npy"
    np.save(filename, np.asarray(logits, dtype=np.float32), allow_pickle=False)
    top = ranked_ids(logits, 10)
    return {
        "step": step,
        "path": str(filename),
        "sha256": sha256_file(filename),
        "shape": list(logits.shape),
        "dtype": "float32",
        "top10": [{"tokenId": int(token), "logit": float(logits[token])} for token in top],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--voice-state-manifest", type=Path, default=DEFAULT_VOICE_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--decode-steps", type=int, default=256,
                        help="Sample speech tokens up to EOS or this cap. The default matches the browser's 256-token limit.")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Text must not be empty.")
    if not 1 <= args.decode_steps <= 256 or args.max_context <= 0 or args.threads <= 0:
        parser.error("Decode steps, context, and threads must be positive and within limits.")
    return args


def main() -> int:
    args = parse_args()
    out = args.output_dir or ROOT / "artifacts" / "nano_lab" / "browser_measurements" / (
        "native_lm_control_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report = {
        "schemaVersion": 1,
        "startedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "running",
        "mode": "native-cpu-lm-control",
        "configuration": {
            "checkpointDir": str(args.checkpoint_dir.resolve()),
            "voiceStateManifest": str(args.voice_state_manifest.resolve()),
            "decodeSteps": args.decode_steps,
            "maxContext": args.max_context,
            "threads": args.threads,
        },
        "text": args.text,
        "seed": args.seed,
        "decodeSteps": args.decode_steps,
        "rawLogitProbeGoal": RAW_LOGIT_PROBES,
        "sampler": "browser_tts/scripts/probe-cpu-pipeline.py:WorkerSampler",
        "onnxruntimeSessionsCreated": 0,
        "gpuUsed": False,
        "audioDecoded": False,
        "outputDir": str(out),
        "logits": [],
        "generatedSpeechTokenIds": [],
        "sampledTokenIdsIncludingEos": [],
    }
    write_report(report_path, report)
    try:
        if str(NANO_LAB) not in sys.path:
            sys.path.insert(0, str(NANO_LAB))
        if str(BROWSER_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(BROWSER_SCRIPTS))
        from checkpoint_reader import CheckpointReader
        from onnx_t3_reference import _load_reference_core, _peak_rss_bytes
        from transformers import AutoTokenizer, __version__ as transformers_version
        import torch

        checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint = checkpoint_dir / "t3_nano_v1.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing native Nano checkpoint: {checkpoint}")
        audio_features, voice_source = load_audio_features(args.voice_state_manifest)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
        token_ids = [int(value) for value in tokenizer.encode(args.text, add_special_tokens=False)]
        if not token_ids or tokenizer.eos_token_id in token_ids:
            raise ValueError("Native GPT-2 tokenizer must produce raw IDs without EOS")
        if args.text == DEFAULT_TEXT and len(token_ids) != 12:
            raise ValueError(f"Default text should produce 12 raw native IDs, received {len(token_ids)}")
        if max(token_ids) >= 50276:
            raise ValueError("Native GPT-2 token ID is outside text_emb.weight")

        # CheckpointReader validates the safetensors header, then this probe
        # seeks only the rows used by the text and the single speech BOS.
        with CheckpointReader(checkpoint) as reader:
            text_rows = read_selected_rows(reader, TEXT_EMB, token_ids)
            bos_row = read_selected_rows(reader, SPEECH_EMB, [SPEECH_BOS])
            text_dtype = reader.header[TEXT_EMB]["dtype"]
            speech_dtype = reader.header[SPEECH_EMB]["dtype"]

        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        checkpoint_hash = sha256_file(checkpoint)
        started = time.perf_counter()
        model = _load_reference_core(checkpoint_dir, "cpu", args.max_context)
        report["checkpoint"] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
            "referenceLoader": "scripts/nano_lab/onnx_t3_reference.py:_load_reference_core",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(next(model.parameters()).device),
            "speechHeadShape": list(model.speech_head.weight.shape),
            "encoderAndDecoderLoaded": False,
            "loadSeconds": round(time.perf_counter() - started, 3),
        }
        report["voiceState"] = voice_source
        report["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "transformersVersion": transformers_version,
            "rawTokenIds": token_ids,
            "rawTokenCount": len(token_ids),
            "eosAdded": False,
        }
        report["selectedEmbeddingRows"] = {
            "textEmbedding": TEXT_EMB,
            "textDtype": text_dtype,
            "textRowsRead": len(token_ids),
            "speechEmbedding": SPEECH_EMB,
            "speechDtype": speech_dtype,
            "speechRowsReadForPrefix": 1,
            "wholeEmbeddingTablesLoaded": False,
        }

        x = torch.cat((
            torch.from_numpy(audio_features),
            torch.from_numpy(text_rows).unsqueeze(0),
            torch.from_numpy(bos_row).reshape(1, 1, HIDDEN),
        ), dim=1)
        sequence_length = int(x.shape[1])
        if sequence_length + args.decode_steps > args.max_context:
            raise ValueError("Input plus decode steps exceed --max-context")
        report["nativeInput"] = {
            "recipe": "public audio_features + raw GPT-2 text_emb rows + exactly one native speech_emb BOS row",
            "audioFeaturesShape": list(audio_features.shape),
            "textEmbeddingShape": [1, len(token_ids), HIDDEN],
            "speechBosId": SPEECH_BOS,
            "speechBosShape": [1, 1, HIDDEN],
            "sequenceLength": sequence_length,
        }

        WorkerSampler, ranked_ids = load_worker_sampler()
        with torch.inference_mode():
            output = model.tfmr(inputs_embeds=x, use_cache=True, return_dict=True)
            cache = output.past_key_values
            logits_tensor = model.speech_head(output.last_hidden_state[:, -1, :])
            logits = np.asarray(logits_tensor.detach().cpu().numpy()[0], dtype=np.float32).copy()
        del x, output, logits_tensor
        if logits.shape != (VOCAB_SIZE,) or not np.isfinite(logits).all():
            raise ValueError(f"Native prefill returned invalid logits: {logits.shape}")
        report["logits"].append(save_logits(out, 0, logits, ranked_ids))
        write_report(report_path, report)

        sampler = WorkerSampler(args.seed)
        history = [SPEECH_BOS]
        with CheckpointReader(checkpoint) as reader, torch.inference_mode():
            for step in range(args.decode_steps):
                token = int(sampler.sample(logits, history))
                report["sampledTokenIdsIncludingEos"].append(token)
                if token == SPEECH_EOS:
                    report["reachedSpeechEos"] = True
                    break
                report["generatedSpeechTokenIds"].append(token)
                history.append(token)
                if step + 1 >= args.decode_steps:
                    break
                token_row = read_selected_rows(reader, SPEECH_EMB, [token])
                token_embedding = torch.from_numpy(token_row).reshape(1, 1, HIDDEN)
                output = model.tfmr(inputs_embeds=token_embedding, past_key_values=cache,
                                    use_cache=True, return_dict=True)
                next_cache = output.past_key_values
                logits_tensor = model.speech_head(output.last_hidden_state[:, -1, :])
                logits = np.asarray(logits_tensor.detach().cpu().numpy()[0], dtype=np.float32).copy()
                if logits.shape != (VOCAB_SIZE,) or not np.isfinite(logits).all():
                    raise ValueError(f"Native decode returned invalid logits: {logits.shape}")
                cache = next_cache
                del next_cache, token_row, token_embedding, output, logits_tensor
                if step + 1 < RAW_LOGIT_PROBES:
                    report["logits"].append(save_logits(out, step + 1, logits, ranked_ids))
                write_report(report_path, report)

        report["status"] = "complete"
        report["stoppedAtDecodeLimit"] = not report.get("reachedSpeechEos", False)
        report["rawLogitCount"] = len(report["logits"])
        report["processPeakRssMiB"] = round(_peak_rss_bytes() / (1024 * 1024), 3)
        report["completedAtUtc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        write_report(report_path, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_report(report_path, report)
        print(f"Native LM control failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
