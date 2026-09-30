"""Subprocess RSS/CUDA benchmark for :mod:`runtime`.

The parent process samples the child process from outside, so Python allocator
and CUDA loading spikes are recorded even when the child frees them before a
stage report.  The child emits one JSON object per completed stage and writes
three matched warm-generation WAV files.

Examples::

    python scripts/nano_lab/runtime_benchmark.py --variant optimized --device cuda
    python scripts/nano_lab/runtime_benchmark.py --variant stock --device cpu --runs 3

The benchmark intentionally does not label a partial ONNX export as an
end-to-end runtime.  ``runtime.onnx_feasibility()`` is included in the report
with the current reason for deferring export work.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
DEFAULT_REF = ROOT / "artifacts" / "references" / "chunks" / "asmr7_chunk_36-46.wav"
DEFAULT_TEXTS = (
    "Hey there. You are doing great. Take a slow breath and let your shoulders relax. I am right here with you.",
    "I left the blue notebook beside the window. Please bring it with you when we meet tomorrow.",
    "You have done enough for today. Take a quiet moment, breathe slowly, and let yourself relax.",
)


def _json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _child(args: argparse.Namespace) -> int:
    # Importing torch after setting thread limits avoids inheriting a host's
    # large default OpenMP pool in long-lived benchmark workers.
    import numpy as np
    import psutil
    import soundfile as sf
    import torch

    sys.path.insert(0, str(SCRIPT.parent))
    from runtime import DEFAULT_CHECKPOINT, NanoEngine, onnx_feasibility
    from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

    torch.set_num_threads(min(4, max(1, int(args.cpu_threads))))
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else DEFAULT_CHECKPOINT
    reference = Path(args.reference).expanduser().resolve() if args.reference else DEFAULT_REF
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()

    def memory() -> dict[str, Any]:
        result = {
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

    report: dict[str, Any] = {
        "schema": "nano-runtime-benchmark-v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "variant": args.variant,
        "device": args.device,
        "dtype": args.dtype,
        "decoder": args.decoder,
        "reference": str(reference),
        "texts": list(DEFAULT_TEXTS),
        "onnx": onnx_feasibility(),
        "before_load": memory(),
    }
    _json_line({"event": "before_load", **report["before_load"]})
    load_started = time.perf_counter()
    try:
        if args.variant == "stock":
            # This branch is deliberately the upstream loader.  Calling
            # NanoEngine(optimized=False) would still delete the GPT-2 text
            # head and therefore would not be a matched stock control.
            if args.decoder != "meanflow":
                raise ValueError("the upstream stock loader always uses s3gen_meanflow")
            if args.quantize:
                raise ValueError("the upstream stock loader does not expose dynamic-int8 quantization")
            if args.unload_voice_encoder or args.unload_decoder_encoder or args.unload_tokenizer:
                raise ValueError("encoder/tokenizer unload flags require --variant optimized")
            model = ChatterboxTurboTTS.from_local(checkpoint, device=args.device, nano=True)
            engine = None
            report["load_report"] = {
                "optimized": False,
                "loader": "chatterbox.tts_turbo.ChatterboxTurboTTS.from_local",
                "device": args.device,
                "dtype": str(next(model.t3.parameters()).dtype),
                "decoder": "meanflow",
                "parameters": {
                    name: sum(parameter.numel() for parameter in getattr(model, name).parameters())
                    for name in ("t3", "s3gen", "ve")
                },
            }
            if args.conditionals_path:
                cond_path = Path(args.conditionals_path).expanduser().resolve()
                model.conds = Conditionals.load(cond_path, map_location="cpu").to(args.device)
        else:
            engine = NanoEngine.from_pretrained(
                checkpoint,
                device=args.device,
                optimized=True,
                dtype=args.dtype,
                decoder=args.decoder,
                cache_conditionals=not args.no_condition_cache,
                unload_voice_encoder=args.unload_voice_encoder,
                unload_decoder_encoder=args.unload_decoder_encoder,
                unload_tokenizer=args.unload_tokenizer,
                conditionals_path=args.conditionals_path,
                quantize=args.quantize,
                cpu_threads=args.cpu_threads,
            )
            model = engine.model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        _json_line({"event": "error", "stage": "load", "error": repr(exc)})
        raise
    report["load_seconds"] = time.perf_counter() - load_started
    if engine is not None:
        report["load_report"] = engine.load_report
    report["loaded"] = memory()
    _json_line(
        {
            "event": "loaded",
            "load_seconds": report["load_seconds"],
            "load_report": report.get("load_report", {}),
            **report["loaded"],
        }
    )

    cond_started = time.perf_counter()
    if args.conditionals_path:
        report["conditioning_seconds"] = 0.0
        report["conditioning_source"] = "cached_file"
    else:
        if engine is not None:
            engine.prepare_conditionals(reference, exaggeration=args.exaggeration, norm_loudness=not args.no_loudness)
        else:
            model.prepare_conditionals(
                str(reference),
                exaggeration=args.exaggeration,
                norm_loudness=not args.no_loudness,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report["conditioning_seconds"] = time.perf_counter() - cond_started
        report["conditioning_source"] = "reference_audio"
    report["conditioned"] = memory()
    _json_line({"event": "conditioned", "conditioning_seconds": report["conditioning_seconds"], **report["conditioned"]})

    report["runs"] = []
    texts = list(DEFAULT_TEXTS)
    if args.runs < len(texts):
        texts = texts[:args.runs]
    seeds = [31 + i * 17 for i in range(len(texts))]
    for index, (text, seed) in enumerate(zip(texts, seeds)):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if engine is not None:
            audio, tokens = engine.generate(
                text,
                n_cfm_steps=args.n_cfm_steps,
                seed=None,
                return_tokens=True,
            )
        else:
            # The upstream API fixes Nano/Turbo meanflow to two CFM steps.
            # Keep this call untouched so stock measurements remain a true
            # reference rather than a reimplementation of the pipeline.
            with torch.inference_mode():
                audio = model.generate(text)
            tokens = torch.empty(0, dtype=torch.long)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        samples = audio.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        sample_rate = engine.sr if engine is not None else model.sr
        audio_seconds = len(samples) / sample_rate
        path = output_dir / f"{args.variant}_{args.decoder}_{index}.wav"
        sf.write(path, samples, sample_rate, subtype="PCM_24")
        row = {
            "index": index,
            "seed": seed,
            "text": text,
            "path": str(path),
            "speech_tokens": int(tokens.numel()),
            "audio_seconds": audio_seconds,
            "generation_seconds": elapsed,
            # RTF is wall time divided by generated audio duration.  Values
            # below one indicate faster-than-real-time generation.
            "rtf": elapsed / audio_seconds if audio_seconds > 0 else None,
            **memory(),
        }
        report["runs"].append(row)
        _json_line({"event": "run", **row})

    report["after_runs"] = memory()
    report["finished_at_unix"] = time.time()
    _json_line({"event": "complete", **report["after_runs"]})
    return 0


def _sample_process(process: "psutil.Process", samples: list[dict[str, float | int]], stop: threading.Event) -> None:
    import psutil

    while not stop.is_set():
        try:
            info = process.memory_info()
            samples.append({"rss_mib": info.rss / 2**20, "vms_mib": info.vms / 2**20, "time": time.time()})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        stop.wait(0.02)


def _parent(args: argparse.Namespace) -> int:
    import psutil

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--child",
        "--variant",
        args.variant,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--decoder",
        args.decoder,
        "--runs",
        str(args.runs),
        "--n-cfm-steps",
        str(args.n_cfm_steps),
        "--output-dir",
        str(output_dir),
        "--cpu-threads",
        str(args.cpu_threads),
        "--exaggeration",
        str(args.exaggeration),
    ]
    for flag, value in (
        ("--checkpoint", args.checkpoint),
        ("--reference", args.reference),
        ("--conditionals-path", args.conditionals_path),
        ("--quantize", args.quantize),
    ):
        if value:
            command.extend([flag, str(value)])
    if args.no_condition_cache:
        command.append("--no-condition-cache")
    if args.no_loudness:
        command.append("--no-loudness")
    if args.unload_voice_encoder:
        command.append("--unload-voice-encoder")
    if args.unload_decoder_encoder:
        command.append("--unload-decoder-encoder")
    if args.unload_tokenizer:
        command.append("--unload-tokenizer")

    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        env={**os.environ, "OMP_NUM_THREADS": str(min(4, args.cpu_threads)), "MKL_NUM_THREADS": str(min(4, args.cpu_threads))},
    )
    process = psutil.Process(child.pid)
    samples: list[dict[str, float | int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_process, args=(process, samples, stop), daemon=True)
    sampler.start()
    events: list[dict[str, Any]] = []
    assert child.stdout is not None
    for line in child.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Keep diagnostics in the terminal, but do not corrupt the report.
            print(line, flush=True)
            continue
        events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
    return_code = child.wait()
    stop.set()
    sampler.join(timeout=2)
    if samples:
        peak_rss = max(item["rss_mib"] for item in samples)
        peak_vms = max(item["vms_mib"] for item in samples)
    else:
        peak_rss = peak_vms = None
    child_report: dict[str, Any] = {}
    for event in events:
        if event.get("event") == "complete":
            child_report["after_runs"] = {k: v for k, v in event.items() if k != "event"}
        elif event.get("event") == "run":
            child_report.setdefault("runs", []).append({k: v for k, v in event.items() if k != "event"})
        elif event.get("event") == "loaded":
            child_report["loaded"] = {k: v for k, v in event.items() if k != "event"}
            if "load_report" in event:
                child_report["load_report"] = event["load_report"]
        elif event.get("event") == "conditioned":
            child_report["conditioned"] = {k: v for k, v in event.items() if k != "event"}
    final = {
        "schema": "nano-runtime-benchmark-v1",
        "variant": args.variant,
        "device": args.device,
        "dtype": args.dtype,
        "decoder": args.decoder,
        "return_code": return_code,
        "wall_seconds": time.perf_counter() - started,
        "subprocess_sampler": {
            "sample_period_seconds": 0.02,
            "peak_rss_mib": peak_rss,
            "peak_vms_mib": peak_vms,
            "sample_count": len(samples),
        },
        **child_report,
    }
    report_path.write_text(json.dumps(final, indent=2, sort_keys=True))
    print(json.dumps({"event": "report", "path": str(report_path), **final["subprocess_sampler"]}, sort_keys=True), flush=True)
    return return_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("stock", "optimized"), default="optimized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--decoder", choices=("meanflow", "original"), default="meanflow")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--conditionals-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "nano_lab" / "runtime_optimized"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-cfm-steps", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--exaggeration", type=float, default=0.0)
    parser.add_argument("--quantize", choices=("dynamic-int8",), default=None)
    parser.add_argument("--no-condition-cache", action="store_true")
    parser.add_argument("--no-loudness", action="store_true")
    parser.add_argument("--unload-voice-encoder", action="store_true")
    parser.add_argument("--unload-decoder-encoder", action="store_true")
    parser.add_argument("--unload-tokenizer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.runs = max(1, min(int(args.runs), len(DEFAULT_TEXTS)))
    if args.child:
        return _child(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
