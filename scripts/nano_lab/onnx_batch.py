"""Research-only persistent ONNX Runtime batch benchmark.

The normal staged launcher starts one process per ORT stage. That keeps peak
RSS low, but it also pays session creation on every request. This command
measures an explicit manifest of cases with one ORT worker that retains one
T3, flow encoder, estimator, and vocoder session across all cases.

Preparation children load only the weights-only conditioning cache. The ORT
worker then runs tokenisation, T3, flow, estimator, and vocoder for every
case. Finish children apply Perth and write WAV files after the worker exits.
The default worker does not import Torch. With --finish-in-worker it also
retains a CPU Perth watermarker. Profiling is intentionally disabled because
profiling changes session behavior and ends a profile per request.

This is a benchmark, not a server and not a default launcher path. Warm worker
compute includes intermediate artifact I/O, but excludes preparation, process
startup, and final watermark/WAV publication. It is not request latency.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import onnx_pipeline as pipeline


STAGES = ("t3", "flow_encoder", "meanflow_estimator", "vocoder")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchCase:
    """One explicit benchmark case from the manifest."""

    case_id: str
    voice: str
    text: str
    seed: int
    steps: int = 2
    tokens_file: str | None = None
    output: str | None = None


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"manifest field {name!r} must be a JSON scalar")


def _load_manifest(path: Path) -> list[BatchCase]:
    """Read a list or ``{"cases": [...]}`` manifest without model imports."""

    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        if payload.get("schema_version") not in (None, SCHEMA_VERSION):
            raise ValueError(f"unsupported batch manifest schema_version: {payload.get('schema_version')!r}")
        payload = payload.get("cases")
    if not isinstance(payload, list) or not payload:
        raise ValueError("batch manifest must contain a non-empty cases list")
    result: list[BatchCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"manifest case {index} must be a JSON object")
        case_id = str(raw.get("id", raw.get("name", f"case_{index:03d}"))).strip()
        voice = str(raw.get("voice", "")).strip()
        text_value = raw.get("text")
        if not isinstance(text_value, str):
            raise ValueError(f"manifest case {case_id!r} requires string text")
        text = text_value
        if not case_id:
            raise ValueError(f"manifest case {index} has an empty id")
        if case_id in seen:
            raise ValueError(f"manifest contains duplicate case id: {case_id!r}")
        seen.add(case_id)
        if not voice:
            raise ValueError(f"manifest case {case_id!r} requires voice")
        if not text.strip():
            raise ValueError(f"manifest case {case_id!r} requires non-empty text")
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest case {case_id!r} requires integer seed") from exc
        steps = int(raw.get("steps", 2))
        if steps != 2:
            raise ValueError("the staged meanflow benchmark requires steps=2")
        tokens_file = raw.get("tokens_file")
        if tokens_file is not None:
            tokens_file = str(_json_value(tokens_file, f"{case_id}.tokens_file"))
        output = raw.get("output")
        if output is not None:
            output = str(_json_value(output, f"{case_id}.output"))
        result.append(
            BatchCase(
                case_id=case_id,
                voice=voice,
                text=text,
                seed=seed,
                steps=steps,
                tokens_file=tokens_file,
                output=output,
            )
        )
    return result


def _slug(value: str) -> str:
    chars = [char if char.isalnum() or char in "._-" else "_" for char in value]
    text = "".join(chars).strip("._")
    return text[:80] or "case"


def _rss_current_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _rss_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _mib(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 2)


class _RssSampler:
    """Small in-process sampler to catch transient ORT allocations."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current_bytes = 0
        self.peak_bytes = 0
        self.case_peak_bytes = 0
        self._case_active = False

    def _sample(self) -> None:
        value = _rss_current_bytes()
        self.current_bytes = value
        self.peak_bytes = max(self.peak_bytes, value, _rss_peak_bytes())
        if self._case_active:
            self.case_peak_bytes = max(self.case_peak_bytes, value)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)
        self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="onnx-batch-rss", daemon=True)
        self._thread.start()

    def begin_case(self) -> None:
        self._sample()
        self.case_peak_bytes = self.current_bytes
        self._case_active = True

    def end_case(self) -> tuple[int, int]:
        self._sample()
        self._case_active = False
        return self.current_bytes, self.case_peak_bytes

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()


def _tree_rss_bytes(pid: int) -> int:
    """Read this parent process tree without importing model packages."""

    total = 0
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(int(value) for value in children.split())
        except (OSError, ValueError):
            continue
    return total


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case_args(options: argparse.Namespace, case: BatchCase, run_dir: Path) -> argparse.Namespace:
    """Build the Namespace expected by the existing stage functions."""

    tokens_file = Path(case.tokens_file).expanduser() if case.tokens_file else None
    if tokens_file is not None and not tokens_file.is_absolute():
        manifest_base = Path(getattr(options, "manifest", Path.cwd())).expanduser().resolve().parent
        tokens_file = manifest_base / tokens_file
    return argparse.Namespace(
        run_dir=run_dir,
        model_dir=Path(options.model_dir).resolve(),
        tokenizer_dir=Path(options.tokenizer_dir).resolve(),
        voice=case.voice,
        text=case.text,
        text_file=None,
        tokens_file=tokens_file.resolve() if tokens_file is not None else None,
        seed=int(case.seed),
        steps=int(case.steps),
        output=None,
        ort_threads=int(options.ort_threads),
        ort_provider=str(options.ort_provider),
        cuda_device_id=int(options.cuda_device_id),
        gpu_mem_limit_mib=int(options.gpu_mem_limit_mib),
        arena_extend_strategy=str(options.arena_extend_strategy),
        cudnn_conv_algo_search=str(options.cudnn_conv_algo_search),
        do_copy_in_default_stream=bool(options.do_copy_in_default_stream),
        ort_profile=False,
        cuda_kv_resident=bool(options.cuda_kv_resident),
        experimental_t3=True,
    )


def _prepare_command(options: argparse.Namespace, case: BatchCase, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(pipeline.__file__).resolve()),
        "prepare",
        "--run-dir",
        str(run_dir),
        "--model-dir",
        str(Path(options.model_dir).resolve()),
        "--voice",
        case.voice,
    ]


def _finish_command(options: argparse.Namespace, case: BatchCase, run_dir: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(pipeline.__file__).resolve()),
        "finish",
        "--run-dir",
        str(run_dir),
        "--model-dir",
        str(Path(options.model_dir).resolve()),
        "--output",
        str(output),
    ]


def _output_path(options: argparse.Namespace, case: BatchCase) -> Path:
    """Resolve the same per-case output path for worker and parent finish."""

    output = Path(case.output).expanduser() if case.output else Path(f"{_slug(case.case_id)}.wav")
    if not output.is_absolute():
        output = Path(options.batch_dir).resolve() / output
    return output.resolve()


def _run_child(command: Sequence[str], *, log_path: Path, parent_pid: int) -> tuple[int, float, int]:
    """Run one serial child and return exit code, elapsed seconds, peak tree RSS."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak = _tree_rss_bytes(parent_pid)
    with log_path.open("w") as log:
        child = subprocess.Popen(
            list(command),
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while child.poll() is None:
            peak = max(peak, _tree_rss_bytes(parent_pid))
            time.sleep(0.05)
        returncode = int(child.returncode)
    peak = max(peak, _tree_rss_bytes(parent_pid))
    return returncode, max(0.0, time.perf_counter() - started), peak


class PersistentRuntimeFactory:
    """Create one runtime per stage and keep it open for the worker lifetime."""

    def __init__(self, *, case_index: int = 0) -> None:
        self.case_index = int(case_index)
        self.runtimes: dict[str, Any] = {}
        self.creation: dict[str, dict[str, Any]] = {}

    def __call__(self, stage: str, model_dir: Path, **kwargs: Any) -> Any:
        if kwargs.get("ort_profile"):
            raise ValueError("persistent batch worker requires profiling disabled")
        existing = self.runtimes.get(stage)
        if existing is not None:
            return existing
        from onnx_staged_runtime import (
            FlowEncoderOrtRuntime,
            MeanflowEstimatorOrtRuntime,
            T3UnifiedOrtRuntime,
            VocoderOrtRuntime,
        )

        classes = {
            "t3": T3UnifiedOrtRuntime,
            "flow_encoder": FlowEncoderOrtRuntime,
            "meanflow_estimator": MeanflowEstimatorOrtRuntime,
            "vocoder": VocoderOrtRuntime,
        }
        if stage not in classes:
            raise ValueError(f"persistent factory received unknown stage {stage!r}")
        started = time.perf_counter()
        runtime = classes[stage](model_dir, **kwargs)
        elapsed = max(0.0, time.perf_counter() - started)
        setattr(runtime, "_nano_persistent", True)
        self.runtimes[stage] = runtime
        self.creation[stage] = {
            "stage": stage,
            "case_index": self.case_index,
            "seconds": elapsed,
            "provider": kwargs.get("ort_provider", "cpu"),
        }
        return runtime

    def set_case(self, case_index: int) -> None:
        self.case_index = int(case_index)

    def close(self) -> None:
        errors: list[str] = []
        for stage, runtime in list(self.runtimes.items()):
            close = getattr(runtime, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # pragma: no cover - cleanup diagnostic
                errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        self.runtimes.clear()
        if errors:
            raise RuntimeError("persistent runtime cleanup failed: " + "; ".join(errors))


def _run_worker_case(
    options: argparse.Namespace,
    case: BatchCase,
    run_dir: Path,
    factory: PersistentRuntimeFactory,
    case_index: int,
) -> dict[str, Any]:
    """Run tokens, flow, estimator, and vocoder with cached stage sessions."""

    args = _case_args(options, case, run_dir)
    factory.set_case(case_index)
    stages = (
        ("tokens", pipeline.stage_tokens),
        ("flow", pipeline.stage_flow),
        ("estimator", pipeline.stage_estimator),
        ("vocoder", pipeline.stage_vocoder),
    )
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    stage_results: dict[str, Any] = {}
    existing_creation = set(factory.creation)
    for stage_name, function in stages:
        stage_started = time.perf_counter()
        stage_started_unix = pipeline._now()
        result: Mapping[str, Any] | None = None
        error: BaseException | None = None
        try:
            result = function(args)
            stage_results[stage_name] = dict(result)
        except BaseException as exc:
            error = exc
        pipeline._stage_report(run_dir, stage_name, stage_started_unix, result, error)
        stage_seconds[stage_name] = max(0.0, time.perf_counter() - stage_started)
        if error is not None:
            raise error
    new_creation = {
        stage: dict(record)
        for stage, record in factory.creation.items()
        if stage not in existing_creation
    }
    generation_seconds = max(0.0, time.perf_counter() - started)
    creation_seconds = {
        stage: float(new_creation[stage]["seconds"]) if stage in new_creation else 0.0
        for stage in STAGES
    }
    return {
        "case_id": case.case_id,
        "voice": case.voice,
        "text": case.text,
        "seed": int(case.seed),
        "run_dir": str(run_dir),
        "stage_seconds": stage_seconds,
        "generation_seconds": generation_seconds,
        "cold_generation_seconds": generation_seconds if case_index == 0 else None,
        "warm_generation_seconds": generation_seconds if case_index > 0 else None,
        "cold_or_warm": "cold" if case_index == 0 else "warm",
        "session_creation_seconds": creation_seconds,
        "session_creation_total_seconds": float(sum(record["seconds"] for record in new_creation.values())),
        "stage_results": stage_results,
    }


def _finish_worker_case(
    options: argparse.Namespace,
    case: BatchCase,
    run_dir: Path,
    output: Path,
    watermarker: Any,
) -> tuple[dict[str, Any], float]:
    """Finish one case inside the persistent worker with one Perth marker."""

    args = _case_args(options, case, run_dir)
    args.output = output
    started = time.perf_counter()
    stage_started_unix = pipeline._now()
    result: Mapping[str, Any] | None = None
    error: BaseException | None = None
    try:
        result = pipeline.stage_finish(args, watermarker=watermarker)
    except BaseException as exc:
        error = exc
    pipeline._stage_report(run_dir, "finish", stage_started_unix, result, error)
    if error is not None:
        raise error
    return dict(result or {}), max(0.0, time.perf_counter() - started)


def _worker(options: argparse.Namespace, cases: Sequence[BatchCase], report_path: Path) -> int:
    """Worker entry point; Torch is allowed only for integrated finishing."""

    if bool(getattr(options, "ort_profile", False)):
        raise SystemExit("persistent batch profiling is disabled; use the fresh staged pipeline for profiles")
    finish_in_worker = bool(getattr(options, "finish_in_worker", False))
    sampler = _RssSampler()
    factory = PersistentRuntimeFactory()
    watermarker: Any | None = None
    watermarker_init_seconds = 0.0
    worker_started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "worker_pid": os.getpid(),
        "ort_profile": False,
        "finish_in_worker": finish_in_worker,
        "watermarker_init_seconds": None,
        "stages_retained": list(STAGES),
        "cases": [],
        "sessions_created": {},
        "rss_sample_interval_seconds": sampler.interval_seconds,
        "ort_provider": options.ort_provider,
        "gpu_arena": {
            "per_session_limit_mib": int(options.gpu_mem_limit_mib) if options.ort_provider == "cuda" else None,
            "session_count": len(STAGES),
            "theoretical_sum_mib": int(options.gpu_mem_limit_mib) * len(STAGES)
            if options.ort_provider == "cuda"
            else None,
            "reference_device_budget_mib": 8192 if options.ort_provider == "cuda" else None,
            "note": "Each ORT session has its own arena; summed limits can add. This is not a total GPU allocation guarantee.",
        },
    }
    sampler.start()
    pipeline.set_runtime_factory(factory)
    _write_json(report_path, report)
    try:
        for index, case in enumerate(cases):
            run_dir = Path(options.batch_dir).resolve() / "cases" / f"{index:03d}_{_slug(case.case_id)}"
            sampler.begin_case()
            request_started = time.perf_counter()
            result = _run_worker_case(options, case, run_dir, factory, index)
            if "torch" in sys.modules:
                if not finish_in_worker:
                    raise RuntimeError("persistent ORT worker unexpectedly imported Torch")
            if finish_in_worker:
                output = _output_path(options, case)
                if watermarker is None:
                    marker_started = time.perf_counter()
                    import perth

                    watermarker = perth.PerthImplicitWatermarker()
                    watermarker_init_seconds = max(0.0, time.perf_counter() - marker_started)
                    report["watermarker_init_seconds"] = watermarker_init_seconds
                finish_result, finish_seconds = _finish_worker_case(
                    options,
                    case,
                    run_dir,
                    output,
                    watermarker,
                )
                result["finish_seconds"] = finish_seconds
                result["finish_result"] = finish_result
                result["output"] = str(output)
                result["audio_seconds"] = finish_result.get("audio_seconds")
                result["watermarker_init_seconds"] = watermarker_init_seconds if index == 0 else 0.0
            result["full_worker_request_seconds"] = max(0.0, time.perf_counter() - request_started)
            current_bytes, case_peak_bytes = sampler.end_case()
            result["rss_current_mib"] = _mib(current_bytes)
            result["rss_peak_mib"] = _mib(case_peak_bytes)
            result["actual_rss_current_mib"] = result["rss_current_mib"]
            result["actual_rss_peak_mib"] = result["rss_peak_mib"]
            report["cases"].append(result)
            report["sessions_created"] = {stage: dict(value) for stage, value in factory.creation.items()}
            _write_json(report_path, report)
        report["sessions_created"] = {stage: dict(value) for stage, value in factory.creation.items()}
        report["watermarker_init_seconds"] = watermarker_init_seconds if finish_in_worker else None
        report["status"] = "ok"
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        sampler.stop()
        report["rss_current_mib"] = _mib(sampler.current_bytes)
        report["rss_peak_mib"] = _mib(sampler.peak_bytes)
        report["torch_imported"] = "torch" in sys.modules
        report["elapsed_seconds"] = max(0.0, time.perf_counter() - worker_started)
        try:
            factory.close()
        except Exception as exc:
            report.setdefault("cleanup_error", f"{type(exc).__name__}: {exc}")
        pipeline.set_runtime_factory(None)
        _write_json(report_path, report)
    return 0 if report["status"] == "ok" else 1


def _run_parent(options: argparse.Namespace, cases: Sequence[BatchCase]) -> int:
    if not options.experimental_t3:
        raise SystemExit("batch requires --experimental-t3 because the long T3 cache is not fully verified")
    if options.cuda_kv_resident and options.ort_provider != "cuda":
        raise SystemExit("--cuda-kv-resident requires --ort-provider cuda")
    if bool(getattr(options, "ort_profile", False)):
        raise SystemExit("persistent batch profiling is disabled; use the fresh staged pipeline for profiles")
    finish_in_worker = bool(getattr(options, "finish_in_worker", False))

    started = time.perf_counter()
    batch_dir = Path(options.batch_dir).expanduser().resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "cases").mkdir(parents=True, exist_ok=True)
    (batch_dir / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = Path(options.manifest).expanduser().resolve()
    (batch_dir / "manifest.json").write_text(manifest_path.read_text())
    report_path = batch_dir / "batch_report.json"
    parent_pid = os.getpid()
    tree_max = _tree_rss_bytes(parent_pid)
    prepare_seconds = 0.0
    finish_seconds = 0.0
    cases_meta: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "manifest": str(manifest_path),
        "batch_dir": str(batch_dir),
        "model_dir": str(Path(options.model_dir).resolve()),
        "python_executable": sys.executable,
        "ort_provider": options.ort_provider,
        "ort_profile": False,
        "cuda_kv_resident": bool(options.cuda_kv_resident),
        "finish_in_worker": finish_in_worker,
        "case_count": len(cases),
        "experimental_t3": True,
        "experimental_t3_limitation": pipeline.EXPERIMENTAL_T3_LIMITATION,
        "pipeline_source_sha256": pipeline._sha256(Path(pipeline.__file__)),
        "batch_source_sha256": pipeline._sha256(Path(__file__)),
        "manifest_sha256": pipeline._sha256(manifest_path),
        "model_manifest_sha256": {name: pipeline._sha256(Path(options.model_dir) / name / "manifest.json") for name in STAGES},
        "warm_compute_is_not_request_latency": True,
        "timing_definition": "generation_seconds covers tokens+flow+estimator+vocoder in the persistent worker; full_worker_request_seconds also includes optional in-worker Perth/master/WAV; end_to_end_seconds includes serial prepare, worker startup, and finish work",
        "cases": [],
    }
    _write_json(report_path, report)
    try:
        # Preparation is intentionally serial. Each child may import Torch, and
        # no preparation child coexists with the persistent ORT worker.
        for index, case in enumerate(cases):
            run_dir = batch_dir / "cases" / f"{index:03d}_{_slug(case.case_id)}"
            command = _prepare_command(options, case, run_dir)
            returncode, elapsed, peak = _run_child(
                command,
                log_path=batch_dir / "logs" / f"prepare_{index:03d}.log",
                parent_pid=parent_pid,
            )
            prepare_seconds += elapsed
            tree_max = max(tree_max, peak)
            stage_report_path = run_dir / "prepare" / "stage_report.json"
            if returncode != 0 or not stage_report_path.exists():
                raise RuntimeError(f"prepare case {case.case_id!r} failed with return code {returncode}")
            stage_report = _read_json(stage_report_path)
            if stage_report.get("status") != "ok":
                raise RuntimeError(f"prepare case {case.case_id!r} failed: {stage_report.get('error')}")
            cases_meta.append(
                {
                    "case_id": case.case_id,
                    "voice": case.voice,
                    "text": case.text,
                    "seed": int(case.seed),
                    "run_dir": str(run_dir),
                    "prepare_seconds": elapsed,
                    "prepare_rss_peak_mib": _mib(peak),
                }
            )

        worker_report_path = batch_dir / "worker_report.json"
        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(manifest_path),
            "--batch-dir",
            str(batch_dir),
            "--model-dir",
            str(Path(options.model_dir).resolve()),
            "--tokenizer-dir",
            str(Path(options.tokenizer_dir).resolve()),
            "--ort-threads",
            str(options.ort_threads),
            "--ort-provider",
            str(options.ort_provider),
            "--cuda-device-id",
            str(options.cuda_device_id),
            "--gpu-mem-limit-mib",
            str(options.gpu_mem_limit_mib),
            "--arena-extend-strategy",
            str(options.arena_extend_strategy),
            "--cudnn-conv-algo-search",
            str(options.cudnn_conv_algo_search),
            "--worker-report",
            str(worker_report_path),
        ]
        if options.do_copy_in_default_stream:
            worker_command.append("--do-copy-in-default-stream")
        else:
            worker_command.append("--no-copy-in-default-stream")
        if options.cuda_kv_resident:
            worker_command.append("--cuda-kv-resident")
        if finish_in_worker:
            worker_command.append("--finish-in-worker")
        worker_started = time.perf_counter()
        returncode, worker_elapsed, worker_peak = _run_child(
            worker_command,
            log_path=batch_dir / "logs" / "worker.log",
            parent_pid=parent_pid,
        )
        tree_max = max(tree_max, worker_peak)
        if returncode != 0 or not worker_report_path.exists():
            raise RuntimeError(f"persistent ORT worker failed with return code {returncode}")
        worker_report = _read_json(worker_report_path)
        if worker_report.get("status") != "ok":
            raise RuntimeError(f"persistent ORT worker failed: {worker_report.get('error')}")
        worker_cases = {str(item["case_id"]): item for item in worker_report.get("cases", [])}
        for item in cases_meta:
            item.update(worker_cases.get(item["case_id"], {}))
        if finish_in_worker:
            # The worker already wrote finish/stage_report.json and output
            # files. Keep finish time in the worker's per-case request timing;
            # no second Perth process is started.
            finish_seconds = sum(float(item.get("finish_seconds", 0.0)) for item in cases_meta)
            for item in cases_meta:
                output = Path(str(item.get("output", ""))).expanduser()
                if not output.is_file():
                    raise RuntimeError(f"worker finish did not write output: {output}")
        else:
            finish_started = time.perf_counter()
            for index, case in enumerate(cases):
                run_dir = Path(cases_meta[index]["run_dir"])
                output = _output_path(options, case)
                returncode, elapsed, peak = _run_child(
                    _finish_command(options, case, run_dir, output),
                    log_path=batch_dir / "logs" / f"finish_{index:03d}.log",
                    parent_pid=parent_pid,
                )
                finish_seconds += elapsed
                tree_max = max(tree_max, peak)
                finish_report_path = run_dir / "finish" / "stage_report.json"
                if returncode != 0 or not finish_report_path.exists():
                    raise RuntimeError(f"finish case {case.case_id!r} failed with return code {returncode}")
                finish_report = _read_json(finish_report_path)
                if finish_report.get("status") != "ok":
                    raise RuntimeError(f"finish case {case.case_id!r} failed: {finish_report.get('error')}")
                item = cases_meta[index]
                item["finish_seconds"] = elapsed
                item["output"] = str(output)
                item["audio_seconds"] = (finish_report.get("result") or {}).get("audio_seconds")
            finish_seconds = max(finish_seconds, time.perf_counter() - finish_started)
        report.update(
            {
                "status": "ok",
                "cases": cases_meta,
                "worker": worker_report,
                "worker_seconds": worker_elapsed,
                "prepare_seconds": prepare_seconds,
                "finish_seconds": finish_seconds,
                "end_to_end_seconds": max(0.0, time.perf_counter() - started),
                "process_tree_rss": {
                    "max_serial_peak_mib": _mib(tree_max),
                    "sampling_interval_seconds": 0.05,
                    "definition": "maximum sampled simultaneous sum of parent and all descendant process RSS",
                },
                "max_tree_peak_mib": _mib(tree_max),
                "gpu_arena": worker_report.get("gpu_arena"),
            }
        )
    except BaseException as exc:
        report.update(
            {
                "status": "error",
                "cases": cases_meta,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "prepare_seconds": prepare_seconds,
                "finish_seconds": finish_seconds,
                "end_to_end_seconds": max(0.0, time.perf_counter() - started),
                "process_tree_rss": {
                    "max_serial_peak_mib": _mib(tree_max),
                    "sampling_interval_seconds": 0.05,
                },
                "max_tree_peak_mib": _mib(tree_max),
            }
        )
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "ok" else 1


def _add_common(parser: argparse.ArgumentParser, *, worker: bool = False) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list or object containing explicit voice/text/seed cases")
    parser.add_argument("--batch-dir", type=Path, required=True, help="directory for case artifacts, logs, and reports")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "nano_lab" / "onnx_staged")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "models" / "chatterbox-nano")
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument("--ort-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--gpu-mem-limit-mib", type=int, default=2048)
    parser.add_argument("--arena-extend-strategy", default="kSameAsRequested")
    parser.add_argument("--cudnn-conv-algo-search", default="HEURISTIC")
    parser.add_argument("--cuda-kv-resident", action="store_true", help="keep T3 KV cache in CUDA OrtValues (CUDA only)")
    parser.add_argument(
        "--finish-in-worker",
        action="store_true",
        help="opt in to Perth/master/WAV finish inside the persistent worker; default keeps finish children serial",
    )
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument("--do-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_true", default=True)
    copy_group.add_argument("--no-copy-in-default-stream", dest="do_copy_in_default_stream", action="store_false")
    parser.add_argument("--ort-profile", action="store_true", help="rejected: persistent workers keep profiling disabled")
    if worker:
        parser.add_argument("--worker-report", type=Path, required=True)
    else:
        parser.add_argument("--experimental-t3", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serial prepare, persistent ORT worker, serial Perth finish")
    _add_common(run)
    worker = sub.add_parser("worker", help="internal child command; use run")
    _add_common(worker, worker=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    manifest = Path(args.manifest).expanduser().resolve()
    cases = _load_manifest(manifest)
    args.manifest = manifest
    args.batch_dir = Path(args.batch_dir).expanduser().resolve()
    args.model_dir = Path(args.model_dir).expanduser().resolve()
    args.tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    if args.command == "worker":
        return _worker(args, cases, Path(args.worker_report).expanduser().resolve())
    return _run_parent(args, cases)


if __name__ == "__main__":
    raise SystemExit(main())
