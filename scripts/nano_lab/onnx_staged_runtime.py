"""Small, pure-ONNX-Runtime adapters for staged Chatterbox-Nano exports.

The exporter in :mod:`onnx_staged` deliberately writes one graph per model
component.  This module only loads those graphs with ONNX Runtime and NumPy;
it never imports the PyTorch model.  This makes it possible to benchmark the
resident set of a stage in a fresh process, after the export process has
released its checkpoint.

The T3 graph is a single prefill/decode graph.  Pass a zero-length KV cache
for prefill, then pass the returned cache and one-token embeddings for each
decode step.  The graph does not own tokenisation, sampling, or conditioning
embedding construction.  Those operations remain Python-side so sampling
semantics and the Perth watermark are not silently changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from file_fingerprint import fingerprint_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "nano_lab" / "onnx_staged"
T3_LAYERS = 12
T3_HEADS = 12
T3_HEAD_DIM = 64
T3_HIDDEN = 768
ORT_PROVIDER_CHOICES = ("cpu", "cuda")
DEFAULT_CUDA_GPU_MEM_LIMIT_MIB = 2048


def _normalise_ort_provider(value: str | None) -> str:
    provider = str(value or "cpu").strip().lower()
    aliases = {"cpu": "cpu", "cpuexecutionprovider": "cpu", "cuda": "cuda", "cudaexecutionprovider": "cuda"}
    if provider not in aliases:
        raise ValueError(f"ort_provider must be 'cpu' or 'cuda', got {value!r}")
    return aliases[provider]


def _cuda_provider_options(
    *,
    cuda_device_id: int = 0,
    gpu_mem_limit_mib: int = DEFAULT_CUDA_GPU_MEM_LIMIT_MIB,
    arena_extend_strategy: str = "kSameAsRequested",
    cudnn_conv_algo_search: str = "HEURISTIC",
    do_copy_in_default_stream: bool = True,
) -> dict[str, Any]:
    """Build bounded CUDA provider options.

    The limit applies to ORT's CUDA arena.  It is intentionally explicit in
    each stage report, because allocator limits do not guarantee that external
    CUDA allocations from another process are bounded.
    """

    if int(cuda_device_id) < 0:
        raise ValueError("cuda_device_id must be >= 0")
    if int(gpu_mem_limit_mib) < 256:
        raise ValueError("gpu_mem_limit_mib must be >= 256")
    return {
        "device_id": int(cuda_device_id),
        # Match the full-precision Torch reference. ORT enables TF32 by
        # default, which changed prefill caches beyond our numeric gate.
        "use_tf32": 0,
        "gpu_mem_limit": int(gpu_mem_limit_mib) * 1024 * 1024,
        "arena_extend_strategy": str(arena_extend_strategy),
        "cudnn_conv_algo_search": str(cudnn_conv_algo_search),
        "do_copy_in_default_stream": bool(do_copy_in_default_stream),
    }


def _provider_request(
    *,
    ort_provider: str,
    cuda_device_id: int,
    gpu_mem_limit_mib: int,
    arena_extend_strategy: str,
    cudnn_conv_algo_search: str,
    do_copy_in_default_stream: bool,
):
    """Resolve a provider request and preload CUDA libraries when required."""

    import onnxruntime as ort

    selected = _normalise_ort_provider(ort_provider)
    available = tuple(ort.get_available_providers())
    if selected == "cpu":
        return ["CPUExecutionProvider"], {
            "requested": "cpu",
            "available": list(available),
            "preload_dlls": {"attempted": False, "directory": None, "status": "not_requested"},
            "cuda_options": None,
        }
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "ort_provider='cuda' was requested, but CUDAExecutionProvider is unavailable; "
            f"available providers: {list(available)}"
        )
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        raise RuntimeError("ort_provider='cuda' requires onnxruntime.preload_dlls(directory='')")
    try:
        # The empty directory asks ORT to resolve the NVIDIA DLLs from the
        # installed package/system search path.  Do this before creating any
        # CUDA session so provider initialization is deterministic.
        preload(directory="")
    except Exception as exc:
        raise RuntimeError(f"CUDA DLL preload failed: {exc}") from exc
    options = _cuda_provider_options(
        cuda_device_id=cuda_device_id,
        gpu_mem_limit_mib=gpu_mem_limit_mib,
        arena_extend_strategy=arena_extend_strategy,
        cudnn_conv_algo_search=cudnn_conv_algo_search,
        do_copy_in_default_stream=do_copy_in_default_stream,
    )
    # CPU remains second in the list so unsupported ONNX operators may fall
    # back per-node.  The session is still required to report CUDA first; a
    # silent all-CPU session is an error below.
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"], {
        "requested": "cuda",
        "available": list(available),
        "preload_dlls": {"attempted": True, "directory": "", "status": "ok"},
        "cuda_options": options,
    }


def _read_manifest(
    model_dir: Path,
    stage: str,
    *,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Read one staged manifest and resolve its graph path.

    Normal inference keeps the verified-manifest gate.  The fitted-adapter
    verifier can opt into a deliberate verification-only read so it can
    compare an exported graph with an independent Torch reference before the
    manifest is marked as numerically verified.  This flag never changes the
    manifest on disk.
    """
    stage_dir = model_dir / stage
    path = stage_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"staged ONNX manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("stage") != stage:
        raise RuntimeError(f"manifest {path} identifies stage {manifest.get('stage')!r}, expected {stage!r}")
    if manifest.get("status") != "exported":
        reason = manifest.get("reason") or manifest.get("error") or "stage was not exported"
        raise RuntimeError(f"staged ONNX stage {stage!r} is unavailable: {reason}")
    graph = manifest.get("graph") or {}
    graph_path = graph.get("path")
    if not graph_path:
        raise RuntimeError(f"staged ONNX stage {stage!r} has no graph path")
    resolved = Path(graph_path)
    if not resolved.is_absolute():
        resolved = stage_dir / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"staged ONNX graph not found: {resolved}")
    acoustic_patch = manifest.get('adapter_patch') if stage == 'meanflow_estimator' else None
    if acoustic_patch is not None:
        _validate_decoder_attention_manifest(manifest, stage_dir, resolved, require_verified=require_verified)
    elif require_verified:
        ordinary = manifest.get("verification", {})
        fitted = manifest.get("fitted_adapter_verification", {})
        ordinary_verified = isinstance(ordinary, Mapping) and ordinary.get("status") == "verified"
        fitted_verified = isinstance(fitted, Mapping) and fitted.get("status") == "verified"
        if fitted_verified:
            # A fitted result is accepted only after the verifier has written
            # the digest of the exact graph it compared.  The fitted field is
            # still marked experimental and keeps its short-context scope.
            fitted_graph_sha = fitted.get("graph_sha256")
            # Fitted verification may be repeated while constructing staged
            # sessions.  The helper caches only the digest and revalidates the
            # graph metadata on each lookup; it does not retain graph bytes.
            actual_graph_sha = fingerprint_file(resolved)
            if fitted_graph_sha != actual_graph_sha:
                fitted_verified = False
        if not ordinary_verified and not fitted_verified:
            raise RuntimeError(f"staged ONNX stage {stage!r} has not passed numerical verification")
    manifest["_stage_dir"] = str(stage_dir)
    manifest["_graph_path"] = str(resolved)
    return manifest


def _validate_decoder_attention_manifest(manifest, stage_dir, graph_path, *, require_verified):
    """Check folded acoustic weights even during verification-only sessions."""
    patch = manifest.get('adapter_patch')
    if not isinstance(patch, Mapping) or patch.get('format') != 'nano_decoder_attention_merged_v1':
        raise RuntimeError('Unsupported meanflow adapter patch')
    def checksum(value, label):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Invalid meanflow adapter SHA-256: ' + label)
        return value
    for key in ('adapter_sha256', 'inventory_sha256', 'model_checkpoint_sha256',
                'initial_conditionals_sha256', 'base_weights_sha256'):
        checksum(patch.get(key), key)
    strength = patch.get('strength')
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not np.isfinite(strength) or strength < 0:
        raise RuntimeError('Invalid meanflow adapter strength')
    expected = {
        'graph_sha256': graph_path,
    }
    for file_key, hash_key in (('weights_file', 'patched_weights_sha256'),
                               ('external_report_file', 'external_report_sha256')):
        name = patch.get(file_key)
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise RuntimeError('Invalid meanflow adapter file path: ' + file_key)
        path = stage_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Meanflow adapter file is missing or symlinked: ' + file_key)
        expected[hash_key] = path
    for key, path in expected.items():
        if checksum(patch.get(key), key) != fingerprint_file(path):
            raise RuntimeError('Meanflow adapter file hash mismatch: ' + key)
    if not require_verified:
        return
    verification = manifest.get('decoder_attention_verification')
    if not isinstance(verification, Mapping) or verification.get('status') != 'verified':
        raise RuntimeError('Folded meanflow adapter has not passed numerical verification')
    for verify_key, patch_key in (('graph_sha256', 'graph_sha256'),
                                  ('weights_sha256', 'patched_weights_sha256'),
                                  ('adapter_sha256', 'adapter_sha256'), ('strength', 'strength')):
        if verification.get(verify_key) != patch.get(patch_key):
            raise RuntimeError('Stale meanflow adapter verification: ' + verify_key)


def _require_decoder_attention_provider(manifest, provider):
    if manifest.get('stage') == 'meanflow_estimator' and manifest.get('adapter_patch'):
        reports = manifest.get('decoder_attention_verification', {}).get('provider_reports', {})
        if _normalise_ort_provider(provider) not in reports:
            raise RuntimeError('Folded meanflow adapter is not verified for the requested ORT provider')


def _session(
    path: str | Path,
    *,
    intra_op_num_threads: int,
    inter_op_num_threads: int,
    providers: Sequence[str] | None,
    ort_provider: str = "cpu",
    cuda_device_id: int = 0,
    gpu_mem_limit_mib: int = DEFAULT_CUDA_GPU_MEM_LIMIT_MIB,
    arena_extend_strategy: str = "kSameAsRequested",
    cudnn_conv_algo_search: str = "HEURISTIC",
    do_copy_in_default_stream: bool = True,
    ort_profile: bool = False,
    profile_prefix: str | Path | None = None,
):
    """Create a conservative ORT session.

    The default is one intra-op thread.  A caller can opt into two threads for
    a measured benchmark, but this module never creates an unbounded ORT pool
    on the user's desktop.
    """

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(int(intra_op_num_threads), 2))
    options.inter_op_num_threads = max(1, min(int(inter_op_num_threads), 2))
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if _normalise_ort_provider(ort_provider) == "cuda":
        # Growing KV shapes must not accumulate shape-specific allocation
        # patterns. Keep the fixed device-memory limit unchanged.
        options.enable_mem_pattern = False
    if ort_profile:
        options.enable_profiling = True
        if profile_prefix is not None:
            profile_path = Path(profile_prefix).expanduser().resolve()
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            options.profile_file_prefix = str(profile_path)
    provider_metadata: dict[str, Any]
    if providers is not None and _normalise_ort_provider(ort_provider) == "cpu" and list(providers) != ["CPUExecutionProvider"]:
        # Preserve the old low-level ``providers=`` escape hatch for callers
        # that intentionally select a non-CPU provider.  New code should use
        # the explicit ``ort_provider`` flag so reports remain unambiguous.
        requested = list(providers)
        provider_metadata = {
            "requested": "custom",
            "available": list(ort.get_available_providers()),
            "preload_dlls": {"attempted": False, "directory": None, "status": "custom_request"},
            "cuda_options": None,
        }
    else:
        requested, provider_metadata = _provider_request(
            ort_provider=ort_provider,
            cuda_device_id=cuda_device_id,
            gpu_mem_limit_mib=gpu_mem_limit_mib,
            arena_extend_strategy=arena_extend_strategy,
            cudnn_conv_algo_search=cudnn_conv_algo_search,
            do_copy_in_default_stream=do_copy_in_default_stream,
        )
    session = ort.InferenceSession(str(path), options, providers=requested)
    session_provider_names = tuple(session.get_providers())
    selected = _normalise_ort_provider(ort_provider)
    if selected == "cuda":
        if "CUDAExecutionProvider" not in session_provider_names or session_provider_names[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "ort_provider='cuda' requested CUDAExecutionProvider, but the session is not CUDA-backed: "
                f"{list(session_provider_names)}"
            )
        # This disables ORT's provider reinitialization fallback.  CPU remains
        # second in the provider list for unsupported individual operators,
        # but an initialization/runtime provider failure is surfaced to the
        # caller instead of being reported as a successful CUDA run.
        fallback_disabled = False
        disable_fallback = getattr(session, "disable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()
            fallback_disabled = True
    else:
        fallback_disabled = False
    # Attach diagnostics without relying on private ORT internals.  StageSession
    # copies this metadata into its JSON-safe info report.
    session._nano_provider_metadata = provider_metadata  # type: ignore[attr-defined]
    session._nano_requested_providers = [  # type: ignore[attr-defined]
        item[0] if isinstance(item, tuple) else item for item in requested
    ]
    session._nano_profile_enabled = bool(ort_profile)  # type: ignore[attr-defined]
    session._nano_profile_prefix = str(profile_prefix) if profile_prefix is not None else None  # type: ignore[attr-defined]
    session._nano_fallback_disabled = fallback_disabled  # type: ignore[attr-defined]
    return session


def _feed_names(session: Any) -> tuple[str, ...]:
    return tuple(item.name for item in session.get_inputs())


def _output_names(session: Any) -> tuple[str, ...]:
    return tuple(item.name for item in session.get_outputs())


def _as_float32(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class OrtRun:
    """Named outputs from one stage call."""

    outputs: tuple[np.ndarray, ...]
    names: tuple[str, ...]

    def as_dict(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.outputs))


class StageSession:
    """Base class for one bounded ORT stage."""

    stage: str

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
        providers: Sequence[str] | None = None,
        ort_provider: str = "cpu",
        cuda_device_id: int = 0,
        gpu_mem_limit_mib: int = DEFAULT_CUDA_GPU_MEM_LIMIT_MIB,
        arena_extend_strategy: str = "kSameAsRequested",
        cudnn_conv_algo_search: str = "HEURISTIC",
        do_copy_in_default_stream: bool = True,
        ort_profile: bool = False,
        profile_prefix: str | Path | None = None,
        cuda_kv_resident: bool = False,
        verification_only: bool = False,
    ) -> None:
        if cuda_kv_resident and _normalise_ort_provider(ort_provider) != "cuda":
            raise ValueError("cuda_kv_resident requires ort_provider='cuda'")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.verification_only = bool(verification_only)
        self.manifest = _read_manifest(
            self.model_dir,
            self.stage,
            require_verified=not self.verification_only,
        )
        if not self.verification_only:
            _require_decoder_attention_provider(self.manifest, ort_provider)
        self.session = _session(
            self.manifest["_graph_path"],
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            providers=providers,
            ort_provider=ort_provider,
            cuda_device_id=cuda_device_id,
            gpu_mem_limit_mib=gpu_mem_limit_mib,
            arena_extend_strategy=arena_extend_strategy,
            cudnn_conv_algo_search=cudnn_conv_algo_search,
            do_copy_in_default_stream=do_copy_in_default_stream,
            ort_profile=ort_profile,
            profile_prefix=profile_prefix,
        )
        self.providers = tuple(self.session.get_providers())
        self.ort_provider = _normalise_ort_provider(ort_provider)
        self.provider_metadata = dict(getattr(self.session, "_nano_provider_metadata", {}))
        self.requested_providers = tuple(getattr(self.session, "_nano_requested_providers", self.providers))
        self.profile_enabled = bool(getattr(self.session, "_nano_profile_enabled", False))
        self.profile_prefix = getattr(self.session, "_nano_profile_prefix", None)
        self.profile_path: str | None = None
        self.fallback_disabled = bool(getattr(self.session, "_nano_fallback_disabled", False))
        self.cuda_kv_resident = bool(cuda_kv_resident)
        self.io_binding_enabled = False
        self.io_binding_calls = 0
        self._profile_ended = False
        self.input_names = _feed_names(self.session)
        self.output_names = _output_names(self.session)

    def _run(self, feeds: Mapping[str, Any]) -> OrtRun:
        unknown = sorted(set(feeds) - set(self.input_names))
        missing = sorted(set(self.input_names) - set(feeds))
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            if missing:
                details.append("missing=" + ",".join(missing))
            raise ValueError("invalid ORT feeds: " + "; ".join(details))
        run_options = None
        if self.ort_provider == "cuda":
            import onnxruntime as ort
            run_options = ort.RunOptions()
            device_id = int(self.provider_metadata["cuda_options"]["device_id"])
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")
        outputs = tuple(np.asarray(value) for value in self.session.run(list(self.output_names), dict(feeds), run_options))
        return OrtRun(outputs=outputs, names=self.output_names)

    def info(self) -> dict[str, Any]:
        import onnxruntime as ort

        return {
            "stage": self.stage,
            "verification_only": self.verification_only,
            "graph": self.manifest.get("_graph_path"),
            "ort_version": getattr(ort, "__version__", "unknown"),
            "ort_provider": self.ort_provider,
            "requested_providers": list(self.requested_providers),
            "providers": list(self.providers),
            "provider_metadata": self.provider_metadata,
            "fallback_disabled": self.fallback_disabled,
            "cuda_memory_pattern_enabled": False if self.ort_provider == "cuda" else None,
            "cuda_arena_shrinkage_after_run": self.ort_provider == "cuda",
            "cuda_kv_resident_requested": self.cuda_kv_resident,
            "cuda_kv_resident_active": self.io_binding_enabled,
            "cuda_kv_cache_location": "cuda_ortvalue" if self.io_binding_enabled else "numpy_cpu",
            "io_binding_calls": self.io_binding_calls,
            "profiling": {
                "enabled": self.profile_enabled,
                "profile_prefix": self.profile_prefix,
                "profile_path": self.profile_path,
                "ended": self._profile_ended,
            },
            "inputs": list(self.input_names),
            "outputs": list(self.output_names),
            "threads": "bounded by caller; default 1 intra-op and 1 inter-op",
        }

    def end_profiling(self) -> str | None:
        """Flush the ORT profile and return its path without parsing it."""

        if self._profile_ended:
            return self.profile_path
        self._profile_ended = True
        if not self.profile_enabled:
            return None
        end = getattr(self.session, "end_profiling", None)
        if not callable(end):
            raise RuntimeError("ORT profiling was enabled but InferenceSession.end_profiling is unavailable")
        path = end()
        self.profile_path = str(Path(path).expanduser().resolve()) if path else None
        return self.profile_path

    def close(self) -> str | None:
        """Flush optional profiling before the child process exits."""

        return self.end_profiling()


class T3UnifiedOrtRuntime(StageSession):
    """Run the unified Nano T3 KV graph.

    ``prefill`` accepts a full embedded sequence and a position vector.  For
    Nano's exact input order the caller assembles speaker projection, prompt
    speech embeddings, text embeddings, and the speech BOS embedding before
    calling this method.  ``decode`` accepts one embedded token and the cache
    returned by ``prefill`` or a prior decode.
    """

    stage = "t3"

    def _run_cuda_iobinding(self, feeds: Mapping[str, Any]) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Run T3 with KV cache OrtValues resident on the CUDA device.

        Embeddings, positions, and the zero-length prefill cache enter through
        CPU bindings.  Once produced, each KV output is bound to CUDA by ORT
        and is passed to the next decode call as an ``OrtValue``.  Only logits
        are copied back to NumPy for the Python sampler.
        """

        import onnxruntime as ort

        if self.ort_provider != "cuda":
            raise RuntimeError("CUDA KV I/O binding requires ort_provider='cuda'")
        io_binding_factory = getattr(self.session, "io_binding", None)
        run_with_binding = getattr(self.session, "run_with_iobinding", None)
        if not callable(io_binding_factory) or not callable(run_with_binding):
            raise RuntimeError("the active ONNX Runtime session does not support I/O binding")
        unknown = sorted(set(feeds) - set(self.input_names))
        missing = sorted(set(self.input_names) - set(feeds))
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            if missing:
                details.append("missing=" + ",".join(missing))
            raise ValueError("invalid ORT feeds: " + "; ".join(details))
        io_binding = io_binding_factory()
        for name, value in feeds.items():
            if isinstance(value, ort.OrtValue):
                io_binding.bind_ortvalue_input(name, value)
            else:
                # NumPy arrays remain CPU-side only for embeddings, positions,
                # and the first zero-length prefill cache.
                io_binding.bind_cpu_input(name, np.asarray(value))
        io_binding.bind_output(self.output_names[0], "cpu", 0)
        device_id = int(self.provider_metadata["cuda_options"]["device_id"])
        for name in self.output_names[1:]:
            io_binding.bind_output(name, "cuda", device_id)
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")
        run_with_binding(io_binding, run_options)
        io_binding.synchronize_outputs()
        outputs = tuple(io_binding.get_outputs())
        if len(outputs) != len(self.output_names):
            raise RuntimeError(f"unified T3 graph returned {len(outputs)} outputs")
        logits_value = outputs[0]
        if not isinstance(logits_value, ort.OrtValue):
            raise RuntimeError("I/O binding returned a non-OrtValue logits output")
        logits = np.asarray(logits_value.numpy())
        cache = tuple(outputs[1:])
        for index, value in enumerate(cache):
            if not isinstance(value, ort.OrtValue):
                raise RuntimeError(f"I/O binding returned cache[{index}] without an OrtValue")
            shape = tuple(int(item) for item in value.shape())
            if len(shape) != 4:
                raise RuntimeError(f"I/O-bound cache[{index}] has invalid shape {shape}")
        self.io_binding_enabled = True
        self.io_binding_calls += 1
        return logits, cache

    @staticmethod
    def _cache_defaults(batch: int, dtype: np.dtype = np.dtype("float32")) -> tuple[np.ndarray, ...]:
        return tuple(np.zeros((batch, T3_HEADS, 0, T3_HEAD_DIM), dtype=dtype) for _ in range(T3_LAYERS * 2))

    @staticmethod
    def _validate_embedded(inputs_embeds: Any, *, one_token: bool | None = None) -> np.ndarray:
        value = _as_float32(inputs_embeds, "inputs_embeds")
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != T3_HIDDEN:
            raise ValueError(f"inputs_embeds must have shape (1,T,{T3_HIDDEN}), got {value.shape}")
        if value.shape[1] < 1:
            raise ValueError("inputs_embeds must contain at least one token")
        if one_token is True and value.shape[1] != 1:
            raise ValueError(f"decode inputs_embeds must have one token, got {value.shape}")
        return value

    @staticmethod
    def _validate_positions(cache_position: Any, length: int) -> np.ndarray:
        value = np.asarray(cache_position, dtype=np.int64)
        if value.ndim != 1 or value.shape[0] != length:
            raise ValueError(f"cache_position must have shape ({length},), got {value.shape}")
        return value

    @staticmethod
    def _validate_cache(cache: Sequence[Any] | None, batch: int) -> tuple[np.ndarray, ...]:
        values = T3UnifiedOrtRuntime._cache_defaults(batch) if cache is None else tuple(np.asarray(x, dtype=np.float32) for x in cache)
        if len(values) != T3_LAYERS * 2:
            raise ValueError(f"T3 cache must have {T3_LAYERS * 2} tensors, got {len(values)}")
        for index, value in enumerate(values):
            if (
                value.ndim != 4
                or value.shape[0] != batch
                or value.shape[1] != T3_HEADS
                or value.shape[3] != T3_HEAD_DIM
            ):
                raise ValueError(f"cache[{index}] has invalid shape {value.shape}")
        return values

    @staticmethod
    def _validate_ort_cache(cache: Sequence[Any], batch: int) -> tuple[Any, ...]:
        """Validate CUDA OrtValue KV tensors without converting them to NumPy."""

        import onnxruntime as ort

        values = tuple(cache)
        if len(values) != T3_LAYERS * 2:
            raise ValueError(f"T3 cache must have {T3_LAYERS * 2} tensors, got {len(values)}")
        for index, value in enumerate(values):
            if not isinstance(value, ort.OrtValue):
                raise TypeError(f"cache[{index}] must be an ONNX Runtime OrtValue for CUDA KV residency")
            shape = tuple(int(item) for item in value.shape())
            if len(shape) != 4 or shape[0] != batch or shape[1] != T3_HEADS or shape[3] != T3_HEAD_DIM:
                raise ValueError(f"cache[{index}] has invalid shape {shape}")
        return values

    @staticmethod
    def cache_length(cache: Sequence[Any]) -> int:
        """Return the sequence length without copying a CUDA OrtValue to CPU."""

        if not cache:
            raise ValueError("T3 cache is empty")
        first = cache[0]
        shape_method = getattr(first, "shape", None)
        shape = shape_method() if callable(shape_method) else getattr(first, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError(f"T3 cache value has no rank-4 shape: {shape!r}")
        return int(shape[2])

    def prefill(self, inputs_embeds: Any, cache_position: Any) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        embeds = self._validate_embedded(inputs_embeds)
        positions = self._validate_positions(cache_position, embeds.shape[1])
        cache = self._validate_cache(None, embeds.shape[0])
        if self.cuda_kv_resident:
            logits, ort_cache = self._run_cuda_iobinding(
                {"inputs_embeds": embeds, "cache_position": positions}
                | {f"past_{index}": value for index, value in enumerate(cache)}
            )
            if len(ort_cache) != T3_LAYERS * 2:
                raise RuntimeError(f"unified T3 graph returned {len(ort_cache)} cache outputs")
            return logits, ort_cache
        result = self._run(
            {"inputs_embeds": embeds, "cache_position": positions}
            | {f"past_{index}": value for index, value in enumerate(cache)}
        )
        if len(result.outputs) != 1 + T3_LAYERS * 2:
            raise RuntimeError(f"unified T3 graph returned {len(result.outputs)} outputs")
        return result.outputs[0], tuple(result.outputs[1:])

    def decode(self, inputs_embeds: Any, cache_position: Any, cache: Sequence[Any]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        embeds = self._validate_embedded(inputs_embeds, one_token=True)
        positions = self._validate_positions(cache_position, 1)
        if self.cuda_kv_resident:
            values = self._validate_ort_cache(cache, embeds.shape[0])
            logits, ort_cache = self._run_cuda_iobinding(
                {"inputs_embeds": embeds, "cache_position": positions}
                | {f"past_{index}": value for index, value in enumerate(values)}
            )
            if len(ort_cache) != T3_LAYERS * 2:
                raise RuntimeError(f"unified T3 graph returned {len(ort_cache)} cache outputs")
            return logits, ort_cache
        values = self._validate_cache(cache, embeds.shape[0])
        result = self._run(
            {"inputs_embeds": embeds, "cache_position": positions}
            | {f"past_{index}": value for index, value in enumerate(values)}
        )
        return result.outputs[0], tuple(result.outputs[1:])


class FlowEncoderOrtRuntime(StageSession):
    """Run the exact S3Gen token-to-condition encoder stage."""

    stage = "flow_encoder"

    def encode(self, speech_tokens: Any, token_lengths: Any) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(speech_tokens, dtype=np.int64)
        lengths = np.asarray(token_lengths, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError(f"speech_tokens must have shape (1,T), got {tokens.shape}")
        if lengths.shape != (1,) or int(lengths[0]) < 1 or int(lengths[0]) > tokens.shape[1]:
            raise ValueError(f"token_lengths must have shape (1,) and be <= T, got {lengths}")
        result = self._run({"speech_tokens": tokens, "token_lengths": lengths})
        if len(result.outputs) != 2:
            raise RuntimeError(f"flow encoder graph returned {len(result.outputs)} outputs")
        return result.outputs[0], result.outputs[1]


class MeanflowEstimatorOrtRuntime(StageSession):
    """Run one deterministic meanflow estimator call.

    The Euler loop and classifier-free guidance remain Python-side.  ``x`` is
    therefore an explicit noise/state input and is never sampled inside the
    graph, which keeps seeds reproducible across ORT providers.
    """

    stage = "meanflow_estimator"

    def estimate(
        self,
        x: Any,
        mask: Any,
        mu: Any,
        t: Any,
        speaker_embedding: Any,
        cond: Any,
        r: Any,
    ) -> np.ndarray:
        x_value = _as_float32(x, "x")
        mask_value = _as_float32(mask, "mask")
        mu_value = _as_float32(mu, "mu")
        cond_value = _as_float32(cond, "cond")
        speaker_value = _as_float32(speaker_embedding, "speaker_embedding")
        t_value = np.asarray(t, dtype=np.float32)
        r_value = np.asarray(r, dtype=np.float32)
        if x_value.ndim != 3 or x_value.shape[0] != 1 or x_value.shape[1] != 80:
            raise ValueError(f"x must have shape (1,80,T), got {x_value.shape}")
        if mu_value.shape != x_value.shape or cond_value.shape != x_value.shape:
            raise ValueError("mu and cond must have the same shape as x")
        if mask_value.shape != (1, 1, x_value.shape[2]):
            raise ValueError(f"mask must have shape (1,1,T), got {mask_value.shape}")
        if speaker_value.shape != (1, 192):
            raise ValueError(f"speaker_embedding must have shape (1,192), got {speaker_value.shape}")
        if t_value.shape != (1,) or r_value.shape != (1,):
            raise ValueError("t and r must have shape (1,)")
        result = self._run(
            {
                "x": x_value,
                "mask": mask_value,
                "mu": mu_value,
                "t": t_value,
                "speaker_embedding": speaker_value,
                "cond": cond_value,
                "r": r_value,
            }
        )
        if len(result.outputs) != 1:
            raise RuntimeError(f"meanflow estimator graph returned {len(result.outputs)} outputs")
        return result.outputs[0]


class VocoderOrtRuntime(StageSession):
    """Run the optional HiFT vocoder stage when a graph was verified."""

    stage = "vocoder"

    def synthesize(self, speech_feat: Any, *, phase_noise: Any, sine_noise: Any) -> np.ndarray:
        features = _as_float32(speech_feat, "speech_feat")
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[1] != 80:
            raise ValueError(f"speech_feat must have shape (1,80,T), got {features.shape}")
        phase = _as_float32(phase_noise,"phase_noise")
        noise = _as_float32(sine_noise,"sine_noise")
        if phase.shape != (1,9,1) or noise.shape != (1,9,features.shape[2]*480):
            raise ValueError("Expected phase (1,9,1) and sine noise (1,9,mel_length*480)")
        result = self._run({"speech_feat": features,"phase_noise":phase,"sine_noise":noise})
        if len(result.outputs) != 1:
            raise RuntimeError(f"vocoder graph returned {len(result.outputs)} outputs")
        return result.outputs[0]


def list_stages(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Read staged manifests without loading ORT or PyTorch models."""

    root = Path(model_dir).expanduser().resolve()
    result: dict[str, Any] = {"model_dir": str(root), "stages": {}}
    for stage in ("t3", "flow_encoder", "meanflow_estimator", "vocoder", "watermark"):
        path = root / stage / "manifest.json"
        if not path.exists():
            result["stages"][stage] = {"status": "not_attempted"}
            continue
        try:
            manifest = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - diagnostic path
            result["stages"][stage] = {"status": "invalid_manifest", "error": str(exc)}
        else:
            result["stages"][stage] = {
                "status": manifest.get("status"),
                "graph": manifest.get("graph"),
                "reason": manifest.get("reason"),
            }
    return result


__all__ = [
    "DEFAULT_CUDA_GPU_MEM_LIMIT_MIB",
    "FlowEncoderOrtRuntime",
    "MeanflowEstimatorOrtRuntime",
    "ORT_PROVIDER_CHOICES",
    "OrtRun",
    "StageSession",
    "T3UnifiedOrtRuntime",
    "VocoderOrtRuntime",
    "_cuda_provider_options",
    "_normalise_ort_provider",
    "list_stages",
]
