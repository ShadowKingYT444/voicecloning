# Isolated GPU ONNX environment

`.venv-nano-ort` contains `onnxruntime-gpu==1.26.0`. It shares the existing CPU
Torch environment, backend dependencies, and local vendor source through its
own `.pth` file. The working CPU and PyTorch CUDA environments were not modified.
Installation output is in `ort_gpu_environment_install.log`. The
`nano-clone-onnx` launcher uses `.venv-nano-cpu` by default. It selects
`.venv-nano-ort` only when `--ort-provider cuda` or `--ort-provider=cuda` is
present. Every synthesis still runs through `bounded_job.py --small-job`.

Version 1.26 uses CUDA 12.8 and cuDNN 9 according to the
[official CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
The local backend already supplies CUDA 12.8 dependencies. The guarded smoke
run now loads the CUDA provider and executes CUDA kernels. It does not promote
the GPU path to the default because the long-context T3 cache has not completed
its full verification.

## Guarded commands

The CPU path remains the default and needs no extra option:

```bash
./nano-clone-onnx \
  --voice asmr_soft \
  --experimental-t3 \
  --text "Take a quiet moment and let yourself relax." \
  --output samples/asmr_onnx.wav
```

Use the isolated GPU environment explicitly for the experimental CUDA path:

```bash
./nano-clone-onnx \
  --ort-provider cuda \
  --cuda-kv-resident \
  --voice asmr_soft \
  --experimental-t3 \
  --text "Take a quiet moment and let yourself relax." \
  --output samples/asmr_onnx_cuda.wav
```

`--cuda-kv-resident` keeps the T3 key/value cache in CUDA `OrtValue` objects
between decode steps. It is opt-in, and it is valid only with
`--ort-provider cuda`. The `run` command still requires `--experimental-t3`
until the long cache reference is verified. Use `--ort-profile` with the CUDA
command when provider kernel evidence is required. A CPU fallback is reported
as a failed CUDA run.

The latest guarded `soft_cuda_resident` smoke report recorded a 1023.09 MiB
full-tree RSS peak and 37.758 seconds wall time for 7.36 seconds of audio.
Its speech tokens and mel output matched the corresponding
`soft_cuda_unprofiled` run. This is a short smoke measurement, not a quality
claim or a completed long-context benchmark. Stage and output metadata are in
`artifacts/nano_lab/onnx_pipeline/soft_cuda_resident/run.json`; the sampled RSS
and matched-run comparison are recorded in `artifacts/nano_lab/RSS_REPORT.md`.

Dataset repair takes priority. Any later GPU provider test must use the existing
resource guard and the single model-job lock. A CPU fallback must not count as
a successful GPU benchmark.

CUDA initial smoke failed cleanly: ORT's preload_dlls(directory="") searches beside its own site-packages and does not follow shared .pth paths for native libraries. Existing NVIDIA CUDA12 libraries are now linked into the isolated ORT environment via a directory symlink; no duplicate package download or global environment change. Keep future CUDA runs guarded by the resource wrapper and verify that CUDA kernels execute.
