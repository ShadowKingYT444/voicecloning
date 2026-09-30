# Environment reconstruction

The original lab used Linux, Python 3.12, an RTX 4060 Laptop GPU, and a systemd user manager with cgroup v2. These notes describe the observed setup. A fresh installation on another machine has not been tested as part of the repository handoff. Run the numerical and output checks again after reconstruction.

The old environments reused packages through absolute `.pth` paths into `voicebox/backend/venv`. Do not copy those environments. The JSON files in [environment](environment/) record observed distribution names and versions without credentials or environment variables. They are evidence, not complete resolver lockfiles.

## Three environments

| Environment | Purpose | Observed key packages |
|---|---|---|
| `.venv-nano` | Native CUDA fitting, inference, and evaluation | Torch/torchaudio `2.11.0+cu128`, transformers `4.57.3`, diffusers `0.39.0`, NumPy `1.26.4` |
| `.venv-nano-cpu` | CPU ONNX, reference preparation, mastering, Small ASR | Torch `2.11.0+cpu`, ONNX Runtime `1.29.0`, NumPy `1.26.4` |
| `.venv-nano-ort` | CUDA ONNX with CPU preparation dependencies | ONNX Runtime GPU `1.26.0`, cuDNN `9.19.0.56`, cuBLAS `12.8.4.1` |

The vendored Chatterbox `pyproject.toml` declares different dependency versions from the measured lab environment. Install the vendored source with `--no-deps` after choosing the environment packages. Preserve its local lazy-import edits.

## Reconstruction recipe

Install Git, Python 3.12, `uv`, FFmpeg, and the GPU driver appropriate to the new machine. Check that `systemctl --user` and `systemd-run --user` work before launching the resource guard. The guard currently uses Linux `/proc` and cgroup v2; it is not a macOS launcher.

The commands below are a starting recipe for the recorded versions. They install packages but do not download speech model weights or run fitting. Review dependency resolution on the target machine. Do not silently substitute a different Torch, ORT, or cuDNN version and reuse an old verification result.

```bash
uv venv --python 3.12 .venv-nano
uv pip install --python .venv-nano/bin/python torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-nano/bin/python -r requirements/nano-common.txt -r requirements/nano-analysis.txt onnxruntime==1.29.0
uv pip install --python .venv-nano/bin/python --no-deps -e vendor/chatterbox

uv venv --python 3.12 .venv-nano-cpu
uv pip install --python .venv-nano-cpu/bin/python torch==2.11.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-nano-cpu/bin/python -r requirements/nano-common.txt -r requirements/nano-analysis.txt onnxruntime==1.29.0
uv pip install --python .venv-nano-cpu/bin/python --no-deps -e vendor/chatterbox

uv venv --python 3.12 .venv-nano-ort
uv pip install --python .venv-nano-ort/bin/python numpy==1.26.4 onnxruntime-gpu==1.26.0 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12==9.19.0.56
```

The CUDA ORT environment can share the new CPU dependencies through a local path file. Its own GPU ORT package takes priority. This reproduces the lab's separation without importing the native CUDA Torch environment into ONNX stages:

```bash
.venv-nano-ort/bin/python - <<'PY'
from pathlib import Path
import sysconfig
root = Path.cwd().resolve()
paths = [root / '.venv-nano-cpu/lib/python3.12/site-packages', root / 'vendor/chatterbox/src']
assert all(p.is_dir() for p in paths)
(Path(sysconfig.get_paths()['purelib']) / 'nano_shared.pth').write_text(''.join(str(p) + '\n' for p in paths))
PY
```

The recorded native cuDNN version was also `9.19.0.56`. Check the resolved native package set against `native-observed.json`; Torch installation may choose another version. The new acoustic CUDA comparison already fails in the recorded environment. Recreating it is for diagnosis, not a claim of correctness.

The core launchers expect these exact environment directory names. Optional listening-page rendering needs Node and a browser executable; inspect `artifacts/nano_lab/render_delivery.mjs` for its host-specific browser path before using it. Voicebox and the earlier baseline packages have separate dependency instructions in their own READMEs.

## Verification before model work

Restore assets first. Then run the small pending checks from [the handoff](EXPERIMENT_HANDOFF.md), using the resource guard. The publication task checked source syntax and repository contents only. It did not install a fresh ML stack, rerun training, or establish new audio quality results.
