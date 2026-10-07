<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Install, update and use (ROCm / CUDA / CPU)

LineFormer needs its own Python environment: mmcv-full 1.7.x, mmdet 2.x and numpy 1.23 do not mix with current
stacks. Keep it separate from the project that calls it, and call it as a command, a subprocess or over HTTP
(`lineformer serve`).

## Why not `pip install git+https://...`

`pyproject.toml` installs only this repository's modules and the `lineformer` commands. torch, mmcv-full and the
vendored mmdetection are not on PyPI in a usable form for this stack: mmcv-full 1.7.2 must be compiled against the
installed torch (with a patch for torch >= 2.10), and torch for ROCm comes from its own index. `install_rocm.sh`
does all of it in the right order.

## Fresh install (AMD GPU, Linux or WSL2)

Needs ROCm in `/opt/rocm` (tested 7.2.0), `git`, `gcc` with C++20, and [uv](https://docs.astral.sh/uv/).

```bash
git clone -b v0.2.0 https://github.com/jehelmich/LineFormer.git ~/LineFormer   # or -b main for the newest
cd ~/LineFormer
VENV=$HOME/lineformer bash rocm/install_rocm.sh     # ~15 min (mmcv build), ~16 GB
```

The script creates the venv (Python 3.11), installs torch 2.14.1+rocm7.2 and torchvision 0.29.1+rocm7.2, swaps in
`/opt/rocm`'s `libhsa-runtime64.so` under WSL, builds mmcv-full 1.7.2 with CPU ops only, installs the pinned
dependencies, pytest, the vendored mmdetection and this repository (editable), and prints
`torch.__version__ torch.version.hip torch.cuda.is_available()` - the last value must be `True`.

The checkpoint is not in the repository: download `iter_3000.pth` from the authors' link in the main README.

NVIDIA GPU or CPU only: the authors' environment (`install.sh`: Python 3.8, torch 1.13.1, CUDA 11.7, mmcv-full via
`mim`) and then `pip install --no-deps -e .` in the checkout. mmcv's compiled MSDA kernel is then used
(`--msda auto`). Not tested by this fork.

### What the script works around (tested setup)

1. WSL only: the `libhsa-runtime64.so` bundled with the torch wheel does not see the GPU
   (`torch.cuda.is_available()` is False). It is replaced by the system one from `/opt/rocm/lib`; the original is
   kept in `$VENV/torch_lib_backup/`. On native Linux this step is skipped (untested).
2. mmcv `setup.py`, patch `mmcv-1.7.2-cpu-ops.patch`: `-std=c++20` for torch >= 2.10 (its headers require C++20),
   and `MMCV_CPU_ONLY=1` skips the GPU branch, which mmcv would otherwise build on its own because it detects ROCm
   (`torch.version.hip`, `ROCM_HOME`) even with `FORCE_CUDA=0`. The build log says "Compiling mmcv._ext only with
   CPU" and `mmcv.ops.get_compiling_cuda_version()` returns "not available".
3. numpy 1.23.5 (mmdet 2.28 still uses `np.int`); opencv-python 4.11.0.86 (5.x needs numpy >= 2); yapf 0.40.1
   (mmcv 1.x calls `FormatCode(verify=...)`, removed later); scipy 1.9.3, scikit-image 0.21.0, matplotlib 3.7.5.
4. No change in the vendored `mmdetection/`.

On a GPU, `msda_compat.py` chooses the MultiScaleDeformableAttention path (`auto` | `compiled` | `pytorch`; option
`msda=` of `infer.load_model`, `--msda`, or env `LINEFORMER_MSDA`). `pytorch` sets
`mmcv.ops.multi_scale_deform_attn.IS_CUDA_AVAILABLE = False`, so GPU tensors take mmcv's own
`multi_scale_deformable_attn_pytorch` branch (the one CPU tensors always take); `auto` tries a tiny call through
the compiled kernel and falls back to `pytorch`. On `cpu` nothing is changed.

## Check an environment

```bash
VENV=$HOME/lineformer
$VENV/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
$VENV/bin/python -m pytest -q          # unit tests, no GPU and no checkpoint needed (or: python tests/run_all.py)
$VENV/bin/lineformer --ckpt iter_3000.pth --out /tmp/lf_check demo/PMC5959982___3_HTML.jpg
$VENV/bin/lineformer --ckpt iter_3000.pth --device cpu --out /tmp/lf_check_cpu demo/PMC5959982___3_HTML.jpg
# expect in both: 3 lines of 621 points in PMC5959982___3_HTML.json (line order can differ between CPU and GPU)
```

To compare a new environment with a reference run on your own images, use `tools/equivalence/run.py` and
`compare.py` (see their docstrings and `docs/VALIDATION.md`).

## Update

The repository is installed editable, so pulled code is live at once:

```bash
cd ~/LineFormer && git fetch --tags && git checkout v0.2.0     # a release; or: git checkout main && git pull
```

* Re-run `uv pip install --python $HOME/lineformer/bin/python --no-deps --no-build-isolation -e .` if
  `pyproject.toml` changed (new modules or commands).
* Re-run `install_rocm.sh` into a NEW `VENV=` path if the pinned versions in it changed; switch over after the new
  venv passes the check above (and, for results that matter, `tools/equivalence` against your old venv).
* Do not upgrade packages inside the venv by hand (`pip install -U ...`): numpy 2, opencv 5 or another torch break
  mmcv 1.x or change results.

## Use

All from the venv. For more than a handful of images use `batch` or `serve`: they run the GPU work in worker
processes (pre-processing workers -> N GPU workers -> post-processing workers, `lineformer_engine.py`), so a caller
never writes its own multi-process GPU code.

Defaults of every form: `--device auto` (cuda:0 if PyTorch sees a GPU, else cpu; printed), kept-queries mode on at
0.3 (only queries whose class score reaches 0.3 are post-processed and returned: the same lines, ~1 GB per GPU
process instead of ~12 GB). `--all-queries` switches it off (all 100 instances, as upstream); `--kept-thr 0.1`
keeps lower-scoring instances in `.instances.npz` / `.masks.npz` (lines still use score > 0.3).

**One job, one command** (`lineformer batch`; list file: `<path>` or `<id><TAB><path>` per line):

```bash
$VENV/bin/lineformer batch --ckpt iter_3000.pth --list images.txt --out out/ --gpu-workers 2 [--instances] [--masks]
```

Per image `out/<id>.json` (`lines` as `get_dataseries`, image path and sha256, the options' fingerprint, timings),
with `--instances` `<id>.instances.npz` (boxes N x 5 with scores, labels; with the default kept mode only the kept
queries) and with `--masks` `<id>.masks.npz` (packed masks of the same N instances); `out/job.json` is the manifest
(options, versions, MSDA path, per-image status and timings, errors). Ids are the file stems unless given
(`--ids parent_stem` for `<dir>__<stem>`); two different images with one id fail the second one, never overwrite.
Writes are atomic and a rerun skips what is done (`--force` recomputes). An unreadable image fails alone; out of
GPU memory or a dead worker ends the run. Exit code 0 all done, 1 some images failed, 2 engine failure, 130
interrupted (Ctrl-C lets in-flight images finish; rerun to resume). `lineformer_jobs.py` has the exact rules.

**A server that owns the GPU** (models loaded once; several callers share it; jobs FIFO, `priority` first):

```bash
$VENV/bin/lineformer serve --ckpt iter_3000.pth --port 8775 --gpu-workers 2
```

Client, from any Python >= 3.8 environment without torch (`lineformer_client.py` is one stdlib-only file: copy it,
or put the checkout on `sys.path`):

```python
from lineformer_client import LineFormerClient, read_lines
lf = LineFormerClient("http://127.0.0.1:8775")
job = lf.submit(["/abs/a.png", {"id": "b2", "path": "/abs/b.png"}], out="/abs/out", instances=True,
                require={"kept_thr": 0.3})          # 409 if the server runs with other model options
summary = lf.wait(job)                               # raises JobFailed unless every image is done or skipped
lines = read_lines("/abs/out", "a")
# lf.status(job), lf.jobs(), lf.cancel(job), lf.health()
```

or `lineformer-client submit --list images.txt --out /abs/out --wait`. The HTTP API (`POST /jobs`,
`GET /jobs/<id>`, `POST /jobs/<id>/cancel`, `GET /jobs`, `GET /health`) is described in `lineformer_serve.py`.
Model options (input size, kept threshold, tiling, device) are per server: a job that needs others needs a second
server on another port. SIGTERM / Ctrl-C drains (in-flight images finish, unfinished jobs end "interrupted";
resubmitting skips what is done). The server binds 127.0.0.1 and has no authentication.

Recommended settings: `--gpu-workers 2` (measured ~20 images/s on the RX 7900 XTX with 8 pre-processing workers,
the default on a machine with >= 24 CPUs: `min(8, max(2, CPUs // 3))`); `--gpu-mem-budget` (default 0.85 of the
device, or a size such as `4G`) is split between the GPU workers - in the kept mode `4G` for two workers leaves the
card to others. With `--all-queries` keep one GPU worker per 24 GB card. `--input-size N|native` and
`--tile CROP --tile-overlap O` are EXPERIMENTAL (`docs/VALIDATION.md`, "Input scale"). `tools/benchmark.py`
measures a configuration on your own images.

**Single process** (the original command; `<stem>.json` per image, `--masks` adds the masks behind the lines):

```bash
$VENV/bin/lineformer --ckpt iter_3000.pth --out out/ --list images.txt
```

**Python**, inside the venv (`infer` is importable from anywhere after the install). The library keeps upstream's
behaviour: the kept-queries mode is off unless asked for (`kept_only=True`, a threshold, or env
`LINEFORMER_KEPT_QUERIES=on|<thr>`).

```python
import cv2, infer
infer.load_model("<repo>/lineformer_swin_t_config.py", "iter_3000.pth", "cuda:0", kept_only=True)  # msda="auto"
lines = infer.get_dataseries(cv2.imread("chart.png"), to_clean=False)                              # as upstream
```

Notes:
- Results match the original CPU stack (`docs/VALIDATION.md`); the order of instances / lines can differ between
  CPU and GPU, so match lines by position, not by index.
- Under WSL every process that initialises the ROCm runtime keeps ~2 CPU cores busy; prefer one long-lived process
  (`serve`) over many short ones.
