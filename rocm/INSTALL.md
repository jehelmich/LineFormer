<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Install, update and use (ROCm / CUDA / CPU)

LineFormer gets its own Python environment: the install pins the exact versions the results were validated with
(Python 3.13, torch 2.14.1, numpy 2.5, OpenCV 5.0, ...). Keep it separate from the project that calls it, and call
it as a command, a subprocess or over HTTP (`lineformer serve`).

## Why not `pip install git+https://...`

`pyproject.toml` installs only this repository's modules and the `lineformer` commands. torch for ROCm comes from
its own index, and mmcv and mmdetection are installed from the checkout: `third_party/mmcv` is the pure-Python part
of mmcv 1.7.2 that inference uses (no compiled ops; see its `NOTICE.md`), `mmdetection/` is mmdet 2.28.2.
`install_rocm.sh` does all of it in the right order. Nothing is compiled.

## Fresh install (AMD GPU, Linux or WSL2)

Needs ROCm in `/opt/rocm` (tested 7.2.0), `git` and [uv](https://docs.astral.sh/uv/). No compiler.

```bash
git clone https://github.com/jehelmich/LineFormer.git ~/LineFormer   # main; or a release: git checkout v0.3.0 (v0.2.0 and earlier build mmcv-full)
cd ~/LineFormer
VENV=$HOME/lineformer bash rocm/install_rocm.sh     # ~16 GB; under a minute with the wheels in uv's cache
```

The script creates the venv (Python 3.13; `PYTHON=` to change it), installs torch 2.14.1+rocm7.2 and torchvision 0.29.1+rocm7.2, swaps in
`/opt/rocm`'s `libhsa-runtime64.so` under WSL, installs the pinned dependencies, pytest, the vendored mmcv subset,
the vendored mmdetection and this repository (all three editable), and prints
`torch.__version__ torch.version.hip torch.cuda.is_available()` - the last value must be `True`.

The checkpoint is not in the repository: download `iter_3000.pth` from the authors' link in the main README and put
it into the checkout (`~/LineFormer/iter_3000.pth`) or `~/.cache/lineformer/`, or point `LINEFORMER_CKPT` at it
(`--ckpt FILE` also works). The command searches `--ckpt`, `$LINEFORMER_CKPT`, the checkout, then the cache, and
names the paths it searched when it finds none.

CPU only: the same script with the CPU wheels of torch (tested):

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cpu TORCH_PKGS="torch==2.14.1 torchvision==0.29.1" \
    VENV=$HOME/lineformer_cpu bash rocm/install_rocm.sh
```

NVIDIA GPU: the same with the CUDA index of PyTorch (`TORCH_INDEX=https://download.pytorch.org/whl/cu<version>`) and
the matching `TORCH_PKGS` (not tested by this fork). MultiScaleDeformableAttention then runs mmcv's pure-PyTorch implementation on the GPU,
as on ROCm (`--msda auto` resolves to `pytorch`: the subset has no compiled kernel). For mmcv's compiled CUDA kernel, use
the authors' environment instead (`install.sh`: Python 3.8, torch 1.13.1, CUDA 11.7, mmcv-full via `mim`, then
`pip install --no-deps -e .` in the checkout; not tested by this fork).

### What the script works around (tested setup)

1. WSL only: the `libhsa-runtime64.so` bundled with the torch wheel does not see the GPU
   (`torch.cuda.is_available()` is False). It is replaced by the system one from `/opt/rocm/lib`; the original is
   kept in `$VENV/torch_lib_backup/`. On native Linux this step is skipped (untested).
2. mmcv: no build. LineFormer inference calls no compiled mmcv op (measured: 0 calls into `mmcv._ext` on CPU and
   GPU), so `third_party/mmcv` holds only the pure-Python modules it loads; the compiled ops that mmdet imports are
   stand-ins that raise `OpUnavailableError` when called. Outputs are bit-identical to the earlier mmcv-full build
   with CPU ops (`docs/VALIDATION.md`). The CPU-only variant above passed the tests and the demo (3 lines of 621
   points).
3. Current dependencies: Python 3.13, numpy 2.5.2, opencv-python 5.0.0.93, scipy 1.18.1, scikit-image 0.26.0,
   matplotlib 3.11.2, any yapf. They needed these patches: mmcv `utils/config.py` (yapf >= 0.40.2 has no
   `FormatCode(verify=)`), mmcv `utils/ext_loader.py` (`pkgutil.find_loader` is gone in Python 3.14), mmdet
   `setup.py` (`exec` into `locals()` no longer works in Python 3.13, PEP 667), and `np.int` -> `int` in five lines
   of mmdet training/dataset code (`datasets/custom.py`, `datasets/openimages.py`,
   `core/bbox/samplers/iou_balanced_neg_sampler.py`; numpy >= 1.24 removed the alias). Results are unchanged: CPU
   outputs are bit-identical to the old stack (Python 3.11, numpy 1.23.5, OpenCV 4.11), see `docs/VALIDATION.md`.
4. The vendored `mmdetection/` has only the patches in 3.

On a GPU, `msda_compat.py` chooses the MultiScaleDeformableAttention path (`auto` | `compiled` | `pytorch`; option
`msda=` of `infer.load_model`, `msda` in the command's settings file, or env `LINEFORMER_MSDA`). `pytorch` sets
`mmcv.ops.multi_scale_deform_attn.IS_CUDA_AVAILABLE = False`, so GPU tensors take mmcv's own
`multi_scale_deformable_attn_pytorch` branch (the one CPU tensors always take); `auto` tries a tiny call through
the compiled kernel and falls back to `pytorch` (always, with the vendored subset); `compiled` raises without a
compiled kernel. On `cpu` nothing is changed.

## Check an environment

```bash
VENV=$HOME/lineformer
$VENV/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
$VENV/bin/python -m pytest -q          # unit tests, no GPU and no checkpoint needed (or: python tests/run_all.py)
$VENV/bin/lineformer --out /tmp/lf_check demo/PMC5959982___3_HTML.jpg
$VENV/bin/lineformer --cpu --out /tmp/lf_check_cpu demo/PMC5959982___3_HTML.jpg
# expect in both: 3 lines of 621 points in PMC5959982___3_HTML.json (line order can differ between CPU and GPU)
```

To compare a new environment with a reference run on your own images, use `tools/equivalence/run.py` and
`compare.py` (see their docstrings and `docs/VALIDATION.md`).

## Update

The repository is installed editable, so pulled code is live at once:

```bash
cd ~/LineFormer && git fetch --tags && git checkout v0.3.0     # a release; or: git checkout main && git pull
```

* Re-run `uv pip install --python $HOME/lineformer/bin/python --no-deps --no-build-isolation -e .` if
  `pyproject.toml` changed (new modules or commands).
* Re-run `install_rocm.sh` into a NEW `VENV=` path if the pinned versions in it changed; switch over after the new
  venv passes the check above (and, for results that matter, `tools/equivalence` against your old venv).
* Do not upgrade packages inside the venv by hand (`pip install -U ...`): the pins are the validated versions;
  another torch, numpy or OpenCV can change results. Validate a new set with `tools/equivalence` first.

## Use

All from the venv. For more than a handful of images use `batch` or `serve`: they run the GPU work in worker
processes (pre-processing workers -> GPU workers -> post-processing workers, `lineformer_engine.py`), so a caller
never writes its own multi-process GPU code.

The three forms share a small set of options:

```
lineformer IMAGE... | --list FILE  --out DIR [--threshold T] [--masks] [--force] [--cpu] [--settings FILE.toml]
lineformer batch    (the same options)
lineformer serve    [--port 8775] [--exit-on-failure] [--threshold T] [--masks] [--cpu] [--settings FILE.toml]
```

* Device: the GPU (`cuda:0`, ROCm too) if PyTorch sees one, else the CPU; `--cpu` forces the CPU.
* `--threshold T` (default 0.3): the kept-queries mode. Only the instances whose class score reaches T are
  post-processed and returned (~1 GB per GPU worker instead of ~12 GB); a line also needs a final score (class x
  mask score) > 0.3, so T <= 0.3 gives the same lines as all queries, a lower T keeps lower-scoring instances in the
  `.npz` outputs, and T > 0.3 drops lines whose class score is below T.
* `--masks` (batch, serve): also write `<id>.masks.npz` and `<id>.instances.npz`; for the single process the masks
  of the instances behind the lines.
* Everything else goes into a settings file, `--settings FILE.toml`: [lineformer.example.toml](../lineformer.example.toml)
  lists every key with its default (model config, input size and tiling, both EXPERIMENTAL, `all_queries` for
  validation, ids, GPU / pre / post workers, threads, memory budget, MSDA path). Unknown keys are an error.
* The flags of v0.2.0 (`--gpu-workers`, `--gpu-mem-budget`, `--kept-thr`, `--instances`, `--input-size`,
  `--device`, ...) still work in this version, with a deprecation warning that names the settings key; a flag and
  the settings file that disagree are an error.

**One job, one command** (`lineformer batch`; list file: `<path>` or `<id><TAB><path>` per line):

```bash
$VENV/bin/lineformer batch --list images.txt --out out/ --masks
```

Per image `out/<id>.json` (`lines` as `get_dataseries`, image path and sha256, the options' fingerprint, timings),
with `--masks` `<id>.instances.npz` (boxes N x 5 with scores, labels; only the kept queries) and `<id>.masks.npz`
(packed masks of the same N instances). Lines and instances are in a geometric order (lines by leftmost x, mean y,
score; line *i* = instance *i*), the same on CPU and GPU. `out/job.json` is the manifest of the job (fork version
and git commit, package versions, the effective settings with the automatic sizing, checkpoint path and sha256,
MSDA path, workers, per-image status, timings and output sha256, start and end times, back-off events, failures).
Ids are the file stems unless given; two different images with one id stop the command before it runs (set
`ids = "parent_stem"` for `<dir>__<stem>`, or give explicit ids).
Writes are atomic and a rerun skips what is done (`--force` recomputes). An unreadable image fails alone. Exit code
0 all done, 1 some images failed, 2 engine failure or bad input, 130 interrupted (Ctrl-C lets in-flight images
finish; rerun to resume). `lineformer_jobs.py` has the exact rules.

GPU workers and memory (`lineformer_engine.py`, "Automatic sizing"): at start the engine measures the free device
memory, keeps max(2 GiB, 10 %) for other processes, and starts min(2, usable / need) GPU workers (need: ~2.4 GB
per worker in the kept mode, ~19.5 GB with `all_queries`; measured peaks with a safety factor of 1.5); each worker
is capped at its share of the usable memory. The decision is printed and stored in the manifest. Not even one
worker fits: the command stops with the numbers (free, headroom, need); free GPU memory or use `--cpu`.
`gpu_workers` / `gpu_mem_budget` in the settings file override it (2 workers are the measured best on the
RX 7900 XTX, ~20 images/s with 8 pre-processing workers, the default on a machine with >= 24 CPUs:
`min(8, max(2, CPUs // 3))`). Out of GPU memory on an image (bounded back-off): that worker stops and is not
restarted, and the image goes once to a remaining worker; a second out of memory of the same image, or one on the
last worker, ends the run with exit code 2. Every back-off event is printed and recorded in the manifest.

**A server that owns the GPU** (models loaded once; several callers share it; jobs FIFO, `priority` first):

```bash
$VENV/bin/lineformer serve --port 8775
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
Model options (input size, threshold, tiling, device) are per server: a job that needs others needs a second
server on another port. `serve --masks` makes every job write the masks and instances. SIGTERM / Ctrl-C drains
(in-flight images finish, unfinished jobs end "interrupted"; resubmitting skips what is done). Out of GPU memory
gets the same back-off as in `batch`. If the engine fails (models do not load, a GPU worker dies, out of memory
after the back-off), its running jobs end "failed" and the server answers 503 until stopped; `--exit-on-failure`
makes it exit at once with code 2 instead, for a supervisor that restarts it. The server binds 127.0.0.1 and has
no authentication.

**Single process** (the original command; `<id>.json` per image with the lines in the model's order, `--masks` adds
the masks behind the lines):

```bash
$VENV/bin/lineformer --list images.txt --out out/
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
