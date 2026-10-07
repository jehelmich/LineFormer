# Install, update and use (ROCm / CUDA / CPU)

LineFormer needs its own Python environment: mmcv-full 1.7.x, mmdet 2.x and numpy 1.23 do not mix with
current stacks. Keep it separate from the project that calls it, and call it as a command or a subprocess.

## Fresh install (AMD GPU, Linux or WSL2)

Needs ROCm in `/opt/rocm` (tested 7.2.0), `git`, `gcc` with C++20, and [uv](https://docs.astral.sh/uv/).

```bash
git clone -b pure-pytorch-inference https://github.com/jehelmich/LineFormer.git ~/LineFormer
cd ~/LineFormer
VENV=$HOME/lineformer bash rocm/install_rocm.sh     # ~15 min (mmcv build), ~16 GB
```

The script creates the venv, installs torch 2.14.1+rocm7.2, swaps in `/opt/rocm`'s `libhsa-runtime64.so` under
WSL, builds mmcv-full 1.7.2 with CPU ops only, installs the vendored mmdetection and this repo (editable), and
prints `torch.__version__ torch.version.hip torch.cuda.is_available()` - the last value must be `True`.

The checkpoint is not in the repository: download `iter_3000.pth` from the link in the main README.

## Update

The repository is installed editable, so pulled code is live at once:

```bash
cd ~/LineFormer && git pull
```

Re-run `uv pip install --python $HOME/lineformer/bin/python --no-deps --no-build-isolation -e .` only if
`pyproject.toml` changed (new modules or commands). Re-run `install_rocm.sh` into a NEW `VENV=` path only if the
pinned versions in it changed; switch over after the new venv passes the check below. Do not upgrade packages
inside the venv by hand (`pip install -U ...`): numpy 2, opencv 5 or a newer torch break mmcv 1.x or change
results; such a change needs the equivalence check.

## Check an environment

```bash
VENV=$HOME/lineformer
$VENV/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
$VENV/bin/python tools/equivalence/tests/test_compare.py
$VENV/bin/python tests/test_engine.py          # job engine pieces, HTTP API + client (no GPU, no checkpoint)
$VENV/bin/lineformer --ckpt iter_3000.pth --device cuda:0 --out /tmp/lf_check demo/PMC5959982___3_HTML.jpg
# expect: 3 lines of 621 points in /tmp/lf_check/PMC5959982___3_HTML.json
```

To compare a new environment with a reference run on your own images, use `tools/equivalence/run.py` and
`compare.py` (see their docstrings).

## Use

Three ways, all from the venv. For anything more than a handful of images use `batch` or `serve`: they run the
GPU work in worker processes (pre-processing workers -> N GPU workers -> post-processing workers,
`lineformer_engine.py`), so a caller never writes its own multi-process GPU code.

**One job, one command** (`lineformer batch`; list file: `<path>` or `<id><TAB><path>` per line):

```bash
$VENV/bin/lineformer batch --ckpt iter_3000.pth --list images.txt --out out/ \
    --kept-only --gpu-workers 2 --pre-workers 8 [--instances] [--masks]
```

Per image `out/<id>.json` (`lines` as `get_dataseries`, image path and sha256, the options' fingerprint, timings),
with `--instances` `<id>.instances.npz` (boxes N x 5 with scores, labels) and with `--masks` `<id>.masks.npz`
(packed masks of the same N instances); `out/job.json` is the manifest (options, versions, MSDA path, per-image
status and timings, errors). Ids are the file stems unless given (`--ids parent_stem` for `<dir>__<stem>`); two
different images with one id fail the second one, never overwrite. Writes are atomic and a rerun skips what is
done (`--force` recomputes). An unreadable image fails alone; out of GPU memory or a dead worker ends the run.
Exit code 0 all done, 1 some images failed, 2 engine failure, 130 interrupted (Ctrl-C lets in-flight images
finish; rerun to resume). `lineformer_jobs.py` has the exact rules.

**A server that owns the GPU** (models loaded once; several callers share it; jobs FIFO, `priority` first):

```bash
$VENV/bin/lineformer serve --ckpt iter_3000.pth --port 8775 --kept-only --gpu-workers 2 --pre-workers 8
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

Recommended settings: `--kept-only` for every consumer of the lines or of instances above the threshold (the same
lines, ~1 GB device memory per GPU worker instead of ~12 GB); `--kept-thr 0.1` (or lower) if low-score instances
are wanted in `.instances.npz` / `.masks.npz` (lines still use score > 0.3); `--gpu-workers 2 --pre-workers 8`
(measured ~20 images/s on the RX 7900 XTX, see NOTES.md); `--gpu-mem-budget` (default 0.85 of the device, or a
size such as `4G`) split between the workers - with kept-only `4G` for two workers leaves the card to others.
Without `--kept-only` keep one GPU worker per 24 GB card. Other input sizes: `--input-size N|native`,
`--tile CROP --tile-overlap O` (native crops, merged; `scale_compat.py`, `tiling.py`).

Single process (the original command, unchanged; `<stem>.json` per image, `--masks` adds the kept masks):

```bash
$VENV/bin/lineformer --ckpt iter_3000.pth --device cuda:0 --out out/ --list images.txt [--kept-only]
```

Python, inside the venv (`infer` is importable from anywhere after the install):

```python
import cv2, infer
from mmdet.apis import inference_detector
infer.load_model("<repo>/lineformer_swin_t_config.py", "iter_3000.pth", "cuda:0")   # msda="auto" by default
lines = infer.get_dataseries(cv2.imread("chart.png"), to_clean=False)               # as upstream
bbox, masks = (lambda r: (r[0][0], r[1][0]))(inference_detector(infer.model, cv2.imread("chart.png")))
```

Notes:
- Results match the original CPU stack (see the README section); the order of instances / lines can differ
  between CPU and GPU, so match lines by position, not by index.
- One GPU process peaks at ~12 GB device memory on 1.5-3.5k px images (mmdet upsamples all 100 candidate masks).
  Run one GPU process per 24 GB card, or several through `lineformer batch|serve --gpu-workers N` (or the
  benchmark runner `tools/throughput/batch_infer.py`), which split `--gpu-mem-budget` between them.
- Kept-queries mode (opt-in, `kept_queries.py`): `infer.load_model(..., kept_only=True)`, `lineformer --kept-only`,
  `batch_infer.py --kept-only 0.3` or env `LINEFORMER_KEPT_QUERIES=on` (the engine, `batch` / `serve`, ignores
  the environment variable: only `--kept-only` switches it on). Only the queries whose class score reaches
  0.3 are upsampled, scored and copied: the same lines from `get_dataseries`, ~0.5 GB peak per process instead of
  ~12 GB, ~0.05 s instead of ~0.27 s GPU time per image. Instances below the threshold are not returned, so leave it
  off for anything that reads low-scoring instances. Check: `tools/equivalence/kept_check.py` (mode off / on / off
  in one process, kept instances must be bit-identical).
- Under WSL every process that initialises the ROCm runtime keeps ~2 CPU cores busy; prefer one long-lived
  process over many short ones.
