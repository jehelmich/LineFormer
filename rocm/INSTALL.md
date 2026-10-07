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
$VENV/bin/lineformer --ckpt iter_3000.pth --device cuda:0 --out /tmp/lf_check demo/PMC5959982___3_HTML.jpg
# expect: 3 lines of 621 points in /tmp/lf_check/PMC5959982___3_HTML.json
```

To compare a new environment with a reference run on your own images, use `tools/equivalence/run.py` and
`compare.py` (see their docstrings).

## Use

Command line (one `<stem>.json` with the lines per image; `--masks` adds the kept masks; existing outputs are
skipped):

```bash
$VENV/bin/lineformer --ckpt iter_3000.pth --device cuda:0 --out out/ --list images.txt
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
  Run one GPU process per 24 GB card, or several through `tools/throughput/batch_infer.py --gpu-workers N`,
  which splits `--gpu-mem-budget` between them.
- Under WSL every process that initialises the ROCm runtime keeps ~2 CPU cores busy; prefer one long-lived
  process over many short ones.
