<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Validation: same results, measured speed

The question: does the fork give the results of the original LineFormer code, on a new stack, on an AMD GPU, with
the kept-queries mode and through the job engine? And how fast is it?

## Environment

AMD Radeon RX 7900 XTX (gfx1100, 24 GB), ROCm 7.2.0, WSL2 Ubuntu 24.04. Python 3.11.17, torch 2.14.1+rocm7.2,
mmcv-full 1.7.2 built with CPU ops only, mmdet 2.28.2 (vendored), numpy 1.23.5, opencv-python 4.11.0.86,
scipy 1.9.3, scikit-image 0.21.0 (`rocm/install_rocm.sh`). On the GPU, MultiScaleDeformableAttention runs mmcv's
pure-PyTorch implementation (`msda_compat.py`, path `pytorch`).

## Method

Reference (run A): the original stack, upstream commit 7952e27, Python 3.8, torch 1.13.1 CPU, mmcv-full 1.7.2 with
its compiled ops. Every candidate run saves all instances (boxes, scores, masks) and the `get_dataseries` lines per
image (`tools/equivalence/run.py`; engine outputs via `to_harness.py`), and `compare.py` judges each image against
acceptance criteria fixed before measuring:

1. the same number of instances with score >= 0.3, matched one-to-one by mask IoU;
2. every matched pair: mask IoU >= 0.98 and |score difference| <= 0.01;
3. the lines: the same number, and >= 99 % of the points within 1 px.

A missing or failed output fails the image. Test set: 72 images, 71 windows of scanned well-log charts (private,
not distributable) and the upstream demo image `demo/PMC5959982___3_HTML.jpg`. 125 instances reach 0.3 in the
reference. All numbers are in-sample: the set was used while developing the fork.

## Equivalence (each candidate against the reference A)

| candidate | PASS | min mask IoU | max \|dscore\| | lines identical | points within 1 px |
|---|---|---|---|---|---|
| new stack on CPU (Python 3.11, torch 2.14.1, mmcv CPU ops) | 72/72 | 0.9978 | 7.5e-4 | 53/72 images | all |
| GPU, all queries (`run.py --device cuda:0 --msda pytorch`) | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |
| GPU, kept-queries mode 0.3 (`run.py --kept-only 0.3`) | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |
| engine, `lineformer batch`, all queries, 1 GPU worker | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |
| engine, `lineformer batch`, kept 0.3, 2 GPU workers | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |

"Lines identical" counts images whose lines are identical point for point; on the others the points differ by at
most 1 px. Between GPU runs: the kept-queries mode returns the 126 instances whose class score reaches 0.3 with
masks bit-identical to the full GPU path, scores within 2.3e-6 (two full GPU runs differ by up to 1.4e-6 from
each other), and identical lines on 72 of 72 images (`tools/equivalence/kept_check.py`). The engine with the kept
mode and two GPU workers gives masks bit-identical to the single-process kept run on 72 of 72 images. Server jobs
(two concurrent jobs and one cancelled mid-run) give lines identical to the batch run on every image
(`tests/integration/serve_check.py`). What was re-checked on the release commit is listed under
[Release check](#release-check-v020).

Demo image, expected output: 3 lines of 621 points each (`lineformer --ckpt iter_3000.pth --out out/
demo/PMC5959982___3_HTML.jpg`), on CPU and GPU.

### Without compiled mmcv (unreleased)

`third_party/mmcv` (the pure-Python subset of mmcv 1.7.2, no `mmcv._ext`) replaces mmcv-full 1.7.2 built with CPU
ops; every other package version is the same. Bit-identity against the mmcv-full stack, all 72 images unless
stated:

| run | masks | boxes (coordinates) | scores | lines |
|---|---|---|---|---|
| CPU, all queries, 12 images (every 7th + demo), against mmcv-full on CPU | identical 12/12 | identical | identical | identical 12/12 |
| CPU, kept 0.3, the same 12 images | identical 12/12 | identical | identical | identical 12/12 |
| GPU, kept 0.3 (`run.py`), against the earlier kept GPU run | identical 72/72 | identical | identical on 53/72, max diff 2.1e-6 | identical 72/72 |
| engine `lineformer batch`, defaults, 2 GPU workers, against the same run | identical 72/72 | identical | identical on 53/72, max diff 2.5e-6 | identical 72/72 |
| GPU, kept 0.3, a second run of the subset against its first | identical 72/72 | identical | identical on 51/72, max diff 1.7e-6 | identical 72/72 |

The GPU score differences are run-to-run variation of the GPU path (last row: the same stack against itself), not
an effect of the subset; on CPU, which is deterministic, everything is bit-identical. A profiler counted 0 calls
into compiled mmcv code on CPU and GPU, before (mmcv-full) and after; on the GPU the only stand-in that runs is
`get_compiling_cuda_version`, called by `msda_compat`'s `auto` probe, which then picks `pytorch`.

## Speed

Per image, 72 images, warm-up excluded, one machine; single runs, so differences of ~1 image/s are noise.

| setup | time per image / rate | peak device memory per process |
|---|---|---|
| original stack, CPU (reference A, detector median) | 5.8 s | - |
| GPU, all queries, one process (detector median) | 0.27 s | 12 GB |
| GPU, kept-queries mode, one process (detector) | 0.056 s | 0.48 GB allocated |
| kept-queries mode, pipelined, 1 GPU worker | 18.7 images/s | 0.48 GB |
| engine `lineformer batch`, kept, 2 GPU workers, 8 pre-processing workers | 20.0 images/s | 0.48 GB allocated, 0.78-0.86 GB reserved per worker |

With the kept mode, one GPU worker keeps the GPU busy ~90 % of the time; a second adds ~10 %; three or four lose to
CPU contention under WSL (every ROCm process keeps ~2 cores busy). With four pre-processing workers, reading and
resizing (~0.2 s per image) limits the rate (15.5-16.7 images/s). Saving all 100 masks per image is
post-processing-bound (2.0 images/s). `tools/benchmark.py` measures the engine on your own images.

## Limits

* No instance in the set has a score near the 0.3 threshold, so the set does not test how the stacks decide such
  borderline cases.
* The order of instances, and so of lines, differs between CPU and GPU (`topk(sorted=False)` in mmdet): match lines
  by position, not by index.
* All results are in-sample, on one GPU (RX 7900 XTX), ROCm 7.2 and WSL2. Native Linux (without the WSL
  `libhsa-runtime64.so` swap the install script makes) and NVIDIA GPUs are untested.
* The kept-queries mode drops instances whose class score is below its threshold; code that reads low-scoring
  instances needs `--all-queries` (or a lower `--kept-thr`).

## Input scale

`--input-size native` and `--tile` are experimental. In an in-sample test on dense chart grids, native-resolution
input made the model segment grid lines as data lines (precision 0.97 -> ~0.2). The model was trained at ~512 px
per chart; results are best near that scale, which the default (`config`, fit 512 x 512) keeps.

## Release check (v0.2.0)

Run on the release commit before tagging, all on the machine above:

1. `python -m pytest` in the installed venv, and the CPU-only set of `.github/workflows/tests.yml` in a separate
   venv (the model-stack tests skipped with their reason);
2. a fresh clone + `rocm/install_rocm.sh` into a new venv, the checks of `rocm/INSTALL.md`, the demo on CPU and GPU
   (3 lines of 621 points);
3. `lineformer batch` with the command-line defaults (device auto, kept 0.3, `--instances --masks`) on the 72
   images -> `tools/equivalence/to_harness.py` -> `compare.py` against the reference A (PASS 72/72 required) and
   against the earlier kept-mode run (masks bit-identical expected);
4. `lineformer serve`: health, one 10-image job, `lineformer_client` wait.
