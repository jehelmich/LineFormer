<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Validation: same results, measured speed

The question: does the fork give the results of the original LineFormer code, on a new stack, on an AMD GPU, with
the kept-queries mode and through the job engine? And how fast is it?

## Environment

Current stack (`rocm/install_rocm.sh`, v0.3.0): Python 3.13.16, torch 2.14.1+rocm7.2, the pure-Python mmcv
1.7.2 subset in `third_party/mmcv` (no compiled ops), mmdet 2.28.2 (vendored), numpy 2.5.2, opencv-python
5.0.0.93, scipy 1.18.1, scikit-image 0.26.0, matplotlib 3.11.2; checked in
[Without compiled mmcv, current dependencies](#without-compiled-mmcv-current-dependencies-v030). The other
measurements on this page were made with the v0.2.0 stack:

AMD Radeon RX 7900 XTX (gfx1100, 24 GB), ROCm 7.2.0, WSL2 Ubuntu 24.04. Python 3.11.17, torch 2.14.1+rocm7.2,
mmcv-full 1.7.2 built with CPU ops only, mmdet 2.28.2 (vendored), numpy 1.23.5, opencv-python 4.11.0.86,
scipy 1.9.3, scikit-image 0.21.0 (`rocm/install_rocm.sh` of v0.2.0). On the GPU, MultiScaleDeformableAttention runs mmcv's
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

Demo image, expected output: 3 lines of 621 points each (`lineformer --out out/ demo/PMC5959982___3_HTML.jpg`,
checkpoint found by the lookup of `lineformer_cli.py`), on CPU and GPU.

### Without compiled mmcv (v0.3.0)

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

### Without compiled mmcv, current dependencies (v0.3.0)

The current stack (see [Environment](#environment): Python 3.13, numpy 2.5.2, OpenCV 5.0.0.93, scipy 1.18.1,
scikit-image 0.26.0, matplotlib 3.11.2, the same torch 2.14.1) with the mmcv subset. Against the reference A, with
the fixed criteria:

| candidate | PASS | min mask IoU | max \|dscore\| | lines identical | points within 1 px |
|---|---|---|---|---|---|
| GPU, kept 0.3 (`run.py --msda pytorch --kept-only 0.3`) | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |
| engine `lineformer batch`, defaults, 2 GPU workers | 72/72 | 0.99995 | 1.1e-5 | 44/72 images | all |
| CPU, all queries, 12 images (every 7th + demo) | 12/12 | 0.99995 | 1.1e-5 | 10/12 images | all |

Bit-identity, the same comparisons as in the table above:

| run | masks | boxes (coordinates) | scores | lines |
|---|---|---|---|---|
| CPU, all queries, 12 images, against mmcv-full + Python 3.11 / numpy 1.23.5 / OpenCV 4.11 on CPU | identical 12/12 | identical | identical | identical 12/12 |
| CPU, kept 0.3, the same 12 images, against the same | identical 12/12 | identical | identical | identical 12/12 |
| GPU, kept 0.3 (`run.py`), against the earlier kept GPU run | identical 72/72 | identical | identical on 55/72, max diff 2.0e-6 | identical 72/72 |
| engine, defaults, 2 GPU workers, against the same run | identical 72/72 | identical | identical on 53/72, max diff 1.3e-6 | identical 72/72 |

The dependency upgrade changed no output: on CPU every array and line is bit-identical to the v0.2.0 stack, on the
GPU only the scores move, within the GPU's run-to-run variation. Nothing had to be pinned back. Speed was not
re-measured: the machine was shared with other GPU and CPU work during these runs; interleaved runs of the two
stacks under the same load showed no difference beyond that noise.

## Threshold sensitivity (v0.3.0)

The acceptance compares instances with score >= 0.3, but no instance of the reference lies near 0.3 (lowest kept
0.3214, highest dropped 0.2288), so "the same instances" was never tested at its own threshold. The runs that
store all 100 queries per image (A; B = the v0.2.0 stack on CPU: Python 3.11, torch 2.14.1, mmcv-full with CPU
ops; C = GPU, all queries, and its repeat C2) were matched again at lower thresholds, without new inference: per
t in {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}, the instances with score >= t of each run, matched one-to-one by mask IoU as
`compare.py` does at 0.3. A flip is a pair (matched over all queries, IoU >= 0.5) whose scores lie on two sides of
t. The kept-queries runs cannot take part (they hold no instance below a class score of 0.3).

Instances with score >= t in A: 142 / 135 / 131 / 128 / 125 / 125. The real instances closest to each threshold:

| t | instances within +-0.02 | closest to t (margin) |
|---|---|---|
| 0.05 | 12 | 0.0486, 0.0514 (0.0014) |
| 0.1 | 7 | 0.0983 (0.0017) |
| 0.15 | 2 | 0.1578 (0.0078) |
| 0.2 | 4 | 0.1984 (0.0016) |
| 0.25 | 0 | 0.2288 (0.0212) |
| 0.3 | 0 | 0.3214 (0.0214) |

Result, every pair at every threshold: the same instance counts on every image, 0 unmatched, 0 flips, every matched
pair with IoU >= 0.98 and |dscore| <= 0.01.

| pair | min IoU | max \|dscore\| (instances >= t) |
|---|---|---|
| A-C (GPU), t = 0.05 ... 0.3 | 0.9999 | 1.1e-5 |
| C-C2 (GPU repeat) | 0.9999 (1.0 for t >= 0.1) | < 5e-7 |
| A-B (CPU, v0.2.0 stack), t = 0.05 ... 0.2 | 0.9944 ... 0.9970 | 0.0030 |
| A-B, t = 0.25 / 0.3 | 0.9978 | 7e-4 |

What this shows: on the GPU the "same instances" holds down to 0.05 with clearance: at 0.05, 0.1 and 0.2 real
instances lie 0.0014-0.0017 from t, and the largest A-C shift of any instance >= 0.05 is 1.1e-5, two orders of
magnitude less. For B there were no flips either, but without that clearance: its largest shift (0.003, on an
instance at 0.2288 in one image) exceeds the margins of the closest instances, so "no flip" for B at 0.05, 0.1 and
0.2 is observed, not guaranteed by margin. What is not shown: 0.25 and 0.3 have no instance within 0.02, so the
threshold 0.3 itself is still not stressed by a near-threshold instance; only 17 instances lie in [0.05, 0.3); the
lines were not recomputed at the lower thresholds; in-sample, the same 72 images.

## torch.compile (measured, not part of the fork)

`torch.compile` of the Swin-T backbone was built and measured on the branch `torch-compile` (TorchInductor, Triton
3.8 on ROCm; kept-queries mode on; 72 images with 23 distinct input shapes at the 512 fit). It is not merged:

* Equivalent: engine, kept 0.3, compiled backbone, PASS 72/72 against A (min IoU 0.99995, max |dscore| 1.1e-5,
  the figures of the uncompiled GPU runs); against the uncompiled kept GPU run masks bit-identical on 72/72 images,
  scores within 8.8e-6, lines identical on 72/72.
* No reliable speed gain: the backbone is ~13 ms of the ~50 ms GPU stage per image, so even a halved backbone gains
  at most ~13 % of the GPU stage (an idle-GPU check on 3 images, not paired, saw ~10 %: 37-39 vs 43 ms). Under the shared load of
  the measurement the paired comparisons overlapped (network per image, medians 568 vs 513 ms in one sitting, 269 vs
  277 ms in another; `tools/benchmark.py` 2.8-4.2 vs 0.8-3.3 images/s with one worker).
* Costs: 30-47 s of compilation per new input shape and per GPU worker with an empty cache (every new aspect ratio
  is a new shape at the 512 fit), ~120 MB of on-disk cache per shape (2.8 GB for the 23); dynamic shapes are not
  usable (932 s for the first shape, the second unfinished after 25 min); compiling more than the backbone costs
  minutes per shape with 25-29 graph breaks (`.item()` in every MSDA call).

Details in `git show torch-compile:docs/VALIDATION.md`.

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
  borderline cases at 0.3; at lower thresholds (0.05-0.2), where such instances exist, no instance flipped
  ([Threshold sensitivity](#threshold-sensitivity-v030)).
* The model's order of instances, and so of lines, differs between CPU and GPU (`topk(sorted=False)` in mmdet).
  The engine (`lineformer batch` / `serve`) sorts them geometrically (since v0.3.0, see
  [Line order](#line-order-v030)); `infer.get_dataseries` and the single-process `lineformer` do not: match
  their lines by position, not by index.
* All results are in-sample, on one GPU (RX 7900 XTX), ROCm 7.2 and WSL2. Native Linux (without the WSL
  `libhsa-runtime64.so` swap the install script makes) and NVIDIA GPUs are untested.
* The kept-queries mode drops instances whose class score is below its threshold; code that reads low-scoring
  instances needs `all_queries = true` in the settings file (or a lower `--threshold`).

## Line order (v0.3.0)

The engine sorts lines by (leftmost x, mean y, -score) and puts line *i* at instance *i* (`lineformer_jobs.py`,
"Order"). Measured on the lines of the 72 test images before the change, reference A (CPU) against the kept GPU
run: in the model's order the lines of 44 of 72 images come in the same order; sorted by the key, 72 of 72. No
key value moved between CPU and GPU on any line. The leftmost x ties exactly between two lines in 11 of the 34
images with several lines (lines starting at the axis); the mean y then decides, and its smallest gap between two
lines of an image is 3.2 px. A swap would need a line's leftmost x or mean y to move past another line's between
runs; this set has no such case, so the margin of the key is not stressed by it. The engine with the order, on CPU
and on the GPU: see [Release validation](#release-validation-v030), item 3.

## Input scale

`input_size = "native"` and `tile` (settings file; `--input-size`, `--tile` before v0.3.0) are experimental. In an in-sample test on dense chart grids, native-resolution
input made the model segment grid lines as data lines (precision 0.97 -> ~0.2). The model was trained at ~512 px
per chart; results are best near that scale, which the default (`config`, fit 512 x 512) keeps.

## Release validation (v0.3.0)

Two runs, both on the current stack of [Environment](#environment), on the machine above while it was shared:

* run R1 on commit `c1efc5e` (the engine before the small command line): items 3 and 4;
* run R2 on commit `6463bcb` (the small command line, automatic sizing and the out-of-memory back-off; the commits
  after it change documentation only): items 1, 2, 5, 6 and 7. The manifests say `git_dirty` true: the working
  tree held the uncommitted version and documentation edits of the release.

1. Tests (R2): `python -m pytest` 80 passed, 0 skipped; `python tests/run_all.py` 80 passed, 0 failed, 0 skipped;
   `ruff check` (ruff 0.16.10) clean. Among them the back-off by fault injection without a GPU
   (`tests/test_autosize_backoff.py`: one out of memory -> the image requeued, the job done with one worker fewer
   and the event in the manifest; the same image twice -> failed; the last worker -> failed; `serve
   --exit-on-failure` -> exit code 2) and the sizing rule with mocked free memory (2, 1 and 0 workers fit).
2. `lineformer batch --list <72 images> --out ... --masks` (R2): no other option, the checkpoint found through
   `LINEFORMER_CKPT`, device auto -> cuda:0, kept 0.3, GPU workers sized automatically (item 5: 1 worker) ->
   `to_harness.py` -> `compare.py`:

   | against | PASS | min mask IoU | max \|dscore\| | lines identical (in order) | points within 1 px |
   |---|---|---|---|---|---|
   | reference A (CPU, original stack) | 72/72 | 0.99995 | 1.1e-5 | 48/72 images | all |
   | the earlier kept GPU run | 72/72 | 1.0 | 7.5e-7 | 45/72 images | all |

   Against the engine run of R1 (2 GPU workers, `--gpu-mem-budget 3G --instances --masks`; the same code path apart
   from the command line), output file by output file: the `lines` of every `<id>.json` identical on 72/72 images
   (byte for byte as JSON), masks identical on 72/72, box coordinates and labels identical on 72/72, scores identical
   on 52/72 (max difference 1.4e-6, the GPU's run-to-run variation). R1 itself against the same references:
   72/72 against A (min IoU 0.99995, max |dscore| 1.1e-5, lines in order 48/72) and against the earlier kept GPU
   run (min IoU 1.0, max |dscore| 1.3e-6, 45/72; order-insensitive: masks and boxes identical on 72/72, scores on
   51/72, lines identical as sets on 72/72). `compare.py` counts identical lines in order; the references keep the
   model's order, so that count measures the order: 48/72 against A (44/72 before the geometric order). The
   manifest names the commit and holds an output sha256 for every image.
3. Line order, CPU against GPU (R1): the same engine on CPU (`--device cpu`, kept 0.3) on 8 of the images (the 7
   windows with 3-4 lines and the demo, 25 lines) against the GPU run of R1: lines identical and in the same
   order on 8/8 images, instances in the same order on 8/8 (masks identical position by position, scores within
   7.2e-6). In the model's order the lines of these 8 images came in different orders on CPU (A) and GPU (8/8).
4. Throughput (R1), `tools/benchmark.py` on the 72 images (kept 0.3, `--gpu-mem-budget 3G`, 8 pre-processing
   workers, 3 timed passes): 1 GPU worker 7.29 / 1.47 / 3.67 images/s, 2 GPU workers 6.05 / 6.16 / 5.57 images/s;
   peak allocated 479-480 MB, reserved 766-834 MB per worker. **Not a clean measurement**: no other process ran in
   WSL, but Windows-side GPU work kept the host CPU at 95-100 % and the GPU busy (engine utilisation summed over
   engines 48-307 %, 17.3-20.9 GB of 24 GB device memory in use by others); the pre-processing median was 0.6-1.2 s
   per image instead of ~0.2 s, so the rate measures the load, not the engine. The [Speed](#speed) table (idle
   machine, v0.2.0 stack) stays the reference; this version adds per image a sort of a few lines and a sha256 over
   the written files. R2 was not timed (item 2 ran at 2.7 images/s with one GPU worker under a load average of
   9-13).
5. Automatic sizing on the real GPU (R2, item 2): other processes held ~18.5 GB of the 24 GB (Windows counter
   before the run); the engine measured `device free 6049 MB of 24517 MB, headroom for other processes 2452 MB,
   usable 3597 MB, need per GPU worker 2358 MB (kept-queries mode) -> 1 GPU worker(s), memory budget 3597 MB`.
   The worker peaked at 479 MB allocated, 654 MB reserved over the 72 images. In a 6-image check before it, with
   7304 MB free, the same rule gave 2 workers (usable 4852 MB); the device's free memory then fell by ~1.6 GB with
   both workers loaded (~0.4 GB reserved each), within the 2 x 2358 MB the rule set aside.
6. Deprecated flags (R2): `lineformer batch --gpu-workers 2 --gpu-mem-budget 3G --kept-thr 0.3 --instances` on 8 of
   the images ran with 2 GPU workers and a 3G budget (no automatic sizing; the manifest names the flags as the
   sources), printed one deprecation line per flag, and gave the outputs of item 2 for these images: `lines`
   identical on 8/8, box coordinates and labels identical on 8/8, scores within 1.2e-7; `.instances.npz` only, no
   masks, as `--instances` wrote before.
7. CPU (R2): `lineformer batch --cpu --masks` on 3 of the images of item 3: lines, masks, boxes, labels and scores
   identical to the CPU run of R1 on 3/3 (bit for bit); the single process `lineformer --cpu` on the demo image:
   3 lines of 621 points.

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
