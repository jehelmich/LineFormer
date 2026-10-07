# ROCm setup notes

What `install_rocm.sh` does and why, for the tested setup: AMD Radeon RX 7900 XTX (gfx1100), WSL2 Ubuntu 24.04,
ROCm 7.2.0 in `/opt/rocm`, gcc 13.3.0. `$VENV` is the virtual environment (the script's default: `$HOME/lineformer`).

## Versions

- Python 3.11.17 (uv-managed: `uv python install 3.11`)
- torch 2.14.1+rocm7.2, torchvision 0.29.1+rocm7.2, triton-rocm 3.8.0 (index https://download.pytorch.org/whl/rocm7.2)
  - the newest ROCm build available; repo.radeon.com rocm-rel-7.2 only goes to torch 2.10.0.
  - GPU check: `torch.cuda.is_available()` True, device "AMD Radeon RX 7900 XTX gfx1100"; a 512x512 matmul differs
    from the CPU by at most 9.9e-05, `F.grid_sample` (bilinear, zeros, align_corners=False) by 0.0.
- mmcv-full 1.7.2 (source tag v1.7.2, CPU ops only, patched `setup.py`), mmdet 2.28.2 (editable, vendored)
- numpy 1.23.5, opencv-python 4.11.0.86, scipy 1.9.3, scikit-image 0.21.0, matplotlib 3.7.5, pillow 12.3.0,
  pycocotools 2.0.11, bresenham 0.2.1, addict 2.4.0, yapf 0.40.1, terminaltables 3.1.10, tqdm 4.70.1,
  chardet 7.6.0, pywavelets 1.8.0, tifffile 2026.3.3

## Workarounds

1. WSL only: the `libhsa-runtime64.so` bundled with the torch wheel does not see the GPU
   (`torch.cuda.is_available()` is False). It is replaced by the system one from `/opt/rocm/lib`; the original is
   kept in `$VENV/torch_lib_backup/`.
2. mmcv `setup.py`, patch `mmcv-1.7.2-cpu-ops.patch`:
   - `-std=c++20` for torch >= 2.10 (the torch 2.14 headers stop with "C++20 or later compatible compiler is
     required").
   - `MMCV_CPU_ONLY=1` skips the GPU branch. Without it `setup.py` builds the HIP extension even with
     `FORCE_CUDA=0`, because it detects ROCm on its own (`torch.version.hip`, `ROCM_HOME`) and
     `torch.cuda.is_available()` is True. The build log then says "Compiling mmcv._ext only with CPU", and
     `mmcv.ops.get_compiling_cuda_version()` returns "not available".
3. numpy pinned to 1.23.5 (mmdet 2.28 still uses `np.int` in training-only code). opencv-python 5.x needs
   numpy >= 2, so 4.11.0.86 is used.
4. yapf 0.40.1 (mmcv 1.x `Config.pretty_text` calls `FormatCode(verify=...)`, which later yapf versions removed).
5. No change in the vendored `mmdetection/` was needed.

## How the GPU path works

`msda_compat.py` chooses the MultiScaleDeformableAttention path (`auto` | `compiled` | `pytorch`). The `pytorch`
route sets `mmcv.ops.multi_scale_deform_attn.IS_CUDA_AVAILABLE = False` (read only by
`MultiScaleDeformableAttention.forward`), so GPU tensors take mmcv's own `multi_scale_deformable_attn_pytorch`
branch. `auto` makes a tiny call through the compiled function (after `get_compiling_cuda_version()` is not
"not available") and falls back to `pytorch`. On `cpu` the flag is left as is.

## Smoke results (demo image `demo/PMC5959982___3_HTML.jpg`)

- CPU before and after the change: bit-identical bboxes, masks and data series.
- GPU (`auto` resolving to `pytorch`, and explicit `pytorch`): 3 instances with score >= 0.3, 3 lines of 621
  points, same masks (IoU 1.0), same scores to 5 decimals, same points, but a different instance order
  (`topk(sorted=False)` in `MaskFormerFusionHead.instance_postprocess`), so the lines come back in a different
  order than on the CPU.
- Timings on a quiet machine, `get_dataseries` per call after one warm-up (includes CPU postprocessing):
  cpu 0.46-0.48 s, GPU (pytorch MSDA) 0.16 s; model load 2.7-3.1 s. Indicative only.
- `install_rocm.sh` was verified end to end into a throwaway venv: exit 0, mmcv CUDA version "not available",
  same GPU smoke counts.

`tools/equivalence/` compares a CPU run and a GPU run image by image (see the docstrings of `run.py` and
`compare.py`).

## Kept-queries mode (`kept_queries.py`, opt-in)

Where the GPU time went without it (pytorch MSDA, `profile_forward.py`, median of 10 images of 1.5-3.5k px): forward
0.247 s, of which the network ~0.05 s (pixel decoder 0.025 s), upsample + scoring of all 100 query masks + the unused
panoptic map ~0.09 s, device->host copies 0.10 s (one per mask, plus several per mask in `mask2bbox`). Peak 12 GB
per process.

The mode patches `simple_test` of the detector and of its fusion head on the model instance (not on the classes):
class scores and `topk` exactly as mmdet computes them, keep class score >= threshold (default 0.3; final score =
class score x mask score <= class score, checked at run time: a mask score > 1 raises), then upsample, crop, rescale
and mask scores for the kept queries only, a loop-free `mask2bbox` (same integer boxes, unit-tested against mmdet's),
one host copy of the kept masks. The panoptic map is skipped (mmdet discards it when there are no stuff classes).
Instances below the threshold are not returned. Enabling raises unless mmdet is 2.28.2, the sha256 of the source of
every reproduced mmdet function matches the recorded one, the model has 1 thing class and 0 stuff classes, and
`instance_on` is set. No change in the vendored `mmdetection/`.

With the mode on, per image (same profile): forward 0.052 s, host copies 0.001 s, `get_dataseries` 0.069 s
(was 0.266 s).

Verification (72 images, RX 7900 XTX, pytorch MSDA):
- CPU, mode off / on / off in one process (`tools/equivalence/kept_check.py`, 3 images): kept masks and boxes
  bit-identical, scores within 6e-8 (sums over fewer rows), dataseries identical.
- GPU, the same check on 72 images: 126 kept instances, all with bit-identical masks and boxes, scores bit-identical on
  94, max difference 2.3e-6; for comparison the unpatched path repeated in the same process was bit-identical on only
  34 of 72 images (its scores vary run to run). Every instance with final score >= 0.3 kept (125; one more kept
  query has class score >= 0.3 and final score < 0.3); highest final score among the dropped: 0.215. Dataseries
  identical on 72 of 72.
- `compare.py` of a `run.py --kept-only 0.3` GPU run against the earlier unpatched GPU runs and against the original
  CPU stack: PASS 72/72 each. Against the unpatched GPU run: all 126 instances bit-identical masks, max score
  difference 1.7e-6 (two unpatched GPU runs differ from each other by up to 1.4e-6), dataseries identical on 72.

Speed (`tools/throughput/batch_infer.py`, 72 images, `--repeat 2`; "off" from the earlier E1/E2 runs with
`--repeat 5`). All "on" runs had the card to themselves except the end of the last one, where another job started:

| config | images/s off | images/s on | GPU stage s/image on | peak allocated / reserved per process on |
|---|---|---|---|---|
| serial (one process) | 2.48 | 5.26 | 0.053 | 0.48 / 0.76 GB |
| E1 pipeline, 1 GPU worker | 3.35 | 18.7 | 0.049 | 0.48 / 0.71 GB |
| E2, 2 GPU workers | - | 20.6 | 0.061 | 0.48 / 0.76 GB |
| E2, 3 GPU workers | - | 20.2 | 0.069 | 0.48 / 0.70 GB |
| E2, 4 GPU workers | - | 15.1 | 0.090 | 0.48 / 0.66 GB |

With the mode on, one pipelined GPU worker keeps the GPU stage busy 90 % of the time; a second worker adds ~10 %;
three or four lose to CPU contention (every ROCm process under WSL keeps ~2 cores busy, plus pre/post workers).
Every timed pass gave the same dataseries as pass 0, and each run passed the acceptance against both references.
