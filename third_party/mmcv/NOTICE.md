<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# mmcv 1.7.2, pure-Python subset (vendored by the LineFormer fork)

Source: [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv) tag `v1.7.2` (commit `4c01b02`), Copyright (c)
OpenMMLab, Apache License 2.0 (`LICENSE` in this directory). This directory holds the part of its `mmcv/` package
that LineFormer inference loads, so that LineFormer installs without compiling mmcv-full.

## How the subset was chosen

The modules in `sys.modules` after a LineFormer run on the original stack (mmcv-full 1.7.2 with compiled CPU ops):
`infer.load_model` + `infer.get_dataseries` with the kept-queries mode off and on, the job engine's
pre-processing / forward / post-processing functions, and `scale_compat`'s native-size pipeline. 148 of those
modules are pure Python outside `mmcv.ops`; they are copied unchanged. A profiler over the same run counted 0 calls
into the compiled extension `mmcv._ext`, and the only `mmcv.ops` code that ran is
`MultiScaleDeformableAttention` with `multi_scale_deformable_attn_pytorch`.

## Changes against mmcv 1.7.2

Unchanged copies: every `.py` file here except the ones in `mmcv/ops/` listed below.

Left out (no LineFormer code path imports them; an import raises `ImportError`):
- `mmcv/ops/` except the files below, and the compiled extension `mmcv/_ext*.so` with its C++/CUDA sources;
- `mmcv/onnx/`, `mmcv/tensorrt/`, `mmcv/engine/` (used by `runner/hooks/evaluation.py` during training only);
- `mmcv/model_zoo/*.json` (the `open-mmlab://`, `torchvision://` and `mmcls://` checkpoint schemes; local paths and
  http URLs work);
- `mmcv/device/ipu/` except `__init__.py` (imports its modules only on Graphcore IPU hardware);
- `mmcv/parallel/distributed_deprecated.py`.

Changed or added in `mmcv/ops/`:
- `point_sample.py`: unchanged (pure PyTorch; the vendored mmdetection imports it).
- `multi_scale_deform_attn.py`: the compiled kernel entry points (`ext_module`) are stand-ins that raise; the rest
  of the file, including `multi_scale_deformable_attn_pytorch` and the `IS_CUDA_AVAILABLE` branch that LineFormer's
  `msda_compat.py` switches, is unchanged.
- `info.py`: `get_compiler_version` and `get_compiling_cuda_version` raise; `get_onnxruntime_op_path` unchanged.
- `__init__.py`: exports only the names that LineFormer, the vendored mmdetection 2.28.2 and the rest of this subset
  import.
- `_unavailable.py` (new) and the stand-in modules `nms.py`, `roi_align.py`, `roi_pool.py`, `deform_conv.py`,
  `modulated_deform_conv.py`, `masked_conv.py`, `corner_pool.py`, `focal_loss.py`, `carafe.py`, `saconv.py`,
  `cc_attention.py`, `sync_bn.py`, `merge_cells.py`: the names exist so that imports succeed; calling a function or
  instantiating a class raises `OpUnavailableError` (a `NotImplementedError`) naming the op. The registry names of
  the full mmcv (`DCN`, `DCNv2`, `SAC`, `carafe`, `CrissCrossAttention`, `MMSyncBN`) are registered with stand-ins
  and fail the same way when a config builds them.

`pyproject.toml` (new) installs the subset as the distribution `mmcv` version `1.7.2+lineformer`; the import name
and `mmcv.__version__` stay `mmcv` / `1.7.2`, which the vendored mmdetection checks.
