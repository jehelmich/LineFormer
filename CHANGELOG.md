<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Changelog

## Unreleased

### Compatibility
- No compiler needed: `third_party/mmcv` vendors the pure-Python part of mmcv 1.7.2 that inference loads (148
  unchanged modules, ~0.9 MB of source; `NOTICE.md` lists every change). `mmcv.ops` keeps the real
  MultiScaleDeformableAttention (pure-PyTorch path) and `point_sample`; the compiled ops that the vendored mmdet
  imports are stand-ins that raise `OpUnavailableError` when called or instantiated. mmcv-full and
  `rocm/mmcv-1.7.2-cpu-ops.patch` are gone; `rocm/install_rocm.sh` takes `TORCH_INDEX` / `TORCH_PKGS` for CPU or
  other torch builds. Outputs are bit-identical to the mmcv-full stack on CPU and on the GPU up to the GPU's
  run-to-run score variation (`docs/VALIDATION.md`).
- `msda_compat`: `auto` resolves to `pytorch` without a compiled kernel, `compiled` raises (unchanged behaviour,
  now also with no mmcv extension at all).
- The engine's manifest reports the `mmcv` distribution version (was `mmcv-full`).

### Dependencies
- `rocm/install_rocm.sh` (and CI): Python 3.13 (was 3.11; `PYTHON=` to change), numpy 2.5.2 (1.23.5),
  opencv-python 5.0.0.93 (4.11.0.86), scipy 1.18.1 (1.9.3), scikit-image 0.26.0 (0.21.0), matplotlib 3.11.2
  (3.7.5), yapf and setuptools unpinned; torch 2.14.1 / torchvision 0.29.1 unchanged (the newest ROCm 7.2 build).
- Patches for them: mmcv `utils/config.py` (yapf >= 0.40.2: no `FormatCode(verify=)`), mmcv `utils/ext_loader.py`
  (`importlib.util.find_spec` for `pkgutil.find_loader`), mmdet `setup.py` (PEP 667: version read into an explicit
  namespace), `np.int` -> `int` in five lines of mmdet dataset/sampler code.
- Results unchanged: CPU outputs bit-identical to the v0.2.0 stack; on the GPU masks, boxes and lines identical, scores
  within the GPU's run-to-run variation; PASS 72/72 against the original stack (`docs/VALIDATION.md`).

### Tests
- `tests/test_mmcv_subset.py`: the import closure (infer, engine, mmdet, the model built from the config) loads no
  compiled mmcv code; every stand-in raises when used; msda_compat without a kernel.
- CI installs the mmcv subset and the vendored mmdet (nothing to compile), so the model-stack tests run there
  instead of being skipped.
- `tests/test_model_cpu.py`: the model built from the config with random weights (no checkpoint), one forward and
  `get_dataseries` on the demo image on CPU, kept-queries mode off and on.

## v0.2.0 (2026-10-07)

First release of the fork, against upstream [TheJaeLal/LineFormer](https://github.com/TheJaeLal/LineFormer) commit
`7952e27`. The model, checkpoint, thresholds and the per-image maths of `infer.get_dataseries` are unchanged; the
results match the original stack within the acceptance in `docs/VALIDATION.md`.

### Compatibility
- `msda_compat.py`, `infer.load_model(..., msda=)`: MultiScaleDeformableAttention on a GPU through mmcv's compiled
  kernel or its pure-PyTorch implementation (`auto` | `compiled` | `pytorch`), so inference runs on GPUs without
  mmcv's compiled ops, e.g. AMD GPUs with ROCm. CPU inference is unchanged.
- `rocm/install_rocm.sh` and `rocm/mmcv-1.7.2-cpu-ops.patch`: Python 3.11, torch 2.14.1+rocm7.2, mmcv-full 1.7.2
  built with CPU ops (C++20 for torch >= 2.10), the vendored mmdet, the WSL HSA runtime swap.

### Performance
- `kept_queries.py`, `infer.load_model(..., kept_only=)`: post-process only the queries whose class score can reach
  the threshold; same lines, detector time per image 0.27 s -> 0.056 s, peak device memory ~12 GB -> 0.5 GB.
- Job engine (`lineformer_engine.py`, `lineformer_jobs.py`): pre-processing workers -> N GPU workers ->
  post-processing workers, atomic outputs, skip-if-done, manifests, loud failure on out of memory; ~20 images/s
  with two GPU workers on one RX 7900 XTX.

### Commands and API
- `lineformer` (single process), `lineformer batch` (one job on the engine), `lineformer serve` (HTTP JSON job
  server on 127.0.0.1), `lineformer-client` / `lineformer_client.py` (stdlib-only client).
- Command-line defaults: `--device auto`; kept-queries mode on at 0.3 (`--all-queries` switches it off;
  `--kept-only` is still accepted; `--instances` / `--masks` then hold only the kept queries); pre-processing
  workers `min(8, max(2, CPUs // 3))`; one GPU worker. The command line does not read `LINEFORMER_KEPT_QUERIES`.
  The library default (`infer.load_model`, `Engine`) stays off.
- Experimental: `scale_compat.py` (`--input-size N|native`) and `tiling.py` (`--tile`): native-resolution input
  segmented grid lines as data lines on dense chart grids (in-sample).

### Tests and tools
- `python -m pytest` or `python tests/run_all.py`: unit tests without GPU or checkpoint; tests that need torch,
  mmcv-full and mmdet are skipped with the reason when those are missing. CI workflow (CPU only) in
  `.github/workflows/tests.yml`.
- `tools/equivalence/`: run, compare (fixed acceptance), kept-mode check, profiler, HTML report, engine output
  converter (`to_harness.py`). `tests/integration/serve_check.py`: load check of a running server.
  `tools/benchmark.py`: engine throughput on your own images.

### Documentation and licensing
- README: fork header, quick start, licensing; the authors' README unchanged below it. `rocm/INSTALL.md`,
  `docs/VALIDATION.md`, `CITATION.cff` (the ICDAR 2023 paper).
- Files added by the fork are Apache-2.0 (`LICENSES/Apache-2.0.txt`, SPDX headers). The upstream code and
  checkpoint have no licence from their authors; see the README.
