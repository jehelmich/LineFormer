<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer) -->
# Changelog

## v0.3.0 (2026-10-08)

### Command line
- A small interface: `lineformer IMAGE... | --list FILE --out DIR [--threshold T] [--masks] [--force] [--cpu]
  [--settings FILE.toml]`, the same for `lineformer batch`, and `lineformer serve [--port] [--exit-on-failure]
  [--threshold] [--masks] [--cpu] [--settings]`. `--help` shows only these.
- `--threshold` is the former `--kept-thr` (default 0.3, same semantics: the class-score threshold of the
  kept-queries mode; a line still needs a final score > 0.3). `--masks` of batch / serve now also writes
  `<id>.instances.npz` (what `--instances` wrote). `--cpu` replaces `--device cpu`; the default stays automatic.
  `serve --masks` makes every job write the masks and instances.
- Settings file (`--settings FILE.toml`, read with `tomllib`; `lineformer.example.toml` lists every key with its
  default): `config`, `input_size`, `tile`, `tile_overlap` (both EXPERIMENTAL), `all_queries` (validation only),
  `msda` (debugging), `ids`, `gpu_workers`, `gpu_mem_budget`, `pre_workers`, `post_workers`, `pre_threads`,
  `post_threads`, `gpu_threads`, `max_inflight`, `progress_s`. Unknown keys and bad values are errors. The
  effective settings (sources included, after the automatic sizing) are in each job manifest (`engine.settings`).
- Deprecated, still accepted in this version with their values applied, a one-line warning naming the settings key
  and hidden from `--help`: `--config`, `--device`, `--msda`, `--all-queries`, `--kept-only` (no effect),
  `--kept-thr`, `--input-size`, `--tile`, `--tile-overlap`, `--ids`, `--instances`, `--gpu-workers`,
  `--gpu-mem-budget`, `--pre-workers`, `--post-workers`, `--pre-threads`, `--post-threads`, `--gpu-threads`,
  `--max-inflight`, `--progress-s`. A flag and the settings file that disagree are an error. `serve`'s `--host`,
  `--drain-timeout`, `--ready-file` and `--verbose` are kept (not shown in `--help`, not deprecated).
- Checkpoint lookup: `--ckpt` is optional (kept, not shown in `--help`); else `$LINEFORMER_CKPT`, else
  `<fork root>/iter_3000.pth`, else `~/.cache/lineformer/iter_3000.pth`; none found is an error that names the
  paths searched and the download link. A path given by the flag or the variable must exist. The path, its source
  and (as before) its sha256 are in the manifest.
- Two images that map to one id stop `lineformer` / `lineformer batch` before anything runs (exit code 2), naming
  the clash and the `ids = "parent_stem"` setting. Before, batch failed the second image and went on. The single
  process reads `<id><TAB><path>` list lines and writes `<id>.json` like batch.

### Engine
- Automatic sizing of the GPU workers (`Engine(gpu_workers=None, gpu_mem_budget=None)`, the command line's
  default; the Python defaults stay 1 worker, 0.85): at start the free device memory is measured
  (`torch.cuda.mem_get_info`, in a subprocess), max(2 GiB, 10 %) is left to other processes, and
  min(2, usable / need) workers start with the usable memory as their budget; need per worker = 1.5 x (measured
  peak + runtime context) = 2358 MB in the kept mode, 19500 MB with all queries (constants in
  `lineformer_engine.py` with their sources). No worker fits: the engine fails with the numbers. Only input size
  `config` without tiling is sized automatically. The decision is logged and in the manifest (`auto_sizing`).
- Bounded back-off on out of device memory (replaces "out of memory ends the engine"): the GPU worker that hit it
  stops (not restarted; the worker count only goes down) and the image is requeued once to a remaining worker. A
  second out of memory of the same image, or one on the last GPU worker, fails the engine as before (message with
  the memory numbers, jobs "failed", batch exit code 2, `serve --exit-on-failure` exits 2). Every event (time,
  image, worker, memory) is logged and in the manifest (`engine.backoff`); `gpu_workers_active` counts the workers
  left. Tested by fault injection without a GPU (`tests/test_autosize_backoff.py`).
- Deterministic output order: `lineformer batch` / `serve` sort the lines and instances of each image by a
  geometric key (leftmost x, mean y, then score; line *i* = instance *i*, the instances without a line after them)
  instead of the model's order, which differs between CPU and GPU. Each `<id>.json` says `"order": "geometric"`.
  The order is not part of the skip-if-done fingerprint: outputs of earlier engines (model order, no `"order"`)
  are still skipped as done. `infer.get_dataseries` is unchanged.
- Job manifest (`<out>/job.json`, `"manifest_version": 2`, written by batch and serve jobs alike): adds `fork`
  (version, git commit, `git_dirty`), `times_utc` (ISO 8601 created / started / finished / written), `failures`
  (every failed image with its full error), per image `outputs_sha256` (the `<id>.json` and each `.npz` the job
  wrote; for skipped images the existing `<id>.json`), the MSDA mode asked for (`msda`, `msda_env`) beside the
  path taken, and the versions of torchvision, scipy, scikit-image, matplotlib and the OpenCV wheel. A failed
  manifest write leaves the previous manifest and no temp file.
- `lineformer serve --exit-on-failure`: when the engine fails (models do not load, a GPU worker dies or raises,
  out of memory) the server marks its running jobs failed, writes their manifests and exits with code 2 instead of
  staying up and answering 503. Default unchanged. `serve()` restores the previous signal handlers on return, and
  `--ready-file` gives the bound port when `--port 0` picks one.

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
- `tests/test_cli.py` (help, settings file, deprecated flags, threshold, checkpoint lookup, id clash) and
  `tests/test_autosize_backoff.py` (sizing arithmetic with mocked free memory; back-off by fault injection in real
  worker processes on the CPU: one out of memory -> requeued and done with one worker fewer, the same image twice ->
  failed, the last worker -> failed, `serve --exit-on-failure` -> 2).
- ruff (`pyproject.toml` `[tool.ruff]`, rules E, F, W, line length 120) on the fork's own files only (those with
  the SPDX header; upstream files and the vendored `mmdetection/`, `third_party/` are not linted); a `lint` job in
  CI. Findings fixed without behaviour change (an unused variable in `scale_compat.py`, long lines, an ambiguous
  name, semicolons in a test).
- `tests/test_mmcv_subset.py`: the import closure (infer, engine, mmdet, the model built from the config) loads no
  compiled mmcv code; every stand-in raises when used; msda_compat without a kernel.
- CI installs the mmcv subset and the vendored mmdet (nothing to compile), so the model-stack tests run there
  instead of being skipped.
- `tests/test_model_cpu.py`: the model built from the config with random weights (no checkpoint), one forward and
  `get_dataseries` on the demo image on CPU, kept-queries mode off and on.

### Documentation
- `docs/VALIDATION.md`: threshold sensitivity (the "same instances" re-matched at 0.05-0.3 on the stored runs: 0
  flips; on the GPU with clearance, on the v0.2.0 CPU stack observed only; 0.3 itself still not stressed), the
  line order, torch.compile (measured on the unmerged branch `torch-compile`: equivalent, no reliable gain, 30-47 s
  of compilation per input shape and worker; not part of the fork) and the release validation of this version.

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
