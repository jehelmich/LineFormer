# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
# Environment for inference on AMD GPUs (ROCm), tested on ROCm 7.2.0 / RX 7900 XTX (gfx1100) under WSL2 Ubuntu 24.04.
# mmcv-full is built with CPU ops only; on the GPU, MultiScaleDeformableAttention runs mmcv's pure-PyTorch
# implementation (see msda_compat.py). Run from the repository root. Needs uv, git, gcc (C++20) and ROCm in /opt/rocm.
set -e
VENV=${VENV:-$HOME/lineformer}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}
REPO=$(pwd)

uv venv --python 3.11 "$VENV"
PY="--python $VENV/bin/python"
uv pip install $PY --index-url "$TORCH_INDEX" torch==2.14.1+rocm7.2 torchvision==0.29.1+rocm7.2

# WSL only: the HSA runtime bundled with the wheel cannot see the GPU through /dev/dxg; use the system one
# (AMD's WSL instructions). Skip this on native Linux.
if grep -qi microsoft /proc/version; then
    TL=$("$VENV/bin/python" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
    mkdir -p "$VENV/torch_lib_backup"
    mv "$TL"/libhsa-runtime64.so* "$VENV/torch_lib_backup/"
    cp "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so)" "$TL/libhsa-runtime64.so"
fi

uv pip install $PY numpy==1.23.5 ninja wheel packaging addict yapf==0.40.1 pyyaml "setuptools<80"

# mmcv-full 1.7.2 with CPU ops only (patch: C++20 for torch >= 2.10, MMCV_CPU_ONLY skips the CUDA/HIP detection)
mkdir -p "$VENV/src"
[ -d "$VENV/src/mmcv" ] || git clone --depth 1 --branch v1.7.2 https://github.com/open-mmlab/mmcv.git "$VENV/src/mmcv"
(cd "$VENV/src/mmcv" && git apply "$REPO/rocm/mmcv-1.7.2-cpu-ops.patch" \
    && MMCV_WITH_OPS=1 FORCE_CUDA=0 MMCV_CPU_ONLY=1 MAX_JOBS=$(nproc) \
       uv pip install $PY --no-build-isolation .)

uv pip install $PY numpy==1.23.5 opencv-python scipy==1.9.3 scikit-image==0.21.0 matplotlib==3.7.5 pillow \
    bresenham==0.2.1 tqdm chardet pycocotools terminaltables six pytest
uv pip install $PY --no-build-isolation -e mmdetection
uv pip install $PY --no-deps --no-build-isolation -e "$REPO"   # infer, msda_compat, the lineformer command

"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
