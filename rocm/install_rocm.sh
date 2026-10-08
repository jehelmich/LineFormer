# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
# Environment for LineFormer inference. Default: AMD GPUs (ROCm), tested on ROCm 7.2.0 / RX 7900 XTX (gfx1100)
# under WSL2 Ubuntu 24.04. Nothing is compiled: mmcv is the pure-Python subset vendored in third_party/mmcv
# (no compiled ops; MultiScaleDeformableAttention runs mmcv's pure-PyTorch implementation, see msda_compat.py).
# Run from the repository root. Needs uv and, for ROCm, ROCm in /opt/rocm.
# Other torch builds: set TORCH_INDEX and TORCH_PKGS, e.g. CPU only:
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu TORCH_PKGS="torch==2.14.1 torchvision==0.29.1"
# The pins below are the versions the equivalence in docs/VALIDATION.md was measured with.
set -e
VENV=${VENV:-$HOME/lineformer}
PYTHON=${PYTHON:-3.13}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}
TORCH_PKGS=${TORCH_PKGS:-"torch==2.14.1+rocm7.2 torchvision==0.29.1+rocm7.2"}
REPO=$(pwd)

uv venv --python "$PYTHON" "$VENV"
PY="--python $VENV/bin/python"
uv pip install $PY --index-url "$TORCH_INDEX" $TORCH_PKGS

# WSL with a ROCm torch only: the HSA runtime bundled with the wheel cannot see the GPU through /dev/dxg; use the
# system one (AMD's WSL instructions). Skipped on native Linux and for non-ROCm torch builds.
if grep -qi microsoft /proc/version && "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.version.hip else 1)" 2>/dev/null; then
    TL=$("$VENV/bin/python" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
    mkdir -p "$VENV/torch_lib_backup"
    mv "$TL"/libhsa-runtime64.so* "$VENV/torch_lib_backup/"
    cp "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so)" "$TL/libhsa-runtime64.so"
fi

uv pip install $PY numpy==2.5.2 opencv-python==5.0.0.93 scipy==1.18.1 scikit-image==0.26.0 matplotlib==3.11.2 \
    pillow addict yapf pyyaml packaging setuptools \
    bresenham==0.2.1 tqdm chardet pycocotools terminaltables six pytest
uv pip install $PY --no-deps -e third_party/mmcv            # pure-Python mmcv 1.7.2 subset, nothing to build
uv pip install $PY --no-deps --no-build-isolation -e mmdetection
uv pip install $PY --no-deps --no-build-isolation -e "$REPO"   # infer, msda_compat, the lineformer command

"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
