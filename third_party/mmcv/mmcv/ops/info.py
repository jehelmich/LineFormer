# Copyright (c) OpenMMLab. All rights reserved.
# Modified by the LineFormer fork (2026): no compiled extension; the two version queries are stand-ins that raise
# (see _unavailable.py). get_onnxruntime_op_path is unchanged.
import glob
import os

from ._unavailable import (get_compiler_version,  # noqa: F401
                           get_compiling_cuda_version)


def get_onnxruntime_op_path():
    wildcard = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '_ext_ort.*.so')

    paths = glob.glob(wildcard)
    if len(paths) > 0:
        return paths[0]
    else:
        return ''
