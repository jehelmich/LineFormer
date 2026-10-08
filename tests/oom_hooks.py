# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Test-only GPU worker hooks (lineformer_engine.Engine(_test_hooks=...)): a fake model without a checkpoint, and
injected out-of-memory errors. Lives in its own module so that the engine's spawned worker processes can import it
(they get the test's sys.path). Never used outside the tests."""
import os

import numpy as np


class FakeGPU:
    """build_model / forward for lineformer_engine.gpu_worker.

    oom: {image id: rule}; rule 'always' (every attempt raises), 'once' (the first attempt in any worker raises; a
    marker file in marker_dir records it). The fake forward returns one horizontal line instance per image."""

    def __init__(self, oom=None, marker_dir=None):
        self.oom = dict(oom or {})
        self.marker_dir = marker_dir

    def build_model(self, mo):
        return None, {'param_devices': ['cpu'], 'msda_path': None, 'kept_thr': mo['kept_thr'],
                      'input_size': mo['input_size']}

    def _first_attempt(self, iid):
        try:
            fd = os.open(os.path.join(self.marker_dir, 'oom_%s' % iid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def forward(self, model, datas, wid, rec):
        import torch
        rule = self.oom.get(rec['id'])
        if rule == 'always' or (rule == 'once' and self._first_attempt(rec['id'])):
            raise torch.OutOfMemoryError('HIP out of memory (injected by the test: image %s, GPU worker %d)'
                                         % (rec['id'], wid))
        return [fake_result(d) for d in datas]


def fake_result(data):
    """One instance (score 0.9): a 3 px horizontal line through the middle of the original image."""
    h, w = data['img_metas'][0].data['ori_shape'][:2]
    mask = np.zeros((h, w), bool)
    y, x0, x1 = h // 2, w // 4, 3 * w // 4
    mask[y - 1:y + 2, x0:x1] = True
    box = np.array([[x0, y - 1, x1, y + 2, 0.9]], np.float32)
    return ([box], [[mask]])
