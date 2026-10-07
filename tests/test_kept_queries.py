"""Unit tests for kept_queries.py on synthetic tensors (no checkpoint needed; CPU).

pytest tests -q      or, without pytest:  python tests/test_kept_queries.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.is_available = lambda: False  # CPU only; under WSL+ROCm the probe would start the HSA runtime

import kept_queries as kq  # noqa: E402
from mmdet.core.mask import mask2bbox  # noqa: E402
from mmdet.models.seg_heads.panoptic_fusion_heads.maskformer_fusion_head import MaskFormerFusionHead  # noqa: E402


def _head():
    return MaskFormerFusionHead(num_things_classes=1, num_stuff_classes=0,
                                test_cfg=dict(panoptic_on=False, semantic_on=False, instance_on=True,
                                              max_per_image=100))


def _inputs(seed, q=100, low=(16, 20), batch=(64, 80), img=(60, 77), ori=(121, 150)):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(q, generator=g) * 2.0 - 2.5  # class logit vs background 0: class scores sigmoid(a)
    mask_cls = torch.stack([a, torch.zeros(q)], 1)
    mask_low = torch.randn(q, *low, generator=g) * 3.0
    meta = {'batch_input_shape': batch, 'img_shape': img + (3,), 'ori_shape': ori + (3,)}
    return mask_cls, mask_low, meta


def _reference(head, mask_cls, mask_low, meta):
    """The unpatched path: MaskFormerHead.simple_test's upsample, then mmdet's fusion head."""
    up = F.interpolate(mask_low[None], size=meta['batch_input_shape'], mode='bilinear', align_corners=False)
    return head.simple_test(mask_cls[None], up, [meta], rescale=True)[0]['ins_results']


def _kept(head, mask_cls, mask_low, meta, thr):
    setattr(head, kq._ATTR, thr)
    try:
        return kq._kept_fusion_simple_test(head, mask_cls[None], mask_low[None], [meta], rescale=True)[0][
            'ins_results']
    finally:
        delattr(head, kq._ATTR)


def test_prefilter_keeps_exactly_the_queries_that_can_reach_the_threshold():
    head = _head()
    for seed in range(5):
        for thr in (0.3, 0.1, 0.5):
            mask_cls, mask_low, meta = _inputs(seed)
            rl, rb, rm = _reference(head, mask_cls, mask_low, meta)
            kl, kb, km = _kept(head, mask_cls, mask_low, meta, thr)
            cls_scores = F.softmax(mask_cls, -1)[:, 0]
            # the reference order is topk(sorted=False); class score of each reference instance = its topk value
            ref_cls = F.softmax(mask_cls, -1)[:, :-1].flatten().topk(100, sorted=False)[0]
            sel = ref_cls >= thr
            assert int(sel.sum()) == int((cls_scores >= thr).sum()) == len(kb)
            assert torch.equal(km, rm[sel]) and torch.equal(kb, rb[sel]) and torch.equal(kl, rl[sel])
            # nothing that was dropped could have reached the threshold, and every final score <= class score
            assert bool((rb[~sel, 4] < thr).all())
            assert bool((rb[:, 4] <= ref_cls).all())
            assert km.shape[1:] == meta['ori_shape'][:2] and km.dtype == torch.bool


def test_no_query_kept():
    head = _head()
    mask_cls, mask_low, meta = _inputs(0)
    mask_cls[:, 0] = -20.0
    kl, kb, km = _kept(head, mask_cls, mask_low, meta, 0.3)
    assert kb.shape == (0, 5) and km.shape == (0,) + meta['ori_shape'][:2] and kl.shape == (0,)


def test_mask2bbox_vectorized_equals_mmdet():
    g = torch.Generator().manual_seed(1)
    m = torch.rand(12, 37, 53, generator=g) > 0.97
    m[3] = False  # empty
    m[4] = False
    m[4, 5, 7] = True  # one pixel
    m[5] = True  # full
    m[6] = False
    m[6, :, 52] = True  # last column
    assert torch.equal(kq.mask2bbox_vectorized(m), mask2bbox(m))
    assert torch.equal(kq.mask2bbox_vectorized(m[:0]), mask2bbox(m[:0]))


def test_resolve_threshold():
    old = os.environ.pop(kq.ENV_VAR, None)
    try:
        assert kq.resolve_threshold(None) is None
        assert kq.resolve_threshold(False) is None
        assert kq.resolve_threshold(True) == 0.3
        assert kq.resolve_threshold(0.25) == 0.25
        for env, want in (('', None), ('off', None), ('0', None), ('on', 0.3), ('1', 0.3), ('0.4', 0.4)):
            os.environ[kq.ENV_VAR] = env
            assert kq.resolve_threshold(None) == want, env
        for bad in (0.0, 1.0, -0.1, 2):
            try:
                kq.resolve_threshold(bad)
            except ValueError:
                continue
            raise AssertionError('threshold %r must be refused' % bad)
        os.environ[kq.ENV_VAR] = 'maybe'
        try:
            kq.resolve_threshold(None)
        except ValueError:
            pass
        else:
            raise AssertionError('a bad env value must raise')
    finally:
        os.environ.pop(kq.ENV_VAR, None)
        if old is not None:
            os.environ[kq.ENV_VAR] = old


def test_unsupported_model_and_changed_mmdet_source_raise():
    try:
        kq.check_supported(torch.nn.Linear(1, 1))
    except RuntimeError as e:
        assert 'not a MaskFormer' in str(e)
    else:
        raise AssertionError('a non-MaskFormer model must be refused')
    saved = kq.SOURCE_SHA256['mask2bbox']
    kq.SOURCE_SHA256['mask2bbox'] = '0' * 64
    try:
        kq.check_supported(torch.nn.Linear(1, 1))
    except RuntimeError as e:
        assert 'mask2bbox differs' in str(e)
    else:
        raise AssertionError('a changed mmdet source must be refused')
    finally:
        kq.SOURCE_SHA256['mask2bbox'] = saved


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print('%d tests passed' % n)
