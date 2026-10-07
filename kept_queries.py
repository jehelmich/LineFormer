# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
# Contains code adapted from OpenMMLab mmdetection (Apache-2.0, Copyright OpenMMLab).
"""Opt-in: post-process only the queries that can reach the score threshold (instance results only).

Mask2Former returns one mask per query (100). mmdet 2.28's test path upsamples all of them to the input and then to
the original image resolution, scores every one at that resolution (``MaskFormerFusionHead.instance_postprocess``),
also runs the panoptic post-processing, and copies every mask to the host one by one
(``MaskFormer.simple_test``; ``mask2bbox`` adds several small copies per mask). On 1.5-3.5k px images that is most
of the GPU time and memory of an image, while only a few queries end up above LineFormer's threshold of 0.3.

The final instance score is ``class score x mask score``, with mask score = mean sigmoid over the pixels of the
binarized mask, which is <= 1. So an instance whose class score is below ``t`` has a final score below ``t`` too.
With this mode on, the queries whose class score is below ``t`` are dropped right after the network, BEFORE the
upsampling; the rest runs on the device as before (same operations, in the same order, on the kept queries
only), and the kept masks reach the host in one copy.

What changes, and what does not:
  * returned: the instances whose CLASS score is >= t, in mmdet's order (topk, sorted=False). Their boxes, scores
    and masks are those of the unpatched path (bit-identical where the device's kernels do not depend on the
    number of queries; the mask score sums can differ in the last float bit on a GPU).
  * NOT returned any more: every instance whose class score is below t. Their final scores would be below t;
    code that reads instances below the threshold (e.g. a "score > 0.1" filter) gets fewer of them.
    N in the (N, 5) box array is the number of kept queries; it can still hold instances with a final score
    below t (class score >= t, mask score < 1), so the usual ``score > 0.3`` filter is still applied downstream.
  * the panoptic result is not computed: mmdet drops it anyway when there are no stuff classes (checked).
  * masks are views into one (N, H, W) bool array (one host copy instead of one per mask).

Activation: ``infer.load_model(..., kept_only=...)``: None reads the environment variable
LINEFORMER_KEPT_QUERIES ("" / "off" / "0" = off, "on" / "1" = 0.3, a number in (0, 1) = that threshold);
False = off; True = 0.3; a float = that threshold. Default: off, which leaves mmdet untouched.

The patch is per model instance (instance attributes for ``simple_test`` on the detector and on its fusion head),
not on the mmdet classes. It is refused (RuntimeError) unless: mmdet is 2.28.2, the source of every mmdet function
whose behaviour is reproduced here has the sha256 recorded below, the model is a MaskFormer/Mask2Former with
exactly one thing class and no stuff class, the test config asks for instance results and no semantic results.
"""
import hashlib
import inspect
import os

import torch
import torch.nn.functional as F

ENV_VAR = 'LINEFORMER_KEPT_QUERIES'
DEFAULT_THR = 0.3
MMDET_VERSION = '2.28.2'
# sha256 of inspect.getsource() of the mmdet code this module reproduces (vendored mmdetection 2.28.2).
SOURCE_SHA256 = {
    'MaskFormer.simple_test': 'f29b4188bb3219c6b44ba0b50326feb9af65574d8514108b677ff58793b0bd24',
    'MaskFormerHead.simple_test': '767f7911d3d43c5c8e54543ed24272a811715bad7f7c3b88c05e7079ed374752',
    'MaskFormerFusionHead.simple_test': '0067d71c63378901a1dbc89bb4d61e57ca066d82a8db5fdeb66e93ccb07d3670',
    'MaskFormerFusionHead.instance_postprocess':
        'af6cffdff90d891f8bc7ff2785b4bf0e09aac4cb280b0d2860c22ab340c0b8b7',
    'bbox2result': '11f806c68ea33c56b6d18f233b248354e7ddc17fd37b08a91bf00d81c2940be2',
    'mask2bbox': 'e1205c333219555ba470fe823771d52a26aacc76018114de11a720a5a012fb35',
}
_ATTR = '_lineformer_kept_thr'
_OFF = ('', 'off', '0', 'false', 'no', 'none')
_ON = ('on', '1', 'true', 'yes')


def resolve_threshold(kept_only=None):
    """kept_only (None = environment) -> threshold in (0, 1), or None for off. Raises on anything else."""
    src = 'kept_only=%r' % (kept_only,)
    if kept_only is None:
        raw = os.environ.get(ENV_VAR, '').strip().lower()
        src = '%s=%r' % (ENV_VAR, raw)
        if raw in _OFF:
            return None
        if raw in _ON:
            return DEFAULT_THR
        try:
            kept_only = float(raw)
        except ValueError:
            raise ValueError('%s: expected off|on or a threshold in (0, 1)' % src) from None
    elif isinstance(kept_only, bool):
        return DEFAULT_THR if kept_only else None
    thr = float(kept_only)
    if not 0.0 < thr < 1.0:
        raise ValueError('%s: the threshold must lie in (0, 1)' % src)
    return thr


def _mmdet_parts():
    import mmdet
    from mmdet.core import bbox2result
    from mmdet.core.mask import mask2bbox
    from mmdet.models.dense_heads.maskformer_head import MaskFormerHead
    from mmdet.models.detectors.maskformer import MaskFormer
    from mmdet.models.seg_heads.panoptic_fusion_heads.maskformer_fusion_head import MaskFormerFusionHead
    funcs = {
        'MaskFormer.simple_test': MaskFormer.simple_test,
        'MaskFormerHead.simple_test': MaskFormerHead.simple_test,
        'MaskFormerFusionHead.simple_test': MaskFormerFusionHead.simple_test,
        'MaskFormerFusionHead.instance_postprocess': MaskFormerFusionHead.instance_postprocess,
        'bbox2result': bbox2result,
        'mask2bbox': mask2bbox,
    }
    return mmdet.__version__, funcs, (MaskFormer, MaskFormerHead, MaskFormerFusionHead)


def check_supported(model):
    """Raise RuntimeError unless the mode reproduces this model's mmdet test path exactly."""
    version, funcs, (MaskFormer, MaskFormerHead, MaskFormerFusionHead) = _mmdet_parts()
    problems = []
    if version != MMDET_VERSION:
        problems.append('mmdet %s (written for %s)' % (version, MMDET_VERSION))
    for name, fn in funcs.items():
        h = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        if h != SOURCE_SHA256[name]:
            problems.append('source of mmdet %s differs from the one this module reproduces' % name)
    if not isinstance(model, MaskFormer):
        problems.append('model is %s, not a MaskFormer/Mask2Former' % type(model).__name__)
    else:
        head = getattr(model, 'panoptic_head', None)
        fusion = getattr(model, 'panoptic_fusion_head', None)
        if type(model).simple_test is not MaskFormer.simple_test:
            problems.append('%s overrides simple_test' % type(model).__name__)
        if head is None or type(head).simple_test is not MaskFormerHead.simple_test:
            problems.append('panoptic_head %s has its own simple_test' % type(head).__name__)
        if not isinstance(fusion, MaskFormerFusionHead) or \
                type(fusion).simple_test is not MaskFormerFusionHead.simple_test or \
                type(fusion).instance_postprocess is not MaskFormerFusionHead.instance_postprocess:
            problems.append('panoptic_fusion_head %s is not mmdet\'s MaskFormerFusionHead' % type(fusion).__name__)
        if (model.num_things_classes, model.num_stuff_classes) != (1, 0):
            problems.append('%d thing / %d stuff classes (supported: 1 / 0)' % (model.num_things_classes,
                                                                                 model.num_stuff_classes))
        cfg = getattr(fusion, 'test_cfg', None) or {}
        if not cfg.get('instance_on', False):
            problems.append('test_cfg.instance_on is not set: the model returns no instances')
        if cfg.get('semantic_on', False):
            problems.append('test_cfg.semantic_on is set')
        if fusion is not None and getattr(fusion, 'num_classes', None) != 1:
            problems.append('fusion head num_classes %r' % getattr(fusion, 'num_classes', None))
    if problems:
        raise RuntimeError('kept-queries mode cannot reproduce this model\'s test path: ' + '; '.join(problems))


def select_queries(mask_cls, num_classes, max_per_image, score_thr):
    """The first lines of MaskFormerFusionHead.instance_postprocess (verbatim), then the class-score filter.

    mask_cls: (num_queries, num_classes + 1) logits of one image.
    Returns (scores, labels, query_indices) of the entries with class score >= score_thr, in topk order.
    """
    num_queries = mask_cls.shape[0]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(num_classes, device=mask_cls.device).\
        unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)
    labels_per_image = labels[top_indices]
    query_indices = top_indices // num_classes
    keep = scores_per_image >= score_thr
    return scores_per_image[keep], labels_per_image[keep], query_indices[keep]


def mask2bbox_vectorized(masks):
    """mmdet.core.mask.mask2bbox without the per-mask loop (and its host round trips): same integer boxes
    [x0, y0, x1 + 1, y1 + 1] as float32, zeros for an empty mask."""
    n, h, w = masks.shape
    x_any = torch.any(masks, dim=1)  # (n, w)
    y_any = torch.any(masks, dim=2)  # (n, h)
    xs = torch.arange(w, device=masks.device)
    ys = torch.arange(h, device=masks.device)
    x0 = torch.where(x_any, xs, w).amin(1) if n else xs.new_zeros(0)
    x1 = torch.where(x_any, xs, -1).amax(1) if n else xs.new_zeros(0)
    y0 = torch.where(y_any, ys, h).amin(1) if n else ys.new_zeros(0)
    y1 = torch.where(y_any, ys, -1).amax(1) if n else ys.new_zeros(0)
    b = torch.stack([x0, y0, x1 + 1, y1 + 1], 1).to(torch.float32)
    valid = (x_any.any(1) & y_any.any(1))[:, None]
    return torch.where(valid, b, torch.zeros_like(b))


def _upsample(m, size):
    """F.interpolate (bilinear, align_corners=False) of (K, h, w) -> (K, *size); K may be 0."""
    if m.shape[0] == 0:
        return m.new_zeros((0,) + tuple(size))
    return F.interpolate(m[None], size=tuple(size), mode='bilinear', align_corners=False)[0]


def _kept_fusion_simple_test(self, mask_cls_results, mask_pred_results, img_metas, rescale=False, **kwargs):
    """MaskFormerHead.simple_test's upsample + MaskFormerFusionHead.simple_test / instance_postprocess, for the
    kept queries only. mask_pred_results are the head's LOW-RESOLUTION mask logits (B, Q, h, w)."""
    thr = getattr(self, _ATTR)
    max_per_image = self.test_cfg.get('max_per_image', 100)
    img_shape = img_metas[0]['batch_input_shape']  # MaskFormerHead.simple_test upsamples the batch to this
    results = []
    for mask_cls, mask_low, meta in zip(mask_cls_results, mask_pred_results, img_metas):
        scores_per_image, labels_per_image, query_indices = select_queries(
            mask_cls, self.num_classes, max_per_image, thr)
        # MaskFormerHead.simple_test: upsample to the batch input
        mask_pred = _upsample(mask_low[query_indices], (img_shape[0], img_shape[1]))
        # MaskFormerFusionHead.simple_test: remove padding, then back to the original resolution
        img_height, img_width = meta['img_shape'][:2]
        mask_pred = mask_pred[:, :img_height, :img_width]
        if rescale:
            ori_height, ori_width = meta['ori_shape'][:2]
            mask_pred = _upsample(mask_pred, (ori_height, ori_width))
        # instance_postprocess from "extract things" on (verbatim)
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        if bool((mask_scores_per_image > 1.0).any()):
            # the filter relies on final score <= class score; never seen, but then dropped queries could matter
            raise RuntimeError('kept-queries mode: a mask score > 1 (%s); the class-score prefilter is not safe'
                               % mask_scores_per_image.max().item())
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox_vectorized(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)
        results.append({'ins_results': (labels_per_image, bboxes, mask_pred_binary)})
    return results


def _kept_detector_simple_test(self, imgs, img_metas, **kwargs):
    """MaskFormer.simple_test for 1 thing class / 0 stuff classes, with one host copy per image for the masks."""
    from mmdet.core import bbox2result
    feats = self.extract_feat(imgs)
    # MaskFormerHead.simple_test without its upsample (moved after the query filter)
    all_cls_scores, all_mask_preds = self.panoptic_head(feats, img_metas)
    mask_cls_results = all_cls_scores[-1]
    mask_pred_results = all_mask_preds[-1]
    results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
    out = []
    for res in results:
        labels_per_image, bboxes, mask_pred_binary = res['ins_results']
        det = torch.cat([bboxes, labels_per_image[:, None].to(bboxes.dtype)], 1).detach().cpu().numpy()
        masks = mask_pred_binary.detach().cpu().numpy()  # one copy for all kept masks
        boxes, labels = det[:, :5], det[:, 5].astype('int64')
        bbox_results = bbox2result(boxes, labels, self.num_things_classes)
        mask_results = [[] for _ in range(self.num_things_classes)]
        for j, label in enumerate(labels):
            mask_results[label].append(masks[j])
        out.append((bbox_results, mask_results))
    return out


def enable(model, score_thr=DEFAULT_THR):
    """Patch this model instance; returns the threshold. Raises if the model / mmdet is not the supported one."""
    thr = resolve_threshold(score_thr)
    if thr is None:
        raise ValueError('enable() needs a threshold, got %r' % (score_thr,))
    check_supported(model)
    fusion = model.panoptic_fusion_head
    setattr(fusion, _ATTR, thr)
    setattr(model, _ATTR, thr)
    fusion.simple_test = _kept_fusion_simple_test.__get__(fusion)
    model.simple_test = _kept_detector_simple_test.__get__(model)
    return thr


def disable(model):
    """Undo enable() on this model instance (no-op if it was not enabled)."""
    for obj, names in ((model, ('simple_test', _ATTR)),
                       (getattr(model, 'panoptic_fusion_head', None), ('simple_test', _ATTR))):
        if obj is None:
            continue
        for n in names:
            if n in obj.__dict__:
                delattr(obj, n)


def get_threshold(model):
    """The kept-queries threshold of this model, or None if the mode is off."""
    return model.__dict__.get(_ATTR)


def configure(model, kept_only=None):
    """Resolve kept_only (None = environment), enable or disable on the model, print the state once if on."""
    thr = resolve_threshold(kept_only)
    disable(model)
    if thr is not None:
        enable(model, thr)
        print('LineFormer kept-queries mode: on, class score >= %g (instances below it are not returned)' % thr)
    return thr
