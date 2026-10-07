"""Choose the test-time input size of a LineFormer model.

The released config resizes every image to fit ``image_size = (512, 512)`` (``Resize`` with ``keep_ratio=True``
inside ``MultiScaleFlipAug``); the test pipeline has no ``Pad``, so the network input is the resized image itself
(e.g. 512 x 256 for a 2:1 image). The mask head predicts at 1/4 of that input and mmdet upsamples the masks to the
original image size. For large, thin-lined images (scans at several hundred dpi) this throws away most of the
resolution before the network sees it.

``build_model(config, ckpt, device, size)`` builds the detector with an overridden test pipeline (an
``mmcv.Config`` change passed to ``init_detector``; the config file is not touched):

  ``'config'``     the pipeline exactly as in the config file.
  ``N`` (int)      fit N x N, keep the aspect ratio (same transform as the config, other scale); no pad.
  ``(W, H)``       fit W x H, keep the aspect ratio; no pad.
  ``'native'``     no resize (``MultiScaleFlipAug(scale_factor=1.0)``, an identity ``Resize``), then ``Pad`` to a
                   multiple of ``size_divisor`` (default 32) with white (255, applied before ``Normalize``). The
                   masks are cropped back to the unpadded image by mmdet (``img_shape``), so they come out at the
                   original size.

``pad_divisor`` adds the same white pad to a sized pipeline; ``None`` (default for sized) keeps the config's no-pad.

``input_meta(model, img)`` runs only the test pipeline and returns what the network will see (``img_shape``,
``pad_shape``, ``scale_factor``, ``ori_shape`` and the input tensor shape); ``record_inputs(model)`` wraps the
model's ``simple_test`` to record the shapes of real forward passes, to check a run after the fact.

Rules: unknown sizes raise; a native pipeline is built from the config's own ``Normalize`` / ``RandomFlip`` /
``ImageToTensor`` / ``Collect`` steps, and a config whose test pipeline does not have the expected
``LoadImageFromFile`` + ``MultiScaleFlipAug`` shape raises instead of guessing. The MSDA path is chosen as in
``infer.load_model`` (msda_compat.configure_msda).
"""
import contextlib
import copy

import mmcv
import numpy as np

from msda_compat import configure_msda

SIZE_DIVISOR = 32


def _parse_size(size):
    if size == 'config' or size == 'native':
        return size
    if isinstance(size, (int, np.integer)) and not isinstance(size, bool) and size > 0:
        return (int(size), int(size))
    if isinstance(size, (tuple, list)) and len(size) == 2 and all(
            isinstance(v, (int, np.integer)) and v > 0 for v in size):
        return (int(size[0]), int(size[1]))
    raise ValueError("size must be 'config', 'native', a positive int or a (W, H) pair, got %r" % (size,))


def test_pipeline_for(cfg, size, pad_divisor=None):
    """The test pipeline (list of dicts) for ``size`` built from ``cfg.data.test.pipeline``."""
    size = _parse_size(size)
    base = copy.deepcopy(cfg.data.test.pipeline)
    if size == 'config':
        return base
    if len(base) != 2 or base[0]['type'] not in ('LoadImageFromFile', 'LoadImageFromWebcam') or \
            base[1]['type'] != 'MultiScaleFlipAug':
        raise ValueError('expected a test pipeline [LoadImageFromFile, MultiScaleFlipAug(...)], got %s'
                         % [p['type'] for p in base])
    aug = base[1]
    inner = [t for t in aug['transforms'] if t['type'] not in ('Resize', 'Pad')]
    if [t['type'] for t in aug['transforms']].count('Resize') != 1:
        raise ValueError('expected exactly one Resize in MultiScaleFlipAug.transforms')
    try:
        norm_at = [t['type'] for t in inner].index('Normalize')
    except ValueError:
        raise ValueError('expected a Normalize step in MultiScaleFlipAug.transforms')
    if size == 'native':
        pad_divisor = SIZE_DIVISOR if pad_divisor is None else pad_divisor
        head = [dict(type='Resize', keep_ratio=True)]
        new_aug = dict(type='MultiScaleFlipAug', scale_factor=1.0, flip=aug.get('flip', False))
    else:
        head = [dict(type='Resize', keep_ratio=True)]
        new_aug = dict(type='MultiScaleFlipAug', img_scale=size, flip=aug.get('flip', False))
    if pad_divisor:
        # before Normalize, so 255 is white paper, not a normalised value; a per-channel tuple, because a scalar
        # pad value fills only the first channel (cv2.copyMakeBorder), which gives a coloured border
        head.append(dict(type='Pad', size_divisor=int(pad_divisor),
                         pad_val=dict(img=(255, 255, 255), masks=0, seg=255)))
    # Resize/Pad go first (as in the config, where Resize is first); the rest keeps its order
    new_aug['transforms'] = head + inner
    return [base[0], new_aug]


def build_config(config, size, pad_divisor=None):
    cfg = mmcv.Config.fromfile(config) if not isinstance(config, mmcv.Config) else copy.deepcopy(config)
    cfg.data.test.pipeline = test_pipeline_for(cfg, size, pad_divisor)
    return cfg


def build_model(config, ckpt, device, size='config', pad_divisor=None, msda=None):
    """Detector whose test pipeline uses ``size`` (see module docstring). Returns the model."""
    from mmdet.apis import init_detector
    cfg = build_config(config, size, pad_divisor)
    configure_msda(device, msda)
    model = init_detector(cfg, ckpt, device=device)
    model.lineformer_input_size = _parse_size(size)
    return model


def input_meta(model, img):
    """What the network receives for ``img`` (ndarray HxWx3 BGR or a path), without running it."""
    from mmdet.apis.inference import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    cfg = model.cfg.copy()
    if isinstance(img, np.ndarray):
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        data = dict(img=img)
    else:
        data = dict(img_info=dict(filename=img), img_prefix=None)
    pipe = Compose(replace_ImageToTensor(cfg.data.test.pipeline))
    out = pipe(data)
    meta = out['img_metas'][0].data
    return dict(ori_shape=tuple(meta['ori_shape']), img_shape=tuple(meta['img_shape']),
                pad_shape=tuple(meta['pad_shape']), scale_factor=np.asarray(meta['scale_factor']).tolist(),
                input_shape=tuple(out['img'][0].data.shape))


@contextlib.contextmanager
def record_inputs(model):
    """Record (input tensor shape, img_metas[0]) of every simple_test call inside the block."""
    if 'simple_test' in vars(model):
        raise RuntimeError('simple_test is already patched on this model instance')
    seen = []
    orig = model.simple_test

    def wrapped(img, img_metas, **kw):
        seen.append((tuple(img.shape), dict(img_metas[0])))
        return orig(img, img_metas, **kw)

    model.simple_test = wrapped
    try:
        yield seen
    finally:
        del model.simple_test  # back to the class method
