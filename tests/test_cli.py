# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""The command line (lineformer_cli.py) without a model: the help of the three forms, the settings file, the
deprecated flags, the threshold, the checkpoint lookup and the id clash check. No GPU, no checkpoint, no torch.

pytest tests/test_cli.py -q      or      python tests/run_all.py
"""
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lineformer_cli as cli  # noqa: E402
import lineformer_engine as engine  # noqa: E402

PUBLIC = {'single': ['--list', '--out', '--threshold', '--masks', '--force', '--cpu', '--settings'],
          'batch': ['--list', '--out', '--threshold', '--masks', '--force', '--cpu', '--settings'],
          'serve': ['--port', '--exit-on-failure', '--threshold', '--masks', '--cpu', '--settings']}
HIDDEN = ['--ckpt', '--config', '--device', '--msda', '--all-queries', '--kept-only', '--kept-thr', '--input-size',
          '--tile', '--tile-overlap', '--ids', '--instances', '--gpu-workers', '--gpu-mem-budget', '--pre-workers',
          '--post-workers', '--pre-threads', '--post-threads', '--gpu-threads', '--max-inflight', '--progress-s',
          '--host', '--drain-timeout', '--ready-file', '--verbose']


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError('%s not raised' % exc.__name__)


def _help(form):
    return cli._parser(form).format_help()


def _options(text):
    return set(re.findall(r'(?m)^\s+(?:-[a-z], )?(--[a-z-]+)', text))


def test_help_shows_only_the_small_interface():
    for form, public in PUBLIC.items():
        text = _help(form)
        assert _options(text) == set(public) | {'--help'}, (form, _options(text))
        assert 'lineformer.example.toml' in text and 'LINEFORMER_CKPT' in text
        for flag in HIDDEN:
            assert not re.search(r'(?m)^\s+%s\b' % re.escape(flag), text), (form, flag)


def _parse(form, argv):
    return cli._parser(form).parse_args(argv)


def _settings(d, text, name='s.toml'):
    p = Path(d) / name
    p.write_text(text, encoding='utf-8')
    return p


def test_example_settings_file_lists_every_key_with_its_default():
    import tomllib
    text = (ROOT / cli.EXAMPLE_SETTINGS).read_text(encoding='utf-8')
    assert cli.load_settings(ROOT / cli.EXAMPLE_SETTINGS) == {}  # everything commented out
    lines = re.findall(r'(?m)^# ([a-z_]+ = [^#\n]*?)\s*(?:#.*)?$', text)
    keys = [ln.split(' = ')[0] for ln in lines]
    assert keys == list(cli.SETTINGS), keys
    uncommented = tomllib.loads('\n'.join(lines))
    with tempfile.TemporaryDirectory() as d:
        got = cli.load_settings(_settings(d, '\n'.join(lines)))
    assert set(got) == set(uncommented) == set(cli.SETTINGS)
    for k, (default, *_rest) in cli.SETTINGS.items():  # the values shown are the defaults where there is a fixed one
        if default is not None and k != 'config':
            assert uncommented[k] == default, (k, uncommented[k], default)


def test_settings_file_checks():
    with tempfile.TemporaryDirectory() as d:
        s = cli.load_settings(_settings(d, 'gpu_workers = 2\ngpu_mem_budget = "3G"\nconfig = "c.py"\n'
                                           'input_size = 1024\nids = "parent_stem"\n'))
        assert s['gpu_workers'] == 2 and s['gpu_mem_budget'] == '3G' and s['input_size'] == 1024
        assert s['config'] == str((Path(d) / 'c.py').resolve())  # relative to the settings file
        e = _raises(cli.SetupError, cli.load_settings, _settings(d, 'gpu_worker = 2\n'))
        assert 'unknown key' in str(e) and 'gpu_worker' in str(e)
        for bad in ('gpu_workers = 0', 'gpu_workers = true', 'gpu_mem_budget = "lots"', 'ids = "hash"',
                    'input_size = "huge"', 'all_queries = "yes"', 'msda = "fast"', 'progress_s = -1'):
            _raises(cli.SetupError, cli.load_settings, _settings(d, bad + '\n'))
        _raises(cli.SetupError, cli.load_settings, _settings(d, 'gpu_workers = \n'))  # not TOML
        _raises(cli.SetupError, cli.load_settings, Path(d) / 'missing.toml')


def test_deprecated_flags_applied_with_a_warning():
    warned = []
    a = _parse('batch', ['--out', 'o', 'x.png', '--gpu-workers', '2', '--gpu-mem-budget', '3G', '--input-size',
                         '1024', '--ids', 'parent_stem', '--pre-workers', '3'])
    values, sources = cli.effective_settings(a, 'batch', warn=warned.append)
    assert values['gpu_workers'] == 2 and values['gpu_mem_budget'] == '3G' and values['input_size'] == 1024
    assert values['ids'] == 'parent_stem' and values['pre_workers'] == 3 and sources['gpu_workers'] == '--gpu-workers'
    assert len(warned) == 5 and all(w.startswith('DEPRECATED') for w in warned)
    assert any('gpu_mem_budget' in w for w in warned) and all('\n' not in w for w in warned)
    with tempfile.TemporaryDirectory() as d:
        s = _settings(d, 'gpu_workers = 1\ngpu_mem_budget = "3072M"\n')
        a = _parse('batch', ['--out', 'o', 'x.png', '--settings', str(s), '--gpu-mem-budget', '3G'])
        values, sources = cli.effective_settings(a, 'batch', warn=lambda m: None)  # the same value: fine
        assert values['gpu_workers'] == 1 and sources['gpu_workers'] == 'settings'
        a = _parse('batch', ['--out', 'o', 'x.png', '--settings', str(s), '--gpu-workers', '2'])
        e = _raises(cli.SetupError, cli.effective_settings, a, 'batch', warn=lambda m: None)
        assert 'disagree' in str(e) and 'gpu_workers' in str(e)
        # keys a form cannot honour: result-changing ones fail, the others are noted
        s2 = _settings(d, 'input_size = "native"\n', 'n.toml')
        a = _parse('single', ['--out', 'o', 'x.png', '--settings', str(s2)])
        assert 'not supported' in str(_raises(cli.SetupError, cli.effective_settings, a, 'single', warn=print))
        notes = []
        a = _parse('single', ['--out', 'o', 'x.png', '--settings', str(s)])
        cli.effective_settings(a, 'single', warn=notes.append)
        assert len(notes) == 2 and all('not used' in n for n in notes)


def test_threshold_and_device():
    a = _parse('batch', ['--out', 'o', 'x.png'])
    v, _ = cli.effective_settings(a, 'batch')
    assert cli._threshold(a, v) == 0.3 and cli._device(a) == 'auto'
    a = _parse('batch', ['--out', 'o', 'x.png', '--threshold', '0.1', '--cpu'])
    assert cli._threshold(a, v) == 0.1 and cli._device(a) == 'cpu'
    a = _parse('batch', ['--out', 'o', 'x.png', '--kept-thr', '0.2', '--kept-only'])
    assert cli._threshold(a, v) == 0.2
    a = _parse('batch', ['--out', 'o', 'x.png', '--kept-thr', '0.2', '--threshold', '0.3'])
    _raises(cli.SetupError, cli._threshold, a, v)
    a = _parse('batch', ['--out', 'o', 'x.png', '--threshold', '1.5'])
    _raises(cli.SetupError, cli._threshold, a, v)
    a = _parse('batch', ['--out', 'o', 'x.png', '--all-queries'])
    v2, _ = cli.effective_settings(a, 'batch', warn=lambda m: None)
    assert cli._threshold(a, v2) is None
    a = _parse('batch', ['--out', 'o', 'x.png', '--all-queries', '--threshold', '0.3'])
    _raises(cli.SetupError, cli._threshold, a, v2)
    a = _parse('batch', ['--out', 'o', 'x.png', '--device', 'cuda:1'])
    assert cli._device(a) == 'cuda:1'
    a = _parse('batch', ['--out', 'o', 'x.png', '--device', 'cuda:1', '--cpu'])
    _raises(cli.SetupError, cli._device, a)


def test_checkpoint_lookup():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        root, home = d / 'fork', d / 'home'
        (root).mkdir()
        (home / '.cache' / 'lineformer').mkdir(parents=True)
        flag, env = d / 'flag.pth', d / 'env.pth'
        e = _raises(cli.SetupError, cli.resolve_ckpt, None, {}, root, home)
        for s in ('no checkpoint found', 'LINEFORMER_CKPT', str(root / 'iter_3000.pth'),
                  str(home / '.cache' / 'lineformer' / 'iter_3000.pth'), 'drive.google.com'):
            assert s in str(e), s
        (home / '.cache' / 'lineformer' / 'iter_3000.pth').write_bytes(b'c')
        assert cli.resolve_ckpt(None, {}, root, home)[1] == 'user cache'
        (root / 'iter_3000.pth').write_bytes(b'r')
        assert cli.resolve_ckpt(None, {}, root, home) == (str((root / 'iter_3000.pth').resolve()), 'fork root')
        _raises(cli.SetupError, cli.resolve_ckpt, None, {'LINEFORMER_CKPT': str(env)}, root, home)  # no fall-through
        env.write_bytes(b'e')
        assert cli.resolve_ckpt(None, {'LINEFORMER_CKPT': str(env)}, root, home)[1] == 'LINEFORMER_CKPT'
        _raises(cli.SetupError, cli.resolve_ckpt, str(flag), {'LINEFORMER_CKPT': str(env)}, root, home)
        flag.write_bytes(b'f')
        assert cli.resolve_ckpt(str(flag), {'LINEFORMER_CKPT': str(env)}, root, home) == (str(flag.resolve()),
                                                                                         '--ckpt')


def _exit_and_stderr(fn, argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            code = fn(argv)
        except SystemExit as e:
            code = e.code
    return code, err.getvalue()


def test_id_clash_stops_before_running():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a').mkdir()
        (Path(d) / 'b').mkdir()
        p1, p2 = str(Path(d) / 'a' / 'chart.png'), str(Path(d) / 'b' / 'chart.png')
        for fn in (cli.main_batch, cli.main_single):
            code, err = _exit_and_stderr(fn, ['--out', str(Path(d) / 'o'), p1, p2])
            assert code == 2 and "id 'chart'" in err and 'ids = "parent_stem"' in err, err
        assert not (Path(d) / 'o').exists()  # nothing ran


def test_engine_from_old_style_flags_and_settings():
    """--gpu-workers 2 --gpu-mem-budget 3G --kept-thr 0.3 --instances (v0.2.0 style) give the engine of v0.2.0, and
    the effective settings go into its engine_info."""
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / 'ck.pth'
        ck.write_bytes(b'0')
        ap = cli._parser('batch')
        a = ap.parse_args(['--out', str(Path(d) / 'o'), 'x.png', '--ckpt', str(ck), '--cpu', '--gpu-workers', '2',
                           '--gpu-mem-budget', '3G', '--kept-thr', '0.3', '--instances'])
        eng, err = _exit_and_stderr(lambda _: cli._make_engine(a, ap), None)
        assert 'DEPRECATED --gpu-workers' in err and 'DEPRECATED --kept-thr' in err and 'gpu_mem_budget' in err
        assert eng.n_gpu == 2 and eng.budget == ('bytes', 3 * 2 ** 30) and eng.mo.kept_thr == 0.3
        assert eng.mo.device == 'cpu' and eng.mo.ckpt == str(ck.resolve())
        eng.prepare()
        st = eng.engine_info['settings']
        assert st['checkpoint'] == {'path': str(ck.resolve()), 'source': '--ckpt'}
        assert st['sources']['gpu_workers'] == '--gpu-workers' and st['threshold'] == 0.3
        assert set(cli.SETTINGS) <= set(st)
        a = ap.parse_args(['--out', str(Path(d) / 'o'), 'x.png', '--ckpt', str(ck), '--cpu'])
        eng = cli._make_engine(a, ap)
        assert eng.n_gpu is None and eng.budget is None  # automatic sizing at start
        assert isinstance(eng, engine.Engine)
