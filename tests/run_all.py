# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Run every unit test without pytest (no GPU, no checkpoint needed):

    python tests/run_all.py

The same set as `python -m pytest` (tests/test_*.py and tools/equivalence/tests/test_*.py). A module whose
dependencies are missing raises unittest.SkipTest at import; it is reported as SKIPPED with the reason. Every
test_* function runs in file order of names; a failure prints its traceback. Exit 1 if any test failed.
"""
import importlib.util
import sys
import time
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / 'tests', ROOT / 'tools' / 'equivalence' / 'tests']


def main():
    n_ok, n_fail, skipped = 0, 0, []
    for d in DIRS:
        for f in sorted(d.glob('test_*.py')):
            rel = f.relative_to(ROOT)
            spec = importlib.util.spec_from_file_location(f.stem, f)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(d))
            try:
                spec.loader.exec_module(mod)
            except unittest.SkipTest as e:
                skipped.append((rel, str(e)))
                print('SKIPPED %s: %s' % (rel, e), flush=True)
                continue
            finally:
                sys.path.remove(str(d))
            for name in sorted(vars(mod)):
                fn = getattr(mod, name)
                if not (name.startswith('test_') and callable(fn)):
                    continue
                t = time.time()
                try:
                    fn()
                except unittest.SkipTest as e:
                    skipped.append(('%s::%s' % (rel, name), str(e)))
                    print('SKIPPED %s::%s: %s' % (rel, name, e), flush=True)
                    continue
                except Exception:
                    n_fail += 1
                    print('FAIL %s::%s' % (rel, name), flush=True)
                    traceback.print_exc()
                    continue
                n_ok += 1
                print('ok %s::%s %.2fs' % (rel, name, time.time() - t), flush=True)
    print('%d passed, %d failed, %d skipped' % (n_ok, n_fail, len(skipped)))
    for what, why in skipped:
        print('  skipped %s: %s' % (what, why))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
