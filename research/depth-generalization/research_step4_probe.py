"""Can the step-4 table-conditioned scaffold accept an episode of another depth?

Imports `research/structure-generalization/common.py` unmodified and asks it to
build its `program`-observation lookup scaffold against episodes of each depth.
The answer is the blocker restated as an executable check rather than a claim.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _step4_common():
    path = _ROOT / 'research' / 'structure-generalization' / 'common.py'
    spec = importlib.util.spec_from_file_location('step4_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step4_lookup_accepts(depth, base, horizon):
    from tcn.generation import Host
    common = _step4_common()
    host = Host.create('logic', seed=0, index=0, split='test',
                       configuration=dict(base) | {'depth': depth, 'horizon': horizon})
    try:
        common.program_scaffold(host)
        return {'accepted': True, 'error': None}
    except Exception as exc:
        return {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
