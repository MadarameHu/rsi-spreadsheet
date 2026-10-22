"""Opt-in diagnostics configuration. No model/output policy is inferred here."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PARALLEL_BUDGET_MODE = 'explicit_parallel_stage_budgets_v1'


def _value(cfg: dict) -> dict:
    value = cfg.get('diagnostics', {})
    if not isinstance(value, dict):
        raise ValueError('diagnostics must be an object')
    unknown = set(value) - {'enabled', 'attribution_mode', 'parallel_attribution'}
    if unknown:
        raise ValueError(f'unsupported diagnostics configuration keys: {sorted(unknown)}')
    flag = value.get('enabled', False)
    if type(flag) is not bool:
        raise ValueError('diagnostics.enabled must be a boolean')
    # Sequential attribution was removed; only this mode exists now. The key is
    # still accepted so configuration files written for the old matrix keep working.
    if value.get('attribution_mode', 'parallel') != 'parallel':
        raise ValueError('diagnostics.attribution_mode must be parallel; sequential attribution was removed')
    return value


def enabled(cfg: dict) -> bool:
    return bool(_value(cfg).get('enabled', False))


def parallel_config(cfg: dict) -> dict:
    """Validate and return the parallel attribution budgets.

    Checked on use, not from enabled(): the budgets are only meaningful for a run
    that actually attributes, and a disabled diagnostics block may still carry
    them while the CLI toggles the flag.
    """
    value = _value(cfg)
    if not value.get('enabled', False):
        raise ValueError('parallel configuration requested without diagnostics')
    settings = value.get('parallel_attribution')
    required = {'max_concurrent', 'worker_budget_min', 'aggregate_budget_min', 'design_budget_min'}
    if not isinstance(settings, dict) or not required <= set(settings) or set(settings) - required - {'runtime', 'reuse_observations_from', 'allow_partial_reuse', 'attribution_revision_from', 'error_library'}:
        raise ValueError('parallel attribution requires explicit max_concurrent, worker_budget_min, aggregate_budget_min, design_budget_min')
    if settings.get('runtime', 'harbor') not in ('harbor', 'local-claude'):
        raise ValueError('parallel attribution runtime must be harbor or local-claude')
    if 'reuse_observations_from' in settings:
        reuse = settings['reuse_observations_from']
        if not isinstance(reuse, str) or not reuse.strip() or not Path(reuse).is_absolute():
            raise ValueError('reuse_observations_from must be an absolute diagnostics directory')
    if 'error_library' in settings and type(settings['error_library']) is not bool:
        raise ValueError('error_library must be a boolean')
    if 'allow_partial_reuse' in settings:
        if type(settings['allow_partial_reuse']) is not bool:
            raise ValueError('allow_partial_reuse must be a boolean')
        if settings['allow_partial_reuse'] and not settings.get('reuse_observations_from'):
            raise ValueError('allow_partial_reuse requires reuse_observations_from')
    if 'attribution_revision_from' in settings:
        revision = settings['attribution_revision_from']
        if not isinstance(revision, str) or not revision.strip() or not Path(revision).is_absolute():
            raise ValueError('attribution_revision_from must be an absolute episode directory')
        if not settings.get('reuse_observations_from') or settings.get('allow_partial_reuse'):
            raise ValueError('attribution revision requires full reuse_observations_from')
    count = settings['max_concurrent']
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError('parallel attribution max_concurrent must be an integer from 1 to 16')
    for key in required - {'max_concurrent'}:
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f'parallel attribution {key} must be an explicit positive finite number')
    return dict(settings)


def add_args(parser) -> None:
    parser.add_argument('--diagnostics', action=argparse.BooleanOptionalAction, default=None,
                        help='opt in to staged authoring and independent correctness diagnostics')


def apply_args(args, cfg) -> None:
    flag = getattr(args, 'diagnostics', None)
    if flag is not None:
        cfg.setdefault('diagnostics', {})['enabled'] = flag
    enabled(cfg)
