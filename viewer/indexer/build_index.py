#!/usr/bin/env python3
"""Build a portable, read-only index for the batch viewer.

Scans ``runs/batch-*`` and the batch work directory and derives each
attempt's stage status from the files that actually exist. Missing stages are a
first-class state: most attempts in this batch stopped early, and the viewer
must say where, not fail.

Rules:

* Never open ``.xlsx`` here. The index only reads small JSON files.
* Never follow symlinks (a run contains a broken ``.venv`` symlink).
* Paths in the index are relative to the configured roots, so the same index is
  valid locally and on another machine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RUN_RE = re.compile(r'^batch-(?P<batch>\d+)-(?P<task>\d+)-(?P<kind>[adr])(?P<n>\d+)$')
KIND_LABEL = {'a': 'analysis', 'd': 'design', 'r': 'retry'}
KIND_ORDER = {'a': 0, 'd': 1, 'r': 2}
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_EVENTS_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {'.venv', 'node_modules', '__pycache__', '.git', 'sessions',
             '.mypy_cache', '.pytest_cache', '.godot'}

STAGE_ORDER = [
    ('observations', '观察'),
    ('aggregate', '汇总'),
    ('design', '出题'),
    ('leak', '泄露检查'),
    ('reference', '参考解'),
    ('gate', '模板 probe'),
    ('trainee', '试做'),
    ('hits', '缺陷归因'),
    ('route', '分流'),
]


def read_json(path: Path | None, limit: int = MAX_JSON_BYTES):
    if path is None or not path.is_file():
        return None, 'missing'
    try:
        if path.stat().st_size > limit:
            return None, 'too large'
        return json.loads(path.read_text(errors='replace')), None
    except (OSError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {exc}'[:200]


def first(*paths: Path | None) -> Path | None:
    for p in paths:
        if p is not None and p.is_file():
            return p
    return None


def _first_rel(ep: Path, *patterns: str) -> str | None:
    for pattern in patterns:
        for hit in sorted(ep.glob(pattern)):
            if hit.is_file():
                return str(hit.relative_to(ep))
    return None


def tree_size(root: Path) -> tuple[int, int]:
    total = files = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.is_symlink():
                    continue
                total += p.stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def events_summary(path: Path) -> dict:
    if not path.is_file():
        return {'state': 'missing'}
    try:
        if path.stat().st_size > MAX_EVENTS_BYTES:
            return {'state': 'present', 'truncated': True}
        lines = [ln for ln in path.read_text(errors='replace').splitlines() if ln.strip()]
        last = {}
        if lines:
            try:
                last = json.loads(lines[-1])
            except ValueError:
                last = {}
        return {'state': 'present', 'n_events': len(lines), 'last': last}
    except OSError as exc:
        return {'state': 'unreadable', 'error': str(exc)[:120]}


def scan_episode(ep: Path) -> dict:
    diag = ep / 'diagnostics'
    work = ep / 'work'
    out: dict = {'episode': ep.name, 'stages': {}, 'metrics': {}, 'files': {},
                 'warnings': []}

    # -- observations ------------------------------------------------------
    obs_p = diag / 'observations/index.json'
    obs, err = read_json(obs_p)
    if obs:
        out['stages']['observations'] = {
            'state': obs.get('status') or 'present',
            'n_observations': len(obs.get('observation_ids') or []),
            'n_workers': len(obs.get('workers') or []),
            'reused_from': bool(obs.get('reuse_observations_from')),
        }
        out['files']['observations'] = str(obs_p.relative_to(ep))
    else:
        out['stages']['observations'] = {'state': 'missing'}
        if obs_p.is_file():
            out['warnings'].append(f'observations index: {err}')

    # -- aggregate / attribution ------------------------------------------
    attr_p = diag / 'attribution.json'
    attr, err = read_json(attr_p)
    if attr:
        claims = attr.get('claims') or []
        goals = attr.get('training_goals') or []
        strengths: dict[str, int] = {}
        for c in claims:
            key = str(c.get('evidence_strength') or 'unknown')
            strengths[key] = strengths.get(key, 0) + 1
        out['stages']['aggregate'] = {
            'state': 'done',
            'n_claims': len(claims),
            'n_goals': len(goals),
            'evidence_strengths': strengths,
            'derivation': attr.get('derivation'),
        }
        out['metrics']['training_goals'] = [
            {'id': g.get('id'), 'ability': str(g.get('ability') or '')[:400]}
            for g in goals
        ]
        out['files']['attribution'] = str(attr_p.relative_to(ep))
    else:
        out['stages']['aggregate'] = {'state': 'missing'}

    # -- design ------------------------------------------------------------
    design_p = first(diag / 'design.json', work / 'design.json')
    design, err = read_json(design_p)
    if design:
        out['stages']['design'] = {
            'state': 'done',
            'primary_goal_id': design.get('primary_goal_id'),
            'n_targets': len(design.get('targets') or []),
            'n_preserves': len(design.get('preserves') or []),
            'n_simplifies': len(design.get('simplifies') or []),
        }
        out['files']['design'] = str(design_p.relative_to(ep))
    else:
        out['stages']['design'] = {'state': 'missing'}

    probes_p = work / 'probes.json'
    probes, _ = read_json(probes_p)
    if probes:
        out['metrics']['authored_probes'] = (
            len(probes) if isinstance(probes, list) else 1)

    # -- leak check --------------------------------------------------------
    leak_p = diag / 'leak-check.json'
    leak, err = read_json(leak_p)
    blocked = sorted(diag.glob('leak-check.blocked-*.json'))
    if leak:
        verdicts = leak.get('verdicts') or []
        out['stages']['leak'] = {
            'state': leak.get('status') or 'present',
            'n_verdicts': len(verdicts),
            'n_leaks': sum(1 for v in verdicts
                           if str(v.get('verdict') or '').lower() in ('leak', 'reject', 'rejected')),
            'model': leak.get('model'),
            'attempts': leak.get('attempts'),
        }
        out['files']['leak'] = str(leak_p.relative_to(ep))
    elif blocked:
        out['stages']['leak'] = {'state': 'blocked', 'n_blocked_files': len(blocked)}
        out['files']['leak'] = str(blocked[-1].relative_to(ep))
    else:
        out['stages']['leak'] = {'state': 'missing'}

    # -- validation: oracle + trainee -------------------------------------
    val_p = diag / 'validation.json'
    val, err = read_json(val_p)
    if val:
        oracle = val.get('oracle') or {}
        oracle_status = str(oracle.get('status') or 'missing')
        out['stages']['reference'] = {
            'state': {'completed': 'done'}.get(oracle_status, oracle_status),
            'status': oracle_status,
            'reward': oracle.get('reward'),
        }
        attempts = val.get('attempts') or []
        solved = val.get('solved_attempts')
        completed = val.get('completed_attempts')
        rate = (solved / completed) if isinstance(solved, int) and completed else None
        counts = {'pending': 0, 'failed': 0, 'completed': 0}
        for a in attempts:
            st = str(a.get('status') or '')
            if st in counts:
                counts[st] += 1
        raw_status = str(val.get('status') or 'present')
        if completed and solved is not None:
            state = 'done'
        elif 'pending' in raw_status:
            state = 'pending'
        elif 'fail' in raw_status:
            state = 'failed'
        else:
            state = 'present'
        out['stages']['trainee'] = {
            'state': state,
            'status': raw_status,
            'requested': val.get('requested_attempts'),
            'solved': solved,
            'completed': completed,
            'pass_rate': rate,
            'counts': counts,
            'difficulty_status': val.get('difficulty_status'),
        }
        target = _first_rel(ep, 'diagnostics/candidate/tests/target.xlsx',
                            'work/out/*/tests/target.xlsx')
        rows = []
        for a in attempts:
            aid = str(a.get('id') or '')
            tail = aid.rsplit('-', 1)[-1]
            idx = int(tail) if tail.isdigit() else -1
            final = _first_rel(
                ep,
                f'work/testruns/{aid}/**/final.xlsx',
                f'work/testruns/{aid}/**/bridge-final.xlsx',
            ) if idx >= 0 else None
            rows.append({'id': aid, 'kind': a.get('kind'),
                         'status': a.get('status'), 'reward': a.get('reward'),
                         'error': str(a.get('error') or '')[:200] or None,
                         'final_xlsx': final})
        out['metrics']['attempts'] = rows
        out['metrics']['target_xlsx'] = target
        out['files']['validation'] = str(val_p.relative_to(ep))
    else:
        out['stages']['reference'] = {'state': 'missing'}
        out['stages']['trainee'] = {'state': 'missing'}

    # -- template gate -----------------------------------------------------
    gate_p = diag / 'template-gate.json'
    gate, err = read_json(gate_p)
    if gate:
        probes = gate.get('probes') or []
        out['stages']['gate'] = {
            'state': gate.get('status') or 'present',
            'eligible': gate.get('eligible_for_training'),
            'n_probes': len(probes),
            'n_issues': len(gate.get('issues') or []),
            'llm_review': gate.get('llm_review'),
        }
        out['metrics']['gate_probes'] = [
            {'id': p.get('id'), 'expected_valid': p.get('expected_valid'),
             'actual_accept': p.get('actual_accept'),
             'execution_status': p.get('execution_status'),
             'reason': str(p.get('reason') or '')[:300]}
            for p in probes
        ]
        out['files']['gate'] = str(gate_p.relative_to(ep))
    else:
        out['stages']['gate'] = {'state': 'missing'}

    # -- defect hits -------------------------------------------------------
    hits_p = diag / 'defect-hits.json'
    hits, err = read_json(hits_p)
    if hits:
        out['stages']['hits'] = {
            'state': 'done',
            'off_target': hits.get('off_target'),
            'attempts': hits.get('attempts'),
            'failed': hits.get('failed'),
            'class_hits': hits.get('class_hits') or {},
            'categories': hits.get('categories') or {},
            'threshold': hits.get('threshold'),
        }
        out['files']['hits'] = str(hits_p.relative_to(ep))
    else:
        out['stages']['hits'] = {'state': 'missing'}

    # -- route -------------------------------------------------------------
    route_p = diag / 'route.json'
    route, err = read_json(route_p)
    if route:
        out['stages']['route'] = {
            'state': 'done',
            'route': route.get('route'),
            'pass_rate': route.get('pass_rate'),
            'band': route.get('band'),
            'solved': route.get('solved'),
            'completed': route.get('completed'),
        }
        out['files']['route'] = str(route_p.relative_to(ep))
    else:
        out['stages']['route'] = {'state': 'missing'}

    # -- synthesis / events ------------------------------------------------
    syn_p = ep / 'synthesis.json'
    syn, _ = read_json(syn_p)
    if syn:
        out['metrics']['synthesis'] = {
            'benchmark': syn.get('benchmark'),
            'original_task': syn.get('original_task'),
            'delivered': syn.get('delivered'),
            'n_delivered': syn.get('n_delivered'),
            'elapsed_sec': syn.get('elapsed_sec'),
            'agent_execution_seconds': syn.get('agent_execution_seconds'),
            'budget_mode': syn.get('budget_mode'),
            'observations_index': syn.get('observations_index'),
        }
    out['metrics']['events'] = events_summary(diag / 'events.jsonl')


    for name in ('candidate', 'design', 'observations', 'probes', 'records',
                 'score-trace', 'template-probes', 'validation', 'leak-check'):
        p = diag / name
        if p.exists():
            # Directories always get a `dir_` prefix: `design`, `observations`
            # and `validation` are also JSON file keys, and a module link must
            # never resolve to a directory.
            out['files'][f'dir_{name}'] = str(p.relative_to(ep))
    for key, p in (('events', diag / 'events.jsonl'),
                   ('synthesis', ep / 'synthesis.json')):
        if p.is_file():
            out['files'][key] = str(p.relative_to(ep))

    return out


def scan_run(run: Path, runs_root: Path) -> dict:
    m = RUN_RE.match(run.name)
    record: dict = {
        'run_id': run.name,
        'rel': str(run.relative_to(runs_root)),
        'task': f'task-{m.group("task")}' if m else None,
        'kind': m.group('kind') if m else None,
        'kind_label': KIND_LABEL.get(m.group('kind'), 'unknown') if m else 'unknown',
        'attempt_no': int(m.group('n')) if m else None,
        'mtime': datetime.fromtimestamp(run.stat().st_mtime, timezone.utc).isoformat(),
        'size_bytes': 0, 'n_files': 0,
        'episodes': [], 'warnings': [],
    }
    size, files = tree_size(run)
    record['size_bytes'], record['n_files'] = size, files

    meta, err = read_json(run / 'run.json')
    if meta:
        record['meta'] = {
            'run_id': meta.get('run_id'),
            'started_at': meta.get('started_at'),
            'benchmark': meta.get('benchmark'),
            'customer': (meta.get('customer') or {}).get('model'),
            'researcher': (meta.get('researcher') or {}).get('model'),
            'judge': (meta.get('judge') or {}).get('model'),
            'attempts': (meta.get('customer') or {}).get('attempts'),
            'band': [((meta.get('score') or {}).get('band_lo')),
                     ((meta.get('score') or {}).get('band_hi'))],
        }
    else:
        record['warnings'].append('run.json missing or unreadable')

    eps = sorted(run.glob('spreadsheet/task-*/ep*'))
    if not eps:
        record['warnings'].append('no episode directory (attempt stopped before run)')
    for ep in eps:
        if not ep.is_dir():
            continue
        scanned = scan_episode(ep)
        scanned['rel'] = str(ep.relative_to(run))
        scanned['task'] = ep.parent.name
        record['episodes'].append(scanned)

    return record


def rollup(run: dict) -> dict:
    """One episode per attempt in this batch; tolerate zero or several.

    The chosen episode's stage payload is copied whole: the route and the
    pass rate live in it, and a state-only rollup silently dropped them.
    """
    stages: dict[str, dict] = {}
    rank = {'done': 5, 'accepted': 5, 'completed': 5, 'reference_completed': 4,
            'present': 4, 'rejected': 4, 'pending': 3, 'blocked': 3,
            'partial': 2, 'error': 2, 'failed': 1, 'missing': 0}
    for key, _label in STAGE_ORDER:
        best_ep, best = None, {'state': 'missing'}
        for ep in run['episodes']:
            payload = ep['stages'].get(key) or {'state': 'missing'}
            if rank.get(payload.get('state'), 0) > rank.get(best.get('state'), 0):
                best_ep, best = ep, payload
        stages[key] = dict(best)
        if best_ep is not None:
            stages[key]['episode'] = best_ep.get('episode')
        stages[key]['n_episodes'] = len(run['episodes'])
    done = [k for k, _ in STAGE_ORDER
            if stages.get(k, {}).get('state') not in ('missing',)]
    files: dict[str, str] = {}
    for ep in run['episodes']:
        for key, rel in (ep.get('files') or {}).items():
            files.setdefault(key, f"{ep['rel']}/{rel}")
    best_ep = max(run['episodes'], key=lambda e: len(
        [1 for s in e['stages'].values() if s.get('state') != 'missing']),
        default=None)
    return {'stages': stages, 'last_stage': done[-1] if done else None,
            'n_stages': len(done), 'files': files,
            'episode_rel': best_ep.get('rel') if best_ep else None}


def build(runs_root: Path, batch_root: Path, pattern: str = '*') -> dict:
    status, status_err = read_json(batch_root / 'status.json')
    status = status or {}
    runs: list[dict] = []
    for run in sorted(runs_root.glob(pattern)):
        if not run.is_dir():
            continue
        record = scan_run(run, runs_root)
        record.update(rollup(record))
        runs.append(record)

    by_task: dict[str, list[dict]] = {}
    for r in runs:
        if r.get('task'):
            by_task.setdefault(r['task'], []).append(r)

    tasks = []
    for task_id in sorted(by_task):
        attempts = sorted(by_task[task_id],
                          key=lambda r: (KIND_ORDER.get(r['kind'], 9),
                                         r.get('attempt_no') or 0))
        entry = status.get(task_id) or {}
        current_run = entry.get('design_run')
        for a in attempts:
            a['current'] = a['run_id'] == current_run
        if current_run is None and attempts:
            # Some tasks ran analysis and design in one pass and status.json
            # carries no design_run. The furthest attempt is the current one.
            best = max(attempts, key=lambda r: (r.get('n_stages') or 0,
                                                KIND_ORDER.get(r['kind'], 0),
                                                r.get('attempt_no') or 0))
            best['current'] = True
            current_run = best['run_id']
        tasks.append({
            'id': task_id,
            'status': entry,
            'current_run': current_run,
            'attempts': attempts,
        })

    summary = summarise(tasks, runs)
    return {
        'schema_version': SCHEMA_VERSION,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'roots': {'runs': runs_root.name, 'batch': batch_root.name},
        'batch': {
            'id': batch_root.name,
            'status_updated': (batch_root / 'status.json').stat().st_mtime
            if (batch_root / 'status.json').is_file() else None,
            'status_error': status_err,
            'summary': summary,
        },
        'tasks': tasks,
        'runs': runs,
    }


def summarise(tasks: list[dict], runs: list[dict]) -> dict:
    routes: dict[str, int] = {}
    leaks: dict[str, int] = {}
    gates: dict[str, int] = {}
    funnel = {key: 0 for key, _ in STAGE_ORDER}
    pass_rates = []
    for t in tasks:
        for a in t['attempts']:
            st = a.get('stages') or {}
            for key, _label in STAGE_ORDER:
                if st.get(key, {}).get('state') not in ('missing',):
                    funnel[key] += 1
            route = (st.get('route') or {}).get('route')
            if route:
                routes[route] = routes.get(route, 0) + 1
            leak = (st.get('leak') or {}).get('state')
            if leak:
                leaks[leak] = leaks.get(leak, 0) + 1
            gate = (st.get('gate') or {}).get('state')
            if gate:
                gates[gate] = gates.get(gate, 0) + 1
            rate = (st.get('trainee') or {}).get('pass_rate')
            if isinstance(rate, (int, float)):
                pass_rates.append(rate)
    return {
        'n_tasks': len(tasks),
        'n_runs': len(runs),
        'n_attempts': sum(len(t['attempts']) for t in tasks),
        'funnel': funnel,
        'routes': routes,
        'leak': leaks,
        'gate': gates,
        'pass_rates': sorted(pass_rates),
    }


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    default_batch = repo / 'batch'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, default=repo / 'runs')
    ap.add_argument('--batch', type=Path, default=default_batch)
    ap.add_argument('--pattern', default='*')
    ap.add_argument('--out', type=Path, default=repo / 'viewer' / 'data' / 'index.json')
    a = ap.parse_args(argv)

    if not a.runs.is_dir():
        raise SystemExit(f'runs root not found: {a.runs}')
    # A fresh checkout has no batch/ yet; an empty index is still valid.
    index = build(a.runs, a.batch, a.pattern)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    s = index['batch']['summary']
    print(f"wrote {a.out}")
    print(f"  tasks {s['n_tasks']}  runs {s['n_runs']}  attempts {s['n_attempts']}")
    print(f"  funnel " + ' '.join(f'{k}={v}' for k, v in s['funnel'].items()))
    print(f"  routes {s['routes']}  leak {s['leak']}  gate {s['gate']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
