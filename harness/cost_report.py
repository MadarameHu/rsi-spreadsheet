#!/usr/bin/env python3
"""Token and cost report for produced episodes, read-only, from records already on disk.

One episode is one produced task. Each model session is read from exactly one source:

  claude-code.txt  last {"type":"result"} line (complete sessions; local-claude and Harbor)
  trajectory.json  Harbor final_metrics when no result line exists (cancelled / partial)
  native-execution.json  Qwen trainee usage (prompt / completion only, no cache split)

Claude Code session jsonl files are never summed: they repeat usage per content block.
Prices come from configs/prices.json (USD per million tokens); an unpriced model gets a
null cost and is listed, never a guessed number. Reused observations (state.json pointing
into another run) are reported but kept out of the episode's new spend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = ROOT / 'configs' / 'prices.json'
FIELDS = ('input', 'cache_write', 'cache_read', 'output')
STAGES = ('observation', 'aggregate', 'design', 'audit', 'source_review', 'trainee', 'other')


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def result_line(path):
    """The Claude Code stream-json result record, or None if the session never finished."""
    try:
        lines = Path(path).read_text(errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{') and '"type"' in line and '"result"' in line:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get('type') == 'result':
                return obj
    return None


def from_result(obj):
    """One entry per model listed in modelUsage."""
    out = []
    for model, u in (obj.get('modelUsage') or {}).items():
        out.append({'model': model, 'input': u.get('inputTokens', 0),
                    'cache_write': u.get('cacheCreationInputTokens', 0),
                    'cache_read': u.get('cacheReadInputTokens', 0),
                    'output': u.get('outputTokens', 0), 'calls': obj.get('num_turns'),
                    'recorded_cost_usd': u.get('costUSD'), 'complete': True})
    return out


def from_trajectory(obj):
    """Harbor ATIF final_metrics; prompt tokens include both cache parts."""
    m = obj.get('final_metrics') or {}
    extra = m.get('extra') or {}
    write = extra.get('total_cache_creation_input_tokens', 0) or 0
    read = extra.get('total_cache_read_input_tokens', m.get('total_cached_tokens', 0)) or 0
    calls = sum(1 for s in obj.get('steps', []) if isinstance(s, dict) and s.get('metrics'))
    return [{'model': (obj.get('agent') or {}).get('model_name'),
             'input': (m.get('total_prompt_tokens') or 0) - write - read,
             'cache_write': write, 'cache_read': read,
             'output': m.get('total_completion_tokens') or 0, 'calls': calls,
             'recorded_cost_usd': None if extra.get('cost_source') == 'litellm_estimate'
             else m.get('total_cost_usd'), 'complete': False}]


def from_native(obj):
    u = obj.get('usage') or {}
    steps = [e.get('step_index') for e in obj.get('native_events', []) if isinstance(e, dict)]
    return [{'model': obj.get('model'), 'input': u.get('prompt_tokens') or 0, 'cache_write': 0,
             'cache_read': 0, 'output': u.get('completion_tokens') or 0,
             'calls': (max(s for s in steps if isinstance(s, int)) + 1)
             if any(isinstance(s, int) for s in steps) else None,
             'recorded_cost_usd': None, 'complete': True}]


def read_session(directory):
    """Usage of the one session recorded in this directory, or [] if none."""
    d = Path(directory)
    if (d / 'native-execution.json').exists():
        obj = _json(d / 'native-execution.json')
        return from_native(obj) if obj else []
    if (d / 'claude-code.txt').exists():
        obj = result_line(d / 'claude-code.txt')
        if obj:
            return from_result(obj)
    obj = _json(d / 'trajectory.json')
    if obj and isinstance(obj.get('agent'), dict) and obj.get('final_metrics'):
        return from_trajectory(obj)
    return []


SKIP_PARTS = {'evidence', 'inputs', 'sessions'}  # snapshot copies and Claude Code's own store
DIAGNOSTIC_STAGES = {'observations': 'observation', 'attribution': 'aggregate', 'design': 'design',
                     'audit-runs': 'audit', 'source-reviews': 'source_review'}


def stage_of(rel):
    parts = rel.parts
    if parts[:1] == ('diagnostics',) and len(parts) > 1:
        return DIAGNOSTIC_STAGES.get(parts[1], 'other')
    if any(p.startswith('trainee') for p in parts):
        return 'trainee'
    return 'other'


def session_dirs(root):
    names = ('claude-code.txt', 'trajectory.json', 'native-execution.json')
    return sorted({p.parent for n in names for p in Path(root).rglob(n)})


def _observation_pointers(diagnostics):
    for state in sorted((Path(diagnostics) / 'observations').glob('sample-*/state.json')):
        target = (_json(state) or {}).get('directory')
        if target and Path(target).resolve() != state.parent.resolve():
            yield Path(target)


def collect(ep):
    """All sessions of one episode, following reused analysis and observation pointers."""
    ep = Path(ep).resolve()
    rows = []
    for d in session_dirs(ep):
        rel = d.relative_to(ep)
        if SKIP_PARTS & set(rel.parts):
            continue
        for u in read_session(d):
            rows.append({**u, 'stage': stage_of(rel), 'path': str(d), 'reused': False})
    diag = ep / 'diagnostics'
    reused = [(t, 'observation') for t in _observation_pointers(diag)]
    source = (_json(diag / 'analysis_reuse.json') or {}).get('analysis_source')
    if source:  # whole analysis (observations + aggregate) produced by another episode
        reused.append((Path(source) / 'attribution', 'aggregate'))
        own = sorted((Path(source) / 'observations').glob('sample-*'))
        pointed = list(_observation_pointers(source))
        reused += [(t, 'observation') for t in (pointed or own)]
    seen = {r['path'] for r in rows}
    for target, stage in reused:
        for d in session_dirs(target):
            if str(d) in seen:
                continue
            seen.add(str(d))
            if SKIP_PARTS & set(d.relative_to(target).parts):
                continue
            for u in read_session(d):
                rows.append({**u, 'stage': stage, 'path': str(d), 'reused': True})
    return rows


def price(row, prices):
    p = prices.get(row['model'])
    if not isinstance(p, dict):
        return None
    return sum(row[f] * p[f] for f in FIELDS) / 1e6


def summarize(rows, prices):
    groups = {}
    for r in rows:
        key = (r['stage'], r['model'], r['reused'])
        g = groups.setdefault(key, {'stage': r['stage'], 'model': r['model'], 'reused': r['reused'],
                                    'sessions': 0, 'incomplete_sessions': 0, 'calls': 0,
                                    **{f: 0 for f in FIELDS}, 'cost_usd': 0.0, 'priced': True})
        g['sessions'] += 1
        g['incomplete_sessions'] += not r['complete']
        g['calls'] += r['calls'] or 0
        for f in FIELDS:
            g[f] += r[f]
        c = price(r, prices)
        if c is None:
            g['priced'] = False
        else:
            g['cost_usd'] += c
    out = sorted(groups.values(), key=lambda g: (STAGES.index(g['stage']), g['reused'], str(g['model'])))
    for g in out:
        g['cost_usd'] = round(g['cost_usd'], 4) if g['priced'] else None
    new = [g for g in out if not g['reused']]
    def total(gs):
        return {**{f: sum(g[f] for g in gs) for f in FIELDS},
                'cost_usd': round(sum(g['cost_usd'] or 0 for g in gs), 4),
                'unpriced_models': sorted({str(g['model']) for g in gs if not g['priced']})}
    return {'groups': out, 'episode_new_spend': total(new), 'including_reused': total(out),
            'incomplete_sessions': sum(g['incomplete_sessions'] for g in out)}


def markdown(ep, s):
    def k(n):
        return f'{n / 1000:,.1f}k'
    lines = [f'### {ep}', '',
             '| stage | model | reused | sessions (incomplete) | calls | input | cache_write | cache_read | output | USD |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for g in s['groups']:
        usd = '-' if g['cost_usd'] is None else f"{g['cost_usd']:.2f}"
        lines.append(f"| {g['stage']} | {g['model']} | {'yes' if g['reused'] else ''} | "
                     f"{g['sessions']} ({g['incomplete_sessions']}) | {g['calls']} | {k(g['input'])} | "
                     f"{k(g['cache_write'])} | {k(g['cache_read'])} | {k(g['output'])} | {usd} |")
    for label, t in (('episode new spend', s['episode_new_spend']), ('including reused', s['including_reused'])):
        miss = f" (unpriced: {', '.join(t['unpriced_models'])})" if t['unpriced_models'] else ''
        lines.append(f"\n**{label}: ${t['cost_usd']:.2f}**{miss}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('episodes', nargs='+', help='episode directories (runs/<run>/<bench>/<task>/<ep>)')
    ap.add_argument('--prices', default=str(DEFAULT_PRICES))
    ap.add_argument('--json', help='write the full report here')
    args = ap.parse_args(argv)
    prices = {k: v for k, v in (_json(args.prices) or {}).items() if not k.startswith('_')}
    report = {}
    for ep in args.episodes:
        rows = collect(ep)
        report[str(Path(ep).resolve())] = {'summary': summarize(rows, prices), 'sessions': rows}
        print(markdown(ep, report[str(Path(ep).resolve())]['summary']), '\n')
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
