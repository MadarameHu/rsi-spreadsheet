#!/usr/bin/env python3
"""Which targeted error class each failed trainee attempt actually fell into. No model calls.

Every failed attempt workbook is recalculated with headless LibreOffice, like the
scorer, and compared with each author wrong-answer probe (diagnostics/probes/probes.json,
named by class id such as "c1-..."). Agreement between an attempt and a probe is the
share of cells, among those wrong in either, where both hold the same value; 1.0
means the attempt reproduces the probe exactly.

  hit        one probe agrees >= THRESHOLD and no other comes within MARGIN of it
  ambiguous  several probes agree >= THRESHOLD within MARGIN: the probes cannot tell
             these classes apart on this data
  unmatched  no probe reaches THRESHOLD (another error, or several classes combined)
  no_value_mismatch  reward 0 while every target value matches (form check etc.)

off_target is true when more than half of the failed attempts are unmatched. Probe
separability (pairwise agreement between probes) is reported too. Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_trace  # noqa: E402

SCHEMA = 'defect-hits-v1'
THRESHOLD = 0.8
MARGIN = 0.1
CLASS = re.compile(r'^(c\d+)\b')


def probe_class(name):
    m = CLASS.match(name)
    return m.group(1) if m else name


def agreement(actual, probe, expected, compare):
    wrong = {k for k in expected if not compare(expected[k], actual.get(k))}
    wrong |= {k for k in expected if not compare(expected[k], probe.get(k))}
    if not wrong:
        return 1.0
    return sum(compare(probe.get(k), actual.get(k)) for k in wrong) / len(wrong)


def classify(actual, expected, probes, compare):
    """actual / expected: {cell: value}; probes: [(name, {cell: value})]."""
    if all(compare(expected[k], actual.get(k)) for k in expected):
        return {'category': 'no_value_mismatch', 'classes': [], 'agreement': {}}
    scores = {n: agreement(actual, v, expected, compare) for n, v in probes}
    best = max(scores.values(), default=0.0)
    close = [n for n, a in scores.items() if a >= THRESHOLD and best - a <= MARGIN]
    category = 'unmatched' if not close else 'hit' if len(close) == 1 else 'ambiguous'
    return {'category': category, 'classes': sorted({probe_class(n) for n in close}), 'probes': close,
            'agreement': {n: round(a, 3) for n, a in sorted(scores.items(), key=lambda x: -x[1])}}


def report(candidate, probes_json, attempts, work_dir=None):
    """attempts: [{'id', 'workbook' (path or None), 'reward'}] in order."""
    candidate = Path(candidate)
    tests = candidate / 'tests'
    _, compare = score_trace._scorer(tests)
    _, cells = score_trace._target_cells(tests)
    keys = [f'{s}!{a}' for s, a in cells]
    expected = dict(zip(keys, score_trace._values(tests / 'target.xlsx', cells).values()))
    probes = [(p['name'], {r['cell']: r['recalculated'] for r in p['rows']})
              for p in json.loads(Path(probes_json).read_text())['probes']]
    failed = [a for a in attempts if a.get('reward') is not None and a['reward'] < 1]
    rows = []
    owned = work_dir is None
    work = Path(tempfile.mkdtemp()) if owned else Path(work_dir)
    try:
        staged = {}
        for a in failed:
            if a.get('workbook') and Path(a['workbook']).is_file():
                path = work / 'in' / f"attempt-{a['id']}.xlsx"
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(a['workbook'], path)
                staged[a['id']] = path
        recalculated = score_trace.recalc(list(staged.values()), work / 'recalculated') if staged else {}
        for a in failed:
            if a['id'] not in staged:
                rows.append({'id': a['id'], 'reward': a['reward'], 'category': 'no_workbook', 'classes': []})
                continue
            actual = dict(zip(keys, score_trace._values(recalculated[f"attempt-{a['id']}"], cells).values()))
            wrong = sum(not compare(expected[k], actual[k]) for k in keys)
            rows.append({'id': a['id'], 'reward': a['reward'], 'wrong_cells': wrong,
                         'total_cells': len(keys), **classify(actual, expected, probes, compare)})
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)
    counts = {}
    for r in rows:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    per_class = {}
    for r in rows:
        for c in r['classes']:
            per_class[c] = per_class.get(c, 0) + 1
    attributable = [r for r in rows if r['category'] in ('hit', 'ambiguous', 'unmatched')]
    separability = {f'{a[:40]} ~ {b[:40]}': round(agreement(va, vb, expected, compare), 3)
                    for i, (a, va) in enumerate(probes) for b, vb in probes[i + 1:]}
    unmatched = counts.get('unmatched', 0)
    return {'schema_version': SCHEMA, 'threshold': THRESHOLD,
            'attempts': len(attempts), 'failed': len(failed),
            'targeted_classes': sorted({probe_class(n) for n, _ in probes}),
            'categories': counts, 'class_hits': dict(sorted(per_class.items())),
            'probe_pair_agreement': separability,
            'off_target': bool(attributable) and unmatched * 2 > len(attributable),
            'note': None if failed else 'no failed attempts: nothing to attribute',
            'rows': rows}


def episode_attempts(ep):
    """Trainee attempts of a host-validated episode, with their collected workbooks."""
    ep = Path(ep)
    validation = json.loads((ep / 'diagnostics/validation.json').read_text())
    target = tomllib.loads((ep / 'diagnostics/candidate/task.toml').read_text())['artifacts'][0].lstrip('/')
    out = []
    for i, a in enumerate(validation.get('attempts', [])):
        book = Path(a['result_path']).parent / 'artifacts' / target if a.get('result_path') else None
        out.append({'id': f'{i:02d}', 'workbook': str(book) if book else None, 'reward': a.get('reward')})
    return out


def write_for_episode(ep):
    ep = Path(ep)
    result = report(ep / 'diagnostics/candidate', ep / 'diagnostics/probes/probes.json', episode_attempts(ep))
    (ep / 'diagnostics/defect-hits.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('episode', nargs='?', help='host-validated episode; writes diagnostics/defect-hits.json')
    ap.add_argument('--candidate', help='task dir, for trials outside an episode')
    ap.add_argument('--probes', help='probes.json for --candidate')
    ap.add_argument('--trials', nargs='*', default=[], help='Harbor trial dirs (contain */*/result.json)')
    ap.add_argument('--out', help='write the report here (with --candidate)')
    args = ap.parse_args(argv)
    if args.episode:
        result = write_for_episode(args.episode)
    else:
        attempts = []
        for t in args.trials:
            res = sorted(Path(t).glob('*/*/result.json'))
            r = json.loads(res[0].read_text()) if res else {}
            reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            book = res[0].parent / 'artifacts/app/data.xlsx' if res else None
            attempts.append({'id': Path(t).name[-2:], 'workbook': str(book) if book else None, 'reward': reward})
        result = report(args.candidate, args.probes, attempts)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
    for r in result['rows']:
        print(r['id'], r['category'], r['classes'], r.get('agreement'))


if __name__ == '__main__':
    main()
