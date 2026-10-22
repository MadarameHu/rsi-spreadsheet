#!/usr/bin/env python3
"""Explicitly reopen one episode whose host probe recalculation failed after authoring.

No model is invoked and no author file changes. The failed synthesis state and the
probe outputs are renamed beside the originals with a numbered suffix, never deleted.
A rerun of the same training job reuses design/finished.json and repeats only the
host recalculation and commit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

PROBE_OUTPUTS = ('probes-recalculated.pending', 'probes.json', 'probes')


def recover(ep_dir):
    if not shutil.which('soffice'):
        raise SystemExit('soffice is still unavailable here; install LibreOffice before recovery')
    diag = Path(ep_dir).resolve() / 'diagnostics'
    state_path = diag / 'synthesis.json'
    raw = state_path.read_bytes()
    state = json.loads(raw)
    if state.get('status') != 'failed' or not ('validate_probes' in state.get('traceback', '')
                                               or 'probes.json' in state.get('error', '')):
        raise ValueError('recovery only applies to a synthesis that failed at probe recalculation or its commit')
    if not (diag / 'design/finished.json').is_file():
        raise ValueError('expected a finished author stage')
    attempt = 1 + len(list(diag.glob('probe_recovery_receipt*.json')))
    receipt_path = diag / f'probe_recovery_receipt-{attempt}.json'
    archived_state = diag / f'failed-synthesis-before-probe-recovery-{attempt}.json'
    moves = [(diag / name, diag / f'{name}.failed-{attempt}') for name in PROBE_OUTPUTS if (diag / name).exists()]
    for target in [receipt_path, archived_state, *(t for _, t in moves)]:
        if target.exists():
            raise ValueError(f'archive target exists; preserve it: {target}')
    for source, target in moves:
        source.rename(target)
    state_path.rename(archived_state)
    receipt = {'origin': 'explicit_probe_recalculation_recovery', 'attempt': attempt, 'recovered_at': time.time(),
               'reason': state.get('error'), 'failure_path': str(archived_state),
               'failure_sha256': hashlib.sha256(raw).hexdigest(),
               'archived_probe_outputs': [str(t) for _, t in moves], 'author_calls_added': 0,
               'next_step': 'rerun the same run_training job; the author stage is reused from design/finished.json'}
    with receipt_path.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-dir', required=True)
    print(json.dumps(recover(parser.parse_args().episode_dir), ensure_ascii=False, indent=2))
