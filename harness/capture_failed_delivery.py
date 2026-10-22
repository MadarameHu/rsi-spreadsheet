#!/usr/bin/env python3
"""Explicitly capture an existing failed run for diagnostics, never promote or rerun.

No author or candidate files are changed. The original failed synthesis envelope
is archived verbatim; its status/error stay unchanged when adding capture pointers.
This command does not invoke any model. Missing/ambiguous artifacts fail explicitly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from common import discover_tasks
from stage_records import _load_json, read_stage_commit, snapshot_tree, validate_design, write_stage_commit


def capture(ep_dir):
    ep = Path(ep_dir).resolve()
    diag = ep / 'diagnostics'
    state_path = diag / 'synthesis.json'
    raw_failure = state_path.read_bytes()
    state = _load_json(raw_failure)
    if state.get('status') != 'failed':
        raise ValueError('explicit capture requires an existing failed synthesis state')
    if state.get('capture_receipt') or (diag / 'capture_receipt.json').exists():
        raise ValueError('already captured; preserve existing capture, do not overwrite')
    a_id = _load_json((diag / 'attribution_commit.json').read_bytes())['commit_id']
    a = read_stage_commit(diag / 'records', a_id)
    raw_design = (ep / 'work/design.json').read_bytes()
    design = _load_json(raw_design)
    validate_design(design, a['payload'])
    candidates = discover_tasks(ep / 'work/out')
    if len(candidates) != 1:
        raise ValueError(f'capture requires exactly one authored candidate; found {len(candidates)}')
    manifest = snapshot_tree(candidates[0], diag / 'candidate')
    if raw_design != (ep / 'work/design.json').read_bytes():
        raise ValueError('design changed during capture; incomplete capture retained')
    manifest_path = diag / 'candidate_manifest.json'
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    design_path = diag / 'design.json'
    with design_path.open('xb') as f: f.write(raw_design)
    failure_archive = diag / 'failed-synthesis-before-capture.json'
    with failure_archive.open('xb') as f: f.write(raw_failure)
    receipt = {'origin':'explicit_failed_delivery_capture', 'captured_at':time.time(),
               'synthesis_status':'failed', 'author_receipt_missing':True,
               'author_calls_added':0, 'candidate_modified':False,
               'failure_path':str(failure_archive),
               'failure_sha256':hashlib.sha256(raw_failure).hexdigest(),
               'candidate_sha256':manifest['tree_sha256'], 'attribution_commit':a_id,
               'reason':'Inspect existing authored bytes after controller failure; no success status or author return code inferred.'}
    receipt_path = diag / 'capture_receipt.json'
    with receipt_path.open('x') as f: json.dump(receipt,f,indent=2);f.write('\n')
    d = write_stage_commit(diag / 'records','design',design,claims=a['payload'],
                           parent_ids=[a_id], references=[design_path,manifest_path,receipt_path,failure_archive],
                           metadata=dict(a['metadata'],candidate_sha256=manifest['tree_sha256'],
                                         capture_origin=receipt['origin'],source_synthesis_status='failed',
                                         author_receipt_missing=True))
    state.update(attribution_commit=a_id, design_commit=d['commit_id'],
                 stage_commit_dir=str(diag/'records'),candidate_manifest=str(manifest_path),
                 input_manifest=str(diag/'inputs.json'), capture_receipt=str(receipt_path))
    # Keep the original failure status and full error. Its original bytes are
    # already archived and hash-bound, not overwritten or retrospectively fixed.
    temp = diag / 'captured-synthesis-state.json'
    with temp.open('x') as f: json.dump(state,f,ensure_ascii=False,indent=2);f.write('\n')
    temp.replace(state_path)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(capture(args.episode_dir),indent=2))
