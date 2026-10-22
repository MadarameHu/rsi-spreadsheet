"""Per-task error class library: defect claims accumulated across analysis runs.

The customer model is one fixed sampling distribution, so each 16-attempt batch
sees a different subset of its error classes. Every committed aggregate's defect
claims are appended here (never overwritten) so later aggregation and design can
see classes a single batch happened not to sample. The library is reference
input only: claims still cite the current batch's observations as evidence.

Enabled with diagnostics.parallel_attribution.error_library = true. The file
lives in the customer model's prep directory, so models never share a library.
"""
from __future__ import annotations
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

SCHEMA = 'error-class-library-v1'
NAME = 'error-classes.json'
NONE_TEXT = '(none: no error class library for this task yet)'


def library_path(cfg, benchmark, task):
    """The library file, or None when the feature is off."""
    if not cfg.get('diagnostics', {}).get('parallel_attribution', {}).get('error_library'):
        return None
    from run_rollout import prep_dir
    return prep_dir(cfg, benchmark, task) / NAME


def prompt_text(path):
    return str(path) if path is not None and Path(path).is_file() else NONE_TEXT


@contextmanager
def _locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = (json.loads(path.read_text()) if path.is_file()
                    else {'schema_version': SCHEMA, 'classes': [], 'ingested_commits': []})
            if data.get('schema_version') != SCHEMA:
                raise ValueError(f'unexpected error class library schema in {path}')
            yield data
            tmp = path.with_name(path.name + '.tmp')
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _origin(commit):
    """A candidate job re-commits its source aggregate; the class sources stay on the source commit."""
    return (commit.get('metadata') or {}).get('source_attribution_commit') or commit['commit_id']


def _strongest(values):
    return 'replay_verified' if 'replay_verified' in values else (
        'hypothesis_only' if 'hypothesis_only' in values else 'not_declared')


def ingest(path, commit, *, episode):
    """Append a committed aggregate's defect claims; return how many new classes were added.

    Class identity is the aggregator's judgment, declared per claim as
    `library_class` (an existing id) or null (a new class); wording varies between
    batches, so text is never used to match. An unknown id is rejected.
    """
    if path is None:
        return 0
    commit_id = _origin(commit)
    with _locked(Path(path)) as data:
        if commit_id in data['ingested_commits']:
            return 0
        known = {entry['id']: entry for entry in data['classes']}
        added = 0
        for claim in commit['payload']['claims']:
            if claim.get('kind') != 'defect':
                continue
            strength = claim.get('evidence_strength', 'not_declared')
            source = {'attribution_commit': commit_id, 'claim_id': claim['id'], 'episode': str(episode),
                      'evidence_strength': strength, 'support_sample_ids': claim.get('support_sample_ids')}
            ref = claim.get('library_class')
            if ref is not None:
                if ref not in known:
                    raise ValueError(f'claim {claim["id"]} references unknown library class {ref}')
                entry = known[ref]
            else:
                entry = known[f'{commit_id[:8]}-{claim["id"]}'] = {
                    'id': f'{commit_id[:8]}-{claim["id"]}', 'statement': claim['statement'],
                    'conditions': claim.get('conditions', []), 'sources': [], 'targeted_by': []}
                data['classes'].append(entry)
                added += 1
            entry['sources'].append(source)
            entry['evidence_strength'] = _strongest([s['evidence_strength'] for s in entry['sources']])
        data['ingested_commits'].append(commit_id)
        return added


def mark_targeted(path, commit, claim_ids, *, episode, design_commit):
    """Record which library classes a committed design selected as targets."""
    if path is None:
        return
    commit_id = _origin(commit)
    with _locked(Path(path)) as data:
        mark = {'episode': str(episode), 'design_commit': design_commit}
        for entry in data['classes']:
            selected = any(source['claim_id'] in claim_ids and source['attribution_commit'] == commit_id
                           for source in entry['sources'])
            if selected and mark not in entry['targeted_by']:
                entry['targeted_by'].append(mark)
