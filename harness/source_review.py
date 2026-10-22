"""Review a frozen source once, independently of candidate authoring.

The full source inventory is a cache shared by candidates, not an author prompt.
Each reviewer still receives one sample or one claim with its dependencies.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import time

from candidate_jobs import validate_job
from common import DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency
from host_validation import execution_slot, locked
from stage_audit import (_load_observations, encoded, read_json, sha, write_json)


@contextmanager
def model_slot(root, max_concurrent=DEFAULT_AUDIT_CONCURRENCY):
    """Share the configured model cap across authors and ready reviewers."""
    cap = validate_audit_concurrency(max_concurrent)
    with execution_slot({'slot_root': str(Path(root) / '.training-model-slots'),
                         'config': {'score': {'n_concurrent': cap}}}):
        yield


def _freeze(job, cfg):
    origin, attribution, inputs = validate_job(job, cfg, original=job['original'])
    observed = _load_observations(origin, origin / 'records', attribution)
    # Candidate IDs and goal/difficulty are deliberately absent: facts are the
    # same for all candidates that share this exact original analysis.
    identity = {'schema_version': 'source-review-input-v1',
                'analysis_identity': job['analysis_identity'],
                'judge': {key: cfg['judge'][key] for key in ('agent', 'model', 'budget_min')}}
    root = origin / 'source-reviews' / hashlib.sha256(encoded(identity)).hexdigest()
    with locked(root / 'prepare.lock'):
        path = root / 'evidence' / 'manifest.json'
        ready = root / 'prepared.json'
        if ready.is_file():
            receipt = read_json(ready)
            if receipt.get('identity') != identity:
                raise ValueError('source review identity changed')
            for name, digest in receipt['files'].items():
                p = Path(name)
                if p.is_symlink() or not p.is_file() or sha(p) != digest:
                    raise ValueError('frozen source review evidence changed')
            return root, path
        destination = root / 'evidence'
        if destination.exists():
            raise ValueError('source evidence freeze was interrupted; retained for explicit recovery')
        destination.mkdir(parents=True)
        expected = {'claims': [c['id'] for c in attribution['payload']['claims']],
                    'observations': observed['observation_ids']}
        manifest = {'snapshots': {name: inputs[name] for name in ('original', 'rollouts')},
                    'commits': {}, 'expected_units': expected,
                    'audit_scope': {'mode': 'full_source'}, 'missing_evidence': []}
        commits = [('attribution', attribution)] + [
            (f"observation-{c['payload']['sample_id']:02d}", c)
            for c in observed['observation_commits']]
        for name, commit in commits:
            payload_path = destination / (name + '.json')
            write_json(payload_path, commit['payload'])
            refs = []
            for ref in commit['references']:
                target = destination / 'references' / ref['sha256']
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copy2(origin / 'records' / ref['path'], target)
                if sha(target) != ref['sha256']:
                    raise ValueError('source reference changed while freezing')
                refs.append({**ref, 'snapshot_path': str(target)})
            manifest['commits'][name] = {'commit_id': commit['commit_id'],
                                        'payload_path': str(payload_path), 'references': refs}
        write_json(path, manifest)
        write_json(destination / 'navigation.json', {
            'schema_version': 'source-review-navigation-v1', 'manifest': str(path),
            'roots': {name: inputs[name]['snapshot_path'] for name in ('original', 'rollouts')},
            'expected_units': expected,
            'attribution_payload': manifest['commits']['attribution']['payload_path'],
            'note': 'Each worker input gives its exact full trace and relevant observation IDs.'})
        write_json(ready, {'identity': identity, 'files': {
            str(p): sha(p) for p in destination.rglob('*') if p.is_file()}})
        return root, path


def goal_units(manifest, goal_id):
    attribution = read_json(manifest['commits']['attribution']['payload_path'])
    goal = next(g for g in attribution['training_goals'] if g['id'] == goal_id)
    claims = [c for c in attribution['claims'] if c['id'] in goal['claim_ids']]
    observations = {oid for claim in claims for key in ('observation_ids', 'counterevidence_ids')
                    for oid in claim[key]}
    return {'claims': [c['id'] for c in claims],
            'observations': [oid for oid in manifest['expected_units']['observations'] if oid in observations]}


class SourceReviewHandle:
    """Join only the source units needed by the authored candidate."""
    def __init__(self, future, hb, cfg, job, *, max_concurrent, slot,
                 recovery_id=None, recovery_from=None):
        self.future, self.hb, self.cfg, self.job = future, hb, cfg, job
        self.max_concurrent, self.slot = max_concurrent, slot
        self.recovery_id, self.recovery_from = recovery_id, recovery_from

    def result_for(self, manifest):
        # Unit-level locks/cache join only shared dependencies. Do not wait on a
        # whole earlier request when this candidate needs only part of it.
        return review_source(self.hb, self.cfg, self.job, max_concurrent=self.max_concurrent,
                             slot=self.slot, required_units=manifest['expected_units'],
                             recovery_id=self.recovery_id, recovery_from=self.recovery_from)


def review_source(hb, cfg, job, *, max_concurrent=DEFAULT_AUDIT_CONCURRENCY,
                  slot=None, required_units=None, recovery_id=None, recovery_from=None):
    """Serialize only duplicate requests for one source, never unrelated sources."""
    from parallel_audit_review import recovery_directory, run_source_review
    max_concurrent = validate_audit_concurrency(max_concurrent)
    root, manifest_path = _freeze(job, cfg)
    status_root, _ = recovery_directory(root, recovery_id, recovery_from)
    manifest = read_json(manifest_path)
    required = goal_units(manifest, job['goal_id']) if required_units is None else {
        group: required_units.get(group, []) for group in ('claims', 'observations')}
    ro = [manifest_path.parent] + [Path(v['snapshot_path']) for v in manifest['snapshots'].values()]
    subset = hashlib.sha256(encoded(required)).hexdigest()
    with locked(status_root / 'subset-locks' / (subset + '.lock')):
        status_path = status_root / 'statuses' / (subset + '.json')
        started = time.time()
        write_json(status_path, {'status': 'running', 'started_at': started,
                                 'manifest_path': str(manifest_path), 'required_units': required})
        try:
            result = run_source_review(hb, cfg, manifest_path, root, ro,
                                       max_concurrent=max_concurrent, slot=slot, required_units=required,
                                       recovery_id=recovery_id, recovery_from=recovery_from)
        except BaseException as exc:
            write_json(status_path, {'status': 'failed', 'started_at': started,
                                     'finished_at': time.time(), 'error': f'{type(exc).__name__}: {exc}'})
            raise
        write_json(status_path, {'status': 'completed', 'started_at': started,
                                 'finished_at': time.time(), 'result': result})
        return result
