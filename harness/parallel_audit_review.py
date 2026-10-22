"""Independent audit workers with explicit evidence dependencies and no retries.

A checks atomic observations per sample. B checks one claim after its own A
dependencies complete. C/D checks the candidate and declares probe expectations.
Only the host merges fields and executes probes; no worker invents another's work.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from common import ROOT, render_task, DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency
import mounts
from stage_records import VERDICTS, _evidence, validate_audit
from stage_records import verify_snapshot

PROMPTS = ROOT / 'agents' / 'stage-audit'


class ParallelReviewError(RuntimeError):
    def __init__(self, message, *, terminals=None, review_files=None):
        super().__init__(message)
        self.terminals = terminals or {}
        self.review_files = review_files or {}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode() + b'\n'


def _once(path, value):
    raw = _encoded(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f'immutable review artifact differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)


def _files(audit, directory):
    result = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'review evidence cannot contain symlinks: {p}')
        if p.is_file():
            result[str(p)] = audit.sha(p)
    return result


def _entries(entries, identifiers, verdict_key, work):
    if not isinstance(entries, list):
        raise ValueError('review verdicts must be an array')
    ids = [v.get('id') if isinstance(v, dict) else None for v in entries]
    if len(ids) != len(set(ids)) or set(ids) != set(identifiers):
        raise ValueError('review verdict IDs do not exactly match assigned units')
    for entry in entries:
        if set(entry) != {'id', verdict_key, 'reason', 'evidence'}:
            raise ValueError('worker verdict fields differ from its assigned role')
        if entry[verdict_key] not in VERDICTS:
            raise ValueError('invalid review verdict')
        if not isinstance(entry['reason'], str) or not entry['reason'].strip():
            raise ValueError('review verdict requires a reason')
        _evidence(entry['evidence'], 'review.evidence', work)


def _validate(audit, kind, payload, plan, assigned, work):
    if not isinstance(payload, dict):
        raise ValueError('review output must be an object')
    if kind in ('observations', 'claims'):
        if set(payload) != {kind}:
            raise ValueError('worker cannot supply another review role')
        _entries(payload[kind], assigned[kind], 'verdict', work)
    else:
        if set(payload) != {'task_checks', 'mappings', 'probes'}:
            raise ValueError('task reviewer cannot supply upstream verdicts')
        _entries(payload['task_checks'], assigned['task_checks'], 'verdict', work)
        _entries(payload['mappings'], assigned['mappings'], 'implementation_verdict', work)
        # Probe shape and advance expectations use the existing audit contract.
        if not isinstance(plan, dict) or not isinstance(plan.get('probes'), list):
            raise ValueError('task review requires an advance probe plan')
        probe_ids = [p['id'] for p in plan['probes']]
        projected = {'claims': [], 'mappings': [],
                     'task_checks': payload['task_checks'], 'probes': payload['probes']}
        validate_audit(projected, {'claims': [], 'mappings': [],
                       'task_checks': assigned['task_checks'], 'probes': probe_ids}, base_dir=work)
        audit._validate_plan(plan, payload, work)


def _load_unit(audit, directory):
    request = audit.read_json(directory / 'input.json')
    terminal = audit.read_json(directory / 'terminal.json')
    if terminal.get('request_sha256') != hashlib.sha256(_encoded(request)).hexdigest():
        raise ValueError('review input does not match terminal identity')
    if terminal.get('status') != 'completed':
        raise ValueError('prior failed review retained without automatic retry')
    recovery_path = directory / 'recovery.json'
    if recovery_path.exists() or 'recovery_sha256' in terminal:
        if not recovery_path.is_file() or audit.sha(recovery_path) != terminal.get('recovery_sha256'):
            raise ValueError('review recovery provenance changed')
        for path, digest in audit.read_json(recovery_path)['prior_files'].items():
            if not Path(path).is_file() or audit.sha(path) != digest:
                raise ValueError('prior failed review evidence changed after recovery')
    model = terminal.get('model_terminal', {})
    if model.get('status') != 'completed' or model.get('trial_count') != 1:
        raise ValueError('review has no normal model terminal')
    work = directory / 'work'
    if {**_files(audit, work), **_files(audit, directory / 'job')} != terminal.get('output_hashes'):
        raise ValueError('completed review evidence changed')
    for dependency in request['source'].get('observation_reviews', {}).values():
        if audit.sha(dependency['raw_path']) != dependency['raw_sha256'] or audit.sha(dependency['terminal_path']) != dependency['terminal_sha256']:
            raise ValueError('upstream observation review changed')
    raw_path, plan_path = work / 'review.json', work / 'probe_plan.json'
    raw = audit.read_json(raw_path)
    plan = audit.read_json(plan_path) if request['kind'] == 'task' else None
    _validate(audit, request['kind'], raw, plan, request['assigned'], work)
    return {'raw': raw, 'plan': plan, 'raw_path': str(raw_path),
            'plan_path': str(plan_path) if plan is not None else None,
            'work': str(work), 'terminal': terminal, 'input': request,
            'terminal_path': str(directory / 'terminal.json')}


def _semantic_request(request):
    if request['kind'] == 'task':
        return {k: v for k, v in request.items()
                if k not in ('implementation_sha256', 'prompt_sha256')}
    source = request['source']
    return {'kind': request['kind'], 'key': request['key'], 'assigned': request['assigned'],
            'judge': request['judge'], 'source_identity': source['source_identity'],
            'observations': source.get('observations'), 'claim': source.get('claim'),
            'training_goals': source.get('training_goals')}


def recovery_directory(run, recovery_id, recovery_from):
    """An explicit recovery gets a separate namespace; never overwrite a trial."""
    if recovery_id is None and recovery_from is None:
        return Path(run), None
    if (not isinstance(recovery_id, str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', recovery_id) is None
            or recovery_from is None):
        raise ValueError('recovery requires a safe recovery_id and an explicit recovery_from directory')
    origin = Path(recovery_from).resolve(strict=True)
    if not origin.is_dir():
        raise ValueError('recovery_from must be a directory')
    destination = Path(run) / 'recoveries' / recovery_id
    if destination.resolve() == origin:
        raise ValueError('recovery must use a new directory')
    return destination, origin


def _prior_unit(audit, directory, request):
    """Validate a reusable completion or bind the exact failed attempt being retried."""
    prior = directory / 'input.json'
    if not prior.is_file():
        blocked = directory / 'blocked.json'
        if blocked.is_file():
            return None, {str(blocked): audit.sha(blocked)}
        if directory.exists():
            raise ValueError('prior interrupted review has no input identity')
        return None, {}
    old = audit.read_json(prior)
    def identity(value):
        result = _semantic_request(value)
        if value['kind'] == 'task':
            # Relocated frozen manifests describe the same explicitly recovered
            # candidate. Keep candidate/source roots, payloads and assignments.
            result = {k: v for k, v in result.items() if k != 'manifest_sha256'}
            result['source'] = {k: v for k, v in result['source'].items()
                                if k not in ('manifest_path', 'navigation_path')}
        return result
    if identity(old) != identity(request):
        raise ValueError('recovery review belongs to different inputs or judge configuration')
    terminal_path = directory / 'terminal.json'
    if not terminal_path.is_file():
        raise ValueError('prior interrupted review has no terminal; explicit terminal recovery is required')
    terminal = audit.read_json(terminal_path)
    if terminal.get('request_sha256') != hashlib.sha256(_encoded(old)).hexdigest():
        raise ValueError('prior review input does not match terminal identity')
    if terminal.get('status') == 'completed':
        return _load_unit(audit, directory), {}
    if terminal.get('status') != 'failed':
        raise ValueError('only a failed terminal can be explicitly retried')
    if {**_files(audit, directory / 'work'), **_files(audit, directory / 'job')} != terminal.get('output_hashes'):
        raise ValueError('prior failed review evidence changed')
    return None, {str(prior): audit.sha(prior), str(terminal_path): audit.sha(terminal_path),
                  **terminal['output_hashes']}


def _run_unit(audit, hb, cfg, manifest_path, run, ro, key, kind, source,
              assigned, slot, reuse_from=None):
    queued_at = time.time()
    lock_dir = run / 'unit-locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / (key + '.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return _run_unit_body(audit, hb, cfg, manifest_path, run, ro, key, kind,
                                  source, assigned, slot, queued_at, reuse_from)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run_unit_body(audit, hb, cfg, manifest_path, run, ro, key, kind, source,
                   assigned, slot, queued_at, reuse_from=None):
    directory = run / 'reviews' / key
    request = {'schema_version': 'parallel-audit-unit-v1', 'key': key, 'kind': kind,
               'source': source, 'assigned': assigned,
               'manifest_sha256': audit.sha(manifest_path),
               'prompt_sha256': audit.sha(PROMPTS / f'parallel-{kind}.md'),
               'judge': {k: cfg['judge'][k] for k in ('agent', 'model', 'budget_min')},
               'implementation_sha256': audit.sha(__file__)}
    request_digest = hashlib.sha256(_encoded(request)).hexdigest()
    terminal_path = directory / 'terminal.json'
    work = directory / 'work'
    raw_path, plan_path = work / 'review.json', work / 'probe_plan.json'
    if terminal_path.is_file():
        cached = _load_unit(audit, directory)
        if _semantic_request(cached['input']) != _semantic_request(request):
            raise ValueError(f'{key}: completed/failed attempt belongs to different inputs')
        return cached
    if directory.exists():
        raise ValueError(f'{key}: interrupted review retained without automatic retry')
    recovered_files = None
    if reuse_from is not None:
        reused, recovered_files = _prior_unit(audit, Path(reuse_from) / 'reviews' / key, request)
        if reused is not None:
            return reused
    work.mkdir(parents=True)
    _once(directory / 'input.json', request)
    if recovered_files is not None:
        _once(directory / 'recovery.json', {'recovery_from': str(reuse_from),
                                           'prior_files': recovered_files})
    terminal = {'key': key, 'kind': kind, 'request_sha256': request_digest,
                'status': 'queued', 'queued_at': queued_at, 'acquired_at': None,
                'failure': None}
    if recovered_files is not None:
        terminal['recovery_sha256'] = audit.sha(directory / 'recovery.json')
    audit.write_json(directory / 'started.json', terminal)
    try:
        rendered = render_task(PROMPTS, directory / 'task', {
            '__MANIFEST__': str(manifest_path),
            '__NAVIGATION__': str(manifest_path.parent / 'navigation.json'),
            '__UNIT_INPUT__': str(directory / 'input.json'), '__WORK__': str(work),
            '__RESULT__': str(raw_path), '__PROBE_PLAN__': str(plan_path)})
        (rendered / 'instruction.md').write_text((rendered / f'parallel-{kind}.md').read_text())
        readonly = list(ro) + [directory / 'input.json']
        readonly.extend(Path(p) for p in source.get('dependency_work_dirs', []))
        readonly.extend(Path(dep['terminal_path']).parent
                        for dep in source.get('observation_reviews', {}).values())
        if source.get('score_trace_path'):
            readonly.append(Path(source['score_trace_path']))
        mount_list = json.loads(mounts.build(hb.bin, ro=readonly, rw=[work], with_docker=False))
        binary = os.environ.get('ADB_CC_BINARY')
        if binary:
            mount_list.append({'type': 'bind', 'source': str(Path(binary).resolve(strict=True)),
                               'target': '/usr/local/bin/claude', 'read_only': True})
        judge = cfg['judge']
        seconds = float(judge['budget_min']) * 60
        with slot() if slot is not None else nullcontext():
            terminal['acquired_at'] = time.time()
            terminal['status'] = 'running'
            audit.write_json(directory / 'started.json', terminal)
            rc = hb.agent_task(rendered, judge['agent'], judge['model'], directory / 'job',
                seconds * 1.5 + 900, json.dumps(mount_list),
                agent_env=[f'ADB_RESULT={raw_path}', f'ADB_DEADLINE_EPOCH={int(time.time()+seconds)}'],
                agent_timeout_mult=seconds / 2400.0)
        actual = audit.reviewer_terminal(directory / 'job', rc)
        terminal['model_terminal'] = actual
        if rc != 0 or actual['trial_count'] != 1 or actual['status'] != 'completed':
            exception = actual.get('exception_info') or {}
            # The full model/CLI failure remains in this terminal and its hashed
            # result file. Do not copy entire transcripts into every parent error.
            raise ValueError(f"{key}: model did not complete normally; status={actual['status']}; "
                             f"exception_type={exception.get('exception_type')}; "
                             f"result_path={actual.get('result_path')}")
        raw = audit.read_json(raw_path)
        plan = audit.read_json(plan_path) if kind == 'task' else None
        _validate(audit, kind, raw, plan, assigned, work)
        terminal['status'] = 'completed'
    except BaseException as exc:
        terminal['status'] = 'failed'
        terminal['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        terminal['finished_at'] = time.time()
        try:
            terminal['output_hashes'] = {**_files(audit, work), **_files(audit, directory / 'job')}
        except Exception as integrity_error:
            terminal.update(status='failed', output_hashes={},
                            failure=f'{type(integrity_error).__name__}: {integrity_error}')
            _once(terminal_path, terminal)
            raise
        _once(terminal_path, terminal)
    return {'raw': raw, 'plan': plan, 'raw_path': str(raw_path),
            'plan_path': str(plan_path) if plan is not None else None,
            'work': str(work), 'terminal': terminal, 'input': request,
            'terminal_path': str(terminal_path)}


def source_identity(manifest):
    """Content identity deliberately excludes candidate and relocated commit IDs."""
    import stage_audit as audit
    payloads = {name: audit.read_json(commit['payload_path'])
                for name, commit in manifest['commits'].items()
                if name == 'attribution' or name.startswith('observation-')}
    return {'original_tree': manifest['snapshots']['original']['tree_sha256'],
            'rollouts_tree': manifest['snapshots']['rollouts']['tree_sha256'],
            'payload_sha256': {name: hashlib.sha256(_encoded(value)).hexdigest()
                               for name, value in sorted(payloads.items())}}


def _context(audit, manifest_path):
    manifest = audit.read_json(manifest_path)
    roots = {k: v['snapshot_path'] for k, v in manifest['snapshots'].items()}
    attribution = audit.read_json(manifest['commits']['attribution']['payload_path'])
    shared = {'source_identity': source_identity(manifest),
              'original_root': roots['original'], 'rollouts_root': roots['rollouts'],
              'manifest_path': str(manifest_path),
              'navigation_path': str(manifest_path.parent / 'navigation.json')}
    return manifest, roots, attribution, shared


def _score_trace(manifest_path, name):
    """Host replay file for A/B; None when the analysis predates score traces."""
    path = Path(manifest_path).parents[3] / 'score-trace' / name
    return str(path) if path.is_file() else None


def _source_units(audit, manifest, attribution, shared):
    expected = manifest['expected_units']
    required = set(expected.get('observations', []))
    owners, units = {}, {}
    for name, commit in manifest['commits'].items():
        if not name.startswith('observation-'):
            continue
        payload = audit.read_json(commit['payload_path'])
        selected = [o for o in payload['observations'] if o['id'] in required]
        if not selected:
            continue
        key = f"sample-{payload['sample_id']:02d}"
        for observation in selected:
            if observation['id'] in owners:
                raise ValueError('observation appears in multiple worker inputs')
            owners[observation['id']] = key
        sources = [{'name': Path(r['original_path']).name, 'path': r['snapshot_path'],
                    'original_path': r['original_path'], 'sha256': r['sha256']}
                   for r in commit['references']
                   if Path(r['original_path']).name in ('attempt.md', 'evidence_index.json')]
        source = {**shared, 'sample_id': payload['sample_id'], 'observations': selected,
                  'payload_path': commit['payload_path'], 'source_files': sources,
                  'fallback_full_trace': str(Path(shared['rollouts_root']) / f"attempt-{payload['sample_id']}.md"),
                  'score_trace_path': _score_trace(shared['manifest_path'], f"sample-{payload['sample_id']:02d}.json")}
        units[key] = ('observations', source, {'observations': [o['id'] for o in selected]})
    if set(owners) != required:
        raise ValueError('required observations do not resolve to frozen workers')
    units = {key: units[key] for key in dict.fromkeys(
        owners[oid] for oid in expected.get('observations', []))}
    claims = {c['id']: c for c in attribution['claims']}
    dependencies = {}
    for cid in expected['claims']:
        claim = claims[cid]
        refs = set(claim.get('observation_ids', [])) | set(claim.get('counterevidence_ids', []))
        if not refs.issubset(owners):
            raise ValueError(f'{cid}: supporting/counterevidence review is missing')
        dependencies[cid] = {owners[oid] for oid in refs}
    return units, claims, dependencies


def _required(manifest, required_units):
    inventory = manifest['expected_units']
    if required_units is None:
        return {group: list(inventory.get(group, [])) for group in ('claims', 'observations')}
    if not isinstance(required_units, dict) or not set(required_units).issubset({'claims', 'observations'}):
        raise ValueError('required_units must contain only claims and observations')
    result = {}
    for group in ('claims', 'observations'):
        ids = required_units.get(group, [])
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('required units must be unique ID arrays')
        if not set(ids).issubset(inventory.get(group, [])):
            raise ValueError(f'unknown required {group}')
        result[group] = [i for i in inventory.get(group, []) if i in ids]
    return result


def _schedule(audit, hb, cfg, manifest_path, run, ro, manifest, attribution,
              shared, max_concurrent, slot, task_unit=None, on_task_ready=None,
              required_units=None, reuse_from=None):
    max_concurrent = validate_audit_concurrency(max_concurrent)
    units, claims, deps = _source_units(audit, manifest, attribution, shared)
    wanted = _required(manifest, required_units)
    pending_claims = list(wanted['claims'])
    samples = {key for key, (_, _, assigned) in units.items()
               if set(assigned['observations']) & set(wanted['observations'])}
    samples.update(key for cid in pending_claims for key in deps[cid])
    pending_atoms = [key for key in units if key in samples]
    pending_task = task_unit is not None
    completed, failures, terminals, review_files = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        while futures or pending_atoms or pending_claims or pending_task:
            # A ready B takes the next free slot before unrelated queued A work.
            for cid in list(pending_claims):
                if deps[cid] & failures.keys():
                    key = 'claim-' + cid
                    failures[key] = 'required observation review failed'
                    blocked = {'key': key, 'status': 'blocked', 'dependencies': sorted(deps[cid])}
                    _once(run / 'reviews' / key / 'blocked.json', blocked)
                    terminals[key] = blocked
                    pending_claims.remove(cid)
            while len(futures) < max_concurrent:
                ready_claim = next((cid for cid in pending_claims if deps[cid].issubset(completed)), None)
                if pending_task:
                    key, unit = 'task', task_unit
                    pending_task = False
                elif ready_claim is not None:
                    cid = ready_claim
                    upstream = {k: {'raw_path': completed[k]['raw_path'],
                                    'raw_sha256': audit.sha(completed[k]['raw_path']),
                                    'terminal_path': completed[k]['terminal_path'],
                                    'terminal_sha256': audit.sha(completed[k]['terminal_path'])}
                                for k in sorted(deps[cid])}
                    source = {**shared, 'claim': claims[cid],
                              'training_goals': [g for g in attribution.get('training_goals', []) if cid in g['claim_ids']],
                              'observation_reviews': upstream,
                              'dependency_work_dirs': [completed[k]['work'] for k in sorted(deps[cid])],
                              'score_trace_path': _score_trace(shared['manifest_path'], 'score_trace_summary.json')}
                    key, unit = 'claim-' + cid, ('claims', source, {'claims': [cid]})
                    pending_claims.remove(cid)
                elif pending_atoms:
                    key = pending_atoms.pop(0)
                    unit = units[key]
                else:
                    break
                future = executor.submit(_run_unit, audit, hb, cfg, manifest_path, run,
                                         ro, key, *unit, slot, **({'reuse_from': reuse_from} if reuse_from is not None else {}))
                futures[future] = key
            if not futures:
                break
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                key = futures.pop(future)
                try:
                    result = future.result()
                    completed[key] = result
                    terminals[key] = result['terminal']
                    review_files[key] = {'raw_path': result['raw_path'], 'plan_path': result['plan_path'],
                                         'terminal_path': result['terminal_path']}
                    if key == 'task' and on_task_ready is not None:
                        on_task_ready(copy.deepcopy(result['raw']), copy.deepcopy(result['plan']), Path(result['work']))
                        completed[key] = _load_unit(audit, Path(result['terminal_path']).parent)
                except Exception as exc:
                    failures[key] = f'{type(exc).__name__}: {exc}'
                    p = run / 'reviews' / key / 'terminal.json'
                    if p.is_file():
                        terminals[key] = audit.read_json(p)
        if failures or pending_claims:
            # Failed attempts remain in per-unit terminal records. This summary is
            # derived and can change as independent completed workers are resumed.
            audit.write_json(run / 'reviews' / 'failed-assembly.json',
                             {'failures': failures, 'pending_claims': pending_claims})
            raise ParallelReviewError('parallel audit incomplete: ' + json.dumps(failures),
                                       terminals=terminals, review_files=review_files)
    return completed


def _source_payload(completed, expected):
    raw, sources = {'observations': [], 'claims': []}, {'observations': {}, 'claims': {}}
    for key in sorted(completed):
        result = completed[key]
        if key == 'task':
            continue
        group = 'observations' if result['input']['kind'] == 'observations' else 'claims'
        for entry in result['raw'][group]:
            if entry['id'] in sources[group]:
                raise ValueError('duplicate source-review judgment')
            raw[group].append(entry)
            sources[group][entry['id']] = {'path': result['raw_path'], 'id': entry['id']}
    for group in ('observations', 'claims'):
        order = expected.get(group, [])
        entries = {item['id']: item for item in raw[group]}
        if set(entries) != set(order):
            raise ValueError('completed source units differ from declared source inventory')
        raw[group] = [entries[identifier] for identifier in order]
        sources[group] = {identifier: sources[group][identifier] for identifier in order}
    return raw, sources


def _source_receipt_dir(run, wanted, subset):
    if not subset:
        return run / 'reviews'
    digest = hashlib.sha256(_encoded(wanted)).hexdigest()
    return run / 'reviews' / 'subsets' / digest


def _record_source(audit, run, manifest_path, completed, required_units=None):
    selected = {k: v for k, v in completed.items() if k != 'task'}
    manifest = audit.read_json(manifest_path)
    wanted = _required(manifest, required_units)
    found = {'claims': set(), 'observations': set()}
    for result in selected.values():
        for group in found:
            found[group].update(item['id'] for item in result['raw'].get(group, []))
    reviewed = {group: [i for i in manifest['expected_units'].get(group, []) if i in found[group]]
                for group in found}
    raw, fields = _source_payload(selected, reviewed)
    location = _source_receipt_dir(run, wanted, required_units is not None)
    raw_path = location / 'source-review.json'
    _once(raw_path, raw)
    record = {'schema_version': 'source-review-v1',
              'source_identity': source_identity(manifest),
              'requested_units': wanted, 'reviewed_units': reviewed,
              'manifest_path': str(manifest_path), 'manifest_sha256': audit.sha(manifest_path),
              'raw_path': str(raw_path), 'raw_sha256': audit.sha(raw_path),
              'terminals': {k: v['terminal'] for k, v in selected.items()},
              'review_files': {k: {'raw_path': v['raw_path'], 'terminal_path': v['terminal_path']}
                               for k, v in selected.items()}, 'field_sources': fields}
    receipt = location / 'source-review-receipt.json'
    _once(receipt, record)
    return {**record, 'receipt_path': str(receipt), 'receipt_sha256': audit.sha(receipt)}


def verify_source_review(record, manifest, *, cfg=None):
    """Re-read all source receipts; a Future's temporary dictionary is no evidence."""
    import stage_audit as audit
    path = Path(record['receipt_path'])
    if audit.sha(path) != record['receipt_sha256']:
        raise ValueError('source-review receipt changed')
    bound = audit.read_json(path)
    if bound.get('source_identity') != source_identity(manifest):
        raise ValueError('source review belongs to different original evidence or attribution')
    if audit.sha(bound['manifest_path']) != bound['manifest_sha256']:
        raise ValueError('source-review frozen manifest changed')
    source_manifest = audit.read_json(bound['manifest_path'])
    if source_identity(source_manifest) != bound['source_identity']:
        raise ValueError('source-review inputs changed')
    evidence_files = {str(path), bound['raw_path'], bound['manifest_path']}
    for name in ('original', 'rollouts'):
        snapshot = verify_snapshot(source_manifest['snapshots'][name])
        evidence_files.update(str(Path(snapshot['snapshot_path']) / e['path'])
                              for e in snapshot['entries'] if e['type'] == 'file')
    for name, commit in source_manifest['commits'].items():
        if name != 'attribution' and not name.startswith('observation-'):
            continue
        evidence_files.add(commit['payload_path'])
        for reference in commit['references']:
            if audit.sha(reference['snapshot_path']) != reference['sha256']:
                raise ValueError('source-review frozen raw evidence changed')
            evidence_files.add(reference['snapshot_path'])
    completed = {}
    for key, files in bound['review_files'].items():
        result = _load_unit(audit, Path(files['terminal_path']).parent)
        if result['terminal'] != bound['terminals'][key] or result['raw_path'] != files['raw_path']:
            raise ValueError('source-review terminal does not match its receipt')
        if result['input']['source']['source_identity'] != bound['source_identity']:
            raise ValueError('source worker belongs to different evidence')
        if cfg is not None and result['input']['judge'] != {k: cfg['judge'][k] for k in ('agent', 'model', 'budget_min')}:
            raise ValueError('source worker belongs to a different judge configuration')
        completed[key] = result
        directory = Path(files['terminal_path']).parent
        evidence_files.update(str(directory / name) for name in ('input.json', 'terminal.json', 'started.json'))
        if (directory / 'recovery.json').is_file():
            evidence_files.add(str(directory / 'recovery.json'))
        evidence_files.update(result['terminal']['output_hashes'])
    reviewed = bound.get('reviewed_units', source_manifest['expected_units'])
    for group in ('claims', 'observations'):
        if not set(reviewed.get(group, [])).issubset(source_manifest['expected_units'].get(group, [])):
            raise ValueError('source receipt contains units outside its full source inventory')
    raw, fields = _source_payload(completed, reviewed)
    if (audit.sha(bound['raw_path']) != bound['raw_sha256']
            or audit.read_json(bound['raw_path']) != raw or fields != bound['field_sources']):
        raise ValueError('source-review assembly differs from independently completed workers')
    for group in ('claims', 'observations'):
        if not set(manifest['expected_units'].get(group, [])).issubset(fields[group]):
            raise ValueError(f'source review lacks required {group}; extend the source review explicitly')
    return {**bound, 'receipt_path': str(path), 'receipt_sha256': record['receipt_sha256'],
            'raw': raw, 'evidence_files': sorted(evidence_files)}


def run_source_review(hb, cfg, manifest_path, run_dir, ro, *,
                      max_concurrent=DEFAULT_AUDIT_CONCURRENCY, slot=None, required_units=None,
                      recovery_id=None, recovery_from=None):
    import stage_audit as audit
    max_concurrent = validate_audit_concurrency(max_concurrent)
    manifest_path, run = Path(manifest_path).resolve(), Path(run_dir).resolve()
    run, reuse_from = recovery_directory(run, recovery_id, recovery_from)
    manifest, roots, attribution, shared = _context(audit, manifest_path)
    wanted = _required(manifest, required_units)
    if reuse_from is not None:
        _once(run / 'recovery.json', {'recovery_id': recovery_id,
                                     'recovery_from': str(reuse_from),
                                     'manifest_sha256': audit.sha(manifest_path)})
    cache_roots = [run]
    if reuse_from is None:
        cache_roots.extend(sorted((run / 'recoveries').glob('*')))
    for cache_root in cache_roots:
        receipt = _source_receipt_dir(cache_root, wanted, required_units is not None) / 'source-review-receipt.json'
        if receipt.is_file():
            validation_manifest = {**manifest, 'expected_units': wanted}
            verified = verify_source_review({'receipt_path': str(receipt), 'receipt_sha256': audit.sha(receipt)},
                                            validation_manifest, cfg=cfg)
            return {k: v for k, v in verified.items() if k not in ('raw', 'evidence_files')}
    completed = _schedule(audit, hb, cfg, manifest_path, run, ro, manifest, attribution,
                          shared, max_concurrent, slot, required_units=required_units, reuse_from=reuse_from)
    return _record_source(audit, run, manifest_path, completed, required_units)


def run_parallel_review(hb, cfg, manifest_path, run_dir, ro, *, max_concurrent=DEFAULT_AUDIT_CONCURRENCY,
                        on_task_ready=None, slot=None, source_review=None, reuse_task_from=None):
    """C/D may finish before the reusable source Future; callback never grades.

    slot is a shared zero-argument contextmanager factory. Callbacks must be
    idempotent when cached C/D is reused, and may launch asynchronous host probes.
    """
    import stage_audit as audit
    max_concurrent = validate_audit_concurrency(max_concurrent)
    manifest_path, run = Path(manifest_path).resolve(), Path(run_dir).resolve()
    manifest, roots, attribution, shared = _context(audit, manifest_path)
    expected = manifest['expected_units']
    design = audit.read_json(manifest['commits']['design']['payload_path'])
    mappings = {m['id']: m for m in design['mappings']}
    claim_map = {c['id']: c for c in attribution['claims']}
    task_source = {**shared, 'candidate_root': roots['candidate'],
                   'candidate_files': [e['path'] for e in manifest['snapshots']['candidate']['entries'] if e['type'] == 'file'],
                   'mappings': [mappings[mid] for mid in expected['mappings']],
                   'claim_statements_as_unverified_design_context': [claim_map[cid] for cid in expected['claims']]}
    task_unit = ('task', task_source, {g: expected[g] for g in ('task_checks', 'mappings', 'probes')})
    if source_review is None:
        completed = _schedule(audit, hb, cfg, manifest_path, run, ro, manifest,
                              attribution, shared, max_concurrent, slot, task_unit, on_task_ready)
        source = _record_source(audit, run, manifest_path, completed)
        task = completed['task']
    else:
        # Source A/B already runs in its own shared-slot-controlled executor.
        # C/D can immediately progress without waiting for that Future.
        try:
            task = _run_unit(audit, hb, cfg, manifest_path, run, ro, 'task', *task_unit, slot,
                             **({'reuse_from': reuse_task_from} if reuse_task_from is not None else {}))
            if on_task_ready is not None:
                on_task_ready(copy.deepcopy(task['raw']), copy.deepcopy(task['plan']), Path(task['work']))
                task = _load_unit(audit, Path(task['terminal_path']).parent)
            if hasattr(source_review, 'result_for'):
                source = source_review.result_for(manifest)
            else:
                source = source_review.result() if hasattr(source_review, 'result') else source_review
        except Exception as exc:
            terminals = dict(getattr(exc, 'terminals', {}))
            files = dict(getattr(exc, 'review_files', {}))
            path = run / 'reviews/task/terminal.json'
            if path.exists():
                terminals['task'] = audit.read_json(path)
                files['task'] = {'terminal_path': str(path),
                                 'raw_path': str(path.parent / 'work/review.json'),
                                 'plan_path': str(path.parent / 'work/probe_plan.json')}
            raise ParallelReviewError(f'candidate/source review incomplete: {exc}',
                                       terminals=terminals, review_files=files) from exc
    verified = verify_source_review(source, manifest, cfg=cfg)
    claims = {c['id']: c for c in verified['raw']['claims']}
    observations = {o['id']: o for o in verified['raw']['observations']}
    implementation = {m['id']: m for m in task['raw']['mappings']}
    raw = {'claims': [claims[cid] for cid in expected['claims']], 'mappings': [],
           'task_checks': task['raw']['task_checks'], 'probes': task['raw']['probes']}
    fields = {'claims': {cid: verified['field_sources']['claims'][cid] for cid in expected['claims']},
              'mappings': {}, 'task_checks': {}, 'probes': {}}
    if 'observations' in expected:
        raw['observations'] = [observations[oid] for oid in expected['observations']]
        fields['observations'] = {oid: verified['field_sources']['observations'][oid] for oid in expected['observations']}
    for mid in expected['mappings']:
        cid = mappings[mid]['claim_id']
        up, impl = claims[cid], implementation[mid]
        raw['mappings'].append({'id': mid, 'upstream_verdict': up['verdict'],
            'implementation_verdict': impl['implementation_verdict'],
            'reason': 'Upstream claim review: ' + up['reason'] + '\nImplementation review: ' + impl['reason'],
            'evidence': up['evidence'] + impl['evidence']})
        fields['mappings'][mid] = {'upstream_verdict': fields['claims'][cid],
                                  'implementation_verdict': {'path': task['raw_path'], 'id': mid},
                                  'reason_and_evidence': 'ordered concatenation; no new judgment'}
    for group in ('task_checks', 'probes'):
        fields[group] = {e['id']: {'path': task['raw_path'], 'id': e['id']} for e in raw[group]}
    validate_audit(raw, {**expected, 'probes': [p['id'] for p in task['plan']['probes']]}, base_dir=run)
    raw_path = run / 'reviews' / 'assembled-review.json'
    _once(raw_path, raw)
    _once(run / 'reviews' / 'field-sources.json', fields)
    return {'raw_path': str(raw_path), 'plan_path': task['plan_path'],
            'terminals': {**verified['terminals'], 'task': task['terminal']},
            'review_files': {**verified['review_files'], 'task': {'raw_path': task['raw_path'],
                              'plan_path': task['plan_path'], 'terminal_path': task['terminal_path']}},
            'field_sources': fields, 'source_identity': verified['source_identity'],
            'evidence_files': verified['evidence_files'],
            'source_receipt': {'receipt_path': verified['receipt_path'], 'receipt_sha256': verified['receipt_sha256']}}
