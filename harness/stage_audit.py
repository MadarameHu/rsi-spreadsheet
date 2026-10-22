#!/usr/bin/env python3
"""Independent diagnostic audit and post-review verifier probes.

One configured judge run reads complete frozen evidence. Its advance probe plan
is frozen before the host runs any probe through Harbor. No token limits,
truncation, schema repair, inference of missing verdicts, or automatic retries.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency,
                    task_pass_threshold, task_timeout_sec)
from stage_records import (read_stage_commit, snapshot_tree, verify_snapshot,
                           validate_audit, write_stage_commit)

PROMPT = ROOT / 'agents' / 'stage-audit'
REVIEW_MODE = 'parallel'
# Receipts written before review_mode existed came from the removed single reviewer.
LEGACY_REVIEW_MODE = 'serial'
TASK_CHECKS = ['spec', 'ground_truth', 'reference']
REQUIRED_PROBES = ['correct_solution', 'incorrect_solution']


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value) + b'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_fingerprint(path):
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f'not a regular evidence directory: {path}')
    entries = []
    for child in sorted(path.rglob('*')):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise ValueError(f'unsupported evidence entry: {child}')
        entries.append([child.relative_to(path).as_posix(),
                        sha(child) if child.is_file() else 'directory'])
    return hashlib.sha256(encoded(entries)).hexdigest()


def _manifest(value):
    return verify_snapshot(value)


def _verify_raw_output(commit, *, expected_path=None):
    """Bind compact model bytes separately from the deterministic stage payload."""
    metadata = commit.get('metadata', {})
    raw_path = metadata.get('raw_output_path')
    raw_sha = metadata.get('raw_output_sha256')
    if raw_path is None and raw_sha is None and expected_path is None:
        return  # Legacy commits predate separate raw/derived records.
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise ValueError('raw model output requires both path and sha256 metadata')
    path = Path(raw_path)
    if expected_path is not None and path.resolve() != Path(expected_path).resolve():
        raise ValueError('worker raw output path differs from committed metadata')
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError('raw model output must be an existing regular absolute file')
    if sha(path) != raw_sha:
        raise ValueError('raw model output differs from committed hash')
    if not any(Path(ref['original_path']).resolve() == path.resolve()
               and ref['sha256'] == raw_sha for ref in commit['references']):
        raise ValueError('raw model output is not bound to immutable stage evidence')


def load_inputs(ep_dir):
    diagnostics = Path(ep_dir) / 'diagnostics'
    synthesis = read_json(diagnostics / 'synthesis.json')
    records = Path(synthesis['stage_commit_dir'])
    attribution = read_stage_commit(records, synthesis['attribution_commit'])
    design = read_stage_commit(records, synthesis['design_commit'])
    if attribution['stage'] != 'attribution' or design['stage'] != 'design':
        raise ValueError('wrong upstream commit stage')
    if attribution['commit_id'] not in design['parent_ids']:
        raise ValueError('design commit does not reference the frozen attribution')
    for name, commit in [('attribution', attribution), ('design', design)]:
        _verify_raw_output(commit)
        if read_json(diagnostics / f'{name}.json') != commit['payload']:
            raise ValueError(f'{name}.json does not match its immutable commit')
    source_path = diagnostics / 'inputs.json'
    source = read_json(source_path)
    if not any(Path(ref['original_path']).resolve() == source_path.resolve() and ref['sha256'] == sha(source_path) for ref in attribution['references']):
        raise ValueError('input manifest is not bound to attribution commit')
    original = _manifest(source['original'])
    rollouts = _manifest(source['rollouts'])
    candidate_path = synthesis.get('candidate_manifest')
    candidate = _manifest(candidate_path) if candidate_path and Path(candidate_path).is_file() else None
    if candidate is not None:
        if design.get('metadata', {}).get('candidate_sha256') != candidate['tree_sha256']:
            raise ValueError('candidate snapshot is not bound to design commit')
        if not any(Path(ref['original_path']).resolve() == Path(candidate_path).resolve() and ref['sha256'] == sha(candidate_path) for ref in design['references']):
            raise ValueError('candidate manifest is not archived in design commit')
    inputs = {'attribution': attribution, 'design': design, 'records_dir': records,
              'original': original, 'rollouts': rollouts, 'candidate': candidate,
              'synthesis': synthesis}
    if attribution['payload'].get('schema_version') == 'parallel-attribution-v1':
        inputs.update(_load_observations(diagnostics, records, attribution))
    return inputs


def _load_observations(diagnostics, records, attribution):
    # Resolve every worker from the immutable aggregate inventory.
    from parallel_records import validate_aggregate, validate_observations
    path = diagnostics / 'observations' / 'index.json'
    index = read_json(path)
    if index.get('schema_version') != 'observation-index-v1' or index.get('status') != 'completed':
        raise ValueError('parallel observation inventory is not completed')
    if not any(Path(ref['original_path']).resolve() == path.resolve() and ref['sha256'] == sha(path)
               for ref in attribution['references']):
        raise ValueError('observation index is not bound to aggregate attribution commit')
    order, workers = index.get('sample_order'), index.get('workers')
    if (not isinstance(order, list) or not order or any(type(i) is not int or i < 0 for i in order)
            or len(set(order)) != len(order) or not isinstance(workers, list)
            or [w.get('sample_id') for w in workers] != order):
        raise ValueError('observation workers do not exactly match declared sample order')
    reuse_root = None
    reuse_sources = set()
    if 'reuse_observations_from' in index:
        origin = index['reuse_observations_from']
        if not isinstance(origin, str) or not Path(origin).is_absolute():
            raise ValueError('reused observations require an absolute source diagnostics directory')
        reuse_root = Path(origin).resolve(strict=True)
        if not reuse_root.is_dir() or reuse_root == diagnostics.resolve():
            raise ValueError('reused observations require a different source diagnostics directory')
        sources = index.get('reuse_observation_sources', [str(reuse_root)])
        if (not isinstance(sources, list) or any(not isinstance(p, str) or not Path(p).is_absolute()
                for p in sources)):
            raise ValueError('reused observation sources require absolute diagnostics directories')
        reuse_sources = {Path(p).resolve(strict=True) for p in sources}
    commits, observations = [], []
    for worker in workers:
        if worker.get('status') != 'completed':
            raise ValueError('incomplete observation worker cannot be silently omitted')
        commit = read_stage_commit(records, worker['commit_id'])
        payload = commit['payload']
        if commit['stage'] != 'observations' or payload.get('sample_id') != worker['sample_id']:
            raise ValueError('observation worker commit stage or sample identity mismatch')
        validate_observations(payload)
        _verify_raw_output(commit, expected_path=worker.get('raw_output_path'))
        output = Path(worker['output_path'])
        directory = Path(worker['directory'])
        job = Path(worker['job_path'])
        observation_root = path.parent.resolve()
        if reuse_root is not None and 'reused_from' in worker:
            declared = worker.get('reused_from')
            if (not isinstance(declared, str) or not Path(declared).is_absolute()
                    or Path(declared).resolve() not in reuse_sources
                    or worker.get('reused_commit_id') != commit['commit_id']):
                raise ValueError('reused worker source or commit identity differs from declared reuse')
            observation_root = Path(declared).resolve() / 'observations'
            # The current records store already verifies every archived byte.
            # Reuse additionally requires the retained live source to remain exact.
            for ref in commit['references']:
                live = Path(ref['original_path'])
                if live.is_symlink() or not live.is_file() or sha(live) != ref['sha256']:
                    raise ValueError('reused worker live evidence differs from immutable commit')
        elif 'reused_from' in worker or 'reused_commit_id' in worker:
            raise ValueError('worker reuse metadata requires a declared reuse source')
        elif reuse_root is not None and not index.get('allow_partial_reuse', False):
            raise ValueError('reused worker source or commit identity differs from declared reuse')
        if (not directory.resolve().is_relative_to(observation_root)
                or not output.resolve().is_relative_to(directory.resolve())
                or job.resolve() != (directory / 'job').resolve() or not job.is_dir()):
            raise ValueError('worker evidence must include its full raw job under observations')
        job_manifest_path = directory / 'job_manifest.json'
        job_manifest = read_json(job_manifest_path)
        if not any(Path(ref['original_path']).resolve() == job_manifest_path.resolve()
                   and ref['sha256'] == sha(job_manifest_path) for ref in commit['references']):
            raise ValueError('worker raw job manifest is not bound to observation commit')
        from stage_records import _tree_entries
        entries = _tree_entries(job)
        if (job_manifest.get('entries') != entries
                or job_manifest.get('tree_sha256') != hashlib.sha256(encoded(entries)).hexdigest()):
            raise ValueError('worker raw job differs from immutable observation evidence')
        raw_path = worker.get('raw_output_path')
        if raw_path is not None and not Path(raw_path).resolve().is_relative_to(directory.resolve()):
            raise ValueError('worker raw model output must remain under its observation directory')
        if read_json(output) != payload:
            raise ValueError('observation output differs from committed worker payload')
        ids = [o['id'] for o in payload['observations']]
        if worker.get('observation_ids') != ids:
            raise ValueError('worker observation IDs differ from immutable output')
        commits.append(commit)
        observations.extend(payload['observations'])
    if attribution['parent_ids'] != [c['commit_id'] for c in commits]:
        raise ValueError('aggregate parents differ from declared worker commits')
    ids = [o['id'] for o in observations]
    if len(set(ids)) != len(ids) or index.get('observation_ids') != ids:
        raise ValueError('observation inventory does not exactly cover all worker observations')
    validate_aggregate(attribution['payload'], observations)
    return {'observation_commits': commits, 'observation_ids': ids,
            'observation_index': str(path), 'observation_index_sha256': sha(path)}


def resolve_scope(inputs, scope='full'):
    """Derive review obligations from frozen design; never from model verdicts.

    The complete input inventory remains available in both modes. A selected
    review includes counterevidence as well as supporting observations, in the
    original inventory order. Required task checks and probes are never reduced.
    """
    if scope not in ('full', 'selected_design'):
        raise ValueError(f'unsupported audit scope: {scope}')
    expected = {'claims': [c['id'] for c in inputs['attribution']['payload']['claims']],
            'mappings': [m['id'] for m in inputs['design']['payload']['mappings']],
            'task_checks': list(TASK_CHECKS), 'probes': list(REQUIRED_PROBES)}
    if 'observation_ids' in inputs:
        expected['observations'] = list(inputs['observation_ids'])
    source = {group: list(ids) for group, ids in expected.items()}
    if scope == 'selected_design':
        mappings = [m for m in inputs['design']['payload']['mappings']
                    if m.get('role') != 'exclude']
        claim_ids = {m['claim_id'] for m in mappings}
        if not claim_ids.issubset(source['claims']):
            raise ValueError('selected design references an unknown claim')
        expected['mappings'] = [m['id'] for m in mappings]
        expected['claims'] = [i for i in source['claims'] if i in claim_ids]
        if 'observations' in source:
            observed = set()
            for claim in inputs['attribution']['payload']['claims']:
                if claim['id'] in claim_ids:
                    observed.update(claim['observation_ids'])
                    observed.update(claim['counterevidence_ids'])
            if not observed.issubset(source['observations']):
                raise ValueError('selected claim references an unknown observation')
            expected['observations'] = [i for i in source['observations'] if i in observed]
    return {'schema_version': 'audit-scope-v1', 'mode': scope,
            'source_units': source, 'expected_units': expected,
            'unreviewed_ids': {group: [i for i in ids if i not in expected[group]]
                               for group, ids in source.items()}}


def expected_units(inputs, scope='full'):
    return resolve_scope(inputs, scope)['expected_units']


def _validate_plan(plan, raw, work):
    if not isinstance(plan, dict) or not isinstance(plan.get('probes'), list):
        raise ValueError('probe plan requires a probes array')
    seen = set()
    for probe in plan['probes']:
        if not isinstance(probe, dict) or not isinstance(probe.get('id'), str):
            raise ValueError('probe plan entry requires an id')
        pid = probe['id']
        if not pid or pid in seen or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in pid):
            raise ValueError(f'invalid or duplicate probe id: {pid}')
        seen.add(pid)
        if 'expected_valid' not in probe or (probe['expected_valid'] is not None and type(probe['expected_valid']) is not bool):
            raise ValueError(f'{pid}: expected_valid must be bool or null')
        if not isinstance(probe.get('reason'), str) or not probe['reason'].strip():
            raise ValueError(f'{pid}: advance independent reason required')
        if not isinstance(probe.get('evidence'), list):
            raise ValueError(f'{pid}: evidence list required')
        for evidence in probe['evidence']:
            if (not isinstance(evidence, dict) or not isinstance(evidence.get('locator'), str)
                    or not evidence['locator'].strip() or not Path(evidence.get('path', '')).is_file()):
                raise ValueError(f'{pid}: invalid independent evidence reference')
        workbook = probe.get('workbook')
        if workbook is not None:
            path = Path(workbook)
            if not path.is_absolute() or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(work.resolve()):
                raise ValueError(f'{pid}: workbook must be a regular file under the audit work directory')
        elif probe['expected_valid'] is not None:
            raise ValueError(f'{pid}: a missing workbook cannot have a decided expectation')
    if not set(REQUIRED_PROBES).issubset(seen):
        raise ValueError('plan must include correct_solution and incorrect_solution, even if missing')
    audit_probes = {p['id']: p for p in raw['probes']}
    if set(audit_probes) != seen:
        raise ValueError('audit and plan probe ids differ')
    for probe in plan['probes']:
        entry = audit_probes[probe['id']]
        if entry['expected_valid'] is not probe['expected_valid']:
            raise ValueError(f'{probe["id"]}: audit and advance plan expectations differ')
        if entry['actual_accept'] is not None or entry['execution_status'] != 'missing':
            raise ValueError('reviewer cannot supply actual probe acceptance before host execution')
    return plan['probes']


def parse_probe_run(job, rc, threshold):
    """Use the actual trial verifier result; infrastructure never becomes rejection."""
    trials = []
    for path in sorted(Path(job).rglob('result.json')):
        value = read_json(path)
        if 'verifier_result' in value or 'exception_info' in value:
            trials.append((path, value))
    if len(trials) != 1:
        return None, 'infra_error', {'error': f'expected exactly one trial result, found {len(trials)}', 'harbor_rc': rc}
    path, value = trials[0]
    rewards = (value.get('verifier_result') or {}).get('rewards') or {}
    reward = rewards.get('reward')
    detail = {'result_path': str(path), 'harbor_rc': rc, 'reward': reward,
              'pass_threshold': threshold, 'exception_info': value.get('exception_info')}
    if rc != 0 or value.get('exception_info'):
        return None, 'infra_error', detail
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        detail['error'] = 'missing or invalid verifier reward'
        return None, 'infra_error', detail
    return bool(reward >= threshold), 'completed', detail


def freeze_probe_inputs(probes, frozen_plan_path, destination):
    """Freeze EVERY workbook before starting any verifier; retain source mapping."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    frozen = {'plan_path': str(frozen_plan_path), 'plan_sha256': sha(frozen_plan_path),
              'frozen_before_any_execution_at': time.time(), 'probes': {}}
    for probe in probes:
        original = probe.get('workbook')
        item = {'id': probe['id'], 'original_workbook': original,
                'snapshot_path': None, 'sha256': None, 'bytes': None}
        if original is not None:
            raw = Path(original).read_bytes()
            target = destination / (probe['id'] + '.xlsx')
            target.write_bytes(raw)
            item.update(snapshot_path=str(target), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        frozen['probes'][probe['id']] = item
    # A source modified while another workbook was copied invalidates the freeze.
    for item in frozen['probes'].values():
        if item['original_workbook'] is not None and sha(item['original_workbook']) != item['sha256']:
            raise ValueError('probe source changed while freezing the complete workbook set')
    write_json(destination / 'manifest.json', frozen)
    return frozen


def _output_hashes(run):
    return {str(p): sha(p) for p in run.rglob('*')
            if p.is_file() and p != run / 'audit.json'}


def verify_cached_envelope(previous, inputs, fingerprint, expected):
    """Only immutable audit-commit metadata can authorize a completed cache hit."""
    if previous.get('status') != 'completed' or not previous.get('audit_commit'):
        raise ValueError('previous audit has no verified completed commit; retained without automatic retry')
    commit = read_stage_commit(inputs['records_dir'], previous['audit_commit'])
    anchor = commit.get('metadata', {}).get('audit_envelope')
    if not isinstance(anchor, dict) or anchor != {k:v for k,v in previous.items() if k != 'audit_commit'}:
        raise ValueError('cached audit envelope does not match immutable commit metadata')
    if (commit.get('stage') != 'audit' or commit.get('payload') != anchor.get('payload')
            or anchor.get('evidence_fingerprint') != fingerprint
            or commit.get('parent_ids') != [inputs['attribution']['commit_id'], inputs['design']['commit_id']]):
        raise ValueError('cached audit payload, parent commits, or evidence version mismatch')
    review_mode = anchor.get('review_mode', LEGACY_REVIEW_MODE)
    if review_mode != commit['metadata'].get('review_mode', LEGACY_REVIEW_MODE):
        raise ValueError('cached audit review mode differs from its immutable metadata')
    descriptor = anchor.get('audit_scope')
    # Old commits did not carry a scope descriptor and always required full review.
    mode = descriptor.get('mode') if isinstance(descriptor, dict) else 'full'
    resolved = resolve_scope(inputs, mode)
    if descriptor is not None:
        if descriptor != resolved or commit['metadata'].get('audit_scope') != resolved:
            raise ValueError('cached audit scope differs from the frozen design evidence closure')
    elif 'audit_scope' in commit['metadata']:
        raise ValueError('cached audit scope metadata differs from its envelope')
    if expected != resolved['expected_units']:
        raise ValueError('requested audit units differ from the verified scope')
    units = anchor.get('expected_units')
    if not isinstance(units, dict):
        raise ValueError('cached audit has no committed expected-unit schema')
    if set(units) != set(expected):
        raise ValueError('cached audit unit groups changed')
    for group in expected:
        if group != 'probes' and units.get(group) != expected[group]:
            raise ValueError('cached audit upstream units changed')
    if not set(REQUIRED_PROBES).issubset(units.get('probes', [])):
        raise ValueError('cached audit lacks required probes')
    validate_audit(commit['payload'], units, base_dir=Path(anchor['work_dir']) / 'work')
    for path, digest in anchor['output_hashes'].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError(f'cached audit evidence changed: {path}')
    recovery = anchor.get('provenance', {}).get('recovery')
    if recovery is not None:
        origin = Path(recovery['recovery_from'])
        for name, digest in [('input_fingerprint.json', recovery['input_sha256']),
                             ('audit.json', recovery['receipt_sha256'])]:
            if not (origin / name).is_file() or sha(origin / name) != digest:
                raise ValueError('cached audit recovery provenance changed')
    if review_mode == REVIEW_MODE and anchor.get('provenance', {}).get('source_receipt'):
        import parallel_audit_review
        run = Path(anchor['work_dir'])
        parallel_audit_review.verify_source_review(anchor['provenance']['source_receipt'],
                                                   read_json(run / 'evidence/manifest.json'))
        # Check exact raw work/job inventories too, including added files.
        task_terminal = anchor.get('provenance', {}).get('review_files', {}).get('task', {}).get('terminal_path')
        task_directory = Path(task_terminal).parent if task_terminal else run / 'reviews/task'
        parallel_audit_review._load_unit(sys.modules[__name__], task_directory)
    return previous


def run_probe(hb, cfg, benchmark, candidate, plan, destination, frozen_input, frozen_plan_sha):
    """Inject a predeclared workbook in a task copy, preserving original verifier."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    planned = dict(plan)
    expected_path = destination / 'expectation.json'
    workbook = frozen_input['snapshot_path']
    planned['workbook_sha256'] = frozen_input['sha256']
    planned['frozen_workbook'] = dict(frozen_input)
    planned['frozen_plan_sha256'] = frozen_plan_sha
    write_json(expected_path, planned)
    expected_sha = sha(expected_path)
    receipt = {'id': plan['id'], 'expected_valid': plan['expected_valid'],
               'actual_accept': None, 'execution_status': 'missing',
               'reason': plan['reason'], 'evidence': list(plan['evidence']) +
               [{'path': str(expected_path), 'locator': 'advance independent expectation'}]}
    if workbook is None:
        write_json(destination / 'execution.json', {'execution_status': 'missing', 'reason': 'no constructed workbook'})
        return receipt
    try:
        if sha(workbook) != frozen_input['sha256']:
            raise ValueError('frozen workbook no longer matches the pre-execution plan')
        task = destination / 'task'
        copied = snapshot_tree(candidate, task)
        config = tomllib.loads((task / 'task.toml').read_text())
        target = plan.get('target_path')
        if target not in config.get('artifacts', []) or not isinstance(target, str) or not Path(target).is_absolute() or Path(target).suffix != '.xlsx':
            raise ValueError('probe target_path must be a declared absolute .xlsx artifact')
        # This is an oracle artifact injector in a COPY, never the candidate oracle.
        solution = task / 'solution'
        solution.mkdir(exist_ok=True)
        injected = solution / 'stage-audit-workbook.xlsx'
        if injected.exists():
            raise ValueError('candidate already has the reserved probe injector filename')
        shutil.copy2(workbook, injected)
        if sha(injected) != planned['workbook_sha256']:
            raise ValueError('workbook changed after independent expectation freeze')
        import shlex
        (solution / 'solve.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncp /solution/stage-audit-workbook.xlsx ' + shlex.quote(target) + '\n')
        original_checks = {entry['path']: entry.get('sha256') for entry in copied['entries']
                           if entry['type'] == 'file' and not entry['path'].startswith('solution/')}
        before = tree_fingerprint(candidate)
        job = destination / 'job'
        command = ['-p', str(task), '-a', 'oracle', '--n-concurrent', '1',
                   '--agent-setup-timeout-multiplier', str(hb.setup_mult)]
        write_json(destination / 'command.json', {'harbor_args': command, 'candidate_tree': before,
                                                 'verifier_unchanged': True})
        rc = hb.run(command, job, task_timeout_sec(cfg, benchmark))
        # Verify every original non-solution byte after execution too.
        for relative, digest in original_checks.items():
            if sha(task / relative) != digest:
                raise ValueError(f'probe altered original non-solution file: {relative}')
        if tree_fingerprint(candidate) != before or sha(expected_path) != expected_sha:
            raise ValueError('candidate or advance expectation changed during execution')
        accept, status, details = parse_probe_run(job, rc, task_pass_threshold(candidate))
        receipt.update(actual_accept=accept, execution_status=status)
    except Exception as exc:
        details = {'error': f'{type(exc).__name__}: {exc}'}
        receipt.update(actual_accept=None, execution_status='infra_error')
    evidence_path = destination / 'execution.json'
    write_json(evidence_path, details)
    receipt['evidence'].append({'path': str(evidence_path), 'locator': 'full execution result and reward provenance'})
    if details.get('result_path'):
        receipt['evidence'].append({'path': details['result_path'], 'locator': 'verifier_result / exception_info'})
    return receipt


def _navigation(manifest, destination):
    """Give direct file locations without summarizing or removing any evidence."""
    roots = {name: value['snapshot_path'] for name, value in manifest['snapshots'].items()}
    files = {name: [entry['path'] for entry in manifest['snapshots'][name]['entries']
                    if entry['type'] == 'file']
             for name in ('candidate', 'original', 'rollouts')}
    required = set(manifest['expected_units'].get('observations', []))
    workers = []
    for name, commit in manifest['commits'].items():
        if not name.startswith('observation-'):
            continue
        payload = read_json(commit['payload_path'])
        identifiers = [item['id'] for item in payload['observations'] if item['id'] in required]
        sources = [{'name': Path(ref['original_path']).name,
                    'path': Path(ref['snapshot_path']).relative_to(destination).as_posix()}
                   for ref in commit['references']
                   if Path(ref['original_path']).name in ('attempt.md', 'evidence_index.json')]
        workers.append({'sample_id': payload['sample_id'], 'required_ids': identifiers,
                        'payload': Path(commit['payload_path']).relative_to(destination).as_posix(),
                        'trace_and_event_index': sources})
    return {'schema_version': 'audit-navigation-v1', 'evidence_root': str(destination),
            'manifest': 'manifest.json', 'scope': manifest['audit_scope']['mode'],
            'expected_units': manifest['expected_units'], 'roots': roots, 'files': files,
            'payloads': {name: Path(commit['payload_path']).relative_to(destination).as_posix()
                         for name, commit in manifest['commits'].items()
                         if not name.startswith('observation-')},
            'workers': workers,
            'path_rule': 'files are relative to roots[group]; payloads are relative to evidence_root',
            'reference_lookup': 'manifest.commits[name].references maps original_path to frozen snapshot_path'}


def _freeze_evidence(inputs, extras, destination, *, scope='full'):
    destination.mkdir(parents=True, exist_ok=False)
    descriptor = resolve_scope(inputs, scope)
    manifest = {'snapshots': {}, 'commits': {}, 'missing_evidence': [],
                'expected_units': descriptor['expected_units'], 'audit_scope': descriptor}
    for name in ('original', 'rollouts', 'candidate'):
        source = inputs[name]
        manifest['snapshots'][name] = source
    for name, path in extras.items():
        if path.is_dir():
            manifest['snapshots'][name] = snapshot_tree(path, destination / name)
        else:
            manifest['missing_evidence'].append(str(path))
    commits = [(name, inputs[name]) for name in ('attribution', 'design')]
    commits.extend((f"observation-{c['payload']['sample_id']:02d}", c)
                   for c in inputs.get('observation_commits', []))
    for name, commit in commits:
        path = destination / f'{name}.json'
        write_json(path, commit['payload'])
        manifest['commits'][name] = {'commit_id': commit['commit_id'], 'payload_path': str(path), 'references': []}
        for ref in commit['references']:
            target = destination / 'references' / ref['sha256']
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(inputs['records_dir'] / ref['path'], target)
            manifest['commits'][name]['references'].append({**ref, 'snapshot_path': str(target)})
    path = destination / 'manifest.json'
    write_json(path, manifest)
    write_json(destination / 'navigation.json', _navigation(manifest, destination))
    return path, manifest


def reviewer_terminal(job, rc):
    """Keep Harbor's actual terminal exception even when no reviewer JSON exists."""
    trials = []
    for path in sorted(Path(job).rglob('result.json')):
        value = read_json(path)
        if isinstance(value, dict) and ('exception_info' in value or 'agent_execution' in value):
            trials.append((path, value))
    result = {'harbor_rc': rc, 'trial_count': len(trials), 'result_path': None,
              'result_sha256': None, 'exception_info': None, 'agent_execution': None,
              'status': 'unknown'}
    if len(trials) == 1:
        path, value = trials[0]
        result.update(result_path=str(path), result_sha256=sha(path),
                      exception_info=value.get('exception_info'),
                      agent_execution=value.get('agent_execution'),
                      status='failed' if rc != 0 or value.get('exception_info') else 'completed')
    return result


def _retain_previous_scope(latest, scope, review_mode=REVIEW_MODE):
    """Keep a prior active envelope before an explicit change of review scope."""
    if not latest.is_file():
        return
    previous = read_json(latest)
    previous_scope = previous.get('audit_scope', {}).get('mode', 'full')
    if previous_scope == scope and previous.get('review_mode', LEGACY_REVIEW_MODE) == review_mode:
        return
    # An old per-run audit.json normally already preserves this exact envelope.
    # Keep an independent history record too when it is missing or differs.
    work_dir = previous.get('work_dir')
    archived = Path(work_dir) / 'audit.json' if work_dir else None
    if archived is not None and archived.is_file() and read_json(archived) == previous:
        return
    history = latest.parent / 'audit-history' / (hashlib.sha256(encoded(previous)).hexdigest() + '.json')
    if history.exists():
        if read_json(history) != previous:
            raise ValueError('prior audit history differs from its content identity')
    else:
        write_json(history, previous)


def _failed_scope_run(diagnostics, inputs, scope, review_mode=REVIEW_MODE):
    """Do not turn changed templates or extras into silent same-scope retries."""
    commits = [inputs[name]['commit_id'] for name in ('attribution', 'design')]
    for identity_path in sorted((diagnostics / 'audit-runs').glob('*/input_fingerprint.json')):
        identity = read_json(identity_path)
        if (identity.get('commits') != commits
                or identity.get('audit_scope', {}).get('mode', 'full') != scope
                or identity.get('review_mode', LEGACY_REVIEW_MODE) != review_mode):
            continue
        receipt = identity_path.parent / 'audit.json'
        if receipt.is_file():
            previous = read_json(receipt)
            if previous.get('status') == 'failed':
                return previous
        else:
            # A prior attempt may have been interrupted before a terminal record.
            # A new scope can run, but this same scope needs explicit recovery.
            raise ValueError('an earlier audit in this scope did not finish; retained without automatic retry')
    return None



def _completed_identity_run(diagnostics, inputs, identity, expected):
    """Reuse the same evidence/config version despite implementation-only changes."""
    ignored = {'implementation', 'template', 'parallel_implementation', 'parallel_template', 'recovery'}
    current = {k: v for k, v in identity.items() if k not in ignored}
    for path in sorted((diagnostics / 'audit-runs').glob('*/input_fingerprint.json')):
        old = read_json(path)
        old.setdefault('review_mode', LEGACY_REVIEW_MODE)
        if {k: v for k, v in old.items() if k not in ignored} != current:
            continue
        receipt = path.parent / 'audit.json'
        if receipt.is_file():
            previous = read_json(receipt)
            if previous.get('status') == 'completed':
                return verify_cached_envelope(previous, inputs, previous['evidence_fingerprint'], expected)
    return None


def _audit_recovery(diagnostics, identity, recovery_id, recovery_from):
    """Bind an explicitly authorized retry to one unchanged failed audit."""
    from parallel_audit_review import recovery_directory
    _, origin = recovery_directory(diagnostics, recovery_id, recovery_from)
    if origin is None:
        return None
    if origin.parent != (diagnostics / 'audit-runs').resolve():
        raise ValueError('audit recovery must reference a prior run of this episode')
    identity_path, receipt_path = origin / 'input_fingerprint.json', origin / 'audit.json'
    previous_identity, previous = read_json(identity_path), read_json(receipt_path)
    ignored = {'implementation', 'template', 'parallel_implementation', 'parallel_template', 'recovery'}
    if ({k: v for k, v in previous_identity.items() if k not in ignored}
            != {k: v for k, v in identity.items() if k not in ignored}):
        raise ValueError('audit recovery cannot change evidence, scope, judge, or execution budgets')
    if previous.get('status') != 'failed' or Path(previous.get('work_dir', '')).resolve() != origin:
        raise ValueError('audit recovery requires a failed run with a terminal receipt')
    for path, digest in previous.get('output_hashes', {}).items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError('prior failed audit evidence changed')
    return {'recovery_id': recovery_id, 'recovery_from': str(origin),
            'input_sha256': sha(identity_path), 'receipt_sha256': sha(receipt_path)}


def _reused_probe(origin, candidate, plan, frozen_input, frozen_plan_sha):
    """Reconstruct a completed probe from its original bound execution evidence."""
    destination = Path(origin) / 'probes' / plan['id']
    execution_path, expectation_path = destination / 'execution.json', destination / 'expectation.json'
    if not execution_path.is_file() or not expectation_path.is_file():
        return None
    details, expectation = read_json(execution_path), read_json(expectation_path)
    if any(expectation.get(k) != v for k, v in plan.items()):
        raise ValueError('recovered probe expectation differs from the immutable task review')
    if (expectation.get('workbook_sha256') != frozen_input['sha256']
            or expectation.get('frozen_plan_sha256') != frozen_plan_sha):
        raise ValueError('recovered probe input or advance plan differs')
    workbook = expectation.get('frozen_workbook', {}).get('snapshot_path')
    if workbook is None:
        return None
    if sha(workbook) != frozen_input['sha256']:
        raise ValueError('recovered probe frozen workbook changed')
    if 'harbor_rc' not in details:
        return None
    accept, status, actual = parse_probe_run(destination / 'job', details['harbor_rc'], task_pass_threshold(candidate))
    if status != 'completed':
        return None
    if actual != details:
        raise ValueError('recovered probe execution differs from the original trial')
    command = read_json(destination / 'command.json')
    if command.get('candidate_tree') != tree_fingerprint(candidate) or command.get('verifier_unchanged') is not True:
        raise ValueError('recovered probe belongs to a different candidate')
    return {'id': plan['id'], 'expected_valid': plan['expected_valid'], 'actual_accept': accept,
            'execution_status': status, 'reason': plan['reason'],
            'evidence': list(plan['evidence']) + [
                {'path': str(expectation_path), 'locator': 'advance independent expectation'},
                {'path': str(execution_path), 'locator': 'full execution result and reward provenance'},
                {'path': details['result_path'], 'locator': 'verifier_result / exception_info'}]}


def _parallel_review_and_probes(hb, cfg, benchmark, ep, inputs, manifest_path, run, ro,
                               *, max_concurrent, slot, source_review, recovery_from=None):
    import host_validation
    import parallel_audit_review
    state = {}
    reusable = None
    if recovery_from is not None:
        origin = Path(recovery_from)
        terminal_path = origin / 'reviews/task/terminal.json'
        if terminal_path.is_file() and read_json(terminal_path).get('status') == 'completed':
            parallel_audit_review._load_unit(sys.modules[__name__], terminal_path.parent)
            reusable = origin
    # This is the existing reference/trainee execution pool, independent from
    # model-generation slots; derive its path before reference prepares a request.
    execution_request = {'slot_root': str(ep.parents[2] / '.validation-slots'), 'config': cfg}
    with ThreadPoolExecutor(max_workers=1) as pool:
        def execute(probes, frozen):
            results = []
            for probe in probes:
                reused = (_reused_probe(reusable, Path(inputs['candidate']['snapshot_path']), probe,
                                        frozen['probes'][probe['id']], frozen['plan_sha256'])
                          if reusable is not None else None)
                if reused is not None:
                    results.append(reused)
                    continue
                with host_validation.execution_slot(execution_request):
                    results.append(run_probe(hb, cfg, benchmark,
                        Path(inputs['candidate']['snapshot_path']), probe,
                        run / 'probes' / probe['id'], frozen['probes'][probe['id']],
                        frozen['plan_sha256']))
            return results

        def task_ready(raw, plan, work):
            probes = _validate_plan(plan, raw, work)
            if state:
                if read_json(state['plan']) != plan:
                    raise ValueError('task callback changed an already frozen probe plan')
                for item in state['frozen']['probes'].values():
                    if item['original_workbook'] is not None and sha(item['original_workbook']) != item['sha256']:
                        raise ValueError('task callback changed an already frozen workbook')
                return
            frozen_path = run / 'frozen_probe_plan.json'
            write_json(frozen_path, plan)
            frozen = freeze_probe_inputs(probes, frozen_path, run / 'probe-inputs')
            state.update(plan=frozen_path, frozen=frozen)
            # All workbooks and the whole plan are immutable before F begins.
            state['future'] = pool.submit(execute, probes, frozen)

        result = parallel_audit_review.run_parallel_review(
            hb, cfg, manifest_path, run, ro, max_concurrent=max_concurrent,
            on_task_ready=task_ready, slot=slot, source_review=source_review,
            **({'reuse_task_from': reusable} if reusable is not None else {}))
        if not state:
            raise ValueError('parallel reviewer did not provide a completed task/probe branch')
        probe_results = state['future'].result()
        if reusable is not None:
            result['evidence_files'] = sorted(set(result.get('evidence_files', [])) |
                                              set(_output_hashes(reusable)) | {str(reusable / 'audit.json')})
    return result, state['plan'], state['frozen'], probe_results


def audit_episode(hb, cfg, benchmark, original, ep_dir, *, scope='full',
                  max_concurrent=DEFAULT_AUDIT_CONCURRENCY,
                  slot=None, source_review=None, recovery_id=None, recovery_from=None):
    review_mode = REVIEW_MODE
    ep_dir = Path(ep_dir).resolve()
    diagnostics = ep_dir / 'diagnostics'
    latest = diagnostics / 'audit.json'
    envelope = {'status': 'failed', 'payload': None, 'expected_units': None, 'failures': [],
                'evidence_fingerprint': None, 'work_dir': None, 'review_mode': review_mode,
                'provenance': {'origin': 'independent_stage_audit', 'is_human_gold': False,
                               'synthesis_status': None, 'capture_receipt': None,
                               'benchmark': benchmark, 'original_task': original,
                               'judge': {k: cfg['judge'][k] for k in ('agent', 'model', 'budget_min')}}}
    run = None
    try:
        if scope not in ('full', 'selected_design'):
            raise ValueError(f'unsupported audit scope: {scope}')
        max_concurrent = validate_audit_concurrency(max_concurrent)
        _retain_previous_scope(latest, scope, review_mode)
        synthesis_path = diagnostics / 'synthesis.json'
        synthesis = read_json(synthesis_path)
        envelope['provenance']['synthesis_status'] = synthesis.get('status')
        envelope['provenance']['capture_receipt'] = synthesis.get('capture_receipt')
        if synthesis.get('status') != 'completed' and not (diagnostics / 'candidate_manifest.json').is_file():
            envelope.update(status='missing', failures=['authoring has no frozen candidate; no judgments invented'])
            write_json(latest, envelope)
            return envelope
        inputs = load_inputs(ep_dir)
        descriptor = resolve_scope(inputs, scope)
        expected = expected_units(inputs, scope)
        envelope['audit_scope'] = descriptor
        envelope['expected_units'] = expected
        extras = {'author_run': ep_dir / 'job', 'attribution_run': diagnostics / 'attribution' / 'job',
                  'design_run': diagnostics / 'design' / 'job'}
        if 'observation_commits' in inputs:
            extras['observations'] = diagnostics / 'observations'
            envelope['provenance']['attribution_mode'] = 'parallel'
            envelope['provenance']['observation_commits'] = [c['commit_id'] for c in inputs['observation_commits']]
            envelope['provenance']['observation_index_sha256'] = inputs['observation_index_sha256']
        fingerprint_data = {
            'audit_scope': descriptor, 'review_mode': review_mode,
            'commits': [inputs[name]['commit_id'] for name in ('attribution', 'design')],
            'snapshots': {name: inputs[name]['tree_sha256'] if inputs[name] else None
                          for name in ('original', 'rollouts', 'candidate')},
            'extras': {name: tree_fingerprint(path) if path.is_dir() else None for name, path in extras.items()},
            'judge': envelope['provenance']['judge'], 'template': tree_fingerprint(PROMPT),
            'implementation': sha(__file__),
            'execution_config': {'task_timeout_sec': task_timeout_sec(cfg, benchmark),
                                 'setup_mult': hb.setup_mult, 'harbor': hb.bin},
        }
        import parallel_audit_review
        fingerprint_data['parallel_implementation'] = sha(parallel_audit_review.__file__)
        fingerprint_data['parallel_template'] = tree_fingerprint(parallel_audit_review.PROMPTS)
        recovery = _audit_recovery(diagnostics, fingerprint_data, recovery_id, recovery_from)
        if recovery is not None:
            if source_review is None:
                raise ValueError('parallel audit recovery requires an explicit reusable source review')
            fingerprint_data['recovery'] = recovery
            envelope['provenance']['recovery'] = recovery
        reused = _completed_identity_run(diagnostics, inputs, fingerprint_data, expected)
        if reused is not None:
            write_json(latest, reused)
            return reused
        if recovery is None:
            failed = _failed_scope_run(diagnostics, inputs, scope, review_mode)
            if failed is not None:
                write_json(latest, failed)
                return failed
        fingerprint = hashlib.sha256(encoded(fingerprint_data)).hexdigest()
        envelope['evidence_fingerprint'] = fingerprint
        run = diagnostics / 'audit-runs' / fingerprint
        envelope['work_dir'] = str(run)
        cached = run / 'audit.json'
        if cached.is_file():
            previous = read_json(cached)
            if inputs['candidate'] is None:
                envelope.update(status='missing', failures=['no frozen candidate manifest; no judgments invented'])
                write_json(latest, envelope)
                return envelope
            verified = verify_cached_envelope(previous, inputs, fingerprint, expected)
            write_json(latest, verified)
            return verified
        if run.exists():
            raise ValueError('an earlier audit with these inputs did not finish; retained without automatic retry')
        run.mkdir(parents=True)
        write_json(run / 'input_fingerprint.json', fingerprint_data)
        if inputs['candidate'] is None:
            envelope.update(status='missing', failures=['no frozen candidate manifest; no judgments invented'])
        else:
            manifest_path, manifest = _freeze_evidence(inputs, extras, run / 'evidence', scope=scope)
            # Ensure live self-test/trajectory trees did not change while freezing.
            for name, path in extras.items():
                if path.is_dir() and tree_fingerprint(path) != fingerprint_data['extras'][name]:
                    raise ValueError(f'{name} changed during audit evidence freeze')
            ro = [run / 'evidence'] + [Path(inputs[name]['snapshot_path']) for name in ('original', 'rollouts', 'candidate')]
            result, frozen_plan_path, frozen_inputs, probe_results = _parallel_review_and_probes(
                hb, cfg, benchmark, ep_dir, inputs, manifest_path, run, ro,
                max_concurrent=max_concurrent, slot=slot, source_review=source_review,
                recovery_from=recovery['recovery_from'] if recovery is not None else None)
            raw_path, plan_path = Path(result['raw_path']), Path(result['plan_path'])
            envelope['provenance']['reviewer_terminals'] = result['terminals']
            envelope['provenance']['field_sources'] = result['field_sources']
            envelope['provenance']['source_identity'] = result.get('source_identity')
            envelope['provenance']['source_receipt'] = result.get('source_receipt')
            envelope['provenance']['review_files'] = result['review_files']
            review_files = list(run.rglob('*'))
            review_files = [p for p in review_files if p.is_file()]
            for path in result.get('evidence_files', []):
                if Path(path) not in review_files:
                    review_files.append(Path(path))
            raw = read_json(raw_path)
            plan = read_json(plan_path)
            if not isinstance(plan, dict) or not isinstance(plan.get('probes'), list):
                raise ValueError('invalid probe plan')
            expected['probes'] = [probe['id'] for probe in plan['probes']]
            validate_audit(raw, expected, base_dir=run)
            task_raw = result.get('review_files', {}).get('task', {}).get('raw_path')
            if task_raw is not None:
                probe_work = Path(task_raw).parent
            elif recovery is None:
                # Legacy parallel runners keep their plan under this run
                # without a separate per-task review_files entry.
                probe_work = run
            else:
                raise ValueError('recovered parallel audit requires task review evidence')
            _validate_plan(plan, raw, probe_work)
            if read_json(frozen_plan_path) != plan:
                raise ValueError('merged parallel plan differs from the plan frozen before probe execution')
            payload = json.loads(json.dumps(raw))
            payload['probes'] = probe_results
            if sha(frozen_plan_path) != frozen_inputs['plan_sha256']:
                raise ValueError('frozen probe plan changed during execution')
            for item in frozen_inputs['probes'].values():
                if item['snapshot_path'] is not None and sha(item['snapshot_path']) != item['sha256']:
                    raise ValueError('frozen probe workbook changed during execution')
            for name in ('original', 'rollouts', 'candidate'):
                verify_snapshot(inputs[name])
            validate_audit(payload, expected, base_dir=run)
            envelope.update(status='completed', payload=payload)
            envelope['output_hashes'] = {**_output_hashes(run), **{str(p): sha(p) for p in review_files}}
            commit = write_stage_commit(inputs['records_dir'], 'audit', payload,
                                        references=[raw_path, plan_path, manifest_path, manifest_path.parent / 'navigation.json',
                                                    frozen_plan_path, run / 'probe-inputs/manifest.json'] + review_files,
                                        parent_ids=[inputs['attribution']['commit_id'], inputs['design']['commit_id']],
                                        metadata={'evidence_fingerprint': fingerprint, 'is_human_gold': False,
                                                  'audit_scope': descriptor, 'review_mode': review_mode,
                                                  'audit_envelope': envelope},
                                        expected_units=expected, base_dir=run)
            envelope['audit_commit'] = commit['commit_id']
    except (Exception, SystemExit) as exc:
        envelope['status'] = 'failed'
        # Raw review and probe receipts remain on disk. A failure to validate or
        # commit them must not make them accepted metric inputs.
        envelope['payload'] = None
        envelope['failures'].append(f'{type(exc).__name__}: {exc}')
    if run is not None:
        if envelope['status'] != 'completed':
            envelope['output_hashes'] = _output_hashes(run)
        if not (run / 'audit.json').exists():
            write_json(run / 'audit.json', envelope)
    write_json(latest, envelope)
    return envelope
