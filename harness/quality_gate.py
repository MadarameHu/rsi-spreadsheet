"""Fail-closed candidate admission from committed audit and reference evidence.

Automatic review is not human gold. Trainee success rates are deliberately not
part of this gate. Evaluation is read-only; only run_quality_gate saves a receipt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_validation
import stage_audit
from common import task_pass_threshold, DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency
from stage_records import _tree_entries, verify_snapshot

SCHEMA = 'quality-gate-v1'


def _check(receipt, identifier, verdict, reason, evidence=None):
    item = {'id': identifier, 'verdict': verdict, 'reason': reason,
            'evidence': evidence or []}
    receipt['checks'].append(item)
    if verdict != 'correct':
        receipt['issues'].append(item)


def _evidence(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _finish(receipt):
    verdicts = [item['verdict'] for item in receipt['issues']]
    status = 'rejected' if 'incorrect' in verdicts else 'blocked' if verdicts else 'accepted'
    receipt.update(status=status, eligible_for_export=status == 'accepted',
                   quality_approved=True if status == 'accepted' else False if status == 'rejected' else None)
    return receipt


def _reference_request(ep, inputs):
    request_path = ep/'diagnostics/validation/request.json'
    request = host_validation.read(request_path)
    identity = {k: v for k, v in request.items() if k != 'request_id'}
    if (request.get('schema_version') != host_validation.SCHEMA
            or host_validation.digest(identity) != request.get('request_id')
            or Path(request['episode']).resolve() != ep
            or inputs['candidate'] is None
            or request.get('candidate_sha256') != inputs['candidate']['tree_sha256']
            or request.get('attribution_commit') != inputs['attribution']['commit_id']
            or request.get('design_commit') != inputs['design']['commit_id']):
        raise ValueError('reference request identity differs from committed candidate or stages')
    return request_path, request, host_validation.case_directory(request, 'oracle')


def freeze_reference_evidence(ep):
    """Freeze one terminal host reference independently of parallel semantic review."""
    from staged_synth import snapshot_or_verify, save_new
    ep = Path(ep).resolve()
    inputs = stage_audit.load_inputs(ep)
    request_path, request, case = _reference_request(ep, inputs)
    result = host_validation.collect_case(case, 'oracle')
    if result['status'] not in ('completed', 'failed'):
        return None  # Never snapshot a still-changing or unknown execution.
    root = ep/'diagnostics/validation'
    manifest = snapshot_or_verify(case, root/'reference-case', root/'reference-case-manifest.json')
    receipt = dict(schema_version='reference-evidence-v1', request_id=request['request_id'],
                   request_sha256=_evidence(request_path)['sha256'],
                   candidate_sha256=inputs['candidate']['tree_sha256'],
                   attribution_commit=inputs['attribution']['commit_id'],
                   design_commit=inputs['design']['commit_id'], case_path=str(case),
                   snapshot=manifest, result=result)
    receipt['reference_evidence_id'] = host_validation.digest(receipt)
    save_new(root/'reference-evidence.json', receipt)
    return receipt


def _parallel_reference_snapshot(ep, inputs, request_path, request, path):
    manifest_path = ep/'diagnostics/validation/reference-evidence.json'
    frozen = stage_audit.read_json(manifest_path)
    identity = {key: value for key, value in frozen.items() if key != 'reference_evidence_id'}
    if (frozen.get('schema_version') != 'reference-evidence-v1'
            or frozen.get('reference_evidence_id') != host_validation.digest(identity)
            or frozen.get('request_id') != request['request_id']
            or frozen.get('request_sha256') != _evidence(request_path)['sha256']
            or frozen.get('candidate_sha256') != inputs['candidate']['tree_sha256']
            or frozen.get('attribution_commit') != inputs['attribution']['commit_id']
            or frozen.get('design_commit') != inputs['design']['commit_id']
            or frozen.get('case_path') != str(path)):
        raise ValueError('independent reference snapshot identity differs from candidate/request')
    snapshot = verify_snapshot(frozen['snapshot'])
    expected_snapshot = ep/'diagnostics/validation/reference-case'
    if (Path(snapshot['source_path']).resolve() != path.resolve()
            or Path(snapshot['snapshot_path']).resolve() != expected_snapshot.resolve()):
        raise ValueError('independent reference snapshot has a different source or destination')
    if _tree_entries(path) != snapshot['entries']:
        raise ValueError('author reference evidence differs from its independent frozen snapshot')
    if host_validation.collect_case(path, 'oracle') != frozen['result']:
        raise ValueError('author reference result differs from independently frozen receipt')
    return manifest_path


def _reference_check(ep, inputs, review, receipt):
    request_path = ep/'diagnostics/validation/request.json'
    if not request_path.is_file():
        _check(receipt, 'reference_execution', 'missing',
               'Author reference execution is required; an independent positive probe is not a substitute.')
        return
    request_path, request, path = _reference_request(ep, inputs)
    if not path.is_dir():
        _check(receipt, 'reference_execution', 'missing', 'Author reference execution has not started.')
        return
    if review.get('review_mode', stage_audit.LEGACY_REVIEW_MODE) != stage_audit.REVIEW_MODE:
        _check(receipt, 'reference_execution', 'missing',
               'Legacy serial audit receipts are read-only; rerun quality review in parallel mode.')
        return
    if not (ep/'diagnostics/validation/reference-evidence.json').is_file():
        _check(receipt, 'reference_execution', 'missing',
               'Parallel review requires an independently frozen terminal author reference.')
        return
    manifest_path = _parallel_reference_snapshot(ep, inputs, request_path, request, path)
    candidate = verify_snapshot(path/'host-candidate.json')
    version = stage_audit.read_json(path/'candidate-version.json')
    if (candidate['tree_sha256'] != receipt['candidate_sha256']
            or version.get('tree_sha256') != receipt['candidate_sha256']
            or stage_audit.read_json(path/'attribution.json') != inputs['attribution']['payload']
            or stage_audit.read_json(path/'design.json') != inputs['design']['payload']):
        raise ValueError('author reference executed a different candidate or stage design')
    result = host_validation.collect_case(path, 'oracle')
    evidence = [_evidence(request_path), _evidence(manifest_path), _evidence(path/'candidate-version.json')]
    if result.get('result_path'):
        evidence.append(_evidence(result['result_path']))
    receipt['reference_execution'] = deepcopy(result)
    if result['status'] != 'completed' or result['reward'] is None:
        _check(receipt, 'reference_execution', 'fault',
               'Author reference has no successful, finished verifier result.', evidence)
    elif result['reward'] < task_pass_threshold(Path(candidate['snapshot_path'])):
        _check(receipt, 'reference_execution', 'incorrect',
               'The finished author reference was rejected by the candidate verifier.', evidence)
    else:
        _check(receipt, 'reference_execution', 'correct',
               'The exact authored reference completed and passed its verifier.', evidence)


def evaluate_quality(ep, *, audit=None):
    """Verify existing evidence and return admission; never run a model or write files."""
    ep = Path(ep).resolve()
    receipt = {'schema_version': SCHEMA, 'episode': str(ep), 'origin': 'automatic_quality_gate',
               'is_human_gold': False, 'status': 'blocked', 'eligible_for_export': False,
               'quality_approved': None, 'candidate_sha256': None, 'candidate_manifest': None,
               'attribution_commit': None, 'design_commit': None, 'audit_commit': None,
               'checks': [], 'issues': [], 'selected_claim_ids': [], 'selected_mapping_ids': [],
               'reference_execution': None, 'difficulty_used_for_admission': False}
    try:
        inputs = stage_audit.load_inputs(ep)
        if inputs['candidate'] is None:
            raise ValueError('no committed candidate')
        receipt.update(candidate_manifest=deepcopy(inputs['candidate']),
                       candidate_sha256=inputs['candidate']['tree_sha256'],
                       attribution_commit=inputs['attribution']['commit_id'],
                       design_commit=inputs['design']['commit_id'])
        if audit is None:
            path = ep/'diagnostics/audit.json'
            if not path.is_file():
                _check(receipt, 'audit', 'missing', 'A completed independent audit is required.')
                return _finish(receipt)
            audit = stage_audit.read_json(path)
        if audit.get('status') != 'completed':
            _check(receipt, 'audit', 'fault' if audit.get('status') == 'failed' else 'missing',
                   'Independent audit has no committed completed result.')
            return _finish(receipt)
        scope_mode = audit.get('audit_scope', {}).get('mode', 'full')
        # Recompute the selected IDs from committed upstream stages; never trust
        # an audit's self-declared subset. The verifier also checks its scope anchor.
        resolved_scope = stage_audit.resolve_scope(inputs, scope_mode)
        review = stage_audit.verify_cached_envelope(
            audit, inputs, audit.get('evidence_fingerprint'), resolved_scope['expected_units'])
        review_mode = review.get('review_mode', stage_audit.LEGACY_REVIEW_MODE)
        if review_mode not in (stage_audit.LEGACY_REVIEW_MODE, stage_audit.REVIEW_MODE):
            raise ValueError('invalid committed audit review_mode')
        receipt['review_mode'] = review_mode
        receipt['audit_scope'] = deepcopy(resolved_scope)
        receipt['audit_commit'] = review['audit_commit']
        receipt['audit_evidence_fingerprint'] = review['evidence_fingerprint']
        _check(receipt, 'audit_integrity', 'correct',
               'Audit commit, upstream parents and frozen execution evidence were verified.')
        payload = review['payload']
        claims = {item['id']: item for item in payload['claims']}
        mappings = {item['id']: item for item in payload['mappings']}
        selected = [item for item in inputs['design']['payload']['mappings'] if item.get('role') != 'exclude']
        if not selected:
            _check(receipt, 'selected_design', 'missing', 'No selected design mapping exists.')
        for item in selected:
            identifier, claim = item['id'], item['claim_id']
            receipt['selected_mapping_ids'].append(identifier)
            if claim not in receipt['selected_claim_ids']:
                receipt['selected_claim_ids'].append(claim)
                judgment = claims[claim]
                _check(receipt, 'claim:'+claim, judgment['verdict'], judgment['reason'], judgment['evidence'])
            judgment = mappings[identifier]
            for field in ('upstream_verdict', 'implementation_verdict'):
                _check(receipt, 'mapping:'+identifier+':'+field, judgment[field],
                       judgment['reason'], judgment['evidence'])
        task_checks = {item['id']: item for item in payload['task_checks']}
        for identifier in stage_audit.TASK_CHECKS:
            judgment = task_checks[identifier]
            _check(receipt, 'task:'+identifier, judgment['verdict'], judgment['reason'], judgment['evidence'])
        required_expectations = {'correct_solution': True, 'incorrect_solution': False}
        for probe in payload['probes']:
            identifier, expected = probe['id'], probe['expected_valid']
            if identifier in required_expectations and expected is not required_expectations[identifier]:
                verdict, reason = 'uncertain', 'Required positive/negative probe lacks the required independent expectation.'
            elif type(expected) is not bool:
                verdict, reason = 'uncertain', 'Verifier probe has no decided independent expectation.'
            elif probe['execution_status'] != 'completed' or type(probe['actual_accept']) is not bool:
                verdict, reason = 'fault', 'Verifier probe has no completed execution result.'
            elif probe['actual_accept'] is not expected:
                verdict, reason = 'incorrect', 'Verifier acceptance contradicts the advance independent expectation.'
            else:
                verdict, reason = 'correct', 'Actual verifier result matches its advance independent expectation.'
            _check(receipt, 'probe:'+identifier, verdict, reason, probe['evidence'])
        _reference_check(ep, inputs, review, receipt)
    except Exception as exc:
        _check(receipt, 'evidence_integrity', 'fault', f'{type(exc).__name__}: {exc}')
    return _finish(receipt)


def run_quality_gate(hb, cfg, benchmark, original, ep, *, scope='full',
                     max_concurrent=DEFAULT_AUDIT_CONCURRENCY, evaluate_only=False,
                     slot=None, source_review=None):
    """Reuse a verified completed audit; otherwise invoke the existing audit stage."""
    max_concurrent = validate_audit_concurrency(max_concurrent)
    if scope not in ('full', 'selected_design'):
        raise ValueError('quality review scope must be full or selected_design')
    receipt = evaluate_quality(ep)
    active_path = Path(ep)/'diagnostics/audit.json'
    switch_scope = False
    if active_path.is_file():
        try:
            active = stage_audit.read_json(active_path)
            switch_scope = (active.get('audit_scope', {}).get('mode', 'full') != scope
                            or active.get('review_mode', stage_audit.LEGACY_REVIEW_MODE)
                            != stage_audit.REVIEW_MODE)
        except (ValueError, TypeError, AttributeError):
            pass  # Malformed records never authorize an automatic retry.
    # Do not repeat a completed review merely because later trainee outputs changed
    # the live self-test tree. Its committed frozen evidence remains authoritative.
    if evaluate_only and switch_scope:
        _check(receipt, 'review_identity', 'fault', 'Active audit scope or mode differs from requested review.')
        receipt = _finish(receipt)
    if not evaluate_only and (switch_scope or (receipt['audit_commit'] is None and not active_path.exists())):
        try:
            existing = stage_audit.audit_episode(hb, cfg, benchmark, original, ep, scope=scope,
                                                 max_concurrent=max_concurrent, slot=slot,
                                                 source_review=source_review)
        except Exception as exc:
            _check(receipt, 'audit_execution', 'fault', f'{type(exc).__name__}: {exc}')
            receipt = _finish(receipt)
        else:
            receipt = evaluate_quality(ep, audit=existing)
    stage_audit.write_json(Path(ep)/'diagnostics/quality.json', receipt)
    return receipt
