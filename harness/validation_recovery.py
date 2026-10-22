"""One explicit replacement only for independently proven unexecuted bridge calls.

This module never contacts a model or repairs a result. Runtime inspection supplies
its proof; ordinary resume uses the immutable routing after explicit preparation.
"""
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json

import host_validation as hv
from stage_records import snapshot_tree, verify_snapshot, _tree_entries
from staged_synth import save_new

SCHEMA = 'validation-recovery-v1'
PROOF_SCHEMA = 'validation-nonexecution-proof-v1'


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'recovery evidence must be a regular file: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('recovery observation and terminal times require timezones')
    return parsed


def _root(request):
    return Path(request['episode']) / 'diagnostics/validation'


def _request(request):
    if (request.get('schema_version') != hv.SCHEMA
            or hv.digest({k: v for k, v in request.items() if k != 'request_id'}) != request.get('request_id')
            or hv.read(_root(request) / 'request.json') != request):
        raise ValueError('recovery requires the unchanged validation request')


def _failed(request, index):
    old = hv.original_case_directory(request, 'trainee', index)
    candidate = verify_snapshot(old / 'host-candidate.json')
    if (Path(candidate['snapshot_path']).resolve() != (old / 'candidate').resolve()
            or candidate['tree_sha256'] != request['candidate_sha256']
            or hv.read(old / 'candidate-version.json').get('tree_sha256') != request['candidate_sha256']):
        raise ValueError('original failed case belongs to a different candidate')
    result = hv.collect_case(old, 'trainee')
    if (result['status'] != 'failed' or result['reward'] is not None
            or result.get('reported_reward') is not None or not result.get('result_path')):
        raise ValueError('recovery requires failed, null-reward, finished original trainee trials')
    trial = Path(result['result_path'])
    bridge = trial.parent / 'agent/bridge-request.json'
    bridge_id = hv.read(bridge).get('request_id')
    if not isinstance(bridge_id, str) or not bridge_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in bridge_id):
        raise ValueError('original failed trial has no valid bridge request ID')
    return old, result, bridge, bridge_id


def _proof(request, proof, indices):
    if (proof.get('schema_version') != PROOF_SCHEMA
            or proof.get('request_id') != request['request_id']
            or proof.get('candidate_sha256') != request['candidate_sha256']):
        raise ValueError('nonexecution proof belongs to a different request or candidate')
    observed = _time(proof['observed_at'])
    if not isinstance(proof.get('remote_host'), str) or not proof['remote_host'].strip():
        raise ValueError('nonexecution proof must identify its inspected host')
    remote = Path(proof['remote_job_root'])
    if not remote.is_absolute():
        raise ValueError('nonexecution proof needs an absolute remote job root')
    checks = proof.get('checks')
    if not isinstance(checks, list) or [c.get('index') for c in checks] != indices:
        raise ValueError('proof checks must exactly match the ordered recovery indices')
    old_cases = []
    for index, check in zip(indices, checks):
        old, result, bridge, bridge_id = _failed(request, index)
        if (check.get('bridge_request_id') != bridge_id
                or check.get('remote_job_path') != str(remote / bridge_id)
                or any(check.get(key) is not False for key in
                       ('remote_job_exists', 'worker_running', 'request_pending'))):
            raise ValueError('proof must explicitly confirm this bridge request never executed and is not pending')
        terminals = [result['finished_at']]
        if result.get('process', {}).get('finished_at'):
            terminals.append(result['process']['finished_at'])
        if any(observed < _time(value) for value in terminals):
            raise ValueError('nonexecution inspection predates the failed attempt terminal')
        old_cases.append((index, old, result, bridge))
    evidence = proof.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('nonexecution proof requires complete raw runtime inspection evidence')
    for item in evidence:
        if not Path(item['path']).is_absolute() or _sha(item['path']) != item['sha256']:
            raise ValueError('raw runtime inspection evidence changed')
    return old_cases


def verify_recovery(request):
    """Return the verified manifest or None; never prepare or execute recovery."""
    root = _root(request)
    active = root / 'active-recovery.json'
    directory = root / 'recovery-01'
    if not active.exists():
        if directory.exists():
            raise ValueError('recovery preparation was interrupted; no implicit recovery or fallback')
        return None
    _request(request)
    pointer = hv.read(active)
    path = directory / 'manifest.json'
    if pointer != {'recovery_id': pointer.get('recovery_id'), 'manifest_path': str(path),
                   'manifest_sha256': _sha(path)}:
        raise ValueError('active recovery pointer changed')
    manifest = hv.read(path)
    if (manifest.get('schema_version') != SCHEMA
            or hv.digest({k: v for k, v in manifest.items() if k != 'recovery_id'}) != pointer['recovery_id']
            or manifest.get('recovery_id') != pointer['recovery_id']
            or manifest.get('request_id') != request['request_id']
            or manifest.get('request_sha256') != _sha(root / 'request.json')
            or manifest.get('candidate_sha256') != request['candidate_sha256']):
        raise ValueError('recovery manifest identity changed')
    for item in manifest['files']:
        if _sha(item['snapshot_path']) != item['sha256']:
            raise ValueError('frozen recovery evidence changed')
        if item.get('verify_live') and _sha(item['source_path']) != item['sha256']:
            raise ValueError('original recovery evidence changed')
    proof = hv.read(directory / 'proof.json')
    old_cases = _proof(request, proof, manifest['indices'])
    snapshots = manifest['old_cases']
    if len(snapshots) != len(old_cases):
        raise ValueError('recovery old case inventory changed')
    routes = {}
    for (index, old, result, bridge), item in zip(old_cases, snapshots):
        snapshot = verify_snapshot(item['snapshot'])
        if (item['index'] != index or item['result'] != result
                or Path(snapshot['source_path']) != old
                or _tree_entries(old) != snapshot['entries']):
            raise ValueError('original failed trial differs from frozen recovery history')
        routes[str(index)] = str(old.with_name(old.name + '--recovery-01'))
    if manifest['routes'] != routes:
        raise ValueError('recovery route differs from the one explicit replacement')
    previous = hv.read(directory / 'previous-validation.json')
    if previous.get('request_id') != request['request_id'] or any(
            previous['attempts'][index] != result for index, _, result, _ in old_cases):
        raise ValueError('recovery history does not match original validation attempts')
    return manifest


def recovery_reference(request):
    manifest = verify_recovery(request)
    if manifest is None:
        return None
    path = _root(request) / 'recovery-01/manifest.json'
    return dict(recovery_id=manifest['recovery_id'], manifest_path=str(path), manifest_sha256=_sha(path))


def prepare_recovery(request_path, proof_path, indices, reason):
    """Explicit caller action under the existing controller lock; no execution."""
    request_path, proof_path = Path(request_path).resolve(), Path(proof_path).resolve()
    request = hv.read(request_path)
    _request(request)
    root = _root(request)
    if request_path != (root / 'request.json').resolve():
        raise ValueError('recovery requires the canonical validation request')
    if (not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < request['attempts'] for i in indices)
            or indices != sorted(set(indices)) or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('explicit recovery requires ordered unique indices and a reason')
    with hv.locked(root / 'controller.lock'):
        if (root / 'active-recovery.json').exists() or (root / 'recovery-01').exists():
            raise ValueError('only one explicit recovery is allowed; resume its existing routed cases')
        from quality_gate import evaluate_quality
        quality = evaluate_quality(request['episode'])
        if quality.get('status') != 'accepted' or quality.get('candidate_sha256') != request['candidate_sha256']:
            raise ValueError('recovery requires the current unchanged candidate to remain quality accepted')
        proof = hv.read(proof_path)
        old_cases = _proof(request, proof, indices)
        previous_path = Path(request['episode']) / 'diagnostics/validation.json'
        previous = hv.read(previous_path)
        if previous.get('request_id') != request['request_id'] or any(
                previous['attempts'][i] != result for i, _, result, _ in old_cases):
            raise ValueError('current validation receipt does not match selected failed trials')
        if any(old.with_name(old.name + '--recovery-01').exists() for _, old, _, _ in old_cases):
            raise ValueError('replacement case already exists without this recovery manifest')
        directory = root / 'recovery-01'
        directory.mkdir()
        files = []
        def freeze(source, name, live=True):
            source, dest = Path(source), directory / name
            raw = source.read_bytes()
            dest.write_bytes(raw)
            item = dict(source_path=str(source), snapshot_path=str(dest),
                        sha256=hashlib.sha256(raw).hexdigest(), verify_live=live)
            files.append(item)
            return item
        freeze(proof_path, 'proof.json')
        freeze(previous_path, 'previous-validation.json', False)
        freeze(root / 'request.json', 'request.json')
        for n, item in enumerate(proof['evidence']):
            freeze(item['path'], f'inspection-{n:02d}.raw')
        snapshots, routes = [], {}
        for index, old, result, bridge in old_cases:
            destination = old.with_name(old.name + '--recovery-01')
            snapshots.append(dict(index=index, result=result,
                                  snapshot=snapshot_tree(old, directory / f'failed-case-{index:02d}')))
            routes[str(index)] = str(destination)
        manifest = dict(schema_version=SCHEMA, request_id=request['request_id'],
                        request_sha256=_sha(request_path), candidate_sha256=request['candidate_sha256'],
                        audit_commit=quality['audit_commit'], created_at=hv.now(), reason=reason,
                        indices=indices, routes=routes, files=files, old_cases=snapshots)
        manifest['recovery_id'] = hv.digest(manifest)
        save_new(directory / 'manifest.json', manifest)
        pointer = dict(recovery_id=manifest['recovery_id'], manifest_path=str(directory / 'manifest.json'),
                       manifest_sha256=_sha(directory / 'manifest.json'))
        save_new(root / 'active-recovery.json', pointer)
        verify_recovery(request)
        return directory / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--proof', required=True, type=Path)
    parser.add_argument('--indices', required=True, nargs='+', type=int)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    print(prepare_recovery(args.request, args.proof, args.indices, args.reason))


if __name__ == '__main__':
    main()
