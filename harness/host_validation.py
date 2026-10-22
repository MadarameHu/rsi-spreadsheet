#!/usr/bin/env python3
"""Reference and trainee checks outside the author's process and time allocation.

One detached controller, immutable candidate copies and existing Harbor calls.
Re-entry collects receipts; an attempt already started is never sampled again.
Execution completion is separate from semantic quality approval.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, Harbor, agent_timeout_sec, load_config, make_harbor, task_cpus,
                    task_pass_threshold, task_timeout_sec, thread_flags)
from stage_records import verify_snapshot
from staged_synth import DEFAULT_ROLLOUT_ATTEMPTS, snapshot_or_verify

SCHEMA = 'host-validation-v1'
# Host runtime state, deliberately outside immutable validation requests.
# Candidate episodes can live anywhere; their path depth does not define a pool.
TARGET_EXECUTION_ROOT = ROOT / 'runs' / '.target-execution-slots'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    pending.write_bytes(encoded(value) + b'\n')
    pending.replace(path)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def execution_slot(request):
    """Share the score cap; one admitted waiter polls so newcomers cannot barge."""
    root = Path(request['slot_root'])
    root.mkdir(parents=True, exist_ok=True)
    cap = int(request['config']['score']['n_concurrent'])
    slot = None
    try:
        # Release admission immediately after acquiring a running slot. Holding
        # it across yield would incorrectly serialize all model execution.
        with locked(root / '.admission.lock'):
            while slot is None:
                for i in range(cap):
                    candidate = (root / f'{i}.lock').open('a')
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                        continue
                    except BaseException:
                        candidate.close()
                        raise
                    slot = candidate
                    break
                if slot is None:
                    time.sleep(0.2)
        yield
    finally:
        if slot is not None:
            try:
                fcntl.flock(slot, fcntl.LOCK_UN)
            finally:
                slot.close()


@contextmanager
def target_execution_slot(request):
    """Share target capacity across runs without changing frozen request identity.

    Oracle and verifier probes retain their existing per-run execution pool.
    The target pool uses the explicit host runtime root and configured score cap;
    current spreadsheet requests use four slots, including legacy requests.
    """
    pool = {'slot_root': str(TARGET_EXECUTION_ROOT),
            'config': request['config']}
    with execution_slot(pool):
        yield


def rollout_settings(cfg, *, attempts=None, max_concurrent=None, previous=None):
    """Resolve host rollout controls without borrowing the author container cap."""
    pipeline = cfg.get('training_pipeline', {})
    if not isinstance(pipeline, dict):
        raise ValueError('training_pipeline must be an object')
    rollout = pipeline.get('rollout', {})
    if not isinstance(rollout, dict):
        raise ValueError('training_pipeline.rollout must be an object')
    previous = previous or {}
    if attempts is None:
        attempts = rollout.get('attempts', previous.get('attempts', DEFAULT_ROLLOUT_ATTEMPTS))
    if type(attempts) is not int or attempts < 1:
        raise ValueError('attempts must be a positive integer')
    cap = cfg['score']['n_concurrent']
    if type(cap) is not int or cap < 1:
        raise ValueError('score.n_concurrent must be a positive integer')
    if max_concurrent is None:
        max_concurrent = rollout.get('max_concurrent',
                                     previous.get('max_concurrent', min(attempts, cap)))
    if type(max_concurrent) is not int or not 1 <= max_concurrent <= cap:
        raise ValueError(f'validation concurrency must be within the configured cap {cap}')
    return attempts, max_concurrent


def prepare_request(hb, cfg, benchmark, episode, attempts=None, max_concurrent=None):
    from stage_audit import load_inputs
    ep = Path(episode).resolve()
    root = ep / 'diagnostics/validation'
    path = root / 'request.json'
    previous = read(path) if path.exists() else None
    attempts, concurrency = rollout_settings(cfg, attempts=attempts, max_concurrent=max_concurrent,
                                             previous=previous)
    upstream = load_inputs(ep)
    manifest = upstream['candidate']
    if manifest is None:
        raise ValueError('no committed candidate to validate')
    safe_cfg = deepcopy(cfg)
    safe_cfg['gateway']['api_key'] = 'env:VALIDATION_CREDENTIAL'
    request = dict(schema_version=SCHEMA, episode=str(ep), benchmark=benchmark,
                   candidate_sha256=manifest['tree_sha256'], candidate_manifest=manifest,
                   attribution_commit=upstream['attribution']['commit_id'],
                   design_commit=upstream['design']['commit_id'], config=safe_cfg,
                   harbor_bin=hb.bin, attempts=attempts, max_concurrent=concurrency,
                   slot_root=str(ep.parents[2] / '.validation-slots'))
    request['request_id'] = digest(request)
    with locked(root / 'prepare.lock'):
        if path.exists():
            if read(path) != request:
                raise ValueError('validation identity changed; retain this run and use a new candidate/version')
        else:
            write(path, request)
    return path


def launch_validation(hb, cfg, benchmark, episode, *, attempts=None, max_concurrent=None,
                      detach=False, phase='all'):
    """The author has returned before this starts. Parent exit does not kill it."""
    _validate_phase(phase)
    path = prepare_request(hb, cfg, benchmark, episode, attempts, max_concurrent)
    env = dict(os.environ, VALIDATION_CREDENTIAL=hb.api_key)
    with (path.parent / 'controller.log').open('a') as output:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(path),
                                 '--phase', phase],
                                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    if detach:
        return {'status': 'launched', 'pid': proc.pid, 'request': str(path)}
    # Do not use a Popen context manager or an author-derived timeout here.
    returncode = proc.wait()
    state = Path(episode) / 'diagnostics/validation.json'
    if not state.is_file():
        raise RuntimeError('validation controller exited without a stage receipt; inspect controller.log')
    result = read(state)
    if result.get('request_id') != read(path)['request_id']:
        raise RuntimeError('validation controller receipt has a different request identity')
    if returncode != 0 and result.get('status') not in ('failed', 'incomplete'):
        raise RuntimeError(f'validation controller exited {returncode}; inspect controller.log')
    return result


def original_case_directory(request, kind, index=0):
    suffix = '' if kind == 'oracle' else f'-{index:02d}'
    return Path(request['episode']) / 'work/testruns' / f"{kind}-host-{request['request_id'][:12]}{suffix}"


def case_directory(request, kind, index=0):
    if kind != 'oracle':
        from validation_recovery import verify_recovery
        recovery = verify_recovery(request)
        if recovery is not None and str(index) in recovery['routes']:
            return Path(recovery['routes'][str(index)])
    return original_case_directory(request, kind, index)


def prepare_case(request, kind, index):
    ep = Path(request['episode'])
    path = case_directory(request, kind, index)
    source = Path(request['candidate_manifest']['snapshot_path'])
    verify_snapshot(request['candidate_manifest'])
    manifest = snapshot_or_verify(source, path / 'candidate', path / 'host-candidate.json')
    if manifest['tree_sha256'] != request['candidate_sha256']:
        raise ValueError('candidate version changed before validation')
    # Same self-test evidence layout as the old wrappers, now written by the host.
    attribution = (ep / 'diagnostics/attribution.json').read_bytes()
    derived = encoded(read(ep / 'diagnostics/design.json'))
    raw = (ep / 'work/design.json').read_bytes()
    fields = dict(manifest, attribution_sha256=hashlib.sha256(attribution).hexdigest(),
                  design_sha256=hashlib.sha256(derived).hexdigest(),
                  model_design_sha256=hashlib.sha256(raw).hexdigest())
    for name, data in [('candidate-version.json', encoded(fields)), ('attribution.json', attribution),
                       ('design.json', derived), ('model-design.json', raw)]:
        dest = path / name
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError(f'changed immutable validation evidence: {dest}')
        if not dest.exists():
            dest.write_bytes(data)
    return path


def collect_case(path, kind):
    """Only a finished trial receipt can supply a score; never read job averages."""
    path = Path(path)
    base = dict(id=path.name, kind=kind, status='pending', reward=None, error=None,
                started_at=None, finished_at=None, result_path=None)
    started = path / 'host-started.json'
    if started.exists():
        base.update(status='incomplete',
                    error='Attempt already started but has no finished trial receipt; not resampled.')
        try:
            base['started_at'] = read(started)['started_at']
        except (ValueError, KeyError, TypeError) as exc:
            base['error'] = f'Unreadable start receipt ({type(exc).__name__}); not resampled.'
    candidates = []
    for f in (path / 'job').glob('*/*/result.json'):
        try:
            data = read(f)
        except ValueError:
            base.update(status='incomplete', error=f'Unreadable trial receipt: {f}; not resampled.')
            continue
        if 'exception_info' in data or 'agent_execution' in data:
            candidates.append((f, data))
    if len(candidates) > 1:
        raise ValueError(f'multiple trial receipts for one validation attempt: {path}')
    if candidates:
        f, data = candidates[0]
        if data.get('finished_at'):
            error = data.get('exception_info')
            reward = ((data.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            valid_reward = type(reward) in (int, float) and math.isfinite(reward)
            base.update(status='completed' if error is None and valid_reward else 'failed',
                        reward=float(reward) if error is None and valid_reward else None,
                        reported_reward=reward if valid_reward or reward is None else repr(reward),
                        error=error or (None if valid_reward else 'No finite verifier reward'),
                        started_at=data.get('started_at') or base['started_at'],
                        finished_at=data['finished_at'], result_path=str(f),
                        result_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    process_receipt = path / 'host-process.json'
    if process_receipt.exists():
        base['process'] = read(process_receipt)
    receipt = path / 'host-result.json'
    if receipt.exists():
        prior = read(receipt)
        if prior['status'] in ('completed', 'failed') and prior != base:
            if base['status'] == 'incomplete' and prior.get('result_path') is None:
                return prior  # explicit runner error, not a reason to retry
            raise ValueError(f'validation receipt or its source changed: {path}')
    return base


def execute_case(hb, cfg, request, kind, index):
    with execution_slot(request):
        return _execute_case(hb, cfg, request, kind, index)


def _execute_case(hb, cfg, request, kind, index):
    path = prepare_case(request, kind, index)
    prior = collect_case(path, kind)
    if prior['status'] != 'pending':
        write(path / 'host-result.json', prior)
        return prior
    # Waiting for target capacity is not an attempt. Acquire before writing the
    # start receipt so an interrupted waiter remains safely resumable as pending.
    with target_execution_slot(request) if kind == 'trainee' else nullcontext():
        return _execute_pending_case(hb, cfg, request, kind, path)


def _execute_pending_case(hb, cfg, request, kind, path):
    candidate = path / 'candidate'
    budget = task_timeout_sec(cfg, request['benchmark'])
    outer = budget * float(cfg['score']['outer_wall_task_multiplier']) + 900
    start = dict(started_at=now(), pid=os.getpid(), kind=kind, candidate_sha256=request['candidate_sha256'],
                 task_timeout_seconds=budget, outer_timeout_seconds=outer)
    with (path / 'host-started.json').open('x') as f:
        json.dump(start, f)
        f.flush()
        os.fsync(f.fileno())
    failure = None
    returncode = None
    try:
        if kind == 'oracle':
            cpus = task_cpus(candidate)
            returncode = hb.run(['-p', str(candidate), '-a', 'oracle', '--n-concurrent', '1',
                    '--agent-setup-timeout-multiplier', str(hb.setup_mult),
                    *thread_flags('--ae', cpus), *thread_flags('--ve', cpus)], path / 'job', outer)
        else:
            declared = agent_timeout_sec(candidate, budget)
            hb.customer(candidate, cfg['customer']['model'], path / 'job', outer,
                        agent_timeout_mult=budget / declared if declared > 0 else 1,
                        agent=cfg['customer']['agent'], closed_book=cfg['customer'].get('closed_book', True))
    except Exception as exc:
        # Retain the error and raw job, without guessing a reward or retrying a call.
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    write(path / 'host-process.json', dict(returncode=returncode, error=failure, finished_at=now()))
    result = collect_case(path, kind)
    if failure and result['status'] == 'incomplete':
        result.update(status='failed', error=failure, finished_at=now())
    verify_snapshot(path / 'host-candidate.json')
    write(path / 'host-result.json', result)
    return result


def summarize(request, oracle, attempts, started_at, status):
    usable = [x for x in attempts if x['status'] == 'completed' and x['reward'] is not None]
    threshold = task_pass_threshold(Path(request['candidate_manifest']['snapshot_path']))
    result = dict(schema_version=SCHEMA, request_id=request['request_id'], status=status,
                pid=os.getpid(), hostname=socket.gethostname(),
                candidate_sha256=request['candidate_sha256'], started_at=started_at,
                finished_at=now() if status != 'running' else None,
                requested_attempts=request['attempts'], completed_attempts=len(usable),
                solved_attempts=sum(x['reward'] >= threshold for x in usable),
                oracle=oracle, attempts=attempts, quality_approved=None,
                difficulty_status='preliminary',
                pending_checks=['independent semantic review', 'targeted verifier probes', 'full difficulty calibration'])
    from validation_recovery import recovery_reference
    recovery = recovery_reference(request)
    if recovery is not None:
        result['recovery'] = recovery
    return result


def _validate_phase(phase):
    if phase not in ('all', 'reference', 'trainee'):
        raise ValueError('validation phase must be all, reference or trainee')


def run_worker(request_path, hb=None, *, phase='all'):
    _validate_phase(phase)
    request_path = Path(request_path)
    request = read(request_path)
    identity = {k: v for k, v in request.items() if k != 'request_id'}
    if request.get('schema_version') != SCHEMA or digest(identity) != request.get('request_id'):
        raise ValueError('validation request identity/hash mismatch')
    ep = Path(request['episode'])
    state_path = ep / 'diagnostics/validation.json'
    root = request_path.parent
    # A second caller waits for the active controller, then only collects results.
    with locked(root / 'controller.lock'):
        try:
            return _run_locked(request, root, state_path, hb, phase)
        except Exception as exc:
            # Preserve the last observed attempt states and the actual failure.
            # Do not infer scores or start replacement attempts while handling errors.
            try:
                state = read(state_path) if state_path.exists() else {}
            except (ValueError, OSError):
                state = {}
            state.update(schema_version=SCHEMA, request_id=request['request_id'],
                         candidate_sha256=request['candidate_sha256'], status='failed',
                         pid=os.getpid(), hostname=socket.gethostname(), finished_at=now(),
                         error={'type': type(exc).__name__, 'message': str(exc)},
                         quality_approved=None)
            write(state_path, state)
            raise


def _run_locked(request, root, state_path, hb, phase):
    ep = Path(request['episode'])
    cfg = deepcopy(request['config'])
    cfg['gateway']['api_key'] = os.environ.get('VALIDATION_CREDENTIAL', '')
    if hb is None:
        if not cfg['gateway']['api_key']:
            raise ValueError('VALIDATION_CREDENTIAL is missing')
        hb = Harbor(request['harbor_bin'], cfg['gateway']['base_url'], cfg['gateway']['api_key'],
                    cfg['score']['setup_timeout_mult'], cfg['customer']['model'],
                    claude_code_version=cfg.get('claude_code_version'))
    # Recheck immutable upstream evidence even on resume.
    from stage_audit import load_inputs
    upstream = load_inputs(ep)
    if (upstream['candidate']['tree_sha256'] != request['candidate_sha256'] or
            upstream['design']['commit_id'] != request['design_commit'] or
            upstream['attribution']['commit_id'] != request['attribution_commit']):
        raise ValueError('upstream changed since validation request')
    previous = read(state_path) if state_path.exists() else {}
    started_at = previous.get('started_at') or now()
    attempts = [collect_case(case_directory(request, 'trainee', i), 'trainee')
                for i in range(request['attempts'])]
    oracle = collect_case(case_directory(request, 'oracle'), 'oracle')
    if (previous.get('status') in ('completed', 'reference_completed')
            and previous.get('request_id') == request['request_id']
            and previous.get('oracle') == oracle and previous.get('attempts') == attempts):
        expected = summarize(request, oracle, attempts, started_at, previous['status'])
        for key in ('pid', 'hostname', 'finished_at'):
            expected[key] = previous.get(key)
        if previous != expected:
            raise ValueError('validation summary differs from its trial receipts')
        if previous['status'] == 'completed' or phase == 'reference':
            return previous
    write(root / 'controller.json', dict(pid=os.getpid(), started_at=now(),
                                        request_id=request['request_id'], phase=phase))
    write(state_path, summarize(request, oracle, attempts, started_at, 'running'))
    if phase != 'trainee' and oracle['status'] == 'pending':
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute_case, hb, cfg, request, 'oracle', 0)
            while not future.done():
                wait([future], timeout=1)
                oracle = collect_case(case_directory(request, 'oracle'), 'oracle')
                if oracle['status'] == 'incomplete' and not future.done():
                    oracle.update(status='running', error=None)
                write(state_path, summarize(request, oracle, attempts, started_at, 'running'))
            oracle = future.result()
    write(state_path, summarize(request, oracle, attempts, started_at, 'running'))
    if oracle['status'] != 'completed' or oracle['reward'] < task_pass_threshold(Path(request['candidate_manifest']['snapshot_path'])):
        status = 'incomplete' if oracle['status'] in ('pending', 'incomplete') else 'failed'
        result = summarize(request, oracle, attempts, started_at, status)
        write(state_path, result)
        return result
    if phase == 'reference':
        result = summarize(request, oracle, attempts, started_at, 'reference_completed')
        write(state_path, result)
        return result
    if 'candidate_job' in cfg.get('training_pipeline', {}):
        # Re-evaluate in this controller process immediately before any trainee
        # scheduling; a parent process may have imported an older gate version.
        try:
            import template_probes
            if template_probes.review_mode(cfg) == 'template':
                quality = template_probes.template_admission(ep)
            else:
                from quality_gate import evaluate_quality
                quality = evaluate_quality(ep)
            accepted = isinstance(quality, dict) and quality.get('status') == 'accepted'
        except Exception as exc:
            quality = {'status': 'blocked', 'issues': [
                {'id': 'quality_gate_execution', 'verdict': 'fault',
                 'reason': f'{type(exc).__name__}: {exc}'}]}
            accepted = False
        if not accepted:
            result = summarize(request, oracle, attempts, started_at, 'quality_blocked')
            result['quality_gate'] = quality
            result['quality_issues'] = quality.get('issues', []) if isinstance(quality, dict) else [
                {'id': 'quality_gate_execution', 'verdict': 'fault', 'reason': 'Invalid quality gate receipt'}]
            write(state_path, result)
            return result
    # Unknown in-flight outcomes consume capacity; never compensate with fresh samples.
    if any(x['status'] == 'incomplete' for x in attempts):
        result = summarize(request, oracle, attempts, started_at, 'incomplete')
        write(state_path, result)
        return result
    with ThreadPoolExecutor(max_workers=request['max_concurrent']) as pool:
        futures = {pool.submit(execute_case, hb, cfg, request, 'trainee', i): i
                   for i in range(request['attempts'])}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                attempts[futures[future]] = future.result()
            for future in pending:
                i = futures[future]
                current = collect_case(case_directory(request, 'trainee', i), 'trainee')
                if current['status'] == 'incomplete':
                    current.update(status='running', error=None)
                attempts[i] = current
            write(state_path, summarize(request, oracle, attempts, started_at, 'running'))
    status = 'completed' if all(x['status'] == 'completed' for x in attempts) else 'incomplete'
    result = summarize(request, oracle, attempts, started_at, status)
    write(state_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--benchmark')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--attempts', type=int)
    parser.add_argument('--phase', choices=('all', 'reference', 'trainee'), default='all')
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = run_worker(args.worker, phase=args.phase)
    else:
        if not all((args.episode, args.benchmark, args.config)):
            parser.error('--episode, --benchmark and --config are required')
        cfg = load_config(args.config)
        hb = make_harbor(cfg, cfg['score']['setup_timeout_mult'])
        result = launch_validation(hb, cfg, args.benchmark, args.episode,
                                   attempts=args.attempts, max_concurrent=args.concurrency, detach=args.detach,
                                   phase=args.phase)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] in ('launched', 'completed', 'reference_completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
