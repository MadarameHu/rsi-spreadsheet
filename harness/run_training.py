#!/usr/bin/env python3
"""Run explicit candidate jobs using existing author, admission and rollout stages.

Analysis is an input. This entrypoint never starts diagnosis or aggregation and
does not apply the benchmark's difficulty-band score to training admission.
Use --export-only after collecting complete native demonstrations on the host.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
import os
from pathlib import Path
import socket
import time

from candidate_jobs import prepare_joblist, run_candidate, validate_job, _digest
from common import (ROOT, DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency,
                    add_path_args, apply_path_args, benchmark_dir,
                    load_config, log, make_harbor)
from host_validation import locked, read, write
from run_episode import validate_authored_candidate
from run_synth import episode_dir
from training_export import export_candidate


def execute_job(cfg, benchmark, run_id, job, episode, *, export_root,
                export_only=False, demonstration_paths=(), quality_scope='full',
                audit_concurrency=DEFAULT_AUDIT_CONCURRENCY, source_executor=None):
    audit_concurrency = validate_audit_concurrency(audit_concurrency)
    validate_job(job, cfg, original=job['original'])
    ep = episode_dir(run_id, benchmark, job['original'], episode)
    current = deepcopy(cfg)
    current.setdefault('training_pipeline', {})['candidate_job'] = deepcopy(job)
    status_path = ep/'training-status.json'
    status = dict(job_id=job['id'], original=job['original'], goal_id=job['goal_id'],
                  difficulty=job['difficulty'], episode=str(ep),
                  pid=os.getpid(), hostname=socket.gethostname())

    def save(phase, **fields):
        status.update(phase=phase, updated_at=time.time(), **fields)
        write(status_path, status)
        log('training', f'{job["original"]} {job["id"][:12]}: {phase}')

    # A simple process lock prevents two resumptions from authoring the same job.
    with locked(ep/'training.lock'):
        previous = read(status_path) if status_path.is_file() else {}
        if previous and previous.get('job_id') != job['id']:
            raise ValueError('existing training status belongs to a different candidate job')
        # Export and candidate reuse must retain the original author timings.
        # Missing legacy timestamps stay missing; a resumed invocation is not
        # another author attempt. run_candidate still verifies all reuse evidence.
        for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at',
                    'requested_review_mode', 'requested_quality_scope',
                    'requested_audit_concurrency', 'shared_model_slots'):
            if key in previous:
                status[key] = previous[key]
        author_attempt_exists = ((ep/'diagnostics/design/started.json').is_file()
                                 or (ep/'diagnostics/synthesis.json').is_file())
        previous_sft = previous.get('export', {}).get('sft', {})
        previous_count = previous_sft.get('count', 0) if previous_sft.get('status') == 'exported' else 0
        # Retain successful export provenance if this new export attempt fails.
        if previous.get('export'):
            status['export'] = previous['export']
        try:
            saved_paths = previous.get('demonstration_paths', [])
            if not isinstance(saved_paths, list) or any(not isinstance(path, str) for path in saved_paths):
                raise ValueError('saved demonstration_paths must be an ordered list of paths')
            supplied_paths = list(demonstration_paths)
            if previous_count and not saved_paths and not supplied_paths:
                raise ValueError('successful legacy SFT export has no saved demonstration_paths; '
                                 'supply its original demonstration descriptors to resume')
            merged_paths = []
            for value in [*saved_paths, *supplied_paths]:
                path = Path(value).resolve()
                if path.is_dir():
                    path = path / 'demonstration.json'
                if str(path) not in merged_paths:
                    merged_paths.append(str(path))
            status['demonstration_paths'] = merged_paths
            if not export_only:
                from source_review import model_slot, review_source, SourceReviewHandle
                slot = partial(model_slot, ROOT / 'runs', max_concurrent=audit_concurrency)
                with ExitStack() as stages:
                    source_future = None
                    pool = source_executor or stages.enter_context(ThreadPoolExecutor(max_workers=1))
                    save('authoring', status='running', requested_review_mode='parallel',
                         requested_quality_scope=quality_scope,
                         requested_audit_concurrency=audit_concurrency,
                         shared_model_slots=audit_concurrency,
                         **({} if author_attempt_exists else {'queued_at': time.time()}))
                    hb = make_harbor(current, float(current['researcher']['agent_setup_timeout_mult']))
                    # Reserve the author slot before source workers enter the
                    # shared pool. Never hold it while waiting for their result.
                    with slot():
                        save('authoring', **({} if author_attempt_exists else {
                            'author_slot_acquired_at': time.time()}))
                        reviewer = make_harbor(current, float(current['judge'].get(
                            'agent_setup_timeout_mult', current['score']['setup_timeout_mult'])))
                        source_future = pool.submit(review_source, reviewer, current, job,
                                                    max_concurrent=audit_concurrency, slot=slot)
                        source_future = SourceReviewHandle(source_future, reviewer, current, job,
                                                           max_concurrent=audit_concurrency, slot=slot)
                        run_candidate(hb, cfg, benchmark, benchmark_dir(benchmark)/job['original'], ep, episode, job)
                    save('reference_quality_rollout', **({} if author_attempt_exists else {
                        'author_slot_released_at': time.time()}))
                    hb = make_harbor(current, float(current['score']['setup_timeout_mult']))
                    validation = validate_authored_candidate(
                        hb, current, benchmark, ep, quality_scope=quality_scope,
                        audit_concurrency=audit_concurrency, audit_slot=slot, source_review=source_future)
                save('exporting', validation=validation)
            else:
                validation_path = ep/'diagnostics/validation.json'
                save('exporting', status='running',
                     validation=read(validation_path) if validation_path.is_file() else {})
            from stage_metrics import summarize_envelope, summarize_audit
            audit_path = ep/'diagnostics/audit.json'
            metrics = (summarize_envelope(read(audit_path), audit_path=audit_path)
                       if audit_path.is_file() else summarize_audit(None, status='missing'))
            write(ep/'diagnostics/metrics.json', metrics)
            exported = export_candidate(ep, export_root, merged_paths)
            rejected = exported.get('sft', {}).get('rejections', [])
            if rejected:
                status['export_attempt'] = exported
                raise ValueError(f'demonstration validation failed: {rejected}')
            if (exported.get('status') != 'blocked'
                    and exported.get('sft', {}).get('count', 0) < previous_count):
                status['export_attempt'] = exported
                raise ValueError('new export would discard previously successful SFT demonstrations')
            # Quality rejection and incomplete sampling remain visible separately.
            validation = status.get('validation', {})
            final = ('quality_blocked' if exported['status'] == 'blocked' else
                     'sampling_incomplete' if validation and validation.get('status') != 'completed' else
                     'completed')
            save('finished', status=final, export=exported)
            return status
        except (Exception, SystemExit) as exc:
            save('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
            raise


def _candidate_concurrency(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('candidate concurrency must be an integer from 1 to 4') from exc
    if not 1 <= number <= 4:
        raise argparse.ArgumentTypeError('candidate concurrency must be an integer from 1 to 4')
    return number


def _audit_concurrency(value):
    try:
        return validate_audit_concurrency(int(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('audit concurrency must be an integer from 1 to 16') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--joblist', required=True)
    parser.add_argument('--sources', help='JSON ordered original/analysis_source objects; required only to prepare')
    parser.add_argument('--repair-from', help='with --prepare-only, append one revision from this rejected episode')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--candidate-concurrency', type=_candidate_concurrency, default=1,
                        help='independent candidate pipelines at once (1-4); target execution keeps its shared cap')
    parser.add_argument('--quality-scope', choices=['full', 'selected_design'], default='selected_design',
                        help='training admission reviews selected target evidence; full also reviews every unused attribution unit')
    parser.add_argument('--audit-concurrency', type=_audit_concurrency, default=DEFAULT_AUDIT_CONCURRENCY,
                        help='ready reviewers per source/candidate (1-16, default 16); '
                             'authors and reviewers share this many model slots; target rollout cap is separate')
    parser.add_argument('--export-root', required=True)
    parser.add_argument('--demonstration', action='append', default=[])
    parser.add_argument('--job-id', action='append', default=[])
    add_path_args(parser)
    args = parser.parse_args()
    apply_path_args(args)
    cfg = load_config(args.config)
    path = Path(args.joblist)
    if args.repair_from and (not args.prepare_only or args.sources):
        parser.error('--repair-from requires --prepare-only and cannot be combined with --sources')
    if args.prepare_only:
        if args.repair_from:
            from repair_jobs import prepare_repair_joblist
            manifest = prepare_repair_joblist(cfg, args.repair_from, path)
            log('training', f'prepared repair job {manifest["jobs"][-1]["id"]}; run only this --job-id')
            return 0
        if not args.sources:
            parser.error('--prepare-only requires --sources')
        manifest = prepare_joblist(cfg, read(args.sources), path)
        log('training', f'prepared {len(manifest["jobs"])} candidate jobs: {path}')
        return 0
    manifest = read(path)
    if manifest.get('joblist_id') != _digest({k: v for k, v in manifest.items() if k != 'joblist_id'}):
        raise ValueError('candidate joblist identity changed')
    jobs = manifest['jobs']
    assignments = manifest.get('episode_assignments', {})
    if assignments and (set(assignments) != {job['id'] for job in jobs}
                        or any(type(n) is not int or n < 1 for n in assignments.values())):
        raise ValueError('episode assignments must map every job to a positive integer')
    if assignments and len({(job['original'], assignments[job['id']]) for job in jobs}) != len(jobs):
        raise ValueError('two jobs cannot share an episode for the same original')
    missing = set(args.job_id) - {job['id'] for job in jobs}
    if missing:
        raise ValueError('requested job ID is absent from joblist')
    plan_path = ROOT/'runs'/args.run_id/('training-plan-'+manifest['joblist_id']+'.json')
    plan = dict(benchmark=args.benchmark, joblist=manifest, export_root=str(Path(args.export_root).resolve()))
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('training plan changed; use a new run ID')
    write(plan_path, plan)
    episodes, failures, scheduled = {}, [], []
    for job in jobs:
        episodes[job['original']] = episodes.get(job['original'], 0)+1
        if args.job_id and job['id'] not in args.job_id:
            continue
        scheduled.append((job, assignments.get(job['id'], episodes[job['original']])))
    # Numbering follows the full immutable joblist, independently of filtering
    # and completion order. A slow first job cannot block later submissions.
    # Source requests may continue caching other observations from the initial
    # goal after the candidate's own dependencies finish. Keep their coordinator
    # outside candidate workers so it cannot block the next candidate in queue.
    with ThreadPoolExecutor(max_workers=args.candidate_concurrency) as source_pool, \
         ThreadPoolExecutor(max_workers=args.candidate_concurrency) as pool:
        futures = {pool.submit(execute_job, cfg, args.benchmark, args.run_id, job, episode,
                               export_root=args.export_root, export_only=args.export_only,
                               demonstration_paths=args.demonstration, quality_scope=args.quality_scope,
                               audit_concurrency=args.audit_concurrency,
                               source_executor=source_pool): job
                   for job, episode in scheduled}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if not isinstance(result, dict):
                    raise ValueError('candidate execution returned no outcome')
                issues = result.get('export', {}).get('quality_gate', {}).get('issues', [])
                if (result.get('status') not in ('completed', 'quality_blocked')
                        or any(issue.get('verdict') == 'fault' for issue in issues)):
                    failures.append(job['id'])
                    log('training', f'{job["original"]}: failed ({result.get("status")}); inspect training-status.json')
            except (Exception, SystemExit) as exc:
                failures.append(job['id'])
                log('training', f'{job["original"]}: failed ({type(exc).__name__}); inspect training-status.json')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
