"""Explicit two-call author protocol; no output clipping, repair or automatic retry."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import mounts
import diagnostics
from common import ROOT, benchmark_dir, discover_tasks, render_task, log
from run_rollout import rollout_bundle_path
from stage_records import (snapshot_tree, verify_snapshot, write_stage_commit,
                           read_stage_commit, validate_design,
                           _load_json, _tree_entries, _encoded)


BUDGET_MODE = diagnostics.PARALLEL_BUDGET_MODE
# Trainee attempts per candidate when training_pipeline.rollout.attempts is unset.
# 16 separates the 2/16..12/16 band from always/never solved; fewer is too coarse.
DEFAULT_ROLLOUT_ATTEMPTS = 16


def agent_execution_evidence(job):
    """Read exactly one Harbor trial; never substitute wall/setup duration."""
    paths = sorted(Path(job).rglob('result.json'))
    evidence = {'status': 'unknown', 'active_seconds': None,
                'result_paths': [str(path) for path in paths], 'source': None}
    try:
        trials = []
        for path in paths:
            raw = path.read_bytes()
            record = _load_json(raw)
            if isinstance(record, dict) and ('agent_execution' in record or 'exception_info' in record):
                trials.append((path, raw, record))
        if len(trials) != 1:
            raise ValueError(f'expected one Harbor trial result, found {len(trials)}')
        path, raw, record = trials[0]
        execution = record.get('agent_execution')
        evidence['source'] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                              'bytes': len(raw), 'agent_execution': execution}
        if not isinstance(execution, dict):
            raise ValueError('missing Harbor trial agent_execution')
        timestamps = [execution.get(name) for name in ('started_at', 'finished_at')]
        if not all(isinstance(value, str) and value for value in timestamps):
            raise ValueError('missing Harbor trial agent_execution timestamps')
        start, end = [datetime.fromisoformat(value.replace('Z', '+00:00')) for value in timestamps]
        seconds = (end - start).total_seconds()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid Harbor trial agent_execution duration')
        evidence.update(status='completed', active_seconds=seconds)
    except (OSError, ValueError, TypeError) as error:
        evidence['error'] = f'{type(error).__name__}: {error}'
    return evidence


def execution_seconds(result):
    evidence = result.get('agent_execution', {})
    if evidence.get('status') != 'completed':
        raise RuntimeError('Harbor agent execution duration unknown; no budget inferred: '
                           + str(evidence.get('error', 'missing duration evidence')))
    source = evidence['source']
    raw = Path(source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256'] or len(raw) != source['bytes']:
        raise ValueError('Harbor agent execution source changed')
    seconds = evidence['active_seconds']
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('invalid recorded Harbor agent execution duration')
    return seconds


def stage_references(diag, stage, result):
    return [diag / stage / 'started.json', diag / stage / 'finished.json',
            Path(result['agent_execution']['source']['path'])]


def load(path):
    return _load_json(Path(path).read_bytes())


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load(path) != value:
            raise ValueError(f'existing artifact differs: {path}')
        return
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save_bytes(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'existing raw artifact differs: {path}')
        return
    with path.open('xb') as f:
        f.write(raw)


def snapshot_or_verify(source, destination, manifest_path):
    if manifest_path.exists():
        manifest = verify_snapshot(manifest_path)
        if manifest['source_path'] != str(source.absolute()) or manifest['snapshot_path'] != str(destination.absolute()):
            raise ValueError('snapshot identity changed')
        verify_snapshot(dict(manifest, snapshot_path=str(source)))
        return manifest
    if destination.exists():
        # Only recover a completely matching copy. Never complete or repair a
        # partial snapshot, since the source may have changed after the crash.
        entries = _tree_entries(source)
        if entries != _tree_entries(destination):
            raise ValueError('incomplete or changed snapshot retained; not repaired')
        manifest = {'schema_version':'1.0', 'source_path':str(source.absolute()),
                    'snapshot_path':str(destination.absolute()), 'entries':entries,
                    'tree_sha256':hashlib.sha256(_encoded(entries)).hexdigest()}
    else:
        manifest = snapshot_tree(source, destination)
    save_new(manifest_path, manifest)
    return manifest


def event(diag, kind, **fields):
    with (diag / 'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), event=kind, **fields), ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _score_trace_text(diag):
    """The design stage's score trace path, or an explicit absence note."""
    trace_dir = Path(diag) / 'score-trace'
    name = 'score_trace_summary.json'
    return str(trace_dir) if (trace_dir / name).is_file() else (
        '(no score traces; the original task has no tests/task_spec.json)')


def validate_probes(work, diag, candidate):
    """Host-recalculate the author's wrong-answer probes; refuse a hollow one.

    Only runs when the author wrote work/probes.json. A probe that matches the
    reference in every target cell proves nothing and is rejected here rather
    than in the design text.
    """
    from score_trace import probe as recalc_probe, validate_probes as check_sources
    path = Path(work) / 'probes.json'
    if not path.is_file():
        return None
    variants = json.loads(path.read_bytes())
    from parallel_attribution import trace_summary
    summary = trace_summary(diag)
    if summary is not None:
        check_sources(variants, summary)
    out = Path(diag) / 'probes-recalculated.pending'
    if out.exists():
        raise ValueError('unfinished probe recalculation; preserve it without replay')
    results = recalc_probe(Path(candidate), variants, out)
    hollow = [item['name'] for item in results if item['replay_failures']]
    if hollow:
        raise ValueError(f'probes failed to replay: {hollow}')
    matched = [item['name'] for item in results if not item['distinguishable_from_reference']]
    if matched:
        raise ValueError(f'probes match the reference in every target cell: {matched}')
    out.rename(Path(diag) / 'probes')
    return results


def render_stage(destination, replacements):
    """Render the design prompt into a new staged copy; templates stay untouched."""
    if destination.exists():
        raise ValueError(f'render destination already exists; preserve it: {destination}')
    replacements = dict(replacements)
    replacements.setdefault('__SCORE_TRACES__', '(no score traces supplied)')
    replacements.setdefault('__PROBES_JSON__', str(destination / 'probes.json'))
    replacements.setdefault('__ERROR_LIBRARY__', '(none: no error class library for this task yet)')
    rendered = render_task(ROOT / 'agents' / 'synth', destination, replacements)
    prompt = (ROOT / 'agents/synth-design-parallel/instruction.md').read_text()
    prompt = prompt.replace('__DIFFICULTY_POLICY__', replacements.get('__DIFFICULTY_POLICY__',
        'The eventual difficulty requirement is __BAND_TEXT__, measured over __ATTEMPTS__ '
        'independent attempts by the customer\'s model. Planning cannot establish a solve '
        'rate. The host measures outcomes separately after authoring is complete.'))
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    prompt = prompt.replace('__CANDIDATE_JOB_CONSTRAINT__', 'No host candidate job is assigned; select a grounded primary goal.')
    import re
    if re.search(r'__[A-Z0-9_]+__', prompt):
        raise ValueError('unresolved stage prompt placeholder')
    (rendered / 'instruction.md').write_text(prompt)
    message = ('Candidate execution is unavailable during authoring. Save one candidate and '
               'design.json, then finish; the independent host validation stage runs tests.')
    for name in ('adb-oracle', 'adb-testrun', 'harbor'):
        (rendered / 'environment/bin' / name).write_text('#!/bin/sh\necho "' + message + '" >&2\nexit 2\n')
    return rendered


def _run_design(hb, cfg, benchmark, source, bundle, ep, diag, allocated_seconds, attribution):
    from run_synth import band_text, search_backend, search_line
    import error_library
    library = error_library.library_path(cfg, benchmark, Path(source).name)
    stage = 'design'
    stage_dir = diag / stage
    active_budget_mode = BUDGET_MODE
    stage_dir.mkdir(exist_ok=True)
    marker = stage_dir / 'started.json'
    finished = stage_dir / 'finished.json'
    if finished.exists():
        result = load(finished)
        if result.get('budget_mode') != active_budget_mode or result.get('allocated_seconds') != allocated_seconds:
            raise ValueError('stage budget mode or allocation changed; no automatic recovery')
        execution_seconds(result)
        return result
    if marker.exists():
        raise RuntimeError(f'{stage} already started without terminal record; inspect retained job, no automatic retry')
    allocated = float(allocated_seconds)
    if not math.isfinite(allocated) or allocated <= 0:
        raise RuntimeError('shared researcher active-time budget exhausted; no additional budget granted')
    work = ep / 'work'
    for name in ('out', 'testruns', '.tmp'):
        (work / name).mkdir(parents=True, exist_ok=True)
    design = work / 'design.json'
    output = design
    search = search_backend(cfg)
    bench = benchmark_dir(benchmark)
    res = cfg['researcher']
    summary = load(bundle / 'summary.json')
    difficulty_policy = None
    if cfg.get('training_pipeline', {}).get('candidate_job') is not None:
        rollout_attempts = cfg['training_pipeline'].get('rollout', {}).get('attempts', DEFAULT_ROLLOUT_ATTEMPTS)
        if type(rollout_attempts) is not int or rollout_attempts < 1:
            raise ValueError('training rollout attempts must be a positive integer')
        difficulty_policy = (
            'Follow the assigned candidate job difficulty.instruction as the task design constraint. '
            'This training job does not require the benchmark solve-rate band. '
            f'After independent quality checks, the host plans {rollout_attempts} independent attempts '
            'by the customer model and records their actual outcomes. A zero score, including zero '
            'successful attempts, does not by itself make a task invalid and is not a training-data '
            'admission gate. Task and scorer quality are reviewed independently. Planning does not '
            'establish a solve rate; do not invent successful attempts or optimize for an unmeasured pass rate.')
    replacements = {
        '__BAND_TEXT__': band_text(cfg), '__SEARCH_LINE__': search_line(cfg, search),
        '__BENCHMARK__': benchmark, '__BENCHMARK_DIR__': str(bench.resolve()),
        '__ORIGINAL_DIR__': str(source), '__ROLLOUT_DIR__': str(bundle),
        '__OUT__': str(work / 'out'), '__WORK__': str(work),
        '__ATTEMPTS__': str(summary.get('attempts', 0)),
        '__ALLOWED_MODEL__': cfg['customer']['model'],
        '__BUDGET_MIN__': str(allocated / 60), '__TOTAL_BUDGET_MIN__': str(res['budget_min']),
        '__GATEWAY_BASE_URL__': cfg['gateway']['base_url'].rstrip('/'),
        '__ATTRIBUTION_JSON__': str(attribution),
        '__DESIGN_JSON__': str(design),
        '__SCORE_TRACES__': _score_trace_text(diag),
        '__PROBES_JSON__': str(work / 'probes.json'),
        '__ERROR_LIBRARY__': error_library.prompt_text(library),
        '__OBSERVATIONS_INDEX__': str(diag / 'observations/index.json'),
        '__CANDIDATE_JOB_CONSTRAINT__': ('Read the host candidate job at ' + str(diag / 'candidate_job.json') +
            '. Its goal_id is mandatory: primary_goal_id must match it. Follow its difficulty.instruction as a design constraint; '
            'the difficulty name is not a measured solve rate. Do not repeat job IDs or variant metadata in your model output. '
            'If the job contains repair, read repair.feedback_dir/quality.json and audit.json, inspect the immutable '
            'parent candidate specified there, and create one corrected candidate in your own output directory. '
            'The prior design is at repair.parent_episode + /diagnostics/design.json; use this exact file path. '
            'The frozen audit also supplies evidence/design.json under its work_dir. '
            'Explain the relevant fixes in design.json without changing the assigned training goal. '
            'Keep all defining goal conditions; repair the independently identified task/scorer defects. '
            'The parent files are read-only and must remain unchanged. Do not run validation yourself.'
            if cfg.get('training_pipeline', {}).get('candidate_job') else 'No host candidate job is assigned; select a grounded primary goal.'),
    }
    if difficulty_policy is not None:
        replacements['__DIFFICULTY_POLICY__'] = difficulty_policy
    rendered = render_stage(stage_dir / 'task', replacements)
    ro = [bench, source, bundle] + ([search] if search else [])
    ro.extend([attribution, diag / 'observations', diag / 'records', diag / 'attribution/work'])
    if (diag / 'score-trace').is_dir():
        ro.append(diag / 'score-trace')
    if library is not None and library.is_file():
        ro.append(library)
    if (diag / 'candidate_job.json').is_file():
        ro.append(diag / 'candidate_job.json')
        from repair_jobs import validate_repair
        ro.extend(validate_repair(load(diag / 'candidate_job.json')))
    if (diag / 'analysis_reuse.json').is_file():
        ro.extend(Path(path) for path in load(diag / 'analysis_reuse.json')['read_only_paths'])
    # Reused observation records retain their original evidence paths.
    # Expose only those inputs/outputs, never the previous candidate or audit.
    index = load(diag / 'observations/index.json')
    if index.get('reuse_observations_from'):
        for worker in index['workers']:
            output_path = Path(worker['output_path'])
            ro.extend([output_path, output_path.parent / 'inputs'])
            if worker.get('raw_output_path'):
                ro.append(Path(worker['raw_output_path']))
    specs = json.loads(mounts.build(hb.bin, ro=ro, rw=[work]))
    for spec in specs:
        if spec.get('target') == '/var/run/docker.sock':
            spec['read_only'] = True
    if os.environ.get('ADB_CC_BINARY'):
        binary = Path(os.environ['ADB_CC_BINARY']).resolve(strict=True)
        specs.append({'type': 'bind', 'source': str(binary), 'target': '/usr/local/bin/claude', 'read_only': True})
    env = {
        'ADB_WORK': work, 'ADB_OUT': work / 'out', 'ADB_TESTRUN_OUT': work / 'testruns',
        'ADB_BENCHMARK_DIR': bench.resolve(), 'ADB_ORIGINAL_DIR': source,
        'ADB_ROLLOUT_DIR': bundle, 'ADB_ALLOWED_MODEL': cfg['customer']['model'],
        'ADB_TRAINEE_AGENT': cfg['customer'].get('agent', 'claude-code'),
        'ADB_MAX_CONCURRENT': res['max_concurrent_docker'], 'ADB_DEADLINE_EPOCH': int(time.time() + allocated),
        'TMPDIR': work / '.tmp', 'HARBOR_REAL': hb.bin, 'ANTHROPIC_BASE_URL': hb.base_url,
        'ADB_CC_VERSION': cfg.get('claude_code_version') or '', 'ADB_SEARCH_SCRIPT': search or '',
        'ADB_SEARCH_MAX_CALLS': res.get('search_max_calls', 100),
        'ADB_ATTRIBUTION_JSON': attribution,
        'ADB_DESIGN_JSON': design,
    }
    for name in ('ADB_DOCKER_REAL', 'QWEN_BRIDGE_URL', 'QWEN_BRIDGE_TOKEN'):
        if os.environ.get(name):
            env[name] = os.environ[name]
    ak = []
    if res['agent'] == 'codex':
        if res.get('web_access'):
            ak.append('web_search=live')
        if res.get('reasoning_effort'):
            ak.append(f"reasoning_effort={res['reasoning_effort']}")
    save_new(marker, {'started_at': time.time(), 'budget_mode': active_budget_mode,
                      'allocated_seconds': allocated,
                      'model': res['model'], 'prompt_sha256': hashlib.sha256((rendered / 'instruction.md').read_bytes()).hexdigest()})
    event(diag, 'stage_started', stage=stage)
    log('diagnostics', f'{stage}: {res["model"]}, allocated agent budget {allocated/60:.1f} min')
    rc = hb.agent_task(rendered, res['agent'], res['model'], stage_dir / 'job',
                       allocated * 1.5 + 900, json.dumps(specs),
                       agent_env=[f'{k}={v}' for k,v in env.items()],
                       agent_timeout_mult=allocated / 1800, agent_kwargs=ak)
    result = {'harbor_rc': rc, 'finished_at': time.time(), 'output': str(output),
              'budget_mode': active_budget_mode, 'allocated_seconds': allocated,
              'agent_execution': agent_execution_evidence(stage_dir / 'job')}
    save_new(finished, result)
    event(diag, 'stage_finished', stage=stage, harbor_rc=rc)
    # A nonzero controller status is retained as evidence. As in the baseline,
    # complete structured deliverables are still eligible for independent review.
    execution_seconds(result)
    return result


def synthesize_staged(hb, cfg, benchmark, task_dir, ep_dir, episode, *, analysis_only=False):
    active_budget_mode = BUDGET_MODE
    parallel = diagnostics.parallel_config(cfg)
    candidate_job = cfg.get('training_pipeline', {}).get('candidate_job')
    job_inputs = None
    if candidate_job is not None:
        from candidate_jobs import validate_job
        _, _, job_inputs = validate_job(candidate_job, cfg, original=task_dir.name)
    ep = ep_dir.resolve()
    diag = ep / 'diagnostics'
    state_path = diag / 'synthesis.json'
    config_fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if (diag / 'started.json').exists():
        identity = load(diag / 'started.json')
        expected = {'config_sha256':config_fingerprint, 'benchmark':benchmark,
                    'original_task':task_dir.name, 'source_path':str(task_dir.resolve()), 'episode':episode}
        if any(identity.get(k) != v for k,v in expected.items()):
            raise ValueError('staged run configuration or source identity changed')
    if state_path.exists():
        state = load(state_path)
        if state['status'] != 'completed':
            raise RuntimeError('diagnostic synthesis failed previously; no automatic retry')
        for name in ('attribution_commit', 'design_commit'):
            read_stage_commit(diag / 'records', state[name])
        if candidate_job is not None:
            design_commit = read_stage_commit(diag / 'records', state['design_commit'])
            if design_commit['payload'].get('primary_goal_id') != candidate_job['goal_id']:
                raise ValueError('committed candidate differs from assigned primary goal')
            if load(diag / 'candidate_job.json') != candidate_job:
                raise ValueError('committed candidate job changed')
        manifest = verify_snapshot(state['candidate_manifest'])
        # Original scorer uses work/out, so refuse if it no longer matches.
        verify_snapshot(dict(manifest, snapshot_path=manifest['source_path']))
        return load(ep / 'synth.json')
    if ((ep / 'synth.json').exists() and not (diag / 'started.json').exists()) or (ep / 'job').exists():
        raise ValueError('baseline episode cannot be reused in diagnostics mode; use a new run ID')
    diag.mkdir(parents=True, exist_ok=True)
    config_fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    start_path = diag / 'started.json'
    if start_path.exists():
        started = load(start_path)
        if started['config_sha256'] != config_fingerprint:
            raise ValueError('configuration changed during staged run')
        if started.get('budget_mode') != active_budget_mode:
            raise ValueError('existing stage budget mode differs; no automatic migration or retry')
    else:
        if (ep / 'work').exists():
            raise ValueError('existing author work must not be mixed with a new staged run')
        started = {'started_at': time.time(), 'config_sha256': config_fingerprint,
                   'benchmark': benchmark, 'original_task': task_dir.name, 'source_path':str(task_dir.resolve()),
                   'episode': episode, 'budget_mode': active_budget_mode,
                   'total_agent_budget_seconds': None, 'explicit_parallel_budgets': parallel}
        save_new(start_path, started)
    metadata = dict(benchmark=benchmark, original_task=task_dir.name, episode=episode,
                    model=cfg['researcher']['model'], config_sha256=config_fingerprint,
                    budget_mode=active_budget_mode, total_agent_budget_seconds=started['total_agent_budget_seconds'])
    try:
        inputs_path = diag / 'inputs.json'
        if inputs_path.exists():
            inputs = load(inputs_path)
            for manifest in inputs.values():
                verify_snapshot(manifest)
        else:
            bundle = (Path(job_inputs['rollouts']['snapshot_path']) if job_inputs is not None else
                      rollout_bundle_path(cfg, benchmark, task_dir.name))
            if not (bundle / 'summary.json').is_file():
                raise ValueError(f'no source rollout summary at {bundle}')
            inputs = {'original': snapshot_tree(task_dir, diag / 'inputs' / task_dir.name),
                      'rollouts': snapshot_tree(bundle, diag / 'inputs' / 'rollouts')}
            save_new(inputs_path, inputs)
        source = Path(inputs['original']['snapshot_path'])
        bundle = Path(inputs['rollouts']['snapshot_path'])
        a_path = diag / 'attribution.json'
        if candidate_job is not None:
            from candidate_jobs import reuse_analysis
            a_commit = reuse_analysis(candidate_job, cfg, source, bundle, diag, metadata)
        else:
            from parallel_attribution import produce_attribution
            a_commit = produce_attribution(hb, cfg, source, bundle, ep, diag, metadata)
        import error_library
        library = error_library.library_path(cfg, benchmark, task_dir.name)
        error_library.ingest(library, a_commit, episode=ep)
        if analysis_only:
            return {'status': 'analysis_completed', 'attribution_commit': a_commit['commit_id']}
        attribution_execution = a_commit['metadata']['stage_execution']
        result = _run_design(hb, cfg, benchmark, source, bundle, ep, diag,
                             float(parallel['design_budget_min']) * 60, a_path)
        if load(a_path) != a_commit['payload']:
            raise ValueError('committed attribution changed while authoring')
        raw = Path(result['output']).read_bytes()
        from compact_synthesis import normalize_design
        payload = normalize_design(_load_json(raw), a_commit['payload'])
        validate_design(payload, a_commit['payload'])
        if candidate_job is not None and payload.get('primary_goal_id') != candidate_job['goal_id']:
            raise ValueError('authored candidate does not match its assigned primary goal')
        delivered = discover_tasks(ep / 'work/out')
        if len(delivered) != 1:
            raise ValueError(f'expected one delivered candidate, found {len(delivered)}; no candidate selected')
        for manifest in inputs.values():
            verify_snapshot(manifest)
        manifest_path = diag / 'candidate_manifest.json'
        candidate_manifest = snapshot_or_verify(delivered[0], diag / 'candidate', manifest_path)
        probe_results = validate_probes(ep / 'work', diag, diag / 'candidate')
        d_path = diag / 'design.json'
        save_new(d_path, payload)
        probe_refs = [diag / 'probes/probes.json'] if probe_results is not None else []
        d_commit = write_stage_commit(diag / 'records', 'design', payload,
                                      references=[Path(result['output']),d_path, manifest_path, *probe_refs] + stage_references(diag, 'design', result), parent_ids=[a_commit['commit_id']],
                                      metadata=dict(metadata, candidate_sha256=candidate_manifest['tree_sha256'], stage_execution=result,
                                                    probe_count=None if probe_results is None else len(probe_results),
                                                    raw_output_path=result['output'],raw_output_sha256=hashlib.sha256(raw).hexdigest()),
                                      claims=a_commit['payload'])
        event(diag, 'design_committed', commit_id=d_commit['commit_id'], candidate_sha256=candidate_manifest['tree_sha256'])
        error_library.mark_targeted(library, a_commit,
                                    [m['claim_id'] for m in payload.get('mappings', [])],
                                    episode=ep, design_commit=d_commit['commit_id'])
        summary = load(bundle / 'summary.json')
        rec = dict(benchmark=benchmark, original_task=task_dir.name, episode=episode,
                   researcher={k: cfg['researcher'][k] for k in ('agent','model','budget_min')},
                   harbor_rc=result['harbor_rc'], elapsed_sec=time.time()-started['started_at'],
                   delivered=[str(delivered[0].relative_to(ep / 'work/out'))], n_delivered=1,
                   original_rollout={k: summary.get(k) for k in ('attempts','n_usable','n_solved','pass_rate','pass_threshold','attempts_excluded')},
                   diagnostics_enabled=True, candidate_manifest=str(manifest_path),
                   budget_mode=active_budget_mode,
                   agent_execution_seconds={'attribution': execution_seconds(attribution_execution),
                                            'design': execution_seconds(result)},
                   stage_harbor_rc={'attribution': attribution_execution.get('harbor_rc'), 'design': result['harbor_rc']})
        if attribution_execution.get('runtime') == 'local-claude':
            rec['stage_runtime'] = {'attribution': 'local-claude', 'design': 'harbor'}
            rec['stage_returncode'] = {'attribution': attribution_execution['returncode'], 'design': result['harbor_rc']}
        rec.update(attribution_mode='parallel', explicit_parallel_budgets=parallel,
                   observations_index=str(diag / 'observations/index.json'),
                   budget_inventory=str(diag / 'observations/input_inventory.json'))
        rec['agent_execution_seconds']['aggregate'] = rec['agent_execution_seconds'].pop('attribution')
        if candidate_job is not None:
            rec['candidate_job'] = candidate_job
            rec['analysis_reuse'] = load(diag / 'analysis_reuse.json')
            rec['source_aggregate_execution_seconds'] = rec['agent_execution_seconds']['aggregate']
            rec['agent_execution_seconds']['aggregate'] = 0.0
            rec['stage_harbor_rc']['attribution'] = None
            if 'stage_returncode' in rec:
                rec['stage_returncode']['attribution'] = None
        if (ep / 'synth.json').exists():
            previous = load(ep / 'synth.json')
            if any(previous.get(k) != v for k,v in rec.items() if k != 'elapsed_sec'):
                raise ValueError('existing synth.json does not match recovered delivery')
            rec = previous
        save_new(ep / 'synth.json', rec)
        save_new(state_path, dict(status='completed', budget_mode=active_budget_mode, attribution_commit=a_commit['commit_id'],
                                 design_commit=d_commit['commit_id'], stage_commit_dir=str(diag / 'records'),
                                 candidate_manifest=str(manifest_path), input_manifest=str(inputs_path)))
        return rec
    except (Exception, SystemExit) as exc:
        import traceback
        if not state_path.exists():
            save_new(state_path, {'status': 'failed', 'budget_mode': active_budget_mode, 'error_type': type(exc).__name__,
                                  'error': str(exc), 'traceback': traceback.format_exc()})
        event(diag, 'synthesis_failed', error_type=type(exc).__name__, error=str(exc))
        raise
