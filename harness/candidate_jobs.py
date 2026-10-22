"""Immutable candidate jobs expand committed training goals without re-analysis.

Difficulty instructions describe the intended task design. They do not assert a
measured solve rate. Every job authors one candidate in its own episode.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from stage_records import (_encoded, _tree_entries, read_stage_commit,
                           verify_snapshot, write_stage_commit)
import staged_synth as st

JOB_SCHEMA = 'candidate-job-v1'
LIST_SCHEMA = 'candidate-joblist-v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _base_config(cfg):
    result = deepcopy(cfg)
    if isinstance(result.get('training_pipeline'), dict):
        result['training_pipeline'].pop('candidate_job', None)
    return result


def _source(original, path):
    """Load analysis commits directly; a candidate or synthesis receipt is not required."""
    from stage_audit import _load_observations, _verify_raw_output
    origin = Path(path)
    if not origin.is_absolute():
        raise ValueError('analysis_source must be an absolute diagnostics directory')
    origin = origin.resolve(strict=True)
    started = st.load(origin / 'started.json')
    if started.get('original_task') != original:
        raise ValueError('analysis source original task identity differs')
    commit = read_stage_commit(origin / 'records', st.load(origin / 'attribution_commit.json')['commit_id'])
    if commit['stage'] != 'attribution' or commit['payload'].get('schema_version') != 'parallel-attribution-v1':
        raise ValueError('candidate jobs require committed parallel attribution')
    if st.load(origin / 'attribution.json') != commit['payload']:
        raise ValueError('analysis attribution differs from immutable commit')
    if not started.get('config_sha256') or commit['metadata'].get('config_sha256') != started['config_sha256']:
        raise ValueError('analysis source config identity differs from attribution commit')
    _verify_raw_output(commit)
    refs = {Path(ref['original_path']).resolve(): ref for ref in commit['references']}
    for required in (origin / 'inputs.json', origin / 'attribution.json'):
        if required not in refs or _sha(required) != refs[required]['sha256']:
            raise ValueError('analysis inputs or attribution are not bound to commit')
    # Raw calculations and execution receipts remain exact, even on job resume.
    for path, ref in refs.items():
        if path.is_symlink() or not path.is_file() or _sha(path) != ref['sha256']:
            raise ValueError('analysis live evidence differs from immutable commit')
    inputs = st.load(origin / 'inputs.json')
    for name in ('original', 'rollouts'):
        verify_snapshot(inputs[name])
    observations = _load_observations(origin, origin / 'records', commit)
    goals = commit['payload'].get('training_goals')
    if not isinstance(goals, list) or not goals:
        raise ValueError('analysis has no declared training goals; no goals invented')
    identity = dict(attribution_commit=commit['commit_id'], config_sha256=started['config_sha256'],
                    inputs_sha256=_sha(origin / 'inputs.json'),
                    original_tree_sha256=inputs['original']['tree_sha256'],
                    rollouts_tree_sha256=inputs['rollouts']['tree_sha256'],
                    observation_index_sha256=observations['observation_index_sha256'])
    return origin, commit, inputs, identity


def prepare_joblist(cfg, sources, path, *, goal_ids=None):
    """Preserve source, declared goal, difficulty and variant order in a frozen list."""
    settings = cfg.get('training_pipeline', {}).get('candidate_generation', {})
    if not isinstance(settings, dict):
        raise ValueError('candidate_generation must be an object')
    difficulties = settings.get('difficulties')
    copies = settings.get('copies_per_goal')
    selected = goal_ids if goal_ids is not None else settings.get('goal_ids')
    if type(copies) is not int or copies < 1:
        raise ValueError('copies_per_goal must be an explicit positive integer')
    if not isinstance(difficulties, list) or not difficulties:
        raise ValueError('difficulties must declare at least one design instruction')
    names = set()
    for difficulty in difficulties:
        if (not isinstance(difficulty, dict) or set(difficulty) != {'name', 'instruction'}
                or any(not isinstance(difficulty[k], str) or not difficulty[k].strip() for k in difficulty)
                or difficulty['name'] in names):
            raise ValueError('difficulties require unique names and nonempty instructions')
        names.add(difficulty['name'])
    if selected is not None and (not isinstance(selected, list) or not selected
            or any(not isinstance(item, str) or not item for item in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('goal_ids must be a nonempty unique list when explicitly selected')
    if not isinstance(sources, list) or not sources:
        raise ValueError('sources must be a nonempty ordered list')
    jobs, origins = [], set()
    base_hash = _digest(_base_config(cfg))
    for source in sources:
        if not isinstance(source, dict) or set(source) != {'original', 'analysis_source'}:
            raise ValueError('source requires original and analysis_source')
        original = source['original']
        if not isinstance(original, str) or not original:
            raise ValueError('original must be a task identifier')
        origin, commit, _, identity = _source(original, source['analysis_source'])
        if original in origins:
            raise ValueError('duplicate original task in candidate job sources')
        origins.add(original)
        goals = commit['payload']['training_goals']
        if selected is not None and set(selected) - {goal['id'] for goal in goals}:
            raise ValueError('selected goal_id is absent from this analysis')
        for goal in goals:
            if selected is not None and goal['id'] not in selected:
                continue
            for difficulty in difficulties:
                for variant in range(copies):
                    job = dict(schema_version=JOB_SCHEMA, original=original, analysis_source=str(origin),
                               analysis_identity=identity, goal_id=goal['id'], difficulty=deepcopy(difficulty),
                               variant_index=variant, base_config_sha256=base_hash)
                    job['id'] = _digest(job)
                    jobs.append(job)
    manifest = dict(schema_version=LIST_SCHEMA, base_config_sha256=base_hash, jobs=jobs)
    manifest['joblist_id'] = _digest(manifest)
    st.save_new(Path(path), manifest)
    return manifest


def validate_job(job, cfg, *, original=None):
    if not isinstance(job, dict) or job.get('schema_version') != JOB_SCHEMA:
        raise ValueError('invalid candidate job schema')
    if job.get('id') != _digest({k: v for k, v in job.items() if k != 'id'}):
        raise ValueError('candidate job identity changed')
    if job.get('base_config_sha256') != _digest(_base_config(cfg)):
        raise ValueError('candidate job base configuration changed')
    if original is not None and job.get('original') != original:
        raise ValueError('candidate job original task differs')
    origin, commit, inputs, identity = _source(job['original'], job['analysis_source'])
    if identity != job['analysis_identity']:
        raise ValueError('candidate job analysis source identity changed')
    if job['goal_id'] not in {goal['id'] for goal in commit['payload']['training_goals']}:
        raise ValueError('candidate job goal is absent from source attribution')
    if 'repair' in job:
        from repair_jobs import validate_repair
        validate_repair(job)
    return origin, commit, inputs


def run_candidate(hb, cfg, benchmark, task_dir, ep_dir, episode, job):
    """Author one candidate through the existing staged protocol, with no analysis calls."""
    current = deepcopy(cfg)
    current.setdefault('training_pipeline', {})['candidate_job'] = deepcopy(job)
    validate_job(job, current, original=Path(task_dir).name)
    return st.synthesize_staged(hb, current, benchmark, Path(task_dir), Path(ep_dir), episode)


def reuse_analysis(job, cfg, source, bundle, diag, metadata):
    """Reuse all worker commits and the aggregate payload; only rebind host provenance."""
    from parallel_attribution import _inventory, _reuse_observations
    from stage_audit import _load_observations
    origin, original_commit, inputs = validate_job(job, cfg, original=metadata['original_task'])
    if origin == diag.resolve():
        raise ValueError('candidate episode must differ from its analysis source')
    for name, current in (('original', source), ('rollouts', bundle)):
        if _tree_entries(current) != inputs[name]['entries']:
            raise ValueError('candidate inputs differ from committed analysis source')
    obs = diag / 'observations'
    obs.mkdir(exist_ok=True)
    inventory = _inventory(source, bundle, cfg['diagnostics']['parallel_attribution'])
    inventory['config_sha256'] = metadata['config_sha256']
    inventory['analysis_reused'] = True
    inventory['maximum_agent_execution_minutes'] = cfg['diagnostics']['parallel_attribution']['design_budget_min']
    inventory['parallel_batch_wall_allocation_minutes_excluding_setup'] = inventory['maximum_agent_execution_minutes']
    inventory_path = obs / 'input_inventory.json'
    st.save_new(inventory_path, inventory)
    worker_origin = origin
    records, read_only = _reuse_observations(worker_origin, diag, source, bundle, inventory)
    order = inventory['sample_order']
    parent_ids = [records[sample]['commit_id'] for sample in order]
    if parent_ids != original_commit['parent_ids']:
        raise ValueError('reused aggregate parents differ from committed worker inventory')
    index = dict(schema_version='observation-index-v1', status='completed', sample_order=order,
                 input_inventory=str(inventory_path), workers=[records[sample] for sample in order],
                 observation_ids=[identifier for sample in order for identifier in records[sample]['observation_ids']],
                 reuse_observations_from=str(worker_origin.resolve()))
    worker_sources = sorted({record['reused_from'] for record in records.values()})
    if worker_sources != [str(worker_origin.resolve())]:
        index['reuse_observation_sources'] = worker_sources
    index_path = obs / 'index.json'
    st.save_new(index_path, index)
    # Specific analysis files are mounted; never the source episode/candidate/audit tree.
    roots = (origin / 'attribution', origin / 'inputs', origin / 'observations')
    for ref in original_commit['references']:
        path = Path(ref['original_path']).resolve()
        if path in read_only or path in (origin / 'attribution.json', origin / 'inputs.json') or any(path.is_relative_to(root) for root in roots):
            read_only.append(path)
        else:
            raise ValueError('aggregate reference is outside the allowed source analysis paths')
    for claim in original_commit['payload']['claims']:
        for evidence in claim.get('evidence', []):
            path = Path(evidence['path']).resolve()
            # Worker evidence is already verified and mounted by _reuse_observations.
            if path not in read_only and not any(path.is_relative_to(root) for root in roots):
                raise ValueError('aggregate evidence is outside the allowed source analysis paths')
            if path not in read_only:
                read_only.append(path)
    provenance = dict(schema_version='analysis-reuse-v1', analysis_source=str(origin),
                      attribution_commit=original_commit['commit_id'], analysis_identity=job['analysis_identity'],
                      candidate_job_id=job['id'], read_only_paths=sorted({str(path) for path in read_only}))
    reuse_path = diag / 'analysis_reuse.json'
    job_path = diag / 'candidate_job.json'
    st.save_new(reuse_path, provenance)
    st.save_new(job_path, job)
    (diag / 'attribution/work').mkdir(parents=True, exist_ok=True)
    a_path = diag / 'attribution.json'
    st.save_new(a_path, original_commit['payload'])
    refs = [a_path, diag / 'inputs.json', index_path, inventory_path, reuse_path, job_path,
            origin / 'records/commits' / (original_commit['commit_id'] + '.json')]
    refs += [Path(ref['original_path']) for ref in original_commit['references']]
    commit = write_stage_commit(diag / 'records', 'attribution', original_commit['payload'],
                               references=refs, parent_ids=parent_ids,
                               metadata={**original_commit['metadata'], **metadata,
                                         'analysis_reused_from': str(origin),
                                         'source_attribution_commit': original_commit['commit_id']})
    pointer = diag / 'attribution_commit.json'
    st.save_new(pointer, {'commit_id': commit['commit_id']})
    _load_observations(diag, diag / 'records', commit)
    return commit
