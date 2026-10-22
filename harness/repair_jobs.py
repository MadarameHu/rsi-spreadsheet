"""Explicit candidate revisions from verified semantic rejections, never retries."""
from copy import deepcopy
from pathlib import Path
import re

from stage_audit import read_json, sha
from staged_synth import save_new


def _require_candidate_rejection(quality):
    """Route existing gate issues without reinterpreting any audit judgment."""
    issues = quality['issues']
    upstream = [item['id'] for item in issues
                if item['id'].startswith(('claim:', 'observation:'))
                or (item['id'].startswith('mapping:')
                    and item['id'].endswith(':upstream_verdict'))]
    if upstream:
        raise ValueError('candidate rewriting cannot repair upstream evidence or attribution; '
                         'a new attribution version is required: ' + ', '.join(upstream))
    # Only the existing gate's candidate checks identify author-repairable work.
    # Faults are rejected by the callers before this routing check.
    def candidate_issue(identifier):
        return (identifier in ('task:spec', 'task:ground_truth', 'task:reference',
                               'reference_execution', 'selected_design')
                or identifier.startswith('probe:')
                or (identifier.startswith('mapping:')
                    and identifier.endswith(':implementation_verdict')))
    unsupported = [item['id'] for item in issues if not candidate_issue(item['id'])]
    if unsupported:
        raise ValueError('candidate rewriting has no supported repair route for gate issues: '
                         + ', '.join(unsupported))


def validate_repair(job):
    from quality_gate import evaluate_quality
    from stage_records import verify_snapshot
    repair = job.get('repair')
    if repair is None:
        return []
    if not isinstance(repair, dict) or repair.get('schema_version') != 'candidate-repair-v1':
        raise ValueError('invalid candidate repair provenance')
    parent = Path(repair['parent_episode'])
    if not parent.is_absolute():
        raise ValueError('repair parent must be an absolute episode path')
    feedback = Path(repair['feedback_dir'])
    for name, digest in repair['feedback_hashes'].items():
        path = feedback / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('repair feedback differs from its frozen version')
    old_job = read_json(parent / 'diagnostics/candidate_job.json')
    if (old_job['id'] != repair['parent_job_id'] or old_job['original'] != job['original']
            or old_job['analysis_identity'] != job['analysis_identity']
            or old_job['goal_id'] != job['goal_id']):
        raise ValueError('repair no longer belongs to its original job and training goal')
    audit = read_json(feedback / 'audit.json')
    checked = evaluate_quality(parent, audit=audit)
    if checked != read_json(feedback / 'quality.json') or checked['status'] != 'rejected':
        raise ValueError('repair requires the exact verified semantic rejection')
    if any(issue['verdict'] == 'fault' for issue in checked['issues']):
        raise ValueError('infrastructure faults require recovery, not candidate rewriting')
    _require_candidate_rejection(checked)
    if (checked['audit_commit'] != repair['audit_commit']
            or checked['candidate_sha256'] != repair['candidate_sha256']):
        raise ValueError('repair rejection identity changed')
    candidate = verify_snapshot(checked['candidate_manifest'])
    # evaluate_quality has already verified this live design against its commit.
    readonly = [feedback, Path(candidate['snapshot_path']), Path(audit['work_dir']),
                parent / 'diagnostics/design.json']
    # Rejections can cite reference execution or shared source reviews outside
    # the candidate audit directory. Expose their verified evidence as well.
    for item in checked['checks']:
        for evidence in item.get('evidence', []):
            if isinstance(evidence, dict) and evidence.get('path'):
                path = Path(evidence['path'])
                if path.is_file():
                    readonly.append(path)
    reference = checked.get('reference_execution') or {}
    if reference.get('result_path'):
        readonly.append(Path(reference['result_path']).parent)
    frozen_reference = parent / 'diagnostics/validation/reference-case'
    if frozen_reference.is_dir():
        readonly.append(frozen_reference)
    for files in audit.get('provenance', {}).get('review_files', {}).values():
        readonly.append(Path(files['terminal_path']).parent)
    source = audit.get('provenance', {}).get('source_receipt')
    if source:
        record = read_json(source['receipt_path'])
        manifest = read_json(record['manifest_path'])
        readonly += [Path(record['manifest_path']).parent,
                     *[Path(v['snapshot_path']) for v in manifest['snapshots'].values()]]
    unique = sorted(set(path.resolve(strict=True) for path in readonly))
    return [p for p in unique if not any(parent.is_dir() and p != parent and p.is_relative_to(parent)
                                        for parent in unique)]


def prepare_repair_joblist(cfg, parent_episode, path):
    """Append one new version after its parent so the episode number is stable."""
    from candidate_jobs import validate_job, _digest, LIST_SCHEMA
    from quality_gate import evaluate_quality
    from host_validation import locked, write
    parent = Path(parent_episode).resolve(strict=True)
    job = read_json(parent / 'diagnostics/candidate_job.json')
    validate_job(job, cfg, original=job['original'])
    audit = read_json(parent / 'diagnostics/audit.json')
    quality = evaluate_quality(parent, audit=audit)
    if quality['status'] != 'rejected' or any(i['verdict'] == 'fault' for i in quality['issues']):
        raise ValueError('repair requires a verified semantic rejection without infrastructure faults')
    _require_candidate_rejection(quality)
    feedback = parent / 'diagnostics/repair-inputs' / quality['audit_commit']
    save_new(feedback / 'quality.json', quality)
    save_new(feedback / 'audit.json', audit)
    revised = deepcopy(job)
    revised.pop('id')
    revised['variant_index'] += 1
    revised['repair'] = {'schema_version': 'candidate-repair-v1',
                         'parent_episode': str(parent), 'parent_job_id': job['id'],
                         'candidate_sha256': quality['candidate_sha256'],
                         'audit_commit': quality['audit_commit'], 'feedback_dir': str(feedback),
                         'feedback_hashes': {name: sha(feedback / name) for name in ('quality.json', 'audit.json')}}
    revised['id'] = _digest(revised)
    validate_job(revised, cfg, original=job['original'])
    match = re.fullmatch(r'ep(\d+)', parent.name)
    if not match:
        raise ValueError('repair parent must have a numbered episode directory')
    parent_number = int(match[1])
    # Reserve a stable new episode after every existing/planned sibling. This
    # also supports a second repair and parents other than ep01.
    with locked(parent.parent / '.repair-episodes.lock'):
        reservations_path = parent.parent / '.repair-episodes.json'
        reservations = read_json(reservations_path) if reservations_path.exists() else {}
        assigned = reservations.get(revised['id'])
        if assigned is None:
            occupied = [parent_number, *reservations.values()]
            occupied += [int(m[1]) for p in parent.parent.glob('ep*')
                         if (m := re.fullmatch(r'ep(\d+)', p.name))]
            for planned in parent.parents[2].glob('training-plan-*.json'):
                listing = read_json(planned)['joblist']
                count = 0
                for item in listing['jobs']:
                    if item['original'] == job['original']:
                        count += 1
                        occupied.append(listing.get('episode_assignments', {}).get(item['id'], count))
            assigned = max(occupied) + 1
            reservations[revised['id']] = assigned
            write(reservations_path, reservations)
        manifest = {'schema_version': LIST_SCHEMA, 'base_config_sha256': job['base_config_sha256'],
                    'jobs': [job, revised],
                    'episode_assignments': {job['id']: parent_number, revised['id']: assigned}}
        manifest['joblist_id'] = _digest(manifest)
        save_new(Path(path), manifest)
    return manifest
