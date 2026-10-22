"""Immutable diagnostics, without inference, correction, retry, or gold labels.

Public schemas (all IDs and prose fields are nonblank strings):
* attribution: claims[{id, statement, observed_as: failure|detour|handled|
  spec_ambiguity, evidence:[{path, locator}], uncertainties:[string]}].
  Parallel attribution and the observations stage use the explicit schemas in
  parallel_records; aggregate references must resolve to observation parents.
* design: mappings[{id, claim_id, feature, check_plan}]. All upstream claims
  must have at least one mapping; an explicit noncoverage feature/check_plan
  can record a deliberate exclusion instead of silently dropping a claim.
* audit: claims[{id, verdict, reason, evidence}], mappings[{id,
  upstream_verdict, implementation_verdict, reason, evidence}],
  task_checks[{id, verdict, reason, evidence}], probes[{id, expected_valid,
  actual_accept, execution_status, reason, evidence}]. Verdicts are
  correct|incorrect|uncertain. Probe validity is bool|null, not a score;
  completed requires a boolean actual_accept; infra_error/missing require
  null actual_accept. Evidence lists may be empty in an audit, but never in
  attribution. The schema checks provenance structure, not semantic truth.

expected_units contains claims/mappings/task_checks/probes and optionally the
parallel observations group, each a list of expected IDs. Every declared group
must match exactly. No missing
unit becomes zero. Additional payload fields are retained, never rewritten.
Evidence paths resolve against base_dir (cwd by default); locators describe
an actual event, cell range, or other precise location. All referenced files
must exist. No evidence content is shortened or interpreted by this module.

snapshot_tree copies EVERY regular file and directory, including dotfiles and
empty directories, directly into destination. It returns an external manifest
and does not insert any files into the candidate. Symlinks and special files
are explicitly rejected. Both source and destination trees are checked after
copying. A failed copy is retained as an incomplete diagnostic, never reused.

write_stage_commit stores exact reference bytes in artifacts/<sha256> and a
content-addressed immutable envelope in commits/<commit_id>.json. Payloads
are preserved as provided; attribution/design are author claims, audits are
machine judgments, and none is human gold. metadata belongs to the caller
(e.g. run_id/source_task/model/candidate_version); no values are inferred.
read_stage_commit verifies envelope, archived files and parent commits. It
uses archived bytes, so later changes to original live files do not rewrite
history. Consumers must not use live evidence paths in place of these bytes.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

VERDICTS = frozenset({'correct', 'incorrect', 'uncertain'})
OBSERVED_AS = frozenset({'failure', 'detour', 'handled', 'spec_ambiguity'})
AUDIT_GROUPS = ('claims', 'mappings', 'task_checks', 'probes')
STAGES = frozenset({'observations', 'attribution', 'design', 'audit'})


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object')
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where} must be a nonblank string')
    return value


def _list(value, where, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f'{where} must be a {"nonempty " if nonempty else ""}list')
    return value


def _entries(value, where, nonempty=False):
    entries = _list(value, where, nonempty)
    seen = set()
    for item in entries:
        _object(item, where)
        identifier = _text(item.get('id'), f'{where}.id')
        if identifier in seen:
            raise ValueError(f'{where}: duplicate id {identifier!r}')
        seen.add(identifier)
    return entries


def _file(path, base_dir=None):
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError('reference path must be a nonblank path')
    raw = _text(os.fspath(path), 'reference path')
    p = Path(raw)
    if not p.is_absolute():
        p = Path(base_dir or Path.cwd()) / p
    if not p.is_file():
        raise ValueError(f'reference file does not exist: {p}')
    return p.absolute()


def _evidence(value, where, base_dir=None, nonempty=False):
    for item in _list(value, where, nonempty):
        _object(item, where)
        _text(item.get('path'), f'{where}.path')
        _text(item.get('locator'), f'{where}.locator')
        path = _file(item['path'], base_dir)
        if 'sha256' in item:
            if item['sha256'] != _sha(path.read_bytes()):
                raise ValueError(f'{where}: reference hash mismatch: {path}')


def validate_attribution(payload, *, base_dir=None):
    """Raise ValueError on invalid structure/references; never modify payload."""
    _object(payload, 'attribution')
    if payload.get('schema_version') == 'parallel-attribution-v1':
        try:
            from .parallel_records import validate_aggregate_structure
        except ImportError:
            from parallel_records import validate_aggregate_structure
        validate_aggregate_structure(payload, base_dir=base_dir)
        return
    for claim in _entries(payload.get('claims'), 'claims', nonempty=True):
        _text(claim.get('statement'), 'claim.statement')
        if not isinstance(claim.get('observed_as'), str) or claim['observed_as'] not in OBSERVED_AS:
            raise ValueError('claim.observed_as is invalid')
        _evidence(claim.get('evidence'), 'claim.evidence', base_dir, nonempty=True)
        for item in _list(claim.get('uncertainties'), 'claim.uncertainties'):
            _text(item, 'claim.uncertainties item')


def _claim_entries(claims):
    return _entries(claims.get('claims') if isinstance(claims, dict) else claims,
                    'upstream claims', nonempty=True)


def validate_design(payload, claims, *, base_dir=None):
    """Validate mappings and complete, explicit coverage of upstream claim IDs."""
    _object(payload, 'design')
    source_claims = {c['id']: c for c in _claim_entries(claims)}
    known = set(source_claims)
    parallel = isinstance(claims, dict) and claims.get('schema_version') == 'parallel-attribution-v1'
    compact = payload.get('schema_version') == 'compact-design-derived-v1'
    if compact:
        for field in ('preserves','simplifies'):
            status = payload.get(field+'_status')
            if status == 'not_declared':
                if payload.get(field) is not None:
                    raise ValueError('undeclared design field must remain null')
            elif status == 'declared':
                for item in _list(payload.get(field), field):
                    _text(item,field+' item')
            else:
                raise ValueError('missing design declaration status')
    targets = 0
    mapped = set()
    for mapping in _entries(payload.get('mappings'), 'mappings', nonempty=True):
        identifier = _text(mapping.get('claim_id'), 'mapping.claim_id')
        if identifier not in known:
            raise ValueError(f'mapping references unknown claim id {identifier!r}')
        mapped.add(identifier)
        _text(mapping.get('feature'), 'mapping.feature')
        _text(mapping.get('check_plan'), 'mapping.check_plan')
        if parallel:
            role = mapping.get('role')
            if not isinstance(role, str) or role not in {'target', 'support', 'preserve', 'exclude'}:
                raise ValueError('mapping.role is invalid')
            if not compact:
                for field in ('preserves', 'simplifies'):
                    for item in _list(mapping.get(field), 'mapping.' + field):
                        _text(item, 'mapping.' + field + ' item')
            if role == 'exclude':
                _text(mapping.get('exclusion_reason'), 'mapping.exclusion_reason')
            if role == 'target':
                if source_claims[identifier].get('kind') not in {'defect', 'capability'}:
                    raise ValueError('target mapping requires a defect or capability claim')
                targets += 1
    compact = payload.get('schema_version') == 'compact-design-derived-v1'
    if compact:
        if not parallel:
            raise ValueError('compact design requires parallel attribution')
        expected_unselected = [c['id'] for c in _claim_entries(claims) if c['id'] not in mapped]
        if payload.get('unselected_claim_ids') != expected_unselected:
            raise ValueError('unselected claim inventory differs from selected targets')
        if any(m.get('role') != 'target' for m in payload['mappings']):
            raise ValueError('compact design mappings must be selected targets')
    if not compact and known - mapped:
        raise ValueError(f'missing claim mappings: {sorted(known - mapped)}')
    if parallel and not targets:
        raise ValueError('parallel design requires at least one target mapping')
    if parallel:
        try:
            from .compact_synthesis import validate_goal_selection
        except ImportError:
            from compact_synthesis import validate_goal_selection
        validate_goal_selection(payload, claims,
                                {m['claim_id'] for m in payload['mappings'] if m.get('role') == 'target'})


def _expected(expected_units):
    _object(expected_units, 'expected_units')
    if set(expected_units) not in (set(AUDIT_GROUPS), set(AUDIT_GROUPS) | {'observations'}):
        raise ValueError(f'expected_units must contain {AUDIT_GROUPS}, optionally observations')
    result = {}
    for group in expected_units:
        values = _list(expected_units[group], f'expected_units.{group}')
        for value in values:
            _text(value, f'expected_units.{group} id')
        if len(set(values)) != len(values):
            raise ValueError(f'expected_units.{group}: duplicate id')
        result[group] = set(values)
    return result


def validate_audit(payload, expected_units, *, base_dir=None):
    """Require exact unit inventory; separate upstream and implementation verdicts."""
    _object(payload, 'audit')
    expected = _expected(expected_units)
    if ('observations' in payload) != ('observations' in expected):
        raise ValueError('audit observations group must match expected_units')
    for group in expected:
        entries = _entries(payload.get(group), group)
        actual = {item['id'] for item in entries}
        if actual != expected[group]:
            raise ValueError(f'{group}: missing ids {sorted(expected[group] - actual)}; '
                             f'unknown ids {sorted(actual - expected[group])}')
        for item in entries:
            _text(item.get('reason'), f'{group}.reason')
            _evidence(item.get('evidence'), f'{group}.evidence', base_dir)
            if group == 'probes':
                for key in ('expected_valid', 'actual_accept'):
                    if key not in item or (item[key] is not None and type(item[key]) is not bool):
                        raise ValueError(f'probe.{key} must be an explicit boolean or null')
                status = item.get('execution_status')
                if not isinstance(status, str) or status not in {'completed', 'infra_error', 'missing'}:
                    raise ValueError('probe.execution_status is invalid')
                if status == 'completed' and type(item['actual_accept']) is not bool:
                    raise ValueError('completed probe requires boolean actual_accept')
                if status != 'completed' and item['actual_accept'] is not None:
                    raise ValueError('noncompleted probe requires null actual_accept')
            else:
                keys = ('upstream_verdict', 'implementation_verdict') if group == 'mappings' else ('verdict',)
                for key in keys:
                    if not isinstance(item.get(key), str) or item[key] not in VERDICTS:
                        raise ValueError(f'{group}.{key} is invalid')


def _tree_entries(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'snapshot source must be a regular directory: {root}')
    entries = []
    def visit(directory):
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f'snapshot does not support symlinks: {path}')
            if stat.S_ISDIR(info.st_mode):
                entries.append({'path': relative, 'type': 'directory'})
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                raw = path.read_bytes()
                entries.append({'path': relative, 'type': 'file', 'sha256': _sha(raw), 'bytes': len(raw)})
            else:
                raise ValueError(f'snapshot does not support special files: {path}')
    visit(root)
    return sorted(entries, key=lambda item: item['path'])


def snapshot_tree(path, destination):
    """Copy a full stable tree into a NEW destination; return manifest externally."""
    source = Path(path).absolute()
    target = Path(destination).absolute()
    if source.is_symlink():
        raise ValueError('snapshot source cannot be a symlink')
    if target.resolve() == source.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError('snapshot destination must be outside the source tree')
    before = _tree_entries(source)
    target.mkdir(parents=True, exist_ok=False)
    for item in before:
        to = target / item['path']
        if item['type'] == 'directory':
            to.mkdir(parents=True, exist_ok=True)
        else:
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item['path'], to)
    after = _tree_entries(source)
    copied = _tree_entries(target)
    if before != after or before != copied:
        raise ValueError('source changed during snapshot or copied tree differs; incomplete destination retained')
    return {'schema_version': '1.0', 'source_path': str(source), 'snapshot_path': str(target),
            'entries': before, 'tree_sha256': _sha(_encoded(before))}


def verify_snapshot(manifest_or_path):
    """Verify the complete snapshot, including added/removed files and directories."""
    manifest = manifest_or_path if isinstance(manifest_or_path, dict) else _load_json(Path(manifest_or_path).read_bytes())
    _object(manifest, 'snapshot manifest')
    if manifest.get('schema_version') != '1.0':
        raise ValueError('unsupported snapshot schema')
    entries = manifest.get('entries')
    if manifest.get('tree_sha256') != _sha(_encoded(entries)):
        raise ValueError('snapshot manifest hash mismatch')
    if _tree_entries(Path(manifest['snapshot_path'])) != entries:
        raise ValueError('snapshot tree hash mismatch')
    return manifest


def _load_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


@contextmanager
def _lock(directory, exclusive):
    with (directory / '.stage-records.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _store(path, raw):
    """Create once; identical content is idempotent, collisions/corruption fail."""
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f'immutable artifact mismatch: {path}')
        return
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_commit(directory, commit_id, ancestors=None):
    if not isinstance(commit_id, str) or not re.fullmatch(r'[0-9a-f]{64}', commit_id):
        raise ValueError('invalid commit id')
    path = directory / 'commits' / (commit_id + '.json')
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'commit file does not exist: {commit_id}')
    commit = _load_json(path.read_bytes())
    body = {k: v for k, v in commit.items() if k != 'commit_id'}
    if commit.get('commit_id') != commit_id or _sha(_encoded(body)) != commit_id:
        raise ValueError('commit hash mismatch')
    for ref in commit['references']:
        if ref['path'] != 'artifacts/' + ref['sha256']:
            raise ValueError('invalid archived reference path')
        archived = directory / ref['path']
        if not archived.is_file() or archived.is_symlink():
            raise ValueError('archived reference missing or symlink')
        raw = archived.read_bytes()
        if len(raw) != ref['bytes'] or _sha(raw) != ref['sha256']:
            raise ValueError('archived reference hash mismatch')
    ancestors = set() if ancestors is None else ancestors
    if commit_id in ancestors:
        raise ValueError('cyclic commit references')
    for parent_id in commit['parent_ids']:
        _read_commit(directory, parent_id, ancestors | {commit_id})
    return commit


def read_stage_commit(directory, commit_id):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'commit directory does not exist: {directory}')
    with _lock(directory, exclusive=False):
        return _read_commit(directory, commit_id)


def write_stage_commit(directory, stage, payload, *, references=(), parent_ids=(),
                       metadata=None, claims=None, expected_units=None, base_dir=None):
    """Validate then freeze payload and all evidence bytes; no payload normalization."""
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f'invalid stage: {stage}')
    # Round-trip creates an independent copy, retaining every provided field.
    payload = _load_json(_encoded(payload))
    metadata = {} if metadata is None else _load_json(_encoded(metadata))
    _object(metadata, 'metadata')
    if stage == 'observations':
        try:
            from .parallel_records import validate_observations
        except ImportError:
            from parallel_records import validate_observations
        validate_observations(payload, base_dir=base_dir)
    elif stage == 'attribution':
        validate_attribution(payload, base_dir=base_dir)
    elif stage == 'design':
        validate_design(payload, claims, base_dir=base_dir)
    else:
        validate_audit(payload, expected_units, base_dir=base_dir)
    parents = list(parent_ids)
    if len(parents) != len(set(parents)):
        raise ValueError('duplicate parent commit id')
    paths = [_file(p, base_dir) for p in references]
    for group in ('observations', 'claims', 'mappings', 'task_checks', 'probes'):
        for item in payload.get(group, []):
            for evidence in item.get('evidence', []):
                paths.append(_file(evidence['path'], base_dir))
    paths = sorted(set(paths), key=str)
    raw_files = {path: path.read_bytes() for path in paths}
    # Verify explicit evidence hashes against the bytes actually about to commit.
    for group in ('observations', 'claims', 'mappings', 'task_checks', 'probes'):
        for item in payload.get(group, []):
            for evidence in item.get('evidence', []):
                if 'sha256' in evidence and evidence['sha256'] != _sha(raw_files[_file(evidence['path'], base_dir)]):
                    raise ValueError('reference changed before commit')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory, exclusive=True):
        parent_commits = []
        for parent_id in parents:
            parent_commits.append(_read_commit(directory, parent_id))
        if stage == 'attribution' and payload.get('schema_version') == 'parallel-attribution-v1':
            if not parent_commits or any(c['stage'] != 'observations' for c in parent_commits):
                raise ValueError('parallel attribution requires observation commits as parents')
            samples = [c['payload']['sample_id'] for c in parent_commits]
            if len(samples) != len(set(samples)):
                raise ValueError('parallel attribution has duplicate parent sample_id')
            try:
                from .parallel_records import validate_aggregate
            except ImportError:
                from parallel_records import validate_aggregate
            observations = [item for c in parent_commits for item in c['payload']['observations']]
            validate_aggregate(payload, observations, base_dir=base_dir)
        (directory / 'artifacts').mkdir(exist_ok=True)
        (directory / 'commits').mkdir(exist_ok=True)
        archived = []
        for path, raw in raw_files.items():
            hash_value = _sha(raw)
            relative = 'artifacts/' + hash_value
            _store(directory / relative, raw)
            archived.append({'original_path': str(path), 'path': relative, 'sha256': hash_value, 'bytes': len(raw)})
        body = {'schema_version': '1.0', 'origin': 'explicit_stage_commit', 'is_human_gold': False,
                'stage': stage, 'payload': payload, 'parent_ids': parents,
                'references': archived, 'metadata': metadata}
        commit_id = _sha(_encoded(body))
        commit = dict(body, commit_id=commit_id)
        _store(directory / 'commits' / (commit_id + '.json'), _encoded(commit) + b'\n')
        return commit
