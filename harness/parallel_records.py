"""Structural contracts for atomic observations and their explicit synthesis.

No semantic verdict, normalization, inference, or model invocation occurs here.
Membership is checked against complete committed observation inventories supplied
by the caller. Evidence files and every declared uncertainty remain unchanged.
"""
from pathlib import Path

try:
    from .stage_records import _entries, _evidence, _list, _object, _text
except ImportError:
    from stage_records import _entries, _evidence, _list, _object, _text

OBSERVATIONS_SCHEMA = 'trajectory-observations-v1'
AGGREGATE_SCHEMA = 'parallel-attribution-v1'
CLAIM_KINDS = frozenset({'defect', 'capability', 'task_issue',
                         'infrastructure_issue', 'uncertain'})


def _texts(value, where):
    for text in _list(value, where):
        _text(text, where + ' item')


def _absolute_evidence(value, where, base_dir):
    _evidence(value, where, base_dir, nonempty=True)
    for item in value:
        if not Path(item['path']).is_absolute():
            raise ValueError(f'{where}: parallel evidence paths must be absolute')


def validate_observations(payload, *, base_dir=None):
    """Check one worker's sample identity and complete atomic observations."""
    _object(payload, 'observations payload')
    if payload.get('schema_version') != OBSERVATIONS_SCHEMA:
        raise ValueError('invalid observations schema_version')
    sample = payload.get('sample_id')
    if type(sample) is not int or sample < 0:
        raise ValueError('sample_id must be a nonnegative integer')
    if 'evidence_limitations' in payload:
        _texts(payload['evidence_limitations'], 'evidence_limitations')
    prefix = f's{sample:02d}-'
    for observation in _entries(payload.get('observations'), 'observations'):
        identifier = observation['id']
        if not identifier.startswith(prefix) or not identifier[len(prefix):].strip():
            raise ValueError(f'observation id must have sample prefix {prefix}')
        _text(observation.get('statement'), 'observation.statement')
        _absolute_evidence(observation.get('evidence'), 'observation.evidence', base_dir)
        _texts(observation.get('uncertainties'), 'observation.uncertainties')


def validate_aggregate_structure(payload, *, base_dir=None, trace_summary=None):
    """Check aggregate structure; caller separately supplies committed membership."""
    _object(payload, 'parallel attribution')
    if payload.get('schema_version') != AGGREGATE_SCHEMA:
        raise ValueError('invalid parallel attribution schema_version')
    for claim in _entries(payload.get('claims'), 'claims', nonempty=True):
        if not isinstance(claim.get('kind'), str) or claim['kind'] not in CLAIM_KINDS:
            raise ValueError('claim.kind is invalid')
        _text(claim.get('statement'), 'claim.statement')
        for field in ('observation_ids', 'counterevidence_ids'):
            ids = _list(claim.get(field), 'claim.' + field,
                        nonempty=(field == 'observation_ids'))
            for identifier in ids:
                _text(identifier, 'claim.' + field + ' item')
            if len(ids) != len(set(ids)):
                raise ValueError(f'claim.{field}: duplicate observation id')
        _texts(claim.get('conditions'), 'claim.conditions')
        _texts(claim.get('uncertainties'), 'claim.uncertainties')
        if 'evidence_strength' in claim and claim['evidence_strength'] not in ('replay_verified', 'hypothesis_only'):
            raise ValueError('claim.evidence_strength is invalid')
        _absolute_evidence(claim.get('evidence'), 'claim.evidence', base_dir)
    try:
        from .compact_synthesis import validate_training_goals
    except ImportError:
        from compact_synthesis import validate_training_goals
    validate_training_goals(payload, trace_summary)


def validate_aggregate(payload, observations, *, base_dir=None, trace_summary=None):
    """Resolve both supporting and contradicting IDs against all worker commits."""
    validate_aggregate_structure(payload, base_dir=base_dir, trace_summary=trace_summary)
    known = {item['id'] for item in _entries(observations, 'committed observations')}
    for claim in payload['claims']:
        for field in ('observation_ids', 'counterevidence_ids'):
            missing = set(claim[field]) - known
            if missing:
                raise ValueError(f'claim.{field}: unknown observation ids {sorted(missing)}')
