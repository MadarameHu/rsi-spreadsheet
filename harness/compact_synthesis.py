"""Deterministic enrichment of compact semantic model outputs.

Raw model outputs are kept separately by the caller. These functions add only
identities, resolved references and declared-field provenance; they never infer
missing judgments or repair labels, text, JSON, or evidence.
"""
from copy import deepcopy
from pathlib import Path
import re

try:
    from .parallel_records import CLAIM_KINDS
except ImportError:
    from parallel_records import CLAIM_KINDS

DESIGN_SCHEMA = 'compact-design-derived-v1'
EVIDENCE_STRENGTHS = ('replay_verified', 'hypothesis_only')


def _object(value, where, allowed=None, required=()):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object')
    if allowed is not None and set(value) - set(allowed):
        raise ValueError(f'{where}: unknown fields {sorted(set(value) - set(allowed))}')
    if set(required) - set(value):
        raise ValueError(f'{where}: missing fields {sorted(set(required) - set(value))}')
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where} must be nonempty text')
    return value


def _list(value, where, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f'{where} must be a {"nonempty " if nonempty else ""}list')
    return value


def _texts(value, where, nonempty=False, unique=False):
    values = _list(value, where, nonempty)
    for item in values:
        _text(item, where + ' item')
    if unique and len(values) != len(set(values)):
        raise ValueError(f'{where}: duplicate reference')
    return deepcopy(values)


def _evidence(value, where, derived=False):
    result = []
    for item in _list(value, where):
        _object(item, where + ' item', {'path', 'locator'} | ({'ref'} if derived else set()), {'path', 'locator'})
        if 'ref' in item:
            _text(item['ref'],where+'.ref')
        _text(item['path'], where + '.path')
        _text(item['locator'], where + '.locator')
        if not Path(item['path']).is_absolute():
            raise ValueError(f'{where}.path must be absolute')
        result.append(deepcopy(item))
    return result


def _uncertainty(item):
    if 'uncertainty' not in item:
        return [], 'not_declared'
    if item['uncertainty'] is None:
        return [], 'explicit_null'
    return [_text(item['uncertainty'], 'uncertainty')], 'declared'


def _sample_id(observation):
    # IDs encode provenance, not a semantic assessment or a vote.
    match = re.fullmatch(r's(\d+)-.+', observation['id'])
    encoded = int(match.group(1)) if match else None
    if 'sample_id' in observation:
        explicit = observation['sample_id']
        if type(explicit) is not int or explicit < 0:
            raise ValueError('observation.sample_id must be a nonnegative integer')
        if encoded is not None and explicit != encoded:
            raise ValueError('observation.sample_id disagrees with its id')
        return explicit
    if encoded is None:
        raise ValueError('observation has no resolvable sample identity')
    return encoded


def enrich_aggregate(raw, observations):
    """Resolve compact claims against a complete, committed observation list."""
    _object(raw, 'aggregate', {'claims', 'training_goals'}, {'claims'})
    known = {}
    for observation in _list(observations, 'observations'):
        _object(observation, 'observation')
        identifier = _text(observation.get('id'), 'observation.id')
        if identifier in known:
            raise ValueError('duplicate observation id')
        known[identifier] = observation
    claims = []
    for index, item in enumerate(_list(raw['claims'], 'claims', True), 1):
        required = {'kind', 'finding', 'conditions', 'supports', 'counterevidence', 'evidence_strength'}
        _object(item, 'claim', required | {'uncertainty', 'evidence', 'library_class'}, required)
        if not isinstance(item['kind'], str) or item['kind'] not in CLAIM_KINDS:
            raise ValueError('claim.kind is invalid')
        if item['evidence_strength'] not in EVIDENCE_STRENGTHS:
            raise ValueError('claim.evidence_strength must be replay_verified or hypothesis_only')
        library_class = item.get('library_class')
        if library_class is not None and (not isinstance(library_class, str) or not library_class.strip()):
            raise ValueError('claim.library_class must be a library class id or null')
        supports = _texts(item['supports'], 'claim.supports', True, True)
        counter = _texts(item['counterevidence'], 'claim.counterevidence', unique=True)
        for identifier in supports + counter:
            if identifier not in known:
                raise ValueError(f'unknown observation reference {identifier}')
        refs = []
        for identifier in supports + counter:
            refs.extend(_evidence(known[identifier].get('evidence'), 'observation.evidence', derived=True))
        refs.extend(_evidence(item.get('evidence', []), 'claim.evidence'))
        evidence, seen = [], set()
        for reference in refs:
            key = (reference['path'], reference['locator'])
            if key not in seen:
                evidence.append(reference)
                seen.add(key)
        if not evidence:
            raise ValueError('claim must resolve to nonempty evidence')
        uncertainty, status = _uncertainty(item)
        samples = list(dict.fromkeys(_sample_id(known[key]) for key in supports))
        claims.append({
            'id': f'c{index}', 'kind': item['kind'],
            'statement': _text(item['finding'], 'claim.finding'),
            'conditions': _texts(item['conditions'], 'claim.conditions'),
            'observation_ids': supports, 'counterevidence_ids': counter,
            'evidence': evidence, 'uncertainties': uncertainty,
            'uncertainty_status': status, 'evidence_strength': item['evidence_strength'],
            **({'library_class': library_class} if 'library_class' in item else {}),
            'support_sample_ids': samples, 'support_sample_count': len(samples),
        })
    result = {'schema_version': 'parallel-attribution-v1', 'claims': claims,
              'derivation': 'compact-aggregate-v1'}
    if 'training_goals' in raw:
        goals = []
        fields = {'claim_refs', 'ability', 'rationale', 'success_signal', 'difficulty_controls'}
        for index, item in enumerate(_list(raw['training_goals'], 'training_goals'), 1):
            _object(item, 'training goal', fields, fields)
            refs = _list(item['claim_refs'], 'training goal.claim_refs', True)
            if any(type(ref) is not int or not 1 <= ref <= len(claims) for ref in refs):
                raise ValueError('training goal.claim_refs must be valid 1-based claim positions')
            if len(refs) != len(set(refs)):
                raise ValueError('training goal.claim_refs: duplicate reference')
            goals.append({
                'id': f'g{index}', 'claim_ids': [claims[ref - 1]['id'] for ref in refs],
                **{field: deepcopy(item[field]) for field in
                   ('ability', 'rationale', 'success_signal', 'difficulty_controls')},
            })
        result['training_goals'] = goals
        validate_training_goals(result)
    return result


def _claim_list(claims):
    if isinstance(claims, dict):
        claims = claims.get('claims')
    known = {}
    for claim in _list(claims, 'claims', True):
        _object(claim, 'claim')
        identifier = _text(claim.get('id'), 'claim.id')
        if identifier in known:
            raise ValueError('duplicate claim id')
        known[identifier] = claim
    return known


SECONDARY = re.compile(r'SECONDARY: appeared in (\d+) of (\d+) attempts and determined the score in '
                       r'(\d+); (\d+) unattributed\.')


def validate_training_goals(attribution, trace_summary=None):
    """Validate declared plans only; absence is not an inferred empty plan.

    trace_summary: the host score trace summary, when one was computed. It gives
    the true attempt count and the FEEDBACK: incidence the model must match.
    """
    if 'training_goals' not in attribution:
        return
    known = _claim_list(attribution)
    goals = _list(attribution['training_goals'], 'training_goals')
    for index, goal in enumerate(goals, 1):
        rationale = goal.get('rationale') if isinstance(goal, dict) else None
        prefix = rationale.split(':')[0] + ':' if isinstance(rationale, str) and ':' in rationale else ''
        if prefix == 'SECONDARY:' and index == 1:
            raise ValueError('the first training goal cannot be SECONDARY:')
    if trace_summary is not None:
        from score_trace import feedback_samples
        attempts = trace_summary.get('attempts')
        unattributed = len(trace_summary.get('not_reproduced') or [])
        feedback = feedback_samples(trace_summary)
        declared = [goal for goal in goals if isinstance(goal, dict)
                    and isinstance(goal.get('rationale'), str) and goal['rationale'].startswith('FEEDBACK:')]
        if declared and len(feedback) < 2:
            raise ValueError('FEEDBACK: requires at least two qualifying traces; '
                             f'only {len(feedback)} found')
        if len(declared) > 1:
            raise ValueError('at most one FEEDBACK: goal is allowed')
        for goal in goals:
            rationale = goal.get('rationale') if isinstance(goal, dict) else None
            if not isinstance(rationale, str) or not rationale.startswith('SECONDARY:'):
                continue
            # The prompt asks the rationale to start with this sentence, not to be only it.
            match = SECONDARY.match(rationale)
            if match is None:
                raise ValueError('SECONDARY: rationale must start with '
                                 '"SECONDARY: appeared in N of M attempts and determined the score in K; U unattributed."')
            n, m, k, u = (int(value) for value in match.groups())
            if not 0 <= k <= n <= m:
                raise ValueError('SECONDARY: requires 0 <= K <= N <= M')
            if m != attempts or u != unattributed:
                raise ValueError(f'SECONDARY: counts must use the score trace facts '
                                 f'(M={attempts}, U={unattributed})')
    fields = {'id', 'claim_ids', 'ability', 'rationale', 'success_signal', 'difficulty_controls'}
    for index, goal in enumerate(goals, 1):
        _object(goal, 'training goal', fields, fields)
        if goal['id'] != f'g{index}':
            raise ValueError('training goal IDs must follow declared priority order')
        ids = _texts(goal['claim_ids'], 'training goal.claim_ids', True, True)
        if any(identifier not in known for identifier in ids):
            raise ValueError('training goal references an unknown claim')
        if not any(known[identifier].get('kind') in {'defect', 'capability'} for identifier in ids):
            raise ValueError('training goal requires a defect or capability claim')
        for field in ('ability', 'rationale', 'success_signal'):
            _text(goal[field], 'training goal.' + field)
        _texts(goal['difficulty_controls'], 'training goal.difficulty_controls')
    # Only aggregates that declare evidence strength are held to it; legacy ones are read as-is.
    if goals and any('evidence_strength' in claim for claim in known.values()):
        if not any(known[i].get('evidence_strength') == 'replay_verified' for i in goals[0]['claim_ids']):
            raise ValueError('the first training goal must reference a replay_verified claim')


def validate_goal_selection(design, attribution, target_ids):
    """Require a grounded primary choice when upstream actually declared goals."""
    goals = attribution.get('training_goals', []) if isinstance(attribution, dict) else []
    selected = 'primary_goal_id' in design
    reason = 'selection_reason' in design
    if selected != reason:
        raise ValueError('primary_goal_id and selection_reason must be declared together')
    if not selected:
        if goals:
            raise ValueError('design must select a primary training goal')
        return
    identifier = _text(design['primary_goal_id'], 'primary_goal_id')
    _text(design['selection_reason'], 'selection_reason')
    if isinstance(attribution, dict):
        validate_training_goals(attribution)
    goal = next((goal for goal in goals if goal['id'] == identifier), None)
    if goal is None:
        raise ValueError('design references an unknown primary training goal')
    known = _claim_list(attribution)
    grounded_targets = {ref for ref in goal['claim_ids']
                        if known[ref].get('kind') in {'defect', 'capability'}}
    if not grounded_targets.intersection(target_ids):
        raise ValueError('design targets must exercise its primary training goal')


def enrich_design(raw, claims):
    """Create mappings for selected targets without inventing exclusion reasons."""
    _object(raw, 'design', {'targets', 'preserves', 'simplifies',
                           'primary_goal_id', 'selection_reason'}, {'targets'})
    known = _claim_list(claims)
    optional = {}
    for field in ('preserves', 'simplifies'):
        # None in the derived artifact means absent, never "nothing to preserve".
        optional[field] = _texts(raw[field], field) if field in raw else None
        optional[field + '_status'] = 'declared' if field in raw else 'not_declared'
    mappings, selected = [], set()
    for index, target in enumerate(_list(raw['targets'], 'targets', True), 1):
        fields = {'claim_id', 'feature', 'test_intent'}
        _object(target, 'target', fields, fields)
        identifier = _text(target['claim_id'], 'target.claim_id')
        if identifier not in known:
            raise ValueError(f'unknown target claim {identifier}')
        if identifier in selected:
            raise ValueError('duplicate target claim')
        if known[identifier].get('kind') not in {'defect', 'capability'}:
            raise ValueError('target must reference a defect or capability')
        selected.add(identifier)
        mappings.append({
            'id': f'm{index}', 'claim_id': identifier, 'role': 'target',
            'feature': _text(target['feature'], 'target.feature'),
            'check_plan': _text(target['test_intent'], 'target.test_intent'),
        })
    validate_goal_selection(raw, claims, selected)
    plan = {field: raw[field] for field in ('primary_goal_id', 'selection_reason') if field in raw}
    return {'schema_version': DESIGN_SCHEMA, 'mappings': mappings, **plan,
            **optional, 'unselected_claim_ids': [key for key in known if key not in selected]}


def normalize_design(raw, claims):
    """Convert compact model output; preserve canonical/legacy artifacts verbatim.

    The caller must still invoke the appropriate canonical validator. This is
    not a fallback that accepts a malformed compact output as a legacy design.
    """
    _object(raw, 'design')
    if 'targets' in raw:
        return enrich_design(raw, claims)
    return deepcopy(raw)
