"""Compact output enrichment preserves semantics and exposes missing fields."""
import copy
from pathlib import Path
import tempfile
import unittest

from harness.compact_synthesis import (enrich_aggregate, enrich_design, normalize_design,
                                      validate_training_goals, validate_goal_selection)
from harness.parallel_records import validate_aggregate


class CompactSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'attempt.md'
        self.path.write_text('Full trajectory; no content clipping.')
        self.first = {'path': str(self.path), 'locator': 'event 3'}
        self.second = {'path': str(self.path), 'locator': 'events 4-20'}
        self.observations = [
            {'id': 's00-o1', 'evidence': [self.first]},
            {'id': 's00-o2', 'evidence': [self.first, self.second]},
            {'id': 's12-o1', 'sample_id': 12, 'evidence': [self.second]},
        ]
        self.raw = {'claims': [{
            'kind': 'defect', 'finding': '  No continuation after a truncated read.  ',
            'conditions': ['When a continuation cursor is returned.'],
            'supports': ['s00-o1', 's00-o2'], 'counterevidence': ['s12-o1'],
            'evidence_strength': 'replay_verified',
        }]}
        self.claims = [{'id': 'c1', 'kind': 'defect'},
                       {'id': 'c2', 'kind': 'capability'},
                       {'id': 'c3', 'kind': 'uncertain'}]
        self.design = {'targets': [{'claim_id': 'c1', 'feature': 'Required data near the end.',
                                    'test_intent': 'A prefix-only answer must fail.'}]}

    def test_aggregate_expands_evidence_without_changing_raw_or_meaning(self):
        raw, observations = copy.deepcopy(self.raw), copy.deepcopy(self.observations)
        result = enrich_aggregate(raw, observations)
        claim = result['claims'][0]
        self.assertEqual(claim['statement'], raw['claims'][0]['finding'])
        self.assertEqual(claim['evidence'], [self.first, self.second])
        self.assertEqual(claim['support_sample_ids'], [0])
        self.assertEqual(claim['support_sample_count'], 1)
        self.assertEqual(claim['counterevidence_ids'], ['s12-o1'])
        self.assertEqual(raw, self.raw)
        self.assertEqual(observations, self.observations)
        validate_aggregate(result, observations)

    def test_uncertainty_absence_null_and_declared_are_distinct(self):
        for extra, status, expected in [({}, 'not_declared', []),
                                        ({'uncertainty': None}, 'explicit_null', []),
                                        ({'uncertainty': 'Cause is unclear.'}, 'declared', ['Cause is unclear.'])]:
            with self.subTest(status=status):
                raw = copy.deepcopy(self.raw)
                raw['claims'][0].update(extra)
                result = enrich_aggregate(raw, self.observations)['claims'][0]
                self.assertEqual(result['uncertainty_status'], status)
                self.assertEqual(result['uncertainties'], expected)

    def test_extra_evidence_and_deterministic_ids_keep_order(self):
        raw = copy.deepcopy(self.raw)
        extra = {'path': str(self.path), 'locator': 'Sheet1!B2:F9'}
        raw['claims'][0]['evidence'] = [self.first, extra]
        raw['claims'].append(dict(raw['claims'][0], kind='capability', supports=['s12-o1', 's00-o1']))
        result = enrich_aggregate(raw, self.observations)
        self.assertEqual([c['id'] for c in result['claims']], ['c1', 'c2'])
        self.assertEqual(result['claims'][0]['evidence'], [self.first, self.second, extra])
        self.assertEqual(result['claims'][1]['support_sample_ids'], [12, 0])

    def test_bad_semantics_are_rejected_not_repaired(self):
        for field, value in [('kind', 'failure'), ('finding', ''), ('conditions', 'always'),
                             ('supports', []), ('supports', ['s99-o1']),
                             ('supports', ['s00-o1', 's00-o1']),
                             ('counterevidence', ['s99-o1']), ('uncertainty', ''),
                             ('counterevidence', None), ('id', 'c1')]:
            raw = copy.deepcopy(self.raw)
            raw['claims'][0][field] = value
            before = copy.deepcopy(raw)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                enrich_aggregate(raw, self.observations)
            self.assertEqual(raw, before)

    def test_sample_identity_and_inventory_are_not_guessed(self):
        observations = copy.deepcopy(self.observations)
        observations[0]['sample_id'] = 1
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            enrich_aggregate(self.raw, observations)
        observations[0]['sample_id'] = True
        with self.assertRaises(ValueError):
            enrich_aggregate(self.raw, observations)
        with self.assertRaisesRegex(ValueError, 'duplicate observation'):
            enrich_aggregate(self.raw, self.observations + [self.observations[0]])

    def test_evidence_validation_preserves_complete_text(self):
        raw = copy.deepcopy(self.raw)
        raw['claims'][0]['finding'] = 'Long finding. ' * 10000
        self.assertEqual(enrich_aggregate(raw, self.observations)['claims'][0]['statement'],
                         raw['claims'][0]['finding'])
        raw['claims'][0]['evidence'] = [{'path': 'relative.txt', 'locator': 'event 1'}]
        with self.assertRaisesRegex(ValueError, 'absolute'):
            enrich_aggregate(raw, self.observations)

    def test_design_selects_only_targets_without_inventing_exclusions(self):
        raw = copy.deepcopy(self.design)
        result = enrich_design(raw, self.claims)
        self.assertEqual(result['schema_version'], 'compact-design-derived-v1')
        self.assertEqual(result['unselected_claim_ids'], ['c2', 'c3'])
        self.assertEqual(result['mappings'], [{'id': 'm1', 'claim_id': 'c1', 'role': 'target',
                                              'feature': raw['targets'][0]['feature'],
                                              'check_plan': raw['targets'][0]['test_intent']}])
        self.assertIsNone(result['preserves'])
        self.assertEqual(result['preserves_status'], 'not_declared')
        self.assertIsNone(result['simplifies'])
        self.assertEqual(raw, self.design)

    def test_design_explicit_empty_differs_from_missing(self):
        raw = dict(self.design, preserves=[], simplifies=['Flat log instead of weekly blocks.'])
        result = enrich_design(raw, {'claims': self.claims})
        self.assertEqual(result['preserves'], [])
        self.assertEqual(result['preserves_status'], 'declared')
        self.assertEqual(result['simplifies'], raw['simplifies'])
        for invalid in (None, 'nothing', [None]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                enrich_design(dict(self.design, preserves=invalid), self.claims)

    def test_design_invalid_targets_are_not_converted(self):
        for identifier in ('c3', 'missing'):
            raw = copy.deepcopy(self.design)
            raw['targets'][0]['claim_id'] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                enrich_design(raw, self.claims)
        for raw in ({'targets': []}, dict(self.design, mappings=[]),
                    {'targets': self.design['targets'] * 2}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_design(raw, self.claims)

    def test_normalize_preserves_legacy_and_derived_artifacts(self):
        for raw in ({'mappings': [{'id': 'legacy'}]}, enrich_design(self.design, self.claims)):
            before = copy.deepcopy(raw)
            result = normalize_design(raw, self.claims)
            self.assertEqual(result, before)
            self.assertIsNot(result, raw)
        self.assertEqual(normalize_design(self.design, self.claims), enrich_design(self.design, self.claims))

    def planned_aggregate(self):
        raw = copy.deepcopy(self.raw)
        raw['claims'].append(dict(raw['claims'][0], kind='infrastructure_issue',
                                  finding='The response was partial; fault is not established.'))
        raw['training_goals'] = [{
            'claim_refs': [1, 2], 'ability': 'Acquire the information needed to answer.',
            'rationale': 'Input coverage precedes aggregation; the causal link is provisional.',
            'success_signal': 'All necessary records are acquired by any valid method.',
            'difficulty_controls': ['Use a small output table.'],
        }]
        return raw

    def test_training_goals_resolve_order_and_preserve_model_judgments(self):
        raw = self.planned_aggregate()
        raw['training_goals'].append(dict(raw['training_goals'][0], claim_refs=[1],
                                           ability='Check acquisition completeness.'))
        before = copy.deepcopy(raw)
        result = enrich_aggregate(raw, self.observations)
        self.assertEqual([g['id'] for g in result['training_goals']], ['g1', 'g2'])
        self.assertEqual(result['training_goals'][0]['claim_ids'], ['c1', 'c2'])
        for field in ('ability', 'rationale', 'success_signal', 'difficulty_controls'):
            self.assertEqual(result['training_goals'][0][field], raw['training_goals'][0][field])
        self.assertEqual(result['claims'][1]['kind'], 'infrastructure_issue')
        self.assertEqual(raw, before)
        validate_aggregate(result, self.observations)

    def test_training_goals_missing_and_explicit_empty_remain_distinct(self):
        self.assertNotIn('training_goals', enrich_aggregate(self.raw, self.observations))
        raw = dict(self.raw, training_goals=[])
        self.assertEqual(enrich_aggregate(raw, self.observations)['training_goals'], [])

    def test_training_goal_references_are_checked_not_repaired(self):
        for refs in ([0], [3], [True], ['1'], [1, 1], [], [2]):
            raw = self.planned_aggregate()
            raw['training_goals'][0]['claim_refs'] = refs
            before = copy.deepcopy(raw)
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                enrich_aggregate(raw, self.observations)
            self.assertEqual(raw, before)

    def test_training_goal_fields_are_validated_without_shortening(self):
        raw = self.planned_aggregate()
        raw['training_goals'][0]['rationale'] *= 10000
        self.assertEqual(enrich_aggregate(raw, self.observations)['training_goals'][0]['rationale'],
                         raw['training_goals'][0]['rationale'])
        for field, value in [('ability', ''), ('rationale', None), ('success_signal', 1),
                             ('difficulty_controls', 'simplify'), ('id', 'g9')]:
            invalid = self.planned_aggregate()
            invalid['training_goals'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                enrich_aggregate(invalid, self.observations)

    def test_primary_goal_requires_matching_target_and_selection_reason(self):
        attribution = enrich_aggregate(self.planned_aggregate(), self.observations)
        raw = dict(self.design, primary_goal_id='g1', selection_reason='Prioritize complete input.')
        derived = enrich_design(raw, attribution)
        self.assertEqual(derived['primary_goal_id'], 'g1')
        self.assertEqual(derived['selection_reason'], raw['selection_reason'])
        validate_goal_selection(derived, attribution, ['c1'])
        for invalid in (self.design, dict(self.design, primary_goal_id='g1'),
                        dict(self.design, selection_reason='Only reason.'),
                        dict(raw, primary_goal_id='g2'), dict(raw, selection_reason='')):
            with self.subTest(raw=invalid), self.assertRaises(ValueError):
                enrich_design(invalid, attribution)
        with self.assertRaisesRegex(ValueError, 'exercise'):
            validate_goal_selection(derived, attribution, ['c2'])

    def test_canonical_training_goal_cannot_bypass_reference_checks(self):
        valid = enrich_aggregate(self.planned_aggregate(), self.observations)
        for field, value in [('id', 'g2'), ('claim_ids', ['missing']), ('claim_ids', ['c2']),
                             ('claim_ids', ['c1', 'c1']), ('ability', '')]:
            invalid = copy.deepcopy(valid)
            invalid['training_goals'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_training_goals(invalid)

    def test_goal_selection_does_not_invent_goals_for_legacy_input(self):
        with self.assertRaisesRegex(ValueError, 'unknown primary'):
            enrich_design(dict(self.design, primary_goal_id='g1', selection_reason='Guess.'), self.claims)
        self.assertNotIn('primary_goal_id', enrich_design(self.design, self.claims))


if __name__ == '__main__':
    unittest.main()
