"""Check planning at canonical commit boundaries, not only raw enrichment."""
import copy
from pathlib import Path
import tempfile
import unittest

from harness.compact_synthesis import enrich_aggregate, enrich_design
from harness.parallel_records import validate_aggregate_structure
from harness.stage_records import validate_design


class TrainingGoalContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = Path(self.temp.name)/'evidence.md'
        p.write_text('A truncated read was not continued.')
        observations = [{'id': 's00-o1', 'evidence': [{'path': str(p), 'locator': 'event:1'}]}]
        self.aggregate = enrich_aggregate({
            'claims': [{'kind': 'defect', 'finding': 'Incomplete acquisition.',
                        'conditions': ['Truncated tool result.'], 'supports': ['s00-o1'], 'counterevidence': [], 'evidence_strength': 'replay_verified'}],
            'training_goals': [{'claim_refs': [1], 'ability': 'Acquire necessary data.',
                'rationale': 'Input is needed for the calculation.', 'success_signal': 'Missing records acquired.',
                'difficulty_controls': ['Small output.']}],
        }, observations)
        self.design = enrich_design({'primary_goal_id': 'g1', 'selection_reason': 'Isolate acquisition.',
            'targets': [{'claim_id': 'c1', 'feature': 'Partial feedback.', 'test_intent': 'Tail records needed.'}]}, self.aggregate)

    def test_canonical_aggregate_rejects_tampered_goal_reference(self):
        invalid = copy.deepcopy(self.aggregate)
        invalid['training_goals'][0]['claim_ids'] = ['c999']
        with self.assertRaisesRegex(ValueError, 'unknown claim'):
            validate_aggregate_structure(invalid)

    def test_canonical_design_cannot_bypass_primary_goal(self):
        validate_design(self.design, self.aggregate)
        invalid = copy.deepcopy(self.design)
        invalid['primary_goal_id'] = 'g999'
        with self.assertRaisesRegex(ValueError, 'unknown primary'):
            validate_design(invalid, self.aggregate)
        del invalid['primary_goal_id']
        del invalid['selection_reason']
        with self.assertRaisesRegex(ValueError, 'select a primary'):
            validate_design(invalid, self.aggregate)

    def test_legacy_payload_stays_unchanged_and_has_no_invented_goal(self):
        legacy = copy.deepcopy(self.aggregate)
        del legacy['training_goals']
        design = copy.deepcopy(self.design)
        del design['primary_goal_id']
        del design['selection_reason']
        before = copy.deepcopy(design)
        validate_design(design, legacy)
        self.assertEqual(design, before)


class EvidenceStrengthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = Path(self.temp.name)/'evidence.md'
        p.write_text('Recalculated mismatch.')
        self.observations = [{'id': f's0{i}-o1', 'evidence': [{'path': str(p), 'locator': 'event:1'}]} for i in range(2)]

    def raw(self, *strengths, g1=(1,)):
        return {'claims': [{'kind': 'defect', 'finding': f'Mechanism {i}.', 'conditions': ['Observed.'],
                            'supports': [f's0{i}-o1'], 'counterevidence': [], 'evidence_strength': s}
                           for i, s in enumerate(strengths)],
                'training_goals': [{'claim_refs': list(g1), 'ability': 'Practice.', 'rationale': 'Decided scores.',
                                    'success_signal': 'Correct values.', 'difficulty_controls': []}]}

    def test_strength_is_required_and_closed_vocabulary(self):
        raw = self.raw('replay_verified')
        del raw['claims'][0]['evidence_strength']
        with self.assertRaisesRegex(ValueError, 'evidence_strength'):
            enrich_aggregate(raw, self.observations)
        with self.assertRaisesRegex(ValueError, 'evidence_strength'):
            enrich_aggregate(self.raw('likely'), self.observations)
        self.assertEqual(enrich_aggregate(self.raw('hypothesis_only', 'replay_verified', g1=(2,)),
                                          self.observations)['claims'][0]['evidence_strength'], 'hypothesis_only')

    def test_first_goal_needs_a_replay_verified_claim(self):
        with self.assertRaisesRegex(ValueError, 'replay_verified'):
            enrich_aggregate(self.raw('hypothesis_only', 'replay_verified', g1=(1,)), self.observations)
        enrich_aggregate(self.raw('hypothesis_only', 'replay_verified', g1=(1, 2)), self.observations)

    def test_library_class_is_optional_and_preserved(self):
        raw = self.raw('replay_verified')
        self.assertNotIn('library_class', enrich_aggregate(raw, self.observations)['claims'][0])
        raw['claims'][0]['library_class'] = 'abc-c1'
        self.assertEqual(enrich_aggregate(raw, self.observations)['claims'][0]['library_class'], 'abc-c1')
        raw['claims'][0]['library_class'] = ''
        with self.assertRaisesRegex(ValueError, 'library_class'):
            enrich_aggregate(raw, self.observations)

    def test_legacy_aggregate_without_strength_is_still_valid(self):
        aggregate = enrich_aggregate(self.raw('replay_verified'), self.observations)
        for claim in aggregate['claims']:
            del claim['evidence_strength']
        validate_aggregate_structure(aggregate)
        aggregate['claims'][0]['evidence_strength'] = 'guess'
        with self.assertRaisesRegex(ValueError, 'evidence_strength'):
            validate_aggregate_structure(aggregate)
