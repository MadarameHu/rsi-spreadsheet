"""Reference -> committed quality admission -> target trials, with no paid calls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import host_validation
import quality_gate
import run_episode
import run_score
import stage_audit
import test_quality_gate as quality_fixtures
from test_host_validation import FakeValidationHarbor


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = quality_fixtures.QualityGateTests(
            'test_accepts_bound_complete_quality_without_requiring_target_success')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.ep = self.fixture.ep
        target_root = patch.object(host_validation, 'TARGET_EXECUTION_ROOT',
                                   self.ep.parent/'host-runtime/target-slots')
        target_root.start()
        self.addCleanup(target_root.stop)
        self.diag = self.fixture.diag
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg['diagnostics'] = {'enabled': True, 'parallel_attribution': {'max_concurrent': 4, 'worker_budget_min': 1, 'aggregate_budget_min': 2, 'design_budget_min': 3}}
        self.events = []

    def pipeline(self, target, reviewer):
        real_gate = quality_gate.run_quality_gate
        def launch(hb, cfg, benchmark, ep, **kwargs):
            phase = kwargs['phase']
            self.events.append(phase)
            request = host_validation.prepare_request(hb, cfg, benchmark, ep)
            return host_validation.run_worker(request, hb, phase=phase)
        def gate(*args, **kwargs):
            self.events.append('quality')
            return real_gate(*args, **kwargs)
        with patch.object(host_validation, 'launch_validation', side_effect=launch), \
             patch.object(quality_gate, 'run_quality_gate', side_effect=gate), \
             patch.object(run_episode, 'make_harbor', return_value=reviewer):
            return run_episode.validate_authored_candidate(target, self.cfg, 'spreadsheet', self.ep)

    def test_real_stages_order_reference_quality_trainee_and_accept_zero_solve_rate(self):
        target = FakeValidationHarbor(reward=0)
        reviewer = quality_fixtures.QualityReviewer()
        result = self.pipeline(target, reviewer)
        self.assertEqual(self.events, ['reference', 'quality', 'trainee'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], 3)
        self.assertEqual(result['solved_attempts'], 0)
        self.assertEqual([kind for kind, _ in target.calls], ['oracle', 'trainee', 'trainee', 'trainee'])
        self.assertGreater(len(reviewer.calls), 1)
        gate = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(gate['status'], 'accepted', gate['issues'])
        self.assertTrue(gate['eligible_for_export'])
        self.assertFalse(gate['difficulty_used_for_admission'])

    def test_failed_reference_stops_before_admission_and_trainee(self):
        target = FakeValidationHarbor(oracle_reward=0)
        reviewer = quality_fixtures.QualityReviewer()
        result = self.pipeline(target, reviewer)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.events, ['reference'])
        self.assertEqual([kind for kind, _ in target.calls], ['oracle'])
        self.assertFalse((self.diag/'quality.json').exists())

    def test_rejected_quality_never_starts_target_trials(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer(judgments={'task:ground_truth': 'incorrect'})
        result = self.pipeline(target, reviewer)
        self.assertEqual(result['status'], 'quality_rejected')
        self.assertEqual(self.events, ['reference', 'quality'])
        self.assertEqual([kind for kind, _ in target.calls], ['oracle'])
        self.assertFalse(result['quality']['eligible_for_export'])

    def test_blocked_quality_never_starts_target_trials(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer(judgments={'mapping:implementation_verdict': 'uncertain'})
        result = self.pipeline(target, reviewer)
        self.assertEqual(result['status'], 'quality_blocked')
        self.assertEqual(self.events, ['reference', 'quality'])
        self.assertEqual([kind for kind, _ in target.calls], ['oracle'])
        self.assertIsNone(result['quality']['quality_approved'])

    def test_pipeline_reentry_keeps_receipts_and_makes_no_new_model_calls(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer()
        first = self.pipeline(target, reviewer)
        review_calls = len(reviewer.calls)
        audit_bytes = (self.diag/'audit.json').read_bytes()
        validation_bytes = (self.diag/'validation.json').read_bytes()
        second = self.pipeline(target, reviewer)
        self.assertEqual(second, first)
        self.assertEqual((self.diag/'audit.json').read_bytes(), audit_bytes)
        self.assertEqual((self.diag/'validation.json').read_bytes(), validation_bytes)
        self.assertEqual(len(target.calls), 4)
        self.assertEqual(len(reviewer.calls), review_calls)

    def test_scoring_after_admission_does_not_launch_another_audit(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer()
        self.pipeline(target, reviewer)
        review_calls = len(reviewer.calls)
        original_audit = (self.diag/'audit.json').read_bytes()
        baseline = {'episode_raw': 0, 'difficulty': 0, 'note': 'Target solve rate is zero.'}
        with patch.object(run_score, '_score_episode_baseline', return_value=baseline), \
             patch.object(run_score, 'make_harbor', return_value=reviewer), \
             patch.object(stage_audit, 'audit_episode', side_effect=AssertionError('must not repeat audit')):
            result = run_score.score_episode(target, self.cfg, 'spreadsheet', 'source', self.ep, None)
        self.assertEqual(result, baseline)
        self.assertEqual((self.diag/'audit.json').read_bytes(), original_audit)
        self.assertEqual(quality_gate.evaluate_quality(self.ep)['status'], 'accepted')
        self.assertEqual(len(reviewer.calls), review_calls)

    def test_cached_admission_recheck_needs_no_harbor_runtime_or_new_audit(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer()
        self.pipeline(target, reviewer)
        original_audit = (self.diag/'audit.json').read_bytes()
        with patch.object(run_score, 'make_harbor', side_effect=RuntimeError('Harbor is unavailable')) as make, \
             patch.object(stage_audit, 'audit_episode', side_effect=AssertionError('must not repeat audit')):
            run_score.run_diagnostic_audit(target, self.cfg, 'spreadsheet', 'source', self.ep)
        make.assert_not_called()
        self.assertEqual((self.diag/'audit.json').read_bytes(), original_audit)
        self.assertEqual(quality_gate.evaluate_quality(self.ep)['status'], 'accepted')

    def test_failed_existing_gate_is_rechecked_without_repeating_review(self):
        target = FakeValidationHarbor()
        reviewer = quality_fixtures.QualityReviewer(judgments={'task:spec': 'incorrect'})
        first = self.pipeline(target, reviewer)
        review_calls = len(reviewer.calls)
        audit_bytes = (self.diag/'audit.json').read_bytes()
        # A mutable saved gate cannot overrule the bound audit on reentry.
        forged = stage_audit.read_json(self.diag/'quality.json')
        forged.update(status='accepted', eligible_for_export=True, quality_approved=True)
        stage_audit.write_json(self.diag/'quality.json', forged)
        second = self.pipeline(target, reviewer)
        self.assertEqual(first['status'], second['status'])
        self.assertEqual(second['status'], 'quality_rejected')
        self.assertEqual((self.diag/'audit.json').read_bytes(), audit_bytes)
        self.assertEqual([kind for kind, _ in target.calls], ['oracle'])
        self.assertEqual(len(reviewer.calls), review_calls)


if __name__ == '__main__':
    unittest.main()
