"""Candidate repair uses an explicit new identity and retains the rejected version."""
from pathlib import Path
import unittest
from unittest.mock import patch

import test_training_runner as fixtures
from test_quality_gate import QualityReviewer
from candidate_jobs import validate_job, run_candidate
from repair_jobs import prepare_repair_joblist, validate_repair, _require_candidate_rejection
from stage_audit import read_json


class RepairJobsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingRunnerTests('test_quality_rejection_produces_no_training_export_or_trainee_call')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def reject(self, judgments=None, **kwargs):
        self.fixture.reviewer = QualityReviewer(
            judgments=judgments if judgments is not None else {'task:ground_truth': 'incorrect'}, **kwargs)
        with self.fixture.context():
            result = self.fixture.execute()
        self.assertEqual(result['status'], 'quality_blocked')

    def prepare(self):
        return prepare_repair_joblist(self.fixture.cfg, self.fixture.ep,
                                     self.fixture.fixture.root / 'repair-jobs.json')

    def test_revision_gets_new_identity_and_authors_without_modifying_parent(self):
        self.reject()
        parent = self.fixture.ep
        before = {p: p.read_bytes() for p in parent.rglob('*') if p.is_file()}
        manifest = self.prepare()
        self.assertEqual(manifest['jobs'][0], self.fixture.job)
        revised = manifest['jobs'][1]
        self.assertNotEqual(revised['id'], self.fixture.job['id'])
        self.assertEqual(revised['analysis_identity'], self.fixture.job['analysis_identity'])
        self.assertEqual(revised['goal_id'], self.fixture.job['goal_id'])
        validate_job(revised, self.fixture.cfg)
        self.assertEqual(self.prepare(), manifest)
        author = fixtures.candidate_fixtures.GoalHarbor(1)
        ep = self.fixture.fixture.root / 'repaired-episode'
        run_candidate(author, self.fixture.cfg, 'bench', self.fixture.fixture.source, ep, 2, revised)
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(author.calls[0]['stage'], 'design')
        mounts = {m['target']: m for m in author.calls[0]['mounts']}
        for path in validate_repair(revised):
            self.assertTrue(mounts[str(path)]['read_only'])
        parent_design = (parent/'diagnostics/design.json').resolve()
        self.assertIn(parent_design, validate_repair(revised))
        self.assertTrue(mounts[str(parent_design)]['read_only'])
        self.assertEqual(Path(mounts[str(parent_design)]['source']).read_bytes(),
                         (parent/'diagnostics/design.json').read_bytes())
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))

    def test_feedback_tampering_rejects_revision(self):
        self.reject()
        job = self.prepare()['jobs'][1]
        (Path(job['repair']['feedback_dir']) / 'quality.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'repair feedback differs'):
            validate_job(job, self.fixture.cfg)

    def test_repair_reserves_after_other_candidates_and_keeps_number_on_resume(self):
        self.reject()
        (self.fixture.ep.parent / 'ep07').mkdir()
        manifest = self.prepare()
        parent, revised = manifest['jobs']
        self.assertEqual(manifest['episode_assignments'][parent['id']], 1)
        self.assertEqual(manifest['episode_assignments'][revised['id']], 8)
        (self.fixture.ep.parent / 'ep09').mkdir()
        self.assertEqual(self.prepare(), manifest)

    def test_incorrect_selected_claim_requires_new_attribution_before_creating_files(self):
        self.reject({'claim:c1': 'incorrect'})
        with self.assertRaisesRegex(ValueError, 'new attribution version.*claim:c1'):
            self.prepare()
        self.assertFalse((self.fixture.ep/'diagnostics/repair-inputs').exists())
        self.assertFalse((self.fixture.fixture.root/'repair-jobs.json').exists())

    def test_upstream_mapping_and_candidate_defect_still_require_new_attribution(self):
        # Parallel review derives upstream_verdict from the claim judgment.
        self.reject({'claim:c1': 'incorrect', 'task:spec': 'incorrect'})
        with self.assertRaisesRegex(ValueError, 'new attribution version.*upstream_verdict'):
            self.prepare()

    def test_existing_upstream_repair_job_is_rejected_on_resume(self):
        self.reject({'claim:c1': 'incorrect'})
        # Reproduce an old job prepared before the repair-route guard existed.
        with patch('repair_jobs._require_candidate_rejection'):
            legacy = self.prepare()['jobs'][1]
        with self.assertRaisesRegex(ValueError, 'new attribution version'):
            validate_job(legacy, self.fixture.cfg)

    def test_implementation_rejection_can_be_repaired(self):
        self.reject({'mapping:implementation_verdict': 'incorrect'})
        revised = self.prepare()['jobs'][1]
        validate_job(revised, self.fixture.cfg)

    def test_scorer_probe_rejection_preserves_existing_revision_identity(self):
        self.reject({}, extra_probe='false_accept')
        manifest = self.prepare()
        revised = manifest['jobs'][1]
        validate_job(revised, self.fixture.cfg)
        self.assertEqual(self.prepare(), manifest)
        self.assertNotIn('repair_route', revised)

    def test_explicit_observation_issue_routes_upstream_without_scanning_audit(self):
        with self.assertRaisesRegex(ValueError, 'new attribution version.*observation:o1'):
            _require_candidate_rejection({'issues': [
                {'id': 'observation:o1', 'verdict': 'incorrect'},
                {'id': 'probe:row1_dates_as_text', 'verdict': 'incorrect'}]})
        # Gate-approved upstream claims are not reconsidered here. Existing
        # unrelated observation/audit judgments are deliberately not inputs.
        _require_candidate_rejection({'issues': [
            {'id': 'probe:row1_dates_as_text', 'verdict': 'incorrect'}]})

    def test_integrity_fault_is_not_sent_to_candidate_author(self):
        self.reject()
        from quality_gate import evaluate_quality
        quality = evaluate_quality(self.fixture.ep)
        quality['issues'].append({'id': 'evidence_integrity', 'verdict': 'fault'})
        with patch('quality_gate.evaluate_quality', return_value=quality):
            with self.assertRaisesRegex(ValueError, 'without infrastructure faults'):
                self.prepare()

    def test_accepted_candidate_is_not_rewritten(self):
        with self.fixture.context():
            self.assertEqual(self.fixture.execute()['status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'semantic rejection'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
