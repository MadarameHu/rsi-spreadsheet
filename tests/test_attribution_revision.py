"""Post-review revisions preserve history and only replay the aggregate stage."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_training_runner as fixtures
from test_quality_gate import QualityReviewer
from test_candidate_jobs import GoalHarbor
import attribution_revision as revision
import candidate_jobs
import diagnostics
import parallel_attribution as pa
import staged_synth as st
from stage_records import _tree_entries, read_stage_commit


class RevisedAggregateHarbor(GoalHarbor):
    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        env = dict(item.split('=', 1) for item in kwargs['agent_env'])
        result = super().agent_task(task, agent, model, job, timeout, mounts, **kwargs)
        if 'ADB_ATTRIBUTION_REVISION_INPUT' in env:
            manifest = st.load(Path(env['ADB_ATTRIBUTION_REVISION_INPUT']))
            assert manifest['origin'] == 'post_review_feedback'
            assert all(Path(e['snapshot_path']).read_bytes() == Path(e['source_path']).read_bytes()
                       for e in manifest['files'].values())
            prompt = (Path(task)/'instruction.md').read_text()
            assert 'NEW POST-REVIEW ATTRIBUTION VERSION' in prompt
            assert 'not human gold' in prompt
            path = Path(env['ADB_RESULT'])
            payload = st.load(path)
            payload['claims'][0]['statement'] += ' Revised based on later review, without a causal assertion.'
            payload['training_goals'][0]['rationale'] = 'Supported correspondence; no causal inference.'
            path.write_text(json.dumps(payload))
        return result


class AttributionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingRunnerTests('test_quality_rejection_produces_no_training_export_or_trainee_call')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.fixture.root
        self.parent = self.fixture.ep
        self.origin = self.fixture.fixture.analysis/'diagnostics'
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg['diagnostics']['parallel_attribution'].update(
            reuse_observations_from=str(self.origin), attribution_revision_from=str(self.parent))
        self.ep = self.root/'revised-analysis'

    def reject(self, judgments=None):
        self.fixture.reviewer = QualityReviewer(judgments=judgments or {'claim:c1': 'incorrect'})
        with self.fixture.context():
            self.fixture.execute()
        self.parent_tree = _tree_entries(self.parent)
        self.origin_tree = _tree_entries(self.origin)

    def run_revision(self, hb=None):
        return st.synthesize_staged(hb or RevisedAggregateHarbor(1), self.cfg, 'bench',
            self.fixture.fixture.source, self.ep, 1, analysis_only=True)

    def test_revision_reuses_all_workers_records_later_feedback_and_preserves_parents(self):
        self.reject()
        hb = RevisedAggregateHarbor(1)
        with patch.object(pa, '_worker', side_effect=AssertionError('diagnosis replay forbidden')):
            result = self.run_revision(hb)
        self.assertEqual(len(hb.calls), 1)
        diag = self.ep/'diagnostics'
        commit = read_stage_commit(diag/'records', result['attribution_commit'])
        original = read_stage_commit(self.origin/'records', st.load(self.origin/'attribution_commit.json')['commit_id'])
        self.assertEqual(commit['parent_ids'], original['parent_ids'])
        record = commit['metadata']['attribution_revision']
        self.assertEqual(record['origin'], 'post_review_model_revision')
        self.assertFalse(record['is_human_gold'])
        self.assertEqual(record['parent_attribution_commit'], original['commit_id'])
        self.assertEqual(record['changed_claim_ids'], ['c1'])
        self.assertEqual(record['changed_training_goal_ids'], ['g1'])
        self.assertEqual(st.load(diag/'attribution/revision.json'), record)
        manifest = st.load(Path(record['input_manifest']))
        quality = st.load(self.parent/'diagnostics/quality.json')
        self.assertEqual(manifest['upstream_issues'],
            [item for item in quality['issues'] if item['id'] in manifest['upstream_issue_ids']])
        self.assertEqual(_tree_entries(self.parent), self.parent_tree)
        self.assertEqual(_tree_entries(self.origin), self.origin_tree)
        self.assertEqual(self.run_revision(hb), result)
        self.assertEqual(len(hb.calls), 1)
        # Original pilot config may use the new analysis without revision flags.
        listing = candidate_jobs.prepare_joblist(self.fixture.cfg,
            [{'original': 'source', 'analysis_source': str(diag)}], self.root/'new-jobs.json', goal_ids=['g1'])
        self.assertNotEqual(listing['jobs'][0]['analysis_identity'], self.fixture.job['analysis_identity'])
        author = GoalHarbor(1)
        candidate_jobs.run_candidate(author, self.fixture.cfg, 'bench', self.fixture.fixture.source,
                                    self.root/'new-candidate', 1, listing['jobs'][0])
        self.assertEqual([call['stage'] for call in author.calls], ['design'])

    def test_candidate_only_rejection_does_not_launch_a_new_aggregate(self):
        self.reject({'task:ground_truth': 'incorrect'})
        hb = RevisedAggregateHarbor(1)
        with self.assertRaisesRegex(ValueError, 'requires an upstream rejection'):
            self.run_revision(hb)
        self.assertEqual(hb.calls, [])

    def test_feedback_tampering_is_rejected_before_any_model_call(self):
        self.reject()
        path = self.parent/'diagnostics/quality.json'
        quality = st.load(path)
        quality['issues'][0]['reason'] = 'forged later judgment'
        path.write_text(json.dumps(quality))
        hb = RevisedAggregateHarbor(1)
        with self.assertRaisesRegex(ValueError, 'feedback differs from verified'):
            self.run_revision(hb)
        self.assertEqual(hb.calls, [])

    def test_different_analysis_source_is_rejected(self):
        self.reject()
        self.cfg['diagnostics']['parallel_attribution']['reuse_observations_from'] = str(self.root)
        with self.assertRaisesRegex(ValueError, 'must reuse the rejected candidate analysis source'):
            self.run_revision()

    def test_frozen_feedback_tampering_is_detected_before_commit(self):
        self.reject()
        inputs = st.load(self.origin/'inputs.json')
        manifest_path = revision.prepare_revision_inputs(self.parent, self.cfg,
            Path(inputs['original']['snapshot_path']), Path(inputs['rollouts']['snapshot_path']), self.ep/'diagnostics')
        manifest = st.load(manifest_path)
        Path(manifest['files']['audit']['snapshot_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'input evidence changed'):
            revision.record_revision(manifest_path, st.load(self.origin/'attribution.json'),
                                     self.origin/'attribution.json')

    def test_revision_config_requires_explicit_full_reuse(self):
        settings = self.cfg['diagnostics']['parallel_attribution']
        settings['attribution_revision_from'] = 'relative'
        with self.assertRaisesRegex(ValueError, 'absolute episode directory'):
            diagnostics.parallel_config(self.cfg)
        settings['attribution_revision_from'] = str(self.parent)
        settings['allow_partial_reuse'] = True
        with self.assertRaisesRegex(ValueError, 'requires full reuse'):
            diagnostics.parallel_config(self.cfg)
