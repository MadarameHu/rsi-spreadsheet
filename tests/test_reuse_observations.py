"""Reuse frozen workers while performing a fresh aggregate, with no API replay."""
import copy
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import test_local_integration as local_tests
import diagnostics
import parallel_attribution as pa
import stage_audit
from stage_records import _tree_entries, read_stage_commit


class ReuseObservationTests(unittest.TestCase):
    setUp = local_tests.LocalIntegrationTests.setUp
    analyze = local_tests.LocalIntegrationTests.analyze

    def prepare_reuse(self):
        self.analyze()
        self.origin = self.ep/'diagnostics'
        self.old_tree = _tree_entries(self.origin)
        self.old_index = json.loads((self.origin/'observations/index.json').read_text())
        self.fake.calls.clear()
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg['diagnostics']['parallel_attribution']['reuse_observations_from'] = str(self.origin)
        # Downstream configuration may change without invalidating input identity.
        self.cfg['researcher']['max_concurrent_docker'] = 3
        self.cfg['diagnostics']['parallel_attribution']['design_budget_min'] = 4
        self.ep = self.root/'ep02'

    def test_reuses_all_sixteen_preserves_commits_and_resumes_without_worker_calls(self):
        self.prepare_reuse()
        with patch.object(pa, '_worker', side_effect=AssertionError('worker replay forbidden')), \
             patch.object(pa, '_run', wraps=pa._run) as aggregate_run:
            result = self.analyze()
            self.assertEqual(self.fake.calls, [None])
            index = json.loads((self.ep/'diagnostics/observations/index.json').read_text())
            self.assertEqual(index['reuse_observations_from'], str(self.origin))
            self.assertEqual(index['observation_ids'],self.old_index['observation_ids'])
            aggregate = read_stage_commit(self.ep/'diagnostics/records',result['attribution_commit'])
            self.assertEqual(aggregate['parent_ids'],[w['commit_id'] for w in self.old_index['workers']])
            self.assertEqual(aggregate_run.call_count, 1)
            readonly = aggregate_run.call_args.args[5]
            self.assertNotIn(self.origin, readonly)
            self.assertNotIn(self.origin/'observations', readonly)
            for old, new in zip(self.old_index['workers'],index['workers']):
                self.assertEqual(old['commit_id'],new['commit_id'])
                self.assertEqual(old['output_path'],new['output_path'])
                self.assertEqual(new['reused_from'],str(self.origin))
                self.assertEqual(Path(old['commit_path']).read_bytes(),Path(new['commit_path']).read_bytes())
                manifest = json.loads(Path(old['input_manifest']).read_text())
                self.assertTrue(all(Path(p) in readonly for p in manifest['allowed_files']))
                self.assertIn(Path(old['output_path']), readonly)
            self.assertEqual(_tree_entries(self.origin),self.old_tree)
            # Call the attribution entry itself to exercise reuse recovery validation.
            inputs = json.loads((self.ep/'diagnostics/inputs.json').read_text())
            recovered = pa.produce_attribution(self.hb,self.cfg,inputs['original']['snapshot_path'],
                inputs['rollouts']['snapshot_path'],self.ep,self.ep/'diagnostics',{})
            self.assertEqual(recovered,aggregate)
            self.assertEqual(self.fake.calls,[None])
            self.assertEqual(_tree_entries(self.origin),self.old_tree)

    def test_rejects_changed_current_source_before_aggregate(self):
        self.prepare_reuse()
        (self.source/'instruction.md').write_text('Different task')
        with self.assertRaisesRegex(ValueError,'source evidence differs'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_changed_rollout_before_aggregate(self):
        self.prepare_reuse()
        (self.bundle/'attempt-0.md').write_text('Different trajectory')
        with self.assertRaisesRegex(ValueError,'source evidence differs'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_changed_live_observation(self):
        self.prepare_reuse()
        worker = self.old_index['workers'][0]
        Path(worker['output_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'committed bytes'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_changed_archived_reference(self):
        self.prepare_reuse()
        worker = self.old_index['workers'][0]
        commit = json.loads(Path(worker['commit_path']).read_text())
        (self.origin/'records'/commit['references'][0]['path']).write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'archived reference hash mismatch'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_changed_worker_evidence(self):
        self.prepare_reuse()
        manifest = json.loads(Path(self.old_index['workers'][0]['input_manifest']).read_text())
        Path(manifest['allowed_files'][0]).write_text('changed evidence')
        with self.assertRaisesRegex(ValueError,'worker input bytes differ'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_incomplete_old_batch(self):
        self.prepare_reuse()
        index_path = self.origin/'observations/index.json'
        changed = dict(self.old_index,status='running')
        index_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'completed with identical sample order'):
            self.analyze()
        self.assertEqual(self.fake.calls,[])

    def test_rejects_relative_reuse_configuration(self):
        self.cfg['diagnostics']['parallel_attribution']['reuse_observations_from'] = 'relative/diagnostics'
        with self.assertRaisesRegex(ValueError,'absolute diagnostics directory'):
            diagnostics.parallel_config(self.cfg)

    def prepare_partial_reuse(self):
        failed = {0, 3, 14, 15}
        worker = pa._worker
        self.fake.barrier = threading.Barrier(12)
        def fail_four(*args):
            sample = args[6]
            if sample in failed:
                return {'sample_id': sample, 'status': 'failed', 'observation_ids': [],
                        'error': 'API 429'}
            return worker(*args)
        with patch.object(pa, '_worker', side_effect=fail_four):
            with self.assertRaisesRegex(RuntimeError, 'observation workers failed'):
                self.analyze()
        self.assertEqual(set(self.fake.calls), set(self.ids) - failed)
        self.origin = self.ep/'diagnostics'
        self.old_tree = _tree_entries(self.origin)
        self.old_index = json.loads((self.origin/'observations/index.json').read_text())
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg['diagnostics']['parallel_attribution'].update(
            reuse_observations_from=str(self.origin), allow_partial_reuse=True)
        self.ep = self.root/'ep02'
        self.fake.calls.clear()
        self.fake.barrier = threading.Barrier(4)
        return failed

    def test_partial_reuse_runs_only_four_failures_and_preserves_twelve_commits(self):
        failed = self.prepare_partial_reuse()
        with patch.object(pa, '_worker', wraps=pa._worker) as worker:
            result = self.analyze()
        self.assertEqual(worker.call_count, 4)
        self.assertEqual(set(self.fake.calls[:-1]), failed)
        self.assertIsNone(self.fake.calls[-1])
        diag = self.ep/'diagnostics'
        index = json.loads((diag/'observations/index.json').read_text())
        self.assertEqual(index['sample_order'], self.ids)
        self.assertEqual(index['status'], 'completed')
        self.assertTrue(index['allow_partial_reuse'])
        for before, after in zip(self.old_index['workers'], index['workers']):
            if before['sample_id'] not in failed:
                self.assertEqual(before['commit_id'], after['commit_id'])
                self.assertEqual(before['output_path'], after['output_path'])
            else:
                self.assertNotIn('reused_from', after)
        aggregate = read_stage_commit(diag/'records', result['attribution_commit'])
        audited = stage_audit._load_observations(diag, diag/'records', aggregate)
        self.assertEqual(len(audited['observation_commits']), 16)
        self.assertEqual(_tree_entries(self.origin), self.old_tree)
        # Re-entering a completed invocation only verifies; no model replay.
        inputs = json.loads((diag/'inputs.json').read_text())
        with patch.object(pa, '_run', side_effect=AssertionError('model replay forbidden')):
            pa.produce_attribution(self.hb, self.cfg, inputs['original']['snapshot_path'],
                inputs['rollouts']['snapshot_path'], self.ep, diag, {})
        self.assertEqual(len(self.fake.calls), 5)

    def test_completed_mixed_source_can_be_reused_without_returning_to_failed_ancestor(self):
        self.prepare_partial_reuse()
        self.analyze()
        mixed_origin = self.ep/'diagnostics'
        self.cfg['diagnostics']['parallel_attribution'].update(
            reuse_observations_from=str(mixed_origin), allow_partial_reuse=False)
        self.ep = self.root/'ep03'
        self.fake.calls.clear()
        with patch.object(pa, '_worker', side_effect=AssertionError('worker replay forbidden')):
            result = self.analyze()
        self.assertEqual(self.fake.calls, [None])
        diag = self.ep/'diagnostics'
        index = json.loads((diag/'observations/index.json').read_text())
        self.assertEqual(set(index['reuse_observation_sources']), {str(self.origin), str(mixed_origin)})
        aggregate = read_stage_commit(diag/'records', result['attribution_commit'])
        self.assertEqual(len(stage_audit._load_observations(diag, diag/'records', aggregate)
                             ['observation_commits']), 16)
        self.assertEqual(_tree_entries(self.origin), self.old_tree)

    def test_partial_reuse_rejects_changed_original_evidence_before_new_workers(self):
        self.prepare_partial_reuse()
        sample = next(w for w in self.old_index['workers'] if w['status'] == 'completed')
        manifest = json.loads(Path(sample['input_manifest']).read_text())
        Path(manifest['allowed_files'][0]).write_text('tampered source evidence')
        with patch.object(pa, '_worker', side_effect=AssertionError('worker replay forbidden')):
            with self.assertRaisesRegex(ValueError, 'worker input bytes differ'):
                self.analyze()
        self.assertEqual(self.fake.calls, [])

    def test_partial_reuse_rejects_running_batch(self):
        self.prepare_partial_reuse()
        path = self.origin/'observations/index.json'
        path.write_text(json.dumps(dict(self.old_index, status='running')))
        with self.assertRaisesRegex(ValueError, 'completed with identical sample order'):
            self.analyze()
        self.assertEqual(self.fake.calls, [])

    def test_partial_reuse_rejects_changed_raw_job_before_any_new_call(self):
        self.prepare_partial_reuse()
        sample = next(w for w in self.old_index['workers'] if w['status'] == 'completed')
        (Path(sample['job_path'])/'claude-code.txt').write_text('changed raw model event')
        with patch.object(pa, '_worker', side_effect=AssertionError('worker replay forbidden')):
            with self.assertRaisesRegex(ValueError, 'committed bytes'):
                self.analyze()
        self.assertEqual(self.fake.calls, [])

    def test_mixed_analysis_flows_into_candidate_and_audit_without_analysis_replay(self):
        import candidate_jobs as jobs
        import local_analysis
        import staged_synth as staged
        from test_candidate_jobs import GoalHarbor
        self.prepare_partial_reuse()
        self.cfg['training_pipeline'] = {'candidate_generation': {
            'difficulties': [{'name': 'focused', 'instruction': 'Reduce output size'}],
            'copies_per_goal': 1}}
        def with_goal(*args):
            result = self.fake(*args)
            env = args[-1]
            if env.get('ADB_SAMPLE_ID') is None:
                path = Path(env['ADB_RESULT'])
                payload = json.loads(path.read_text())
                payload['training_goals'] = [{'claim_refs': [1], 'ability': 'Complete grouping',
                    'rationale': 'Supported by complete source evidence',
                    'success_signal': 'All records included', 'difficulty_controls': ['Fewer output cells']}]
                path.write_text(json.dumps(payload))
            return result
        with patch.object(local_analysis, 'run_local', side_effect=with_goal):
            self.analyze()
        analysis = self.ep
        manifest = jobs.prepare_joblist(self.cfg,
            [{'original': 'source', 'analysis_source': str(analysis/'diagnostics')}], self.root/'jobs.json')
        author = GoalHarbor(1)
        ep = self.root/'candidate'
        with patch.object(pa, '_worker', side_effect=AssertionError('worker replay forbidden')), \
             patch.object(pa, '_run', side_effect=AssertionError('analysis replay forbidden')), \
             patch.object(pa.mounts, 'build', side_effect=self.real_mount_builder), \
             patch.object(staged, 'benchmark_dir', return_value=self.source.parent):
            jobs.run_candidate(author, self.cfg, 'bench', self.source, ep, 1, manifest['jobs'][0])
        self.assertEqual([call['stage'] for call in author.calls], ['design'])
        self.assertEqual(len(self.fake.calls), 5)
        upstream = stage_audit.load_inputs(ep)
        self.assertEqual(len(upstream['observation_commits']), 16)
        self.assertEqual(_tree_entries(self.origin), self.old_tree)

    def test_partial_reuse_requires_explicit_valid_source(self):
        settings = self.cfg['diagnostics']['parallel_attribution']
        settings['allow_partial_reuse'] = True
        with self.assertRaisesRegex(ValueError, 'requires reuse_observations_from'):
            diagnostics.parallel_config(self.cfg)
        settings['allow_partial_reuse'] = 'true'
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            diagnostics.parallel_config(self.cfg)


if __name__ == '__main__':
    unittest.main()
