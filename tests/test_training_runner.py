"""Training jobs reuse analysis and resume real stages without paid model calls."""
from contextlib import ExitStack
from copy import deepcopy
import itertools
import json
import tempfile
import threading
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import host_validation
import run_episode
import run_training
import stage_audit
import test_candidate_jobs as candidate_fixtures
from test_host_validation import FakeValidationHarbor
from test_quality_gate import QualityReviewer
import test_training_export as export_fixtures


class TrainingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = candidate_fixtures.CandidateJobsTests(
            'test_one_job_only_authors_and_keeps_same_analysis_payload')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg['researcher']['agent_setup_timeout_mult'] = 1
        self.cfg['researcher']['max_concurrent_docker'] = 3
        self.cfg['customer']['agent'] = 'fixture'
        self.cfg['gateway']['api_key'] = 'fixture-key'
        self.cfg['score'].update(n_concurrent=3, setup_timeout_mult=2, outer_wall_task_multiplier=2)
        self.cfg['judge'] = {'agent': 'claude-code', 'model': 'judge', 'budget_min': 1}
        self.cfg['task_timeout_sec'] = {'bench': 60}
        self.cfg['training_pipeline']['rollout'] = {'attempts': 3}
        self.fixture.cfg = self.cfg
        self.job = self.fixture.joblist()['jobs'][0]
        self.ep = self.fixture.root/'runs/train/bench/source/ep01'
        self.exports = self.fixture.root/'exports'
        self.author = candidate_fixtures.GoalHarbor(1)
        self.target = FakeValidationHarbor(reward=0)
        self.reviewer = QualityReviewer()
        # run_training builds author, source reviewer, then trainee harbors per job.
        self.clients = itertools.cycle([lambda: self.author, lambda: self.reviewer, lambda: self.target])

    def context(self):
        stack = ExitStack()
        stack.enter_context(patch.object(run_training, 'episode_dir', return_value=self.ep))
        stack.enter_context(patch.object(run_training, 'benchmark_dir', return_value=self.fixture.source.parent))
        stack.enter_context(patch.object(run_training, 'make_harbor', side_effect=lambda *_: next(self.clients)()))
        stack.enter_context(patch.object(run_episode, 'make_harbor', return_value=self.reviewer))
        stack.enter_context(patch.object(run_training, 'log'))
        def launch(hb, cfg, benchmark, ep, **kwargs):
            path = host_validation.prepare_request(hb, cfg, benchmark, ep)
            return host_validation.run_worker(path, hb, phase=kwargs['phase'])
        stack.enter_context(patch.object(host_validation, 'launch_validation', side_effect=launch))
        return stack

    def execute(self, **kwargs):
        return run_training.execute_job(self.cfg, 'bench', 'train', self.job, 1,
                                        export_root=self.exports, **kwargs)

    def test_candidate_job_reuses_analysis_and_exports_quality_passed_zero_solve_task(self):
        analysis_bytes = (self.fixture.analysis/'diagnostics/attribution.json').read_bytes()
        with self.context():
            result = self.execute()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['export']['rl']['status'], 'exported')
        self.assertEqual(result['validation']['solved_attempts'], 0)
        self.assertEqual(result['export']['sft']['status'], 'pending')
        self.assertEqual(len(self.fixture.analysis_hb.calls), 2)
        self.assertEqual([call['stage'] for call in self.author.calls], ['design'])
        self.assertEqual((self.fixture.analysis/'diagnostics/attribution.json').read_bytes(), analysis_bytes)
        self.assertEqual(stage_audit.read_json(self.ep/'diagnostics/attribution.json'),
                         stage_audit.read_json(self.fixture.analysis/'diagnostics/attribution.json'))

    def test_quality_rejection_produces_no_training_export_or_trainee_call(self):
        self.reviewer = QualityReviewer(judgments={'task:ground_truth': 'incorrect'})
        with self.context():
            result = self.execute()
        self.assertEqual(result['status'], 'quality_blocked')
        self.assertEqual(result['validation']['status'], 'quality_rejected')
        self.assertEqual(result['export']['status'], 'blocked')
        self.assertFalse(self.exports.exists())
        self.assertEqual([kind for kind, _ in self.target.calls], ['oracle'])

    def test_resume_reuses_author_reference_review_and_trainee_receipts(self):
        with self.context():
            first = self.execute()
            review_calls = len(self.reviewer.calls)
            audit = (self.ep/'diagnostics/audit.json').read_bytes()
            validation = (self.ep/'diagnostics/validation.json').read_bytes()
            second = self.execute()
        self.assertEqual(first['export']['output_dir'], second['export']['output_dir'])
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(len(self.author.calls), 1)
        self.assertEqual(len(self.target.calls), 4)
        self.assertEqual(len(self.reviewer.calls), review_calls)
        self.assertEqual(len(self.fixture.analysis_hb.calls), 2)
        self.assertEqual((self.ep/'diagnostics/audit.json').read_bytes(), audit)
        self.assertEqual((self.ep/'diagnostics/validation.json').read_bytes(), validation)
        for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at'):
            self.assertEqual(second[key], first[key], key)

    def test_export_only_uses_existing_admission_without_harbor_or_models(self):
        with self.context():
            first = self.execute()
            review_calls = len(self.reviewer.calls)
        with patch.object(run_training, 'episode_dir', return_value=self.ep), \
             patch.object(run_training, 'make_harbor', side_effect=AssertionError('no Harbor')), \
             patch.object(run_training, 'run_candidate', side_effect=AssertionError('no author')), \
             patch.object(run_training, 'validate_authored_candidate', side_effect=AssertionError('no validation')), \
             patch.object(run_training, 'log'):
            result = self.execute(export_only=True)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['export']['output_dir'], first['export']['output_dir'])
        self.assertEqual(len(self.author.calls), 1)
        self.assertEqual(len(self.target.calls), 4)
        self.assertEqual(len(self.reviewer.calls), review_calls)
        for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at'):
            self.assertEqual(result[key], first[key], key)

    def test_legacy_resume_does_not_invent_missing_author_timings(self):
        with self.context():
            self.execute()
            path = self.ep/'training-status.json'
            previous = host_validation.read(path)
            for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at'):
                del previous[key]
            host_validation.write(path, previous)
            resumed = self.execute()
            exported = self.execute(export_only=True)
        for status in (resumed, exported):
            for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at'):
                self.assertNotIn(key, status)
        self.assertEqual(len(self.author.calls), 1)

    def test_failed_author_resume_preserves_partial_timings(self):
        self.author.wrong_goal = True
        with self.context():
            with self.assertRaisesRegex(ValueError, 'assigned primary goal'):
                self.execute()
            path = self.ep/'training-status.json'
            first = host_validation.read(path)
            with self.assertRaisesRegex(RuntimeError, 'failed previously'):
                self.execute()
            failed = host_validation.read(path)
        self.assertEqual(failed['queued_at'], first['queued_at'])
        self.assertEqual(failed['author_slot_acquired_at'], first['author_slot_acquired_at'])
        self.assertNotIn('author_slot_released_at', failed)
        self.assertEqual(len(self.author.calls), 1)

    def test_finished_author_recovery_keeps_timings_before_synthesis_receipt(self):
        with self.context():
            first = self.execute()
            (self.ep/'diagnostics/synthesis.json').unlink()
            resumed = self.execute()
        self.assertEqual(resumed['status'], 'completed')
        for key in ('queued_at', 'author_slot_acquired_at', 'author_slot_released_at'):
            self.assertEqual(resumed[key], first[key], key)
        self.assertEqual(len(self.author.calls), 1)

    def test_author_failure_is_preserved_and_stops_validation_and_export(self):
        self.author.wrong_goal = True
        with self.context():
            with self.assertRaisesRegex(ValueError, 'assigned primary goal'):
                self.execute()
        result = stage_audit.read_json(self.ep/'training-status.json')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['phase'], 'failed')
        self.assertEqual(self.target.calls, [])
        # Source review may overlap authoring; no candidate (task) review may start.
        self.assertFalse(any('/reviews/task/' in v for call in self.reviewer.calls for v in call['kwargs']['agent_env']))
        self.assertFalse(self.exports.exists())


class TrainingExportResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = export_fixtures.TrainingExportTests('test_sft_keeps_original_messages_and_input_order_and_copies_all_evidence')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.job = {'id': 'fixture-job', 'original': 'source', 'goal_id': 'g1', 'difficulty': {'name': 'focused'}}
        host_validation.write(self.fixture.ep/'diagnostics/validation.json', {'status': 'completed'})
        for name, value in [('episode_dir', self.fixture.ep), ('validate_job', None), ('log', None)]:
            context = patch.object(run_training, name, return_value=value)
            context.start(); self.addCleanup(context.stop)
        for name in ('make_harbor', 'run_candidate', 'validate_authored_candidate'):
            context = patch.object(run_training, name, side_effect=AssertionError('export-only cannot run models'))
            context.start(); self.addCleanup(context.stop)

    def execute(self, paths=()):
        return run_training.execute_job({}, 'bench', 'train', self.job, 1,
            export_root=self.fixture.output, export_only=True, demonstration_paths=paths)

    def test_resume_reuses_successful_demonstration_paths_and_merges_in_order(self):
        first_path = self.fixture.demo('first')
        first = self.execute([first_path])
        self.assertEqual(first['export']['sft']['count'], 1)
        again = self.execute()
        self.assertEqual(again['export']['output_dir'], first['export']['output_dir'])
        self.assertEqual(again['export']['sft']['count'], 1)
        self.assertEqual(again['demonstration_paths'], [str(first_path.resolve())])
        second_path = self.fixture.demo('second')
        merged = self.execute([second_path, first_path, second_path])
        self.assertEqual(merged['demonstration_paths'], [str(first_path.resolve()), str(second_path.resolve())])
        rows = [json.loads(line) for line in Path(merged['export']['sft']['path']).read_text().splitlines()]
        self.assertEqual([row['request_id'] for row in rows], ['first', 'second'])
        self.assertEqual(self.execute()['export']['sft']['count'], 2)

    def test_saved_descriptor_is_reverified_and_failure_preserves_prior_success(self):
        descriptor = self.fixture.demo('first')
        first = self.execute([descriptor])
        times = {'queued_at': 100, 'author_slot_acquired_at': 101, 'author_slot_released_at': 110}
        host_validation.write(self.fixture.ep/'training-status.json', {**first, **times})
        descriptor.write_text('{"schema_version":"invalid"}')
        with self.assertRaisesRegex(ValueError, 'demonstration validation failed'):
            self.execute()
        state = host_validation.read(self.fixture.ep/'training-status.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['export'], first['export'])
        self.assertEqual(state['export']['sft']['count'], 1)
        self.assertTrue(state['export_attempt']['sft']['rejections'])
        self.assertEqual({key: state[key] for key in times}, times)

    def test_legacy_success_without_saved_paths_cannot_silently_downgrade(self):
        descriptor = self.fixture.demo('first')
        first = self.execute([descriptor])
        path = self.fixture.ep/'training-status.json'
        legacy = host_validation.read(path)
        del legacy['demonstration_paths']
        host_validation.write(path, legacy)
        with self.assertRaisesRegex(ValueError, 'successful legacy SFT export'):
            self.execute()
        failed = host_validation.read(path)
        self.assertEqual(failed['export'], first['export'])
        self.assertEqual(failed['export']['sft']['status'], 'exported')


class TrainingSchedulingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root/'config.json'; self.config.write_text('{}')
        self.jobs = [{'id': 'first', 'original': 'source-a'}, {'id': 'second', 'original': 'source-b'},
                     {'id': 'third', 'original': 'source-a'}]
        self.manifest = {'jobs': self.jobs}
        self.manifest['joblist_id'] = run_training._digest(self.manifest)
        self.path = self.root/'jobs.json'; self.path.write_text(json.dumps(self.manifest))
        self.argv = ['run_training', '--config', str(self.config), '--benchmark', 'bench', '--run-id', 'schedule',
                     '--joblist', str(self.path), '--export-root', str(self.root/'exports'), '--export-only']

    def invoke(self, execute, extra=(), log=None):
        with patch.object(run_training, 'ROOT', self.root), \
             patch.object(run_training, 'execute_job', side_effect=execute), \
             patch.object(run_training, 'log', side_effect=log), \
             patch.object(sys, 'argv', [*self.argv, *extra]):
            return run_training.main()

    def test_blocked_first_job_does_not_block_later_jobs_or_failure_collection(self):
        first_started, third_finished, failure_collected = threading.Event(), threading.Event(), threading.Event()
        first_unblocked = threading.Event()
        seen, mutex = {}, threading.Lock()
        def execute(cfg, benchmark, run_id, job, episode, **kwargs):
            with mutex:
                seen[job['id']] = episode
            if job['id'] == 'first':
                first_started.set()
                if not failure_collected.wait(5) or not third_finished.wait(5):
                    raise AssertionError('later jobs or failure collection waited for first job')
                first_unblocked.set()
            elif job['id'] == 'second':
                if not first_started.wait(5):
                    raise AssertionError('first job did not start')
                raise RuntimeError('independent second-job failure')
            else:
                third_finished.set()
            return {'status': 'completed'}
        def log(topic, message):
            if 'source-b: failed' in message:
                failure_collected.set()
        result = self.invoke(execute, ['--candidate-concurrency', '2'], log)
        self.assertEqual(result, 1)
        self.assertTrue(failure_collected.is_set())
        self.assertTrue(third_finished.is_set())
        self.assertTrue(first_unblocked.is_set(), 'first job must complete normally after later-job progress')
        self.assertEqual(seen, {'first': 1, 'second': 1, 'third': 2})

    def test_filter_and_concurrency_do_not_change_episode_identity_or_plan(self):
        seen = []
        def execute(cfg, benchmark, run_id, job, episode, **kwargs):
            seen.append((job['id'], episode))
            self.assertNotIn('candidate_concurrency', cfg)
            return {'status': 'completed'}
        self.assertEqual(self.invoke(execute, ['--candidate-concurrency', '2', '--job-id', 'third']), 0)
        plan = next((self.root/'runs/schedule').glob('training-plan-*.json'))
        before = plan.read_bytes()
        self.assertEqual(seen, [('third', 2)])
        self.assertEqual(self.invoke(execute, ['--job-id', 'third']), 0)
        self.assertEqual(plan.read_bytes(), before)
        self.assertEqual(self.path.read_text(), json.dumps(self.manifest))

    def test_cli_rejects_candidate_concurrency_outside_one_to_four(self):
        import contextlib
        import io
        for value in ('0', '5', '-1', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(lambda *args, **kwargs: self.fail('must not execute'),
                                ['--candidate-concurrency', value])
                self.assertEqual(error.exception.code, 2)

    def test_cli_audit_concurrency_defaults_to_sixteen_and_allows_override(self):
        seen = []
        def execute(*args, **kwargs):
            seen.append(kwargs['audit_concurrency'])
            return {'status': 'completed'}
        self.assertEqual(self.invoke(execute), 0)
        self.assertEqual(seen, [16] * 3)
        seen.clear()
        self.assertEqual(self.invoke(execute, ['--audit-concurrency', '8']), 0)
        self.assertEqual(seen, [8] * 3)

    def test_cli_rejects_audit_concurrency_outside_one_to_sixteen(self):
        import contextlib
        import io
        for value in ('0', '17', '-1', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(lambda *args, **kwargs: self.fail('must not execute'),
                                ['--audit-concurrency', value])
                self.assertEqual(error.exception.code, 2)

    def test_explicit_repair_assignment_survives_job_filtering(self):
        self.manifest['episode_assignments'] = {'first': 7, 'second': 1, 'third': 9}
        self.manifest['joblist_id'] = run_training._digest({
            k: v for k, v in self.manifest.items() if k != 'joblist_id'})
        self.path.write_text(json.dumps(self.manifest))
        seen = []
        def execute(cfg, benchmark, run_id, job, episode, **kwargs):
            seen.append((job['id'], episode))
            return {'status': 'completed'}
        self.assertEqual(self.invoke(execute, ['--job-id', 'third']), 0)
        self.assertEqual(seen, [('third', 9)])

    def test_cli_sampling_fault_returns_failure_and_keeps_other_jobs_running(self):
        seen = []
        def execute(cfg, benchmark, run_id, job, episode, **kwargs):
            seen.append(job['id'])
            return {'status': 'sampling_incomplete' if job['id'] == 'first' else 'completed'}
        self.assertEqual(self.invoke(execute, ['--candidate-concurrency', '2']), 1)
        self.assertEqual(set(seen), {'first', 'second', 'third'})

    def test_cli_distinguishes_semantic_rejection_from_review_fault(self):
        for verdict, expected in [('incorrect', 0), ('fault', 1)]:
            with self.subTest(verdict=verdict):
                outcome = {'status': 'quality_blocked', 'export': {
                    'quality_gate': {'issues': [{'id': 'check', 'verdict': verdict}]}}}
                self.assertEqual(self.invoke(lambda *args, **kwargs: outcome), expected)

    def test_cli_missing_outcome_is_not_success(self):
        self.assertEqual(self.invoke(lambda *args, **kwargs: None), 1)


if __name__ == '__main__':
    unittest.main()
