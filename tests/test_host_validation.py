"""Host validation preserves attempt identity across completion and interruption."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import host_validation as validation
import stage_audit
from stage_records import snapshot_tree


class FakeValidationHarbor:
    bin = '/fixture/harbor'
    api_key = 'fixture-secret-not-for-disk'
    base_url = 'http://fixture.invalid'
    setup_mult = 2

    def __init__(self, *, synchronize=False, oracle_reward=1, exception=None,
                 reward=0, returncode=0, raises=False):
        self.calls = []
        self.active = self.maximum = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(3) if synchronize else None
        self.oracle_reward = oracle_reward
        self.exception = exception
        self.reward = reward
        self.returncode = returncode
        self.raises = raises

    @staticmethod
    def receipt(job, reward=1, exception=None):
        path = Path(job)/'run/trial/result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Real Harbor trial structure; deliberately not a job-average result.
        path.write_text(json.dumps({
            'started_at': '2026-09-24T01:00:00+00:00',
            'finished_at': '2026-09-24T01:00:02+00:00',
            'agent_execution': {'started_at': '2026-09-24T01:00:00+00:00',
                                'finished_at': '2026-09-24T01:00:01+00:00'},
            'exception_info': exception,
            'verifier_result': {'rewards': {'reward': reward}},
        }))
        return path

    def run(self, args, job, timeout):
        with self.lock:
            self.calls.append(('oracle', str(job)))
        self.receipt(job, self.oracle_reward)
        return self.returncode

    def customer(self, task, model, job, timeout, **kwargs):
        with self.lock:
            self.calls.append(('trainee', str(job)))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            if self.barrier:
                self.barrier.wait(timeout=5)
            if self.raises:
                raise RuntimeError('transport interrupted without finished trial')
            self.receipt(job, self.reward, self.exception)
            # common.Harbor.customer returns parsed reward, not process status.
            return self.reward
        finally:
            with self.lock:
                self.active -= 1


class HostValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        target_root = patch.object(validation, 'TARGET_EXECUTION_ROOT',
                                   self.root/'runs/.target-execution-slots')
        target_root.start()
        self.addCleanup(target_root.stop)
        self.ep = self.root/'runs/run/spreadsheet/task/ep01'
        self.diag = self.ep/'diagnostics'
        self.diag.mkdir(parents=True)
        (self.ep/'work').mkdir()
        source = self.root/'authored'
        source.mkdir()
        (source/'task.toml').write_text('[agent]\ntimeout_sec=30\n[environment]\ncpus=1\n[metadata]\npass_threshold=1.0\n')
        (source/'instruction.md').write_text('Compute the complete required output.')
        manifest = snapshot_tree(source, self.diag/'candidate')
        self.upstream = {'candidate': manifest,
                         'attribution': {'commit_id': 'attribution-commit'},
                         'design': {'commit_id': 'design-commit'}}
        validation.write(self.diag/'attribution.json', {'claims': [{'id': 'c1'}]})
        validation.write(self.diag/'design.json', {'primary_goal_id': 'g1', 'mappings': []})
        validation.write(self.ep/'work/design.json', {'primary_goal_id': 'g1', 'targets': []})
        self.cfg = {'gateway': {'api_key': 'fixture-secret-not-for-disk',
                                'base_url': 'http://fixture.invalid'},
                    'score': {'n_concurrent': 3, 'outer_wall_task_multiplier': 2,
                              'setup_timeout_mult': 2},
                    'researcher': {'max_concurrent_docker': 3},
                    'customer': {'model': 'target', 'agent': 'fixture', 'closed_book': True},
                    # Three fake trainees keep controller tests small; the default is tested separately.
                    'training_pipeline': {'rollout': {'attempts': 3}, 'review': 'llm'},
                    'task_timeout_sec': {'spreadsheet': 60}}
        mock = patch.object(stage_audit, 'load_inputs', side_effect=lambda _: deepcopy(self.upstream))
        mock.start()
        self.addCleanup(mock.stop)

    def request(self, hb, attempts=None, max_concurrent=None):
        return validation.prepare_request(hb, self.cfg, 'spreadsheet', self.ep,
                                          attempts=attempts, max_concurrent=max_concurrent)

    def seed_finished(self, request, kind, index=0):
        path = validation.prepare_case(request, kind, index)
        validation.write(path/'host-started.json', {'started_at': validation.now()})
        FakeValidationHarbor.receipt(path/'job')
        return path

    def run_configured_rollouts(self, count):
        self.cfg['score']['n_concurrent'] = count
        self.cfg['researcher']['max_concurrent_docker'] = 1
        self.cfg['training_pipeline'] = {'rollout': {'attempts': count, 'max_concurrent': count}}
        hb = FakeValidationHarbor()
        hb.barrier = threading.Barrier(count)
        path = self.request(hb)
        request_bytes = path.read_bytes()
        reference = validation.run_worker(path, hb, phase='reference')
        self.assertEqual(reference['status'], 'reference_completed')
        self.assertEqual(reference['requested_attempts'], count)
        self.assertEqual(reference['completed_attempts'], 0)
        self.assertTrue(all(a['status'] == 'pending' for a in reference['attempts']))
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])
        request = validation.read(path)
        oracle_dir = validation.case_directory(request, 'oracle')
        oracle_files = {p.relative_to(oracle_dir): p.read_bytes()
                        for p in oracle_dir.rglob('*') if p.is_file()}
        state_path = self.diag/'validation.json'
        reference_bytes = state_path.read_bytes()
        self.assertEqual(validation.run_worker(path, hb, phase='reference'), reference)
        self.assertEqual(state_path.read_bytes(), reference_bytes)
        self.assertEqual(self.request(hb), path)
        result = validation.run_worker(path, hb, phase='trainee')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], count)
        self.assertEqual(result['solved_attempts'], 0)
        self.assertEqual(hb.maximum, count)
        self.assertEqual([k for k, _ in hb.calls].count('oracle'), 1)
        self.assertEqual([k for k, _ in hb.calls].count('trainee'), count)
        self.assertEqual(path.read_bytes(), request_bytes)
        self.assertEqual({p.relative_to(oracle_dir): p.read_bytes()
                          for p in oracle_dir.rglob('*') if p.is_file()}, oracle_files)
        self.assertIsNone(result['quality_approved'])
        validation.run_worker(path, hb, phase='trainee')
        self.assertEqual(len(hb.calls), count + 1)

    def test_four_configured_rollouts_have_independent_reference_and_trainee_phases(self):
        self.run_configured_rollouts(4)

    def test_eight_configured_rollouts_ignore_author_container_limit(self):
        self.run_configured_rollouts(8)

    def test_trainee_only_does_not_start_missing_reference_or_model_calls(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        result = validation.run_worker(path, hb, phase='trainee')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['oracle']['status'], 'pending')
        self.assertEqual(hb.calls, [])
        self.assertFalse(validation.case_directory(validation.read(path), 'oracle').exists())

    def test_reference_phase_does_not_replace_an_unknown_started_oracle(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        request = validation.read(path)
        oracle_dir = validation.prepare_case(request, 'oracle', 0)
        validation.write(oracle_dir/'host-started.json', {'started_at': validation.now()})
        result = validation.run_worker(path, hb, phase='reference')
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['oracle']['reward'])
        self.assertEqual(hb.calls, [])

    def test_legacy_request_reuses_its_existing_concurrency_and_identity(self):
        self.cfg['researcher']['max_concurrent_docker'] = 1
        hb = FakeValidationHarbor()
        path = self.request(hb, max_concurrent=1)
        original = path.read_bytes()
        self.assertEqual(self.request(hb), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(validation.read(path)['attempts'], 3)
        self.assertEqual(validation.read(path)['max_concurrent'], 1)

    def test_unset_rollout_defaults_to_sixteen_attempts_capped_by_concurrency(self):
        self.assertEqual(validation.rollout_settings({'score': {'n_concurrent': 16}}), (16, 16))
        self.assertEqual(validation.rollout_settings({'score': {'n_concurrent': 4}}), (16, 4))

    def test_rollout_config_rejects_invalid_values_without_calls(self):
        for rollout in ({'attempts': 0}, {'attempts': True}, {'attempts': 4.0},
                        {'max_concurrent': 0}, {'max_concurrent': True},
                        {'max_concurrent': 4}, None):
            with self.subTest(rollout=rollout):
                self.cfg['training_pipeline'] = {'rollout': rollout}
                with self.assertRaises(ValueError):
                    self.request(FakeValidationHarbor())

    def test_launch_passes_reference_phase_without_changing_request_identity(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        original = path.read_bytes()
        with patch.object(validation.subprocess, 'Popen', return_value=Mock(pid=123)) as popen:
            validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep,
                                         phase='reference', detach=True)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index('--phase')+1], 'reference')
        self.assertEqual(path.read_bytes(), original)

    def test_training_all_phase_cannot_bypass_latest_quality_gate(self):
        self.cfg['training_pipeline']['candidate_job'] = {'id': 'fixture-job'}
        hb = FakeValidationHarbor()
        path = self.request(hb)
        gate = {'status': 'rejected', 'issues': [{'id': 'extra_probe', 'verdict': 'incorrect'}]}
        with patch('quality_gate.evaluate_quality', return_value=gate) as evaluate:
            result = validation.run_worker(path, hb, phase='all')
        self.assertEqual(result['status'], 'quality_blocked')
        self.assertEqual(result['quality_issues'], gate['issues'])
        self.assertEqual(result['quality_gate'], gate)
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])
        self.assertTrue(all(item['status'] == 'pending' for item in result['attempts']))
        evaluate.assert_called_once_with(self.ep)

    def test_training_reference_then_trainee_rechecks_quality_before_calls(self):
        self.cfg['training_pipeline']['candidate_job'] = {'id': 'fixture-job'}
        hb = FakeValidationHarbor()
        path = self.request(hb)
        with patch('quality_gate.evaluate_quality') as evaluate:
            reference = validation.run_worker(path, hb, phase='reference')
        self.assertEqual(reference['status'], 'reference_completed')
        evaluate.assert_not_called()
        with patch('quality_gate.evaluate_quality', return_value={'status': 'blocked', 'issues': []}):
            blocked = validation.run_worker(path, hb, phase='trainee')
        self.assertEqual(blocked['status'], 'quality_blocked')
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])
        with patch('quality_gate.evaluate_quality', return_value={'status': 'accepted', 'issues': []}) as evaluate:
            completed = validation.run_worker(path, hb, phase='trainee')
        self.assertEqual(completed['status'], 'completed')
        self.assertEqual(len(hb.calls), 4)
        evaluate.assert_called_once_with(self.ep)
        with patch('quality_gate.evaluate_quality') as evaluate:
            self.assertEqual(validation.run_worker(path, hb, phase='trainee'), completed)
        evaluate.assert_not_called()
        self.assertEqual(len(hb.calls), 4)

    def test_training_quality_evaluation_fault_is_explicitly_blocked(self):
        self.cfg['training_pipeline']['candidate_job'] = {'id': 'fixture-job'}
        hb = FakeValidationHarbor()
        path = self.request(hb)
        with patch('quality_gate.evaluate_quality', side_effect=RuntimeError('cannot verify gate evidence')):
            result = validation.run_worker(path, hb)
        self.assertEqual(result['status'], 'quality_blocked')
        self.assertIn('cannot verify gate evidence', str(result['quality_issues']))
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])

    def test_quality_blocked_controller_exit_cannot_return_to_old_export_path(self):
        hb = FakeValidationHarbor()
        request = validation.read(self.request(hb))
        expected = {'request_id': request['request_id'], 'status': 'quality_blocked', 'quality_issues': []}
        validation.write(self.diag/'validation.json', expected)
        process = Mock(pid=321)
        process.wait.return_value = 1
        with patch.object(validation.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'controller exited 1'):
                validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep)
        self.assertEqual(validation.read(self.diag/'validation.json'), expected)

    def test_oracle_precedes_three_concurrent_trainees_and_reentry_makes_no_calls(self):
        hb = FakeValidationHarbor(synchronize=True)
        request_path = self.request(hb)
        self.assertNotIn(hb.api_key, request_path.read_text())
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], 3)
        self.assertEqual(result['solved_attempts'], 0)
        self.assertEqual(hb.calls[0][0], 'oracle')
        self.assertEqual(hb.maximum, 3)
        self.assertEqual(len(hb.calls), 4)
        self.assertIsNone(result['quality_approved'])
        again = validation.run_worker(request_path, hb)
        self.assertEqual(again['status'], 'completed')
        self.assertEqual(len(hb.calls), 4)

    def test_multiple_controllers_do_not_duplicate_model_calls(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: validation.run_worker(request_path, hb), range(2)))
        self.assertTrue(all(result['status'] == 'completed' for result in results))
        self.assertEqual(len(hb.calls), 4)

    def test_finished_trials_without_host_receipts_are_collected_not_repeated(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        request = validation.read(request_path)
        self.seed_finished(request, 'oracle')
        finished = self.seed_finished(request, 'trainee', 0)
        self.assertFalse((finished/'host-result.json').exists())
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([kind for kind, _ in hb.calls], ['trainee', 'trainee'])
        self.assertEqual(result['attempts'][0]['reward'], 1)

    def test_started_without_terminal_does_not_launch_replacement_samples(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        request = validation.read(request_path)
        self.seed_finished(request, 'oracle')
        path = validation.prepare_case(request, 'trainee', 0)
        validation.write(path/'host-started.json', {'started_at': validation.now()})
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(hb.calls, [])
        self.assertEqual(result['attempts'][0]['status'], 'incomplete')
        self.assertEqual(result['completed_attempts'], 0)

    def test_partial_started_marker_is_unknown_not_permission_to_resample(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        request = validation.read(request_path)
        self.seed_finished(request, 'oracle')
        path = validation.prepare_case(request, 'trainee', 0)
        (path/'host-started.json').write_text('{"started_at":')
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(hb.calls, [])
        self.assertIsNone(result['attempts'][0]['reward'])

    def test_partial_unbound_trial_result_stays_incomplete_without_resampling(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        request = validation.read(request_path)
        self.seed_finished(request, 'oracle')
        path = validation.prepare_case(request, 'trainee', 0)
        validation.write(path/'host-started.json', {'started_at': validation.now()})
        trial = path/'job/run/trial/result.json'
        trial.parent.mkdir(parents=True)
        trial.write_text('{"finished_at":')
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(hb.calls, [])
        self.assertIsNone(result['attempts'][0]['reward'])
        self.assertTrue(result['attempts'][0]['error'])
        self.assertEqual(trial.read_text(), '{"finished_at":')

    def test_corrupt_bound_trial_result_is_rejected_without_overwriting_receipt(self):
        hb = FakeValidationHarbor()
        request_path = self.request(hb)
        validation.run_worker(request_path, hb)
        request = validation.read(request_path)
        path = validation.case_directory(request, 'trainee', 0)
        receipt = path/'host-result.json'
        original = receipt.read_bytes()
        trial = path/'job/run/trial/result.json'
        trial.write_text('{"finished_at":')
        with self.assertRaises(ValueError):
            validation.run_worker(request_path, hb)
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(len(hb.calls), 4)
        self.assertEqual(validation.read(self.diag/'validation.json')['status'], 'failed')

    def test_launch_detaches_controller_and_passes_credentials_only_in_environment(self):
        hb = FakeValidationHarbor()
        process = Mock(pid=321)
        with patch.object(validation.subprocess, 'Popen', return_value=process) as popen:
            result = validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep, detach=True)
        self.assertEqual(result['status'], 'launched')
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(popen.call_args.kwargs['stdin'], validation.subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs['env']['VALIDATION_CREDENTIAL'], hb.api_key)
        self.assertNotIn(hb.api_key, str(popen.call_args.args))
        self.assertNotIn(hb.api_key, Path(result['request']).read_text())
        process.wait.assert_not_called()

    def test_failed_controller_cannot_return_stale_completed_or_running_receipt(self):
        hb = FakeValidationHarbor()
        request = validation.read(self.request(hb))
        for status in ('completed', 'running'):
            with self.subTest(status=status):
                validation.write(self.diag/'validation.json',
                                 {'request_id': request['request_id'], 'status': status})
                process = Mock(pid=321)
                process.wait.return_value = 7
                with patch.object(validation.subprocess, 'Popen', return_value=process):
                    with self.assertRaisesRegex(RuntimeError, 'controller exited 7'):
                        validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep)
                self.assertEqual(validation.read(self.diag/'validation.json')['status'], status)
                self.assertEqual(hb.calls, [])

    def test_launch_rejects_receipt_for_a_different_request_even_after_zero_exit(self):
        hb = FakeValidationHarbor()
        self.request(hb)
        validation.write(self.diag/'validation.json',
                         {'request_id': 'different-request', 'status': 'completed'})
        process = Mock(pid=321)
        process.wait.return_value = 0
        with patch.object(validation.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'different request identity'):
                validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep)
        self.assertEqual(hb.calls, [])

    def test_failed_controller_returns_its_matching_explicit_failure_receipt(self):
        hb = FakeValidationHarbor()
        request = validation.read(self.request(hb))
        for status in ('failed', 'incomplete'):
            with self.subTest(status=status):
                expected = {'request_id': request['request_id'], 'status': status,
                            'error': 'Actual controller failure'}
                validation.write(self.diag/'validation.json', expected)
                process = Mock(pid=321)
                process.wait.return_value = 1
                with patch.object(validation.subprocess, 'Popen', return_value=process):
                    result = validation.launch_validation(hb, self.cfg, 'spreadsheet', self.ep)
                self.assertEqual(result, expected)

    def test_configured_concurrency_cap_is_enforced_before_execution(self):
        hb = FakeValidationHarbor()
        with self.assertRaisesRegex(ValueError, 'configured cap'):
            self.request(hb, max_concurrent=4)
        self.assertEqual(hb.calls, [])

    def test_execution_slots_share_one_cap_across_request_controllers(self):
        request = validation.read(self.request(FakeValidationHarbor()))
        second = dict(request, request_id='another-request-in-the-same-run')
        barrier = threading.Barrier(3)
        mutex = threading.Lock()
        counts = {'active': 0, 'maximum': 0}
        def enter(index):
            with validation.execution_slot(request if index < 3 else second):
                with mutex:
                    counts['active'] += 1
                    counts['maximum'] = max(counts['maximum'], counts['active'])
                try:
                    barrier.wait(timeout=5)
                finally:
                    with mutex:
                        counts['active'] -= 1
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(enter, range(6)))
        self.assertEqual(counts['maximum'], 3)
        self.assertEqual(counts['active'], 0)

    def test_oracle_rejection_does_not_start_trainees(self):
        hb = FakeValidationHarbor(oracle_reward=0)
        result = validation.run_worker(self.request(hb), hb)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])

    def test_exception_keeps_reward_null_and_is_not_retried(self):
        hb = FakeValidationHarbor(reward=1, exception={'exception_type': 'AgentTimeoutError'})
        request_path = self.request(hb)
        result = validation.run_worker(request_path, hb)
        self.assertNotEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], 0)
        self.assertTrue(all(case['reward'] is None for case in result['attempts']))
        validation.run_worker(request_path, hb)
        self.assertEqual(len(hb.calls), 4)

    def test_transport_failure_without_trial_is_retained_without_retry(self):
        hb = FakeValidationHarbor(raises=True)
        request_path = self.request(hb)
        first = validation.run_worker(request_path, hb)
        self.assertTrue(all(case['reward'] is None for case in first['attempts']))
        validation.run_worker(request_path, hb)
        self.assertEqual(len(hb.calls), 4)

    def test_nonfinite_reward_is_a_failed_null_score_not_a_write_crash(self):
        hb = FakeValidationHarbor(reward=float('nan'))
        result = validation.run_worker(self.request(hb), hb)
        self.assertNotEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], 0)
        self.assertTrue(all(case['reward'] is None for case in result['attempts']))

    def test_oracle_process_nonzero_keeps_finished_trial_score_and_process_error_separate(self):
        hb = FakeValidationHarbor(reward=1, returncode=7)
        request_path = self.request(hb)
        result = validation.run_worker(request_path, hb)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['oracle']['reward'], 1)
        self.assertEqual(result['oracle']['process']['returncode'], 7)
        self.assertTrue(all(case['process']['returncode'] is None for case in result['attempts']))
        validation.run_worker(request_path, hb)
        self.assertEqual(len(hb.calls), 4)

    def test_candidate_bytes_changed_after_request_are_rejected_before_calls(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        (self.diag/'candidate/instruction.md').write_text('Changed task.')
        with self.assertRaises(ValueError):
            validation.run_worker(path, hb)
        self.assertEqual(hb.calls, [])
        state = validation.read(self.diag/'validation.json')
        self.assertIn(state['status'], ('failed', 'incomplete'))
        self.assertTrue(state.get('error'))
        self.assertIsNotNone(state.get('finished_at'))

    def test_unexpected_future_exception_records_terminal_failure_and_reraises(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        execute = validation.execute_case
        def fail_one(harbor, cfg, request, kind, index):
            if kind == 'trainee' and index == 0:
                raise RuntimeError('unexpected execution worker failure')
            return execute(harbor, cfg, request, kind, index)
        with patch.object(validation, 'execute_case', side_effect=fail_one):
            with self.assertRaisesRegex(RuntimeError, 'unexpected execution worker failure'):
                validation.run_worker(path, hb)
        state = validation.read(self.diag/'validation.json')
        self.assertIn(state['status'], ('failed', 'incomplete'))
        self.assertIn('unexpected execution worker failure', str(state.get('error')))
        self.assertIsNotNone(state.get('finished_at'))

    def test_completed_resume_preserves_stage_finish_time_and_receipt_bytes(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        first = validation.run_worker(path, hb)
        state_path = self.diag/'validation.json'
        original_bytes = state_path.read_bytes()
        request = validation.read(path)
        case_paths = [validation.case_directory(request, 'oracle')]
        case_paths += [validation.case_directory(request, 'trainee', i) for i in range(3)]
        receipts = {case/'host-result.json': (case/'host-result.json').read_bytes()
                    for case in case_paths}
        second = validation.run_worker(path, hb)
        self.assertEqual(second['finished_at'], first['finished_at'])
        self.assertEqual(state_path.read_bytes(), original_bytes)
        self.assertTrue(all(receipt.read_bytes() == data for receipt, data in receipts.items()))
        self.assertEqual(len(hb.calls), 4)

    def test_completed_summary_tamper_is_rejected_without_reusing_false_metrics(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        original = validation.run_worker(path, hb)
        for field, value in [('solved_attempts', 99), ('completed_attempts', 99),
                             ('quality_approved', True), ('difficulty_status', 'approved')]:
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed[field] = value
                validation.write(self.diag/'validation.json', changed)
                with self.assertRaises(ValueError):
                    validation.run_worker(path, hb)
                rejected = validation.read(self.diag/'validation.json')
                self.assertEqual(rejected['status'], 'failed')
                self.assertTrue(rejected.get('error'))
                self.assertIsNone(rejected['quality_approved'])
                self.assertEqual(len(hb.calls), 4)

    def test_configuration_identity_change_is_rejected(self):
        hb = FakeValidationHarbor()
        self.request(hb)
        self.cfg['customer']['model'] = 'different-target'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.request(hb)
        self.assertEqual(hb.calls, [])

    def test_request_file_tamper_is_rejected_even_when_worker_called_directly(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        request = validation.read(path)
        request['config']['customer']['model'] = 'different-target'
        validation.write(path, request)
        with self.assertRaises(ValueError):
            validation.run_worker(path, hb)
        self.assertEqual(hb.calls, [])

    def test_changed_upstream_commit_is_rejected_before_calls(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        request = validation.read(path)
        self.assertEqual(request['attribution_commit'], self.upstream['attribution']['commit_id'])
        self.assertEqual(request['design_commit'], self.upstream['design']['commit_id'])
        self.upstream['design']['commit_id'] = 'different-design-commit'
        with self.assertRaisesRegex(ValueError, 'upstream changed'):
            validation.run_worker(path, hb)
        self.assertEqual(hb.calls, [])

    def test_changed_finished_trial_is_rejected_instead_of_reused_or_repeated(self):
        hb = FakeValidationHarbor()
        path = self.request(hb)
        validation.run_worker(path, hb)
        request = validation.read(path)
        trial = validation.case_directory(request, 'trainee', 0)/'job/run/trial/result.json'
        changed = validation.read(trial)
        changed['verifier_result']['rewards']['reward'] = 1
        validation.write(trial, changed)
        with self.assertRaisesRegex(ValueError, 'receipt or its source changed'):
            validation.run_worker(path, hb)
        self.assertEqual(len(hb.calls), 4)


if __name__ == '__main__':
    unittest.main()
