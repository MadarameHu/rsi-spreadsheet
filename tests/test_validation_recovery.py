"""Explicit bridge nonexecution recovery preserves failures and never repeats scores."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

import test_host_validation as fixtures
import host_validation as hv
import validation_recovery as recovery


class FailedBridge(fixtures.FakeValidationHarbor):
    def __init__(self):
        super().__init__(reward=None, exception={'exception_type': 'RuntimeError',
                                                'exception_message': 'Native bridge HTTP 502'})

    def customer(self, task, model, job, timeout, **kwargs):
        result = super().customer(task, model, job, timeout, **kwargs)
        bridge = Path(job)/'run/trial/agent/bridge-request.json'
        bridge.parent.mkdir()
        hv.write(bridge, {'request_id': Path(job).parent.name, 'model': model, 'seed': 0,
                          'instruction': 'original complete instruction'})
        return result


class ValidationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.HostValidationTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.hb = FailedBridge()
        self.request_path = self.f.request(self.hb, attempts=2, max_concurrent=2)
        self.request = hv.read(self.request_path)
        self.original = hv.run_worker(self.request_path, self.hb)
        self.assertEqual(self.original['status'], 'incomplete')
        self.raw = self.f.root/'runtime-check.json'
        hv.write(self.raw, {'checked': 'both exact request dirs absent, no workers or pending calls'})
        self.proof_path = self.f.root/'proof.json'
        self.proof = dict(schema_version=recovery.PROOF_SCHEMA,
            request_id=self.request['request_id'], candidate_sha256=self.request['candidate_sha256'],
            observed_at=hv.now(), remote_host='fixture-remote', remote_job_root='/remote/jobs',
            checks=[dict(index=i, bridge_request_id=hv.original_case_directory(self.request, 'trainee', i).name,
                         remote_job_path='/remote/jobs/'+hv.original_case_directory(self.request, 'trainee', i).name,
                         remote_job_exists=False, worker_running=False, request_pending=False) for i in range(2)],
            evidence=[dict(path=str(self.raw), sha256=recovery._sha(self.raw))])
        hv.write(self.proof_path, self.proof)
        self.quality = dict(status='accepted', candidate_sha256=self.request['candidate_sha256'], audit_commit='audit-commit')

    def prepare(self):
        with patch('quality_gate.evaluate_quality', return_value=self.quality):
            return recovery.prepare_recovery(self.request_path, self.proof_path, [0, 1],
                                             'Explicit recovery after independent nonexecution inspection')

    def test_new_cases_run_once_and_old_failures_and_request_remain_unchanged(self):
        request_bytes = self.request_path.read_bytes()
        old_dirs = [hv.original_case_directory(self.request, 'trainee', i) for i in range(2)]
        old_bytes = {p: p.read_bytes() for d in old_dirs for p in d.rglob('*') if p.is_file()}
        oracle = hv.case_directory(self.request, 'oracle')
        oracle_bytes = {p: p.read_bytes() for p in oracle.rglob('*') if p.is_file()}
        previous = (self.f.diag/'validation.json').read_bytes()
        manifest_path = self.prepare()
        manifest = recovery.verify_recovery(self.request)
        self.assertEqual((manifest_path.parent/'previous-validation.json').read_bytes(), previous)
        self.assertEqual(hv.case_directory(self.request, 'oracle'), oracle)
        for i, old in enumerate(old_dirs):
            self.assertEqual(hv.case_directory(self.request, 'trainee', i), old.with_name(old.name+'--recovery-01'))
        fresh = fixtures.FakeValidationHarbor(reward=1)
        completed = hv.run_worker(self.request_path, fresh, phase='trainee')
        self.assertEqual(completed['status'], 'completed')
        self.assertEqual(completed['solved_attempts'], 2)
        self.assertEqual(completed['recovery']['recovery_id'], manifest['recovery_id'])
        self.assertEqual([kind for kind, _ in fresh.calls], ['trainee', 'trainee'])
        self.assertEqual(hv.run_worker(self.request_path, fresh, phase='trainee'), completed)
        self.assertEqual(len(fresh.calls), 2)
        self.assertEqual(self.request_path.read_bytes(), request_bytes)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in {**old_bytes, **oracle_bytes}.items()))
        with self.assertRaisesRegex(ValueError, 'only one explicit recovery'):
            self.prepare()

    def test_only_selected_failed_index_is_replaced(self):
        self.proof['checks'] = self.proof['checks'][:1]
        hv.write(self.proof_path, self.proof)
        with patch('quality_gate.evaluate_quality', return_value=self.quality):
            recovery.prepare_recovery(self.request_path, self.proof_path, [0], 'explicit one-index recovery')
        self.assertEqual(hv.case_directory(self.request, 'trainee', 1),
                         hv.original_case_directory(self.request, 'trainee', 1))
        hb = fixtures.FakeValidationHarbor(reward=1)
        result = hv.run_worker(self.request_path, hb, phase='trainee')
        self.assertEqual(len(hb.calls), 1)
        self.assertEqual([a['status'] for a in result['attempts']], ['completed', 'failed'])
        self.assertIsNone(result['attempts'][1]['reward'])
        self.assertEqual(result['status'], 'incomplete')

    def test_preparation_waits_for_existing_controller_lock(self):
        started, quality_checked = threading.Event(), threading.Event()
        def gate(*args):
            quality_checked.set()
            return self.quality
        def prepare():
            started.set()
            return recovery.prepare_recovery(self.request_path, self.proof_path, [0, 1], 'explicit recovery')
        with patch('quality_gate.evaluate_quality', side_effect=gate), ThreadPoolExecutor(max_workers=1) as pool:
            with hv.locked(self.request_path.parent/'controller.lock'):
                future = pool.submit(prepare)
                self.assertTrue(started.wait(2))
                self.assertFalse(quality_checked.wait(0.03))
            self.assertTrue(future.result(timeout=3).is_file())
        self.assertTrue(quality_checked.is_set())

    def test_original_candidate_tamper_is_rejected_before_recovery(self):
        old = hv.original_case_directory(self.request, 'trainee', 0)
        (old/'candidate/instruction.md').write_text('different candidate')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.request_path.parent/'recovery-01').exists())

    def test_plain_resume_does_not_repeat_failed_calls_without_recovery(self):
        self.assertIsNone(recovery.verify_recovery(self.request))
        hb = fixtures.FakeValidationHarbor(reward=1)
        result = hv.run_worker(self.request_path, hb, phase='trainee')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(hb.calls, [])
        self.assertNotIn('recovery', result)

    def test_unknown_or_present_remote_execution_is_rejected(self):
        for key, value in [('remote_job_exists', True), ('worker_running', True),
                           ('request_pending', True), ('request_pending', None)]:
            with self.subTest(key=key, value=value):
                proof = deepcopy(self.proof)
                proof['checks'][0][key] = value
                hv.write(self.proof_path, proof)
                with self.assertRaisesRegex(ValueError, 'never executed and is not pending'):
                    self.prepare()
                self.assertFalse((self.request_path.parent/'recovery-01').exists())

    def test_request_id_candidate_and_bridge_id_must_match(self):
        for key in ('request_id', 'candidate_sha256', 'bridge_request_id'):
            with self.subTest(key=key):
                proof = deepcopy(self.proof)
                if key == 'bridge_request_id':
                    proof['checks'][0][key] = 'different'
                else:
                    proof[key] = 'different'
                hv.write(self.proof_path, proof)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_inspection_before_terminal_is_rejected(self):
        self.proof['observed_at'] = '2000-01-01T00:00:00+00:00'
        hv.write(self.proof_path, self.proof)
        with self.assertRaisesRegex(ValueError, 'predates'):
            self.prepare()

    def test_true_zero_score_or_reported_zero_cannot_be_resampled(self):
        for status, reward, reported in [('completed', 0.0, 0.0), ('failed', None, 0.0),
                                         ('incomplete', None, None)]:
            with self.subTest(status=status, reported=reported):
                original = hv.collect_case
                def collect(path, kind):
                    value = original(path, kind)
                    if kind == 'trainee':
                        value.update(status=status, reward=reward, reported_reward=reported)
                    return value
                with patch.object(hv, 'collect_case', side_effect=collect):
                    with self.assertRaisesRegex(ValueError, 'failed, null-reward, finished'):
                        self.prepare()

    def test_quality_must_still_be_accepted(self):
        self.quality['status'] = 'blocked'
        with self.assertRaisesRegex(ValueError, 'quality accepted'):
            self.prepare()

    def test_raw_proof_tampering_is_rejected(self):
        self.raw.write_text('changed runtime inspection')
        with self.assertRaisesRegex(ValueError, 'inspection evidence changed'):
            self.prepare()

    def test_frozen_or_original_evidence_tampering_blocks_routing(self):
        path = self.prepare()
        old = hv.original_case_directory(self.request, 'trainee', 0)
        (old/'extra-unreviewed.txt').write_text('added after recovery')
        with self.assertRaisesRegex(ValueError, 'frozen recovery history'):
            hv.case_directory(self.request, 'trainee', 0)

    def test_frozen_proof_tampering_and_interrupted_prepare_never_fall_back(self):
        path = self.prepare()
        (path.parent/'proof.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen recovery evidence changed'):
            hv.case_directory(self.request, 'trainee', 0)
        # Simulate losing the last activation step; old failed cases are not reused.
        (self.request_path.parent/'active-recovery.json').unlink()
        with self.assertRaisesRegex(ValueError, 'preparation was interrupted'):
            hv.case_directory(self.request, 'trainee', 0)


if __name__ == '__main__':
    unittest.main()
