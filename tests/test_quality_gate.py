"""Admission depends on committed semantic/probe/reference evidence, not solve rate."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import host_validation
import quality_gate
import stage_audit
import test_stage_audit as fixtures
from test_host_validation import FakeValidationHarbor


class QualityReviewer(fixtures.FakeHarbor):
    def __init__(self, judgments=None, false_accept=False, unknown_probe=False, extra_probe=None, **kwargs):
        super().__init__(**kwargs)
        self.judgments = judgments or {}
        self.false_accept = false_accept
        self.unknown_probe = unknown_probe
        self.extra_probe = extra_probe

    def agent_task(self, *args, **kwargs):
        rc = super().agent_task(*args, **kwargs)
        path = Path(next(v.split('=', 1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
        payload = stage_audit.read_json(path)
        # Each parallel unit writes only its own role.
        for item in payload.get('claims', []):
            # A wrong unrelated conclusion must not reject a correctly targeted task.
            item['verdict'] = self.judgments.get('claim:'+item['id'],
                                                'incorrect' if item['id'] == 'c2' else 'correct')
        if 'task_checks' not in payload:
            stage_audit.write_json(path, payload)
            return rc
        for item in payload['mappings']:
            item['implementation_verdict'] = self.judgments.get('mapping:implementation_verdict', 'correct')
        for item in payload['task_checks']:
            item['verdict'] = self.judgments.get('task:'+item['id'], 'correct')
        if self.unknown_probe:
            payload['probes'][0]['expected_valid'] = None
            plan = stage_audit.read_json(path.parent/'probe_plan.json')
            plan['probes'][0]['expected_valid'] = None
            stage_audit.write_json(path.parent/'probe_plan.json', plan)
        if self.extra_probe:
            expected = None if self.extra_probe == 'uncertain' else False
            workbook = path.parent/'targeted_requirement.xlsx'
            workbook.write_bytes(b'wrong')
            evidence = payload['probes'][0]['evidence']
            payload['probes'].append(dict(id='targeted_requirement', expected_valid=expected,
                actual_accept=None, execution_status='missing',
                reason='Independent targeted requirement counterexample.', evidence=evidence))
            plan = stage_audit.read_json(path.parent/'probe_plan.json')
            plan['probes'].append(dict(id='targeted_requirement', workbook=str(workbook),
                target_path='/app/data.xlsx', expected_valid=expected,
                reason='Independent targeted requirement counterexample.', evidence=evidence))
            stage_audit.write_json(path.parent/'probe_plan.json', plan)
        stage_audit.write_json(path, payload)
        return rc

    def run(self, args, job, timeout):
        rc = super().run(args, job, timeout)
        if self.false_accept and 'incorrect_solution' in str(job):
            path = Path(job)/'result.json'
            value = stage_audit.read_json(path)
            value['verifier_result']['rewards']['reward'] = 1
            stage_audit.write_json(path, value)
        if 'targeted_requirement' in str(job) and self.extra_probe in ('false_accept', 'infra'):
            path = Path(job)/'result.json'
            value = stage_audit.read_json(path)
            if self.extra_probe == 'false_accept':
                value['verifier_result']['rewards']['reward'] = 1
            else:
                value['exception_info'] = {'exception_type': 'EnvironmentSetupError'}
            stage_audit.write_json(path, value)
        return rc


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AuditTests('test_compact_raw_and_derived_worker_records_are_both_preserved')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.parallel_fixture(compact=True)
        self.ep = self.fixture.ep
        self.diag = self.fixture.diag
        (self.ep/'work').mkdir(exist_ok=True)
        (self.ep/'work/design.json').write_bytes((self.diag/'design.json').read_bytes())
        self.cfg = {**self.fixture.cfg,
                    'gateway': {'api_key': 'fixture-key', 'base_url': 'http://fixture.invalid'},
                    'researcher': {'max_concurrent_docker': 3},
                    'score': {'n_concurrent': 3, 'outer_wall_task_multiplier': 2, 'setup_timeout_mult': 2},
                    'customer': {'model': 'target', 'agent': 'fixture'},
                    'training_pipeline': {'rollout': {'attempts': 3}, 'review': 'llm'}}

    def reference(self, reward=1, exception=None):
        hb = FakeValidationHarbor(oracle_reward=reward)
        request_path = host_validation.prepare_request(hb, self.cfg, 'spreadsheet', self.ep)
        request = host_validation.read(request_path)
        result = host_validation.execute_case(hb, self.cfg, request, 'oracle', 0)
        if exception:
            # Simulate a genuine exceptional trial before it is reviewed/frozen.
            path = Path(result['result_path'])
            raw = stage_audit.read_json(path)
            raw['exception_info'] = exception
            stage_audit.write_json(path, raw)
            case = host_validation.case_directory(request, 'oracle')
            (case/'host-result.json').unlink()
            host_validation.write(case/'host-result.json', host_validation.collect_case(case, 'oracle'))
        quality_gate.freeze_reference_evidence(self.ep)
        return request

    def review(self, *, scope='full', **kwargs):
        hb = QualityReviewer(**kwargs)
        result = stage_audit.audit_episode(hb, self.cfg, 'spreadsheet', 'source', self.ep, scope=scope)
        self.assertEqual(result['status'], 'completed', result['failures'])
        return hb, result

    def test_selected_scope_checks_selected_units_and_all_extra_probes(self):
        self.reference()
        _, audit = self.review(scope='selected_design', extra_probe='pass')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'accepted', result['issues'])
        self.assertEqual(result['audit_scope']['mode'], 'selected_design')
        self.assertEqual(result['selected_claim_ids'], ['c1'])
        self.assertEqual(audit['expected_units']['claims'], ['c1'])
        self.assertEqual(audit['audit_scope']['unreviewed_ids']['claims'], ['c2'])
        self.assertIn('probe:targeted_requirement', [item['id'] for item in result['checks']])

    def test_selected_scope_extra_false_accept_still_rejects(self):
        self.reference()
        self.review(scope='selected_design', extra_probe='false_accept')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('probe:targeted_requirement', [item['id'] for item in result['issues']])

    def test_scope_subset_cannot_be_self_reported_to_skip_a_selected_claim(self):
        self.reference()
        _, audit = self.review(scope='selected_design')
        tampered = deepcopy(audit)
        tampered['audit_scope']['expected_units']['claims'] = []
        tampered['expected_units']['claims'] = []
        tampered['payload']['claims'] = []
        result = quality_gate.evaluate_quality(self.ep, audit=tampered)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['audit_commit'])

    def test_run_quality_gate_passes_scope_and_reuses_same_scope(self):
        self.reference()
        reviewer = QualityReviewer()
        result = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep,
                                               scope='selected_design')
        self.assertEqual(result['status'], 'accepted', result['issues'])
        self.assertEqual(result['audit_scope']['mode'], 'selected_design')
        calls = len(reviewer.calls)
        again = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep,
                                              scope='selected_design')
        self.assertEqual(again, result)
        self.assertEqual(len(reviewer.calls), calls)

    def test_explicit_scope_change_after_failed_full_review_keeps_previous_evidence(self):
        self.reference()
        failed = stage_audit.audit_episode(QualityReviewer(malformed=True), self.cfg,
                                           'spreadsheet', 'source', self.ep, scope='full')
        self.assertEqual(failed['status'], 'failed')
        retained = Path(failed['work_dir']) / 'audit.json'
        before = retained.read_bytes()
        reviewer = QualityReviewer()
        result = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep,
                                               scope='selected_design')
        self.assertEqual(result['status'], 'accepted', result['issues'])
        self.assertEqual(result['audit_scope']['mode'], 'selected_design')
        self.assertEqual(retained.read_bytes(), before)
        self.assertGreater(len(reviewer.calls), 0)

    def test_accepts_bound_complete_quality_without_requiring_target_success(self):
        self.reference()
        _, audit = self.review()
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'accepted', result['issues'])
        self.assertTrue(result['eligible_for_export'])
        self.assertTrue(result['quality_approved'])
        self.assertFalse(result['difficulty_used_for_admission'])
        self.assertEqual(result['selected_claim_ids'], ['c1'])
        self.assertEqual(result['selected_mapping_ids'], ['m1'])
        self.assertEqual(result['audit_commit'], audit['audit_commit'])
        self.assertFalse((self.diag/'quality.json').exists())

    def test_uncertain_task_blocks_without_claiming_confirmed_defect(self):
        self.reference()
        self.review(judgments={'task:ground_truth': 'uncertain'})
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])
        self.assertFalse(result['eligible_for_export'])

    def test_incorrect_selected_claim_rejects_candidate(self):
        self.reference()
        self.review(judgments={'claim:c1': 'incorrect'})
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertFalse(result['quality_approved'])

    def test_incorrect_mapping_implementation_rejects_candidate(self):
        self.reference()
        self.review(judgments={'mapping:implementation_verdict': 'incorrect'})
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('mapping:m1:implementation_verdict', [i['id'] for i in result['issues']])

    def test_false_accept_probe_rejects_candidate(self):
        self.reference()
        self.review(false_accept=True)
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('probe:incorrect_solution', [i['id'] for i in result['issues']])

    def test_additional_targeted_probe_false_accept_rejects_candidate(self):
        self.reference()
        self.review(extra_probe='false_accept')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertFalse(result['eligible_for_export'])
        self.assertIn('probe:targeted_requirement', [i['id'] for i in result['issues']])

    def test_additional_targeted_probe_matching_expectation_allows_admission(self):
        self.reference()
        self.review(extra_probe='pass')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'accepted', result['issues'])
        check = next(i for i in result['checks'] if i['id'] == 'probe:targeted_requirement')
        self.assertEqual(check['verdict'], 'correct')

    def test_additional_targeted_probe_unknown_expectation_blocks_admission(self):
        self.reference()
        self.review(extra_probe='uncertain')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])
        self.assertIn('probe:targeted_requirement', [i['id'] for i in result['issues']])

    def test_additional_targeted_probe_execution_failure_blocks_admission(self):
        self.reference()
        self.review(extra_probe='infra')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])
        self.assertIn('probe:targeted_requirement', [i['id'] for i in result['issues']])

    def test_probe_infrastructure_failure_blocks_without_becoming_wrong_answer(self):
        self.reference()
        self.review(infra=True)
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])

    def test_required_probe_with_unknown_expectation_blocks(self):
        self.reference()
        self.review(unknown_probe=True)
        self.assertEqual(quality_gate.evaluate_quality(self.ep)['status'], 'blocked')

    def test_independent_positive_probe_does_not_replace_author_reference(self):
        self.review()
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('reference_execution', [i['id'] for i in result['issues']])

    def test_finished_but_rejected_author_reference_is_confirmed_failure(self):
        self.reference(reward=0)
        self.review()
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('reference_execution', [i['id'] for i in result['issues']])

    def test_reference_infrastructure_exception_blocks_without_inventing_zero(self):
        self.reference(exception={'exception_type': 'EnvironmentSetupError'})
        self.review()
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])
        self.assertIsNone(result['reference_execution']['reward'])

    def test_tampered_audit_summary_is_not_trusted(self):
        self.reference()
        _, audit = self.review()
        altered = deepcopy(audit)
        altered['payload']['claims'][0]['verdict'] = 'incorrect'
        result = quality_gate.evaluate_quality(self.ep, audit=altered)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['quality_approved'])
        self.assertIsNone(result['audit_commit'])

    def test_frozen_oracle_tampering_blocks_admission(self):
        request = self.reference()
        self.review()
        path = host_validation.case_directory(request, 'oracle')/'job/run/trial/result.json'
        raw = stage_audit.read_json(path)
        raw['verifier_result']['rewards']['reward'] = 0
        stage_audit.write_json(path, raw)
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertTrue(any('immutable audit snapshot' in i['reason']
                            or 'frozen snapshot' in i['reason'] for i in result['issues']))

    def test_changed_actual_probe_evidence_is_rejected_by_audit_commit_binding(self):
        self.reference()
        _, review = self.review()
        path = Path(review['work_dir'])/'probes/incorrect_solution/job/result.json'
        raw = stage_audit.read_json(path)
        raw['verifier_result']['rewards']['reward'] = 1
        stage_audit.write_json(path, raw)
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['audit_commit'])

    def test_changed_candidate_is_not_admitted_using_old_audit(self):
        self.reference()
        self.review()
        (self.diag/'candidate/instruction.md').write_text('Changed contract.')
        result = quality_gate.evaluate_quality(self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['eligible_for_export'])

    def test_later_zero_reward_trainees_do_not_invalidate_or_repeat_paid_audit(self):
        self.reference()
        reviewer, _ = self.review()
        later = self.ep/'work/testruns/trainee-later/job/run/trial/result.json'
        later.parent.mkdir(parents=True)
        stage_audit.write_json(later, {'finished_at': 'now', 'exception_info': None,
                                      'verifier_result': {'rewards': {'reward': 0}}})
        with patch.object(stage_audit, 'audit_episode', side_effect=AssertionError('no repeated audit')):
            result = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep)
        self.assertEqual(result['status'], 'accepted', result['issues'])
        self.assertEqual(stage_audit.read_json(self.diag/'quality.json'), result)

    def test_missing_audit_runs_existing_stage_once_and_saves_gate(self):
        self.reference()
        reviewer = QualityReviewer()
        result = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep)
        self.assertEqual(result['status'], 'accepted', result['issues'])
        calls = len(reviewer.calls)
        again = quality_gate.run_quality_gate(reviewer, self.cfg, 'spreadsheet', 'source', self.ep)
        self.assertEqual(again, result)
        self.assertEqual(len(reviewer.calls), calls)

    def test_malformed_or_failed_audit_never_triggers_automatic_retry(self):
        (self.diag/'audit.json').write_text('{broken')
        with patch.object(stage_audit, 'audit_episode', side_effect=AssertionError('no automatic retry')):
            result = quality_gate.run_quality_gate(None, self.cfg, 'spreadsheet', 'source', self.ep)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['eligible_for_export'])


if __name__ == '__main__':
    unittest.main()
