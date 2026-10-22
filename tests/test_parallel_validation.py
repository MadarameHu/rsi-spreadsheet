"""Parallel audit ordering and independently frozen reference evidence, no models."""
import copy
from contextlib import contextmanager
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import test_stage_audit as fixtures
import test_quality_gate as quality_fixtures
import stage_audit
import quality_gate
import parallel_audit_review
import run_episode
import host_validation


class ParallelAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.AuditTests('test_complete_probes_keep_advance_expectations_and_original_bytes')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.cfg = {**self.f.cfg, 'score': {'n_concurrent': 4}}
        self.hb = fixtures.FakeHarbor()
        self.probe_started = threading.Event()
        original = self.hb.run
        def execute(*args):
            run = Path(args[1]).parents[2]
            frozen = stage_audit.read_json(run/'probe-inputs/manifest.json')
            self.assertEqual(len(frozen['probes']), 2)
            self.assertTrue(all(Path(v['snapshot_path']).is_file() for v in frozen['probes'].values()))
            self.probe_started.set()
            return original(*args)
        self.hb.run = execute

    def runner(self, hb, cfg, manifest, run, ro, *, on_task_ready, **kwargs):
        self.assertEqual(kwargs['max_concurrent'], 16)
        self.assertNotIn('selftests', stage_audit.read_json(manifest)['snapshots'])
        expected = stage_audit.read_json(manifest)['expected_units']
        work = run/'reviews/task/work'
        work.mkdir(parents=True)
        stage_audit.write_json(work.parent/'input.json', {'key': 'task', 'kind': 'task', 'assigned': {
            'task_checks': expected['task_checks'], 'mappings': expected['mappings']}})
        task_raw = work/'review.json'
        hb.agent_task(None, 'fake', 'fake', work.parent/'job', 1, '[]', agent_env=[f'ADB_RESULT={task_raw}'])
        payload = stage_audit.read_json(task_raw)
        evidence = payload['task_checks'][0]['evidence']
        for kind in ('claims', 'observations'):
            if kind in expected:
                payload[kind] = [dict(id=i, verdict='correct', reason='independent', evidence=evidence)
                                 for i in expected[kind]]
        for mapping in payload['mappings']:
            mapping['upstream_verdict'] = 'correct'
        raw = run/'reviews/assembled-review.json'
        stage_audit.write_json(raw, payload)
        plan = stage_audit.read_json(work/'probe_plan.json')
        on_task_ready(stage_audit.read_json(raw), plan, work)
        # Simulated slow source review: probes must already run without waiting for this return.
        self.assertTrue(self.probe_started.wait(3), 'probes waited for source review instead of task review')
        on_task_ready(stage_audit.read_json(raw), plan, work)  # idempotence
        return dict(raw_path=str(raw), plan_path=str(work/'probe_plan.json'),
                    terminals={'task': {'status': 'completed'}}, review_files={},
                    field_sources={}, source_identity=None, evidence_files=[])

    def call(self):
        with patch.object(parallel_audit_review, 'run_parallel_review', side_effect=self.runner):
            return stage_audit.audit_episode(self.hb, self.cfg, 'spreadsheet', 'source',
                                             self.f.ep)

    def test_probes_start_after_complete_freeze_before_source_join_and_cache(self):
        first = self.call()
        self.assertEqual(first['status'], 'completed', first['failures'])
        self.assertEqual(first['review_mode'], 'parallel')
        self.assertEqual(len(self.hb.probe_calls), 2)
        second = self.call()
        self.assertEqual(first, second)
        self.assertEqual(len(self.hb.calls), 1)
        job_files = [p for p in first['output_hashes'] if '/job/' in p]
        self.assertTrue(job_files)
        commit = stage_audit.read_stage_commit(self.f.diag/'records', first['audit_commit'])
        archived = {r['original_path'] for r in commit['references']}
        self.assertTrue(set(job_files).issubset(archived))

    def test_parallel_failure_does_not_retry_after_template_change(self):
        with patch.object(parallel_audit_review, 'run_parallel_review', side_effect=ValueError('bad branch')) as runner:
            result = stage_audit.audit_episode(self.hb, self.cfg, 'spreadsheet', 'source', self.f.ep)
            self.assertEqual(result['status'], 'failed')
            with patch.object(stage_audit, 'tree_fingerprint', return_value='changed-template'):
                second = stage_audit.audit_episode(self.hb, self.cfg, 'spreadsheet', 'source', self.f.ep)
            self.assertEqual(second, result)
            self.assertEqual(runner.call_count, 1)

    def test_mode_tampering_is_rejected(self):
        result = self.call()
        self.assertEqual(result['status'], 'completed', result['failures'])
        result['review_mode'] = 'serial'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            stage_audit.verify_cached_envelope(result, stage_audit.load_inputs(self.f.ep),
                                               result['evidence_fingerprint'], stage_audit.expected_units(stage_audit.load_inputs(self.f.ep)))

    def test_completed_audit_reused_after_implementation_changes(self):
        first = self.call()
        real_sha = stage_audit.sha
        def changed(path):
            return 'new-implementation' if str(path) in (stage_audit.__file__, parallel_audit_review.__file__) else real_sha(path)
        with patch.object(stage_audit, 'sha', side_effect=changed):
            second = self.call()
        self.assertEqual(first, second)
        self.assertEqual(len(self.hb.calls), 1)


    def test_real_parallel_scheduler_commits_all_branches(self):
        self.f.parallel_fixture(compact=True)
        hb = self.hb
        def unit(task, agent, model, job, timeout, mounts, **kwargs):
            hb.calls.append(str(job))
            raw = Path(next(v.split('=', 1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
            work = raw.parent
            request = stage_audit.read_json(work.parent/'input.json')
            assigned, kind = request['assigned'], request['kind']
            evidence_path = work/'calculation.txt'
            evidence_path.write_text('Independent full calculation')
            evidence = [{'path': str(evidence_path), 'locator': 'full calculation'}]
            if kind in ('claims', 'observations'):
                value = {kind: [dict(id=i, verdict='correct', reason='evidence agrees', evidence=evidence) for i in assigned[kind]]}
            else:
                value = {'task_checks': [dict(id=i, verdict='correct', reason='checked', evidence=evidence) for i in assigned['task_checks']],
                         'mappings': [dict(id=i, implementation_verdict='correct', reason='covered', evidence=evidence) for i in assigned['mappings']], 'probes': []}
                plans = []
                for pid, valid in [('correct_solution', True), ('incorrect_solution', False)]:
                    workbook = work/(pid+'.xlsx')
                    workbook.write_bytes(b'correct' if valid else b'wrong')
                    value['probes'].append(dict(id=pid, expected_valid=valid, actual_accept=None, execution_status='missing', reason='independent expectation', evidence=evidence))
                    plans.append(dict(id=pid, expected_valid=valid, workbook=str(workbook), target_path='/app/data.xlsx', reason='independent expectation', evidence=evidence))
                stage_audit.write_json(work/'probe_plan.json', {'probes': plans})
            stage_audit.write_json(raw, value)
            Path(job).mkdir()
            stage_audit.write_json(Path(job)/'result.json', {'exception_info': None, 'agent_execution': {'duration': 1}})
            (Path(job)/'raw-trajectory.txt').write_text('All original model turns retained')
            return 0
        hb.agent_task = unit
        result = stage_audit.audit_episode(hb, self.cfg, 'spreadsheet', 'source', self.f.ep)
        self.assertEqual(result['status'], 'completed', result['failures'])
        self.assertEqual(len(result['payload']['observations']), 3)
        self.assertTrue(all(p['execution_status'] == 'completed' for p in result['payload']['probes']))
        self.assertGreaterEqual(len(hb.calls), 4)
        second = stage_audit.audit_episode(hb, self.cfg, 'spreadsheet', 'source', self.f.ep)
        self.assertEqual(result, second)


class ParallelReferenceTests(unittest.TestCase):
    def setUp(self):
        self.f = quality_fixtures.QualityGateTests('test_tampered_audit_summary_is_not_trusted')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def verify(self):
        ep = self.f.ep.resolve()
        inputs = stage_audit.load_inputs(ep)
        request_path, request, case = quality_gate._reference_request(ep, inputs)
        return quality_gate._parallel_reference_snapshot(ep, inputs, request_path, request, case)

    def test_reference_freeze_reuse_and_trainee_siblings(self):
        self.f.reference()
        first = quality_gate.freeze_reference_evidence(self.f.ep)
        self.assertEqual(first, quality_gate.freeze_reference_evidence(self.f.ep))
        case = Path(first['case_path'])
        (case.parent/'new-trainee').mkdir()
        self.assertTrue(self.verify().is_file())

    def test_reference_live_oracle_changes_are_rejected(self):
        self.f.reference()
        frozen = quality_gate.freeze_reference_evidence(self.f.ep)
        (Path(frozen['case_path'])/'unreviewed.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.verify()

    def test_reference_receipt_tampering_is_rejected(self):
        self.f.reference()
        quality_gate.freeze_reference_evidence(self.f.ep)
        path = self.f.ep/'diagnostics/validation/reference-evidence.json'
        value = stage_audit.read_json(path)
        value['result']['reward'] = 9
        stage_audit.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.verify()

    def test_episode_independent_reference_and_audit_join_before_gate(self):
        barrier = threading.Barrier(2)
        events = []
        token = object()
        @contextmanager
        def slot():
            self.fail('episode held a model slot while waiting on dependencies')
            yield
        def launch(*args, phase, **kwargs):
            if phase == 'reference':
                barrier.wait(timeout=3)
                events.append('reference')
                return {'status': 'reference_completed'}
            self.assertEqual(events[-1], 'gate')
            events.append('trainee')
            return {'status': 'completed'}
        def audit(*args, **kwargs):
            self.assertEqual(kwargs['max_concurrent'], 16)
            self.assertIs(kwargs['slot'], slot)
            self.assertIs(kwargs['source_review'], token)
            barrier.wait(timeout=3)
            events.append('audit')
            return {'status': 'completed'}
        def gate(*args, **kwargs):
            self.assertEqual(kwargs['max_concurrent'], 16)
            self.assertEqual(set(events), {'reference', 'audit', 'freeze'})
            self.assertTrue(kwargs['evaluate_only'])
            events.append('gate')
            return {'eligible_for_export': True}
        with patch.object(host_validation, 'launch_validation', side_effect=launch), \
             patch.object(stage_audit, 'audit_episode', side_effect=audit), \
             patch.object(quality_gate, 'freeze_reference_evidence', side_effect=lambda ep: events.append('freeze')), \
             patch.object(quality_gate, 'run_quality_gate', side_effect=gate), \
             patch.object(run_episode, 'make_harbor', return_value=object()):
            result = run_episode.validate_authored_candidate(object(), self.f.cfg, 'spreadsheet', self.f.ep,
                         audit_slot=slot, source_review=token)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(events[-1], 'trainee')
