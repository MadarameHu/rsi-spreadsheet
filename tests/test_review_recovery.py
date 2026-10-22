"""Explicit review recovery reuses verified completions and preserves failed trials."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import unittest
from unittest.mock import patch

import test_parallel_audit_review as parallel_fixtures
import test_audit_scope as scope_fixtures

ReviewHarbor, parallel, audit = parallel_fixtures.ReviewHarbor, parallel_fixtures.parallel, parallel_fixtures.audit
FakeHarbor = scope_fixtures.FakeHarbor


class SourceRecoveryTests(unittest.TestCase):
    setUp = parallel_fixtures.ParallelReviewTests.setUp
    parallel_fixture = parallel_fixtures.ParallelReviewTests.parallel_fixture
    inputs = parallel_fixtures.ParallelReviewTests.inputs

    def test_failed_units_retry_once_in_new_directory_and_completed_units_are_reused(self):
        run, manifest = self.inputs(source_only=True)
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(ReviewHarbor(fail='sample-03'), self.cfg, manifest, run, [manifest.parent])
        original = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
        good = ReviewHarbor()
        result = parallel.run_source_review(good, self.cfg, manifest, run, [manifest.parent],
            recovery_id='resume-01', recovery_from=run)
        self.assertEqual(set(good.calls), {'sample-03', 'claim-c1', 'claim-c2'})
        self.assertEqual(Path(result['review_files']['sample-00']['terminal_path']),
                         (run / 'reviews/sample-00/terminal.json').resolve())
        self.assertIn('/recoveries/resume-01/', result['review_files']['sample-03']['terminal_path'])
        self.assertTrue(all(p.read_bytes() == data for p, data in original.items()))
        verified = parallel.verify_source_review(result, audit.read_json(manifest), cfg=self.cfg)
        self.assertEqual(len(verified['raw']['observations']), 3)
        cached = parallel.run_source_review(good, self.cfg, manifest, run, [manifest.parent],
            recovery_id='resume-01', recovery_from=run)
        self.assertEqual(cached['receipt_sha256'], result['receipt_sha256'])
        self.assertEqual(len(good.calls), 3)
        default = parallel.run_source_review(good, self.cfg, manifest, run, [manifest.parent])
        self.assertEqual(default['receipt_sha256'], result['receipt_sha256'])
        self.assertEqual(len(good.calls), 3)

    def test_explicit_recovery_does_not_automatically_repeat_a_second_failure(self):
        run, manifest = self.inputs(source_only=True)
        bad = ReviewHarbor(fail='sample-03')
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(bad, self.cfg, manifest, run, [manifest.parent])
        for _ in range(2):
            with self.assertRaises(parallel.ParallelReviewError):
                parallel.run_source_review(bad, self.cfg, manifest, run, [manifest.parent],
                    recovery_id='retry-once', recovery_from=run)
        self.assertEqual(bad.calls.count('sample-03'), 2)
        self.assertEqual(bad.calls.count('sample-00'), 1)

    def test_tampered_failed_evidence_cannot_be_silently_retried(self):
        run, manifest = self.inputs(source_only=True)
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(ReviewHarbor(fail='sample-03'), self.cfg, manifest, run, [manifest.parent])
        (run / 'reviews/sample-03/work/review.json').write_text('{}')
        good = ReviewHarbor()
        with self.assertRaisesRegex(parallel.ParallelReviewError, 'prior failed review evidence changed'):
            parallel.run_source_review(good, self.cfg, manifest, run, [manifest.parent],
                recovery_id='resume-01', recovery_from=run)
        self.assertEqual(good.calls, [])

    def test_recovery_keeps_exact_required_subset(self):
        run, manifest = self.inputs(split=True, source_only=True)
        wanted = {'claims': ['c1']}
        bad = ReviewHarbor(fail='sample-03')
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(bad, self.cfg, manifest, run, [manifest.parent], required_units=wanted)
        good = ReviewHarbor()
        result = parallel.run_source_review(good, self.cfg, manifest, run, [manifest.parent],
            required_units=wanted, recovery_id='resume-01', recovery_from=run)
        self.assertEqual(good.calls, ['sample-03', 'claim-c1'])
        self.assertEqual(result['requested_units']['claims'], ['c1'])
        self.assertEqual(result['reviewed_units']['observations'], ['s03-o1'])

    def test_recovery_rejects_different_judge_without_model_calls(self):
        run, manifest = self.inputs(source_only=True)
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(ReviewHarbor(fail='sample-03'), self.cfg, manifest, run, [manifest.parent])
        changed = deepcopy(self.cfg)
        changed['judge']['budget_min'] += 1
        good = ReviewHarbor()
        with self.assertRaisesRegex(parallel.ParallelReviewError, 'different inputs or judge'):
            parallel.run_source_review(good, changed, manifest, run, [manifest.parent],
                recovery_id='resume-01', recovery_from=run)
        self.assertEqual(good.calls, [])


class AuditRecoveryTests(unittest.TestCase):
    setUp = scope_fixtures.AuditScopeTests.setUp
    parallel_fixture = scope_fixtures.AuditScopeTests.parallel_fixture

    def test_source_failure_recovery_reuses_completed_task_and_probe_results(self):
        from concurrent.futures import Future
        self.parallel_fixture(compact=True)
        inputs = audit.load_inputs(self.ep)
        manifest, _ = audit._freeze_evidence(inputs, {}, self.ep / 'source-audit/evidence')
        source = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest,
                                            self.ep / 'source-audit', [manifest.parent])
        class ProbeHarbor(ReviewHarbor):
            run = FakeHarbor.run
            def __init__(self):
                super().__init__()
                self.probe_calls = []
                self.infra = False
        failed_source = Future()
        failed_source.set_exception(RuntimeError('source cache needs explicit recovery'))
        first = ProbeHarbor()
        with patch('host_validation.execution_slot', side_effect=lambda *args: nullcontext()):
            old = audit.audit_episode(first, self.cfg, 'spreadsheet', 'source', self.ep,
                scope='selected_design', source_review=failed_source)
        self.assertEqual(old['status'], 'failed')
        self.assertEqual(first.calls, ['task'])
        self.assertEqual(len(first.probe_calls), 2)
        origin = Path(old['work_dir'])
        original = {p: p.read_bytes() for p in origin.rglob('*') if p.is_file()}
        good = ProbeHarbor()
        result = audit.audit_episode(good, self.cfg, 'spreadsheet', 'source', self.ep,
            scope='selected_design', source_review=source,
            recovery_id='source-recovery-01', recovery_from=origin)
        self.assertEqual(result['status'], 'completed', result['failures'])
        self.assertEqual(good.calls, [])
        self.assertEqual(good.probe_calls, [])
        self.assertTrue(all(p.read_bytes() == raw for p, raw in original.items()))
        self.assertEqual(result['provenance']['review_files']['task']['terminal_path'],
                         str(origin / 'reviews/task/terminal.json'))
        self.assertEqual([p['execution_status'] for p in result['payload']['probes']], ['completed', 'completed'])
        cached = audit.audit_episode(good, self.cfg, 'spreadsheet', 'source', self.ep,
            scope='selected_design', source_review=source)
        self.assertEqual(cached['audit_commit'], result['audit_commit'])

    def test_parallel_task_recovery_only_calls_task_and_reuses_source_receipt(self):
        self.parallel_fixture(compact=True)
        inputs = audit.load_inputs(self.ep)
        manifest, _ = audit._freeze_evidence(inputs, {}, self.ep / 'source-audit/evidence')
        source = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest,
                                            self.ep / 'source-audit', [manifest.parent])
        class ProbeHarbor(ReviewHarbor):
            run = FakeHarbor.run
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.probe_calls = []
                self.infra = False
        old = audit.audit_episode(ProbeHarbor(fail='task'), self.cfg, 'spreadsheet', 'source', self.ep,
            scope='selected_design', source_review=source)
        origin = Path(old['work_dir'])
        original = {p: p.read_bytes() for p in origin.rglob('*') if p.is_file()}
        good = ProbeHarbor()
        with patch('host_validation.execution_slot', side_effect=lambda *args: nullcontext()):
            result = audit.audit_episode(good, self.cfg, 'spreadsheet', 'source', self.ep,
                scope='selected_design', source_review=source,
                recovery_id='resume-01', recovery_from=origin)
        self.assertEqual(result['status'], 'completed', result['failures'])
        self.assertEqual(good.calls, ['task'])
        self.assertEqual(len(good.probe_calls), 2)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in original.items()))
        again = audit.audit_episode(good, self.cfg, 'spreadsheet', 'source', self.ep,
            scope='selected_design', source_review=source)
        self.assertEqual(again['audit_commit'], result['audit_commit'])
        self.assertEqual(good.calls, ['task'])

    def test_recovery_cannot_change_budget_or_scope(self):
        self.parallel_fixture(compact=True)
        old = audit.audit_episode(FakeHarbor(malformed=True), self.cfg, 'spreadsheet', 'source', self.ep,
                                  scope='selected_design')
        for scope, cfg in [('full', self.cfg), ('selected_design', deepcopy(self.cfg))]:
            if scope == 'selected_design':
                cfg['judge']['budget_min'] += 1
            good = FakeHarbor()
            result = audit.audit_episode(good, cfg, 'spreadsheet', 'source', self.ep,
                scope=scope, recovery_id='resume-01', recovery_from=old['work_dir'])
            self.assertEqual(result['status'], 'failed')
            self.assertIn('cannot change evidence, scope, judge, or execution budgets', result['failures'][0])
            self.assertEqual(good.calls, [])

    def test_parallel_recovery_requires_explicit_source_reuse(self):
        self.parallel_fixture(compact=True)
        old = audit.audit_episode(ReviewHarbor(fail='task'), self.cfg, 'spreadsheet', 'source', self.ep,
                                  scope='selected_design')
        result = audit.audit_episode(ReviewHarbor(), self.cfg, 'spreadsheet', 'source', self.ep,
            scope='selected_design', recovery_id='resume-01', recovery_from=old['work_dir'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('explicit reusable source review', result['failures'][0])


if __name__ == '__main__':
    unittest.main()
