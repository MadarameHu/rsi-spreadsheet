from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import test_stage_audit as fixtures
import parallel_audit_review as parallel

audit = fixtures.audit


class ReviewHarbor:
    bin = '/fixture/harbor'
    setup_mult = 1

    def __init__(self, *, block=None, fail=None, claim_verdict='correct'):
        self.calls = []
        self.events = []
        self.active = self.maximum = 0
        self.lock = threading.Lock()
        self.block = block or {}
        self.fail = fail
        self.claim_verdict = claim_verdict

    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        raw_path = Path(next(x.split('=', 1)[1] for x in kwargs['agent_env'] if x.startswith('ADB_RESULT=')))
        work = raw_path.parent
        request = audit.read_json(work.parent / 'input.json')
        key, kind = request['key'], request['kind']
        with self.lock:
            self.calls.append(key)
            self.events.append((key, 'started', time.monotonic()))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            if key in self.block:
                entered, release = self.block[key]
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('fixture release timed out')
            Path(job).mkdir()
            audit.write_json(Path(job) / 'result.json', {'exception_info': None, 'agent_execution': {'duration': 1}})
            proof = work / 'proof.txt'
            proof.write_text('independent check against actual input')
            evidence = [{'path': str(proof), 'locator': 'calculation'}]
            def verdict(i, field='verdict'):
                return {'id': i, field: self.claim_verdict if kind == 'claims' else 'correct',
                        'reason': f'{kind} independent evidence', 'evidence': evidence}
            if kind in ('observations', 'claims'):
                payload = {kind: [verdict(i) for i in request['assigned'][kind]]}
            else:
                payload = {'task_checks': [verdict(i) for i in request['assigned']['task_checks']],
                           'mappings': [verdict(i, 'implementation_verdict') for i in request['assigned']['mappings']],
                           'probes': []}
                plan = []
                for pid, expected in [('correct_solution', True), ('incorrect_solution', False)]:
                    workbook = work / (pid + '.xlsx')
                    workbook.write_bytes(b'fixture workbook')
                    payload['probes'].append({'id': pid, 'expected_valid': expected, 'actual_accept': None,
                        'execution_status': 'missing', 'reason': 'independent expectation', 'evidence': evidence})
                    plan.append({'id': pid, 'expected_valid': expected, 'workbook': str(workbook),
                        'target_path': '/app/data.xlsx', 'reason': 'independent expectation', 'evidence': evidence})
                audit.write_json(work / 'probe_plan.json', {'probes': plan})
            if key == self.fail:
                raw_path.write_text('{malformed')
            else:
                audit.write_json(raw_path, payload)
            return 0
        finally:
            with self.lock:
                self.active -= 1
                self.events.append((key, 'finished', time.monotonic()))


class ParallelReviewTests(unittest.TestCase):
    setUp = fixtures.AuditTests.setUp
    parallel_fixture = fixtures.AuditTests.parallel_fixture

    def test_parent_failure_references_full_terminal_without_copying_transcript(self):
        run, manifest = self.inputs()
        full_error = 'complete-raw-transcript-' * 1000
        actual = {'status': 'failed', 'trial_count': 1, 'result_path': '/evidence/result.json',
                  'exception_info': {'exception_type': 'NetworkConnectionError',
                                     'exception_message': full_error}}
        with patch.object(audit, 'reviewer_terminal', return_value=actual):
            with self.assertRaises(ValueError) as error:
                parallel._run_unit(audit, ReviewHarbor(), self.cfg, manifest, run,
                    [manifest.parent], 'sample-00', 'observations', {}, {'observations': ['o1']}, None)
        self.assertIn('NetworkConnectionError', str(error.exception))
        self.assertIn('/evidence/result.json', str(error.exception))
        self.assertNotIn('complete-raw-transcript', str(error.exception))
        terminal = audit.read_json(run/'reviews/sample-00/terminal.json')
        self.assertEqual(terminal['model_terminal'], actual)
        self.assertNotIn('complete-raw-transcript', terminal['failure'])

    def test_source_scheduler_can_run_sixteen_ready_reviews_together(self):
        run = self.ep / 'sixteen-review-scheduler'
        identifiers = [f'o{i}' for i in range(16)]
        units = {f'sample-{i:02d}': ('observations', {}, {'observations': [oid]})
                 for i, oid in enumerate(identifiers)}
        manifest = {'expected_units': {'claims': [], 'observations': identifiers}}
        ready = threading.Barrier(16)
        def execute(*args):
            key = args[6]
            ready.wait(timeout=5)  # fails if an inner executor still admits only four
            return {'terminal': {'status': 'completed'}, 'raw_path': key + '.json',
                    'plan_path': None, 'terminal_path': key + '/terminal.json'}
        with patch.object(parallel, '_source_units', return_value=(units, {}, {})), \
             patch.object(parallel, '_run_unit', side_effect=execute):
            result = parallel._schedule(audit, object(), self.cfg, run / 'manifest.json',
                run, [], manifest, {}, {}, 16, None)
        self.assertEqual(set(result), set(units))

    def inputs(self, *, split=False, source_only=False):
        self.parallel_fixture(compact=True)
        inputs = audit.load_inputs(self.ep)
        run = self.ep / 'parallel-audit'
        manifest, _ = audit._freeze_evidence(inputs, {}, run / 'evidence')
        if split:
            data = audit.read_json(run / 'evidence/attribution.json')
            data['claims'][0].update(observation_ids=['s03-o1'], counterevidence_ids=[])
            data['claims'][1].update(observation_ids=['s00-o1'], counterevidence_ids=[])
            audit.write_json(run / 'evidence/attribution.json', data)
        if source_only:
            value = audit.read_json(manifest)
            value['snapshots'].pop('candidate')
            value['commits'].pop('design')
            value['expected_units'] = {k: value['expected_units'][k] for k in ('claims', 'observations')}
            audit.write_json(manifest, value)
        return run, manifest

    def test_parallel_merge_preserves_claim_and_implementation_disagreement(self):
        run, manifest = self.inputs()
        hb = ReviewHarbor(claim_verdict='incorrect')
        callbacks = []
        result = parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent],
            on_task_ready=lambda raw, plan, work: callbacks.append((raw, plan, work)))
        payload = audit.read_json(result['raw_path'])
        self.assertEqual(payload['mappings'][0]['upstream_verdict'], 'incorrect')
        self.assertEqual(payload['mappings'][0]['implementation_verdict'], 'correct')
        self.assertEqual(len(payload['observations']), 3)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(hb.calls), 5)
        self.assertLessEqual(hb.maximum, 4)
        self.assertTrue(result['evidence_files'])
        self.assertIn('upstream_verdict', result['field_sources']['mappings']['m1'])
        again = parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent])
        self.assertEqual(again['raw_path'], result['raw_path'])
        self.assertEqual(len(hb.calls), 5)

    def test_ready_claim_is_not_queued_behind_unrelated_observation(self):
        run, manifest = self.inputs(split=True, source_only=True)
        hb = ReviewHarbor()
        parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent], max_concurrent=1)
        self.assertEqual(hb.calls, ['sample-03', 'claim-c1', 'sample-00', 'claim-c2'])

    def test_task_ready_callback_precedes_slow_source_completion(self):
        run, manifest = self.inputs()
        entered, release, callback = threading.Event(), threading.Event(), threading.Event()
        hb = ReviewHarbor(block={'sample-00': (entered, release)})
        with ThreadPoolExecutor(1) as ex:
            future = ex.submit(parallel.run_parallel_review, hb, self.cfg, manifest, run, [manifest.parent],
                               on_task_ready=lambda *args: callback.set())
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(callback.wait(3))
                self.assertFalse(future.done())
            finally:
                release.set()
            future.result()

    def test_failed_atomic_review_is_not_retried_or_converted_to_uncertain(self):
        run, manifest = self.inputs()
        hb = ReviewHarbor(fail='sample-03')
        for _ in range(2):
            with self.assertRaises(parallel.ParallelReviewError) as error:
                parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent])
            self.assertIn('sample-03', error.exception.terminals)
        self.assertEqual(hb.calls.count('sample-03'), 1)
        self.assertFalse(any(c.startswith('claim-') for c in hb.calls))
        self.assertFalse((run / 'reviews/assembled-review.json').exists())

    def test_completed_workbook_or_worker_job_tamper_invalidates_resume(self):
        run, manifest = self.inputs()
        hb = ReviewHarbor()
        parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent])
        (run / 'reviews/task/work/correct_solution.xlsx').write_bytes(b'changed')
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent])
        self.assertEqual(len(hb.calls), 5)

    def test_source_only_result_reused_by_candidate_without_source_model_calls(self):
        run, manifest = self.inputs()
        source_manifest = run / 'evidence/source-manifest.json'
        source = audit.read_json(manifest)
        source['snapshots'].pop('candidate')
        source['commits'].pop('design')
        source['expected_units'] = {k: source['expected_units'][k] for k in ('claims', 'observations')}
        audit.write_json(source_manifest, source)
        hb = ReviewHarbor()
        source_result = parallel.run_source_review(hb, self.cfg, source_manifest,
                                                   run / 'source-only', [manifest.parent])
        self.assertEqual(len(hb.calls), 4)
        result = parallel.run_parallel_review(hb, self.cfg, manifest, run, [manifest.parent],
                                              source_review=source_result)
        self.assertEqual(len(hb.calls), 5)
        self.assertEqual(len(audit.read_json(result['raw_path'])['claims']), 2)
        # A temporary dictionary cannot override persisted source judgments.
        fake = copy.deepcopy(source_result)
        fake['terminals'] = {}
        verified = parallel.verify_source_review(fake, audit.read_json(manifest))
        self.assertEqual(len(verified['terminals']), 4)

    def test_future_does_not_delay_candidate_callback(self):
        run, manifest = self.inputs()
        source = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest,
                                             run / 'source-only', [manifest.parent])
        source_future = Future()
        callback = threading.Event()
        hb = ReviewHarbor()
        with ThreadPoolExecutor(1) as ex:
            future = ex.submit(parallel.run_parallel_review, hb, self.cfg, manifest, run,
                [manifest.parent], source_review=source_future, on_task_ready=lambda *args: callback.set())
            try:
                self.assertTrue(callback.wait(3))
                self.assertFalse(future.done())
            finally:
                source_future.set_result(source)
            future.result()
        self.assertEqual(hb.calls, ['task'])

    def test_source_identity_ignores_commit_id_but_rejects_changed_payload(self):
        run, manifest = self.inputs(source_only=True)
        result = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest, run, [manifest.parent])
        candidate_manifest = copy.deepcopy(audit.read_json(manifest))
        candidate_manifest['commits']['attribution']['commit_id'] = 'relocated-commit'
        parallel.verify_source_review(result, candidate_manifest)
        value = audit.read_json(candidate_manifest['commits']['attribution']['payload_path'])
        value['claims'][0]['statement'] = 'different claim'
        altered = run / 'altered-attribution.json'
        audit.write_json(altered, value)
        candidate_manifest['commits']['attribution']['payload_path'] = str(altered)
        with self.assertRaisesRegex(ValueError, 'different original evidence'):
            parallel.verify_source_review(result, candidate_manifest)

    def test_source_output_and_terminal_hashes_are_checked(self):
        run, manifest = self.inputs(source_only=True)
        result = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest, run, [manifest.parent])
        worker = Path(result['review_files']['sample-03']['raw_path'])
        worker.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            parallel.verify_source_review(result, audit.read_json(manifest))

    def test_complete_source_cache_survives_implementation_change_and_preserves_order(self):
        run, manifest = self.inputs(source_only=True)
        hb = ReviewHarbor()
        result = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent])
        expected = audit.read_json(manifest)['expected_units']
        raw = audit.read_json(result['raw_path'])
        self.assertEqual([x['id'] for x in raw['observations']], expected['observations'])
        self.assertEqual([x['id'] for x in raw['claims']], expected['claims'])
        with patch.object(parallel, '__file__', str(self.root / 'new-code-not-needed.py')):
            cached = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent])
        self.assertEqual(cached['receipt_sha256'], result['receipt_sha256'])
        self.assertEqual(len(hb.calls), 4)

    def test_source_cache_does_not_mix_judge_configurations(self):
        run, manifest = self.inputs(source_only=True)
        hb = ReviewHarbor()
        parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent])
        different = copy.deepcopy(self.cfg)
        different['judge']['model'] = 'another-judge'
        with self.assertRaisesRegex(ValueError, 'different judge'):
            parallel.run_source_review(hb, different, manifest, run, [manifest.parent])
        self.assertEqual(len(hb.calls), 4)

    def test_shared_slot_is_acquired_before_model_deadline_starts(self):
        run, manifest = self.inputs(source_only=True)
        occupancy = {'active': 0, 'maximum': 0}
        lock, gate = threading.Lock(), threading.Semaphore(1)
        @contextmanager
        def slot():
            with gate:
                with lock:
                    occupancy['active'] += 1
                    occupancy['maximum'] = max(occupancy['maximum'], occupancy['active'])
                try:
                    yield
                finally:
                    with lock:
                        occupancy['active'] -= 1
        hb = ReviewHarbor()
        result = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent], slot=slot)
        self.assertEqual(occupancy['maximum'], 1)
        for terminal in result['terminals'].values():
            self.assertLessEqual(terminal['queued_at'], terminal['acquired_at'])
            self.assertLessEqual(terminal['acquired_at'], terminal['finished_at'])

    def test_long_full_trace_is_directly_addressable_without_slicing(self):
        run, manifest = self.inputs(source_only=True)
        full = ('event:1 tool result and original reasoning\n' * 10000) + 'FINAL-EVIDENCE'
        trace = manifest.parent / 'complete-attempt.md'
        trace.write_text(full)
        index = manifest.parent / 'complete-evidence-index.json'
        audit.write_json(index, {'events': [{'event': 1, 'line': 1}]})
        value = audit.read_json(manifest)
        value['commits']['observation-03']['references'].extend([
            {'original_path': '/original/attempt.md', 'snapshot_path': str(trace), 'sha256': audit.sha(trace)},
            {'original_path': '/original/evidence_index.json', 'snapshot_path': str(index), 'sha256': audit.sha(index)}])
        audit.write_json(manifest, value)
        result = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest, run, [manifest.parent])
        request = audit.read_json(run / 'reviews/sample-03/input.json')
        path = next(x['path'] for x in request['source']['source_files'] if x['name'] == 'attempt.md')
        self.assertEqual(Path(path).read_text(), full)
        self.assertEqual(request['source']['observations'][0]['id'], 's03-o1')
        self.assertIn(str(trace), parallel.verify_source_review(result, value)['evidence_files'])
        # B sees the completed A terminal/input directory, not only its work.
        claim_mounts = [call.kwargs['ro'] for call in self.mount_mock.call_args_list
                        if any(str(p).endswith('/claim-c1/input.json') for p in call.kwargs['ro'])]
        self.assertEqual(len(claim_mounts), 1)
        self.assertIn((run / 'reviews/sample-03').resolve(), claim_mounts[0])

    def test_reviewers_receive_host_score_trace_read_only(self):
        run, manifest = self.inputs(source_only=True)
        traces = Path(manifest).resolve().parents[3] / 'score-trace'
        traces.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: [p.unlink() for p in traces.glob('*.json')])
        sample = traces / 'sample-03.json'
        summary = traces / 'score_trace_summary.json'
        audit.write_json(sample, {'sample_id': 3})
        audit.write_json(summary, {'rows': []})
        parallel.run_source_review(ReviewHarbor(), self.cfg, manifest, run, [manifest.parent])
        observation = audit.read_json(run / 'reviews/sample-03/input.json')['source']
        claim = audit.read_json(run / 'reviews/claim-c1/input.json')['source']
        self.assertEqual(observation['score_trace_path'], str(sample))
        self.assertEqual(claim['score_trace_path'], str(summary))
        mounted = [p for call in self.mount_mock.call_args_list for p in call.kwargs['ro']]
        self.assertIn(sample, mounted)
        self.assertIn(summary, mounted)

    def test_reviewers_without_score_trace_get_null(self):
        run, manifest = self.inputs(source_only=True)
        parallel.run_source_review(ReviewHarbor(), self.cfg, manifest, run, [manifest.parent])
        self.assertIsNone(audit.read_json(run / 'reviews/sample-03/input.json')['source']['score_trace_path'])

    def test_shared_slot_bounds_models_across_two_candidate_reviews(self):
        run, manifest = self.inputs()
        source = parallel.run_source_review(ReviewHarbor(), self.cfg, manifest,
                                             run / 'source-only', [manifest.parent])
        gate = threading.Semaphore(1)
        @contextmanager
        def slot():
            with gate:
                yield
        entered, release = threading.Event(), threading.Event()
        hb = ReviewHarbor(block={'task': (entered, release)})
        with ThreadPoolExecutor(2) as ex:
            first = ex.submit(parallel.run_parallel_review, hb, self.cfg, manifest,
                run / 'candidate-a', [manifest.parent], source_review=source, slot=slot)
            self.assertTrue(entered.wait(3))
            second = ex.submit(parallel.run_parallel_review, hb, self.cfg, manifest,
                run / 'candidate-b', [manifest.parent], source_review=source, slot=slot)
            release.set()
            first.result()
            second.result()
        self.assertEqual(hb.maximum, 1)
        self.assertEqual(hb.calls, ['task', 'task'])

    def test_subset_reviews_reuse_whole_sample_workers_and_keep_separate_receipts(self):
        run, manifest = self.inputs(source_only=True)
        hb = ReviewHarbor()
        first = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent],
                                           required_units={'claims': ['c1']})
        # c1 depends on both samples; sample-00 has an additional observation.
        self.assertEqual(first['reviewed_units']['observations'], ['s03-o1', 's00-o1', 's00-o2'])
        self.assertEqual(len(hb.calls), 3)
        revision = self.root / 'new-implementation.py'
        revision.write_text('new implementation identity')
        with patch.object(parallel, '__file__', str(revision)):
            second = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent],
                                                required_units={'claims': ['c2']})
        self.assertEqual(len(hb.calls), 4)
        self.assertEqual(hb.calls.count('sample-00'), 1)
        self.assertEqual(hb.calls.count('sample-03'), 1)
        self.assertNotEqual(first['receipt_path'], second['receipt_path'])
        self.assertEqual(second['reviewed_units']['claims'], ['c2'])
        self.assertEqual(audit.read_json(first['raw_path'])['claims'][0]['id'], 'c1')

    def test_unrelated_failed_sample_does_not_block_required_subset(self):
        run, manifest = self.inputs(split=True, source_only=True)
        hb = ReviewHarbor(fail='sample-00')
        with self.assertRaises(parallel.ParallelReviewError):
            parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent],
                                       required_units={'claims': ['c2']})
        result = parallel.run_source_review(hb, self.cfg, manifest, run, [manifest.parent],
                                            required_units={'claims': ['c1']})
        self.assertEqual(result['reviewed_units']['claims'], ['c1'])
        self.assertEqual(hb.calls.count('sample-00'), 1)
        self.assertNotIn('claim-c2', hb.calls)

    def test_unrelated_running_sample_does_not_block_required_subset(self):
        run, manifest = self.inputs(split=True, source_only=True)
        entered, release = threading.Event(), threading.Event()
        hb = ReviewHarbor(block={'sample-00': (entered, release)})
        with ThreadPoolExecutor(2) as ex:
            unrelated = ex.submit(parallel.run_source_review, hb, self.cfg, manifest, run,
                [manifest.parent], required_units={'claims': ['c2']})
            try:
                self.assertTrue(entered.wait(3))
                needed = ex.submit(parallel.run_source_review, hb, self.cfg, manifest, run,
                    [manifest.parent], required_units={'claims': ['c1']})
                self.assertEqual(needed.result(timeout=3)['reviewed_units']['claims'], ['c1'])
                self.assertFalse(unrelated.done())
                started = audit.read_json(run / 'reviews/sample-00/started.json')
                self.assertEqual(started['status'], 'running')
                self.assertIsNotNone(started['acquired_at'])
            finally:
                release.set()
            unrelated.result()

    def test_overlapping_subsets_lock_only_shared_units_and_do_not_duplicate_calls(self):
        run, manifest = self.inputs(source_only=True)
        entered, release = threading.Event(), threading.Event()
        hb = ReviewHarbor(block={'sample-00': (entered, release)})
        with ThreadPoolExecutor(2) as ex:
            one = ex.submit(parallel.run_source_review, hb, self.cfg, manifest, run,
                [manifest.parent], required_units={'claims': ['c1']})
            try:
                self.assertTrue(entered.wait(3))
                two = ex.submit(parallel.run_source_review, hb, self.cfg, manifest, run,
                    [manifest.parent], required_units={'claims': ['c2']})
            finally:
                release.set()
            one.result()
            two.result()
        self.assertEqual(sorted(hb.calls), ['claim-c1', 'claim-c2', 'sample-00', 'sample-03'])

    def test_candidate_uses_result_for_instead_of_waiting_for_full_future(self):
        run, manifest = self.inputs(split=True)
        candidate_manifest = manifest.parent / 'candidate-scope.json'
        value = audit.read_json(manifest)
        value['expected_units']['claims'] = ['c1']
        value['expected_units']['observations'] = ['s03-o1']
        audit.write_json(candidate_manifest, value)
        hb = ReviewHarbor()
        outer = self
        class Handle:
            def result(self):
                raise AssertionError('must not wait for full-source Future')
            def result_for(self, candidate):
                return parallel.run_source_review(hb, outer.cfg, manifest, run / 'source',
                    [manifest.parent], required_units={g: candidate['expected_units'][g]
                                                      for g in ('claims', 'observations')})
        result = parallel.run_parallel_review(hb, self.cfg, candidate_manifest, run / 'candidate',
                                              [manifest.parent], source_review=Handle())
        self.assertEqual(set(hb.calls), {'task', 'sample-03', 'claim-c1'})
        self.assertEqual([x['id'] for x in audit.read_json(result['raw_path'])['claims']], ['c1'])


if __name__ == '__main__':
    unittest.main()
