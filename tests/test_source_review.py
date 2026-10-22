"""Source facts are reusable without a candidate; model work has a shared cap."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import source_review
import run_training
import test_candidate_jobs as fixtures
from host_validation import read, write


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateJobsTests('test_one_job_only_authors_and_keeps_same_analysis_payload')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = deepcopy(self.fixture.cfg)
        self.cfg['judge'] = {'agent': 'claude-code', 'model': 'judge', 'budget_min': 1}
        self.cfg['researcher']['agent_setup_timeout_mult'] = 1
        self.cfg['score']['setup_timeout_mult'] = 1
        self.fixture.cfg = self.cfg
        self.jobs = self.fixture.joblist()['jobs']

    def test_source_freeze_has_no_candidate_dependency_and_reuses_across_goals(self):
        first, path = source_review._freeze(self.jobs[0], self.cfg)
        second, second_path = source_review._freeze(self.jobs[-1], self.cfg)
        self.assertEqual(first, second)
        self.assertEqual(path, second_path)
        manifest = read(path)
        self.assertEqual(set(manifest['snapshots']), {'original', 'rollouts'})
        self.assertNotIn('design', manifest['commits'])
        self.assertEqual(manifest['expected_units']['claims'], ['c1'])
        self.assertFalse((self.fixture.analysis / 'diagnostics/synthesis.json').exists())
        trace = Path(manifest['snapshots']['rollouts']['snapshot_path']) / 'attempt-0.md'
        self.assertEqual(trace.read_bytes(), (self.fixture.bundle / 'attempt-0.md').read_bytes())

    def test_modified_frozen_source_is_rejected_before_a_model_can_run(self):
        root, path = source_review._freeze(self.jobs[0], self.cfg)
        (path.parent / 'attribution.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen source review evidence changed'):
            source_review._freeze(self.jobs[1], self.cfg)

    def test_model_slots_limit_independent_candidate_workers_together(self):
        root = self.fixture.root / 'slots'
        active, peak = 0, 0
        mutex = threading.Lock()
        four_started, release = threading.Event(), threading.Event()
        def work():
            nonlocal active, peak
            with source_review.model_slot(root, max_concurrent=4):
                with mutex:
                    active += 1
                    peak = max(peak, active)
                    if active == 4:
                        four_started.set()
                if not release.wait(5):
                    raise AssertionError('worker release did not arrive')
                with mutex:
                    active -= 1
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(work) for _ in range(8)]
            try:
                self.assertTrue(four_started.wait(5))
                self.assertEqual(peak, 4)
            finally:
                release.set()
            for future in futures:
                future.result()
        self.assertEqual(peak, 4)

    def test_default_model_slots_allow_sixteen_and_block_seventeenth(self):
        root = self.fixture.root / 'sixteen-slots'
        active, peak = 0, 0
        mutex = threading.Lock()
        ready, overflow, release = [threading.Event() for _ in range(3)]
        def work():
            nonlocal active, peak
            with source_review.model_slot(root):
                with mutex:
                    active += 1
                    peak = max(peak, active)
                    if active == 16:
                        ready.set()
                    if active > 16:
                        overflow.set()
                try:
                    self.assertTrue(release.wait(5))
                finally:
                    with mutex:
                        active -= 1
        with ThreadPoolExecutor(max_workers=17) as pool:
            futures = [pool.submit(work) for _ in range(17)]
            try:
                self.assertTrue(ready.wait(5), 'a hidden lower cap blocked 16 workers')
                self.assertFalse(overflow.wait(.1), 'seventeenth worker bypassed shared cap')
            finally:
                release.set()
            for future in futures:
                future.result()
        self.assertEqual(peak, 16)

    def test_author_and_source_overlap_and_validation_receives_future(self):
        ep = self.fixture.root / 'runs/train/bench/source/ep01'
        source_started, author_started, author_finished = [threading.Event() for _ in range(3)]
        def source(*args, **kwargs):
            self.assertEqual(kwargs['max_concurrent'], 16)
            self.assertEqual(kwargs['slot'].keywords['max_concurrent'], 16)
            source_started.set()
            self.assertTrue(author_started.wait(5))
            self.assertTrue(author_finished.wait(5))
            return {'source': 'checked', 'field_sources': {'claims': {}, 'observations': {}}}
        def author(*args, **kwargs):
            author_started.set()
            self.assertTrue(source_started.wait(5), 'source review must start before author finishes')
            author_finished.set()
        def validate(*args, **kwargs):
            self.assertEqual(kwargs['audit_concurrency'], 16)
            self.assertEqual(kwargs['audit_slot'].keywords['max_concurrent'], 16)
            checked = kwargs['source_review'].result_for({'expected_units': {}})
            self.assertEqual(checked['source'], 'checked')
            # All slots must be released by author before joining downstream.
            return {'status': 'completed'}
        with patch.object(run_training, 'ROOT', self.fixture.root), \
             patch.object(run_training, 'episode_dir', return_value=ep), \
             patch.object(run_training, 'benchmark_dir', return_value=self.fixture.source.parent), \
             patch.object(run_training, 'make_harbor', return_value=object()), \
             patch.object(run_training, 'run_candidate', side_effect=author), \
             patch.object(run_training, 'validate_authored_candidate', side_effect=validate), \
             patch.object(run_training, 'export_candidate', return_value={'status': 'blocked', 'sft': {}}), \
             patch.object(source_review, 'review_source', side_effect=source), \
             patch.object(run_training, 'log'):
            result = run_training.execute_job(self.cfg, 'bench', 'train', self.jobs[0], 1,
                export_root=self.fixture.root / 'exports')
        self.assertEqual(result['status'], 'quality_blocked')
        self.assertEqual(result['requested_audit_concurrency'], 16)
        self.assertEqual(result['shared_model_slots'], 16)
        self.assertLessEqual(result['queued_at'], result['author_slot_acquired_at'])
        self.assertLessEqual(result['author_slot_acquired_at'], result['author_slot_released_at'])

    def test_candidate_returns_without_waiting_on_unrelated_source_work(self):
        ep = self.fixture.root / 'runs/train/bench/source/ep01'
        started, release = threading.Event(), threading.Event()
        def source(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError('candidate incorrectly waited for unrelated source work')
        def author(*args, **kwargs):
            self.assertTrue(started.wait(5))
        with ThreadPoolExecutor(max_workers=1) as source_pool, \
             patch.object(run_training, 'ROOT', self.fixture.root), \
             patch.object(run_training, 'episode_dir', return_value=ep), \
             patch.object(run_training, 'benchmark_dir', return_value=self.fixture.source.parent), \
             patch.object(run_training, 'make_harbor', return_value=object()), \
             patch.object(run_training, 'run_candidate', side_effect=author), \
             patch.object(run_training, 'validate_authored_candidate', return_value={'status': 'completed'}), \
             patch.object(run_training, 'export_candidate', return_value={'status': 'exported', 'sft': {}}), \
             patch.object(source_review, 'review_source', side_effect=source), \
             patch.object(run_training, 'log'):
            try:
                result = run_training.execute_job(self.cfg, 'bench', 'train', self.jobs[0], 1,
                    export_root=self.fixture.root / 'exports', source_executor=source_pool)
                self.assertEqual(result['status'], 'completed')
                self.assertFalse(release.is_set())
            finally:
                release.set()


if __name__ == '__main__':
    unittest.main()
