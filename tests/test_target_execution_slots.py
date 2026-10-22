"""Different runs share target execution slots; cached legacy trials stay exact."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import test_host_validation as fixtures
import host_validation as hv


class HoldingHarbor(fixtures.FakeValidationHarbor):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.four_started = threading.Event()
        self.over_capacity = threading.Event()

    def customer(self, task, model, job, timeout, **kwargs):
        with self.lock:
            self.calls.append(('trainee', str(job)))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if self.active == 4:
                self.four_started.set()
            if self.active > 4:
                self.over_capacity.set()
        try:
            if not self.release.wait(10):
                raise RuntimeError('test did not release target executions')
            self.receipt(job, 0)
            return 0
        finally:
            with self.lock:
                self.active -= 1


class TargetExecutionSlotTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.HostValidationTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.cfg['score']['n_concurrent'] = 4
        self.f.cfg['training_pipeline'] = {'rollout': {'attempts': 4, 'max_concurrent': 4}}

    def second_request(self, hb):
        ep = self.f.root / 'runs/another-run/spreadsheet/task/ep01'
        for relative in ('diagnostics/attribution.json', 'diagnostics/design.json', 'work/design.json'):
            hv.write(ep / relative, hv.read(self.f.ep / relative))
        return hv.prepare_request(hb, self.f.cfg, 'spreadsheet', ep)

    def test_two_runs_admit_only_four_targets_and_waiters_have_no_start_receipt(self):
        hb = HoldingHarbor()
        paths = [self.f.request(hb), self.second_request(hb)]
        requests = [hv.read(p) for p in paths]
        request_bytes = [p.read_bytes() for p in paths]
        self.assertNotEqual(requests[0]['slot_root'], requests[1]['slot_root'])
        for path in paths:
            self.assertEqual(hv.run_worker(path, hb, phase='reference')['status'], 'reference_completed')
        original_slot = hv.target_execution_slot
        all_waiting = threading.Event()
        entered, mutex = [], threading.Lock()

        @contextmanager
        def record_waiter(request):
            with mutex:
                entered.append(request['request_id'])
                if len(entered) == 8:
                    all_waiting.set()
            with original_slot(request):
                yield

        with patch.object(hv, 'target_execution_slot', side_effect=record_waiter), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(hv.run_worker, p, hb, phase='trainee') for p in paths]
            try:
                self.assertTrue(hb.four_started.wait(5))
                self.assertTrue(all_waiting.wait(5))
                self.assertFalse(hb.over_capacity.is_set())
                self.assertEqual(hb.maximum, 4)
                started = [hv.original_case_directory(request, 'trainee', i) / 'host-started.json'
                           for request in requests for i in range(4)]
                self.assertEqual(sum(p.exists() for p in started), 4)
            finally:
                hb.release.set()
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual([r['status'] for r in results], ['completed', 'completed'])
        self.assertEqual([r['completed_attempts'] for r in results], [4, 4])
        self.assertEqual(hb.maximum, 4)
        self.assertEqual(sum(kind == 'trainee' for kind, _ in hb.calls), 8)
        self.assertEqual([p.read_bytes() for p in paths], request_bytes)

    def test_reference_keeps_original_pool_when_target_pool_is_full(self):
        hb = fixtures.FakeValidationHarbor()
        path = self.f.request(hb)
        request = hv.read(path)
        # These are actual file locks, just as another run would hold them.
        held = ExitStack()
        for _ in range(4):
            held.enter_context(hv.target_execution_slot(request))
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(hv.run_worker, path, hb, phase='reference')
            try:
                result = future.result(timeout=3)
            finally:
                held.close()
        self.assertEqual(result['status'], 'reference_completed')
        self.assertEqual([kind for kind, _ in hb.calls], ['oracle'])

    def test_completed_legacy_request_and_trials_reuse_without_slots_or_calls(self):
        hb = fixtures.FakeValidationHarbor()
        path = self.f.request(hb)
        before = path.read_bytes()
        request = hv.read(path)
        self.assertEqual(request['slot_root'], str(self.f.root / 'runs/run/.validation-slots'))
        result = hv.run_worker(path, hb)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(hb.calls), 5)
        oracle = hv.original_case_directory(request, 'oracle')
        frozen = {p: p.read_bytes() for p in oracle.rglob('*') if p.is_file()}
        with patch.object(hv, 'target_execution_slot', side_effect=AssertionError('cached trial acquired target slot')):
            self.assertEqual(hv.run_worker(path, hb, phase='trainee'), result)
            cached = hv.execute_case(hb, self.f.cfg, request, 'trainee', 0)
            self.assertEqual(cached, result['attempts'][0])
        self.assertEqual(len(hb.calls), 5)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in frozen.items()))

    def test_target_pool_keeps_explicit_lower_cap_and_does_not_mutate_request(self):
        hb = fixtures.FakeValidationHarbor()
        path = self.f.request(hb)
        request = hv.read(path)
        request['config']['score']['n_concurrent'] = 2
        before = deepcopy(request)
        original_slot = hv.execution_slot
        calls = []

        @contextmanager
        def capture(pool_request):
            calls.append(deepcopy(pool_request))
            with original_slot(pool_request):
                yield

        with patch.object(hv, 'execution_slot', side_effect=capture):
            with hv.target_execution_slot(request):
                pass
        self.assertEqual(calls[0]['config']['score']['n_concurrent'], 2)
        self.assertEqual(calls[0]['slot_root'], str(self.f.root / 'runs/.target-execution-slots'))
        self.assertEqual(request, before)

    def test_arbitrary_episode_depths_use_the_same_explicit_runtime_root(self):
        request = hv.read(self.f.request(fixtures.FakeValidationHarbor()))
        configured = self.f.root / 'separate-runtime-root/target-slots'
        calls = []
        real_slot = hv.execution_slot

        @contextmanager
        def capture(pool_request):
            calls.append(deepcopy(pool_request))
            with real_slot(pool_request):
                yield

        with patch.object(hv, 'TARGET_EXECUTION_ROOT', configured), \
                patch.object(hv, 'execution_slot', side_effect=capture):
            for episode in ('/ep01', '/arbitrary/deeper/custom/location/ep01'):
                custom = {**request, 'episode': episode}
                before = deepcopy(custom)
                with hv.target_execution_slot(custom):
                    self.assertTrue(configured.is_dir())
                self.assertEqual(custom, before)
        self.assertEqual([call['slot_root'] for call in calls], [str(configured)] * 2)


if __name__ == '__main__':
    unittest.main()
