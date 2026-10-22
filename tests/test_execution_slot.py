"""OS admission locking: no newcomer barging, unchanged cap, crash recovery."""
import fcntl
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import host_validation as validation


class ExecutionSlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {'slot_root': str(self.root), 'config': {'score': {'n_concurrent': 1}}}

    def test_admitted_waiter_cannot_be_overtaken_by_later_pollers(self):
        occupied = (self.root / '0.lock').open('a')
        fcntl.flock(occupied, fcntl.LOCK_EX)
        polling, permit_poll, newcomer_acquired = threading.Event(), threading.Event(), threading.Event()
        order, errors, threads = [], [], []
        lock = threading.Lock()
        real_sleep = time.sleep

        def controlled_sleep(seconds):
            if threading.current_thread().name == 'admitted-waiter':
                polling.set()
                if not permit_poll.wait(3):
                    raise RuntimeError('fixture did not release admitted waiter')
            else:
                real_sleep(0.001)

        def worker(name, started=None):
            if started is not None:
                started.set()
            try:
                with validation.execution_slot(self.request):
                    with lock:
                        order.append(name)
                    if name != 'first':
                        newcomer_acquired.set()
            except BaseException as error:
                errors.append(error)

        with patch.object(validation.time, 'sleep', side_effect=controlled_sleep):
            first = threading.Thread(target=worker, args=('first',), name='admitted-waiter')
            threads.append(first)
            first.start()
            try:
                self.assertTrue(polling.wait(3))
                started_events = []
                for number in range(6):
                    started = threading.Event()
                    thread = threading.Thread(target=worker, args=(f'new-{number}', started))
                    threads.append(thread)
                    started_events.append(started)
                    thread.start()
                self.assertTrue(all(started.wait(3) for started in started_events))
                fcntl.flock(occupied, fcntl.LOCK_UN)
                occupied.close()
                # The old implementation lets a later poller take this now-free
                # slot while the first waiter's polling loop is paused.
                self.assertFalse(newcomer_acquired.wait(0.15))
            finally:
                permit_poll.set()
                if not occupied.closed:
                    fcntl.flock(occupied, fcntl.LOCK_UN)
                    occupied.close()
                for thread in threads:
                    thread.join(3)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(order[0], 'first')
        self.assertEqual(len(order), 7)

    def test_admission_is_released_during_work_and_configured_cap_is_preserved(self):
        self.request['config']['score']['n_concurrent'] = 3
        release, full = threading.Event(), threading.Event()
        guard = threading.Lock()
        state = {'active': 0, 'maximum': 0, 'entered': 0}
        errors = []

        def worker():
            try:
                with validation.execution_slot(self.request):
                    with guard:
                        state['active'] += 1
                        state['entered'] += 1
                        state['maximum'] = max(state['maximum'], state['active'])
                        if state['active'] == 3:
                            full.set()
                    release.wait(3)
                    with guard:
                        state['active'] -= 1
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(9)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(full.wait(3), 'admission must not remain held during work')
            self.assertEqual(state['active'], 3)
            self.assertEqual(state['entered'], 3)
        finally:
            release.set()
            for thread in threads:
                thread.join(3)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(state['maximum'], 3)
        self.assertEqual(state['entered'], 9)

    def test_python_exception_releases_running_slot(self):
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with validation.execution_slot(self.request):
                raise RuntimeError('test failure')
        with validation.execution_slot(self.request):
            pass

    def child(self, code):
        setup = "import os,sys,time\nsys.path.insert(0,sys.argv[1])\nimport host_validation as v\nrequest={'slot_root':sys.argv[2],'config':{'score':{'n_concurrent':1}}}\n"
        return subprocess.Popen([sys.executable, '-c', setup + code,
                                 str(Path(validation.__file__).parent), str(self.root)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def assert_child_line(self, child, expected):
        self.assertTrue(select.select([child.stdout], [], [], 5)[0], 'child did not reach lock phase')
        self.assertEqual(child.stdout.readline().strip(), expected)

    def test_abrupt_process_exit_releases_running_slot(self):
        child = self.child("with v.execution_slot(request):\n print('acquired',flush=True)\n os._exit(17)\n")
        try:
            self.assert_child_line(child, 'acquired')
            self.assertEqual(child.wait(timeout=5), 17)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        with (self.root / '0.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        with validation.execution_slot(self.request):
            pass

    def test_killed_waiter_releases_admission_lock(self):
        with (self.root / '0.lock').open('a') as occupied:
            fcntl.flock(occupied, fcntl.LOCK_EX)
            child = self.child("real_sleep=time.sleep\ndef waiting(seconds):\n print('waiting',flush=True)\n real_sleep(60)\nv.time.sleep=waiting\nwith v.execution_slot(request):\n print('acquired',flush=True)\n")
            try:
                self.assert_child_line(child, 'waiting')
                with (self.root / '.admission.lock').open('a') as admission:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child.kill()
                child.wait(timeout=5)
                with (self.root / '.admission.lock').open('a') as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(admission, fcntl.LOCK_UN)
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)
                fcntl.flock(occupied, fcntl.LOCK_UN)
        with validation.execution_slot(self.request):
            pass


if __name__ == '__main__':
    unittest.main()
