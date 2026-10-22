"""Real offline process trees: a leader exit must not hide TERM-ignoring children."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
from common import Harbor, terminate_process_group
from host_validation import execution_slot


class ProcessCleanupTests(unittest.TestCase):
    def test_permission_denied_cleanup_is_explicitly_unconfirmed(self):
        proc = Mock(pid=2147483000, returncode=None)
        proc.wait.side_effect = subprocess.TimeoutExpired('fixture', .01)
        with patch('common.os.killpg', side_effect=PermissionError(1, 'Operation not permitted')):
            result = terminate_process_group(proc, 0, 0)
        self.assertIsNone(result['host_group_absent'])
        self.assertEqual(result['signals_sent'], [])
        self.assertTrue(result['errors'])
        self.assertIsNone(result['leader_returncode'])

    def test_harbor_preserves_original_timeout_when_cleanup_permission_is_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = Mock(pid=2147483000, returncode=None)
            original = subprocess.TimeoutExpired('original-call', 9)
            proc.wait.side_effect = original
            hb = Harbor('unused-fixture', 'http://127.0.0.1:1', 'env:TEST', 1, 'test')
            with patch('common.subprocess.Popen', return_value=proc), \
                 patch('common.os.killpg', side_effect=PermissionError(1, 'Operation not permitted')), \
                 patch('common.terminate_process_group', side_effect=lambda p: terminate_process_group(p, 0, 0)):
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    hb.run([], root/'job', 9)
            self.assertIs(raised.exception, original)
            receipt = json.loads((root/'job/host-process-cleanup.json').read_text())
            self.assertIsNone(receipt['host_group_absent'])
            self.assertTrue(receipt['errors'])

    def test_harbor_timeout_cleans_surviving_descendants_before_slot_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_path = root/'descendant.pid'
            script = root/'fake-harbor'
            child = 'trap "" TERM; echo $$ > '+shlex.quote(str(pid_path))+'; while :; do sleep 1; done'
            script.write_text('#!/bin/sh\ntrap "exit 0" TERM\n/bin/sh -c '+shlex.quote(child)+
                              ' &\necho original-stdout\necho original-stderr >&2\nwait\n')
            script.chmod(0o700)
            hb = Harbor(str(script), 'http://127.0.0.1:1', 'env:TEST', 1, 'test')
            request = {'slot_root': str(root/'slots'), 'config': {'score': {'n_concurrent': 1}}}
            try:
                with execution_slot(request):
                    with patch('common.terminate_process_group',
                               side_effect=lambda proc: terminate_process_group(proc, .2, 2)):
                        with self.assertRaises(subprocess.TimeoutExpired):
                            hb.run([], root/'job', 5)
                    self.assertTrue(pid_path.is_file())
                    child_pid = int(pid_path.read_text())
                    status = subprocess.run(['ps', '-o', 'stat=', '-p', str(child_pid)],
                                            capture_output=True, text=True).stdout.strip()
                    self.assertTrue(not status or status.startswith('Z'), status)
                cleanup = json.loads((root/'job/host-process-cleanup.json').read_text())
                self.assertEqual(cleanup['signals_sent'], ['SIGTERM', 'SIGKILL'])
                self.assertEqual(cleanup['external_container_cleanup'], 'unverified')
                log = (root/'job/harbor.log').read_text()
                self.assertIn('original-stdout', log)
                self.assertIn('original-stderr', log)
                with execution_slot(request):
                    pass
            finally:
                if pid_path.exists():
                    try:
                        os.kill(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_success_preserves_returncode_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root/'fake-harbor'
            script.write_text('#!/bin/sh\necho complete\nexit 7\n')
            script.chmod(0o700)
            hb = Harbor(str(script), 'http://127.0.0.1:1', 'env:TEST', 1, 'test')
            self.assertEqual(hb.run([], root/'job', 10), 7)
            self.assertEqual((root/'job/harbor.log').read_text(), 'complete\n')
            self.assertFalse((root/'job/host-process-cleanup.json').exists())


if __name__ == '__main__':
    unittest.main()
