import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
from common import Harbor
from stage_terminal import read_terminal


class StageTerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.job = Path(self.tmp.name) / 'job'
        self.job.mkdir()

    def receipt(self, name='trial', **overrides):
        path = self.job / '2026-09-22__15-05-06' / name / 'result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {'task_name': 'task', 'trial_name': name,
                 'finished_at': '2026-09-22T15:12:54.579661Z',
                 'exception_info': None}
        value.update(overrides)
        path.write_text(json.dumps(value))
        return path

    def test_unique_success_ignores_job_summary_and_hashes_exact_receipt(self):
        path = self.receipt()
        (path.parent.parent / 'result.json').write_text(json.dumps({
            'n_total_trials': 1, 'finished_at': '2026-09-22T15:13:12.315595', 'stats': {}}))
        result = read_terminal(self.job)
        self.assertEqual(result['agent_terminal_status'], 'completed')
        self.assertIsNone(result['exception_type'])
        self.assertIsNone(result['exception_message'])
        self.assertEqual(result['finished_at'], '2026-09-22T15:12:54.579661Z')
        self.assertEqual(result['trial_result_path'], str(path.resolve()))
        self.assertEqual(result['trial_result_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_timeout_preserves_actual_exception(self):
        self.receipt(exception_info={'exception_type': 'AgentTimeoutError',
                                     'exception_message': 'Agent execution timed out after 600 seconds'})
        result = read_terminal(self.job)
        self.assertEqual(result['agent_terminal_status'], 'timeout')
        self.assertEqual(result['exception_type'], 'AgentTimeoutError')
        self.assertEqual(result['exception_message'], 'Agent execution timed out after 600 seconds')

    def test_other_exception_is_error_even_if_output_exists(self):
        self.receipt(exception_info={'exception_type': 'HTTPError', 'exception_message': 'HTTP 502'})
        (self.job / 'output.json').write_text('{"claims": []}')
        self.assertEqual(read_terminal(self.job)['agent_terminal_status'], 'error')

    def test_no_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'found 0'):
            read_terminal(self.job)

    def test_multiple_receipts_rejected(self):
        self.receipt('one')
        self.receipt('two')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            read_terminal(self.job)

    def test_unfinished_rejected(self):
        for finished in (None, ''):
            with self.subTest(finished=finished):
                self.receipt(finished_at=finished)
                with self.assertRaisesRegex(ValueError, 'has not terminated'):
                    read_terminal(self.job)

    def test_missing_exception_field_is_not_success(self):
        path = self.receipt()
        value = json.loads(path.read_text())
        del value['exception_info']
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'missing exception_info'):
            read_terminal(self.job)

    def test_invalid_terminal_timestamp_rejected(self):
        self.receipt(finished_at='still running')
        with self.assertRaisesRegex(ValueError, 'invalid Harbor trial finished_at'):
            read_terminal(self.job)

    def test_empty_exception_object_is_not_success(self):
        self.receipt(exception_info={})
        with self.assertRaisesRegex(ValueError, 'invalid Harbor terminal exception_info'):
            read_terminal(self.job)

    def test_harbor_disables_only_telemetry_preserving_model_environment(self):
        harbor = Harbor('harbor', 'https://model.invalid', 'test-key', 1.0, 'test-model')
        with patch.dict(os.environ, {'HARBOR_TELEMETRY': '1', 'UNRELATED_OPTION': 'keep'}):
            before = dict(os.environ)
            env = harbor._env()
            self.assertEqual(env['HARBOR_TELEMETRY'], '0')
            self.assertEqual(env['ANTHROPIC_BASE_URL'], harbor.base_url)
            self.assertEqual(env['ANTHROPIC_API_KEY'], harbor.api_key)
            self.assertEqual(env['UNRELATED_OPTION'], 'keep')
            self.assertEqual(dict(os.environ), before)


if __name__ == '__main__':
    unittest.main()
