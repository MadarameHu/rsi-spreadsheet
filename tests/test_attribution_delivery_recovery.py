"""A failed final writer can deliver its exact literal without another model call."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_local_integration as fixtures
import attribution_revision as revision
import local_analysis
import parallel_attribution as pa
import staged_synth as st
from stage_records import read_stage_commit


class DeliveryRecoveryTests(unittest.TestCase):
    setUp = fixtures.LocalIntegrationTests.setUp
    analyze = fixtures.LocalIntegrationTests.analyze
    tool_id = 'tool-complete-final-delivery'

    def fail_writer(self, *, unsafe=False, wrong_path=False):
        def timeout(*args):
            result = self.fake(*args)
            env = args[-1]
            if env.get('ADB_SAMPLE_ID') is not None:
                return result
            output = Path(env['ADB_RESULT'])
            payload = st.load(output)
            self.expected_output = json.dumps(payload, indent=2).encode()
            output.unlink()
            value = "__import__('os').system('false')" if unsafe else repr(payload)
            destination = str(output)+'-other' if wrong_path else str(output)
            script = ('import json\nR = '+repr(str(env['ADB_ROLLOUT_DIR']))+'\nout = '+value
                      +'\ndest = '+repr(destination)+'\nwith open(dest, "w") as f:\n'
                      '    json.dump(out, f, indent=2)\nprint("written", dest)')
            command = ('mkdir -p "'+str(output.parent)+'" && cat > /tmp/write_output.py << \'PYEOF\'\n'
                       +script+'\nPYEOF\npython3 /tmp/write_output.py')
            tool = {'type': 'tool_use', 'id': self.tool_id, 'name': 'Bash', 'input': {'command': command}}
            rows = [{'type': 'assistant', 'message': {'content': [tool]}},
                    {'type': 'user', 'message': {'content': [{'type': 'tool_result',
                     'tool_use_id': self.tool_id, 'is_error': True,
                     'content': 'Exit code 2\noperation not permitted: /tmp/write_output.py'}]}}]
            job = Path(args[0])/'job'
            transcript = job/'claude-code.txt'
            transcript.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            record = st.load(job/'local-result.json')
            record.update(status='timeout', returncode=143, active_seconds=args[3],
                          exception_type='TimeoutError', exception_message='local agent exceeded allocated wall time',
                          process_cleanup={'host_group_absent': True},
                          transcript=local_analysis._identity(transcript))
            (job/'local-result.json').write_text(json.dumps(record))
            return local_analysis.read_local_result(job)
        with patch.object(local_analysis, 'run_local', side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, 'output is missing.*agent_terminal_status=timeout'):
                self.analyze()
        self.diag = self.ep/'diagnostics'
        self.transcript_sha = local_analysis._identity(self.diag/'attribution/job/claude-code.txt')['sha256']
        self.before = {path: path.read_bytes() for path in self.diag.rglob('*') if path.is_file()}

    def recover(self):
        return revision.recover_delivery(self.diag, expected_tool_call_id=self.tool_id,
                                         expected_transcript_sha256=self.transcript_sha)

    def test_exact_literal_materialization_preserves_timeout_and_every_original_file(self):
        self.fail_writer()
        with patch.object(pa, '_worker', side_effect=AssertionError('no model')), \
             patch.object(pa, '_run', side_effect=AssertionError('no model')):
            receipt = self.recover()
        self.assertEqual(receipt['origin'], 'post_failure_delivery_recovery')
        self.assertEqual(receipt['agent_terminal_status'], 'timeout')
        self.assertEqual((self.diag/'attribution/work/output.json').read_bytes(), self.expected_output)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in self.before.items()))
        commit = read_stage_commit(self.diag/'records', receipt['attribution_commit'])
        self.assertEqual(len(commit['parent_ids']), 16)
        self.assertEqual(commit['metadata']['stage_execution']['terminal']['agent_terminal_status'], 'timeout')
        self.assertEqual(commit['metadata']['post_failure_delivery_recovery']['method'],
                         'restricted_ast_literal_materialization')
        self.assertEqual(self.recover(), receipt)
        self.assertEqual(len(self.fake.calls), 17)

    def test_nonliteral_expression_is_rejected_without_execution_or_output(self):
        self.fail_writer(unsafe=True)
        with self.assertRaisesRegex(ValueError, 'AST whitelist'):
            self.recover()
        self.assertFalse((self.diag/'attribution/work/output.json').exists())

    def test_wrong_destination_is_rejected(self):
        self.fail_writer(wrong_path=True)
        with self.assertRaisesRegex(ValueError, 'source or output path differs'):
            self.recover()
        self.assertFalse((self.diag/'attribution/work/output.json').exists())

    def test_explicit_tool_id_and_transcript_hash_are_required(self):
        self.fail_writer()
        with self.assertRaisesRegex(ValueError, 'exact final Bash'):
            revision.recover_delivery(self.diag, expected_tool_call_id='wrong',
                                      expected_transcript_sha256=self.transcript_sha)
        with self.assertRaisesRegex(ValueError, 'explicit expected hash'):
            revision.recover_delivery(self.diag, expected_tool_call_id=self.tool_id,
                                      expected_transcript_sha256='0'*64)

    def test_completed_worker_tampering_blocks_materialization(self):
        self.fail_writer()
        index = st.load(self.diag/'observations/index.json')
        Path(index['workers'][0]['raw_output_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'raw model output differs'):
            self.recover()
        self.assertFalse((self.diag/'attribution/work/output.json').exists())

    def test_recovered_literal_or_original_failure_cannot_be_changed(self):
        self.fail_writer()
        receipt = self.recover()
        output = self.diag/'attribution/work/output.json'
        original = output.read_bytes()
        output.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'complete original model literal'):
            revision.validate_delivery_recovery(self.diag, receipt['receipt_path'])
        output.write_bytes(original)
        (self.diag/'attribution/error.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'original failed invocation evidence changed'):
            revision.validate_delivery_recovery(self.diag, receipt['receipt_path'])
