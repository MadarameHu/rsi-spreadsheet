import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import leak_check as lc  # noqa: E402

lc.RETRY_SLEEP = 0

INSTRUCTION = 'Fill B4:B21 with the rate.\nDo not multiply   by 100.'
CLASSES = [{'class_id': 'c1', 'statement': 'no key filter'}, {'class_id': 'c5', 'statement': 'scaled by 100'}]


def reply(items):
    return lambda *a: {'content': [{'type': 'text', 'text': 'x ' + json.dumps(items)}], 'usage': {'output_tokens': 9}}


class LeakCheckTests(unittest.TestCase):
    def run_check(self, caller):
        return lc.check(INSTRUCTION, CLASSES, tempfile.mkdtemp(), base_url='http://x', caller=caller)

    def test_verified_quote_rejects(self):
        r = self.run_check(reply([{'class_id': 'c1', 'leaked': False, 'quote': None},
                                  {'class_id': 'c5', 'leaked': True, 'quote': 'Do not multiply by 100.'}]))
        self.assertEqual(r['status'], 'rejected')
        self.assertTrue(r['verdicts'][1]['quote_verified'])

    def test_no_leak_accepts(self):
        r = self.run_check(reply([{'class_id': 'c1', 'leaked': False}, {'class_id': 'c5', 'leaked': False}]))
        self.assertEqual(r['status'], 'accepted')

    def test_quote_not_in_instruction_blocks(self):
        r = self.run_check(reply([{'class_id': 'c1', 'leaked': False},
                                  {'class_id': 'c5', 'leaked': True, 'quote': 'never scale'}]))
        self.assertEqual((r['status'], r['verdicts'][1]['quote_verified']), ('blocked', False))

    def test_invalid_output_or_network_blocks(self):
        for caller in (reply([{'class_id': 'c1', 'leaked': False}]),
                       lambda *a: {'content': [{'type': 'text', 'text': 'no json'}]}):
            self.assertEqual(self.run_check(caller)['status'], 'blocked')

        def down(*a):
            raise urllib.error.URLError('refused')
        r = self.run_check(down)
        self.assertEqual(r['status'], 'blocked')
        self.assertIn('URLError', r['error'])


if __name__ == '__main__':
    unittest.main()
