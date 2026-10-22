"""The parallel author produces artifacts; execution belongs to the host."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import staged_synth as staged


class AuthorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sources = [staged.ROOT/'agents/synth/instruction.md',
                   staged.ROOT/'agents/synth-design-parallel/instruction.md']
        placeholders = set(re.findall(r'__[A-Z0-9_]+__', '\n'.join(p.read_text() for p in sources)))
        self.values = {key: str(self.root/key.strip('_')) for key in placeholders}
        self.values['__BUDGET_MIN__'] = '30'
        self.values['__GATEWAY_BASE_URL__'] = 'http://fixture.invalid'

    def test_parallel_entrypoints_actually_reject_execution(self):
        rendered = staged.render_stage(self.root/'parallel', self.values)
        for name in ('adb-oracle', 'adb-testrun', 'harbor'):
            with self.subTest(name=name):
                result = subprocess.run(['/bin/sh', str(rendered/'environment/bin'/name),
                                         'run', '-p', '/nonexistent-candidate'],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Save one candidate and design.json, then finish', result.stderr)
                self.assertIn('independent host validation', result.stderr)
        self.assertFalse((rendered/'environment/bin/adb-snapshot-task').exists())

    def test_parallel_prompt_keeps_evidence_contract_and_ends_after_authoring(self):
        rendered = staged.render_stage(self.root/'parallel', self.values)
        prompt = (rendered/'instruction.md').read_text()
        for key in ('__ATTRIBUTION_JSON__', '__ORIGINAL_DIR__', '__OBSERVATIONS_INDEX__',
                    '__ROLLOUT_DIR__', '__OUT__', '__DESIGN_JSON__'):
            self.assertIn(self.values[key], prompt)
        for phrase in ('primary_goal_id', 'selection_reason', 'test_intent', 'reference',
                       'scoring logic', 'not a trusted answer key',
                       'Do not execute the reference', 'Do not poll for validation results',
                       'Finish your response after saving the deliverables'):
            self.assertIn(phrase, prompt)
        for phrase in ('launch three', 'wait for their results', 'After testing,',
                       'Test waits still count', 'Use adb-oracle or adb-testrun',
                       'Read several of them', 'Run `adb-oracle'):
            self.assertNotIn(phrase, prompt)
        self.assertNotRegex(prompt, r'__[A-Z0-9_]+__')

if __name__ == '__main__':
    unittest.main()
