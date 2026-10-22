"""Minimal model output enrichment and original-evidence references, offline."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from harness.compact_observations import enrich_observations
from harness.worker_inputs import prepare_worker_inputs, validate_worker_output


class CompactObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'instruction.md').write_text('Task\nSecond line\nThird line\n')
        (self.source/'system_prompt.md').write_text('System\nSecond line\n')
        self.attempt = self.root/'attempt.md'
        self.attempt.write_bytes(b'# Original\r\n## Official score and execution summary\r\nscore\r\n## Original task prompt\r\ntask\r\n### step 1 / event 0 / thought\r\nthink\r\n### step 1 / event 1 / tool_call\r\ncall\r\n### step 1 / event 2 / tool_response\r\nresponse\r\n')
        self.prepared = prepare_worker_inputs(self.source, self.attempt, self.root/'worker', 11)

    def enrich(self, refs=None, **kwargs):
        item = {'finding': 'Exact statement 中文 ', 'evidence': refs or ['event:0']}
        item.update(kwargs)
        return enrich_observations({'observations': [item]}, self.prepared['manifest'])

    def test_exact_text_no_mutation_and_uncertainty_states(self):
        raw = {'observations': [{'finding': 'Exact statement 中文 ', 'evidence': ['event:0']},
                                {'finding': 'B', 'evidence': ['event:1'], 'uncertainty': None},
                                {'finding': 'C', 'evidence': ['event:2'], 'uncertainty': 'unknown'}]}
        before = copy.deepcopy(raw)
        output = enrich_observations(raw, self.prepared['manifest'])
        self.assertEqual(raw, before)
        self.assertEqual([x['id'] for x in output['observations']], ['s11-o1','s11-o2','s11-o3'])
        self.assertEqual([x['uncertainty_status'] for x in output['observations']], ['not_declared','explicit_null','declared'])
        self.assertEqual([x['uncertainties'] for x in output['observations']], [[],[],['unknown']])
        self.assertEqual(output['observations'][0]['statement'], raw['observations'][0]['finding'])
        self.assertEqual(output['limitations_status'], 'not_declared')
        self.assertNotIn('evidence_limitations', output)

    def test_event_range_score_full_and_source_lines(self):
        refs = ['event:0','events:0-2','trace:score','trace:all','task:lines:1-3','system:lines:2']
        evidence = self.enrich(refs)['observations'][0]['evidence']
        self.assertEqual([e['locator'] for e in evidence], ['event:0; lines 6-7','events:0-2; lines 6-11',
                         'trace:score; lines 2-3','trace:all; lines 1-11','task:lines:1-3; lines 1-3','system:lines:2; lines 2-2'])
        self.assertEqual(evidence[0]['path'], self.prepared['attempt'])
        self.assertEqual(Path(self.prepared['attempt']).read_bytes(), self.attempt.read_bytes())

    def test_invalid_references_and_unknown_fields_not_repaired(self):
        for ref in ['event:3','events:0-3','events:2-0','task:lines:0-1','system:lines:1-3',
                    'other:1','EVENT:0','event:0 ', 'input:Sheet!A1']:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.enrich([ref])
        for payload in [{'observations': [], 'sample_id': 11},
                        {'observations': [{'finding': 'x', 'evidence': ['event:0'], 'id': 's11-o1'}]},
                        {'observations': [{'finding': 'x', 'evidence': []}]},
                        {'observations': [{'finding': 'x', 'evidence': ['event:0'], 'uncertainty': []}]}]:
            before = copy.deepcopy(payload)
            with self.assertRaises(ValueError):
                enrich_observations(payload, self.prepared['manifest'])
            self.assertEqual(before, payload)

    def test_missing_event_and_score_sections_rejected(self):
        self.attempt.write_text('Full original trajectory without event headings\n')
        self.prepared = prepare_worker_inputs(self.source, self.attempt, self.root/'noevents', 11)
        index = json.loads(Path(self.prepared['evidence_index']).read_bytes())
        self.assertEqual(index['events'], [])
        self.enrich(['trace:all'])
        for ref in ['event:0', 'trace:score']:
            with self.assertRaises(ValueError):
                self.enrich([ref])

    def test_event_range_requires_every_event(self):
        self.attempt.write_text('### step 1 / event 0 / thought\nx\n### step 2 / event 2 / text\ny\n')
        self.prepared = prepare_worker_inputs(self.source, self.attempt, self.root/'gap', 11)
        with self.assertRaises(ValueError):
            self.enrich(['events:0-2'])

    def test_index_is_mechanical_and_hash_protected(self):
        index_path = Path(self.prepared['evidence_index'])
        index = json.loads(index_path.read_bytes())
        self.assertNotIn('finding', json.dumps(index))
        self.assertEqual(index['events'][0]['event'], 0)
        index['events'][0]['line_start'] = 1
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            self.enrich()

    def test_workbook_existing_sheet_and_in_bounds_coordinates(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest('openpyxl not installed')
        (self.source/'environment').mkdir()
        wb = Workbook()
        wb.active.title = 'Schedule_Weekly'
        wb.active['H26'] = 'End'
        wb.save(self.source/'environment'/'input.xlsx')
        self.prepared = prepare_worker_inputs(self.source, self.attempt, self.root/'xlsx', 11)
        self.enrich(['input:Schedule_Weekly!A23:H26'])
        for ref in ['input:Missing!A1', 'input:Schedule_Weekly!A23:I26',
                    'input:Schedule_Weekly!A23:H27', 'input:Schedule_Weekly!H26:A1']:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.enrich([ref])

    def test_legacy_and_compact_cli(self):
        raw = {'observations': [{'finding': 'x', 'evidence': ['event:0']}], 'limitations': []}
        canonical = enrich_observations(raw, self.prepared['manifest'])
        validate_worker_output(canonical, self.prepared['manifest'])
        validate_worker_output(raw, self.prepared['manifest'])
        isolated = self.root/'cli'
        isolated.mkdir()
        repo = Path(__file__).resolve().parents[1]
        for name in ['worker_inputs.py','compact_observations.py','parallel_records.py','stage_records.py']:
            shutil.copyfile(repo/'harness'/name, isolated/name)
        output = self.root/'output.json'
        output.write_text(json.dumps(raw))
        result = subprocess.run([sys.executable, str(isolated/'worker_inputs.py'), '--validate', str(output),
                                 '--manifest', self.prepared['manifest']], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('sample 11, 1 observations', result.stdout)
        self.assertEqual(json.loads(output.read_text()), raw)


if __name__ == '__main__':
    unittest.main()
