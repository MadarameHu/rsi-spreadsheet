import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from stage_records import write_stage_commit, read_stage_commit
from capture_failed_delivery import capture

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.ep=Path(self.tmp.name)/'ep';self.diag=self.ep/'diagnostics'
        self.diag.mkdir(parents=True);self.work=self.ep/'work';self.work.mkdir()
        self.failure=b'{"status":"failed","error":"TimeoutExpired","traceback":"complete original trace"}\n'
        (self.diag/'synthesis.json').write_bytes(self.failure)
        evidence=self.diag/'evidence.txt';evidence.write_text('source')
        self.a=write_stage_commit(self.diag/'records','attribution',{'claims':[{'id':'c1','statement':'source claim','observed_as':'failure','evidence':[{'path':str(evidence),'locator':'line 1'}],'uncertainties':[]}]})
        (self.diag/'attribution_commit.json').write_text(json.dumps({'commit_id':self.a['commit_id']}))
        (self.work/'design.json').write_text(json.dumps({'mappings':[{'id':'m1','claim_id':'c1','feature':'feature','check_plan':'probe'}]}))
        self.candidate=self.work/'out/task';self.candidate.mkdir(parents=True)
        (self.candidate/'task.toml').write_text('version="1.0"\n');(self.candidate/'instruction.md').write_text('candidate')
    def tearDown(self):self.tmp.cleanup()
    def test_capture_keeps_failure_and_original_bytes(self):
        before={p.name:p.read_bytes() for p in self.candidate.iterdir()}
        receipt=capture(self.ep);state=json.loads((self.diag/'synthesis.json').read_text())
        self.assertEqual(state['status'],'failed');self.assertEqual(receipt['author_calls_added'],0)
        self.assertTrue(receipt['author_receipt_missing'])
        self.assertEqual((self.diag/'failed-synthesis-before-capture.json').read_bytes(),self.failure)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.candidate.iterdir()})
        commit=read_stage_commit(self.diag/'records',state['design_commit'])
        self.assertEqual(commit['metadata']['source_synthesis_status'],'failed')
        self.assertEqual(commit['parent_ids'],[self.a['commit_id']])
        with self.assertRaises(ValueError):capture(self.ep)
    def test_no_candidate_selection_from_multiple_deliveries(self):
        extra=self.work/'out/other';extra.mkdir();(extra/'task.toml').write_text('');(extra/'instruction.md').write_text('')
        with self.assertRaises(ValueError):capture(self.ep)
        self.assertEqual((self.diag/'synthesis.json').read_bytes(),self.failure)
    def test_malformed_design_is_not_repaired(self):
        (self.work/'design.json').write_text('broken json')
        with self.assertRaises(ValueError):capture(self.ep)
        self.assertEqual((self.work/'design.json').read_text(),'broken json')
        self.assertEqual((self.diag/'synthesis.json').read_bytes(),self.failure)

if __name__=='__main__':unittest.main()
