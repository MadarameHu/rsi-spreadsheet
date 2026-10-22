import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import diagnostics
import staged_synth as staged
import stage_audit
import stage_metrics
from test_parallel_attribution import FakeHarbor as ObservationHarbor
from test_stage_audit import FakeHarbor as AuditHarbor


class FullAuthorHarbor(ObservationHarbor):
    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        env = dict(item.split('=', 1) for item in kwargs['agent_env'])
        if 'ADB_DESIGN_JSON' not in env:
            rc = super().agent_task(task, agent, model, job, timeout, mounts, **kwargs)
            if 'ADB_OBSERVATIONS_INDEX' in env:
                output = Path(env['ADB_RESULT'])
                calculation = output.parent / 'calculation.txt'
                calculation.write_text('Independent aggregation check against source events')
                payload = json.loads(output.read_text())
                payload['claims'][0].setdefault('evidence',[]).append({'path':str(calculation),'locator':'complete calculation'})
                output.write_text(json.dumps(payload))
                index = json.loads(Path(env['ADB_OBSERVATIONS_INDEX']).read_text())
                read_only = [Path(item['target']) for item in json.loads(mounts) if item.get('read_only')]
                for worker in index['workers']:
                    for key in ('output_path','job_path','commit_path'):
                        path = Path(worker[key])
                        assert any(path == target or target in path.parents for target in read_only), key
            return rc
        self.calls.append({'stage': 'design', 'timeout': timeout, 'kwargs': kwargs, 'mounts': json.loads(mounts)})
        attribution = Path(env['ADB_ATTRIBUTION_JSON'])
        assert (attribution.parent / 'attribution_commit.json').is_file()
        assert all(w['status'] == 'completed' for w in json.loads((attribution.parent / 'observations/index.json').read_text())['workers'])
        payload = {'mappings': [{'id': 'm1', 'claim_id': 'c1', 'role': 'target', 'feature': 'Exercise the grounded capability',
                                'check_plan': 'Independent positive and negative workbook', 'preserves': ['Known input semantics'], 'simplifies': ['Output size']}]}
        if getattr(self,'compact',False):
            payload={'targets':[{'claim_id':'c1','feature':'Exercise the grounded capability',
                                 'test_intent':'Independent positive and negative workbook'}]}
        Path(env['ADB_DESIGN_JSON']).write_text(json.dumps(payload))
        candidate = Path(env['ADB_OUT']) / 'candidate-task'
        candidate.mkdir()
        (candidate / 'task.toml').write_text('artifacts = ["/app/data.xlsx"]\n[metadata]\npass_threshold=1.0\n')
        for name in ('instruction.md', 'tests/test.sh', 'environment/Dockerfile', 'solution/solve.sh'):
            path = candidate / name; path.parent.mkdir(exist_ok=True); path.write_text('fixture ' + name)
        # Authoring only saves its candidate. Host validation starts after return.
        start = datetime(2026, 9, 22, tzinfo=timezone.utc)
        result = job / 'run/trial/result.json'; result.parent.mkdir(parents=True)
        result.write_text(json.dumps({'agent_execution': {'started_at': start.isoformat(), 'finished_at': (start+timedelta(seconds=10)).isoformat()}, 'exception_info': None}))
        (job / 'run/result.json').write_text('{"stats":{}}')
        return 0


class ParallelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'bench/source'; self.source.mkdir(parents=True)
        (self.source/'instruction.md').write_text('complete original task')
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        self.ids = [2, 0, 3, 1]
        (self.bundle/'summary.json').write_text(json.dumps({'attempt_ids':self.ids,'attempts':4}))
        for i in self.ids: (self.bundle/f'attempt-{i}.md').write_text(f'complete sample {i}\n'*2000)
        self.cfg = {'diagnostics': {'enabled': True, 'attribution_mode':'parallel', 'parallel_attribution':{
                    'max_concurrent':4,'worker_budget_min':1,'aggregate_budget_min':2,'design_budget_min':3}},
                    'customer':{'model':'target','attempts':4},'score':{'band_lo':0.1,'band_hi':0.9,'n_concurrent':4},
                    'gateway':{'base_url':'http://fake'},'researcher':{'agent':'claude-code','model':'author','budget_min':45,'max_concurrent_docker':1},
                    'judge':{'agent':'claude-code','model':'judge','budget_min':4},'task_timeout_sec':{'bench':100}}
        self.ep = self.root/'ep01'
        for obj, name, value in ((staged,'benchmark_dir',self.source.parent),(staged,'rollout_bundle_path',self.bundle)):
            context=patch.object(obj,name,return_value=value);context.start();self.addCleanup(context.stop)

    def test_complete_observations_aggregate_design_audit_and_metrics(self):
        author=FullAuthorHarbor(4)
        rec=staged.synthesize_staged(author,self.cfg,'bench',self.source,self.ep,1)
        self.assertEqual(len(author.calls),6)
        self.assertEqual(author.maximum,4)
        self.assertEqual(author.calls[-1]['stage'],'design')
        self.assertEqual(author.calls[-1]['kwargs']['agent_timeout_mult'],180/1800)
        self.assertEqual(rec['budget_mode'],diagnostics.PARALLEL_BUDGET_MODE)
        self.assertEqual(rec['agent_execution_seconds'],{'aggregate':10.0,'design':10.0})
        self.assertTrue(any(m['target']==str(self.ep/'diagnostics/observations') and m.get('read_only') for m in author.calls[-1]['mounts']))
        for directory in ('records','attribution/work'):
            self.assertTrue(any(m['target']==str(self.ep/'diagnostics'/directory) and m.get('read_only') for m in author.calls[-1]['mounts']))
        inventory=json.loads((self.ep/'diagnostics/observations/input_inventory.json').read_text())
        self.assertEqual(inventory['maximum_agent_execution_minutes'],9)
        self.assertEqual(inventory['parallel_batch_wall_allocation_minutes_excluding_setup'],6)
        reviewer=AuditHarbor()
        audit=stage_audit.audit_episode(reviewer,self.cfg,'bench','source',self.ep)
        self.assertEqual(audit['status'],'completed',audit['failures'])
        review_calls=len(reviewer.calls)
        self.assertEqual(len(reviewer.probe_calls),2)
        metrics=stage_metrics.summarize_envelope(audit)
        self.assertEqual(metrics['metrics']['observations']['denominator'],4)
        self.assertEqual(metrics['metrics']['probes']['accuracy'],1)
        self.assertEqual(staged.synthesize_staged(author,self.cfg,'bench',self.source,self.ep,1),rec)
        self.assertEqual(len(author.calls),6)
        self.assertEqual(stage_audit.audit_episode(reviewer,self.cfg,'bench','source',self.ep)['audit_commit'],audit['audit_commit'])
        self.assertEqual(len(reviewer.calls),review_calls)

    def test_compact_raw_derived_snapshot_and_audit_end_to_end(self):
        author=FullAuthorHarbor(4);author.compact=True
        staged.synthesize_staged(author,self.cfg,'bench',self.source,self.ep,1)
        diag=self.ep/'diagnostics'
        index=json.loads((diag/'observations/index.json').read_text())
        for worker in index['workers']:
            raw=json.loads(Path(worker['raw_output_path']).read_text())
            derived=json.loads(Path(worker['output_path']).read_text())
            self.assertNotIn('sample_id',raw)
            self.assertEqual(derived['sample_id'],worker['sample_id'])
            self.assertEqual(raw['observations'][0]['finding'],derived['observations'][0]['statement'])
            self.assertEqual(derived['observations'][0]['uncertainty_status'],'not_declared')
            self.assertNotIn('path',raw['observations'][0]['evidence'][0])
        raw_claim=json.loads((diag/'attribution/work/output.json').read_text())['claims'][0]
        claim=json.loads((diag/'attribution.json').read_text())['claims'][0]
        self.assertNotIn('id',raw_claim);self.assertEqual(claim['id'],'c1')
        raw_design=json.loads((self.ep/'work/design.json').read_text())
        design=json.loads((diag/'design.json').read_text())
        self.assertIn('targets',raw_design);self.assertIn('mappings',design)
        self.assertEqual(design['preserves_status'],'not_declared')
        reviewer=AuditHarbor();audit=stage_audit.audit_episode(reviewer,self.cfg,'bench','source',self.ep)
        self.assertEqual(audit['status'],'completed',audit['failures'])
        metrics=stage_metrics.summarize_envelope(audit)
        self.assertEqual(metrics['metrics']['observations']['denominator'],4)
        self.assertEqual(metrics['metrics']['mappings']['total'],1)
        self.assertEqual(metrics['metrics']['mappings']['denominator'],0)
        self.assertIsNone(metrics['metrics']['mappings']['accuracy'])
        self.assertEqual(len(author.calls),6)

    def test_explicit_budget_validation_precedes_all_model_invocations(self):
        for key in ('worker_budget_min','aggregate_budget_min','design_budget_min'):
            for value in (None, False, 0, float('inf')):
                cfg=copy.deepcopy(self.cfg);cfg['diagnostics']['parallel_attribution'][key]=value
                author=FullAuthorHarbor(4)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    staged.synthesize_staged(author,cfg,'bench',self.source,self.ep,1)
                self.assertEqual(author.calls,[])
                self.assertFalse(self.ep.exists())
        cfg=copy.deepcopy(self.cfg);cfg['diagnostics']['parallel_attribution']['max_concurrent']=17
        with self.assertRaises(ValueError):diagnostics.parallel_config(cfg)

    def test_sequential_attribution_is_rejected(self):
        self.assertFalse(diagnostics.enabled({}))
        cfg=copy.deepcopy(self.cfg);cfg['diagnostics']['attribution_mode']='sequential'
        with self.assertRaises(ValueError):diagnostics.enabled(cfg)


if __name__=='__main__':unittest.main()
