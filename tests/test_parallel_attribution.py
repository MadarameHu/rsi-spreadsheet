import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys,tempfile,threading,time,unittest,subprocess
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
import parallel_attribution as pa
from stage_records import read_stage_commit

class FakeHarbor:
    bin='/fake/harbor'
    base_url='http://fake'
    def __init__(self,n=1,fail=None,invalid=None,bad_aggregate=False,timeout_sample=None):
        self.lock=threading.Lock();self.barrier=threading.Barrier(n)
        self.active=0;self.maximum=0;self.calls=[]
        self.fail=fail;self.invalid=invalid;self.bad_aggregate=bad_aggregate;self.timeout_sample=timeout_sample
    def agent_task(self,task,agent,model,job,timeout,specs,**kwargs):
        env=dict(x.split('=',1) for x in kwargs['agent_env'])
        sample=int(env['ADB_SAMPLE_ID']) if 'ADB_SAMPLE_ID' in env else None
        with self.lock:
            self.calls.append(dict(sample=sample,timeout=timeout,kwargs=kwargs,mounts=json.loads(specs)))
            self.active+=1;self.maximum=max(self.maximum,self.active)
        try:
            if sample is not None:
                self.barrier.wait(timeout=5);time.sleep((20-sample)*0.0005)
                if sample==self.fail:raise RuntimeError('simulated service failure')
                payload={'schema_version':'trajectory-observations-v1','sample_id':sample,'observations':[{
                    'id':f's{sample:02d}-o1','statement':'Observed action','evidence':[{'path':env['ADB_ATTEMPT'],'locator':'event1'}],'uncertainties':[]}]}
            else:
                index=json.loads(Path(env['ADB_OBSERVATIONS_INDEX']).read_text())
                assert index['status']=='completed'
                assert all(w['status']=='completed' and Path(w['commit_path']).is_file() for w in index['workers'])
                payload={'schema_version':'parallel-attribution-v1','claims':[{'id':'c1','kind':'capability','statement':'Grounded conditional capability',
                    'observation_ids':['missing'] if self.bad_aggregate else index['observation_ids'],'counterevidence_ids':[],
                    'conditions':['observed conditions'],'uncertainties':[],
                    'evidence':[{'path':str(Path(env['ADB_ROLLOUT_DIR'])/f"attempt-{index['sample_order'][0]}.md"),'locator':'event1'}]}]}
            if getattr(self,'compact',False):
                if sample is not None:
                    payload={'observations':[{'finding':'Observed action','evidence':['trace:all']}]}
                else:
                    payload={'claims':[{'kind':'capability','finding':'Grounded conditional capability',
                        'supports':index['observation_ids'],'counterevidence':[],'conditions':['observed conditions'],'evidence_strength':'replay_verified'}]}
            Path(env['ADB_RESULT']).write_text('invalid json' if sample is not None and sample==self.invalid else json.dumps(payload))
            if sample is not None and sample != self.invalid:
                subprocess.run([sys.executable,env['ADB_WORKER_VALIDATOR'],'--validate',env['ADB_RESULT'],'--manifest',env['ADB_INPUT_MANIFEST']],check=True,capture_output=True,text=True)
            trial=job/'run/trial/result.json';trial.parent.mkdir(parents=True)
            start=datetime(2026,9,22,tzinfo=timezone.utc)
            trial.write_text(json.dumps({'agent_execution':{'started_at':start.isoformat(),'finished_at':(start+timedelta(seconds=10)).isoformat()},'exception_info':({'exception_type':'AgentTimeoutError','exception_message':'configured limit'} if sample is not None and sample==self.timeout_sample else None),'finished_at':(start+timedelta(seconds=12)).isoformat()}))
            (job/'run/result.json').write_text('{"stats":{}}')
            (job/'raw.log').write_bytes(b'complete model transcript'*10000)
            return 0
        finally:
            with self.lock:self.active-=1

class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir();(self.source/'instruction.md').write_text('original')
        self.bundle=self.root/'bundle';self.bundle.mkdir();self.ids=[3,0,2,1];self.inputs(self.ids)
        self.ep=self.root/'ep';self.diag=self.ep/'diagnostics'
        self.cfg={'diagnostics':{'enabled':True,'attribution_mode':'parallel','parallel_attribution':{
            'max_concurrent':4,'worker_budget_min':1,'aggregate_budget_min':2,'design_budget_min':3}},
            'researcher':{'agent':'claude-code','model':'author','budget_min':45}}
    def tearDown(self):self.temp.cleanup()
    def inputs(self,ids):
        (self.bundle/'summary.json').write_text(json.dumps({'attempt_ids':ids,'attempts':len(ids)}))
        for i in ids:(self.bundle/f'attempt-{i}.md').write_bytes(f'FULL {i}\n'.encode()*30000)
    def run_pipeline(self,hb):return pa.produce_attribution(hb,self.cfg,self.source,self.bundle,self.ep,self.diag,{'benchmark':'spreadsheet'})
    def test_parallel_order_full_inputs_and_parent_commits(self):
        hb=FakeHarbor(4);commit=self.run_pipeline(hb)
        self.assertEqual(hb.maximum,4);self.assertEqual(len(hb.calls),5);self.assertIsNone(hb.calls[-1]['sample'])
        index=pa.st.load(self.diag/'observations/index.json')
        self.assertEqual(index['sample_order'],self.ids);self.assertEqual([w['sample_id'] for w in index['workers']],self.ids)
        self.assertEqual(index['observation_ids'],[f's{i:02d}-o1' for i in self.ids])
        self.assertEqual(commit['parent_ids'],[w['commit_id'] for w in index['workers']])
        for call in hb.calls:
            self.assertTrue(all(m['target']!='/var/run/docker.sock' for m in call['mounts']))
            self.assertEqual(len([m for m in call['mounts'] if not m.get('read_only')]),1)
            if call['sample'] is not None:
                views=[m for m in call['mounts'] if Path(m['target']).name=='inputs']
                self.assertEqual(len(views),1)
                self.assertEqual((Path(views[0]['target'])/'attempt.md').read_bytes(),(self.bundle/f"attempt-{call['sample']}.md").read_bytes())
                self.assertFalse(any(Path(m['target']) in (self.source,self.bundle,self.bundle/'summary.json') for m in call['mounts']))
                self.assertEqual(call['timeout'],990)
        self.assertEqual(hb.calls[-1]['timeout'],1080)
        worker=read_stage_commit(self.diag/'records',index['workers'][0]['commit_id'])
        ref=next(r for r in worker['references'] if r['original_path'].endswith('raw.log'))
        self.assertEqual((self.diag/'records'/ref['path']).read_bytes(),b'complete model transcript'*10000)
        self.assertEqual(self.run_pipeline(hb),commit);self.assertEqual(len(hb.calls),5)
    def test_timeout_artifact_commit_is_distinct_from_agent_completion(self):
        hb=FakeHarbor(4,timeout_sample=2);self.run_pipeline(hb)
        index=pa.st.load(self.diag/'observations/index.json')
        worker=next(w for w in index['workers'] if w['sample_id']==2)
        self.assertEqual(worker['artifact_status'],'committed')
        self.assertEqual(worker['agent_terminal_status'],'timeout')
        self.assertEqual(worker['terminal']['exception_type'],'AgentTimeoutError')
        self.assertEqual(len(hb.calls),5)

    def test_poisoned_parent_metadata_not_in_worker_mounts(self):
        (self.source/'_meta.json').write_text('{"prior_judgment":"POISON","stratum":"0/16"}')
        (self.source/'tests').mkdir();(self.source/'tests/target.xlsx').write_bytes(b'POISON')
        (self.source/'solution').mkdir();(self.source/'solution/solve.py').write_text('POISON')
        summary=pa.st.load(self.bundle/'summary.json');summary['rewards']=[0]*4
        (self.bundle/'summary.json').write_text(json.dumps(summary))
        hb=FakeHarbor(4);self.run_pipeline(hb)
        for call in hb.calls[:-1]:
            view=next(Path(m['target']) for m in call['mounts'] if Path(m['target']).name=='inputs')
            self.assertEqual({str(p.relative_to(view)) for p in view.rglob('*') if p.is_file()},
                             {'manifest.json','task/instruction.md','attempt.md','evidence_index.json'})
            self.assertNotIn('POISON',''.join(p.read_text() for p in view.rglob('*') if p.is_file()))

    def test_aggregate_binds_root_input_manifest(self):
        self.diag.mkdir(parents=True)
        manifest=self.diag/'inputs.json';manifest.write_text('{"original":{},"rollouts":{}}')
        commit=self.run_pipeline(FakeHarbor(4))
        reference=next(r for r in commit['references'] if r['original_path']==str(manifest.resolve()))
        self.assertEqual((self.diag/'records'/reference['path']).read_bytes(),manifest.read_bytes())
    def test_sixteen_workers(self):
        self.ids=[3,0,2,1]+list(range(4,16));self.inputs(self.ids)
        self.cfg['diagnostics']['parallel_attribution']['max_concurrent']=16
        hb=FakeHarbor(16);self.run_pipeline(hb)
        self.assertEqual(hb.maximum,16);self.assertEqual(len(hb.calls),17)
        self.assertEqual(pa.st.load(self.diag/'observations/index.json')['sample_order'],self.ids)
    def test_sixteen_samples_obey_lower_configured_concurrency(self):
        self.ids=[3,0,2,1]+list(range(4,16));self.inputs(self.ids)
        hb=FakeHarbor(4);self.run_pipeline(hb)
        self.assertEqual(hb.maximum,4);self.assertEqual(len(hb.calls),17)
    def test_failure_collects_all_no_aggregate_no_retry(self):
        hb=FakeHarbor(4,fail=2)
        with self.assertRaisesRegex(RuntimeError,'aggregate not started'):self.run_pipeline(hb)
        index=pa.st.load(self.diag/'observations/index.json')
        self.assertEqual(index['status'],'failed');self.assertEqual(len(hb.calls),4)
        self.assertEqual(sum(w['status']=='completed' for w in index['workers']),3)
        self.assertTrue((self.diag/'observations/sample-02/invocation_error.json').is_file())
        with self.assertRaisesRegex(RuntimeError,'failed previously'):self.run_pipeline(hb)
        self.assertEqual(len(hb.calls),4)
    def test_invalid_worker_raw_retained(self):
        hb=FakeHarbor(4,invalid=1)
        with self.assertRaises(RuntimeError):self.run_pipeline(hb)
        self.assertEqual((self.diag/'observations/sample-01/work/output.json').read_text(),'invalid json')
        self.assertEqual(len(hb.calls),4)
    def test_invalid_aggregate_no_retry(self):
        hb=FakeHarbor(4,bad_aggregate=True)
        with self.assertRaises(ValueError):self.run_pipeline(hb)
        raw=(self.diag/'attribution/work/output.json').read_bytes();self.assertIn(b'missing',raw)
        with self.assertRaisesRegex(RuntimeError,'aggregate failed previously'):self.run_pipeline(hb)
        self.assertEqual(len(hb.calls),5);self.assertEqual((self.diag/'attribution/work/output.json').read_bytes(),raw)
    def test_inventory_validation_before_calls(self):
        hb=FakeHarbor()
        for ids in ([3,0,2],[3,0,2,2],[]):
            (self.bundle/'summary.json').write_text(json.dumps({'attempt_ids':ids}))
            with self.assertRaises(ValueError):self.run_pipeline(hb)
        self.assertEqual(hb.calls,[])
    def test_explicit_budgets_required(self):
        hb=FakeHarbor()
        for value in (None,False,0,-1,float('inf')):
            self.cfg['diagnostics']['parallel_attribution']['worker_budget_min']=value
            with self.assertRaises(ValueError):self.run_pipeline(hb)
        self.assertEqual(hb.calls,[])
    def test_local_recovery_after_models_finish(self):
        hb=FakeHarbor(4);original=pa.st.save_new
        def crash(path,value):
            if Path(path).name=='attribution_commit.json':raise KeyboardInterrupt('process loss')
            return original(path,value)
        with patch.object(pa.st,'save_new',side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):self.run_pipeline(hb)
        self.assertEqual(len(hb.calls),5)
        self.assertEqual(self.run_pipeline(hb)['stage'],'attribution');self.assertEqual(len(hb.calls),5)
    def test_unfinished_batch_no_new_invocations(self):
        obs=self.diag/'observations';obs.mkdir(parents=True);(obs/'index.json').write_text('{"status":"running"}')
        hb=FakeHarbor()
        with self.assertRaisesRegex(RuntimeError,'unfinished invocations'):self.run_pipeline(hb)
        self.assertEqual(hb.calls,[])
    def test_existing_untracked_work_retained_without_model_replay(self):
        directory=self.diag/'observations/sample-03/work';directory.mkdir(parents=True)
        original=directory/'output.json';original.write_text('untracked raw bytes')
        hb=FakeHarbor()
        with self.assertRaises(RuntimeError):self.run_pipeline(hb)
        self.assertEqual(original.read_text(),'untracked raw bytes')
        self.assertEqual({call['sample'] for call in hb.calls},{0,1,2})
    def test_changed_input_reuse_rejected(self):
        hb=FakeHarbor(4);self.run_pipeline(hb);(self.bundle/'attempt-0.md').write_text('changed')
        with self.assertRaises(ValueError):self.run_pipeline(hb)
        self.assertEqual(len(hb.calls),5)

if __name__=='__main__':unittest.main()
