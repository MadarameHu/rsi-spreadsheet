import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import mounts
import stage_audit as audit
from stage_records import snapshot_tree, write_stage_commit


class FakeHarbor:
    bin = '/fixture/bin/harbor'
    setup_mult = 4

    def __init__(self, malformed=False, infra=False):
        self.calls = []
        self.probe_calls = []
        self.malformed = malformed
        self.infra = infra

    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        self.calls.append(dict(agent=agent, model=model, timeout=timeout, kwargs=kwargs))
        result = Path(next(v.split('=', 1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
        work = result.parent
        if self.malformed:
            result.write_text('{invalid json')
            return 0
        evidence = [{'path': str(work/'calculation.txt'), 'locator': 'full independent calculation'}]
        (work/'calculation.txt').write_text('Independent expectation from task, before verifier.')
        # Parallel review runs one role per unit; each unit writes only its assigned role.
        request = audit.read_json(work.parent/'input.json')
        kind, assigned = request['kind'], request['assigned']
        if kind == 'observations':
            payload = {'observations': [dict(id=i, verdict='correct', reason='raw sample agrees', evidence=evidence)
                                        for i in assigned['observations']]}
        elif kind == 'claims':
            payload = {'claims': [dict(id=i, verdict='correct', reason='source agrees', evidence=evidence)
                                  for i in assigned['claims']]}
        else:
            payload = {'mappings': [{'id':i,'implementation_verdict':'uncertain',
                                     'reason':'implementation uncertain','evidence':evidence}
                                    for i in assigned['mappings']],
                       'task_checks': [{'id':x,'verdict':'uncertain','reason':'not fully established','evidence':evidence}
                                       for x in assigned['task_checks']], 'probes': []}
        plans = []
        for pid, valid in ([('correct_solution',True),('incorrect_solution',False)] if kind == 'task' else []):
            workbook = work/(pid+'.xlsx')
            workbook.write_bytes(b'correct' if valid else b'wrong')
            payload['probes'].append(dict(id=pid,expected_valid=valid,actual_accept=None,
                                         execution_status='missing',reason='advance judgement',evidence=evidence))
            plans.append(dict(id=pid,workbook=str(workbook),target_path='/app/data.xlsx',
                              expected_valid=valid,reason='advance judgement',evidence=evidence))
        audit.write_json(result,payload)
        if kind == 'task':
            audit.write_json(work/'probe_plan.json',{'probes':plans})
        Path(job).mkdir()
        (Path(job)/'full-trajectory.json').write_text('complete reviewer output')
        audit.write_json(Path(job)/'result.json',{'exception_info':None,
                                                'agent_execution':{'duration':120.0}})
        return 0

    def run(self, args, job, timeout):
        task = Path(args[args.index('-p')+1])
        expectation = task.parent/'expectation.json'
        self.probe_calls.append(audit.read_json(expectation))
        self.assert_frozen = expectation.is_file()
        reward = 1.0 if (task/'solution/stage-audit-workbook.xlsx').read_bytes() == b'correct' else 0.0
        Path(job).mkdir()
        audit.write_json(Path(job)/'result.json',{'verifier_result':{'rewards':{'reward':reward}},
                                               'exception_info':{'exception_type':'SetupError'} if self.infra else None})
        return 0


def review_output(result, unit='task'):
    return Path(result['work_dir'])/'reviews'/unit/'work/review.json'


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ep = self.root/'ep01'
        self.diag = self.ep/'diagnostics'
        self.diag.mkdir(parents=True)
        source = self.root/'source'; source.mkdir()
        (source/'instruction.md').write_text('Original full task')
        rollout = self.root/'rollout';rollout.mkdir()
        (rollout/'attempt-0.json').write_text('complete original trajectory'*1000)
        inputs = {'original':snapshot_tree(source,self.diag/'original'),
                  'rollouts':snapshot_tree(rollout,self.diag/'rollouts')}
        audit.write_json(self.diag/'inputs.json',inputs)
        ap = {'claims':[{'id':'c1','statement':'one observed mechanism','observed_as':'failure',
                          'evidence':[{'path':str(self.diag/'original/instruction.md'),'locator':'full task'}],
                          'uncertainties':[]}]}
        audit.write_json(self.diag/'attribution.json',ap)
        records = self.diag/'records'
        ac = write_stage_commit(records,'attribution',ap,
                                references=[self.diag/'attribution.json',self.diag/'inputs.json'])
        candidate = self.root/'authored';candidate.mkdir()
        (candidate/'task.toml').write_text('artifacts = ["/app/data.xlsx"]\n[metadata]\npass_threshold = 1.0\n')
        for name in ['instruction.md','tests/test.sh','tests/target.xlsx','environment/Dockerfile','solution/solve.sh']:
            p=candidate/name;p.parent.mkdir(exist_ok=True);p.write_text('original '+name)
        cm=snapshot_tree(candidate,self.diag/'candidate')
        audit.write_json(self.diag/'candidate_manifest.json',cm)
        dp={'mappings':[{'id':'m1','claim_id':'c1','feature':'feature','check_plan':'check'}]}
        audit.write_json(self.diag/'design.json',dp)
        dc=write_stage_commit(records,'design',dp,claims=ap,parent_ids=[ac['commit_id']],
                              references=[self.diag/'design.json',self.diag/'candidate_manifest.json'],
                              metadata={'candidate_sha256':cm['tree_sha256']})
        audit.write_json(self.diag/'synthesis.json',dict(status='completed',stage_commit_dir=str(records),
                         attribution_commit=ac['commit_id'],design_commit=dc['commit_id'],
                         candidate_manifest=str(self.diag/'candidate_manifest.json')))
        self.cfg={'judge':{'agent':'claude-code','model':'configured-judge','budget_min':37},
                  'task_timeout_sec':{'spreadsheet':900},'score':{'n_concurrent':4}}
        self.mount = patch.object(mounts,'build',return_value='[]')
        self.mount_mock = self.mount.start();self.addCleanup(self.mount.stop)

    def call(self,hb):
        return audit.audit_episode(hb,self.cfg,'spreadsheet','source',self.ep)

    def parallel_fixture(self, compact=False, bind_raw=True):
        from stage_records import _tree_entries
        import hashlib
        records = self.diag/'records'
        workers, commits, ids = [], [], []
        evidence=[{'path':str(self.diag/'original/instruction.md'),'locator':'original full task'}]
        for sample in (3, 0):
            directory = self.diag/'observations'/f'sample-{sample:02d}'
            (directory/'work').mkdir(parents=True)
            (directory/'job').mkdir()
            raw=directory/'job/full.json';raw.write_text('complete worker trajectory')
            entries=_tree_entries(directory/'job')
            manifest=directory/'job_manifest.json'
            audit.write_json(manifest,{'entries':entries,'tree_sha256':hashlib.sha256(audit.encoded(entries)).hexdigest()})
            oid=f's{sample:02d}-o1'
            payload={'schema_version':'trajectory-observations-v1','sample_id':sample,
                     'observations':[dict(id=oid,statement='one observable action',evidence=evidence,uncertainties=[])]}
            if sample == 0:
                payload['observations'].append(dict(id='s00-o2',statement='unused observable action',
                                                    evidence=evidence,uncertainties=[]))
            worker_ids=[item['id'] for item in payload['observations']]
            output=directory/('observations.json' if compact else 'work/output.json')
            audit.write_json(output,payload)
            references=[output,manifest,raw]
            metadata={}
            raw_output=directory/'work/output.json'
            if compact:
                audit.write_json(raw_output,{'observations':[
                    {'finding':o['statement'],'evidence':['event:3']} for o in payload['observations']]})
                metadata={'raw_output_path':str(raw_output),'raw_output_sha256':audit.sha(raw_output)}
                if bind_raw:
                    references.append(raw_output)
            commit=write_stage_commit(records,'observations',payload,references=references,metadata=metadata)
            commits.append(commit['commit_id']);ids.extend(worker_ids)
            workers.append(dict(sample_id=sample,status='completed',directory=str(directory),
                                output_path=str(output),commit_id=commit['commit_id'],
                                job_path=str(directory/'job'),observation_ids=worker_ids))
            if compact:
                workers[-1]['raw_output_path']=str(raw_output)
        index=self.diag/'observations/index.json'
        audit.write_json(index,dict(schema_version='observation-index-v1',status='completed',
                                    sample_order=[3,0],workers=workers,observation_ids=ids))
        payload={'schema_version':'parallel-attribution-v1','claims':[dict(id='c1',kind='defect',
                 statement='conditional grouping defect',observation_ids=[ids[0]],counterevidence_ids=[ids[1]],
                 conditions=['when repeated'],evidence=evidence,uncertainties=[])]}
        if compact:
            payload['claims'].append(dict(payload['claims'][0],id='c2',kind='capability',
                                           statement='another supported ability'))
        audit.write_json(self.diag/'attribution.json',payload)
        ac=write_stage_commit(records,'attribution',payload,parent_ids=commits,
                              references=[self.diag/'inputs.json',self.diag/'attribution.json',index])
        design={'mappings':[dict(id='m1',claim_id='c1',feature='feature',check_plan='check',
                                role='target',preserves=[],simplifies=[])]}
        if compact:
            design.update(schema_version='compact-design-derived-v1',unselected_claim_ids=['c2'],
                          preserves=None,preserves_status='not_declared',
                          simplifies=None,simplifies_status='not_declared')
        audit.write_json(self.diag/'design.json',design)
        cm=audit.read_json(self.diag/'candidate_manifest.json')
        dc=write_stage_commit(records,'design',design,claims=payload,parent_ids=[ac['commit_id']],
                              references=[self.diag/'design.json',self.diag/'candidate_manifest.json'],
                              metadata={'candidate_sha256':cm['tree_sha256']})
        synthesis=audit.read_json(self.diag/'synthesis.json')
        synthesis.update(attribution_commit=ac['commit_id'],design_commit=dc['commit_id'])
        audit.write_json(self.diag/'synthesis.json',synthesis)
        return ids

    def test_compact_raw_and_derived_worker_records_are_both_preserved(self):
        self.parallel_fixture(compact=True)
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'completed',result['failures'])
        manifest=audit.read_json(Path(result['work_dir'])/'evidence/manifest.json')
        refs=manifest['commits']['observation-03']['references']
        self.assertTrue(any(r['original_path'].endswith('/work/output.json') for r in refs))
        self.assertTrue(any(r['original_path'].endswith('/observations.json') for r in refs))
        from stage_metrics import summarize_envelope
        metrics=summarize_envelope(result)['metrics']
        self.assertEqual(result['expected_units']['claims'],['c1','c2'])
        self.assertEqual(result['expected_units']['mappings'],['m1'])
        self.assertEqual(metrics['mappings']['total'],1)
        self.assertEqual(metrics['mappings']['missing'],0)
        self.assertEqual(metrics['mappings']['uncertain'],1)
        self.assertIsNone(metrics['mappings']['accuracy'])

    def rebind_parallel_index(self, index):
        """Bind a fixture index through new real immutable aggregate/design commits."""
        path = self.diag/'observations/index.json'
        audit.write_json(path, index)
        records = self.diag/'records'
        synthesis = audit.read_json(self.diag/'synthesis.json')
        old_a = audit.read_stage_commit(records, synthesis['attribution_commit'])
        old_d = audit.read_stage_commit(records, synthesis['design_commit'])
        ac = write_stage_commit(records, 'attribution', old_a['payload'],
            parent_ids=old_a['parent_ids'],
            references=[self.diag/'inputs.json', self.diag/'attribution.json', path])
        dc = write_stage_commit(records, 'design', old_d['payload'], claims=ac['payload'],
            parent_ids=[ac['commit_id']], metadata=old_d['metadata'],
            references=[self.diag/'design.json', self.diag/'candidate_manifest.json'])
        synthesis.update(stage_commit_dir=str(records), attribution_commit=ac['commit_id'],
                         design_commit=dc['commit_id'],
                         candidate_manifest=str(self.diag/'candidate_manifest.json'))
        audit.write_json(self.diag/'synthesis.json', synthesis)

    def reused_parallel_fixture(self):
        self.parallel_fixture(compact=True)
        origin = self.diag
        self.ep = self.root/'reused-ep'
        self.diag = self.ep/'diagnostics'
        shutil.copytree(origin, self.diag)
        index = audit.read_json(self.diag/'observations/index.json')
        index['reuse_observations_from'] = str(origin)
        for worker in index['workers']:
            worker.update(reused_from=str(origin), reused_commit_id=worker['commit_id'])
        self.rebind_parallel_index(index)
        return origin

    def test_reused_workers_are_verified_and_frozen_without_rewriting_source_paths(self):
        origin = self.reused_parallel_fixture()
        hb = FakeHarbor()
        result = self.call(hb)
        self.assertEqual(result['status'], 'completed', result['failures'])
        manifest = audit.read_json(Path(result['work_dir'])/'evidence/manifest.json')
        references = manifest['commits']['observation-03']['references']
        source = origin/'observations/sample-03/job/full.json'
        frozen = next(ref for ref in references if ref['original_path'] == str(source))
        self.assertEqual(Path(frozen['snapshot_path']).read_bytes(), source.read_bytes())
        self.assertEqual(audit.sha(frozen['snapshot_path']), frozen['sha256'])
        inputs = audit.load_inputs(self.ep)
        self.assertEqual(inputs['attribution']['parent_ids'],
                         [c['commit_id'] for c in inputs['observation_commits']])

    def test_reused_worker_forged_source_or_commit_is_rejected(self):
        self.reused_parallel_fixture()
        original = audit.read_json(self.diag/'observations/index.json')
        for field, value in [('reused_from', str(self.root)),
                             ('reused_commit_id', 'not-the-committed-worker')]:
            index = json.loads(json.dumps(original))
            index['workers'][0][field] = value
            self.rebind_parallel_index(index)
            hb = FakeHarbor()
            result = self.call(hb)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('source or commit identity', str(result['failures']))
            self.assertEqual(hb.calls, [])

    def test_reused_worker_cannot_escape_declared_origin_observations(self):
        self.reused_parallel_fixture()
        index = audit.read_json(self.diag/'observations/index.json')
        index['reuse_observations_from'] = str(self.root)
        for worker in index['workers']:
            worker['reused_from'] = str(self.root)
        self.rebind_parallel_index(index)
        hb = FakeHarbor()
        result = self.call(hb)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('full raw job under observations', str(result['failures']))
        self.assertEqual(hb.calls, [])

    def test_reused_worker_live_job_tamper_is_rejected(self):
        origin = self.reused_parallel_fixture()
        (origin/'observations/sample-03/job/full.json').write_text('changed live evidence')
        hb = FakeHarbor()
        result = self.call(hb)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('live evidence differs', str(result['failures']))
        self.assertEqual(hb.calls, [])

    def test_reuse_worker_metadata_without_index_source_is_rejected(self):
        self.reused_parallel_fixture()
        index = audit.read_json(self.diag/'observations/index.json')
        del index['reuse_observations_from']
        self.rebind_parallel_index(index)
        hb = FakeHarbor()
        result = self.call(hb)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('requires a declared reuse source', str(result['failures']))
        self.assertEqual(hb.calls, [])

    def test_compact_raw_output_tamper_rejected_before_reviewer(self):
        self.parallel_fixture(compact=True)
        (self.diag/'observations/sample-03/work/output.json').write_text('{"observations":[]}')
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('raw model output differs',str(result['failures']))
        self.assertEqual(hb.calls,[])

    def test_compact_raw_output_not_archived_rejected_before_reviewer(self):
        self.parallel_fixture(compact=True,bind_raw=False)
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('raw model output is not bound',str(result['failures']))
        self.assertEqual(hb.calls,[])

    def test_raw_attribution_and_design_tamper_rejected_before_reviewer(self):
        for stage in ('attribution','design'):
            with self.subTest(stage=stage):
                synthesis=audit.read_json(self.diag/'synthesis.json')
                records=self.diag/'records'
                ac=audit.read_stage_commit(records,synthesis['attribution_commit'])
                dc=audit.read_stage_commit(records,synthesis['design_commit'])
                source=ac if stage=='attribution' else dc
                raw=self.diag/(stage+'-model-output.json')
                audit.write_json(raw,{'raw_semantic_fields':'original'})
                commit=write_stage_commit(records,stage,source['payload'],
                    claims=ac['payload'] if stage=='design' else None,
                    parent_ids=source['parent_ids'],
                    references=[Path(ref['original_path']) for ref in source['references']]+[raw],
                    metadata={**source['metadata'],'raw_output_path':str(raw),
                              'raw_output_sha256':audit.sha(raw)})
                synthesis[stage+'_commit']=commit['commit_id']
                if stage=='attribution':
                    replacement=write_stage_commit(records,'design',dc['payload'],claims=ac['payload'],
                        parent_ids=[commit['commit_id']],metadata=dc['metadata'],
                        references=[Path(ref['original_path']) for ref in dc['references']])
                    synthesis['design_commit']=replacement['commit_id']
                audit.write_json(self.diag/'synthesis.json',synthesis)
                raw.write_text('{"raw_semantic_fields":"tampered"}')
                hb=FakeHarbor();result=self.call(hb)
                self.assertEqual(result['status'],'failed')
                self.assertIn('raw model output differs',str(result['failures']))
                self.assertEqual(hb.calls,[])
                # Restore exact committed bytes so the next stage tests its own guard.
                audit.write_json(raw,{'raw_semantic_fields':'original'})

    def test_raw_output_metadata_requires_complete_identity(self):
        for metadata in ({'raw_output_path':'/tmp/unknown.json'},
                         {'raw_output_sha256':'a'*64}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError,'both path and sha256'):
                audit._verify_raw_output({'metadata':metadata,'references':[]})

    def test_parallel_all_observations_audited_once_and_evidence_frozen(self):
        ids=self.parallel_fixture()
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'completed',result['failures'])
        self.assertEqual(result['expected_units']['observations'],ids)
        self.assertEqual([o['id'] for o in result['payload']['observations']],ids)
        frozen=audit.read_json(Path(result['work_dir'])/'evidence/manifest.json')
        self.assertIn('observations',frozen['snapshots'])
        self.assertIn('observation-03',frozen['commits'])
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['audit_commit'],result['audit_commit'])
        self.assertEqual(len(hb.calls),calls)

    def test_parallel_omitted_observation_is_not_accepted_or_repaired(self):
        self.parallel_fixture()
        class MissingObservation(FakeHarbor):
            def agent_task(self,*args,**kwargs):
                rc=super().agent_task(*args,**kwargs)
                path=Path(next(v.split('=',1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
                raw=audit.read_json(path)
                if 'observations' in raw:
                    raw['observations'].pop();audit.write_json(path,raw)
                return rc
        hb=MissingObservation();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['payload'])

    def test_parallel_raw_worker_tamper_rejected_before_reviewer(self):
        self.parallel_fixture()
        (self.diag/'observations/sample-03/job/full.json').write_text('changed')
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('raw job differs',str(result['failures']))
        self.assertEqual(hb.calls,[])

    def test_parallel_index_tamper_rejected_before_reviewer(self):
        self.parallel_fixture()
        path=self.diag/'observations/index.json';index=audit.read_json(path)
        index['workers'].pop();audit.write_json(path,index)
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('index is not bound',str(result['failures']))
        self.assertEqual(hb.calls,[])

    def test_missing_terminal_receipt_rejects_valid_output_without_probes_or_retry(self):
        class MissingReceipt(FakeHarbor):
            def agent_task(self,*args,**kwargs):
                rc=super().agent_task(*args,**kwargs)
                job=Path(args[3])
                (job/'result.json').unlink()
                return rc
        hb=MissingReceipt();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('model did not complete normally',str(result['failures']))
        self.assertTrue(review_output(result).is_file())
        self.assertIsNone(result['payload'])
        self.assertEqual(hb.probe_calls,[])
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['status'],'failed')
        self.assertEqual(len(hb.calls),calls)

    def test_timeout_keeps_harbor_exception_and_draft_without_metrics(self):
        class TimedOut(FakeHarbor):
            def agent_task(self,task,agent,model,job,timeout,mounts,**kwargs):
                self.calls.append({})
                path=Path(next(v.split('=',1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
                audit.write_json(path.parent/'audit-draft.json',{'completed_ids':['c1'],'pending_ids':['m1']})
                Path(job).mkdir()
                audit.write_json(Path(job)/'result.json',{'exception_info':{'exception_type':'AgentTimeoutError'},
                                                        'agent_execution':{'duration':2400}})
                return 0
        hb=TimedOut();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertIn('AgentTimeoutError',str(result['failures']))
        self.assertNotIn('FileNotFoundError',str(result['failures']))
        self.assertIsNone(result['payload'])
        self.assertEqual(hb.probe_calls,[])
        self.assertTrue((review_output(result).parent/'audit-draft.json').is_file())
        calls=len(hb.calls)
        self.call(hb);self.assertEqual(len(hb.calls),calls)

    def test_complete_probes_keep_advance_expectations_and_original_bytes(self):
        before=audit.tree_fingerprint(self.diag/'candidate')
        hb=FakeHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'completed',result['failures'])
        self.assertEqual([p['actual_accept'] for p in result['payload']['probes']],[True,False])
        self.assertEqual([p['expected_valid'] for p in result['payload']['probes']],[True,False])
        self.assertEqual(audit.tree_fingerprint(self.diag/'candidate'),before)
        self.assertTrue(hb.assert_frozen)
        self.assertEqual(hb.calls[0]['model'],'configured-judge')
        self.assertEqual(hb.calls[0]['kwargs']['agent_timeout_mult'],37*60/2400)
        self.assertNotIn('agent_kwargs',hb.calls[0]['kwargs'])
        self.assertFalse(result['provenance']['is_human_gold'])
        self.assertEqual(result['provenance']['synthesis_status'],'completed')
        self.assertIsNone(result['provenance']['capture_receipt'])
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['audit_commit'],result['audit_commit'])
        self.assertEqual(len(hb.calls),calls)
        self.assertEqual(len(hb.probe_calls),2)
        self.assertFalse(self.mount_mock.call_args.kwargs['with_docker'])

    def test_commit_failure_retains_raw_evidence_but_cannot_produce_accuracy(self):
        from stage_metrics import summarize_envelope
        hb=FakeHarbor()
        with patch.object(audit, 'write_stage_commit', side_effect=OSError('commit failed')):
            result=self.call(hb)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['payload'])
        self.assertEqual(len(hb.probe_calls), 2)
        self.assertTrue(review_output(result).is_file())
        self.assertTrue((Path(result['work_dir'])/'probes/correct_solution/execution.json').is_file())
        self.assertTrue(all(item['accuracy'] is None for item in summarize_envelope(result)['metrics'].values()))

    def test_original_mapping_label_failure_is_preserved_without_conversion(self):
        class OriginalLabels(FakeHarbor):
            def agent_task(self, *args, **kwargs):
                rc=super().agent_task(*args, **kwargs)
                path=Path(next(v.split('=',1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
                raw=audit.read_json(path)
                if 'mappings' in raw:
                    raw['mappings'][0].update(upstream_verdict='supported', implementation_verdict='partially_implemented')
                    audit.write_json(path, raw)
                return rc
        hb=OriginalLabels();result=self.call(hb)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(hb.probe_calls, [])
        self.assertEqual(audit.read_json(review_output(result))['mappings'][0]['upstream_verdict'], 'supported')
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['status'], 'failed')
        self.assertEqual(len(hb.calls), calls)

    def test_all_probe_workbooks_frozen_before_first_execution(self):
        class ChangesLaterSource(FakeHarbor):
            def run(self,args,job,timeout):
                rc=super().run(args,job,timeout)
                if len(self.probe_calls)==1:
                    task=Path(args[args.index('-p')+1])
                    planned=audit.read_json(task.parent/'expectation.json')
                    original=Path(planned['workbook']).parent/'incorrect_solution.xlsx'
                    original.write_bytes(b'correct')
                return rc
        hb=ChangesLaterSource();result=self.call(hb)
        self.assertEqual(result['status'],'completed',result['failures'])
        self.assertEqual([p['actual_accept'] for p in result['payload']['probes']],[True,False])
        run=Path(result['work_dir'])
        manifest=audit.read_json(run/'probe-inputs/manifest.json')
        negative=manifest['probes']['incorrect_solution']
        self.assertEqual(Path(negative['original_workbook']).read_bytes(),b'correct')
        self.assertEqual(Path(negative['snapshot_path']).read_bytes(),b'wrong')
        self.assertEqual(manifest['plan_sha256'],audit.sha(run/'frozen_probe_plan.json'))
        self.assertEqual(audit.read_json(run/'reviews/task/work/probe_plan.json'),audit.read_json(run/'frozen_probe_plan.json'))
        self.assertEqual(hb.probe_calls[1]['workbook'],negative['original_workbook'])

    def test_mutated_cached_verdict_rejected_against_immutable_commit(self):
        from stage_records import read_stage_commit
        hb=FakeHarbor();first=self.call(hb)
        cached=Path(first['work_dir'])/'audit.json'
        mutated=audit.read_json(cached)
        mutated['payload']['claims'][0]['verdict']='incorrect'
        calls=len(hb.calls)
        audit.write_json(cached,mutated)
        second=self.call(hb)
        self.assertEqual(second['status'],'failed')
        self.assertEqual(len(hb.calls),calls)
        self.assertEqual(read_stage_commit(self.diag/'records',first['audit_commit'])['payload']['claims'][0]['verdict'],'correct')

    def test_mutating_cache_hash_list_cannot_hide_tampered_evidence(self):
        hb=FakeHarbor();first=self.call(hb)
        cached=Path(first['work_dir'])/'audit.json'
        mutated=audit.read_json(cached)
        raw=review_output(first)
        raw.write_text('{}')
        mutated.setdefault('output_hashes',{})[str(raw)]=audit.sha(raw)
        audit.write_json(cached,mutated)
        calls=len(hb.calls)
        second=self.call(hb)
        self.assertEqual(second['status'],'failed')
        self.assertEqual(len(hb.calls),calls)

    def test_malformed_json_is_failed_preserved_and_never_retried(self):
        hb=FakeHarbor(malformed=True);r=self.call(hb)
        self.assertEqual(r['status'],'failed')
        self.assertEqual(review_output(r).read_text(),'{invalid json')
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['status'],'failed')
        self.assertEqual(len(hb.calls),calls)
        self.assertEqual(len(hb.probe_calls),0)

    def test_infrastructure_is_not_false_rejection(self):
        r=self.call(FakeHarbor(infra=True))
        self.assertEqual(r['status'],'completed',r['failures'])
        self.assertTrue(all(p['execution_status']=='infra_error' and p['actual_accept'] is None
                            for p in r['payload']['probes']))

    def test_mutated_frozen_candidate_fails_without_agent(self):
        (self.diag/'candidate/instruction.md').write_text('changed')
        hb=FakeHarbor();r=self.call(hb)
        self.assertEqual(r['status'],'failed');self.assertEqual(hb.calls,[])

    def test_tampered_cached_output_not_reused_or_retried(self):
        hb=FakeHarbor();r=self.call(hb)
        review_output(r).write_text('{}')
        calls=len(hb.calls)
        self.assertEqual(self.call(hb)['status'],'failed')
        self.assertEqual(len(hb.calls),calls)

    def test_missing_candidate_has_no_invented_verdict(self):
        audit.write_json(self.diag/'synthesis.json',{'status':'failed'})
        (self.diag/'candidate_manifest.json').unlink()
        hb=FakeHarbor();r=self.call(hb)
        self.assertEqual(r['status'],'missing');self.assertIsNone(r['payload'])
        self.assertEqual(hb.calls,[])

    def test_missing_audit_envelope_metrics_integration(self):
        from stage_metrics import summarize_envelope
        audit.write_json(self.diag/'synthesis.json',{'status':'failed'})
        (self.diag/'candidate_manifest.json').unlink()
        hb=FakeHarbor();envelope=self.call(hb)
        self.assertIsNone(envelope['expected_units'])
        metrics=summarize_envelope(envelope)
        self.assertEqual(metrics['status'],'missing')
        self.assertTrue(all(value['accuracy'] is None and value['denominator'] is None
                            for value in metrics['metrics'].values()))
        self.assertEqual(hb.calls,[])

    def test_completed_audit_preserves_failed_synthesis_capture_provenance(self):
        receipt=self.diag/'capture-receipt.json'
        audit.write_json(receipt,{'author_receipt_missing':True})
        synthesis=audit.read_json(self.diag/'synthesis.json')
        synthesis.update(status='failed',capture_receipt=str(receipt))
        audit.write_json(self.diag/'synthesis.json',synthesis)
        result=self.call(FakeHarbor())
        self.assertEqual(result['status'],'completed',result['failures'])
        self.assertEqual(result['provenance']['synthesis_status'],'failed')
        self.assertEqual(result['provenance']['capture_receipt'],str(receipt))
        self.assertEqual(audit.read_json(self.diag/'synthesis.json')['status'],'failed')

    def test_schema_failure_is_not_repaired(self):
        class BadSchemaHarbor(FakeHarbor):
            def agent_task(self,*args,**kwargs):
                rc=super().agent_task(*args,**kwargs)
                path=Path(next(v.split('=',1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
                raw=audit.read_json(path)
                if 'claims' in raw:
                    raw['claims']=[];audit.write_json(path,raw)
                return rc
        hb=BadSchemaHarbor();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(audit.read_json(review_output(result,'claim-c1'))['claims'],[])

    def test_reviewer_cannot_predeclare_actual_verifier_result(self):
        class PrematureResult(FakeHarbor):
            def agent_task(self,*args,**kwargs):
                rc=super().agent_task(*args,**kwargs)
                path=Path(next(v.split('=',1)[1] for v in kwargs['agent_env'] if v.startswith('ADB_RESULT=')))
                raw=audit.read_json(path)
                raw['probes'][0].update(actual_accept=True,execution_status='completed')
                audit.write_json(path,raw)
                return rc
        hb=PrematureResult();result=self.call(hb)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(hb.probe_calls,[])

    def test_actual_accept_uses_existing_threshold_and_no_reward_is_unknown(self):
        job=self.root/'trial';job.mkdir()
        audit.write_json(job/'result.json',{'exception_info':None,'verifier_result':{'rewards':{'reward':0.75}}})
        self.assertEqual(audit.parse_probe_run(job,0,0.5)[:2],(True,'completed'))
        self.assertEqual(audit.parse_probe_run(job,0,1.0)[:2],(False,'completed'))
        audit.write_json(job/'result.json',{'exception_info':None,'verifier_result':{'rewards':{}}})
        self.assertEqual(audit.parse_probe_run(job,0,1.0)[:2],(None,'infra_error'))


if __name__=='__main__':
    unittest.main()
