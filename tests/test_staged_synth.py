import json
from datetime import datetime, timedelta, timezone
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import staged_synth as st
import run_synth
import diagnostics
from test_parallel_attribution import FakeHarbor as ObservationHarbor


class FakeHarbor(ObservationHarbor):
    """Real parallel attribution (workers + aggregate), scripted design call."""
    def __init__(self, n=2, *, rc=0, invalid_design=False, missing_duration=False,
                 design_seconds=200.5, **kwargs):
        super().__init__(n, **kwargs)
        self.rc = rc
        self.invalid_design = invalid_design
        self.missing_duration = missing_duration
        self.design_seconds = design_seconds
        self.design_calls = []

    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        env = dict(x.split('=', 1) for x in kwargs['agent_env'])
        if 'ADB_DESIGN_JSON' not in env:
            return super().agent_task(task, agent, model, job, timeout, mounts, **kwargs)
        self.design_calls.append(dict(task=task, kwargs=kwargs, timeout=timeout, job=job))
        attribution = Path(env['ADB_ATTRIBUTION_JSON'])
        assert (attribution.parent / 'attribution_commit.json').is_file()
        payload = {'mappings': [{'id': 'm1', 'claim_id': 'c1', 'role': 'target',
                                 'feature': 'Exercise the grounded capability',
                                 'check_plan': 'Independent positive and negative workbook',
                                 'preserves': ['Known input semantics'], 'simplifies': ['Output size']}]}
        Path(env['ADB_DESIGN_JSON']).write_text('bad json' if self.invalid_design else json.dumps(payload))
        out = Path(env['ADB_OUT']) / 'new-task'
        out.mkdir()
        (out / 'task.toml').write_text('version = "1.0"\n')
        (out / 'instruction.md').write_text('task')
        (out / '.hidden').write_bytes(b'x' * 200000)
        start = datetime(2026, 9, 22, tzinfo=timezone.utc)
        execution = {'started_at': start.isoformat(),
                     'finished_at': None if self.missing_duration else
                     (start + timedelta(seconds=self.design_seconds)).isoformat()}
        trial = job / 'run/trial/result.json'
        trial.parent.mkdir(parents=True)
        # Days of setup before agent execution must not count as active time.
        trial.write_text(json.dumps({'agent_execution': execution, 'exception_info': None,
            'started_at': (start - timedelta(days=2)).isoformat(),
            'agent_setup': {'started_at': (start - timedelta(days=2)).isoformat(), 'finished_at': start.isoformat()}}))
        (job / 'run/result.json').write_text('{"stats": {}}')
        return self.rc


class StagedTests(unittest.TestCase):
    DESIGN_SECONDS = 3 * 60

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'bench/source'
        self.source.mkdir(parents=True)
        (self.source / 'instruction.md').write_text('original')
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        (self.bundle / 'summary.json').write_text(json.dumps({'attempt_ids': [1, 0], 'attempts': 2}))
        for i in (1, 0):
            (self.bundle / f'attempt-{i}.md').write_bytes(f'FULL EVIDENCE {i}\n'.encode() * 20000)
        self.cfg = {'diagnostics': {'enabled': True, 'attribution_mode': 'parallel', 'parallel_attribution': {
                        'max_concurrent': 4, 'worker_budget_min': 1, 'aggregate_budget_min': 2,
                        'design_budget_min': 3}}, 'customer': {'model': 'target', 'attempts': 2},
                    'score': {'band_lo': 0.1, 'band_hi': 0.9}, 'gateway': {'base_url': 'http://fake'},
                    'researcher': {'agent': 'claude-code', 'model': 'author', 'budget_min': 45, 'max_concurrent_docker': 1}}
        self.ep = self.root / 'episode'
        self.diag = self.ep / 'diagnostics'
        self.patches = [patch.object(st, 'benchmark_dir', return_value=self.source.parent),
                        patch.object(st, 'rollout_bundle_path', return_value=self.bundle)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def synth(self, hb, cfg=None):
        return st.synthesize_staged(hb, cfg or self.cfg, 'bench', self.source, self.ep, 1)

    def crash_on_synthesis_receipt(self, hb):
        original = st.save_new
        def crash(path, value):
            if Path(path).name == 'synthesis.json':
                raise KeyboardInterrupt('simulated process loss after synth.json')
            return original(path, value)
        with patch.object(st, 'save_new', side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):
                self.synth(hb)

    def test_design_after_committed_attribution_full_bytes_and_reuse(self):
        hb = FakeHarbor()
        rec = self.synth(hb)
        self.assertEqual(rec['n_delivered'], 1)
        self.assertEqual(rec['attribution_mode'], 'parallel')
        self.assertEqual(len(hb.design_calls), 1)
        calls = len(hb.calls)
        self.assertEqual((self.diag / 'inputs/rollouts/attempt-0.md').read_bytes(),
                         (self.bundle / 'attempt-0.md').read_bytes())
        self.assertEqual((self.diag / 'candidate/.hidden').stat().st_size, 200000)
        events = [json.loads(x) for x in (self.diag / 'events.jsonl').read_text().splitlines()]
        kinds = [e['event'] for e in events]
        design_started = next(i for i, e in enumerate(events) if e['event'] == 'stage_started' and e.get('stage') == 'design')
        self.assertLess(kinds.index('attribution_committed'), design_started)
        self.assertEqual(self.synth(hb), rec)
        self.assertEqual((len(hb.calls), len(hb.design_calls)), (calls, 1))
        changed = dict(self.cfg, researcher=dict(self.cfg['researcher'], model='other'))
        with self.assertRaises(ValueError): self.synth(hb, changed)
        (self.ep / 'work/out/new-task/instruction.md').write_text('mutated')
        with self.assertRaises(ValueError): self.synth(hb)

    def test_bad_aggregate_stops_before_design_no_retry(self):
        hb = FakeHarbor(bad_aggregate=True)
        with self.assertRaises(ValueError): self.synth(hb)
        self.assertEqual(hb.design_calls, [])
        calls = len(hb.calls)
        self.assertEqual(st.load(self.diag / 'synthesis.json')['status'], 'failed')
        with self.assertRaises(RuntimeError): self.synth(hb)
        self.assertEqual(len(hb.calls), calls)

    def test_finished_authors_recover_local_finalization_without_model_retry(self):
        hb = FakeHarbor()
        self.crash_on_synthesis_receipt(hb)
        calls = len(hb.calls)
        recovered = self.synth(hb)
        self.assertEqual(recovered['n_delivered'], 1)
        self.assertEqual((len(hb.calls), len(hb.design_calls)), (calls, 1))

    def test_design_budget_uses_only_trial_active_seconds(self):
        hb = FakeHarbor()
        # Orchestration wall time must not consume the design allocation.
        with patch.object(st.time, 'time', side_effect=range(100000, 2000000, 10000)):
            rec = self.synth(hb)
        call = hb.design_calls[0]
        self.assertEqual(call['timeout'], self.DESIGN_SECONDS * 1.5 + 900)
        self.assertEqual(call['kwargs']['agent_timeout_mult'], self.DESIGN_SECONDS / 1800)
        self.assertEqual(rec['agent_execution_seconds'], {'aggregate': 10.0, 'design': 200.5})
        self.assertEqual(rec['budget_mode'], st.BUDGET_MODE)
        self.assertNotIn('deadline', st.load(self.diag / 'started.json'))
        state = st.load(self.diag / 'synthesis.json')
        commit = st.read_stage_commit(self.diag / 'records', state['design_commit'])
        evidence = commit['metadata']['stage_execution']
        self.assertEqual(evidence['harbor_rc'], 0)
        self.assertEqual(evidence['agent_execution']['status'], 'completed')
        self.assertTrue(Path(evidence['agent_execution']['source']['path']).is_file())
        self.assertEqual(commit['metadata']['budget_mode'], st.BUDGET_MODE)

    def test_nonzero_harbor_rc_does_not_discard_valid_structured_delivery(self):
        hb = FakeHarbor(rc=124)
        rec = self.synth(hb)
        self.assertEqual(rec['n_delivered'], 1)
        self.assertEqual(rec['harbor_rc'], 124)
        self.assertEqual(rec['stage_harbor_rc']['design'], 124)
        state = st.load(self.diag / 'synthesis.json')
        commit = st.read_stage_commit(self.diag / 'records', state['design_commit'])
        self.assertEqual(commit['metadata']['stage_execution']['harbor_rc'], 124)
        calls = len(hb.calls)
        self.assertEqual(self.synth(hb), rec)
        self.assertEqual(len(hb.calls), calls)

    def test_unknown_design_duration_retains_delivery_without_completed_claim(self):
        hb = FakeHarbor(missing_duration=True)
        with self.assertRaisesRegex(RuntimeError, 'duration unknown'):
            self.synth(hb)
        self.assertEqual(len(hb.design_calls), 1)
        self.assertEqual(st.load(self.diag / 'synthesis.json')['status'], 'failed')
        self.assertIsNone(st.load(self.diag / 'design/finished.json')['agent_execution']['active_seconds'])
        self.assertTrue((self.ep / 'work/design.json').is_file())
        self.assertTrue((self.ep / 'work/out/new-task/instruction.md').is_file())
        self.assertFalse((self.ep / 'synth.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'failed previously'):
            self.synth(hb)
        self.assertEqual(len(hb.design_calls), 1)

    def test_nonzero_rc_with_incomplete_output_remains_failed(self):
        hb = FakeHarbor(rc=124, invalid_design=True)
        with self.assertRaises(ValueError):
            self.synth(hb)
        self.assertEqual(st.load(self.diag / 'synthesis.json')['status'], 'failed')
        self.assertEqual(st.load(self.diag / 'design/finished.json')['harbor_rc'], 124)
        self.assertEqual(len(hb.design_calls), 1)

    def test_old_failed_state_is_preserved_without_migration(self):
        hb = FakeHarbor()
        state = self.diag / 'synthesis.json'
        state.parent.mkdir(parents=True)
        original = b'{"status":"failed","error":"old wall deadline"}'
        state.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, 'failed previously'):
            self.synth(hb)
        self.assertEqual(state.read_bytes(), original)
        self.assertEqual(hb.calls, [])

    def test_old_unfinished_budget_mode_is_not_automatically_migrated(self):
        hb = FakeHarbor()
        self.diag.mkdir(parents=True)
        original = {'started_at': 1, 'deadline': 2,
                    'config_sha256': st.hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest(),
                    'benchmark': 'bench', 'original_task': self.source.name,
                    'source_path': str(self.source.resolve()), 'episode': 1}
        st.save_new(self.diag / 'started.json', original)
        before = (self.diag / 'started.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'budget mode differs'):
            self.synth(hb)
        self.assertEqual((self.diag / 'started.json').read_bytes(), before)
        self.assertFalse((self.diag / 'synthesis.json').exists())
        self.assertEqual(hb.calls, [])

    def test_changed_duration_source_rejected_on_local_finalization_recovery(self):
        hb = FakeHarbor()
        self.crash_on_synthesis_receipt(hb)
        calls = len(hb.calls)
        (hb.design_calls[0]['job'] / 'run/trial/result.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.synth(hb)
        self.assertEqual(len(hb.calls), calls)
        self.assertEqual(st.load(self.diag / 'synthesis.json')['status'], 'failed')

    def test_duration_requires_unique_trial_and_valid_execution_timestamps(self):
        job = self.root/'job'; job.mkdir()
        self.assertEqual(st.agent_execution_evidence(job)['status'], 'unknown')
        trial = job/'result.json'
        for execution in (None, {}, {'started_at':'bad','finished_at':'also-bad'},
                          {'started_at':'2026-09-22T00:01:00Z','finished_at':'2026-09-22T00:00:00Z'}):
            trial.write_text(json.dumps({'agent_execution':execution}))
            evidence = st.agent_execution_evidence(job)
            self.assertEqual(evidence['status'],'unknown')
            self.assertIsNone(evidence['active_seconds'])
        trial.write_text(json.dumps({'agent_execution':{'started_at':'2026-09-22T00:00:00Z','finished_at':'2026-09-22T00:01:00Z'}}))
        self.assertEqual(st.agent_execution_evidence(job)['active_seconds'],60)
        second = job/'second/result.json';second.parent.mkdir();second.write_bytes(trial.read_bytes())
        self.assertEqual(st.agent_execution_evidence(job)['status'],'unknown')

    def test_existing_baseline_refused(self):
        self.ep.mkdir(); (self.ep/'synth.json').write_text('{}')
        with self.assertRaises(ValueError): self.synth(FakeHarbor())

    def run_design(self, hb, allocated_seconds):
        return st._run_design(hb, self.cfg, 'bench', self.source, self.bundle, self.ep, self.diag,
                              allocated_seconds, self.diag/'attribution.json')

    def test_design_budget_expired_no_call(self):
        self.diag.mkdir(parents=True)
        hb = FakeHarbor()
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'): self.run_design(hb, 0)
        self.assertEqual(hb.calls, [])

    def test_unfinished_design_not_restarted(self):
        (self.diag/'design').mkdir(parents=True)
        (self.diag/'design/started.json').write_text('{}')
        hb = FakeHarbor()
        with self.assertRaisesRegex(RuntimeError, 'already started'): self.run_design(hb, 1e15)
        self.assertEqual(hb.calls, [])

    def test_default_dispatch_and_strict_flag(self):
        self.assertFalse(diagnostics.enabled({}))
        for bad in (True,{'enabled':'false'},{'enabled':1},{'retry':True}):
            with self.assertRaises(ValueError): diagnostics.enabled({'diagnostics':bad})
        with patch.object(run_synth,'_synthesize_baseline',return_value={'baseline':True}) as baseline:
            self.assertEqual(run_synth.synthesize(None,{},'bench',self.source,self.root/'ep',1),{'baseline':True})
            baseline.assert_called_once()

    def test_design_prompt_rendered_into_copy_with_disabled_execution_tools(self):
        template = st.ROOT/'agents/synth-design-parallel/instruction.md'
        before = (template.read_bytes(), (run_synth.PROMPT/'instruction.md').read_bytes())
        self.synth(FakeHarbor())
        self.assertEqual((template.read_bytes(), (run_synth.PROMPT/'instruction.md').read_bytes()), before)
        self.assertFalse((run_synth.PROMPT/'staged-instruction.md').exists())
        rendered = self.diag/'design/task'
        self.assertNotRegex((rendered/'instruction.md').read_text(), r'__[A-Z0-9_]+__')
        for name in ('adb-oracle', 'adb-testrun', 'harbor'):
            self.assertIn('Candidate execution is unavailable', (rendered/'environment/bin'/name).read_text())

    def test_error_library_ingests_aggregate_and_records_targeted_class(self):
        import error_library
        cfg = json.loads(json.dumps(self.cfg)); cfg['diagnostics']['parallel_attribution']['error_library'] = True
        library = self.root / 'prep' / error_library.NAME
        with patch.object(error_library, 'library_path', return_value=library):
            rec = self.synth(FakeHarbor(), cfg)
        self.assertEqual(rec['n_delivered'], 1)
        data = json.loads(library.read_text())
        state = st.load(self.diag / 'synthesis.json')
        self.assertEqual(data['ingested_commits'], [state['attribution_commit']])
        self.assertEqual(data['classes'], [])   # the fixture aggregate holds only a capability claim
        self.assertIn(str(library), (self.diag / 'design/task/instruction.md').read_text())

if __name__ == '__main__': unittest.main()
