"""Offline integration of local analysis, receipts, commits and the CLI boundary."""
import copy
import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import diagnostics
import local_analysis
import parallel_attribution as pa
import run_episode
import run_synth
import staged_synth as staged
from stage_records import read_stage_commit


class FakeLocal:
    def __init__(self, count):
        self.calls = []
        self.prompts = {}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(count)

    def __call__(self, directory, prompt, ro, seconds, model, base_url, api_key, env):
        sample = env.get('ADB_SAMPLE_ID')
        with self.lock:
            self.calls.append(sample)
            self.prompts[sample] = Path(prompt).read_text()
        if sample is not None:
            self.barrier.wait(timeout=10)
            payload = {'observations': [{'finding': 'Observed grouped output', 'evidence': ['trace:all']}]}
            manifest = json.loads(Path(env['ADB_INPUT_MANIFEST']).read_text())
            assert manifest
            assert all(Path(p).exists() for p in ro)
        else:
            index = json.loads(Path(env['ADB_OBSERVATIONS_INDEX']).read_text())
            assert index['status'] == 'completed'
            assert all(Path(w['commit_path']).is_file() for w in index['workers'])
            payload = {'claims': [{'kind': 'capability', 'finding': 'Grouped output under observed conditions',
                                  'conditions': ['Observed records'], 'supports': index['observation_ids'],
                                  'counterevidence': [], 'evidence_strength': 'replay_verified'}]}
        Path(env['ADB_RESULT']).write_text(json.dumps(payload))
        job = directory/'job'
        job.mkdir(parents=True)
        transcript, stderr = job/'claude-code.txt', job/'stderr.txt'
        transcript.write_text(json.dumps({'type': 'result', 'is_error': False})+'\n')
        stderr.write_text('')
        record = {'runtime': 'local-claude', 'started_at': '2026-09-23T00:00:00+00:00',
                  'finished_at': '2026-09-23T00:00:01+00:00', 'active_seconds': 1.0,
                  'status': 'completed', 'returncode': 0, 'output': str(env['ADB_RESULT']),
                  'transcript': local_analysis._identity(transcript), 'stderr': local_analysis._identity(stderr)}
        (job/'local-result.json').write_text(json.dumps(record))
        return local_analysis.read_local_result(job)


class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root/'bench/source'
        self.source.mkdir(parents=True)
        (self.source/'instruction.md').write_text('Group source records completely.')
        self.bundle = self.root/'prep/bundle'
        self.bundle.mkdir(parents=True)
        self.ids = [3, 0, 2, 1] + list(range(4, 16))
        (self.bundle/'summary.json').write_text(json.dumps({'attempt_ids': self.ids, 'attempts': 16}))
        for i in self.ids:
            (self.bundle/f'attempt-{i}.md').write_text(f'Full trajectory {i}\n' * 100)
        self.ep = self.root/'ep01'
        self.cfg = {'diagnostics': {'enabled': True, 'attribution_mode': 'parallel', 'parallel_attribution': {
            'runtime': 'local-claude', 'max_concurrent': 16, 'worker_budget_min': 1,
            'aggregate_budget_min': 2, 'design_budget_min': 3}},
            'researcher': {'agent': 'claude-code', 'model': 'author', 'budget_min': 45,
                           'max_concurrent_docker': 1, 'agent_setup_timeout_mult': 1},
            'customer': {'model': 'target', 'attempts': 16},
            'judge': {'agent': 'claude-code', 'model': 'reviewer', 'budget_min': 4},
            'score': {'n_concurrent': 1, 'setup_timeout_mult': 1, 'band_lo': 0.1, 'band_hi': 0.9},
            'gateway': {'base_url': 'http://127.0.0.1:12345', 'api_key': 'fake'}}
        self.hb = SimpleNamespace(**self.cfg['gateway'])  # deliberately no bin or agent_task
        self.fake = FakeLocal(16)
        self.real_mount_builder = pa.mounts.build
        for context in (patch.object(local_analysis, 'run_local', side_effect=self.fake),
                        patch.object(pa.mounts, 'build', side_effect=AssertionError('no local mounts')),
                        patch.object(staged, 'rollout_bundle_path', return_value=self.bundle)):
            context.start()
            self.addCleanup(context.stop)

    def analyze(self):
        return run_synth.synthesize(self.hb, self.cfg, 'bench', self.source, self.ep, 1, analysis_only=True)

    def test_sixteen_local_workers_aggregate_and_resume_without_model_replay(self):
        with patch.object(staged, '_run_design', side_effect=AssertionError('no author stage')), \
             patch.dict(os.environ, {'ADB_CC_BINARY': '/missing/old-docker-binary'}):
            result = self.analyze()
            self.assertEqual(result['status'], 'analysis_completed')
            self.assertEqual(len(self.fake.calls), 17)
            self.assertIsNone(self.fake.calls[-1])
            diag = self.ep/'diagnostics'
            index = json.loads((diag/'observations/index.json').read_text())
            self.assertEqual(index['sample_order'], self.ids)
            self.assertEqual(index['status'], 'completed')
            aggregate = read_stage_commit(diag/'records', result['attribution_commit'])
            self.assertEqual(aggregate['parent_ids'], [w['commit_id'] for w in index['workers']])
            for worker in index['workers']:
                raw = json.loads(Path(worker['raw_output_path']).read_text())
                derived = json.loads(Path(worker['output_path']).read_text())
                self.assertNotIn('sample_id', raw)
                self.assertEqual(derived['sample_id'], worker['sample_id'])
                self.assertEqual(raw['observations'][0]['finding'], derived['observations'][0]['statement'])
                self.assertEqual(worker['terminal']['runtime'], 'local-claude')
                self.assertEqual(worker['terminal']['agent_terminal_status'], 'completed')
                read_stage_commit(diag/'records', worker['commit_id'])
            self.assertEqual(self.analyze(), result)
            self.assertEqual(len(self.fake.calls), 17)
            self.assertFalse((diag/'design.json').exists())
            self.assertFalse((diag/'audit.json').exists())

    def test_completed_local_receipt_recovery_does_not_use_harbor(self):
        self.analyze()
        directory = self.ep/'diagnostics/observations/sample-00'
        old = json.loads((directory/'finished.json').read_text())
        with patch.object(pa, 'read_terminal', side_effect=AssertionError('not a Harbor result')):
            recovered = pa._run(self.hb, self.cfg, directory, 'unused', {}, [], 60,
                                self.ep/'diagnostics', 'observation-0', {})
        self.assertEqual(recovered, old)
        self.assertEqual(len(self.fake.calls), 17)
        (directory/'job/claude-code.txt').write_text('changed transcript')
        with self.assertRaisesRegex(ValueError, 'transcript evidence changed'):
            pa._run(self.hb, self.cfg, directory, 'unused', {}, [], 60,
                    self.ep/'diagnostics', 'observation-0', {})
        self.assertEqual(len(self.fake.calls), 17)

    def test_cli_analysis_only_skips_harbor_design_scoring_and_audit(self):
        argv = ['run_episode', '--benchmark', 'bench', '--task', 'source', '--run-id', 'local-test', '--analysis-only']
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            for name, value in [('load_config', self.cfg), ('load_meta', {}),
                                ('sampled_tasks', [self.source]), ('rollout_bundle_path', self.bundle),
                                ('rubric_path', self.root/'prep/rubrics/source/rubric.json'),
                                ('episode_dir', self.ep)]:
                stack.enter_context(patch.object(run_episode, name, return_value=value))
            stack.enter_context(patch.object(run_episode, 'ROOT', self.root))
            for name in ('make_harbor', 'resolve_harbor_bin', 'score_episode',
                         'run_diagnostic_audit', 'write_diagnostic_run_summary'):
                stack.enter_context(patch.object(run_episode, name, side_effect=AssertionError(name+' forbidden')))
            stack.enter_context(patch.object(staged, '_run_design', side_effect=AssertionError('design forbidden')))
            self.assertEqual(run_episode.main(), 0)
        runtime = json.loads((self.ep/'runtime-status.json').read_text())
        self.assertEqual(runtime['status'], 'completed')
        self.assertEqual(runtime['runtime'], 'local-claude')
        self.assertEqual(len(self.fake.calls), 17)
        self.assertFalse((self.ep/'diagnostics/design.json').exists())
        self.assertFalse((self.ep/'diagnostics/audit.json').exists())

    def test_local_analysis_can_continue_into_harbor_design_and_keep_runtime_identity(self):
        from test_parallel_integration import FullAuthorHarbor
        author = FullAuthorHarbor(1)
        author.api_key = 'fake'
        author.compact = True
        with patch.object(pa.mounts, 'build', side_effect=self.real_mount_builder), \
             patch.object(staged, 'benchmark_dir', return_value=self.source.parent):
            rec = run_synth.synthesize(author, self.cfg, 'bench', self.source, self.ep, 1)
            self.assertEqual(rec['stage_runtime'], {'attribution': 'local-claude', 'design': 'harbor'})
            self.assertEqual(rec['stage_returncode'], {'attribution': 0, 'design': 0})
            self.assertIsNone(rec['stage_harbor_rc']['attribution'])
            self.assertEqual(len(author.calls), 1)
            self.assertEqual(author.calls[0]['stage'], 'design')
            self.assertEqual(len(self.fake.calls), 17)
            self.assertEqual(run_synth.synthesize(author, self.cfg, 'bench', self.source, self.ep, 1), rec)
            self.assertEqual(len(author.calls), 1)
            self.assertEqual(len(self.fake.calls), 17)

    def test_preloaded_context_keeps_complete_text_and_isolates_each_sample(self):
        original = '完整原题：按成员分组。\n"quoted"\n' + 'source instruction ' * 1000
        system = 'Original system prompt: preserve all source records.\n' * 500
        (self.source/'instruction.md').write_text(original)
        (self.source/'system_prompt.md').write_text(system)
        poison = 'HIDDEN_PRIOR_JUDGMENT_MUST_NOT_REACH_WORKERS'
        (self.source/'_meta.json').write_text(json.dumps({'judgment': poison}))
        (self.source/'solution').mkdir()
        (self.source/'solution/solve.py').write_text('# '+poison)
        texts = {}
        for sample in self.ids:
            texts[sample] = (f'<<SAMPLE_{sample:02d}_ONLY>>\n' + '中文 "quoted" \\ backslash\n') * 6000
            (self.bundle/f'attempt-{sample}.md').write_text(texts[sample])
        self.analyze()
        marker = 'You can use these contents directly; file tools remain available for checks.\n'
        diag = self.ep/'diagnostics'
        for sample in self.ids:
            prompt = self.fake.prompts[sample]
            context = json.loads(prompt.split(marker, 1)[1])
            inputs = diag/'observations'/f'sample-{sample:02d}'/'inputs'
            paths = [inputs/'task/instruction.md', inputs/'task/system_prompt.md',
                     inputs/'evidence_index.json', inputs/'attempt.md']
            self.assertEqual([item['path'] for item in context], [str(path) for path in paths])
            self.assertEqual([item['text'] for item in context], [path.read_text() for path in paths])
            self.assertEqual(context[0]['text'], original)
            self.assertEqual(context[1]['text'], system)
            self.assertEqual(context[-1]['text'], texts[sample])
            self.assertNotIn(poison, prompt)
            for other in set(self.ids)-{sample}:
                self.assertNotIn(f'<<SAMPLE_{other:02d}_ONLY>>', prompt)
            started = json.loads((inputs.parent/'started.json').read_text())
            self.assertEqual(started['prompt_sha256'], hashlib.sha256(prompt.encode()).hexdigest())
        aggregate = json.loads(self.fake.prompts[None].split(marker, 1)[1])
        index = json.loads((diag/'observations/index.json').read_text())
        self.assertEqual(len(aggregate), 17)
        self.assertEqual(aggregate[0]['text'], original)
        self.assertEqual([item['path'] for item in aggregate[1:]], [w['output_path'] for w in index['workers']])
        for item, worker in zip(aggregate[1:], index['workers']):
            self.assertEqual(item['text'], Path(worker['output_path']).read_text())
            self.assertEqual(json.loads(item['text'])['sample_id'], worker['sample_id'])
        self.assertEqual(len(self.fake.calls), 17)

    def test_unknown_runtime_rejected_before_any_invocation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['diagnostics']['parallel_attribution']['runtime'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'runtime'):
            diagnostics.parallel_config(cfg)
        self.assertEqual(self.fake.calls, [])


if __name__ == '__main__':
    unittest.main()
