"""Candidate expansion reuses evidence without rerunning analysis or leaking old tasks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import candidate_jobs as jobs
import staged_synth as staged
import stage_audit
from test_parallel_integration import FullAuthorHarbor, ObservationHarbor


class GoalHarbor(FullAuthorHarbor):
    wrong_goal = False

    def agent_task(self, task, agent, model, job, timeout, mounts, **kwargs):
        env = dict(item.split('=', 1) for item in kwargs['agent_env'])
        index = json.loads(Path(env['ADB_OBSERVATIONS_INDEX']).read_text()) if 'ADB_OBSERVATIONS_INDEX' in env else {}
        # The generic integration fixture assumes newly executed worker job mounts.
        runner = ObservationHarbor.agent_task if index.get('reuse_observations_from') else FullAuthorHarbor.agent_task
        result = runner(self, task, agent, model, job, timeout, mounts, **kwargs)
        if 'ADB_OBSERVATIONS_INDEX' in env and 'ADB_DESIGN_JSON' not in env:
            path = Path(env['ADB_RESULT'])
            payload = json.loads(path.read_text())
            payload['training_goals'] = [dict(id=f'g{i}', claim_ids=['c1'], ability=f'ability {i}',
                rationale='supported by observations', success_signal='complete correct grouping',
                difficulty_controls=['reduce unrelated output size']) for i in (1, 2, 3)]
            path.write_text(json.dumps(payload))
        if 'ADB_DESIGN_JSON' in env:
            path = Path(env['ADB_DESIGN_JSON'])
            payload = json.loads(path.read_text())
            job_path = Path(env['ADB_ATTRIBUTION_JSON']).parent / 'candidate_job.json'
            chosen = json.loads(job_path.read_text())['goal_id'] if job_path.exists() else 'g1'
            payload.update(primary_goal_id='g2' if self.wrong_goal else chosen,
                           selection_reason='Exercise the assigned ability with controlled output size')
            path.write_text(json.dumps(payload))
        return result


class CandidateJobsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root / 'bench/source'; self.source.mkdir(parents=True)
        (self.source / 'instruction.md').write_text('Complete the original task.')
        self.bundle = self.root / 'rollouts'; self.bundle.mkdir()
        (self.bundle / 'summary.json').write_text(json.dumps({'attempt_ids': [0], 'attempts': 1}))
        (self.bundle / 'attempt-0.md').write_text('complete trajectory\n' * 10)
        self.cfg = {'diagnostics': {'enabled': True, 'attribution_mode': 'parallel', 'parallel_attribution': {
                    'max_concurrent': 1, 'worker_budget_min': 1, 'aggregate_budget_min': 2, 'design_budget_min': 3}},
                    'customer': {'model': 'target', 'attempts': 1}, 'score': {'band_lo': 0.1, 'band_hi': 0.9},
                    'gateway': {'base_url': 'http://fake'},
                    'researcher': {'agent': 'claude-code', 'model': 'author', 'budget_min': 45, 'max_concurrent_docker': 1}}
        for name, result in [('benchmark_dir', self.source.parent), ('rollout_bundle_path', self.bundle)]:
            p = patch.object(staged, name, return_value=result); p.start(); self.addCleanup(p.stop)
        self.analysis = self.root / 'analysis'
        self.analysis_hb = GoalHarbor(1)
        staged.synthesize_staged(self.analysis_hb, self.cfg, 'bench', self.source, self.analysis, 1,
                                 analysis_only=True)
        self.assertEqual(len(self.analysis_hb.calls), 2)
        self.cfg['training_pipeline'] = {'candidate_generation': {
            'difficulties': [{'name': 'focused', 'instruction': 'Reduce unrelated output size'},
                             {'name': 'expanded', 'instruction': 'Add repeated records while preserving the target'}],
            'copies_per_goal': 2}, 'review': 'llm'}
        self.sources = [{'original': 'source', 'analysis_source': str(self.analysis / 'diagnostics')}]

    def joblist(self, **kwargs):
        return jobs.prepare_joblist(self.cfg, self.sources, self.root / 'joblist.json', **kwargs)

    def test_goal_difficulty_variant_order_and_immutable_replay(self):
        result = self.joblist()
        self.assertEqual(len(result['jobs']), 12)
        expected = [(g, d, i) for g in ('g1', 'g2', 'g3') for d in ('focused', 'expanded') for i in (0, 1)]
        self.assertEqual([(j['goal_id'], j['difficulty']['name'], j['variant_index']) for j in result['jobs']], expected)
        self.assertEqual(len({job['id'] for job in result['jobs']}), 12)
        original = (self.root / 'joblist.json').read_bytes()
        self.assertEqual(self.joblist(), result)
        self.assertEqual((self.root / 'joblist.json').read_bytes(), original)
        self.assertFalse((self.analysis / 'diagnostics/synthesis.json').exists())
        self.assertFalse((self.analysis / 'diagnostics/design.json').exists())

    def test_explicit_goal_filter_is_not_a_fixed_two_goals(self):
        result = self.joblist(goal_ids=['g3'])
        self.assertEqual(len(result['jobs']), 4)
        self.assertTrue(all(job['goal_id'] == 'g3' for job in result['jobs']))
        with self.assertRaises(ValueError):
            self.joblist(goal_ids=['g4'])

    def test_one_job_only_authors_and_keeps_same_analysis_payload(self):
        job = self.joblist()['jobs'][0]
        # Deliberate old artifacts must not become mounted author inputs.
        (self.analysis / 'work/out/old-candidate').mkdir(parents=True)
        (self.analysis / 'work/out/old-candidate/secret').write_text('prior candidate')
        (self.analysis / 'diagnostics/audit.json').write_text('prior audit')
        hb = GoalHarbor(1)
        ep = self.root / 'candidate-1'
        rec = jobs.run_candidate(hb, self.cfg, 'bench', self.source, ep, 1, job)
        self.assertEqual([call['stage'] for call in hb.calls], ['design'])
        self.assertEqual(rec['agent_execution_seconds']['aggregate'], 0.0)
        self.assertEqual(rec['source_aggregate_execution_seconds'], 10.0)
        self.assertEqual(rec['n_delivered'], 1)
        self.assertEqual(staged.load(ep / 'diagnostics/attribution.json'),
                         staged.load(self.analysis / 'diagnostics/attribution.json'))
        self.assertNotEqual(staged.load(ep / 'diagnostics/attribution_commit.json'),
                            staged.load(self.analysis / 'diagnostics/attribution_commit.json'))
        upstream = stage_audit.load_inputs(ep)
        self.assertEqual(upstream['design']['payload']['primary_goal_id'], 'g1')
        targets = [Path(m['target']) for m in hb.calls[0]['mounts']]
        for private in (self.analysis / 'work/out/old-candidate/secret', self.analysis / 'diagnostics/audit.json'):
            self.assertFalse(any(private == target or target in private.parents for target in targets))
        spec = next(m for m in hb.calls[0]['mounts'] if m['target'] == str(ep / 'diagnostics/candidate_job.json'))
        self.assertTrue(spec['read_only'])
        prompt = (ep / 'diagnostics/design/task/instruction.md').read_text()
        self.assertIn('goal_id is mandatory', prompt)
        self.assertEqual(jobs.run_candidate(hb, self.cfg, 'bench', self.source, ep, 1, job), rec)
        self.assertEqual(len(hb.calls), 1)

    def test_multiple_jobs_each_make_one_design_call_without_new_analysis(self):
        selected = self.joblist()['jobs'][:2]
        hb = GoalHarbor(1)
        for index, job in enumerate(selected, 1):
            ep = self.root / f'candidate-{index}'
            rec = jobs.run_candidate(hb, self.cfg, 'bench', self.source, ep, index, job)
            self.assertEqual(rec['candidate_job']['variant_index'], index - 1)
            inventory = staged.load(ep / 'diagnostics/observations/input_inventory.json')
            self.assertTrue(inventory['analysis_reused'])
            self.assertEqual(inventory['maximum_agent_execution_minutes'], 3)
        self.assertEqual([call['stage'] for call in hb.calls], ['design', 'design'])

    def test_analysis_that_reused_workers_can_supply_a_new_candidate_job(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['diagnostics']['parallel_attribution']['reuse_observations_from'] = str(self.analysis / 'diagnostics')
        source_ep = self.root / 'second-analysis'
        hb = GoalHarbor(1)
        staged.synthesize_staged(hb, cfg, 'bench', self.source, source_ep, 1, analysis_only=True)
        self.assertEqual(len(hb.calls), 1)
        manifest = jobs.prepare_joblist(cfg, [{'original': 'source', 'analysis_source': str(source_ep / 'diagnostics')}],
                                        self.root / 'second-joblist.json')
        author = GoalHarbor(1)
        ep = self.root / 'reused-worker-candidate'
        jobs.run_candidate(author, cfg, 'bench', self.source, ep, 1, manifest['jobs'][0])
        self.assertEqual([call['stage'] for call in author.calls], ['design'])
        upstream = stage_audit.load_inputs(ep)
        self.assertEqual(len(upstream['observation_ids']), 1)

    def test_training_prompt_uses_rollout_count_and_quality_not_pass_band(self):
        self.cfg['training_pipeline']['rollout'] = {'attempts': 8, 'max_concurrent': 4}
        self.cfg['training_pipeline']['candidate_generation']['difficulties'][0]['instruction'] = '保留目标能力，减少无关输出格子'
        job = self.joblist()['jobs'][0]
        ep = self.root / 'training-prompt'
        jobs.run_candidate(GoalHarbor(1), self.cfg, 'bench', self.source, ep, 1, job)
        prompt = (ep / 'diagnostics/design/task/instruction.md').read_text()
        self.assertIn('host plans 8 independent attempts', prompt)
        self.assertIn('successful attempts, does not by itself make a task invalid', prompt)
        self.assertIn('not a training-data admission gate', prompt)
        self.assertNotIn('The eventual difficulty requirement', prompt)
        self.assertNotIn('measured over 1', prompt)
        started = staged.load(ep / 'diagnostics/started.json')
        inventory = staged.load(ep / 'diagnostics/observations/input_inventory.json')
        commit = stage_audit.load_inputs(ep)['attribution']
        self.assertEqual(inventory['config_sha256'], started['config_sha256'])
        self.assertEqual(commit['metadata']['config_sha256'], started['config_sha256'])

    def test_legacy_author_prompt_keeps_the_existing_pass_band(self):
        ep = self.root / 'legacy-prompt'
        staged.synthesize_staged(GoalHarbor(1), self.cfg, 'bench', self.source, ep, 1)
        prompt = (ep / 'diagnostics/design/task/instruction.md').read_text()
        self.assertIn('The eventual difficulty requirement', prompt)
        self.assertIn('measured over 1 independent attempts', prompt)
        self.assertNotIn('not a training-data admission gate', prompt)

    def test_wrong_goal_is_retained_as_failed_not_reassigned(self):
        job = self.joblist()['jobs'][0]
        hb = GoalHarbor(1); hb.wrong_goal = True
        ep = self.root / 'wrong-goal'
        with self.assertRaisesRegex(ValueError, 'assigned primary goal'):
            jobs.run_candidate(hb, self.cfg, 'bench', self.source, ep, 1, job)
        self.assertEqual(len(hb.calls), 1)
        self.assertEqual(staged.load(ep / 'diagnostics/synthesis.json')['status'], 'failed')
        self.assertEqual(staged.load(ep / 'work/design.json')['primary_goal_id'], 'g2')

    def test_config_job_and_source_tamper_are_rejected_without_model_calls(self):
        job = self.joblist()['jobs'][0]
        hb = GoalHarbor(1)
        changed = copy.deepcopy(self.cfg); changed['researcher']['model'] = 'different'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            jobs.run_candidate(hb, changed, 'bench', self.source, self.root / 'bad-config', 1, job)
        changed_job = copy.deepcopy(job); changed_job['difficulty']['instruction'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'job identity changed'):
            jobs.run_candidate(hb, self.cfg, 'bench', self.source, self.root / 'bad-job', 1, changed_job)
        raw = self.analysis / 'diagnostics/attribution/work/output.json'
        raw.write_text('{"claims":[]}')
        with self.assertRaises(ValueError):
            jobs.run_candidate(hb, self.cfg, 'bench', self.source, self.root / 'bad-source', 1, job)
        self.assertEqual(hb.calls, [])

    def test_generation_settings_change_cannot_replace_existing_joblist(self):
        self.joblist()
        before = (self.root / 'joblist.json').read_bytes()
        self.cfg['training_pipeline']['candidate_generation']['copies_per_goal'] = 3
        with self.assertRaisesRegex(ValueError, 'artifact differs'):
            self.joblist()
        self.assertEqual((self.root / 'joblist.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
