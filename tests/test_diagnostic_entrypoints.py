"""Entry point routing tests; no model, network, or Docker execution."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from harness import run_episode, run_score


class InlinePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, values):
        return [function(value) for value in values]


class DiagnosticEntrypointsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ep = self.root / 'runs' / 'run1' / 'bench' / 'task1' / 'ep01'
        self.ep.mkdir(parents=True)
        self.cfg = {
            'diagnostics': {'enabled': True, 'parallel_attribution': {'max_concurrent': 4, 'worker_budget_min': 1, 'aggregate_budget_min': 2, 'design_budget_min': 3}},
            'researcher': {'agent_setup_timeout_mult': 3, 'max_concurrent_docker': 1},
            'judge': {'agent_setup_timeout_mult': 9, 'model': 'judge-model'},
            'score': {'setup_timeout_mult': 6, 'n_concurrent': 1, 'band_lo': 0.125, 'band_hi': 0.75},
            'customer': {'model': 'customer-model'},
        }
        self.hb = object()
        self.audit_hb = object()
        self.audit = Mock(side_effect=self.successful_audit)
        self.fake_stage_audit = types.SimpleNamespace(audit_episode=self.audit)

    def successful_audit(self, hb, cfg, benchmark, original, ep_dir):
        envelope = {'status': 'completed', 'payload': {group: [] for group in
                     ('claims', 'mappings', 'task_checks', 'probes')},
                    'expected_units': {group: [] for group in
                     ('claims', 'mappings', 'task_checks', 'probes')},
                    'failures': [], 'provenance': {'source_task': original}}
        directory = ep_dir / 'diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'audit.json').write_text(json.dumps(envelope))
        return envelope

    def audit_context(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {'stage_audit': self.fake_stage_audit}))
        make = stack.enter_context(patch.object(run_score, 'make_harbor', return_value=self.audit_hb))
        stack.enter_context(patch.object(run_score, 'log'))
        return stack, make

    def test_invalid_delivery_still_runs_one_audit_and_keeps_score(self):
        stack, make = self.audit_context()
        with stack:
            result = run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        self.assertEqual(result['gate'], 0)
        self.assertEqual(result['episode_raw'], 0)
        self.assertEqual(json.loads((self.ep / 'score.json').read_text()), result)
        self.audit.assert_called_once_with(self.audit_hb, self.cfg, 'bench', 'task1', self.ep)
        make.assert_called_once_with(self.cfg, 9.0)
        metrics = json.loads((self.ep / 'diagnostics/metrics.json').read_text())
        self.assertEqual(metrics['status'], 'completed')
        self.assertIsNone(metrics['metrics']['claims']['accuracy'])

    def score_with_difficulty(self, summary, judge=True):
        task = self.ep / 'work/out/task-new'
        task.mkdir(parents=True, exist_ok=True)
        (self.ep / 'diagnostics').mkdir(exist_ok=True)
        stack, make = self.audit_context()
        with stack, patch.object(run_score, 'discover_tasks', return_value=[task]), \
                patch.object(run_score, 'run_difficulty', return_value=summary), \
                patch.object(run_score, 'instruction_diff', return_value={'spans': [], 'n_spans': 1}), \
                patch.object(run_score, 'run_judge') as old_judge:
            result = run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None, judge)
        self.audit.assert_called_once()
        old_judge.assert_not_called()
        return result

    def test_difficulty_zero_runs_audit_without_old_judge(self):
        result = self.score_with_difficulty({'pass_rate': 0.0})
        self.assertEqual(result['difficulty'], 0)
        self.assertEqual(result['episode_raw'], 0.0)

    def test_unscorable_difficulty_runs_audit_without_inventing_score(self):
        result = self.score_with_difficulty({'pass_rate': None, 'unscorable': 'infra faults'})
        self.assertIsNone(result['episode_raw'])
        self.assertIsNone(result['difficulty'])

    def test_rollout_only_deferred_judge_still_runs_audit(self):
        result = self.score_with_difficulty({'pass_rate': 0.5}, judge=False)
        self.assertEqual(result['difficulty'], 1)
        self.assertIsNone(result['episode_raw'])

    def test_original_exception_preserved_and_audit_runs_once(self):
        for error in (RuntimeError('original crash'), SystemExit('original stop')):
            self.audit.reset_mock()
            stack, _ = self.audit_context()
            with stack, patch.object(run_score, '_score_episode_baseline', side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
            self.assertIs(caught.exception, error)
            self.audit.assert_called_once()

    def test_audit_exception_cannot_replace_original_exception(self):
        original = RuntimeError('score remains primary')
        self.audit.side_effect = SystemExit('audit failed')
        stack, _ = self.audit_context()
        with stack, patch.object(run_score, '_score_episode_baseline', side_effect=original):
            with self.assertRaises(RuntimeError) as caught:
                run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        self.assertIs(caught.exception, original)
        envelope = json.loads((self.ep / 'diagnostics/audit.json').read_text())
        self.assertEqual(envelope['status'], 'failed')
        self.assertIsNone(envelope['payload'])
        self.assertEqual(envelope['failures'][0]['error_type'], 'SystemExit')
        self.assertIn('audit failed', envelope['failures'][0]['traceback'])
        metrics = json.loads((self.ep / 'diagnostics/metrics.json').read_text())
        self.assertEqual(metrics['status'], 'failed')
        self.assertIsNone(metrics['metrics']['claims']['accuracy'])

    def test_audit_failure_preserves_score_and_prior_audit_evidence(self):
        expected = {'episode_raw': 0.5, 'sentinel': object()}
        previous = self.ep / 'diagnostics/audit.json'
        previous.parent.mkdir()
        original_bytes = b'{"status":"completed","original":"exact bytes"}\n'
        previous.write_bytes(original_bytes)
        self.audit.side_effect = ValueError('audit schema failed')
        stack, _ = self.audit_context()
        with stack, patch.object(run_score, '_score_episode_baseline', return_value=expected):
            actual = run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        self.assertIs(actual, expected)
        envelope = json.loads(previous.read_text())
        preserved = Path(envelope['failures'][0]['previous_envelope_path'])
        self.assertEqual(preserved.read_bytes(), original_bytes)
        from harness.stage_metrics import summarize_envelope
        metrics = summarize_envelope(envelope)
        self.assertEqual(metrics['status'], 'failed')
        self.assertIsNone(metrics['metrics']['claims']['accuracy'])

    def test_disabled_baseline_skips_audit_harbor_and_artifacts(self):
        self.cfg['diagnostics']['enabled'] = False
        stack, make = self.audit_context()
        with stack:
            result = run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        self.assertEqual(result['gate'], 0)
        self.audit.assert_not_called()
        make.assert_not_called()
        self.assertFalse((self.ep / 'diagnostics').exists())

    def test_judge_missing_optional_setup_uses_scoring_setup(self):
        del self.cfg['judge']['agent_setup_timeout_mult']
        stack, make = self.audit_context()
        with stack:
            run_score.run_diagnostic_audit(self.hb, self.cfg, 'bench', 'task1', self.ep)
        make.assert_called_once_with(self.cfg, 6.0)

    def test_preflight_source_rollout_always_required_but_rubric_optional(self):
        task = self.root / 'task1'
        rollout = self.root / 'prep/task1/rollout'
        rubric = self.root / 'private/rubric.json'
        with patch.object(run_episode, 'rollout_bundle_path', return_value=rollout), \
                patch.object(run_episode, 'rubric_path', return_value=rubric):
            with self.assertRaisesRegex(SystemExit, 'no customer rollout'):
                run_episode.preflight(self.cfg, 'bench', [task], require_rubric=False)
            rollout.mkdir(parents=True)
            (rollout / 'summary.json').write_text('{}')
            run_episode.preflight(self.cfg, 'bench', [task], require_rubric=False)
            with self.assertRaisesRegex(SystemExit, 'no rubric'):
                run_episode.preflight(self.cfg, 'bench', [task])

    def episode_main(self, flags, *, synth_error=None):
        cfg = copy.deepcopy(self.cfg)
        cfg['diagnostics']['enabled'] = False  # CLI override must actually work.
        task = self.root / 'task1'
        args = ['run_episode.py', '--benchmark', 'bench', '--run-id', 'run1', *flags]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', args))
            stack.enter_context(patch.object(run_episode, 'ROOT', self.root))
            stack.enter_context(patch.object(run_episode, 'load_config', return_value=cfg))
            stack.enter_context(patch.object(run_episode, 'load_meta'))
            stack.enter_context(patch.object(run_episode, 'sampled_tasks', return_value=[task]))
            preflight = stack.enter_context(patch.object(run_episode, 'preflight'))
            stack.enter_context(patch.object(run_episode, 'make_harbor', return_value=self.hb))
            stack.enter_context(patch.object(run_episode, 'episode_dir', return_value=self.ep))
            stack.enter_context(patch.object(run_episode, 'rollout_bundle_path', return_value=self.root/'prep/task/rollout'))
            stack.enter_context(patch.object(run_episode, 'rubric_path', return_value=self.root/'rubrics/task/rubric.json'))
            stack.enter_context(patch.object(run_episode.cf, 'ThreadPoolExecutor', InlinePool))
            stack.enter_context(patch.object(run_episode, 'log'))
            synth = stack.enter_context(patch.object(run_episode, 'synthesize', side_effect=synth_error))
            audit = stack.enter_context(patch.object(run_episode, 'run_diagnostic_audit'))
            validate = stack.enter_context(patch.object(
                run_episode, 'validate_authored_candidate',
                return_value={'status': 'completed', 'quality_approved': True}))
            score = stack.enter_context(patch.object(run_episode, 'score_episode', return_value={'episode_dir': str(self.ep)}))
            result = run_episode.main()
        return result, cfg, preflight, synth, validate, audit, score

    def test_synth_only_diagnostics_validates_once_without_private_rubric(self):
        result, cfg, preflight, synth, validate, audit, score = self.episode_main(['--synth-only', '--diagnostics'])
        self.assertEqual(result, 0)
        self.assertTrue(cfg['diagnostics']['enabled'])
        self.assertFalse(preflight.call_args.kwargs['require_rubric'])
        synth.assert_called_once()
        validate.assert_called_once()
        score.assert_not_called()
        manifest = json.loads((self.root / 'runs/run1/run.json').read_text())
        self.assertIs(manifest['diagnostics_enabled'], True)
        self.assertTrue((self.root / 'runs/run1/diagnostics-summary.json').is_file())

    def test_baseline_synth_only_keeps_rubric_and_skips_audit(self):
        _, _, preflight, synth, validate, audit, score = self.episode_main(['--synth-only'])
        self.assertTrue(preflight.call_args.kwargs['require_rubric'])
        validate.assert_not_called()
        score.assert_not_called()
        manifest = json.loads((self.root / 'runs/run1/run.json').read_text())
        self.assertNotIn('diagnostics_enabled', manifest)
        self.assertFalse((self.root / 'runs/run1/diagnostics-summary.json').exists())

    def test_full_episode_leaves_audit_to_score_wrapper(self):
        _, _, preflight, synth, validate, audit, score = self.episode_main(['--diagnostics'])
        self.assertTrue(preflight.call_args.kwargs['require_rubric'])
        synth.assert_called_once()
        validate.assert_called_once()
        score.assert_called_once()

    def test_synthesis_failure_audits_once_and_never_scores(self):
        for error in (RuntimeError('synth crash'), SystemExit('synth failure')):
            with self.subTest(error=type(error).__name__):
                _, _, _, synth, validate, audit, score = self.episode_main(['--diagnostics'], synth_error=error)
                synth.assert_called_once()
                audit.assert_called_once()
                score.assert_not_called()

    def test_score_cli_diagnostics_override_is_applied(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['diagnostics']['enabled'] = False
        args = ['run_score.py', '--run-id', 'run1', '--benchmark', 'bench', '--diagnostics']
        with patch.object(sys, 'argv', args), patch.object(run_score, 'ROOT', self.root), \
                patch.object(run_score, 'load_config', return_value=cfg), \
                patch.object(run_score, 'make_harbor', return_value=self.hb), \
                patch.object(run_score, 'episode_dir', return_value=self.ep), \
                patch.object(run_score.cf, 'ThreadPoolExecutor', InlinePool), \
                patch.object(run_score, 'score_episode') as score:
            self.assertEqual(run_score.main(), 0)
        self.assertTrue(cfg['diagnostics']['enabled'])
        score.assert_called_once()
        self.assertTrue((self.root / 'runs/run1/diagnostics-summary.json').is_file())

    def test_metrics_export_failure_does_not_replace_audit_or_score(self):
        result = {'episode_raw': 0.25}
        old_metrics = self.ep / 'diagnostics/metrics.json'
        old_metrics.parent.mkdir()
        old_metrics.write_text('{"stale": true}')
        stack, _ = self.audit_context()
        with stack, patch.object(run_score, '_score_episode_baseline', return_value=result), \
                patch('stage_metrics.summarize_envelope', side_effect=ValueError('export error')):
            actual = run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        self.assertIs(actual, result)
        self.assertEqual(json.loads((self.ep / 'diagnostics/audit.json').read_text())['status'], 'completed')
        errors = [json.loads(p.read_text()) for p in (self.ep / 'diagnostics/export-errors').glob('*.json')]
        failure = next(e for e in errors if e.get('kind') == 'metrics_export')
        self.assertFalse(old_metrics.exists())
        self.assertEqual(Path(failure['previous_artifact_path']).read_text(), '{"stale": true}')

    def test_existing_run_mode_mismatch_refuses_before_any_write(self):
        path = self.root / 'runs/run1/run.json'
        for existing, flags in (({}, ['--diagnostics', '--score-only']),
                                ({'diagnostics_enabled': True}, ['--score-only'])):
            with self.subTest(existing=existing):
                original = json.dumps(existing).encode()
                path.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'run diagnostics mode differs'):
                    self.episode_main(flags)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((self.ep / 'score.json').exists())

    def test_matching_diagnostic_run_preserves_manifest_exactly(self):
        self.episode_main(['--diagnostics', '--synth-only'])
        path = self.root / 'runs/run1/run.json'
        doc = json.loads(path.read_text())
        doc['started_at'] = 'original start'
        original = json.dumps(doc, separators=(',', ':')).encode()
        path.write_bytes(original)
        self.episode_main(['--diagnostics', '--synth-only'])
        self.assertEqual(path.read_bytes(), original)

    def test_diagnostic_plan_changes_refuse_without_rewriting_manifest(self):
        self.episode_main(['--diagnostics', '--synth-only'])
        path = self.root / 'runs/run1/run.json'
        initial = json.loads(path.read_text())
        changes = [('tasks', ['other-task']), ('episodes', [2]),
                   ('customer', {'model': 'another-model'}),
                   ('prep_root', 'another-source'),
                   ('diagnostic_plan', {'source_paths': ['another-path'], 'config': self.cfg})]
        for field, value in changes:
            with self.subTest(field=field):
                doc = copy.deepcopy(initial)
                doc[field] = value
                original = json.dumps(doc).encode()
                path.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'diagnostic run plan changed'):
                    self.episode_main(['--diagnostics', '--score-only'])
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((self.ep / 'score.json').exists())

    def test_matching_baseline_retains_manifest_rewrite_behavior(self):
        path = self.root / 'runs/run1/run.json'
        path.write_text('{"started_at":"old", "legacy_extra":true}')
        self.episode_main(['--synth-only'])
        doc = json.loads(path.read_text())
        self.assertNotEqual(doc['started_at'], 'old')
        self.assertNotIn('legacy_extra', doc)

    def test_score_only_cannot_bypass_episode_mode_guards(self):
        (self.ep / 'synth.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'baseline episode'):
            self.episode_main(['--diagnostics', '--score-only'])
        self.assertFalse((self.root / 'runs/run1/run.json').exists())
        (self.ep / 'diagnostics').mkdir()
        with self.assertRaisesRegex(ValueError, 'diagnostic episode'):
            self.episode_main(['--score-only'])
        self.assertFalse((self.ep / 'score.json').exists())

    def test_standalone_score_cli_rejects_both_episode_mode_mismatches(self):
        for diagnostic in (True, False):
            with self.subTest(diagnostic=diagnostic):
                cfg = copy.deepcopy(self.cfg)
                cfg['diagnostics'] = ({'enabled': diagnostic} if not diagnostic
                                      else copy.deepcopy(self.cfg['diagnostics']))
                if diagnostic:
                    (self.ep / 'synth.json').write_text('{}')
                else:
                    (self.ep / 'diagnostics').mkdir()
                args = ['run_score.py', '--run-id', 'run1', '--benchmark', 'bench']
                with patch.object(sys, 'argv', args), patch.object(run_score, 'ROOT', self.root), \
                        patch.object(run_score, 'load_config', return_value=cfg), \
                        patch.object(run_score, 'episode_dir', return_value=self.ep), \
                        patch.object(run_score, 'make_harbor') as make, \
                        patch.object(run_score, 'score_episode') as score:
                    with self.assertRaisesRegex(ValueError, 'episode cannot be reused'):
                        run_score.main()
                make.assert_not_called()
                score.assert_not_called()
                self.assertFalse((self.ep / 'score.json').exists())

    def test_direct_score_guard_does_not_launch_audit_or_modify_score(self):
        (self.ep / 'synth.json').write_text('{}')
        original = b'{"original":true}'
        (self.ep / 'score.json').write_bytes(original)
        with patch.object(run_score, 'run_diagnostic_audit') as audit:
            with self.assertRaisesRegex(ValueError, 'baseline episode'):
                run_score.score_episode(self.hb, self.cfg, 'bench', 'task1', self.ep, None)
        audit.assert_not_called()
        self.assertEqual((self.ep / 'score.json').read_bytes(), original)

    def test_run_summary_failure_is_separately_recorded(self):
        with patch('stage_metrics.write_run_summary', side_effect=ValueError('summary error')), \
                patch.object(run_score, 'log'):
            run_score.write_diagnostic_run_summary(self.root)
        errors = list((self.root / 'export-errors').glob('*.json'))
        self.assertEqual(len(errors), 1)
        self.assertEqual(json.loads(errors[0].read_text())['kind'], 'run_summary_export')


if __name__ == '__main__':
    unittest.main()
