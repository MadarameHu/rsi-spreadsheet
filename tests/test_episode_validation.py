"""CLI routing checks for authoring, host execution validation and formal audit."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import run_episode
import run_score


class EpisodeValidationRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.task = self.root/'bench/source'
        self.task.mkdir(parents=True)
        self.bundle = self.root/'prep/rollouts/source'
        self.bundle.mkdir(parents=True)
        (self.bundle/'summary.json').write_text('{}')
        self.rubric = self.root/'prep/rubrics/source/rubric.json'
        self.calls = []
        self.run_number = 0
        self.cfg = {
            'diagnostics': {'enabled': True, 'attribution_mode': 'parallel',
                            'parallel_attribution': {'runtime': 'local-claude', 'max_concurrent': 16,
                                                     'worker_budget_min': 1, 'aggregate_budget_min': 1,
                                                     'design_budget_min': 1}},
            'researcher': {'agent_setup_timeout_mult': 1, 'max_concurrent_docker': 3},
            'customer': {'model': 'target', 'attempts': 16},
            'judge': {'model': 'reviewer'},
            'score': {'setup_timeout_mult': 1, 'n_concurrent': 3},
        }

    def invoke(self, *flags, validation_status='completed', validation_error=None):
        self.run_number += 1
        run_id = f'route-{self.run_number}'
        self.ep = self.root/'runs'/run_id/'bench/source/ep01'
        cfg = copy.deepcopy(self.cfg)
        self.calls = []

        def author(*args, **kwargs):
            self.calls.append('author')

        def validate(*args, **kwargs):
            self.calls.append('validation')
            self.assertEqual(args[2], 'bench')
            self.assertEqual(args[3], self.ep)
            if validation_error is not None:
                raise validation_error
            return {'status': validation_status, 'quality_approved': None}

        def baseline(*args, **kwargs):
            self.calls.append('score')
            return {'episode_dir': str(self.ep.relative_to(self.root)), 'episode_raw': 0}

        def audit(*args, **kwargs):
            self.calls.append('audit')

        argv = ['run_episode', '--benchmark', 'bench', '--task', 'source', '--run-id', run_id, *flags]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(run_episode, 'ROOT', self.root))
            for name, value in [('load_config', cfg), ('load_meta', {}), ('sampled_tasks', [self.task]),
                                ('rollout_bundle_path', self.bundle), ('rubric_path', self.rubric),
                                ('episode_dir', self.ep), ('make_harbor', SimpleNamespace())]:
                stack.enter_context(patch.object(run_episode, name, return_value=value))
            for name, effect in [('synthesize', author), ('validate_authored_candidate', validate),
                                 ('run_diagnostic_audit', audit)]:
                stack.enter_context(patch.object(run_episode, name, side_effect=effect))
            # Keep the real score wrapper: the formal audit belongs to its finally block.
            stack.enter_context(patch.object(run_score, '_score_episode_baseline', side_effect=baseline))
            stack.enter_context(patch.object(run_score, 'run_diagnostic_audit', side_effect=audit))
            stack.enter_context(patch.object(run_episode, 'write_diagnostic_run_summary'))
            stack.enter_context(patch.object(run_episode, 'log'))
            result = run_episode.main()
        return result, json.loads((self.ep/'runtime-status.json').read_text())

    def test_validation_only_skips_author_audit_and_score_without_rubric(self):
        self.assertFalse(self.rubric.exists())
        result, runtime = self.invoke('--validation-only')
        self.assertEqual(result, 0)
        self.assertEqual(self.calls, ['validation'])
        self.assertEqual(runtime['status'], 'completed')
        self.assertEqual(runtime['phase'], 'host-validation')
        self.assertFalse(self.rubric.exists())

    def test_parallel_synth_only_runs_author_then_host_validation_without_full_audit(self):
        result, runtime = self.invoke('--synth-only')
        self.assertEqual(result, 0)
        self.assertEqual(self.calls, ['author', 'validation'])
        self.assertEqual(runtime['status'], 'completed')

    def test_validation_failure_or_exception_blocks_all_downstream_stages(self):
        self.rubric.parent.mkdir(parents=True)
        self.rubric.write_text('{}')
        for flags in [(), ('--synth-only',), ('--validation-only',)]:
            for failure in ['failed', 'incomplete', RuntimeError('validation interrupted')]:
                with self.subTest(flags=flags, failure=str(failure)):
                    kwargs = ({'validation_error': failure} if isinstance(failure, Exception)
                              else {'validation_status': failure})
                    result, runtime = self.invoke(*flags, **kwargs)
                    self.assertEqual(result, 1)
                    self.assertEqual(self.calls, (['validation'] if '--validation-only' in flags
                                                  else ['author', 'validation']))
                    self.assertEqual(runtime['status'], 'failed')
                    self.assertEqual(runtime['phase'], 'host-validation')

    def test_default_parallel_orders_author_validation_score_then_audit(self):
        self.rubric.parent.mkdir(parents=True)
        self.rubric.write_text('{}')
        result, runtime = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(self.calls, ['author', 'validation', 'score', 'audit'])
        self.assertEqual(runtime['status'], 'completed')
        self.assertTrue((self.ep.parents[1]/'episodes.json').is_file())


if __name__ == '__main__':
    unittest.main()
