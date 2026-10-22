import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import cost_report  # noqa: E402

PRICES = {'opus': {'input': 5.0, 'cache_write': 6.25, 'cache_read': 0.5, 'output': 25.0},
          'qwen': {'input': 0.0, 'cache_write': 0.0, 'cache_read': 0.0, 'output': 0.0}}


def put(path, obj, text=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else json.dumps(obj))


def result(model, inp, write, read, out, turns=3):
    return json.dumps({'type': 'result', 'num_turns': turns, 'modelUsage': {model: {
        'inputTokens': inp, 'cacheCreationInputTokens': write, 'cacheReadInputTokens': read,
        'outputTokens': out, 'costUSD': 1.0}}})


class CostReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ep = self.tmp / 'run/bench/task/ep01'
        d = self.ep / 'diagnostics'
        put(d / 'attribution/job/claude-code.txt', None,
            '{"type":"system","subtype":"init"}\n' + result('opus', 10, 1000, 100000, 2000))
        agent = d / 'design/job/t/task__x/agent'
        put(agent / 'claude-code.txt', None, '{"type":"assistant"}\n' + result('opus', 0, 0, 0, 1000))
        put(agent / 'trajectory.json', {'agent': {'model_name': 'opus'}, 'final_metrics': {}})
        put(agent / 'sessions/projects/-app/s.jsonl', None, result('opus', 9, 9, 9, 999999))
        put(d / 'audit-runs/fp/evidence/design_run/agent/claude-code.txt', None, result('opus', 0, 0, 0, 999999))
        put(d / 'audit-runs/fp/reviews/task/job/t/task__y/agent/trajectory.json',
            {'agent': {'model_name': 'opus'}, 'steps': [{'metrics': {'prompt_tokens': 1}}, {}, {'metrics': {'prompt_tokens': 1}}],
             'final_metrics': {'total_prompt_tokens': 1300, 'total_completion_tokens': 50,
                               'extra': {'total_cache_creation_input_tokens': 200,
                                         'total_cache_read_input_tokens': 1000,
                                         'cost_source': 'litellm_estimate'}}})
        put(self.ep / 'work/testruns/trainee-host-a-00/job/t/c__z/agent/native-execution.json',
            {'model': 'qwen', 'usage': {'prompt_tokens': 400, 'completion_tokens': 90},
             'native_events': [{'step_index': 0}, {'step_index': 2}]})
        other = self.tmp / 'old/diagnostics/observations/sample-00'
        put(other / 'job/claude-code.txt', None, result('kimi', 100, 0, 500, 40))
        put(d / 'observations/sample-00/state.json', {'directory': str(other)})

    def by_stage(self):
        return {(r['stage'], r['reused']): r for r in cost_report.collect(self.ep)}

    def test_sources_stages_and_skipped_copies(self):
        rows = self.by_stage()
        self.assertEqual(set(rows), {('aggregate', False), ('design', False), ('audit', False),
                                     ('trainee', False), ('observation', True)})
        self.assertEqual(rows[('design', False)]['output'], 1000)  # result line, not session jsonl
        audit = rows[('audit', False)]
        self.assertEqual((audit['input'], audit['cache_write'], audit['cache_read'], audit['calls'],
                          audit['complete'], audit['recorded_cost_usd']), (100, 200, 1000, 2, False, None))
        self.assertEqual((rows[('trainee', False)]['input'], rows[('trainee', False)]['calls']), (400, 3))

    def test_costs_keep_reused_and_unpriced_separate(self):
        s = cost_report.summarize(cost_report.collect(self.ep), PRICES)
        expected = (10 * 5 + 1000 * 6.25 + 100000 * 0.5 + 2000 * 25 + 1000 * 25
                    + 100 * 5 + 200 * 6.25 + 1000 * 0.5 + 50 * 25) / 1e6
        self.assertAlmostEqual(s['episode_new_spend']['cost_usd'], round(expected, 4))
        self.assertEqual(s['episode_new_spend']['unpriced_models'], [])
        self.assertEqual(s['including_reused']['unpriced_models'], ['kimi'])
        self.assertEqual(s['incomplete_sessions'], 1)


if __name__ == '__main__':
    unittest.main()
