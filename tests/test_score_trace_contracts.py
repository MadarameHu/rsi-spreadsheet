"""Mechanical score-trace contracts on worker, aggregate and design outputs; offline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import score_trace
from compact_observations import enrich_observations
from compact_synthesis import validate_training_goals
from worker_inputs import prepare_worker_inputs


def record(reward=0.0, status='reproduced', flags=(), after=0):
    return {'sample_id': 0, 'official': {'reward': reward}, 'reconciliation': {'status': status},
            'last_edit_feedback': {'flags': list(flags), 'tool_calls_after': after}}


def row(sample, reward=0.0, status='reproduced', flags=(), after=0):
    return {'sample_id': sample, 'official_reward': reward, 'reconciliation': status,
            'last_feedback_flags': list(flags), 'tool_calls_after_last_edit': after}


class FeedbackConditionTests(unittest.TestCase):
    def test_only_error_or_zero_left_uncorrected(self):
        self.assertTrue(score_trace.feedback_condition(record(flags=['error_value'])))
        self.assertTrue(score_trace.feedback_condition(record(flags=['all_zero', 'all_identical'])))
        self.assertFalse(score_trace.feedback_condition(record(flags=['all_identical'])))
        self.assertFalse(score_trace.feedback_condition(record(flags=['error_value'], after=1)))
        self.assertFalse(score_trace.feedback_condition(record(flags=['error_value'], after=None)))

    def test_samples_exclude_solved_and_unattributed(self):
        summary = {'rows': [row(0, flags=['all_zero']), row(1, reward=1.0, flags=['all_zero']),
                            row(2, status='not_reproduced', flags=['error_value']),
                            row(3, flags=['all_identical']), row(4, flags=['error_value'])]}
        self.assertEqual(score_trace.feedback_samples(summary), [0, 4])


class WorkerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'instruction.md').write_text('Task\n')
        self.attempt = self.root / 'attempt.md'
        self.attempt.write_text('# A\n### step 1 / event 0 / tool_call\ncall\n### step 1 / event 1 / tool_response\nok\n')

    def prepare(self, trace=None):
        self.count = getattr(self, 'count', 0) + 1
        path = None
        if trace is not None:
            path = self.root / f'trace-{self.count}.json'
            path.write_text(json.dumps(trace))
        return prepare_worker_inputs(self.source, self.attempt, self.root / f'worker-{self.count}', 0, trace=path)

    def enrich(self, prepared, *findings):
        raw = {'observations': [{'finding': text, 'evidence': ['event:0']} for text in findings]}
        return enrich_observations(raw, prepared['manifest'])

    def test_trace_is_copied_only_when_supplied(self):
        self.assertIsNone(self.prepare()['trace'])
        prepared = self.prepare(record())
        self.assertTrue(Path(prepared['trace']).is_file())
        self.assertIn(prepared['trace'], prepared['allowed_files'])

    def test_attributed_failure_requires_one_ordered_score_trace(self):
        prepared = self.prepare(record())
        self.enrich(prepared, 'SCORE TRACE: REASON 1: x\nREASON 2: y')
        for findings in [('plain',), ('SCORE TRACE: no reasons',), ('SCORE TRACE: REASON 2: skipped',),
                         ('SCORE TRACE: REASON 1: a', 'SCORE TRACE: REASON 1: b')]:
            with self.subTest(findings=findings), self.assertRaises(ValueError):
                self.enrich(prepared, *findings)

    def test_feedback_required_exactly_when_condition_holds(self):
        prepared = self.prepare(record(flags=['error_value']))
        with self.assertRaisesRegex(ValueError, 'FEEDBACK'):
            self.enrich(prepared, 'SCORE TRACE: REASON 1: x')
        self.enrich(prepared, 'SCORE TRACE: REASON 1: x', 'FEEDBACK: no correction')

    def test_unattributed_or_untraced_forbids_prefixes(self):
        for prepared in (self.prepare(), self.prepare(record(status='not_reproduced'))):
            self.enrich(prepared, 'plain observation')
            with self.assertRaises(ValueError):
                self.enrich(prepared, 'SCORE TRACE: REASON 1: x')

    def test_solved_attempt_has_no_score_trace(self):
        prepared = self.prepare(record(reward=1.0))
        self.enrich(prepared, 'plain observation')
        with self.assertRaises(ValueError):
            self.enrich(prepared, 'SCORE TRACE: REASON 1: x')


class GoalContractTests(unittest.TestCase):
    def attribution(self, *rationales):
        claims = [{'id': 'c1', 'kind': 'defect'}]
        goals = [{'id': f'g{i}', 'claim_ids': ['c1'], 'ability': 'a', 'rationale': text,
                  'success_signal': 's', 'difficulty_controls': []} for i, text in enumerate(rationales, 1)]
        return {'claims': claims, 'training_goals': goals}

    def summary(self, feedback=2, attempts=16, unattributed=()):
        rows = [row(i, flags=['error_value']) for i in range(feedback)]
        return {'attempts': attempts, 'not_reproduced': list(unattributed), 'rows': rows}

    def test_secondary_format_and_counts(self):
        good = 'SECONDARY: appeared in 2 of 16 attempts and determined the score in 1; 0 unattributed.'
        validate_training_goals(self.attribution('primary', good), self.summary())
        # Trailing explanation after the required opening sentence is allowed.
        validate_training_goals(self.attribution('primary', good + ' It appeared in s01 and s02.'),
                                self.summary())
        for text in ['SECONDARY: twice', good.replace('in 1;', 'in 3;'), good.replace('of 16', 'of 15'),
                     good.replace('0 unattributed', '1 unattributed')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_training_goals(self.attribution('primary', text), self.summary())

    def test_first_goal_cannot_be_secondary(self):
        text = 'SECONDARY: appeared in 2 of 16 attempts and determined the score in 0; 0 unattributed.'
        with self.assertRaisesRegex(ValueError, 'first training goal'):
            validate_training_goals(self.attribution(text), self.summary())

    def test_feedback_goal_needs_two_traces(self):
        validate_training_goals(self.attribution('primary', 'FEEDBACK: two'), self.summary(feedback=2))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            validate_training_goals(self.attribution('primary', 'FEEDBACK: one'), self.summary(feedback=1))

    def test_legacy_validation_without_trace_is_unchanged(self):
        validate_training_goals(self.attribution('primary', 'SECONDARY: free text'))


class ProbeSourceTests(unittest.TestCase):
    summary = {'rows': [row(0), row(3, reward=1.0)]}

    def probe(self, source):
        return [{'name': 'p', 'source': source, 'edits': [{'tool': 'fill_formula', 'arguments': {}}]}]

    def test_sources_must_cite_unsolved_known_attempts(self):
        score_trace.validate_probes(self.probe('s00'), self.summary)
        score_trace.validate_probes(self.probe('authored'), self.summary)
        for source in ('s03', 's09', 'control', '', 's0,x'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                score_trace.validate_probes(self.probe(source), self.summary)

    def test_host_recalculation_commits_a_probe_file(self):
        import staged_synth
        with tempfile.TemporaryDirectory() as root:
            work, diag = Path(root, 'work'), Path(root, 'diag')
            work.mkdir(), diag.mkdir()
            (work / 'probes.json').write_text('[]')
            def fake(candidate, variants, out):
                out.mkdir(parents=True)
                (out / 'probes.json').write_text('{}')
                return []
            original = score_trace.probe
            score_trace.probe = fake
            try:
                staged_synth.validate_probes(work, diag, Path(root, 'candidate'))
            finally:
                score_trace.probe = original
            self.assertTrue((diag / 'probes/probes.json').is_file())


if __name__ == '__main__':
    unittest.main()
