import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import error_library as lib
import diagnostics


def commit(cid, *claims):
    return {'commit_id': cid, 'payload': {'claims': [
        dict({'id': f'c{i}', 'statement': f'Mechanism {i}', 'conditions': ['x'],
              'support_sample_ids': [i]}, **c) for i, c in enumerate(claims, 1)]}}


class ErrorLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'prep/model/spreadsheet/task-1' / lib.NAME

    def test_off_by_default_and_needs_boolean(self):
        cfg = {'diagnostics': {'parallel_attribution': {}}}
        self.assertIsNone(lib.library_path(cfg, 'b', 't'))
        self.assertEqual(lib.ingest(None, commit('a' * 64, {'kind': 'defect'}), episode='/ep'), 0)
        self.assertEqual(lib.prompt_text(None), lib.NONE_TEXT)
        base = {'enabled': True, 'attribution_mode': 'parallel', 'parallel_attribution': {
            'max_concurrent': 1, 'worker_budget_min': 1, 'aggregate_budget_min': 1, 'design_budget_min': 1}}
        base['parallel_attribution']['error_library'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'error_library'):
            diagnostics.parallel_config({'diagnostics': base})
        base['parallel_attribution']['error_library'] = True
        self.assertTrue(diagnostics.parallel_config({'diagnostics': base})['error_library'])

    def test_path_is_per_customer_model(self):
        cfg = {'diagnostics': {'parallel_attribution': {'error_library': True}}}
        with patch('run_rollout.prep_dir', side_effect=lambda c, b, t: Path('/p') / b / t):
            self.assertEqual(lib.library_path(cfg, 'spreadsheet', 'task-1'), Path('/p/spreadsheet/task-1') / lib.NAME)

    def test_identity_is_declared_by_the_aggregator_not_by_wording(self):
        first = commit('a' * 64, {'kind': 'defect', 'evidence_strength': 'hypothesis_only'},
                       {'kind': 'capability'}, {'kind': 'defect'})
        self.assertEqual(lib.ingest(self.path, first, episode='/ep1'), 2)
        self.assertEqual(lib.ingest(self.path, first, episode='/ep1'), 0)   # recovery replay
        ids = [c['id'] for c in json.loads(self.path.read_text())['classes']]
        # Next batch: same mechanism reworded and now replay-verified, plus a new mechanism.
        second = commit('b' * 64, {'kind': 'defect', 'statement': 'Reworded', 'library_class': ids[0],
                                   'evidence_strength': 'replay_verified'},
                        {'kind': 'defect', 'statement': 'New mechanism', 'library_class': None})
        self.assertEqual(lib.ingest(self.path, second, episode='/ep2'), 1)
        data = json.loads(self.path.read_text())
        self.assertEqual([c['statement'] for c in data['classes']], ['Mechanism 1', 'Mechanism 3', 'New mechanism'])
        self.assertEqual([len(c['sources']) for c in data['classes']], [2, 1, 1])
        self.assertEqual(data['classes'][0]['evidence_strength'], 'replay_verified')
        self.assertEqual(data['classes'][1]['evidence_strength'], 'not_declared')
        with self.assertRaisesRegex(ValueError, 'unknown library class'):
            lib.ingest(self.path, commit('c' * 64, {'kind': 'defect', 'library_class': 'nope'}), episode='/ep3')
        self.assertEqual(json.loads(self.path.read_text())['ingested_commits'], ['a' * 64, 'b' * 64])
        self.assertEqual(lib.prompt_text(self.path), str(self.path))

    def test_candidate_recommit_adds_nothing_and_marks_its_source(self):
        first = commit('a' * 64, {'kind': 'defect'}, {'kind': 'defect'})
        lib.ingest(self.path, first, episode='/ep1')
        recommit = dict(first, commit_id='b' * 64, metadata={'source_attribution_commit': 'a' * 64})
        self.assertEqual(lib.ingest(self.path, recommit, episode='/ep2'), 0)
        for _ in range(2):
            lib.mark_targeted(self.path, recommit, ['c1'], episode='/ep2', design_commit='d1')
        classes = json.loads(self.path.read_text())['classes']
        self.assertEqual([len(c['sources']) for c in classes], [1, 1])
        self.assertEqual([len(c['targeted_by']) for c in classes], [1, 0])
        self.assertEqual(classes[0]['targeted_by'][0], {'episode': '/ep2', 'design_commit': 'd1'})

if __name__ == '__main__':
    unittest.main()
