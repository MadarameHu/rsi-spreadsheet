"""Contract and immutable-link tests; no models, inferred labels, or repairs."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from harness import parallel_records as parallel
from harness import stage_records as records


class ParallelRecordsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root / 'original.txt'
        self.evidence.write_text('Full original evidence.\n' * 10000)
        self.refs = [{'path': str(self.evidence), 'locator': 'attempt 0, event 1'}]
        self.worker = {'schema_version': parallel.OBSERVATIONS_SCHEMA, 'sample_id': 0,
                       'observations': [{'id': 's00-o1', 'statement': 'An observed output.',
                                         'evidence': self.refs, 'uncertainties': []}]}
        self.aggregate = {'schema_version': parallel.AGGREGATE_SCHEMA,
                          'claims': [{'id': 'c1', 'kind': 'defect', 'statement': 'A conditional weakness.',
                                      'observation_ids': ['s00-o1'], 'counterevidence_ids': [],
                                      'conditions': ['Under the recorded condition.'],
                                      'evidence': self.refs, 'uncertainties': []}]}
        self.design = {'mappings': [{'id': 'm1', 'claim_id': 'c1', 'feature': 'A targeted check.',
                                     'check_plan': 'Check the target cell.', 'role': 'target',
                                     'preserves': ['Target operation'], 'simplifies': []}]}

    def test_observations_preserve_raw_fields_and_empty_inventory(self):
        self.worker['extra'] = {'complete_text': 'not shortened' * 10000}
        before = copy.deepcopy(self.worker)
        parallel.validate_observations(self.worker)
        self.assertEqual(self.worker, before)
        empty = dict(self.worker, observations=[])
        parallel.validate_observations(empty)

    def test_sample_and_observation_identity_are_not_repaired(self):
        for sample in (-1, True, 0.5, '0'):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                parallel.validate_observations(dict(self.worker, sample_id=sample))
        for identifier in ('s01-o1', 's00-', 's00- ', ''):
            value = copy.deepcopy(self.worker)
            value['observations'][0]['id'] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                parallel.validate_observations(value)
        value = copy.deepcopy(self.worker)
        value['observations'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parallel.validate_observations(value)

    def test_optional_evidence_limitations_preserved_without_side_effects(self):
        for limitations in ([], ['An embedded tool response reports truncation.']):
            value = copy.deepcopy(self.worker)
            value['evidence_limitations'] = limitations
            before = copy.deepcopy(value)
            source_before = self.evidence.read_bytes()
            files_before = sorted(self.root.rglob('*'))
            parallel.validate_observations(value)
            self.assertEqual(value, before)
            self.assertEqual(self.evidence.read_bytes(), source_before)
            self.assertEqual(sorted(self.root.rglob('*')), files_before)

    def test_evidence_limitations_require_list_of_nonempty_strings(self):
        for invalid in (None, True, 1, 'limitation', {}, (), [None], [False],
                        [42], [{}], [[]], [''], ['   ']):
            value = copy.deepcopy(self.worker)
            value['evidence_limitations'] = invalid
            before = copy.deepcopy(value)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, 'evidence_limitations'):
                parallel.validate_observations(value)
            self.assertEqual(value, before)

    def test_atomic_evidence_and_uncertainty_required(self):
        for field, invalid in [('statement', ''), ('evidence', []), ('uncertainties', None),
                               ('uncertainties', [''])]:
            value = copy.deepcopy(self.worker)
            value['observations'][0][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                parallel.validate_observations(value)
        value = copy.deepcopy(self.worker)
        value['observations'][0]['evidence'][0]['path'] = self.evidence.name
        with self.assertRaisesRegex(ValueError, 'absolute'):
            parallel.validate_observations(value, base_dir=self.root)

    def test_aggregate_kinds_conditions_and_links(self):
        for kind in parallel.CLAIM_KINDS:
            value = copy.deepcopy(self.aggregate)
            value['claims'][0]['kind'] = kind
            parallel.validate_aggregate(value, self.worker['observations'])
        for field, invalid in [('kind', 'failure'), ('conditions', None),
                               ('observation_ids', []), ('counterevidence_ids', None)]:
            value = copy.deepcopy(self.aggregate)
            value['claims'][0][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                parallel.validate_aggregate(value, self.worker['observations'])

    def test_support_and_counterevidence_both_require_known_unique_ids(self):
        for field in ('observation_ids', 'counterevidence_ids'):
            for invalid in (['s01-absent'], ['s00-o1', 's00-o1']):
                value = copy.deepcopy(self.aggregate)
                value['claims'][0][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    parallel.validate_aggregate(value, self.worker['observations'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parallel.validate_aggregate(self.aggregate, self.worker['observations'] * 2)

    def test_conflicting_observations_are_preserved_without_voting(self):
        opposite = dict(self.worker['observations'][0], id='s01-o1', statement='An opposing output.')
        value = copy.deepcopy(self.aggregate)
        value['claims'][0]['counterevidence_ids'] = ['s01-o1']
        before = copy.deepcopy(value)
        parallel.validate_aggregate(value, self.worker['observations'] + [opposite])
        self.assertEqual(value, before)

    def test_commits_bind_observation_parents_and_full_evidence(self):
        directory = self.root / 'records'
        parent = records.write_stage_commit(directory, 'observations', self.worker)
        with self.assertRaisesRegex(ValueError, 'observation commits'):
            records.write_stage_commit(directory, 'attribution', self.aggregate)
        child = records.write_stage_commit(directory, 'attribution', self.aggregate,
                                           parent_ids=[parent['commit_id']])
        self.assertEqual(records.read_stage_commit(directory, child['commit_id'])['payload'], self.aggregate)
        self.assertEqual((directory / parent['references'][0]['path']).read_bytes(), self.evidence.read_bytes())
        value = copy.deepcopy(self.aggregate)
        value['claims'][0]['counterevidence_ids'] = ['s01-uncommitted']
        with self.assertRaisesRegex(ValueError, 'unknown observation'):
            records.write_stage_commit(directory, 'attribution', value, parent_ids=[parent['commit_id']])

    def test_concurrent_workers_share_evidence_without_archive_race(self):
        directory = self.root / 'records'
        def commit(sample):
            value = copy.deepcopy(self.worker)
            value['sample_id'] = sample
            value['observations'][0]['id'] = f's{sample:02d}-o1'
            return records.write_stage_commit(directory, 'observations', value)
        with ThreadPoolExecutor(max_workers=16) as pool:
            commits = list(pool.map(commit, range(16)))
        self.assertEqual(len({c['commit_id'] for c in commits}), 16)
        self.assertEqual(len(list((directory / 'artifacts').iterdir())), 1)
        for commit_record in commits:
            records.read_stage_commit(directory, commit_record['commit_id'])

    def test_optional_observation_audit_exact_inventory(self):
        payload = {'claims': [], 'mappings': [], 'task_checks': [], 'probes': [],
                   'observations': [{'id': 's00-o1', 'verdict': 'uncertain',
                                     'reason': 'Evidence not decisive.', 'evidence': self.refs}]}
        expected = {'claims': [], 'mappings': [], 'task_checks': [], 'probes': [],
                    'observations': ['s00-o1']}
        records.validate_audit(payload, expected)
        with self.assertRaisesRegex(ValueError, 'observations group'):
            records.validate_audit(payload, {k: v for k, v in expected.items() if k != 'observations'})
        for defect in ('missing', 'unknown', 'duplicate', 'verdict'):
            value = copy.deepcopy(payload)
            if defect == 'missing':
                value['observations'] = []
            elif defect == 'unknown':
                value['observations'][0]['id'] = 's01-o1'
            elif defect == 'duplicate':
                value['observations'] *= 2
            else:
                value['observations'][0]['verdict'] = 'supported'
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                records.validate_audit(value, expected)
        commit = records.write_stage_commit(self.root / 'records', 'audit', payload,
                                            expected_units=expected)
        self.assertEqual(len(commit['references']), 1)

    def test_parallel_design_records_roles_and_explicit_exclusion(self):
        records.validate_design(self.design, self.aggregate)
        value = copy.deepcopy(self.aggregate)
        value['claims'].append(dict(value['claims'][0], id='c2', kind='infrastructure_issue'))
        design = copy.deepcopy(self.design)
        design['mappings'].append(dict(design['mappings'][0], id='m2', claim_id='c2', role='exclude',
                                        exclusion_reason='A service failure is not a capability target.'))
        records.validate_design(design, value)
        del design['mappings'][1]['exclusion_reason']
        with self.assertRaisesRegex(ValueError, 'exclusion_reason'):
            records.validate_design(design, value)

    def test_parallel_design_targets_only_capabilities_or_defects(self):
        for kind in ('task_issue', 'infrastructure_issue', 'uncertain'):
            value = copy.deepcopy(self.aggregate)
            value['claims'][0]['kind'] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'defect or capability'):
                records.validate_design(self.design, value)
        for role in ('support', 'preserve', 'exclude'):
            design = copy.deepcopy(self.design)
            design['mappings'][0].update(role=role, exclusion_reason='Explicitly omitted.')
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'at least one target'):
                records.validate_design(design, self.aggregate)

    def test_parallel_design_requires_preserves_and_simplifies_without_target_cap(self):
        for field in ('preserves', 'simplifies'):
            design = copy.deepcopy(self.design)
            del design['mappings'][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                records.validate_design(design, self.aggregate)
        design = copy.deepcopy(self.design)
        design['mappings'] = [dict(design['mappings'][0], id=f'm{i}') for i in range(20)]
        records.validate_design(design, self.aggregate)


if __name__ == '__main__':
    unittest.main()
