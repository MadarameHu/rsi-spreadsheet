import copy
from pathlib import Path
import unittest

import test_stage_audit as fixtures
from stage_records import write_stage_commit

audit = fixtures.audit
FakeHarbor = fixtures.FakeHarbor


class AuditScopeTests(unittest.TestCase):
    setUp = fixtures.AuditTests.setUp
    parallel_fixture = fixtures.AuditTests.parallel_fixture

    def call(self, hb, scope='selected_design'):
        return audit.audit_episode(hb, self.cfg, 'spreadsheet', 'source', self.ep, scope=scope)

    def test_selected_closure_preserves_counterevidence_and_full_source_order(self):
        ids = self.parallel_fixture(compact=True)
        inputs = audit.load_inputs(self.ep)
        full = audit.resolve_scope(inputs)
        selected = audit.resolve_scope(inputs, 'selected_design')
        self.assertEqual(selected['source_units'], full['expected_units'])
        self.assertEqual(selected['expected_units']['claims'], ['c1'])
        self.assertEqual(selected['expected_units']['observations'], ids[:2])
        self.assertEqual(selected['unreviewed_ids']['observations'], [ids[2]])
        self.assertEqual(selected['unreviewed_ids']['claims'], ['c2'])
        self.assertEqual(selected['expected_units']['task_checks'], audit.TASK_CHECKS)
        self.assertEqual(selected['expected_units']['probes'], audit.REQUIRED_PROBES)
        self.assertTrue(all(not ids for ids in full['unreviewed_ids'].values()))

    def test_excluded_mappings_are_unreviewed_support_mappings_remain(self):
        self.parallel_fixture(compact=True)
        inputs = audit.load_inputs(self.ep)
        inputs['design']['payload']['mappings'] = [
            {'id': 'm1', 'claim_id': 'c1', 'role': 'exclude'},
            {'id': 'm2', 'claim_id': 'c2', 'role': 'support'}]
        selected = audit.resolve_scope(inputs, 'selected_design')
        self.assertEqual(selected['expected_units']['claims'], ['c2'])
        self.assertEqual(selected['expected_units']['mappings'], ['m2'])
        self.assertEqual(selected['unreviewed_ids']['mappings'], ['m1'])

    def test_unknown_scope_and_unknown_evidence_are_rejected(self):
        self.parallel_fixture()
        inputs = audit.load_inputs(self.ep)
        with self.assertRaisesRegex(ValueError, 'unsupported audit scope'):
            audit.resolve_scope(inputs, 'arbitrary_subset')
        inputs['attribution']['payload']['claims'][0]['counterevidence_ids'].append('missing')
        with self.assertRaisesRegex(ValueError, 'unknown observation'):
            audit.resolve_scope(inputs, 'selected_design')

    def test_selected_run_binds_scope_but_freezes_all_raw_workers_and_claims(self):
        ids = self.parallel_fixture(compact=True)
        hb = FakeHarbor()
        result = self.call(hb)
        self.assertEqual(result['status'], 'completed', result['failures'])
        run = Path(result['work_dir'])
        manifest = audit.read_json(run / 'evidence/manifest.json')
        navigation = audit.read_json(run / 'evidence/navigation.json')
        commit = audit.read_stage_commit(self.diag / 'records', result['audit_commit'])
        self.assertEqual(navigation['expected_units'], manifest['expected_units'])
        self.assertEqual([i for worker in navigation['workers'] for i in worker['required_ids']], ids[:2])
        self.assertEqual(len(navigation['workers']), 2)
        for group, files in navigation['files'].items():
            self.assertEqual(files, [entry['path'] for entry in manifest['snapshots'][group]['entries']
                                     if entry['type'] == 'file'])
            self.assertTrue(all((Path(navigation['roots'][group]) / name).is_file() for name in files))
        self.assertTrue(all((Path(navigation['evidence_root']) / worker['payload']).is_file()
                            for worker in navigation['workers']))
        self.assertIn(str(run / 'evidence/navigation.json'), [r['original_path'] for r in commit['references']])
        self.assertEqual(manifest['audit_scope'], result['audit_scope'])
        self.assertEqual(commit['metadata']['audit_scope'], result['audit_scope'])
        self.assertEqual(audit.read_json(run / 'input_fingerprint.json')['audit_scope'], result['audit_scope'])
        self.assertEqual([c['id'] for c in result['payload']['claims']], ['c1'])
        self.assertEqual([o['id'] for o in result['payload']['observations']], ids[:2])
        frozen_claims = audit.read_json(manifest['commits']['attribution']['payload_path'])['claims']
        self.assertEqual([c['id'] for c in frozen_claims], ['c1', 'c2'])
        self.assertIn('observation-03', manifest['commits'])
        self.assertIn('observation-00', manifest['commits'])
        self.assertEqual(audit.tree_fingerprint(self.diag / 'observations'),
                         audit.tree_fingerprint(Path(manifest['snapshots']['observations']['snapshot_path'])))
        calls = len(hb.calls)
        self.assertEqual(self.call(hb)['audit_commit'], result['audit_commit'])
        self.assertEqual(len(hb.calls), calls)
        self.assertEqual(len(hb.probe_calls), 2)

    def test_failed_full_can_switch_scope_without_overwriting_or_merging_history(self):
        self.parallel_fixture(compact=True)
        bad = FakeHarbor(malformed=True)
        previous = self.call(bad, 'full')
        old_run = Path(previous['work_dir'])
        original_files = {p: p.read_bytes() for p in old_run.rglob('*') if p.is_file()}
        good = FakeHarbor()
        current = self.call(good)
        self.assertEqual(previous['status'], 'failed')
        self.assertEqual(current['status'], 'completed', current['failures'])
        self.assertNotEqual(current['evidence_fingerprint'], previous['evidence_fingerprint'])
        self.assertEqual(audit.read_json(self.diag / 'audit.json'), current)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in original_files.items()))
        # Switching back does not retry the already failed full scope, even when
        # its original implementation fingerprint was from a previous version.
        calls = len(good.calls)
        returned = self.call(good, 'full')
        self.assertEqual(returned, previous)
        self.assertEqual(len(good.calls), calls)

    def test_failed_same_scope_is_not_retried_after_implementation_identity_changes(self):
        self.parallel_fixture()
        hb = FakeHarbor(malformed=True)
        previous = self.call(hb)
        fingerprint = Path(previous['work_dir']) / 'input_fingerprint.json'
        identity = audit.read_json(fingerprint)
        identity['implementation'] = 'previous-version'
        audit.write_json(fingerprint, identity)
        calls = len(hb.calls)
        self.assertEqual(self.call(hb), previous)
        self.assertEqual(len(hb.calls), calls)

    def test_missing_prior_per_run_envelope_is_preserved_in_history_on_scope_switch(self):
        self.parallel_fixture()
        prior = {'status': 'failed', 'payload': None, 'work_dir': None, 'failures': ['legacy']}
        audit.write_json(self.diag / 'audit.json', prior)
        result = self.call(FakeHarbor())
        self.assertEqual(result['status'], 'completed', result['failures'])
        histories = list((self.diag / 'audit-history').glob('*.json'))
        self.assertEqual(len(histories), 1)
        self.assertEqual(audit.read_json(histories[0]), prior)

    def test_forged_committed_subset_cannot_omit_counterevidence(self):
        self.parallel_fixture(compact=True)
        result = self.call(FakeHarbor())
        inputs = audit.load_inputs(self.ep)
        old = audit.read_stage_commit(self.diag / 'records', result['audit_commit'])
        forged = copy.deepcopy(result)
        forged.pop('audit_commit')
        removed = forged['audit_scope']['expected_units']['observations'].pop()
        forged['audit_scope']['unreviewed_ids']['observations'].insert(0, removed)
        forged['expected_units']['observations'].remove(removed)
        forged['payload']['observations'] = [o for o in forged['payload']['observations'] if o['id'] != removed]
        commit = write_stage_commit(self.diag / 'records', 'audit', forged['payload'],
            references=[Path(r['original_path']) for r in old['references']], parent_ids=old['parent_ids'],
            metadata={**old['metadata'], 'audit_scope': forged['audit_scope'], 'audit_envelope': forged},
            expected_units=forged['expected_units'], base_dir=Path(forged['work_dir']) / 'work')
        forged['audit_commit'] = commit['commit_id']
        with self.assertRaisesRegex(ValueError, 'frozen design evidence closure'):
            audit.verify_cached_envelope(forged, inputs, forged['evidence_fingerprint'],
                                         audit.expected_units(inputs, 'selected_design'))

    def test_legacy_full_commit_without_scope_still_verifies(self):
        self.parallel_fixture()
        result = self.call(FakeHarbor(), 'full')
        inputs = audit.load_inputs(self.ep)
        old = audit.read_stage_commit(self.diag / 'records', result['audit_commit'])
        legacy = copy.deepcopy(result)
        legacy.pop('audit_commit')
        legacy.pop('audit_scope')
        metadata = {k: v for k, v in old['metadata'].items() if k != 'audit_scope'}
        metadata['audit_envelope'] = legacy
        commit = write_stage_commit(self.diag / 'records', 'audit', legacy['payload'],
            references=[Path(r['original_path']) for r in old['references']], parent_ids=old['parent_ids'],
            metadata=metadata, expected_units=legacy['expected_units'],
            base_dir=Path(legacy['work_dir']) / 'work')
        legacy['audit_commit'] = commit['commit_id']
        self.assertEqual(audit.verify_cached_envelope(legacy, inputs, legacy['evidence_fingerprint'],
                                                     audit.expected_units(inputs)), legacy)


if __name__ == '__main__':
    unittest.main()
