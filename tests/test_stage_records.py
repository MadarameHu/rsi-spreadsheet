"""Behavioral tests for immutable stage evidence and missing-result handling."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness import stage_records as records


class StageRecordsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root / 'trajectory.json'
        self.evidence.write_text('{"events":[{"id":3,"value":"完整证据"}]}')
        self.claims = {'claims': [{'id': 'c1', 'statement': 'The output was blank.',
                                  'observed_as': 'failure',
                                  'evidence': [{'path': str(self.evidence), 'locator': 'events[0]'}],
                                  'uncertainties': []}]}
        self.design = {'mappings': [{'id': 'm1', 'claim_id': 'c1',
                                     'feature': 'A required output cell.',
                                     'check_plan': 'Inspect the output after execution.'}]}
        item = lambda identifier: {'id': identifier, 'verdict': 'uncertain',
                                   'reason': 'The available evidence is insufficient.', 'evidence': []}
        self.audit = {'claims': [item('c1')],
                      'mappings': [{'id': 'm1', 'upstream_verdict': 'uncertain',
                                    'implementation_verdict': 'correct',
                                    'reason': 'Implementation is visible, causality is not.', 'evidence': []}],
                      'task_checks': [item('spec')],
                      'probes': [{'id': 'p1', 'expected_valid': None, 'actual_accept': None,
                                  'execution_status': 'missing', 'reason': 'No execution record exists.',
                                  'evidence': []}]}
        self.expected = {'claims': ['c1'], 'mappings': ['m1'], 'task_checks': ['spec'], 'probes': ['p1']}

    def test_validation_preserves_payload_and_unknown_fields(self):
        self.claims['extra_original_detail'] = {'full_text': 'x' * 20000}
        original = copy.deepcopy(self.claims)
        records.validate_attribution(self.claims)
        records.validate_design(self.design, self.claims)
        records.validate_audit(self.audit, self.expected)
        self.assertEqual(self.claims, original)
        self.assertIsNone(self.audit['probes'][0]['actual_accept'])

    def test_missing_evidence_file_and_locator_are_rejected(self):
        for change in ({'path': str(self.root / 'absent')}, {'locator': ''}):
            with self.subTest(change=change):
                value = copy.deepcopy(self.claims)
                value['claims'][0]['evidence'][0].update(change)
                with self.assertRaises(ValueError):
                    records.validate_attribution(value)

    def test_relative_reference_and_supplied_hash(self):
        value = copy.deepcopy(self.claims)
        evidence = value['claims'][0]['evidence'][0]
        evidence['path'] = self.evidence.name
        evidence['sha256'] = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        records.validate_attribution(value, base_dir=self.root)
        self.evidence.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            records.validate_attribution(value, base_dir=self.root)

    def test_blank_required_fields_and_invalid_observation(self):
        for field in ('id', 'statement', 'observed_as'):
            value = copy.deepcopy(self.claims)
            value['claims'][0][field] = ' '
            with self.subTest(field=field), self.assertRaises(ValueError):
                records.validate_attribution(value)
        for field in ('feature', 'check_plan', 'claim_id'):
            value = copy.deepcopy(self.design)
            value['mappings'][0][field] = ''
            with self.subTest(field=field), self.assertRaises(ValueError):
                records.validate_design(value, self.claims)
        self.audit['claims'][0]['reason'] = ''
        with self.assertRaises(ValueError):
            records.validate_audit(self.audit, self.expected)

    def test_empty_attribution_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            records.validate_attribution({'claims': []})
        self.claims['claims'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            records.validate_attribution(self.claims)
        self.design['mappings'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            records.validate_design(self.design, [self.claims['claims'][0]])

    def test_design_unknown_and_missing_claims(self):
        self.design['mappings'][0]['claim_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'unknown'):
            records.validate_design(self.design, self.claims)
        self.design['mappings'][0]['claim_id'] = 'c1'
        another = dict(self.claims['claims'][0], id='c2')
        with self.assertRaisesRegex(ValueError, 'missing'):
            records.validate_design(self.design, self.claims['claims'] + [another])

    def test_audit_exact_inventory_no_missing_or_unknown_or_duplicate(self):
        for group in self.expected:
            for defect in ('missing', 'unknown', 'duplicate'):
                value = copy.deepcopy(self.audit)
                if defect == 'missing':
                    value[group] = []
                elif defect == 'unknown':
                    value[group][0]['id'] = 'not-expected'
                else:
                    value[group].append(copy.deepcopy(value[group][0]))
                with self.subTest(group=group, defect=defect), self.assertRaises(ValueError):
                    records.validate_audit(value, self.expected)
        bad = dict(self.expected, probes=['p1', 'p1'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            records.validate_audit(self.audit, bad)

    def test_audit_missing_groups_and_bad_verdict(self):
        value = copy.deepcopy(self.audit)
        del value['probes']
        with self.assertRaises(ValueError):
            records.validate_audit(value, self.expected)
        self.audit['mappings'][0]['implementation_verdict'] = 'pass'
        with self.assertRaises(ValueError):
            records.validate_audit(self.audit, self.expected)

    def test_probe_execution_status_has_no_synthetic_failure(self):
        for status, actual in [('done', True), ('completed', None), ('missing', False),
                               ('infra_error', False), ('completed', 0)]:
            value = copy.deepcopy(self.audit)
            value['probes'][0].update(execution_status=status, actual_accept=actual)
            with self.subTest(status=status, actual=actual), self.assertRaises(ValueError):
                records.validate_audit(value, self.expected)
        for status, actual in [('missing', None), ('infra_error', None), ('completed', False)]:
            value = copy.deepcopy(self.audit)
            value['probes'][0].update(execution_status=status, actual_accept=actual)
            records.validate_audit(value, self.expected)
            self.assertEqual(value['probes'][0]['actual_accept'], actual)
        del value['probes'][0]['expected_valid']
        with self.assertRaises(ValueError):
            records.validate_audit(value, self.expected)

    def test_snapshot_preserves_hidden_files_empty_dirs_and_no_manifest_pollution(self):
        source = self.root / 'source'
        (source / '.hidden-dir' / 'empty').mkdir(parents=True)
        (source / '.hidden-dir' / '.hidden-file').write_bytes(b'complete\x00binary')
        (source / 'manifest.json').write_text('original candidate file')
        target = self.root / 'snapshot'
        manifest = records.snapshot_tree(source, target)
        self.assertEqual((target / 'manifest.json').read_text(), 'original candidate file')
        self.assertTrue((target / '.hidden-dir' / 'empty').is_dir())
        self.assertEqual((target / '.hidden-dir' / '.hidden-file').read_bytes(), b'complete\x00binary')
        records.verify_snapshot(manifest)
        external = self.root / 'version.json'
        external.write_text(json.dumps(manifest))
        records.verify_snapshot(external)
        with self.assertRaises(FileExistsError):
            records.snapshot_tree(source, target)

    def test_snapshot_detects_modified_removed_and_added_files(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'file').write_text('original')
        for index, defect in enumerate(('modify', 'remove', 'add')):
            target = self.root / f'snapshot-{index}'
            manifest = records.snapshot_tree(source, target)
            if defect == 'modify':
                (target / 'file').write_text('changed')
            elif defect == 'remove':
                (target / 'file').unlink()
            else:
                (target / '.new').write_text('new')
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                records.verify_snapshot(manifest)

    def test_snapshot_source_changes_during_copy_are_not_committed(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'file').write_text('before')
        copy2 = records.shutil.copy2
        def mutate(src, dst):
            result = copy2(src, dst)
            Path(src).write_text('after')
            return result
        with patch.object(records.shutil, 'copy2', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                records.snapshot_tree(source, self.root / 'mixed')
        self.assertTrue((self.root / 'mixed').exists())

    def test_snapshot_rejects_symlinks_and_nested_destination(self):
        source = self.root / 'source'
        source.mkdir()
        with self.assertRaisesRegex(ValueError, 'outside'):
            records.snapshot_tree(source, source / 'nested')
        (source / 'link').symlink_to(self.evidence)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            records.snapshot_tree(source, self.root / 'snapshot')

    def test_commit_freezes_full_evidence_and_is_idempotent(self):
        directory = self.root / 'commits'
        original = copy.deepcopy(self.claims)
        commit = records.write_stage_commit(directory, 'attribution', self.claims,
                                             references=[self.evidence], metadata={'source_task': 'task1'})
        same = records.write_stage_commit(directory, 'attribution', self.claims,
                                           references=[self.evidence], metadata={'source_task': 'task1'})
        self.assertEqual(commit['commit_id'], same['commit_id'])
        self.assertEqual(self.claims, original)
        self.assertFalse(commit['is_human_gold'])
        self.assertEqual(len(commit['references']), 1)
        self.evidence.write_text('live source changed later')
        loaded = records.read_stage_commit(directory, commit['commit_id'])
        self.assertEqual(loaded['payload'], original)
        archive = directory / loaded['references'][0]['path']
        self.assertIn('完整证据', archive.read_text())

    def test_commit_detects_archived_evidence_and_envelope_corruption(self):
        for defect in ('archive', 'envelope'):
            directory = self.root / defect
            commit = records.write_stage_commit(directory, 'attribution', self.claims)
            if defect == 'archive':
                path = directory / commit['references'][0]['path']
            else:
                path = directory / 'commits' / (commit['commit_id'] + '.json')
            if defect == 'archive':
                path.write_bytes(b'changed')
            else:
                changed = copy.deepcopy(commit)
                changed['payload']['claims'][0]['statement'] = 'changed'
                path.write_text(json.dumps(changed))
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                records.read_stage_commit(directory, commit['commit_id'])

    def test_commit_unknown_parent_and_parent_corruption(self):
        directory = self.root / 'records'
        with self.assertRaisesRegex(ValueError, 'commit file'):
            records.write_stage_commit(directory, 'attribution', self.claims, parent_ids=['0' * 64])
        parent = records.write_stage_commit(directory, 'attribution', self.claims)
        child = records.write_stage_commit(directory, 'design', self.design, claims=self.claims,
                                            parent_ids=[parent['commit_id']])
        records.read_stage_commit(directory, child['commit_id'])
        (directory / parent['references'][0]['path']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            records.read_stage_commit(directory, child['commit_id'])

    def test_audit_commit_keeps_missing_results_null(self):
        directory = self.root / 'records'
        commit = records.write_stage_commit(directory, 'audit', self.audit, expected_units=self.expected)
        loaded = records.read_stage_commit(directory, commit['commit_id'])
        self.assertIsNone(loaded['payload']['probes'][0]['actual_accept'])
        self.assertEqual(loaded['payload']['probes'][0]['execution_status'], 'missing')


if __name__ == '__main__':
    unittest.main()
