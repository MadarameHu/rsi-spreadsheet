"""Single-trajectory isolation and strict evidence validation; no model calls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from harness import worker_inputs as worker


class WorkerInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'instruction.md').write_bytes(b'Original instruction\r\n')
        self.attempt = self.root/'attempt-11.md'
        self.attempt.write_bytes(('Full trajectory 中文\r\n' * 10000).encode())
        self.directory = self.root/'worker'

    def prepare(self):
        return worker.prepare_worker_inputs(self.source, self.attempt, self.directory, 11)

    def payload(self, prepared):
        return {'schema_version': 'trajectory-observations-v1', 'sample_id': 11,
                'observations': [{'id': 's11-o1', 'statement': 'An atomic observation.',
                                  'evidence': [{'path': prepared['attempt'], 'locator': 'event 1'}],
                                  'uncertainties': []}]}

    def test_only_allowlisted_inputs_and_exact_bytes(self):
        (self.source/'system_prompt.md').write_bytes(b'system\r\n')
        (self.source/'environment').mkdir()
        (self.source/'environment'/'input.xlsx').write_bytes(b'full workbook\x00\xff')
        for name in ('_meta.json', 'summary.json', 'solution/solve.sh', 'tests/test.py',
                     'environment/secret.txt', 'attempt-12.md'):
            path = self.source/name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text('POISON: all 16 failed; assume failure')
        prepared = self.prepare()
        root = Path(prepared['root'])
        self.assertEqual({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()},
                         {'task/instruction.md', 'task/system_prompt.md', 'task/environment/input.xlsx',
                          'attempt.md', 'manifest.json','evidence_index.json'})
        manifest = json.loads(Path(prepared['manifest']).read_bytes())
        self.assertEqual(manifest['sample_id'], 11)
        self.assertEqual(prepared['allowed_files'], manifest['allowed_files'])
        for entry in manifest['files']:
            raw = Path(entry['source_path']).read_bytes()
            self.assertEqual(Path(entry['path']).read_bytes(), raw)
            self.assertEqual(entry['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(entry['bytes'], len(raw))
        self.assertEqual(Path(prepared['attempt']).read_bytes(), self.attempt.read_bytes())

    def test_optional_files_absent_and_resume_never_overwrites(self):
        prepared = self.prepare()
        before = {p: p.stat().st_mtime_ns for p in Path(prepared['root']).rglob('*') if p.is_file()}
        self.assertEqual(self.prepare(), prepared)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.attempt.write_bytes(b'changed original')
        with self.assertRaisesRegex(ValueError, 'current source identity'):
            self.prepare()
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def test_manifest_tampering_rejected_on_resume(self):
        prepared = self.prepare()
        path = Path(prepared['manifest'])
        original = json.loads(path.read_bytes())
        for field, value in [('sample_id', 12), ('policy_version', 'untrusted'),
                             ('allowed_files', [str(self.attempt)])]:
            modified = dict(original, **{field: value})
            path.write_text(json.dumps(modified))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
        path.write_text(json.dumps(original))
        original['files'][0]['source_path'] = str(self.attempt)
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'current source identity'):
            self.prepare()

    def test_output_validation_preserves_bad_paths_and_all_content(self):
        prepared = self.prepare()
        good = self.payload(prepared)
        good['extra'] = 'Full arbitrary raw field' * 10000
        expected = copy.deepcopy(good)
        worker.validate_worker_output(good, prepared['manifest'])
        self.assertEqual(good, expected)
        for path in (str(self.attempt), str(self.source/'instruction.md'),
                     str(Path(prepared['root'])/'missing.md'),
                     str(Path(prepared['root'])/'task'/'..'/'attempt.md')):
            payload = copy.deepcopy(good)
            payload['observations'][0]['evidence'][0]['path'] = path
            before = copy.deepcopy(payload)
            with self.subTest(path=path), self.assertRaises(ValueError):
                worker.validate_worker_output(payload, prepared['manifest'])
            self.assertEqual(payload, before)

    def test_snapshot_modification_extra_file_and_symlink_rejected(self):
        prepared = self.prepare()
        attempt = Path(prepared['attempt'])
        original = attempt.read_bytes()
        attempt.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            worker.validate_worker_output(self.payload(prepared), prepared['manifest'])
        attempt.write_bytes(original)
        extra = Path(prepared['root'])/'summary.json'
        extra.write_text('all failed')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.prepare()
        extra.unlink()
        attempt.unlink()
        attempt.symlink_to(self.attempt)
        with self.assertRaisesRegex(ValueError, 'symlinked'):
            self.prepare()

    def test_wrong_sample_and_structure_rejected(self):
        prepared = self.prepare()
        payload = self.payload(prepared)
        payload['sample_id'] = 12
        payload['observations'][0]['id'] = 's12-o1'
        with self.assertRaisesRegex(ValueError, 'sample_id differs'):
            worker.validate_worker_output(payload, prepared['manifest'])
        payload = self.payload(prepared)
        payload['observations'][0]['uncertainties'] = None
        with self.assertRaises(ValueError):
            worker.validate_worker_output(payload, prepared['manifest'])

    def test_manifest_cannot_add_external_evidence(self):
        prepared = self.prepare()
        path = Path(prepared['manifest'])
        manifest = json.loads(path.read_bytes())
        manifest['files'].append({'relative_path': '../attempt-11.md', 'path': str(self.attempt),
                                  'source_path': str(self.attempt), 'sha256': 'unused', 'bytes': 0})
        manifest['allowed_files'].append(str(self.attempt))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'outside allowlist'):
            worker.validate_worker_output(self.payload(prepared), path)

    def test_standalone_cli_with_only_three_modules(self):
        prepared = self.prepare()
        isolated = self.root/'tools'
        isolated.mkdir()
        for name in ('worker_inputs.py', 'parallel_records.py', 'stage_records.py', 'compact_observations.py'):
            shutil.copyfile(Path(worker.__file__).parent/name, isolated/name)
        output = self.root/'output.json'
        output.write_text(json.dumps(self.payload(prepared)))
        command = [sys.executable, str(isolated/'worker_inputs.py'), '--validate', str(output),
                   '--manifest', prepared['manifest']]
        result = subprocess.run(command, cwd=isolated, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('sample 11, 1 observations', result.stdout)
        bad = self.payload(prepared)
        bad['observations'][0]['evidence'][0]['path'] = str(self.attempt)
        output.write_text(json.dumps(bad))
        original = output.read_bytes()
        result = subprocess.run(command, cwd=isolated, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('not in input manifest', result.stderr)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
