"""Exact, allowlisted single-trajectory inputs and evidence validation.

Only input files are copied; no judgments, score summaries, or other attempts
are injected. Existing snapshots are checked against their sources, never repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import tempfile

try:
    from .parallel_records import validate_observations
except ImportError:
    from parallel_records import validate_observations

POLICY_VERSION = 'single-trajectory-inputs-v1'
_REQUIRED = frozenset({'task/instruction.md', 'attempt.md'})
_ALLOWED = _REQUIRED | {'task/system_prompt.md', 'task/environment/input.xlsx', 'evidence_index.json',
                        'score-trace.json'}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build_evidence_index(contents):
    """Locate complete original text mechanically, without rewriting its bytes."""
    lines = contents['attempt.md'].decode('utf-8').splitlines()
    headings = []
    seen = set()
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r'### step (\d+) / event (\d+) / (.+)', line)
        if match:
            event = int(match[2])
            if event in seen:
                raise ValueError('duplicate event number in original trajectory')
            seen.add(event)
            headings.append({'event': event, 'step': int(match[1]),
                             'type': match[3], 'line_start': number})
    for offset, event in enumerate(headings):
        event['line_end'] = headings[offset+1]['line_start']-1 if offset+1 < len(headings) else len(lines)
    score_sections = []
    for number, line in enumerate(lines, 1):
        if line == '## Official score and execution summary':
            end = next((i for i in range(number+1, len(lines)+1)
                        if re.match(r'^#{1,2} ', lines[i-1])), len(lines)+1)-1
            score_sections.append({'line_start': number, 'line_end': end})
    sources = {}
    for alias, relative in [('trace', 'attempt.md'), ('task', 'task/instruction.md'),
                            ('system', 'task/system_prompt.md'), ('input', 'task/environment/input.xlsx')]:
        if relative in contents:
            sources[alias] = {'relative_path': relative}
            if alias != 'input':
                sources[alias]['line_count'] = len(contents[relative].decode('utf-8').splitlines())
    return {'schema_version': 'worker-evidence-index-v1', 'sources': sources,
            'events': headings, 'score_sections': score_sections}


def _read_manifest(manifest_path):
    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('worker manifest must be a regular file')
    root = path.parent.resolve(strict=True)
    payload = json.loads(path.read_bytes())
    if payload.get('policy_version') != POLICY_VERSION:
        raise ValueError('worker manifest policy_version mismatch')
    if type(payload.get('sample_id')) is not int or payload['sample_id'] < 0:
        raise ValueError('worker manifest sample_id invalid')
    files = payload.get('files')
    if not isinstance(files, list):
        raise ValueError('worker manifest files must be a list')
    seen = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError('worker manifest file entry must be an object')
        relative = entry.get('relative_path')
        if relative not in _ALLOWED or relative in seen:
            raise ValueError('worker manifest file outside allowlist or duplicated')
        seen.add(relative)
        target = root / relative
        if entry.get('path') != str(target):
            raise ValueError('worker manifest evidence path mismatch')
        if not isinstance(entry.get('source_path'), str) or not Path(entry['source_path']).is_absolute():
            raise ValueError('worker manifest source_path must be absolute')
        if target.is_symlink() or not target.is_file() or target.resolve() != target:
            raise ValueError('worker input missing, symlinked, or outside snapshot')
        raw = target.read_bytes()
        if type(entry.get('bytes')) is not int or entry['bytes'] != len(raw) or entry.get('sha256') != _digest(raw):
            raise ValueError('worker input bytes differ from manifest')
    if not _REQUIRED <= seen:
        raise ValueError('worker manifest missing required files')
    allowed_files = [entry['path'] for entry in files]
    if payload.get('allowed_files') != allowed_files:
        raise ValueError('worker manifest allowed_files mismatch')
    expected_paths = {root / relative for relative in seen} | {path.resolve()}
    expected_dirs = {parent for entry in expected_paths for parent in entry.parents if parent != root and root in parent.parents}
    for entry in root.rglob('*'):
        if entry.is_symlink() or entry not in expected_paths | expected_dirs:
            raise ValueError('unexpected file or directory in worker input snapshot')
    if 'evidence_index.json' in seen:
        content = {entry['relative_path']: (root / entry['relative_path']).read_bytes()
                   for entry in files if entry['relative_path'] != 'evidence_index.json'}
        if json.loads((root / 'evidence_index.json').read_bytes()) != build_evidence_index(content):
            raise ValueError('worker evidence index differs from original input locations')
    return payload


def prepare_worker_inputs(source: Path, attempt: Path, directory: Path, sample: int,
                          trace: Path | None = None) -> dict:
    """Create an exact input snapshot or verify an unchanged existing snapshot.

    trace: this attempt's single score trace, when the host computed one. Other
    attempts' traces and the cross-attempt summary are never included.
    """
    if type(sample) is not int or sample < 0:
        raise ValueError('sample must be a nonnegative integer')
    source, attempt = Path(source).resolve(strict=True), Path(attempt).resolve(strict=True)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = directory / 'inputs'
    candidates = [('task/instruction.md', source/'instruction.md', True),
                  ('task/system_prompt.md', source/'system_prompt.md', False),
                  ('task/environment/input.xlsx', source/'environment'/'input.xlsx', False),
                  ('attempt.md', attempt, True),
                  ('score-trace.json', Path(trace) if trace else None, False)]
    entries, contents = [], {}
    for relative, original, required in candidates:
        if original is None and not required:
            continue
        if not original.exists() and not required:
            continue
        if not original.is_file():
            raise ValueError(f'worker source input is not a file: {original}')
        raw = original.read_bytes()
        contents[relative] = raw
        entries.append({'relative_path': relative, 'path': str(root/relative),
                        'source_path': str(original.resolve(strict=True)),
                        'sha256': _digest(raw), 'bytes': len(raw)})
    index_raw = (json.dumps(build_evidence_index(contents), ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    contents['evidence_index.json'] = index_raw
    entries.append({'relative_path': 'evidence_index.json', 'path': str(root/'evidence_index.json'),
                    'source_path': str(root/'evidence_index.json'), 'derived': True,
                    'sha256': _digest(index_raw), 'bytes': len(index_raw)})
    manifest = {'policy_version': POLICY_VERSION, 'sample_id': sample,
                'files': entries, 'allowed_files': [entry['path'] for entry in entries]}
    manifest_path = root/'manifest.json'
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            raise ValueError('existing worker inputs must be a directory')
        existing = _read_manifest(manifest_path)
        if existing != manifest:
            raise ValueError('existing worker manifest differs from current source identity')
    else:
        # Temporary staging retains partial evidence on any failure. It is never
        # reused or repaired. Publish only once all full input bytes are written.
        staging = Path(tempfile.mkdtemp(prefix='inputs-pending-', dir=directory))
        for relative, raw in contents.items():
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (staging/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        staging.rename(root)
        _read_manifest(manifest_path)
    return {'root': str(root), 'task_dir': str(root/'task'),
            'attempt': str(root/'attempt.md'), 'manifest': str(manifest_path),
            'evidence_index': str(root/'evidence_index.json'),
            'trace': str(root/'score-trace.json') if 'score-trace.json' in {e['relative_path'] for e in entries} else None,
            'allowed_files': manifest['allowed_files']}


def validate_worker_output(payload, manifest_path):
    """Validate without rewriting payloads or permitting implicit path repair."""
    manifest = _read_manifest(manifest_path)
    if isinstance(payload, dict) and 'schema_version' not in payload:
        try:
            from .compact_observations import enrich_observations
        except ImportError:
            from compact_observations import enrich_observations
        enrich_observations(payload, manifest_path)
        return
    validate_observations(payload)
    if payload['sample_id'] != manifest['sample_id']:
        raise ValueError('worker output sample_id differs from input manifest')
    allowed = set(manifest['allowed_files'])
    for observation in payload['observations']:
        for evidence in observation['evidence']:
            if evidence['path'] not in allowed:
                raise ValueError(f"worker evidence path not in input manifest: {evidence['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', type=Path, required=True, metavar='OUTPUT')
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.validate.read_bytes())
        validate_worker_output(payload, args.manifest)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(1, f'Validation failed: {error}\n')
    print(f"Valid worker output: sample {_read_manifest(args.manifest)['sample_id']}, {len(payload['observations'])} observations")


if __name__ == '__main__':
    main()
