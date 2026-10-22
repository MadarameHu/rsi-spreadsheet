#!/usr/bin/env python3
"""Fail if internal infrastructure, credentials, or machine paths leak into the tree.

This is a publication gate, not a linter. It checks the files that would be
committed; ``runs/``, ``prep/`` and other ignored trees are not scanned.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', '.venv', '.pytest_cache', 'runs', 'prep',
             'jobs', 'archive', 'outputs', 'batch', 'data', 'node_modules'}
SKIP_FILES = {Path('tools/check_clean.py')}
SUFFIXES = {'.py', '.sh', '.md', '.json', '.toml', '.txt', '.yml', '.yaml',
            '.html', '.css', '.js', '.cfg', '.ini'}

RULES = [
    (re.compile(r'\b' + 'd' + 's' + 'w' + r'\b', re.I), 'internal host name'),
    (re.compile('/' + 'Users' + '/'), 'absolute home path'),
    (re.compile('/' + 'mnt' + '/'), 'absolute mount path'),
    (re.compile('/' + 'root' + '/'), 'absolute root path'),
    (re.compile('/' + 'home' + '/'), 'absolute home path'),
    (re.compile(r'\b' + 'hj' + 'w' + r'\b|' + 'hujia' + 'wang', re.I), 'personal identifier'),
    (re.compile('host' + r'\.' + 'docker' + r'\.' + 'internal', re.I), 'container host alias'),
    (re.compile(r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'private IP literal'),
    (re.compile('s' + 'k' + r'-[A-Za-z0-9]{8,}'), 'credential literal'),
    (re.compile('AUTODATA' + 'BENCH_API_KEY|ADB_VALI' + 'DATION_API_KEY'), 'credential variable'),
    (re.compile(r'\b' + 's' + 's' + 'h' + r'\b', re.I), 'remote shell'),
    (re.compile(r'\b' + 'r' + 's' + 'y' + 'n' + 'c' + r'\b', re.I), 'remote copy'),
    (re.compile(r'\b' + 't' + 'u' + 'n' + 'n' + 'e' + 'l' + r'\b', re.I), 'tunnel'),
    (re.compile('batch-' + '20' + '26'), 'internal batch identifier'),
    (re.compile(r'127\.0\.0\.1:(' + '189' + '80|' + '189' + '81)'), 'internal port'),
]

# A value assigned to api_key outside test fixtures is a credential.
KEY_VALUE = re.compile(r'api_key["\']?\s*[:=]\s*["\'][^"\']+["\']')


def files():
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or rel in SKIP_FILES:
            continue
        if path.suffix.lower() not in SUFFIXES:
            continue
        yield rel, path


def main() -> int:
    hits = []
    for rel, path in files():
        try:
            text = path.read_text(errors='replace')
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, why in RULES:
                if pattern.search(line):
                    hits.append((rel, lineno, why, line.strip()[:160]))
            if KEY_VALUE.search(line) and not str(rel).startswith('tests/'):
                hits.append((rel, lineno, 'credential value', line.strip()[:160]))
    if hits:
        for rel, lineno, why, line in hits:
            print(f'{rel}:{lineno}: {why}: {line}')
        print(f'\nFAILED: {len(hits)} hit(s)')
        return 1
    print('clean')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
