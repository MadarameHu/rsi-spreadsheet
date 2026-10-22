"""Run the existing Claude tool loop locally with OS-enforced evidence isolation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

from common import terminate_process_group

RUNTIME = 'local-claude'
TOOLS = 'Bash,Read,Write,Edit,Glob,Grep'


def _identity(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _local_url(base_url):
    url = urlsplit(base_url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost'} or url.username or url.password:
        raise ValueError('local analysis requires the existing loopback HTTP relay')
    if not url.port:
        raise ValueError('relay requires an explicit port')
    return urlunsplit((url.scheme, f'127.0.0.1:{url.port}', url.path, url.query, url.fragment))


def sandbox_profile(ro, writable, cli, port):
    """No user-data reads beyond explicit inputs; descendants inherit restrictions."""
    runtime_ro = ['/bin', '/sbin', '/usr', '/System/Library', '/Library/Apple',
                  '/Library/Developer', '/Library/Frameworks', '/opt/homebrew',
                  '/private/etc', '/private/var/db', '/dev']
    rules = ['(version 1)', '(allow default)', '(deny file-read*)', '(deny file-write*)',
             '(allow file-read-metadata)', '(allow file-read* (literal "/"))',
             '(allow file-write* (literal "/dev/null") (literal "/dev/tty"))',
             '(deny network*)', f'(allow network-outbound (remote ip "localhost:{int(port)}"))']
    for item in [*runtime_ro, Path(sys.prefix), *ro, cli]:
        path = Path(item).resolve()
        kind = 'subpath' if path.is_dir() else 'literal'
        rules.append(f'(allow file-read* ({kind} {json.dumps(str(path))}))')
    for item in writable:
        rules.append(f'(allow file-read* file-write* (subpath {json.dumps(str(Path(item).resolve()))}))')
    return '\n'.join(rules) + '\n'


def read_local_result(job_dir):
    job = Path(job_dir).resolve()
    receipt = job / 'local-result.json'
    record = json.loads(receipt.read_text())
    if record.get('runtime') != RUNTIME or not record.get('finished_at'):
        raise ValueError('invalid local process receipt')
    for key in ('transcript', 'stderr'):
        expected = record[key]
        actual = _identity(expected['path'])
        if actual != expected:
            raise ValueError(f'local {key} evidence changed')
    seconds = record['active_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('invalid local process duration')
    source = _identity(receipt)
    terminal = {'runtime': RUNTIME, 'agent_terminal_status': record['status'],
                'exception_type': record.get('exception_type'), 'exception_message': record.get('exception_message'),
                'finished_at': record['finished_at'], 'trial_result_path': str(receipt),
                'trial_result_sha256': source['sha256']}
    return {'runtime': RUNTIME, 'returncode': record['returncode'], 'finished_at': record['finished_at'],
            'output': record['output'], 'agent_execution': {'status': 'completed', 'active_seconds': seconds,
                                                         'source': source}, 'terminal': terminal}


def run_local(directory, prompt_path, ro, seconds, model, base_url, api_key, extra_env=None):
    directory = Path(directory).resolve()
    work, job = directory/'work', directory/'job'
    job.mkdir(parents=True, exist_ok=True)
    if (job/'local-result.json').exists() or (job/'claude-code.txt').exists():
        raise RuntimeError('existing local invocation; do not automatically replay')
    work.mkdir(parents=True, exist_ok=True)
    runtime_home, runtime_tmp = directory/'runtime-home', directory/'runtime-tmp'
    runtime_home.mkdir(exist_ok=True)
    runtime_tmp.mkdir(exist_ok=True)
    started, monotonic_start = _now(), time.monotonic()
    rc, error_type, error_message, status, proc = None, None, None, 'error', None
    cleanup = None
    transcript, stderr = job/'claude-code.txt', job/'stderr.txt'
    with transcript.open('xb') as out, stderr.open('xb') as err:
        try:
            if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
                raise RuntimeError('local analysis requires macOS sandbox-exec; no unisolated fallback')
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('seconds must be positive and finite')
            relay = _local_url(base_url)
            cli = shutil.which('claude')
            if not cli:
                raise RuntimeError('existing Claude Code CLI is missing')
            prompt = Path(prompt_path).resolve(strict=True)
            inputs = [Path(p).resolve(strict=True) for p in ro]
            profile = sandbox_profile([*inputs, prompt], [work, runtime_home, runtime_tmp], cli, urlsplit(relay).port)
            # Python may inspect cwd when an agent cd's to the stage root.
            # A literal directory rule permits listing, not reading descendants.
            profile += f'(allow file-read* (literal {json.dumps(str(directory))}))\n'
            profile_path = job/'sandbox.sb'
            profile_path.write_text(profile)
            # Do not inherit user hooks, credentials, proxies, Node/Python injection or shell startup settings.
            env = {'PATH': str(Path(sys.executable).parent)+':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                   'TMPDIR': str(runtime_tmp), 'CLAUDE_CODE_TMPDIR': str(runtime_tmp),
                   'CLAUDE_CODE_SHELL': '/bin/zsh', 'TMPPREFIX': str(runtime_tmp/'shell'),
                   'CLAUDE_CONFIG_DIR': str(runtime_home),
                   'XDG_CONFIG_HOME': str(runtime_home), 'XDG_CACHE_HOME': str(runtime_home/'cache'),
                   'ANTHROPIC_BASE_URL': relay, 'ANTHROPIC_API_KEY': str(api_key),
                   'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_AUTOUPDATER': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                   'ADB_WORK': str(work), 'ADB_RESULT': str(work/'output.json'),
                   'ADB_DEADLINE_EPOCH': str(int(time.time()+seconds))}
            for key, value in (extra_env or {}).items():
                if not key.startswith('ADB_'):
                    raise ValueError('only task-scoped ADB_ environment additions are accepted')
                env[key] = str(value)
            command = ['/usr/bin/sandbox-exec', '-f', str(profile_path), cli, '--bare',
                       '--setting-sources', '', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
                       '--disable-slash-commands', '--no-session-persistence', '--permission-mode', 'dontAsk',
                       '--tools', TOOLS, '--allowedTools', TOOLS, '--model', model,
                       '--output-format', 'stream-json', '--verbose', '--print']
            with prompt.open('rb') as stdin:
                proc = subprocess.Popen(command, stdin=stdin, stdout=out, stderr=err,
                                        cwd=work, env=env, start_new_session=True)
                try:
                    rc = proc.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    cleanup = terminate_process_group(proc)
                    rc = proc.returncode
                    status, error_type, error_message = 'timeout', 'TimeoutError', 'local agent exceeded allocated wall time'
            if status != 'timeout':
                out.flush()
                results = []
                for line in transcript.read_text().splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict) and value.get('type') == 'result':
                        results.append(value)
                if rc == 0 and len(results) == 1 and results[0].get('is_error') is False:
                    status = 'completed'
                else:
                    error_type, error_message = 'LocalAgentError', f'CLI return code {rc}; valid successful result receipt absent'
        except Exception as error:
            error_type, error_message = type(error).__name__, str(error)
            if proc is not None and cleanup is None:
                cleanup = terminate_process_group(proc)
                rc = proc.returncode
    record = {'runtime': RUNTIME, 'started_at': started, 'finished_at': _now(),
              'active_seconds': time.monotonic()-monotonic_start, 'status': status,
              'returncode': rc, 'exception_type': error_type, 'exception_message': error_message,
              'output': str(work/'output.json'), 'transcript': _identity(transcript), 'stderr': _identity(stderr)}
    if cleanup is not None:
        record['process_cleanup'] = cleanup
    with (job/'local-result.json').open('x') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return read_local_result(job)
