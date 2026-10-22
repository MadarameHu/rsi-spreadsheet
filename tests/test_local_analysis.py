import json
import os
import signal
from pathlib import Path
import subprocess
import shutil
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
from local_analysis import run_local, read_local_result, sandbox_profile, _local_url
from common import terminate_process_group


@unittest.skipUnless(sys.platform == 'darwin', 'actual macOS sandbox integration')
class LocalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.directory = self.root/'sample'
        self.directory.mkdir()
        self.inputs = self.directory/'inputs'
        self.inputs.mkdir()
        self.prompt = self.inputs/'prompt.txt'
        self.prompt.write_text('complete unchanged prompt')
        self.cli = self.root/'fake-claude'

    def fake(self, body):
        self.cli.write_text('#!'+sys.executable+'\n'+body)
        self.cli.chmod(0o755)

    def run_agent(self, seconds=5):
        with patch('local_analysis.shutil.which', return_value=str(self.cli)):
            return run_local(self.directory, self.prompt, [self.inputs], seconds,
                             'model', 'http://127.0.0.1:12345', 'local-relay', {})

    def test_success_and_recovery(self):
        self.fake('import json,sys,os\nfrom pathlib import Path\nassert sys.stdin.read()=="complete unchanged prompt"\nassert "--bare" in sys.argv\nassert not any("max-token" in a for a in sys.argv)\nPath(os.environ["ADB_RESULT"]).write_text("{}")\nprint(json.dumps({"type":"result","is_error":False}))\n')
        result = self.run_agent()
        self.assertEqual(result['terminal']['agent_terminal_status'], 'completed')
        self.assertEqual(result, read_local_result(self.directory/'job'))
        self.assertTrue(Path(result['output']).is_file())
        with self.assertRaisesRegex(RuntimeError, 'automatically replay'):
            self.run_agent()
        (self.directory/'job'/'claude-code.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            read_local_result(self.directory/'job')

    def test_timeout_keeps_partial_output(self):
        # Start the timeout assertion after the child actually writes. Sandbox
        # startup can exceed a tiny test budget while real model jobs run.
        self.cli.write_text('#!/bin/sh\nprintf "partial raw output\\n"\nsleep 30\n')
        self.cli.chmod(0o755)
        real_popen = subprocess.Popen
        transcript = self.directory/'job'/'claude-code.txt'

        def launch_ready(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if 'partial raw output' in transcript.read_text():
                    return proc
                if proc.poll() is not None:
                    break
                time.sleep(.01)
            terminate_process_group(proc)
            self.fail('sandbox child did not produce the fixture transcript')

        with patch('local_analysis.subprocess.Popen', side_effect=launch_ready):
            result = self.run_agent(seconds=.1)
        self.assertEqual(result['terminal']['agent_terminal_status'], 'timeout')
        self.assertIn('partial raw output', (self.directory/'job'/'claude-code.txt').read_text())
        self.assertIsNotNone(result['returncode'])

    def test_process_exit_without_model_result_is_error(self):
        self.fake('print("partial output")\n')
        result = self.run_agent()
        self.assertEqual(result['terminal']['agent_terminal_status'], 'error')
        self.assertEqual(result['returncode'], 0)

    def test_timeout_kills_term_ignoring_descendant_after_leader_exits(self):
        self.cli.write_text('''#!/bin/sh
trap 'exit 0' TERM
/bin/sh -c 'trap "" TERM; echo $$ > "$ADB_WORK/child.pid"; while :; do sleep 1; done' &
echo original-transcript
echo original-stderr >&2
wait
''')
        self.cli.chmod(0o755)
        pid_path = self.directory/'work/child.pid'
        try:
            with patch('local_analysis.terminate_process_group',
                       side_effect=lambda proc: terminate_process_group(proc, .2, 2)):
                result = self.run_agent(seconds=5)
            self.assertEqual(result['terminal']['agent_terminal_status'], 'timeout')
            self.assertTrue(pid_path.is_file())
            status = subprocess.run(['ps', '-o', 'stat=', '-p', pid_path.read_text().strip()],
                                    capture_output=True, text=True).stdout.strip()
            self.assertTrue(not status or status.startswith('Z'), status)
            receipt = json.loads((self.directory/'job/local-result.json').read_text())
            self.assertEqual(receipt['process_cleanup']['signals_sent'], ['SIGTERM', 'SIGKILL'])
            self.assertIn('original-transcript', (self.directory/'job/claude-code.txt').read_text())
            self.assertIn('original-stderr', (self.directory/'job/stderr.txt').read_text())
        finally:
            if pid_path.exists():
                try:
                    os.kill(int(pid_path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_launch_error_is_recorded(self):
        with patch('local_analysis.shutil.which', return_value=None):
            result = run_local(self.directory, self.prompt, [self.inputs], 5,
                               'model', 'http://127.0.0.1:12345', 'local-relay', {})
        self.assertEqual(result['terminal']['exception_type'], 'RuntimeError')
        self.assertTrue((self.directory/'job'/'local-result.json').is_file())

    def test_real_sandbox_protects_other_tasks_and_inputs(self):
        protected = self.root/'another-task-secret'
        protected.write_text('should remain unreadable')
        self.fake('''import json,os,socket
from pathlib import Path
p=Path(os.environ['ADB_WORK'])
assert Path(%r).read_text()=='complete unchanged prompt'
for target in [%r]:
 try: Path(target).read_bytes()
 except PermissionError: pass
 else: raise AssertionError('unexpected read access: '+target)
for target in [%r, %r]:
 try: Path(target).write_text('no')
 except PermissionError: pass
 else: raise AssertionError('unexpected write access: '+target)
p.joinpath('output.json').write_text('{}')
s=socket.socket()
try: s.connect(('1.1.1.1',80))
except PermissionError: pass
else: raise AssertionError('unexpected network access')
print(json.dumps({'type':'result','is_error':False}))
''' % (str(self.prompt), str(protected), str(self.prompt), str(protected)))
        result = self.run_agent()
        self.assertEqual(result['terminal']['agent_terminal_status'], 'completed',
                         (self.directory/'job'/'stderr.txt').read_text())
        self.assertEqual(protected.read_text(), 'should remain unreadable')
        self.assertEqual(self.prompt.read_text(), 'complete unchanged prompt')

    def test_import_from_stage_root_does_not_expose_job_files(self):
        self.fake("""import os,json,subprocess,sys
from pathlib import Path
root=Path(os.environ['ADB_WORK']).parent
code="import openpyxl; from pathlib import Path;\\ntry:Path('job/claude-code.txt').read_bytes()\\nexcept PermissionError:pass\\nelse:raise AssertionError('job became readable')"
r=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
assert r.returncode==0,r.stderr
print(json.dumps({'type':'result','is_error':False}))
""")
        result = self.run_agent()
        self.assertEqual(result['terminal']['agent_terminal_status'], 'completed',
                         (self.directory/'job'/'stderr.txt').read_text())

    def test_zsh_large_heredoc_from_read_only_inputs(self):
        self.fake("""import os,json,subprocess,shlex
from pathlib import Path
work=Path(os.environ['ADB_WORK'])
inputs=work.parent/'inputs'
assert os.environ['CLAUDE_CODE_SHELL']=='/bin/zsh'
body='x'*20000
command="cat > "+shlex.quote(os.environ['ADB_RESULT'])+" <<'EOF'\\n"+body+"\\nEOF"
r=subprocess.run([os.environ['CLAUDE_CODE_SHELL'],'-c',command],cwd=inputs,capture_output=True,text=True)
assert r.returncode==0,r.stderr
assert Path(os.environ['ADB_RESULT']).read_text()==body+'\\n'
assert sorted(p.name for p in inputs.iterdir())==['prompt.txt']
try:inputs.joinpath('should-not-write').write_text('blocked')
except PermissionError:pass
else:raise AssertionError('inputs became writable')
print(json.dumps({'type':'result','is_error':False}))
""")
        result = self.run_agent()
        self.assertEqual(result['terminal']['agent_terminal_status'], 'completed',
                         (self.directory/'job'/'stderr.txt').read_text())
        self.assertEqual(self.prompt.read_text(), 'complete unchanged prompt')

    def test_relay_port_allowed_but_other_loopback_ports_denied(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        work = self.directory/'work'
        work.mkdir()
        profile = sandbox_profile([], [work], sys.executable, port)
        code = ("import socket; s=socket.socket(); s.connect(('127.0.0.1',"+str(port)+")); s.close(); "
                "s=socket.socket();\ntry: s.connect(('127.0.0.1',"+str(port+1 if port<65535 else port-1)+"))\n"
                "except PermissionError: pass\nelse: raise AssertionError('other port accessible')")
        r = subprocess.run(['/usr/bin/sandbox-exec','-p',profile,sys.executable,'-c',code],
                           cwd=work, env={'PATH':'/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE':'1'},
                           capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_real_cli_version_starts_with_profile(self):
        cli = Path(shutil.which('claude') or '/usr/local/bin/claude')
        if not cli.exists():
            self.skipTest('local Claude CLI unavailable')
        work = self.directory/'work'
        work.mkdir()
        profile = sandbox_profile([self.inputs], [work], cli, 12345)
        r = subprocess.run(['/usr/bin/sandbox-exec','-p',profile,str(cli),'--version'],
                           cwd=work, env={"PATH":"/usr/bin:/bin", "TMPDIR":str(work), "CLAUDE_CONFIG_DIR":str(work)}, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('Claude Code', r.stdout)


class URLTests(unittest.TestCase):
    def test_only_loopback(self):
        self.assertEqual(_local_url('http://127.0.0.1:12345'), 'http://127.0.0.1:12345')
        for url in ['https://example.com', 'http://127.0.0.1', 'http://user:pass@127.0.0.1:12345']:
            with self.assertRaises(ValueError):
                _local_url(url)


if __name__ == '__main__':
    unittest.main()
