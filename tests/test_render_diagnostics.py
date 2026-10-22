import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import subprocess
import time
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
from render_diagnostics import HTML, build_data, render


class RenderDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ep = self.root/'ep01'
        self.diag = self.ep/'diagnostics'
        self.work = self.diag/'observations/sample-00'
        self.work.mkdir(parents=True)
        self.put(self.ep/'runtime-status.json', {'status': 'running', 'pid': os.getpid()})
        self.put(self.diag/'observations/index.json', {
            'sample_order': list(range(16)), 'status': 'running',
            'workers': [{'sample_id': i, 'status': 'pending'} for i in range(16)]})

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_live_written_does_not_imply_commit_or_terminal(self):
        self.put(self.work/'started.json', {'env': {'SECRET': 'not for display'}})
        self.put(self.work/'work/output.json', {'observations': [{'finding': 'f', 'evidence': ['event:3']}]})
        data = build_data(self.ep)
        self.assertEqual(data['progress'], {'total': 16, 'written': 1, 'committed': 0,
                         'failed': 0, 'reused': 0, 'stage': '单路轨迹分析', 'batch_status': 'running'})
        self.assertEqual(data['workers'][0]['status'], 'running')
        self.assertEqual(data['workers'][0]['terminal'], {})
        self.assertEqual(data['workers'][0]['artifact_status'], 'written_uncommitted')
        self.assertNotIn('SECRET', json.dumps(data))
        self.assertIsNone(data['metrics']['value'])
        self.assertEqual(data['stages']['attribution']['status'], 'not_started')

    def test_committed_output_uses_index_paths_and_keeps_raw_separate(self):
        raw = self.work/'work/specific.json'
        derived = self.work/'derived.json'
        self.put(raw, {'observations': [{'finding': 'model words', 'evidence': ['event:3']}]})
        self.put(derived, {'observations': [{'id': 's00-o1', 'statement': 'model words'}]})
        self.put(self.work/'state.json', {'status': 'completed', 'artifact_status': 'committed',
                 'commit_id': 'abc', 'raw_output_path': str(raw), 'output_path': str(derived)})
        data = build_data(self.ep)
        self.assertEqual(data['progress']['committed'], 1)
        worker = data['workers'][0]
        self.assertNotIn('id', worker['raw']['value']['observations'][0])
        self.assertEqual(worker['derived']['value']['observations'][0]['id'], 's00-o1')

    def test_failed_output_retains_partial_raw_and_error(self):
        output = self.work/'work/output.json'
        output.parent.mkdir()
        output.write_text('{"observations": [', encoding='utf-8')
        self.put(self.work/'state.json', {'status': 'failed', 'artifact_status': 'rejected',
                 'error_type': 'ValueError', 'error': 'invalid output'})
        self.put(self.work/'finished.json', {'terminal': {'agent_terminal_status': 'timeout',
                 'exception_type': 'AgentTimeoutError'}, 'extra_env': {'SECRET': 'hidden'}})
        data = build_data(self.ep)
        self.assertEqual(data['progress']['failed'], 1)
        self.assertEqual(data['workers'][0]['raw']['raw'], '{"observations": [')
        self.assertEqual(data['workers'][0]['terminal']['agent_terminal_status'], 'timeout')
        self.assertNotIn('SECRET', json.dumps(data))

    def test_full_model_text_cannot_escape_embedded_data(self):
        finding = '</script><script>alert("x")</script>&中文' + 'long finding ' * 2000
        self.put(self.work/'work/output.json', {'observations': [{'finding': finding, 'evidence': []}]})
        page = render(self.ep, self.root/'view')
        text = page.read_text()
        self.assertNotIn(finding, text)
        embedded = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)[1]
        restored = json.loads(embedded)
        self.assertEqual(restored['workers'][0]['raw']['value']['observations'][0]['finding'], finding)
        self.assertNotIn('innerHTML', text)
        self.assertEqual(json.loads((page.parent/'data.json').read_text()), restored)

    def test_complete_task_manifest_and_index_visible(self):
        self.put(self.work/'inputs/manifest.json', {'files': [{'relative_path': 'attempt.md', 'bytes': 100000}]})
        self.put(self.work/'inputs/evidence_index.json', {'events': [{'event': 3, 'line_start': 8, 'line_end': 50}]})
        task = self.work/'inputs/task/instruction.md'
        task.parent.mkdir()
        task.write_text('完整原题\nNo truncation.', encoding='utf-8')
        data = build_data(self.ep)
        self.assertEqual(data['task']['text'], '完整原题\nNo truncation.')
        self.assertEqual(data['workers'][0]['evidence_index']['value']['events'][0]['event'], 3)

    def test_full_trajectory_and_actual_prompt_not_truncated(self):
        trace = 'trace <script>not executable</script>' * 10000
        path = self.work/'inputs/attempt.md'
        path.parent.mkdir()
        path.write_text(trace, encoding='utf-8')
        prompt = self.work/'task/instruction.md'
        prompt.parent.mkdir()
        prompt.write_text('Actual analyst prompt', encoding='utf-8')
        worker = build_data(self.ep)['workers'][0]
        self.assertEqual(worker['input_texts']['attempt']['text'], trace)
        self.assertEqual(worker['input_texts']['agent_prompt']['text'], 'Actual analyst prompt')
        self.assertFalse(worker['input_texts']['original_system_prompt']['exists'])

    def test_external_index_reference_not_read(self):
        secret = self.root/'secrets.json'
        self.put(secret, {'password': 'must not read'})
        self.put(self.work/'state.json', {'raw_output_path': str(secret)})
        data = build_data(self.ep)
        self.assertNotIn('must not read', json.dumps(data))
        self.assertFalse(data['workers'][0]['raw']['exists'])

    def test_missing_audit_envelope_does_not_advance_stage(self):
        self.put(self.diag/'audit.json', {'status': 'missing', 'payload': None})
        self.put(self.diag/'metrics.json', {'status': 'missing', 'metrics': {'accuracy': None}})
        self.assertEqual(build_data(self.ep)['progress']['stage'], '单路轨迹分析')

    def test_dead_controller_marks_active_workers_interrupted_without_rewriting_records(self):
        self.put(self.work/'started.json', {})
        self.put(self.diag/'attribution/started.json', {})
        with patch('render_diagnostics.os.kill', side_effect=ProcessLookupError):
            data = build_data(self.ep)
        self.assertEqual(data['runtime']['status'], 'interrupted')
        self.assertEqual(data['runtime']['recorded_status'], 'running')
        self.assertEqual(data['progress']['batch_status'], 'running')
        self.assertEqual(data['workers'][0]['status'], 'interrupted')
        self.assertEqual(data['workers'][0]['recorded_status'], 'running')
        self.assertEqual(data['stages']['attribution']['status'], 'interrupted')
        self.assertEqual(json.loads((self.ep/'runtime-status.json').read_text())['status'], 'running')

    def test_legacy_run_does_not_claim_live_status(self):
        (self.ep/'runtime-status.json').unlink()
        self.put(self.work/'started.json', {})
        data = build_data(self.ep)
        self.assertEqual(data['runtime']['status'], 'unknown')
        self.assertEqual(data['workers'][0]['status'], 'unknown')
        self.assertEqual(data['workers'][0]['recorded_status'], 'running')

    def test_terminal_runtime_and_worker_states_not_overwritten_by_pid(self):
        for status in ('completed', 'failed', 'interrupted'):
            self.put(self.ep/'runtime-status.json', {'status': status, 'pid': os.getpid()})
            self.put(self.work/'state.json', {'status': 'completed', 'artifact_status': 'committed'})
            with patch('render_diagnostics.os.kill') as kill:
                data = build_data(self.ep)
            kill.assert_not_called()
            self.assertEqual(data['runtime']['status'], status)
            self.assertEqual(data['workers'][0]['status'], 'completed')

    def test_nonlocal_or_invalid_pid_cannot_certify_running(self):
        for record in ({'pid': os.getpid(), 'hostname': 'other-host.invalid'},
                       {'pid': -1}, {'pid': True}, {}):
            self.put(self.ep/'runtime-status.json', dict(status='running', **record))
            with patch('render_diagnostics.os.kill') as kill:
                data = build_data(self.ep)
            kill.assert_not_called()
            self.assertEqual(data['runtime']['status'], 'unknown')

    def test_auto_refresh_only_for_confirmed_active_snapshot(self):
        page = render(self.ep, self.root/'view', refresh_seconds=10)
        self.assertIn('setTimeout(()=>location.reload(), 10000)', page.read_text())
        self.put(self.ep/'runtime-status.json', {'status': 'failed', 'pid': os.getpid()})
        page = render(self.ep, self.root/'view', refresh_seconds=10)
        self.assertNotIn('setTimeout', page.read_text())

    def test_watcher_leaves_final_interrupted_snapshot_when_controller_exits(self):
        controller = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: controller.poll() is None and controller.kill())
        self.put(self.ep/'runtime-status.json', {'status': 'running', 'pid': controller.pid})
        out = self.root/'watch-view'
        watcher = subprocess.Popen([sys.executable,
            str(Path(__file__).resolve().parents[1]/'harness/render_diagnostics.py'),
            '--episode', str(self.ep), '--output-dir', str(out), '--watch', '--interval', '0.05'],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: watcher.poll() is None and watcher.kill())
        deadline = time.monotonic() + 5
        while not (out/'index.html').is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((out/'index.html').is_file())
        controller.terminate()
        controller.wait(timeout=5)
        _, error = watcher.communicate(timeout=5)
        self.assertEqual(watcher.returncode, 0, error)
        self.assertEqual(json.loads((out/'data.json').read_text())['runtime']['status'], 'interrupted')
        self.assertNotIn('setTimeout', (out/'index.html').read_text())

    def test_recovery_history_preserves_retry_and_reuse_without_resetting_counts(self):
        recovery = {'recorded_at': '2026-09-23T13:45:02+00:00',
                    'repeated_sample_ids': [0], 'reused_sample_ids': list(range(1, 16)),
                    'reason': 'runtime fix; preserve prior failed invocation',
                    'automatic_model_retry': False, 'moved_records': [{'archived_path': 'recovery-01/sample-00'}]}
        self.put(self.diag/'recovery-01/recovery.json', recovery)
        self.put(self.diag/'recovery-02/recovery.json', {'repeated_sample_ids': [0], 'reason': 'second recovery'})
        self.put(self.work/'state.json', {'status': 'completed', 'artifact_status': 'committed'})
        data = build_data(self.ep)
        self.assertEqual([item['value'] for item in data['recoveries']],
                         [recovery, {'repeated_sample_ids': [0], 'reason': 'second recovery'}])
        self.assertEqual(data['progress']['committed'], 1)
        self.assertEqual(data['workers'][0]['status'], 'completed')
        page = render(self.ep, self.root/'recovery-view')
        text = page.read_text()
        embedded = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)[1]
        self.assertEqual(json.loads(embedded)['recoveries'][0]['value'], recovery)
        self.assertIn('不代表首次 16 路全部成功', text)
        self.assertIn('重跑 sample_ids', text)
        self.assertIn('复用 sample_ids', text)

    def test_recovery_missing_or_malformed_does_not_invent_history(self):
        self.assertEqual(build_data(self.ep)['recoveries'], [])
        path = self.diag/'recovery-01/recovery.json'
        path.parent.mkdir()
        path.write_text('{partial', encoding='utf-8')
        recovery = build_data(self.ep)['recoveries'][0]
        self.assertIsNone(recovery['value'])
        self.assertEqual(recovery['raw'], '{partial')
        self.assertIn('read_error', recovery)

    def legacy_trial(self, exception=None, finished=True):
        directory = self.diag/'design'
        trial = directory/'job/run/trial/result.json'
        payload = {'exception_info': exception, 'agent_execution': {'finished_at': '2026-09-23T14:31:36Z'}}
        if finished:
            payload['finished_at'] = '2026-09-23T14:31:40Z'
        self.put(trial, payload)
        self.put(directory/'finished.json', {'agent_execution': {'status': 'completed',
                 'source': {'path': str(trial), 'sha256': hashlib.sha256(trial.read_bytes()).hexdigest()}}})
        return trial

    def test_legacy_verified_trial_completion_and_timeout_keep_artifact_status_separate(self):
        for exception, status in [(None, 'completed'),
            ({'exception_type': 'AgentTimeoutError', 'exception_message': 'agent exceeded 1800s'}, 'timeout'),
            ({'exception_type': 'AgentExecutionError', 'exception_message': 'agent failed'}, 'error')]:
            self.legacy_trial(exception)
            self.put(self.diag/'design.json', {'targets': []})
            stage = build_data(self.ep)['stages']['design']
            self.assertEqual(stage['status'], 'artifact_available')
            self.assertEqual(stage['terminal']['agent_terminal_status'], status)
            self.assertEqual(stage['terminal']['exception_type'], exception['exception_type'] if exception else None)
            self.assertEqual(stage['terminal']['finished_at'], '2026-09-23T14:31:40Z')

    def test_legacy_trial_tampering_or_missing_terminal_remains_unknown(self):
        trial = self.legacy_trial()
        trial.write_text('{}', encoding='utf-8')
        terminal = build_data(self.ep)['stages']['design']['terminal']
        self.assertEqual(terminal['agent_terminal_status'], 'unknown')
        self.assertIn('SHA-256', terminal['verification_error'])
        self.legacy_trial(finished=False)
        terminal = build_data(self.ep)['stages']['design']['terminal']
        self.assertEqual(terminal['agent_terminal_status'], 'unknown')
        self.assertIn('结束时间', terminal['verification_error'])
        self.put(self.diag/'design/finished.json', {'harbor_rc': 0})
        terminal = build_data(self.ep)['stages']['design']['terminal']
        self.assertEqual(terminal['agent_terminal_status'], 'unknown')
        self.assertIn('来源', terminal['verification_error'])

    def test_explicit_terminal_remains_authoritative_without_legacy_source_access(self):
        self.put(self.diag/'design/finished.json', {'terminal': {'agent_terminal_status': 'timeout'},
                 'agent_execution': {'source': {'path': '/not-readable', 'sha256': 'bad'}}})
        terminal = build_data(self.ep)['stages']['design']['terminal']
        self.assertEqual(terminal, {'agent_terminal_status': 'timeout'})

    def test_evidence_directory_cannot_be_output(self):
        with self.assertRaises(ValueError):
            render(self.ep, self.ep/'view')

    def render_training_plan(self, name, raw, derived):
        return self.render_browser_function('trainingPlan', {'raw': {'value': raw}, 'derived': {'value': derived}}, name)

    def render_browser_function(self, function, *args):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node is required to execute browser rendering assertions')
        # Exercise the actual DOM renderer without adding a browser dependency.
        helpers = HTML.split('function el(', 1)[1].split("document.getElementById('time')", 1)[0]
        script = '''const document={createElement(tag){return {tag,children:[],
          appendChild(child){this.children.push(child)}}}};
        ''' + 'function el(' + helpers
        script += '\nconst root=document.createElement("section");\n'
        script += function+'(root,'+','.join(json.dumps(arg) for arg in args)+');\n'
        script += 'console.log(JSON.stringify(root));'
        result = subprocess.run([node], input=script, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_training_goals_display_model_order_derived_ids_and_full_fields(self):
        goals = [{'ability': 'Read complete input', 'claim_ids': ['c1', 'c3'],
                  'rationale': 'Known recoverable input gap', 'success_signal': 'Fetch decisive omitted rows',
                  'difficulty_controls': ['Small output', 'Simple grouping']},
                 {'ability': 'Write valid dimensions', 'claim_ids': ['c5'],
                  'rationale': 'Execution prerequisite', 'success_signal': 'Correct matrix width',
                  'difficulty_controls': ['Provide complete source']}]
        derived = {'training_goals': [dict(goal, id='g'+str(i+1)) for i, goal in enumerate(goals)]}
        raw = {'training_goals': goals}
        self.put(self.diag/'attribution/work/output.json', raw)
        self.put(self.diag/'attribution.json', derived)
        stage = build_data(self.ep)['stages']['attribution']
        tree = self.render_training_plan('attribution', stage['raw']['value'], stage['derived']['value'])
        ordered = [child for child in tree['children'] if child['tag'] == 'ol'][0]
        titles = [item['children'][0]['textContent'] for item in ordered['children']]
        self.assertEqual(titles, ['g1 · Read complete input', 'g2 · Write valid dimensions'])
        displayed = json.dumps(tree, ensure_ascii=False)
        for goal in goals:
            for key in ('ability', 'rationale', 'success_signal'):
                self.assertIn(goal[key], displayed)
        self.assertIn('Small output', displayed)
        self.assertIn('c3', displayed)
        self.assertIn('目标 ID 由脚本生成', displayed)
        self.assertNotIn('id', stage['raw']['value']['training_goals'][0])

    def test_training_plan_legacy_missing_and_raw_only_have_honest_provenance(self):
        tree = self.render_training_plan('attribution', {'claims': [{'id': 'c1'}]}, {'claims': []})
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('不从缺陷列表推断优先级', displayed)
        self.assertNotIn('"tag": "ol"', displayed)
        tree = self.render_training_plan('attribution', {'training_goals': [{'ability': 'Read'}]}, None)
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('尚无派生目标 ID', displayed)
        self.assertNotIn('g1', displayed)
        tree = self.render_training_plan('design', {'targets': [{'claim_id': 'c1'}]}, None)
        self.assertIn('不从 targets 的数量或顺序推断', json.dumps(tree, ensure_ascii=False))

    def test_design_primary_goal_and_reason_display_raw_choice_without_rewriting(self):
        raw = {'primary_goal_id': 'g2', 'selection_reason': 'One main difficulty <script> literal'}
        tree = self.render_training_plan('design', raw, dict(raw, primary_goal_id='g1'))
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('主目标：g2', displayed)
        self.assertNotIn('主目标：g1', displayed)
        self.assertIn(raw['selection_reason'], displayed)
        self.assertIn('模型原始输出', displayed)

    def reused_record(self):
        from stage_records import write_stage_commit
        source = (self.root/'old-episode/diagnostics').resolve()
        worker = source/'observations/sample-00'
        raw = {'observations': []}
        derived = {'schema_version': 'trajectory-observations-v1', 'sample_id': 0,
                   'observations': [], 'evidence_limitations': []}
        references = []
        for relative, value in [('work/output.json', raw), ('observations.json', derived),
                                ('inputs/manifest.json', {'files': []}), ('inputs/evidence_index.json', {'events': []})]:
            path = worker/relative
            self.put(path, value)
            references.append(path)
        for relative, value in [('inputs/attempt.md', 'complete old trajectory'),
                                ('inputs/task/instruction.md', 'original task')]:
            path = worker/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
            references.append(path)
        commit = write_stage_commit(self.diag/'records', 'observations', derived,
                   references=references, metadata={'stage_execution': {'terminal': {'agent_terminal_status': 'completed'}}})
        record = {'sample_id': 0, 'status': 'completed', 'artifact_status': 'committed',
                  'commit_id': commit['commit_id'], 'reused_commit_id': commit['commit_id'],
                  'reused_from': str(source), 'commit_path': str((self.diag/'records/commits'/f"{commit['commit_id']}.json").resolve()),
                  'raw_output_path': str(worker/'work/output.json'), 'output_path': str(worker/'observations.json'),
                  'input_manifest': str(worker/'inputs/manifest.json')}
        self.put(self.diag/'observations/index.json', {'sample_order': [0], 'workers': [record]})
        return record, commit

    def test_reused_analysis_reads_verified_archive_and_not_changed_external_originals(self):
        record, _ = self.reused_record()
        self.put(Path(record['raw_output_path']), {'secret': 'changed external original'})
        data = build_data(self.ep)
        worker = data['workers'][0]
        self.assertEqual(data['progress']['reused'], 1)
        self.assertEqual(worker['status'], 'reused')
        self.assertEqual(worker['raw']['value'], {'observations': []})
        self.assertEqual(worker['input_texts']['attempt']['text'], 'complete old trajectory')
        self.assertEqual(data['task']['text'], 'original task')
        self.assertEqual(worker['terminal']['agent_terminal_status'], 'completed')
        self.assertNotIn('changed external original', json.dumps(data))

    def test_reused_analysis_rejects_forged_pointer_and_tampered_archive(self):
        record, commit = self.reused_record()
        original_path = record['raw_output_path']
        record['raw_output_path'] = str(self.root/'secret.json')
        self.put(Path(record['raw_output_path']), {'secret': 'must not read'})
        self.put(self.diag/'observations/index.json', {'sample_order': [0], 'workers': [record]})
        worker = build_data(self.ep)['workers'][0]
        self.assertEqual(worker['status'], 'failed')
        self.assertEqual(worker['error_type'], 'ReuseVerificationError')
        self.assertNotIn('must not read', json.dumps(worker))
        record['raw_output_path'] = original_path
        self.put(self.diag/'observations/index.json', {'sample_order': [0], 'workers': [record]})
        archived = self.diag/'records'/commit['references'][0]['path']
        archived.write_text('tampered')
        data = build_data(self.ep)
        self.assertEqual(data['progress']['reused'], 0)
        self.assertEqual(data['workers'][0]['artifact_status'], 'rejected')
        self.assertIn('hash mismatch', data['workers'][0]['error'])

    def test_author_completed_does_not_complete_running_host_validation(self):
        self.put(self.diag/'design.json', {'targets': []})
        self.put(self.diag/'design/finished.json', {'terminal': {'agent_terminal_status': 'completed'}})
        validation = {'schema_version': 'host-validation-v1', 'status': 'running',
                      'candidate_sha256': 'candidate-version', 'requested_attempts': 3,
                      'completed_attempts': 0, 'solved_attempts': 0, 'quality_approved': None,
                      'oracle': {'id': 'oracle-1', 'kind': 'oracle', 'status': 'completed', 'reward': 1},
                      'attempts': [{'id': 'qwen-1', 'kind': 'trainee', 'status': 'running', 'reward': None}]}
        self.put(self.diag/'validation.json', validation)
        data = build_data(self.ep)
        self.assertEqual(data['stages']['design']['terminal']['agent_terminal_status'], 'completed')
        self.assertEqual(data['validation']['value'], validation)
        self.assertEqual(data['progress']['stage'], '脚本执行验证')
        tree = self.render_browser_function('showValidation', data['validation'])
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('执行中', displayed)
        self.assertIn('得分：未知（无有效评分）', displayed)
        self.assertIn('参考解 · oracle-1', displayed)
        self.assertIn('目标模型 · qwen-1', displayed)
        self.assertIn('待完成', displayed)

    def test_author_timeout_and_completed_host_validation_remain_independent(self):
        self.legacy_trial({'exception_type': 'AgentTimeoutError', 'exception_message': 'author timeout'})
        self.put(self.diag/'design.json', {'targets': []})
        self.put(self.diag/'validation.json', {'schema_version': 'host-validation-v1', 'status': 'completed',
                 'requested_attempts': 1, 'completed_attempts': 1, 'solved_attempts': 0,
                 'quality_approved': None, 'oracle': {'id': 'ref', 'kind': 'oracle', 'status': 'completed', 'reward': 1},
                 'attempts': [{'id': 'qwen', 'kind': 'trainee', 'status': 'completed', 'reward': 0}]})
        data = build_data(self.ep)
        self.assertEqual(data['stages']['design']['terminal']['agent_terminal_status'], 'timeout')
        self.assertEqual(data['validation']['value']['status'], 'completed')
        self.assertIsNone(data['validation']['value']['quality_approved'])
        tree = self.render_browser_function('showValidation', data['validation'])
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('执行检查已完成', displayed)
        self.assertIn('得分：0', displayed)
        self.assertIn('不判定归因', displayed)
        self.assertNotIn('全面验证通过', displayed)
        self.assertNotIn('待完成', displayed)

    def test_interrupted_validation_missing_scores_stay_unknown_and_old_runs_unstarted(self):
        missing = build_data(self.ep)['validation']
        tree = self.render_browser_function('showValidation', missing)
        self.assertIn('未开始', json.dumps(tree, ensure_ascii=False))
        self.put(self.diag/'validation.json', {'schema_version': 'host-validation-v1', 'status': 'incomplete',
                 'requested_attempts': 1, 'completed_attempts': 0, 'solved_attempts': None, 'oracle': None,
                 'attempts': [{'id': 'qwen', 'kind': 'trainee', 'status': 'interrupted', 'reward': None,
                               'error': 'process stopped'}]})
        validation = build_data(self.ep)['validation']
        self.assertIsNone(validation['value']['attempts'][0]['reward'])
        tree = self.render_browser_function('showValidation', validation)
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('执行不完整', displayed)
        self.assertIn('做对 未知 次', displayed)
        self.assertIn('得分：未知（无有效评分）', displayed)
        self.assertNotIn('得分：0', displayed)
        self.assertIn('process stopped', displayed)

    def test_validation_stage_error_is_visible_without_trial_error(self):
        self.put(self.diag/'validation.json', {'schema_version': 'host-validation-v1', 'status': 'failed',
                 'oracle': None, 'attempts': [], 'error': {'type': 'OSError', 'message': 'worker archive unavailable'}})
        tree = self.render_browser_function('showValidation', build_data(self.ep)['validation'])
        displayed = json.dumps(tree, ensure_ascii=False)
        self.assertIn('验证阶段错误', displayed)
        self.assertIn('worker archive unavailable', displayed)

    def test_independent_validation_refresh_uses_local_liveness_or_explicit_remote_record(self):
        self.put(self.ep/'runtime-status.json', {'status': 'completed'})
        validation = {'schema_version': 'host-validation-v1', 'status': 'running',
                      'pid': os.getpid(), 'hostname': socket.gethostname()}
        self.put(self.diag/'validation.json', validation)
        page = render(self.ep, self.root/'validation-view', refresh_seconds=5)
        self.assertIn('setTimeout(()=>location.reload(), 5000)', page.read_text())
        with patch('render_diagnostics.os.kill', side_effect=ProcessLookupError):
            page = render(self.ep, self.root/'validation-view', refresh_seconds=5)
        self.assertNotIn('setTimeout', page.read_text())
        runtime = json.loads((page.parent/'data.json').read_text())['validation']['runtime']
        self.assertEqual(runtime['status'], 'interrupted')
        self.assertEqual(runtime['recorded_status'], 'running')
        validation['hostname'] = 'docker-worker-other-host'
        self.put(self.diag/'validation.json', validation)
        with patch('render_diagnostics.os.kill') as kill:
            page = render(self.ep, self.root/'validation-view', refresh_seconds=5)
        kill.assert_not_called()
        self.assertIn('setTimeout', page.read_text())
        document = json.loads((page.parent/'data.json').read_text())['validation']
        self.assertEqual(document['runtime']['status'], 'unknown')
        tree = self.render_browser_function('showValidation', document)
        self.assertIn('仅显示记录状态', json.dumps(tree, ensure_ascii=False))

    def test_watch_continues_after_episode_completion_until_independent_validation_exits(self):
        self.put(self.ep/'runtime-status.json', {'status': 'completed'})
        controller = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: controller.poll() is None and controller.kill())
        self.put(self.diag/'validation.json', {'schema_version': 'host-validation-v1', 'status': 'running',
                 'pid': controller.pid, 'hostname': socket.gethostname()})
        out = self.root/'independent-watch'
        watcher = subprocess.Popen([sys.executable,
            str(Path(__file__).resolve().parents[1]/'harness/render_diagnostics.py'),
            '--episode', str(self.ep), '--output-dir', str(out), '--watch', '--interval', '0.05'],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: watcher.poll() is None and watcher.kill())
        deadline = time.monotonic() + 5
        while not (out/'index.html').is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((out/'index.html').is_file())
        self.assertIsNone(watcher.poll())
        self.assertIn('setTimeout', (out/'index.html').read_text())
        controller.terminate()
        controller.wait(timeout=5)
        _, error = watcher.communicate(timeout=5)
        self.assertEqual(watcher.returncode, 0, error)
        data = json.loads((out/'data.json').read_text())
        self.assertEqual(data['runtime']['status'], 'completed')
        self.assertEqual(data['validation']['runtime']['status'], 'interrupted')
        self.assertNotIn('setTimeout', (out/'index.html').read_text())


if __name__ == '__main__':
    unittest.main()
