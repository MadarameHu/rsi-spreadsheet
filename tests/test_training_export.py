"""Offline export checks: raw-source bindings, failed-demo isolation and ordering."""
import copy
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
import quality_gate
from stage_records import snapshot_tree, verify_snapshot, write_stage_commit
from training_export import export_candidate, verify_export, _encoded, _validate_recorded_history


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ep = self.root/'episode'
        self.output = self.root/'exports'
        source = self.root/'authored'
        self.put(source/'instruction.md', 'Count visits.')
        self.put(source/'environment/input.xlsx', b'initial-workbook-fixture')
        self.put(source/'environment/Dockerfile', 'FROM python:3.12\n')
        self.put(source/'tests/target.xlsx', b'ground-truth-fixture')
        self.put(source/'tests/verify.py', 'print("verifier")\n')
        self.put(source/'solution/solve.py', 'print("oracle script")\n')
        self.put(source/'task.toml', '[metadata]\npass_threshold = 1.0\n')
        self.manifest = snapshot_tree(source, self.ep/'candidate')
        self.put(self.ep/'diagnostics/synthesis.json', {'stage_commit_dir': str(self.ep/'diagnostics/records')})
        self.gate = {'schema_version': 'quality-gate-v1', 'status': 'accepted', 'eligible_for_export': True,
                     'quality_approved': True, 'candidate_manifest': self.manifest,
                     'candidate_sha256': self.manifest['tree_sha256'], 'checks': [], 'issues': []}
        self.gate_mock = patch.object(quality_gate, 'evaluate_quality', side_effect=lambda ep: copy.deepcopy(self.gate))
        self.gate_mock.start()
        self.addCleanup(self.gate_mock.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (dict, list)):
            path.write_text(json.dumps(value, ensure_ascii=False))
        elif isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)
        return path

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def demo(self, request_id='native-one', reward=1):
        root = self.root/'demos'/request_id
        case = root/'case'
        candidate = snapshot_tree(self.ep/'candidate', case/'candidate')
        self.put(case/'host-candidate.json', candidate)
        trial_path = case/'job/run/trial/result.json'
        agent = trial_path.parent/'agent'
        native_job = root/'native-job'
        request = {'request_id': request_id, 'model': 'qwen', 'seed': 1, 'instruction': 'Count visits.'}
        system = self.put(root/'system.md', 'Use spreadsheet tools.')
        runner = self.put(root/'runner.py', '''TOOL_RESPONSE_CHARS = 10240
def tool_markup_segments(content):
    return []
def strip_tool_call_markup(content):
    return content
def normalize_api_history(state):
    return None
def truncate_tool_feedback(text):
    return text
def parse_calls(message, turn):
    return [{'id': c['id'], 'name': c['function']['name'], 'raw_arguments': c['function']['arguments'], 'protocol': 'native'} for c in message.get('tool_calls', [])]
''')
        assistant = {'role': 'assistant', 'content': None, 'tool_calls': [
                     {'id': 'call-1', 'type': 'function', 'function': {'name': 'inspect_range', 'arguments': '{}'}}]}
        response = {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"rows": 3}'}
        final = {'role': 'assistant', 'content': 'Saved workbook.'}
        messages = [{'role': 'system', 'content': system.read_text()}, {'role': 'user', 'content': request['instruction']},
                    assistant, response, final]
        tools = [{'type': 'function', 'function': {'name': 'inspect_range', 'parameters': {'type': 'object'}}}]
        events = [{'event_index': 0, 'step_index': 1, 'event_kind': 'tool_call', 'tool_name': 'inspect_range',
                   'call_id': 'call-1', 'response_event_index': 1},
                  {'event_index': 1, 'step_index': 1, 'event_kind': 'tool_response', 'tool_name': 'inspect_range',
                   'call_id': 'call-1', 'response_to_event_index': 0, 'raw': response['content']}]
        native = {'schema': 'adb-native-execution-v1', 'request_id': request_id, 'model': 'qwen',
                  'execution_completed': True, 'termination_reason': 'model_final', 'runtime_errors': [],
                  'source_runner_sha256': self.sha(runner), 'system_prompt_sha256': self.sha(system),
                  'final_workbook_sha256': hashlib.sha256(b'actual-final-workbook').hexdigest(), 'native_events': events}
        for path in [agent/'bridge-request.json', native_job/'request.json']:
            self.put(path, request)
        for path in [agent/'native-execution.json', native_job/'response.json']:
            self.put(path, native)
        for path in [agent/'bridge-input.xlsx', native_job/'input.xlsx']:
            self.put(path, (self.ep/'candidate/environment/input.xlsx').read_bytes())
        for path in [agent/'bridge-final.xlsx', native_job/'final.xlsx']:
            self.put(path, b'actual-final-workbook')
        sampling = {'extra_body': {}, 'temperature': 0.6, 'seed': 1}
        cfg = {'model': 'qwen', 'sampling': sampling}
        self.put(native_job/'rollout/state.json', {'messages': messages, 'tools': tools, 'events': events,
                 'initial_prompt_message_count': 2, 'initial_run_config': cfg,
                 'active_run_config': cfg, 'config_segments': [{'configuration': cfg}]})
        self.put(native_job/'rollout/result.json', {'events': events})
        for i, message in enumerate([assistant, final], 1):
            payload = dict(model='qwen', messages=messages[:2] if i == 1 else messages[:4], tools=tools,
                           tool_choice='auto', temperature=0.6, stream=False, seed=1)
            fingerprint = hashlib.sha256(_encoded({'payload': payload, 'sampling': sampling})).hexdigest()
            self.put(native_job/f'rollout/turn-{i:04d}.json', {'turn': i, 'request_sha256': fingerprint,
                     'response': {'choices': [{'message': message}]}})
        trial = {'finished_at': '2026-09-24T01:00:00Z', 'exception_info': None,
                 'config': {'task': {'path': str(case/'candidate')}, 'agent': {'name': 'native-qwen'}},
                 'agent_result': {'metadata': {'bridge_request_id': request_id}},
                 'verifier_result': {'rewards': {'reward': reward}}}
        self.put(trial_path, trial)
        return self.put(root/'demonstration.json', {'schema_version': 'native-training-demo-v1',
                         'case_dir': str(case), 'native_job_dir': str(native_job), 'trial_result_path': str(trial_path),
                         'runner_path': str(runner), 'system_prompt_path': str(system)})

    def test_zero_target_success_still_exports_complete_rl_and_pending_sft(self):
        self.put(self.ep/'diagnostics/validation.json', {'solved_attempts': 0, 'completed_attempts': 3})
        before = {p: p.read_bytes() for p in self.ep.rglob('*') if p.is_file()}
        result = export_candidate(self.ep, self.output)
        self.assertEqual(result['rl']['status'], 'exported')
        self.assertEqual(result['sft'], {'status': 'pending', 'count': 0, 'path': None, 'rejections': []})
        self.assertFalse((Path(result['output_dir'])/'sft.jsonl').exists())
        task = Path(result['rl']['path'])
        for relative in ['instruction.md', 'environment/input.xlsx', 'environment/Dockerfile',
                         'tests/target.xlsx', 'tests/verify.py', 'solution/solve.py']:
            self.assertEqual((task/relative).read_bytes(), (self.ep/'candidate'/relative).read_bytes())
        verify_snapshot(Path(result['output_dir'])/'rl/candidate-manifest.json')
        self.assertEqual(before, {p: p.read_bytes() for p in self.ep.rglob('*') if p.is_file()})
        self.assertEqual(export_candidate(self.ep, self.output), result)

    def test_rejected_or_unverified_gate_never_exports_training_payload(self):
        for status, eligible in [('blocked', False), ('rejected', False), ('accepted', False)]:
            self.gate.update(status=status, eligible_for_export=eligible)
            result = export_candidate(self.ep, self.output)
            self.assertEqual(result['status'], 'blocked')
            self.assertIsNone(result['output_dir'])
            self.assertFalse(self.output.exists())

    def parallel_evidence(self):
        from host_validation import digest
        raw = self.put(self.ep/'audit/reviews/claim-1/job/turn-0001.json',
                       {'response': 'complete unedited reviewer response'})
        feedback = self.put(self.ep/'audit/reviews/probe/job/tool-feedback.json',
                            {'feedback': 'complete probe tool response'})
        payload = {key: [] for key in ('claims', 'mappings', 'task_checks', 'probes')}
        commit = write_stage_commit(self.ep/'diagnostics/records', 'audit', payload,
            references=[raw], expected_units=payload,
            metadata={'audit_envelope': {'output_hashes': {str(p): self.sha(p) for p in (raw, feedback)}}})
        self.gate.update(review_mode='parallel', audit_commit=commit['commit_id'])
        root = self.ep/'diagnostics/validation'
        request = {'candidate_sha256': self.manifest['tree_sha256'], 'episode': str(self.ep)}
        request['request_id'] = digest(request)
        self.put(root/'request.json', request)
        case = self.ep/'oracle-case'
        self.put(case/'job/result.json', {'reward': 1})
        self.put(case/'job/stdout.txt', 'complete original reference execution')
        (case/'empty-artifacts').mkdir()
        snapshot = snapshot_tree(case, root/'reference-case')
        self.put(root/'reference-case-manifest.json', snapshot)
        receipt = dict(schema_version='reference-evidence-v1', request_id=request['request_id'],
            request_sha256=self.sha(root/'request.json'), candidate_sha256=self.manifest['tree_sha256'],
            attribution_commit=None, design_commit=None, case_path=str(case), snapshot=snapshot,
            result={'status': 'completed', 'reward': 1})
        receipt['reference_evidence_id'] = digest(receipt)
        self.put(root/'reference-evidence.json', receipt)
        return raw, feedback, root

    def test_parallel_export_preserves_raw_review_and_reference_and_moves_independently(self):
        raw, feedback, root = self.parallel_evidence()
        result = export_candidate(self.ep, self.output)
        destination = Path(result['output_dir'])
        mapping = json.loads((destination/'provenance/source-paths.json').read_text())
        for source in (raw, feedback, root/'reference-evidence.json', root/'request.json',
                       root/'reference-case/job/result.json', root/'reference-case/job/stdout.txt'):
            self.assertEqual((destination/mapping[str(source)]).read_bytes(), source.read_bytes())
        moved = self.root/'moved-export'
        shutil.copytree(destination, moved)
        self.ep.rename(self.root/'unavailable-original-episode')
        verify_export(moved)
        (moved/mapping[str(feedback)]).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            verify_export(moved)

    def test_missing_or_changed_parallel_evidence_refuses_export(self):
        raw, feedback, root = self.parallel_evidence()
        sources = [feedback, self.ep/'diagnostics/records/artifacts'/self.sha(raw),
                   root/'reference-case/job/result.json', root/'reference-case-manifest.json',
                   root/'reference-evidence.json', root/'request.json']
        for source in sources:
            original = source.read_bytes()
            for mode in ('missing', 'changed'):
                with self.subTest(source=source.name, mode=mode):
                    if mode == 'missing':
                        source.unlink()
                    else:
                        source.write_text('{}')
                    with self.assertRaises((ValueError, OSError, KeyError)):
                        export_candidate(self.ep, self.output)
                    self.assertFalse(self.output.exists())
                    source.write_bytes(original)

    def test_repair_export_contains_parent_candidate_feedback_and_raw_audit(self):
        raw, feedback, root = self.parallel_evidence()
        parent = self.ep
        old_job = {'id': 'parent-job', 'original': 'task-1', 'analysis_identity': 'analysis-1', 'goal_id': 'goal-1'}
        self.put(parent/'diagnostics/candidate_job.json', old_job)
        # The mocked quality gate stands in for commit validation; repair still
        # requires the parent's actual design file for its read-only author input.
        self.put(parent/'diagnostics/design.json', {'primary_goal_id': 'goal-1'})
        parent_gate = dict(self.gate, status='rejected', eligible_for_export=False)
        frozen = parent/'repair-feedback'
        self.put(frozen/'quality.json', parent_gate)
        self.put(frozen/'audit.json', {'work_dir': str(parent/'audit')})
        child = self.root/'repaired-episode'
        child_manifest = snapshot_tree(parent/'candidate', child/'candidate')
        child_gate = dict(self.gate, candidate_manifest=child_manifest)
        child_gate.pop('audit_commit')
        child_gate.pop('review_mode')
        job = dict(old_job, id='new-job', repair={
            'schema_version': 'candidate-repair-v1', 'parent_episode': str(parent),
            'parent_job_id': old_job['id'], 'candidate_sha256': parent_gate['candidate_sha256'],
            'audit_commit': parent_gate['audit_commit'], 'feedback_dir': str(frozen),
            'feedback_hashes': {name: self.sha(frozen/name) for name in ('quality.json', 'audit.json')}})
        self.put(child/'diagnostics/candidate_job.json', job)
        self.put(child/'diagnostics/synthesis.json', {'stage_commit_dir': str(child/'diagnostics/records')})
        self.gate_mock.stop()
        with patch.object(quality_gate, 'evaluate_quality',
                          side_effect=lambda ep, **kw: copy.deepcopy(parent_gate if Path(ep) == parent else child_gate)):
            result = export_candidate(child, self.output)
        destination = Path(result['output_dir'])
        mapping = json.loads((destination/'provenance/source-paths.json').read_text())
        for source in (raw, feedback, frozen/'quality.json', frozen/'audit.json', parent/'candidate/instruction.md',
                       root/'reference-case/job/result.json'):
            self.assertEqual((destination/mapping[str(source)]).read_bytes(), source.read_bytes())
        parent.rename(self.root/'unavailable-parent')
        child.rename(self.root/'unavailable-child')
        verify_export(destination)

    def recovery_fixture(self):
        import host_validation as hv
        import validation_recovery as recovery
        payload = {key: [] for key in ('claims', 'mappings', 'task_checks', 'probes')}
        audit = write_stage_commit(self.ep/'diagnostics/records', 'audit', payload, expected_units=payload)
        self.gate['audit_commit'] = audit['commit_id']
        request = {'schema_version': hv.SCHEMA, 'episode': str(self.ep), 'attempts': 1,
                   'candidate_sha256': self.manifest['tree_sha256']}
        request['request_id'] = hv.digest(request)
        request_path = self.put(self.ep/'diagnostics/validation/request.json', request)
        old = hv.original_case_directory(request, 'trainee', 0)
        candidate = snapshot_tree(self.ep/'candidate', old/'candidate')
        self.put(old/'host-candidate.json', candidate)
        self.put(old/'candidate-version.json', candidate)
        self.put(old/'job/run/trial/result.json', {'finished_at': '2000-01-01T00:00:00+00:00',
            'exception_info': {'exception_type': 'RuntimeError', 'exception_message': 'bridge HTTP 502'},
            'verifier_result': {'rewards': {'reward': None}}})
        self.put(old/'job/run/trial/agent/bridge-request.json', {'request_id': 'failed-bridge-call'})
        self.put(old/'job/raw-output.txt', 'entire unmodified original failure trace')
        (old/'empty-artifacts').mkdir()
        self.put(self.ep/'diagnostics/validation.json', {'request_id': request['request_id'],
            'attempts': [hv.collect_case(old, 'trainee')]})
        raw = self.put(self.root/'inspection.txt', 'No exact remote request directory, worker or pending request.')
        proof = self.put(self.root/'proof.json', dict(schema_version=recovery.PROOF_SCHEMA,
            request_id=request['request_id'], candidate_sha256=request['candidate_sha256'],
            observed_at=hv.now(), remote_host='fixture-remote', remote_job_root='/remote/jobs',
            checks=[dict(index=0, bridge_request_id='failed-bridge-call',
                         remote_job_path='/remote/jobs/failed-bridge-call', remote_job_exists=False,
                         worker_running=False, request_pending=False)],
            evidence=[{'path': str(raw), 'sha256': self.sha(raw)}]))
        return request_path, proof, old, raw

    def test_explicit_recovery_has_new_export_identity_and_complete_offline_evidence(self):
        import validation_recovery as recovery
        request, proof, old, raw = self.recovery_fixture()
        original = export_candidate(self.ep, self.output)
        original_dir = Path(original['output_dir'])
        original_bytes = {p: p.read_bytes() for p in original_dir.rglob('*') if p.is_file()}
        expected_identity = hashlib.sha256(_encoded({'candidate': self.manifest['tree_sha256'],
            'gate': self.gate, 'rows': [], 'rejections': []})).hexdigest()
        self.assertEqual(original_dir.name, expected_identity)
        self.assertNotIn('recovery', original)
        manifest = recovery.prepare_recovery(request, proof, [0], 'Explicit verified nonexecution recovery')
        result = export_candidate(self.ep, self.output)
        destination = Path(result['output_dir'])
        self.assertNotEqual(destination, original_dir)
        self.assertEqual(result['recovery']['recovery_id'], json.loads(manifest.read_text())['recovery_id'])
        self.assertEqual(original_bytes, {p: p.read_bytes() for p in original_dir.rglob('*') if p.is_file()})
        mapping = json.loads((destination/'provenance/source-paths.json').read_text())
        for source in (manifest, proof, raw, request, old/'job/raw-output.txt', old/'job/run/trial/result.json'):
            self.assertEqual((destination/mapping[str(source)]).read_bytes(), source.read_bytes())
        self.put(self.ep/'diagnostics/validation.json', {'status': 'new recovery progress'})
        self.assertEqual(export_candidate(self.ep, self.output), result)
        moved = self.root/'moved-recovery-export'
        shutil.copytree(destination, moved)
        self.ep.rename(self.root/'unavailable-recovery-episode')
        verify_export(moved)
        (moved/'provenance/validation-recovery/inspection-00.raw').unlink()
        with self.assertRaisesRegex(ValueError, 'integrity'):
            verify_export(moved)

    def test_recovery_missing_changed_or_untracked_evidence_refuses_export(self):
        import validation_recovery as recovery
        request, proof, old, raw = self.recovery_fixture()
        manifest = recovery.prepare_recovery(request, proof, [0], 'Explicit verified nonexecution recovery')
        for source in (manifest.parent/'proof.json', manifest.parent/'inspection-00.raw',
                       old/'job/raw-output.txt', manifest.parent.parent/'active-recovery.json'):
            original = source.read_bytes()
            for mode in ('missing', 'changed'):
                with self.subTest(source=source.name, mode=mode):
                    if mode == 'missing':
                        source.unlink()
                    else:
                        source.write_text('{}')
                    with self.assertRaises((ValueError, KeyError, OSError)):
                        export_candidate(self.ep, self.output)
                    self.assertFalse(self.output.exists())
                    source.write_bytes(original)
        self.put(manifest.parent/'untracked.txt', 'not anchored by recovery identity')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            export_candidate(self.ep, self.output)

    def test_sft_keeps_original_messages_and_input_order_and_copies_all_evidence(self):
        second, first = self.demo('second'), self.demo('first')
        result = export_candidate(self.ep, self.output, [second, first])
        self.assertEqual(result['sft']['count'], 2)
        rows = [json.loads(line) for line in Path(result['sft']['path']).read_text().splitlines()]
        self.assertEqual([row['request_id'] for row in rows], ['second', 'first'])
        for row, descriptor in zip(rows, [second, first]):
            native = Path(json.loads(descriptor.read_text())['native_job_dir'])
            self.assertEqual(row['messages'], json.loads((native/'rollout/state.json').read_text())['messages'])
            for ref in row['evidence']:
                local = Path(result['output_dir'])/ref['path']
                self.assertEqual(self.sha(local), ref['sha256'])

    def test_failed_oracle_or_incomplete_demo_is_pending_without_affecting_rl(self):
        for variant in ['zero', 'oracle', 'exception', 'digest-only', 'missing-tools']:
            with self.subTest(variant=variant):
                descriptor_path = self.demo(variant, reward=0 if variant == 'zero' else 1)
                descriptor = json.loads(descriptor_path.read_text())
                trial_path = Path(descriptor['trial_result_path'])
                trial = json.loads(trial_path.read_text())
                if variant == 'oracle':
                    trial['config']['agent']['name'] = 'oracle'
                    self.put(trial_path, trial)
                elif variant == 'exception':
                    trial['exception_info'] = {'exception_type': 'TimeoutError'}
                    self.put(trial_path, trial)
                elif variant == 'digest-only':
                    self.put(descriptor_path, {'schema': 'adb-native-digest-v1', 'steps': []})
                elif variant == 'missing-tools':
                    state_path = Path(descriptor['native_job_dir'])/'rollout/state.json'
                    state = json.loads(state_path.read_text())
                    state.pop('tools')
                    self.put(state_path, state)
                result = export_candidate(self.ep, self.output, [descriptor_path])
                self.assertEqual(result['rl']['status'], 'exported')
                self.assertEqual(result['sft']['status'], 'pending')
                self.assertEqual(len(result['sft']['rejections']), 1)

    def test_source_binding_mismatches_reject_sft(self):
        for variant in ['initial', 'final', 'request', 'turn', 'feedback', 'runner', 'system', 'candidate']:
            with self.subTest(variant=variant):
                descriptor_path = self.demo('mismatch-'+variant)
                descriptor = json.loads(descriptor_path.read_text())
                native = Path(descriptor['native_job_dir'])
                if variant in ['initial', 'final']:
                    self.put(native/('input.xlsx' if variant == 'initial' else 'final.xlsx'), b'wrong workbook')
                elif variant == 'request':
                    request = json.loads((native/'request.json').read_text())
                    request['request_id'] = 'unrelated'
                    self.put(native/'request.json', request)
                elif variant == 'turn':
                    self.put(native/'rollout/turn-0001.json', {'choices': [{'message': {'role': 'assistant', 'content': 'fabricated'}}]})
                elif variant == 'feedback':
                    state = json.loads((native/'rollout/state.json').read_text())
                    state['messages'][3]['content'] = 'fabricated feedback'
                    self.put(native/'rollout/state.json', state)
                elif variant in ['runner', 'system']:
                    self.put(Path(descriptor['runner_path' if variant == 'runner' else 'system_prompt_path']), 'changed')
                elif variant == 'candidate':
                    self.put(Path(descriptor['case_dir'])/'candidate/instruction.md', 'different task')
                result = export_candidate(self.ep, self.output, [descriptor_path])
                self.assertEqual(result['sft']['count'], 0)
                self.assertTrue(result['sft']['rejections'])

    def test_credential_bearing_demo_is_not_copied_and_episode_cannot_be_destination(self):
        path = self.demo('credential')
        descriptor = json.loads(path.read_text())
        trial_path = Path(descriptor['trial_result_path'])
        trial = json.loads(trial_path.read_text())
        trial['config']['agent']['env'] = {'API_KEY': 'private-value'}
        self.put(trial_path, trial)
        result = export_candidate(self.ep, self.output, [path])
        self.assertEqual(result['sft']['status'], 'pending')
        self.assertNotIn('private-value', ''.join(p.read_text(errors='ignore') for p in Path(result['output_dir']).rglob('*') if p.is_file()))
        with self.assertRaisesRegex(ValueError, 'outside'):
            export_candidate(self.ep, self.ep/'exports')

    def test_frozen_runner_canonical_history_and_each_original_request_fingerprint(self):
        runner = self.put(self.root/'serialized-runner.py', '''TOOL_RESPONSE_CHARS = 10240
def tool_markup_segments(content):
    return []
def strip_tool_call_markup(content):
    return content
def normalize_api_history(state):
    return None
def truncate_tool_feedback(text):
    return text
def parse_calls(message, turn):
    return [{'id': c['id'], 'name': c['function']['name'], 'raw_arguments': c['function']['arguments'], 'protocol': 'native'} for c in message.get('tool_calls', [])]
''')
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]
        initial = [{'role': 'system', 'content': 'Use tools'}, {'role': 'user', 'content': 'Read records'}]
        assistant = {'role': 'assistant', 'content': None, 'tool_calls': [
                     {'id': 'c1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]}
        feedback = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'actual output'}
        final = {'role': 'assistant', 'content': 'Done'}
        sampling = {'extra_body': {}, 'temperature': 0.6, 'seed': 12}
        cfg = {'model': 'qwen', 'sampling': sampling}
        state = {'initial_prompt_message_count': 2, 'initial_run_config': cfg,
                 'active_run_config': cfg, 'config_segments': [{'configuration': cfg}],
                 'messages': initial+[assistant, feedback, final], 'events': [
                 {'step_index': 1, 'event_index': 0, 'event_kind': 'tool_call', 'call_id': 'c1', 'tool_name': 'read'},
                 {'step_index': 1, 'event_index': 1, 'event_kind': 'tool_response', 'call_id': 'c1',
                  'response_to_event_index': 0, 'tool_name': 'read', 'raw': 'actual output'}]}
        turns = []
        for i, (prefix, message) in enumerate([(initial, assistant), (initial+[assistant, feedback], final)], 1):
            payload = {'model': 'qwen', 'messages': prefix, 'tools': tools, 'tool_choice': 'auto',
                       'temperature': 0.6, 'stream': False, 'seed': 12}
            fingerprint = hashlib.sha256(_encoded({'payload': payload, 'sampling': sampling})).hexdigest()
            turns.append({'turn': i, 'request_sha256': fingerprint,
                          'response': {'choices': [{'message': dict(message, provider_only_metadata='not part of model history')}]}})
        _validate_recorded_history(state, tools, turns, runner)
        for variant in ['tools', 'context', 'feedback', 'turn-order']:
            with self.subTest(variant=variant):
                altered_state, altered_tools, altered_turns = copy.deepcopy(state), copy.deepcopy(tools), copy.deepcopy(turns)
                if variant == 'tools':
                    altered_tools[0]['function']['parameters']['description'] = 'changed schema'
                elif variant == 'context':
                    altered_state['messages'][0]['content'] = 'changed context'
                elif variant == 'feedback':
                    altered_state['events'][1]['raw'] = 'changed feedback'
                else:
                    altered_turns.reverse()
                with self.assertRaises(ValueError):
                    _validate_recorded_history(altered_state, altered_tools, altered_turns, runner)



if __name__ == '__main__':
    unittest.main()
