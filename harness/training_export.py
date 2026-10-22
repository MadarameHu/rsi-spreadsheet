"""Export quality-gated candidates and verified, source-bound tool demonstrations.

No model calls, semantic repairs or inferred successful demonstrations. Native
demonstration descriptors locate raw records; their assertions are not trusted.
"""
from __future__ import annotations

import hashlib
import ast
import json
import math
from pathlib import Path
import re

from common import task_pass_threshold
from stage_records import read_stage_commit, snapshot_tree, verify_snapshot


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _secret_check(raw):
    """Refuse credential-bearing evidence instead of changing its raw bytes."""
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return
    sensitive = re.compile(r'(?i)(?:[\w-]*(?:api[_-]?key|access[_-]?token|secret[_-]?key)|authorization|password)')
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if sensitive.fullmatch(key) and isinstance(item, str) and item.strip():
                    if not (item.startswith(('env:', '${', '$')) or item in ('present', 'missing', 'REDACTED')):
                        raise ValueError('credential-bearing evidence cannot be exported')
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    try:
        check(json.loads(text))
    except json.JSONDecodeError:
        pass
    if re.search(r'\bsk-[A-Za-z0-9_-]{20,}\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', text):
        raise ValueError('credential-bearing evidence cannot be exported')


def _store(path, raw):
    _secret_check(raw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f'export artifact already exists with different bytes: {path.name}')
    else:
        path.write_bytes(raw)


def _commit_evidence(ep, gate, ancestors=()):
    if ep in ancestors:
        raise ValueError('cyclic candidate repair provenance')
    synthesis = _read(ep/'diagnostics/synthesis.json')
    records = Path(synthesis['stage_commit_dir'])
    files, visited, originals, commits = {}, set(), {}, {}
    def visit(commit_id):
        if commit_id in visited:
            return
        commit = read_stage_commit(records, commit_id)
        commits[commit_id] = commit
        visited.add(commit_id)
        commit_raw = (records/'commits'/f'{commit_id}.json').read_bytes()
        if json.loads(commit_raw) != commit:
            raise ValueError('stage commit changed during export')
        files[f'records/commits/{commit_id}.json'] = commit_raw
        for reference in commit['references']:
            raw = (records/reference['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference['sha256'] or len(raw) != reference['bytes']:
                raise ValueError('archived evidence changed during export')
            files['records/'+reference['path']] = raw
            originals[reference['original_path']] = 'records/'+reference['path']
        for parent in commit['parent_ids']:
            visit(parent)
    for key in ('attribution_commit', 'design_commit', 'audit_commit'):
        if gate.get(key):
            visit(gate[key])
    audit = commits.get(gate.get('audit_commit'), {})
    anchor = audit.get('metadata', {}).get('audit_envelope', {})
    for source, digest in anchor.get('output_hashes', {}).items():
        relative = originals.get(source)
        if relative is None:
            path = Path(source)
            if path.is_symlink() or not path.is_file():
                raise ValueError('missing raw audit evidence: '+source)
            relative = 'audit/artifacts/'+digest
            files[relative] = path.read_bytes()
            originals[source] = relative
        if hashlib.sha256(files[relative]).hexdigest() != digest:
            raise ValueError('raw audit evidence hash mismatch: '+source)
    if gate.get('review_mode') == 'parallel':
        from host_validation import digest
        root = ep/'diagnostics/validation'
        receipt = _read(root/'reference-evidence.json')
        request = _read(root/'request.json')
        identity = {key: value for key, value in receipt.items() if key != 'reference_evidence_id'}
        if (receipt.get('schema_version') != 'reference-evidence-v1'
                or receipt.get('reference_evidence_id') != digest(identity)
                or receipt.get('request_sha256') != _sha(root/'request.json')
                or request.get('request_id') != receipt.get('request_id')
                or digest({k:v for k,v in request.items() if k != 'request_id'}) != request.get('request_id')
                or any(receipt.get(key) != gate.get(key) for key in
                       ('candidate_sha256', 'attribution_commit', 'design_commit'))):
            raise ValueError('independent reference receipt identity mismatch')
        snapshot = verify_snapshot(receipt['snapshot'])
        if (_read(root/'reference-case-manifest.json') != snapshot
                or Path(snapshot['snapshot_path']).resolve() != (root/'reference-case').resolve()):
            raise ValueError('independent reference manifest mismatch')
        for name in ('request.json', 'reference-evidence.json', 'reference-case-manifest.json'):
            relative = 'reference/'+name
            files[relative] = (root/name).read_bytes()
            originals[str(root/name)] = relative
        if (hashlib.sha256(files['reference/request.json']).hexdigest() != receipt['request_sha256']
                or json.loads(files['reference/reference-evidence.json']) != receipt
                or json.loads(files['reference/reference-case-manifest.json']) != snapshot):
            raise ValueError('reference records changed during export')
        for entry in snapshot['entries']:
            if entry['type'] != 'file':
                continue
            source = Path(snapshot['snapshot_path'])/entry['path']
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
                raise ValueError('independent reference changed during export')
            relative = 'reference/case/'+entry['path']
            files[relative] = raw
            originals[str(source)] = relative
            originals[str(Path(snapshot['source_path'])/entry['path'])] = relative
    job_path = ep/'diagnostics/candidate_job.json'
    if job_path.is_file():
        files['candidate-job.json'] = job_path.read_bytes()
        originals[str(job_path)] = 'candidate-job.json'
        job = _read(job_path)
        repair = job.get('repair')
        if repair is not None:
            from repair_jobs import validate_repair
            validate_repair(job)
            parent = Path(repair['parent_episode']).resolve()
            feedback = Path(repair['feedback_dir'])
            if not re.fullmatch(r'[A-Za-z0-9_-]+', repair['parent_job_id']):
                raise ValueError('unsafe parent job identifier')
            prefix = 'repair/'+repair['parent_job_id']+'/'
            for name, digest in repair['feedback_hashes'].items():
                relative_name = Path(name)
                if relative_name.is_absolute() or '..' in relative_name.parts:
                    raise ValueError('unsafe repair feedback path')
                raw = (feedback/name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError('repair feedback changed during export')
                relative = prefix+'feedback/'+name
                files[relative] = raw
                originals[str(feedback/name)] = relative
            parent_gate = _read(feedback/'quality.json')
            parent_files = _commit_evidence(parent, parent_gate, (*ancestors, ep))
            parent_mapping = json.loads(parent_files.pop('source-paths.json'))
            files.update({prefix+name: raw for name, raw in parent_files.items()})
            originals.update({source: prefix+relative.removeprefix('provenance/')
                              for source, relative in parent_mapping.items()})
            candidate = verify_snapshot(parent_gate['candidate_manifest'])
            files[prefix+'candidate-manifest.json'] = _encoded(candidate)+b'\n'
            for entry in candidate['entries']:
                if entry['type'] != 'file':
                    continue
                source = Path(candidate['snapshot_path'])/entry['path']
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
                    raise ValueError('parent candidate changed during export')
                relative = prefix+'candidate/'+entry['path']
                files[relative] = raw
                originals[str(source)] = relative
    files['source-paths.json'] = _encoded({source: 'provenance/'+relative
                                        for source, relative in sorted(originals.items())})+b'\n'
    return files


def verify_export(directory):
    """Verify a moved export without accessing any original absolute source path."""
    from stage_records import _tree_entries
    directory = Path(directory)
    manifest = _read(directory/'integrity.json')
    actual = [item for item in _tree_entries(directory) if item['path'] != 'integrity.json']
    if (manifest.get('schema_version') != 'training-export-integrity-v1'
            or manifest.get('entries') != actual
            or manifest.get('sha256') != hashlib.sha256(_encoded(actual)).hexdigest()):
        raise ValueError('training export integrity mismatch')
    for relative in _read(directory/'provenance/source-paths.json').values():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not (directory/path).is_file():
            raise ValueError('training export source mapping is not self-contained')
    for reference in (directory/'provenance').rglob('*manifest.json'):
        if reference.name == 'reference-case-manifest.json':
            verify_snapshot(dict(_read(reference), snapshot_path=str(reference.parent/'case')))
        elif reference.name == 'candidate-manifest.json':
            verify_snapshot(dict(_read(reference), snapshot_path=str(reference.parent/'candidate')))
    recovery = directory/'provenance/validation-recovery/manifest.json'
    if recovery.is_file():
        pointer = _read(directory/'provenance/active-validation-recovery.json')
        original_root = Path(pointer['manifest_path']).parent
        for item in _read(recovery)['old_cases']:
            snapshot = item['snapshot']
            relative = Path(snapshot['snapshot_path']).relative_to(original_root)
            verify_snapshot(dict(snapshot, snapshot_path=str(recovery.parent/relative)))
    return manifest


def _recovery_evidence(ep, gate, files):
    """Copy only an explicitly prepared, verified immutable recovery history."""
    root = ep/'diagnostics/validation'
    active = root/'active-recovery.json'
    directory = root/'recovery-01'
    if not active.exists() and not directory.exists():
        return None, []
    from validation_recovery import verify_recovery
    from stage_records import _tree_entries
    request = _read(root/'request.json')
    manifest = verify_recovery(request)
    if manifest is None:
        raise ValueError('recovery evidence exists without a verified recovery manifest')
    if (manifest['candidate_sha256'] != gate['candidate_sha256']
            or manifest['audit_commit'] != gate.get('audit_commit')):
        raise ValueError('recovery belongs to a different admitted candidate or audit')
    pointer_raw = active.read_bytes()
    pointer = json.loads(pointer_raw)
    manifest_raw = (directory/'manifest.json').read_bytes()
    if (json.loads(manifest_raw) != manifest
            or hashlib.sha256(manifest_raw).hexdigest() != pointer['manifest_sha256']
            or pointer['recovery_id'] != manifest['recovery_id']):
        raise ValueError('recovery identity changed during export')
    expected = {'manifest.json': pointer['manifest_sha256']}
    expected_directories = set()
    originals = {str(directory/'manifest.json'): 'provenance/validation-recovery/manifest.json',
                 str(active): 'provenance/active-validation-recovery.json'}
    for item in manifest['files']:
        relative = Path(item['snapshot_path']).relative_to(directory).as_posix()
        expected[relative] = item['sha256']
        for source in (item['source_path'], item['snapshot_path']):
            originals[source] = 'provenance/validation-recovery/'+relative
    for case in manifest['old_cases']:
        snapshot = case['snapshot']
        prefix = Path(snapshot['snapshot_path']).relative_to(directory)
        expected_directories.add(prefix.as_posix())
        for item in snapshot['entries']:
            if item['type'] == 'directory':
                expected_directories.add((prefix/item['path']).as_posix())
            if item['type'] == 'file':
                relative = (prefix/item['path']).as_posix()
                expected[relative] = item['sha256']
                for source in (snapshot['source_path'], snapshot['snapshot_path']):
                    originals[str(Path(source)/item['path'])] = 'provenance/validation-recovery/'+relative
    for name in expected:
        expected_directories.update(parent.as_posix() for parent in Path(name).parents if parent != Path('.'))
    before = _tree_entries(directory)
    if ({item['path']: item['sha256'] for item in before if item['type'] == 'file'} != expected
            or {item['path'] for item in before if item['type'] == 'directory'} != expected_directories):
        raise ValueError('recovery directory differs from its complete frozen evidence inventory')
    for relative, digest in expected.items():
        raw = (directory/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('recovery raw evidence changed during export')
        files['validation-recovery/'+relative] = raw
    if before != _tree_entries(directory) or active.read_bytes() != pointer_raw:
        raise ValueError('recovery evidence changed during export')
    files['active-validation-recovery.json'] = pointer_raw
    mapping = json.loads(files['source-paths.json'])
    mapping.update(originals)
    files['source-paths.json'] = _encoded(mapping)+b'\n'
    directories = ['validation-recovery/'+item['path'] for item in before if item['type'] == 'directory']
    return pointer, directories


def _path(descriptor, base, key):
    value = descriptor.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError('missing raw demonstration source: '+key)
    path = Path(value)
    path = path if path.is_absolute() else base/path
    if path.is_symlink() or not path.exists():
        raise ValueError('missing or symlink demonstration source: '+key)
    return path.resolve()


def _validate_recorded_history(state, tools, turns, runner):
    """Check frozen runner serialization against raw turns; export no rewritten text."""
    wanted = {'tool_markup_segments', 'strip_tool_call_markup', 'parse_calls',
              'normalize_api_history', 'truncate_tool_feedback'}
    parsed = ast.parse(runner.read_text())
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise ValueError('unsupported native runner history serialization')
    constants = [node for node in parsed.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'TOOL_RESPONSE_CHARS' for target in node.targets)]
    if len(constants) != 1 or not isinstance(constants[0].value, ast.Constant):
        raise ValueError('runner observation limit is not statically recorded')
    namespace = {'json': json, 're': re, 'utc': lambda: 'verification-only',
                 '__builtins__': {name: value for name, value in [('len', len), ('isinstance', isinstance),
                     ('str', str), ('dict', dict), ('list', list), ('enumerate', enumerate), ('reversed', reversed),
                     ('ValueError', ValueError), ('TypeError', TypeError)]},
                 'TOOL_RESPONSE_CHARS': constants[0].value.value}
    # Only the five serialization helpers are loaded, never the runner main loop,
    # tool classes, network client, or model. Their source hash was checked above.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(runner), 'exec'), namespace)
    messages = state.get('messages')
    count = state.get('initial_prompt_message_count')
    if not isinstance(messages, list) or type(count) is not int or count < 1:
        raise ValueError('missing recorded initial message boundary')
    history = {'messages': messages[:count], 'history_normalizations': []}
    cfg = state.get('initial_run_config') or {}
    if state.get('active_run_config') != cfg or len(state.get('config_segments', [])) != 1:
        raise ValueError('changed native configuration requires an explicitly supported history adapter')
    sampling = cfg['sampling']
    names = {tool['function']['name'] for tool in tools}
    events = state['events']
    for expected_turn, turn in enumerate(turns, 1):
        if turn.get('turn') != expected_turn:
            raise ValueError('raw model turns are missing or out of order')
        namespace['normalize_api_history'](history)
        payload = dict(sampling['extra_body'], model=cfg['model'], messages=history['messages'], tools=tools,
                       tool_choice='auto', temperature=sampling['temperature'], stream=False)
        if sampling.get('seed') is not None:
            payload['seed'] = sampling['seed']
        fingerprint = hashlib.sha256(_encoded({'payload': payload, 'sampling': sampling})).hexdigest()
        if fingerprint != turn.get('request_sha256'):
            raise ValueError('raw turn request fingerprint differs: tools, original context or sampling changed')
        choices = turn['response']['choices']
        if len(choices) != 1:
            raise ValueError('raw model turn must contain one response')
        raw = choices[0]['message']
        calls = namespace['parse_calls'](raw, expected_turn)
        content = raw.get('content')
        if calls and any(call['protocol'] == 'text_fallback' for call in calls):
            content = namespace['strip_tool_call_markup'](content or '')
        assistant = {'role': 'assistant', 'content': content}
        reasoning = raw.get('reasoning_content') or raw.get('reasoning')
        if isinstance(reasoning, str):
            assistant['reasoning_content'] = reasoning
        if calls:
            assistant['tool_calls'] = [{'id': call['id'], 'type': 'function', 'function': {
                'name': call['name'], 'arguments': call['raw_arguments'] if isinstance(call['raw_arguments'], str)
                else json.dumps(call['raw_arguments'])}} for call in calls]
        history['messages'].append(assistant)
        namespace['normalize_api_history'](history)
        executed = [event for event in events if event.get('step_index') == expected_turn and event.get('event_kind') == 'tool_call']
        if len(executed) != len(calls):
            raise ValueError('raw assistant calls and actual execution count differ')
        for call, event in zip(calls, executed):
            if event.get('call_id') != call['id'] or event.get('tool_name') != call['name']:
                raise ValueError('executed tool call does not match the raw model turn or schema')
            if call['name'] not in names and event.get('execution_status') != 'error':
                raise ValueError('unknown tool lacks its actual rejection feedback')
            responses = [response for response in events if response.get('event_kind') == 'tool_response'
                         and response.get('response_to_event_index') == event['event_index']]
            if len(responses) != 1 or responses[0].get('call_id') != call['id']:
                raise ValueError('executed tool call has no unique matching response')
            feedback = namespace['truncate_tool_feedback'](responses[0]['raw'])
            history['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': feedback})
    if history['messages'] != messages:
        raise ValueError('saved messages differ from the original runner history')
    if not any(event.get('event_kind') == 'tool_call' for event in events):
        raise ValueError('no real executed tool demonstration')


def _demonstration(path, candidate):
    """Validate raw Harbor/native sources; a descriptor alone cannot certify them."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path/'demonstration.json'
    descriptor = _read(path)
    if descriptor.get('schema_version') != 'native-training-demo-v1':
        raise ValueError('unsupported demonstration; a digest or oracle is not an SFT trajectory')
    base = path.parent
    case = _path(descriptor, base, 'case_dir')
    native_job = _path(descriptor, base, 'native_job_dir')
    result_path = _path(descriptor, base, 'trial_result_path')
    if not result_path.is_relative_to(case):
        raise ValueError('trial result does not belong to the validation case')
    manifest_path = case/'host-candidate.json'
    if not manifest_path.exists():
        manifest_path = case/'candidate-version.json'
    manifest = verify_snapshot(manifest_path)
    if manifest['tree_sha256'] != candidate['tree_sha256']:
        raise ValueError('demonstration belongs to a different candidate version')
    trial = _read(result_path)
    if not trial.get('finished_at') or 'exception_info' not in trial or trial['exception_info'] is not None:
        raise ValueError('trial did not finish successfully')
    if trial.get('config', {}).get('agent', {}).get('name') == 'oracle':
        raise ValueError('oracle execution is not a model tool demonstration')
    task_path = trial.get('config', {}).get('task', {}).get('path')
    if not task_path or Path(task_path).resolve() != Path(manifest['snapshot_path']).resolve():
        raise ValueError('raw trial task path is not bound to the candidate snapshot')
    reward = (trial.get('verifier_result') or {}).get('rewards', {}).get('reward')
    agent = result_path.parent/'agent'
    request = _read(agent/'bridge-request.json')
    native = _read(agent/'native-execution.json')
    remote_response = _read(native_job/'response.json')
    # bridge server may add workbook transport bytes after the worker wrote its
    # execution receipt; all execution fields must still match exactly.
    if native != {key: value for key, value in remote_response.items() if key != 'workbook_b64'} or request != _read(native_job/'request.json'):
        raise ValueError('Harbor/native raw request or response mismatch')
    request_id = trial.get('agent_result', {}).get('metadata', {}).get('bridge_request_id')
    if not request_id or request_id != request.get('request_id') or request_id != native.get('request_id'):
        raise ValueError('request identity is not bound to the original trial')
    if native.get('execution_completed') is not True or native.get('runtime_errors') != [] or native.get('termination_reason') != 'model_final':
        raise ValueError('native execution has an error or non-final terminal')
    initial = Path(candidate['snapshot_path'])/'environment/input.xlsx'
    if not initial.is_file() or not (_sha(initial) == _sha(agent/'bridge-input.xlsx') == _sha(native_job/'input.xlsx')):
        raise ValueError('initial workbook is not bound to the candidate')
    if not (_sha(agent/'bridge-final.xlsx') == _sha(native_job/'final.xlsx') == native.get('final_workbook_sha256')):
        raise ValueError('final workbook does not match the executed result')
    runner = _path(descriptor, base, 'runner_path')
    system = _path(descriptor, base, 'system_prompt_path')
    if _sha(runner) != native.get('source_runner_sha256') or _sha(system) != native.get('system_prompt_sha256'):
        raise ValueError('runner or system prompt differs from actual tool environment')
    state_path = native_job/'rollout/state.json'
    native_result_path = native_job/'rollout/result.json'
    state, native_result = _read(state_path), _read(native_result_path)
    if state.get('task_id') is not None and state['task_id'] != 'adb-'+request_id:
        raise ValueError('native state belongs to a different task/session')
    if state.get('seed_sha256') is not None and state['seed_sha256'] != _sha(initial):
        raise ValueError('native state initial workbook hash differs')
    if native_result.get('events', state.get('events')) != native.get('native_events'):
        raise ValueError('raw execution events differ from the Harbor receipt')
    messages, tools = state.get('messages'), state.get('tools')
    extra_files = []
    if descriptor.get('collection_manifest'):
        collection_path = _path(descriptor, base, 'collection_manifest')
        collection = _read(collection_path)
        if collection.get('request_id') != request_id:
            raise ValueError('collection manifest belongs to another native request')
        extra_files.append(collection_path)
        for reference in collection.get('files', []):
            relative = Path(reference['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe collected source path')
            source = (base/relative).resolve()
            if not source.is_relative_to(base.resolve()) or _sha(source) != reference['sha256']:
                raise ValueError('collected native source has changed')
            extra_files.append(source)
    if tools is None:
        tools_path = _path(descriptor, base, 'tools_path')
        tool_config_path = _path(descriptor, base, 'tool_config_path')
        if _sha(tool_config_path) != state.get('initial_run_config', {}).get('tool_config_sha256'):
            raise ValueError('tool configuration differs from the original native environment')
        tools = _read(tools_path)
        extra_files += [tools_path, tool_config_path]
    turn_names = state.get('turn_files')
    if turn_names is not None:
        if not isinstance(turn_names, list) or not all(isinstance(name, str) and re.fullmatch(r'turn-\d{4,}\.json', name) for name in turn_names):
            raise ValueError('invalid original turn file list')
        turns_paths = [native_job/'rollout'/name for name in turn_names]
    else:
        turns_paths = sorted((native_job/'rollout').glob('turn-*.json'))
    if not turns_paths:
        raise ValueError('original model turn files are missing')
    _validate_recorded_history(state, tools, [_read(turn) for turn in turns_paths], runner)
    if not any(message.get('role') == 'system' and message.get('content') == system.read_text() for message in messages):
        raise ValueError('complete original system context is missing')
    if not any(message.get('role') == 'user' and message.get('content') in
               (request['instruction'], 'Task:\n'+request['instruction']+'\n') for message in messages):
        raise ValueError('original task request is missing from messages')
    files = [path, manifest_path, result_path, state_path, native_result_path, runner, system,
             agent/'bridge-request.json', agent/'native-execution.json', agent/'bridge-input.xlsx',
             agent/'bridge-final.xlsx', native_job/'request.json', native_job/'response.json',
             native_job/'input.xlsx', native_job/'final.xlsx', *turns_paths, *extra_files]
    files = list(dict.fromkeys(files))
    if type(reward) not in (int, float) or not math.isfinite(reward) or reward < task_pass_threshold(Path(candidate['snapshot_path'])):
        raise ValueError('no verified successful demonstration reward')
    row = dict(schema_version='tool-sft-v1', candidate_sha256=candidate['tree_sha256'],
               request_id=request_id, model=native.get('model'), reward=reward, messages=messages, tools=tools)
    return row, files


def export_candidate(ep, output_root, demonstration_paths=()):
    """Export an eligible RL task and zero or more proven successful SFT rows."""
    from quality_gate import evaluate_quality
    ep, output_root = Path(ep).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(ep):
        raise ValueError('export destination must be outside original episode records')
    gate = evaluate_quality(ep)
    if gate.get('eligible_for_export') is not True or gate.get('status') != 'accepted':
        return dict(status='blocked', candidate_sha256=gate.get('candidate_sha256'), quality_gate=gate,
                    rl={'status': 'blocked', 'path': None},
                    sft={'status': 'pending', 'count': 0, 'path': None, 'rejections': []}, output_dir=None)
    candidate = verify_snapshot(gate['candidate_manifest'])
    if gate.get('candidate_sha256') != candidate['tree_sha256']:
        raise ValueError('quality gate candidate identity mismatch')
    files = _commit_evidence(ep, gate)
    recovery, recovery_directories = _recovery_evidence(ep, gate, files)
    rows, demos, rejections = [], {}, []
    for index, path in enumerate(demonstration_paths):
        try:
            row, sources = _demonstration(path, candidate)
            evidence = []
            pending = {}
            for number, source in enumerate(sources):
                raw = source.read_bytes()
                _secret_check(raw)
                dest = f'demonstrations/{index:04d}/{number:03d}-{source.name}'
                pending[dest] = raw
                evidence.append({'path': 'provenance/'+dest, 'source_path': str(source), 'sha256': hashlib.sha256(raw).hexdigest()})
            _secret_check(_encoded(row))
            row['evidence'] = evidence
            rows.append(row)
            demos.update(pending)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rejections.append({'source': str(path), 'reason': f'{type(exc).__name__}: {exc}'})
    files.update(demos)
    files['quality-gate.json'] = _encoded(gate)+b'\n'
    for item in candidate['entries']:
        if item['type'] == 'file':
            if Path(item['path']).name == '.env':
                raise ValueError('private .env files cannot be included in training exports')
            _secret_check((Path(candidate['snapshot_path'])/item['path']).read_bytes())
    for raw in files.values():
        _secret_check(raw)
    identity_fields = {'candidate': candidate['tree_sha256'],
                       'gate': gate, 'rows': rows, 'rejections': rejections}
    if recovery is not None:
        identity_fields['recovery_id'] = recovery['recovery_id']
    identity = hashlib.sha256(_encoded(identity_fields)).hexdigest()
    destination = output_root/candidate['tree_sha256']/identity
    task = destination/'rl/task'
    copied = dict(candidate, source_path=candidate['snapshot_path'], snapshot_path=str(task))
    if task.exists():
        verify_snapshot(copied)
    else:
        snapshot_tree(candidate['snapshot_path'], task)
        verify_snapshot(copied)
    _store(destination/'rl/candidate-manifest.json', _encoded(copied)+b'\n')
    for relative in recovery_directories:
        (destination/'provenance'/relative).mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        manifest = Path(name)
        if manifest.name not in ('reference-case-manifest.json', 'candidate-manifest.json'):
            continue
        folder = 'case' if manifest.name == 'reference-case-manifest.json' else 'candidate'
        for entry in json.loads(raw)['entries']:
            if entry['type'] == 'directory':
                (destination/'provenance'/manifest.parent/folder/entry['path']).mkdir(parents=True, exist_ok=True)
    for relative, raw in files.items():
        _store(destination/'provenance'/relative, raw)
    sft_path = None
    if rows:
        sft_path = destination/'sft.jsonl'
        _store(sft_path, b''.join(_encoded(row)+b'\n' for row in rows))
    result = dict(schema_version='training-export-v1', status='exported', candidate_sha256=candidate['tree_sha256'],
                  output_dir=str(destination), quality_gate=gate,
                  rl={'status': 'exported', 'path': str(task)},
                  sft={'status': 'exported' if rows else 'pending', 'count': len(rows),
                       'path': str(sft_path) if sft_path else None, 'rejections': rejections})
    if recovery is not None:
        result['recovery'] = recovery
    _store(destination/'export.json', _encoded(result)+b'\n')
    from stage_records import _tree_entries
    entries = [item for item in _tree_entries(destination) if item['path'] != 'integrity.json']
    _store(destination/'integrity.json', _encoded(dict(schema_version='training-export-integrity-v1',
           entries=entries, sha256=hashlib.sha256(_encoded(entries)).hexdigest()))+b'\n')
    verify_export(destination)
    return result
