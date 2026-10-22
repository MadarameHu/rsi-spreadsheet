"""Explicit post-review attribution versions; original judgments remain immutable."""
from pathlib import Path
import ast
import hashlib
import json
import re

import staged_synth as st
from stage_records import _tree_entries, read_stage_commit


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_revision_inputs(parent_episode, cfg, source, bundle, diag):
    """Validate an upstream rejection and freeze complete feedback in the new run."""
    from candidate_jobs import _source
    from quality_gate import evaluate_quality
    parent = Path(parent_episode)
    if not parent.is_absolute():
        raise ValueError('attribution_revision_from must be an absolute episode directory')
    parent = parent.resolve(strict=True)
    job_path = parent/'diagnostics/candidate_job.json'
    job = st.load(job_path)
    origin, commit, inputs, identity = _source(job['original'], job['analysis_source'])
    config = cfg['diagnostics']['parallel_attribution']
    if Path(config.get('reuse_observations_from', '')).resolve() != origin:
        raise ValueError('attribution revision must reuse the rejected candidate analysis source')
    if identity != job['analysis_identity']:
        raise ValueError('attribution revision parent analysis identity changed')
    if Path(diag).resolve() in (origin, parent/'diagnostics'):
        raise ValueError('attribution revision requires a new analysis directory')
    for name, current in [('original', source), ('rollouts', bundle)]:
        if _tree_entries(current) != inputs[name]['entries']:
            raise ValueError('attribution revision source evidence differs from parent')
    quality_path, audit_path = [parent/'diagnostics'/name for name in ('quality.json', 'audit.json')]
    quality = st.load(quality_path)
    verified = evaluate_quality(parent, audit=st.load(audit_path))
    if verified != quality:
        raise ValueError('attribution revision feedback differs from verified quality receipt')
    upstream = [item for item in quality['issues'] if item['id'].startswith(('claim:', 'observation:'))
                or (item['id'].startswith('mapping:') and item['id'].endswith(':upstream_verdict'))]
    if (quality['status'] != 'rejected' or not upstream
            or any(item['verdict'] == 'fault' for item in quality['issues'])):
        raise ValueError('attribution revision requires an upstream rejection without infrastructure faults')
    candidate_commit = read_stage_commit(parent/'diagnostics/records', quality['attribution_commit'])
    if candidate_commit['payload'] != commit['payload']:
        raise ValueError('candidate rejected attribution differs from declared source analysis')
    root = Path(diag)/'attribution/revision-inputs'
    paths = {'parent_attribution': origin/'attribution.json',
             'parent_attribution_commit': origin/'records/commits'/(commit['commit_id']+'.json'),
             'parent_job': job_path, 'quality': quality_path, 'audit': audit_path}
    raw_path = commit.get('metadata', {}).get('raw_output_path')
    if raw_path:
        paths['parent_model_output'] = Path(raw_path)
    entries, contents = {}, {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('attribution revision requires complete regular source files')
        raw = path.read_bytes()
        target = root/(name+'.json')
        entries[name] = {'source_path': str(path), 'snapshot_path': str(target),
                         'sha256': _sha(raw), 'bytes': len(raw)}
        contents[target] = raw
    evidence = []
    # The rejected units' evidence is kept complete, including reviewer findings.
    # This is mechanical provenance capture; it does not rewrite any judgment.
    for item in upstream:
        for ref in item.get('evidence', []):
            path = Path(ref['path'])
            if path.is_symlink() or not path.is_file():
                raise ValueError('attribution revision feedback evidence is missing')
            raw = path.read_bytes()
            target = root/'evidence'/(_sha(raw)+path.suffix)
            entry = {'source_path': str(path), 'snapshot_path': str(target),
                     'sha256': _sha(raw), 'bytes': len(raw)}
            if entry not in evidence:
                evidence.append(entry)
                contents[target] = raw
    manifest = {'schema_version': 'attribution-revision-input-v1',
                'origin': 'post_review_feedback', 'is_human_gold': False,
                'parent_episode': str(parent), 'parent_analysis_source': str(origin),
                'parent_attribution_commit': commit['commit_id'],
                'rejected_candidate_attribution_commit': quality['attribution_commit'],
                'audit_commit': quality['audit_commit'],
                'upstream_issue_ids': [item['id'] for item in upstream],
                'upstream_issues': upstream,
                'files': entries, 'evidence': evidence}
    for path, raw in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        st.save_bytes(path, raw)
    manifest_path = root/'manifest.json'
    st.save_new(manifest_path, manifest)
    return manifest_path


def revision_prompt(manifest_path):
    return ('\n\nThis invocation creates a NEW POST-REVIEW ATTRIBUTION VERSION. '
            'Read the complete revision-input manifest at '+str(manifest_path)+'. '
            'Begin with its upstream_issues (copied verbatim from the quality receipt) '
            'and the parent_attribution file. The complete quality and audit files are '
            'available for provenance checks; candidate/scorer defects belong to a later stage. '
            'It identifies the frozen original attribution, original model output, independently '
            'verified quality/audit feedback, and complete supporting evidence. Original judgments '
            'are historical records; the feedback was written later and is not human gold or part '
            'of the original model diagnosis. Recheck rejected assertions against the original '
            'task and trajectories, including quantifiers, causal wording and counterevidence. '
            'Revise affected claims and dependent training goals where justified; retain supported '
            'content and stable claim/goal IDs where possible. Do not invent findings to satisfy '
            'a rejection, and do not alter worker observations or any parent file. Output the '
            'complete normal capability-synthesis schema, not a patch. Do not produce or repair '
            'a candidate task or scorer in this attribution stage. The host separately records '
            'that this output was informed by later review.\n')


def record_revision(manifest_path, payload, raw_path):
    """Record structural changes mechanically, without interpreting correctness."""
    manifest_path = Path(manifest_path)
    manifest = st.load(manifest_path)
    for entry in [*manifest['files'].values(), *manifest['evidence']]:
        for name in ('snapshot_path', 'source_path'):
            path = Path(entry[name])
            if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != entry['sha256']:
                raise ValueError('attribution revision input evidence changed')
    old = st.load(Path(manifest['files']['parent_attribution']['snapshot_path']))
    def changes(field):
        before = {item['id']: item for item in old.get(field, [])}
        after = {item['id']: item for item in payload.get(field, [])}
        order = list(before) + [key for key in after if key not in before]
        return [key for key in order if before.get(key) != after.get(key)]
    receipt = {'schema_version': 'attribution-revision-v1',
               'origin': 'post_review_model_revision', 'is_human_gold': False,
               'parent_episode': manifest['parent_episode'],
               'parent_analysis_source': manifest['parent_analysis_source'],
               'parent_attribution_commit': manifest['parent_attribution_commit'],
               'audit_commit': manifest['audit_commit'],
               'input_manifest': str(manifest_path), 'input_manifest_sha256': _sha(manifest_path.read_bytes()),
               'raw_output_path': str(raw_path), 'raw_output_sha256': _sha(Path(raw_path).read_bytes()),
               'changed_claim_ids': changes('claims'), 'changed_training_goal_ids': changes('training_goals')}
    path = manifest_path.parent.parent/'revision.json'
    st.save_new(path, receipt)
    refs = [path, manifest_path] + [Path(e['snapshot_path']) for e in manifest['files'].values()]
    refs += [Path(e['snapshot_path']) for e in manifest['evidence']]
    return receipt, list(dict.fromkeys(refs))


def _literal(node, values):
    """Decode data only. No Python evaluation, calls, attributes or comprehensions."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.List):
        return [_literal(item, values) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            key = _literal(key, values)
            if not isinstance(key, str) or key in result:
                raise ValueError('delivery literal requires unique string dictionary keys')
            result[key] = _literal(value, values)
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left, values), _literal(node.right, values)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    raise ValueError('delivery script is outside the data-only AST whitelist')


def _decode_delivery(diag, tool_id, transcript_sha256):
    from local_analysis import read_local_result
    directory = diag/'attribution'
    finished = st.load(directory/'finished.json')
    actual = read_local_result(directory/'job')
    if any(finished.get(key) != value for key, value in actual.items()):
        raise ValueError('delivery recovery invocation receipt differs from original failure')
    terminal = actual['terminal']['agent_terminal_status']
    process = st.load(directory/'job/local-result.json')
    if (terminal not in ('timeout', 'failed') or actual['returncode'] == 0
            or process.get('process_cleanup', {}).get('host_group_absent') is not True):
        raise ValueError('delivery recovery requires a terminated failed local invocation')
    transcript = directory/'job/claude-code.txt'
    raw = transcript.read_bytes()
    if _sha(raw) != transcript_sha256:
        raise ValueError('delivery recovery transcript does not match explicit expected hash')
    tools, results = [], []
    for line in raw.decode().splitlines():
        entry = json.loads(line)
        content = entry.get('message', {}).get('content', [])
        if not isinstance(content, list):
            continue
        tools += [item for item in content if item.get('type') == 'tool_use']
        results += [item for item in content if item.get('type') == 'tool_result']
    chosen = [tool for tool in tools if tool.get('id') == tool_id]
    if len(chosen) != 1 or chosen[0] != tools[-1] or chosen[0].get('name') != 'Bash':
        raise ValueError('delivery recovery requires the exact final Bash tool call')
    tool = chosen[0]
    command = tool.get('input', {}).get('command', '')
    work = directory/'work'
    pattern = (r'^mkdir -p "'+re.escape(str(work))+r'" && cat > '
               r'(?P<temporary>/tmp/[A-Za-z0-9_.-]+\.py) << \'PYEOF\'\n'
               r'(?P<script>.*?)\nPYEOF\npython3 (?P=temporary)\s*$')
    match = re.fullmatch(pattern, command, re.S)
    if match is None:
        raise ValueError('delivery command is outside the single-file writer whitelist')
    failures = [item for item in results if item.get('tool_use_id') == tool_id]
    if (len(failures) != 1 or failures[0].get('is_error') is not True
            or 'operation not permitted: '+match['temporary'] not in str(failures[0].get('content'))):
        raise ValueError('delivery recovery requires recorded rejection of the temporary path')
    script = match['script']
    tree = ast.parse(script)
    expected_import = ast.parse('import json').body[0]
    expected_tail = ast.parse('with open(dest, "w") as f:\n    json.dump(out, f, indent=2)\nprint("written", dest)').body
    same = lambda left, right: ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)
    if (len(tree.body) != 6 or not same(tree.body[0], expected_import)
            or any(not same(a, b) for a, b in zip(tree.body[-2:], expected_tail))):
        raise ValueError('delivery script is outside the data-only AST whitelist')
    values = {}
    for name, node in zip(('R', 'out', 'dest'), tree.body[1:4]):
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != name):
            raise ValueError('delivery script requires literal R, out and dest assignments')
        values[name] = _literal(node.value, values)
    output = work/'output.json'
    inputs = st.load(diag/'inputs.json')
    if values['R'] != inputs['rollouts']['snapshot_path'] or values['dest'] != str(output):
        raise ValueError('delivery literal source or output path differs from this invocation')
    if actual['output'] != str(output) or not isinstance(values['out'], dict):
        raise ValueError('delivery literal does not target the declared aggregate output')
    error = st.load(directory/'error.json')
    if str(output) not in error.get('error', '') and 'output is missing' not in error.get('error', ''):
        raise ValueError('delivery recovery is restricted to missing-output failures')
    raw_output = json.dumps(values['out'], indent=2, allow_nan=False).encode()
    return finished, tool, script, raw_output


def _validate_delivery_payload(diag, raw_output):
    from compact_synthesis import enrich_aggregate
    from parallel_records import validate_aggregate
    from stage_audit import _load_observations
    from stage_records import verify_snapshot
    inputs = st.load(diag/'inputs.json')
    for name in ('original', 'rollouts'):
        verify_snapshot(inputs[name])
    path = diag/'observations/index.json'
    index = st.load(path)
    commits = [read_stage_commit(diag/'records', worker['commit_id']) for worker in index['workers']]
    observations = [item for commit in commits for item in commit['payload']['observations']]
    raw = st._load_json(raw_output)
    payload = raw if raw.get('schema_version') == 'parallel-attribution-v1' else enrich_aggregate(raw, observations)
    validate_aggregate(payload, observations)
    prospective = {'payload': payload, 'parent_ids': [c['commit_id'] for c in commits],
                   'references': [{'original_path': str(path), 'sha256': _sha(path.read_bytes())}]}
    _load_observations(diag, diag/'records', prospective)
    for commit in commits:
        for ref in commit['references']:
            path = Path(ref['original_path'])
            if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != ref['sha256']:
                raise ValueError('delivery recovery observation evidence changed')
    return index, commits, observations, payload


def recover_delivery(diag, *, expected_tool_call_id, expected_transcript_sha256):
    """Explicitly materialize the exact model-authored JSON literal; execute no code."""
    from host_validation import locked
    from parallel_attribution import _commit_aggregate
    diag = Path(diag).resolve(strict=True)
    root = diag/'attribution/delivery-recovery'
    receipt_path = root/'receipt.json'
    with locked(diag/'attribution/delivery-recovery.lock'):
        if receipt_path.is_file():
            validated = validate_delivery_recovery(diag, receipt_path)
            if (validated['tool_call_id'] != expected_tool_call_id
                    or validated['transcript_sha256'] != expected_transcript_sha256):
                raise ValueError('existing delivery recovery has a different explicit identity')
            return validated
        finished, tool, script, raw = _decode_delivery(diag, expected_tool_call_id, expected_transcript_sha256)
        index, commits, observations, _ = _validate_delivery_payload(diag, raw)
        output = Path(finished['output'])
        if output.exists() or (diag/'attribution_commit.json').exists():
            raise ValueError('delivery recovery refuses an existing untracked output or aggregate')
        prior = [diag/'attribution'/name for name in
                 ('error.json', 'finished.json', 'started.json', 'task/instruction.md',
                  'job/local-result.json', 'job/claude-code.txt', 'job/stderr.txt')]
        prior += [path for path in (diag/'started.json', diag.parent/'runtime-status.json') if path.is_file()]
        originals = {str(path): _sha(path.read_bytes()) for path in prior}
        root.mkdir(parents=True, exist_ok=True)
        tool_path, script_path = root/'tool-call.json', root/'original-writer.py'
        st.save_new(tool_path, tool)
        st.save_bytes(script_path, script.encode())
        source = {'schema_version': 'attribution-delivery-recovery-input-v1',
                  'origin': 'post_failure_delivery_recovery', 'method': 'restricted_ast_literal_materialization',
                  'tool_call_id': expected_tool_call_id, 'transcript_sha256': expected_transcript_sha256,
                  'tool_call_sha256': _sha(tool_path.read_bytes()), 'script_sha256': _sha(script.encode()),
                  'agent_terminal_status': finished['terminal']['agent_terminal_status'],
                  'output_path': str(output), 'output_sha256': _sha(raw), 'original_failure_files': originals}
        source_path = root/'input.json'
        st.save_new(source_path, source)
        # No arbitrary shell/Python is executed. This serializes only the complete
        # data expression that the failed, archived tool call would have dumped.
        st.save_bytes(output, raw)
        revision_input = diag/'attribution/revision-inputs/manifest.json'
        commit = _commit_aggregate(diag, finished, st.load(diag/'started.json'),
            diag/'observations/index.json', diag/'observations/input_inventory.json',
            index['sample_order'], commits, observations,
            revision_input=revision_input if revision_input.is_file() else None,
            delivery_recovery=source, recovery_refs=[source_path, tool_path, script_path])
        receipt = {'schema_version': 'attribution-delivery-recovery-v1',
                   'origin': 'post_failure_delivery_recovery', 'status': 'recovered',
                   'agent_terminal_status': source['agent_terminal_status'],
                   'attribution_commit': commit['commit_id'], 'input_receipt': str(source_path),
                   'input_receipt_sha256': _sha(source_path.read_bytes()),
                   'tool_call_id': expected_tool_call_id, 'transcript_sha256': expected_transcript_sha256}
        st.save_new(receipt_path, receipt)
        return validate_delivery_recovery(diag, receipt_path)


def validate_delivery_recovery(diag, receipt_path):
    """Read-only check for controllers that explicitly accept a recovered artifact."""
    from stage_audit import _load_observations
    diag, receipt_path = Path(diag).resolve(strict=True), Path(receipt_path).resolve(strict=True)
    root = diag/'attribution/delivery-recovery'
    if receipt_path != root/'receipt.json':
        raise ValueError('delivery recovery receipt belongs to a different analysis')
    receipt = st.load(receipt_path)
    if (receipt.get('schema_version') != 'attribution-delivery-recovery-v1'
            or receipt.get('origin') != 'post_failure_delivery_recovery' or receipt.get('status') != 'recovered'):
        raise ValueError('invalid explicit delivery recovery receipt')
    source_path = root/'input.json'
    if (receipt.get('input_receipt') != str(source_path)
            or receipt.get('input_receipt_sha256') != _sha(source_path.read_bytes())):
        raise ValueError('delivery recovery input receipt changed')
    source = st.load(source_path)
    for name, digest in source['original_failure_files'].items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != digest:
            raise ValueError('original failed invocation evidence changed after recovery')
    finished, tool, script, raw = _decode_delivery(diag, receipt['tool_call_id'], receipt['transcript_sha256'])
    index, commits, observations, payload = _validate_delivery_payload(diag, raw)
    output = Path(finished['output'])
    if output.read_bytes() != raw or source['output_sha256'] != _sha(raw):
        raise ValueError('recovered output differs from the complete original model literal')
    if (st.load(root/'tool-call.json') != tool or (root/'original-writer.py').read_text() != script
            or source['tool_call_sha256'] != _sha((root/'tool-call.json').read_bytes())
            or source['script_sha256'] != _sha(script.encode())):
        raise ValueError('delivery recovery tool-call evidence changed')
    commit = read_stage_commit(diag/'records', receipt['attribution_commit'])
    if (st.load(diag/'attribution_commit.json')['commit_id'] != commit['commit_id']
            or commit['payload'] != payload or st.load(diag/'attribution.json') != payload
            or commit['metadata'].get('post_failure_delivery_recovery') != source
            or commit['metadata']['stage_execution'] != finished
            or receipt['agent_terminal_status'] != finished['terminal']['agent_terminal_status']):
        raise ValueError('delivery recovery commit or original terminal identity changed')
    _load_observations(diag, diag/'records', commit)
    for ref in commit['references']:
        path = Path(ref['original_path'])
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != ref['sha256']:
            raise ValueError('delivery recovery committed evidence changed')
    return {**receipt, 'receipt_path': str(receipt_path)}
