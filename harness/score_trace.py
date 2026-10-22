"""Mechanical score trace: replay each attempt's successful edits, recalculate, compare.

Read-only on sources. Replays recorded successful edit calls onto a copy of the
original input workbook, recalculates with headless LibreOffice, scores with the
task's own vendored scorer, and reconciles against the official score receipt.
No model calls, no judgments: the output is evidence for diagnosis workers.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import openpyxl
from openpyxl.formula.translate import Translator
from openpyxl.utils import column_index_from_string, get_column_letter, range_boundaries

SCHEMA = 'score-trace-v1'
SUMMARY_NAME = 'score_trace_summary.json'
WORKER_NAME = 'score-trace.json'
EDIT_TOOLS = {'fill_formula', 'write_range', 'manage_sheet'}
READ_TOOLS = {'inspect_range', 'list_sheets', 'find_cells'}
EVENT_HEAD = re.compile(r'^### step (\d+) / event (\d+) / (\w+)\s*$', re.M)
FENCE = re.compile(r'```json\n(.*?)\n```', re.S)


def _first_json(text):
    match = FENCE.search(text)
    return json.loads(match.group(1)) if match else {}


def _event_text(block):
    match = re.search(r'#### Event text\n```\n(.*?)\n```', block, re.S)
    return match.group(1) if match else ''


def parse_attempt(path):
    """Return the official receipt, usage summary and ordered tool events."""
    text = Path(path).read_text()
    head = text[:EVENT_HEAD.search(text).start()] if EVENT_HEAD.search(text) else text
    summary = _first_json(head)
    termination = re.search(r'termination: (\w+)', head)
    heads = list(EVENT_HEAD.finditer(text))
    events = []
    for index, match in enumerate(heads):
        block = text[match.end(): heads[index + 1].start() if index + 1 < len(heads) else len(text)]
        if match.group(3) not in ('tool_call', 'tool_response'):
            continue
        meta = _first_json(block)
        events.append({'step': int(match.group(1)), 'event': int(match.group(2)), 'kind': match.group(3),
                       'tool': meta.get('tool_name'), 'status': meta.get('execution_status'),
                       'arguments': meta.get('arguments'), 'response_event': meta.get('response_event_index'),
                       'metrics': meta.get('metrics') or {}, 'text': _event_text(block)})
    score = summary.get('official_score') or {}
    return {'reward': score.get('reward'), 'msg': score.get('msg', ''), 'usage': summary.get('usage') or {},
            'duration_seconds': summary.get('duration_seconds'), 'termination': termination.group(1) if termination else None,
            'events': events}


def parse_trial(trial_dir):
    """Same shape as parse_attempt, for a host validation trial (adb-native-digest-v1)."""
    trial_dir = Path(trial_dir)
    digest = json.loads((trial_dir / 'agent/trajectory.json').read_text())
    details_path = trial_dir / 'verifier/details.json'
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    events, pending = [], None
    for step in digest.get('steps', []):
        message, extra = str(step.get('message', '')), step.get('extra') or {}
        if message.startswith('[tool_call]'):
            body = re.search(r'<tool_call>\n(.*?)\n</tool_call>', message, re.S)
            try:
                call = json.loads(body.group(1)) if body else {}
            except ValueError:
                call = {}
            pending = {'step': extra.get('native_step'), 'event': step['step_id'], 'kind': 'tool_call',
                       'tool': extra.get('tool_name') or call.get('name'), 'status': extra.get('execution_status'),
                       'arguments': call.get('arguments'), 'response_event': None, 'metrics': {}, 'text': message}
            events.append(pending)
        elif message.startswith('[tool_response]'):
            if pending:
                pending['response_event'] = step['step_id']
            events.append({'step': extra.get('native_step'), 'event': step['step_id'], 'kind': 'tool_response',
                           'tool': pending['tool'] if pending else extra.get('tool_name'),
                           'status': pending['status'] if pending else extra.get('execution_status'), 'arguments': None,
                           'response_event': None, 'metrics': {}, 'text': message[len('[tool_response]'):].strip()})
            pending = None
    return {'reward': details.get('reward'), 'msg': details.get('msg', ''), 'usage': digest.get('final_metrics') or {},
            'duration_seconds': None, 'termination': None, 'events': events}


def _sheet(wb, name, create=False):
    if name in wb.sheetnames:
        return wb[name]
    folded = {title.casefold(): title for title in wb.sheetnames}
    if name and name.casefold() in folded:
        return wb[folded[name.casefold()]]
    if create:
        return wb.create_sheet(name)
    raise KeyError(f'sheet not found: {name!r}')


def _grid(data, rows, cols, anchor):
    """Shape write_range data as the native tool does: scalar, 1D row, or 2D, expanded to the range."""
    if isinstance(data, str) and data.lstrip().startswith('['):
        try:
            data = json.loads(data)
        except ValueError:
            pass
    if not isinstance(data, list):
        grid = [[data]]
    elif not any(isinstance(item, list) for item in data):
        grid = [[item] for item in data] if not anchor and cols == 1 and rows == len(data) else [data]
    else:
        grid = data
    if anchor or (len(grid) == rows and len(grid[0]) == cols):
        return grid
    if len(grid) == 1 and len(grid[0]) == 1:
        return [[grid[0][0]] * cols for _ in range(rows)]
    if len(grid) == 1 and len(grid[0]) == cols:
        return [list(grid[0]) for _ in range(rows)]
    if len(grid[0]) == 1 and len(grid) == rows:
        return [[row[0]] * cols for row in grid]
    raise ValueError('write_range data shape does not match its range')


def apply_edit(wb, tool, args, reported_formula=None):
    """Replay one recorded successful edit; return the cells it wrote.

    reported_formula is the start cell's formula as the tool response recorded it,
    i.e. after the native tool's own normalization of the submitted template.
    """
    if tool == 'fill_formula':
        ws = _sheet(wb, args.get('sheet_name') or wb.sheetnames[0])
        start = args['start_cell'].replace('$', '').split('!')[-1]
        col = re.match(r'([A-Z]+)(\d+)', start.upper())
        start_col, start_row = column_index_from_string(col.group(1)), int(col.group(2))
        end_row = int(args.get('end_row') or start_row)
        end_col = column_index_from_string(str(args.get('end_col') or col.group(1)).replace('$', '').upper())
        template, written = reported_formula or args['formula_template'], []
        if not template.startswith('='):
            template = '=' + template
        for row in range(start_row, end_row + 1):
            for column in range(start_col, end_col + 1):
                address = f'{get_column_letter(column)}{row}'
                ws[address] = Translator(template, origin=start.upper()).translate_formula(address)
                written.append(f'{ws.title}!{address}')
        return written
    if tool == 'write_range':
        ws = _sheet(wb, args.get('sheet_name') or wb.sheetnames[0], create=True)
        label = args['range'].replace('$', '').split('!')[-1]
        min_col, min_row, max_col, max_row = range_boundaries(label)
        written = []
        for r_offset, values in enumerate(_grid(args.get('data'), max_row - min_row + 1, max_col - min_col + 1, ':' not in label)):
            for c_offset, value in enumerate(values):
                cell = ws.cell(min_row + r_offset, min_col + c_offset)
                if value is None:
                    continue
                cell.value = None if value == '' else value[1:] if isinstance(value, str) and value.startswith("'") else value
                written.append(f'{ws.title}!{cell.coordinate}')
        return written
    if tool == 'manage_sheet':
        action = args.get('action')
        if action == 'create' and args.get('sheet_name') not in wb.sheetnames:
            wb.create_sheet(args['sheet_name'])
        elif action == 'rename':
            _sheet(wb, args['sheet_name']).title = args['new_name']
        elif action == 'delete':
            wb.remove(_sheet(wb, args['sheet_name']))
        return []
    raise ValueError(f'unsupported edit tool: {tool}')


def recalc(paths, outdir):
    """Recalculate workbooks with headless LibreOffice, as the scorer does."""
    if not paths:
        raise ValueError('no workbooks to recalculate')
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([shutil.which('soffice') or 'soffice', f'-env:UserInstallation=file://{profile}', '--headless',
                        '--norestore', '--nologo', '--nolockcheck', '--calc', '--convert-to', 'xlsx', '--outdir',
                        str(outdir), *map(str, paths)], check=True, capture_output=True, timeout=600)
    return {Path(p).stem: outdir / f'{Path(p).stem}.xlsx' for p in paths}


def _scorer(tests):
    sys.path.insert(0, str(Path(tests) / 'vendor'))
    # The scorer is imported from a hashed input snapshot; bytecode must not be written into it.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        from async_reward_api.eval import compute_reward, compare_cell_value
    finally:
        sys.dont_write_bytecode = previous
    return compute_reward, compare_cell_value


def _target_cells(tests):
    spec = json.loads((Path(tests) / 'task_spec.json').read_text())
    cells = []
    for item in spec.get('target_ranges') or []:
        min_col, min_row, max_col, max_row = range_boundaries(item['range'])
        for row in range(min_row, max_row + 1):
            for column in range(min_col, max_col + 1):
                cells.append((item['sheet'], f'{get_column_letter(column)}{row}'))
    return spec['answer_position'], cells


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _values(path, cells, data_only=True):
    wb = openpyxl.load_workbook(path, data_only=data_only)
    out = {}
    for sheet, address in cells:
        try:
            out[(sheet, address)] = _sheet(wb, sheet)[address].value
        except KeyError:
            out[(sheet, address)] = None
    return out


FEEDBACK_ERROR = re.compile(r'#(?:VALUE!|NAME\?|REF!|DIV/0!|N/A|NUM!|NULL!)')
FEEDBACK_FLAGS = ('error_value', 'all_zero')


def _feedback_values(text):
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    samples = payload.get('samples') if isinstance(payload, dict) else None
    return [s.get('value') for s in samples] if isinstance(samples, list) else None


def last_feedback(events):
    """Mechanical flags on the last successful edit response; not a judgment."""
    responses = [e for e in events if e['kind'] == 'tool_response' and e['tool'] in EDIT_TOOLS and e['status'] == 'success']
    if not responses:
        return None
    last = responses[-1]
    values = _feedback_values(last['text'])
    flags = []
    if values is not None:
        if any(isinstance(v, str) and FEEDBACK_ERROR.search(v) for v in values):
            flags.append('error_value')
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and len(numeric) == len(values) and all(v == 0 for v in numeric):
            flags.append('all_zero')
        if numeric and len(numeric) == len(values) and len(set(numeric)) == 1 and len(values) > 1:
            flags.append('all_identical')
    later_calls = sum(1 for e in events if e['kind'] == 'tool_call' and e['event'] > last['event'])
    return {'event': last['event'], 'sample_values': values, 'flags': flags, 'tool_calls_after': later_calls}


def _reported_formula(event, responses):
    """Start-cell formula as recorded by the tool response, when it reports one."""
    response = responses.get(event.get('response_event'))
    if event['tool'] != 'fill_formula' or not response:
        return None
    try:
        samples = json.loads(response['text']).get('samples') or []
    except (ValueError, AttributeError):
        return None
    start = (event['arguments'] or {}).get('start_cell', '').replace('$', '').split('!')[-1].upper()
    for sample in samples:
        if str(sample.get('address', '')).split('!')[-1].upper() == start and isinstance(sample.get('formula'), str):
            return sample['formula']
    return None


def _replay(events, source, destination):
    """Copy the input workbook and replay recorded successful edits in order."""
    shutil.copyfile(source, destination)
    wb = openpyxl.load_workbook(destination)
    edits, failures = [], []
    responses = {e['event']: e for e in events if e['kind'] == 'tool_response'}
    for event in events:
        if event['kind'] != 'tool_call' or event['tool'] not in EDIT_TOOLS or event['status'] != 'success':
            continue
        try:
            reported = _reported_formula(event, responses)
            cells = apply_edit(wb, event['tool'], event['arguments'] or {}, reported)
            edits.append({'event': event['event'], 'tool': event['tool'], 'arguments': event['arguments'],
                          'formula_as_written': reported,
                          'cells_written': len(cells), 'range': f'{cells[0]}:{cells[-1].split("!")[-1]}' if cells else None})
        except Exception as exc:
            failures.append({'event': event['event'], 'tool': event['tool'], 'error': f'{type(exc).__name__}: {exc}'})
    wb.save(destination)
    return edits, failures


def _receipt_cell(msg):
    match = re.search(r'mismatch at ([A-Z]+\d+): your workbook has (.*)$', msg or '')
    return (match.group(1), match.group(2)) if match else (None, None)


def _same_as_receipt(value, shown):
    if shown is None:
        return None
    if value is None:
        return shown == 'None'
    if isinstance(value, (str, bool)):
        return shown in (str(value), repr(value))
    try:
        return float(shown) == float(value)
    except ValueError:
        return False


def _call_signature(event):
    return json.dumps([event['tool'], event['arguments']], sort_keys=True, ensure_ascii=False)


def process_stats(attempt):
    """Mechanical process counts used as comparison evidence for efficiency notes."""
    calls = [e for e in attempt['events'] if e['kind'] == 'tool_call']
    seen, identical, streak, longest = set(), 0, 0, 0
    previous = None
    for call in calls:
        signature = _call_signature(call)
        identical += signature in seen
        seen.add(signature)
        streak = streak + 1 if call['status'] == 'error' and signature == previous else (1 if call['status'] == 'error' else 0)
        longest = max(longest, streak)
        previous = signature
    usage = attempt['usage']
    return {'tool_calls': len(calls), 'read_calls': sum(c['tool'] in READ_TOOLS for c in calls),
            'edit_calls': sum(c['tool'] in EDIT_TOOLS for c in calls),
            'failed_calls': sum(c['status'] == 'error' for c in calls), 'identical_repeat_calls': identical,
            'longest_identical_error_streak': longest, 'completion_tokens': usage.get('completion_tokens'),
            'total_tokens': usage.get('total_tokens'), 'duration_seconds': attempt['duration_seconds']}


def trace(task_dir, sources, out_dir, *, worker_dir=None):
    """Write one score trace per attempt or trial plus a summary; returns the summary.

    sources: attempt-*.md trajectory digests, or host trial dirs holding agent/trajectory.json.
    worker_dir: when given, also publish the single-attempt view each diagnosis
    worker may read, with the cross-attempt median removed.
    """
    task_dir, sources, out_dir = Path(task_dir), Path(sources), Path(out_dir)
    tests, source = task_dir / 'tests', task_dir / 'environment/input.xlsx'
    compute_reward, compare = _scorer(tests)
    answer_position, cells = _target_cells(tests)
    expected = _values(tests / 'target.xlsx', cells)
    replay_dir, recalc_dir = out_dir / 'replayed', out_dir / 'recalculated'
    replay_dir.mkdir(parents=True, exist_ok=True)
    attempts = {}
    trials = sorted(sources.glob('agent/trajectory.json')) or sorted(sources.glob('job/*/candidate__*/agent/trajectory.json'))
    if trials:
        label = next((part for part in reversed(sources.parts) if re.search(r'-\d+$', part)), '0')
        found = [(int(label.split('-')[-1]), t.parent.parent, parse_trial(t.parent.parent)) for t in trials[:1]]
    else:
        book = sorted(sources.glob('attempt-*.md'), key=lambda p: int(p.stem.split('-')[1]))
        found = [(int(p.stem.split('-')[1]), p, parse_attempt(p)) for p in book]
    if not found:
        raise ValueError(f'no attempt-*.md digests or trial trajectories under {sources}')
    for sample, path, attempt in found:
        workbook = replay_dir / f'sample-{sample:02d}.xlsx'
        edits, failures = _replay(attempt['events'], source, workbook)
        attempts[sample] = {'path': path, 'attempt': attempt, 'workbook': workbook, 'edits': edits, 'replay_failures': failures}
    recalculated = recalc([a['workbook'] for a in attempts.values()], recalc_dir)
    for sample, item in attempts.items():
        item['recalculated'] = recalculated.get(f'sample-{sample:02d}')
    return _write_traces(attempts, cells, expected, compare, compute_reward, tests, answer_position, out_dir,
                         worker_dir=Path(worker_dir) if worker_dir else None)


def _reconcile(item, reward, receipt_cell, receipt_shown, mismatched, actual):
    """Compare the replay with the official receipt; disagreement is not explained here."""
    official = item['attempt']['reward']
    checks = {'reward_matches': official is None or float(official) == reward}
    if receipt_cell:
        first = mismatched[0][1] if mismatched else None
        cell_value = next((v for (s, a), v in actual.items() if a == receipt_cell), None)
        checks['receipt_cell_is_first_mismatch'] = first == receipt_cell
        checks['receipt_value_matches'] = _same_as_receipt(cell_value, receipt_shown)
    agreed = all(v is not False for v in checks.values()) and not item['replay_failures']
    return {'status': 'reproduced' if agreed else 'not_reproduced', **checks}


def _final_edits(edits, cells):
    """Last recorded edit covering each target cell: the actions that determined the score."""
    covering = {}
    for edit in edits:
        if not edit['range']:
            continue
        sheet, span = edit['range'].split('!')
        min_col, min_row, max_col, max_row = range_boundaries(span)
        for target_sheet, address in cells:
            col, row = range_boundaries(address)[:2]
            if target_sheet.casefold() == sheet.casefold() and min_col <= col <= max_col and min_row <= row <= max_row:
                covering[(target_sheet, address)] = edit['event']
    events = sorted(set(covering.values()))
    return [{'event': e, 'target_cells': sum(v == e for v in covering.values()),
             **{k: v for k, v in next(x for x in edits if x['event'] == e).items() if k in ('tool', 'arguments', 'range', 'formula_as_written')}}
            for e in events], len(cells) - len(covering)


def _write_traces(attempts, cells, expected, compare, compute_reward, tests, answer_position, out_dir,
                  *, worker_dir=None):
    summary = []
    for sample, item in sorted(attempts.items()):
        recalculated = item['recalculated']
        if recalculated and recalculated.exists():
            reward, msg = compute_reward(tests / 'target.xlsx', recalculated, answer_position)
            actual = _values(recalculated, cells)
            formulas = _values(recalculated, cells, data_only=False)
        else:
            reward, msg, actual, formulas = 0.0, 'recalculation produced no workbook', {c: None for c in cells}, {c: None for c in cells}
        mismatched = [c for c in cells if not compare(expected[c], actual[c])]
        receipt_cell, receipt_shown = _receipt_cell(item['attempt']['msg'])
        final, uncovered = _final_edits(item['edits'], cells)
        record = {
            'schema_version': SCHEMA, 'sample_id': sample, 'source': str(item['path']),
            'official': {'reward': item['attempt']['reward'], 'msg': item['attempt']['msg'],
                         'termination': item['attempt']['termination']},
            'replay': {'reward': reward, 'msg': msg, 'replayed_edits': len(item['edits']),
                       'replay_failures': item['replay_failures']},
            'reconciliation': _reconcile(item, reward, receipt_cell, receipt_shown, mismatched, actual),
            'final_edits': final, 'target_cells_never_written': uncovered,
            'cells': {'total': len(cells), 'mismatched': len(mismatched),
                      'rows': [{'cell': f'{s}!{a}', 'formula': _jsonable(formulas[(s, a)]),
                                'recalculated': _jsonable(actual[(s, a)]), 'expected': _jsonable(expected[(s, a)]),
                                'matches': (s, a) not in mismatched} for s, a in cells]},
            'last_edit_feedback': last_feedback(item['attempt']['events']),
            'process': process_stats(item['attempt']),
        }
        summary.append(record)
    result = _summarize(summary, out_dir)
    for record in summary:
        # The only cross-attempt figure a worker sees: medians over officially solved attempts.
        record['solved_process_median'] = result['solved_process_median']
        (out_dir / f"sample-{record['sample_id']:02d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n')
        if worker_dir is not None:
            single = {k: v for k, v in record.items() if k != 'solved_process_median'}
            target = worker_dir / f'sample-{record["sample_id"]:02d}' / WORKER_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(single, ensure_ascii=False, indent=1) + '\n')
    return result


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def _summarize(records, out_dir):
    solved = [r for r in records if r['official']['reward'] == 1]
    baseline = {'solved_attempts': len(solved),
                **{name: _median(r['process'][name] for r in solved)
                   for name in ('tool_calls', 'completion_tokens', 'duration_seconds')}}
    rows = []
    for r in records:
        first = next((c for c in r['cells']['rows'] if not c['matches']), None)
        feedback = r['last_edit_feedback'] or {}
        rows.append({'sample_id': r['sample_id'], 'official_reward': r['official']['reward'],
                     'replay_reward': r['replay']['reward'], 'reconciliation': r['reconciliation']['status'],
                     'mismatched_cells': r['cells']['mismatched'], 'first_mismatch': first,
                     'final_formulas': [e.get('formula_as_written') or (e.get('arguments') or {}).get('formula_template')
                                        for e in r['final_edits'] if e['tool'] == 'fill_formula'],
                     'target_cells_never_written': r['target_cells_never_written'],
                     'last_feedback_flags': feedback.get('flags', []), 'tool_calls_after_last_edit': feedback.get('tool_calls_after'),
                     'process': r['process']})
    summary = {'schema_version': SCHEMA, 'attempts': len(records),
               'reproduced': sum(r['reconciliation'] == 'reproduced' for r in rows),
               'not_reproduced': [r['sample_id'] for r in rows if r['reconciliation'] != 'reproduced'],
               'solved_process_median': baseline, 'rows': rows}
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    return summary


def feedback_condition(record):
    """The mechanical FEEDBACK: trigger: a visible failure left uncorrected."""
    feedback = record.get('last_edit_feedback') or {}
    return (bool(set(feedback.get('flags') or []) & set(FEEDBACK_FLAGS))
            and feedback.get('tool_calls_after') == 0)


def feedback_samples(summary):
    """Sample IDs of unsolved, attributed attempts that satisfy FEEDBACK:."""
    rows = summary.get('rows', []) if isinstance(summary, dict) else []
    return [row['sample_id'] for row in rows
            if row.get('reconciliation') == 'reproduced' and (row.get('official_reward') or 0) < 1
            and set(row.get('last_feedback_flags') or []) & set(FEEDBACK_FLAGS)
            and row.get('tool_calls_after_last_edit') == 0]


def validate_probes(variants, summary):
    """Reject probes whose declared source cannot come from these attempts."""
    rows = {row['sample_id']: row for row in summary.get('rows', [])}
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or not isinstance(variant.get('name'), str) or not variant['name'].strip():
            raise ValueError(f'probe {index} requires a nonempty name')
        source = variant.get('source')
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f'probe {index} requires a declared source')
        if source == 'authored':
            continue
        identifiers = source.split(',')
        if any(not item.startswith('s') or not item[1:].isdigit() for item in identifiers):
            raise ValueError(f'probe {index} source must be "authored" or comma-separated sample ids')
        for item in identifiers:
            sample = int(item[1:])
            if sample not in rows:
                raise ValueError(f'probe {index} cites unknown attempt {item}')
            if (rows[sample].get('official_reward') or 0) >= 1:
                raise ValueError(f'probe {index} cites solved attempt {item}')
        if not isinstance(variant.get('edits'), list) or not variant['edits']:
            raise ValueError(f'probe {index} requires a nonempty edits list')


def probe(task_dir, variants, out_dir):
    """Recalculate named edit lists on the task input; used for counterfactuals and wrong-answer probes.

    variants: [{"name": "weekday-only", "edits": [{"tool": "fill_formula", "arguments": {...}}]}]
    """
    task_dir, out_dir = Path(task_dir), Path(out_dir)
    tests = task_dir / 'tests'
    _, compare = _scorer(tests)
    _, cells = _target_cells(tests)
    expected = _values(tests / 'target.xlsx', cells)
    (out_dir / 'replayed').mkdir(parents=True, exist_ok=True)
    books = {}
    for index, variant in enumerate(variants):
        events = [{'kind': 'tool_call', 'event': n, 'tool': e['tool'], 'status': 'success', 'arguments': e['arguments']}
                  for n, e in enumerate(variant['edits'])]
        path = out_dir / 'replayed' / f'probe-{index:02d}.xlsx'
        _, failures = _replay(events, task_dir / 'environment/input.xlsx', path)
        books[index] = (path, failures)
    recalculated = recalc([p for p, _ in books.values()], out_dir / 'recalculated')
    results = []
    for index, variant in enumerate(variants):
        actual = _values(recalculated[f'probe-{index:02d}'], cells)
        wrong = [c for c in cells if not compare(expected[c], actual[c])]
        results.append({'name': variant['name'], 'replay_failures': books[index][1], 'mismatched_cells': len(wrong),
                        'total_cells': len(cells), 'distinguishable_from_reference': bool(wrong),
                        'rows': [{'cell': f'{s}!{a}', 'recalculated': _jsonable(actual[(s, a)]),
                                  'expected': _jsonable(expected[(s, a)])} for s, a in cells]})
    (out_dir / 'probes.json').write_text(json.dumps({'schema_version': SCHEMA, 'probes': results}, ensure_ascii=False, indent=1) + '\n')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path, help='original task dir with environment/ and tests/')
    parser.add_argument('--rollouts', type=Path, help='dir with attempt-N.md digests or one host trial dir')
    parser.add_argument('--probes', type=Path, help='JSON list of named edit lists to recalculate instead of attempts')
    parser.add_argument('--trace-summary', type=Path, help='score trace summary a probe file must cite')
    parser.add_argument('--worker-dir', type=Path, help='also write one filleted single-attempt trace per sample')
    parser.add_argument('--out', required=True, type=Path, help='new output dir; must not exist')
    args = parser.parse_args(argv)
    if bool(args.rollouts) == bool(args.probes):
        raise SystemExit('pass exactly one of --rollouts or --probes')
    if args.out.exists():
        raise SystemExit(f'output exists; preserve it: {args.out}')
    args.out.mkdir(parents=True)
    if args.probes:
        variants = json.loads(args.probes.read_text())
        if args.trace_summary:
            validate_probes(variants, json.loads(args.trace_summary.read_text()))
        results = probe(args.task, variants, args.out)
        print(json.dumps([{k: r[k] for k in ('name', 'mismatched_cells', 'total_cells')} for r in results], ensure_ascii=False))
        return 0
    summary = trace(args.task, args.rollouts, args.out, worker_dir=args.worker_dir)
    print(json.dumps({k: summary[k] for k in ('attempts', 'reproduced', 'not_reproduced')}, ensure_ascii=False))
    return 0 if not summary['not_reproduced'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
