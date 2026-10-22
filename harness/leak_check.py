#!/usr/bin/env python3
"""One-call check that a task instruction does not hand the solver the fix for a targeted class.

Input is only the instruction text and the targeted classes (claim statements); no
reference, verifier or trajectories. The model answers per class with a verbatim
quote; the host accepts a leak only when the quote occurs in the instruction.

  accepted  no class leaked
  rejected  at least one class leaked with a verified quote
  blocked   call failed, output invalid, or a leak quote is not in the instruction

One call, no retry. Raw request and response are written next to the receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

SCHEMA = 'leak-check-v1'
DEFAULT_MODEL = 'claude-opus-5.5'
RETRY_SLEEP = 10

PROMPT = """You check a spreadsheet task instruction before it is used as training data.
A target model has been observed to make specific errors (the classes below). The task
is meant to give the model a fair chance to make each of these errors. An instruction
LEAKS a class when it tells the solver how to avoid that error, or gives a value or
example that lets the solver detect the error without reasoning it out. Examples of
leaks: "do not multiply by 100" against a class that scales a fraction by 100; "the copy
in K9 must test K3 against C9:E9" against a reference-anchoring class; "10 June is a
Wednesday" or "there are 5 Saturdays" against a weekday-code class; naming the exact
column to use against a class about choosing the wrong column.
Not a leak: stating what the output means, where it goes, its units or format, as long
as that statement is needed to define the task and does not spell out the avoidance of
the specific error.

Answer for EVERY class. Output only a JSON array, one object per class, in order:
[{"class_id": "...", "leaked": true or false, "quote": "verbatim instruction text that leaks it, or null", "reason": "one sentence"}]
The quote must be copied exactly from the instruction.

Targeted classes:
__CLASSES__

Instruction:
<<<
__INSTRUCTION__
>>>
"""


def _norm(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def targeted_classes(diag):
    """Target mappings of a committed design, with their claim statements."""
    diag = Path(diag)
    design = json.loads((diag / 'design.json').read_text())
    claims = {c['id']: c for c in json.loads((diag / 'attribution.json').read_text())['claims']}
    out = []
    for m in design.get('mappings', []):
        if m.get('role') == 'target' and m.get('claim_id') in claims:
            out.append({'class_id': m['claim_id'], 'statement': claims[m['claim_id']]['statement']})
    return out


def _extract_json(text):
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end < start:
        raise ValueError('no JSON array in response')
    return json.loads(text[start:end + 1])


def judge(instruction, classes, answer):
    """Validate the model answer against the instruction; never repair it."""
    if not isinstance(answer, list):
        raise ValueError('answer is not a list')
    wanted = [c['class_id'] for c in classes]
    got = [a.get('class_id') if isinstance(a, dict) else None for a in answer]
    if got != wanted:
        raise ValueError(f'class ids {got} differ from targeted {wanted}')
    text = _norm(instruction)
    verdicts, status = [], 'accepted'
    for a in answer:
        if not isinstance(a.get('leaked'), bool):
            raise ValueError(f"{a['class_id']}: leaked is not a boolean")
        v = {'class_id': a['class_id'], 'leaked': a['leaked'], 'quote': a.get('quote'),
             'reason': a.get('reason'), 'quote_verified': None}
        if a['leaked']:
            v['quote_verified'] = bool(a.get('quote')) and _norm(a['quote']) in text
            if v['quote_verified']:
                status = 'rejected'
            elif status == 'accepted':
                status = 'blocked'
        verdicts.append(v)
    return status, verdicts


def call(base_url, api_key, model, prompt, timeout=600):
    body = json.dumps({'model': model, 'max_tokens': 4000,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    request = urllib.request.Request(base_url.rstrip('/') + '/v1/messages', data=body, headers={
        'content-type': 'application/json', 'x-api-key': api_key or 'local-relay',
        'anthropic-version': '2023-06-01'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check(instruction, classes, out_dir, *, base_url, api_key=None, model=DEFAULT_MODEL, caller=call):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listed = '\n'.join(f"- {c['class_id']}: {c['statement']}" for c in classes)
    prompt = PROMPT.replace('__CLASSES__', listed).replace('__INSTRUCTION__', instruction)
    (out_dir / 'leak-check-prompt.txt').write_text(prompt)
    receipt = {'schema_version': SCHEMA, 'model': model, 'classes': classes, 'status': 'blocked',
               'verdicts': [], 'error': None, 'usage': None, 'started_at': time.time()}
    if not classes:
        receipt.update(status='accepted', error=None, note='no targeted classes')
        return receipt
    error = None
    for attempt in range(3):
        try:
            response = caller(base_url, api_key, model, prompt)
            (out_dir / 'leak-check-response.json').write_text(json.dumps(response, ensure_ascii=False, indent=1))
            receipt['usage'] = response.get('usage')
            text = ''.join(b.get('text', '') for b in response.get('content', []) if b.get('type') == 'text')
            receipt['status'], receipt['verdicts'] = judge(instruction, classes, _extract_json(text))
            receipt['attempts'] = attempt + 1
            error = None
            break
        except urllib.error.HTTPError as exc:
            error = f'HTTPError: {exc}'
            if exc.code < 500:  # a client-side rejection is not retried
                break
            time.sleep(RETRY_SLEEP * (attempt + 1))
        except urllib.error.URLError as exc:
            error = f'URLError: {exc}'
            time.sleep(RETRY_SLEEP * (attempt + 1))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            break
    if error:
        receipt['error'] = error
    receipt['finished_at'] = time.time()
    return receipt


def run_for_episode(cfg, ep, *, base_url=None):
    """Write diagnostics/leak-check.json for the committed candidate; reuse a finished one."""
    diag = Path(ep) / 'diagnostics'
    path = diag / 'leak-check.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('status') != 'blocked':  # a transport-blocked check is retried, not cached
            return cached
        path.rename(diag / f"leak-check.blocked-{int(time.time())}.json")
    model = (cfg.get('training_pipeline') or {}).get('leak_check_model', DEFAULT_MODEL)
    receipt = check((diag / 'candidate/instruction.md').read_text(), targeted_classes(diag),
                    diag / 'leak-check', base_url=base_url or cfg['gateway']['base_url'],
                    api_key=cfg['gateway'].get('api_key'), model=model)
    path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + '\n')
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--diagnostics', required=True, help='episode diagnostics dir giving the targeted classes')
    ap.add_argument('--instruction', help='instruction to check (default: that episode candidate)')
    ap.add_argument('--out', required=True, help='receipt directory')
    ap.add_argument('--base-url', default=os.environ.get('ADB_MODEL_ENDPOINT', ''))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    diag = Path(args.diagnostics)
    instruction = Path(args.instruction or diag / 'candidate/instruction.md').read_text()
    receipt = check(instruction, targeted_classes(diag), args.out, base_url=args.base_url, model=args.model)
    receipt['instruction_path'] = str(args.instruction or diag / 'candidate/instruction.md')
    (Path(args.out) / 'leak-check.json').write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'status': receipt['status'], 'error': receipt['error'],
                      'leaks': [(v['class_id'], v['quote']) for v in receipt['verdicts'] if v['leaked']]},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
