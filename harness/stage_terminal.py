"""Read a stage's actual Harbor terminal receipt, separately from its artifacts."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


def read_terminal(job_dir: str | Path) -> dict:
    """Require exactly one finished trial; never infer success from an output file.

    Harbor writes both job and trial ``result.json`` files. The job summary is
    not a trial receipt. Older receipts can omit task_name, so the trial-specific
    execution/exception fields also identify a receipt.
    """
    trials = []
    for path in sorted(Path(job_dir).rglob('result.json')):
        raw = path.read_bytes()
        value = json.loads(raw)
        if isinstance(value, dict) and any(
                key in value for key in ('task_name', 'agent_execution', 'exception_info')):
            trials.append((path, raw, value))
    if len(trials) != 1:
        raise ValueError(f'expected exactly one Harbor terminal trial result, found {len(trials)} in {job_dir}')

    path, raw, value = trials[0]
    finished_at = value.get('finished_at')
    if not isinstance(finished_at, str) or not finished_at.strip():
        raise ValueError(f'Harbor trial has not terminated: missing finished_at in {path}')
    try:
        datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'invalid Harbor trial finished_at in {path}') from exc
    if 'exception_info' not in value:
        raise ValueError(f'Harbor terminal trial is missing exception_info in {path}')

    exception = value['exception_info']
    exception_type = exception_message = None
    status = 'completed'
    if exception is not None:
        if not isinstance(exception, dict) or not isinstance(exception.get('exception_type'), str) or not exception['exception_type']:
            raise ValueError(f'invalid Harbor terminal exception_info in {path}')
        exception_type = exception['exception_type']
        exception_message = exception.get('exception_message')
        status = 'timeout' if exception_type.endswith('TimeoutError') else 'error'
    return {
        'agent_terminal_status': status,
        'exception_type': exception_type,
        'exception_message': exception_message,
        'finished_at': finished_at,
        'trial_result_path': str(path.resolve()),
        'trial_result_sha256': hashlib.sha256(raw).hexdigest(),
    }
