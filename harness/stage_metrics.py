#!/usr/bin/env python3
"""Deterministic diagnostic metrics from explicit automatic audit judgments.

These are automatic estimates, not human-validated accuracy or AutoDataBench's
paper score. No model calls, text truncation, token limits, or semantic inference
are performed. Missing evidence never receives an invented zero score.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re


GROUPS = ("claims", "mappings", "task_checks", "probes")
PARALLEL_GROUPS = (*GROUPS, "observations")
VERDICTS = ("correct", "incorrect", "uncertain")
METRIC_NAMES = {
    "claims": "attribution_correctness",
    "observations": "observation_correctness",
    "mappings": "design_correctness",
    "task_checks": "task_correctness",
    "probes": "verifier_correctness",
}
RULES = {
    "accuracy": "correct / (correct + incorrect); empty denominator is null",
    "false_accept_rate": "FP / (FP + TN); independently invalid completed probes only",
    "false_reject_rate": "FN / (FN + TP); independently valid completed probes only",
    "mapping_conjunction": (
        "incorrect if either upstream_verdict or implementation_verdict is incorrect; "
        "correct if both are correct; otherwise uncertain"
    ),
    "probe": (
        "Only execution_status=completed with boolean expected_valid and actual_accept "
        "enters accuracy. Completed probes with expected_valid=null are uncertain; "
        "infra_error and missing executions are reported separately."
    ),
    "evaluation_coverage": (
        "(correct + incorrect + uncertain) / declared units; infrastructure and missing "
        "executions are excluded from its numerator"
    ),
    "decidable_coverage": "(correct + incorrect) / declared units",
    "scope_coverage": (
        "source_total counts all source units; scope_total counts assigned review units. "
        "out_of_scope_count is source minus assigned scope. unreviewed_count counts source units "
        "without an accepted correct/incorrect/uncertain verdict. Missing/infra executions and "
        "failed audit payloads have no accepted verdicts. source_review_coverage is accepted "
        "reviewed units / source_total; accuracy remains null without decided judgments."
    ),
    "empty_or_failed": (
        "Completed empty lists have observed counts of zero and null rates. A failed "
        "audit without payload has unknown counts/rates (null); explicitly declared "
        "expected units, if supplied, give its known missing-unit count."
    ),
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mapping_verdict(upstream_verdict, implementation_verdict):
    """Public three-valued conjunction, with no semantic judgment."""
    if upstream_verdict not in VERDICTS or implementation_verdict not in VERDICTS:
        raise ValueError("mapping verdicts must be correct, incorrect, or uncertain")
    if "incorrect" in (upstream_verdict, implementation_verdict):
        return "incorrect"
    if upstream_verdict == implementation_verdict == "correct":
        return "correct"
    return "uncertain"


def _expected(expected_units):
    if expected_units is None:
        return None
    if not isinstance(expected_units, dict) or set(expected_units) not in (set(GROUPS), set(PARALLEL_GROUPS)):
        raise ValueError("expected_units must declare four groups, optionally observations")
    for group, ids in expected_units.items():
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError(f"expected_units.{group} must be a list of nonempty IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate expected_units ID in {group}")
    return expected_units


def _groups(expected_units=None, payload=None):
    declared = expected_units if expected_units is not None else payload
    return PARALLEL_GROUPS if isinstance(declared, dict) and "observations" in declared else GROUPS


def _units(payload, expected_units):
    if not isinstance(payload, dict):
        raise ValueError("audit payload must be an object")
    groups = _groups(expected_units, payload)
    missing = set(groups) - set(payload)
    if missing:
        raise ValueError("audit payload lacks groups: " + ", ".join(sorted(missing)))
    if expected_units is not None and ("observations" in payload) != ("observations" in expected_units):
        raise ValueError("observation group differs from explicitly expected units")
    for group in groups:
        items = payload[group]
        if not isinstance(items, list):
            raise ValueError(f"payload.{group} must be a list")
        ids = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError(f"payload.{group} requires object units with nonempty IDs")
            ids.append(item["id"])
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate unit ID in {group}")
        if expected_units is not None and set(ids) != set(expected_units[group]):
            raise ValueError(f"payload.{group} IDs differ from explicitly expected units")


def _unknown_metric(group, expected_ids):
    known_total = len(expected_ids) if expected_ids is not None else None
    result = {
        "name": METRIC_NAMES[group], "total": known_total, "correct": None,
        "incorrect": None, "uncertain": None, "missing": known_total,
        "infra_error": None, "evaluated": None, "numerator": None,
        "denominator": None, "accuracy": None, "evaluation_coverage": None,
        "decidable_coverage": None, "status": "audit_payload_missing", "units": None,
    }
    if group == "probes":
        result.update(execution_status_counts=None, expected_valid_missing=None,
                      TP=None, TN=None, FP=None, FN=None,
                      false_accept_rate=None, false_reject_rate=None,
                      invalid_denominator=None, valid_denominator=None)
    return result


def _support_metric(group, items):
    counts = {v: 0 for v in VERDICTS}
    units = []
    for item in items:
        if group == "mappings":
            verdict = mapping_verdict(item.get("upstream_verdict"), item.get("implementation_verdict"))
        else:
            verdict = item.get("verdict")
            if verdict not in VERDICTS:
                raise ValueError(f"{group}.{item['id']}: invalid or missing verdict")
        counts[verdict] += 1
        units.append({"audit": copy.deepcopy(item), "effective_verdict": verdict})
    denominator = counts["correct"] + counts["incorrect"]
    return {
        "name": METRIC_NAMES[group], "status": "computed", "total": len(items),
        **counts, "missing": 0, "infra_error": 0, "evaluated": len(items),
        "numerator": counts["correct"], "denominator": denominator,
        "accuracy": ratio(counts["correct"], denominator),
        "evaluation_coverage": ratio(len(items), len(items)),
        "decidable_coverage": ratio(denominator, len(items)), "units": units,
    }


def _probe_metric(items):
    counts = {v: 0 for v in (*VERDICTS, "missing", "infra_error")}
    matrix = {key: 0 for key in ("TP", "TN", "FP", "FN")}
    execution = {key: 0 for key in ("completed", "infra_error", "missing")}
    units = []
    expected_missing = 0
    for item in items:
        label = f"probes.{item['id']}"
        if "expected_valid" not in item or "actual_accept" not in item:
            raise ValueError(f"{label}: expected_valid and actual_accept must be explicit")
        expected, actual = item["expected_valid"], item["actual_accept"]
        if expected is not None and type(expected) is not bool:
            raise ValueError(f"{label}: expected_valid must be bool or null")
        if actual is not None and type(actual) is not bool:
            raise ValueError(f"{label}: actual_accept must be bool or null")
        status = item.get("execution_status")
        if status not in execution:
            raise ValueError(f"{label}: invalid or missing execution_status")
        if status == "completed" and type(actual) is not bool:
            raise ValueError(f"{label}: completed execution requires boolean actual_accept")
        if status != "completed" and actual is not None:
            raise ValueError(f"{label}: missing/infra_error execution requires null actual_accept")
        execution[status] += 1
        expected_missing += expected is None
        if status != "completed":
            outcome = status
        elif expected is None:
            outcome = "uncertain"
        else:
            outcome = "correct" if actual == expected else "incorrect"
            matrix[("T" if actual == expected else "F") + ("P" if actual else "N")] += 1
        counts[outcome] += 1
        units.append({"audit": copy.deepcopy(item), "effective_verdict": outcome})
    denominator = counts["correct"] + counts["incorrect"]
    evaluated = denominator + counts["uncertain"]
    return {
        "name": METRIC_NAMES["probes"], "status": "computed", "total": len(items),
        **counts, **matrix, "evaluated": evaluated,
        "execution_status_counts": execution, "expected_valid_missing": expected_missing,
        "invalid_denominator": matrix["FP"] + matrix["TN"],
        "valid_denominator": matrix["FN"] + matrix["TP"],
        "false_accept_rate": ratio(matrix["FP"], matrix["FP"] + matrix["TN"]),
        "false_reject_rate": ratio(matrix["FN"], matrix["FN"] + matrix["TP"]),
        "numerator": counts["correct"], "denominator": denominator,
        "accuracy": ratio(counts["correct"], denominator),
        "evaluation_coverage": ratio(evaluated, len(items)),
        "decidable_coverage": ratio(denominator, len(items)), "units": units,
    }


def _scope_metrics(metrics, expected_units, audit_scope):
    """Add mechanical review coverage; omitted source units never acquire verdicts."""
    descriptor = copy.deepcopy(audit_scope)
    if descriptor is not None:
        if (not isinstance(descriptor, dict) or descriptor.get('schema_version') != 'audit-scope-v1'
                or descriptor.get('mode') not in ('full', 'selected_design')):
            raise ValueError('invalid audit scope descriptor')
        source = _expected(descriptor.get('source_units'))
        scoped = _expected(descriptor.get('expected_units'))
        if source is None or scoped is None or set(source) != set(scoped) or set(source) != set(metrics):
            raise ValueError('audit scope source and expected groups differ')
        unreviewed = descriptor.get('unreviewed_ids')
        if not isinstance(unreviewed, dict) or set(unreviewed) != set(source):
            raise ValueError('audit scope requires explicit unreviewed IDs')
        for group in source:
            if not set(scoped[group]).issubset(source[group]):
                raise ValueError('review scope contains units absent from source')
            if unreviewed[group] != [identifier for identifier in source[group] if identifier not in scoped[group]]:
                raise ValueError('unreviewed IDs differ from source minus scope')
            if descriptor['mode'] == 'full' and unreviewed[group]:
                raise ValueError('full scope cannot omit source units')
            actual = (expected_units[group] if expected_units is not None else
                      [unit['audit']['id'] for unit in metrics[group]['units']]
                      if metrics[group]['units'] is not None else None)
            if actual is not None:
                if group == 'probes':
                    if not set(scoped[group]).issubset(actual):
                        raise ValueError('review omitted required scope probes')
                elif set(actual) != set(scoped[group]):
                    raise ValueError('audit units differ from declared scope')
    else:
        source = expected_units
        scoped = expected_units
    for group, metric in metrics.items():
        source_ids = list(source[group]) if source is not None else None
        scope_ids = list(scoped[group]) if scoped is not None else None
        # Extra probes are declared by the frozen probe plan, beyond the fixed
        # required positive/negative pair. Every extra probe remains in scope.
        if descriptor is not None and group == 'probes':
            declared = (expected_units[group] if expected_units is not None else
                        [unit['audit']['id'] for unit in metric['units']] if metric['units'] is not None else [])
            for identifier in declared:
                if identifier not in source_ids:
                    source_ids.append(identifier)
                if identifier not in scope_ids:
                    scope_ids.append(identifier)
        source_total = len(source_ids) if source_ids is not None else metric['total']
        scope_total = len(scope_ids) if scope_ids is not None else metric['total']
        out_of_scope = ([identifier for identifier in source_ids if identifier not in scope_ids]
                        if source_ids is not None else None)
        reviewed = {unit['audit']['id'] for unit in metric['units'] or []
                    if unit['effective_verdict'] in VERDICTS}
        known_source_ids = source_ids if source_ids is not None else (
            [unit['audit']['id'] for unit in metric['units']] if metric['units'] is not None else None)
        unreviewed = ([identifier for identifier in known_source_ids if identifier not in reviewed]
                      if known_source_ids is not None else None)
        metric.update(source_total=source_total, scope_total=scope_total,
                      out_of_scope_count=(len(out_of_scope) if out_of_scope is not None else
                                          0 if source_total is not None else None),
                      unreviewed_count=len(unreviewed) if unreviewed is not None else None,
                      unreviewed_ids=unreviewed,
                      source_review_coverage=ratio(len(reviewed), source_total))
    return descriptor


def summarize_audit(payload, *, provenance=None, expected_units=None,
                    status="completed", failures=None, audit_scope=None):
    """Summarize one explicit audit without pooling tasks, origins, or versions.

    ``provenance`` is preserved verbatim, never completed from guesses. Callers
    should supply source task, candidate/test version, evaluator, and evidence
    paths there. ``expected_units`` is optional; when provided, IDs must match
    exactly. Failed audits without payload retain unknown metrics as null.
    """
    expected_units = _expected(expected_units)
    if not isinstance(status, str) or not status:
        raise ValueError("audit status must be an explicit nonempty string")
    groups = _groups(expected_units, payload)
    if payload is None:
        if status == "completed":
            raise ValueError("completed audit cannot have a missing payload")
        metrics = {group: _unknown_metric(group, expected_units[group] if expected_units is not None else None)
                   for group in groups}
    else:
        _units(payload, expected_units)
        metrics = {group: _support_metric(group, payload[group]) for group in groups if group != "probes"}
        metrics["probes"] = _probe_metric(payload["probes"])
    descriptor = _scope_metrics(metrics, expected_units, audit_scope)
    scope_mode = descriptor["mode"] if descriptor is not None else "full"
    return {
        "schema_version": 1, "evaluation_type": "automatic_estimate",
        "scope": ("Automatic audit estimates; not human gold or the AutoDataBench paper score. "
                  + ("Only selected-design units were assigned review; other source units are unreviewed, not uncertain."
                     if scope_mode == "selected_design" else "Full review scope.")),
        "scope_mode": scope_mode, "audit_scope": descriptor,
        "status": status, "provenance": copy.deepcopy(provenance),
        "failures": copy.deepcopy(failures), "expected_units": copy.deepcopy(expected_units),
        "rules": copy.deepcopy(RULES), "metrics": metrics,
    }


def summarize_envelope(envelope, *, audit_path=None):
    """Consume the stable ``diagnostics/audit.json`` envelope, without inference."""
    if not isinstance(envelope, dict):
        raise ValueError("audit envelope must be an object")
    status = envelope.get("status")
    if status not in ("completed", "failed", "missing"):
        raise ValueError("audit envelope status must be completed, failed, or missing")
    if "payload" not in envelope:
        raise ValueError("audit envelope must explicitly declare payload, including null")
    if status != "completed" and envelope["payload"] is not None:
        raise ValueError("failed or missing audit cannot supply accepted metric payload")
    failures = envelope.get("failures")
    if not isinstance(failures, list):
        raise ValueError("audit envelope failures must be a list")
    result = summarize_audit(
        envelope["payload"], provenance=envelope.get("provenance"),
        expected_units=envelope.get("expected_units"), status=status, failures=failures,
        audit_scope=envelope.get("audit_scope"),
    )
    result["audit_context"] = {key: copy.deepcopy(envelope[key]) for key in
                               ("evidence_fingerprint", "work_dir") if key in envelope}
    result["audit_path"] = str(audit_path) if audit_path is not None else None
    return result


def _read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _failure(kind, path, exception=None):
    result = {"kind": kind, "path": str(path)}
    if exception is not None:
        result.update(error_type=type(exception).__name__, reason=str(exception))
    return result


def _evidence_paths(value):
    """Collect explicit path fields only; never infer paths from narrative text."""
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("path", "original_path", "work_dir", "audit_path") and isinstance(child, str):
                paths.append(child)
            else:
                paths.extend(_evidence_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_evidence_paths(child))
    return sorted(set(paths))


def _episode_paths(run_dir, metadata):
    """Include planned-but-absent episodes as well as every existing ep directory."""
    paths = {path for path in run_dir.glob("*/*/ep*")
             if path.is_dir() and re.fullmatch(r"ep\d+", path.name)}
    if metadata is None:
        return sorted(paths), []
    errors = []
    benchmark, tasks, episodes = (metadata.get(k) for k in ("benchmark", "tasks", "episodes"))
    if any(value is not None for value in (benchmark, tasks, episodes)):
        if not (isinstance(benchmark, str) and benchmark and isinstance(tasks, list)
                and all(isinstance(task, str) and task for task in tasks)
                and isinstance(episodes, list)
                and all(type(episode) is int and episode >= 0 for episode in episodes)):
            errors.append({"kind": "invalid_run_plan", "path": str(run_dir / "run.json"),
                           "reason": "Plan requires benchmark, task-name list, and integer episode list."})
        elif any(Path(name).name != name or name in (".", "..") for name in [benchmark, *tasks]):
            errors.append({"kind": "invalid_run_plan", "path": str(run_dir / "run.json"),
                           "reason": "Plan benchmark/task names must be single directory components."})
        else:
            paths.update(run_dir / benchmark / task / f"ep{episode:02d}"
                         for task in tasks for episode in episodes)
    return sorted(paths), errors


def episode_summary(episode_dir, *, run_dir=None):
    """One episode including a failed/missing audit and all recorded score values.

    No historical audit is substituted for a missing current audit. This keeps
    different candidate/test versions from being combined into one measurement.
    """
    episode_dir = Path(episode_dir).resolve()
    audit_path = episode_dir / "diagnostics" / "audit.json"
    score_path = episode_dir / "score.json"
    failures = []
    envelope = None
    score = None
    audit_sha256 = None
    if audit_path.is_file():
        try:
            audit_bytes = audit_path.read_bytes()
            audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
            envelope = json.loads(audit_bytes)
            if not isinstance(envelope, dict):
                raise ValueError("audit envelope must be an object")
            result = summarize_envelope(envelope, audit_path=audit_path)
        except (OSError, ValueError, TypeError) as exc:
            failures.append(_failure("invalid_audit", audit_path, exc))
            prior_failures = envelope.get("failures", []) if isinstance(envelope, dict) else []
            if isinstance(prior_failures, list):
                failures = copy.deepcopy(prior_failures) + failures
            expected = envelope.get("expected_units") if isinstance(envelope, dict) else None
            try:
                expected = _expected(expected)
            except ValueError:
                expected = None
            result = summarize_audit(None, status="failed", failures=failures,
                                     expected_units=expected,
                                     provenance=envelope.get("provenance") if isinstance(envelope, dict) else None)
    else:
        failures.append(_failure("audit_missing", audit_path))
        result = summarize_audit(None, status="missing", failures=failures)
    if score_path.is_file():
        try:
            score = _read_object(score_path)
        except (OSError, ValueError, TypeError) as exc:
            failures.append(_failure("invalid_score", score_path, exc))
    else:
        failures.append(_failure("score_missing", score_path))
    if not episode_dir.is_dir():
        failures.append(_failure("episode_missing", episode_dir))
    # Audit failure details belong to the audit; all additional discovery errors
    # are also made visible on the episode row, with no missing-data score fill.
    row_failures = copy.deepcopy(result["failures"] or [])
    for failure in failures:
        if failure not in row_failures:
            row_failures.append(failure)
    relative = episode_dir.relative_to(Path(run_dir).resolve()) if run_dir else None
    location = {"episode_dir": str(episode_dir), "relative_episode_dir": str(relative) if relative else None}
    if relative is not None and len(relative.parts) == 3:
        location.update(benchmark=relative.parts[0], source_task=relative.parts[1], episode=relative.parts[2])
    return {
        "location": location, "score": score, "diagnostics": result,
        "audit_envelope": envelope, "audit_sha256": audit_sha256,
        "evidence_paths": sorted(set([str(audit_path), str(score_path)] + _evidence_paths(envelope))),
        "failures": row_failures,
    }


def build_run_summary(run_dir):
    """Detailed per-episode report; no aggregate correctness or paper score."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    metadata = None
    failures = []
    metadata_path = run_dir / "run.json"
    if metadata_path.is_file():
        try:
            metadata = _read_object(metadata_path)
        except (OSError, ValueError, TypeError) as exc:
            failures.append(_failure("invalid_run_metadata", metadata_path, exc))
    else:
        failures.append(_failure("run_metadata_missing", metadata_path))
    paths, plan_errors = _episode_paths(run_dir, metadata)
    failures.extend(plan_errors)
    rows = [episode_summary(path, run_dir=run_dir) for path in paths]
    return {
        "schema_version": 1, "evaluation_type": "automatic_estimate",
        "scope": "Automatic audit estimates; not human gold or the AutoDataBench paper score. "
                 "Each episode remains separate, including difficulty-zero and missing-audit episodes.",
        "run_dir": str(run_dir), "run_metadata": metadata, "rules": copy.deepcopy(RULES),
        "episode_count": len(rows), "failures": failures, "episodes": rows,
    }


COUNT_FIELDS = ("accuracy", "numerator", "denominator", "total", "correct", "incorrect", "uncertain",
                "missing", "infra_error", "evaluated", "evaluation_coverage", "decidable_coverage",
                "source_total", "scope_total", "out_of_scope_count", "unreviewed_count", "source_review_coverage")


def _flat_row(row, groups=GROUPS):
    score = row["score"] or {}
    diag = row["diagnostics"]
    result = {key: row["location"].get(key) for key in
              ("benchmark", "source_task", "episode", "episode_dir")}
    result.update(status=diag["status"], evaluation_type=diag["evaluation_type"], scope_mode=diag["scope_mode"],
                  difficulty=score.get("difficulty"), episode_raw=score.get("episode_raw"),
                  provenance=json.dumps(diag["provenance"], ensure_ascii=False),
                  audit_context=json.dumps(diag.get("audit_context"), ensure_ascii=False),
                  evidence_paths=json.dumps(row["evidence_paths"], ensure_ascii=False),
                  audit_sha256=row["audit_sha256"], failures=json.dumps(row["failures"], ensure_ascii=False))
    for group in groups:
        for field in COUNT_FIELDS:
            result[f"{group}_{field}"] = diag["metrics"].get(group, {}).get(field)
    return result


def _md(value):
    if value is None:
        return "null"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\n", "<br>")


def _metric_cell(group):
    display = {key: "null" if value is None else value for key, value in group.items()}
    return (f"{display['numerator']}/{display['denominator']}; accuracy={display['accuracy']}; "
            f"uncertain={display['uncertain']}; missing={display['missing']}; infra={display['infra_error']}; "
            f"coverage={display['evaluation_coverage']}; decidable={display['decidable_coverage']}; "
            f"source={display['source_total']}; scope={display['scope_total']}; "
            f"out_of_scope={display['out_of_scope_count']}; unreviewed={display['unreviewed_count']}; "
            f"source_review_coverage={display['source_review_coverage']}")


def write_run_summary(run_dir):
    """Write diagnostics-summary.{json,csv,md}; return the detailed report."""
    report = build_run_summary(run_dir)
    run_dir = Path(report["run_dir"])
    json_path = run_dir / "diagnostics-summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    groups = PARALLEL_GROUPS if any("observations" in row["diagnostics"]["metrics"]
                                    for row in report["episodes"]) else GROUPS
    rows = [_flat_row(row, groups) for row in report["episodes"]]
    columns = (["benchmark", "source_task", "episode", "episode_dir", "status", "evaluation_type", "scope_mode",
                "difficulty", "episode_raw", "provenance", "audit_context", "evidence_paths",
                "audit_sha256", "failures"] + [f"{group}_{field}" for group in groups for field in COUNT_FIELDS])
    with (run_dir / "diagnostics-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        # Explicit literal null avoids making an unknown metric look like zero.
        writer.writerows({key: "null" if value is None else value for key, value in row.items()} for row in rows)
    labels = {"claims": "Claims", "mappings": "Mappings", "task_checks": "Task checks",
              "probes": "Probes", "observations": "Observations"}
    headings = ["Episode", "Audit status", "Review scope", "Paper difficulty (context only)"] + [labels[g] for g in groups] + ["Provenance/version", "Evidence paths", "Failures"]
    lines = ["# Automatic diagnostics estimates", "", report["scope"], "",
             "Accuracy uses correct / (correct + incorrect). Uncertain judgments remain outside this denominator. "
             "Coverage includes uncertain judgments but excludes missing or infrastructure-failed executions. "
             "Decidable coverage counts only correct and incorrect judgments. Empty denominators are null. "
             "Units outside review scope are unreviewed, not uncertain or incorrect; source coverage uses the full source count.", "",
             f"Episodes: {report['episode_count']}", "",
             "Run/discovery failures: " + _md(json.dumps(report["failures"], ensure_ascii=False)), "",
             "| " + " | ".join(headings) + " |",
             "| " + " | ".join("---" for _ in headings) + " |"]
    for row in report["episodes"]:
        diag = row["diagnostics"]
        values = [row["location"]["relative_episode_dir"], diag["status"], diag["scope_mode"],
                  (row["score"] or {}).get("difficulty")]
        values.extend(_metric_cell(diag["metrics"][group]) if group in diag["metrics"] else "not applicable" for group in groups)
        values.extend(json.dumps(value, ensure_ascii=False) for value in
                      (diag["provenance"], row["evidence_paths"], row["failures"]))
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")
    (run_dir / "diagnostics-summary.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = write_run_summary(args.run_dir)
    print(json.dumps({"run_dir": report["run_dir"], "episodes": report["episode_count"],
                      "report": str(Path(report["run_dir"]) / "diagnostics-summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
