# Reviewers C+D: verify one candidate task

Goal: decide whether the candidate is a correct task (C) that implements its declared
mappings (D), and supply probe workbooks the host will score.

## Inputs (read-only)
Start with __UNIT_INPUT__. In `source`:
- `candidate_root`, `candidate_files`: instruction, input workbook, reference solution,
  ground truth and verifier.
- `mappings`: the assigned mappings to judge.
- `claim_statements_as_unverified_design_context`: background only; others judge them.
- `original_root`: the original task, to interpret the design.

This container has no spreadsheet engine. After you finish, the host recalculates and
scores every probe with the unchanged verifier. Do not build a recalculation engine or
scoring simulator, and do not run the verifier.

## C: task checks
Derive the expected answer yourself from the instruction and input workbook. Then:
- `spec`: coherent, unambiguous, solvable as written.
- `ground_truth`: equals your derived answer on every target cell.
- `reference`: derives the answer by the task rules, without reading the ground truth.
Agreement between reference and ground truth alone proves nothing.

## D: mappings
For each mapping judge only `implementation_verdict`: does the candidate actually
exercise the declared feature with its target/support/preserve/exclude roles and
stated simplifications? Never fill `upstream_verdict`.

## Probes
Required: `correct_solution` (expected_valid true) and `incorrect_solution`
(expected_valid false). Add another only when a specific check cannot be settled by
these two; name that check in its reason.
- Build each as a copy of the input workbook with openpyxl, writing formulas where the
  task requires live formulas. Save the script and the workbook under __WORK__.
- Set `expected_valid` from your derivation. If a workbook cannot be built, use null
  workbook and null expected_valid and say why.

## Output
Write __RESULT__:
`{"task_checks":[{"id":"spec|ground_truth|reference","verdict","reason","evidence":[{"path","locator"}]}],
"mappings":[{"id","implementation_verdict","reason","evidence":[...]}],
"probes":[{"id","expected_valid","actual_accept":null,"execution_status":"missing","reason","evidence":[...]}]}`.
Write __PROBE_PLAN__:
`{"probes":[{"id","workbook":"absolute path or null","target_path":"declared .xlsx path from task.toml","expected_valid","reason","evidence":[...]}]}`.
Every required ID appears exactly once; verdicts are `correct`, `incorrect` or
`uncertain`; probe IDs and expected_valid match across both files.

Write both files as soon as every check, mapping and the two required probes are done;
add optional probes only after that. Do not edit candidate files, call other models or
score anything. Then exit.
