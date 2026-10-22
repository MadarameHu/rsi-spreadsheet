# Reviewer A: verify one sample's observations

Goal: for each assigned observation, decide whether the attempt's evidence supports it.

## Inputs (read-only)
Start with __UNIT_INPUT__. In `source`:
- `observations`: the observations assigned to you, with their cited evidence.
- `source_files`: the full `attempt.md` and its `evidence_index.json` (event and line
  locations, no judgments). If empty, use `fallback_full_trace`; if that is missing too,
  the trace is missing evidence. Never invent one.
- `score_trace_path`: the host's replay of this attempt's edits, or null. It holds
  per-cell formula, recalculated and expected values, replay reward,
  `reconciliation`, `last_edit_feedback.flags` and process counts. These are
  measurements. "no Harbor rerun is implied" in attempt.md means the model was not
  re-run; this replay still exists.
- `original_root`: the original task and input workbook.

## Judging
- Behavior (what the model called, wrote, saw): check it against the full raw event,
  not only the index.
- Numbers (mismatched cells, values, rewards, flags; `SCORE TRACE:` and `FEEDBACK:`
  observations): compare with `score_trace_path`. Do not recompute formulas.
- Reasons stated as hypotheses: check that the cited formula fragment or argument
  exists and can produce the stated value.
- `correct`: supported. `incorrect`: contradicted. `uncertain`: the needed evidence is
  absent; say which.
- Judge only the assigned observations of this sample. Do not judge claims, goals or
  the candidate.

## Output
Write __RESULT__ as `{"observations":[{"id","verdict","reason","evidence":[{"path","locator"}]}]}`.
Each assigned ID appears exactly once; `verdict` is `correct`, `incorrect` or `uncertain`;
`path` is an existing absolute file and `locator` names the event or lines.
Write it as soon as every assigned observation is judged. Notes go under __WORK__.
Do not call other models or run scoring. Then exit.
