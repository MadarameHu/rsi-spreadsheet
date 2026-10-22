# Evidence-based observations from one complete attempt

Read __INPUT_MANIFEST__ — the inventory of the files supplied to you, with their
sources and hashes — and the `evidence_index.json` beside it, which holds
mechanical event and line locations in the trajectory and no judgments. Then read
the original task files at __ORIGINAL__ and your assigned complete trajectory at
__ATTEMPT__. Only this attempt and the allowlisted original inputs are supplied.
Do not seek other attempts, cross-sample scores, reference answers, scorers or
historical audits. You have __BUDGET_MIN__ minutes of agent execution allocation.
No Docker daemon or task execution runtime is mounted. Work directory: __WORK__.

Score trace: __SCORE_TRACE__. When supplied, it is this attempt's recorded
successful edits replayed on the original input, recalculated by the scoring
engine and compared with the official receipt. For every target cell it reports
the formula, the recalculated value, the expected value and whether the two
match. It is measurement, not judgment: use its values, do not recompute them.

Fields that need a definition:
- `reconciliation.status`: `reproduced` when the replayed reward and the receipt
  agree; `not_reproduced` otherwise.
- `last_edit_feedback`: the tool response to the last successful edit.
  `sample_values` are the values it returned; `flags` are set by the host —
  `error_value` (an Excel error such as `#VALUE!`), `all_zero` (every value is 0),
  `all_identical` (two or more values, all equal); `tool_calls_after` counts the
  calls made after it. `null` means no edit ever succeeded.
- `process`: counts over all tool calls. `failed_calls` returned an error;
  `identical_repeat_calls` repeated an earlier call with the same tool and
  arguments; `longest_identical_error_streak` is the longest run of the same
  failing call in a row.

## Score trace

This applies when a trace is supplied for an attempt whose official reward is
below 1. With no trace, or when `reconciliation.status` is `not_reproduced`, the
attempt is unattributed: record that as a limitation, let no defect observation
rest on it, and write no `SCORE TRACE:` observation.

Otherwise one observation must start its `finding` with `SCORE TRACE:` and report
only what the trace does not already state. The trace already holds the mismatched
cells, the recalculated and expected values, `final_edits`,
`target_cells_never_written`, `last_edit_feedback` and the call counts: cite those
by name where they matter, never copy their values. Add what they cannot say —
every separate reason the submitted content produces the measured value, one per
line as `REASON 1:`, `REASON 2:` and so on in order with no gaps, each naming the
formula fragment or call argument it comes from and whether fixing that alone
removes the mismatch or only works together with another reason. At least one
reason is required; an attributed unsolved attempt that reports no reason is
rejected. State each as a hypothesis unless the trace alone settles it; never the
model's inner causes.

For such an attributed attempt, when `last_edit_feedback.flags` contains
`error_value` or `all_zero` and `last_edit_feedback.tool_calls_after` is 0, write
that as its own observation starting with `FEEDBACK:`, cited to the events showing
the response and the absence of a later correction. `all_identical` alone does not
trigger it. Write no `FEEDBACK:` observation otherwise.

The validator enforces these prefixes mechanically.

## Your output

Write only these semantic fields to __OUTPUT__:

```json
{
  "observations": [
    {
      "finding": "One observable behavior and its relevant result",
      "evidence": ["event:3", "events:4-20"],
      "uncertainty": "What cannot be established about this observation"
    }
  ],
  "limitations": ["A limitation of the supplied evidence, when applicable"]
}
```

`finding` and a nonempty `evidence` list are required for each observation.
`uncertainty` is optional text or null; omission means not declared, not certain.
`limitations` is optional. Emit no IDs, sample/schema fields, paths, statistics,
terminal-state fields, summaries, or other fields. The script adds identity,
versions and evidence paths without changing your findings. An empty observation
list is valid when no supported behavioral proposition can be made.

Use exact references from the evidence index:
- `event:3` or `events:4-20`: one event or every event in an inclusive range.
- `trace:all`: the entire supplied trajectory, useful for absence claims.
- `trace:score`: the explicitly indexed official-score section, when present.
- `task:lines:1-3` or `system:lines:1-2`: original text line ranges.
- `input:Schedule_Weekly!A23:H26`: an existing sheet and in-bounds cell range.

## How to observe

Inspect the cited evidence. For “did not continue reading,” cite both the
truncation response and the relevant subsequent events (or entire trajectory),
not merely the truncation response. Scratch calculations may assist inspection;
cite their supporting original events/cells, not scratch files. References only
establish location; the reviewer separately checks evidential support.

Record correct operations and errors where evidenced, without category quotas
or an observation-count target. Combine an action, its result and its score
when they support the same proposition; do not split them into repetitive
observations. Separate distinct propositions. Do not turn metadata, task
requirements, scores or evidence availability into capability observations.
Scores and terminal states remain context, never independent proof of a defect.
Do not infer cross-sample prevalence or capability categories; the main Agent
compares attempts later. Preserve contradictory evidence.

Distinguish recorded tool/service failures from customer actions. A successful
tool response proves only that recorded call succeeded. Put shared evidence
limitations in `limitations`; put proposition-specific uncertainty in
`uncertainty`. The harness does not clip the supplied trajectory. A tool response
within it may itself be truncated; describe the resulting limitation if relevant.
Do not invent missing events, call models, run candidate tasks, or modify inputs.

Within this invocation you may validate your saved output:

```sh
python __VALIDATOR__ --validate __OUTPUT__ --manifest __INPUT_MANIFEST__
```

The validator checks format and reference existence, not semantic correctness.
You may correct your own output within this invocation. The host does not repair
judgments or retry the model. Finish after saving the output.
