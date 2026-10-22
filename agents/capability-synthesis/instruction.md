# Aggregate observations and propose focused training goals

Original task: __ORIGINAL__
Complete original customer trajectories and metadata: __ROLLOUTS__
Committed worker observation index: __INDEX__
Score traces: __SCORE_TRACES__ — `score_trace_summary.json` (one row per attempt
plus the solved-attempt median) and `sample-NN.json` (each attempt's full trace).
Writable work directory: __WORK__
Explicit agent execution allocation: __BUDGET_MIN__ minutes.

Host-computed counts — use them as given, do not recount: __TRACE_FACTS__

Error class library: __ERROR_LIBRARY__ — defect classes accumulated from earlier
batches of this original task. The customer model samples a different subset of
its errors in each batch. For each defect claim set `library_class` to the `id` of
the library class describing the same mechanism, or null when the mechanism is new
to the library (always null when there is no library). It is reference only: every
claim must still be supported by this batch's observations.

Score trace fields:
- `official_reward` / `replay_reward`: the receipt's score / the host's replay of
  the final formulas.
- `reconciliation`: `reproduced` when the two agree; otherwise the attempt is
  unattributed. `not_reproduced` lists those attempts.
- `mismatched_cells`: target cells whose recalculated value differs from the
  expected value; `first_mismatch` is the first of them with its formula.
- `target_cells_never_written`: target cells no successful edit ever wrote.
- `last_feedback_flags`: host flags on the response to the last successful
  edit — `error_value` (an Excel error such as `#VALUE!`), `all_zero` (every
  value is 0), `all_identical` (two or more values, all equal).
- `tool_calls_after_last_edit`: calls made after that response; `null` means no
  edit ever succeeded.
- `process`: per-attempt counts — tool, read, edit and failed calls, identical
  repeats (same tool and arguments as an earlier call), longest run of the same
  failing call, tokens and duration.
- `solved_process_median`: medians of tool calls, completion tokens and duration
  over solved attempts only; `solved_attempts` gives how many.

Read the index and every declared worker output. Every worker has already
finished and its atomic observations have been committed. Workers are listed in
original declared sample order. You may reread all complete original attempts
to check observations and contradictions. No Docker daemon or evaluation
services are available. Do not run tasks or call another model.

The score traces are host-computed measurement: each attempt's recorded edits
replayed, recalculated and compared with its official receipt. Use them, not
hand evaluation of formulas, to decide what a submission produced.

## Your output

Write JSON to __OUTPUT__:

```json
{
  "claims": [
    {
      "kind": "defect",
      "finding": "Testable conclusion supported by the observations",
      "supports": ["s00-o1"],
      "counterevidence": [],
      "evidence_strength": "replay_verified",
      "library_class": null,
      "conditions": ["The conditions under which the conclusion applies"],
      "uncertainty": "Only if some part of this judgment remains uncertain"
    }
  ],
  "training_goals": [
    {
      "claim_refs": [1],
      "ability": "The transferable ability to practice, under its relevant conditions",
      "rationale": "Why this goal should be tried before others, based on evidence and task dependencies; identify uncertain causal assumptions",
      "success_signal": "Observable behavior or output that would demonstrate the ability, and distinguish its absence from an unrelated failure",
      "difficulty_controls": ["An unrelated demand to reduce while retaining the target condition"]
    }
  ]
}
```

Allowed kinds are exactly `defect`, `capability`, `task_issue`,
`infrastructure_issue`, or `uncertain`. Every observation ID in either reference
list must resolve to a committed worker observation. `supports` is nonempty;
`counterevidence` is required and can be empty. An empty list does not establish
that no counterevidence exists. `conditions` is a list of strings. `uncertainty`
is optional text or null; omission is recorded as not declared, never certainty.
`evidence_strength` is required: `replay_verified` when at least half of the
supporting attempts have host-recalculated evidence for the mechanism (the score
trace values, or a worker reason the trace alone settles); `hypothesis_only`
otherwise. A reason a worker states as a hypothesis, a counterfactual fix that
was never recalculated, an edit that never reached the workbook, or a purely
behavioral reading of the trajectory is not recalculated evidence.
The host assigns claim IDs in output order and expands original evidence from
supports and counterevidence mechanically. Do not assign your own claim IDs or
repeat source paths, sample counts, hashes, timing, or schema metadata. Only when you find additional
original evidence not already represented by those observations, add an optional
`evidence` list of {"path": "/absolute/existing/file", "locator": "precise event or workbook coordinates"}.
Conclusions may combine multiple observations; explain
their applicable conditions and preserve counterevidence. If worker observations
are unsupported, describe that uncertainty without modifying their artifacts.
Do not turn a numerical majority into a verdict or interpret absence of failure
as a capability. Positive capability claims require observed successful actions.
Distinguish consistent strengths, conditional success, defects, task ambiguity,
and infrastructure errors. A zero reward alone does not identify a weakness.
Tool pagination, truncation or other environmental constraints are not inherently
faults: distinguish the environment's condition from the model's response, and
whether a workable recovery path was available. Do not silently relabel worker
evidence. Preserve uncertainty when fault or recoverability is unresolved.

## Group the claims by what determined the score

Group the unsolved attempts by the `REASON n:` lines of their score traces; an
attempt with several reasons belongs to each group. Every attributed unsolved
attempt carries at least one reason, so every one of them lands in a group. Write
one claim per mechanism, concrete enough that a wrong submission of that kind can
be written down, and list the attempts it covers. Treat an attempt marked
`not_reproduced` as an infrastructure uncertainty and keep it out of defect counts.

## Order the goals, and the first one

After the claims, propose a short ordered list of training goals. This is part of
your synthesis, not a new task or a request to another model. Order goals by
evidence reliability, whether the ability is a prerequisite for later steps,
whether it can be exercised and checked independently, and whether unrelated
difficulty can be reduced enough for the target model to have a chance to succeed.
Frequency alone is not priority. Co-occurring failures do not establish causality;
say when a prerequisite or downstream-effect interpretation remains uncertain.
Do not force a total ranking over all claims or assign subjective numeric scores.

The first goal (g1) must rest on a mechanism that decided the score, be specific
enough to derive wrong-answer probes from ("write correct conditional formulas"
is too broad), be consistent with the solved attempts, and state in `rationale`
which unsolved attempts it accounts for and which it does not. g1 must reference
at least one `replay_verified` claim. Do not merge
mechanisms into one goal to raise that count. A problem that occurred but
determined no attempt's score cannot be the first goal: put it later, with a
`rationale` starting exactly
`SECONDARY: appeared in N of M attempts and determined the score in K; U unattributed.`
M and U are the host-computed counts above; K counts only attributed attempts.
The host checks 0 ≤ K ≤ N ≤ M and that M and U match the score traces.

The `FEEDBACK:` condition is `last_feedback_flags` containing `error_value` or
`all_zero` with `tool_calls_after_last_edit` equal to 0. When the host-computed
counts say a `FEEDBACK:` goal is required, that pattern is its own goal, never g1,
with a `rationale` starting `FEEDBACK:`. Otherwise no goal starts with `FEEDBACK:`.

Efficiency is judged here, not per attempt: combine the traces with the process
counts. Correctness comes first; reading headers, inspecting data and checking a
written result are useful exploration, never waste, and no goal may suggest
removing them. Propose at most one efficiency goal, always `SECONDARY:`, and only
for a step the trace and counts settle mechanically: a byte-identical repeated
call, a rewrite of an already successful target with the same content, or a
repeat of an unchanged failing call. Quote the counts; otherwise propose no
efficiency goal. Do not propose an efficiency goal from elevated totals alone.

Each goal must describe an ability to practice, not merely restate a failure or
collect several unrelated weaknesses. Use `claim_refs` as unique 1-based positions
in your own claims list (integers, not IDs). Include at least one defect or observed
capability; environmental/task/uncertain claims may supply context but cannot be
the sole training target. The host resolves references and assigns g1, g2, ... in
your declared priority order. Do not generate goal IDs yourself. All five fields
shown above are required for each goal. `difficulty_controls` is a list of strings.
If no grounded trainable goal can be proposed, explicitly return an empty list;
do not fabricate a goal to satisfy the format.

Prefer an observable outcome to a mandatory tool sequence: a valid alternative
strategy can demonstrate the ability. Separate evidence of target ability from
whole-task success; for example, complete input acquisition can succeed even if
later writing fails. Use observed positive abilities to reduce unrelated demands,
but never assume stable mastery from a single successful action. Leave concrete
task construction and actual difficulty measurement to the design stage.

No fixed number of claims is required. Preserve all relevant grounded findings
rather than forcing every event into a weakness. Propose training goals, but do
not construct a new task at this stage. Do not invent evidence or write a private reasoning transcript.
The host saves your exact output and will not repair JSON or infer missing
references. Save the output and finish when your evidence-grounded assessment is
complete.
