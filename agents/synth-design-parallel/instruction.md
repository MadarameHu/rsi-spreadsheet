# Design from a committed capability and defect synthesis

Deliver exactly one new training task for __BENCHMARK__, aimed at the customer's
model __ALLOWED_MODEL__.

Inputs:
- __ATTRIBUTION_JSON__: the committed claims and ordered training goals.
- __ORIGINAL_DIR__: the original instruction, input, runtime, reference and verifier.
- __OBSERVATIONS_INDEX__: the atomic observations and their evidence links.
- __SCORE_TRACES__: host-computed replays of the original attempts —
  `score_trace_summary.json` (one row per attempt) and `sample-NN.json` (full
  trace). Use `final_formulas` (the formulas each attempt last wrote) and
  `first_mismatch` (the first wrong target cell with its formula, recalculated
  and expected value). Facts, not judgments.
- __ROLLOUT_DIR__: complete original customer attempts, available for evidence checks.
- __ERROR_LIBRARY__: error classes accumulated from earlier analysis batches of this
  original task (statement, conditions, evidence_strength, which designs targeted
  them). The customer model samples a different subset of its errors each batch,
  so a class absent from this batch may still be live. Reference only.
- __BENCHMARK_DIR__: native task templates, available if the original lacks a detail.

Read the aggregation and original task/environment first. Consult the observation
index and original evidence as needed to verify the selected goal or resolve a
contradiction. You are not required to re-analyze all trajectories or read unrelated
benchmark tasks. The aggregation is a proposal, not a trusted answer key. Do not
overwrite its claims or evidence.

## The candidate task

Write the task in its own directory directly under __OUT__. Use __WORK__ for
scratch files. Keep grading data out of the solver-visible inputs.

Your explicit design allocation is __BUDGET_MIN__ minutes. Worker, aggregation,
and design allocations are configured separately. Use this authoring allocation
to inspect evidence and create the candidate. The host runs and times validation
as a separate stage after your completion.

Choose a concrete training target supported by a defect or demonstrated capability
claim. State the conditions under which the ability or weakness was observed.
Use supported positive abilities to control unrelated difficulty. Missing errors
are not evidence of ability; a repeated symptom is not automatically its cause.
Resolve task/specification ambiguity and isolate infrastructure failures rather
than turning them into model defects. Do not try to exercise every observation
or every aggregate claim merely to inflate coverage.

__CANDIDATE_JOB_CONSTRAINT__

Read the aggregation's ordered `training_goals`. Select exactly one primary goal
for this candidate, using its host-assigned goal ID. Start from the proposed
priority but check its evidence and feasibility. When a host candidate job is
assigned, use its goal_id and explain feasibility and design choices in
`selection_reason`; do not substitute another goal. Without an assigned job,
explain any change of priority in `selection_reason`. Ranking is a proposal, not a causal
verdict. If no grounded goal is available, report the limitation instead of
inventing one. Legacy aggregations without training_goals may omit the two new
selection fields; that compatibility path does not apply when goals are present.

Keep one primary goal. Within the same task, also exercise the other error
classes that share its construct (for example, other wrong ways of writing the
same conditional count), taken from this batch's defect claims or from the error
class library: add a target for each such claim of this batch, and a wrong-answer
probe for each class, so one attempt can reveal several classes. Do not add
classes from unrelated constructs, and do not remove conditions under which these
classes occurred merely to make the task easier; a harder task is acceptable.
Reduce only demands unrelated to these classes, such as input layout
interpretation, output size or tool formatting. These controls must be reflected in
the actual generated task, not just claimed in the design text. Retain relevant
environmental constraints only with a feasible response/recovery path. Do not
confuse normal partial feedback with an infrastructure failure.

A goal whose `rationale` starts with `SECONDARY:` is a constraint, never the
primary goal and never part of the reward. You may reflect it in the instruction,
for example by stating a plausible value range for the results; never limit
steps or tool calls. For a secondary problem that can prevent any write at all,
such as an addressing parameter, choose one and say which in
`selection_reason`: avoid the condition entirely, or make it the declared
primary goal. A goal whose `rationale` starts with `FEEDBACK:` is not a primary
goal for this design.

## Write design.json

Write __DESIGN_JSON__ with this structure before finishing:

```json
{
  "primary_goal_id": "g1",
  "selection_reason": "Why this goal is the best next exercise, how other demands are controlled, and why any auxiliary targets are necessary",
  "targets": [
    {
      "claim_id": "c1",
      "feature": "The concrete feature exercising this grounded capability or defect",
      "test_intent": "Observable evidence of the target ability, plus independently valid and targeted invalid answers to verify the scoring distinction"
    }
  ],
  "preserves": ["The requirement or demonstrated ability deliberately retained"],
  "simplifies": ["The unrelated difficulty deliberately simplified"]
}
```

When the analysis declares training goals, `primary_goal_id` and
`selection_reason` are both required: never write one without the other, and never
omit both. Choose at least one target linked to a defect or capability. Do not repeat a
claim in targets. Each target needs its claim_id, concrete feature and test_intent.
At least one selected target must belong to the primary goal's defect/capability
claims. There is no coverage quota: retain only targets needed for this goal.
`preserves` and
`simplifies` are optional lists of strings describing deliberate choices for this
task as a whole; omit them if not declared. An explicitly empty list and an
omitted field are recorded differently. Do not write mapping IDs, candidate IDs,
roles, version metadata, hashes, file inventories, or self-reported test scores.
The host assigns IDs and records actual execution results. Claims you do not
select are mechanically recorded as unselected, not incorrect or excluded for
an invented reason. Do not account for every claim or provide exclusion reasons.
Update the design artifact when the task changes; never retroactively edit the
source attribution.

## Wrong-answer probes

Write __PROBES_JSON__: the submissions that should score 0, each
`{"name": "...", "source": "s02,s14" or "authored", "edits":
[{"tool": "fill_formula", "arguments": {...}}]}`, expressed against your new
workbook with the tool names and arguments the customer model would use.
`source` lists only original attempts that did NOT solve the task (from the
score traces, `solved` is false); a probe that cites a solved attempt is rejected.
Take each probe first from the wrong final formulas recorded in the score traces for
the targeted mechanisms, adapted to the new layout; add an authored probe only for
a mechanism the traces do not contain, such as a library class, naming the library
class id in `name`. Give one probe per distinct
reason. The host recalculates every probe on your workbook and rejects the
candidate if a probe equals the reference in every target cell, so choose data
in which each probe changes at least one target cell. Do not state in
`test_intent` how many cells a probe changes; the host measures that.
In every `formula_template`, a string literal is a plain double quote inside the
formula: write it as a single JSON escape `\"` and nothing more. The decoded
formula must read `=COUNTIF(A:A,"Yes")`, never `=COUNTIF(A:A,\"Yes\")`. Keep each
comparison operator in front of its operand; do not drop `=` or `<>` before a
quoted literal. A probe the host cannot replay is rejected and the whole candidate
is lost, so re-read every template once before writing the file.

Design the checks so that the host can distinguish whether the primary ability
was demonstrated, whether the whole task succeeded, and whether unrelated failures
obscured the result. Accept valid alternative strategies based on their outcomes,
not on a prescribed tool sequence. A zero whole-task score alone does not establish
absence of the primary ability. Scores and observed difficulty must come from the
subsequent independent execution, never from the planning text.

Produce the task instruction, input data, explicit output contract, reference
solution and scoring logic using the existing task format. The host supplies
runtime templates, captures versions and runs validation; you remain responsible
for the actual task content and solver/scorer logic.

Reuse the original's runnable native format and entrypoints: `task.toml`,
`instruction.md`, `environment/`, `solution/` with its reference entrypoint, and
`tests/` with its verifier entrypoint. Do not redesign runtime plumbing. Author
the task content yourself, in a meaningfully different work scenario; changing
only names or values is insufficient. State output paths, structure, units,
ordering and missing-value rules explicitly. Check every requirement the verifier
enforces against the instruction, and cover the instruction's required outputs.
In spreadsheet tasks, comparing only recalculated values cannot enforce a
requirement to submit formulas. If the instruction requires formulas, inspect
the submitted workbook with formulas preserved and reject literal values in the
required formula cells in addition to checking their calculated results. Do not
assume a reused value-comparison verifier already checks this. Accept equivalent
valid formulas unless the instruction explicitly requires a particular form.
After authoring, the host builds three workbooks from your finished reference
output and runs them through your unchanged verifier; the candidate is rejected
if the verifier accepts any of them: every reference target formula replaced by
its correct value; every reference target formula replaced by a constant formula
such as `=42` or `="x"`; every cell your reference writes outside the answer
position reverted to the input. So a formula check must require each target
formula to reference cells, and every output the instruction requires must lie
inside the scored answer position.
The instruction must not hand the solver the fix for a targeted class. Define
inputs, outputs, formats and meanings, but do not state how to avoid a targeted
error (for example "do not multiply by 100" against a scale class, or "the copy
in K9 must test K3 against C9:E9" against an anchoring class) and do not give
check values that reveal it (for example "10 June is a Wednesday" or "there are
5 Saturdays" against a date-code class). If a definition that blocks a class is
unavoidable, drop that class from the targets and its probes. Before the
reference run, a separate checker reads only your instruction and the targeted
classes; a candidate whose instruction states the fix for any targeted class is
rejected without trainee attempts.
The verifier must run unattended with dependencies already supplied, without
network installs. Use reasonable resources for the actual work. Where the native
manifest has source provenance, record authored origin instead of claiming an
upstream task identity. Reference execution must compute from the task inputs,
not copy a grading answer. Preserve the benchmark's scoring and solve conventions;
do not change the solve threshold to make a difficult candidate appear successful.

__DIFFICULTY_POLICY__

Your responsibility ends when one complete candidate, its matching design.json
and its probes are saved. You may inspect inputs, generate artifacts and perform static checks
such as parsing manifests or checking source syntax. Do not execute the reference
solution, verifier or customer model during authoring. Candidate execution belongs
to the independent host validation stage; adb-oracle, adb-testrun and direct Harbor
entrypoints are disabled here. Do not bypass this boundary through direct binaries,
Docker, APIs or another model. Do not poll for validation results.

Finish your response after saving the deliverables, with their paths and any
unresolved limitations. The host freezes the candidate, executes validation and
records actual results after you finish. An accepted reference alone will not
independently establish the correctness of the answer or scorer. Do not self-report
an accuracy score or claim that unobserved target successes occurred. No human
step is needed for this handoff.
