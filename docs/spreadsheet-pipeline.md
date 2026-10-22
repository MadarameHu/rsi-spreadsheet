# Spreadsheet pipeline

## Stages

1. **Observe.** The target model attempts each source task through the fixed
   spreadsheet tool surface. Each attempt records tool calls, workbook
   revisions, verifier output, and a final workbook.
2. **Attribute.** The record is reduced to a hidden rubric. Each entry names a
   failure mode, the samples that support it, and the decision a new task must
   put the target model in front of.
3. **Author.** A synthesis agent writes one new task: instruction, starting
   workbook, verifier, reference solution, and probes that exercise the
   intended failure mode.
4. **Gate.** Four checks run before any difficulty measurement:
   - leak check: the instruction must not give away the answer or an anchor;
   - reference: the reference solution must pass the task's own verifier;
   - template probes: deterministic probes check that the verifier accepts the
     intended form and rejects shortcut forms;
   - optional model review: enabled explicitly, off by default.
5. **Difficulty.** The target model attempts the new task K times under its own
   verifier. The solve rate becomes the difficulty term; a task outside the
   configured band is routed away from RL.
6. **Route.** In-band deliveries go to RL; out-of-band deliveries go to SFT.
   Defect attribution compares failed attempts with the author's probes and
   reports matches, ambiguity, and off-target failures.
7. **Aggregate.** Episodes are averaged per source task first, then across
   tasks, so a source task with more repeats does not weigh more.

## Artifacts

An episode directory contains:

```
<episode>/
  diagnostics/
    observations/index.json
    attribution.json
    design.json
    leak-check.json
    template-gate.json
    validation.json
    route.json
    defect-hits.json
    candidate_manifest.json
  work/
    design.json
    probes.json
    candidate/
```

The viewer in `viewer/` indexes these files and exposes them without loading
workbooks into the browser.

## Configuration

`configs/default.json` is the tracked default. Copy it to a local, uncommitted
file and fill in model access there. The tracked file contains no credentials or
machine-specific values.

Relevant knobs:

| Key | Meaning |
|---|---|
| `training_pipeline.review` | `template` (default) or `llm` |
| `training_pipeline.rollout.attempts` | attempts per difficulty measurement |
| `score.band_lo`, `score.band_hi` | difficulty band |
| `score.quality_coverage_target` | rubric coverage target |

## Design notes

- The hidden rubric is never shown to the authoring agent.
- Difficulty is measured by execution, not by reading the task.
- A failed reference or a leaking instruction is rejected before spending
  attempts.
- Template probes are deterministic; they do not call a model.
- Defect attribution is diagnostic only and never changes admission.
