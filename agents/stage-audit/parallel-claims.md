# Reviewer B: verify one claim's inference

Goal: decide whether the assigned claim, and the training goals that rest on it,
follow from its supporting and counter observations.

## Inputs (read-only)
Start with __UNIT_INPUT__. In `source`:
- `claim`: the claim, with `observation_ids`, `counterevidence_ids` and `conditions`.
- `training_goals`: goals declared on this claim.
- `observation_reviews`: reviewer A's completed verdict files for those observations.
- `score_trace_path`: `score_trace_summary.json`, one host-replayed row per attempt
  (rewards, mismatched cells, feedback flags, process counts), or null.
- `rollouts_root`, `original_root`, `navigation_path`: complete raw evidence.

## Judging
- Facts: accept A's `correct` observations and the score-trace numbers. Re-read raw
  events only for observations A marked `uncertain` or `incorrect`, or for a premise
  no observation covers. Do not recompute formulas or build a simulator.
- Inference, the actual task:
  - Does the claim's scope match its evidence? "all attempts" requires checking all.
  - Does each counterevidence observation fit the claim's `conditions`, or break it?
  - Does a successful attempt contradict a claimed inability?
  - Frequency is not causation.
  - For each training goal: is it supported by the behavior the claim describes?
- `correct`: follows. `incorrect`: contradicted. `uncertain`: an unresolved alternative
  remains; name it. Do not rewrite the claim or goals.

## Output
Write __RESULT__ as `{"claims":[{"id","verdict","reason","evidence":[{"path","locator"}]}]}`
with the assigned ID once, a literal verdict, and existing absolute paths with locators.
Write it as soon as the verdict is formed; refine it only if a later check changes it.
Notes go under __WORK__. Do not call other models or run scoring. Then exit.
