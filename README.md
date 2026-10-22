# rsi-spreadsheet

A spreadsheet task-data pipeline. It observes a target model attempting existing
tasks, reduces that record to a hidden rubric, authors new tasks, gates them,
measures difficulty by execution, and routes each delivery to SFT or RL.

This repository ships **code only**: no benchmark task data, no model
transcripts, no prebuilt runs, and no hosted service.

## Pipeline

| Stage | What happens | Main entry point |
|---|---|---|
| Observe | The target model attempts each source task; its tool calls, workbooks and verifier outcomes are recorded. | `harness/run_rollout.py` |
| Attribute | The record is reduced to a hidden rubric: failure modes, supporting samples, and what each mode requires. | `harness/run_analyst.py` |
| Author | A synthesis agent writes one new task from the rubric, plus probes and a reference solution. | `harness/run_episode.py` |
| Gate | Leak check, reference execution, deterministic template probes, and an optional model review. | `harness/leak_check.py`, `harness/template_probes.py`, `harness/quality_gate.py` |
| Difficulty | The target model attempts the new task K times under its own verifier; the solve rate decides the band. | `harness/run_episode.py` |
| Route | In-band tasks go to RL, out-of-band tasks to SFT; defect attribution explains failures. | `harness/defect_hits.py`, `harness/error_library.py` |
| Aggregate | Episodes are averaged per source task, then across tasks. | `harness/aggregate.py` |

Two invariants drive the design:

- **Hidden rubric.** The authoring agent never sees the rubric derived from the
  target model's record.
- **Measurement by execution.** Difficulty is the target model's measured solve
  rate, not a judgement about how hard the task reads.

## Layout

```
harness/      pipeline modules
agents/       authoring, synthesis, audit and analysis prompts
rubrics/      gate and format definitions
configs/      default configuration and token pricing
docs/         pipeline, local run, and viewer notes
tests/        unit and integration tests
viewer/       local read-only audit console
```

## Quick start

Requires Python 3.11+, Docker, and Harbor.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

cp configs/default.json configs/work.local.json
# Fill in model access in configs/work.local.json.
# It is gitignored; never commit it.
```

Bring your own benchmark data under `benchmarks/<name>/` following the contract
in `docs/local-run.md`. This repository does not ship any task data.

```bash
python3 harness/run_rollout.py --benchmark <name>
python3 harness/run_analyst.py --benchmark <name>
python3 harness/run_episode.py --benchmark <name> --episodes 3
python3 harness/aggregate.py --run-id <run_id>
```

Output lands under `runs/<run_id>/<benchmark>/<task>/<ep>/`, one directory per
episode.

## Viewer

A local, read-only console for inspecting a batch of episodes: task list, stage
status, rubric evidence, and workbook diffs.

```bash
./viewer/serve.sh
# open the loopback URL it prints
```

See `docs/viewer.md`.

## Tests

```bash
python3 -m unittest discover -s tests
python3 tools/check_clean.py
```

`check_clean.py` rejects internal host names, absolute home paths, credential
values, and other material that must not appear in this repository.

## What this repository is not

- It is not a benchmark dataset.
- It does not include a remote job runner or a deployment target.
- It does not ship credentials, endpoints, or machine-specific paths.

## Attribution

The pipeline is derived from [AutoDataBench](https://github.com/StarDewXXX/AutoDataBench)
(Apache-2.0). The spreadsheet task format and scoring semantics follow
Spreadsheet-RL (CC BY-SA 4.0). See `NOTICE` and `LICENSE`.
