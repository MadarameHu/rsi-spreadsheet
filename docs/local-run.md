# Running locally

## Prerequisites

- Python 3.11 or newer
- Docker
- Harbor
- The dependencies in `requirements.txt`

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

`configs/default.json` is a tracked default with empty values. Copy it to a
local file and fill in model access there:

```bash
cp configs/default.json configs/work.local.json
```

`configs/*.local.json` is gitignored. Do not commit local configuration.

## Benchmark data

This repository does not ship benchmark data. Put your source tasks under
`benchmarks/<name>/`, one directory per task:

```
benchmarks/<name>/
  _meta.json
  <task>/
    instruction.md
    task.toml
    environment/
      input.xlsx
    tests/
      target.xlsx
      verify.py
```

Every task must be solvable through the fixed spreadsheet tool surface. Keep
the starting workbook in the task environment and the target workbook in the
test directory; the verifier runs outside the target environment.

## Commands

```bash
python3 harness/run_rollout.py --benchmark <name>
python3 harness/run_analyst.py --benchmark <name>
python3 harness/run_episode.py --benchmark <name> --episodes 3
python3 harness/aggregate.py --run-id <run_id>
```

Episode output is written under `runs/<run_id>/<benchmark>/<task>/<ep>/`.

## Tests

```bash
python3 -m unittest discover -s tests
python3 tools/check_clean.py
```

The macOS sandbox integration tests skip on other platforms. The rest of the
suite is platform-independent.

## Troubleshooting

- **Reference gate fails** — run the reference solution against the task's own
  verifier before delivery; a task whose reference has never passed is not
  finished.
- **Template gate fails** — inspect the probe evidence in
  `diagnostics/template-gate.json`; the verifier may be accepting a shortcut.
- **Difficulty outside the band** — adjust the task, not the band. A task solved
  on every attempt or on none is not useful training data.
- **Defect attribution is off-target** — the author's probes did not cover the
  failures the target model actually produced; widen the probes.
