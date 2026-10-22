# Viewer

A local, read-only console for a batch of episodes. It reads run directories and
serves JSON, text, workbook previews, and cell diffs. There is no write endpoint.

## Start

```bash
./viewer/serve.sh
```

The script builds `viewer/data/index.json` and starts a loopback-only server.
Use `./viewer/serve.sh stop` to stop it.

## Data

The indexer scans `runs/batch-*` by default:

```
runs/batch-<date>-<task>-<kind><n>/
  spreadsheet/<task>/<episode>/
    diagnostics/
    work/
```

An optional `batch/status.json` supplies the task-level stage, current run, and
gate hints. If it is absent, the viewer derives everything from the run
directories.

## Pages

| Route | Contents |
|---|---|
| `#/b/<batch>` | funnel, pass-rate distribution, task table, all attempts |
| `#/t/<task>` | attempt timeline, error library, training goals |
| `#/t/<task>/a/<run>` | stage rail and module cards |
| `#/t/<task>/a/<run>/m/<module>` | records for one module |
| `#/tree/<root>/<path>` | raw file tree |
| `#/x/<root>/<path>` | workbook preview and cell diff |

Every field can be traced to its source file and JSON Pointer.

## Security

- Read-only.
- Paths are resolved under the configured data roots; absolute paths, `..`, and
  symlink escapes are rejected.
- Credential-looking files and session directories are not served.
- JSON, text, and workbook sizes are capped.
