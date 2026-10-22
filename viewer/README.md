# Viewer

A local, read-only console for a batch of pipeline episodes.

## Start

```bash
./viewer/serve.sh
./viewer/serve.sh stop
```

The script builds `viewer/data/index.json` from `runs/` and starts a
loopback-only server on a local port. It prints the URL.

## Data

By default it indexes `runs/batch-*`:

```
runs/batch-<id>-<task>-<kind><n>/
  spreadsheet/<task>/<episode>/
    diagnostics/
    work/
```

An optional `batch/status.json` supplies task-level stage and gate hints. If it
is absent, the viewer derives everything from the run directories.

## Pages

| Route | Contents |
|---|---|
| `#/b/<batch>` | funnel, pass-rate distribution, task table, all attempts |
| `#/t/<task>` | attempt timeline, error library, training goals |
| `#/t/<task>/a/<run>` | stage rail and module cards |
| `#/t/<task>/a/<run>/m/<module>` | records for one module |
| `#/tree/<root>/<path>` | raw file tree |
| `#/x/<root>/<path>` | workbook preview and cell diff |

Each field can be traced to its source file and JSON Pointer.

## Security

- Read-only; no write endpoint.
- Paths are confined to the configured data roots; absolute paths, `..`, and
  symlink escapes are rejected.
- Credential-looking files and session directories are not served.
- JSON, text, and workbook sizes are capped.
