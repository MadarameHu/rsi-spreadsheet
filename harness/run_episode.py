#!/usr/bin/env python3
"""Run whole episodes end to end: synthesize, then score.

An episode is one original task turned into one new task and scored. This driver
exists because episodes are the unit of repetition: the point of v2's one-task-at-
a-time design is that the same (benchmark, original task) pair can be run many
times, and the spread across repeats is itself a number we need. v1's lesson was
that the researcher's run-to-run variance exceeded the differences between the
models being compared, so a single episode per task cannot separate them.

  # one episode per sampled task, 3 repeats
  python3 harness/run_episode.py --benchmark tb-science --episodes 3

  # just one task, one episode, in an existing run
  python3 harness/run_episode.py --benchmark tb-science --task dna-storage-codec \
      --episodes 1 --run-id 20260908-1200

Episodes run concurrently up to --episode-concurrency; within one episode the
phases are strictly ordered (synthesize, then roll out, then judge, because each
needs the previous one's output). Two separate caps bound the container count:
--episode-concurrency times researcher.max_concurrent_docker bounds the nested
containers a researcher can spawn, and score.n_concurrent bounds the customer
attempts across all episodes of this driver. Running episodes one at a time is
correct but far too slow: on tb-science one episode is ~45 min of authoring plus
up to 2 h of rollouts plus ~40 min of judging, so eight of them serially is a day
and a half.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, Harbor, DEFAULT_AUDIT_CONCURRENCY, validate_audit_concurrency,
                    add_budget_args, add_path_args, apply_budget_args,
                    apply_path_args, load_config, load_meta, log, resolve_harbor_bin,
                    sampled_tasks, rel, make_harbor)
from run_analyst import rubric_path
from run_rollout import rollout_bundle_path
from run_score import (guard_episode_mode, guard_run_mode, run_diagnostic_audit,
                       score_episode, write_diagnostic_run_summary)
import diagnostics
from run_synth import episode_dir, synthesize


def validate_authored_candidate(hb, cfg, benchmark, ep, *, quality_scope='full',
                                audit_concurrency=DEFAULT_AUDIT_CONCURRENCY,
                                audit_slot=None, source_review=None):
    from host_validation import launch_validation
    import template_probes
    if template_probes.review_mode(cfg) == 'template':
        # Default: leak check, reference run, deterministic verifier probes, then measured
        # difficulty. No LLM review is started in this mode.
        import leak_check
        leak = leak_check.run_for_episode(cfg, ep)
        if leak['status'] != 'accepted':
            return {'status': 'leak_' + leak['status'], 'leak_check': leak}
        reference = launch_validation(hb, cfg, benchmark, ep, phase='reference')
        if reference['status'] not in ('reference_completed', 'completed'):
            return reference
        gate = template_probes.run_gate(hb, cfg, benchmark, ep)
        if not gate['eligible_for_training']:
            return {'status': 'template_' + gate['status'], 'template_gate': gate, 'reference': reference}
        result = launch_validation(hb, cfg, benchmark, ep, phase='trainee')
        if result.get('status') == 'completed':
            template_probes.write_route(ep, result, cfg)
            try:
                import defect_hits
                defect_hits.write_for_episode(ep)
            except Exception as exc:  # diagnostic only; never changes admission
                log('defect-hits', f'{ep}: {type(exc).__name__}: {exc}')
        return result
    from quality_gate import run_quality_gate, freeze_reference_evidence
    import stage_audit
    audit_concurrency = validate_audit_concurrency(audit_concurrency)
    reviewer = make_harbor(cfg, float(cfg['judge'].get('agent_setup_timeout_mult',
                                                     cfg['score']['setup_timeout_mult'])))

    def reference_stage():
        result = launch_validation(hb, cfg, benchmark, ep, phase='reference')
        freeze_reference_evidence(ep)
        return result
    # Reference execution is independent of the review units.
    # Every future settles before admission; no stage is retried here.
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        reference_future = pool.submit(reference_stage)
        audit_future = pool.submit(stage_audit.audit_episode, reviewer, cfg, benchmark,
                                   Path(ep).parent.name, ep, scope=quality_scope,
                                   max_concurrent=audit_concurrency,
                                   slot=audit_slot, source_review=source_review)
        results, errors = {}, []
        for future in cf.as_completed((reference_future, audit_future)):
            try:
                results[future] = future.result()
            except (Exception, SystemExit) as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        reference = results[reference_future]
    if reference['status'] not in ('reference_completed', 'completed'):
        return reference
    quality = run_quality_gate(reviewer, cfg, benchmark, Path(ep).parent.name, ep,
                               scope=quality_scope, max_concurrent=audit_concurrency,
                               evaluate_only=True)
    if not quality['eligible_for_export']:
        return {'status': 'quality_' + quality['status'], 'quality': quality,
                'reference': reference}
    return launch_validation(hb, cfg, benchmark, ep, phase='trainee')


def preflight(cfg: dict, benchmark: str, tasks: list[Path], *, require_rubric: bool = True) -> None:
    """Refuse to start when the prep stages have not been run.

    Checked up front for the whole set rather than per task: discovering halfway
    through a multi-hour run that half the rubrics are missing wastes the half
    that already ran.
    """
    missing_rollout = [t.name for t in tasks
                       if not (rollout_bundle_path(cfg, benchmark, t.name) / "summary.json").is_file()]
    missing_rubric = ([t.name for t in tasks if not rubric_path(cfg, benchmark, t.name).is_file()]
                      if require_rubric else [])
    if missing_rollout:
        raise SystemExit(
            f"{benchmark}: no customer rollout bundle for {missing_rollout}.\n"
            f"  run: python3 harness/run_rollout.py --benchmark {benchmark}")
    if missing_rubric:
        raise SystemExit(
            f"{benchmark}: no rubric for {missing_rubric}.\n"
            f"  run: python3 harness/run_analyst.py --benchmark {benchmark}")


def main() -> int:
    ap = argparse.ArgumentParser(description="run episodes end to end: synthesize then score")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--episodes", type=int, default=1, help="repeats per task")
    ap.add_argument("--first-episode", type=int, default=1,
                    help="episode index to start at (to add repeats to an existing run)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--validation-only", action="store_true",
                    help="validate the committed candidate: reference, quality admission, then configured trainee attempts")
    ap.add_argument("--analysis-only", action="store_true",
                    help="run parallel local analysis and aggregation without starting Harbor")
    ap.add_argument("--score-only", action="store_true",
                    help="score episodes that already exist under --run-id")
    ap.add_argument("--rollout-only", action="store_true",
                    help="like --score-only but stop after the customer attempts: measure "
                         "difficulty, finalise the out-of-band episodes at 0, and leave the "
                         "in-band ones for a later --score-only pass to judge. Use it to keep "
                         "every trainee container off the box before any judge starts")
    ap.add_argument("--episode-concurrency", type=int, default=1,
                    help="episodes to run at once (default 1). Each concurrent episode can "
                         "spawn up to researcher.max_concurrent_docker nested containers "
                         "while authoring, so raise this and that cap together, not either alone")
    add_budget_args(ap)
    add_path_args(ap)
    diagnostics.add_args(ap)
    args = ap.parse_args()
    apply_path_args(args)

    cfg = load_config(args.config)
    apply_budget_args(args, cfg)
    diagnostics.apply_args(args, cfg)
    diagnostic_mode = diagnostics.enabled(cfg)
    parallel_mode = diagnostic_mode
    if args.validation_only and (not parallel_mode or args.analysis_only or args.score_only or
                                 args.rollout_only or args.synth_only):
        ap.error('--validation-only requires parallel diagnostics without other stage-only flags')
    if args.analysis_only and (not diagnostic_mode
                              or diagnostics.parallel_config(cfg).get('runtime') != 'local-claude'
                              or args.score_only or args.rollout_only):
        ap.error('--analysis-only requires parallel local-claude diagnostics without scoring flags')
    bench = args.benchmark
    load_meta(bench)
    tasks = sampled_tasks(bench)
    if args.task:
        want = set(args.task)
        tasks = [t for t in tasks if t.name in want]
        missing = want - {t.name for t in tasks}
        if missing:
            raise SystemExit(f"no such sampled task(s) in {bench}: {sorted(missing)}")
    preflight(cfg, bench, tasks,
              require_rubric=not ((args.synth_only or args.analysis_only or args.validation_only) and diagnostic_mode))

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    eps = list(range(args.first_episode, args.first_episode + args.episodes))
    log("episode", f"run_id={run_id}  {bench}: {len(tasks)} task(s) x {len(eps)} episode(s) "
                   f"= {len(tasks) * len(eps)} episode(s)")

    # Which models produced this run. Without it a report is uninterpretable: the
    # difficulty term depends on the customer model and quality on its rubric.
    run_root = ROOT / "runs" / run_id
    existing = guard_run_mode(run_root, diagnostic_mode)
    plan = {
        "run_id": run_id,
        **({"diagnostics_enabled": True} if diagnostic_mode else {}),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": bench,
        "tasks": [t.name for t in tasks],
        "episodes": eps,
        "episode_concurrency": args.episode_concurrency,
        "customer": cfg["customer"],
        "researcher": cfg["researcher"],
        "judge": cfg["judge"],
        "score": cfg["score"],
        "prep_root": rel(rollout_bundle_path(cfg, bench, tasks[0].name).parents[1]) if tasks else "",
        "rubrics_root": rel(rubric_path(cfg, bench, tasks[0].name).parents[2]) if tasks else "",
    }
    if diagnostic_mode:
        plan["diagnostic_plan"] = {
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "source_paths": [str(t.resolve()) for t in tasks],
        }
    jobs = [(ep_i, t) for ep_i in eps for t in tasks]
    for ep_i, task in jobs:
        ep = episode_dir(run_id, bench, task.name, ep_i)
        guard_episode_mode(ep, diagnostic_mode)
        marker = ep / "diagnostics/started.json"
        if diagnostic_mode and marker.exists():
            identity = json.loads(marker.read_text())
            expected = {
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "benchmark": bench, "original_task": task.name,
                "source_path": str(task.resolve()), "episode": ep_i,
            }
            if any(identity.get(k) != v for k, v in expected.items()):
                raise ValueError("diagnostic episode configuration or source identity changed")
    if existing is not None and diagnostic_mode:
        # Legacy diagnostic manifests lack diagnostic_plan; their existing fields
        # and per-episode started identities still remain authoritative.
        keys = set(plan) - {"started_at"}
        if "diagnostic_plan" not in existing:
            keys.remove("diagnostic_plan")
        changed = sorted(k for k in keys if existing.get(k) != plan[k])
        if changed:
            raise ValueError(f"diagnostic run plan changed ({', '.join(changed)}); use a new run ID")
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")

    if args.analysis_only:
        hb_synth = SimpleNamespace(base_url=cfg['gateway']['base_url'], api_key=cfg['gateway']['api_key'])
        hb_score = None
    else:
        hb_synth = make_harbor(cfg, float(cfg["researcher"]["agent_setup_timeout_mult"]))
        hb_score = make_harbor(cfg, float(cfg["score"]["setup_timeout_mult"]))
    n_ep = max(1, args.episode_concurrency)
    log("episode", f"episode concurrency {n_ep}, customer-attempt pool "
                   f"{cfg['score']['n_concurrent']}, researcher nested-docker cap "
                   f"{cfg['researcher']['max_concurrent_docker']}")

    records: list[dict] = []
    failures: list[str] = []
    lock = threading.Lock()

    def one_episode(job) -> None:
        ep_i, t = job
        ep = episode_dir(run_id, bench, t.name, ep_i)
        ep.mkdir(parents=True, exist_ok=True)
        def runtime_status(status, reason=None, phase=None):
            payload = {'status': status, 'pid': os.getpid(), 'hostname': socket.gethostname(),
                       'updated_at': time.time(),
                       'runtime': 'local-claude' if args.analysis_only else 'episode-controller', 'reason': reason,
                       'phase': phase}
            path = ep/'runtime-status.json'
            temporary = path.with_suffix('.pending')
            temporary.write_text(json.dumps(payload, indent=2)+'\n')
            temporary.replace(path)
        runtime_status('running')
        if not (args.score_only or args.rollout_only or args.validation_only):
            try:
                if args.analysis_only:
                    synthesize(hb_synth, cfg, bench, t, ep, ep_i, analysis_only=True)
                    runtime_status('completed')
                    return
                synthesize(hb_synth, cfg, bench, t, ep, ep_i)
            except SystemExit as e:
                log("episode", f"{t.name} ep{ep_i:02d}: synthesis FAILED {e}")
                failures.append(str(e))
                runtime_status('failed', str(e))
                if diagnostic_mode and not args.analysis_only and not (parallel_mode and args.synth_only):
                    run_diagnostic_audit(hb_synth, cfg, bench, t.name, ep)
                return
            except Exception as e:                                   # noqa: BLE001
                log("episode", f"{t.name} ep{ep_i:02d}: synthesis CRASHED "
                               f"{type(e).__name__}: {e}")
                failures.append(str(e))
                runtime_status('failed', f'{type(e).__name__}: {e}')
                if diagnostic_mode and not args.analysis_only and not (parallel_mode and args.synth_only):
                    run_diagnostic_audit(hb_synth, cfg, bench, t.name, ep)
                return
        if parallel_mode and not (args.score_only or args.rollout_only):
            if os.environ.get('ADB_STOP_AFTER_DESIGN') == '1':
                # Batch scheduling: design in parallel, validate serially later via --validation-only.
                runtime_status('completed', phase='design')
                return
            runtime_status('running', phase='host-validation')
            try:
                validation = validate_authored_candidate(hb_score, cfg, bench, ep)
                if validation['status'] != 'completed':
                    failures.append(f'{t.name}: host validation {validation["status"]}')
                    runtime_status('failed', failures[-1], phase='host-validation')
                    return
            except (Exception, SystemExit) as exc:
                failures.append(f'{t.name}: host validation {type(exc).__name__}: {exc}')
                runtime_status('failed', failures[-1], phase='host-validation')
                return
        if args.validation_only:
            runtime_status('completed', 'Execution checks completed; full quality approval remains separate.',
                           phase='host-validation')
            return
        if args.synth_only:
            if diagnostic_mode and not parallel_mode:
                run_diagnostic_audit(hb_synth, cfg, bench, t.name, ep)
            runtime_status('completed', 'Candidate retained; author terminal and validation results are separate.',
                           phase='candidate-and-validation')
            return
        try:
            rec = score_episode(hb_score, cfg, bench, t.name, ep, pool,
                                judge=not args.rollout_only)
        except SystemExit as e:
            log("episode", f"{t.name} ep{ep_i:02d}: scoring FAILED {e}")
            failures.append(str(e))
            runtime_status('failed', str(e), phase='scoring')
            return
        except Exception as e:                                       # noqa: BLE001
            log("episode", f"{t.name} ep{ep_i:02d}: scoring CRASHED "
                           f"{type(e).__name__}: {e}")
            failures.append(str(e))
            runtime_status('failed', f'{type(e).__name__}: {e}', phase='scoring')
            return
        with lock:
            records.append(rec)
        runtime_status('completed', phase='scoring')

    # Two pools: one for episodes, one shared by every episode's customer attempts.
    # The attempt pool has to be shared, or each episode would get its own and the
    # live-container count would be multiplied by the episode concurrency.
    with cf.ThreadPoolExecutor(max_workers=int(cfg["score"]["n_concurrent"])) as pool, \
            cf.ThreadPoolExecutor(max_workers=n_ep) as ep_pool:
        list(ep_pool.map(one_episode, jobs))

    if records and not args.synth_only:
        idx = ROOT / "runs" / run_id / bench / "episodes.json"
        recs = []
        for r in records:
            sj = ROOT / r["episode_dir"] / "score.json"
            recs.append(json.loads(sj.read_text()) if sj.is_file() else r)
        idx.write_text(json.dumps(recs, indent=2, ensure_ascii=False) + "\n")
        log("episode", f"wrote {rel(idx)}")
    if diagnostic_mode and not args.analysis_only:
        write_diagnostic_run_summary(run_root)
    log("episode", f"done. run_id={run_id}  "
                   f"aggregate with: python3 harness/aggregate.py --run-id {run_id}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
