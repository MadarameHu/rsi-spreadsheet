"""Deterministic metrics never turn missing observations into incorrect labels."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from harness import stage_metrics as metrics


def empty_payload():
    return {group: [] for group in metrics.GROUPS}


def judgment(identifier, verdict):
    return {"id": identifier, "verdict": verdict, "reason": "Evidence considered.", "evidence": []}


def probe(identifier, expected, actual, status="completed"):
    return {"id": identifier, "expected_valid": expected, "actual_accept": actual,
            "execution_status": status, "reason": "Execution record.", "evidence": []}


class MetricsTests(unittest.TestCase):
    def scoped_fixture(self):
        payload = empty_payload()
        payload['claims'] = [judgment('c1', 'correct')]
        payload['observations'] = [judgment('o1', 'correct'), judgment('o2', 'uncertain')]
        payload['probes'] = [probe('correct_solution', True, True), probe('incorrect_solution', False, False)]
        expected = {group: [item['id'] for item in items] for group, items in payload.items()}
        source = copy.deepcopy(expected)
        source['claims'].append('c2')
        source['observations'].append('o3')
        scope = {'schema_version': 'audit-scope-v1', 'mode': 'selected_design',
                 'source_units': source, 'expected_units': copy.deepcopy(expected),
                 'unreviewed_ids': {group: [i for i in ids if i not in expected[group]] for group, ids in source.items()}}
        return payload, expected, scope

    def test_scoped_metrics_keep_unreviewed_distinct_from_uncertain(self):
        payload, expected, scope = self.scoped_fixture()
        result = metrics.summarize_audit(payload, expected_units=expected, audit_scope=scope)
        claim = result['metrics']['claims']
        self.assertEqual((claim['source_total'], claim['scope_total'], claim['unreviewed_count']), (2, 1, 1))
        self.assertEqual((claim['accuracy'], claim['uncertain'], claim['source_review_coverage']), (1, 0, .5))
        self.assertEqual(claim['unreviewed_ids'], ['c2'])
        observed = result['metrics']['observations']
        self.assertEqual((observed['source_total'], observed['scope_total'], observed['unreviewed_count']), (3, 2, 1))
        self.assertEqual(observed['uncertain'], 1)
        self.assertAlmostEqual(observed['source_review_coverage'], 2 / 3)
        self.assertEqual(result['scope_mode'], 'selected_design')
        self.assertIn('unreviewed, not uncertain', result['scope'])

    def test_all_extra_probes_remain_in_source_and_scope_totals(self):
        payload, expected, scope = self.scoped_fixture()
        payload['probes'].append(probe('extra', False, True))
        expected['probes'].append('extra')
        result = metrics.summarize_envelope({'status': 'completed', 'payload': payload,
            'expected_units': expected, 'audit_scope': scope, 'failures': []})
        probes = result['metrics']['probes']
        self.assertEqual((probes['source_total'], probes['scope_total'], probes['unreviewed_count']), (3, 3, 0))
        self.assertEqual(probes['incorrect'], 1)
        self.assertEqual(probes['source_review_coverage'], 1)
        direct = metrics.summarize_audit(payload, audit_scope=scope)['metrics']['probes']
        self.assertEqual((direct['source_total'], direct['scope_total'], direct['unreviewed_count']), (3, 3, 0))

    def test_failed_scope_counts_all_source_units_unreviewed_with_null_accuracy(self):
        _, expected, scope = self.scoped_fixture()
        result = metrics.summarize_audit(None, status='failed', expected_units=expected, audit_scope=scope)
        claim = result['metrics']['claims']
        self.assertEqual((claim['source_total'], claim['scope_total'], claim['unreviewed_count']), (2, 1, 2))
        self.assertEqual(claim['out_of_scope_count'], 1)
        self.assertEqual(claim['source_review_coverage'], 0)
        self.assertIsNone(claim['uncertain'])
        self.assertIsNone(claim['accuracy'])

    def test_full_failed_review_does_not_claim_all_source_units_were_reviewed(self):
        expected = {group: [group + '-1'] for group in metrics.GROUPS}
        result = metrics.summarize_audit(None, status='failed', expected_units=expected)
        for item in result['metrics'].values():
            self.assertEqual(item['source_total'], 1)
            self.assertEqual(item['scope_total'], 1)
            self.assertEqual(item['out_of_scope_count'], 0)
            self.assertEqual(item['unreviewed_count'], 1)
            self.assertEqual(item['source_review_coverage'], 0)
            self.assertIsNone(item['accuracy'])

    def test_in_scope_probe_fault_is_unreviewed_not_out_of_scope(self):
        payload, expected, scope = self.scoped_fixture()
        payload['probes'][1] = probe('incorrect_solution', False, None, 'infra_error')
        result = metrics.summarize_audit(payload, expected_units=expected, audit_scope=scope)
        item = result['metrics']['probes']
        self.assertEqual(item['out_of_scope_count'], 0)
        self.assertEqual(item['unreviewed_count'], 1)
        self.assertEqual(item['unreviewed_ids'], ['incorrect_solution'])
        self.assertEqual(item['source_review_coverage'], .5)
        self.assertEqual(item['uncertain'], 0)

    def test_scope_cannot_silently_shrink_or_relabel_omitted_units(self):
        payload, expected, scope = self.scoped_fixture()
        scope['unreviewed_ids']['claims'] = []
        with self.assertRaisesRegex(ValueError, 'unreviewed IDs'):
            metrics.summarize_audit(payload, expected_units=expected, audit_scope=scope)

    def test_parallel_observation_metric_is_separate_from_aggregate_claims(self):
        payload=empty_payload()
        payload['claims']=[judgment('c1','incorrect')]
        payload['observations']=[judgment('s00-o1','correct'),judgment('s01-o1','incorrect'),
                                 judgment('s02-o1','uncertain')]
        expected={group:[unit['id'] for unit in units] for group,units in payload.items()}
        result=metrics.summarize_audit(payload,expected_units=expected)
        observed=result['metrics']['observations']
        self.assertEqual(observed['name'],'observation_correctness')
        self.assertEqual((observed['numerator'],observed['denominator']),(1,2))
        self.assertEqual(observed['accuracy'],.5)
        self.assertEqual(observed['uncertain'],1)
        self.assertEqual(observed['evaluation_coverage'],1)
        self.assertAlmostEqual(observed['decidable_coverage'],2/3)
        self.assertEqual(result['metrics']['claims']['accuracy'],0)
        failed=metrics.summarize_audit(None,status='failed',expected_units=expected)
        self.assertEqual(failed['metrics']['observations']['missing'],3)
        self.assertIsNone(failed['metrics']['observations']['accuracy'])

    def test_parallel_dropped_or_extra_group_or_observation_rejected(self):
        payload=empty_payload();payload['observations']=[judgment('s00-o1','correct')]
        expected={group:[unit['id'] for unit in units] for group,units in payload.items()}
        bad=copy.deepcopy(payload);del bad['observations']
        with self.assertRaises(ValueError):
            metrics.summarize_audit(bad,expected_units=expected)
        bad=copy.deepcopy(payload);bad['observations']=[]
        with self.assertRaises(ValueError):
            metrics.summarize_audit(bad,expected_units=expected)
        bad=copy.deepcopy(payload);bad['observations']*=2
        with self.assertRaises(ValueError):
            metrics.summarize_audit(bad,expected_units=expected)
        with self.assertRaises(ValueError):
            metrics.summarize_audit(payload,expected_units={g:[] for g in metrics.GROUPS})
        self.assertEqual(set(metrics.summarize_audit(empty_payload())['metrics']),set(metrics.GROUPS))

    def test_all_nine_mapping_pairs(self):
        expected = {
            ("correct", "correct"): "correct", ("correct", "incorrect"): "incorrect",
            ("correct", "uncertain"): "uncertain", ("incorrect", "correct"): "incorrect",
            ("incorrect", "incorrect"): "incorrect", ("incorrect", "uncertain"): "incorrect",
            ("uncertain", "correct"): "uncertain", ("uncertain", "incorrect"): "incorrect",
            ("uncertain", "uncertain"): "uncertain",
        }
        for pair, verdict in expected.items():
            with self.subTest(pair=pair):
                self.assertEqual(metrics.mapping_verdict(*pair), verdict)

    def test_uncertainty_excluded_from_accuracy_and_reported_in_coverage(self):
        payload = empty_payload()
        payload["claims"] = [judgment("c1", "correct"), judgment("c2", "incorrect"),
                             judgment("c3", "uncertain")]
        group = metrics.summarize_audit(payload)["metrics"]["claims"]
        self.assertEqual((group["numerator"], group["denominator"]), (1, 2))
        self.assertEqual(group["accuracy"], .5)
        self.assertEqual(group["uncertain"], 1)
        self.assertEqual(group["evaluation_coverage"], 1)
        self.assertAlmostEqual(group["decidable_coverage"], 2 / 3)

    def test_probe_matrix_uncertainty_and_execution_failures_are_separate(self):
        payload = empty_payload()
        payload["probes"] = [probe("tp", True, True), probe("tn", False, False),
                             probe("fp", False, True), probe("fn", True, False),
                             probe("unknown", None, True),
                             probe("infra", True, None, "infra_error"),
                             probe("missing", False, None, "missing")]
        group = metrics.summarize_audit(payload)["metrics"]["probes"]
        self.assertEqual([group[k] for k in ("TP", "TN", "FP", "FN")], [1, 1, 1, 1])
        self.assertEqual((group["numerator"], group["denominator"]), (2, 4))
        self.assertEqual((group["uncertain"], group["infra_error"], group["missing"]), (1, 1, 1))
        self.assertAlmostEqual(group["evaluation_coverage"], 5 / 7)
        self.assertAlmostEqual(group["decidable_coverage"], 4 / 7)
        self.assertEqual(group["execution_status_counts"], {"completed": 5, "infra_error": 1, "missing": 1})
        self.assertEqual(group['false_accept_rate'], .5)
        self.assertEqual(group['false_reject_rate'], .5)
        self.assertEqual(group['valid_denominator'], 2)
        self.assertEqual(group['invalid_denominator'], 2)

    def test_error_rates_keep_class_denominators_and_unknowns_separate(self):
        payload=empty_payload()
        payload['probes']=[probe('tp1',True,True),probe('tp2',True,True),probe('fn',True,False),
                           probe('fp',False,True),probe('infra',False,None,'infra_error')]
        group=metrics.summarize_audit(payload)['metrics']['probes']
        self.assertEqual(group['false_accept_rate'],1)
        self.assertAlmostEqual(group['false_reject_rate'],1/3)
        payload['probes']=[probe('tp',True,True)]
        group=metrics.summarize_audit(payload)['metrics']['probes']
        self.assertIsNone(group['false_accept_rate'])
        self.assertEqual(group['false_reject_rate'],0)

    def test_failed_envelope_cannot_supply_accepted_metrics(self):
        with self.assertRaisesRegex(ValueError, 'accepted metric payload'):
            metrics.summarize_envelope({'status':'failed','payload':empty_payload(),'failures':['commit failed']})

    def test_probe_rejects_numeric_booleans_and_contradictory_status(self):
        bad_probes = [probe("p", 1, True), probe("p", False, 0), probe("p", True, None),
                      probe("p", True, False, "infra_error"), probe("p", False, True, "missing"),
                      probe("p", None, None, "unrecognized")]
        for bad in bad_probes:
            payload = empty_payload()
            payload["probes"] = [bad]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                metrics.summarize_audit(payload)

    def test_empty_and_uncertain_denominators_are_null(self):
        payload = empty_payload()
        payload["task_checks"] = [judgment("t", "uncertain")]
        result = metrics.summarize_audit(payload)
        for group in result["metrics"].values():
            self.assertIsNone(group["accuracy"])
            self.assertEqual(group["denominator"], 0)
        self.assertIsNone(result["metrics"]["claims"]["evaluation_coverage"])
        self.assertEqual(result["metrics"]["task_checks"]["evaluation_coverage"], 1)

    def test_failed_audit_preserves_failure_and_unknown_counts(self):
        failure = {"kind": "infra_error", "reason": "Connection lost before any audit payload."}
        result = metrics.summarize_audit(None, status="failed", failures=[failure])
        self.assertEqual(result["failures"], [failure])
        for group in result["metrics"].values():
            for key in ("total", "missing", "correct", "incorrect", "denominator", "accuracy", "evaluation_coverage"):
                self.assertIsNone(group[key])
        expected = {group: [group + "1"] for group in metrics.GROUPS}
        result = metrics.summarize_audit(None, status="failed", expected_units=expected)
        for group in result["metrics"].values():
            self.assertEqual(group["missing"], 1)
            self.assertIsNone(group["accuracy"])

    def test_rejects_missing_duplicate_or_mismatched_ids_and_verdicts(self):
        payload = empty_payload()
        payload["claims"] = [judgment("c", "correct")]
        expected = {group: [] for group in metrics.GROUPS}
        with self.assertRaisesRegex(ValueError, "differ"):
            metrics.summarize_audit(payload, expected_units=expected)
        payload["claims"].append(judgment("c", "incorrect"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            metrics.summarize_audit(payload)
        payload["claims"] = [judgment("c", "missing")]
        with self.assertRaisesRegex(ValueError, "verdict"):
            metrics.summarize_audit(payload)
        del payload["claims"]
        with self.assertRaisesRegex(ValueError, "lacks"):
            metrics.summarize_audit(payload)
        with self.assertRaisesRegex(ValueError, "missing payload"):
            metrics.summarize_audit(None)

    def test_preserves_full_evidence_and_explicit_version_identity_without_mutation(self):
        payload = empty_payload()
        payload["claims"] = [judgment("c", "correct")]
        payload["claims"][0].update(reason="长证据" * 10000,
                                     evidence=[{"path": "/source/task.json", "locator": "events[400]"}])
        provenance = {"source_task": "source1", "origin": "author", "candidate_version": "draft-v1",
                      "tested_version": "tested-v2", "model": "auditor", "extra": ["完整"]}
        original = copy.deepcopy(payload)
        result = metrics.summarize_audit(payload, provenance=provenance)
        self.assertEqual(result["provenance"], provenance)
        self.assertEqual(result["metrics"]["claims"]["units"][0]["audit"], original["claims"][0])
        self.assertEqual(payload, original)
        result["provenance"]["extra"].append("changed")
        result["metrics"]["claims"]["units"][0]["audit"]["reason"] = "changed"
        self.assertEqual(payload, original)
        self.assertEqual(provenance["extra"], ["完整"])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "run1"
        self.run.mkdir()
        (self.run / "run.json").write_text(json.dumps({"run_id": "run1", "benchmark": "sheet",
                                                      "tasks": ["t1", "t2"], "episodes": [1, 2]}))

    def episode(self, task, number, *, difficulty=0, version="v1"):
        ep = self.run / "sheet" / task / f"ep{number:02d}"
        (ep / "diagnostics").mkdir(parents=True)
        (ep / "score.json").write_text(json.dumps({"difficulty": difficulty, "episode_raw": 0.0}))
        payload = empty_payload()
        payload["claims"] = [judgment("c1", "correct")]
        payload["claims"][0]["evidence"] = [{"path": "/source/full.json", "locator": "events[300]"}]
        envelope = {"status": "completed", "payload": payload, "failures": [],
                    "expected_units": {group: [item["id"] for item in payload[group]] for group in metrics.GROUPS},
                    "provenance": {"source_task": task, "origin": "author", "candidate_version": version},
                    "evidence_fingerprint": "fingerprint-" + version, "work_dir": str(ep / "audit-job")}
        (ep / "diagnostics" / "audit.json").write_text(json.dumps(envelope))
        return ep

    def test_selected_scope_export_has_source_counts_and_unreviewed_labels(self):
        ep = self.episode('t1', 1)
        path = ep / 'diagnostics/audit.json'
        envelope = json.loads(path.read_text())
        source = copy.deepcopy(envelope['expected_units'])
        source['claims'].append('c2')
        envelope['audit_scope'] = {'schema_version': 'audit-scope-v1', 'mode': 'selected_design',
            'source_units': source, 'expected_units': copy.deepcopy(envelope['expected_units']),
            'unreviewed_ids': {group: ['c2'] if group == 'claims' else [] for group in source}}
        path.write_text(json.dumps(envelope))
        metrics.write_run_summary(self.run)
        with (self.run / 'diagnostics-summary.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['scope_mode'], 'selected_design')
        self.assertEqual(row['claims_source_total'], '2')
        self.assertEqual(row['claims_scope_total'], '1')
        self.assertEqual(row['claims_unreviewed_count'], '1')
        self.assertEqual(row['claims_source_review_coverage'], '0.5')
        self.assertEqual(row['claims_uncertain'], '0')
        markdown = (self.run / 'diagnostics-summary.md').read_text()
        self.assertIn('selected_design', markdown)
        self.assertIn('unreviewed=1', markdown)

    def test_mixed_parallel_and_sequential_export_has_optional_fifth_metric(self):
        ep=self.episode('t1',1)
        self.episode('t1',2)
        path=ep/'diagnostics/audit.json'
        envelope=json.loads(path.read_text())
        envelope['payload']['observations']=[judgment('s00-o1','uncertain')]
        envelope['expected_units']['observations']=['s00-o1']
        path.write_text(json.dumps(envelope))
        report=metrics.write_run_summary(self.run)
        self.assertEqual(report['episodes'][0]['diagnostics']['metrics']['observations']['uncertain'],1)
        self.assertNotIn('observations',report['episodes'][1]['diagnostics']['metrics'])
        with (self.run/'diagnostics-summary.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['observations_denominator'],'0')
        self.assertEqual(rows[0]['observations_uncertain'],'1')
        self.assertEqual(rows[1]['observations_denominator'],'null')
        md=(self.run/'diagnostics-summary.md').read_text()
        self.assertIn('Observations',md)
        self.assertIn('not applicable',md)
        envelope['payload']['observations']=[]
        path.write_text(json.dumps(envelope))
        failed=metrics.episode_summary(ep,run_dir=self.run)['diagnostics']
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(failed['metrics']['observations']['missing'],1)
        self.assertIsNone(failed['metrics']['observations']['accuracy'])

    def test_every_planned_episode_included_and_versions_remain_separate(self):
        self.episode("t1", 1, difficulty=0, version="v1")
        self.episode("t1", 2, difficulty=1, version="v2")
        report = metrics.build_run_summary(self.run)
        self.assertEqual(report["episode_count"], 4)
        rows = report["episodes"]
        self.assertEqual(rows[0]["score"]["difficulty"], 0)
        self.assertEqual(rows[0]["diagnostics"]["metrics"]["claims"]["accuracy"], 1)
        self.assertEqual([row["diagnostics"]["provenance"]["candidate_version"] for row in rows[:2]], ["v1", "v2"])
        self.assertEqual([row["diagnostics"]["status"] for row in rows[2:]], ["missing", "missing"])
        self.assertIsNone(rows[2]["diagnostics"]["metrics"]["claims"]["accuracy"])
        self.assertIn("episode_missing", [entry["kind"] for entry in rows[2]["failures"]])

    def test_invalid_audit_is_explicit_failure_with_no_metrics(self):
        ep = self.episode("t1", 1)
        path = ep / "diagnostics" / "audit.json"
        envelope = json.loads(path.read_text())
        envelope["payload"]["claims"][0]["verdict"] = "yes"
        envelope["failures"] = [{"kind": "prior_attempt", "reason": "x" * 20000}]
        path.write_text(json.dumps(envelope))
        row = metrics.episode_summary(ep, run_dir=self.run)
        self.assertEqual(row["diagnostics"]["status"], "failed")
        self.assertIsNone(row["diagnostics"]["metrics"]["claims"]["accuracy"])
        self.assertEqual(row["failures"][0]["reason"], "x" * 20000)
        self.assertEqual(row["diagnostics"]["provenance"], envelope["provenance"])
        self.assertEqual(row["audit_envelope"], envelope)
        self.assertIn("invalid_audit", [entry["kind"] for entry in row["failures"]])

    def test_current_missing_audit_does_not_substitute_historical_version(self):
        ep = self.episode("t1", 1)
        current = ep / "diagnostics" / "audit.json"
        historic = ep / "diagnostics" / "audit-runs" / "old"
        historic.mkdir(parents=True)
        current.rename(historic / "audit.json")
        row = metrics.episode_summary(ep, run_dir=self.run)
        self.assertEqual(row["diagnostics"]["status"], "missing")
        self.assertIsNone(row["diagnostics"]["provenance"])
        self.assertIsNone(row["diagnostics"]["metrics"]["claims"]["accuracy"])

    def test_nonobject_audit_produces_failure_row_instead_of_crashing_report(self):
        ep = self.episode("t1", 1)
        (ep / "diagnostics" / "audit.json").write_text('["invalid envelope"]')
        report = metrics.build_run_summary(self.run)
        self.assertEqual(report["episode_count"], 4)
        row = report["episodes"][0]
        self.assertEqual(row["diagnostics"]["status"], "failed")
        self.assertIsNone(row["diagnostics"]["metrics"]["claims"]["accuracy"])
        self.assertEqual(row["audit_envelope"], ["invalid envelope"])

    def test_cli_exports_json_csv_markdown_evidence_failures_and_nulls(self):
        ep = self.episode("t1", 1)
        cli = Path(metrics.__file__).resolve()
        result = subprocess.run([sys.executable, str(cli), "--run-dir", str(self.run)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["episodes"], 4)
        report = json.loads((self.run / "diagnostics-summary.json").read_text())
        self.assertEqual(report["episode_count"], 4)
        with (self.run / "diagnostics-summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["difficulty"], "0")
        self.assertEqual(rows[0]["claims_accuracy"], "1.0")
        self.assertEqual(rows[1]["claims_accuracy"], "null")
        self.assertIn("/source/full.json", json.loads(rows[0]["evidence_paths"]))
        self.assertIn(str((ep / "diagnostics" / "audit.json").resolve()), json.loads(rows[0]["evidence_paths"]))
        md = (self.run / "diagnostics-summary.md").read_text()
        self.assertIn("not human gold", md)
        self.assertIn("audit_missing", md)
        self.assertIn("sheet/t1/ep01", md)
        self.assertIn("/source/full.json", md)


if __name__ == "__main__":
    unittest.main()
