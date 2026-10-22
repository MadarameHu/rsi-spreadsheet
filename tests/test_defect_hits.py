from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import defect_hits as dh  # noqa: E402


def eq(a, b):
    return (a in (None, '') and b in (None, '')) or a == b


EXPECTED = {f'A{i}': i for i in range(1, 11)}
ZERO9 = {f'A{i}': 0 for i in range(1, 10)}
PROBES = [('c1-shift', {**EXPECTED, **ZERO9}),
          ('c2-scale', {k: v * 100 for k, v in EXPECTED.items()}),
          ('c3-near-c1', {**EXPECTED, **ZERO9, 'A10': 0})]


class DefectHitTests(unittest.TestCase):
    def test_exact_reproduction_is_a_hit(self):
        r = dh.classify({k: v * 100 for k, v in EXPECTED.items()}, EXPECTED, PROBES, eq)
        self.assertEqual((r['category'], r['classes']), ('hit', ['c2']))

    def test_close_probes_are_ambiguous(self):
        r = dh.classify({**EXPECTED, **ZERO9}, EXPECTED, PROBES, eq)
        self.assertEqual((r['category'], r['classes']), ('ambiguous', ['c1', 'c3']))

    def test_unexplained_error_is_unmatched(self):
        r = dh.classify({**EXPECTED, 'A9': -1, 'A10': -1}, EXPECTED, PROBES, eq)
        self.assertEqual((r['category'], r['classes']), ('unmatched', []))

    def test_no_value_mismatch(self):
        self.assertEqual(dh.classify(dict(EXPECTED), EXPECTED, PROBES, eq)['category'], 'no_value_mismatch')

    def test_class_id_from_probe_name(self):
        self.assertEqual(dh.probe_class('c12-foo bar'), 'c12')
        self.assertEqual(dh.probe_class('authored'), 'authored')


if __name__ == '__main__':
    unittest.main()
