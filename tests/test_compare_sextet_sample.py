"""Exercise the shipped reference checker through the command a user runs."""
import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / 'sample_output'


class SextetComparisonTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='sextet example ')
        self.addCleanup(tmp.cleanup)
        self.run_dir = Path(tmp.name)
        for path in [*REFERENCE.glob('*.csv'), REFERENCE / 'manifest.json']:
            shutil.copyfile(path, self.run_dir / path.name)

    def compare(self):
        return subprocess.run([sys.executable, str(REFERENCE / 'compare.py'), str(self.run_dir)],
                              capture_output=True, text=True, cwd=self.run_dir)

    def edit_row(self, name, field, select, value):
        path = self.run_dir / (name + '.csv')
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        next(r for r in rows if select(r))[field] = value
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_reference(self):
        result = self.compare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('1,925 histories, 30 selected rates', result.stdout)

    def test_production_outside_mc_uncertainties(self):
        self.edit_row('cross_sections', 'cross_section_pb', lambda r: r['process'] == 'p p > x1', '2.5')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Production cross section', result.stderr)

    def test_uncalculated_is_distinct_from_zero(self):
        self.edit_row('ranking', 'rate_pb', lambda r: r['rate_pb'] == '', '0')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('coverage disagree', result.stderr)

    def test_changed_partial_width(self):
        self.edit_row('decays', 'width_gev', lambda r: r['process'] == 'x1 > u u', '0.01')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('decays', result.stderr)
