"""Exercise the shipped reference checker through the command a user runs."""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / 'output' / 'triplet_sextet'


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

    def test_ranked_final_state_and_decay_history(self):
        direct = lambda row: row['production_process'] == 'p p > x1' and row['final_state'] == 'u u'
        for key, replacement in [('final_pdgs', '[21, 21]'), ('final_state', 'g g'),
                                 ('history', 'x1 → (g g)'),
                                 ('history_key', '[[9700001, [[21, []], [21, []]]]]')]:
            with self.subTest(field=key):
                shutil.copyfile(REFERENCE / 'ranking.csv', self.run_dir / 'ranking.csv')
                self.edit_row('ranking', key, direct, replacement)
                result = self.compare()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)

    def test_production_error_is_checked_even_when_rate_error_is_correct(self):
        self.edit_row('ranking', 'production_error_pb', lambda r: r['final_state'] == 'u u', '999')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('production_error_pb', result.stderr)

    def test_zero_decay_branching_fraction(self):
        self.edit_row('decays', 'branching_ratio', lambda r: r['process'] == 'x1 > u u', '0')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('decays', result.stderr)
        self.assertIn('BR', result.stderr)

    def test_fraction_must_equal_partial_width_over_total(self):
        # Inside the cross-machine tolerance; conflicts with this run's own widths.
        self.edit_row('decays', 'branching_ratio', lambda r: r['process'] == 'x1 > u u', '.33925')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('partial / total width', result.stderr)

    def test_cascade_must_use_saved_primary_branch(self):
        direct = lambda r: r['process'] == 'x1 > u u'
        self.edit_row('decays', 'branching_ratio', direct, '.33925')
        self.edit_row('decays', 'width_gev', direct, str(.33925 * .87965))
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cascade BR from primary decays', result.stderr)

    def test_primary_decay_label(self):
        self.edit_row('decays', 'process', lambda r: r['process'] == 'x1 > u u', 'x1 > g g')
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('process label', result.stderr)

    def test_unrecognized_excerpt_cannot_skip_missing_modes(self):
        path = self.run_dir / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['source_ranking_sha256'] = 'unrelated source'
        path.write_text(json.dumps(manifest))
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing primary decay', result.stderr)
