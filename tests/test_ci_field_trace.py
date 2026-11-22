"""Credit-free CI checks for forensic evidence and benchmark log redaction."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ci import trace_ufo
from ci.run_benchmark import Benchmark


class FieldTraceTests(unittest.TestCase):
    def test_requires_marker_and_valid_zero_spin_leg_from_same_attempt(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            base = root / 'model' / 'run' / 'export' / 'key'
            old, new = base / 'attempt-old', base / 'attempt-current'
            field = new / 'CI_UFO' / trace_ufo.FIELD_FILE
            field.parent.mkdir(parents=True)
            old.mkdir()
            (old / 'feynrules.log').write_text(trace_ufo.GUARD)
            (new / 'feynrules.log').write_text('SetDelayed::write')
            os.utime(old, (4, 4))
            os.utime(new, (8, 8))
            leg = {'position': 1, 'incoming_leg_id': '1', 'outgoing_leg_id': '1',
                   'incoming_field': 'System`someField[q]',
                   'outgoing_field': 'qbar', 'incoming_symbol_or_head': 'someField',
                   'outgoing_symbol_or_head': 'qbar', 'incoming_spin_code': 0, 'outgoing_spin_code': 0}
            report = {'stage': 'MakeOutgoingParticles', 'vertex_index': 31, 'number_of_legs': 3,
                      'unclassified_outgoing_legs': 1, 'legs': [leg], 'unused': name,
                      'match_level': 'first_unclassified', 'first_unclassified_vertex_index': 31}
            field.write_text(json.dumps(report))
            benchmark = SimpleNamespace(output=root, report={'commands': [{'action': 'ufo', 'status': 'failed'}]})
            self.assertEqual(trace_ufo.export_evidence(benchmark)['observation'], 'inconclusive')
            (new / 'feynrules.log').write_text(trace_ufo.GUARD + ' vertex 31')
            result = trace_ufo.export_evidence(benchmark)
            self.assertEqual(result['observation'], 'guard_reproduced_unknown_fields')
            self.assertEqual(result['fields']['legs'], [leg])
            self.assertNotIn(name, json.dumps(result))
            leg['outgoing_spin_code'] = 1
            field.write_text(json.dumps(report))
            result = trace_ufo.export_evidence(benchmark)
            self.assertTrue(result['invalid_field_report'])
            self.assertEqual(result['observation'], 'inconclusive')

    def test_baseline_signature_counts_and_ambiguous_matches(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / trace_ufo.FIELD_FILE
            rows = []
            for i, spin in enumerate((0, 0, 0, 0, 1, 0), 1):
                row = {'position': i}
                for side in ('incoming', 'outgoing'):
                    row.update({side + '_leg_id': str(i), side + '_field': 'phi',
                                side + '_symbol_or_head': 'phi', side + '_spin_code': spin})
                rows.append(row)
            data = dict(stage='MakeOutgoingParticles', vertex_index=43, number_of_legs=6,
                        unclassified_outgoing_legs=5, legs=rows, match_level='exact_ordered_signature',
                        first_unclassified_vertex_index=5, ordered_signature_match_count=2,
                        six_leg_multiset_match_count=7, vertices_in_replay=600)
            path.write_text(json.dumps(data))
            result = trace_ufo.field_report(path)
            self.assertEqual(result['matching_vertices_at_selected_level'], 2)
            self.assertTrue(result['ambiguous_at_selected_level'])
            rows[0]['outgoing_spin_code'], rows[4]['outgoing_spin_code'] = 1, 0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                trace_ufo.field_report(path)
            data.update(match_level='six_leg_multiset', ordered_signature_match_count=0)
            path.write_text(json.dumps(data))
            self.assertEqual(trace_ufo.field_report(path)['matching_vertices_at_selected_level'], 7)

    def test_cli_preserves_syntax_cursors_and_excerpt_on_inconclusive_export(self):
        secret = 'private0123456789'
        with TemporaryDirectory(prefix='field trace ') as name, patch.dict(
                os.environ, {'WOLFRAMINIT': '-pwfile !cloudlm.wolfram.com -entitlement ' + secret}):
            root = Path(name)
            model = Benchmark(root, 'licensed', {'commands': []})
            sample = 'GC_30058 = Invalid[' + secret + '] at ' + name + '/private log with spaces'
            error = (f'Invalid generated UFO Python in {root}/saved results/CI_UFO/vertices.py, '
                     f'line 5017, column 42, source {sample!r}: invalid syntax {secret}. '
                     "Conversion diagnostic: more context (line 5018, column 3, source 'V_9 = ?')")

            def fail(command, *, stdout, **kwargs):
                stdout.write(('irrelevant log text\n' * 9000 + error + '\n').encode())
                return subprocess.CompletedProcess(command, 1)

            with patch('ci.run_benchmark.subprocess.run', side_effect=fail), self.assertRaises(AssertionError) as caught:
                model.cli('ufo')
            result = trace_ufo.export_evidence(model)
            self.assertEqual(result['observation'], 'inconclusive')
            evidence = result['cli_failure']
            self.assertEqual(evidence['module'], 'vertices.py')
            self.assertEqual([(r['line'], r['column']) for r in evidence['locations']], [(5017, 42), (5018, 3)])
            self.assertIn('GC_30058', evidence['locations'][0]['source'])
            self.assertIn('invalid syntax', evidence['reason'])
            published = json.dumps(result) + str(caught.exception)
            for private in (name, secret, 'private log with spaces', 'irrelevant log text'):
                self.assertNotIn(private, published)
            self.assertLess(len(published), 1400)

    def test_keep_actionable_numerical_or_dependency_failure(self):
        with TemporaryDirectory() as name:
            log = Path(name) / 'command.log'
            log.write_text('Run files: ' + name + '\nC0004: integration missed 1% accuracy; log ' + name +
                           '/process with spaces/private.log\n')
            result = trace_ufo.cli_failure(log, 'command failed')
            self.assertIn('C0004: integration missed 1% accuracy', result['reason'])
            self.assertNotIn(name, result['reason'])
            self.assertNotIn('private.log', result['reason'])
            log.write_text('MISSING madgraph: could not find executable\nOK python: available\n')
            self.assertEqual(trace_ufo.cli_failure(log, 'command failed')['reason'],
                             'MISSING madgraph: could not find executable')


if __name__ == '__main__':
    unittest.main()
