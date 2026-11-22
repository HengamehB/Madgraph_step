"""Bounded-memory exports, atomic failures, and indexed history references."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import csv
import errno
import hashlib
import html
import io
import json
import shlex
import sys
import tempfile
import tracemalloc
import unittest

from pipeline_storage import JsonArray, RunStore, digest, encode, decode, public, write_json, write_csv
from pipeline_reports import RANKING_COLUMNS, export_ranking, publish, write_history_page


class GeneratedRows:
    def __init__(self, count):
        self.count = count
        self.iterations = 0

    def __len__(self):
        raise AssertionError('Streaming writers must not materialize/size the iterable')

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError('One-pass source was consumed twice')
        for index in range(self.count):
            yield dict(id=index, text='x' * 1024, rate=None)


class FixtureRows:
    def __init__(self, root):
        self.path = root / 'histories.sqlite'
        self.path.write_bytes(b'fixture indexed store')
        self.revision = 1
        self.iterations = 0
        self.page_calls = 0
        self.forbid_scan = False
        self.data = [dict(history_id='h1', production_id='C1', history='X > a a',
                          history_ref=[1], rate_status='calculated', rate_pb=2., branching_ratio=.5),
                     dict(history_id='h2', production_id='C2', history='Y > a a',
                          history_ref=[2], rate_status='uncalculated', rate_pb=None, branching_ratio=.3)]

    def reference(self):
        return dict(path=str(self.path), schema='fixture-v1', fingerprint='a' * 64, revision=self.revision)

    def counts(self):
        return dict(total=len(self.data), calculated=sum(r['rate_pb'] is not None for r in self.data),
                    discovery_complete=True, revision=self.revision)

    def iter_rows(self):
        if self.forbid_scan:
            raise AssertionError('Progress/page rendering scanned the full row collection')
        self.iterations += 1
        yield from self.data

    def iter_nodes(self):
        yield dict(node_id=1, key='aa', pdg=900, children=[2, 2], branching_ratio=.5,
                   final_counts={'22': 2}, flags=[])
        yield dict(node_id=2, key='a', pdg=22, children=[], branching_ratio=1.,
                   final_counts={'22': 1}, flags=[])

    def page(self, *, limit=100, cursor=None, min_relative_rate=0, include_uncalculated=True):
        self.page_calls += 1
        return dict(rows=self.data[:min(limit, 1)], next_cursor='revision-bound-next',
                    revision=self.revision, total=len(self.data), page_size=limit)


@dataclass
class FixtureRanking:
    rows: object
    coverage: dict
    omitted_probability: dict
    notices: tuple = ()
    uncomputed_production_ids: tuple = ('C2',)
    calculation_limits: dict = None


class StreamingStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='streamed history tests ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_json_and_csv_memory_is_bounded_for_one_pass_large_sources(self):
        for extension in ('json', 'csv'):
            with self.subTest(extension=extension):
                rows = GeneratedRows(20000)  # More than 20 MB of output.
                tracemalloc.start()
                try:
                    if extension == 'json':
                        result = write_json(self.root / 'large.json', {'rows': JsonArray(rows)})
                    else:
                        result = write_csv(self.root / 'large.csv', rows, ('id', 'text', 'rate'), fixed=True)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertGreater(result['size_bytes'], 20_000_000)
                self.assertLess(peak, 4_000_000)
                self.assertEqual(rows.iterations, 1)
                with (self.root / ('large.' + extension)).open('rb') as stream:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(65536), b''):
                        hasher.update(chunk)
                self.assertEqual(result['sha256'], hasher.hexdigest())

    def test_streamed_digest_preserves_existing_canonical_fingerprints(self):
        value = {'β': [1, 2.5, 1+2j, None, True], 'path': self.root, 3: ('a', 'b')}
        expected = hashlib.sha256(json.dumps(public(value), sort_keys=True, ensure_ascii=False,
                                            allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest(value), expected)

    def test_interrupted_writer_retains_previous_file_and_removes_temporary(self):
        target = self.root / 'rows.json'
        target.write_text('previous complete result')
        failure = OSError(errno.ENOSPC, 'No space left on device')
        def rows():
            yield {'id': 1}
            raise failure
        with self.assertRaises(OSError) as caught:
            write_json(target, JsonArray(rows()))
        self.assertIs(caught.exception, failure)
        self.assertEqual(target.read_text(), 'previous complete result')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['rows.json'])

    def test_fixed_csv_schema_keeps_nulls_and_rejects_unannounced_columns(self):
        target = self.root / 'rows.csv'
        write_csv(target, iter(({'id': 1, 'rate': 3.}, {'id': 2})), ('id', 'rate'), fixed=True)
        with target.open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [{'id': '1', 'rate': '3.0'}, {'id': '2', 'rate': ''}])
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'fields not in fieldnames'):
            write_csv(target, iter(({'id': 3, 'unexpected': 1},)), ('id', 'rate'), fixed=True)
        self.assertEqual(target.read_bytes(), before)

    def test_native_fast_paths_preserve_exact_json_and_csv_values(self):
        import numpy as np
        @dataclass
        class Row:
            history: tuple
            values: dict
            scalar: object
            _hidden: str = 'excluded'
        rows = [Row(((900, ((22, ()), (22, ()))),),
                    {'path': self.root, 'complex': 1+2j, 'fraction': Fraction(2, 3),
                     'array': np.array([1+2j, 3-4j]), 'text': 'α "quoted"\nline',
                     'none': None, 'set': frozenset((1, 2))}, np.complex64(2+3j))]
        for native in (False, True):
            write_json(self.root / f'{native}.json', JsonArray(iter(rows), native=native))
            write_csv(self.root / f'{native}.csv', iter(rows), ('history', 'values', 'scalar'),
                      fixed=True, native=native)
        self.assertEqual((self.root / 'True.json').read_bytes(), (self.root / 'False.json').read_bytes())
        self.assertEqual((self.root / 'True.csv').read_bytes(), (self.root / 'False.csv').read_bytes())

    def test_native_history_forest_is_not_recursively_copied(self):
        tree = (22, ())
        for _ in range(7):
            tree = (900, (tree, tree))
        row = {'history_key': (tree, tree), 'rate_pb': None, 'flags': ('α',)}
        expected = public(row)
        with patch('pipeline_storage.public', side_effect=AssertionError('Native containers copied recursively')):
            write_json(self.root / 'native.json', JsonArray(iter((row,)), native=True))
            write_csv(self.root / 'native.csv', iter((row,)), tuple(row), fixed=True, native=True)
        self.assertEqual(json.loads((self.root / 'native.json').read_text()), [expected])
        with (self.root / 'native.csv').open() as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(json.loads(saved['history_key']), expected['history_key'])
        self.assertEqual(saved['rate_pb'], '')

    def test_generic_mapping_key_conventions_remain_unchanged(self):
        row = {'data': {None: 'none', True: 'true', 5: 'integer', (1, 2): 'tuple'}}
        write_json(self.root / 'keys.json', JsonArray(iter((row,))))
        write_csv(self.root / 'keys.csv', iter((row,)), ('data',), fixed=True)
        self.assertEqual(json.loads((self.root / 'keys.json').read_text()), [public(row)])
        with (self.root / 'keys.csv').open() as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(json.loads(saved['data']), public(row)['data'])

    def test_fast_and_general_writers_reject_nonfinite_values_atomically(self):
        for native in (False, True):
            for value in (float('nan'), float('inf'), [float('-inf')], {'a': complex(float('nan'), 1)}):
                for extension in ('json', 'csv'):
                    with self.subTest(native=native, value=value, extension=extension):
                        target = self.root / ('finite.' + extension)
                        target.write_text('previous complete result')
                        with self.assertRaises(ValueError):
                            if extension == 'json':
                                write_json(target, JsonArray(iter(({'value': value},)), native=native))
                            else:
                                write_csv(target, iter(({'value': value},)), ('value',), fixed=True, native=native)
                        self.assertEqual(target.read_text(), 'previous complete result')

    def test_checkpoint_stores_only_checked_relative_reference(self):
        root = self.root
        class SQLiteHistoryRows:
            __module__ = 'history_store'
            def __init__(self, reference):
                self.ref = reference
            def reference(self, *, root=None, checksum=False):
                from run_paths import portable_history_reference
                return portable_history_reference(self.ref, root) if root is not None else self.ref
            def __iter__(self):
                raise AssertionError('Checkpoint traversed histories')
            @classmethod
            def from_reference(cls, reference, *, root=None):
                from run_paths import portable_history_reference
                return cls(portable_history_reference(reference, root, absolute=True))
        reference = dict(path=str((root / 'history.sqlite').resolve()), schema='test-v1',
                         fingerprint='a' * 64, revision=7)
        rows = SQLiteHistoryRows(reference)
        with patch.dict(sys.modules, {'history_store': SimpleNamespace(SQLiteHistoryRows=SQLiteHistoryRows)}):
            value = encode(rows, root)
            self.assertLess(len(json.dumps(value)), 500)
            restored = decode(value, root).reference()
            self.assertEqual(str(restored.pop('path')), reference['path'])
            self.assertEqual(restored.pop('path_base'), 'run')
            self.assertEqual(restored, {key: item for key, item in reference.items() if key != 'path'})
            with patch.object(SQLiteHistoryRows, 'from_reference',
                              return_value=SQLiteHistoryRows({**reference, 'revision': 8})):
                with self.assertRaisesRegex(ValueError, 'revision changed'):
                    decode(value, root)
            outside = SQLiteHistoryRows({**reference, 'path': str(root.parent / 'elsewhere.sqlite')})
            with self.assertRaisesRegex(ValueError, 'outside'):
                encode(outside, root)


class StreamingReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='streamed report tests ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = FixtureRows(self.root)
        self.ranking = FixtureRanking(self.rows, {'C1': .5, 'C2': .3}, {'C1': .5, 'C2': .7},
                                      calculation_limits={})
        self.project = SimpleNamespace(run_directory=self.root, ranking=self.ranking,
            config=SimpleNamespace(model_file=self.root / "model ; quoted.tex", collider={}, decays={}, ranking={}),
            _channels=None, _decays=None, cross_sections=None, failures=[], inputs={'model_name': 'Fixture'},
            complete=False, requested_complete=False, _signature='a' * 64, _stage_keys={},
            _store=RunStore(self.root), export_complete=False)

    def test_complete_exports_include_uncalculated_rows_and_all_dag_nodes(self):
        result = export_ranking(self.project)
        saved = json.loads((self.root / 'ranking.json').read_text())
        self.assertEqual(len(saved['rows']), 2)
        self.assertEqual(set(saved['rows'][0]), set(RANKING_COLUMNS))
        self.assertEqual(set(saved['rows'][1]), set(RANKING_COLUMNS))
        self.assertIsNone(saved['rows'][1]['production_cross_section_pb'])
        self.assertIsNone(saved['rows'][1]['rate_pb'])
        self.assertEqual(len(json.loads((self.root / 'history_nodes.json').read_text())['nodes']), 2)
        for name, record in result['files'].items():
            self.assertEqual((self.root / name).stat().st_ino, (self.root / record['path']).stat().st_ino)
        self.rows.forbid_scan = True
        self.assertEqual(export_ranking(self.project)['identity'], result['identity'])

    def test_export_progress_counts_existing_passes_and_cached_exports_do_not_scan(self):
        from workflow_progress import report_progress
        progress = io.StringIO()
        with report_progress(stream=progress):
            result = export_ranking(self.project)
        self.assertEqual(self.rows.iterations, 2, 'Only the JSON and CSV passes may traverse rows')
        text = progress.getvalue()
        for name in ('ranking.json', 'ranking.csv', 'history_nodes.json'):
            self.assertIn('Saving ' + name, text)
        self.assertIn('2/2 rows', text)
        self.assertIsNone(json.loads((self.root / 'ranking.json').read_text())['rows'][1]['rate_pb'])
        self.rows.forbid_scan = True
        cached_progress = io.StringIO()
        with report_progress(stream=cached_progress):
            self.assertEqual(export_ranking(self.project)['identity'], result['identity'])
        self.assertNotIn('Saving ranking', cached_progress.getvalue())

    def test_compatibility_link_replacement_preserves_another_contained_file(self):
        card = self.root / 'param_card.dat'
        card.write_text('physical parameters must remain unchanged\n')
        compatibility = self.root / 'ranking.json'
        compatibility.symlink_to(card.resolve())
        export_ranking(self.project)
        self.assertFalse(compatibility.is_symlink())
        self.assertEqual(card.read_text(), 'physical parameters must remain unchanged\n')
        self.assertEqual(len(json.loads(compatibility.read_text())['rows']), 2)

    def test_archiving_compatibility_link_preserves_physical_target(self):
        card = self.root / 'param_card.dat'
        content = 'physical parameters must remain unchanged\n'
        card.write_text(content)
        compatibility = self.root / 'ranking.json'
        compatibility.symlink_to(card.resolve())
        self.project.ranking = None
        publish(self.project, status='discovering')
        self.assertEqual(card.read_text(), content)
        self.assertFalse(compatibility.is_symlink())
        archived = list((self.root / 'previous_results').glob('*/ranking.json'))
        self.assertEqual(len(archived), 1)
        self.assertTrue(archived[0].is_symlink())

    def test_progress_never_exports_or_materializes_rows_and_marks_stale_export(self):
        export_ranking(self.project)
        self.rows.revision += 1
        self.rows.forbid_scan = True
        for status in ('discovering', 'integrating', 'selected_integrating', 'failed', 'interrupted'):
            manifest = publish(self.project, status=status)
            self.assertFalse(manifest['export_complete'])
            self.assertNotIn('ranking', manifest['current_tables'])
            self.assertEqual(manifest['history_store']['revision'], 2)
        self.assertEqual(self.rows.page_calls, 5)

    def test_fresh_process_archive_does_not_rehash_or_rescan_unchanged_bundle(self):
        original = export_ranking(self.project)
        self.rows.forbid_scan = True
        self.project.ranking = None  # A fresh Pipeline has not hydrated yet.
        with patch('pipeline_reports.file_digest', side_effect=AssertionError('Unnecessary full-file rehash')):
            publish(self.project, status='discovering')
            self.project.ranking = self.ranking
            self.assertEqual(export_ranking(self.project)['identity'], original['identity'])

    def test_archive_metadata_refresh_does_not_bless_tampered_bundle(self):
        original = export_ranking(self.project)
        target = self.root / 'ranking.json'
        target.write_text('tampered')
        self.project.ranking = None
        publish(self.project, status='discovering')
        self.project.ranking = self.ranking
        before = self.rows.iterations
        repaired = export_ranking(self.project)
        self.assertEqual(repaired['identity'], original['identity'])
        self.assertEqual(self.rows.iterations, before + 1)  # Only damaged JSON.
        self.assertEqual(len(json.loads(target.read_text())['rows']), 2)

    def test_disk_full_export_preserves_old_bundle_and_can_resume(self):
        original = export_ranking(self.project)
        old_json = (self.root / 'ranking.json').read_bytes()
        self.rows.revision += 1
        self.rows.data[1]['rate_pb'] = 1.
        error = OSError(errno.ENOSPC, 'No space left on device')
        with patch('pipeline_reports.write_csv', side_effect=error):
            with self.assertRaises(OSError) as caught:
                publish(self.project, status='complete')
        self.assertIs(caught.exception, error)
        self.assertFalse(self.project.export_complete)
        self.assertEqual((self.root / 'ranking.json').read_bytes(), old_json)
        self.assertEqual(json.loads((self.root / 'ranking_exports.json').read_text())['identity'], original['identity'])
        self.assertFalse(json.loads((self.root / 'export_state.json').read_text())['complete'])
        before = self.rows.iterations
        final = export_ranking(self.project)
        self.assertEqual(self.rows.iterations, before + 1, 'Finished JSON should be reused; only CSV needs another pass')
        self.assertNotEqual(final['identity'], original['identity'])
        self.assertEqual(json.loads((self.root / 'ranking.json').read_text())['rows'][1]['rate_pb'], 1.)

    def test_bounded_html_page_has_real_quoted_cli_command_and_rejects_stale_source(self):
        self.rows.forbid_scan = True
        page = self.rows.page(limit=10)
        path = write_history_page(self.project, page)
        text = html.unescape(path.read_text())
        command = text.split('<pre>', 1)[1].split('</pre>', 1)[0]
        tokens = shlex.split(command)
        self.assertEqual(tokens[1:3], ['histories', str(self.project.config.model_file)])
        self.assertTrue(Path(tokens[0]).is_absolute())
        self.assertIn('--report-page', tokens)
        self.assertNotIn('<td>h2</td>', text)
        self.project.config.output = self.root / 'custom output'
        self.project.config.path = self.root / 'custom .config'
        self.project.config.path.touch()
        text = html.unescape(write_history_page(self.project, page).read_text())
        tokens = shlex.split(text.split('<pre>', 1)[1].split('</pre>', 1)[0])
        self.assertEqual(tokens[tokens.index('--output') + 1], str(self.project.config.output))
        self.assertEqual(tokens[tokens.index('--config') + 1], str(self.project.config.path))
        self.rows.revision += 1
        with self.assertRaisesRegex(ValueError, 'stale'):
            write_history_page(self.project, page)

    def test_completion_is_recomputed_after_exports_and_settings_are_recorded(self):
        self.project.config.explicit_decays = {'history_ids': ['h1'], 'cut_decays': True}
        def completion():
            self.project.complete = self.project.export_complete
            self.project.requested_complete = self.project.export_complete
        self.project._update_completion = completion
        manifest = publish(self.project, status='complete')
        self.assertTrue(manifest['complete'])
        self.assertTrue(manifest['requested_complete'])
        self.assertTrue(manifest['export_complete'])
        self.assertEqual(manifest['settings']['explicit_decays'], self.project.config.explicit_decays)

    def test_real_sqlite_reference_exports_pages_and_numerical_revision(self):
        from cascade_ranking import rank_final_states
        from history_store import SQLiteHistoryRows
        from test_cascade_ranking import fixture, X, Y
        channels, decays, rates = fixture((Y, Y))
        ranking = rank_final_states(channels, decays, None, (X,), show=False,
                                    storage_path=self.root / 'real-history.sqlite')
        self.project.ranking = ranking
        checkpoint = encode(ranking, self.root)
        self.assertLess(len(json.dumps(checkpoint)), 4000)
        restored = decode(checkpoint, self.root)
        self.assertIsInstance(restored.rows, SQLiteHistoryRows)
        self.assertEqual(len(restored.rows), 3)
        first_export = export_ranking(self.project)
        saved = json.loads((self.root / 'ranking.json').read_text())
        self.assertEqual(len(saved['rows']), 3)
        self.assertTrue(all(row['rate_pb'] is None for row in saved['rows']))
        first_page = ranking.rows.page(limit=1)
        write_history_page(self.project, first_page)
        second_page = ranking.rows.page(limit=1, cursor=first_page['next_cursor'])
        self.assertNotEqual(first_page['rows'][0]['history_id'], second_page['rows'][0]['history_id'])
        ranking.rows.update_rates(rates)
        ranking.refresh()
        with self.assertRaisesRegex(ValueError, 'revision'):
            decode(checkpoint, self.root)
        with self.assertRaisesRegex(ValueError, 'stale'):
            write_history_page(self.project, first_page)
        updated = export_ranking(self.project)
        self.assertNotEqual(first_export['identity'], updated['identity'])
        saved = json.loads((self.root / 'ranking.json').read_text())
        self.assertEqual([round(row['rate_pb'], 6) for row in saved['rows']], [6.4, 3.2, .4])
        nodes = json.loads((self.root / 'history_nodes.json').read_text())['nodes']
        node_ids = {node['node_id'] for node in nodes}
        self.assertTrue(all(set(row['history_ref']) <= node_ids for row in saved['rows']))
        with patch.object(SQLiteHistoryRows, 'iter_rows', side_effect=AssertionError('Unnecessary full export')):
            self.assertEqual(export_ranking(self.project)['identity'], updated['identity'])


if __name__ == '__main__':
    unittest.main()
