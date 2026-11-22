"""Durable discovery, canonical identities and bounded traversal regressions."""
from pathlib import Path
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch
import errno
import io
import itertools
import sqlite3
import tempfile
import unittest

import history_store
from cascade_ranking import _prepare_context, rank_final_states
from history_store import SQLiteHistoryRows, estimate_history_combinations, replacement_indices
from test_cascade_ranking import fixture, X, Y
from workflow_progress import report_progress


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'catalogue with spaces.sqlite'

    def rank(self, data=None, **kwargs):
        return rank_final_states(*(data or fixture((Y,Y,Y))), stable_particles=(X,),
                                 storage_path=self.path, show=False, **kwargs)

    def test_seekable_combinations(self):
        for n in range(1,7):
            for count in range(1,5):
                expected = list(itertools.combinations_with_replacement(range(n),count))
                for start in range(len(expected)+1):
                    self.assertEqual(list(replacement_indices(n,count,start)),expected[start:])
        # Huge choice count must not allocate range(n) as a tuple.
        self.assertEqual(next(replacement_indices(10**12,2,0)),(0,0))

    def test_expected_combinations_match_nested_identical_and_mixed_forests(self):
        data = fixture()
        finals = [(X,), (Y,), (X,X), (Y,Y), (X,Y)]
        data[0].channels = tuple(SimpleNamespace(id=f'C{i}',process=f'production {i}',final_pdgs=p)
                                for i,p in enumerate(finals))
        context = _prepare_context(data[0],data[1],None,stable_particles=(X,))
        expected = estimate_history_combinations(context)
        self.assertEqual(expected['particle_choices'][Y],2)
        self.assertEqual(expected['productions'],dict(C0=1,C1=2,C2=1,C3=3,C4=2))
        result = self.rank(data[:2])
        self.assertEqual(expected['total_combinations'],len(result.rows))
        self.assertEqual(len(result.rows),9)

    def test_expected_counts_do_not_construct_enormous_forests(self):
        def modes(pdg, ancestors):
            return [((-1,),.5),((-2,),.5)] if pdg == 1 else [((pdg-1,pdg-1),1.)]
        context = dict(terminal=lambda p:p<0, decay_choices=modes,
                       channels=[dict(id='C1',final_pdgs=(10,10))])
        with patch.object(history_store,'_connect',side_effect=AssertionError('Count must not open a store')):
            expected = estimate_history_combinations(context)
        self.assertGreater(expected['total_combinations'],10**30)
        self.assertEqual(len(expected['particle_choices']),12)
        choices = expected['particle_choices'][10]
        self.assertEqual(expected['total_combinations'],choices*(choices+1)//2)

    def test_finite_four_body_tree_count_exceeds_decimal_conversion_limit_safely(self):
        def modes(pdg, ancestors):
            return [((0,0),.5),((1,1),.5)] if pdg == 2 else [((pdg-1,)*4,1.)]
        context = dict(terminal=lambda p:p<2, decay_choices=modes,
                       channels=[dict(id='C1',process='p p > p10',final_pdgs=(10,))],
                       particles={},mode_records=[],total_widths={},stable=[],
                       limits=dict(min_probability=0),provenance={})
        expected = estimate_history_combinations(context)
        self.assertGreater(expected['total_combinations'].bit_length(),15000)
        # Exercise the diagnostic preflight, but forbid creating any database or
        # expanding a single forest. Quiet library calls must also reach storage
        # safely, because f-string arguments are evaluated before message().
        output = io.StringIO()
        with redirect_stderr(output), patch.object(history_store,'_connect',side_effect=RuntimeError('stop before storage')) as connect:
            with self.assertRaisesRegex(RuntimeError,'stop before storage'):
                history_store.build_history_store(context,self.path)
            self.assertEqual(connect.call_count,1)
        self.assertEqual(output.getvalue(),'')
        with report_progress(stream=output), patch.object(history_store,'_connect',side_effect=RuntimeError('stop before storage')):
            with self.assertRaisesRegex(RuntimeError,'stop before storage'):
                history_store.build_history_store(context,self.path)
        self.assertIn('between 10^',output.getvalue())
        self.assertIn('no automatic truncation',output.getvalue())
        self.assertLess(len(output.getvalue()),600)
        self.assertFalse(self.path.exists())

    def test_counter_passes_ancestry_to_decay_validation(self):
        def modes(pdg, ancestors):
            if len(ancestors) >= 2:
                raise ValueError('depth reached')
            return [((pdg-1,),1.)]
        context = dict(terminal=lambda p:p==0, decay_choices=modes,
                       channels=[dict(id='C1',final_pdgs=(3,))])
        with self.assertRaisesRegex(ValueError,'depth reached'):
            estimate_history_combinations(context)

    def test_progress_reports_candidates_and_distinguishes_filtered_rows(self):
        output = io.StringIO()
        with report_progress(stream=output):
            result = self.rank(fixture((Y,Y,Y))[:2],min_probability=.5,batch_size=1)
        messages = output.getvalue()
        self.assertIn('4 candidate combinations across 1 production channels',messages)
        self.assertIn('min_probability filter',messages)
        self.assertIn('Expanding decays of y',messages)
        self.assertIn('[2/2 choices]',messages)
        self.assertIn('[4/4 combinations; 1 saved]',messages)
        self.assertIn('Indexing 1 saved histories',messages)
        self.assertIn('[2/2 indexes]',messages)
        self.assertEqual(len(result.rows),1)
        self.assertIsNone(result.rows[0]['rate_pb'])

    def test_null_is_not_computed_zero_and_updates_are_idempotent(self):
        data = fixture((Y,Y))
        result = self.rank(data[:2])
        self.assertEqual(result.rows.counts()['calculated'],0)
        self.assertTrue(all(row['rate_pb'] is None for row in result.rows))

        before = result.rows.counts()['revision']
        result.rows.update_rates(statuses={'C1':{'status':'not_calculated'}})
        self.assertEqual(result.rows.counts()['revision'],before)
        rate = dict(id='C1',cross_section_pb=0.,error_pb=0.,target_met=True)
        result.rows.update_rates([rate])
        calculated_revision = result.rows.counts()['revision']
        result.rows.update_rates([rate])
        self.assertEqual(result.rows.counts()['revision'],calculated_revision)
        self.assertTrue(all(row['rate_pb'] == 0 and row['rate_status']=='computed' for row in result.rows))
        result.rows.update_rates(statuses={'C1':dict(status='failed',error='interrupted integration')})
        self.assertEqual(result.rows.production_statuses()['C1']['status'],'failed')
        self.assertTrue(all(row['rate_pb'] is None for row in result.rows))

    def test_public_json_rate_record(self):
        data = fixture((Y,Y))
        result = self.rank((data[0],data[1],vars(data[2])))
        self.assertEqual(result.rows.counts()['calculated'],3)
        self.assertAlmostEqual(sum(row['rate_pb'] for row in result.rows),10.)

    def test_keyset_pages_ties_nulls_and_stale_revision(self):
        data = fixture((Y,Y,Y))
        data[0].channels += (SimpleNamespace(id='C2',process='p p > x x',final_pdgs=(X,X)),)
        result = self.rank(data)
        ids, cursor = [],None
        while True:
            page = result.rows.page(limit=2,cursor=cursor)
            ids.extend(row['history_id'] for row in page['rows'])
            cursor = page['next_cursor']
            if cursor is None:
                break
        self.assertEqual(ids,[row['history_id'] for row in result.rows])
        self.assertEqual(len(ids),len(set(ids)))
        first = result.rows.page(limit=1)
        result.rows.update_rates([dict(id='C2',cross_section_pb=1.,error_pb=.01)])
        with self.assertRaisesRegex(ValueError,'another revision'):
            result.rows.page(limit=1,cursor=first['next_cursor'])
        with self.assertRaises(ValueError):
            result.rows.page(limit=1001)
        with self.assertRaises(ValueError):
            result.rows.page(cursor='bad token')

    def test_ids_do_not_use_rank_channel_labels_or_numerical_parameters(self):
        data = fixture((Y,Y))
        data[0].channels += (SimpleNamespace(id='C2',process='p p > x x',final_pdgs=(X,X)),)
        initial = self.rank(data)
        ids = {row['history_id'] for row in initial.rows}
        # Reorder inventory and swap both labels, using the same SQLite file.
        first,second = data[0].channels
        data[0].channels = (SimpleNamespace(id='C1',process=second.process,final_pdgs=second.final_pdgs),
                           SimpleNamespace(id='C2',process=first.process,final_pdgs=first.final_pdgs))
        data[2].rates[0]['id'] = 'C2'
        renamed = self.rank(data)
        self.assertEqual(ids,{row['history_id'] for row in renamed.rows})
        data[1].particles[Y]['mass_gev'] = 500.
        data[2].rates[0]['cross_section_pb'] = 20.
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            self.rank(data)
        other = rank_final_states(*data,stable_particles=(X,),show=False)
        self.assertEqual(ids,{row['history_id'] for row in other.rows})

    def test_interrupt_rolls_back_only_uncommitted_batch_and_resumes(self):
        original = history_store._revision
        def interrupt(db):
            if db.execute('SELECT COUNT(*) FROM histories').fetchone()[0] == 4:
                raise KeyboardInterrupt
            return original(db)
        output = io.StringIO()
        with patch.object(history_store,'_revision',side_effect=interrupt), report_progress(stream=output):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.rank(batch_size=2)
        partial = raised.exception.partial_ranking
        self.assertEqual(len(partial.rows),2)
        self.assertFalse(partial.rows.counts()['discovery_complete'])
        self.assertIsNone(partial.omitted_probability['C1'])
        self.assertIn('interrupted (',output.getvalue())
        self.assertIn('[2/4 combinations; 2 saved]',output.getvalue())
        prefix = {r['history_id'] for r in partial.rows}
        output = io.StringIO()
        with report_progress(stream=output):
            resumed = self.rank(batch_size=2)
        self.assertIn('... [2/4 combinations]',output.getvalue())
        self.assertIn('[4/4 combinations; 4 saved]',output.getvalue())
        self.assertEqual(len(resumed.rows),4)
        self.assertTrue(resumed.rows.counts()['discovery_complete'])
        self.assertTrue(prefix.issubset({r['history_id'] for r in resumed.rows}))
        self.assertAlmostEqual(resumed.coverage['C1'],1.)

    def test_disk_full_retains_committed_cursor_and_can_resume(self):
        original = history_store._revision
        def disk_full(db):
            if db.execute('SELECT COUNT(*) FROM histories').fetchone()[0] == 4:
                raise OSError(errno.ENOSPC,'simulated full disk')
            return original(db)
        with patch.object(history_store,'_revision',side_effect=disk_full):
            with self.assertRaises(OSError) as raised:
                self.rank(batch_size=2)
        self.assertEqual(raised.exception.category,'resource_limit')
        self.assertEqual(len(raised.exception.partial_ranking.rows),2)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT row_count,next_ordinal FROM productions').fetchone(),(2,2))
        self.assertEqual(len(self.rank(batch_size=2).rows),4)

    def test_interrupted_postprocessing_is_not_complete_discovery(self):
        original = history_store._put
        def interrupt(db,key,value):
            if key == 'discovery_complete' and value is True:
                raise KeyboardInterrupt
            return original(db,key,value)
        with patch.object(history_store,'_put',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.rank(batch_size=2)
        self.assertEqual(len(raised.exception.partial_ranking.rows),4)
        self.assertFalse(raised.exception.partial_ranking.rows.counts()['discovery_complete'])
        resumed = self.rank(batch_size=2)
        self.assertTrue(resumed.rows.counts()['discovery_complete'])
        self.assertEqual(len(resumed.rows),4)

    def test_small_render_budget_preserves_full_dag(self):
        result = self.rank(max_render_nodes=1)
        row = result.rows[0]
        self.assertTrue(row['render_truncated'])
        self.assertIsNone(row['history_key'])
        self.assertTrue(row['history_ref'])
        nodes = {n['node_id']:n for n in result.rows.iter_nodes()}
        self.assertTrue(all(root in nodes for root in row['history_ref']))
        self.assertEqual(len(result.rows),4)

    def test_render_change_invalidates_views_without_changing_histories(self):
        first = self.rank(max_render_nodes=1)
        old = first.rows.reference()
        ids = {r['history_id'] for r in first.rows}
        new = self.rank(max_render_nodes=100)
        self.assertEqual(ids,{r['history_id'] for r in new.rows})
        self.assertNotEqual(old['revision'],new.rows.reference()['revision'])
        self.assertTrue(all(not r['render_truncated'] for r in new.rows))
        self.assertTrue(all(not r['render_truncated'] for r in first.rows))
        with self.assertRaisesRegex(ValueError,'revision'):
            SQLiteHistoryRows.from_reference(old)

    def test_width_values_invalidate_unverified_library_store(self):
        data = fixture()
        del data[1].width_digest
        del data[2].width_digest
        self.rank(data)
        data[1].total_widths[Y] = 100.
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            self.rank(data)

    def test_checkpoint_revision_and_fingerprint_are_verified(self):
        result = self.rank()
        reference = result.rows.reference()
        self.assertEqual(len(SQLiteHistoryRows.from_reference(reference)),4)
        result.rows.update_rates(statuses={'C1':{'status':'running'}})
        with self.assertRaisesRegex(ValueError,'revision'):
            SQLiteHistoryRows.from_reference(reference)
        reference['fingerprint'] = 'different'
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            SQLiteHistoryRows.from_reference(reference)

    def test_unchanged_rebuild_does_not_touch_database(self):
        first = self.rank().rows.reference()
        second = self.rank().rows.reference()
        self.assertEqual(first,second)

    def test_external_edits_invalidate_reference_and_cursor(self):
        rows = self.rank().rows
        reference = rows.reference()
        cursor = rows.page(limit=1)['next_cursor']
        with sqlite3.connect(rows.path) as db:
            db.execute('UPDATE histories SET rate=999')
        self.assertEqual(rows.counts()['revision'],reference['revision'])
        with self.assertRaisesRegex(ValueError,'outside its recorded revision'):
            SQLiteHistoryRows.from_reference(reference)
        with self.assertRaisesRegex(ValueError,'another revision'):
            rows.page(limit=1,cursor=cursor)


if __name__ == '__main__':
    unittest.main()
