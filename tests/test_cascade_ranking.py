"""Probability and physics checks independent of an installed generator."""
from types import SimpleNamespace
import unittest

from cascade_ranking import CascadeError, rank_final_states


X, Y = 9000001, 9000002


def fixture(produced=(Y,), modes=None, widths=None):
    records = [(X, 'x', X, 100., True, 1, 1, 0),
               (Y, 'y', Y, 400., True, 1, 1, 0),
               (25, 'h', 25, 125., False, 1, 1, 0),
               (23, 'z', 23, 91., False, 3, 1, 0),
               (5, 'b', -5, 4.7, False, 2, 3, -1/3),
               (-5, 'b~', 5, 4.7, False, 2, 3, 1/3),
               (11, 'e-', -11, .0005, False, 2, 1, -1),
               (-11, 'e+', 11, .0005, False, 2, 1, 1),
               (15, 'ta-', -15, 1.777, False, 2, 1, -1),
               (-15, 'ta+', 15, 1.777, False, 2, 1, 1),
               (22, 'a', 22, 0., False, 3, 1, 0)]
    particles = {p: dict(name=n, anti_pdg=a, mass_gev=m, is_new=new, spin=s, color=c, charge=q)
                 for p, n, a, m, new, s, c, q in records}
    mode_rows = modes if modes is not None else [(Y, (X, 25), 1.), (25, (5, -5), .8), (25, (15, -15), .2)]
    decays = SimpleNamespace(modes=tuple(dict(pdg=p, daughters=d, branching_ratio=b) for p, d, b in mode_rows),
                             total_widths=widths or {X: 0., Y: 1., 25: .004, 23: 2.5}, particles=particles,
                             model_digest='model', benchmark_digest='point', width_digest='width',
                             approximation='Tree-level, maximum four primary daughters.')
    channel = SimpleNamespace(id='C1', process='p p > ' + ' '.join(particles[p]['name'] for p in produced),
                              final_pdgs=produced)
    channels = SimpleNamespace(channels=(channel,), particles=particles, model_digest='model', benchmark_digest='point')
    rates = SimpleNamespace(rates=(dict(id='C1', final_pdgs=produced, cross_section_pb=10., error_pb=.2, target_met=True),),
                            particles=particles, model_digest='model', benchmark_digest='point', width_digest='width')
    return channels, decays, rates


def rank(data, **kwargs):
    return rank_final_states(*data, stable_particles=(X,), show=False, **kwargs)


class CascadeRankingTests(unittest.TestCase):
    def test_scale_comparison_notice_preserves_default_and_topology_ids(self):
        from madgraph_runner import DEFAULT_COLLIDER
        data = fixture()
        default = rank(data)
        self.assertEqual(DEFAULT_COLLIDER['dynamical_scale_choice'], 3)
        self.assertTrue(any('Equal dynamical_scale_choice integers do not ensure' in n
                            for n in default.notices))
        self.assertTrue(any('not solely a resonance-window effect' in n for n in default.notices))
        data[0].collider = {'dynamical_scale_choice': 4}
        common = rank(data)
        self.assertTrue(any('same function of incoming parton momenta' in n for n in common.notices))
        self.assertEqual([r['history_id'] for r in default.rows],
                         [r['history_id'] for r in common.rows])
        self.assertEqual([r['rate_pb'] for r in default.rows],
                         [r['rate_pb'] for r in common.rows])

    def test_stable_x_and_heavy_sm_cascade(self):
        result = rank(fixture())
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0]['final_pdgs'], (-5, 5, X))
        self.assertAlmostEqual(result.rows[0]['rate_pb'], 8.)
        self.assertAlmostEqual(result.rows[0]['error_pb'], .16)
        self.assertIn('h →', result.rows[0]['history'])
        self.assertAlmostEqual(result.coverage['C1'], 1.)
        self.assertAlmostEqual(result.omitted_probability['C1'], 0.)

    def test_identical_parents_have_multinomial_not_duplicates(self):
        result = rank(fixture((Y, Y)))
        self.assertEqual(len(result.rows), 3)
        self.assertEqual([round(row['branching_ratio'], 8) for row in result.rows], [.64, .32, .04])
        self.assertAlmostEqual(result.coverage['C1'], 1.)
        mixed = next(row for row in result.rows if -5 in row['final_pdgs'] and -15 in row['final_pdgs'])
        self.assertAlmostEqual(mixed['rate_pb'], 10 * 2 * .8 * .2)

    def test_identical_daughters_have_multinomial_factor(self):
        data = fixture(modes=[(Y, (25, 25), 1.), (25, (5, -5), .8), (25, (15, -15), .2)])
        result = rank(data)
        self.assertEqual(len(result.rows), 3)
        self.assertAlmostEqual(result.rows[1]['branching_ratio'], .32)

    def test_three_identical_parents_and_conjugate_heavy_sm_decays(self):
        result = rank(fixture((Y, Y, Y)))
        self.assertEqual(len(result.rows), 4)
        self.assertAlmostEqual(result.coverage['C1'], 1.)
        weights = sorted(row['branching_ratio'] for row in result.rows)
        for actual, expected in zip(weights, sorted((.8 ** 3, 3 * .8 ** 2 * .2, 3 * .8 * .2 ** 2, .2 ** 3))):
            self.assertAlmostEqual(actual, expected)
        data = fixture(modes=[(Y, (X, 6, -6), 1.), (6, (5, 24), 1.), (24, (-11, 12), 1.)])
        particles = data[1].particles
        for pdg, label, anti, mass, spin, color, charge in (
                (6, 't', -6, 172.5, 2, 3, 2/3), (-6, 't~', 6, 172.5, 2, 3, -2/3),
                (24, 'w+', -24, 80.4, 3, 1, 1), (-24, 'w-', 24, 80.4, 3, 1, -1),
                (12, 've', -12, 0., 2, 1, 0), (-12, 've~', 12, 0., 2, 1, 0)):
            particles[pdg] = dict(name=label, anti_pdg=anti, mass_gev=mass, spin=spin, color=color, charge=charge, is_new=False)
        particles[Y]['mass_gev'] = 600.
        data[1].total_widths.update({6: 1.4, 24: 2.085})
        result = rank(data)
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]['final_pdgs'], (-12, -11, -5, 5, 11, 12, X))
        self.assertIn('w-', result.rows[0]['history'])

    def test_three_body_mode_and_different_histories_stay_separate(self):
        data = fixture(modes=[(Y, (X, 23), .7), (Y, (X, 11, -11), .3), (23, (11, -11), 1.)])
        result = rank(data)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0]['final_pdgs'], result.rows[1]['final_pdgs'])
        self.assertNotEqual(result.rows[0]['history_key'], result.rows[1]['history_key'])
        self.assertAlmostEqual(result.rows[0]['rate_pb'], 7.)

    def test_missing_and_zero_width_do_not_imply_stability(self):
        data = fixture()
        with self.assertRaisesRegex(CascadeError, 'Zero calculated width.*does not establish stability'):
            rank_final_states(*data, show=False)
        del data[1].total_widths[X]
        with self.assertRaisesRegex(CascadeError, 'No calculated total width'):
            rank_final_states(*data, show=False)
        data[1].total_widths[X] = .01
        with self.assertRaisesRegex(CascadeError, 'No usable primary decay modes'):
            rank_final_states(*data, show=False)

    def test_stable_declaration_cannot_override_nonzero_decay(self):
        data = fixture()
        data[1].total_widths[X] = .1
        with self.assertRaisesRegex(CascadeError, 'declared stable.*nonzero'):
            rank(data)

    def test_truncated_tables_and_pruning_do_not_renormalize(self):
        data = fixture(modes=[(Y, (X, 25), .9), (25, (5, -5), .8), (25, (15, -15), .1)])
        complete = rank(data)
        self.assertAlmostEqual(complete.coverage['C1'], .81)
        self.assertAlmostEqual(complete.omitted_probability['C1'], .19)
        pruned = rank(data, min_probability=.1)
        self.assertEqual(len(pruned.rows), 1)
        self.assertAlmostEqual(pruned.rows[0]['branching_ratio'], .72)
        self.assertAlmostEqual(pruned.omitted_probability['C1'], .28)

    def test_relative_cutoff_filters_display_only(self):
        data = fixture(modes=[(Y, (X, 25), 1.), (25, (5, -5), .99999), (25, (15, -15), .00001)])
        result = rank(data)
        self.assertEqual(len(result.visible_rows()), 1)
        self.assertEqual(len(result.visible_rows(show_all=True)), 2)
        self.assertEqual(len(result.rows), 2)
        self.assertAlmostEqual(result.coverage['C1'], 1.)

    def test_broad_threshold_and_overlap_flags(self):
        data = fixture()
        data[1].particles[Y]['mass_gev'] = 230.
        data[1].total_widths[Y] = 10.
        result = rank(data)
        self.assertTrue(any('Γ/m' in flag for flag in result.rows[0]['flags']))
        self.assertTrue(any('near decay threshold' in flag for flag in result.rows[0]['flags']))
        data = fixture(modes=[(Y, (X, 22), 1.)])
        data[1].particles[Y]['mass_gev'] = 129.
        result = rank(data)
        self.assertTrue(any('overlapping resonances' in flag for flag in result.rows[0]['flags']))

    def test_production_cascade_overlap_is_detected_without_summing(self):
        data = fixture()
        other = SimpleNamespace(id='C2', process='p p > x h', final_pdgs=(X, 25))
        data[0].channels += (other,)
        data[2].rates += (dict(id='C2', final_pdgs=(X, 25), cross_section_pb=3., error_pb=.1),)
        result = rank(data)
        self.assertEqual(len(result.rows), 4)
        parent_rows = [row for row in result.rows if row['production_id'] == 'C1']
        self.assertTrue(all(any('production overlap: C2' in flag for flag in row['flags']) for row in parent_rows))
        daughter_rows = [row for row in result.rows if row['production_id'] == 'C2']
        self.assertTrue(all(any('production overlap: C1' in flag for flag in row['flags']) for row in daughter_rows))
        self.assertEqual(set(result.coverage), {'C1', 'C2'})

    def test_stale_benchmark_and_widths_rejected(self):
        data = fixture()
        data[2].benchmark_digest = 'other'
        with self.assertRaisesRegex(CascadeError, 'benchmark_digest differs'):
            rank(data)
        data[2].benchmark_digest = 'point'
        data[2].width_digest = 'other'
        with self.assertRaisesRegex(CascadeError, 'different decay widths'):
            rank(data)

    def test_depth_and_resource_limits_are_explicit_errors(self):
        with self.assertRaisesRegex(CascadeError, 'max_depth'):
            rank(fixture(), max_depth=1)
        with self.assertWarnsRegex(DeprecationWarning, 'max_histories'):
            result = rank(fixture((Y, Y)), max_histories=2)
        self.assertEqual(len(result.rows), 3)

    def test_closed_and_duplicate_primary_modes_rejected(self):
        data = fixture()
        data[1].particles[Y]['mass_gev'] = 150.
        with self.assertRaisesRegex(CascadeError, 'Kinematically closed'):
            rank(data)
        data = fixture(modes=[(Y, (X, 25), .4), (Y, (25, X), .6)])
        with self.assertRaisesRegex(CascadeError, 'Duplicate primary decay'):
            rank(data)

    def test_missing_rates_are_reported_as_uncomputed(self):
        data = fixture()
        data[0].channels += (SimpleNamespace(id='C2', process='p p > x x', final_pdgs=(X, X)),)
        result = rank(data)
        self.assertEqual(result.uncomputed_production_ids, ('C2',))
        self.assertEqual(set(result.coverage), {'C1', 'C2'})
        missing = next(row for row in result.rows if row['production_id'] == 'C2')
        self.assertIsNone(missing['rate_pb'])
        self.assertIsNone(missing['error_pb'])
        self.assertIsNone(missing['production_cross_section_pb'])
        self.assertEqual(missing['rate_status'], 'not_calculated')


if __name__ == '__main__':
    unittest.main()
