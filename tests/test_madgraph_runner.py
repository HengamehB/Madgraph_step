"""Fast physics-facing tests; integration tests require the bundled runtime."""
import tempfile
import unittest
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io
from pathlib import Path

from madgraph_runner import (_candidate_finals, _tree_signatures, _read_decays, _collider,
    _card_digest, _full_card_digest, _digest, _artifact, _copy_ignore, _reported_decays, _retain_small_colored_widths, MadGraphError, validate_provenance, ProductionChannels, DecayResults, CrossSectionResults)


class TreeEnumerationTests(unittest.TestCase):
    def test_new_particles_only_keeps_five_channels_and_internal_sm_exchange(self):
        # All proton couplings enter through an internal SM Higgs. Filtering
        # vertices instead of outgoing finals would wrongly remove the pairs.
        anti = {5: -5, -5: 5, 25: 25, 900: 900, 901: 901}
        vertices = [(-5, 5, 25), (-5, 5, 900), (-5, 5, 901),
                    (25, 900, 900), (25, 901, 901), (25, 900, 901)]
        expected = [(900,), (901,), (900, 900), (900, 901), (901, 901)]
        scoped = _candidate_finals(vertices, anti, [5, -5], {900, 901}, 2,
                                   new_particles_only=True)
        self.assertEqual(scoped, expected)
        all_finals = _candidate_finals(vertices, anti, [5, -5], {900, 901}, 2)
        self.assertIn((25, 900), all_finals)
        self.assertEqual(all_finals, _candidate_finals(vertices, anti, [5, -5], {900, 901}, 2,
                                                     new_particles_only=False))

    def test_new_particles_only_includes_antiparticles(self):
        anti = {1: -1, -1: 1, 25: 25, 900: -900, -900: 900}
        vertices = [(-1, 1, 25), (25, -900, 900), (-1, 1, 900), (-1, 1, -900)]
        finals = _candidate_finals(vertices, anti, [1, -1], {900}, 2,
                                   new_particles_only=True)
        self.assertIn((-900,), finals)
        self.assertIn((-900, 900), finals)
        self.assertTrue(all(set(final) <= {-900, 900} for final in finals))

    def test_new_particles_only_disconnected_sector_has_no_channels(self):
        anti = {5: -5, -5: 5, 25: 25, 900: 900, 901: 901}
        vertices = [(-5, 5, 25), (900, 900, 901)]
        self.assertEqual(_candidate_finals(vertices, anti, [5, -5], {900, 901}, 2,
                                          new_particles_only=True), [])

    def test_external_artifact_mutation_during_decays_is_an_execution_failure(self):
        from madgraph_runner import compute_decays, ArtifactChangedError
        from workflow_errors import error_category
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            particles = root / 'particles.py'
            particles.write_text('# original UFO\n')
            def changed_during_job(*args):
                particles.write_text('# externally changed UFO\n')
                return {}, root
            with patch('madgraph_runner._run_job', side_effect=changed_during_job):
                with self.assertRaisesRegex(ArtifactChangedError, 'changed during decay') as caught:
                    compute_decays(root, new_particles=['x'], show=False)
            self.assertEqual(error_category(caught.exception), 'execution_failure')

    def test_conjugate_crossing_and_three_vertex_tree(self):
        # b bbar h, h X X and h Z Z produce XX, XXh, XXZ (the latter
        # only when a Z-fermion vertex is present). Keep connected trees only.
        anti = {5: -5, -5: 5, 25: 25, 23: 23, 900: 900}
        vertices = [(-5, 5, 25), (25, 900, 900), (25, 23, 23)]
        finals = _candidate_finals(vertices, anti, [5, -5], {900}, 3)
        self.assertIn((900, 900), finals)
        self.assertIn((25, 900, 900), finals)
        self.assertNotIn((23, 900, 900), finals)
        self.assertEqual(len(finals), len(set(finals)))

    def test_antiparticle_initial_state_and_quartic(self):
        anti = {1: -1, -1: 1, 11: -11, -11: 11, 900: -900, -900: 900}
        finals = _candidate_finals([(1, -11, 900), (-1, 11, -900)], anti, [1, -1], {900, -900}, 2)
        self.assertEqual(finals, [(-900, 900)])

    def test_self_conjugate_gluing_removes_two_copies(self):
        signatures = _tree_signatures([(1, 1, 1)], {1: 1})
        self.assertEqual(signatures, {(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1)})

    def test_no_loop_external_states(self):
        # Joining two hXX vertices once has four external legs; joining twice
        # would be a loop, which the tree builder must never introduce.
        signatures = _tree_signatures([(25, 900, 900)], {25: 25, 900: 900})
        self.assertNotIn((25, 25), signatures)

    def test_contact_five_point(self):
        anti = {1: -1, -1: 1, 900: 900}
        self.assertIn((900, 900, 900), _candidate_finals([(-1, 1, 900, 900, 900)], anti, [1, -1], {900}))

    def test_configured_four_particle_production_and_resource_bound(self):
        from madgraph_runner import ProductionResourceLimit
        anti = {1: -1, -1: 1, 900: 900}
        contact = [(-1, 1, 900, 900, 900, 900)]
        self.assertEqual(_candidate_finals(contact, anti, [1, -1], {900}, 3), [])
        self.assertIn((900, 900, 900, 900), _candidate_finals(contact, anti, [1, -1], {900}, 4))
        self.assertEqual(_collider({'max_final': 4})['max_final'], 4)
        with self.assertRaisesRegex(ProductionResourceLimit, 'max_tree_signatures=1'):
            _tree_signatures([(1, 1, 1)], {1: 1}, 6, max_signatures=1)


class CardTests(unittest.TestCase):
    def test_scratch_installation_omits_foreign_build_products(self):
        from madgraph_runner import _source_copy_ignore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in {'makefile': b'all: build\n', 'matrix.f': b'      END\n',
                                  'matrix.o': b'object', 'libhelas.a': b'archive',
                                  'gensym': b'\x7fELFcompiled', 'script': b'#!/bin/sh\nexit 0\n'}.items():
                (root / name).write_bytes(content)
            skipped = _source_copy_ignore(root, [p.name for p in root.iterdir()])
            self.assertEqual(skipped, {'matrix.o', 'libhelas.a', 'gensym'})

    def test_source_copy_keeps_valid_relative_links(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            (source / 'target.inc').write_text('source')
            (source / 'valid.inc').symlink_to('target.inc')
            (source / 'optional.inc').symlink_to('not_installed.inc')
            destination = Path(directory) / 'copy'
            shutil.copytree(source, destination, ignore=_copy_ignore)
            self.assertEqual((destination / 'valid.inc').read_text(), 'source')
            self.assertFalse((destination / 'optional.inc').exists())

    def test_slha_widths_are_not_branching_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            card = Path(directory) / 'param_card.dat'
            card.write_text('BLOCK MASS\n 900 600\nDECAY 900 2.0D+00\n 2.5D-01 2 25 25 # hh\n 7.5e-1 3 1 -1 25\nBLOCK OTHER\n 1 2\n')
            result = _read_decays(card)
        self.assertEqual(result[900]['total'], 2.0)
        self.assertEqual(result[900]['channels'], [((25, 25), .5), ((1, -1, 25), 1.5)])

    def test_unknown_collider_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            _collider({'enrgy_tev': 14})

    def test_nonbundled_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _collider({'pdf': 'lhapdf'})

    def test_collider_type_errors_are_diagnostic_before_jobs(self):
        for values in ({'max_final': True}, {'max_final': 2.0}, {'energy_tev': '13.6'},
                       {'accuracy': float('nan')}, {'ptj': True}, {'dynamical_scale_choice': 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _collider(values)

    def test_new_particles_only_requires_actual_python_boolean(self):
        self.assertIs(_collider()['new_particles_only'], False)
        self.assertIs(_collider({'new_particles_only': True})['new_particles_only'], True)
        for value in ('true', 'False', 0, 1, None, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'new_particles_only must be True or False'):
                _collider({'new_particles_only': value})

    def test_default_is_half_ht_and_no_photon_parton(self):
        settings = _collider()
        self.assertEqual(settings['dynamical_scale_choice'], 3)
        self.assertNotIn(22, settings['parton_pdgs'])
        self.assertEqual(settings['seed'], 1)

    def test_benchmark_and_width_fingerprints_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            card = Path(directory) / 'card.dat'
            card.write_text('BLOCK MASS\n 900 6.0D2 # example\nDECAY 900 1.0\n 1.0 2 25 25\n')
            benchmark, width = _card_digest(card), _card_digest(card, widths=True)
            card.write_text('block mass\n 900 600.000\nDECAY 900 2.0\n 1.0 2 25 25\n')
            self.assertEqual(_card_digest(card), benchmark)
            self.assertNotEqual(_card_digest(card, widths=True), width)
            card.write_text('BLOCK MASS\n 900 601\nDECAY 900 1\n 1 2 25 25\n')
            self.assertNotEqual(_card_digest(card), benchmark)

    def test_provenance_rejects_edited_benchmark_and_decay_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'particles.py').write_text('# fixture UFO')
            card = root / 'param_card.dat'
            text = 'BLOCK MASS\n 900 600\nDECAY 900 1\n 1 2 25 25\n'
            card.write_text(text)
            benchmark, width, model = _card_digest(card), _card_digest(card, widths=True), _digest(root)
            channels = ProductionChannels((), root, ('x',), _collider(), card, model, root, benchmark_digest=benchmark)
            decays = DecayResults((), {900: 1}, card, root, model, benchmark_digest=benchmark, width_digest=width)
            rates = CrossSectionResults((), dict(channels.collider), model_digest=model, benchmark_digest=benchmark, width_digest=width)
            self.assertTrue(validate_provenance(channels, decays, rates))
            card.write_text(text.replace('DECAY 900 1', 'DECAY 900 2'))
            with self.assertRaisesRegex(ValueError, 'decay parameter card'):
                validate_provenance(channels, decays, rates)
            card.write_text(text.replace('900 600', '900 601'))
            with self.assertRaisesRegex(ValueError, 'channel parameter card'):
                validate_provenance(channels, decays, rates)

    def test_export_fingerprint_rejects_edits_before_first_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            particles = root / 'particles.py'
            particles.write_text('# original UFO')
            card = root / 'param_card.dat'
            original = 'BLOCK MASS\n 900 600\nDECAY 900 1\n 1 2 25 25\n'
            card.write_text(original)
            model = SimpleNamespace(path=root, new_particles=('x',), parameter_card=card,
                                    parameter_values={}, benchmark_only=True,
                                    saved_model_digest=_digest(root), saved_parameter_card_digest=_full_card_digest(card))
            self.assertEqual(_artifact(model)[2], card.resolve())
            model.parameter_values['MX'] = 1000.0
            with self.assertRaisesRegex(ValueError, 'cannot override parameter_values'):
                _artifact(model)
            model.parameter_values.clear()
            for change in (original.replace('900 600', '900 601'), original.replace('DECAY 900 1', 'DECAY 900 2')):
                card.write_text(change)
                with self.assertRaisesRegex(ValueError, 'changed after export'):
                    _artifact(model)
            card.write_text(original)
            particles.write_text('# changed UFO')
            with self.assertRaisesRegex(ValueError, 'changed after export'):
                _artifact(model, allow_unsealed=True)

    def test_unsealed_benchmark_allowed_only_for_initial_export_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'particles.py').write_text('# original UFO')
            card = root / 'param_card.dat'
            card.write_text('BLOCK MASS\n 900 600\nDECAY 900 0\n')
            model = SimpleNamespace(path=root, new_particles=('x',), parameter_card=card,
                                    parameter_values={}, benchmark_only=True)
            self.assertEqual(_artifact(model, allow_unsealed=True)[2], card.resolve())
            with self.assertRaisesRegex(ValueError, 'saved fingerprint'):
                _artifact(model)

    def test_source_width_only_edit_invalidates_channel_and_decay_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'particles.py').write_text('# fixture UFO')
            stored, source = root / 'stored.dat', root / 'source.dat'
            card_text = 'BLOCK MASS\n 900 600\nDECAY 900 1\n 1 2 25 25\n'
            stored.write_text(card_text)
            source.write_text(card_text)
            benchmark, width, model = _card_digest(stored), _card_digest(stored, widths=True), _digest(root)
            channels = ProductionChannels((), root, ('x',), _collider(), stored, model, root,
                                          benchmark_digest=benchmark, source_parameter_card=source,
                                          source_benchmark_digest=benchmark, source_width_digest=width)
            self.assertTrue(validate_provenance(channels))
            source.write_text(card_text.replace('DECAY 900 1', 'DECAY 900 2'))
            self.assertEqual(_card_digest(source), benchmark)
            with self.assertRaisesRegex(ValueError, 'source widths changed'):
                validate_provenance(channels)
            channels.source_parameter_card = None
            decays = DecayResults((), {900: 1.}, stored, root, model, benchmark_digest=benchmark,
                                 width_digest=width, source_parameter_card=source,
                                 source_benchmark_digest=benchmark, source_width_digest=width)
            with self.assertRaisesRegex(ValueError, 'source widths changed'):
                validate_provenance(channels, decays)

    def test_fingerprints_do_not_hide_missing_decay_tails(self):
        with tempfile.TemporaryDirectory() as directory:
            card = Path(directory) / 'card.dat'
            card.write_text('DECAY 900 2.0\n .4 2 25 25\n .5 3 1 -1 25\n')
            data = _read_decays(card)[900]
        self.assertAlmostEqual(sum(width for _, width in data['channels']) / data['total'], .9)


class DecayCoverageTests(unittest.TestCase):
    def test_total_width_summary_includes_offshell_width_without_two_body_modes(self):
        result = DecayResults((), {6: 1.4, -6: 1.4, 25: .004, 900: .002}, Path('card'), Path('.'), 'model',
                              particles={6: {'name': 't', 'anti_pdg': -6}, -6: {'name': 'tbar', 'anti_pdg': 6},
                                         25: {'name': 'h', 'anti_pdg': 25}, 900: {'name': 'y', 'anti_pdg': 900}},
                              omitted_probability={900: .05}, coverage_notices=('Unknown higher-body tail.',))
        rows = result.total_width_rows()
        self.assertEqual([row['name'] for row in rows], ['t', 'h', 'y'])
        self.assertEqual(rows[-1]['width_gev'], .002)
        self.assertEqual(rows[-1]['omitted_probability'], .05)
        output = io.StringIO()
        with patch('madgraph_runner._table') as table, contextlib.redirect_stdout(output):
            result.show()
        self.assertEqual(table.call_count, 1)
        self.assertIn(('y', '0.002', '0.05'), table.call_args.args[1])
        self.assertIn('Unknown higher-body tail.', output.getvalue())


    def test_reporting_cut_preserves_total_and_does_not_renormalize(self):
        data = {900: {'total': 2.0, 'channels': [((1, -1), 1.8), ((25, 25), .2)]}}
        names = {900: 'x', -900: 'xbar', 1: 'd', -1: 'dbar', 25: 'h'}
        anti = {900: -900, -900: 900, 1: -1, -1: 1, 25: 25}
        result = _reported_decays(data, {900}, {900}, names, anti, {900: {'color': 1}}, .15)
        self.assertEqual(result['total_widths'], {900: 2.0, -900: 2.0})
        self.assertEqual(result['omitted_probability'], {900: .1, -900: .1})
        self.assertEqual([r['branching_ratio'] for r in result['modes']], [.9, .9])
        self.assertEqual(result['modes'][1]['daughters'], (-1, 1))
        self.assertEqual(data[900]['channels'], [((1, -1), 1.8), ((25, 25), .2)])

    def test_small_colored_width_is_retained_and_flagged(self):
        result = _reported_decays({900: {'total': .05, 'channels': [((1, 25), .05)]}},
                                 {900}, {900}, {900: 'x', 1: 'd', 25: 'h'},
                                 {900: 900, 1: 1, 25: 25}, {900: {'color': 3}}, 1e-8)
        self.assertEqual(result['modes'][0]['width_gev'], .05)
        self.assertEqual(result['modes'][0]['branching_ratio'], 1.)
        self.assertIn('hadronization', result['coverage_notices'][0])

    def test_unknown_adaptive_tail_is_distinct_from_reported_cut(self):
        result = _reported_decays({900: {'total': 2., 'channels': [((25, 25), 1.8)]}},
                                 {900}, {900}, {900: 'x', 25: 'h'},
                                 {900: 900, 25: 25}, {900: {'color': 1}}, 1e-8)
        self.assertEqual(result['omitted_probability'][900], 0.)
        self.assertEqual(result['modes'][0]['branching_ratio'], .9)
        self.assertIn('unknown', result['coverage_notices'][-1])

    def test_scratch_compatibility_patch_retains_both_colored_contributions(self):
        source_text = """def analytic(value, particle, decay_to, logger):
    if 0 < value < 0.1 and particle['color'] !=1:
        logger.warning("partial width of particle %s lower than QCD scale:%s. Set it to zero. (%s)" % (particle.get('name'), value, decay_to))
        value = 0
    return value

def numerical(width, particle, entries, logger):
    result = []
    for BR in entries:
        if 0 < BR.value * width <0.1 and particle['color'] !=1:
            logger.warning("partial width of particle %s lower than QCD scale:%s. Set it to zero. (%s)" % (particle.get('name'), BR.value * width, BR.lhacode[1:]))
            continue
        result.append(BR.value * width)
    return result
"""
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory)
            source = engine / 'madgraph' / 'interface' / 'madgraph_interface.py'
            source.parent.mkdir(parents=True)
            source.write_text(source_text)
            _retain_small_colored_widths(engine)
            namespace = {}
            exec(compile(source.read_text(), str(source), 'exec'), namespace)
            log = SimpleNamespace(warning=lambda *args: None)
            particle = {'color': 3, 'name': 'x'}
            self.assertEqual(namespace['analytic'](.05, particle, [1, 25], log), .05)
            self.assertEqual(namespace['numerical'](.5, particle, [SimpleNamespace(value=.1, lhacode=(2, 1, 25))], log), [.05])
            with self.assertRaisesRegex(MadGraphError, 'compatibility review'):
                _retain_small_colored_widths(engine)


if __name__ == '__main__':
    unittest.main()
