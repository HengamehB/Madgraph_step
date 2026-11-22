"""Selected-history identity, syntax and exact primary-topology regression tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from selected_decay import (canonical_forest, render_history, primary_keys,
                            diagram_definition, PrimaryDiagramError,
                            integrate_selected_history, capture_primary_diagrams,
                            restrict_primary_amplitudes)


def leg(pdg, number):
    return dict(id=pdg, number=number)


class SelectedDecayTests(unittest.TestCase):
    def test_default_scale_is_preserved_and_external_momenta_caveat_is_reported(self):
        from madgraph_runner import _collider, scale_diagnostics
        collider = _collider()
        self.assertEqual(collider['dynamical_scale_choice'], 3)
        diagnostic = scale_diagnostics(collider, has_decay_chain=True)
        self.assertFalse(diagnostic['common_incoming_prescription'])
        self.assertIn('do not ensure equal numerical', diagnostic['note'])
        self.assertIn('PDFs', diagnostic['note'])
        # Resolving a parent at rest into two daughters along the beam keeps
        # total incoming energy fixed but changes sum(mT)/2 from m/2 to zero.
        parent_energy, parent_pz = 100., 0.
        daughter_momenta = [(50., 50.), (50., -50.)]
        production_scale = (parent_energy**2-parent_pz**2)**.5 / 2
        decay_scale = sum((energy**2-pz**2)**.5 for energy,pz in daughter_momenta) / 2
        self.assertNotEqual(production_scale, decay_scale)

    def test_explicit_incoming_scale_choice_is_distinguished_from_rate_equality(self):
        from madgraph_runner import _collider, scale_diagnostics
        diagnostic = scale_diagnostics(_collider({'dynamical_scale_choice': 4}), has_decay_chain=True)
        self.assertTrue(diagnostic['common_incoming_prescription'])
        self.assertIn('incoming parton momenta', diagnostic['definition'])
        self.assertIn('does not imply', diagnostic['note'])

    def test_identical_mixed_parents_have_two_explicit_clauses(self):
        forest = [(9002, [(11, []), (-11, [])]), (9002, [(13, []), (-13, [])])]
        names = {9002: 'y', 11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+'}
        result = render_history(forest, names)
        self.assertEqual(result, 'p p > y y, (y > mu+ mu-), (y > e+ e-)')
        self.assertEqual(result, render_history(list(reversed(forest)), names))
        self.assertEqual(len(primary_keys(canonical_forest(forest))), 2)

    def test_nested_occurrences_keep_their_own_subdecays(self):
        names = {9002: 'y', 9001: 'x', 25: 'h', 5: 'b', -5: 'b~', 13: 'mu-', -13: 'mu+'}
        forest = [(9002, [(25, [(5, []), (-5, [])]), (9001, [])]),
                  (9002, [(25, [(13, []), (-13, [])]), (9001, [])])]
        text = render_history(forest, names, 'QED<=8 NP<=8')
        self.assertEqual(text.count('y > h x QED<=8 NP<=8'), 2)
        self.assertIn('(y > h x QED<=8 NP<=8, (h > b~ b QED<=8 NP<=8))', text)
        self.assertIn('(y > h x QED<=8 NP<=8, (h > mu+ mu- QED<=8 NP<=8))', text)

    def test_stable_roots_have_no_decay_clause(self):
        self.assertEqual(render_history([(9001, []), (9001, [])], {9001: 'x'}), 'p p > x x')

    def test_current_number_reuse_and_artificial_identity_match(self):
        # The internal current reuses daughter number 2; a global edge map
        # would wrongly see that number three times and reject this tree.
        primary = dict(vertices=[dict(id=10, legs=[leg(5, 2), leg(-5, 3), leg(25, 2)]),
                                 dict(id=20, legs=[leg(25, 2), leg(9001, 4), leg(9002, 1)])])
        chain = dict(vertices=[dict(id=10, legs=[leg(-5, 8), leg(5, 6), leg(25, 6)]),
                               dict(id=20, legs=[leg(9001, 9), leg(25, 6), leg(9002, 6)]),
                               dict(id=0, legs=[leg(9002, 6), leg(9002, 1)])])
        self.assertEqual(diagram_definition(primary, 9002, None),
                         diagram_definition(chain, 9002, None))

    def test_distinct_intermediates_and_interactions_do_not_match(self):
        def diagram(mediator, vertex):
            return dict(vertices=[dict(id=vertex, legs=[leg(5, 2), leg(-5, 3), leg(mediator, 2)]),
                                  dict(id=20, legs=[leg(mediator, 2), leg(9001, 4), leg(9002, 1)])])
        expected = diagram_definition(diagram(25, 10), 9002, None)
        self.assertNotEqual(expected, diagram_definition(diagram(23, 10), 9002, None))
        self.assertNotEqual(expected, diagram_definition(diagram(25, 11), 9002, None))

    def test_disconnected_reductions_fail_closed(self):
        bad = dict(vertices=[dict(id=10, legs=[leg(5, 2), leg(-5, 3), leg(25, 2)]),
                             dict(id=20, legs=[leg(9001, 4), leg(9001, 5), leg(9002, 1)])])
        with self.assertRaisesRegex(PrimaryDiagramError, 'Disconnected'):
            diagram_definition(bad, 9002, None)

    def test_analytic_modes_keep_reconstruction_provenance(self):
        modes = [dict(pdg=9002, daughters=[11, -11], branching_ratio=.2)]
        provenance = dict(schema='test-provenance', settings={'max_body': 2})
        capture_primary_diagrams(SimpleNamespace(), modes, provenance)
        self.assertEqual(modes[0]['primary_provenance'], provenance)
        self.assertNotIn('primary_diagrams', modes[0])

    def test_primary_filter_excludes_extra_diagrams_before_export(self):
        class Amp(dict):
            def set(self, name, value):
                self[name] = value
        class Process:
            def get_initial_ids(self): return [9002]
            def get_final_ids(self): return [9001, 5, -5]
        def graph(mediator):
            return dict(vertices=[dict(id=10, legs=[leg(5, 2), leg(-5, 3), leg(mediator, 2)]),
                                  dict(id=20, legs=[leg(mediator, 2), leg(9001, 4), leg(9002, 1)])])
        allowed = Amp(process=Process(), diagrams=[graph(25)])
        generated = Amp(process=Process(), diagrams=[graph(25), graph(23)])
        class Particle(dict):
            def get_amplitudes(self, count): return [allowed]
        class Model:
            def get_particle(self, pdg): return Particle(self_antipart=True)
        class DecayModel(Model):
            def __init__(self, *args): pass
            def read_param_card(self, card): pass
        command = SimpleNamespace(_curr_model=Model(), _curr_amps=[dict(amplitudes=[],
            decay_chains=[dict(amplitudes=[generated], decay_chains=[])])])
        modules = {'madgraph.core': SimpleNamespace(base_objects=SimpleNamespace(DiagramList=list)),
                   'mg5decay.decay_objects': SimpleNamespace(DecayModel=DecayModel)}
        forest = canonical_forest([[9002, [[9001, []], [5, []], [-5, []]]]])
        modes = [dict(pdg=9002, daughters=[9001, 5, -5], primary_provenance={'version': 1})]
        with patch.dict('sys.modules', modules):
            report = restrict_primary_amplitudes(command, '/test/card', forest, modes, {'version': 1})
            self.assertEqual(report[0]['excluded'], 1)
            self.assertEqual(generated['diagrams'], [graph(25)])
            with self.assertRaisesRegex(PrimaryDiagramError, 'provenance'):
                restrict_primary_amplitudes(command, '/test/card', forest, modes, {'version': 2})
            generated['diagrams'] = [graph(23)]
            with self.assertRaisesRegex(PrimaryDiagramError, 'misses 1'):
                restrict_primary_amplitudes(command, '/test/card', forest, modes, {'version': 1})

    def test_public_selected_rate_has_no_manual_branching_factor(self):
        class Channels(list):
            pass
        channels = Channels([SimpleNamespace(id='production', final_pdgs=(9002, 9002),
                                             process='p p > y y')])
        channels.particles = {9002: {'name': 'y'}, 11: {'name': 'e-'}, -11: {'name': 'e+'},
                              13: {'name': 'mu-'}, -13: {'name': 'mu+'}}
        channels.collider = {}
        channels.ufo = Path('/test/model')
        channels.runtime = None
        channels.model_digest = 'model'
        channels.benchmark_digest = 'benchmark'
        decays = SimpleNamespace(modes=tuple(dict(pdg=9002, daughters=daughters,
                    branching_ratio=.2, primary_provenance={'schema': 'test'})
                    for daughters in ([11, -11], [13, -13])),
                    parameter_card=Path('/test/card'), settings={}, width_digest='width')
        row = dict(production_id='production', history_key=[
            [9002, [[11, []], [-11, []]]], [9002, [[13, []], [-13, []]]]], rate_pb=None)
        with TemporaryDirectory() as folder, patch('madgraph_runner.validate_provenance'), \
                patch('madgraph_runner._run_job') as job:
            job.return_value = (dict(cross_section_pb=3., error_pb=.01, target_met=True), Path(folder))
            result = integrate_selected_history(channels, decays, row)
            self.assertEqual(result['cross_section_pb'], 3.)
            self.assertIsNone(result['factorized_reference_pb'])
            self.assertEqual(result['method'], 'madgraph_decay_chain')
            self.assertEqual(result['scale_diagnostics']['dynamical_scale_choice'], 3)
            self.assertIn('not automatically', result['comparison_notice'])
            self.assertEqual(job.call_count, 1)
            decays.modes[0].pop('primary_provenance')
            with self.assertRaisesRegex(PrimaryDiagramError, 'recompute decays'):
                integrate_selected_history(channels, decays, row)
            self.assertEqual(job.call_count, 1)


if __name__ == '__main__':
    unittest.main()
