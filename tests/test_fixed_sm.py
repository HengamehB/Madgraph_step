"""Fixed SM assumptions retain consistent coefficients without vacuum search."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fixed_sm import normalize_fixed_sm, check_fixed_sm_model, fixed_sm_report
from feynrules_export import ModelResolutionError, prepare_model
from model_builder import ModelDefinition


def scalar(colour='1', weak=1, charge='0', kind='real'):
    return dict(spin='scalar', su2=weak, su3=colour, hypercharge=charge, boson_kind=kind)


def zero_inputs(definition):
    inputs = definition.required_inputs()
    inputs['coefficients'] = {
        row['key']: 0. if row['multiplicity'] == 1 else [0.] * row['multiplicity']
        for row in definition.operator_inputs()}
    return inputs


class FixedSMInputsTests(unittest.TestCase):
    def setUp(self):
        self.definition = ModelDefinition(1, scalar(), selected_operators=['X^2', 'D(X)^2'])

    def test_normalization_preserves_coefficients_and_does_not_mutate(self):
        values = {'coefficients': {'x': [1 + 2j, -3.]}, 'sm_inputs': {'aS': .12}}
        original = deepcopy(values)
        result = normalize_fixed_sm(values, self.definition.fields)
        self.assertEqual(values, original)
        self.assertEqual(result['coefficients'], values['coefficients'])
        self.assertEqual(result['higgs_mass'], 125.)
        self.assertEqual(result['vevs'], {'X': 0.})
        self.assertEqual(result['sm_inputs']['aS'], .12)
        self.assertGreater(result['sm_inputs']['Gf'], 0.)
        self.assertNotIn('vacuum', result)

    def test_no_fermion_vev_entries_are_invented(self):
        fields = [SimpleNamespace(name='X', spin='scalar'), SimpleNamespace(name='Y', spin='fermion')]
        self.assertEqual(normalize_fixed_sm({}, fields)['vevs'], {'X': 0.})

    def test_contradictory_inputs_rejected_before_export(self):
        cases = [({'vevs': {'X': 1.}}, 'must be zero'),
                 ({'vevs': {'Y': 0.}}, 'Unknown scalar'),
                 ({'higgs_mass': 126.}, 'must be 125'),
                 ({'observed_higgs_mass': 124.}, 'must be 125'),
                 ({'vacuum': {'mode': 'auto'}}, 'remove old vacuum'),
                 ({'_resolved_vacuum': {}}, 'remove old vacuum'),
                 ({'sm_inputs': {'lam': .13}}, 'not independent'),
                 ({'sm_inputs': {'MH': 125.}}, 'not independent'),
                 ({'sm_inputs': {'Gf': -1.}}, 'must be positive'),
                 ({'sm_inputs': {'MZ': 1.}}, 'real SM'),
                 ({'sm_scheme': 'measured'}, '(?i)electroweak|scheme')]
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ModelResolutionError, message):
                normalize_fixed_sm(values, self.definition.fields)

    def test_report_states_assumptions_not_stability(self):
        report = fixed_sm_report(normalize_fixed_sm({}, self.definition.fields))
        self.assertEqual(report['status'], 'assumed_fixed_sm')
        self.assertFalse(report['potential_minimized'])
        self.assertFalse(report['search_complete'])
        self.assertIsNone(report['global_minimum'])
        self.assertEqual(report['boundedness'], 'not_evaluated')
        self.assertEqual(report['assumptions']['scalar_vevs'], {'X': 0.})
        self.assertEqual(report['assumptions']['higgs_mass_gev'], 125.)


class FixedSMCompilerTests(unittest.TestCase):
    def test_singlet_portal_masses_without_any_vacuum_solver(self):
        definition = ModelDefinition(1, scalar(), selected_operators=['X^2', 'D(X)^2', 'H H† X^2'])
        inputs = zero_inputs(definition)
        inputs['coefficients'].update({'X^2': -20000., 'D(X)^2': .5, 'H H† X^2': -.01})
        with patch('subprocess.Popen', side_effect=AssertionError('external process called')):
            lag, resolved = definition.build(inputs)
            model = prepare_model(lag, resolved)
        v2 = 1 / (math.sqrt(2) * resolved['sm_inputs']['Gf'])
        self.assertEqual(model.masses['H'], 125.)
        self.assertAlmostEqual(model.masses['X'] ** 2, 40000. + .01 * v2)
        self.assertEqual(model.inputs['coefficients'], resolved['coefficients'])

    def test_bsm_mixing_remains_supported_with_fixed_higgs(self):
        definition = ModelDefinition(2, scalar(), scalar(), selected_operators=[
            'X^2', 'Y^2', 'D(X)^2', 'D(Y)^2', 'D(X) D(Y)', 'X Y'])
        inputs = zero_inputs(definition)
        inputs['coefficients'].update({'X^2': -20000., 'Y^2': -180000.,
            'D(X)^2': .5, 'D(Y)^2': .5, 'D(X) D(Y)': .1, 'X Y': 500.})
        model = prepare_model(*definition.build(inputs))
        self.assertEqual(model.masses['H'], 125.)
        self.assertNotEqual(model.basis['transformation'][1, 2], 0.)

    def test_unbounded_quartic_is_not_a_hidden_global_stability_check(self):
        definition = ModelDefinition(1, scalar(), selected_operators=['X^2', 'D(X)^2', 'X^4'])
        inputs = zero_inputs(definition)
        # L contains +X^4, so V has an unbounded negative quartic, despite a
        # positive curvature at the assumed expansion point. This workflow
        # deliberately does not attempt to certify the global potential.
        inputs['coefficients'].update({'X^2': -20000., 'D(X)^2': .5, 'X^4': .01})
        with patch('subprocess.Popen', side_effect=AssertionError('external process called')):
            model = prepare_model(*definition.build(inputs))
        self.assertAlmostEqual(model.masses['X'], 200.)
        self.assertEqual(fixed_sm_report(model.inputs)['boundedness'], 'not_evaluated')

    def test_uncancelled_tadpole_still_rejected(self):
        definition = ModelDefinition(1, scalar(), selected_operators=['X^2', 'D(X)^2', 'H H† X'])
        inputs = zero_inputs(definition)
        inputs['coefficients'].update({'X^2': -20000., 'D(X)^2': .5, 'H H† X': -1.})
        with self.assertRaisesRegex(ModelResolutionError, 'not stationary'):
            prepare_model(*definition.build(inputs))

    def test_higgs_mixing_rejected_even_with_cancelled_tadpole(self):
        definition = ModelDefinition(1, scalar(), selected_operators=['X', 'X^2', 'D(X)^2', 'H H† X'])
        for coefficient in (-1., -1e-20):
            inputs = zero_inputs(definition)
            v2 = 1 / (math.sqrt(2) * inputs['sm_inputs']['Gf'])
            inputs['coefficients'].update({'X': -coefficient * v2 / 2, 'X^2': -20000.,
                'D(X)^2': .5, 'H H† X': coefficient})
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ModelResolutionError, 'mix the Higgs'):
                prepare_model(*definition.build(inputs))

    def test_coloured_mixing_is_not_mistaken_for_higgs_mixing(self):
        spec = scalar(colour='3', kind='complex')
        definition = ModelDefinition(2, spec, spec, selected_operators=[
            'X X†', 'Y Y†', 'X Y†', 'D(X) D(X†)', 'D(Y) D(Y†)',
            'H H† X X†', 'H H† X Y†', 'H H† Y Y†'])
        inputs = zero_inputs(definition)
        inputs['coefficients'].update({'X X†': -40000., 'Y Y†': -360000., 'X Y†': .05,
            'D(X) D(X†)': 1., 'D(Y) D(Y†)': 1., 'H H† X X†': -.01,
            'H H† X Y†': -.01, 'H H† Y Y†': -.01})
        model = prepare_model(*definition.build(inputs))
        self.assertEqual(model.masses['H'], 125.)
        self.assertAlmostEqual(model.masses['X'], 200.7556619434269)
        self.assertAlmostEqual(model.masses['Y'], 600.2527883183927)

    def test_generic_complex_scalar_higgs_mixing_rejected(self):
        definition = ModelDefinition(1, scalar(kind='complex'), selected_operators=[
            'X', 'X X†', 'D(X) D(X†)', 'H H† X'])
        for coefficient in (-1., -1e-20):
            inputs = zero_inputs(definition)
            v2 = 1 / (math.sqrt(2) * inputs['sm_inputs']['Gf'])
            # Native scalar RREF convention: H†H X = h^2(x_R+i x_I)
            # before the h -> v+h shift; both rows include their conjugate.
            inputs['coefficients'].update({'X': -coefficient * v2, 'X X†': -40000.,
                'D(X) D(X†)': 1., 'H H† X': coefficient})
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ModelResolutionError, 'mix the Higgs'):
                prepare_model(*definition.build(inputs))

    def test_doublet_higgs_mixing_rejected_after_tadpole_cancellation(self):
        definition = ModelDefinition(1, scalar(weak=2, kind='complex', charge='1/2'))
        for coupling in (.02 + .01j, 1e-20 + 1e-20j):
            inputs = zero_inputs(definition)
            v2 = 1 / (math.sqrt(2) * inputs['sm_inputs']['Gf'])
            inputs['coefficients'].update({'X X†': -40000., 'D(X) D(X†)': 1.,
                'H X†': -coupling.conjugate() * v2 / 2, 'H H†^2 X': coupling})
            with self.subTest(coupling=coupling), self.assertRaisesRegex(ModelResolutionError, 'mix the Higgs'):
                prepare_model(*definition.build(inputs))

    def test_generic_weak_multiplet_keeps_higgs(self):
        from test_generic_scalar_compiler import benchmark
        lag, inputs = benchmark()
        inputs['sm_scheme'] = 'fixed_sm'
        model = prepare_model(lag, inputs)
        self.assertAlmostEqual(model.masses['h'], 125.)
        self.assertIs(check_fixed_sm_model(model), model)

    def test_no_physical_higgs_mass_overwrite(self):
        prepared = SimpleNamespace(inputs={'sm_scheme': 'fixed_sm'}, manual=None,
                                   compiler=None, basis=None, operators=(), masses={'H': 126.})
        with self.assertRaisesRegex(ModelResolutionError, 'exported mass was not overwritten'):
            check_fixed_sm_model(prepared)


if __name__ == '__main__':
    unittest.main()
