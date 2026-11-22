"""Independent real-representation, Takagi, and physical-degree checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import fractional_matrix_power

from feynrules_export import ModelResolutionError, CompilerResolutionError, prepare_model as prepare
from generic_majorana_compiler import (classify, resolve_block, component_embedding,
                                        uses_generic, supports_fields)
from generic_fermion_compiler import weak_generators
from model_builder import ModelDefinition
from workflow_errors import ResourceLimitError


def benchmark(dimension=3, colour='1', two=False, kind='majorana'):
    spec = dict(spin='fermion', fermion_kind=kind, su2=dimension, su3=colour, hypercharge='0')
    definition = ModelDefinition(2 if two else 1, spec, spec if two else None, selected_operators='all')
    lagrangian = definition.lagrangian
    coefficients = {}
    for row in lagrangian.operators:
        category, _ = classify(row, definition.fields)
        value = 0.
        if category == 'generic_majorana_mass':
            value = -100. if len(row['factors']) == 1 else 0.
        elif category == 'generic_majorana_kinetic':
            bare = next(p['name'][2:-1] for p in row['factors'] if p['name'].startswith('D('))
            dagger = next(p['name'][:-1] for p in row['factors'] if p['name'].endswith('†'))
            value = .5 if bare == dagger else 0.
        coefficients[definition.coefficient_key(row['id'])] = value if row['multiplicity'] == 1 else [value]*row['multiplicity']
    inputs = definition.required_inputs()
    inputs.update(model_name='GenericMajoranaTest', coefficients=coefficients)
    return definition.build(inputs)


class GenericMajoranaTests(unittest.TestCase):
    def test_real_invariant_and_component_embedding_for_arbitrary_odd_dimensions(self):
        for dimension in (1, 3, 5, 7, 11):
            with self.subTest(dimension=dimension):
                data = component_embedding(dimension)
                b, c = data['transformation'], data['invariant_metric']
                np.testing.assert_allclose(b.conj().T@b, np.eye(dimension), atol=1e-13)
                np.testing.assert_allclose(c@c, np.eye(dimension), atol=1e-13)
                mass_metric = np.eye(dimension)
                for q in range(1, (dimension+1)//2):
                    mass_metric[2*q-1:2*q+1, 2*q-1:2*q+1] = [[0., 1.], [1., 0.]]
                np.testing.assert_allclose(b.T@c@b, mass_metric, atol=1e-13)
                tx, ty, tz = weak_generators(dimension)
                for generator in (tx, ty, tz):
                    np.testing.assert_allclose(generator.T@c+c@generator, 0., atol=1e-13)
                charges = [0] + [q*sign for q in range(1, (dimension+1)//2) for sign in (1, -1)]
                np.testing.assert_allclose(b.conj().T@tz@b, np.diag(charges), atol=1e-13)
                # Charge-conjugating the negative-weight Weyl currents gives
                # exactly the same right-handed ladder as the left-handed one.
                up = b.conj().T@(tx+1j*ty)@b
                j = (dimension-1)//2
                for q in range(1, j+1):
                    lower_left = 0 if q == 1 else 2*q-3
                    lower_right = 0 if q == 1 else 2*q-2
                    expected = np.sqrt((j+q)*(j-q+1))
                    self.assertAlmostEqual(up[2*q-1, lower_left], expected)
                    self.assertAlmostEqual(-up[lower_right, 2*q], expected)

    def test_complex_kinetic_and_takagi_masses_match_independent_singular_values(self):
        kinetic = np.array([[1.7, .15+.22j], [.15-.22j, 2.3]])
        mass = np.array([[200+50j, 35-20j], [35-20j, 350-80j]])
        result = resolve_block(('X', 'Y'), kinetic, mass)
        inv = fractional_matrix_power(kinetic, -.5)
        expected = np.linalg.svd(inv.T@mass@inv, compute_uv=False)
        np.testing.assert_allclose(sorted(result['masses']), sorted(expected), rtol=1e-12)
        transform = result['transformation']
        np.testing.assert_allclose(transform.conj().T@kinetic@transform, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(transform.T@mass@transform, np.diag(result['masses']), atol=1e-10)
        # Gauge currents are proportional to the species identity; the same
        # transform must leave every current canonical, including mixed species.
        generator = weak_generators(5)[0]
        combined = np.kron(transform, np.eye(5))
        np.testing.assert_allclose(combined.conj().T@np.kron(kinetic, generator)@combined,
                                   np.kron(np.eye(2), generator), atol=1e-12)

    def test_catalogue_normalization_offdiagonal_mass_and_kinetic_order(self):
        lagrangian, inputs = benchmark(two=True)
        mapping = {'X^2': -100-25j, 'Y^2': -175+40j, 'X Y': -35+20j,
                   'D(X) X†': .85, 'D(Y) Y†': 1.15, 'D(X) Y†': .15-.22j}
        for row in lagrangian.operators:
            if row['field_content'] in mapping:
                inputs['coefficients'][row['id']] = mapping[row['field_content']]
        model = prepare(lagrangian, inputs)
        result = model.compiler.fermion_basis['X/Y']
        np.testing.assert_allclose(result['mass_matrix'], [[200+50j, 35-20j], [35-20j, 350-80j]])
        np.testing.assert_allclose(result['kinetic'], [[1.7, .15+.22j], [.15-.22j, 2.3]])
        self.assertEqual(set(model.coverage), set(lagrangian.selected_ids))

    def test_degree_inventory_and_native_adjoint_source(self):
        for dimension, colour in ((3, '1'), (5, '8'), (9, '1')):
            lagrangian, inputs = benchmark(dimension, colour, kind='weyl')
            model = prepare(lagrangian, inputs)
            self.assertTrue(uses_generic(model.fields))
            with tempfile.TemporaryDirectory() as temporary:
                files, expression, particles = model.compiler.write_sources(model, Path(temporary), Path(__file__).resolve().parents[1]/'vendor/FeynRules')
                self.assertEqual(len(particles), (dimension+1)//2)
                # One Majorana plus Dirac charge pairs account for exactly n
                # left-handed Weyl components, not n independent Dirac fields.
                self.assertEqual(sum(1 if p['name']==p['antiname'] else 2 for p in particles), dimension)
                self.assertEqual([p['charge'] for p in particles], list(range((dimension+1)//2)))
                self.assertTrue(all(abs(p['mass']-200.)<1e-12 for p in particles))
                self.assertEqual(expression, 'LSM + LGenericMajorana')
                source = files[-1].read_text()
                self.assertIn('I*1/2*pmMJF1q0bar', source)
                self.assertIn('-1/2*pmMJM1', source)
                if colour == '8':
                    self.assertIn('Indices -> {Index[Gluon]}', source)
                    self.assertIn('.DC[pmMJF1q0,mu]', source)
                    self.assertTrue(all(p['color']==8 for p in particles))

    def test_nonzero_yukawa_is_explicit_gap_and_zero_is_accounted(self):
        lagrangian, inputs = benchmark()
        model = prepare(lagrangian, inputs)
        self.assertTrue(any(op.kind=='zero_verified' for op in model.operators))
        row = next(row for row in lagrangian.operators if row['multiplicity']>1)
        inputs['coefficients'][row['id']] = [.1]*row['multiplicity']
        with self.assertRaisesRegex(CompilerResolutionError, 'Yukawa'):
            prepare(lagrangian, inputs)

    def test_singlet_convention_is_preserved_and_budget_is_explicit(self):
        lagrangian, inputs = benchmark(1)
        fields = lagrangian._source.models[0]
        self.assertTrue(supports_fields(fields))
        self.assertFalse(uses_generic(fields))
        lagrangian, inputs = benchmark(5)
        inputs['compiler_limits'] = {'max_weak_generator_entries': 74}
        with self.assertRaisesRegex(ResourceLimitError, 'requires 75 entries'):
            prepare(lagrangian, inputs)
        inputs['compiler_limits']['max_weak_generator_entries'] = 75
        self.assertIsNotNone(prepare(lagrangian, inputs))

    def test_ghost_rejection_and_small_mass_scaling(self):
        with self.assertRaisesRegex(ModelResolutionError, 'positive definite'):
            resolve_block(('X',), [[-1]], [[10]])
        for size in (1e-100, 1e-12, 1., 1e100):
            result = resolve_block(('X',), [[2]], [[(3+4j)*size]])
            self.assertAlmostEqual(result['masses'][0]/size, 2.5)

    def test_degenerate_complex_pair_and_exact_zero_mode(self):
        for size in (1e-100, 1., 1e100):
            mass = np.array([[0., 200j], [200j, 0.]])*size
            result = resolve_block(('X', 'Y'), np.eye(2), mass)
            np.testing.assert_allclose(result['masses']/size, [200., 200.], rtol=1e-12)
            transform = result['transformation']
            np.testing.assert_allclose(transform.T@(mass/size)@transform, np.eye(2)*200., atol=1e-10)
        result = resolve_block(('X', 'Y'), np.eye(2), [[0., 0.], [0., 200j]])
        np.testing.assert_allclose(result['masses'], [0., 200.], atol=1e-12)

    def test_unequal_input_field_rescaling_preserves_the_public_spectrum(self):
        for rescaling in (1e-10, 1e10):
            lagrangian, inputs = benchmark(two=True)
            for row in lagrangian.operators:
                copies = sum(part['count'] for part in row['factors']
                             if part['name'].replace('D(', '').replace(')', '').rstrip('†') == 'X')
                factor = rescaling**copies
                coefficient = inputs['coefficients'][row['id']]
                inputs['coefficients'][row['id']] = ([value*factor for value in coefficient]
                                                   if isinstance(coefficient, list) else coefficient*factor)
            model = prepare(lagrangian, inputs)
            np.testing.assert_allclose([mass for name, mass in model.masses.items() if name != 'H'], 200., rtol=1e-12)


if __name__ == '__main__': unittest.main()
