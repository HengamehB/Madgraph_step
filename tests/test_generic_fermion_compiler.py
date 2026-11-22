"""Analytic gauge identities and independent chiral mass-basis checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import fractional_matrix_power

from lagrangian_maker import make_lagrangian
from feynrules_export import ModelResolutionError
from generic_fermion_compiler import classify, prepare, resolve_block, weak_generators, uses_generic


def benchmark(su2=3, hypercharge='0', su3='1', two=False):
    spec = dict(spin='fermion', fermion_kind='dirac', su2=su2, su3=su3, hypercharge=hypercharge)
    result = make_lagrangian(2 if two else 1, spec, spec if two else None, show=False)
    lag = result.select('all', show=False)
    fields = result.models[0]
    coefficients = {}
    for row in lag.operators:
        values = [0.] * row['multiplicity']
        kind, _ = classify(row, fields)
        factors = [p['name'] for p in row['factors']]
        if kind == 'generic_dirac_mass':
            values[0] = -200. if len({v.split('_')[0] for v in factors}) == 1 else 0.
        if kind == 'generic_dirac_kinetic':
            bare = next(v[2:-1] for v in factors if v.startswith('D('))
            dagger = next(v[:-1] for v in factors if v.endswith('†'))
            values[0] = .5 if bare == dagger else 0.
        coefficients[row['id']] = values[0] if len(values) == 1 else values
    return lag, dict(model_name='GenericDiracTest', coefficients=coefficients, widths={f.name: 0. for f in fields}, higgs_mass=125.)


class GenericFermionTests(unittest.TestCase):
    def test_uniform_field_rescaling_preserves_mixed_chiral_physics(self):
        left = np.array([[2., .1+.2j], [.1-.2j, 1.3]])
        right = np.array([[1.2, -.12j], [.12j, 1.8]])
        mass = np.array([[200+40j, 30-20j], [-70j, 350-10j]])
        reference = resolve_block(('X','Y'), left, right, mass)
        for scale in (1e-150, 1e-40, 1e40, 1e150):
            with self.subTest(scale=scale):
                result = resolve_block(('X','Y'), scale*left, scale*right, scale*mass)
                np.testing.assert_allclose(result['masses'], reference['masses'], rtol=1e-12)
                for name in ('left_transformation','right_conjugate_transformation'):
                    np.testing.assert_allclose(result[name]*np.sqrt(scale), reference[name], rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(resolve_block(('X',), [[1e-14]], [[1e-14]], [[2e-12]])['masses'][0], 200.)

    def test_hermiticity_is_checked_relative_to_small_kinetic_scale(self):
        left = 1e-30 * np.array([[1., .1j],[.1j,1.]])
        with self.assertRaisesRegex(ModelResolutionError, 'Hermitian'):
            resolve_block(('X','Y'), left, 1e-30*np.eye(2), 1e-28*np.eye(2))

    def test_large_multiplets_have_disjoint_particle_and_class_identities(self):
        import re
        lag, inputs = benchmark(101, '1', two=True)
        model = prepare(lag, inputs)
        with tempfile.TemporaryDirectory() as temporary:
            files, _, particles = model.compiler.write_sources(model, Path(temporary), Path(__file__).resolve().parents[1] / 'vendor/FeynRules')
            self.assertEqual(len({p['pdg'] for p in particles}), 202)
            ids = re.findall(r'F\[(\d+)\] ==', files[-1].read_text())
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all('color' in p and 'charge' in p for p in particles))

    def test_generators_commutators_and_casimir_for_arbitrary_dimensions(self):
        for n in range(1, 9):
            with self.subTest(dimension=n):
                x, y, z = weak_generators(n)
                np.testing.assert_allclose(x@y-y@x, 1j*z, atol=1e-13)
                np.testing.assert_allclose(y@z-z@y, 1j*x, atol=1e-13)
                np.testing.assert_allclose(z@x-x@z, 1j*y, atol=1e-13)
                j = (n-1)/2
                np.testing.assert_allclose(x@x+y@y+z@z, j*(j+1)*np.eye(n), atol=1e-13)
                for g in (x, y, z):
                    np.testing.assert_allclose(g, g.conj().T, atol=1e-13)

    def test_triplet_and_quartet_complete_component_inventory(self):
        for n, y in ((3, '0'), (4, '1/2'), (7, '1')):
            with self.subTest(dimension=n):
                lag, inputs = benchmark(n, y)
                model = prepare(lag, inputs)
                self.assertTrue(uses_generic(model.fields))
                self.assertEqual(set(model.coverage), set(lag.selected_ids))
                self.assertEqual(len(model.masses), n+1)
                self.assertTrue(all(abs(m-200.) < 1e-12 for name, m in model.masses.items() if name != 'H'))
                with tempfile.TemporaryDirectory() as temporary:
                    files, expression, particles = model.compiler.write_sources(model, Path(temporary), Path(__file__).resolve().parents[1] / 'vendor/FeynRules')
                    self.assertEqual(len(particles), n)
                    self.assertEqual(len({p['pdg'] for p in particles}), n)
                    self.assertEqual(len({p['name'] for p in particles}), n)
                    self.assertTrue(all(p['spin'] == 2 for p in particles))
                    self.assertIn('FlavorExpand -> {SU2D,SU2W,pmGFWeak', files[-1].read_text())
                    self.assertEqual(expression, 'LSM + LGenericFermion')

    def test_complex_kinetic_mass_mixing_matches_independent_singular_values(self):
        left = np.array([[2., .1+.2j], [.1-.2j, 1.3]])
        right = np.array([[1.2, -.12j], [.12j, 1.8]])
        mass = np.array([[200+40j, 30-20j], [-70j, 350-10j]])
        result = resolve_block(('X', 'Y'), left, right, mass)
        a = fractional_matrix_power(left, -.5)
        b = fractional_matrix_power(right, -.5)
        expected = np.linalg.svd(a.T @ mass @ b, compute_uv=False)
        np.testing.assert_allclose(sorted(result['masses']), sorted(expected), rtol=1e-12)
        tl, tr = result['left_transformation'], result['right_conjugate_transformation']
        np.testing.assert_allclose(tl.conj().T @ left @ tl, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(tr.conj().T @ right @ tr, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(tl.T @ mass @ tr, np.diag(result['masses']), atol=1e-10)

    def test_catalogue_directed_mixing_convention(self):
        lag, inputs = benchmark(3, '1', two=True)
        for row in lag.operators:
            text = row['field_content']
            if text == 'X_L Y_Rc': inputs['coefficients'][row['id']] = -30-20j
            if text == 'X_Rc Y_L': inputs['coefficients'][row['id']] = 10j
            if text == 'Y_L Y_Rc': inputs['coefficients'][row['id']] = -350.
            if text == 'D(X_L) Y_L†': inputs['coefficients'][row['id']] = .1+.15j
        model = prepare(lag, inputs)
        block = model.compiler.fermion_basis['X/Y']
        np.testing.assert_allclose(block['mass_matrix'], [[200, 30+20j], [-10j, 350]])
        np.testing.assert_allclose(block['left_kinetic'], [[1, .1-.15j], [.1+.15j, 1]])
        self.assertLess(max(block['residuals'].values()), 1e-9)

    def test_zero_operators_accounted_nonzero_yukawa_or_majorana_rejected(self):
        lag, inputs = benchmark()
        model = prepare(lag, inputs)
        zeros = [op for op in model.operators if op.kind == 'zero_verified']
        self.assertTrue(zeros)
        self.assertTrue(all(all(c == 0 for c in op.coefficients) for op in zeros))
        for op in zeros[:2]:
            altered = {**inputs, 'coefficients': dict(inputs['coefficients'])}
            altered['coefficients'][op.operator_id] = 1. if len(op.coefficients) == 1 else [1.] * len(op.coefficients)
            with self.assertRaisesRegex(ModelResolutionError, 'compiler gap'):
                prepare(lag, altered)

    def test_configured_generator_budget_is_reported_precisely(self):
        from workflow_errors import ResourceLimitError, error_category
        lag, inputs = benchmark(4, '1/2')
        inputs['compiler_limits'] = {'max_weak_generator_entries': 47}
        with self.assertRaisesRegex(ResourceLimitError, 'requires 48 entries') as caught:
            prepare(lag, inputs)
        self.assertEqual(error_category(caught.exception), 'resource_limit')
        inputs['compiler_limits']['max_weak_generator_entries'] = 48
        self.assertIsNotNone(prepare(lag, inputs))

    def test_ghost_kinetic_input_is_rejected(self):
        lag, inputs = benchmark(4, '1/2')
        for row in lag.operators:
            if row['field_content'] == 'D(X_L) X_L†': inputs['coefficients'][row['id']] = -.5
        with self.assertRaisesRegex(ModelResolutionError, 'not positive definite'):
            prepare(lag, inputs)


class DiracSpeciesRescalingTests(unittest.TestCase):
    def test_independent_chiral_species_units_preserve_complex_mixing(self):
        left = np.array([[2., .1 + .2j], [.1 - .2j, 1.3]])
        right = np.array([[1.2, -.12j], [.12j, 1.8]])
        mass = np.array([[200 + 40j, 30 - 20j], [-70j, 350 - 10j]])
        reference = resolve_block(('X', 'Y'), left, right, mass)
        # This spectator Yukawa checks that the two canonical rotations retain
        # their relative phases, in addition to matching the singular masses.
        yukawa = np.array([[.3 + .1j, -.2j], [.15, -.4 + .07j]])
        expected_yukawa = reference['left_transformation'].T @ yukawa @ reference['right_conjugate_transformation']
        for left_scale, right_scale in ((np.array([1e-60, 1e60]), np.array([1e30, 1e-30])),
                                         (np.array([1e60, 1e-60]), np.array([1e-30, 1e30]))):
            with self.subTest(left=left_scale.tolist(), right=right_scale.tolist()):
                scaled_left = (left_scale[:, None] * left) * left_scale[None, :]
                scaled_right = (right_scale[:, None] * right) * right_scale[None, :]
                scaled_mass = (left_scale[:, None] * mass) * right_scale[None, :]
                actual = resolve_block(('X', 'Y'), scaled_left, scaled_right, scaled_mass)
                np.testing.assert_allclose(actual['masses'], reference['masses'], rtol=2e-13)
                tl, tr = actual['left_transformation'], actual['right_conjugate_transformation']
                np.testing.assert_allclose(left_scale[:, None] * tl, reference['left_transformation'], rtol=2e-12, atol=1e-14)
                np.testing.assert_allclose(right_scale[:, None] * tr, reference['right_conjugate_transformation'], rtol=2e-12, atol=1e-14)
                scaled_yukawa = (left_scale[:, None] * yukawa) * right_scale[None, :]
                np.testing.assert_allclose(tl.T @ scaled_yukawa @ tr, expected_yukawa, rtol=3e-12, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
