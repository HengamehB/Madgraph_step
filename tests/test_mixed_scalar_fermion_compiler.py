"""Independent mixed-spin vacuum, mass and scalar/pseudoscalar coupling tests."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
import numpy as np
from scipy.linalg import eigvalsh

from lagrangian_maker import make_lagrangian
from feynrules_export import ModelResolutionError, CompilerResolutionError
from mixed_scalar_fermion_compiler import prepare, classify, uses_generic


def benchmark(su2=3, su3='1', hypercharge='1', reverse=False):
    scalar = dict(spin='scalar', su2=1, su3='1', hypercharge='0', boson_kind='real')
    fermion = dict(spin='fermion', su2=su2, su3=su3, hypercharge=hypercharge, fermion_kind='dirac')
    result = make_lagrangian(2, fermion if reverse else scalar, scalar if reverse else fermion, show=False)
    lag = result.select('all', show=False)
    s, f = ('Y', 'X') if reverse else ('X', 'Y')
    quadratic, quartic, portal2 = -80000., -.01, -.05
    coefficients = {}
    given = {s: 0., s+'^2': quadratic, s+'^3': 0., s+'^4': quartic,
             'H H† '+s: 0., 'H H† '+s+'^2': portal2,
             f'D({s})^2': .75, f'{f}_L {f}_Rc': -80+12j,
             f'D({f}_L) {f}_L†': 1., f'D({f}_Rc) {f}_Rc†': .25}
    for row in lag.operators:
        kind, _ = classify(row, result.models[0])
        value = given.get(row['field_content'], 0.)
        if kind == 'mixed_scalar_dirac_yukawa': value = -.4+.3j
        coefficients[row['id']] = value if row['multiplicity'] == 1 else [value] * row['multiplicity']
    return lag, dict(model_name='MixedScalarDiracTest', coefficients=coefficients, higgs_mass=125.,
                     vevs={s: 0.}, widths={s: 0., f: 0.})


class MixedScalarFermionTests(unittest.TestCase):
    def test_scalar_field_rescaling_preserves_public_masses_couplings_and_vacuum_check(self):
        from feynrules_export import prepare_model
        lag, inputs = benchmark()
        reference = prepare_model(lag, deepcopy(inputs))
        for scale in (1e-10, 1e10):
            with self.subTest(scale=scale):
                changed = deepcopy(inputs)
                for row in lag.operators:
                    power = sum(part['count'] for part in row['factors']
                                if part['name'] in ('X', 'D(X)'))
                    value = changed['coefficients'][row['id']]
                    changed['coefficients'][row['id']] = ([v * scale**power for v in value]
                        if isinstance(value, list) else value * scale**power)
                changed['vevs']['X'] /= scale
                result = prepare_model(lag, changed)
                np.testing.assert_allclose(list(result.masses.values()),
                                           list(reference.masses.values()), rtol=1e-12)
                np.testing.assert_allclose(result.compiler.fermion_basis['Y']['physical_scalar_yukawas'],
                    reference.compiler.fermion_basis['Y']['physical_scalar_yukawas'], rtol=1e-12)
                tadpole = next(row['id'] for row in lag.operators if row['field_content'] == 'X')
                changed['coefficients'][tadpole] += 1e5 * scale
                with self.assertRaisesRegex(ModelResolutionError, 'not stationary'):
                    prepare_model(lag, changed)

    def test_zero_scalar_vev_and_generalized_scalar_masses(self):
        lag, inputs = benchmark()
        model = prepare(lag, inputs)
        self.assertTrue(uses_generic(model.fields))
        b = model.compiler.scalar_basis
        v = b['vev']
        expected_hessian = np.diag([125.**2, 160000 + .05*v*v])
        np.testing.assert_allclose(b['hessian'], expected_hessian, rtol=1e-12)
        expected = np.sqrt(eigvalsh(expected_hessian, np.diag([1., 1.5])))
        np.testing.assert_allclose(sorted(b['masses']), expected, rtol=1e-12)
        np.testing.assert_allclose(b['transformation'].T @ b['kinetic'] @ b['transformation'], np.eye(2), atol=1e-12)

    def test_complex_mass_phase_and_all_scalar_yukawas_match_derivatives(self):
        lag, inputs = benchmark()
        model = prepare(lag, inputs)
        b = model.compiler.fermion_basis['Y']
        expected_mass = 80-12j
        self.assertAlmostEqual(model.compiler.physical_mass, abs(expected_mass))
        expected_yukawa = (-.4+.3j) * expected_mass.conjugate() / abs(expected_mass)
        self.assertAlmostEqual(model.compiler.effective_yukawa, expected_yukawa)
        couplings = b['physical_scalar_yukawas']
        mixing = model.compiler.scalar_basis['transformation'][1]
        np.testing.assert_allclose(couplings, expected_yukawa * mixing, atol=1e-14)
        self.assertAlmostEqual(sum(abs(g)**2 for g in couplings), abs(-.4+.3j)**2/1.5)
        # d |M-y*s|/ds = -Re(y e^-i arg M): checks phase and VEV counting.
        step = 1e-3
        for index, coupling in enumerate(couplings):
            mass = lambda displacement: abs(expected_mass - (-.4+.3j)*mixing[index]*displacement)
            derivative = (mass(step)-mass(-step))/(2*step)
            self.assertAlmostEqual(derivative, -coupling.real, places=8)

    def test_all_selected_operators_accounted_in_both_field_orders(self):
        for reverse in (False, True):
            lag, inputs = benchmark(reverse=reverse)
            model = prepare(lag, inputs)
            self.assertEqual(set(model.coverage), set(lag.selected_ids))
            self.assertEqual(len(model.masses), 5)
            self.assertTrue(any(op.kind == 'mixed_scalar_dirac_yukawa' for op in model.operators))
            self.assertTrue(any(op.kind == 'zero_verified' for op in model.operators))

    def test_nonstationary_vacuum_and_unsupported_nonzero_yukawa_rejected(self):
        lag, inputs = benchmark()
        linear = next(row['id'] for row in lag.operators if row['field_content']=='X')
        inputs['coefficients'][linear] += 1e5
        with self.assertRaisesRegex(ModelResolutionError, 'not stationary'):
            prepare(lag, inputs)
        lag, inputs = benchmark()
        row = next(row for row in lag.operators if row['multiplicity'] > 1)
        inputs['coefficients'][row['id']] = [.1]*row['multiplicity']
        with self.assertRaisesRegex(CompilerResolutionError, 'compiler gap'):
            prepare(lag, inputs)

    def test_source_loads_scalar_sm_once_and_yukawa_contains_only_fluctuations(self):
        lag, inputs = benchmark(su2=2, su3='6', hypercharge='1/2')
        model = prepare(lag, inputs)
        with tempfile.TemporaryDirectory() as temporary:
            files, expression, particles = model.compiler.write_sources(model, Path(temporary), Path(__file__).resolve().parents[1]/'vendor/FeynRules')
            self.assertEqual(len(files), 4)
            self.assertEqual(sum(file.name.startswith('SM') for file in files), 1)
            self.assertEqual(len(particles), 3)
            sm = files[0].read_text()
            self.assertIn('{pmGFT2,pmGFWeak2}', sm)
            self.assertIn('pmR00*H', sm)
            yukawa = files[-1].read_text()
            self.assertNotIn('pmR10*H', yukawa)
            self.assertIn('pmR11*pmS1', yukawa)
            self.assertNotIn('pmVEV1*', yukawa)
            self.assertIn('ProjM', yukawa)
            self.assertIn('HC[', yukawa)
            self.assertEqual(expression, 'LSM + LNew + LGenericFermion + LMixedScalarFermion')


if __name__ == '__main__': unittest.main()
