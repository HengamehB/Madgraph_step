"""Independent native-colour algebra and physical/source checks."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import eigh, expm

from lagrangian_maker import make_lagrangian
from feynrules_export import CompilerResolutionError, ModelResolutionError
from workflow_errors import ResourceLimitError, error_category
from generic_color_compiler import (basis_fingerprint, basis_labels, classify,
    colour_tensors, invariant_basis, prepare, uses_generic)

ROOT = Path(__file__).resolve().parents[1]


def benchmark(colour='6', charge='1', kind='complex', spin='scalar'):
    spec = dict(spin=spin, su2=1, su3=colour, hypercharge=charge)
    spec['boson_kind' if spin == 'scalar' else 'fermion_kind'] = kind
    result = make_lagrangian(1, spec, show=False)
    lagrangian = result.select('all', show=False)
    values = dict(model_name='NativeColourTest', coefficients={}, vevs={'X': 0.}, widths={'X': 0.},
                  higgs_mass=125., sm_inputs={'Gf': 1.16637e-5})
    for row in lagrangian.operators:
        counts = {p['name']: p['count'] for p in row['factors']}
        opkind, _ = classify(row, result.models[0])
        coefficient = 0. if row['multiplicity'] == 1 else [0.] * row['multiplicity']
        if opkind == 'colour_scalar_kinetic' and (counts == {'D(X)':2} and kind == 'real' or
                                                 counts == {'D(X)':1,'D(X†)':1}):
            coefficient = .5 if kind == 'real' else 1.
        elif opkind == 'colour_dirac_kinetic':
            coefficient = .5
        elif opkind == 'colour_dirac_mass':
            coefficient = -200.
        elif counts == {'X': 2} and kind == 'real':
            coefficient = -20000.
        elif counts == {'X': 1, 'X†': 1}:
            coefficient = -40000.
        elif opkind == 'colour_scalar_potential' and sum(counts.values()) == 4 and 'H' not in counts:
            coefficient = -.01 if row['multiplicity'] == 1 else [-.01] * row['multiplicity']
        values['coefficients'][row['id']] = coefficient
    return lagrangian, values


def mixed_benchmark(colour='6'):
    spec = dict(spin='scalar', su2=1, su3=colour, hypercharge='1', boson_kind='complex')
    result = make_lagrangian(2, spec, spec, show=False)
    lagrangian = result.select('all', show=False)
    values = dict(model_name='MixedNativeColour', coefficients={}, vevs={'X': 0., 'Y': 0.},
                  widths={'X': 0., 'Y': 0.}, higgs_mass=125., sm_inputs={'Gf': 1.16637e-5})
    chosen = {'D(X) D(X†)':1.2, 'D(Y) D(Y†)':.8, 'D(X) D(Y†)':.1+.05j,
              'X X†':-40000., 'Y Y†':-90000., 'X Y†':-2000.-1000j}
    for row in lagrangian.operators:
        values['coefficients'][row['id']] = chosen.get(row['field_content'],
            0. if row['multiplicity'] == 1 else [0.] * row['multiplicity'])
        if row['field_content'] == 'X X† Y Y†':
            values['coefficients'][row['id']] = [-.02] + [0.] * (row['multiplicity'] - 1)
    return lagrangian, values


class NativeColourTests(unittest.TestCase):
    def test_sextet_generators_and_clebsch_gordan_normalization(self):
        tensors = colour_tensors()
        np.testing.assert_allclose(np.einsum('aij,bji->ab', tensors['T'], tensors['T']), .5 * np.eye(8), atol=1e-14)
        np.testing.assert_allclose(np.einsum('aij,bji->ab', tensors['T6'], tensors['T6']), 2.5 * np.eye(8), atol=1e-14)
        np.testing.assert_allclose(np.einsum('Aij,Bij->AB', tensors['K6'], tensors['K6']), np.eye(6), atol=1e-14)
        np.testing.assert_allclose(np.einsum('aij,ajk->ik', tensors['T6'], tensors['T6']), (10/3) * np.eye(6), atol=1e-14)
        for a in range(8):
            for b in range(8):
                np.testing.assert_allclose(tensors['T6'][a] @ tensors['T6'][b] - tensors['T6'][b] @ tensors['T6'][a],
                                          1j * np.einsum('c,cij->ij', tensors['f'][a,b], tensors['T6']), atol=1e-14)

    def test_native_bases_match_exact_catalogue_counts(self):
        for colour, charge, kind, count in [('6','1','complex',2), ('8','0','real',1), ('8','1','complex',3)]:
            with self.subTest(colour=colour, kind=kind):
                lagrangian, values = benchmark(colour, charge, kind)
                model = prepare(lagrangian, values)
                self.assertAlmostEqual(model.masses['X'], 200.)
                row = next(row for row in lagrangian.operators if not row['derivatives'] and
                           sum(part['count'] for part in row['factors']) == 4 and 'H' not in row['field_content'])
                self.assertEqual(row['multiplicity'], count)
                self.assertEqual(len(invariant_basis(row, model.fields)[0]), count)
                self.assertEqual(len(basis_labels(row, model.fields)), count)
                self.assertEqual(len(basis_fingerprint(row, model.fields)), 64)

    def test_native_quartics_and_sextet_determinant_are_group_invariant(self):
        rng = np.random.default_rng(17)
        tensors = colour_tensors()
        for colour in ('6', '8'):
            lagrangian, _ = benchmark(colour, '0' if colour == '6' else '1')
            fields = lagrangian._source.models[0]
            generators = tensors['T6'] if colour == '6' else -1j * tensors['f']
            transform = expm(1j * np.einsum('a,aij->ij', rng.normal(size=8), generators))
            x = rng.normal(size=abs(int(colour))) + 1j * rng.normal(size=abs(int(colour)))
            rotated = transform @ x
            for row in lagrangian.operators:
                if row['derivatives'] or 'H' in row['field_content']:
                    continue
                _, polys = invariant_basis(row, fields)
                def evaluate(poly, x):
                    return sum(coefficient * math.prod((x[index].conjugate() if token.endswith('†') else x[index])
                                                       for token,index in monomial) for monomial,coefficient in poly.items())
                for polynomial in polys:
                    self.assertAlmostEqual(evaluate(polynomial, x), evaluate(polynomial, rotated), places=8)

    def test_real_octet_cubic_is_d_symbol_and_has_correct_normalization(self):
        lagrangian, values = benchmark('8', '0', 'real')
        row = next(row for row in lagrangian.operators if row['field_content'] == 'X^3')
        labels = basis_labels(row, lagrangian._source.models[0])
        self.assertIn('dSUN[', labels[0])
        self.assertNotIn('f[', labels[0])
        values['coefficients'][row['id']] = .1
        self.assertAlmostEqual(prepare(lagrangian, values).masses['X'], 200.)

    def test_native_source_retains_sextet_colour_and_covariant_kinetic_term(self):
        lagrangian, values = benchmark()
        model = prepare(lagrangian, values)
        with tempfile.TemporaryDirectory() as folder:
            files, expression, particles = model.compiler.write_sources(model, Path(folder), ROOT/'vendor'/'FeynRules')
            source = files[1].read_text()
        self.assertIn('AddGaugeRepresentation[{SU3C -> {T6,Sextet}}]', source)
        self.assertIn('Indices -> {Index[Sextet]}', source)
        self.assertIn('DC[pmGC1bar[cc],mu]*DC[pmGC1[cc],mu]', source)
        self.assertIn('T6[', source)
        self.assertEqual(expression, 'LSM + LColour')
        self.assertEqual(particles[0]['name'], 'x')

    def test_antirepresentation_names_and_charge_are_conjugated(self):
        lagrangian, values = benchmark('6bar')
        model = prepare(lagrangian, values)
        with tempfile.TemporaryDirectory() as folder:
            files, _, particles = model.compiler.write_sources(model, Path(folder), ROOT/'vendor'/'FeynRules')
            source = files[1].read_text()
        self.assertEqual((particles[0]['name'], particles[0]['antiname']), ('xbar','x'))
        self.assertIn('Y -> -1', source)

    def test_native_dirac_masses_and_chiral_normalization(self):
        for colour in ('6','8'):
            lagrangian, values = benchmark(colour, '1', 'dirac', 'fermion')
            model = prepare(lagrangian, values)
            self.assertAlmostEqual(model.masses['X'], 200.)

    def test_zero_verified_requires_complete_vector_and_nonzero_gap_is_reported(self):
        lagrangian, values = benchmark('6', '1/3')
        unsupported = [row for row in lagrangian.operators if classify(row, lagrangian._source.models[0])[1]]
        self.assertTrue(unsupported)
        model = prepare(lagrangian, deepcopy(values))
        self.assertTrue(any(op.kind == 'zero_verified' for op in model.operators))
        row = unsupported[0]
        values['coefficients'][row['id']] = [0.] * (row['multiplicity'] - 1)
        with self.assertRaises(ModelResolutionError):
            prepare(lagrangian, deepcopy(values))
        values['coefficients'][row['id']] = [0.] * row['multiplicity']
        values['coefficients'][row['id']][0] = .1
        with self.assertRaisesRegex(ModelResolutionError, 'compiler gap'):
            prepare(lagrangian, values)

    def test_negative_kinetics_and_nonzero_vevs_are_rejected(self):
        lagrangian, values = benchmark('8', '0')
        kinetic = next(row for row in lagrangian.operators if row['derivatives'])
        changed = deepcopy(values); changed['coefficients'][kinetic['id']] = -1
        with self.assertRaisesRegex(ModelResolutionError, 'kinetic matrix is not positive definite'):
            prepare(lagrangian, changed)
        changed = deepcopy(values); changed['vevs']['X'] = 1
        with self.assertRaisesRegex(ModelResolutionError, 'zero new-field VEV'):
            prepare(lagrangian, changed)

    def test_resource_limits_raise_during_discovery_without_truncation(self):
        lagrangian, values = benchmark()
        fields = lagrangian._source.models[0]
        row = next(row for row in lagrangian.operators if row['multiplicity'] == 2)
        for limits in ({'max_colour_candidate_networks': 1}, {'max_power_monomials': 1},
                       {'max_output_terms': 1}):
            with self.subTest(limits=limits):
                with self.assertRaises(ResourceLimitError):
                    classify(row, fields, limits)
                changed = deepcopy(values); changed['compiler_limits'] = limits
                with self.assertRaises(ResourceLimitError):
                    prepare(lagrangian, changed)
        self.assertEqual(len(invariant_basis(row, fields, {'max_colour_candidate_networks': 10000})[0]), 2)

    def test_implementation_gap_category_is_distinct_from_physical_input(self):
        lagrangian, values = benchmark('6', '1/3')
        row = next(row for row in lagrangian.operators if classify(row, lagrangian._source.models[0])[1])
        values['coefficients'][row['id']] = [.1] * row['multiplicity']
        with self.assertRaises(CompilerResolutionError) as caught:
            prepare(lagrangian, values)
        self.assertEqual(error_category(caught.exception), 'compiler_failure')

    def test_exact_mixed_sextet_kinetic_and_mass_basis(self):
        lagrangian, values = mixed_benchmark()
        model = prepare(lagrangian, values)
        k = np.array([[1.2, .1-.05j], [.1+.05j, .8]])
        m = np.array([[40000., 2000.-1000j], [2000.+1000j, 90000.]])
        rotation = model.compiler.scalar_transformation
        np.testing.assert_allclose(rotation.conj().T @ k @ rotation, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(rotation.conj().T @ m @ rotation,
                                   np.diag([model.masses['X']**2, model.masses['Y']**2]), atol=1e-10)
        np.testing.assert_allclose(sorted([model.masses['X']**2, model.masses['Y']**2]), eigh(m,k)[0])
        with tempfile.TemporaryDirectory() as folder:
            files, _, _ = model.compiler.write_sources(model, Path(folder), ROOT/'vendor'/'FeynRules')
            source = files[1].read_text()
        # Both original fields occur as superpositions in the same quartic;
        # quadratic-only replacement would leave the interactions incorrect.
        self.assertIn('pmGC1[c1]', source)
        self.assertIn('pmGC2[c1]', source)
        self.assertNotIn('{index}', source)

    def test_complex_neutral_octet_holomorphic_mass_and_kinetic_mixing(self):
        lagrangian, values = benchmark('8', '0')
        selected = {'X^2': -1000.+500j, 'D(X)^2':.05+.025j}
        for row in lagrangian.operators:
            if row['field_content'] in selected:
                values['coefficients'][row['id']] = selected[row['field_content']]
        model = prepare(lagrangian, values)
        k = np.array([[1.1,-.05],[-.05,.9]])
        m = np.array([[42000.,1000.],[1000.,38000.]])
        r = model.compiler.scalar_transformation
        np.testing.assert_allclose(r.T @ k @ r, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(r.T @ m @ r, np.diag([model.masses['X1']**2,model.masses['X2']**2]), atol=1e-10)
        self.assertTrue(all(state.real for state in model.compiler.scalar_states))
        self.assertEqual({state.name for state in model.compiler.scalar_states}, {'x1','x2'})

    def test_all_two_field_scalar_contraction_ranks(self):
        total = 0
        for first, second in [('3','6'),('3','8'),('6','8'),('6','6bar'),('8','8')]:
            def spec(rep):
                return dict(spin='scalar',su2=1,su3=rep,hypercharge='0',boson_kind='complex')
            result = make_lagrangian(2, spec(first), spec(second), show=False)
            for row in result.select('all', show=False).operators:
                if row['derivatives'] or any(p['name'].removesuffix('†') not in ('X','Y','H') for p in row['factors']):
                    continue
                self.assertEqual(len(invariant_basis(row, result.models[0])[0]), row['multiplicity'])
                total += 1
        self.assertEqual(total, 98)

    def test_uniform_field_rescaling_preserves_spectrum_and_all_field_maps(self):
        lagrangian, inputs = mixed_benchmark()
        reference = prepare(lagrangian, deepcopy(inputs))
        for scale in (1e-150, 1e-40, 1e40, 1e150):
            changed = deepcopy(inputs)
            for row in lagrangian.operators:
                count = sum(p['count'] for p in row['factors']
                            if p['name'].removeprefix('D(').removesuffix(')').removesuffix('†') in ('X','Y'))
                factor = scale**(count/2)
                value = changed['coefficients'][row['id']]
                changed['coefficients'][row['id']] = [z*factor for z in value] if isinstance(value,list) else value*factor
            with self.subTest(scale=scale):
                model = prepare(lagrangian, changed)
                for name, mass in reference.masses.items():
                    self.assertAlmostEqual(model.masses[name], mass, places=10)
                for token, mapping in reference.compiler.scalar_maps.items():
                    self.assertEqual(set(model.compiler.scalar_maps[token]),set(mapping))
                    for key, value in mapping.items():
                        self.assertAlmostEqual(model.compiler.scalar_maps[token][key]*math.sqrt(scale), value, places=12)
                # A quartic coefficient and its four raw-field factors must
                # cancel the field reparameterization exactly.
                row = next(row for row in lagrangian.operators if row['field_content']=='X X† Y Y†')
                physical = changed['coefficients'][row['id']][0] * math.prod(
                    model.compiler.scalar_maps[token][0,bar] for token,bar in
                    [('X',False),('X†',True),('Y',False),('Y†',True)])
                expected = inputs['coefficients'][row['id']][0] * math.prod(
                    reference.compiler.scalar_maps[token][0,bar] for token,bar in
                    [('X',False),('X†',True),('Y',False),('Y†',True)])
                np.testing.assert_allclose(physical,expected,rtol=1e-12,atol=0)

    def test_tiny_physical_mass_and_interaction_are_not_pruned(self):
        lagrangian, values = benchmark('8','0','real')
        for row in lagrangian.operators:
            if row['field_content']=='X^2':values['coefficients'][row['id']]=-5e-41
            if row['field_content']=='X^3':values['coefficients'][row['id']]=1e-70
        model = prepare(lagrangian, values)
        self.assertAlmostEqual(model.masses['X']/1e-20,1.,places=12)
        with tempfile.TemporaryDirectory() as folder:
            files,_,_=model.compiler.write_sources(model,Path(folder),ROOT/'vendor'/'FeynRules')
            source=files[1].read_text()
        self.assertIn('Value -> 1*^-70',source)
        self.assertIn('dSUN[',source)


class NativeColourSpeciesRescalingTests(unittest.TestCase):
    def test_independent_extreme_species_units_preserve_mixed_physics(self):
        lagrangian, inputs = mixed_benchmark()
        reference = prepare(lagrangian, deepcopy(inputs))
        reference_rotation = reference.compiler.scalar_transformation
        expected_quartic = -.02 * np.einsum('a,b,c,d->abcd', reference_rotation[0],
            reference_rotation[0].conj(), reference_rotation[1], reference_rotation[1].conj())
        for scales in (np.array([1e-60, 1e60]), np.array([1e60, 1e-60])):
            with self.subTest(scales=scales.tolist()):
                changed = deepcopy(inputs)
                by_name = dict(zip(('X', 'Y'), scales))
                for row in lagrangian.operators:
                    factor = 1.
                    for part in row['factors']:
                        token = part['name']
                        if token.startswith('D('):
                            token = token[2:-1]
                        factor *= by_name.get(token.removesuffix('†'), 1.) ** part['count']
                    coefficient = changed['coefficients'][row['id']]
                    changed['coefficients'][row['id']] = ([value * factor for value in coefficient]
                        if isinstance(coefficient, list) else coefficient * factor)
                actual = prepare(lagrangian, changed)
                np.testing.assert_allclose([actual.masses[name] for name in ('X', 'Y')],
                    [reference.masses[name] for name in ('X', 'Y')], rtol=2e-13)
                normalized = scales[:, None] * actual.compiler.scalar_transformation
                np.testing.assert_allclose(normalized, reference_rotation, rtol=2e-12, atol=1e-14)
                for token in ('X', 'X†', 'Y', 'Y†'):
                    source = token.removesuffix('†')
                    expected_map = reference.compiler.scalar_maps[token]
                    actual_map = actual.compiler.scalar_maps[token]
                    self.assertEqual(set(actual_map), set(expected_map))
                    for key, value in expected_map.items():
                        np.testing.assert_allclose(by_name[source] * actual_map[key], value, rtol=2e-12, atol=1e-14)
                rotation = actual.compiler.scalar_transformation
                quartic = -.02 * np.einsum('a,b,c,d->abcd', rotation[0], rotation[0].conj(),
                                            rotation[1], rotation[1].conj())
                np.testing.assert_allclose(quartic, expected_quartic, rtol=3e-12, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
