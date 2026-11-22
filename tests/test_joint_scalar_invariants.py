"""Exact catalogue counts and independent finite-group tests of joint scalars."""
from copy import deepcopy
from fractions import Fraction
import unittest

import numpy as np
from scipy.linalg import expm

from component_invariants import InvariantVerificationError
from generic_color_compiler import _REPS
from joint_scalar_invariants import (invariant_basis, kinetic_basis, basis_labels,
    basis_fingerprint, conjugate_polynomial, _basis, _ordered_weak,
    _exact_weak_coefficient, _limits)
from lagrangian_maker import make_lagrangian
from test_colour_export_validation import reference_tensors
from workflow_errors import ResourceLimitError


def catalogue(weak=2, colour='6', charge='1/2', kind='complex'):
    result = make_lagrangian(1, dict(spin='scalar', su2=weak, su3=colour,
                            hypercharge=charge, boson_kind=kind), show=False)
    return result.models[0], result.select('all', show=False).operators


def row_named(rows, content):
    return next(row for row in rows if row['field_content'] == content)


def weak_generators(dimension):
    plus = np.zeros((dimension, dimension))
    for index in range(1, dimension):
        plus[index - 1, index] = np.sqrt(index * (dimension - index))
    return ((plus + plus.T) / 2, (plus - plus.T) / (2j),
            np.diag(np.arange((dimension - 1) / 2, -dimension / 2, -1)))


def evaluate(poly, components):
    value = 0j
    for monomial, coefficient in poly.items():
        term = coefficient
        for token, weak, colour in monomial:
            term *= components[token][weak, colour]
        value += term
    return value


def group_check(test, row, fields):
    """Transform full weak and colour fields before evaluating each polynomial."""
    _, polynomials = invariant_basis(row, fields)
    rng = np.random.default_rng(725)
    reference = reference_tensors()
    colour_generators = {1: np.zeros((8, 1, 1)), 3: reference['T'],
                         6: reference['T6'], 8: -1j * reference['f']}
    for dimension in (3, 6):
        colour_generators[-dimension] = -colour_generators[dimension].conj()
    angles_weak, angles_colour = rng.normal(size=3), rng.normal(size=8)
    original, rotated = {}, {}
    descriptors = [('H', 2, 1, False)] + [(f.name, f.su2, _REPS[f.su3], f.kind == 'real') for f in fields]
    for name, weak, colour, real in descriptors:
        raw = rng.normal(size=(weak, abs(colour))) + 1j * rng.normal(size=(weak, abs(colour)))
        if real:
            c = np.fliplr(np.diag((-1.) ** np.arange(weak)))
            raw = (raw + c @ raw.conj()) / np.sqrt(2)
        uweak = expm(1j * sum(angle * matrix for angle, matrix in zip(angles_weak, weak_generators(weak))))
        ucolour = expm(1j * np.einsum('a,aij->ij', angles_colour, colour_generators[colour]))
        original[name] = raw
        rotated[name] = uweak @ raw @ ucolour.T
        original[name + '†'] = raw.conj()
        rotated[name + '†'] = rotated[name].conj()
    for index, polynomial in enumerate(polynomials):
        with test.subTest(content=row['field_content'], invariant=index):
            before, after = evaluate(polynomial, original), evaluate(polynomial, rotated)
            np.testing.assert_allclose(after, before, rtol=2e-10, atol=2e-9)
            if row['field_content'] == row['conjugate_content']:
                test.assertAlmostEqual(before.imag, 0., delta=2e-9)
                conjugated = conjugate_polynomial(polynomial, fields)
                all_keys = set(polynomial) | set(conjugated)
                test.assertLess(max((abs(polynomial.get(m, 0) - conjugated.get(m, 0)) for m in all_keys), default=0), 1e-9)


class JointScalarInvariantTests(unittest.TestCase):
    def test_joint_bose_counts_exceed_separately_reduced_counts(self):
        # A weak doublet alone has one norm quartic. Multiplying that by the
        # separately Bose-reduced colour sextet/octet bases gives only 2/3,
        # whereas the correct joint scalar spaces have 3/6 invariants.
        for colour, expected in (('3', 2), ('6', 3), ('8', 6)):
            fields, rows = catalogue(colour=colour)
            row = row_named(rows, 'X^2 X†^2')
            basis, polynomials = invariant_basis(row, fields)
            self.assertEqual(row['multiplicity'], expected)
            self.assertEqual(len(basis), expected)
            self.assertEqual(len(polynomials), expected)
            group_check(self, row, fields)

    def test_full_higgs_components_and_non_norm_higgs_contractions(self):
        fields, rows = catalogue(colour='8')
        for content, expected in (('H H† X X†', 2), ('H X X†^2', 2), ('H^2 X†^2', 1)):
            row = row_named(rows, content)
            basis, polynomials = invariant_basis(row, fields)
            self.assertEqual(len(basis), expected)
            indices = {weak for poly in polynomials for monomial in poly for token, weak, _ in monomial if token.startswith('H')}
            self.assertEqual(indices, {0, 1})
            group_check(self, row, fields)

    def test_real_weak_triplet_adjoint_and_nontrivial_reality(self):
        fields, rows = catalogue(weak=3, colour='8', charge='0', kind='real')
        for content, expected in (('X^2', 1), ('X^3', 1), ('X^4', 3)):
            row = row_named(rows, content)
            self.assertEqual(len(invariant_basis(row, fields)[0]), expected)
            group_check(self, row, fields)
        basis, polynomials = kinetic_basis(row_named(rows, 'D(X)^2'), fields)
        self.assertEqual(len(basis), 1)
        self.assertTrue(polynomials[0])

    def test_antirepresentations_and_mixed_colour_cubic(self):
        for colour in ('3bar', '6bar'):
            fields, rows = catalogue(colour=colour, charge='-1/2')
            group_check(self, row_named(rows, 'X^2 X†^2'), fields)
            group_check(self, row_named(rows, 'H H† X X†'), fields)
        # X_i X_j Y_A uses an antisextet Y, requiring the opposite K6 tensor
        # orientation from a sextet multiplied by two antitriplets.
        x = dict(spin='scalar', su2=2, su3='3', hypercharge='1/3', boson_kind='complex')
        y = dict(spin='scalar', su2=3, su3='6bar', hypercharge='-2/3', boson_kind='complex')
        result = make_lagrangian(2, x, y, show=False)
        row = row_named(result.select('all', show=False).operators, 'X^2 Y')
        basis, _ = invariant_basis(row, result.models[0])
        self.assertEqual(len(basis), 1)
        group_check(self, row, result.models[0])

    def test_kinetic_normalization_and_source_substitution(self):
        fields, rows = catalogue(colour='6')
        basis, polynomials = kinetic_basis(row_named(rows, 'D(X) D(X†)'), fields)
        expected = {tuple(sorted((('X', weak, colour), ('X†', weak, colour)))): 1.
                    for weak in range(2) for colour in range(6)}
        self.assertEqual(polynomials[0], expected)
        symbols = {(token, weak): ('xbar' if token.endswith('†') else 'x') + str(weak) + '[{index}]'
                   for token in ('X', 'X†') for weak in range(2)}
        expression = basis[0].render(symbols)
        self.assertNotIn('{index}', expression)
        self.assertIn('IndexDelta[Index[Sextet,c1],Index[Sextet,c2]]', expression)
        self.assertIn('x0[c1]', expression)
        self.assertIn('xbar1[c2]', expression)

    def test_exact_count_failure_and_explicit_budgets(self):
        fields, rows = catalogue(colour='6')
        row = row_named(rows, 'X^2 X†^2')
        wrong = dict(row, multiplicity=row['multiplicity'] + 1)
        with self.assertRaisesRegex(InvariantVerificationError, 'exact catalogue'):
            invariant_basis(wrong, fields)
        for budget in ('max_letter_components', 'max_weight_combinations', 'max_zero_weight_states',
                       'max_matrix_entries', 'max_matrix_nonzeros', 'max_power_monomials',
                       'max_colour_candidate_networks', 'max_output_terms'):
            with self.subTest(budget=budget), self.assertRaises(ResourceLimitError):
                invariant_basis(row, fields, {budget: 1})
        self.assertEqual(len(invariant_basis(row, fields)[0]), 3)

    def test_weak_group_factors_render_as_exact_rational_roots(self):
        for value, expected in ((1., '1'), (-.5, '-1/2'),
                                (np.sqrt(2), 'Sqrt[2]'),
                                (-np.sqrt(2 / 3), '-Sqrt[2/3]')):
            self.assertEqual(_exact_weak_coefficient(value), expected)
        for dimensions in ((2, 2, 3), (3, 3, 3, 3), (4, 4, 4, 4),
                           (5, 5, 3, 3), (6, 6, 2, 2)):
            for invariant in _ordered_weak(dimensions, _limits(None)):
                for _, value in invariant:
                    rendered = _exact_weak_coefficient(value)
                    self.assertNotIn('.', rendered)
                    # Parse only the exact syntax emitted above, without eval.
                    sign = -1 if rendered.startswith('-') else 1
                    magnitude = rendered.lstrip('-')
                    recovered = (np.sqrt(float(Fraction(magnitude[5:-1])))
                                 if magnitude.startswith('Sqrt[')
                                 else float(Fraction(magnitude)))
                    self.assertAlmostEqual(sign * recovered, value, delta=abs(value) * 5e-14)
        with self.assertRaises(InvariantVerificationError):
            _exact_weak_coefficient(float('nan'))
        with self.assertRaises(InvariantVerificationError):
            _exact_weak_coefficient(np.pi)

    def test_fingerprint_and_basis_are_deterministic(self):
        fields, rows = catalogue(colour='6')
        row = row_named(rows, 'X^2 X†^2')
        before = basis_labels(row, fields)
        fingerprint = basis_fingerprint(row, fields)
        _basis.cache_clear(); _ordered_weak.cache_clear()
        self.assertEqual(before, basis_labels(row, fields))
        self.assertEqual(fingerprint, basis_fingerprint(row, fields))
        self.assertEqual(len(fingerprint), 64)


if __name__ == '__main__':
    unittest.main()
