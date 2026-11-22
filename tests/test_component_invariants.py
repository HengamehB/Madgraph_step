"""Independent representation and physical regression checks for components."""

from collections import Counter
from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm

from component_invariants import (
    ComponentResourceLimit, InvariantVerificationError, Limits,
    _letter_rep, _su2, _su2_conjugation, _su3, _su3_conjugation,
    component_invariants, gauge_conjugation,
)
from group_theory import su3_character


def letter(name, *, lorentz=(0, 0), weak=0, colour=(0, 0), nf=1,
           derivative=False, fermion=False, count=1):
    return dict(name=name, lorentz=lorentz, weak=weak, colour=colour,
                nf=nf, derivative=derivative, fermion=fermion, count=count)


class ComponentInvariantTests(unittest.TestCase):
    def test_su2_charge_conjugation_and_pseudoreality(self):
        for n in range(7):
            matrix = _su2_conjugation(n).toarray()
            e = _su2(n).raising[0].toarray()
            np.testing.assert_allclose(matrix @ matrix.conjugate(), (-1) ** n * np.eye(n + 1))
            np.testing.assert_allclose(e @ matrix + matrix @ e.T, 0, atol=1e-12)

    def test_su3_charge_conjugation_maps_harmonic_bases(self):
        for p, q in ((1, 0), (2, 0), (1, 1), (2, 1), (2, 2)):
            forward = _su3_conjugation(p, q).toarray()
            reverse = _su3_conjugation(q, p).toarray()
            np.testing.assert_allclose(reverse @ forward.conjugate(), np.eye(len(forward)), atol=1e-12)
            for old, new in zip(_su3(p, q).raising, _su3(q, p).raising):
                np.testing.assert_allclose(new.toarray() @ forward + forward @ old.toarray().T, 0, atol=1e-11)
        data = gauge_conjugation(1, (2, 1))
        self.assertEqual(data["su2_square"], -1)
        self.assertEqual(data["conjugate_colour_dynkin"], [1, 2])

    def test_general_su3_weights_and_lie_algebra(self):
        for p, q in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1), (2, 2), (3, 1)):
            with self.subTest(p=p, q=q):
                rep = _su3(p, q)
                self.assertEqual(Counter(rep.weights), su3_character(p, q))
                e1, e2 = [generator.toarray() for generator in rep.raising]
                for root, e in enumerate((e1, e2)):
                    cartan = np.diag([weight[root] for weight in rep.weights])
                    np.testing.assert_allclose(e @ e.T - e.T @ e, cartan, atol=1e-11)
                e12 = e1 @ e2 - e2 @ e1
                np.testing.assert_allclose(e1 @ e12 - e12 @ e1, 0, atol=1e-11)
                np.testing.assert_allclose(e2 @ e12 - e12 @ e2, 0, atol=1e-11)

    def test_distinct_doublet_epsilon(self):
        result = component_invariants([letter("X", weak=1), letter("Y", weak=1)], 1)
        terms = result["basis"][0]["terms"]
        self.assertEqual(len(terms), 2)
        self.assertAlmostEqual(sum(term["coefficient"] for term in terms), 0)
        self.assertTrue(result["verified"])

    def test_identical_boson_doublet_epsilon_vanishes(self):
        result = component_invariants([letter("H", weak=1, count=2)], 0)
        self.assertEqual(result["basis"], [])

    def test_colour_epsilon_and_bose_statistics(self):
        distinct = [letter(name, colour=(1, 0)) for name in "XYZ"]
        result = component_invariants(distinct, 1)
        self.assertEqual(len(result["basis"][0]["terms"]), 6)
        self.assertEqual(component_invariants([letter("X", colour=(1, 0), count=3)], 0)["basis"], [])

    def test_sextet_cubic_and_adjoint_quadratic(self):
        self.assertEqual(component_invariants([letter("S", colour=(2, 0), count=3)], 1)["multiplicity"], 1)
        self.assertEqual(component_invariants([letter("A", colour=(1, 1), count=2)], 1)["multiplicity"], 1)

    def test_higher_colour_rep_conjugate_bilinear(self):
        result = component_invariants([letter("X", colour=(2, 1)), letter("Xd", colour=(1, 2))], 1)
        self.assertEqual(result["multiplicity"], 1)

    def test_joint_colour_weak_quartic(self):
        result = component_invariants([
            letter("X", weak=1, colour=(1, 0), count=2),
            letter("Xd", weak=1, colour=(0, 1), count=2),
        ], 2)
        self.assertEqual(result["multiplicity"], 2)

    def test_displayed_polynomial_is_invariant_under_finite_transformations(self):
        # This independently checks the conversion from normalized Fock
        # vectors to ordinary polynomial coefficients, including repeated
        # bosonic components; it does not just recheck the SVD residual.
        descriptors = [letter("S", colour=(2, 0), count=3)]
        result = component_invariants(descriptors, 1)
        rep, _ = _letter_rep(descriptors[0], Limits())
        values = np.random.default_rng(23).normal(size=rep.dimension)
        terms = result["basis"][0]["terms"]

        def evaluate(vector):
            return sum(term["coefficient"] * np.prod([
                vector[index - 1] for factor in term["factors"]
                for index in factor["components"]
            ]) for term in terms)

        original = evaluate(values)
        for generator in rep.raising:
            transformed = expm(0.17 * generator.toarray()) @ values
            self.assertAlmostEqual(evaluate(transformed), original, places=11)

    def test_flavour_antisymmetric_charged_scalar_yukawa(self):
        for nf, expected in ((1, 0), (2, 1), (3, 3)):
            result = component_invariants([
                letter("L", lorentz=(1, 0), weak=1, nf=nf, fermion=True, count=2),
                letter("X"),
            ], expected)
            self.assertEqual(result["multiplicity"], expected)

    def test_flavour_symmetric_triplet_yukawa(self):
        result = component_invariants([
            letter("L", lorentz=(1, 0), weak=1, nf=3, fermion=True, count=2),
            letter("Delta", weak=2),
        ], 6)
        self.assertEqual(result["multiplicity"], 6)

    def test_vector_derivative_preserves_full_representation(self):
        result = component_invariants([letter("DV", lorentz=(1, 1), derivative=True, count=2)], 4)
        self.assertEqual(result["multiplicity"], 4)
        result = component_invariants([
            letter("V", lorentz=(1, 1), count=2),
            letter("DV", lorentz=(1, 1), derivative=True),
        ], 2)
        self.assertEqual(result["multiplicity"], 2)

    def test_fermion_kinetic(self):
        result = component_invariants([
            letter("psid", lorentz=(0, 1), fermion=True),
            letter("Dpsi", lorentz=(1, 0), derivative=True, fermion=True),
        ], 1)
        self.assertEqual(result["multiplicity"], 1)

    def test_count_mismatch_and_resource_limits_are_explicit(self):
        with self.assertRaises(InvariantVerificationError):
            component_invariants([letter("X")], 2)
        with self.assertRaises(ComponentResourceLimit):
            component_invariants([letter("X", weak=3)], limits=replace(Limits(), max_letter_components=2))
        with self.assertRaises(ComponentResourceLimit):
            component_invariants([letter("X", weak=1), letter("Y", weak=1)],
                                 limits=replace(Limits(), max_zero_weight_states=1))


if __name__ == "__main__":
    unittest.main()
