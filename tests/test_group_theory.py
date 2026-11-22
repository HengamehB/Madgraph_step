"""Independent identities/regressions for the exact group-theory engine.

Run ``python -m unittest test_group_theory -v`` in this directory.
"""

from collections import Counter
from itertools import combinations, combinations_with_replacement, product
from math import comb
import unittest

from group_theory import (
    character_dimension,
    conjugate_character,
    decompose_character,
    decompose_su2,
    decompose_su3,
    direct_product,
    exterior_power,
    field_character,
    singlet_multiplicity,
    singlet_product_multiplicity,
    su2_character,
    su2_dimension,
    su2_tensor_product,
    su3_character,
    su3_dimension,
    su3_tensor_product,
    symmetric_power,
    tensor_product,
)


class CharacterTests(unittest.TestCase):
    def test_irrep_dimensions_and_duals(self):
        for n in range(12):
            self.assertEqual(character_dimension(su2_character(n)), su2_dimension(n))
            self.assertEqual(su2_character(n), conjugate_character(su2_character(n)))
        for p, q in product(range(7), repeat=2):
            char = su3_character(p, q)
            self.assertEqual(character_dimension(char), su3_dimension(p, q))
            self.assertEqual(conjugate_character(char), su3_character(q, p))

    def test_known_colour_products(self):
        self.assertEqual(su3_tensor_product((1, 0), (1, 0)), {(2, 0): 1, (0, 1): 1})
        self.assertEqual(su3_tensor_product((1, 0), (0, 1)), {(1, 1): 1, (0, 0): 1})
        self.assertEqual(
            su3_tensor_product((1, 1), (1, 1)),
            {(0, 0): 1, (1, 1): 2, (3, 0): 1, (0, 3): 1, (2, 2): 1},
        )

    def test_tensor_decomposition_dimension(self):
        labels = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
        for left, right in product(labels, repeat=2):
            decomposition = su3_tensor_product(left, right)
            self.assertEqual(
                sum(m * su3_dimension(*rep) for rep, m in decomposition.items()),
                su3_dimension(*left) * su3_dimension(*right),
            )
        for n, m in product(range(6), repeat=2):
            self.assertEqual(
                decompose_su2(tensor_product(su2_character(n), su2_character(m))),
                su2_tensor_product(n, m),
            )

    def test_powers_against_basis_enumeration(self):
        # The adjoint has a repeated zero weight. Enumerating basis vectors is
        # independent of the Newton-identity implementation and catches the
        # common mistake of treating a repeated weight as a single state.
        char = su3_character(1, 1)
        basis = [w for w, m in char.items() for _ in range(m)]
        for k in range(5):
            for exterior, choose in (
                (False, combinations_with_replacement),
                (True, combinations),
            ):
                expected = Counter()
                for indices in choose(range(len(basis)), k):
                    expected[tuple(sum(basis[i][j] for i in indices) for j in range(2))] += 1
                actual = exterior_power(char, k) if exterior else symmetric_power(char, k)
                self.assertEqual(actual, expected)
                expected_dimension = comb(8, k) if exterior else comb(8 + k - 1, k)
                self.assertEqual(character_dimension(actual), expected_dimension)

    def test_colour_powers(self):
        fundamental = su3_character(1, 0)
        self.assertEqual(decompose_su3(symmetric_power(fundamental, 3)), {(3, 0): 1})
        self.assertEqual(decompose_su3(exterior_power(fundamental, 2)), {(0, 1): 1})
        self.assertEqual(decompose_su3(exterior_power(fundamental, 3)), {(0, 0): 1})
        self.assertEqual(exterior_power(fundamental, 4), {})

    def test_weyl_projection(self):
        for n in range(8):
            self.assertEqual(singlet_multiplicity(su2_character(n), ("SU2",)), int(n == 0))
        for p, q in product(range(5), repeat=2):
            self.assertEqual(singlet_multiplicity(su3_character(p, q), ("SU3",)), int(p == q == 0))
        adjoint = su3_character(1, 1)
        self.assertEqual(singlet_multiplicity(tensor_product(adjoint, adjoint), ("SU3",)), 1)
        self.assertEqual(singlet_multiplicity(tensor_product(adjoint, adjoint, adjoint), ("SU3",)), 2)

    def test_identical_field_statistics(self):
        higgs = field_character(weak=1)
        self.assertEqual(singlet_multiplicity(symmetric_power(higgs, 2)), 0)
        majorana = field_character(lorentz=(1, 0))
        self.assertEqual(singlet_multiplicity(exterior_power(majorana, 2)), 1)
        weak_doublet_fermion = field_character(lorentz=(1, 0), weak=1)
        self.assertEqual(singlet_multiplicity(exterior_power(weak_doublet_fermion, 2)), 0)
        weak_triplet_fermion = field_character(lorentz=(1, 0), weak=2)
        self.assertEqual(singlet_multiplicity(exterior_power(weak_triplet_fermion, 2)), 1)
        # Two flavours allow the antisymmetric flavour contraction absent
        # for a single identical Weyl doublet.
        two_flavours = Counter({w: 2 * m for w, m in weak_doublet_fermion.items()})
        self.assertEqual(singlet_multiplicity(exterior_power(two_flavours, 2)), 1)

    def test_product_projection_matches_full_convolution(self):
        chars = [
            field_character(lorentz=(1, 0), weak=1, colour=(1, 0)),
            field_character(lorentz=(1, 0), colour=(0, 1)),
            field_character(weak=1),
        ]
        self.assertEqual(singlet_product_multiplicity(chars), 1)
        for k in range(1, 4):
            factors = chars[:k]
            self.assertEqual(
                singlet_product_multiplicity(factors),
                singlet_multiplicity(tensor_product(*factors)),
            )
        self.assertEqual(singlet_product_multiplicity([]), 1)
        self.assertEqual(singlet_product_multiplicity([Counter()]), 0)

    def test_general_decomposition_and_statistics(self):
        char = field_character(lorentz=(1, 0), weak=1, colour=(1, 0))
        antisymmetric = exterior_power(char, 2)
        reps = decompose_character(antisymmetric)
        reconstructed = Counter()
        for (l, r, w, p, q), multiplicity in reps.items():
            irrep = field_character((l, r), w, (p, q))
            reconstructed.update({weight: multiplicity * m for weight, m in irrep.items()})
        self.assertEqual(reconstructed, antisymmetric)
        self.assertEqual(
            decompose_character(direct_product(su2_character(2), su3_character(1, 1)), ("SU2", "SU3")),
            {(2, 1, 1): 1},
        )

    def test_cache_returns_are_mutation_safe(self):
        char = su3_character(1, 0)
        char[(100, 100)] = 200
        self.assertEqual(character_dimension(su3_character(1, 0)), 3)
        field = field_character()
        field.clear()
        self.assertEqual(character_dimension(field_character()), 1)


if __name__ == "__main__":
    unittest.main()
