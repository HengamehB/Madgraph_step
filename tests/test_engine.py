"""Physics regressions for the SM-extension catalogue.

Run with ``python -m unittest test_engine -v``.  These tests use known
renormalizable interactions, flavour symmetries and dimension sum rules.
Derivative multiplicities are deliberately checked before IBP/EOM reduction.
"""

from collections import Counter
from fractions import Fraction
from itertools import product
import unittest

from lagrangian_engine import (
    FieldSpec,
    enumerate_operators,
    expand_models,
    make_letters,
    new_letters,
    pair_channels,
    parse_charge,
    parse_su3,
)


def content(row):
    return Counter({factor["name"]: factor["count"] for factor in row["factors"]})


def lookup(rows, expected, *, include_hc=False):
    expected = Counter(expected)
    found = [row for row in rows if content(row) == expected]
    if not found and include_hc:
        def conjugate(name):
            if name.startswith("D("):
                return "D(" + conjugate(name[2:-1]) + ")"
            return name[:-1] if name.endswith("†") else name + "†"
        hc = Counter({conjugate(name): n for name, n in expected.items()})
        found = [row for row in rows if content(row) == hc]
    if len(found) != 1:
        raise AssertionError(f"Expected one row for {dict(expected)}; found {len(found)}")
    return found[0]


class EngineTests(unittest.TestCase):
    def test_real_scalar_singlet_complete_known_catalogue(self):
        model = [FieldSpec("X", "scalar", 1, (0, 0), 0, "real")]
        rows = enumerate_operators(model)
        expected = {
            (1, "X"), (2, "X^2"), (3, "X^3"), (3, "H H† X"),
            (4, "X^4"), (4, "H H† X^2"), (4, "D(X)^2"),
        }
        self.assertEqual({(r["dimension"], r["field_content"]) for r in rows}, expected)
        self.assertTrue(all(r["multiplicity"] == 1 for r in rows))

    def test_charged_scalar_flavour_antisymmetry(self):
        model = [FieldSpec("X", "scalar", 1, (0, 0), 1)]
        for nf in (1, 2, 3, 4):
            rows = enumerate_operators(model, sm_generations=nf)
            matches = [r for r in rows if content(r) == Counter({"L": 2, "X": 1})]
            if nf == 1:
                self.assertEqual(matches, [])
            else:
                self.assertEqual(matches[0]["multiplicity"], nf * (nf - 1) // 2)

    def test_triplet_and_doubly_charged_flavour_symmetry(self):
        triplet = [FieldSpec("X", "scalar", 3, (0, 0), 1)]
        doubly_charged = [FieldSpec("X", "scalar", 1, (0, 0), 2)]
        for nf in (1, 3):
            expected = nf * (nf + 1) // 2
            rows = enumerate_operators(triplet, sm_generations=nf)
            self.assertEqual(lookup(rows, {"L": 2, "X": 1})["multiplicity"], expected)
            self.assertEqual(lookup(rows, {"H": 2, "X†": 1})["dimension"], 3)
            rows = enumerate_operators(doubly_charged, sm_generations=nf)
            self.assertEqual(
                lookup(rows, {"ec": 2, "X†": 1}, include_hc=True)["multiplicity"], expected
            )

    def test_coloured_doublet_quartics_and_yukawa(self):
        rows = enumerate_operators([FieldSpec("X", "scalar", 2, (1, 0), "1/6")])
        self.assertEqual(lookup(rows, {"X": 2, "X†": 2})["multiplicity"], 2)
        self.assertEqual(lookup(rows, {"H": 1, "H†": 1, "X": 1, "X†": 1})["multiplicity"], 2)
        self.assertEqual(lookup(rows, {"L": 1, "X": 1, "dc": 1})["multiplicity"], 9)

    def test_sterile_weyl_mass_neutrino_portal_and_kinetic(self):
        rows = enumerate_operators([FieldSpec("X", "fermion", 1, (0, 0), 0)])
        self.assertEqual(len(rows), 3)
        mass = lookup(rows, {"X": 2})
        self.assertEqual((mass["dimension"], mass["multiplicity"]), (3, 1))
        portal = lookup(rows, {"H": 1, "L": 1, "X": 1})
        self.assertEqual((portal["dimension"], portal["multiplicity"]), (4, 3))
        kinetic = lookup(rows, {"D(X)": 1, "X†": 1})
        self.assertEqual((kinetic["dimension"], kinetic["derivatives"], kinetic["multiplicity"]), (4, 1, 1))

    def test_bosonic_antisymmetric_gauge_contractions_vanish(self):
        # epsilon_abc X^a X^b X^c is zero for one commuting triplet.
        for su2, su3 in ((3, (0, 0)), (1, (1, 0))):
            rows = enumerate_operators([FieldSpec("X", "scalar", su2, su3, 0)])
            self.assertFalse(any(content(r) == Counter({"X": 3}) for r in rows))
        octet_rows = enumerate_operators([FieldSpec("X", "scalar", 1, (1, 1), 0, "real")])
        self.assertEqual(lookup(octet_rows, {"X": 3})["multiplicity"], 1)

    def test_vector_derivative_counts_before_ibp(self):
        model = [FieldSpec("X", "vector", 1, (0, 0), 0, "real")]
        rows = enumerate_operators(model)
        self.assertEqual(lookup(rows, {"D(X)": 2})["multiplicity"], 4)
        self.assertEqual(lookup(rows, {"D(X)": 1, "X": 2})["multiplicity"], 2)
        self.assertEqual(lookup(rows, {"B+": 1, "D(X)": 1})["multiplicity"], 1)
        self.assertEqual(lookup(rows, {"L": 1, "L†": 1, "X": 1})["multiplicity"], 9)
        mixing_model = model + [FieldSpec("Y", "vector", 1, (0, 0), 0, "real")]
        mixing = enumerate_operators(mixing_model)
        self.assertEqual(lookup(mixing, {"D(X)": 1, "D(Y)": 1})["multiplicity"], 4)
        self.assertEqual(lookup(mixing, {"X": 1, "Y": 1})["dimension"], 2)

    def test_scalar_kinetic_mixing_with_higgs(self):
        model = [FieldSpec("X", "scalar", 2, (0, 0), "1/2")]
        rows = enumerate_operators(model)
        self.assertEqual(lookup(rows, {"D(H)": 1, "D(X†)": 1})["multiplicity"], 1)
        self.assertEqual(lookup(rows, {"H": 1, "X†": 1})["dimension"], 2)

    def test_exact_dimension_and_catalogue_consistency(self):
        model = [FieldSpec("X", "scalar", 1, (0, 0), 0, "real")]
        rows = enumerate_operators(model)
        exact = enumerate_operators(model, exact_dimension=4)
        self.assertEqual(
            {r["field_content"] for r in exact},
            {r["field_content"] for r in rows if r["dimension"] == 4},
        )
        for bound in (1, 2, 3, 4):
            bounded = enumerate_operators(model, max_dimension=bound)
            self.assertEqual(
                {r["field_content"] for r in bounded},
                {r["field_content"] for r in rows if r["dimension"] <= bound},
            )
        with self.assertRaises(ValueError):
            enumerate_operators(model, max_dimension=5)

    def test_spin_any_expands_nine_models_and_fixes_charges(self):
        specification = dict(spin="any", su2=2, su3="3bar", hypercharge="1/6")
        models = expand_models([dict(specification, name="X"), dict(specification, name="Y")])
        self.assertEqual(len(models), 9)
        self.assertEqual({tuple(f.spin for f in m) for m in models}, set(product(("scalar", "fermion", "vector"), repeat=2)))
        self.assertTrue(all((f.su2, f.su3, f.hypercharge) == (2, (0, 1), Fraction(1, 6)) for m in models for f in m))
        self.assertEqual(len(expand_models([specification])), 3)
        with self.assertRaises(ValueError):
            parse_charge("any")
        with self.assertRaises(ValueError):
            parse_su3("any")

    def test_pair_channel_dimensions_and_identical_statistics(self):
        models = [
            [FieldSpec("X", "scalar", 2, (1, 0), 0)],
            [FieldSpec("X", "fermion", 2, (1, 0), 0)],
            [FieldSpec("X", "vector", 1, (0, 0), 0, "real"),
             FieldSpec("Y", "vector", 1, (0, 0), 0, "real")],
        ]
        for model in models:
            letters = {f.name: f for f in new_letters(model)}
            totals = Counter()
            for row in pair_channels(model):
                j_left, j_right = (Fraction(v.strip()) for v in row["lorentz"].strip("()").split(","))
                colour_dim = int(row["su3"].split()[0])
                dimension = (2 * j_left + 1) * (2 * j_right + 1) * row["su2"] * colour_dim
                totals[row["pair"]] += row["multiplicity"] * dimension
            for pair, actual in totals.items():
                a, b = (letters[name] for name in pair.split(" × "))
                def dim(letter):
                    p, q = letter.colour
                    return (letter.lorentz[0] + 1) * (letter.lorentz[1] + 1) * (letter.weak + 1) * (p + 1) * (q + 1) * (p + q + 2) // 2
                da, db = dim(a), dim(b)
                expected = da * db if a != b else da * (da - 1 if a.fermion else da + 1) // 2
                self.assertEqual(actual, expected, pair)
        vector_channels = pair_channels(models[-1])
        self.assertEqual(len([r for r in vector_channels if r["pair"] == "X × X"]), 2)
        self.assertEqual(len([r for r in vector_channels if r["pair"] == "X × Y"]), 4)

    def test_every_reported_row_is_neutral_and_contains_new_fields(self):
        model = [FieldSpec("X", "scalar", 2, (1, 0), "1/6"),
                 FieldSpec("Y", "fermion", 1, (0, 0), 0)]
        letters = {f.name: f for f in make_letters(model)}
        rows = enumerate_operators(model)
        self.assertEqual(len({r["field_content"] for r in rows}), len(rows))
        self.assertTrue(rows)
        for row in rows:
            factors = [(letters[f["name"]], f["count"]) for f in row["factors"]]
            self.assertEqual(sum(f.charge * n for f, n in factors), 0)
            self.assertEqual(sum(f.dim2 * n for f, n in factors), 2 * row["dimension"])
            self.assertTrue(any(f.new for f, _ in factors))
            self.assertGreater(row["multiplicity"], 0)
            self.assertLessEqual(row["dimension"], 4)

    def test_exact_charge_and_reality_validation(self):
        self.assertEqual(parse_charge("-2/3"), Fraction(-2, 3))
        for charge in (0.5, True, "0.5", "1/0"):
            with self.assertRaises(ValueError):
                parse_charge(charge)
        with self.assertRaises(ValueError):
            FieldSpec("X", "scalar", 2, (0, 0), 0, "real")
        with self.assertRaises(ValueError):
            FieldSpec("X", "fermion", 1, (1, 0), 0, "majorana")
        with self.assertRaises(ValueError):
            FieldSpec("X", "vector", 1, (0, 0), 1, "real")


if __name__ == "__main__":
    unittest.main()
