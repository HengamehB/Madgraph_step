"""Checks that kind scans create valid, separately selectable physical models."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import unittest

from lagrangian_engine import FieldSpec, expand_models
from lagrangian_maker import make_lagrangian


def particle(spin="scalar", **changes):
    specification = dict(spin=spin, su2=1, su3="1", hypercharge="0")
    specification.update(changes)
    return specification


def kinds(models):
    return [tuple((field.spin, field.kind) for field in model) for model in models]


class AnyKindTests(unittest.TestCase):
    def test_scalar_and_fermion_kind_scans_include_all_real_rep_options(self):
        for representation in ({}, {"su2": 3}, {"su3": "8"}, {"su2": 3, "su3": (1, 1)}):
            with self.subTest(representation=representation):
                scalars = expand_models([particle(boson_kind="any", **representation)])
                fermions = expand_models([particle("fermion", fermion_kind="any", **representation)])
                self.assertEqual([model[0].kind for model in scalars], ["complex", "real"])
                self.assertEqual([model[0].kind for model in fermions], ["weyl", "dirac", "majorana"])

    def test_scans_filter_real_and_majorana_for_each_reality_obstruction(self):
        for representation in ({"hypercharge": "1/2"}, {"su2": 2}, {"su3": "3"}, {"su3": "3bar"}):
            with self.subTest(representation=representation):
                scalars = expand_models([particle(boson_kind="any", **representation)])
                fermions = expand_models([particle("fermion", fermion_kind="any", **representation)])
                vectors = expand_models([particle("vector", boson_kind="any", **representation)])
                self.assertEqual([model[0].kind for model in scalars], ["complex"])
                self.assertEqual([model[0].kind for model in fermions], ["weyl", "dirac"])
                self.assertEqual([model[0].kind for model in vectors], ["complex"])

    def test_explicit_invalid_real_or_majorana_is_still_an_error(self):
        for representation in ({"hypercharge": "1"}, {"su2": 2}, {"su3": "3"}):
            for spin, kind_key, kind in (("scalar", "boson_kind", "real"),
                                         ("vector", "boson_kind", "real"),
                                         ("fermion", "fermion_kind", "majorana")):
                specification = particle(spin, **{kind_key: kind}, **representation)
                with self.subTest(specification=specification), self.assertRaises(ValueError):
                    expand_models([specification])
                with self.subTest(api=specification), self.assertRaises(ValueError):
                    make_lagrangian(1, specification, show=False)

    def test_spin_and_kind_scans_form_cartesian_products_without_changing_defaults(self):
        specification = particle("any", boson_kind="any", fermion_kind="any")
        expected = [("scalar", "complex"), ("scalar", "real"),
                    ("fermion", "weyl"), ("fermion", "dirac"), ("fermion", "majorana"),
                    ("vector", "complex"), ("vector", "real")]
        self.assertEqual(kinds(expand_models([specification])), [(item,) for item in expected])
        # Test the 49-model expansion without enumerating 49 operator catalogues.
        models = expand_models([specification, specification])
        self.assertEqual(kinds(models), list(product(expected, repeat=2)))
        self.assertTrue(all([field.name for field in model] == ["X", "Y"] for model in models))
        self.assertEqual(kinds(expand_models([particle("any")])),
                         [(("scalar", "complex"),), (("fermion", "weyl"),), (("vector", "complex"),)])

    def test_scanned_kinds_keep_charges_and_representations_fixed(self):
        x = particle("any", su2=2, su3=[1, 0], hypercharge="1/6",
                     boson_kind="any", fermion_kind="any")
        y = particle("fermion", su2=3, su3="8", fermion_kind="any")
        original = deepcopy((x, y))
        models = expand_models([x, y])
        self.assertEqual(len(models), 12)
        for model in models:
            self.assertEqual((model[0].su2, model[0].su3, model[0].hypercharge),
                             (2, (1, 0), Fraction(1, 6)))
            self.assertEqual((model[1].su2, model[1].su3, model[1].hypercharge),
                             (3, (1, 1), Fraction(0)))
        self.assertEqual((x, y), original)
        for key in ("su2", "su3", "hypercharge"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                expand_models([particle(boson_kind="any", **{key: "any"})])

    def test_engine_normalizes_case_and_whitespace_without_mutating_inputs(self):
        x = particle(" ScAlAr ", boson_kind=" AnY ")
        y = particle(" FeRmIoN ", fermion_kind=" aNy ")
        original = deepcopy((x, y))
        models = expand_models([x, y])
        self.assertEqual([tuple(field.kind for field in model) for model in models],
                         list(product(("complex", "real"), ("weyl", "dirac", "majorana"))))
        self.assertEqual((x, y), original)

    def test_concrete_fields_can_be_combined_with_kind_scans(self):
        concrete = FieldSpec("X", "scalar", 1, (0, 0), 0, "real")
        models = expand_models([concrete, particle("fermion", fermion_kind="any")])
        self.assertEqual(len(models), 3)
        self.assertTrue(all(model[0] is concrete for model in models))
        self.assertEqual([model[1].kind for model in models], ["weyl", "dirac", "majorana"])

    def test_six_catalogues_are_independent_and_selectable(self):
        x = particle(" ScAlAr ", boson_kind=" AnY ")
        y = particle(" FERmion ", fermion_kind=" aNy ")
        original = deepcopy((x, y))
        result = make_lagrangian(2, x, y, sm_generations=1, show=False)
        self.assertEqual(len(result.models), 6)
        self.assertEqual(len(result.catalogues), 6)
        self.assertEqual(len({catalogue["label"] for catalogue in result.catalogues}), 6)
        self.assertEqual((x, y), original)
        with self.assertRaises(ValueError):
            result.select("all", show=False)
        for scenario, (model, catalogue) in enumerate(zip(result.models, result.catalogues), 1):
            with self.subTest(scenario=scenario):
                self.assertEqual(catalogue["index"], scenario)
                for field in model:
                    self.assertIn(f"{field.name}: {field.kind} {field.spin}", catalogue["label"])
                rows = {row["field_content"]: row for row in catalogue["operators"]}
                self.assertTrue(catalogue["pairs"])
                self.assertTrue(rows)
                # Changing type changes the field content: complex scalars
                # have X†, and Dirac fermions have two independent Weyl fields.
                self.assertEqual(any("X†" in text for text in rows), model[0].kind == "complex")
                self.assertEqual(any("Y_Rc" in text for text in rows), model[1].kind == "dirac")
                if model[1].kind != "dirac":
                    self.assertEqual(rows["Y^2"]["dimension"], 3)
                    self.assertEqual(rows["H L Y"]["dimension"], 4)
                selected = result.select("all", scenario=scenario, show=False)
                self.assertEqual(selected.scenario, scenario)
                self.assertEqual(selected.operators, tuple(catalogue["operators"]))
                self.assertEqual(selected.selected_ids, tuple(row["id"] for row in catalogue["operators"]))
                self.assertTrue(selected.latex)


if __name__ == "__main__":
    unittest.main()
