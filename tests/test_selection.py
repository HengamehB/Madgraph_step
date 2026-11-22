"""Integration checks for assembling a Lagrangian from generated families."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

from lagrangian_maker import make_lagrangian


def scalar(**changes):
    value = dict(spin="scalar", su2=1, su3="1", hypercharge="0", boson_kind="real")
    value.update(changes)
    return value


def majorana():
    return dict(spin="fermion", su2=1, su3="1", hypercharge="0", fermion_kind="majorana")


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = make_lagrangian(2, scalar(), majorana(), show=False)
        cls.rows = cls.result.catalogues[0]["operators"]
        cls.by_content = {row["field_content"]: row for row in cls.rows}
        cls.ids = tuple(row["id"] for row in cls.rows)

    def test_select_all_preserves_catalogue_and_has_one_coefficient_family_per_row(self):
        selected = self.result.select("ALL", show=False)
        self.assertEqual(selected.scenario, 1)
        self.assertEqual(selected.selected_ids, self.ids)
        self.assertIsInstance(selected.operators, tuple)
        self.assertIsInstance(selected.coefficients, tuple)
        self.assertEqual(selected.operators, tuple(self.rows))
        self.assertEqual(len(selected.coefficients), len(self.rows))
        self.assertEqual({coefficient["operator_id"] for coefficient in selected.coefficients}, set(self.ids))
        for row, coefficient in zip(selected.operators, selected.coefficients):
            self.assertEqual(coefficient["dimension"], 4 - row["dimension"])
            self.assertEqual(coefficient["count"], row["multiplicity"])
            self.assertTrue(coefficient["latex"])
        self.assertIn("SM", selected.latex)

    def test_subset_is_in_catalogue_order_without_duplicates_or_input_mutation(self):
        requested = [self.ids[-1], self.ids[0], self.ids[-1]]
        before = requested[:]
        selected = self.result.select(requested, show=False)
        self.assertEqual(selected.selected_ids, (self.ids[0], self.ids[-1]))
        self.assertEqual(requested, before)
        self.assertEqual(len(selected.coefficients), 2)

    def test_single_id_and_empty_selection(self):
        selected = self.result.select(self.ids[0], show=False)
        self.assertEqual(selected.selected_ids, (self.ids[0],))
        empty = self.result.select([], show=False)
        self.assertEqual(empty.selected_ids, ())
        self.assertEqual(empty.operators, ())
        self.assertEqual(empty.coefficients, ())
        self.assertIn("SM", empty.latex)
        self.assertNotIn("+", empty.latex)

    def test_majorana_mass_has_complex_coefficient_and_hermitian_conjugate(self):
        row = self.by_content["Y^2"]
        selected = self.result.select(row["id"], show=False)
        coefficient, = selected.coefficients
        self.assertEqual((coefficient["kind"], coefficient["count"], coefficient["dimension"]),
                         ("complex", 1, 1))
        self.assertIn(coefficient["latex"], selected.latex)
        self.assertEqual(selected.latex.count(r"\mathrm{h.c.}"), 1)

    def test_neutrino_portal_keeps_three_independent_flavour_couplings(self):
        row = self.by_content["H L Y"]
        selected = self.result.select(row["id"], show=False)
        coefficient, = selected.coefficients
        self.assertEqual((coefficient["kind"], coefficient["count"], coefficient["dimension"]),
                         ("complex", 3, 0))
        self.assertIn(r"\sum", selected.latex)
        self.assertIn("3", selected.latex)
        self.assertIn(coefficient["latex"], selected.latex)
        self.assertEqual(selected.latex.count(r"\mathrm{h.c.}"), 1)

    def test_real_scalar_mass_uses_real_coefficient_without_double_counting_hc(self):
        selected = self.result.select(self.by_content["X^2"]["id"], show=False)
        coefficient, = selected.coefficients
        self.assertEqual((coefficient["kind"], coefficient["count"], coefficient["dimension"]),
                         ("real", 1, 2))
        self.assertNotIn(r"^{*}", selected.latex)
        self.assertNotIn(r"^{\dagger}", selected.latex)
        self.assertNotIn(r"\mathrm{h.c.}", selected.latex)

    def test_add_returns_new_selection_without_changing_original(self):
        original = self.result.select(self.ids[0], show=False)
        before = deepcopy((original.selected_ids, original.operators, original.coefficients, original.latex))
        extended = original.add([self.ids[-1], self.ids[0]], show=False)
        self.assertIsNot(extended, original)
        self.assertEqual(extended.selected_ids, (self.ids[0], self.ids[-1]))
        self.assertEqual((original.selected_ids, original.operators, original.coefficients, original.latex), before)
        self.assertEqual(extended.scenario, original.scenario)
        self.assertEqual(extended.add("all", show=False).selected_ids, self.ids)

    def test_selected_rows_are_independent_copies(self):
        original = deepcopy(self.rows[0])
        selected = self.result.select(self.ids[0], show=False)
        selected.operators[0]["notes"] = "changed locally"
        selected.operators[0]["factors"][0]["name"] = "changed locally"
        self.assertEqual(self.rows[0], original)

    def test_multiple_spin_scenarios_require_an_explicit_choice(self):
        result = make_lagrangian(1, scalar(spin="any"), sm_generations=1, show=False)
        with self.assertRaises(ValueError):
            result.select("all", show=False)
        for scenario, catalogue in enumerate(result.catalogues, 1):
            selected = result.select("all", scenario=scenario, show=False)
            self.assertEqual(selected.scenario, scenario)
            self.assertEqual(selected.operators, tuple(catalogue["operators"]))
            self.assertEqual(selected.selected_ids, tuple(row["id"] for row in catalogue["operators"]))
            self.assertEqual(selected.add([], show=False).scenario, scenario)
        selected = result.select([], scenario=1, show=False)
        with self.assertRaises(TypeError):
            selected.add("all", scenario=2, show=False)

    def test_bad_ids_scenarios_and_types_are_rejected_without_prompting(self):
        with patch("builtins.input", side_effect=AssertionError("Selection must not prompt")):
            for operators in ("O9999", [self.ids[0], "O9999"], 42, None, {}, [42]):
                with self.subTest(operators=operators), self.assertRaises((TypeError, ValueError)):
                    self.result.select(operators, show=False)
            for scenario in (0, 2, -1, True, 1.0, "1"):
                with self.subTest(scenario=scenario), self.assertRaises((TypeError, ValueError)):
                    self.result.select("all", scenario=scenario, show=False)
            with self.assertRaises((TypeError, ValueError)):
                self.result.select("all", show="yes")
        selected = self.result.select([], show=False)
        with self.assertRaises((TypeError, ValueError)):
            selected.add("O9999", show=False)

    def test_exact_dimension_four_selection_excludes_mass_terms(self):
        result = make_lagrangian(1, majorana(), exact_dimension=4, sm_generations=1, show=False)
        selected = result.select("all", show=False)
        self.assertEqual({row["field_content"] for row in selected.operators}, {"H L X", "D(X) X†"})
        self.assertEqual({row["dimension"] for row in selected.operators}, {4})
        self.assertEqual({coefficient["dimension"] for coefficient in selected.coefficients}, {0})

    def test_selection_and_add_display_latex_without_prompting(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=AssertionError("Selection must not prompt")), redirect_stdout(output):
            selected = self.result.select(self.ids[0])
            added = selected.add(self.ids[-1])
        self.assertIn(selected.latex, output.getvalue())
        self.assertIn(added.latex, output.getvalue())


if __name__ == "__main__":
    unittest.main()
