"""Generic user inputs bind numbers to physics, without calling external tools."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import unittest
from unittest.mock import patch

from feynrules_export import ModelResolutionError, prepare_model
from model_builder import ModelDefinition


def spec(spin="scalar", kind="real", **changes):
    value = dict(spin=spin, su2=1, su3="1", hypercharge="0")
    value["fermion_kind" if spin == "fermion" else "boson_kind"] = kind
    value.update(changes)
    return value


def explicit_zero_inputs(definition):
    """Deliberate test choice, never a production default."""
    inputs = definition.required_inputs()
    inputs["coefficients"] = {
        row["key"]: 0.0 if row["multiplicity"] == 1 else [0.0] * row["multiplicity"]
        for row in definition.operator_inputs()}
    return inputs


class ModelBuilderTests(unittest.TestCase):
    def test_template_has_every_coefficient_unset_for_each_spin(self):
        for spin, kind in (("scalar", "real"), ("scalar", "complex"),
                           ("fermion", "weyl"), ("fermion", "dirac"),
                           ("fermion", "majorana"), ("vector", "real")):
            with self.subTest(spin=spin, kind=kind):
                definition = ModelDefinition(1, spec(spin, kind))
                inputs = definition.required_inputs()
                self.assertEqual(len(inputs["coefficients"]), len(definition.lagrangian.operators))
                for value in inputs["coefficients"].values():
                    self.assertTrue(value is None or all(item is None for item in value))
                self.assertEqual(inputs["stable_particles"], [])

    def test_readable_coefficients_build_a_scalar_physical_model(self):
        definition = ModelDefinition(1, spec(), selected_operators=["X^2", "D(X)^2", "X^4"])
        inputs = definition.required_inputs()
        inputs["coefficients"] = {"X^2": -20000.0, "D(X)^2": 0.5, "X^4": -0.01}
        before = deepcopy(inputs)
        lagrangian, backend = definition.build(inputs)
        model = prepare_model(lagrangian, backend)
        self.assertAlmostEqual(model.masses["X"], 200.0)
        self.assertEqual(inputs, before)
        self.assertEqual(set(backend["coefficients"]), set(lagrangian.selected_ids))

    def test_same_interface_resolves_weyl_majorana_and_dirac(self):
        for kind in ("weyl", "majorana", "dirac"):
            definition = ModelDefinition(1, spec("fermion", kind))
            inputs = explicit_zero_inputs(definition)
            if kind == "dirac":
                inputs["coefficients"].update({"X_L X_Rc": -300.0,
                    "D(X_L) X_L†": 0.5, "D(X_Rc) X_Rc†": 0.5})
            else:
                inputs["coefficients"].update({"X^2": -150.0, "D(X) X†": 0.5})
            model = prepare_model(*definition.build(inputs))
            masses = [mass for name, mass in model.masses.items() if name.startswith("X")]
            self.assertEqual(len(masses), 2 if kind == "dirac" else 1)
            for mass in masses:
                self.assertAlmostEqual(mass, 300.0)

    def test_same_interface_resolves_scalar_fermion_yukawa(self):
        definition = ModelDefinition(2, spec(), spec("fermion", "majorana"))
        inputs = explicit_zero_inputs(definition)
        inputs["coefficients"].update({"X^2": -20000.0, "D(X)^2": 0.5,
            "Y^2": -150.0, "D(Y) Y†": 0.5, "X Y^2": -0.2 + 0.1j})
        model = prepare_model(*definition.build(inputs))
        yukawa = next(operator for operator in model.operators if operator.kind == "weyl_yukawa")
        self.assertEqual(yukawa.coefficients, (-0.2 + 0.1j,))

    def test_vector_requires_physical_specification_and_preserves_invariant_order(self):
        definition = ModelDefinition(1, spec("vector"))
        inputs = explicit_zero_inputs(definition)
        self.assertNotIn("X", inputs["unresolved"])
        inputs["coefficients"].update({"X^2": 600.0 ** 2 / 2,
                                        "D(X)^2": [-0.5, 0.0, 0.5, 0.0]})
        with self.assertRaisesRegex(ModelResolutionError, "vector_specifications.X"):
            definition.build(inputs)
        inputs["vector_specifications"] = {"X": {"kind": "proca"}}
        model = prepare_model(*definition.build(inputs))
        self.assertAlmostEqual(model.masses["X"], 600.0)
        current = next(row for row in definition.operator_inputs() if row["key"] == "Q Q† X")
        self.assertEqual(current["invariant_labels"],
                         ("11", "22", "33", "Re12", "Im12", "Re13", "Im13", "Re23", "Im23"))
        self.assertTrue(current["real"])

    def test_stale_signature_blocks_reused_scalar_fermion_key(self):
        scalar = ModelDefinition(1, spec(), selected_operators=["X^2"])
        fermion = ModelDefinition(1, spec("fermion", "majorana"), selected_operators=["X^2"])
        inputs = explicit_zero_inputs(scalar)
        with self.assertRaisesRegex(ModelResolutionError, "stale model_signature"):
            fermion.build(inputs)

    def test_signature_changes_for_representation_kind_and_selection(self):
        baseline = ModelDefinition(1, spec()).signature
        alternatives = [ModelDefinition(1, spec(su2=3)),
                        ModelDefinition(1, spec(kind="complex")),
                        ModelDefinition(1, spec(), selected_operators=["X^2"])]
        self.assertTrue(all(item.signature != baseline for item in alternatives))
        self.assertEqual(baseline, ModelDefinition(1, spec()).signature)

    def test_missing_signature_is_not_silently_attached(self):
        definition = ModelDefinition(1, spec())
        inputs = explicit_zero_inputs(definition)
        del inputs["model_signature"]
        with self.assertRaisesRegex(ModelResolutionError, "missing or stale"):
            definition.build(inputs)

    def test_missing_extra_duplicate_and_invalid_coefficients_are_named(self):
        definition = ModelDefinition(1, spec())
        inputs = explicit_zero_inputs(definition)
        del inputs["coefficients"]["X"]
        inputs["coefficients"]["invented portal"] = 0.1
        with self.assertRaisesRegex(ModelResolutionError, "missing=\\['X'\\].*invented portal"):
            definition.build(inputs)
        inputs = explicit_zero_inputs(definition)
        inputs["coefficients"][definition.lagrangian.operators[0]["id"]] = 0.0
        with self.assertRaisesRegex(ModelResolutionError, "duplicate=\\['X'\\]"):
            definition.build(inputs)
        inputs = explicit_zero_inputs(definition)
        inputs["coefficients"].update({"X": None, "X^2": float("nan"), "X^3": 1j})
        with self.assertRaises(ModelResolutionError) as caught:
            definition.build(inputs)
        self.assertIn("'X'", str(caught.exception))
        self.assertIn("'X^2' must be finite", str(caught.exception))
        self.assertIn("'X^3' must be real", str(caught.exception))

    def test_flavour_vector_length_and_complex_values(self):
        definition = ModelDefinition(1, spec("fermion", "majorana"))
        inputs = explicit_zero_inputs(definition)
        inputs["coefficients"]["H L X"] = [0.1, 0.2]
        with self.assertRaisesRegex(ModelResolutionError, "'H L X' needs 3 coefficients"):
            definition.build(inputs)
        inputs["coefficients"]["H L X"] = [0.1j, 0.2, 0.3]
        _, backend = definition.build(inputs)
        row = next(row for row in definition.lagrangian.operators if row["field_content"] == "H L X")
        self.assertEqual(backend["coefficients"][row["id"]], [0.1j, 0.2, 0.3])

    def test_operator_metadata_dimensions_and_conjugation_follow_actual_spin(self):
        scalar = ModelDefinition(1, spec())
        majorana = ModelDefinition(1, spec("fermion", "majorana"))
        a = next(row for row in scalar.operator_inputs() if row["key"] == "X^2")
        b = next(row for row in majorana.operator_inputs() if row["key"] == "X^2")
        self.assertEqual((a["units"], a["add_hc"]), ("GeV^2", False))
        self.assertEqual((b["units"], b["add_hc"]), ("GeV", True))
        portal = next(row for row in majorana.operator_inputs() if row["key"] == "H L X")
        self.assertEqual(portal["invariant_labels"], ("electron", "muon", "tau"))

    def test_doublet_contractions_and_weyl_flavour_order_are_explicit(self):
        definition = ModelDefinition(1, spec(kind="complex", su2=2, hypercharge="1/2"))
        rows = {row["key"]: row for row in definition.operator_inputs()}
        self.assertEqual(rows["H H† X X†"]["invariant_labels"],
                         ("(H† H) (X† X)", "(H† X) (X† H)"))
        yukawa = rows["Q X uc"]
        self.assertEqual(yukawa["invariant_labels"], ("11", "12", "13", "21", "22", "23", "31", "32", "33"))
        self.assertIn("charge-conjugate-singlet", yukawa["hint"])
        self.assertTrue(yukawa["add_hc"])

    def test_nonphysical_generation_count_reports_limit_with_matching_labels(self):
        vector = ModelDefinition(1, spec("vector"), sm_generations=2)
        current = next(row for row in vector.operator_inputs() if row["key"] == "Q Q† X")
        self.assertEqual(current["multiplicity"], 4)
        self.assertEqual(current["invariant_labels"], ("11", "22", "Re12", "Im12"))
        self.assertIn("three SM generations", vector.required_inputs()["unresolved"]["SM"])
        fermion = ModelDefinition(1, spec("fermion", "majorana"), sm_generations=1)
        portal = next(row for row in fermion.operator_inputs() if row["key"] == "H L X")
        self.assertEqual(len(portal["invariant_labels"]), portal["multiplicity"])
        self.assertTrue(portal["obstruction"])
        self.assertIn("SM", fermion.required_inputs()["unresolved"])

    def test_unsupported_representations_can_be_inspected_without_external_tools(self):
        with patch("subprocess.run", side_effect=AssertionError("external tool invoked")):
            definition = ModelDefinition(1, spec(kind="complex", su3="10"))
            inputs = definition.required_inputs()
            self.assertTrue(inputs["unresolved"])
            output = StringIO()
            with redirect_stdout(output):
                definition.show_inputs()
            self.assertIn("physical_inputs =", output.getvalue())
            self.assertIn("Automatic export unresolved", output.getvalue())
            scope = {}
            exec(output.getvalue(), scope)
            self.assertEqual(scope["physical_inputs"], {key: value for key, value in inputs.items()
                                                       if key not in {"vevs", "higgs_mass"}})

    def test_manual_feynrules_hook_accepts_readable_operator_map(self):
        definition = ModelDefinition(1, spec(su3="8"), selected_operators=["X^2"])
        inputs = explicit_zero_inputs(definition)
        inputs["feynrules"] = dict(physical_basis=True, model_files=["user.fr"],
                                    operator_map={"X^2": "X[a]*X[a]"})
        lagrangian, backend = definition.build(inputs)
        self.assertEqual(backend["feynrules"]["operator_map"],
                         {lagrangian.selected_ids[0]: "X[a]*X[a]"})
        self.assertIn("X^2", inputs["feynrules"]["operator_map"])

    def test_any_requires_a_concrete_scenario(self):
        with self.assertRaisesRegex(ValueError, "Multiple particle-type scenarios"):
            ModelDefinition(1, spec("any"))
        definition = ModelDefinition(1, spec("any"), scenario=1)
        self.assertEqual(definition.lagrangian.scenario, 1)

    def test_unknown_physical_keys_and_invalid_scalar_values(self):
        definition = ModelDefinition(1, spec())
        inputs = explicit_zero_inputs(definition)
        inputs["masses"] = {"X": 200.0}
        with self.assertRaisesRegex(ModelResolutionError, "Unknown physical input keys: masses"):
            definition.build(inputs)
        del inputs["masses"]
        inputs["widths"]["X"] = -1.0
        with self.assertRaisesRegex(ModelResolutionError, "widths.X must be nonnegative"):
            definition.build(inputs)
        inputs["widths"]["X"] = 0.0
        inputs["vevs"]["Y"] = 0.0
        with self.assertRaisesRegex(ModelResolutionError, "Unknown scalar VEV fields"):
            definition.build(inputs)
        del inputs["vevs"]["Y"]
        inputs["sm_inputs"]["Gf"] = 0.0
        with self.assertRaisesRegex(ModelResolutionError, "Gf.*must be positive"):
            definition.build(inputs)

    def test_selection_can_use_ids_or_keys_and_never_drops_extra_coefficients(self):
        complete = ModelDefinition(1, spec())
        coefficient_id = next(row["operator_id"] for row in complete.operator_inputs() if row["key"] == "X^2")
        by_key = ModelDefinition(1, spec(), selected_operators=["X^2"])
        by_id = ModelDefinition(1, spec(), selected_operators=[coefficient_id])
        self.assertEqual(by_key.signature, by_id.signature)
        inputs = explicit_zero_inputs(by_key)
        inputs["coefficients"]["X^4"] = 0.0
        with self.assertRaisesRegex(ModelResolutionError, "extra=\\['X\\^4'\\]"):
            by_key.build(inputs)


if __name__ == "__main__":
    unittest.main()
