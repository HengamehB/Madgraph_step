import math
from copy import deepcopy
import unittest

import numpy as np

from lagrangian_maker import make_lagrangian
from feynrules_export import required_inputs, prepare_model, ModelResolutionError
from neutral_fermion_compiler import takagi


def model(kind="majorana", scalar=False):
    fermion = dict(spin="fermion", fermion_kind=kind, su2=1, su3=1, hypercharge=0)
    if scalar:
        scalar_spec = dict(spin="scalar", boson_kind="real", su2=1, su3=1, hypercharge=0)
        result = make_lagrangian(2, scalar_spec, fermion, show=False)
    else:
        result = make_lagrangian(1, fermion, show=False)
    return result.select("all", show=False)


def defaults(lagrangian):
    inputs = required_inputs(lagrangian)
    for row in lagrangian.operators:
        counts = {f["name"]: f["count"] for f in row["factors"]}
        value = 0 if row["multiplicity"] == 1 else [0] * row["multiplicity"]
        if row["derivatives"]:
            if len(counts) == 1:
                value = .5
            elif len(counts) == 2:
                derivative = next(k for k in counts if k.startswith("D("))
                bare = derivative[2:-1]
                if bare + "†" in counts:
                    value = .5
        if counts in ({"X": 2}, {"Y": 2}):
            value = -150 if any(f.spin == "fermion" and f.name in counts for f in lagrangian._source.models[0]) else -20000
        if counts in ({"X_L": 1, "X_Rc": 1}, {"Y_L": 1, "Y_Rc": 1}):
            value = -300
        inputs["coefficients"][row["id"]] = value
    return inputs


class TakagiTests(unittest.TestCase):
    def test_takagi_mass_scale_is_relative_including_exact_zero_modes(self):
        matrix = np.zeros((4, 4), complex)
        matrix[2:, 2:] = [[1+2j, 3+4j], [3+4j, 5-1j]]
        _, reference = takagi(matrix)
        for scale in (1e-100, 1e100):
            with self.subTest(scale=scale):
                u, masses = takagi(scale * matrix)
                np.testing.assert_allclose(masses / scale, reference, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(u.T @ matrix @ u, np.diag(masses / scale), atol=1e-12)
                np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
        asymmetric = 1e-100 * np.array([[1., 2.], [3., 4.]])
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            takagi(asymmetric)

    def test_complex_symmetric_mass(self):
        m = np.array([[1+2j,3+4j],[3+4j,5-1j]])
        u, masses = takagi(m)
        np.testing.assert_allclose(u.conj().T@u,np.eye(2),atol=1e-12)
        np.testing.assert_allclose(u.T@m@u,np.diag(masses),atol=1e-12)
        self.assertTrue(np.all(masses >= 0))

    def test_degenerate_dirac_pair_and_zero_modes(self):
        m = np.zeros((5,5), complex)
        m[3,4] = m[4,3] = 300
        u, masses = takagi(m)
        np.testing.assert_allclose(sorted(masses),[0,0,0,300,300],atol=1e-12)
        np.testing.assert_allclose(u.T@m@u,np.diag(masses),atol=1e-12)


class NeutralCompilerTests(unittest.TestCase):
    def test_unrelated_mass_scale_does_not_hide_a_small_nonzero_tadpole(self):
        lagrangian = model(scalar=True)
        inputs = defaults(lagrangian)
        linear = next(row for row in lagrangian.operators if row['field_content'] == 'X')
        inputs['coefficients'][linear['id']] = 1e-20
        with self.assertRaisesRegex(ModelResolutionError, 'not stationary'):
            prepare_model(lagrangian, inputs)

    def test_neutral_portal_field_rescaling_preserves_masses_and_active_mixing(self):
        lagrangian = model()
        inputs = defaults(lagrangian)
        portal = next(row for row in lagrangian.operators if row['multiplicity'] == 3)
        inputs['coefficients'][portal['id']] = [1e-4, 2e-4j, 0.]
        reference = prepare_model(lagrangian, inputs).compiler.fermion_basis
        for scale in (1e-10, 1e10):
            with self.subTest(scale=scale):
                changed = deepcopy(inputs)
                for row in lagrangian.operators:
                    power = sum(part['count'] for part in row['factors']
                                if part['name'] in ('X', 'X†', 'D(X)'))
                    value = changed['coefficients'][row['id']]
                    changed['coefficients'][row['id']] = ([v * scale**power for v in value]
                        if isinstance(value, list) else value * scale**power)
                actual = prepare_model(lagrangian, changed).compiler.fermion_basis
                np.testing.assert_allclose(actual['masses'], reference['masses'], rtol=1e-8, atol=1e-12)
                np.testing.assert_allclose(abs(actual['transformation'][:3]),
                                           abs(reference['transformation'][:3]), atol=1e-10, rtol=1e-8)

    def test_tiny_physical_mass_is_retained(self):
        lagrangian = model()
        inputs = defaults(lagrangian)
        mass = next(row for row in lagrangian.operators if row['field_content'] == 'X^2')
        inputs['coefficients'][mass['id']] = -.5e-100
        self.assertAlmostEqual(prepare_model(lagrangian, inputs).masses['X'] / 1e-100, 1.)

    def test_all_majorana_operators_resolved(self):
        lagrangian = model()
        inputs = defaults(lagrangian)
        result = prepare_model(lagrangian,inputs)
        self.assertEqual(required_inputs(lagrangian)["unresolved"],{})
        self.assertEqual(set(result.coverage),set(lagrangian.selected_ids))
        self.assertAlmostEqual(result.masses["X"],300)

    def test_neutrino_portal_mass_and_mixing(self):
        lagrangian = model()
        inputs = defaults(lagrangian)
        portal = next(r for r in lagrangian.operators if r["multiplicity"] == 3)
        inputs["coefficients"][portal["id"]] = [1e-4,0,0]
        result = prepare_model(lagrangian, inputs)
        v = result.basis["vev"]
        md = 1e-4*v/math.sqrt(2)
        expected = (math.sqrt(300**2 + 4*md**2)-300)/2
        self.assertAlmostEqual(result.masses["nu_e"],expected,places=12)
        b = result.compiler.fermion_basis
        t = b["transformation"]
        np.testing.assert_allclose(t.conj().T@b["kinetic"]@t,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(t.T@b["mass_matrix"]@t,np.diag(b["masses"]),atol=1e-10)

    def test_neutral_dirac_pair_and_weyl(self):
        for kind in ("dirac","weyl"):
            lagrangian = model(kind)
            result = prepare_model(lagrangian,defaults(lagrangian))
            masses = [v for k,v in result.masses.items() if k.startswith("X")]
            np.testing.assert_allclose(masses,[300,300] if kind == "dirac" else [300],atol=1e-10)

    def test_scalar_majorana_yukawa_preserved(self):
        lagrangian = model(scalar=True)
        inputs = defaults(lagrangian)
        yukawa = next(r for r in lagrangian.operators if r["field_content"] == "X Y^2")
        inputs["coefficients"][yukawa["id"]] = -.2 + .1j
        result = prepare_model(lagrangian, inputs)
        op = next(o for o in result.operators if o.operator_id == yukawa["id"])
        self.assertEqual(op.kind,"weyl_yukawa")
        self.assertEqual(op.coefficients,(-.2+.1j,))

    def test_flavour_count_is_exact(self):
        lagrangian = model()
        inputs = defaults(lagrangian)
        portal = next(r for r in lagrangian.operators if r["multiplicity"] == 3)
        inputs["coefficients"][portal["id"]] = [0,0]
        with self.assertRaisesRegex(ModelResolutionError,"3 coefficients"):
            prepare_model(lagrangian,inputs)


if __name__ == "__main__":
    unittest.main()
