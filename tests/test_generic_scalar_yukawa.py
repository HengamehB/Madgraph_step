"""Independent exchange/normalization checks for lepton-scalar invariants."""
import math
import unittest

from lagrangian_maker import make_lagrangian
from generic_scalar_compiler import coordinates
from generic_scalar_yukawa import classify,invariant_basis,basis_fingerprint


def example(dimension,charge):
    r=make_lagrangian(1,dict(spin='scalar',su2=dimension,su3='1',hypercharge=str(charge)),show=False)
    row=next(row for row in r.select('all',show=False).operators if classify(row,r.models[0]))
    return row,tuple(r.models[0])


class LeptonScalarTests(unittest.TestCase):
    def test_singly_charged_singlet_is_flavour_antisymmetric(self):
        row,fields=example(1,1);basis=invariant_basis(row,fields)
        self.assertEqual(len(basis),3)
        self.assertEqual(set(basis[0]),{(1,1,2,2,1),(1,2,2,1,1)})
        self.assertAlmostEqual(basis[0][(1,1,2,2,1)],1)
        self.assertAlmostEqual(basis[0][(1,2,2,1,1)],-1)
        self.assertTrue(all(a!=b for p in basis for _,a,_,b,_ in p))

    def test_triplet_symmetric_tensor_has_expected_clebsch_factors(self):
        row,fields=example(3,1);basis=invariant_basis(row,fields);c=coordinates(fields)
        self.assertEqual(len(basis),6)
        xr,xi,xp,xpp=[c.labels.index(label) for label in ('x3r','x3i','x2','x1')]
        expected={(1,1,1,1,xr):1,(1,1,1,1,xi):1j,
                  (1,1,2,1,xp):-2,(2,1,2,1,xpp):math.sqrt(2)}
        self.assertEqual(set(expected),set(basis[0]))
        for key,value in expected.items():self.assertAlmostEqual(basis[0][key],value)
        self.assertAlmostEqual(basis[1][(1,1,2,2,xp)],-1)
        self.assertAlmostEqual(basis[1][(1,2,2,1,xp)],-1)

    def test_doubly_charged_singlets_keep_six_complex_flavour_entries(self):
        for charge in (-2,2):
            row,fields=example(1,charge);basis=invariant_basis(row,fields)
            self.assertEqual(len(basis),6)
            self.assertEqual([next(iter(p))[:4] for p in basis],
                             [(1,1,1,1),(1,1,1,2),(1,1,1,3),(1,2,1,2),(1,2,1,3),(1,3,1,3)])
            self.assertTrue(all(abs(next(iter(p.values()))-1)<1e-12 for p in basis))

    def test_basis_fingerprint_is_stable_across_generous_budget_changes(self):
        row,fields=example(3,-1)
        self.assertEqual(basis_fingerprint(row,fields),basis_fingerprint(row,fields,{'max_output_terms':200000}))


if __name__=='__main__':unittest.main()
