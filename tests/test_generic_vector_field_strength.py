"""Independent representation, Bose-rank, charge and source checks for FVV."""
import math
from collections import defaultdict
import unittest
import numpy as np

from generic_vector_field_strength import invariant_basis,adjoint_intertwiner,source_candidates
from generic_vector_compiler import classify
from generic_scalar_compiler import coordinates
from component_invariants import ComponentResourceLimit
from lagrangian_maker import make_lagrangian
from test_generic_vector_compiler import vector_benchmark
from feynrules_export import prepare_model


class VectorFieldStrengthTests(unittest.TestCase):
    def test_cartesian_adjoint_has_expected_charged_components(self):
        s=adjoint_intertwiner()
        # Stock SM.fr defines Wi1=(W-+W+)/sqrt(2),
        # Wi2=(W--W+)/(i sqrt(2)); spherical charges must be +1,0,-1.
        charged=np.array([[1/math.sqrt(2),0,1/math.sqrt(2)],
                          [1j/math.sqrt(2),0,-1j/math.sqrt(2)],[0,1,0]])
        np.testing.assert_allclose(s@charged,np.diag([-1j,1j,1j]),atol=1e-14)

    def test_every_catalogue_fvv_rank_and_electric_charge(self):
        for n,kind,y in ((1,'complex',1),(2,'complex','1/2'),(3,'real',0),(4,'complex','1/2'),(5,'real',0),(7,'real',0)):
            l,_=vector_benchmark(n,kind,y);fields=l._source.models[0];c=coordinates(tuple(fields))
            for row in l.operators:
                if classify(row,fields)[0]!='generic_vector_field_strength':continue
                basis=invariant_basis(row,fields)
                self.assertEqual(len(basis['basis']),row['multiplicity'])
                for tensor in basis['tensors']:
                    polynomial=defaultdict(complex)
                    for (a,i,j),z in tensor.items():
                        charges=({1:1/math.sqrt(2),-1:1/math.sqrt(2)},
                                 {1:1j/math.sqrt(2),-1:-1j/math.sqrt(2)},{0:1})[a] if basis['strength'].startswith('W') else {0:1}
                        for charge,g in charges.items():
                            for (left,),x in c.components[basis['names'][0]][i].items():
                                for (right,),v in c.components[basis['names'][1]][j].items():
                                    for mu,nu,sign in ((0,1,1),(1,0,-1)):
                                        key=(charge,tuple(sorted((4*left+mu,4*right+nu))))
                                        polynomial[key]+=sign*g*z*x*v
                    for (charge,monomial),z in polynomial.items():
                        if abs(z)>1e-10:
                            self.assertEqual(charge+sum(c.charges[i//4] for i in monomial),0)

    def test_two_different_weak_representations_are_generated(self):
        real=lambda n:dict(spin='vector',su2=n,su3=1,hypercharge=0,boson_kind='real')
        r=make_lagrangian(2,real(1),real(3),show=False);l=r.select('all',show=False)
        row=next(row for row in l.operators if row['field_content']=='W+ X Y')
        self.assertEqual(len(invariant_basis(row,r.models[0])['basis']),row['multiplicity'])

    def test_source_retains_dual_and_nonabelian_field_strength(self):
        l,inputs=vector_benchmark();fields=l._source.models[0]
        row=next(row for row in l.operators if classify(row,fields)[0]=='generic_vector_field_strength')
        inputs['coefficients'][row['id']]=1e-20+2e-20j
        prepared=prepare_model(l,inputs)
        op=next(o for o in prepared.operators if o.operator_id==row['id'])
        self.assertEqual(op.coefficients,(1e-20+2e-20j,))
        source=' '.join(source_candidates(invariant_basis(row,fields),repr,lambda n,k,mu:f'{n}{k}[{mu}]'))
        self.assertIn('FS[Wi,mu,nu,',source)
        self.assertIn('I/2*Eps[mu,nu,rho,sigma]*FS[Wi,rho,sigma,',source)

    def test_field_strength_resource_budget_is_explicit(self):
        l,_=vector_benchmark();fields=l._source.models[0]
        row=next(row for row in l.operators if classify(row,fields)[0]=='generic_vector_field_strength')
        with self.assertRaises(ComponentResourceLimit):invariant_basis(row,fields,dict(max_power_monomials=1))


if __name__=='__main__':unittest.main()
