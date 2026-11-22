"""Lorentz-basis, ghost constraint and charge-sector vector mixing checks."""
import unittest
import numpy as np
from scipy.linalg import eigvalsh
from lagrangian_maker import make_lagrangian
from generic_vector_compiler import classify
from generic_vector_invariants import invariant_basis
from feynrules_export import prepare_model,ModelResolutionError,CompilerResolutionError


def vector_benchmark(n=3,kind='real',charge=0):
    r=make_lagrangian(1,dict(spin='vector',su2=n,su3='1',hypercharge=charge,boson_kind=kind),show=False)
    l=r.select('all',show=False);coeff={}
    for row in l.operators:
        values=[0.]*row['multiplicity'];category,_=classify(row,r.models[0])
        if category=='generic_vector_kinetic':values=[-.5,0,.5,0]
        if row['field_content']==('X^2' if kind=='real' else 'X X†'):values=[20000.]
        coeff[row['id']]=values if len(values)>1 else values[0]
    return l,dict(coefficients=coeff,widths={'X':0.},vector_specifications={'X':{'kind':'proca'}})


class VectorCompilerTests(unittest.TestCase):
    def test_uniformly_small_vector_quadratic_forms_keep_the_same_mass(self):
        l,inputs=vector_benchmark()
        for key,value in inputs['coefficients'].items():
            inputs['coefficients'][key]=[x*1e-20 for x in value] if isinstance(value,list) else value*1e-20
        model=prepare_model(l,inputs)
        for label,mass in model.masses.items():self.assertAlmostEqual(mass,125. if label=='H' else 200.)

    def test_triplet_sm_currents_have_hermitian_flavour_basis(self):
        from generic_vector_currents import gauge_basis,flavour_basis
        l,_=vector_benchmark();row=next(r for r in l.operators if r['field_content']=='L L† X')
        gauge=gauge_basis(row,l._source.models[0]);c=prepare_model(*vector_benchmark()).compiler.coordinates
        for (a,b,i),z in gauge.items():self.assertAlmostEqual(gauge[(b,a,c.bars[i])],z.conjugate())
        self.assertEqual(len(flavour_basis(row)),9)
        self.assertEqual(flavour_basis(row)[6],((1,2,1j),(2,1,-1j)))

    def test_complete_lorentz_weak_basis_has_catalogue_rank(self):
        for n,kind,charge in ((1,'complex',1),(2,'complex','1/2'),(3,'real',0)):
            l,_=vector_benchmark(n,kind,charge)
            for row in l.operators:
                category,_=classify(row,l._source.models[0])
                if category in ('generic_vector_potential','generic_vector_cubic_derivative'):
                    self.assertEqual(len(invariant_basis(row,l._source.models[0])['basis']),row['multiplicity'])

    def test_arbitrary_weak_components_have_resolved_masses(self):
        for n,kind,charge,states in ((3,'real',0,2),(4,'complex','1/2',5)):
            l,inputs=vector_benchmark(n,kind,charge);p=prepare_model(l,inputs)
            self.assertEqual(len(p.masses),states+1)
            for label,mass in p.masses.items():self.assertAlmostEqual(mass,125. if label=='H' else 200.)
            self.assertEqual(set(p.coverage),set(l.selected_ids))

    def test_ghost_free_constraint_is_not_guessed_from_one_coefficient(self):
        l,inputs=vector_benchmark();row=next(r for r in l.operators if classify(r,l._source.models[0])[0]=='generic_vector_kinetic')
        inputs['coefficients'][row['id']]=[-.5,.1,.5,0]
        with self.assertRaisesRegex(ModelResolutionError,'ghost-free matrix'):prepare_model(l,inputs)
        inputs['coefficients'][row['id']]=[-.5,1e-20,.5,0]
        with self.assertRaisesRegex(ModelResolutionError,'ghost-free matrix'):prepare_model(l,inputs)
        inputs['coefficients'][row['id']]=[.5,0,-.5,0]
        with self.assertRaisesRegex(ModelResolutionError,'positive definite'):prepare_model(l,inputs)

    def test_same_charge_full_kinetic_and_mass_mixing(self):
        spec=dict(spin='vector',su2=1,su3='1',hypercharge=1,boson_kind='complex')
        r=make_lagrangian(2,spec,spec,show=False);l=r.select('all',show=False);coeff={}
        for row in l.operators:
            kind,_=classify(row,r.models[0]);values=[0.]*row['multiplicity'];label=row['field_content']
            if kind=='generic_vector_kinetic':values=[-.1,0,.1,0] if 'X' in label and 'Y' in label else [-.5,0,.5,0]
            if label=='X X†':values=[20000.]
            elif label=='Y Y†':values=[45000.]
            elif label in ('X Y†','X† Y'):values=[3000.]
            coeff[row['id']]=values if len(values)>1 else values[0]
        model=prepare_model(l,dict(coefficients=coeff,widths={'X':0.,'Y':0.},
                                  vector_specifications={'X':{'kind':'proca'},'Y':{'kind':'proca'}}))
        expected=np.sqrt(eigvalsh([[20000,3000],[3000,45000]],[[.5,.1],[.1,.5]]))
        np.testing.assert_allclose(sorted(v for k,v in model.masses.items() if k!='H'),expected,rtol=1e-9)

    def test_nonzero_unlowered_currents_are_explicit_compiler_errors(self):
        l,inputs=vector_benchmark();row=next(r for r in l.operators if classify(r,l._source.models[0])[0] is None)
        p=prepare_model(l,inputs);self.assertIn('zero_verified',p.coverage[row['id']])
        inputs['coefficients'][row['id']]=[.1]*row['multiplicity'] if row['multiplicity']>1 else .1
        with self.assertRaises(CompilerResolutionError):prepare_model(l,inputs)


if __name__=='__main__':unittest.main()
