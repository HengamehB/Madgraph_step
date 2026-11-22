"""Independent quadratic-form and conjugation checks for generic multiplets."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import eigvalsh

from lagrangian_maker import make_lagrangian
from feynrules_export import prepare_model, ModelResolutionError, BackendCapabilityError, _verify_exported_particles
from generic_scalar_compiler import (coordinates, invariant_basis, classify, conjugate, add, uses_generic)
from component_invariants import ComponentResourceLimit


def benchmark(su2=4, kind='complex', charge='1/2'):
    result=make_lagrangian(1,dict(spin='scalar',su2=su2,su3='1',hypercharge=charge,boson_kind=kind),show=False)
    lag=result.select('all',show=False)
    coefficients={}
    for row in lag.operators:
        values=[0.]*row['multiplicity']
        category,_=classify(row,result.models[0])
        if category=='generic_scalar_kinetic' and 'H' not in row['field_content']:values[0]=1.
        if sum(p['count'] for p in row['factors'])==2 and not row['derivatives'] and 'H' not in row['field_content']:values[0]=-40000.
        coefficients[row['id']]=values if len(values)>1 else values[0]
    return lag,dict(model_name='GenericScalarTest',coefficients=coefficients,vevs={'X':0.},widths={'X':0.},higgs_mass=125.)


class GenericScalarTests(unittest.TestCase):
    def test_tiny_complex_kinetic_mixing_phase_survives_basis_projection(self):
        spec=dict(spin='scalar',su2=2,su3=1,hypercharge='1/2')
        result=make_lagrangian(2,spec,spec,show=False);lag=result.select('all',show=False);coefficients={}
        for row in lag.operators:
            kind,_=classify(row,result.models[0]);values=[0.]*row['multiplicity'];label=row['field_content']
            if kind=='generic_scalar_kinetic' and 'H' not in label:
                values[0]=.2+1e-20j if 'X' in label and 'Y' in label else 1.
            if label=='X X†':values[0]=-40000.
            if label=='Y Y†':values[0]=-90000.
            coefficients[row['id']]=values if len(values)>1 else values[0]
        compiler=prepare_model(lag,dict(coefficients=coefficients)).compiler
        cross=[z for a,b,z in compiler.kinetic_rows if a[0]!=b[0]]
        self.assertTrue(cross)
        for z in cross:self.assertAlmostEqual(abs(z.imag)/1e-20,1.)

    def test_tiny_selected_coupling_survives_physical_polynomial_and_source(self):
        lag,inputs=benchmark()
        row=next(r for r in lag.operators if r['field_content']=='H H† X X†')
        values=[0.]*row['multiplicity'];values[0]=1e-20
        inputs['coefficients'][row['id']]=values
        model=prepare_model(lag,inputs)
        self.assertTrue(any(0<abs(z)<1e-13 for m,z in model.compiler.potential.items() if len(m)==4))
        with tempfile.TemporaryDirectory() as folder:
            files,_,_=model.compiler.write_sources(model,Path(folder),Path(__file__).resolve().parents[1]/'vendor'/'FeynRules')
            self.assertIn('*^-20',files[-1].read_text())

    def test_uniformly_small_positive_quadratic_forms_are_normalized(self):
        lag,inputs=benchmark()
        for row in lag.operators:
            value=inputs['coefficients'][row['id']]
            inputs['coefficients'][row['id']]=[x*1e-20 for x in value] if isinstance(value,list) else value*1e-20
        model=prepare_model(lag,inputs)
        for label,mass in model.masses.items():self.assertAlmostEqual(mass,125. if label=='h' else 200.)

    def test_selected_component_resource_budget_is_enforced(self):
        lag,inputs=benchmark()
        inputs['compiler_limits']={'max_letter_components':1}
        with self.assertRaises(ComponentResourceLimit) as error:
            prepare_model(lag,inputs)
        self.assertEqual(error.exception.category,'resource_limit')

    def test_mixed_scalar_sector_retains_doublet_yukawa_flavour_order(self):
        r=make_lagrangian(2,dict(spin='scalar',su2=2,su3='1',hypercharge='1/2'),
                          dict(spin='scalar',su2=1,su3='1',hypercharge='0',boson_kind='real'),show=False)
        lag=r.select('all',show=False);coefficients={}
        for row in lag.operators:
            kind,_=classify(row,r.models[0]);values=[0.]*row['multiplicity']
            if kind=='generic_scalar_kinetic' and 'H' not in row['field_content']:values[0]=1.
            if row['field_content'] in ('X X†','Y^2'):values[0]=-40000.
            if kind.startswith('generic_scalar_yukawa_'):values[8]=.1+.02j
            coefficients[row['id']]=values if len(values)>1 else values[0]
        model=prepare_model(lag,dict(coefficients=coefficients,vevs={'X':0.,'Y':0.},higgs_mass=125.))
        yukawas=[op for op in model.operators if op.kind.startswith('generic_scalar_yukawa_')]
        self.assertEqual(len(yukawas),3)
        self.assertTrue(all(op.coefficients[8]==.1+.02j for op in yukawas))
        with tempfile.TemporaryDirectory() as folder:
            files,_,_=model.compiler.write_sources(model,Path(folder),Path(__file__).resolve().parents[1]/'vendor'/'FeynRules')
            source=files[-1].read_text()
        self.assertIn('LLbar[sp,1,3].lR[sp,3]',source)
        self.assertIn('QLbar[sp,2,3,cc].uR[sp,3,cc]',source)
        self.assertIn('InteractionOrder -> {NP, 1}',source)

    def test_quartet_complete_potential_and_physical_degrees_of_freedom(self):
        lag,inputs=benchmark()
        model=prepare_model(lag,inputs)
        self.assertTrue(uses_generic(model.fields))
        self.assertEqual(set(model.coverage),set(lag.selected_ids))
        self.assertEqual(len(model.masses),6) # Higgs, three charged, two neutral
        for label,mass in model.masses.items():
            self.assertAlmostEqual(mass,125. if label=='h' else 200.)
        self.assertEqual(sum(1 if model.compiler.coordinates.charges[i]==0 else 2
                             for i in model.compiler.eigenmasses if i!=0),8)

    def test_all_selfconjugate_quartet_invariants_are_hermitian(self):
        lag,_=benchmark(); fields=tuple(lag._source.models[0]); bars=coordinates(fields).bars
        for row in lag.operators:
            if row['field_content']==row['conjugate_content']:
                for polynomial in invariant_basis(row,fields):
                    self.assertFalse(add(polynomial,conjugate(polynomial,bars),-1),row['id'])

    def test_real_quintet_reality_reduces_five_components(self):
        lag,inputs=benchmark(5,'real','0')
        model=prepare_model(lag,inputs)
        self.assertEqual(len(model.masses),4) # Higgs plus charge 0,1,2 states
        for label,mass in model.masses.items():
            self.assertAlmostEqual(mass,125. if label=='h' else 200.)

    def test_complete_charge_block_mass_and_kinetic_mixing(self):
        spec=dict(spin='scalar',su2=2,su3='1',hypercharge='1/2')
        result=make_lagrangian(2,spec,spec,show=False)
        rows=[r for r in result.select('all',show=False).operators
              if sum(p['count'] for p in r['factors'])==2 and not any('H' in p['name'] for p in r['factors'])]
        lag=result.select([r['id'] for r in rows],show=False)
        values={}
        for row in rows:
            names=' '.join(p['name'] for p in row['factors'])
            if row['derivatives']:
                value=.2 if 'X' in names and 'Y' in names else 2. if 'X' in names else 1.
            else:
                value=-20000. if 'X' in names and 'Y' in names else -160000. if 'X' in names else -90000.
            values[row['id']]=value
        model=prepare_model(lag,dict(coefficients=values,vevs={'X':0.,'Y':0.},higgs_mass=125.,widths={'X':0.,'Y':0.}))
        expected=np.sqrt(eigvalsh([[160000,20000],[20000,90000]],[[2,.2],[.2,1]]))
        masses=sorted(v for k,v in model.masses.items() if k!='h')
        np.testing.assert_allclose(masses,sorted(list(expected)*3),rtol=1e-9)

    def test_nonstationary_vacuum_and_ghost_are_rejected(self):
        lag,inputs=benchmark()
        for row in lag.operators:
            if row['field_content']=='H H†^2 X':inputs['coefficients'][row['id']]=1.
        with self.assertRaisesRegex(ModelResolutionError,'not stationary'):prepare_model(lag,inputs)
        lag,inputs=benchmark()
        for row in lag.operators:
            if row['derivatives']:inputs['coefficients'][row['id']]=-1.
        with self.assertRaisesRegex(ModelResolutionError,'positive definite'):prepare_model(lag,inputs)

    def test_higgs_kinetic_mixing_is_covered_at_zero_but_nonzero_needs_gauge_basis(self):
        lag,inputs=benchmark(2,'complex','-1/2')
        for row in lag.operators:
            if row['derivatives'] and 'H' in row['field_content']:inputs['coefficients'][row['id']]=0.
        model=prepare_model(lag,inputs)
        row=next(r for r in lag.operators if r['derivatives'] and 'H' in r['field_content'])
        self.assertIn(row['id'],model.coverage)
        inputs['coefficients'][row['id']]=1.
        with self.assertRaisesRegex(ModelResolutionError,'compiler gap'):prepare_model(lag,inputs)

    def test_true_colour_backend_limit_is_distinct(self):
        lag=make_lagrangian(1,dict(spin='scalar',su2=1,su3='10',hypercharge='1'),show=False).select('all',show=False)
        with self.assertRaisesRegex(BackendCapabilityError,'bundled FeynRules UFO and MadGraph'):
            prepare_model(lag,{'coefficients':{}})

    def test_export_coverage_checks_conjugate_mass_and_spin(self):
        expected=[dict(name='x',antiname='xbar',pdg=900,spin=1,mass=200.)]
        data={900:dict(name='x',spin=1,mass_gev=200.),-900:dict(name='xbar')}
        _verify_exported_particles(expected,data)
        with self.assertRaisesRegex(ModelResolutionError,'conjugate'):
            _verify_exported_particles(expected,{900:data[900]})
        data[900]['mass_gev']=201.
        with self.assertRaisesRegex(ModelResolutionError,'mass'):_verify_exported_particles(expected,data)


if __name__=='__main__':unittest.main()
