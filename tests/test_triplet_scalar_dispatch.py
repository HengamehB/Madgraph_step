"""Native-triplet fallback preserves legacy models and exposes full tensors."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from model_builder import ModelDefinition
from feynrules_export import prepare_model
from generic_color_compiler import invariant_basis,kinetic_basis,uses_generic
from gauge_matter_compiler import prepare as legacy_prepare

ROOT=Path(__file__).resolve().parents[1]


def specification(colour='3'):
    return dict(spin='scalar',su2=1,su3=colour,hypercharge=1,boson_kind='complex')


def inputs_for(definition):
    inputs=definition.required_inputs()
    for row in definition.operator_inputs():
        value=[0.]*row['multiplicity']
        if row['label'] in ('D(X) D(X†)','D(Y) D(Y†)'):value[0]=1.
        if row['label']=='X X†':value[0]=-40000.
        if row['label']=='Y Y†':value[0]=-90000.
        inputs['coefficients'][row['key']]=value if len(value)>1 else value[0]
    return inputs


def evaluate(polynomial,fields):
    return sum(z*np.prod([fields[token][colour] for token,colour in monomial])
               for monomial,z in polynomial.items())


class TripletScalarDispatchTests(unittest.TestCase):
    def test_existing_fundamental_and_antifundamental_keep_identity_and_spectrum(self):
        for colour in ('3','3bar'):
            definition=ModelDefinition(1,specification(colour));inputs=inputs_for(definition)
            inputs['coefficients']['H H† X X†']=-.03
            inputs['coefficients']['X^2 X†^2']=-.02
            lag,values=definition.build(inputs)
            self.assertFalse(uses_generic(definition.fields,lag.operators))
            self.assertFalse(uses_generic(definition.fields))
            prepared=prepare_model(lag,values);legacy=legacy_prepare(lag,values)
            self.assertEqual(type(prepared.compiler).__name__,'GaugeMatterCompiler')
            self.assertEqual(prepared.masses,legacy.masses)
            with tempfile.TemporaryDirectory() as folder:
                files,_,particles=prepared.compiler.write_sources(prepared,Path(folder),ROOT/'vendor'/'FeynRules')
            self.assertEqual(particles[0]['pdg'],9200001)
            self.assertEqual(particles[0]['name'],'x' if colour=='3' else 'xbar')

    def test_legacy_norm_coefficients_equal_native_tensor_coefficients(self):
        x=np.array([.2+.3j,-.7+.1j,.4-.5j]);norm=np.vdot(x,x).real
        for colour in ('3','3bar'):
            definition=ModelDefinition(1,specification(colour));fields={'X':x,'X†':x.conj()}
            for row in definition.lagrangian.operators:
                basis,polys=(kinetic_basis if row['derivatives'] else invariant_basis)(row,definition.fields)
                self.assertEqual(len(basis),1)
                power=2 if row['field_content']=='X^2 X†^2' else 1
                # Native Higgs-pair factors are kept separately as H†H;
                # the remaining colour norm has exactly the legacy coefficient.
                self.assertAlmostEqual(evaluate(polys[0],fields),norm**power)

    def test_public_two_triplets_lower_both_colour_quartics_and_holomorphic_term(self):
        definition=ModelDefinition(2,specification(),specification());inputs=inputs_for(definition)
        inputs['coefficients']['X X† Y Y†']=[-.02,-.03]
        inputs['coefficients']['X^2 Y†^2']=.01+.02j
        self.assertEqual(inputs['unresolved'],{})
        self.assertTrue(uses_generic(definition.fields,definition.lagrangian.operators))
        descriptions=definition.operator_inputs()
        self.assertTrue(all(row['kind'].startswith('colour_') for row in descriptions))
        cross=next(row for row in descriptions if row['label']=='X X† Y Y†')
        self.assertEqual(len(cross['invariant_labels']),2)
        self.assertEqual(len(cross['basis_fingerprint']),64)
        lag,values=definition.build(inputs);prepared=prepare_model(lag,values)
        self.assertEqual(type(prepared.compiler).__name__,'GenericColorCompiler')
        self.assertEqual(set(prepared.coverage),set(lag.selected_ids))
        np.testing.assert_allclose([prepared.masses['X'],prepared.masses['Y']],[200,300])
        x=np.array([.2+.3j,-.7+.1j,.4-.5j]);y=np.array([-.1+.2j,.3+.4j,.7-.1j])
        components={'X':x,'X†':x.conj(),'Y':y,'Y†':y.conj()}
        row=next(r for r in lag.operators if r['field_content']=='X X† Y Y†')
        _,polys=invariant_basis(row,definition.fields)
        expected=[np.vdot(x,x).real*np.vdot(y,y).real,abs(np.vdot(x,y))**2]
        np.testing.assert_allclose([evaluate(p,components) for p in polys],expected,atol=1e-14)
        holomorphic=next(r for r in lag.operators if r['field_content']=='X^2 Y†^2')
        _,polys=invariant_basis(holomorphic,definition.fields)
        self.assertAlmostEqual(evaluate(polys[0],components),np.vdot(y,x)**2)
        with tempfile.TemporaryDirectory() as folder:
            files,_,_=prepared.compiler.write_sources(prepared,Path(folder),ROOT/'vendor'/'FeynRules')
            source=files[-1].read_text()
        for operator in (row,holomorphic):
            for i in range(operator['multiplicity']):
                self.assertIn('pmGCO'+operator['id'][1:]+'n'+str(i+1),source)
        self.assertIn('IndexDelta[Index[Colour,c1],Index[Colour,c4]]',source)

    def test_legacy_supported_selection_stays_legacy_even_for_two_triplets(self):
        operators=['D(X) D(X†)','D(Y) D(Y†)','X X†','Y Y†','X^2 X†^2','Y^2 Y†^2']
        definition=ModelDefinition(2,specification(),specification(),selected_operators=operators)
        self.assertFalse(uses_generic(definition.fields,definition.lagrangian.operators))
        lag,values=definition.build(inputs_for(definition))
        self.assertEqual(type(prepare_model(lag,values).compiler).__name__,'GaugeMatterCompiler')


if __name__=='__main__':unittest.main()
