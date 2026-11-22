"""Physical checks for joint native-colour/weak scalar lowering."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import eigh

from lagrangian_maker import make_lagrangian
from generic_joint_scalar_compiler import prepare, classify, basis_fingerprint
from feynrules_export import CompilerResolutionError, ModelResolutionError
from workflow_errors import ResourceLimitError

ROOT=Path(__file__).resolve().parents[1]


def benchmark(colour='6',weak=2,kind='complex',charge='1/2',mixed=False):
    spec=dict(spin='scalar',su2=weak,su3=colour,hypercharge=charge,boson_kind=kind)
    result=make_lagrangian(2 if mixed else 1,spec,*([spec] if mixed else []),show=False)
    lagrangian=result.select('all',show=False)
    values=dict(model_name='JointScalarValidation',coefficients={},vevs={'X':0.,'Y':0.},widths={'X':0.,'Y':0.})
    for row in lagrangian.operators:
        value=0. if row['multiplicity']==1 else [0.]*row['multiplicity']
        for name,mass in [('X',200.),('Y',300.)]:
            if row['field_content']==(f'D({name})^2' if kind=='real' else f'D({name}) D({name}†)'):
                value=.5 if kind=='real' else 1.
            if row['field_content']==(f'{name}^2' if kind=='real' else f'{name} {name}†'):
                value=-mass*mass/(2 if kind=='real' else 1)
        if mixed:
            value={'D(X) D(Y†)':.1+.05j,'X Y†':-2000.-1000j}.get(row['field_content'],value)
        values['coefficients'][row['id']]=value
    return lagrangian,values


class JointScalarCompilerTests(unittest.TestCase):
    def test_public_tex_workflow_discovers_joint_basis_and_saves_physical_maps(self):
        from model_builder import ModelDefinition
        from tex_inputs import BEGIN, END, input_bindings
        from pipeline import Pipeline
        from pipeline_storage import public
        spec = dict(spin='scalar', su2=2, su3='6', hypercharge='1/2', boson_kind='complex')
        definition = ModelDefinition(1, spec)
        lagrangian, values = benchmark()
        inputs = definition.required_inputs()
        inputs.pop('model_signature')
        self.assertEqual(inputs['unresolved'], {})
        inputs['coefficients'] = {definition.coefficient_key(k): v for k, v in values['coefficients'].items()}
        inputs['coefficients']['H H† X X†'] = [-.03, -.02]
        inputs['coefficients']['X^2 X†^2'] = [-.02, -.01, -.03]
        descriptions = definition.operator_inputs()
        quartic = next(row for row in descriptions if row['label'] == 'X^2 X†^2')
        self.assertEqual(quartic['kind'], 'joint_scalar_potential')
        self.assertEqual(len(quartic['invariant_labels']), 3)
        self.assertEqual(len(quartic['basis_fingerprint']), 64)
        source = (BEGIN + '\n\\begin{verbatim}\nX = ' + repr(spec) +
                  '\nphysical_inputs = ' + repr(inputs) +
                  '\noutput = "external output with spaces"\n_input_bindings = ' + repr(input_bindings(definition)) +
                  '\n\\end{verbatim}\n' + END + '\n')
        with tempfile.TemporaryDirectory(prefix='joint public input ') as temporary:
            path = Path(temporary) / 'sextet doublet.tex'
            path.write_text(source)
            project = Pipeline(model_file=path)
            prepared = project.validate()
            saved = json.loads((project.run_directory / 'spectrum.json').read_text())
            self.assertEqual(saved['coordinates'], public(prepared.compiler.coordinates))
            self.assertEqual(saved['raw_field_basis'], public(prepared.compiler.raw_field_basis))
            self.assertEqual(len(saved['transformation']), 2)
            self.assertTrue(project.run_directory.is_relative_to(Path(temporary).resolve() / 'external output with spaces'))
            self.assertEqual(set(prepared.coverage), set(lagrangian.selected_ids))
            self.assertTrue(all(row == 'resolved and numerically normalized' or row.startswith('zero_verified:')
                                for row in prepared.coverage.values()))

    def test_discovery_honours_custom_construction_limits(self):
        from model_builder import ModelDefinition
        for weak in (1, 2):
            with self.subTest(weak=weak):
                spec = dict(spin='scalar', su2=weak, su3='6', hypercharge='1/2', boson_kind='complex')
                model = ModelDefinition(1, spec, selected_operators=['X^2 X†^2'],
                                        compiler_limits={'max_output_terms': 1})
                with self.assertRaises(ResourceLimitError):
                    model.required_inputs()

    def test_complex_sextet_doublet_portal_splits_charges(self):
        lagrangian,values=benchmark()
        row=next(r for r in lagrangian.operators if r['field_content']=='H H† X X†')
        values['coefficients'][row['id']]=[-.03,-.02]
        model=prepare(lagrangian,values)
        v=(math.sqrt(2)*1.16637e-5)**-.5
        expected=[math.sqrt(40000+.025*v*v),math.sqrt(40000+.015*v*v)]
        np.testing.assert_allclose([s.mass for s in model.compiler.scalar_states],expected,rtol=1e-14)
        self.assertEqual([(s.rep,s.charge,s.real) for s in model.compiler.scalar_states],[(6,1,False),(6,0,False)])
        self.assertEqual(len(basis_fingerprint(row,model.fields)),64)

    def test_real_adjoint_triplet_and_neutral_complex_inventory(self):
        for weak,kind,charge,expected in [(3,'real','0',[(1,False),(0,True)]),(2,'complex','1/2',[(1,False),(0,True),(0,True)])]:
            with self.subTest(weak=weak,kind=kind):
                model=prepare(*benchmark('8',weak,kind,charge))
                self.assertEqual([(s.charge,s.real) for s in model.compiler.scalar_states],expected)
                np.testing.assert_allclose([s.mass for s in model.compiler.scalar_states],200,rtol=1e-14)
                for raw,mapping in model.compiler.raw_field_basis.items():
                    if raw[0].endswith('†'):continue
                    conjugate={(i,False if model.compiler.scalar_states[i].real else not bar):z.conjugate() for (i,bar),z in mapping.items()}
                    self.assertEqual(conjugate,model.compiler.raw_field_basis[raw[0]+'†',raw[1]])

    def test_each_charge_block_has_exact_generalized_masses_and_rotations(self):
        model=prepare(*benchmark('6',mixed=True))
        k=np.array([[1,.1-.05j],[.1+.05j,1]])
        m=np.array([[40000,2000.-1000j],[2000.+1000j,90000]])
        expected=np.sqrt(eigh(m,k,eigvals_only=True))
        states=model.compiler.scalar_states
        for charge in (0,1):
            indices=[i for i,s in enumerate(states) if s.charge==charge]
            rotation=model.compiler.transformation[np.ix_(indices,indices)]
            np.testing.assert_allclose(sorted(states[i].mass for i in indices),expected,rtol=1e-14)
            np.testing.assert_allclose(rotation.conj().T@k@rotation,np.eye(2),atol=1e-14)
            np.testing.assert_allclose(rotation.conj().T@m@rotation,np.diag([states[i].mass**2 for i in indices]),atol=1e-9)

    def test_native_source_keeps_exact_generators_and_rational_charge(self):
        model=prepare(*benchmark('6bar'))
        with tempfile.TemporaryDirectory() as folder:
            files,_,particles=model.compiler.write_sources(model,Path(folder),ROOT/'vendor'/'FeynRules')
            source=files[1].read_text()
        self.assertIn('QuantumNumbers -> {Y -> -1/2}',source)
        self.assertIn('pmJST2[2,1,2] = -I*Sqrt[1]/2',source)
        self.assertIn('AddGaugeRepresentation[{SU3C -> {T6,Sextet}}]',source)
        self.assertEqual([(p['charge'],p['color']) for p in particles],[(-1.,6),(0.,6)])
        self.assertEqual(len({p['pdg'] for p in particles}),len(particles))

    def test_explicit_zero_unlowered_rows_are_preserved_and_shape_checked(self):
        lagrangian,values=benchmark('3',charge='1/6')
        unlowered=[r for r in lagrangian.operators if classify(r,lagrangian._source.models[0])[0] is None]
        self.assertTrue(unlowered)
        model=prepare(lagrangian,values)
        self.assertEqual(len(model.operators),len(lagrangian.operators))
        self.assertTrue(all(model.coverage[r['id']].startswith('zero_verified') for r in unlowered))
        bad=deepcopy(values);row=unlowered[0]
        bad['coefficients'][row['id']]=1. if row['multiplicity']==1 else [1.]+[0.]*(row['multiplicity']-1)
        with self.assertRaises(CompilerResolutionError):prepare(lagrangian,bad)
        bad['coefficients'][row['id']]=[]
        with self.assertRaises(ModelResolutionError):prepare(lagrangian,bad)

    def test_generator_budget_is_explicit(self):
        lagrangian,values=benchmark();values['compiler_limits']={'max_weak_generator_entries':1}
        with self.assertRaises(ResourceLimitError):prepare(lagrangian,values)


@unittest.skipUnless(os.environ.get('RUN_JOINT_EXTERNAL')=='1' or os.environ.get('MADGRAPH_STEP_EXTERNAL_TESTS')=='1','opt-in real Mathematica/FeynRules and MadGraph export')
class JointScalarExternalTests(unittest.TestCase):
    def test_sextet_doublet_full_external_validation(self):
        from uuid import uuid4
        lagrangian,values=benchmark()
        for row in lagrangian.operators:
            if row['field_content']=='H H† X X†':values['coefficients'][row['id']]=[-.03,-.02]
            if row['field_content']=='X^2 X†^2':values['coefficients'][row['id']]=[-.02]*row['multiplicity']
        output=ROOT/'runs'/('joint_external_'+uuid4().hex[:8])
        model=prepare(lagrangian,values)
        artifact=model.export_ufo(output/'export',analytic_decays=False,timeout=600)
        self.assertEqual({p['name'] for p in artifact.particles},{'x1','x2'})
        self.assertTrue((output/'export'/'madgraph_import'/'param_card.dat').is_file())


if __name__=='__main__':unittest.main()
