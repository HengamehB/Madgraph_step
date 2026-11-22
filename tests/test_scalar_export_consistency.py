"""Light eigenmode stability and consistent Higgs interaction orders."""
from pathlib import Path
import re
import tempfile
import unittest

from feynrules_export import prepare_model,ModelResolutionError
from generic_color_compiler import _REPS
from lagrangian_engine import FieldSpec
from lagrangian_maker import make_lagrangian
from native_colour_basis import resolve_scalars

ROOT=Path(__file__).resolve().parents[1]


class ScalarExportConsistencyTests(unittest.TestCase):
    def test_legacy_norm_doublet_and_triplet_higgs_orders_leave_quadratics_unchanged(self):
        for weak,colour,charge,kind,prefix,suffix in (
                (1,'3',1,'complex','pmGM','n1'),
                (2,'1','1/2','complex','pmDO','n1'),
                (3,'1',0,'real','pmTO','')):
            result=make_lagrangian(1,dict(spin='scalar',su2=weak,su3=colour,hypercharge=charge,boson_kind=kind),show=False)
            lagrangian=result.select('all',show=False);coefficients={};selected={}
            mass='X^2' if kind=='real' else 'X X†'
            kinetic='D(X)^2' if kind=='real' else 'D(X) D(X†)'
            portal='H H† X^2' if kind=='real' else 'H H† X X†'
            for row in lagrangian.operators:
                values=[0.]*row['multiplicity']
                if row['field_content']==mass:values[0]=-20000. if kind=='real' else -40000.;selected['mass']=row
                if row['field_content']==kinetic:values[0]=.5 if kind=='real' else 1.;selected['kinetic']=row
                if row['field_content']==portal:values[0]=-.03;selected['portal']=row
                coefficients[row['id']]=values if len(values)>1 else values[0]
            model=prepare_model(lagrangian,dict(coefficients=coefficients,widths={'X':0.}))
            with tempfile.TemporaryDirectory() as folder:
                files,_,_=model.compiler.write_sources(model,Path(folder),ROOT/'vendor'/'FeynRules')
                source=files[-1].read_text()
            for label,row in selected.items():
                symbol=prefix+row['id'][1:]+suffix
                declaration=next(line for line in source.splitlines() if line.startswith(symbol+' =='))
                with self.subTest(weak=weak,label=label):
                    if label=='portal':self.assertIn('InteractionOrder -> {{NP, 1}, {QED, 2}}',declaration)
                    else:self.assertNotIn('InteractionOrder',declaration)

    def test_heavy_spectator_does_not_hide_native_scalar_tachyon(self):
        fields=tuple(FieldSpec(name,'scalar',1,(2,0),0,'complex') for name in ('X','Y'))
        for negative_mass_squared in (-1e-6,-1e-20):
            for field_scale in (1.,1e-20):
                kinetic={};potential={}
                for colour in range(6):
                    for name,mass_squared in (('X',negative_mass_squared),('Y',1e6)):
                        key=tuple(sorted(((name,colour),(name+'†',colour))))
                        kinetic[key]=field_scale
                        potential[key]=-mass_squared*field_scale
                with self.subTest(mass_squared=negative_mass_squared,field_scale=field_scale):
                    with self.assertRaisesRegex(ModelResolutionError,'tachyonic'):
                        resolve_scalars(fields,_REPS,kinetic,potential)

    def test_exact_massless_mode_with_heavy_spectator_remains_valid(self):
        fields=tuple(FieldSpec(name,'scalar',1,(2,0),0,'complex') for name in ('X','Y'))
        kinetic={};potential={}
        for colour in range(6):
            for name,mass_squared in (('X',0.),('Y',1e6)):
                key=tuple(sorted(((name,colour),(name+'†',colour))))
                kinetic[key]=1.
                if mass_squared:potential[key]=-mass_squared
        states,_,_=resolve_scalars(fields,_REPS,kinetic,potential)
        self.assertEqual([state.mass for state in states],[0.,1000.])

    def test_native_and_joint_portals_compensate_the_sm_vev_order(self):
        for weak in (1,2):
            result=make_lagrangian(1,dict(spin='scalar',su2=weak,su3='6',hypercharge=0 if weak==1 else '1/2'),show=False)
            lagrangian=result.select('all',show=False);coefficients={}
            portal=None
            for row in lagrangian.operators:
                value=[0.]*row['multiplicity']
                if row['field_content']=='D(X) D(X†)':value[0]=1.
                if row['field_content']=='X X†':value[0]=-40000.
                if row['field_content']=='H H† X X†':value[0]=-.03;portal=row
                coefficients[row['id']]=value if len(value)>1 else value[0]
            model=prepare_model(lagrangian,dict(coefficients=coefficients,widths={'X':0.}))
            with tempfile.TemporaryDirectory() as folder:
                files,_,_=model.compiler.write_sources(model,Path(folder),ROOT/'vendor'/'FeynRules')
                source=files[-1].read_text()
            symbol=('pmGCO' if weak==1 else 'pmJSO')+portal['id'][1:]+'n1'
            declaration=next(line for line in source.splitlines() if line.startswith(symbol+' =='))
            order=int(re.search(r'\{QED, (\d+)\}',declaration).group(1))
            # SM.fr assigns QED=-1 to vev. The physical hXX vertex contains
            # one vev and hhXX none, so their QED orders must be 1 and 2.
            self.assertEqual((order-1,order),(1,2))


if __name__=='__main__':unittest.main()
