"""Native-colour scalar multiplets with arbitrary weak representations.

Joint weak/colour tensors are Bose-symmetrized together. Only weak indices
are expanded in the exported model. Colour stays native, while every raw
weak component is transformed to its verified charge/colour mass block.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math

import numpy as np

from component_invariants import _su2_conjugation
from generic_color_compiler import _REPS, _index
import generic_colour_yukawa as yukawa
from lagrangian_engine import FieldSpec
from native_colour_basis import ScalarState, resolve_scalars

BASIS_VERSION = 'joint-native-colour-weak-scalar-v1'


def supports_fields(fields):
    return bool(fields) and all(f.spin == 'scalar' and f.su3 in _REPS for f in fields)


def uses_generic(fields):
    return supports_fields(fields) and any(f.su3 != (0,0) for f in fields) and any(f.su2 > 1 for f in fields)


def classify(row, fields, limits=None):
    from joint_scalar_invariants import invariant_basis, kinetic_basis
    from workflow_errors import ResourceLimitError
    kind = yukawa.classify(row, fields)
    if kind:
        return kind, ''
    names = {'H','H†'} | {token for f in fields for token in (f.name,f.name+'†')}
    tokens = [p['name'] for p in row['factors'] for _ in range(p['count'])]
    try:
        if not row['derivatives'] and set(tokens) <= names:
            invariant_basis(row, fields, limits)
            return 'joint_scalar_potential', ''
        if row['derivatives'] == 2 and len(tokens) == 2 and all(t.startswith('D(') and t[2:-1] in names for t in tokens):
            kinetic_basis(row, fields, limits)
            return 'joint_scalar_kinetic', ''
    except ResourceLimitError:
        raise
    except (ValueError,RuntimeError) as exc:
        if getattr(exc,'category',None)=='resource_limit':raise
        return None, 'compiler gap: joint invariant construction failed: '+str(exc)
    return None, 'compiler gap: this nonzero fermion or derivative family needs joint Lorentz lowering'


def basis_labels(row, fields, limits=None):
    from joint_scalar_invariants import invariant_basis, kinetic_basis
    basis,_ = (kinetic_basis if row['derivatives'] else invariant_basis)(row, fields, limits)
    return tuple(item.render() for item in basis)


def basis_fingerprint(row, fields, limits=None):
    import hashlib,json
    from joint_scalar_invariants import basis_fingerprint as joint_fingerprint
    return hashlib.sha256(json.dumps(dict(version=BASIS_VERSION,basis=joint_fingerprint(row,fields,limits)),
                                    sort_keys=True).encode()).hexdigest()


@dataclass
class WeakCoordinates:
    fields: tuple
    flat_fields: tuple
    raw_maps: dict
    labels: dict
    sources: dict


def weak_coordinates(fields):
    """Raw weak components and direct conjugates in native-colour coordinates."""
    flat, raw, labels, sources = [], {}, {}, {}
    for f in fields:
        conjugation = _su2_conjugation(f.su2-1).toarray()
        for weak in range(f.su2):
            if (f.name,weak) in raw:
                continue
            charge = f.hypercharge + Fraction(f.su2-1-2*weak,2)
            name=f.name+'W'+str(weak)
            kind = 'real' if f.kind=='real' and charge==0 else 'complex'
            flat.append(FieldSpec(name,'scalar',1,f.su3,charge,kind))
            labels[name]=f.name+str(weak+1);sources[name]=f.name
            phase = 1j if kind=='real' and conjugation[weak,weak]<0 else 1.
            raw[f.name,weak]={name:phase}
            if f.kind=='real' and charge!=0:
                reflected=f.su2-1-weak
                raw[f.name,reflected]={name+'†':complex(conjugation[reflected,weak])}
        for weak in range(f.su2):
            raw[f.name+'†',weak]={token[:-1] if token.endswith('†') else token+'†':complex(z).conjugate()
                                   for token,z in raw[f.name,weak].items()}
    return WeakCoordinates(tuple(fields),tuple(flat),raw,labels,sources)


def _physical_polynomial(polynomial, coordinates, vev, *, derivative_higgs=False):
    """Project H to unitary gauge and retain the scalar quadratic polynomial."""
    values=defaultdict(list)
    for monomial, coefficient in polynomial.items():
        new=[leg for leg in monomial if leg[0] not in ('H','H†')]
        higgs=[leg for leg in monomial if leg[0] in ('H','H†')]
        if len(new)>2 or any(weak!=1 for _,weak,_ in higgs):continue
        if derivative_higgs and (len(new)!=1 or not higgs):continue
        factor=(vev/math.sqrt(2))**len(higgs)
        if derivative_higgs:factor*=len(higgs)/vev
        maps=[coordinates.raw_maps[token,weak] for token,weak,_ in new]
        for terms in product(*(mapping.items() for mapping in maps)):
            key=tuple(sorted((token,old[2]) for (token,_),old in zip(terms,new)))
            values[key].append(coefficient*factor*math.prod(z for _,z in terms))
    return {key:complex(math.fsum(z.real for z in terms),math.fsum(z.imag for z in terms))
            for key,terms in values.items()}


def _add(target, source, factor=1):
    for key,value in source.items():
        target[key]=target.get(key,0j)+factor*value
    for key in [key for key,value in target.items() if value==0]:target.pop(key)


def _conjugate_raw(polynomial):
    return {tuple(sorted((token[:-1] if token.endswith('†') else token+'†',weak,colour)
                         for token,weak,colour in monomial)):complex(value).conjugate()
            for monomial,value in polynomial.items()}


@dataclass
class JointScalarCompiler:
    fields: tuple
    operators: tuple
    coordinates: WeakCoordinates
    bases: dict
    scalar_states: tuple
    raw_field_basis: dict
    transformation: object
    higgs_mass: float
    stock_sm_neutrinos: bool=True

    @property
    def masses(self):
        return {'H':self.higgs_mass,**{state.label:state.mass for state in self.scalar_states}}

    def write_sources(self, model, source, fr):
        from feynrules_export import _wl,_internal,CompilerResolutionError,ModelResolutionError
        from generic_fermion_compiler import weak_generators
        sm=(fr/'Models'/'SM'/'SM.fr').read_text()
        marker='Mass            -> {MH,125}'
        if marker not in sm:raise CompilerResolutionError('Bundled SM Higgs mass declaration changed')
        sm=sm.replace(marker,f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        dimensions=sorted({f.su2 for f in self.fields})
        representations=['{Ta,SU2D}']+[f'{{pmJST{n},pmJSWeak{n}}}' for n in dimensions if n>1]
        marker='Representations   -> {Ta,SU2D}'
        if marker not in sm:raise CompilerResolutionError('Bundled SM weak declaration changed')
        sm=sm.replace(marker,'Representations   -> {'+','.join(representations)+'}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        parameters,constants=[],{}
        def scalar(value):
            value=float(value)
            if value==int(value):return str(int(value))
            if value<0:return '-('+scalar(-value)+')'
            if value not in constants:
                symbol=f'pmJSN{len(constants)+1}';constants[value]=symbol
                parameters.append(_internal(symbol,value))
            return constants[value]
        def literal(value):
            value=complex(value)
            return scalar(value.real) if value.imag==0 else '('+scalar(value.real)+'+I*('+scalar(value.imag)+'))'
        definitions,classes,particles=[],[],[]
        for n in dimensions:
            definitions.append(f'IndexRange[Index[pmJSWeak{n}]] = Unfold[Range[{n}]];')
            if n==1:continue
            definitions.append(f'pmJST{n}[Index[SU2W,a_Integer],Index[pmJSWeak{n},b_Integer],Index[pmJSWeak{n},c_Integer]] := pmJST{n}[a,b,c];')
            for a,generator in enumerate(weak_generators(n),1):
                for i in range(n):
                    for j in range(n):
                        if a==3:value=str(Fraction(n-1-2*i,2)) if i==j else '0'
                        elif abs(i-j)==1:
                            k=max(i,j);value=f'Sqrt[{k*(n-k)}]/2'
                            if a==2:value=('-I*' if j>i else 'I*')+value
                        else:value='0'
                        definitions.append(f'pmJST{n}[{a},{i+1},{j+1}] = {value};')
        if any(abs(_REPS[f.su3])==6 for f in self.fields):
            definitions.extend(('IndexRange[Index[Sextet]] = NoUnfold[Range[6]];',
                                'IndexStyle[Sextet,sx];','AddGaugeRepresentation[{SU3C -> {T6,Sextet}}];'))
        lepton_numbers, lepton_note = yukawa.lepton_labels(self.operators, self.raw_field_basis)
        for index,state in enumerate(self.scalar_states,1):
            symbol=f'pmJSP{index}';name=state.name
            anti=name if state.real else name.removesuffix('bar') if name.endswith('bar') else name+'bar'
            colour=f'Indices -> {{Index[{_index(state.rep)}]}},' if state.rep!=1 else ''
            lepton = f', LeptonNumber -> {lepton_numbers[index-1]}' if index-1 in lepton_numbers else ''
            width=model.inputs['widths'][state.source]
            classes.append(f'S[{50000+index}] == {{ClassName -> {symbol}, SelfConjugate -> {str(state.real)}, {colour}'
                           f'QuantumNumbers -> {{Q -> {state.charge}{lepton}}}, Mass -> {{pmJSM{index},{_wl(state.mass)}}},'
                           f'Width -> {{pmJSW{index},{_wl(width)}}}, ParticleName -> {_wl(name)},'
                           +('' if state.real else f'AntiParticleName -> {_wl(anti)},')
                           +f'PDG -> {9700000+index}, PropagatorLabel -> {_wl(name)}, PropagatorType -> D,'
                           f'PropagatorArrow -> {"None" if state.real else "Forward"}}}')
            particles.append(dict(name=name,antiname=anti,pdg=9700000+index,source=state.source,spin=1,
                                  mass=state.mass,width=width,charge=float(state.charge),color=state.rep))
        symbols={('H',0):'0',('H†',0):'0',('H',1):'((vev+H)/Sqrt[2])',('H†',1):'((vev+H)/Sqrt[2])'}
        for raw,mapping in self.raw_field_basis.items():
            terms=[]
            for (index,bar),value in mapping.items():
                state=self.scalar_states[index]
                symbol=f'pmJSP{index+1}'+('bar' if bar and not state.real else '')
                if state.rep!=1:symbol+='[{index}]'
                terms.append('('+literal(value)+')*'+symbol)
            symbols[raw]='('+('+'.join(terms) or '0')+')'
        covariant={}
        for fi,f in enumerate(self.fields,1):
            base=f'pmJSU{fi}';rep=_REPS[f.su3];conjugated=rep<0
            colour=',Index['+_index(rep)+']' if abs(rep)!=1 else ''
            index=f'Index[pmJSWeak{f.su2}]'+colour
            c=_su2_conjugation(f.su2-1).toarray()
            rules=[]
            for weak in range(f.su2):
                raw=(f.name+'†',f.su2-1-weak) if conjugated else (f.name,weak)
                value=symbols[raw].replace('{index}','cc')
                if conjugated:value='('+literal(c[weak,f.su2-1-weak])+')*'+value
                args=',cc_' if colour else ''
                rules.append(f'{base}[{weak+1}{args}] :> {value}')
            classes.append(f'S[{50000+len(self.scalar_states)+fi}] == {{ClassName -> {base}, SelfConjugate -> False, Unphysical -> True,'
                           f'Indices -> {{{index}}}, FlavorIndex -> pmJSWeak{f.su2}, QuantumNumbers -> {{Y -> {-f.hypercharge if conjugated else f.hypercharge}}},'
                           'Definitions -> {'+','.join(rules)+'}}')
            for weak in range(f.su2):
                i=f.su2-1-weak if conjugated else weak
                phase=c[i,weak] if conjugated else 1.
                for dagger in (False,True):
                    suffix='bar' if dagger!=conjugated else ''
                    args=f'{i+1}'+(',{index}' if colour else '')
                    covariant[f.name+('†' if dagger else ''),weak]='('+literal(phase)+f')*DC[{base}{suffix}[{args}],mu]'
        terms=[];has_np=False
        for op in self.operators:
            if op.kind in yukawa.KINDS:
                declarations, interactions = yukawa.render(op, symbols)
                parameters.extend(declarations)
                terms.extend(interactions)
                has_np |= bool(interactions)
                continue
            if op.kind not in ('joint_scalar_kinetic','joint_scalar_potential'):continue
            for i,(coefficient,invariant) in enumerate(zip(op.coefficients,self.bases[op.operator_id]),1):
                if coefficient==0:continue
                parameter=f'pmJSO{op.operator_id[1:]}n{i}'
                order=op.kind=='joint_scalar_potential' and sum(n for _,n in op.factors)>2
                declaration=_internal(parameter,coefficient,order=order)
                higgs=sum(count for token,count in op.factors if token in ('H','H†'))
                if order and higgs:
                    declaration=declaration.replace('InteractionOrder -> {NP, 1}',
                        f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
                parameters.append(declaration);has_np|=order
                expression=invariant.render(covariant if op.kind=='joint_scalar_kinetic' else symbols)
                term=parameter+'*('+expression+')'
                terms.append('('+term+'+HC['+term+'])' if op.add_hc else term)
        smfile=source/'SM_joint_scalar.fr';smfile.write_text(sm)
        extension=source/'generic_joint_scalar.fr'
        extension.write_text('M$ModelName = '+_wl(model.model_name)+';\n'
            +lepton_note+
            'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}'+(',{NP,1}' if has_np else '')+'};\n'
            +'\n'.join(definitions)+'\nM$ClassesDescription = {\n'+',\n'.join(classes)+'\n};\n'
            +'M$Parameters = {\n'+',\n'.join(parameters)+'\n};\n'
            +'LJointScalar := Block[{mu,cc,'+','.join(f'c{i}' for i in range(1,24))+'}, ExpandIndices['
            +(' + '.join(terms) or '0')+', FlavorExpand -> {SU2D,SU2W,'+','.join(f'pmJSWeak{n}' for n in dimensions)+'}]];\n')
        return [smfile,extension],'LSM + LJointScalar',particles


def prepare(lagrangian, inputs, runtime=None):
    from copy import deepcopy
    from feynrules_export import _fields,_coefficients,_number,ResolvedOperator,PreparedModel,CompilerResolutionError,ModelResolutionError
    from joint_scalar_invariants import invariant_basis,kinetic_basis
    fields=tuple(_fields(lagrangian))
    if not uses_generic(fields):return None
    inputs=deepcopy(inputs)
    from compiler_limits import DEFAULTS,normalize_limits
    from workflow_errors import ResourceLimitError
    limits=normalize_limits(inputs.get('compiler_limits'))
    entries=sum(3*n*n for n in {f.su2 for f in fields})
    budget=limits.get('max_weak_generator_entries',DEFAULTS['max_weak_generator_entries'])
    if entries>budget:
        raise ResourceLimitError(f'Joint weak generators require {entries} entries; max_weak_generator_entries={budget}')
    if any(_number(inputs.get('vevs',{}).get(f.name,0.),'vevs.'+f.name,real=True)!=0 for f in fields):
        raise CompilerResolutionError('Joint scalar compiler currently requires a colour/electromagnetism-preserving zero new-field VEV vacuum')
    gf=_number(inputs.get('sm_inputs',{}).get('Gf',1.16637e-5),'Gf',real=True)
    mh=_number(inputs.get('higgs_mass',125.),'higgs_mass',real=True)
    if gf<=0 or mh<=0:raise ModelResolutionError('Gf and higgs_mass must be positive')
    vev=(math.sqrt(2)*gf)**-.5
    coefficients=_coefficients(lagrangian,inputs)
    coordinates=weak_coordinates(fields)
    operators,bases,kinetic,quadratic,higgs_mixing=[],{},{},{},{}
    for row in lagrangian.operators:
        values=coefficients[row['id']]
        # Complete coefficient vectors are shape-validated above. An exact zero
        # needs no tensor construction, including otherwise unresolved families.
        kind,reason=('zero_verified','') if all(value==0 for value in values) else classify(row,fields,inputs.get('compiler_limits'))
        if reason:
            if any(value!=0 for value in values):raise CompilerResolutionError(f'{row["id"]}: {reason}')
            kind='zero_verified'
        op=ResolvedOperator(row['id'],kind,tuple((p['name'],p['count']) for p in row['factors']),values,
                            add_hc=row['field_content']!=row['conjugate_content'])
        operators.append(op)
        # Matter Yukawas use the same mass-basis map at emission; with zero
        # coloured VEVs they do not enter either scalar quadratic form.
        if kind=='zero_verified' or kind in yukawa.KINDS:continue
        if kind=='joint_scalar_kinetic' and any(p['name'] in ('D(H)','D(H†)') for p in row['factors']):
            if any(value!=0 for value in values):raise CompilerResolutionError('Nonzero Higgs/new-multiplet kinetic mixing requires a gauge/Goldstone transformation')
            operators[-1]=ResolvedOperator(op.operator_id,'zero_verified',op.factors,op.coefficients)
            continue
        basis,polynomials=(kinetic_basis if kind=='joint_scalar_kinetic' else invariant_basis)(row,fields,inputs.get('compiler_limits'))
        bases[row['id']]=basis
        for coefficient,polynomial in zip(values,polynomials):
            if coefficient==0:continue
            value={m:coefficient*z for m,z in polynomial.items()}
            if op.add_hc:_add(value,_conjugate_raw(value))
            _add(kinetic if kind=='joint_scalar_kinetic' else quadratic,
                 _physical_polynomial(value,coordinates,vev))
            if kind=='joint_scalar_potential':_add(higgs_mixing,_physical_polynomial(value,coordinates,vev,derivative_higgs=True))
    states,flat_maps,transformation=resolve_scalars(coordinates.flat_fields,_REPS,kinetic,quadratic)
    if higgs_mixing:raise CompilerResolutionError('Joint coloured model needs simultaneous SM Higgs/singlet interaction transformation')
    renamed=[]
    for state in states:
        base=coordinates.labels[state.source]
        suffix=state.label.removeprefix(state.source)
        suffix={'1':'r','2':'i'}.get(suffix,suffix)
        name=(base+suffix).lower()+('bar' if state.name.endswith('bar') else '')
        renamed.append(ScalarState(base+suffix,coordinates.sources[state.source],state.rep,state.charge,state.real,name,state.mass))
    raw_maps={}
    for raw,mapping in coordinates.raw_maps.items():
        result={}
        for token,z in mapping.items():_add(result,flat_maps[token],z)
        raw_maps[raw]=result
    inputs.setdefault('widths',{})
    for f in fields:
        width=_number(inputs['widths'].get(f.name,0.),'widths.'+f.name,real=True)
        if width<0:raise ModelResolutionError('Widths must be nonnegative')
        inputs['widths'][f.name]=width
    compiler=JointScalarCompiler(fields,tuple(operators),coordinates,bases,tuple(renamed),raw_maps,transformation,float(mh))
    return PreparedModel(lagrangian,inputs,tuple(operators),fields,None,runtime,compiler=compiler)
