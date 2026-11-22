"""Massive colour-singlet vector matter with declared Proca dynamics.

Arbitrary weak representations and complete nonderivative vector invariants
are generated algorithmically. All four covariant quadratic derivative terms
are retained: their differences contain physical gauge-field interactions.
The ghost-free constraint and positive kinetic/mass matrices are checked before
charge-sector diagonalization. Gauge/current mixing remains a compiler gap.
"""
from dataclasses import dataclass
from collections import Counter
from fractions import Fraction
from itertools import product
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from generic_scalar_compiler import coordinates,add,mul,conjugate,_quadratic_matrix,_shift
from generic_vector_invariants import invariant_basis,BASIS_VERSION


def supports_fields(fields):
    return bool(fields) and all(f.spin=='vector' and f.su3==(0,0) for f in fields)


def uses_generic(fields):
    return supports_fields(fields) and not (len(fields)==1 and fields[0].su2==1 and fields[0].kind=='real' and fields[0].hypercharge==0)


def classify(row,fields):
    names={n for f in fields for n in (f.name,f.name+'†')}
    counts=Counter({p['name']:p['count'] for p in row['factors']})
    if row['derivatives']==2 and sum(counts.values())==2 and row['multiplicity']==4 and all(n.startswith('D(') and n[2:-1] in names for n in counts):
        return 'generic_vector_kinetic',''
    if not row['derivatives'] and set(counts)<=names|{'H','H†'}:
        return 'generic_vector_potential',''
    if row['derivatives']==1 and sum(counts.values())==3 and all(
            n in names or n.startswith('D(') and n[2:-1] in names for n in counts):
        return 'generic_vector_cubic_derivative',''
    from generic_vector_currents import classify as current_classify
    if current_classify(row,fields) is not None:return 'generic_vector_current',''
    from generic_vector_field_strength import classify as strength_classify
    if strength_classify(row,fields) is not None:return 'generic_vector_field_strength',''
    return None,'compiler gap: Higgs/vector derivative terms, gauge kinetic mixing or different-species SM fermion currents need additional lowering'


def basis_labels(row,fields,limits=None):
    kind,reason=classify(row,fields)
    if kind=='generic_vector_kinetic':
        return ('D_mu V_nu D^mu W^nu','(D_mu V^mu)(D_nu W^nu)',
                'D_mu V_nu D^nu W^mu','epsilon(mu,nu,rho,sigma) D_mu V_nu D_rho W_sigma')
    if kind=='generic_vector_current':
        from generic_vector_currents import basis_labels as current_labels
        return current_labels(row,fields,limits)
    if kind=='generic_vector_field_strength':
        from generic_vector_field_strength import basis_labels as strength_labels
        return strength_labels(row,fields,limits)
    if kind in ('generic_vector_potential','generic_vector_cubic_derivative'):
        c=coordinates(tuple(fields));basis=invariant_basis(row,fields,limits)['basis']
        return tuple(' + '.join(f'({z.real:g}{z.imag:+g}j)*'+'*'.join('h' if i==0 else ('D'+str(i%20//4-1)+'(' if i%20//4 else '')+c.labels[i//20]+f'[{i%4}]'+(')' if i%20//4 else '') for i in m)
                               for m,z in sorted(p.items())) for p in basis)
    return tuple(reason for _ in range(row['multiplicity']))


def basis_fingerprint(row,fields,limits=None):
    import hashlib,json
    return hashlib.sha256(json.dumps(dict(version=BASIS_VERSION,labels=basis_labels(row,fields,limits)),sort_keys=True).encode()).hexdigest()


def _gauge_basis(row,fields,limits):
    from generic_scalar_compiler import invariant_basis as scalar_basis
    fake=dict(row,multiplicity=1)
    return scalar_basis(fake,fields,limits)[0]


def _covariant_rows(row,fields,c,polynomial):
    factors=[p['name'][2:-1] for p in row['factors'] for _ in range(p['count'])]
    left,right=[n.removesuffix('†') for n in factors];leftbar,rightbar=[n.endswith('†') for n in factors]
    alternatives=[];keys=[]
    for i,j in product(range(len(c.components[left])),range(len(c.components[right]))):
        a=c.components[left][i];b=c.components[right][j]
        if leftbar:a=conjugate(a,c.bars)
        if rightbar:b=conjugate(b,c.bars)
        candidate=mul(a,b);alternatives.append(candidate);keys.append(((left,i,leftbar),(right,j,rightbar)))
        if row['field_content']!=row['conjugate_content']:
            alternatives.append(conjugate(candidate,c.bars));keys.append(((left,i,not leftbar),(right,j,not rightbar)))
    monomials=sorted(set(polynomial)|set().union(*(p.keys() for p in alternatives)))
    matrix=np.array([[p.get(m,0) for p in alternatives] for m in monomials],complex)
    target=np.array([polynomial.get(m,0) for m in monomials],complex)
    solution=np.linalg.lstsq(matrix,target,rcond=1e-11)[0]
    if not np.allclose(matrix@solution,target,atol=1e-10*max(abs(target),default=0.),rtol=1e-9):raise ValueError('Covariant vector quadratic reconstruction failed')
    scale=max(abs(solution),default=0.)
    # This projection is applied to dimensionless invariant tensors, before
    # multiplying any physical input. Remove its numerical null components;
    # arbitrarily small real or imaginary user coefficients remain untouched.
    solution=np.array([complex(z.real if abs(z.real)>1e-12*scale else 0.,
                               z.imag if abs(z.imag)>1e-12*scale else 0.) for z in solution])
    return tuple((a,b,complex(z)) for (a,b),z in zip(keys,solution) if z!=0)


@dataclass
class GenericVectorCompiler:
    fields:tuple
    operators:tuple
    coordinates:object
    bases:dict
    kinetic_rows:tuple
    transformation:np.ndarray
    eigenmasses:dict
    vev:float
    higgs_mass:float
    stock_sm_neutrinos:bool=True

    @property
    def masses(self):
        return {'H':self.higgs_mass,**{self.coordinates.labels[i]:m for i,m in self.eigenmasses.items()}}

    def write_sources(self,model,source,fr):
        from feynrules_export import _wl,_internal,CompilerResolutionError,ModelResolutionError
        from generic_fermion_compiler import weak_generators
        c=self.coordinates;params=[];constants={};has_np=False
        def literal(value):
            z=complex(value)
            def scalar(x):
                if x<0:return '-('+scalar(-x)+')'
                if x==0:return '0'
                if round(x)!=0 and abs(x-round(x))<1e-13*x:return str(round(x))
                q=Fraction(x).limit_denominator(1000)
                if q and abs(float(q)-x)<1e-14*x:return str(q)
                q=Fraction(x*x).limit_denominator(1000)
                if q and abs(math.sqrt(float(q))-x)<1e-14*x:return 'Sqrt['+str(q)+']'
                key=format(x,'.15g')
                if key not in constants:
                    symbol=f'pmGVN{len(constants)+1}';constants[key]=symbol;params.append(_internal(symbol,x))
                return constants[key]
            return scalar(z.real) if z.imag==0 else '('+scalar(z.real)+'+I*('+scalar(z.imag)+'))'
        symbols={i:f'pmGV{i}' for i in self.eigenmasses}
        for i in self.eigenmasses:symbols[c.bars[i]]=symbols[i] if c.bars[i]==i else symbols[i]+'bar'
        def component(index,mu):
            if index==0:return '(vev+H)'
            return '('+(' + '.join(f'({literal(self.transformation[index,j])})*{symbols[j]}[{mu}]'
                                   for j in range(1,len(c.labels)) if self.transformation[index,j]!=0) or '0')+')'
        def gauge_component(name,k,mu):
            return '('+(' + '.join('('+literal(z)+')*'+component(m[0],mu)
                                  for m,z in c.components[name][k].items()) or '0')+')'
        definitions=[];classes=[];particles=[];dimensions=sorted({f.su2 for f in self.fields})
        for n in dimensions:
            definitions.append(f'IndexRange[Index[pmGVWeak{n}]] = Unfold[Range[{n}]];')
            if n>1:
                definitions.append(f'pmGVT{n}[Index[SU2W,a_Integer],Index[pmGVWeak{n},b_Integer],Index[pmGVWeak{n},d_Integer]] := pmGVT{n}[a,b,d];')
                for a,matrix in enumerate(weak_generators(n),1):
                    for i,j in product(range(n),repeat=2):definitions.append(f'pmGVT{n}[{a},{i+1},{j+1}] = {literal(matrix[i,j])};')
        for i,mass in self.eigenmasses.items():
            real=c.bars[i]==i;name=c.labels[i];anti=name if real else name+'bar';pdg=9700000+i
            width=float(model.inputs['widths'].get(c.sources[i],0.))
            extra='' if real else f'AntiParticleName -> {_wl(anti)}, QuantumNumbers -> {{Q -> {str(c.charges[i])}}},'
            classes.append(f'V[{700+i}] == {{ClassName -> {symbols[i]}, SelfConjugate -> {str(real)}, {extra}'
                           f'Mass -> {{pmGVM{i},{_wl(mass)}}}, Width -> {{pmGVW{i},{_wl(width)}}},'
                           f'ParticleName -> {_wl(name)}, PDG -> {pdg}, PropagatorLabel -> {_wl(name)},'
                           f'PropagatorType -> Sine, PropagatorArrow -> {"None" if real else "Forward"}}}')
            particles.append(dict(name=name,antiname=anti,pdg=pdg,source=c.sources[i],spin=3,color=1,
                                  charge=float(c.charges[i]),mass=mass,width=width))
        original={}
        for fi,f in enumerate(self.fields,1):
            symbol=f'pmGVU{fi}';original[f.name]=symbol
            defs=[f'{symbol}[mu_,{k+1}] -> '+gauge_component(f.name,k,'mu') for k in range(f.su2)]
            classes.append(f'V[{800+max(self.eigenmasses)+fi}] == {{ClassName -> {symbol}, SelfConjugate -> False, Unphysical -> True,'
                           f'Indices -> {{Index[pmGVWeak{f.su2}]}}, FlavorIndex -> pmGVWeak{f.su2}, QuantumNumbers -> {{Y -> {str(f.hypercharge)}}},'
                           'Definitions -> {'+', '.join(defs)+'}}')
        terms=[]
        for structure,left,right,z in self.kinetic_rows:
            def field(item,mu):return original[item[0]]+('bar' if item[2] else '')+f'[{mu},{item[1]+1}]'
            if structure==0:term=f'DC[{field(left,"nu")},mu]*DC[{field(right,"nu")},mu]'
            elif structure==1:term=f'DC[{field(left,"mu")},mu]*DC[{field(right,"nu")},nu]'
            elif structure==2:term=f'DC[{field(left,"nu")},mu]*DC[{field(right,"mu")},nu]'
            else:term=f'Eps[mu,nu,rho,sigma]*DC[{field(left,"nu")},mu]*DC[{field(right,"sigma")},rho]'
            terms.append('('+literal(z)+')*'+term)
        for op in self.operators:
            if op.kind=='generic_vector_field_strength':
                from generic_vector_field_strength import source_candidates
                basis=self.bases[op.operator_id]
                candidates=source_candidates(basis,literal,gauge_component)
                for k,(value,transform) in enumerate(zip(op.coefficients,basis['transform']),1):
                    if value==0:continue
                    name=f'pmGVF{op.operator_id[1:]}n{k}';params.append(_internal(name,value,order=True));has_np=True
                    expression='('+' + '.join('('+literal(z)+')*'+e for z,e in zip(transform,candidates) if abs(z)>1e-10)+')'
                    term=name+'*'+expression
                    terms.append('('+term+'+HC['+term+'])' if op.add_hc else term)
                continue
            if op.kind=='generic_vector_current':
                from generic_vector_currents import current
                species,gauge,flavour_basis=self.bases[op.operator_id]
                for k,(value,flavour) in enumerate(zip(op.coefficients,flavour_basis),1):
                    if value==0:continue
                    name=f'pmGVJ{op.operator_id[1:]}n{k}';params.append(_internal(name,value,order=True));has_np=True
                    pieces=['('+literal(z*phase)+')*('+current(species,a,f,b,g)+')*'+component(i,'mu')
                            for f,g,phase in flavour for (a,b,i),z in sorted(gauge.items())]
                    term=name+'*('+' + '.join(pieces)+')'
                    terms.append('('+term+'+HC['+term+'])' if op.add_hc else term)
                continue
            if op.kind not in ('generic_vector_potential','generic_vector_cubic_derivative'):continue
            basis=self.bases[op.operator_id];names=basis['names'];vector_slots=basis['vectors'];derivative_slots=basis['derivative_slots'];candidate_sources=[]
            for tensor,contraction in basis['candidates']:
                lorentz={};n_lorentz=len(vector_slots)+len(derivative_slots)
                prefactor=''
                if contraction=='epsilon':
                    lorentz=dict(enumerate(('mu','nu','rho','sigma')));prefactor='Eps[mu,nu,rho,sigma]*'
                else:
                    for (a,b),mu in zip(contraction,('mu','nu')):
                        lorentz[a]=lorentz[b]=mu
                pieces=[]
                for indices,value in tensor.items():
                    factors=[]
                    for slot,(name,k) in enumerate(zip(names,indices)):
                        mu=lorentz[vector_slots.index(slot)] if slot in vector_slots else 'mu'
                        if slot in derivative_slots:
                            from component_invariants import _su2_conjugation
                            derivative=lorentz[len(vector_slots)+derivative_slots.index(slot)]
                            bare=name.removesuffix('†');n=len(c.components[bare]);symbol=original[bare]
                            factor=1
                            if name.endswith('†'):
                                factor=_su2_conjugation(n-1).toarray()[k,n-1-k];k=n-1-k;symbol+='bar'
                            factors.append('('+literal(factor)+f')*DC[{symbol}[{mu},{k+1}],{derivative}]')
                        else:factors.append(gauge_component(name,k,mu))
                    pieces.append('('+literal(value)+')*'+prefactor+'*'.join(factors))
                candidate_sources.append('('+' + '.join(pieces)+')')
            hermitian=[]
            for index,phase in basis['hermitian_candidates']:
                e=candidate_sources[index]
                hermitian.append(e if phase==0 else '('+e+'+HC['+e+'])' if phase==1 else 'I*('+e+'-HC['+e+'])')
            for k,(value,transform) in enumerate(zip(op.coefficients,basis['transform']),1):
                if value==0:continue
                name=f'pmGVO{op.operator_id[1:]}n{k}';interaction=len(names)>2
                declaration=_internal(name,value,order=interaction)
                higgs=sum(n in ('H','H†') for n in names)
                if interaction and higgs:declaration=declaration.replace('InteractionOrder -> {NP, 1}',f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
                params.append(declaration);has_np|=interaction
                expression='('+' + '.join('('+literal(z)+')*'+e for z,e in zip(transform,hermitian) if abs(z)>1e-10)+')'
                term=name+'*'+expression
                terms.append('('+term+'+HC['+term+'])' if op.add_hc else term)
        sm=(fr/'Models'/'SM'/'SM.fr').read_text()
        sm=sm.replace('Mass            -> {MH,125}',f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        reps=['{Ta,SU2D}']+[f'{{pmGVT{n},pmGVWeak{n}}}' for n in dimensions if n>1]
        sm=sm.replace('Representations   -> {Ta,SU2D}','Representations   -> {'+', '.join(reps)+'}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile=source/'SM_generic_vector.fr';smfile.write_text(sm)
        extension=source/'generic_vector.fr'
        extension.write_text('M$ModelName = '+_wl(model.model_name)+';\n'
            +'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}'+(',{NP,1}' if has_np else '')+'};\n'
            +'\n'.join(definitions)+'\nM$ClassesDescription = {\n'+',\n'.join(classes)+'\n};\n'
            +'M$Parameters = {\n'+',\n'.join(params)+'\n};\n'
            +'LGenericVector := Block[{mu,nu,rho,sigma,s1,s2,s3,cc}, ExpandIndices['+' + '.join(terms)
            +', FlavorExpand -> {SU2D,SU2W,'+', '.join(f'pmGVWeak{n}' for n in dimensions)+'}]];\n')
        return [smfile,extension],'LSM + LGenericVector',particles


def prepare(lagrangian,inputs,runtime=None):
    from feynrules_export import _fields,_coefficients,_number,ResolvedOperator,PreparedModel,ModelResolutionError,CompilerResolutionError
    fields=tuple(_fields(lagrangian))
    if not supports_fields(fields):return None
    for f in fields:
        spec=inputs.get('vector_specifications',{}).get(f.name,{})
        if not isinstance(spec,dict) or spec.get('kind')!='proca':
            raise ModelResolutionError(f'{f.name}: declare vector_specifications[{f.name!r}]={{"kind":"proca"}} for a massive effective vector theory')
    c=coordinates(fields);size=len(c.labels);coefficients=_coefficients(lagrangian,inputs)
    gf=_number(inputs.get('sm_inputs',{}).get('Gf',1.16637e-5),'Gf',real=True)
    mh=_number(inputs.get('higgs_mass',125.),'higgs_mass',real=True)
    if gf<=0 or mh<=0:raise ModelResolutionError('Gf and Higgs mass must be positive')
    vev=(math.sqrt(2)*gf)**-.5;operators=[];bases={};potential={};kinetics=[{} for _ in range(4)];kinetic_rows=[]
    for row in lagrangian.operators:
        kind,reason=classify(row,fields);values=coefficients[row['id']]
        factors=tuple((p['name'],p['count']) for p in row['factors'])
        add_hc=row['field_content']!=row['conjugate_content']
        if reason:
            if all(z==0 for z in values):operators.append(ResolvedOperator(row['id'],'zero_verified',factors,values));continue
            raise CompilerResolutionError(f"{row['id']}: {reason}")
        operators.append(ResolvedOperator(row['id'],kind,factors,values,add_hc=add_hc))
        try:
            if kind=='generic_vector_field_strength':
                from generic_vector_field_strength import invariant_basis as strength_basis
                bases[row['id']]=strength_basis(row,fields,inputs.get('compiler_limits'))
            elif kind=='generic_vector_current':
                from generic_vector_currents import classify as current_classify,gauge_basis,flavour_basis
                species,_=current_classify(row,fields)
                bases[row['id']]=(species,gauge_basis(row,fields,inputs.get('compiler_limits')),flavour_basis(row))
            elif kind=='generic_vector_kinetic':
                gauge=_gauge_basis(row,fields,inputs.get('compiler_limits'))
                # Each catalogue bilinear has an independent gauge tensor.
                # Compensated summation catches a genuinely nonzero tiny
                # longitudinal input that ordinary +/- large coefficients
                # would erase. Its scale is the smallest contributing input,
                # so a tiny declared ghost term is not treated as roundoff.
                longitudinal_value=complex(math.fsum(z.real for z in values[:3]),
                                           math.fsum(z.imag for z in values[:3]))
                longitudinal_scale=min((abs(z) for z in values[:3] if z!=0),default=0.)
                if abs(longitudinal_value)>1e-12*longitudinal_scale:
                    raise ModelResolutionError(f"{row['id']}: Proca ghost-free matrix constraint c1+c2+c3=0 failed")
                unit_rows=_covariant_rows(row,fields,c,gauge)
                for k,value in enumerate(values):
                    p={m:z*value for m,z in gauge.items()}
                    if add_hc:p=add(p,conjugate(p,c.bars))
                    kinetics[k]=add(kinetics[k],p)
                    kinetic_rows.extend((k,a,b,z*value) for a,b,z in unit_rows if value!=0)
                    if add_hc:
                        kinetic_rows.extend((k,(a[0],a[1],not a[2]),(b[0],b[1],not b[2]),(z*value).conjugate())
                                            for a,b,z in unit_rows if value!=0)
            else:
                basis=invariant_basis(row,fields,inputs.get('compiler_limits'));bases[row['id']]=basis
                for p,value in zip(basis['basis'],values):
                    if kind!='generic_vector_potential':continue
                    scalar={tuple(i//20 for i in m):z*value for m,z in p.items() if all(i%20==0 for i in m)}
                    if add_hc:scalar=add(scalar,conjugate(scalar,c.bars))
                    potential=add(potential,scalar)
        except ModelResolutionError:
            raise
        except (ValueError,RuntimeError) as exc:
            if getattr(exc,'category','')=='resource_limit':raise
            raise CompilerResolutionError(f"{row['id']}: vector invariant construction failed: {exc}") from exc
    potential=_shift(potential,[vev]+[0.]*(size-1),size)
    ghost=add(add(kinetics[0],kinetics[1]),kinetics[2]);transformation=np.zeros((size,size),complex);masses={}
    groups={}
    for i,q in enumerate(c.charges):
        if i and q>=0:groups.setdefault(q,[]).append(i)
    for charge,indices in groups.items():
        real=charge==0
        k=-_quadratic_matrix(kinetics[0],indices,c.bars,real)
        longitudinal=_quadratic_matrix(ghost,indices,c.bars,real)
        m=_quadratic_matrix(potential,indices,c.bars,real)
        scale=max(abs(k).max(),abs(longitudinal).max())
        if not np.allclose(longitudinal,0,atol=1e-10*scale):
            raise ModelResolutionError(f'Charge {charge}: Proca ghost-free matrix constraint c1+c2+c3=0 failed')
        if np.diag(k).real.min()<=0:raise ModelResolutionError(f'Charge {charge}: vector kinetic matrix is not positive definite')
        scaling=1/np.sqrt(np.diag(k).real);normalized_k=scaling[:,None]*k*scaling[None,:]
        normalized_m=scaling[:,None]*m*scaling[None,:]
        keigen=np.linalg.eigvalsh(normalized_k)
        if keigen.min()<=0:raise ModelResolutionError(f'Charge {charge}: vector kinetic matrix is not positive definite')
        if keigen.min()<=1e-12*keigen.max():raise CompilerResolutionError('Vector kinetic matrix is too ill-conditioned for floating-point diagonalization')
        eigen,rotation=eigh(normalized_m,normalized_k);rotation=scaling[:,None]*rotation
        if eigen.min()<=0:raise ModelResolutionError(f'Charge {charge}: a massive Proca model requires strictly positive mass squared')
        _,assignment=linear_sum_assignment(-abs(rotation));rotation=rotation[:,assignment];eigen=eigen[assignment]
        for j in range(len(indices)):
            pivot=int(np.argmax(abs(rotation[:,j])));rotation[:,j]/=rotation[pivot,j]/abs(rotation[pivot,j])
        if not np.allclose(rotation.conj().T@k@rotation,np.eye(len(indices)),atol=1e-8) or not np.allclose(rotation.conj().T@m@rotation,np.diag(eigen),atol=1e-6,rtol=1e-8):
            raise CompilerResolutionError('Vector canonical mass-basis residual failed')
        for a,i in enumerate(indices):
            masses[i]=math.sqrt(float(eigen[a]))
            for b,j in enumerate(indices):
                transformation[i,j]=rotation[a,b]
                if not real:transformation[c.bars[i],c.bars[j]]=rotation[a,b].conjugate()
    compiler=GenericVectorCompiler(fields,tuple(operators),c,bases,tuple(kinetic_rows),transformation,masses,vev,mh)
    return PreparedModel(lagrangian,inputs,tuple(operators),fields,{'masses':[mh]},runtime,compiler=compiler)
