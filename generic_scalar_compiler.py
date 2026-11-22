"""Invariant-driven scalar electroweak multiplets and their physical mass basis.

This compiler does not enumerate a list of allowed weak representations. It
constructs each selected scalar invariant, imposes conjugation/reality, and
diagonalizes the complete quadratic forms separately in each charge sector.
The Higgs alone carries the electroweak VEV.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from itertools import product
import math

import numpy as np
from scipy.linalg import eigh

from component_invariants import component_invariants, _su2_conjugation, Limits
from lagrangian_engine import make_letters

BASIS_VERSION = 'scalar-components-hermitian-rref-v1'
TOL = 1e-10


def supports_fields(fields):
    return bool(fields) and all(f.spin == 'scalar' and f.su3 == (0, 0) for f in fields)


def uses_generic(fields):
    """Keep established coefficient conventions in existing specialized sectors."""
    if not supports_fields(fields):
        return False
    if all(f.kind == 'real' and f.su2 == 1 and f.hypercharge == 0 for f in fields):
        return False
    if len(fields) == 1:
        f = fields[0]
        if ((f.su2 == 1 and f.hypercharge not in (0,1,-1,2,-2)) or (f.su2 == 2 and f.kind == 'complex' and f.hypercharge == Fraction(1,2))
                or (f.su2 == 3 and f.kind == 'real' and f.hypercharge == 0)):
            return False
    return True


def basis_labels(row, fields, limits=None):
    """Physical-component expressions define the numerical coefficient order."""
    kind, _ = classify(row, fields)
    if kind and kind.startswith('generic_scalar_yukawa_'):
        return tuple(f'{i}{j}' for i in range(1,4) for j in range(1,4))
    if kind == 'generic_scalar_lepton_yukawa':
        from generic_scalar_yukawa import basis_labels as lepton_labels
        return lepton_labels(row,fields,limits)
    c = coordinates(tuple(fields))
    return tuple(' + '.join(f'({z.real:g}{z.imag:+g}j)*'+'*'.join(c.labels[i] for i in m)
                            for m,z in sorted(poly.items())) for poly in invariant_basis(row,fields,limits))


def classify(row, fields):
    names = {'H', 'H†'} | {n for f in fields for n in (f.name, f.name + '†')}
    counts = Counter({v['name']: v['count'] for v in row['factors']})
    if not row['derivatives'] and set(counts) <= names:
        return 'generic_scalar_potential', ''
    if row['derivatives'] == 2 and sum(counts.values()) == 2 and all(
            n.startswith('D(') and n[2:-1] in names for n in counts):
        return 'generic_scalar_kinetic', ''
    if not row['derivatives'] and row['multiplicity'] == 9:
        for field in fields:
            if field.su2 != 2 or abs(field.hypercharge) != Fraction(1,2):
                continue
            for doublet,singlet,dagger in (('L','ec',True),('Q','dc',True),('Q','uc',False)):
                if field.hypercharge < 0:dagger=not dagger
                if counts == {doublet:1,singlet:1,field.name+('†' if dagger else ''):1}:
                    return f'generic_scalar_yukawa_{field.name}_{singlet}', ''
    from generic_scalar_yukawa import classify as lepton_classify
    if lepton_classify(row,fields) is not None:
        return 'generic_scalar_lepton_yukawa', ''
    return None, 'compiler gap: non-scalar or non-quadratic-derivative contraction needs Lorentz/fermion lowering'


def add(a, b, scale=1):
    out = dict(a)
    for monomial, value in b.items():
        out[monomial] = out.get(monomial, 0j) + scale * value
    return {m: complex(c) for m, c in out.items() if c != 0}


def mul(a, b):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            key = tuple(sorted(ma + mb))
            out[key] = out.get(key, 0j) + ca * cb
    return {m: complex(c) for m, c in out.items() if c != 0}


def conjugate(poly, bars):
    return {tuple(sorted(bars[i] for i in m)): complex(c).conjugate() for m, c in poly.items()}


def substitute(poly, maps):
    out = {}
    for monomial, coefficient in poly.items():
        term = {(): coefficient}
        for index in monomial:
            term = mul(term, maps[index])
        out = add(out, term)
    return out


@dataclass
class Coordinates:
    fields: tuple
    labels: list
    charges: list
    bars: list
    sources: list
    components: dict


def coordinates(fields):
    labels, charges, bars, sources = ['h'], [Fraction(0)], [0], ['H']
    components = {'H': [{}, {(0,): 1 / math.sqrt(2)}]}

    def real(label, source):
        i = len(labels)
        labels.append(label); charges.append(Fraction(0)); bars.append(i); sources.append(source)
        return {(i,): 1.}

    def charged(label, charge, source):
        i = len(labels)
        labels.extend((label, label + 'bar')); charges.extend((charge, -charge))
        bars.extend((i + 1, i)); sources.extend((source, source))
        return {(i,): 1.}, {(i + 1,): 1.}

    for field in fields:
        n = field.su2
        entries = [None] * n
        c = _su2_conjugation(n - 1).toarray()
        for i in range(n):
            if entries[i] is not None:
                continue
            q = Fraction(n - 1 - 2 * i, 2) + field.hypercharge
            label = f'{field.name.lower()}{i + 1}'
            if field.kind == 'real':
                if q == 0:
                    entries[i] = {m: v * (1 if c[i, i] > 0 else 1j)
                                  for m, v in real(label, field.name).items()}
                else:
                    positive, negative = charged(label, abs(q), field.name)
                    entries[i] = positive if q > 0 else negative
                    entries[n - 1 - i] = {m: v * c[n - 1 - i, i]
                                         for m, v in (negative if q > 0 else positive).items()}
            elif q == 0:
                entries[i] = add(real(label + 'r', field.name), real(label + 'i', field.name), 1j)
                entries[i] = {m: v / math.sqrt(2) for m, v in entries[i].items()}
            else:
                positive, negative = charged(label, abs(q), field.name)
                entries[i] = positive if q > 0 else negative
        components[field.name] = entries
    for name, entries in tuple(components.items()):
        c = _su2_conjugation(len(entries) - 1).toarray()
        components[name + '†'] = [
            {m: coefficient * c[i, len(entries)-1-i]
             for m, coefficient in conjugate(entries[len(entries)-1-i], bars).items()}
            for i in range(len(entries))]
    return Coordinates(tuple(fields), labels, charges, bars, sources, components)


def _canonical_span(polynomials, bars, hermitian, expected):
    """Deterministic real row reduction fixes rotations of degenerate SVD bases."""
    candidates = []
    for p in polynomials:
        if hermitian:
            candidates.extend((add(p, conjugate(p, bars), 1),
                               {m: -1j*c for m, c in add(p, conjugate(p, bars), -1).items()}))
        else:
            candidates.append(p)
    monomials = sorted(set().union(*(p.keys() for p in candidates)))
    complex_rows = np.array([[p.get(m, 0) for m in monomials] for p in candidates], complex)
    if not len(monomials):
        raise ValueError('No physical scalar invariant remains after component expansion')
    # Reality uses a real vector space; non-Hermitian invariants may be complex.
    matrix = np.concatenate((complex_rows.real, complex_rows.imag), axis=1) if hermitian else complex_rows
    matrix = matrix.copy()
    pivot = 0
    for column in range(matrix.shape[1]):
        choices = np.flatnonzero(np.abs(matrix[pivot:, column]) > 1e-9)
        if not len(choices):
            continue
        selected = pivot + int(choices[np.argmax(np.abs(matrix[pivot + choices, column]))])
        matrix[[pivot, selected]] = matrix[[selected, pivot]]
        matrix[pivot] /= matrix[pivot, column]
        for other in range(len(matrix)):
            if other != pivot:
                matrix[other] -= matrix[other, column] * matrix[pivot]
        pivot += 1
        if pivot == len(matrix):
            break
    if pivot != expected:
        raise ValueError(f'Physical scalar basis rank {pivot} differs from catalogue multiplicity {expected}')
    rows = matrix[:pivot]
    if hermitian:
        rows = rows[:, :len(monomials)] + 1j*rows[:, len(monomials):]
    # These are dimensionless numerical basis tensors, not physical inputs.
    # Remove null-space roundoff here, before any user coefficient multiplies
    # the basis; physical polynomial arithmetic below never prunes couplings.
    def clean(z):
        return complex(0. if abs(z.real)<1e-11 else z.real,0. if abs(z.imag)<1e-11 else z.imag)
    return tuple({m: clean(c) for m, c in zip(monomials, row) if abs(c) > 1e-9} for row in rows)


@lru_cache(maxsize=256)
def _basis_cached(fields, factors, multiplicity, hermitian, kinetic, limits):
    coords = coordinates(fields)
    letters = {v.name: v for v in make_letters(fields)}
    descriptors = []
    for name, count in factors:
        original = name[2:-1] if kinetic else name
        letter = letters[original]
        descriptors.append(dict(name=original, count=count, lorentz=(0, 0), weak=letter.weak,
                                colour=letter.colour, nf=1, derivative=False, fermion=False,
                                hypercharge=str(letter.charge)))
    # Rename duplicate descriptors only by aggregating counts; two identical
    # differentiated letters represent the same commuting scalar factor.
    combined = {}
    for d in descriptors:
        if d['name'] in combined:
            combined[d['name']]['count'] += d['count']
        else:
            combined[d['name']] = dict(d)
    result = component_invariants(list(combined.values()), expected_multiplicity=multiplicity, limits=limits)
    polynomials = []
    for invariant in result['basis']:
        polynomial = {}
        for term in invariant['terms']:
            piece = {(): term['coefficient']}
            for factor in term['factors']:
                for component in factor['components']:
                    info = result['components'][factor['letter']][component - 1]
                    weak = info['indices']['weak'] - 1
                    piece = mul(piece, coords.components[factor['letter']][weak])
            polynomial = add(polynomial, piece)
        polynomials.append(polynomial)
    return _canonical_span(polynomials, coords.bars, hermitian, multiplicity)


def invariant_basis(row, fields, limits=None):
    if isinstance(limits, dict):
        from compiler_limits import component_limits
        limits = component_limits(limits)
    if limits is not None and not isinstance(limits, Limits):
        raise ValueError('compiler_limits must be a mapping of component construction budgets')
    kind, reason = classify(row, fields)
    if reason:
        raise ValueError(reason)
    factors = tuple((v['name'], v['count']) for v in row['factors'])
    return _basis_cached(tuple(fields), factors, row['multiplicity'],
                         row['field_content'] == row['conjugate_content'],
                         kind == 'generic_scalar_kinetic', limits)


def _quadratic_matrix(poly, indices, bars, real):
    matrix = np.zeros((len(indices), len(indices)), complex)
    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            monomial = tuple(sorted((i if real else bars[i], j)))
            value = poly.get(monomial, 0)
            matrix[a, b] = value * (2 if real and i == j else 1)
    scale=max(abs(matrix).max(),np.finfo(float).tiny)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10*scale, rtol=1e-8):
        raise ValueError('Quadratic scalar form is not Hermitian')
    if real and np.max(abs(matrix.imag), initial=0) > 1e-8:
        raise ValueError('Neutral real scalar quadratic form has imaginary entries')
    return matrix.real if real else matrix


def _shift(poly, vacuum, size):
    maps = [{(i,): 1} for i in range(size)]
    for i,value in enumerate(vacuum):
        if value:
            maps[i] = {(): value, (i,): 1}
    return substitute(poly, maps)


@dataclass
class GenericScalarCompiler:
    fields: tuple
    operators: tuple
    coordinates: Coordinates
    bases: dict
    transformation: np.ndarray
    eigenmasses: dict
    potential: dict
    kinetic_rows: tuple
    vev: float
    higgs_mass: float
    mu_h_squared: float
    vacuum: tuple
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {self.coordinates.labels[i]: mass for i, mass in self.eigenmasses.items()}

    def write_sources(self, model, source, fr):
        from feynrules_export import _wl, _internal, ModelResolutionError, CompilerResolutionError
        c = self.coordinates
        params = [_internal('pmGSBareH',self.higgs_mass),_internal('pmGSMuH2',self.mu_h_squared)]
        constants = {}
        def scalar(value, order=False, qed=0):
            value=float(value)
            if value < 0:
                return '-('+scalar(-value,order,qed)+')'
            if value == 0:
                return '0'
            if not order and round(value)!=0 and abs(value-round(value)) < 1e-14*abs(value):
                return str(round(value))
            rational=Fraction(value).limit_denominator(1000)
            if not order and rational and abs(float(rational)-value)<1e-14*abs(value):
                return str(rational)
            squared=Fraction(value*value).limit_denominator(1000)
            if not order and squared and abs(math.sqrt(float(squared))-value)<1e-14*abs(value):
                return 'Sqrt['+str(squared)+']'
            # FeynRules' field collector does not accept a machine Complex
            # atom as a tensor coefficient. Real internal constants keep I
            # symbolic and also make conjugation exact during validation.
            key=(format(value,'.15g'),order,qed)
            if key not in constants:
                symbol=f'pmGSN{len(constants)+1}'
                declaration=_internal(symbol,value,order=order)
                if order and qed:
                    declaration=declaration.replace('InteractionOrder -> {NP, 1}',
                                                    f'InteractionOrder -> {{{{NP, 1}}, {{QED, {qed}}}}}')
                constants[key]=symbol; params.append(declaration)
            return constants[key]
        def literal(value, order=False, qed=0):
            value=complex(value)
            return scalar(value.real,order,qed) if value.imag==0 else '('+scalar(value.real,order,qed)+'+I*('+scalar(value.imag,order,qed)+'))'
        physical = [i for i in self.eigenmasses]
        symbols = {i: ('H' if i == 0 else f'pmGS{i}') for i in physical}
        for i in physical:
            symbols[c.bars[i]] = symbols[i] if c.bars[i] == i else symbols[i] + 'bar'
        def expression(index, radial=False):
            terms = [f'({literal(self.transformation[index,j])})*{symbols[j]}'
                     for j in range(len(c.labels)) if self.transformation[index,j] != 0]
            if radial and self.vacuum[index]:
                terms.insert(0, 'vev' if index == 0 else literal(self.vacuum[index]))
            return '(' + (' + '.join(terms) or '0') + ')'
        def polynomial(poly, radial=False, interactions=False):
            return '(' + (' + '.join('('+literal(value,interactions and len(m)>=3,m.count(0) if interactions else 0)+')' + ''.join('*'+expression(i,radial) for i in m)
                                    for m,value in sorted(poly.items())) or '0') + ')'
        sm = (fr/'Models'/'SM'/'SM.fr').read_text()
        marker = 'Phi[2] -> (vev + H + I G0)/Sqrt[2]'
        if marker not in sm:
            raise CompilerResolutionError('Bundled SM Higgs template changed')
        sm = sm.replace(marker, f'Phi[2] -> (vev + {expression(0)} + I G0)/Sqrt[2]')
        sm = sm.replace('Value            -> MH^2/(2*vev^2)', 'Value            -> pmGSBareH^2/(2*vev^2)')
        sm = sm.replace('+ muH^2 Phibar[ii] Phi[ii]', '+ pmGSMuH2 Phibar[ii] Phi[ii]')
        sm = sm.replace('Mass            -> {MH,125}', f'Mass            -> {{MH,{_wl(self.eigenmasses[0])}}}')
        # Numerical generators in the canonical highest-weight basis, for any
        # weak dimension. FR expands them before producing UFO vertices.
        dimensions = sorted({f.su2 for f in self.fields if f.su2 > 1})
        representations = ['{Ta,SU2D}'] + [f'{{pmGST{n},pmGSWeak{n}}}' for n in dimensions]
        sm = sm.replace('Representations   -> {Ta,SU2D}', 'Representations   -> {'+', '.join(representations)+'}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile = source/'SM_generic_scalar.fr'; smfile.write_text(sm)
        classes, particles, definitions = [], [], []
        for n in dimensions:
            definitions.append(f'IndexRange[Index[pmGSWeak{n}]] = Unfold[Range[{n}]];')
            definitions.append(f'pmGST{n}[Index[SU2W,a_Integer],Index[pmGSWeak{n},b_Integer],Index[pmGSWeak{n},d_Integer]] := pmGST{n}[a,b,d];')
            raising = np.diag([math.sqrt(k*(n-k)) for k in range(1,n)],1)
            matrices = ((raising+raising.T)/2, (raising-raising.T)/(2j), np.diag([(n-1)/2-i for i in range(n)]))
            # Definitions are evaluated by ExpandIndices; no numerical group
            # index ever leaks into the final UFO color structures.
            for a,matrix in enumerate(matrices,1):
                for i in range(n):
                    for j in range(n):
                        definitions.append(f'pmGST{n}[{a},{i+1},{j+1}] = {literal(matrix[i,j])};')
        for i in physical:
            if i == 0:
                continue
            real = c.bars[i] == i
            name = c.labels[i]; anti = name if real else name+'bar'
            width = float(model.inputs.get('widths',{}).get(c.sources[i],0.))
            mass = self.eigenmasses[i]; pdg = 9500000+i
            extra = '' if real else f'AntiParticleName -> {_wl(anti)}, QuantumNumbers -> {{Q -> {str(c.charges[i])}}},'
            classes.append(f'S[{500+i}] == {{ClassName -> {symbols[i]}, SelfConjugate -> {str(real)}, {extra}'
                           f' Mass -> {{pmGSM{i},{_wl(mass)}}},'
                           f' Width -> {{pmGSW{i},{_wl(width)}}}, ParticleName -> {_wl(name)}, PDG -> {pdg},'
                           f' PropagatorLabel -> {_wl(name)}, PropagatorType -> D, PropagatorArrow -> {"None" if real else "Forward"}}}')
            particles.append(dict(name=name,antiname=anti,pdg=pdg,source=c.sources[i],spin=1,mass=mass,width=width,
                                  charge=float(c.charges[i]),color=1))
        original_symbols = {}
        for fi,f in enumerate(self.fields,1):
            symbol = f'pmGSU{fi}'; original_symbols[f.name]=symbol
            defs = []
            for k,p in enumerate(c.components[f.name],1):
                defs.append(f'{symbol}[{k}] -> {polynomial(p,True)}')
            # Even a real spherical multiplet uses complex indexed components;
            # its reality relation is enforced by the explicit definitions.
            index = f'pmGSWeak{f.su2}' if f.su2>1 else f'pmGSSinglet{fi}'
            if f.su2 == 1:
                definitions.append(f'IndexRange[Index[{index}]] = Unfold[Range[1]];')
            classes.append(f'S[{600+max(physical)+fi}] == {{ClassName -> {symbol}, SelfConjugate -> False, Unphysical -> True,'
                           f' Indices -> {{Index[{index}]}}, FlavorIndex -> {index}, QuantumNumbers -> {{Y -> {str(f.hypercharge)}}},'
                           ' Definitions -> {'+', '.join(defs)+'}}')
        # Covariant kinetic tensors are reconstructed from their gauge-index
        # invariant basis, rather than differentiating the unitary-gauge Higgs.
        kinetic_terms=[]
        for left,right,value in self.kinetic_rows:
            def cov(name,k,bar):
                s=original_symbols[name] + ('bar' if bar else '')
                return f'DC[{s}[{k+1}],mu]'
            kinetic_terms.append(f'({literal(value)})*'+cov(left[0],left[1],left[2])+'*'+cov(right[0],right[1],right[2]))
        potential_expression=polynomial(self.potential,True,True)
        has_np=any(len(m)>=3 for m in self.potential)
        yukawa_terms=[]
        for op in self.operators:
            if op.kind=='generic_scalar_lepton_yukawa':
                from generic_scalar_yukawa import classify as lepton_classify, bilinear
                row=next(r for r in model.lagrangian.operators if r['id']==op.operator_id)
                fermion,_=lepton_classify(row,self.fields)
                for index,(coefficient,basis) in enumerate(zip(op.coefficients,self.bases[op.operator_id]),1):
                    if coefficient==0:continue
                    parameter=f'pmGSL{op.operator_id[1:]}n{index}'
                    params.append(_internal(parameter,coefficient,order=True));has_np=True
                    pieces=[f'({literal(z)})*('+bilinear(fermion,a,f,b,g)+')*'+expression(k)
                            for (a,f,b,g,k),z in sorted(basis.items())]
                    term=parameter+'*('+' + '.join(pieces)+')'
                    yukawa_terms.append('('+term+'+HC['+term+'])')
                continue
            if not op.kind.startswith('generic_scalar_yukawa_'):
                continue
            name,species=op.kind.removeprefix('generic_scalar_yukawa_').split('_')
            field=next(f for f in self.fields if f.name==name)
            doublet=c.components[name if field.hypercharge>0 else name+'†']
            for generation,(left,right) in enumerate(product(range(1,4),repeat=2)):
                coefficient=op.coefficients[generation]
                if coefficient == 0:
                    continue
                parameter=f'pmGSY{op.operator_id[1:]}n{generation+1}'
                params.append(_internal(parameter,coefficient,order=True));has_np=True
                pieces=[]
                for k in range(2):
                    scalar_component=doublet[k]
                    if species == 'ec':
                        fermion=f'LLbar[sp,{k+1},{left}].lR[sp,{right}]'
                    elif species == 'dc':
                        fermion=f'QLbar[sp,{k+1},{left},cc].dR[sp,{right},cc]'
                    else:
                        fermion=f'QLbar[sp,{k+1},{left},cc].uR[sp,{right},cc]'
                        scalar_component=conjugate(doublet[1-k],c.bars)
                        if k:scalar_component={m:-z for m,z in scalar_component.items()}
                    pieces.append(fermion+'*'+polynomial(scalar_component,True))
                # Match the established catalogue Weyl-to-Dirac convention:
                # the catalogue operator is the HC of these chiral currents.
                term=parameter+'*HC['+' + '.join(pieces)+']'
                yukawa_terms.append('('+term+'+HC['+term+'])')
        weak_indices=['SU2D','SU2W']+[f'pmGSWeak{n}' for n in dimensions]+[
            f'pmGSSinglet{i}' for i,f in enumerate(self.fields,1) if f.su2==1]
        extension=source/'generic_scalar.fr'
        extension.write_text('M$ModelName = '+_wl(model.model_name)+';\n'
                             'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}'+(',{NP,1}' if has_np else '')+'};\n'
                             +'\n'.join(definitions)+'\nM$ClassesDescription = {\n'+',\n'.join(classes)+'\n};\n'
                             +'M$Parameters = {\n'+',\n'.join(params)+'\n};\n'
                             +'(* '+BASIS_VERSION+': complete scalar invariant basis and charge-block mass diagonalization. *)\n'
                             +'LGenericScalar := Block[{mu,sp,cc,s1,s2}, ExpandIndices['+potential_expression
                             +' + '+(' + '.join(kinetic_terms+yukawa_terms) or '0')+', FlavorExpand -> {'+', '.join(weak_indices)+'}]];\n')
        return [smfile,extension], 'LSM + LGenericScalar', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import _fields,_coefficients,_number,PreparedModel,ResolvedOperator,ModelResolutionError,CompilerResolutionError
    fields=tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    if any(inputs.get('vevs',{}).values()):
        raise ModelResolutionError('All new scalar VEVs must be zero in the fixed electroweak vacuum.')
    gf=_number(inputs.get('sm_inputs',{}).get('Gf',1.16637e-5),'Gf',real=True)
    mh=_number(inputs.get('higgs_mass',125),'higgs_mass',real=True)
    if gf<=0 or mh<=0:
        raise ModelResolutionError('Gf and Higgs mass must be positive')
    v=(math.sqrt(2)*gf)**-.5
    c=coordinates(fields); size=len(c.labels)
    vacuum=[0.]*size; vacuum[0]=v
    coefficients=_coefficients(lagrangian,inputs)
    potential={}; kinetic={(0,0):.5}; bases={}; operators=[]; kinetic_rows=[]
    for row in lagrangian.operators:
        kind,reason=classify(row,fields)
        values=coefficients[row['id']]
        if reason:
            if all(value==0 for value in values):
                operators.append(ResolvedOperator(row['id'],'zero_verified',tuple((p['name'],p['count']) for p in row['factors']),values))
                continue
            raise CompilerResolutionError(f"{row['id']}: {reason}")
        if kind.startswith('generic_scalar_yukawa_'):
            operators.append(ResolvedOperator(row['id'],kind,tuple((p['name'],p['count']) for p in row['factors']),values))
            continue
        if kind=='generic_scalar_lepton_yukawa':
            from generic_scalar_yukawa import invariant_basis as lepton_basis
            try:
                bases[row['id']]=lepton_basis(row,fields,inputs.get('compiler_limits'))
            except (ValueError,RuntimeError) as exc:
                if getattr(exc,'category','')=='resource_limit':raise
                raise CompilerResolutionError(f"{row['id']}: lepton invariant construction failed: {exc}") from exc
            operators.append(ResolvedOperator(row['id'],kind,tuple((p['name'],p['count']) for p in row['factors']),values))
            continue
        try:
            basis=invariant_basis(row,fields,inputs.get('compiler_limits'))
        except (ValueError,RuntimeError) as exc:
            if getattr(exc, 'category', '') == 'resource_limit':
                raise
            raise CompilerResolutionError(f"{row['id']}: scalar invariant construction failed: {exc}") from exc
        bases[row['id']]=basis
        polynomial={}
        for p,value in zip(basis,values): polynomial=add(polynomial,p,value)
        if row['field_content']!=row['conjugate_content']:
            polynomial=add(polynomial,conjugate(polynomial,c.bars))
        operators.append(ResolvedOperator(row['id'],kind,tuple((p['name'],p['count']) for p in row['factors']),values))
        if kind=='generic_scalar_potential':
            potential=add(potential,polynomial)
        else:
            factors=[p['name'][2:-1] for p in row['factors'] for _ in range(p['count'])]
            if any(n in ('H','H†') for n in factors):
                if any(value!=0 for value in values):
                    raise CompilerResolutionError(f"{row['id']}: compiler gap: nonzero Higgs-multiplet kinetic mixing requires gauge/Goldstone redefinitions")
                continue
            kinetic=add(kinetic,polynomial)
            # Any quadratic scalar invariant is a bilinear. Reconstruct its
            # coefficients in the original complex spherical components by a
            # checked linear solve, including real-field identifications.
            left,right=[n.removesuffix('†') for n in factors]
            leftbar,rightbar=[n.endswith('†') for n in factors]
            alternatives=[]; keys=[]
            for i,j in product(range(len(c.components[left])),range(len(c.components[right]))):
                a=c.components[left][i]; b=c.components[right][j]
                if leftbar:a=conjugate(a,c.bars)
                if rightbar:b=conjugate(b,c.bars)
                candidate=mul(a,b)
                alternatives.append(candidate); keys.append(((left,i,leftbar),(right,j,rightbar)))
                if row['field_content']!=row['conjugate_content']:
                    alternatives.append(conjugate(candidate,c.bars))
                    keys.append(((left,i,not leftbar),(right,j,not rightbar)))
            monomials=sorted(set().union(*(p.keys() for p in (*basis,*alternatives))))
            matrix=np.array([[p.get(m,0) for p in alternatives] for m in monomials],complex)
            for unit,value in zip(basis,values):
                target=np.array([unit.get(m,0) for m in monomials],complex)
                solution=np.linalg.lstsq(matrix,target,rcond=1e-11)[0]
                if not np.allclose(matrix@solution,target,atol=1e-10*max(abs(target),default=0.),rtol=1e-9):
                    raise CompilerResolutionError('Covariant kinetic reconstruction failed')
                # Only dimensionless basis-projection roundoff is removed.
                # Apply user coefficients afterwards, retaining tiny complex
                # mixing phases as well as uniformly small kinetic forms.
                solution_scale=max(abs(solution),default=0.)
                unit_rows=[(a,b,complex(z.real if abs(z.real)>1e-12*solution_scale else 0.,
                                      z.imag if abs(z.imag)>1e-12*solution_scale else 0.))
                           for (a,b),z in zip(keys,solution)]
                kinetic_rows.extend((a,b,z*value) for a,b,z in unit_rows if z!=0 and value!=0)
                if row['field_content']!=row['conjugate_content']:
                    kinetic_rows.extend(((a[0],a[1],not a[2]),(b[0],b[1],not b[2]),(z*value).conjugate())
                                        for a,b,z in unit_rows if z!=0 and value!=0)
    shifted=_shift(potential,vacuum,size)
    tadpole_scales={}
    for monomial,z in potential.items():
        for index in set(monomial)-{0}:
            remaining=list(monomial);remaining.remove(index)
            term=abs(z)*monomial.count(index)*math.prod(abs(vacuum[i]) for i in remaining)
            tadpole_scales[index]=tadpole_scales.get(index,0.)+term
    tadpoles={c.labels[m[0]]:z for m,z in shifted.items() if len(m)==1 and m[0]!=0
              and abs(z)>1e-10*tadpole_scales.get(m[0],0.)}
    if tadpoles:
        raise ModelResolutionError(f'The supplied multiplet vacuum is not stationary: {tadpoles}')
    # Fix the Higgs quadratic input by its tadpole, as the existing singlet
    # compiler does; no BSM tadpole counterterm is introduced.
    mu=mh*mh/2-shifted.get((0,),0).real/v
    sm={(0,0):mu/2,(0,0,0,0):-mh*mh/(8*v*v)}
    full=_shift(add(potential,sm),vacuum,size)
    transformation=np.zeros((size,size),complex); masses={}
    groups={}
    for i,q in enumerate(c.charges):
        if q>=0:groups.setdefault(q,[]).append(i)
    for charge,indices in sorted(groups.items()):
        real=charge==0
        try:
            k=_quadratic_matrix(kinetic,indices,c.bars,real)
            m=-_quadratic_matrix(full,indices,c.bars,real)
            if np.diag(k).real.min()<=0:raise ValueError('kinetic matrix is not positive definite')
            # Remove arbitrary input field normalizations before testing
            # conditioning; a uniformly tiny positive kinetic form is valid.
            scale=1/np.sqrt(np.diag(k).real)
            normalized_k=scale[:,None]*k*scale[None,:]
            normalized_m=scale[:,None]*m*scale[None,:]
            kinetic_eigenvalues=np.linalg.eigvalsh(normalized_k)
            if kinetic_eigenvalues.min()<=0:
                raise ValueError('kinetic matrix is not positive definite')
            if kinetic_eigenvalues.min()<=1e-12*kinetic_eigenvalues.max():
                raise CompilerResolutionError('Scalar kinetic matrix is too ill-conditioned for reliable floating-point diagonalization')
            eigen,rotation=eigh(normalized_m,normalized_k)
            rotation=scale[:,None]*rotation
        except CompilerResolutionError:
            raise
        except (ValueError,np.linalg.LinAlgError) as exc:
            raise ModelResolutionError(f'Charge {charge}: invalid scalar quadratic form: {exc}') from exc
        uncertainty=np.array([1e-10*np.sum(abs(rotation[:,j,None]*m*rotation[None,:,j]))
                              for j in range(len(indices))])
        if np.any(eigen < -uncertainty):
            raise ModelResolutionError(f'Charge {charge}: tachyonic scalar spectrum')
        # Deterministic overlap assignment keeps the SM-like neutral state H.
        from scipy.optimize import linear_sum_assignment
        _,assignment=linear_sum_assignment(-abs(rotation))
        rotation=rotation[:,assignment]; eigen=eigen[assignment]
        for j in range(len(indices)):
            pivot=int(np.argmax(abs(rotation[:,j])))
            phase=rotation[pivot,j]/abs(rotation[pivot,j])
            rotation[:,j]/=phase
        if not np.allclose(rotation.conj().T@k@rotation,np.eye(len(indices)),atol=1e-8):
            raise CompilerResolutionError('Scalar kinetic normalization residual failed')
        if not np.allclose(rotation.conj().T@m@rotation,np.diag(eigen),atol=1e-6,rtol=1e-8):
            raise CompilerResolutionError('Scalar mass diagonalization residual failed')
        for a,i in enumerate(indices):
            masses[i]=math.sqrt(max(0.,float(eigen[a])))
            for b,j in enumerate(indices):
                transformation[i,j]=rotation[a,b]
                if not real:transformation[c.bars[i],c.bars[j]]=rotation[a,b].conjugate()
    compiler=GenericScalarCompiler(fields,tuple(operators),c,bases,transformation,masses,potential,tuple(kinetic_rows),v,mh,float(mu),tuple(vacuum))
    return PreparedModel(lagrangian,inputs,tuple(operators),fields,{'masses':[masses[0]]},runtime,compiler=compiler)
