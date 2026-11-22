"""Automatic massive neutral-vector effective models in a fixed invariant basis.

A single real singlet Proca field is supported, including all SM flavour
currents, Higgs current, hypercharge kinetic mixing and generalized-Proca cubic
self-interaction. This is a massive-vector effective theory, not a declaration
of a new anomaly-free gauge symmetry or a UV completion.

Nine real current coefficients use [11,22,33,Re12,Im12,Re13,Im13,Re23,Im23],
where Reij = J_ij+J_ji and Imij = i(J_ij-J_ji). Right-handed conjugate Weyl
currents include the charge-conjugation minus sign and transpose.
"""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


def supports_fields(fields):
    return (len(fields) == 1 and fields[0].spin == 'vector' and fields[0].kind == 'real'
            and fields[0].su2 == 1 and fields[0].su3 == (0, 0) and fields[0].hypercharge == 0)


def classify(row, fields):
    if not supports_fields(fields):
        return None, 'neutral-vector compiler requires one real gauge-singlet vector'
    x = fields[0].name
    c = Counter({p['name']: p['count'] for p in row['factors']})
    families = [(Counter({x: 2}), 'mass'),
                (Counter({x: 2, 'H': 1, 'H†': 1}), 'portal'),
                (Counter({x: 4}), 'quartic'),
                (Counter({f'D({x})': 2}), 'kinetic'),
                (Counter({f'D({x})': 1, x: 2}), 'cubic_derivative'),
                (Counter({f'D({x})': 1, 'H': 1, 'H†': 1}), 'higgs_divergence'),
                (Counter({'D(H)': 1, 'H†': 1, x: 1}), 'higgs_current'),
                (Counter({'B+': 1, f'D({x})': 1}), 'hypercharge_mixing')]
    for counts, kind in families:
        if c == counts:
            return kind, ''
    for fermion in ('L', 'Q', 'uc', 'dc', 'ec'):
        if c == Counter({fermion: 1, fermion + '†': 1, x: 1}):
            return 'current_' + fermion, ''
    return None, 'this neutral-vector contraction is not in the documented physical invariant basis'


def hermitian_current(values):
    """Return the matrix represented by the documented nine Hermitian invariants."""
    matrix = np.diag(np.asarray(values[:3], dtype=complex))
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        matrix[i, j] = values[3 + 2*k] + 1j * values[4 + 2*k]
        matrix[j, i] = matrix[i, j].conjugate()
    return matrix


@dataclass
class VectorCurrentCompiler:
    field: object
    operators: tuple
    vector_basis: dict
    higgs_mass: float
    vev: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {'H': self.higgs_mass, 'Z': self.vector_basis['masses'][1],
                self.field.name: self.vector_basis['masses'][2]}

    def write_sources(self, model, source, fr):
        from feynrules_export import _wl, _internal, ModelResolutionError
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        sm = sm.replace('Mass            -> {MH,125}', f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        sm = sm.replace('Mass            -> {MZ, 91.1876}', f'Mass            -> {{pmPhysicalZ,{_wl(self.masses["Z"])}}}')
        # MZ would otherwise survive only in ghost declarations and is pruned
        # by RemoveGhosts, leaving dangling MW/sw2 parameter expressions.
        sm = sm.replace('MZ^2', 'pmBareZ^2').replace('MZ^4', 'pmBareZ^4').replace('(MW/MZ)^2', '(MW/pmBareZ)^2')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs, skip={'MZ'})
        smfile = source / 'SM_vector.fr'
        smfile.write_text(sm)
        parameters, terms, comments = [], [], []
        lookup = {}
        for op in self.operators:
            higgs = sum(count for name, count in op.factors
                        if (name[2:-1] if name.startswith('D(') else name) in ('H', 'H†'))
            for k, value in enumerate(op.coefficients, 1):
                name = f'pmVC{op.operator_id[1:]}n{k}'
                lookup[(op.kind, k)] = name
                parameters.append(_internal(name, value,
                    order=op.kind not in ('mass', 'kinetic', 'hypercharge_mixing'), qed_order=higgs))
            comments.append(f'(* {op.operator_id}: {op.kind}; every supplied coefficient retained. *)')
        def c(kind, k=1):
            return lookup.get((kind, k), '0')
        # Quadratic pieces are replaced only after the numerical congruences
        # T^T K T=I and T^T M2 T=diag(m2) have been verified in Python.
        terms += [f'{c("mass")}*pmV[mu]*pmV[mu]',
                  f'{c("portal")}*(vev+H)^2/2*pmV[mu]*pmV[mu]',
                  f'{c("quartic")}*pmV[mu]*pmV[mu]*pmV[nu]*pmV[nu]',
                  f'({c("cubic_derivative",1)}-{c("cubic_derivative",2)}/2)*pmV[mu]*pmV[mu]*del[pmV[nu],nu]']
        # i V.Hdagger.DH + h.c. = -Re(c) gZ(v+h)^2 Z.V/2
        # + Im(c) div(V) Hdagger.H after integration by parts.
        hc = next((o.coefficients[0] for o in self.operators if o.kind == 'higgs_current'), 0)
        parameters.append(_internal('pmHCurrent', complex(hc).real, order=True, qed_order=2))
        terms.append('-pmHCurrent*ee/(2*sw*cw)*(vev+H)^2*Z[mu]*pmV[mu]')
        comments += ['(* Higgs derivative basis: i V_mu Hdagger D_mu H + h.c.; div(V) Hdagger H. *)',
                     '(* Its longitudinal combination is required to cancel before export. *)',
                     '(* Cubic basis: V^2 div(V), V_mu V_nu d_mu V_nu; second = -first/2 modulo a boundary term. *)',
                     '(* B+ D(V) basis: (Bmunu Vmunu + i dual(B)munu Vmunu)/2 plus h.c.; dual piece is a boundary term. *)']
        for op in self.operators:
            if not op.kind.startswith('current_'):
                continue
            kind = op.kind[8:]
            field = {'L': 'LL', 'Q': 'QL', 'uc': 'uR', 'dc': 'dR', 'ec': 'lR'}[kind]
            doublet = kind in ('L', 'Q')
            colour = kind in ('Q', 'uc', 'dc')
            def current(i, j):
                if kind in ('uc', 'dc', 'ec'):
                    i, j = j, i
                    sign = '-'
                else:
                    sign = ''
                a = f'sp1,' + ('ii,' if doublet else '') + str(i) + (',cc' if colour else '')
                b = f'sp2,' + ('ii,' if doublet else '') + str(j) + (',cc' if colour else '')
                return f'({sign}{field}bar[{a}].{field}[{b}]*Ga[mu,sp1,sp2]*pmV[mu])'
            for k in range(3):
                if op.coefficients[k] != 0:
                    terms.append(c(op.kind, k+1) + '*' + current(k+1, k+1))
            for k, (i, j) in enumerate(((1, 2), (1, 3), (2, 3))):
                base = current(i, j)
                if op.coefficients[3+2*k] != 0:
                    terms.append(f'{c(op.kind,4+2*k)}*({base}+HC[{base}])')
                if op.coefficients[4+2*k] != 0:
                    terms.append(f'I*{c(op.kind,5+2*k)}*({base}-HC[{base}])')
        transform = self.vector_basis['transformation']
        symbols = ['A', 'Z', 'pmV']
        rules = []
        for i, symbol in enumerate(symbols):
            pieces = []
            for j, target in enumerate(symbols):
                if abs(transform[i, j]) > 1e-15:
                    pname = f'pmVT{i}{j}'
                    parameters.append(_internal(pname, transform[i, j]))
                    pieces.append(f'{pname}*{target}[a]')
            rules.append(f'{symbol}[a_] :> (' + '+'.join(pieces) + ')')
        parameters.append(_internal('pmBareZ', self.vector_basis['bare_z']))
        pdg, name = 9300001, self.field.name.lower()
        width = model.inputs.get('widths', {}).get(self.field.name, 0.)
        classes = f'''V[301] == {{ClassName -> pmV, SelfConjugate -> True,
Mass -> {{pmVMass,{_wl(self.masses[self.field.name])}}}, Width -> {{pmVWidth,{_wl(width)}}},
ParticleName -> {_wl(name)}, PDG -> {pdg}, PropagatorLabel -> {_wl(self.field.name)},
PropagatorType -> Sine, PropagatorArrow -> None, FullName -> "Massive singlet vector"}}'''
        extension = source / 'vector_current.fr'
        has_np = any(any(value != 0 for value in op.coefficients)
                     for op in self.operators
                     if op.kind.startswith('current_') or op.kind in ('portal', 'quartic', 'cubic_derivative', 'higgs_current'))
        extension.write_text('M$ModelName = '+_wl(model.model_name)+';\n'
          'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}' + (',{NP,1}' if has_np else '') + '};\n'
          'M$ClassesDescription = {'+classes+'};\nM$Parameters = {\n'+',\n'.join(parameters)+'};\n'
          +'\n'.join(comments)+'\n'
          'LVectorExtra := Block[{mu,nu,sp1,sp2,ii,cc}, ExpandIndices['+' + '.join(terms)+', FlavorExpand->{SU2D,SU2W}]];\n'
          'LVectorRotated := Expand[ExpandIndices[LSM+LVectorExtra, FlavorExpand->{SU2D,SU2W}] /. {'+', '.join(rules)+'}];\n'
          'LVectorQuadratic := Select[Expand[GetQuadraticTerms[LVectorRotated]], !FreeQ[#, A|Z|pmV]&];\n'
          'LVectorCanonical := Block[{mu,nu}, -1/4*(del[A[nu],mu]-del[A[mu],nu])^2'
          ' -1/4*(del[Z[nu],mu]-del[Z[mu],nu])^2 + pmPhysicalZ^2/2*Z[mu]*Z[mu]'
          ' -1/4*(del[pmV[nu],mu]-del[pmV[mu],nu])^2 + pmVMass^2/2*pmV[mu]*pmV[mu]];\n')
        particles = [dict(name=name, antiname=name, pdg=pdg, source=self.field.name, spin=3,
                          mass=self.masses[self.field.name], width=width)]
        return [smfile, extension], 'LVectorRotated - LVectorQuadratic + LVectorCanonical', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import (_fields, _coefficients, _number, PreparedModel,
                                 ResolvedOperator, ModelResolutionError, CompilerResolutionError)
    fields = tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    f = fields[0]
    specification = inputs.get('vector_specifications', {}).get(f.name)
    if not isinstance(specification, dict) or specification.get('kind') != 'proca':
        raise ModelResolutionError(f'{f.name}: set vector_specifications[{f.name!r}]={{"kind":"proca"}} for a massive-vector effective theory')
    if lagrangian._source.sm_generations != 3:
        raise CompilerResolutionError('Vector flavour currents require all three SM generations')
    coefs = _coefficients(lagrangian, inputs)
    operators = []
    expected = {'kinetic': 4, 'cubic_derivative': 2}
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        if reason:
            raise CompilerResolutionError(f'{row["id"]}: {reason}')
        multiplicity = 9 if kind.startswith('current_') else expected.get(kind, 1)
        if row['multiplicity'] != multiplicity:
            raise CompilerResolutionError(f'{row["id"]}: expected {multiplicity} invariants in the documented vector basis')
        operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                                          coefs[row['id']], add_hc=row['field_content'] != row['conjugate_content']))
    bykind = {o.kind: o.coefficients for o in operators}
    def coefficient(kind, default=0):
        return bykind.get(kind, (default,))[0]
    if 'kinetic' not in bykind:
        raise ModelResolutionError(f'{f.name}: select the four-invariant vector kinetic operator')
    a, b, c, d = bykind['kinetic']
    if not math.isclose(b+c, -a, rel_tol=1e-10, abs_tol=1e-12):
        raise ModelResolutionError(f'{f.name}: Proca constraint c2+c3=-c1 is required; otherwise the longitudinal mode has an additional kinetic pole')
    z = -2*a
    if z <= 0:
        raise ModelResolutionError(f'{f.name}: negative vector kinetic residue; require -2*c1 > 0')
    hc = complex(coefficient('higgs_current'))
    longitudinal = coefficient('higgs_divergence') + hc.imag
    if abs(longitudinal) > 1e-12:
        raise CompilerResolutionError(f'{f.name}: div(V) Hdagger H + Im(c_Hcurrent) must cancel; nonzero longitudinal scalar-vector mixing requires derivative field redefinitions and induced higher-dimension terms not supported by this compiler')
    gf = _number(inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5), 'Gf', real=True)
    mz = _number(inputs.get('sm_inputs', {}).get('MZ', 91.1876), 'MZ', real=True)
    inv_alpha = _number(inputs.get('sm_inputs', {}).get('aEWM1', 127.9), 'aEWM1', real=True)
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    if min(gf, mz, inv_alpha, mh) <= 0:
        raise ModelResolutionError('Gf, MZ, aEWM1 and Higgs mass must be positive')
    discriminant = mz**4/4 - math.pi/(math.sqrt(2)*inv_alpha*gf)*mz*mz
    if discriminant < 0:
        raise ModelResolutionError('SM inputs do not admit a real weak mixing angle')
    mw2 = mz*mz/2 + math.sqrt(discriminant)
    cw = math.sqrt(mw2)/mz
    sw = math.sqrt(1-cw*cw)
    ee = math.sqrt(4*math.pi/inv_alpha)
    vev = (math.sqrt(2)*gf)**-.5
    k = np.diag([1., 1., z])
    mixing = -2*complex(coefficient('hypercharge_mixing')).real
    k[0, 2] = k[2, 0] = mixing*cw
    k[1, 2] = k[2, 1] = -mixing*sw
    ke, ku = np.linalg.eigh(k)
    if min(ke) <= 0:
        raise ModelResolutionError('Hypercharge/vector mixing produces a nonpositive kinetic matrix')
    inverse = (ku * ke**-.5) @ ku.T
    m = np.diag([0., mz*mz, 2*coefficient('mass')+coefficient('portal')*vev*vev])
    m[1, 2] = m[2, 1] = -hc.real*ee/(2*sw*cw)*vev*vev
    eigen, rotation = np.linalg.eigh(inverse @ m @ inverse)
    if min(eigen) < -1e-8 or np.count_nonzero(eigen > 1e-8) != 2:
        raise ModelResolutionError('Vector mass matrix must have exactly one photon zero mode and two positive massive modes')
    # Photon first; name the massive state of greatest old-Z overlap Z.
    t = inverse @ rotation
    zero = int(np.argmin(abs(eigen)))
    others = [i for i in range(3) if i != zero]
    zi = max(others, key=lambda i: abs(t[1, i]))
    xi = next(i for i in others if i != zi)
    order = [zero, zi, xi]
    t, eigen = t[:, order], np.maximum(eigen[order], 0.)
    eigen[0] = 0.  # Preserve the exact unbroken electromagnetic zero mode.
    for i in range(3):
        if t[i, i] < 0:
            t[:, i] *= -1
    if not np.allclose(t.T@k@t, np.eye(3), atol=1e-10) or not np.allclose(t.T@m@t, np.diag(eigen), atol=1e-7):
        raise CompilerResolutionError('Neutral-vector normalization/mass diagonalization failed numerical verification')
    basis = dict(transformation=t, kinetic=k, mass_matrix=m, masses=np.sqrt(eigen), bare_z=mz)
    compiler = VectorCurrentCompiler(f, tuple(operators), basis, mh, vev)
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, {'masses':[mh]}, runtime, compiler=compiler)
