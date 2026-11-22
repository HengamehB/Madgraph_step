"""Vectorlike Dirac multiplets in arbitrary weak representations.

The coefficient basis uses conventional invariant chiral bilinears. Equal gauge
representations permit full Hermitian kinetic mixing and complex Dirac mass
mixing. Canonical chiral rotations commute with every gauge generator, so the
complete covariant currents in the mass basis are exactly canonical. Nonzero
Majorana, SM mixing and Yukawa operators require a further fermion-invariant
lowerer; they are diagnosed, never silently discarded or called backend limits.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import math

import numpy as np

BASIS_VERSION = 'dirac-covariant-bilinears-v1'
_COLOURS = {(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)}


def supports_fields(fields):
    return bool(fields) and all(f.spin == 'fermion' and f.kind == 'dirac' and f.su3 in _COLOURS for f in fields)


def uses_generic(fields):
    if not supports_fields(fields):
        return False
    # Preserve the established Majorana/SM-neutrino convention for singlets.
    if all(f.su2 == 1 and f.su3 == (0, 0) and f.hypercharge == 0 for f in fields):
        return False
    return len(fields) > 1 or any(f.su2 > 1 or f.su3 in {(2, 0), (0, 2), (1, 1)} for f in fields)


def _representation(field):
    return field.su2, field.su3, field.hypercharge


def _tokens(fields):
    return {f.name + suffix: (f, suffix) for f in fields for suffix in ('_L', '_Rc')}


def classify(row, fields):
    tokens = _tokens(fields)
    factors = list(Counter({p['name']: p['count'] for p in row['factors']}).elements())
    if row['multiplicity'] != 1:
        return None, 'compiler gap: fermion flavour/Yukawa invariant lowering is not implemented for this operator'
    if row['derivatives'] == 1 and len(factors) == 2:
        differentiated = [name[2:-1] for name in factors if name.startswith('D(')]
        daggered = [name[:-1] for name in factors if name.endswith('†')]
        if len(differentiated) == len(daggered) == 1 and differentiated[0] in tokens and daggered[0] in tokens:
            a, chirality = tokens[differentiated[0]]
            b, other = tokens[daggered[0]]
            if chirality == other and _representation(a) == _representation(b):
                return 'generic_dirac_kinetic', ''
    if not row['derivatives'] and len(factors) == 2 and all(name in tokens for name in factors):
        (a, left), (b, right) = [tokens[name] for name in factors]
        if {left, right} == {'_L', '_Rc'} and _representation(a) == _representation(b):
            return 'generic_dirac_mass', ''
    return None, 'compiler gap: selected Majorana, SM-mixing or Yukawa contraction needs general fermion/Lorentz lowering'


def basis_labels(row, fields):
    kind, reason = classify(row, fields)
    if kind == 'generic_dirac_kinetic':
        factors = list(Counter({p['name']: p['count'] for p in row['factors']}).elements())
        bare = next(name[2:-1] for name in factors if name.startswith('D('))
        dagger = next(name[:-1] for name in factors if name.endswith('†'))
        return (f'i {dagger}† barsigma(mu) D_mu {bare} + h.c.; gauge identity contraction; diagonal canonical coefficient 1/2',)
    if kind == 'generic_dirac_mass':
        token = next(p['name'] for p in row['factors'] if p['name'].endswith('_L'))
        field = _tokens(fields)[token][0]
        return (f'sum(i=1..{field.su2}) (-1)^(i-1) L_i Rc_({field.su2}+1-i) + h.c.; native colour identity; Rc=C.T times charge-conjugate right field; coefficient -M before normalization',)
    return tuple(f'unlowered invariant {i+1}: {reason}' for i in range(row['multiplicity']))


def weak_generators(dimension):
    """Hermitian generators in the highest-weight basis (Q=T3+Y)."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError('Weak representation dimension must be a positive integer')
    raising = np.diag([math.sqrt(k * (dimension-k)) for k in range(1, dimension)], 1)
    return ((raising + raising.T) / 2, (raising - raising.T) / (2j),
            np.diag([(dimension-1)/2-i for i in range(dimension)]))


def _inverse_kinetic(matrix, label):
    from feynrules_export import ModelResolutionError
    # Each field normalization is arbitrary, independently for both chiralities.
    # Diagonal preconditioning avoids losing a small species beside a large one.
    diagonal = matrix.diagonal().real
    if np.any(diagonal <= 0):
        raise ModelResolutionError(f'{label}: fermion kinetic matrix is not positive definite; canonical diagonal coefficient is 1/2 plus h.c.')
    scale = np.sqrt(diagonal)
    normalized = (matrix / scale[:, None]) / scale[None, :]
    if not np.allclose(normalized, normalized.conj().T, atol=1e-12, rtol=1e-12):
        raise ModelResolutionError('Dirac kinetic matrices must be Hermitian')
    eigen, vectors = np.linalg.eigh(normalized)
    if eigen.min() <= 0:
        raise ModelResolutionError(f'{label}: fermion kinetic matrix is not positive definite; canonical diagonal coefficient is 1/2 plus h.c.')
    return ((vectors / np.sqrt(eigen)) @ vectors.conj().T) / scale[:, None]


def resolve_block(names, left, right, mass):
    """Resolve Weyl L.T M Rc with K_L,K_Rc Hermitian kinetic forms."""
    from feynrules_export import ModelResolutionError, CompilerResolutionError
    left, right, mass = map(lambda a: np.asarray(a, complex), (left, right, mass))
    size = len(names)
    if any(a.shape != (size, size) for a in (left, right, mass)):
        raise ModelResolutionError('Dirac kinetic/mass matrices have inconsistent dimensions')
    if any(not np.isfinite(a).all() for a in (left, right, mass)):
        raise ModelResolutionError('Dirac kinetic/mass matrices must be finite')
    for kinetic in (left, right):
        scale = float(np.max(abs(kinetic)))
        if scale and not np.allclose(kinetic / scale, kinetic.conj().T / scale, atol=1e-12, rtol=1e-12):
            raise ModelResolutionError('Dirac kinetic matrices must be Hermitian')
    zl, zr = _inverse_kinetic(left, 'Left chirality'), _inverse_kinetic(right, 'Charge-conjugate right chirality')
    u, masses, vh = np.linalg.svd(zl.T @ mass @ zr)
    tl, tr = zl @ u.conj(), zr @ vh.conj().T
    # Label each eigenstate by maximum total overlap with the input species.
    def unit_columns(matrix):
        scaled = matrix / np.max(abs(matrix), axis=0)
        return scaled / np.linalg.norm(scaled, axis=0)
    weights = abs(unit_columns(np.sqrt(left.diagonal().real)[:, None] * tl)) + abs(unit_columns(np.sqrt(right.diagonal().real)[:, None] * tr))
    order = max(permutations(range(size)), key=lambda p: sum(weights[i, p[i]] for i in range(size)))
    tl, tr, masses = tl[:, order], tr[:, order], masses[list(order)]
    for i in range(size):
        pivot = tl[i, i] if tl[i, i] != 0 else tl[np.argmax(abs(tl[:, i])), i]
        phase = np.exp(-1j * np.angle(pivot))
        tl[:, i] *= phase
        tr[:, i] *= phase.conjugate()
    residuals = dict(left_kinetic=float(np.linalg.norm(tl.conj().T @ left @ tl - np.eye(size))),
                     right_kinetic=float(np.linalg.norm(tr.conj().T @ right @ tr - np.eye(size))),
                     mass=float(np.linalg.norm(tl.T @ mass @ tr - np.diag(masses))))
    mass_scale = float(np.max(masses))
    mass_error = tl.T @ mass @ tr - np.diag(masses)
    relative_mass_error = float(np.max(abs(mass_error / mass_scale))) if mass_scale else float(np.max(abs(mass_error)))
    if max(residuals['left_kinetic'], residuals['right_kinetic'], relative_mass_error) > 1e-8:
        raise CompilerResolutionError('Dirac normalization/mass diagonalization numerical residual exceeds tolerance')
    return dict(names=tuple(names), left_kinetic=left, right_kinetic=right, mass_matrix=mass,
                left_transformation=tl, right_conjugate_transformation=tr, masses=masses, residuals=residuals)


@dataclass
class GenericFermionCompiler:
    fields: tuple
    operators: tuple
    fermion_basis: dict
    physical_masses: dict
    higgs_mass: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {'H': self.higgs_mass, **{f.name.lower() + str(i+1): self.physical_masses[f.name]
                                      for f in self.fields for i in range(f.su2)}}

    def write_sources(self, model, source, fr):
        from feynrules_export import _wl, _internal, ModelResolutionError
        source, fr = Path(source), Path(fr)
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        marker = 'Mass            -> {MH,125}'
        if marker not in sm:
            raise ModelResolutionError('Bundled SM Higgs template changed')
        sm = sm.replace(marker, f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        dimensions = sorted({f.su2 for f in self.fields})
        weak_reps = ['{Ta,SU2D}'] + [f'{{pmGFT{n},pmGFWeak{n}}}' for n in dimensions if n > 1]
        sm = sm.replace('Representations   -> {Ta,SU2D}', 'Representations   -> {' + ','.join(weak_reps) + '}')
        params, constants = [], {}
        # This literal helper is used only for dimensionless group generators.
        # Physical chiral rotations commute with these equal-representation
        # gauge couplings; canonical fields are declared directly below.
        def scalar(value):
            value = float(value)
            if value < 0:
                return '-(' + scalar(-value) + ')'
            if abs(value-round(value)) < 1e-14 * max(1, abs(value)):
                return str(round(value))
            ratio = Fraction(value).limit_denominator(1000)
            if abs(float(ratio)-value) < 1e-14:
                return str(ratio)
            square = Fraction(value*value).limit_denominator(1000)
            if abs(math.sqrt(float(square))-value) < 1e-14:
                return 'Sqrt[' + str(square) + ']'
            key = format(value, '.15g')
            if key not in constants:
                name = f'pmGFN{len(constants)+1}'
                constants[key] = name
                params.append(_internal(name, value))
            return constants[key]
        def literal(value):
            value = complex(value)
            return scalar(value.real) if abs(value.imag) < 1e-14 else '(' + scalar(value.real) + '+I*(' + scalar(value.imag) + '))'
        definitions, classes, particles, terms = [], [], [], []
        for n in dimensions:
            definitions.append(f'IndexRange[Index[pmGFWeak{n}]] = Unfold[Range[{n}]];')
            if n == 1:
                continue
            definitions.append(f'pmGFT{n}[Index[SU2W,a_Integer],Index[pmGFWeak{n},b_Integer],Index[pmGFWeak{n},c_Integer]] := pmGFT{n}[a,b,c];')
            for a, generator in enumerate(weak_generators(n), 1):
                for i in range(n):
                    for j in range(n):
                        definitions.append(f'pmGFT{n}[{a},{i+1},{j+1}] = {literal(generator[i,j])};')
        if any(f.su3 in {(2, 0), (0, 2)} for f in self.fields):
            definitions.extend(('IndexRange[Index[Sextet]] = NoUnfold[Range[6]];',
                                'IndexStyle[Sextet,sx];', 'AddGaugeRepresentation[{SU3C -> {T6,Sextet}}];'))
        pdg_offset, class_offset = 100, 1000
        for fi, field in enumerate(self.fields, 1):
            n = field.su2
            conjugated = field.su3 in {(0, 1), (0, 2)}
            hypercharge = -field.hypercharge if conjugated else field.hypercharge
            colour = ('Colour' if field.su3 in {(1, 0), (0, 1)} else
                      'Sextet' if field.su3 in {(2, 0), (0, 2)} else
                      'Gluon' if field.su3 == (1, 1) else None)
            physical_symbols = []
            for i in range(n):
                symbol = f'pmGFF{fi}n{i+1}'
                physical_symbols.append(symbol)
                label = field.name.lower() + str(n-i if conjugated else i+1)
                name, anti = (label + 'bar', label) if conjugated else (label, label + 'bar')
                q = Fraction(n-1-2*i, 2) + hypercharge
                mass = self.physical_masses[field.name]
                width = float(model.inputs.get('widths', {}).get(field.name, 0.))
                pdg = 9600000 + pdg_offset + i+1
                color_part = f'Indices -> {{Index[{colour}]}},' if colour else ''
                classes.append(f'F[{10000+class_offset+i}] == {{ClassName -> {symbol}, SelfConjugate -> False, {color_part}'
                               f'QuantumNumbers -> {{Q -> {str(q)}}}, Mass -> {{pmGFM{fi}n{i+1},{_wl(mass)}}},'
                               f'Width -> {{pmGFW{fi}n{i+1},{_wl(width)}}}, ParticleName -> {_wl(name)},'
                               f'AntiParticleName -> {_wl(anti)}, PDG -> {pdg}, PropagatorLabel -> {_wl(name)},'
                               'PropagatorType -> Straight, PropagatorArrow -> Forward}')
                particles.append(dict(name=name, antiname=anti, pdg=pdg, source=field.name, spin=2,
                                      mass=mass, width=width, charge=float(q),
                                      color={None: 1, 'Colour': 3, 'Sextet': 6, 'Gluon': 8}[colour]))
                terms.append(f'-pmGFM{fi}n{i+1}*{symbol}bar.{symbol}')
            symbol = f'pmGFU{fi}'
            index = f'Index[pmGFWeak{n}]' + (f',Index[{colour}]' if colour else '')
            args = ',cc_' if colour else ''
            target = ',cc' if colour else ''
            rules = [f'{symbol}[sp_,{i+1}{args}] :> {physical_symbols[i]}[sp{target}]' for i in range(n)]
            classes.append(f'F[{1000+fi}] == {{ClassName -> {symbol}, SelfConjugate -> False, Unphysical -> True,'
                           f'Indices -> {{{index}}}, FlavorIndex -> pmGFWeak{n}, QuantumNumbers -> {{Y -> {str(hypercharge)}}},'
                           'Definitions -> {' + ','.join(rules) + '}}')
            terms.append(f'I*{symbol}bar.Ga[mu].DC[{symbol},mu]')
            pdg_offset += max(100, n)
            class_offset += max(1000, n)
        smfile = source / 'SM_generic_fermion.fr'
        smfile.write_text(sm)
        extension = source / 'generic_fermion.fr'
        indices = ['SU2D', 'SU2W'] + [f'pmGFWeak{n}' for n in dimensions]
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                             'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}};\n'
                             + '\n'.join(definitions) + '\nM$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
                             + 'M$Parameters = {\n' + ',\n'.join(params) + '\n};\n'
                             + '(* ' + BASIS_VERSION + ': equivalent-representation chiral transformations commute with covariant generators. *)\n'
                             + 'LGenericFermion := Block[{mu}, ExpandIndices[' + ' + '.join(terms)
                             + ', FlavorExpand -> {' + ','.join(indices) + '}]];\n')
        return [smfile, extension], 'LSM + LGenericFermion', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import _fields, _coefficients, _number, PreparedModel, ResolvedOperator, ModelResolutionError, CompilerResolutionError
    from workflow_errors import ResourceLimitError
    fields = tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    limit = inputs.get('compiler_limits', {}).get('max_weak_generator_entries', 100000)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ModelResolutionError('compiler_limits.max_weak_generator_entries must be a positive integer')
    entries = sum(f.su2*f.su2*3 for f in fields)
    if entries > limit:
        raise ResourceLimitError(f'Weak generator construction requires {entries} entries, exceeding compiler_limits.max_weak_generator_entries={limit}')
    if any(inputs.get('vevs', {}).get(f.name, 0) != 0 for f in fields):
        raise ModelResolutionError('Fermion fields cannot have ordinary c-number vacuum expectation values')
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    if mh <= 0:
        raise ModelResolutionError('higgs_mass must be positive')
    coefficients = _coefficients(lagrangian, inputs)
    operators, issues = [], {}
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        values = coefficients[row['id']]
        if reason:
            if all(value == 0 for value in values):
                kind = 'zero_verified'
            else:
                issues[row['id']] = reason
                continue
        operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                         values, expressions=basis_labels(row, fields), add_hc=row['field_content'] != row['conjugate_content']))
    if issues:
        raise CompilerResolutionError('Unresolved selected Dirac multiplet operators:\n' + '\n'.join(f'{key}: {reason}' for key, reason in issues.items()))
    groups = {}
    for field in fields:
        groups.setdefault(_representation(field), []).append(field)
    blocks, physical = {}, {}
    tokens = _tokens(fields)
    for members in groups.values():
        names = [f.name for f in members]
        positions = {name: i for i, name in enumerate(names)}
        size = len(names)
        left, right, mass = (np.zeros((size, size), complex) for _ in range(3))
        for op in operators:
            factors = list(Counter(dict(op.factors)).elements())
            if op.kind == 'generic_dirac_kinetic':
                bare = next(name[2:-1] for name in factors if name.startswith('D('))
                dagger = next(name[:-1] for name in factors if name.endswith('†'))
                (a, suffix), (b, _) = tokens[bare], tokens[dagger]
                if a.name not in positions or b.name not in positions:
                    continue
                i, j = positions[b.name], positions[a.name]
                matrix = left if suffix == '_L' else right
                coefficient = complex(op.coefficients[0])
                matrix[i, j] += coefficient
                matrix[j, i] += coefficient.conjugate()
            elif op.kind == 'generic_dirac_mass':
                by_suffix = {tokens[name][1]: tokens[name][0].name for name in factors}
                if by_suffix['_L'] in positions and by_suffix['_Rc'] in positions:
                    mass[positions[by_suffix['_L']], positions[by_suffix['_Rc']]] -= op.coefficients[0]
        block = resolve_block(names, left, right, mass)
        n = members[0].su2
        block['weak_conjugation'] = np.fliplr(np.diag([(-1.)**i for i in range(n)]))
        block['gauge_representation'] = dict(su2=n, su3=members[0].su3, hypercharge=str(members[0].hypercharge))
        block['mass_convention'] = 'L.T C Rc; Rc=C.T times charge-conjugate physical right field, with native colour identity'
        blocks['/'.join(names)] = block
        physical.update(zip(names, map(float, block['masses'])))
    compiler = GenericFermionCompiler(fields, tuple(operators), blocks, physical, mh)
    return PreparedModel(lagrangian, dict(inputs), tuple(operators), fields, None, runtime=runtime, compiler=compiler)
