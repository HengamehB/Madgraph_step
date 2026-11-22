"""Real-representation Weyl multiplets with canonical Majorana/Dirac components.

For integral weak isospin j and hypercharge zero, the invariant metric is
C[m,-m]=(-1)**(j-m). After a common Takagi transformation in species space,
the zero-weight component is Majorana and each ±q pair is one charged Dirac
field. The explicit embedding below fixes all phases and normalizations.
It lowers the full covariant current, including native adjoint colour, without
duplicating the charged degrees of freedom.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import math

import numpy as np

BASIS_VERSION = 'real-weyl-takagi-spherical-majorana-dirac-v1'


def supports_fields(fields):
    return bool(fields) and all(f.spin == 'fermion' and f.kind in ('weyl', 'majorana')
                               and f.hypercharge == 0 and f.su2 % 2 == 1
                               and f.su3 in {(0, 0), (1, 1)} for f in fields)


def uses_generic(fields):
    # Keep the established singlet/SM-neutrino conventions and portal compiler.
    return supports_fields(fields) and any(f.su2 != 1 or f.su3 != (0, 0) for f in fields)


def _representation(field):
    return field.su2, field.su3, field.hypercharge


def classify(row, fields):
    if not supports_fields(fields):
        return None, 'compiler gap: this real-Weyl compiler requires odd weak dimension, Y=0, and colour 1 or 8'
    tokens = {f.name: f for f in fields}
    factors = list(Counter({part['name']: part['count'] for part in row['factors']}).elements())
    if row['multiplicity'] == 1 and len(factors) == 2:
        if row['derivatives'] == 1:
            differentiated = [name[2:-1] for name in factors if name.startswith('D(')]
            daggered = [name[:-1] for name in factors if name.endswith('†')]
            if (len(differentiated) == len(daggered) == 1 and differentiated[0] in tokens
                    and daggered[0] in tokens and _representation(tokens[differentiated[0]]) == _representation(tokens[daggered[0]])):
                return 'generic_majorana_kinetic', ''
        elif not row['derivatives'] and all(name in tokens for name in factors):
            if _representation(tokens[factors[0]]) == _representation(tokens[factors[1]]):
                return 'generic_majorana_mass', ''
    return None, 'compiler gap: nonzero SM-mixing or Yukawa operators require additional fermion/Lorentz invariant lowering'


def basis_labels(row, fields):
    kind, reason = classify(row, fields)
    if kind == 'generic_majorana_kinetic':
        return ('i chi_b^dagger barsigma(mu) D_mu chi_a + h.c.; native gauge identity; canonical diagonal coefficient 1/2',)
    if kind == 'generic_majorana_mass':
        return ('chi_a.T C chi_b + h.c.; C[m,-m]=(-1)^(j-m), native colour identity; diagonal coefficient -M_aa/2 and off-diagonal coefficient -M_ab',)
    return tuple(f'unlowered invariant {index+1}: {reason}' for index in range(row['multiplicity']))


def component_embedding(dimension):
    """Map canonical (N_L, psi_1L, psi_1Rc, ...) to spherical Weyl weights.

    psi_qRc is the left-handed charge conjugate of the physical right-handed
    charged component. This map sends C to one Majorana unit and Dirac swap
    blocks; it fixes the W-current phases, including the neutral/charged pair.
    """
    if type(dimension) is not int or dimension < 1 or dimension % 2 != 1:
        raise ValueError('Real weak Weyl representations need a positive odd dimension')
    j = (dimension - 1) // 2
    phase = 1j if j % 2 else 1.
    embedding = np.zeros((dimension, dimension), complex)
    embedding[j, 0] = phase
    labels = ['neutral_L']
    for q in range(1, j + 1):
        embedding[j-q, 2*q-1] = phase
        embedding[j+q, 2*q] = phase * (-1)**q
        labels.extend((f'charge_{q}_L', f'charge_{q}_Rc'))
    metric = np.fliplr(np.diag([(-1.)**i for i in range(dimension)]))
    return dict(weak_dimension=dimension, weights=tuple(range(j, -j-1, -1)),
                canonical_weyl_labels=tuple(labels), transformation=embedding,
                invariant_metric=metric, canonical_mass_metric=embedding.T @ metric @ embedding)


def resolve_block(names, kinetic, mass):
    """Normalize K and Takagi-diagonalize the symmetric species mass form."""
    from neutral_fermion_compiler import takagi
    from feynrules_export import ModelResolutionError, CompilerResolutionError
    kinetic, mass = np.asarray(kinetic, complex), np.asarray(mass, complex)
    n = len(names)
    if any(value.shape != (n, n) for value in (kinetic, mass)):
        raise ModelResolutionError('Weyl kinetic/mass matrices have inconsistent dimensions')
    if any(not np.isfinite(value).all() for value in (kinetic, mass)):
        raise ModelResolutionError('Weyl kinetic/mass matrices must be finite')
    if not np.allclose(kinetic, kinetic.conj().T, rtol=1e-12, atol=0):
        raise ModelResolutionError('Weyl kinetic matrix must be Hermitian')
    if not np.allclose(mass, mass.T, rtol=1e-12, atol=0):
        raise ModelResolutionError('Majorana mass matrix must be symmetric')
    diagonal = kinetic.diagonal().real
    if np.any(diagonal <= 0):
        raise ModelResolutionError('Weyl kinetic matrix must be positive definite; canonical coefficient is 1/2 plus h.c.')
    scaling = 1/np.sqrt(diagonal)
    normalized = scaling[:, None]*kinetic*scaling[None, :]
    eigen, vectors = np.linalg.eigh(normalized)
    if eigen.min() <= 0:
        raise ModelResolutionError('Weyl kinetic matrix must be positive definite; canonical coefficient is 1/2 plus h.c.')
    if eigen.min() <= 1e-12*max(abs(eigen)):
        raise CompilerResolutionError('Weyl kinetic matrix is too ill-conditioned for reliable floating-point normalization')
    inverse = scaling[:, None]*((vectors / np.sqrt(eigen)) @ vectors.conj().T)
    canonical = inverse.T @ mass @ inverse
    mass_scale = float(np.max(abs(canonical)))
    try:
        rotation, masses = takagi(canonical / mass_scale if mass_scale else canonical)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise CompilerResolutionError('Takagi diagonalization failed: ' + str(exc)) from exc
    if mass_scale:
        masses *= mass_scale
    transform = inverse @ rotation
    overlap = np.sqrt(diagonal)[:, None]*transform
    overlap /= np.max(abs(overlap), axis=0)
    weights = abs(overlap / np.linalg.norm(overlap, axis=0))
    order = max(permutations(range(n)), key=lambda p: sum(weights[i, p[i]] for i in range(n)))
    transform, masses = transform[:, order], masses[list(order)]
    for i in range(n):
        pivot = transform[i, i] if abs(transform[i, i]) else transform[np.argmax(abs(transform[:, i])), i]
        if pivot.real < 0 or pivot.real == 0 and pivot.imag < 0:
            transform[:, i] *= -1
    kinetic_residual = float(np.max(abs(transform.conj().T @ kinetic @ transform - np.eye(n))))
    mass_residual = transform.T @ mass @ transform - np.diag(masses)
    relative_mass_residual = float(np.max(abs(mass_residual / mass_scale if mass_scale else mass_residual)))
    if max(kinetic_residual, relative_mass_residual) > 2e-8:
        raise CompilerResolutionError('Weyl kinetic/Takagi numerical residual exceeds tolerance')
    return dict(names=tuple(names), kinetic=kinetic, mass_matrix=mass, transformation=transform,
                masses=masses, residuals=dict(kinetic=kinetic_residual, mass_relative=relative_mass_residual))


@dataclass
class GenericMajoranaCompiler:
    fields: tuple
    operators: tuple
    fermion_basis: dict
    component_basis: dict
    physical_masses: dict
    higgs_mass: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {'H': self.higgs_mass, **{f.name.lower() + str(q): self.physical_masses[f.name]
                                      for f in self.fields for q in range((f.su2 + 1)//2)}}

    def write_sources(self, model, source, fr):
        from feynrules_export import _wl, ModelResolutionError, CompilerResolutionError
        source, fr = Path(source), Path(fr)
        sm = (fr/'Models'/'SM'/'SM.fr').read_text()
        marker = 'Mass            -> {MH,125}'
        if marker not in sm:
            raise CompilerResolutionError('Bundled SM Higgs declaration changed')
        sm = sm.replace(marker, f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        classes, particles, terms = [], [], []
        serial = 0
        for fi, field in enumerate(self.fields, 1):
            j = (field.su2 - 1)//2
            color = 8 if field.su3 == (1, 1) else 1
            mass = self.physical_masses[field.name]
            width = model.inputs.get('widths', {}).get(field.name, 0.)
            for q in range(j + 1):
                serial += 1
                symbol = f'pmMJF{fi}q{q}'
                name, anti = field.name.lower() + str(q), field.name.lower() + str(q) + 'bar'
                if q == 0:
                    anti = name
                pdg = 9700000 + serial
                index = 'Indices -> {Index[Gluon]},' if color == 8 else ''
                classes.append(f'F[{20000+serial}] == {{ClassName -> {symbol}, SelfConjugate -> {str(q == 0)}, {index}'
                               f'QuantumNumbers -> {{Q -> {q}, Y -> 0}}, Mass -> {{pmMJM{serial},{_wl(mass)}}},'
                               f'Width -> {{pmMJW{serial},{_wl(width)}}}, ParticleName -> {_wl(name)},'
                               + (f'AntiParticleName -> {_wl(anti)},' if q else '')
                               + f'PDG -> {pdg}, PropagatorLabel -> {_wl(name)}, PropagatorType -> Straight,'
                               + ('PropagatorArrow -> Forward}' if q else 'PropagatorArrow -> None}'))
                particles.append(dict(name=name, antiname=anti, pdg=pdg, source=field.name,
                                      spin=2, color=color, charge=q, mass=mass, width=width))
                derivative = 'DC' if color == 8 else 'del'
                factor = '1/2*' if q == 0 else ''
                terms.extend((f'I*{factor}{symbol}bar.Ga[mu].{derivative}[{symbol},mu]',
                              f'-{factor}pmMJM{serial}*{symbol}bar.{symbol}'))
                if q:
                    terms.extend((f'ee*{q}*{symbol}bar.Ga[mu].{symbol}*A[mu]',
                                  f'ee*cw/sw*{q}*{symbol}bar.Ga[mu].{symbol}*Z[mu]'))
                    lower = f'pmMJF{fi}q{q-1}'
                    squared = Fraction((j+q)*(j-q+1), 2)
                    ladder = 'Sqrt[' + str(squared) + ']'
                    current = f'ee/sw*{ladder}*{symbol}bar.Ga[mu].{lower}*W[mu]'
                    terms.append(f'({current}+HC[{current}])')
        smfile, extension = source/'SM_generic_majorana.fr', source/'generic_majorana.fr'
        smfile.write_text(sm)
        comments = '\n'.join('(* ' + op.operator_id + ': ' + op.kind + ' *)' for op in self.operators)
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                             'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}};\n'
                             'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
                             '(* ' + BASIS_VERSION + ': complete canonical covariant current after Takagi species rotation and spherical charge pairing. *)\n'
                             + comments + '\nLGenericMajorana := Block[{mu}, ExpandIndices['
                             + ' + '.join(terms) + ', FlavorExpand -> {SU2D,SU2W}]];\n')
        return [smfile, extension], 'LSM + LGenericMajorana', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import (PreparedModel, ResolvedOperator, _fields, _coefficients, _number,
                                 ModelResolutionError, CompilerResolutionError)
    from compiler_limits import normalize_limits
    from workflow_errors import ResourceLimitError
    fields = tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    limits = normalize_limits(inputs.get('compiler_limits'))
    budget = limits.get('max_weak_generator_entries', 100000)
    entries = sum(3*f.su2**2 for f in fields)
    if entries > budget:
        raise ResourceLimitError(f'Weak generator construction requires {entries} entries, exceeding compiler_limits.max_weak_generator_entries={budget}')
    if any(inputs.get('vevs', {}).get(f.name, 0) != 0 for f in fields):
        raise ModelResolutionError('Fermion fields cannot have ordinary c-number VEVs')
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    if mh <= 0:
        raise ModelResolutionError('higgs_mass must be positive')
    coefficients = _coefficients(lagrangian, inputs)
    operators, unresolved = [], {}
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        values = coefficients[row['id']]
        if reason:
            if all(value == 0 for value in values):
                kind = 'zero_verified'
            else:
                unresolved[row['id']] = reason
                continue
        operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                         values, expressions=basis_labels(row, fields), add_hc=row['field_content'] != row['conjugate_content']))
    if unresolved:
        raise CompilerResolutionError('Unresolved real-Weyl operators:\n' + '\n'.join(f'{key}: {reason}' for key, reason in unresolved.items()))
    groups = {}
    for field in fields:
        groups.setdefault(_representation(field), []).append(field)
        width = _number(inputs.get('widths', {}).get(field.name, 0.), 'widths.' + field.name, real=True)
        if width < 0:
            raise ModelResolutionError('Widths must be nonnegative')
    blocks, masses = {}, {}
    for members in groups.values():
        names = [f.name for f in members]
        position = {name: i for i, name in enumerate(names)}
        kinetic, mass = (np.zeros((len(names), len(names)), complex) for _ in range(2))
        for op in operators:
            factors = list(Counter(dict(op.factors)).elements())
            if op.kind == 'generic_majorana_kinetic':
                bare = next(name[2:-1] for name in factors if name.startswith('D('))
                dagger = next(name[:-1] for name in factors if name.endswith('†'))
                if bare in position and dagger in position:
                    i, j = position[dagger], position[bare]
                    kinetic[i, j] += op.coefficients[0]
                    kinetic[j, i] += complex(op.coefficients[0]).conjugate()
            elif op.kind == 'generic_majorana_mass' and all(name in position for name in factors):
                i, j = [position[name] for name in factors]
                mass[i, j] -= op.coefficients[0]
                mass[j, i] -= op.coefficients[0]
        block = resolve_block(names, kinetic, mass)
        block['gauge_representation'] = dict(su2=members[0].su2, su3=members[0].su3, hypercharge='0')
        block['mass_convention'] = 'L=-1/2 chi_a.T M_ab C chi_b+h.c.; C[m,-m]=(-1)^(j-m); see component_basis for physical Majorana/Dirac pairing'
        blocks['/'.join(names)] = block
        masses.update(zip(names, map(float, block['masses'])))
    components = {field.name: component_embedding(field.su2) for field in fields}
    compiler = GenericMajoranaCompiler(fields, tuple(operators), blocks, components, masses, mh)
    return PreparedModel(lagrangian, dict(inputs), tuple(operators), fields, None, runtime=runtime, compiler=compiler)
