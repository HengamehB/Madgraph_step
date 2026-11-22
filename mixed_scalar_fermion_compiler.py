"""Real singlet scalar plus vectorlike Dirac multiplet in the physical basis.

A stationary scalar VEV shifts the complete complex Dirac mass. Independent
chiral normalization and its mass phase are applied to the scalar Yukawa before
Higgs/singlet mixing. The vacuum contribution is included exactly once in the
canonical fermion mass; the exported Yukawa contains scalar fluctuations only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import numpy as np

from generic_fermion_compiler import (supports_fields as dirac_supports, classify as dirac_classify,
                                     basis_labels as dirac_labels, GenericFermionCompiler, resolve_block)

BASIS_VERSION = 'real-singlet-dirac-chiral-and-scalar-mixing-v1'


def _parts(fields):
    return tuple(f for f in fields if f.spin == 'scalar'), tuple(f for f in fields if f.spin == 'fermion')


def supports_fields(fields):
    scalars, fermions = _parts(fields)
    return (len(fields) == 2 and len(scalars) == len(fermions) == 1
            and scalars[0].kind == 'real' and scalars[0].su2 == 1
            and scalars[0].su3 == (0, 0) and scalars[0].hypercharge == 0
            and dirac_supports(fermions))


def uses_generic(fields):
    if not supports_fields(fields):
        return False
    fermion = _parts(fields)[1][0]
    return not (fermion.su2 == 1 and fermion.su3 == (0, 0) and fermion.hypercharge == 0)


def classify(row, fields):
    from feynrules_export import _classify
    if not supports_fields(fields):
        return None, 'compiler gap: mixed scalar/fermion compiler needs one neutral real singlet and one vectorlike Dirac multiplet'
    scalars, fermions = _parts(fields)
    scalar_kind, scalar_reason = _classify(row, scalars)
    if not scalar_reason:
        return scalar_kind, ''
    kind, reason = dirac_classify(row, fermions)
    if not reason:
        return kind, ''
    scalar, fermion = scalars[0], fermions[0]
    counts = Counter({part['name']: part['count'] for part in row['factors']})
    if not row['derivatives'] and row['multiplicity'] == 1 and counts == {
            scalar.name: 1, fermion.name + '_L': 1, fermion.name + '_Rc': 1}:
        return 'mixed_scalar_dirac_yukawa', ''
    return None, 'compiler gap: nonzero Majorana, SM-mixing or other mixed Yukawa contraction requires additional fermion/Lorentz lowering'


def basis_labels(row, fields):
    kind, reason = classify(row, fields)
    if kind in ('generic_dirac_kinetic', 'generic_dirac_mass'):
        return dirac_labels(row, _parts(fields)[1])
    if kind == 'mixed_scalar_dirac_yukawa':
        scalar, fermion = _parts(fields)[0][0], _parts(fields)[1][0]
        return (f'{scalar.name} times conventional Dirac L.T C Rc + h.c.; scalar VEV enters mass before chiral normalization and phase removal; native colour identity',)
    if kind in ('kinetic', 'potential'):
        return (row['field_content'] + '; existing real-singlet monomial convention, no implicit factorial',)
    return tuple(f'unlowered invariant {i+1}: {reason}' for i in range(row['multiplicity']))


@dataclass
class MixedScalarFermionCompiler:
    scalar_fields: tuple
    fermion_fields: tuple
    scalar_operators: tuple
    fermion_operators: tuple
    scalar_basis: dict
    fermion_basis: dict
    physical_mass: float
    effective_yukawa: complex
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        result = dict(zip(self.scalar_basis['names'], map(float, self.scalar_basis['masses'])))
        field = self.fermion_fields[0]
        result.update({field.name.lower() + str(i+1): self.physical_mass for i in range(field.su2)})
        return result

    def write_sources(self, model, source, fr):
        from feynrules_export import PreparedModel, _internal, _wl, CompilerResolutionError
        source, fr = Path(source), Path(fr)
        scalar_model = PreparedModel(model.lagrangian, model.inputs, self.scalar_operators,
                                    self.scalar_fields, self.scalar_basis, runtime=model.runtime)
        scalar_files, _, scalar_particles = scalar_model._automatic_sources(source, fr)
        field = self.fermion_fields[0]
        fermion_compiler = GenericFermionCompiler(self.fermion_fields, self.fermion_operators,
                             self.fermion_basis, {field.name: self.physical_mass}, float(self.scalar_basis['masses'][0]))
        fermion_model = PreparedModel(model.lagrangian, model.inputs, self.fermion_operators,
                                     self.fermion_fields, None, runtime=model.runtime, compiler=fermion_compiler)
        fermion_files, _, fermion_particles = fermion_compiler.write_sources(fermion_model, source, fr)
        # Load the SM with scalar transformations exactly once; register the
        # same arbitrary weak generator/index used by the Dirac source.
        if field.su2 > 1:
            sm = scalar_files[0].read_text()
            marker = 'Representations   -> {Ta,SU2D}'
            if marker not in sm:
                raise CompilerResolutionError('Bundled SM weak-representation declaration changed')
            sm = sm.replace(marker, f'Representations   -> {{{{Ta,SU2D}},{{pmGFT{field.su2},pmGFWeak{field.su2}}}}}')
            scalar_files[0].write_text(sm)
        parameters = [_internal('pmSFYR', self.effective_yukawa.real, order=True),
                      _internal('pmSFYI', self.effective_yukawa.imag, order=True)]
        scalar_expression = '(' + ' + '.join(f'pmR1{j}*{symbol}' for j, symbol in enumerate(('H', 'pmS1'))
                                             if self.scalar_basis['transformation'][1, j] != 0) + ')'
        if scalar_expression == '()':
            raise CompilerResolutionError('Scalar physical-basis transformation lost the singlet direction')
        terms = []
        if self.effective_yukawa != 0:
            for index in range(1, field.su2 + 1):
                symbol = f'pmGFF1n{index}'
                term = f'(pmSFYR+I*pmSFYI)*{scalar_expression}*{symbol}bar.ProjM.{symbol}'
                terms.append(f'({term}+HC[{term}])')
        yukawa_file = source / 'mixed_scalar_fermion.fr'
        yukawa_file.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                              'M$InteractionOrderHierarchy = {{QCD,1},{QED,2},{NP,1}};\n'
                              'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
                              '(* ' + BASIS_VERSION + ': Yukawa fluctuations only; VEV is already in canonical Dirac mass. *)\n'
                              'LMixedScalarFermion := ' + (' + '.join(terms) or '0') + ';\n')
        return [*scalar_files, fermion_files[-1], yukawa_file], 'LSM + LNew + LGenericFermion + LMixedScalarFermion', [*scalar_particles, *fermion_particles]


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import (PreparedModel, ResolvedOperator, _fields, _coefficients, _scalar_basis,
                                 ModelResolutionError, CompilerResolutionError, _number)
    from compiler_limits import normalize_limits
    from workflow_errors import ResourceLimitError
    fields = tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    inputs = deepcopy(inputs)
    scalar_fields, fermion_fields = _parts(fields)
    scalar, fermion = scalar_fields[0], fermion_fields[0]
    limits = normalize_limits(inputs.get('compiler_limits'))
    budget = limits.get('max_weak_generator_entries', 100000)
    entries = 3 * fermion.su2 ** 2
    if entries > budget:
        raise ResourceLimitError(f'Weak generator construction requires {entries} entries, exceeding compiler_limits.max_weak_generator_entries={budget}')
    coefficients = _coefficients(lagrangian, inputs)
    scalar_ops, fermion_ops, issues = [], [], {}
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        values = coefficients[row['id']]
        if reason:
            if all(value == 0 for value in values):
                kind = 'zero_verified'
            else:
                issues[row['id']] = reason
                continue
        operator = ResolvedOperator(row['id'], kind, tuple((v['name'], v['count']) for v in row['factors']),
                                    values, expressions=basis_labels(row, fields),
                                    add_hc=row['field_content'] != row['conjugate_content'])
        (scalar_ops if kind in ('kinetic', 'potential') else fermion_ops).append(operator)
    if issues:
        raise CompilerResolutionError('Unresolved mixed scalar/fermion operators:\n' + '\n'.join(f'{key}: {reason}' for key, reason in issues.items()))
    scalar_basis = _scalar_basis(scalar_fields, scalar_ops, inputs)
    kinetic = {'_L': 0., '_Rc': 0.}
    mass_coefficient, yukawa_coefficient = 0j, 0j
    for op in fermion_ops:
        if op.kind == 'generic_dirac_mass':
            mass_coefficient += complex(op.coefficients[0])
        elif op.kind == 'mixed_scalar_dirac_yukawa':
            yukawa_coefficient += complex(op.coefficients[0])
        elif op.kind == 'generic_dirac_kinetic':
            token = next(name[2:-1] for name, _ in op.factors if name.startswith('D('))
            suffix = '_Rc' if token.endswith('_Rc') else '_L'
            kinetic[suffix] += 2 * complex(op.coefficients[0]).real
    scalar_vev = scalar_basis['vevs'][scalar.name]
    effective_mass = -(mass_coefficient + scalar_vev * yukawa_coefficient)
    block = resolve_block((fermion.name,), [[kinetic['_L']]], [[kinetic['_Rc']]], [[effective_mass]])
    left, right = block['left_transformation'][0, 0], block['right_conjugate_transformation'][0, 0]
    effective_yukawa = yukawa_coefficient * left * right
    block.update(bare_mass_coefficient=mass_coefficient, scalar_yukawa_coefficient=yukawa_coefficient,
                 scalar_vev=scalar_vev, effective_yukawa=effective_yukawa,
                 physical_scalar_yukawas=effective_yukawa * scalar_basis['transformation'][1, :],
                 mass_convention='L.T C Rc; mass=-(bare coefficient + scalar VEV times Yukawa coefficient) before chiral normalization')
    inputs.setdefault('widths', {})
    for field in fields:
        width = _number(inputs['widths'].get(field.name, 0.), 'widths.' + field.name, real=True)
        if width < 0:
            raise ModelResolutionError('Widths must be nonnegative')
        inputs['widths'][field.name] = width
    if inputs.get('vevs', {}).get(fermion.name, 0) != 0:
        raise ModelResolutionError('Fermion fields cannot have ordinary c-number VEVs')
    compiler = MixedScalarFermionCompiler(scalar_fields, fermion_fields, tuple(scalar_ops), tuple(fermion_ops),
                 scalar_basis, {fermion.name: block}, float(block['masses'][0]), effective_yukawa)
    return PreparedModel(lagrangian, inputs, tuple(scalar_ops + fermion_ops), fields, scalar_basis,
                         runtime=runtime, compiler=compiler)
