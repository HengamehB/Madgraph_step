"""Deterministic, explicitly recorded starting coefficients, not fitted physics.

Defaults are expressed in each compiler's actual invariant basis. A default is
not a claim that the resulting potential has a stable or exportable vacuum.
Unknown conventions remain unresolved instead of receiving an arbitrary zero.
"""
from collections import Counter

from generic_colour_yukawa import KINDS as COLOURED_YUKAWAS

PROFILE = 'coefficient-seeds-v1'
REFERENCE_MASSES = {'X': 200., 'Y': 600.}
SM_DEFAULTS = {'Gf': 1.16637e-5, 'MZ': 91.1876, 'aEWM1': 127.9}


def _tokens(row):
    return [part['name'] for part in row['factors'] for _ in range(part['count'])]


def _bare(token):
    if token.startswith('D(') and token.endswith(')'):
        token = token[2:-1]
    return token.removesuffix('†').removesuffix('_L').removesuffix('_Rc')


def _fit(polynomials, target):
    """Project a canonical norm onto the actual basis and verify reconstruction."""
    import numpy as np
    monomials = sorted(set(target).union(*(set(p) for p in polynomials)))
    matrix = np.array([[p.get(m, 0) for p in polynomials] for m in monomials], complex)
    right = np.array([target.get(m, 0) for m in monomials], complex)
    # Hermitian coefficients are real; complex entries in the component basis
    # must still reconstruct both real and imaginary target components.
    real_matrix = np.concatenate((matrix.real, matrix.imag))
    real_right = np.concatenate((right.real, right.imag))
    answer, _, rank, _ = np.linalg.lstsq(real_matrix, real_right, rcond=1e-12)
    if rank != len(polynomials) or not np.allclose(matrix @ answer, right, rtol=1e-10, atol=1e-12):
        return None
    from fractions import Fraction
    values = []
    for value in answer:
        if abs(value) <= 1e-12:
            values.append(0.)
            continue
        rational = float(Fraction(float(value)).limit_denominator(10000))
        values.append(rational if abs(rational - value) <= 1e-13 * abs(value) else float(value))
    return values


def _scalar_norm(row, info, definition, name):
    """Coefficient vector for +1/2 real or +1 complex covariant norm."""
    fields, limits = definition.fields, definition.compiler_limits
    field = next(f for f in fields if f.name == name)
    kind = info['kind']
    if kind.startswith('generic_scalar_') or kind in ('generic_vector_kinetic', 'generic_vector_potential'):
        from generic_scalar_compiler import coordinates, invariant_basis
        coords = coordinates(tuple(fields))
        fake = dict(row, multiplicity=1) if kind == 'generic_vector_kinetic' else row
        if kind == 'generic_vector_potential':
            from generic_vector_invariants import invariant_basis as vector_basis
            basis = [{tuple(index // 20 for index in monomial): value for monomial, value in polynomial.items()
                      if all(index % 20 == 0 for index in monomial)}
                     for polynomial in vector_basis(row, fields, limits)['basis']]
        else:
            basis = invariant_basis(fake, fields, limits)
        target = {}
        for index, source in enumerate(coords.sources):
            if source != name:
                continue
            other = coords.bars[index]
            if index <= other:
                target[tuple(sorted((index, other)))] = .5 if index == other else 1.
        return _fit(basis, target)
    if kind.startswith('joint_scalar_'):
        from generic_joint_scalar_compiler import weak_coordinates, _physical_polynomial
        from joint_scalar_invariants import kinetic_basis, invariant_basis
        from generic_color_compiler import _REPS
        coords = weak_coordinates(fields)
        basis = (kinetic_basis if row['derivatives'] else invariant_basis)(row, fields, limits)[1]
        physical = [_physical_polynomial(p, coords, 1.) for p in basis]
        target = {}
        for flat in coords.flat_fields:
            if coords.sources[flat.name] != name:
                continue
            for colour in range(abs(_REPS[flat.su3])):
                second = flat.name if flat.kind == 'real' else flat.name + '†'
                target[tuple(sorted(((flat.name, colour), (second, colour))))] = .5 if flat.kind == 'real' else 1.
        return _fit(physical, target)
    if kind.startswith('colour_scalar_'):
        from generic_color_compiler import kinetic_basis, invariant_basis, _REPS
        basis = (kinetic_basis if row['derivatives'] else invariant_basis)(row, fields, limits)[1]
        second = name if field.kind == 'real' else name + '†'
        target = {tuple(sorted(((name, i), (second, i)))): .5 if field.kind == 'real' else 1.
                  for i in range(abs(_REPS[field.su3]))}
        return _fit(basis, target)
    if info['multiplicity'] == 1 and kind in {'kinetic', 'potential', 'triplet_kinetic',
            'triplet_potential', 'scalar_covariant_kinetic', 'doublet_kinetic',
            'doublet_potential', 'norm_potential', 'proca_kinetic'}:
        return [.5 if field.kind == 'real' else 1.]
    return None


def coefficient_default(definition, info):
    """Return value/role/reason, or an explicit unresolved basis diagnostic."""
    row = next(row for row in definition.lagrangian.operators if row['id'] == info['operator_id'])
    tokens = _tokens(row)
    names = [_bare(token) for token in tokens]
    fields = {f.name: f for f in definition.fields}
    kind = info['kind'] or ''
    count, dimension = len(tokens), info['dimension']
    result = dict(profile=PROFILE, role='interaction', units=info['units'], value=None,
                  reason=info.get('obstruction') or 'No verified default for this contraction basis.')
    if not kind or info.get('obstruction'):
        return result
    diagonal = len(set(names)) == 1 and names[0] in fields if names else False
    field = fields[names[0]] if diagonal else None
    value = None
    if 'kinetic' in kind or kind == 'doublet_kinetic_mixing':
        result['role'] = 'kinetic'
        if not diagonal:
            value = [0.] * info['multiplicity']
            result['reason'] = 'Declared zero kinetic mixing; diagonal norms are normalized separately.'
        elif field.spin == 'fermion' and info['multiplicity'] == 1:
            differentiated = [t[2:-1] for t in tokens if t.startswith('D(')]
            daggered = [t[:-1] for t in tokens if t.endswith('†')]
            if differentiated == daggered:
                value = [.5]
                result['reason'] = 'Canonical Weyl kinetic norm, including the declared Hermitian conjugate.'
        elif field.spin == 'scalar':
            value = _scalar_norm(row, info, definition, field.name)
            result['reason'] = 'Canonical scalar norm projected onto the displayed invariant basis.'
        elif field.spin == 'vector' and info['multiplicity'] == 4:
            norm = (_scalar_norm(row, info, definition, field.name) if kind == 'generic_vector_kinetic'
                    else [.5 if field.kind == 'real' else 1.])
            if norm is not None:
                value = [-norm[0], 0., norm[0], 0.]
                result['reason'] = 'Canonical Proca kinetic norm with c1+c2+c3=0; no longitudinal ghost term.'
    elif row['derivatives'] == 0 and count == 2:
        result['role'] = 'quadratic'
        if not diagonal:
            value = [0.] * info['multiplicity']
            result['reason'] = 'Declared zero mass mixing.'
        elif field.spin == 'scalar':
            norm = _scalar_norm(row, info, definition, field.name)
            if norm is not None:
                value = [-REFERENCE_MASSES[field.name]**2 * z for z in norm]
                result['reason'] = f'Reference mass {REFERENCE_MASSES[field.name]:g} GeV before VEV shifts/mixing; actual masses are derived.'
        elif field.spin == 'fermion' and info['multiplicity'] == 1 and kind in {
                'weyl_mass', 'dirac_mass', 'generic_dirac_mass', 'generic_majorana_mass', 'colour_dirac_mass'}:
            value = [-REFERENCE_MASSES[field.name] * (.5 if len(set(tokens)) == 1 else 1.)]
            result['reason'] = f'Reference mass {REFERENCE_MASSES[field.name]:g} GeV in the declared Weyl-pair convention; actual masses are derived.'
        elif field.spin == 'vector' and info['multiplicity'] == 1 and kind in {'mass', 'potential', 'norm_potential'}:
            value = [REFERENCE_MASSES[field.name]**2 * (.5 if field.kind == 'real' else 1.)]
            result['reason'] = 'Positive Proca reference mass squared in the displayed norm convention.'
        elif field.spin == 'vector' and kind == 'generic_vector_potential':
            norm = _scalar_norm(row, info, definition, field.name)
            if norm is not None:
                value = [REFERENCE_MASSES[field.name]**2 * z for z in norm]
                result['reason'] = 'Proca reference mass projected onto the actual Lorentz/weak invariant basis.'
    elif kind == 'higgs_divergence':
        value = [0.] * info['multiplicity']
        result['role'] = 'constrained_interaction'
        result['reason'] = 'Zero Higgs divergence matches the real Higgs-current seed and preserves the Proca constraint.'
    elif dimension > 0:
        value = [0.] * info['multiplicity']
        result['role'] = 'dimensionful_interaction'
        result['reason'] = 'Declared zero dimensionful interaction seed (linear/cubic/mixing), in the displayed units.'
    elif kind in COLOURED_YUKAWAS:
        value = [.01] * info['multiplicity']
        result['role'] = 'dimensionless_interaction'
        result['reason'] = 'Small real seed for each independent complex flavour coupling in the stated order.'
    else:
        scalar_potential = 'potential' in kind and all(name in fields or name == 'H' for name in names)
        sign = -1. if scalar_potential and all(fields[n].spin == 'scalar' for n in names if n in fields) else 1.
        value = [sign * .01] * info['multiplicity']
        result['role'] = 'quartic' if scalar_potential else 'dimensionless_interaction'
        result['reason'] = ('Small Lagrangian quartic seed (L=-V); full boundedness/vacuum still requires resolution.'
                            if sign < 0 else 'Small dimensionless interaction seed in the displayed real/Hermitian invariant basis.')
    if value is not None:
        result['value'] = value[0] if info['multiplicity'] == 1 else value
    return result
