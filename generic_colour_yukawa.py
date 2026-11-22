"""Physical flavour bases for two native-colour scalar–fermion–fermion invariants."""

from fractions import Fraction
from hashlib import sha256
from itertools import combinations_with_replacement, product
import json
from math import isqrt


SEXTET = 'sextet_qq_yukawa'
TRIPLET = 'triplet_eu_yukawa'
KINDS = frozenset((SEXTET, TRIPLET))
BASIS_VERSION = 'coloured-scalar-sm-yukawa-v1'


def classify(row, fields):
    if row['derivatives']:
        return None
    counts = {factor['name']: factor['count'] for factor in row['factors']}
    for field in fields:
        if field.spin != 'scalar' or field.kind != 'complex':
            continue
        rep = (field.su3, field.su2, field.hypercharge)
        if rep == ((2, 0), 3, Fraction(1, 3)) and counts == {'Q': 2, field.name + '†': 1}:
            return SEXTET
        if rep == ((1, 0), 1, Fraction(-1, 3)) and counts == {'ec': 1, 'uc': 1, field.name: 1}:
            return TRIPLET
    return None


def normalization(kind):
    if kind == SEXTET:
        return ('One half times the full symmetric complex matrix Y_QQ[i,j], with normalized sextet '
                'colour Clebsches and i sigma2 sigmaA in the weak triplet. In descending T3 order '
                'the weak bilinears are sqrt(2) u_i u_j, u_i d\'_j + d\'_i u_j, sqrt(2) d\'_i d\'_j, '
                'where d\'_i = sum_k CKM[i,k] d_k. Quarks are left-handed. Add the Hermitian conjugate.')
    if kind == TRIPLET:
        return ('lambda_R[i,j] S_a ubar_i^a P_L C(lbar_j)^T plus its Hermitian conjugate; '
                'the colour contraction is the identity. Both physical fermions are right-handed. '
                'Row-major order: up-quark generation i, charged-lepton generation j.')
    raise ValueError(f'Unknown coloured-scalar Yukawa family {kind!r}')


def _flavours(kind, count):
    if kind == SEXTET:
        n = (isqrt(8 * count + 1) - 1) // 2
        if n > 0 and n * (n + 1) // 2 == count:
            return tuple(combinations_with_replacement(range(1, n + 1), 2))
    elif kind == TRIPLET:
        n = isqrt(count)
        if n > 0 and n * n == count:
            return tuple(product(range(1, n + 1), repeat=2))
    raise ValueError(f'{kind} is incompatible with catalogue multiplicity {count}')


def basis_labels(row, fields):
    kind = classify(row, fields)
    label = 'Y_QQ' if kind == SEXTET else 'lambda_R'
    return tuple(f'{label}[{i},{j}]' for i, j in _flavours(kind, row['multiplicity']))


def basis_fingerprint(row, fields):
    kind = classify(row, fields)
    binding = (BASIS_VERSION, kind, row['factors'], basis_labels(row, fields), normalization(kind))
    return sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def _quark_components(weak, i):
    """Mass-basis components of the ith SM doublet in the up-diagonal convention."""
    if weak == 0:
        return (('uq', i, None),)
    return tuple(('dq', k, f'CKM[{i},{k}]') for k in range(1, 4))


def _qq_term(scalar, symbols, i, j):
    # The symmetric square of two doublets has the same positive raising
    # matrix elements as the spin-one kinetic term. The sigmaA convention
    # gives sqrt(2) times the unit-normalized symmetric weak tensor.
    components = []
    for a, b in product(range(2), repeat=2):
        scalar_component = symbols[scalar, a + b].replace('{index}', 'c3')
        weak_factor = 'Sqrt[2]' if a == b else '1'
        for (left, f, v), (right, g, w) in product(_quark_components(a, i), _quark_components(b, j)):
            bilinear = f'ProjM[s1,s2]*CC[{left}bar][s1,{f},c1].{right}[s2,{g},c2]'
            components.append('*'.join(x for x in (weak_factor, v, w, scalar_component, bilinear) if x))
    # Off-diagonal flavour terms occur twice in 1/2 sum_ij Y_ij Q_i Q_j.
    # FeynRules converts differentiated fields to all-outgoing particles.
    # K6 therefore connects the outgoing sextet and two outgoing antiquarks.
    factor = '1/2' if i == j else '1'
    return f'{factor}*K6[c3,c1,c2]*(' + '+'.join(components) + ')'


def render(op, symbols):
    """Emit the normal and HC terms after the scalar kinetic/mass rotations."""
    from feynrules_export import _internal

    scalar = next(name for name, count in op.factors if name not in ('Q', 'ec', 'uc'))
    parameters, terms = [], []
    flavours = _flavours(op.kind, len(op.coefficients))
    for index, ((i, j), value) in enumerate(zip(flavours, op.coefficients), 1):
        if value == 0:
            continue
        parameter = f'pmCYO{op.operator_id[1:]}n{index}'
        parameters.append(_internal(parameter, value, order=True))
        if op.kind == SEXTET:
            expression = _qq_term(scalar, symbols, i, j)
        else:
            component = symbols[scalar, 0].replace('{index}', 'c1')
            expression = component + f'*ProjM[s1,s2]*uqbar[s1,{i},c1].CC[l][s2,{j}]'
        term = f'{parameter}*({expression})'
        terms.append(f'({term}+HC[{term}])')
    return parameters, terms


def lepton_labels(operators, scalar_maps):
    """Label physical leptoquarks from nonzero Yukawas and actual field rotations."""
    possibilities = {}
    for op in operators:
        if op.kind != TRIPLET or not any(op.coefficients):
            continue
        source = next(name for name, _ in op.factors if name not in ('ec', 'uc'))
        for (state, bar), value in scalar_maps[source, 0].items():
            if value:
                possibilities.setdefault(state, set()).add(-1 if bar else 1)
    if not possibilities:
        return {}, ''
    charges = {state: next(iter(labels)) for state, labels in possibilities.items() if len(labels) == 1}
    notes = ['LeptonNumber labels follow the active right-handed leptoquark couplings.']
    if set(possibilities) != set(charges):
        notes.append('States with incompatible particle and antiparticle projections have no assigned lepton number.')

    # A label inferred from one vertex is not a constraint on the model.
    # Identify the selected terms which can violate these conventional charges.
    raw_charges = {}
    for (token, _), mapping in scalar_maps.items():
        if token.endswith('†'):
            continue
        for (state, bar), value in mapping.items():
            if value:
                raw_charges.setdefault(token, set()).add(charges.get(state, 0) * (-1 if bar else 1))
    sm_charges = {'L': 1, 'ec': -1}
    violations = []
    for op in operators:
        if not any(op.coefficients):
            continue
        totals = {0}
        for token, count in op.factors:
            if token.startswith('D('):
                token = token[2:-1]
            base = token.removesuffix('†')
            values = raw_charges.get(base, {sm_charges.get(base, 0)})
            sign = -1 if token.endswith('†') else 1
            for _ in range(count):
                totals = {a + sign*b for a in totals for b in values}
        if totals - {0}:
            violations.append(op.operator_id)
    if violations:
        notes.append('These selected terms can violate the assigned lepton numbers: ' + ', '.join(violations) + '.')
    return charges, '(* ' + ' '.join(notes) + ' *)\n'
