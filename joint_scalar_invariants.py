"""Joint SU(2) x native-SU(3) scalar invariants, before weak symmetry breaking.

Weak singlets are constructed on distinguishable ordered legs. Their products
with native colour networks are Bose symmetrized only after combining BOTH
groups. Separately reducing either group's identical legs would lose invariants.

Polynomial keys are sorted tuples of (token, weak_index, colour_index), with
zero-based component indices. Dagger tokens denote direct component conjugates,
not SU(2) dual-converted fields. Colour singlets have colour_index=0. For real
integer-spin weak multiplets, HC reflects the weak index with the canonical C
sign. Native adjoint colour indices are Cartesian and have real conjugation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
import hashlib
import json
import math

import numpy as np

from compiler_limits import DEFAULTS, normalize_limits
from generic_color_compiler import _REPS, _anti, _networks, _tensor, NativeInvariant
from workflow_errors import ResourceLimitError
from component_invariants import InvariantVerificationError

BASIS_VERSION = 'joint-ordered-weak-native-colour-hermitian-v1'
_TOL = 1e-11


def _limits(limits):
    values = normalize_limits(vars(limits) if hasattr(limits, '__dataclass_fields__') else limits)
    return tuple(sorted({**DEFAULTS, **values}.items()))


def _bound(value, name, limits, operation):
    if value > limits[name]:
        raise ResourceLimitError(f'{operation} requires {value} entries; {name}={limits[name]}. '
                                 'Increase compiler_limits explicitly; no incomplete basis is returned.')


def _clean(poly):
    return {m: complex(0 if abs(z.real) < _TOL else z.real,
                       0 if abs(z.imag) < _TOL else z.imag)
            for m, z in poly.items() if abs(z) > _TOL}


def _sum(left, right, scale=1):
    result = dict(left)
    for monomial, value in right.items():
        result[monomial] = result.get(monomial, 0j) + scale * value
    return _clean(result)


def _rref(matrix):
    """Deterministic pivots; this fixes the basis even in degenerate subspaces."""
    result = np.array(matrix, dtype=float, copy=True)
    row, pivots = 0, []
    for column in range(result.shape[1]):
        if row == len(result):
            break
        pivot = row + int(np.argmax(abs(result[row:, column])))
        if abs(result[pivot, column]) <= _TOL:
            continue
        result[[row, pivot]] = result[[pivot, row]]
        result[row] /= result[row, column]
        for other in range(len(result)):
            if other != row:
                result[other] -= result[other, column] * result[row]
        pivots.append(column)
        row += 1
    result[abs(result) < _TOL] = 0
    return result, tuple(pivots)


def _ordered_multiplicity(dimensions):
    spins = Counter({0: 1})
    for dimension in dimensions:
        updated = Counter()
        for twice_j, count in spins.items():
            for output in range(abs(twice_j - dimension + 1), twice_j + dimension, 2):
                updated[output] += count
        spins = updated
    return spins[0]


def _exact_weak_coefficient(value):
    """Recover a group coefficient as an exact signed square root of a rational.

    Rescaling weight states by square roots of binomial coefficients makes
    the raising matrix rational. Its RREF kernel, normalized to first entry
    one, therefore has rational squared coefficients. Recover only that
    known algebraic class, within floating-point roundoff; physical input
    coefficients must never pass through this conversion.
    """
    value = float(value)
    if not math.isfinite(value):
        raise InvariantVerificationError('Non-finite ordered weak coefficient')
    if value == 0:
        return '0'
    squared = Fraction(value * value).limit_denominator(10**6)
    if not math.isclose(math.sqrt(squared), abs(value), rel_tol=5e-14, abs_tol=0.):
        raise InvariantVerificationError(
            f'Cannot recover exact rational-square weak coefficient {value!r}; '
            'no approximate FeynRules coefficient is emitted')
    numerator, denominator = squared.numerator, squared.denominator
    root_n, root_d = math.isqrt(numerator), math.isqrt(denominator)
    if root_n * root_n == numerator and root_d * root_d == denominator:
        result = str(root_n) if root_d == 1 else f'{root_n}/{root_d}'
    else:
        result = f'Sqrt[{numerator}]' if denominator == 1 else f'Sqrt[{numerator}/{denominator}]'
    return ('-' if value < 0 else '') + result


@lru_cache(maxsize=128)
def _ordered_weak(dimensions, limits_key):
    """Sparse ordered singlet coefficient tensors, with no Bose reduction."""
    limits = dict(limits_key)
    if not dimensions:
        return ((((), 1.),),)
    total = sum(d - 1 for d in dimensions)
    if total % 2:
        return ()
    _bound(math.prod(dimensions[:-1]), 'max_weight_combinations', limits, 'Ordered weak weight matching')
    states = []
    for prefix in product(*(range(d) for d in dimensions[:-1])):
        last = total // 2 - sum(prefix)
        if 0 <= last < dimensions[-1]:
            states.append(prefix + (last,))
            _bound(len(states), 'max_zero_weight_states', limits, 'Ordered weak zero-weight space')
    targets, entries = {}, []
    for column, indices in enumerate(states):
        for leg, (index, dimension) in enumerate(zip(indices, dimensions)):
            if index:
                target = indices[:leg] + (index - 1,) + indices[leg + 1:]
                row = targets.setdefault(target, len(targets))
                entries.append((row, column, math.sqrt(index * (dimension - index))))
                _bound(len(entries), 'max_matrix_nonzeros', limits, 'Ordered weak raising matrix')
    _bound(len(targets) * len(states), 'max_matrix_entries', limits, 'Ordered weak raising matrix')
    matrix = np.zeros((len(targets), len(states)))
    for row, column, value in entries:
        matrix[row, column] += value
    reduced, pivots = _rref(matrix)
    free = [column for column in range(len(states)) if column not in pivots]
    expected = _ordered_multiplicity(dimensions)
    if len(free) != expected:
        raise InvariantVerificationError(f'Ordered weak rank yields {len(free)} singlets, exact tensor product requires {expected}')
    _bound(len(states) * len(free), 'max_output_terms', limits, 'Ordered weak basis')
    result = []
    for column in free:
        vector = np.zeros(len(states)); vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row, column]
        first = next((value for value in vector if abs(value) > _TOL), 1.)
        vector /= first
        if matrix.size and np.max(abs(matrix @ vector)) > 1e-8 * max(1., np.max(abs(matrix))):
            raise InvariantVerificationError('Ordered weak invariant failed its raising-generator residual')
        result.append(tuple((state, float(value)) for state, value in zip(states, vector) if abs(value) > _TOL))
    return tuple(result)


@dataclass(frozen=True)
class JointInvariant:
    legs: tuple
    colour_positions: tuple
    network: tuple
    weak_terms: tuple
    mode: str = 'plain'

    def render(self, symbols=None):
        """Render native colour tensors times sparse weak-component products.

        symbols[(token, weak_index)] is a FeynRules expression. For coloured
        fields it should contain ``{index}``, replaced by the native colour
        index. Maps already incorporate physical rotations and Higgs projection.
        """
        symbols = symbols or {}
        colour = NativeInvariant(self.network, (), (), 0).render()
        colour_slots = {position: local for local, position in enumerate(self.colour_positions)}
        terms = []
        for indices, coefficient in self.weak_terms:
            factors = [f'({_exact_weak_coefficient(coefficient)})', colour]
            for position, (token, weak) in enumerate(zip(self.legs, indices)):
                expression = symbols.get((token, weak), f'{token}[{weak + 1}]')
                if position in colour_slots:
                    index = f'c{colour_slots[position] + 1}'
                    expression = expression.replace('{index}', index) if '{index}' in expression else expression + f'[{index}]'
                factors.append('(' + expression + ')')
            terms.append('*'.join(factors))
        base = '(' + '+'.join(terms) + ')'
        if self.mode == 'real':
            return f'(({base})+HC[{base}])/2'
        if self.mode == 'imag':
            return f'(({base})-HC[{base}])/(2*I)'
        return base


def _leg_data(factors, fields):
    table = {f.name: f for f in fields if f.spin == 'scalar'}
    legs, dimensions, reps = [], [], []
    hypercharge = Fraction(0)
    for token, count in sorted(factors):
        base = token.removesuffix('†')
        if base == 'H':
            dimension, rep, charge = 2, 1, Fraction(1, 2)
        elif base in table and table[base].su3 in _REPS:
            field = table[base]
            dimension, rep, charge = field.su2, _REPS[field.su3], field.hypercharge
        else:
            raise ValueError(f'Joint scalar contraction requires a supported scalar leg: {token}')
        if token.endswith('†'):
            rep, charge = _anti(rep), -charge
        hypercharge += count * charge
        legs.extend([token] * count); dimensions.extend([dimension] * count); reps.extend([rep] * count)
    if len(legs) > 4:
        raise ValueError('Joint scalar lowering currently covers at most four scalar legs')
    if hypercharge:
        raise ValueError('Joint scalar contraction has nonzero hypercharge')
    return tuple(legs), tuple(dimensions), tuple(reps)


def conjugate_polynomial(poly, fields):
    """Hermitian conjugation in the documented direct weak-component basis."""
    real = {f.name: f.su2 for f in fields if f.kind == 'real'}
    result = {}
    for monomial, value in poly.items():
        factors, sign = [], 1
        for token, weak, colour in monomial:
            if token in real:
                sign *= (-1) ** weak
                weak = real[token] - 1 - weak
            else:
                token = token[:-1] if token.endswith('†') else token + '†'
            factors.append((token, weak, colour))
        key = tuple(sorted(factors))
        result[key] = result.get(key, 0j) + sign * complex(value).conjugate()
    return _clean(result)


@lru_cache(maxsize=128)
def _basis(fields, factors, expected, hermitian, limits_key):
    limits = dict(limits_key)
    legs, dimensions, reps = _leg_data(factors, fields)
    for token, weak, colour in zip(legs, dimensions, reps):
        _bound(weak * abs(colour), 'max_letter_components', limits, f'{token} component representation')
    positions = tuple(i for i, rep in enumerate(reps) if rep != 1)
    coloured_reps = tuple(reps[i] for i in positions)
    networks = _networks(coloured_reps)
    _bound(len(networks), 'max_colour_candidate_networks', limits, 'Native colour candidate networks')
    _bound(math.prod(abs(rep) for rep in coloured_reps), 'max_power_monomials', limits, 'Native colour tensor')
    weak_basis = _ordered_weak(dimensions, limits_key)
    # Convert canonical SU(2) dual legs to direct component conjugates. C has
    # precisely one nonzero entry (-1)^i at the reflected index of each row.
    weak_basis_direct = []
    for tensor in weak_basis:
        terms = []
        for indices, coefficient in tensor:
            direct = list(indices)
            for leg, token in enumerate(legs):
                if token.endswith('†'):
                    coefficient *= (-1) ** indices[leg]
                    direct[leg] = dimensions[leg] - 1 - indices[leg]
            terms.append((tuple(direct), coefficient))
        terms.sort()
        normalization = terms[0][1] if terms else 1.
        weak_basis_direct.append(tuple((indices, value / normalization) for indices, value in terms))
    basis, polynomials, vectors, seen = [], [], [], set()
    stored = 0
    for network in networks:
        colour_tensor = _tensor(network, len(positions))
        colour_terms = [(indices, complex(colour_tensor[indices])) for indices in np.ndindex(colour_tensor.shape)
                        if abs(colour_tensor[indices]) > _TOL]
        for weak_terms in weak_basis_direct:
            _bound(len(colour_terms) * len(weak_terms), 'max_power_monomials', limits, 'Joint candidate polynomial')
            polynomial = {}
            for (weak_indices, weak_value), (colour_indices, colour_value) in product(weak_terms, colour_terms):
                colours = dict(zip(positions, colour_indices))
                monomial = tuple(sorted((token, weak, colours.get(leg, 0))
                                        for leg, (token, weak) in enumerate(zip(legs, weak_indices))))
                polynomial[monomial] = polynomial.get(monomial, 0j) + weak_value * colour_value
            polynomial = _clean(polynomial)
            if hermitian:
                conjugated = conjugate_polynomial(polynomial, fields)
                candidates = [('real', {m: z / 2 for m, z in _sum(polynomial, conjugated).items()}),
                              ('imag', {m: z / (2j) for m, z in _sum(polynomial, conjugated, -1).items()})]
            else:
                candidates = [('plain', polynomial)]
            for mode, poly in candidates:
                if not poly:
                    continue
                key = tuple((m, round(z.real, 11), round(z.imag, 11)) for m, z in sorted(poly.items()))
                if key in seen:
                    continue
                seen.add(key)
                stored += len(poly)
                _bound(stored, 'max_output_terms', limits, 'Joint distinct candidate expansion')
                residual = dict(poly)
                original_norm = math.sqrt(sum(abs(z) ** 2 for z in poly.values()))
                # Reorthogonalize to avoid rank inflation in redundant colour
                # networks. Hermitian tensors form a REAL vector space.
                for _ in range(2):
                    for previous in vectors:
                        overlap = sum(z.conjugate() * residual.get(m, 0j) for m, z in previous.items())
                        if hermitian:
                            overlap = overlap.real
                        for m, z in previous.items():
                            residual[m] = residual.get(m, 0j) - overlap * z
                norm = math.sqrt(sum(abs(z) ** 2 for z in residual.values()))
                if norm > 1e-9 * max(1., original_norm):
                    basis.append(JointInvariant(legs, positions, network, weak_terms, mode))
                    polynomials.append(poly)
                    vectors.append({m: z / norm for m, z in residual.items() if abs(z) > _TOL * norm})
                    _bound(sum(map(len, vectors)), 'max_matrix_entries', limits, 'Joint rank verification')
    if len(basis) != expected:
        raise InvariantVerificationError(f'Joint weak-colour Bose-symmetrized rank {len(basis)} differs '
                                         f'from exact catalogue multiplicity {expected}; no incomplete basis is returned')
    return tuple(basis), tuple(polynomials)


def invariant_basis(row, fields, limits=None):
    if row['derivatives']:
        raise ValueError('Use kinetic_basis for two-derivative scalar kinetic operators')
    return _basis(tuple(fields), tuple((p['name'], p['count']) for p in row['factors']),
                  row['multiplicity'], row['field_content'] == row['conjugate_content'], _limits(limits))


def kinetic_basis(row, fields, limits=None):
    if row['derivatives'] != 2 or sum(p['count'] for p in row['factors']) != 2 or any(
            not p['name'].startswith('D(') or not p['name'].endswith(')') for p in row['factors']):
        raise ValueError('Joint scalar kinetic lowering requires exactly two once-differentiated scalar legs')
    stripped = dict(row, derivatives=0, factors=[dict(name=p['name'][2:-1], count=p['count']) for p in row['factors']])
    return invariant_basis(stripped, fields, limits)


def basis_labels(row, fields, limits=None):
    function = kinetic_basis if row['derivatives'] else invariant_basis
    return tuple(invariant.render() for invariant in function(row, fields, limits)[0])


def basis_fingerprint(row, fields, limits=None):
    payload = dict(version=BASIS_VERSION, expressions=basis_labels(row, fields, limits))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
