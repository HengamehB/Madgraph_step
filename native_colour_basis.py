"""Exact gauge-block scalar kinetic/mass resolution with native colour indices.

Only flavour coordinates are diagonalized. Colour indices stay in their
native irreducible representation, so all exported interactions can use the
same transformation without introducing component-colour UFO vertices.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ScalarState:
    label: str
    source: str
    rep: int
    charge: object
    real: bool
    name: str
    mass: float = 0.


def _coordinates(fields, reps):
    states, expressions = [], {}
    for f in fields:
        if f.spin != 'scalar':
            continue
        rep = reps[f.su3]
        if f.kind == 'real':
            index = len(states)
            states.append(ScalarState(f.name, f.name, abs(rep), f.hypercharge, True, f.name.lower()))
            expressions[f.name] = {((index, False)): 1.}
        elif rep in (1, 8) and f.hypercharge == 0:
            index = len(states)
            states.extend(ScalarState(f.name + str(i), f.name, rep, f.hypercharge, True,
                                      f.name.lower() + str(i)) for i in (1, 2))
            expressions[f.name] = {(index, False): 1 / math.sqrt(2), (index + 1, False): 1j / math.sqrt(2)}
        else:
            conjugate = rep < 0 or (rep in (1, 8) and f.hypercharge < 0)
            index = len(states)
            states.append(ScalarState(f.name, f.name, abs(rep), -f.hypercharge if conjugate else f.hypercharge,
                                      False, f.name.lower() + ('bar' if conjugate else '')))
            expressions[f.name] = {(index, conjugate): 1.}
        expressions[f.name + '†'] = {(i, False if states[i].real else not bar): complex(z).conjugate()
                                      for (i, bar), z in expressions[f.name].items()}
    return states, expressions


def _substitute(polynomial, expressions):
    contributions = defaultdict(list)
    for monomial, coefficient in polynomial.items():
        for terms in product(*(expressions[token].items() for token, _ in monomial)):
            key = tuple(sorted((index, bar, old[1]) for ((index, bar), _), old in zip(terms, monomial)))
            contributions[key].append(coefficient * math.prod(value for _, value in terms))
    result = {key: complex(math.fsum(z.real for z in values), math.fsum(z.imag for z in values))
              for key, values in contributions.items()}
    return {m: z for m, z in result.items() if z != 0}


def _relative_close(left, right, tolerance=1e-10):
    scale = max(float(np.max(abs(left), initial=0)), float(np.max(abs(right), initial=0)))
    return not scale or float(np.max(abs(left / scale - right / scale), initial=0)) <= tolerance


def _matrix(polynomial, indices, real, dimension):
    from feynrules_export import CompilerResolutionError, ModelResolutionError
    matrices = []
    for colour in range(dimension):
        matrix = np.zeros((len(indices), len(indices)), complex)
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                if real:
                    key = tuple(sorted(((i, False, colour), (j, False, colour))))
                    matrix[a, b] = polynomial.get(key, 0.) * (2 if i == j else 1)
                else:
                    key = tuple(sorted(((i, True, colour), (j, False, colour))))
                    matrix[a, b] = polynomial.get(key, 0.)
        if real and not _relative_close(matrix, matrix.real):
            raise ModelResolutionError('Scalar quadratic form is not real in real-field coordinates')
        matrix = matrix.real if real else matrix
        if not _relative_close(matrix, matrix.conj().T):
            raise ModelResolutionError('Scalar quadratic form is not Hermitian')
        matrices.append(matrix)
    if any(not _relative_close(matrix, matrices[0]) for matrix in matrices[1:]):
        raise CompilerResolutionError('Native scalar quadratic form unexpectedly splits colour components')
    return matrices[0]


def resolve_scalars(fields, reps, kinetic, quadratic):
    """Return physical states and raw-token -> physical-field linear maps.

    Input polynomials retain explicit colour component labels only for the
    numerical verification of the quadratic form. The returned transformation
    acts identically on every colour component.
    """
    from feynrules_export import CompilerResolutionError, ModelResolutionError
    states, expressions = _coordinates(fields, reps)
    kin, mass = _substitute(kinetic, expressions), _substitute(quadratic, expressions)
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.rep, state.charge, state.real].append(index)
    for polynomial in (kin, mass):
        for monomial, value in polynomial.items():
            if len(monomial) == 0:
                continue
            if len(monomial) == 1:
                raise ModelResolutionError('The supplied zero new-field VEV vacuum has a nonzero scalar tadpole')
            (i, ibar, ic), (j, jbar, jc) = monomial
            left, right = states[i], states[j]
            if (left.rep, left.charge, left.real) != (right.rep, right.charge, right.real) or ic != jc:
                raise CompilerResolutionError('Native scalar quadratic term mixes different gauge blocks or colour indices')
            if not left.real and ibar == jbar:
                raise CompilerResolutionError('Native scalar quadratic term violates the conserved gauge representation')
    transformation = np.zeros((len(states), len(states)), complex)
    physical = list(states)
    for (rep, charge, real), indices in groups.items():
        k = _matrix(kin, indices, real, rep)
        m = -_matrix(mass, indices, real, rep)
        try:
            diagonal = k.diagonal().real
            if np.any(diagonal <= 0):
                raise ValueError('kinetic matrix is not positive definite')
            scale = np.sqrt(diagonal)
            normalized_k = (k / scale[:, None]) / scale[None, :]
            normalized_m = (m / scale[:, None]) / scale[None, :]
            if np.linalg.eigvalsh(normalized_k).min() <= 0:
                raise ValueError('kinetic matrix is not positive definite')
            eigenvalues, normalized_rotation = eigh(normalized_m, normalized_k)
            rotation = normalized_rotation / scale[:, None]
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ModelResolutionError(f'Colour {rep}, charge {charge}: invalid scalar quadratic form: {exc}') from exc
        # A heavy spectator must not hide a tachyon in a light eigenmode.
        # Bound each Rayleigh quotient's cancellation error using the terms
        # contributing to that mode, in normalization-independent coordinates.
        uncertainty = np.array([1e-10 * np.sum(abs(
            normalized_rotation[:, j, None] * normalized_m * normalized_rotation[None, :, j]))
            for j in range(len(indices))])
        if np.any(eigenvalues < -uncertainty):
            raise ModelResolutionError(f'Colour {rep}, charge {charge}: tachyonic scalar spectrum')
        # Assign by largest overlap, then fix the arbitrary phase/sign. This
        # keeps original labels in the diagonal limit without changing rates.
        _, assignment = linear_sum_assignment(-abs(normalized_rotation))
        rotation, eigenvalues = rotation[:, assignment], eigenvalues[assignment]
        normalized_rotation = normalized_rotation[:, assignment]
        for column in range(len(indices)):
            pivot = int(np.argmax(abs(normalized_rotation[:, column])))
            rotation[:, column] /= rotation[pivot, column] / abs(rotation[pivot, column])
        if not np.allclose(rotation.conj().T @ k @ rotation, np.eye(len(indices)), atol=1e-9, rtol=1e-9):
            raise CompilerResolutionError('Native scalar canonical-kinetic residual failed')
        if not _relative_close(rotation.conj().T @ m @ rotation, np.diag(eigenvalues), tolerance=1e-9):
            raise CompilerResolutionError('Native scalar diagonal-mass residual failed')
        for a, i in enumerate(indices):
            state = states[i]
            physical[i] = ScalarState(state.label, state.source, rep, charge, real, state.name,
                                      math.sqrt(max(0., float(eigenvalues[a]))))
            for b, j in enumerate(indices):
                transformation[i, j] = rotation[a, b]
    resolved = {}
    for token, terms in expressions.items():
        mapping = defaultdict(complex)
        for (i, bar), coefficient in terms.items():
            for j in range(len(states)):
                value = transformation[i, j].conjugate() if bar else transformation[i, j]
                mapping[j, bar] += coefficient * value
        resolved[token] = {key: value for key, value in mapping.items() if value != 0}
    return tuple(physical), resolved, transformation
