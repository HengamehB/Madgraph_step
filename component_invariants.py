"""Numerical component polynomials for gauge/Lorentz invariant operators.

This optional backend supplements the exact character counter.  It constructs
actual invariant polynomials in a documented orthonormal component basis, using
NumPy/SciPy linear algebra.  SU(3) (p,q) is the traceless part of
Sym^p(3) x Sym^q(3bar), built from scratch in a normalized bosonic Fock basis.
No external symbolic algebra system or pre-existing operator code is used.

``component_invariants`` accepts one descriptor per distinct letter::

    {"name": "X", "lorentz": (0, 0), "weak": 1,
     "colour": (1, 0), "nf": 1, "derivative": False,
     "fermion": False, "count": 2}

Lorentz and weak labels are Dynkin labels (twice spin, or dimension minus one),
not dimensions.  ``derivative=True`` tensors one full (1,1) Lorentz vector onto
the supplied undifferentiated Lorentz representation.  Hypercharge must already
sum to zero; an optional ``hypercharge`` key enables a second check here.
Different derivative letters must have different names.

Returned coefficients are floating point, with a checked generator residual
and, if supplied, a checked exact singlet multiplicity.  The basis is before
integration by parts/Bianchi/EOM reduction.  It does not choose a Hermitian
basis or impose a component reality condition for real fields.  All indices in
the result are one-based; fermion components multiply in the stated order.
Explicit resource limits raise an exception rather than omit contractions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from math import comb, factorial, prod, sqrt
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg, sparse


class ComponentResourceLimit(RuntimeError):
    """Explicit construction exceeded its user-configurable resource budget."""
    category = 'resource_limit'


class InvariantVerificationError(RuntimeError):
    """Numerical construction failed an exact-count or residual check."""
    category = 'compiler_failure'


@dataclass(frozen=True)
class Limits:
    max_letter_components: int = 1200
    max_colour_ambient: int = 2500
    max_power_monomials: int = 200_000
    max_weight_combinations: int = 1_000_000
    max_zero_weight_states: int = 1800
    max_matrix_entries: int = 8_000_000
    max_matrix_nonzeros: int = 1_000_000
    max_output_terms: int = 100_000

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer resource limit')


@dataclass
class _Rep:
    weights: tuple[tuple[int, ...], ...]
    raising: tuple[sparse.csr_matrix, ...]
    labels: tuple[Any, ...]
    # For SU(3), columns give each weight-basis vector in the ambient Fock basis.
    embedding: sparse.csr_matrix | None = None
    ambient_labels: tuple[Any, ...] = ()

    @property
    def dimension(self):
        return len(self.weights)


def _nat(value, label):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _phase_fix(matrix: np.ndarray) -> np.ndarray:
    """Choose deterministic signs; degenerate SVD subspaces remain arbitrary."""
    matrix = matrix.copy()
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        if len(vector):
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0:
                matrix[:, column] *= -1
    return matrix


def _nullspace(matrix: np.ndarray, tolerance: float):
    rows, columns = matrix.shape
    if columns == 0:
        return np.empty((0, 0)), np.empty(0), 0
    if rows == 0:
        return np.eye(columns), np.empty(0), 0
    # scipy.linalg.null_space uses a full U even for tall matrices.  Avoid that
    # potentially enormous allocation: only a full V is needed here.
    _, singular, vh = linalg.svd(
        matrix, full_matrices=rows < columns, check_finite=True,
        lapack_driver="gesvd",
    )
    threshold = tolerance * max(1.0, float(singular[0]) if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    return _phase_fix(vh[rank:].T), singular, rank


@lru_cache(maxsize=64)
def _su2(n: int) -> _Rep:
    n = _nat(n, "SU(2) Dynkin label")
    rows = list(range(n))
    columns = list(range(1, n + 1))
    values = [sqrt(k * (n - k + 1)) for k in columns]
    raising = sparse.csr_matrix((values, (rows, columns)), shape=(n + 1, n + 1))
    return _Rep(
        tuple((n - 2 * k,) for k in range(n + 1)),
        (raising,),
        tuple({"twice_m": n - 2 * k} for k in range(n + 1)),
    )


def _occupations(n: int):
    return tuple((i, j, n - i - j) for i in range(n, -1, -1)
                 for j in range(n - i, -1, -1))


def _colour_weight(z, w):
    d = tuple(a - b for a, b in zip(z, w))
    return d[0] - d[1], d[1] - d[2]


@lru_cache(maxsize=64)
def _su3(p: int, q: int) -> _Rep:
    """Harmonic-tensor SU(3) irrep with orthonormal fixed-weight columns."""
    p, q = _nat(p, "SU(3) p"), _nat(q, "SU(3) q")
    ambient = tuple(product(_occupations(p), _occupations(q)))
    lookup = {state: index for index, state in enumerate(ambient)}
    by_weight = defaultdict(list)
    for index, (z, w) in enumerate(ambient):
        by_weight[_colour_weight(z, w)].append(index)
    target = tuple(product(_occupations(p - 1), _occupations(q - 1))) if p and q else ()
    target_by_weight = defaultdict(list)
    for index, (z, w) in enumerate(target):
        target_by_weight[_colour_weight(z, w)].append(index)
    embedding_rows, embedding_columns, embedding_values = [], [], []
    weights, labels = [], []
    for weight in sorted(by_weight, reverse=True):
        indices = by_weight[weight]
        lower = target_by_weight.get(weight, [])
        lower_lookup = {target[index]: row for row, index in enumerate(lower)}
        trace = np.zeros((len(lower), len(indices)))
        for column, ambient_index in enumerate(indices):
            z, w = ambient[ambient_index]
            for i in range(3):
                if z[i] and w[i]:
                    new_z, new_w = list(z), list(w)
                    new_z[i] -= 1
                    new_w[i] -= 1
                    trace[lower_lookup[(tuple(new_z), tuple(new_w))], column] += sqrt(z[i] * w[i])
        kernel, _, _ = _nullspace(trace, 1e-12)
        for local_column in range(kernel.shape[1]):
            column = len(weights)
            weights.append(weight)
            labels.append({"weight": weight, "degeneracy": local_column + 1})
            for row, ambient_index in enumerate(indices):
                coefficient = float(kernel[row, local_column])
                if abs(coefficient) > 1e-14:
                    embedding_rows.append(ambient_index)
                    embedding_columns.append(column)
                    embedding_values.append(coefficient)
    expected_dimension = (p + 1) * (q + 1) * (p + q + 2) // 2
    if len(weights) != expected_dimension:
        raise InvariantVerificationError("SU(3) trace-free tensor dimension check failed")
    embedding = sparse.csr_matrix(
        (embedding_values, (embedding_rows, embedding_columns)),
        shape=(len(ambient), len(weights)),
    )
    generators = []
    for i, j in ((0, 1), (1, 2)):
        rows, columns, values = [], [], []
        for column, (z, w) in enumerate(ambient):
            if z[j]:
                new_z = list(z)
                new_z[j] -= 1
                new_z[i] += 1
                rows.append(lookup[(tuple(new_z), w)])
                columns.append(column)
                values.append(sqrt(z[j] * (z[i] + 1)))
            if w[i]:
                new_w = list(w)
                new_w[i] -= 1
                new_w[j] += 1
                rows.append(lookup[(z, tuple(new_w))])
                columns.append(column)
                values.append(-sqrt(w[i] * (w[j] + 1)))
        ambient_generator = sparse.csr_matrix(
            (values, (rows, columns)), shape=(len(ambient), len(ambient)),
        )
        generator = (embedding.T @ ambient_generator @ embedding).tocsr()
        generator.data[np.abs(generator.data) < 1e-13] = 0.0
        generator.eliminate_zeros()
        # The harmonic subspace must be preserved by SU(3), not only have the
        # right dimension.  Check the residual in the full tensor realization.
        residue = ambient_generator @ embedding - embedding @ generator
        if residue.nnz and max(abs(residue.data)) > 1e-10:
            raise InvariantVerificationError("SU(3) harmonic-subspace generator check failed")
        generators.append(generator)
    return _Rep(tuple(weights), tuple(generators), tuple(labels), embedding, ambient)


def _factor_operator(operator, factor, dimensions):
    left = prod(dimensions[:factor])
    right = prod(dimensions[factor + 1:])
    return sparse.kron(
        sparse.eye(left, format="csr"),
        sparse.kron(operator, sparse.eye(right, format="csr"), format="csr"),
        format="csr",
    )


def _letter_rep(descriptor, limits):
    lorentz = tuple(_nat(x, "Lorentz Dynkin label") for x in descriptor.get("lorentz", (0, 0)))
    colour = tuple(_nat(x, "Colour Dynkin label") for x in descriptor.get("colour", (0, 0)))
    if len(lorentz) != 2 or len(colour) != 2:
        raise ValueError("lorentz and colour must each contain two Dynkin labels")
    weak = _nat(descriptor.get("weak", 0), "Weak Dynkin label")
    nf = _nat(descriptor.get("nf", 1), "Number of flavours")
    if nf == 0:
        raise ValueError("nf must be positive")
    ambient = comb(colour[0] + 2, 2) * comb(colour[1] + 2, 2)
    if ambient > limits.max_colour_ambient:
        raise ComponentResourceLimit(
            f"SU(3){colour} needs {ambient:,} ambient tensor components; "
            f"limit is {limits.max_colour_ambient:,}. Increase Limits.max_colour_ambient explicitly."
        )
    colour_dimension = (colour[0] + 1) * (colour[1] + 1) * (sum(colour) + 2) // 2
    dimension = prod(n + 1 for n in lorentz) * (weak + 1) * colour_dimension * nf
    if descriptor.get("derivative", False):
        dimension *= 4
    if dimension > limits.max_letter_components:
        raise ComponentResourceLimit(
            f"{descriptor['name']} has {dimension:,} components; "
            f"limit is {limits.max_letter_components:,}. Increase Limits.max_letter_components explicitly."
        )
    factors = [_su2(lorentz[0]), _su2(lorentz[1]), _su2(weak), _su3(*colour)]
    keys = ["lorentz_left", "lorentz_right", "weak", "colour"]
    rank_slots = [(0,), (1,), (2,), (3, 4)]
    factors.append(_Rep(tuple((0,) for _ in range(nf)), (), tuple(range(1, nf + 1))))
    keys.append("flavour")
    rank_slots.append(())
    if descriptor.get("derivative", False):
        factors += [_su2(1), _su2(1)]
        keys += ["derivative_left", "derivative_right"]
        rank_slots += [(0,), (1,)]
    dimensions = [factor.dimension for factor in factors]
    weights, labels = [], []
    for indices in product(*(range(d) for d in dimensions)):
        weight = [0] * 5
        for factor, slots, component in zip(factors, rank_slots, indices):
            for coordinate, slot in enumerate(slots):
                weight[slot] += factor.weights[component][coordinate]
        weights.append(tuple(weight))
        labels.append({key: component + 1 for key, component in zip(keys, indices)})
    generators = [sparse.csr_matrix((dimension, dimension)) for _ in range(5)]
    for index, (factor, slots) in enumerate(zip(factors, rank_slots)):
        for local_root, slot in enumerate(slots):
            generators[slot] += _factor_operator(factor.raising[local_root], index, dimensions)
    return _Rep(tuple(weights), tuple(g.tocsr() for g in generators), tuple(labels)), colour


def _power_states(rep, count, fermion, name, limits):
    dimension = rep.dimension
    size = comb(dimension, count) if fermion else comb(dimension + count - 1, count)
    if size > limits.max_power_monomials:
        raise ComponentResourceLimit(
            f"{name}^{count} requires {size:,} component monomials; "
            f"limit is {limits.max_power_monomials:,}. Increase Limits.max_power_monomials explicitly."
        )
    iterator = combinations(range(dimension), count) if fermion else combinations_with_replacement(range(dimension), count)
    by_weight = defaultdict(list)
    for indices in iterator:
        weight = tuple(sum(rep.weights[index][axis] for index in indices) for axis in range(5))
        by_weight[weight].append(indices)
    return dict(by_weight)


def _zero_weight_states(groups, limits):
    """Join weight buckets, avoiding the much larger full component product."""
    if not groups:
        return [()]
    # Put the largest weight dictionary last; the final weight is looked up.
    order = sorted(range(len(groups)), key=lambda i: len(groups[i]))
    ordered = [groups[i] for i in order]
    weight_combinations = prod(len(group) for group in ordered[:-1])
    if weight_combinations > limits.max_weight_combinations:
        raise ComponentResourceLimit(
            f"Weight matching requires {weight_combinations:,} combinations; "
            f"limit is {limits.max_weight_combinations:,}. Increase Limits.max_weight_combinations explicitly."
        )
    result = []
    for prefix_weights in product(*(group.keys() for group in ordered[:-1])):
        needed = tuple(-sum(weight[axis] for weight in prefix_weights) for axis in range(5))
        tail = ordered[-1].get(needed)
        if tail is None:
            continue
        buckets = [group[weight] for group, weight in zip(ordered[:-1], prefix_weights)] + [tail]
        size = prod(len(bucket) for bucket in buckets)
        if len(result) + size > limits.max_zero_weight_states:
            raise ComponentResourceLimit(
                f"The operator has more than {limits.max_zero_weight_states:,} zero-weight component monomials. "
                "Increase Limits.max_zero_weight_states and, if necessary, the matrix limits explicitly."
            )
        for combined in product(*buckets):
            original = [None] * len(groups)
            for ordered_index, original_index in enumerate(order):
                original[original_index] = combined[ordered_index]
            result.append(tuple(original))
    return result


def _transitions(rep):
    result = []
    for generator in rep.raising:
        matrix = generator.tocsc()
        result.append(tuple(
            tuple((int(matrix.indices[k]), float(matrix.data[k]))
                  for k in range(matrix.indptr[column], matrix.indptr[column + 1]))
            for column in range(rep.dimension)
        ))
    return tuple(result)


def _raise_monomial(indices, transitions, fermion):
    if fermion:
        occupied = set(indices)
        for removed, source in enumerate(indices):
            remainder = indices[:removed] + indices[removed + 1:]
            for target, value in transitions[source]:
                if target in occupied and target != source:
                    continue
                inserted = sum(index < target for index in remainder)
                sign = -1 if (removed + inserted) % 2 else 1
                new_indices = tuple(sorted(remainder + (target,)))
                yield new_indices, sign * value
    else:
        occupations = Counter(indices)
        for source, population in occupations.items():
            for target, value in transitions[source]:
                if source == target:
                    coefficient = population * value
                else:
                    coefficient = sqrt(population * (occupations.get(target, 0) + 1)) * value
                new_indices = list(indices)
                new_indices.remove(source)
                new_indices.append(target)
                yield tuple(sorted(new_indices)), coefficient


def _constraint_matrix(states, reps, descriptors, limits):
    target_rows = {}
    rows, columns, values = [], [], []
    transition_lists = [_transitions(rep) for rep in reps]
    for column, combined in enumerate(states):
        for letter_index, indices in enumerate(combined):
            fermion = bool(descriptors[letter_index].get("fermion", False))
            for root in range(5):
                transitions = transition_lists[letter_index][root]
                for changed, coefficient in _raise_monomial(indices, transitions, fermion):
                    target = combined[:letter_index] + (changed,) + combined[letter_index + 1:]
                    row = target_rows.setdefault((root, target), len(target_rows))
                    rows.append(row)
                    columns.append(column)
                    values.append(coefficient)
                    if len(values) > limits.max_matrix_nonzeros:
                        raise ComponentResourceLimit(
                            f"Generator matrix exceeded {limits.max_matrix_nonzeros:,} entries. "
                            "Increase Limits.max_matrix_nonzeros explicitly."
                        )
    shape = (len(target_rows), len(states))
    # The SVD also needs an n x n right-singular-vector matrix.
    entries = max(prod(shape), shape[1] ** 2)
    if entries > limits.max_matrix_entries:
        raise ComponentResourceLimit(
            f"Generator/SVD matrix needs up to {entries:,} dense entries for shape {shape}; "
            f"limit is {limits.max_matrix_entries:,}. Increase Limits.max_matrix_entries explicitly."
        )
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()
    matrix.sum_duplicates()
    return matrix


def _colour_legend(colour):
    rep = _su3(*colour)
    embedding = rep.embedding.tocsc()
    basis = []
    for column, weight in enumerate(rep.weights):
        terms = []
        for k in range(embedding.indptr[column], embedding.indptr[column + 1]):
            z, w = rep.ambient_labels[embedding.indices[k]]
            terms.append({"coefficient": float(embedding.data[k]), "z_occupations": list(z), "w_occupations": list(w)})
        basis.append({"index": column + 1, "weight": list(weight), "terms": terms})
    return {"dynkin": list(colour), "basis": basis,
            "convention": "Each ambient ket is z1^n1 z2^n2 z3^n3 w1^m1 w2^m2 w3^m3 / sqrt(product(n_i! m_i!)); w transforms in 3bar."}


@lru_cache(maxsize=64)
def _su2_conjugation(n):
    """C maps dual components into the canonical SU(2) highest-weight basis."""
    n = _nat(n, "SU(2) Dynkin label")
    indices = np.arange(n + 1)
    return sparse.csr_matrix(
        ((-1.0) ** indices, (indices, n - indices)), shape=(n + 1, n + 1),
    )


@lru_cache(maxsize=64)
def _su3_conjugation(p, q):
    """C_(p,q): conjugate (p,q) components -> canonical (q,p) components."""
    original, conjugate = _su3(p, q), _su3(q, p)
    target = {label: index for index, label in enumerate(conjugate.ambient_labels)}
    swap = sparse.csr_matrix(
        (np.ones(len(original.ambient_labels)),
         ([target[(w, z)] for z, w in original.ambient_labels],
          np.arange(len(original.ambient_labels)))),
        shape=(len(conjugate.ambient_labels), len(original.ambient_labels)),
    )
    matrix = (conjugate.embedding.T @ swap @ original.embedding).tocsr()
    matrix.data[np.abs(matrix.data) < 1e-13] = 0
    matrix.eliminate_zeros()
    identity_error = matrix.T @ matrix - sparse.eye(original.dimension, format="csr")
    if identity_error.nnz and max(abs(identity_error.data)) > 1e-10:
        raise InvariantVerificationError("SU(3) gauge-conjugation matrix is not orthogonal")
    for old_generator, new_generator in zip(original.raising, conjugate.raising):
        residue = new_generator @ matrix + matrix @ old_generator.T
        if residue.nnz and max(abs(residue.data)) > 1e-10:
            raise InvariantVerificationError("SU(3) gauge-conjugation intertwiner check failed")
    return matrix


def _sparse_matrix_data(matrix):
    coo = matrix.tocoo()
    return {"shape": list(matrix.shape), "entries": [
        {"row": int(row) + 1, "column": int(column) + 1, "value": float(value)}
        for row, column, value in zip(coo.row, coo.col, coo.data)
    ]}


def gauge_conjugation(weak: int, colour: tuple[int, int]) -> dict[str, Any]:
    """Return explicit gauge charge-conjugation intertwiners as sparse data.

    Rows index the canonical conjugate representation, columns the original.
    For canonical gauge components, ``barX = (Cweak tensor Ccolour) X*``.
    This is a *forward* convention: use the inverse matrix for the return map.
    SU(2)'s same canonical C obeys C C* = (-1)^weak, reflecting pseudoreality.
    Lorentz Hermitian conjugation and Grassmann factor reversal are separate.
    """
    weak = _nat(weak, "Weak Dynkin label")
    p, q = (_nat(x, "Colour Dynkin label") for x in colour)
    return {
        "weak_dynkin": weak,
        "colour_dynkin": [p, q],
        "conjugate_colour_dynkin": [q, p],
        "weak_matrix": _sparse_matrix_data(_su2_conjugation(weak)),
        "colour_matrix": _sparse_matrix_data(_su3_conjugation(p, q)),
        "forward_relation": "barX[a_bar,c_bar] = sum_(a,c) Cweak[a_bar,a] Ccolour[c_bar,c] conjugate(X[a,c]); flavour is unchanged.",
        "reverse_relation": "Use the inverse intertwiners to return from barX to X. Reapplying the same SU(2) C gives (-1)^weak, not +1 for pseudoreal representations.",
        "su2_square": (-1) ** weak,
        "colour_square_relation": "C_(q,p) conjugate(C_(p,q)) = identity.",
        "real_field_relation": "For a self-conjugate gauge representation with even weak Dynkin label, one possible reality convention is X=(Cweak tensor Ccolour) conjugate(X). A field-phase convention can change the overall phase of this real structure.",
        "scope": "Gauge indices only. Lorentz Hermitian conjugation exchanges left/right slots (and derivative left/right slots); its spinor/Cartesian convention and reversal of fermion order must also be applied. These matrices alone do not make an arbitrary component polynomial Hermitian.",
    }


def _expression(terms, digits=10):
    pieces = []
    for term in terms:
        coefficient = term["coefficient"]
        factors = []
        for factor in term["factors"]:
            factors.extend(f"{factor['letter']}[{component}]" for component in factor["components"])
        monomial = " ".join(factors) or "1"
        magnitude = abs(coefficient)
        prefactor = "" if abs(magnitude - 1) < 10 ** (-digits) else f"{magnitude:.{digits}g} "
        pieces.append(("- " if coefficient < 0 else "+ ") + prefactor + monomial)
    if not pieces:
        return "0"
    joined = " ".join(pieces)
    return joined[2:] if pieces[0].startswith("+ ") else "-" + joined[2:]


def component_invariants(
    descriptors: Sequence[Mapping[str, Any]],
    expected_multiplicity: int | None = None,
    *,
    limits: Limits | None = None,
    tolerance: float = 1e-10,
    coefficient_cutoff: float = 1e-11,
) -> dict[str, Any]:
    """Construct and verify an explicit invariant-polynomial basis.

    A basis term is ``coefficient * product(letter[component])`` in descriptor
    order and increasing component order within each letter.  Thus fermionic
    signs are unambiguous.  ``components`` and ``colour_bases`` fully define the
    component convention.  Bosonic Fock normalizations have already been
    converted into ordinary polynomial coefficients in ``basis[*]['terms']``.

    Supply the independent exact character count whenever available.  A
    mismatch or numerical residual raises InvariantVerificationError and
    returns no unchecked basis.  Resource limits affect only this optional
    explicit expansion, never the exact operator catalog.
    """
    limits = limits or Limits()
    if not 0 < tolerance < 1e-3 or not 0 <= coefficient_cutoff < tolerance:
        raise ValueError("Use 0 <= coefficient_cutoff < tolerance < 1e-3")
    descriptors = [dict(item) for item in descriptors]
    if not descriptors:
        raise ValueError("Provide at least one field letter")
    names = [item.get("name") for item in descriptors]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Letter names must be nonempty and distinct")
    for descriptor in descriptors:
        descriptor["count"] = _nat(descriptor.get("count", 1), "Letter count")
        if not descriptor["count"]:
            raise ValueError("Letter counts must be positive")
    if expected_multiplicity is not None:
        expected_multiplicity = _nat(expected_multiplicity, "Expected multiplicity")
    if all("hypercharge" in item for item in descriptors):
        charge = sum(Fraction(str(item["hypercharge"])) * item["count"] for item in descriptors)
        if charge:
            raise ValueError(f"The supplied monomial has nonzero hypercharge {charge}")
    reps, colours = [], []
    for descriptor in descriptors:
        rep, colour = _letter_rep(descriptor, limits)
        reps.append(rep)
        colours.append(colour)
    groups = [_power_states(rep, item["count"], bool(item.get("fermion", False)), item["name"], limits)
              for rep, item in zip(reps, descriptors)]
    states = _zero_weight_states(groups, limits)
    matrix = _constraint_matrix(states, reps, descriptors, limits)
    kernel, singular, rank = _nullspace(matrix.toarray(), tolerance)
    multiplicity = kernel.shape[1]
    if expected_multiplicity is not None and multiplicity != expected_multiplicity:
        raise InvariantVerificationError(
            f"Numerical singlet multiplicity {multiplicity} disagrees with exact character count "
            f"{expected_multiplicity}; no component basis is accepted."
        )
    output_size = len(states) * multiplicity
    if output_size > limits.max_output_terms:
        raise ComponentResourceLimit(
            f"Basis expansion may contain {output_size:,} terms; limit is {limits.max_output_terms:,}. "
            "Increase Limits.max_output_terms explicitly."
        )
    scale = max(1.0, float(singular[0]) if singular.size else 0.0)
    basis, max_residual = [], 0.0
    for index in range(multiplicity):
        vector = kernel[:, index].copy()
        vector[np.abs(vector) < coefficient_cutoff] = 0
        residual = float(np.max(np.abs(matrix @ vector))) / scale if matrix.shape[0] else 0.0
        if residual > 10 * tolerance:
            raise InvariantVerificationError(f"Component coefficient residual {residual:g} exceeds tolerance")
        max_residual = max(max_residual, residual)
        coefficients = []
        for amplitude, state in zip(vector, states):
            normalization = 1.0
            for indices, descriptor in zip(state, descriptors):
                if not descriptor.get("fermion", False):
                    normalization *= sqrt(prod(factorial(n) for n in Counter(indices).values()))
            coefficients.append(amplitude / normalization)
        normalization = max((abs(x) for x in coefficients), default=1.0) or 1.0
        terms = []
        for coefficient, state in zip(coefficients, states):
            coefficient = float(coefficient / normalization)
            if not coefficient:
                continue
            terms.append({"coefficient": coefficient,
                          "factors": [{"letter": descriptor["name"], "components": [i + 1 for i in indices]}
                                      for descriptor, indices in zip(descriptors, state)]})
        basis.append({"index": index + 1, "terms": terms, "expression": _expression(terms),
                      "relative_generator_residual": residual})
    return {
        "basis": basis,
        "multiplicity": multiplicity,
        "expected_multiplicity": expected_multiplicity,
        "verified": True,
        "components": {descriptor["name"]: [
            {"index": index + 1, "weight": list(weight), "indices": dict(label)}
            for index, (weight, label) in enumerate(zip(rep.weights, rep.labels))
        ] for descriptor, rep in zip(descriptors, reps)},
        "colour_bases": {f"{p},{q}": _colour_legend((p, q))
                         for p, q in sorted(set(colours) | {(q, p) for p, q in colours})},
        "gauge_conjugation": {
            descriptor["name"]: gauge_conjugation(descriptor.get("weak", 0), colour)
            for descriptor, colour in zip(descriptors, colours)
        },
        "metadata": {
            "arithmetic": "NumPy/SciPy float64; numerical component coefficients",
            "tolerance": tolerance,
            "coefficient_cutoff": coefficient_cutoff,
            "max_relative_generator_residual": max_residual,
            "zero_weight_monomials": len(states),
            "constraint_shape": list(matrix.shape),
            "constraint_rank": rank,
            "exact_multiplicity_checked": expected_multiplicity is not None,
            "hypercharge_checked": all("hypercharge" in item for item in descriptors),
            "basis_scope": "Local singlets before integration by parts, Bianchi identities, equations of motion, or Hermitian-basis selection.",
            "component_order": "Lorentz-left, Lorentz-right, weak, colour, flavour, then derivative-left/right if present; all indices one-based.",
            "su2_convention": "For Dynkin label n, component k=1..n+1 has weight n-2(k-1). Raising maps k to k-1 with coefficient sqrt((k-1)(n-k+2)).",
            "conjugate_basis_convention": "Every letter uses its own canonical highest-weight basis. Dagger-letter components are not assumed to equal componentwise conjugates with the same index. Explicit gauge intertwiners are supplied in gauge_conjugation; they do not constitute a full Lorentz/Grassmann Hermitian-conjugation map or choose a Hermitian operator basis.",
            "fermion_order": list(names),
            "normalization": "Each displayed polynomial has maximum absolute coefficient 1; independent basis rotations and overall scales are conventional.",
        },
    }


# A descriptive alias is convenient in notebooks.
explicit_invariants = component_invariants
