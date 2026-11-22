"""Exact, cached character algebra for SU(2) and SU(3).

No floating point arithmetic or external representation-theory package is used.
Irreducible representations are specified by Dynkin labels: SU(2) ``n`` has
dimension ``n + 1``; SU(3) ``(p, q)`` has dimension
``(p + 1) * (q + 1) * (p + q + 2) // 2``.  Thus 3=(1,0), 3bar=(0,1),
6=(2,0), and 8=(1,1).  Weights are tuples in the fundamental-weight basis.

A character is a ``Counter`` mapping weight tuples to integer multiplicities.
SU(2) characters use 1-tuples and SU(3) characters use 2-tuples.  The standard
``field_character`` ordering is (Lorentz left, Lorentz right, weak, colour p,
colour q).  Lorentz labels refer to the complexified Lorentz algebra
sl(2,C) + sl(2,C), whose finite-dimensional character rules are those of
SU(2) x SU(2).  Hypercharge is tracked separately by the operator enumerator.

Public functions return fresh Counters, so callers cannot corrupt caches.
Symmetric/exterior powers use Newton's identities with Adams operations;
this includes repeated weights and hence enforces identical-field statistics.
Singlets are extracted by the Weyl-denominator constant-term formula.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from numbers import Integral
from typing import Mapping, Sequence

Weight = tuple[int, ...]
Character = Counter[Weight]
FrozenCharacter = tuple[tuple[Weight, int], ...]
DEFAULT_GROUPS = ("SU2", "SU2", "SU2", "SU3")


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


def _freeze(character: Mapping[Weight, int]) -> FrozenCharacter:
    """Validate and canonicalize a character for the memoization keys."""
    result = []
    rank = None
    for weight, multiplicity in character.items():
        if not isinstance(weight, tuple) or not all(
            isinstance(x, Integral) for x in weight
        ):
            raise ValueError("Character weights must be tuples of integers")
        if rank is None:
            rank = len(weight)
        elif rank != len(weight):
            raise ValueError("Character weights must all have the same rank")
        if not isinstance(multiplicity, Integral):
            raise ValueError("Weight multiplicities must be integers")
        if multiplicity:
            result.append((tuple(int(x) for x in weight), int(multiplicity)))
    return tuple(sorted(result))


def _thaw(character: FrozenCharacter) -> Character:
    return Counter(dict(character))


def _rank(character: FrozenCharacter) -> int:
    return len(character[0][0]) if character else 0


def su2_dimension(n: int) -> int:
    return _nonnegative_integer(n, "SU(2) Dynkin label") + 1


def su3_dimension(p: int, q: int) -> int:
    p = _nonnegative_integer(p, "SU(3) p")
    q = _nonnegative_integer(q, "SU(3) q")
    return (p + 1) * (q + 1) * (p + q + 2) // 2


@lru_cache(maxsize=256)
def _su2_character(n: int) -> FrozenCharacter:
    return tuple(((w,), 1) for w in range(-n, n + 1, 2))


def su2_character(n: int) -> Character:
    """Return the SU(2) irrep of highest weight ``n`` (spin n/2)."""
    return _thaw(_su2_character(_nonnegative_integer(n, "SU(2) Dynkin label")))


@lru_cache(maxsize=256)
def _su3_character(p: int, q: int) -> FrozenCharacter:
    # Integral Gelfand--Tsetlin patterns for the U(3) partition (p+q,q,0):
    #             p+q     q     0
    #                  a     b
    #                     c
    # Interlacing: p+q >= a >= q >= b >= 0 and a >= c >= b.
    # Their GL(3) weights are (c, a+b-c, p+2q-a-b).  Differences give
    # the two SU(3) Dynkin coordinates, preserving all weight degeneracies.
    result: Character = Counter()
    for a in range(q, p + q + 1):
        for b in range(q + 1):
            for c in range(b, a + 1):
                result[(2 * c - a - b, 2 * (a + b) - c - p - 2 * q)] += 1
    return _freeze(result)


def su3_character(p: int, q: int) -> Character:
    """Return an exact SU(3) irrep character, including weight multiplicities."""
    return _thaw(
        _su3_character(
            _nonnegative_integer(p, "SU(3) p"),
            _nonnegative_integer(q, "SU(3) q"),
        )
    )


def character_dimension(character: Mapping[Weight, int]) -> int:
    """Total representation dimension (sum of weight multiplicities)."""
    return sum(character.values())


@lru_cache(maxsize=4096)
def _tensor_two(left: FrozenCharacter, right: FrozenCharacter) -> FrozenCharacter:
    if not left or not right:
        return ()
    if _rank(left) != _rank(right):
        raise ValueError("Tensor factors must use the same weight-space rank")
    result: Character = Counter()
    for w_left, m_left in left:
        for w_right, m_right in right:
            result[tuple(a + b for a, b in zip(w_left, w_right))] += m_left * m_right
    return _freeze(result)


def tensor_product(*characters: Mapping[Weight, int]) -> Character:
    """Tensor representations of the *same* group by weight convolution.

    For a product of different groups use ``direct_product`` instead.
    At least one factor is required so the weight-space rank is unambiguous.
    """
    if not characters:
        raise ValueError("tensor_product needs at least one character")
    result = _freeze(characters[0])
    for character in characters[1:]:
        result = _tensor_two(result, _freeze(character))
    return _thaw(result)


@lru_cache(maxsize=1024)
def _direct_product(characters: tuple[FrozenCharacter, ...]) -> FrozenCharacter:
    result: FrozenCharacter = (((), 1),)
    for character in characters:
        result = tuple(
            (left + right, ml * mr)
            for left, ml in result
            for right, mr in character
        )
    return tuple(sorted(result))


def direct_product(*characters: Mapping[Weight, int]) -> Character:
    """Outer product for independent groups; concatenate their weight tuples."""
    return _thaw(_direct_product(tuple(_freeze(c) for c in characters)))


@lru_cache(maxsize=1024)
def _field_character(lorentz: tuple[int, int], weak: int, colour: tuple[int, int]):
    return _direct_product(
        (
            _su2_character(lorentz[0]),
            _su2_character(lorentz[1]),
            _su2_character(weak),
            _su3_character(*colour),
        )
    )


def field_character(
    lorentz: tuple[int, int] = (0, 0),
    weak: int = 0,
    colour: tuple[int, int] = (0, 0),
) -> Character:
    """Character for a field's Lorentz x SU(2)-weak x SU(3)-colour irrep.

    Examples: a scalar has lorentz=(0,0); a left Weyl fermion (1,0);
    its Hermitian conjugate (0,1); a Lorentz vector (1,1).
    """
    if len(lorentz) != 2 or len(colour) != 2:
        raise ValueError("lorentz and colour must each contain two Dynkin labels")
    lorentz = tuple(_nonnegative_integer(n, "Lorentz Dynkin label") for n in lorentz)
    colour = tuple(_nonnegative_integer(n, "Colour Dynkin label") for n in colour)
    weak = _nonnegative_integer(weak, "Weak Dynkin label")
    return _thaw(_field_character(lorentz, weak, colour))


def conjugate_character(character: Mapping[Weight, int]) -> Character:
    """Dual character, with every weight negated.

    For Hermitian conjugation of a Lorentz representation, ALSO interchange
    Lorentz left and right: use ``field_character`` with swapped labels.
    """
    return Counter({tuple(-x for x in w): m for w, m in _freeze(character)})


@lru_cache(maxsize=2048)
def _power(character: FrozenCharacter, k: int, exterior: bool) -> FrozenCharacter:
    rank = _rank(character)
    if k == 0:
        return (((0,) * rank, 1),)
    if k == 1:
        return character
    if not character or (exterior and k > sum(m for _, m in character)):
        return ()
    # Newton: k h_k = sum_{i=1}^k psi_i(chi) h_{k-i}; for exterior
    # powers insert (-1)^(i-1). psi_i multiplies each weight by i, leaving
    # its multiplicity unchanged. All divisions are exact over Z.
    numerator: Character = Counter()
    for i in range(1, k + 1):
        adams = tuple((tuple(i * x for x in w), m) for w, m in character)
        term = _tensor_two(adams, _power(character, k - i, exterior))
        sign = -1 if exterior and i % 2 == 0 else 1
        for w, m in term:
            numerator[w] += sign * m
    if any(m % k for m in numerator.values()):
        raise ArithmeticError("Nonintegral character from Newton identity")
    return tuple(sorted((w, m // k) for w, m in numerator.items() if m))


def symmetric_power(character: Mapping[Weight, int], k: int) -> Character:
    """Character of Sym^k(V), including identical-boson statistics."""
    frozen = _freeze(character)
    if any(m < 0 for _, m in frozen):
        raise ValueError("Symmetric powers require nonnegative multiplicities")
    return _thaw(_power(frozen, _nonnegative_integer(k, "Power"), False))


def exterior_power(character: Mapping[Weight, int], k: int) -> Character:
    """Character of wedge^k(V), including identical-fermion statistics."""
    frozen = _freeze(character)
    if any(m < 0 for _, m in frozen):
        raise ValueError("Exterior powers require nonnegative multiplicities")
    return _thaw(_power(frozen, _nonnegative_integer(k, "Power"), True))


# Short aliases convenient in notebooks.
sym_power = symmetric_power
wedge_power = exterior_power


def _normalized_groups(groups: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(g.upper().replace("(", "").replace(")", "") for g in groups)
    if any(g not in ("SU2", "SU3") for g in normalized):
        raise ValueError("Only SU2 and SU3 factors are supported")
    return normalized


@lru_cache(maxsize=64)
def _denominator(groups: tuple[str, ...]) -> FrozenCharacter:
    # det Weyl denominator: product over positive roots (1 - e^{-alpha}).
    # We return weights with the opposite sign, i.e. coefficient probes of
    # chi at sum(alpha), to compute [e^0] chi * product(1-e^{-alpha}).
    factors = []
    for group in groups:
        if group == "SU2":
            factors.append((((0,), 1), ((2,), -1)))
        else:
            result: FrozenCharacter = (((0, 0), 1),)
            for root in ((2, -1), (-1, 2), (1, 1)):
                result = _tensor_two(result, (((0, 0), 1), (root, -1)))
            factors.append(result)
    return _direct_product(tuple(factors))


def singlet_multiplicity(
    character: Mapping[Weight, int], groups: Sequence[str] = DEFAULT_GROUPS
) -> int:
    """Multiplicity of the trivial representation of all specified groups.

    Use groups=("SU2",) for rank-one SU(2), groups=("SU3",) for SU(3),
    or the default for ``field_character``.  This exact Weyl-denominator
    projection is much cheaper than fully decomposing a product character.
    It requires a genuine group character, rather than an arbitrary set of
    weights; no Weyl-group-size division is needed with this convention.
    """
    groups = _normalized_groups(groups)
    rank = sum(1 if g == "SU2" else 2 for g in groups)
    if character and any(len(w) != rank for w in character):
        raise ValueError(f"Expected weight rank {rank} for groups {groups!r}")
    return sum(character.get(w, 0) * sign for w, sign in _denominator(groups))


@lru_cache(maxsize=4096)
def _singlet_product(
    characters: tuple[FrozenCharacter, ...], groups: tuple[str, ...]
) -> int:
    if any(not character for character in characters):
        return 0
    if not characters:
        return 1
    if len(characters) == 1:
        character = dict(characters[0])
        return sum(character.get(w, 0) * sign for w, sign in _denominator(groups))
    # Do not form the final, usually largest, tensor character. Convolve the
    # smallest factors first, then obtain only the Weyl projection's needed
    # coefficients by dictionary lookup between the two remaining factors.
    factors = list(characters)
    while len(factors) > 2:
        factors.sort(key=len)
        factors.append(_tensor_two(factors.pop(0), factors.pop(0)))
    small, large = sorted(factors, key=len)
    large = dict(large)
    result = 0
    for target, sign in _denominator(groups):
        coefficient = 0
        for weight, multiplicity in small:
            complementary = tuple(t - w for t, w in zip(target, weight))
            coefficient += multiplicity * large.get(complementary, 0)
        result += sign * coefficient
    return result


def singlet_product_multiplicity(
    characters: Sequence[Mapping[Weight, int]], groups: Sequence[str] = DEFAULT_GROUPS
) -> int:
    """Project a list of tensor factors without constructing its final product.

    Prefer this to ``singlet_multiplicity(tensor_product(*characters))`` in
    operator scans.  It uses exact cached partial convolutions and coefficient
    lookups.  For identical particles, pass the appropriate symmetric or
    exterior power as ONE factor, not independent copies of the field.
    The empty product has one singlet; a zero-dimensional factor gives zero.
    """
    groups = _normalized_groups(groups)
    rank = sum(1 if g == "SU2" else 2 for g in groups)
    frozen = tuple(sorted(_freeze(c) for c in characters))
    if any(c and _rank(c) != rank for c in frozen):
        raise ValueError(f"Expected weight rank {rank} for groups {groups!r}")
    return _singlet_product(frozen, groups)


@lru_cache(maxsize=1024)
def _decompose_character(
    character: FrozenCharacter, groups: tuple[str, ...]
) -> tuple[tuple[Weight, int], ...]:
    remainder = _thaw(character)
    result: Character = Counter()
    while remainder:
        candidates = [w for w, m in remainder.items() if m > 0 and min(w, default=0) >= 0]
        if not candidates:
            raise ValueError("Input is not a nonnegative representation character")
        highest = max(candidates, key=lambda w: (sum(w), w))
        multiplicity = remainder[highest]
        result[highest] += multiplicity
        factors = []
        offset = 0
        for group in groups:
            if group == "SU2":
                factors.append(_su2_character(highest[offset]))
                offset += 1
            else:
                factors.append(_su3_character(*highest[offset:offset + 2]))
                offset += 2
        for weight, m in _direct_product(tuple(factors)):
            remainder[weight] -= multiplicity * m
            if remainder[weight] < 0:
                raise ValueError("Input is not a valid representation character")
            if not remainder[weight]:
                del remainder[weight]
    return _freeze(result)


def decompose_character(
    character: Mapping[Weight, int], groups: Sequence[str] = DEFAULT_GROUPS
) -> Counter[Weight]:
    """Irrep multiplicities for any direct product of SU(2)/SU(3) factors.

    Keys concatenate the highest Dynkin labels in the same order as character
    weights.  By default they are ``(lorentz_l, lorentz_r, weak, colour_p,
    colour_q)``.  Setting groups=("SU2", "SU3") returns keys (weak,p,q).
    Highest-weight subtraction is exact, including repeated irrep channels.
    """
    groups = _normalized_groups(groups)
    frozen = _freeze(character)
    rank = sum(1 if g == "SU2" else 2 for g in groups)
    if frozen and _rank(frozen) != rank:
        raise ValueError(f"Expected weight rank {rank} for groups {groups!r}")
    return _thaw(_decompose_character(frozen, groups))


def decompose_su2(character: Mapping[Weight, int]) -> Counter[int]:
    """Decompose an SU(2) character into Dynkin-label multiplicities."""
    remainder = _thaw(_freeze(character))
    if any(len(w) != 1 for w in remainder):
        raise ValueError("An SU(2) character must have rank one")
    result: Counter[int] = Counter()
    while remainder:
        highest = max(w[0] for w in remainder)
        multiplicity = remainder[(highest,)]
        if highest < 0 or multiplicity < 0:
            raise ValueError("Input is not a nonnegative SU(2) representation")
        result[highest] += multiplicity
        for weight, m in _su2_character(highest):
            remainder[weight] -= multiplicity * m
            if remainder[weight] < 0:
                raise ValueError("Input is not a valid SU(2) character")
            if not remainder[weight]:
                del remainder[weight]
    return result


def decompose_su3(character: Mapping[Weight, int]) -> Counter[tuple[int, int]]:
    """Decompose an SU(3) character by exact highest-weight subtraction."""
    remainder = _thaw(_freeze(character))
    if any(len(w) != 2 for w in remainder):
        raise ValueError("An SU(3) character must have rank two")
    result: Counter[tuple[int, int]] = Counter()
    while remainder:
        candidates = [w for w, m in remainder.items() if m > 0 and min(w) >= 0]
        if not candidates:
            raise ValueError("Input is not a nonnegative SU(3) representation")
        highest = max(candidates, key=lambda w: (sum(w), w))
        multiplicity = remainder[highest]
        result[highest] += multiplicity
        for weight, m in _su3_character(*highest):
            remainder[weight] -= multiplicity * m
            if remainder[weight] < 0:
                raise ValueError("Input is not a valid SU(3) character")
            if not remainder[weight]:
                del remainder[weight]
    return result


@lru_cache(maxsize=1024)
def _su2_tensor_product(n: int, m: int) -> tuple[tuple[int, int], ...]:
    return tuple((label, 1) for label in range(abs(n - m), n + m + 1, 2))


def su2_tensor_product(n: int, m: int) -> Counter[int]:
    """SU(2) Clebsch--Gordan decomposition as {Dynkin label: multiplicity}."""
    n = _nonnegative_integer(n, "SU(2) Dynkin label")
    m = _nonnegative_integer(m, "SU(2) Dynkin label")
    return Counter(dict(_su2_tensor_product(n, m)))


@lru_cache(maxsize=1024)
def _su3_tensor_product(left: tuple[int, int], right: tuple[int, int]):
    character = _thaw(_tensor_two(_su3_character(*left), _su3_character(*right)))
    return tuple(sorted(decompose_su3(character).items()))


def su3_tensor_product(
    left: tuple[int, int], right: tuple[int, int]
) -> Counter[tuple[int, int]]:
    """SU(3) tensor-product decomposition in unambiguous Dynkin notation."""
    if len(left) != 2 or len(right) != 2:
        raise ValueError("Each SU(3) irrep needs two Dynkin labels")
    left = tuple(_nonnegative_integer(x, "SU(3) Dynkin label") for x in left)
    right = tuple(_nonnegative_integer(x, "SU(3) Dynkin label") for x in right)
    # Symmetric cache key for commutative tensor products.
    if left > right:
        left, right = right, left
    return Counter(dict(_su3_tensor_product(left, right)))


def clear_caches() -> None:
    """Release memoized characters after an unusually large exploratory scan."""
    for function in (
        _su2_character,
        _su3_character,
        _tensor_two,
        _direct_product,
        _field_character,
        _power,
        _denominator,
        _singlet_product,
        _decompose_character,
        _su2_tensor_product,
        _su3_tensor_product,
    ):
        function.cache_clear()
