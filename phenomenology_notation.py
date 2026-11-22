"""Conventional field expressions for recognized renormalizable structures.

This is a presentation layer: it never changes the operator catalogue or takes
an integration-by-parts quotient. Fermions use two-component Weyl notation,
including Majorana fields. Unsupported contractions return ``None`` so callers
can retain an explicitly indexed invariant tensor rather than invent a formula.
"""
from collections import Counter
from itertools import permutations
from math import prod

from lagrangian_engine import make_letters


def _base(name):
    if name in ("uc", "dc", "ec"):
        return name[0] + "^{c}"
    if name.endswith("_Rc"):
        return name[:-3] + "_{R}^{c}"
    if name.endswith("_L"):
        return name[:-2] + "_{L}"
    if "_" in name:
        stem, sub = name.split("_", 1)
        return stem + "_{" + sub + "}"
    return name


def _symbol(name, *, lower=(), upper=()):
    """Group decorated names before adding explicit component indices."""
    dagger = name.endswith("†")
    stem = _base(name[:-1] if dagger else name)
    if lower or upper or (dagger and "^" in stem):
        stem = "{" + stem + "}"
    if lower:
        stem += "_{" + " ".join(lower) + "}"
    superscript = ([r"\dagger"] if dagger else []) + list(upper)
    if superscript:
        stem += "^{" + " ".join(superscript) + "}"
    return stem


def _singlet(letter):
    return letter.weak == 0 and letter.colour == (0, 0) and letter.charge == 0


def _scalar(letter):
    return (not letter.fermion and not letter.field_strength
            and letter.lorentz == (0, 0))


def _product(expressions):
    counts = Counter(expressions)
    return r"\,".join(
        expression if count == 1 else "{" + expression + "}^{" + str(count) + "}"
        for expression, count in counts.items()
    )


def _flavours(letters):
    indices = iter(("i", "j", "k", "l"))
    assigned = [next(indices) if letter.nf > 1 else None for letter in letters]
    ranges = {index: letter.nf for letter, index in zip(letters, assigned) if index}
    return assigned, ranges


def _answer(row, expression, *, index_ranges=None, components=None):
    ranges = index_ranges or {}
    result = dict(
        expression=expression,
        self_conjugate=row["field_content"] == row["conjugate_content"],
        multiplicity=row["multiplicity"],
        summed_indices=tuple(ranges),
        coefficient_suffix="".join(ranges),
        index_ranges=ranges,
    )
    if components is not None:
        result["components"] = components
    return result


def _potential(row, letters):
    """Singlet scalars and ordinary Y=+1/2 colourless scalar doublets."""
    if any(letter.derivative or not _scalar(letter) for letter in letters):
        return None
    singlets = [letter for letter in letters if _singlet(letter)]
    doublets = [letter for letter in letters if not _singlet(letter)]
    if any(letter.weak != 1 or letter.colour != (0, 0)
           or letter.charge != (-1 if letter.name.endswith("†") else 1) / 2
           for letter in doublets):
        return None
    daggers = [letter for letter in doublets if letter.name.endswith("†")]
    undagged = [letter for letter in doublets if not letter.name.endswith("†")]
    if len(daggers) != len(undagged):
        return None
    singlet_counts = Counter(letter.name for letter in singlets)
    singlet_product = []
    for name in sorted(singlet_counts):
        if name.endswith("†"):
            continue
        pairs = min(singlet_counts[name], singlet_counts[name + "†"])
        if pairs:
            singlet_product.extend([r"\left(" + _symbol(name + "†") + " "
                                    + _symbol(name) + r"\right)"] * pairs)
            singlet_counts[name] -= pairs
            singlet_counts[name + "†"] -= pairs
    singlet_product.extend(_symbol(name) for name, count in sorted(singlet_counts.items())
                           for _ in range(count))
    products = set()
    for ordering in permutations(undagged):
        bilinears = sorted(r"\left(" + _symbol(left.name) + " " + _symbol(right.name)
                           + r"\right)" for left, right in zip(daggers, ordering))
        products.add(_product(singlet_product + bilinears))
    components = sorted(products)
    if len(components) != row["multiplicity"]:
        return None
    if len(components) == 1:
        return _answer(row, components[0])
    # At most four scalar factors occur here. For self-conjugate content the
    # two degree-four invariants are (A†A)(B†B) and (A†B)(B†A), individually
    # Hermitian; non-self-conjugate mixed products keep their h.c. unchanged.
    return _answer(row, components[0], components=components)


def _norm_potential(row, letters):
    """Unique products of scalar norms in arbitrary gauge representations."""
    if row["multiplicity"] != 1 or any(letter.derivative or not _scalar(letter) for letter in letters):
        return None
    lookup = {letter.name: letter for letter in letters}
    counts = Counter(letter.name for letter in letters)
    factors = []
    for name in sorted(counts):
        letter = lookup[name]
        if _singlet(letter):
            factors.extend([_symbol(name)] * counts[name])
            counts[name] = 0
        elif letter.conjugate == name:
            if counts[name] % 2:
                return None
            factors.extend([r"\left(" + _symbol(name) + r"\cdot " + _symbol(name)
                            + r"\right)"] * (counts[name] // 2))
            counts[name] = 0
        elif not name.endswith("†"):
            if counts[name] != counts[name + "†"]:
                return None
            factors.extend([r"\left(" + _symbol(name + "†") + " " + _symbol(name)
                            + r"\right)"] * counts[name])
            counts[name] = counts[name + "†"] = 0
    if any(counts.values()):
        return None
    return _answer(row, _product(factors))


def _scalar_kinetic(row, letters, lookup):
    if (len(letters) != 2 or row["multiplicity"] != 1
            or any(not letter.derivative or not _scalar(letter) for letter in letters)):
        return None
    bare = [lookup[letter.name[2:-1]] for letter in letters]
    derivatives = [r"\partial" if _singlet(letter) else "D" for letter in bare]
    if all(_singlet(letter) for letter in bare):
        expression = (derivatives[0] + r"_{\mu} " + _symbol(bare[0].name)
                      + r"\," + derivatives[1] + r"^{\mu} " + _symbol(bare[1].name))
        return _answer(row, expression)
    if (all(letter.conjugate == letter.name for letter in bare)
            and bare[0].weak == bare[1].weak and bare[0].colour == bare[1].colour):
        expression = (r"\left(D_{\mu} " + _symbol(bare[0].name) + r"\right)\cdot "
                      + r"\left(D^{\mu} " + _symbol(bare[1].name) + r"\right)")
        return _answer(row, expression)
    # For a conjugate pair the familiar Hermitian inner product carries all
    # representation indices, in any SU(2)/SU(3) representation.
    daggers = [letter for letter in bare if letter.name.endswith("†")]
    undagged = [letter for letter in bare if not letter.name.endswith("†")]
    if len(daggers) != 1 or len(undagged) != 1:
        return None
    left, right = daggers[0], undagged[0]
    if (left.weak != right.weak or left.colour != right.colour[::-1]
            or left.charge != -right.charge):
        return None
    expression = (r"\left(D_{\mu} " + _symbol(left.name[:-1]) + r"\right)^{\dagger}"
                  + r"\left(D^{\mu} " + _symbol(right.name) + r"\right)")
    return _answer(row, expression)


def _fermion_kinetic(row, letters, lookup):
    if (len(letters) != 2 or not all(letter.fermion for letter in letters)
            or sum(letter.derivative for letter in letters) != 1):
        return None
    differentiated = next(letter for letter in letters if letter.derivative)
    other = next(letter for letter in letters if not letter.derivative)
    bare = lookup[differentiated.name[2:-1]]
    if (bare.weak != other.weak or bare.colour != other.colour[::-1]
            or bare.charge != -other.charge or bare.lorentz != other.lorentz[::-1]):
        return None
    if row["multiplicity"] != bare.nf * other.nf:
        return None
    left, right = (other, bare) if bare.lorentz == (1, 0) else (bare, other)
    if not left.name.endswith("†") or right.name.endswith("†"):
        return None
    (fl, fr), ranges = _flavours((left, right))
    left_symbol = _symbol(left.name, lower=[fl] if fl else [])
    right_symbol = _symbol(right.name, lower=[fr] if fr else [])
    derivative = r"\partial" if _singlet(bare) else "D"
    if bare == right:
        expression = ("i " + left_symbol + r"\bar{\sigma}^{\mu} "
                      + derivative + r"_{\mu} " + right_symbol)
    else:
        expression = (r"i\left(" + derivative + r"_{\mu} " + left_symbol
                      + r"\right)\bar{\sigma}^{\mu} " + right_symbol)
    # Even diagonal first-derivative rows are kept complex plus h.c.: making
    # them Hermitian here would silently apply integration by parts.
    return _answer(row, expression, index_ranges=ranges)


def _fermion_pair(row, letters):
    """Gauge-singlet fermion mass or a singlet-scalar Yukawa coupling."""
    if any(letter.derivative or letter.field_strength for letter in letters):
        return None
    fermions = [letter for letter in letters if letter.fermion]
    scalars = [letter for letter in letters if not letter.fermion]
    if len(fermions) != 2 or len(scalars) > 1:
        return None
    if any(not _singlet(letter) for letter in fermions):
        return None
    if any(not _scalar(letter) or not _singlet(letter) for letter in scalars):
        return None
    first, second = fermions
    if first.lorentz != second.lorentz or first.lorentz not in ((1, 0), (0, 1)):
        return None
    if row["multiplicity"] != prod(letter.nf for letter in fermions):
        return None
    (ff, fs), ranges = _flavours(fermions)
    alpha = r"\alpha" if first.lorentz == (1, 0) else r"\dot{\alpha}"
    first_symbol = _symbol(first.name, lower=[ff] if ff else [], upper=[alpha])
    second_symbol = _symbol(second.name, lower=([fs] if fs else []) + [alpha])
    expression = _product([_symbol(letter.name) for letter in scalars]
                          + [first_symbol + " " + second_symbol])
    return _answer(row, expression, index_ranges=ranges)


def _neutrino_portal(row, letters):
    if (len(letters) != 3 or any(letter.derivative or letter.field_strength for letter in letters)):
        return None
    scalars = [letter for letter in letters if _scalar(letter)]
    fermions = [letter for letter in letters if letter.fermion]
    if len(scalars) != 1 or len(fermions) != 2:
        return None
    scalar = scalars[0]
    singlets = [letter for letter in fermions if _singlet(letter)]
    doublets = [letter for letter in fermions if letter.weak == 1 and letter.colour == (0, 0)]
    if len(singlets) != 1 or len(doublets) != 1:
        return None
    singlet, doublet = singlets[0], doublets[0]
    if (scalar.weak != 1 or scalar.colour != (0, 0)
            or scalar.charge != -doublet.charge
            or doublet.lorentz != singlet.lorentz):
        return None
    if row["multiplicity"] != singlet.nf * doublet.nf:
        return None
    (fd, fn), ranges = _flavours((doublet, singlet))
    alpha = r"\alpha" if doublet.lorentz == (1, 0) else r"\dot{\alpha}"
    fermion = _symbol(doublet.name, lower=[fd] if fd else [], upper=["a", alpha])
    sterile = _symbol(singlet.name, lower=([fn] if fn else []) + [alpha])
    if scalar.name.endswith("†") != doublet.name.endswith("†"):
        # A daggered doublet and an undaggered doublet use the Hermitian
        # inner product; no epsilon tensor is needed in this convention.
        if scalar.name.endswith("†"):
            scalar_symbol = _symbol(scalar.name, lower=["a"])
        else:
            fermion = _symbol(doublet.name, lower=([fd] if fd else []) + ["a"], upper=[alpha])
            scalar_symbol = _symbol(scalar.name, upper=["a"])
        expression = scalar_symbol + " " + fermion + " " + sterile
    else:
        if scalar.name.endswith("†"):
            fermion = _symbol(doublet.name, lower=([fd] if fd else []) + ["a"], upper=[alpha])
            scalar_symbol = _symbol(scalar.name, lower=["b"])
            epsilon = r"\epsilon^{ab} "
        else:
            scalar_symbol = _symbol(scalar.name, upper=["b"])
            epsilon = r"\epsilon_{ab} "
        expression = epsilon + fermion + " " + scalar_symbol + " " + sterile
    return _answer(row, expression, index_ranges=ranges)


def _sm_yukawa(row, letters):
    """A Higgs-like new scalar doublet coupled to SM quarks or leptons."""
    if len(letters) != 3 or any(letter.derivative for letter in letters):
        return None
    scalars = [letter for letter in letters if _scalar(letter)]
    fermions = [letter for letter in letters if letter.fermion]
    if len(scalars) != 1 or len(fermions) != 2:
        return None
    scalar = scalars[0]
    if not scalar.new or scalar.weak != 1 or scalar.colour != (0, 0):
        return None
    doublets = [letter for letter in fermions if letter.species in ("Q", "L")]
    singlets = [letter for letter in fermions if letter.species in ("uc", "dc", "ec")]
    if len(doublets) != 1 or len(singlets) != 1:
        return None
    doublet, singlet = doublets[0], singlets[0]
    if (doublet.weak != 1 or singlet.weak != 0
            or doublet.colour != singlet.colour[::-1]
            or doublet.charge + scalar.charge + singlet.charge != 0
            or doublet.lorentz != singlet.lorentz):
        return None
    if row["multiplicity"] != doublet.nf * singlet.nf:
        return None
    (fd, fs), ranges = _flavours((doublet, singlet))
    alpha = r"\alpha" if doublet.lorentz == (1, 0) else r"\dot{\alpha}"
    d_lower, d_upper = ([fd] if fd else []), [alpha]
    s_lower, s_upper = ([fs] if fs else []) + [alpha], []
    if doublet.colour == (1, 0):
        d_upper.append("A")
        s_lower.append("A")
    elif doublet.colour == (0, 1):
        d_lower.append("A")
        s_upper.append("A")
    if scalar.name.endswith("†") == doublet.name.endswith("†"):
        if scalar.name.endswith("†"):
            d_lower.append("a")
            scalar_symbol = _symbol(scalar.name, lower=["b"])
            epsilon = r"\epsilon^{ab} "
        else:
            d_upper.append("a")
            scalar_symbol = _symbol(scalar.name, upper=["b"])
            epsilon = r"\epsilon_{ab} "
    else:
        if scalar.name.endswith("†"):
            d_upper.append("a")
            scalar_symbol = _symbol(scalar.name, lower=["a"])
        else:
            d_lower.append("a")
            scalar_symbol = _symbol(scalar.name, upper=["a"])
        epsilon = ""
    expression = (epsilon + _symbol(doublet.name, lower=d_lower, upper=d_upper)
                  + " " + scalar_symbol + " "
                  + _symbol(singlet.name, lower=s_lower, upper=s_upper))
    return _answer(row, expression, index_ranges=ranges)


def format_operator(model, row, sm_generations=3):
    """Return conventional notation metadata, or None for unsupported tensors.

    ``expression`` has neither a coefficient nor a Hermitian-conjugate term.
    ``index_ranges`` supplies SM flavour sums. If ``components`` is present,
    its entries are separately contracted monomials requiring separate
    coefficients, rather than one unresolved contraction index.
    """
    lookup = {letter.name: letter for letter in make_letters(model, sm_generations)}
    letters = [lookup[factor["name"]] for factor in row["factors"]
               for _ in range(factor["count"])]
    for formatter in (_potential, _norm_potential, _fermion_pair, _neutrino_portal, _sm_yukawa):
        result = formatter(row, letters)
        if result is not None:
            return result
    for formatter in (_scalar_kinetic, _fermion_kinetic):
        result = formatter(row, letters, lookup)
        if result is not None:
            return result
    return None
