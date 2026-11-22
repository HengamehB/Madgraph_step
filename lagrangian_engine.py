"""From-scratch SM extension catalogue: exact characters, full first derivative jets.

All group labels in the public API are ordered SU(2), SU(3), U(1).
Derivative rows form a spanning set, not a quotient by IBP or equations of motion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
import math
import re

from group_theory import (
    field_character, tensor_product, symmetric_power, exterior_power,
    singlet_product_multiplicity, decompose_character,
)


def parse_su3(value):
    """Unambiguous common dimension aliases or arbitrary nonnegative Dynkin (p,q)."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        p, q = value
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (p, q)):
            raise ValueError("SU(3) Dynkin labels must be nonnegative integers.")
        return p, q
    s = str(value).strip().lower().replace(" ", "").replace("¯", "bar")
    aliases = {"1": (0, 0), "3": (1, 0), "3bar": (0, 1), "-3": (0, 1),
               "6": (2, 0), "6bar": (0, 2), "-6": (0, 2), "8": (1, 1),
               "10": (3, 0), "10bar": (0, 3), "-10": (0, 3)}
    if s in aliases:
        return aliases[s]
    match = re.fullmatch(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", s)
    if match:
        return tuple(map(int, match.groups()))
    raise ValueError("Use 1, 3, 3bar, 6, 6bar, 8, 10, 10bar or Dynkin labels (p,q).")


def parse_charge(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("Enter exact hypercharge as an integer or fraction, e.g. 1/6.")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    s = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+(?:/\d+)?", s):
        raise ValueError("Hypercharge must be an exact integer or fraction; 'any' is only for spin or field type.")
    try:
        return Fraction(s)
    except ZeroDivisionError as exc:
        raise ValueError("Hypercharge denominator must be nonzero.") from exc


def colour_dimension(rep):
    p, q = rep
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def colour_label(rep):
    p, q = rep
    return f"{colour_dimension(rep)} [{p},{q}]"


def colour_display_label(rep):
    """Use familiar colour names, retaining Dynkin labels for other irreps."""
    common = {(0, 0): "1", (1, 0): "3", (0, 1): "3bar", (2, 0): "6",
              (0, 2): "6bar", (1, 1): "8", (3, 0): "10", (0, 3): "10bar"}
    return common.get(tuple(rep), f"{colour_dimension(rep)} (Dynkin {tuple(rep)})")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    spin: str
    su2: int
    su3: tuple
    hypercharge: Fraction
    kind: str = "complex"

    def __post_init__(self):
        if not re.fullmatch(r"[XYZ][A-Za-z0-9_]*", self.name):
            raise ValueError("New field names must start with X, Y, or Z and contain letters/digits/_.")
        if self.spin not in ("scalar", "fermion", "vector"):
            raise ValueError("Concrete spin must be scalar, fermion or vector; expand 'any' first.")
        if isinstance(self.su2, bool) or not isinstance(self.su2, int) or self.su2 < 1:
            raise ValueError("SU(2) must be a positive integer representation dimension.")
        object.__setattr__(self, "su3", parse_su3(self.su3))
        object.__setattr__(self, "hypercharge", parse_charge(self.hypercharge))
        if self.spin == "fermion" and self.kind == "complex":
            object.__setattr__(self, "kind", "weyl")
        allowed = ("weyl", "dirac", "majorana") if self.spin == "fermion" else ("complex", "real")
        if self.kind not in allowed:
            raise ValueError(f"{self.spin} kind must be one of {allowed}.")
        if self.kind in ("real", "majorana"):
            if self.hypercharge or self.su2 % 2 == 0 or self.su3[0] != self.su3[1]:
                raise ValueError("A single real/Majorana multiplet needs Y=0, odd SU(2) dimension, and SU(3) p=q.")


def expand_models(specifications):
    """Expand spin/type choices at fixed gauge quantum numbers.

    An 'any' kind includes real/Majorana fields only in admissible real gauge
    representations. Explicitly requested invalid kinds still raise an error.
    """
    if len(specifications) not in (1, 2):
        raise ValueError("Choose exactly one or two new multiplets.")
    choices = []
    for index, spec in enumerate(specifications):
        if isinstance(spec, FieldSpec):
            choices.append((spec,))
            continue
        spin = str(spec["spin"]).strip().lower()
        if spin not in ("any", "scalar", "fermion", "vector"):
            raise ValueError("Spin must be scalar, fermion, vector or any.")
        spins = ("scalar", "fermion", "vector") if spin == "any" else (spin,)
        weak = spec["su2"]
        if isinstance(weak, bool) or not isinstance(weak, int) or weak < 1:
            raise ValueError("SU(2) must be a positive integer representation dimension.")
        colour = parse_su3(spec["su3"])
        charge = parse_charge(spec["hypercharge"])
        permits_real = charge == 0 and weak % 2 == 1 and colour[0] == colour[1]
        alternatives = []
        for concrete_spin in spins:
            if concrete_spin == "fermion":
                requested = str(spec.get("fermion_kind", "weyl")).strip().lower()
                kinds = ("weyl", "dirac", "majorana")
            else:
                requested = str(spec.get("boson_kind", "complex")).strip().lower()
                kinds = ("complex", "real")
            if requested == "any":
                kinds = tuple(kind for kind in kinds
                              if kind not in ("real", "majorana") or permits_real)
            else:
                kinds = (requested,)
            alternatives.extend(FieldSpec(spec.get("name", ("X", "Y")[index]), concrete_spin,
                                          weak, colour, charge, kind) for kind in kinds)
        choices.append(tuple(alternatives))
    models = list(product(*choices))
    for model in models:
        if len({x.name for x in model}) != len(model):
            raise ValueError("New field names must be distinct.")
    return models


def model_label(model):
    return "; ".join(f"{f.name}: {f.kind} {f.spin} "
                     f"(SU(2)={f.su2}, SU(3)={colour_display_label(f.su3)}, hypercharge={f.hypercharge})"
                     for f in model)


@dataclass(frozen=True)
class Letter:
    name: str
    species: str
    lorentz: tuple
    weak: int
    colour: tuple
    charge: Fraction
    dim2: int
    fermion: bool = False
    nf: int = 1
    new: bool = False
    derivative: bool = False
    conjugate: str = ""
    field_strength: bool = False

    @property
    def signature(self):
        return (self.lorentz, self.weak, self.colour, self.nf, self.derivative, self.fermion)


def _matter_pair(name, species, lorentz, weak, colour, charge, dim2, *, nf=1, new=False, real=False):
    first = Letter(name, species, lorentz, weak, colour, Fraction(charge), dim2,
                   dim2 == 3, nf, new, conjugate=name if real else name + "†")
    if real:
        return [first]
    second = replace(first, name=name + "†", lorentz=lorentz[::-1], colour=colour[::-1],
                     charge=-first.charge, conjugate=name)
    return [first, second]


def new_letters(model):
    letters = []
    for field in model:
        args = dict(species=field.name, weak=field.su2 - 1, colour=field.su3,
                    charge=field.hypercharge, new=True)
        if field.spin == "fermion":
            if field.kind == "dirac":
                letters += _matter_pair(field.name + "_L", lorentz=(1, 0), dim2=3, **args)
                args.update(colour=field.su3[::-1], charge=-field.hypercharge)
                letters += _matter_pair(field.name + "_Rc", lorentz=(1, 0), dim2=3, **args)
            else:
                letters += _matter_pair(field.name, lorentz=(1, 0), dim2=3, **args)
        else:
            letters += _matter_pair(field.name, lorentz=(0, 0) if field.spin == "scalar" else (1, 1),
                                    dim2=2, real=field.kind == "real", **args)
    if len({letter.name for letter in letters}) != len(letters):
        raise ValueError("New field names collide with generated conjugate or Dirac component names.")
    return letters


def make_letters(model, sm_generations=3):
    if not isinstance(sm_generations, int) or isinstance(sm_generations, bool) or sm_generations < 1:
        raise ValueError("SM generations must be a positive integer.")
    matter = _matter_pair("H", "H", (0, 0), 1, (0, 0), Fraction(1, 2), 2)
    # All fermions are left-handed; uc, dc and ec are conjugates of RH SM fields.
    for name, weak, colour, charge in (
        ("Q", 1, (1, 0), Fraction(1, 6)), ("uc", 0, (0, 1), Fraction(-2, 3)),
        ("dc", 0, (0, 1), Fraction(1, 3)), ("L", 1, (0, 0), Fraction(-1, 2)),
        ("ec", 0, (0, 0), Fraction(1)),
    ):
        matter += _matter_pair(name, name, (1, 0), weak, colour, charge, 3, nf=sm_generations)
    matter += new_letters(model)
    letters = matter[:]
    for letter in matter:
        letters.append(replace(letter, name=f"D({letter.name})", dim2=letter.dim2 + 2,
                               derivative=True, conjugate=f"D({letter.conjugate})"))
    for name, weak, colour in (("B", 0, (0, 0)), ("W", 2, (0, 0)), ("G", 0, (1, 1))):
        letters.append(Letter(name + "+", name, (2, 0), weak, colour, Fraction(0), 4,
                              conjugate=name + "-", field_strength=True))
        letters.append(Letter(name + "-", name, (0, 2), weak, colour, Fraction(0), 4,
                              conjugate=name + "+", field_strength=True))
    return sorted(letters, key=lambda x: (x.dim2, x.name))


@lru_cache(maxsize=512)
def _character(signature):
    lorentz, weak, colour, nf, derivative, fermion = signature
    char = field_character(lorentz=lorentz, weak=weak, colour=colour)
    if derivative:
        char = tensor_product(char, field_character(lorentz=(1, 1), weak=0, colour=(0, 0)))
    return tuple(sorted((weight, multiplicity * nf) for weight, multiplicity in char.items()))


@lru_cache(maxsize=1024)
def _power(signature, count):
    char = Counter(dict(_character(signature)))
    power = exterior_power if signature[-1] else symmetric_power
    return tuple(sorted(power(char, count).items()))


@lru_cache(maxsize=32768)
def _count_invariants(factors):
    # Charge and center tests happen before this expensive, representation-only cached step.
    chars = [Counter(dict(_power(signature, count))) for signature, count in factors]
    chars.sort(key=len)
    return singlet_product_multiplicity(chars)


def _content(counts):
    return " ".join(name if n == 1 else f"{name}^{n}" for name, n in counts)


def _latex_name(name):
    if name.startswith("D("):
        return r"(D " + _latex_name(name[2:-1]) + ")"
    if name.endswith("†"):
        return "(" + _latex_name(name[:-1]) + r")^{\dagger}"
    if name in ("B+", "B-", "W+", "W-", "G+", "G-"):
        return name[0] + "_{" + name[1] + "}"
    if name in ("uc", "dc", "ec"):
        return name[0] + "^c"
    if "_" in name:
        stem, sub = name.split("_", 1)
        return stem + "_{" + sub + "}"
    return name


def pair_channels(model):
    """All XX, XY, YY and conjugate channels, with identical-letter statistics.

    Lorentz labels are (j_L,j_R), not rest-frame angular momentum. These are local
    products without derivatives; orbital angular momentum is not included.
    """
    rows = []
    letters = sorted(new_letters(model), key=lambda x: x.name)
    for a, b in combinations_with_replacement(letters, 2):
        first = Counter(dict(_character(a.signature)))
        if a == b:
            char = (exterior_power if a.fermion else symmetric_power)(first, 2)
            stats = ("Identical fermions: exchanging the two fields, including all indices, changes the sign."
                     if a.fermion else
                     "Identical bosons: exchanging the two fields, including all indices, leaves the product unchanged.")
        else:
            char = tensor_product(first, Counter(dict(_character(b.signature))))
            stats = "Different fields: no identical-particle exchange restriction."
        for (left, right, weak, p, q), multiplicity in sorted(decompose_character(char).items()):
            rows.append(dict(pair=f"{a.name} × {b.name}", lorentz=f"({Fraction(left, 2)}, {Fraction(right, 2)})",
                             su2=weak + 1, su3=colour_label((p, q)), hypercharge=str(a.charge + b.charge),
                             multiplicity=multiplicity, statistics=stats))
    return rows


def enumerate_operators(model, max_dimension=4, exact_dimension=None, sm_generations=3):
    """Return all invariant field-content families in a d<=4 spanning catalogue.

    Multiplicity is the exact dimension of the LOCAL polynomial invariant space,
    including flavour, before IBP/Bianchi/EOM reduction. One representative per HC
    pair of field contents. Individual Clebsch tensors are optional, not implicit.
    """
    if max_dimension not in (1, 2, 3, 4):
        raise ValueError("This engine is designed and tested only for dimensions 1 through 4.")
    if exact_dimension is not None and exact_dimension not in range(1, max_dimension + 1):
        raise ValueError("exact_dimension must be between 1 and max_dimension.")
    if len(model) not in (1, 2) or len({f.name for f in model}) != len(model):
        raise ValueError("Provide one or two distinct FieldSpec multiplets.")
    letters = make_letters(model, sm_generations)
    by_name = {x.name: x for x in letters}
    rows = []
    dimension_limit = max_dimension * 2
    charge_unit = math.lcm(*(x.charge.denominator for x in letters))
    charges = [int(x.charge * charge_unit) for x in letters]

    def visit(start, slots, dim2, charge):
        if slots and dim2 % 2 == 0 and (exact_dimension is None or dim2 == 2 * exact_dimension):
            chosen = [letters[i] for i in slots]
            if charge == 0 and any(x.new for x in chosen) and sum(x.fermion for x in chosen) % 2 == 0:
                # Exact center selection rules in all four nonabelian factors.
                center = [sum(x.lorentz[k] + int(x.derivative) for x in chosen) % 2 for k in (0, 1)]
                center += [sum(x.weak for x in chosen) % 2, sum(x.colour[0] - x.colour[1] for x in chosen) % 3]
                if not any(center) and not (len(chosen) == 1 and chosen[0].derivative):
                    counts = tuple(sorted(Counter(x.name for x in chosen).items()))
                    conjugates = tuple(sorted((by_name[name].conjugate, n) for name, n in counts))
                    if counts <= conjugates:
                        factors = tuple(sorted((by_name[name].signature, n) for name, n in counts))
                        multiplicity = _count_invariants(factors)
                        if multiplicity:
                            new_species = sorted({x.species for x in chosen if x.new})
                            has_sm = any(not x.new for x in chosen)
                            sector = (" + ".join(new_species) + " ↔ SM") if has_sm else (
                                "X–Y interaction" if len(new_species) > 1 else new_species[0] + " only")
                            derivatives = sum(x.derivative for x in chosen)
                            content = _content(counts)
                            tex = " ".join(_latex_name(name) if n == 1 else "(" + _latex_name(name) + ")^{" + str(n) + "}" for name, n in counts)
                            notes = ""
                            if derivatives:
                                notes = "Local contractions before IBP/Bianchi/EOM; derivative rows can be redundant or total derivatives."
                            if any(x.new and x.lorentz == (1, 1) for x in chosen):
                                notes += " Matter-vector power counting only; no gauge completion inferred."
                            self_conjugate = counts == conjugates
                            operator = f"[{content}]_(singlet)"
                            latex = r"\left[" + tex + r"\right]_{\mathbf{1}}"
                            if multiplicity > 1:
                                operator = f"[{content}]_(singlet,a), a=1,...,{multiplicity}"
                                latex = (r"\left[" + tex + r"\right]_{\mathbf{1},a},\quad a=1,\ldots,"
                                         + str(multiplicity))
                            rows.append(dict(dimension=dim2 // 2, coefficient_dimension=4 - dim2 // 2, field_content=content,
                                             derivatives=derivatives, multiplicity=multiplicity, sector=sector,
                                             operator=operator, latex=latex,
                                             hc_relation="self-conjugate field content; choose Hermitian combinations" if self_conjugate else "add Hermitian conjugate",
                                             notes=notes.strip(), factors=[{"name": name, "count": n} for name, n in counts],
                                             conjugate_content=_content(conjugates),
                                             coupling_parameters=f"{multiplicity} real before derivative relations" if self_conjugate else f"{multiplicity} complex before derivative relations"))
        for index in range(start, len(letters)):
            next_dimension = dim2 + letters[index].dim2
            if next_dimension > dimension_limit:
                break
            visit(index, slots + (index,), next_dimension, charge + charges[index])

    visit(0, (), 0, 0)
    rows.sort(key=lambda r: (r["dimension"], r["derivatives"], r["field_content"]))
    for index, row in enumerate(rows, 1):
        row["id"] = f"O{index:04d}"
    return rows


def explicit_operator(model, row, sm_generations=3, **limits):
    """Numerically construct component contractions and verify their exact count.

    This optional path requires NumPy and SciPy. Core enumeration is exact integer
    arithmetic and does not use this numerical calculation.
    """
    from component_invariants import component_invariants
    lookup = {letter.name: letter for letter in make_letters(model, sm_generations)}
    descriptors = []
    for factor in row["factors"]:
        letter = lookup[factor["name"]]
        descriptors.append(dict(name=letter.name, lorentz=letter.lorentz, weak=letter.weak,
                                colour=letter.colour, nf=letter.nf, derivative=letter.derivative,
                                fermion=letter.fermion, count=factor["count"]))
    return component_invariants(descriptors, expected_multiplicity=row["multiplicity"], **limits)


def cache_info():
    return {name: str(func.cache_info()) for name, func in (("characters", _character), ("powers", _power), ("counts", _count_invariants))}
