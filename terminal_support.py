"""Validated terminal prompts and plain-text catalogue display; no UI dependencies."""
from __future__ import annotations
import re

def _ask(prompt, parser, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        try:
            answer = input(f"{prompt}{suffix}: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input is unavailable. Supply particle dictionaries to make_lagrangian(), "
                "or edit a model.txt file and use the command-line setup helper."
            ) from exc
        if not answer and default is not None:
            answer = str(default)
        try:
            return parser(answer)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            print(f"  Please try again: {exc}")


def _choice(options):
    def parse(text):
        value = text.lower().strip()
        if value not in options:
            raise ValueError("choose " + ", ".join(options))
        return value
    return parse


def _positive_int(text):
    if not re.fullmatch(r"[1-9]\d*", str(text).strip()):
        raise ValueError("enter a positive integer")
    return int(text)


def _hypercharge(text):
    if str(text).strip().lower() == "any":
        raise ValueError("'any' is available for spin or field type; specify an exact hypercharge")
    from lagrangian_engine import parse_charge
    return parse_charge(text)


def collect_inputs():
    """Ask for one or two fields. ``any`` expands spin/type, never charges."""
    from lagrangian_engine import parse_su3

    print("Gauge order: SU(2)_L × SU(3)_c × U(1)_Y. Hypercharge convention: Q = T3 + Y.")
    print("'any' scans spin or field type at fixed gauge quantum numbers.\n")
    count = int(_ask("Number of new particles (1 or 2)", _choice(("1", "2")), "1"))
    fields = []
    for name in ("X", "Y")[:count]:
        print(f"\nParticle {name}")
        spin = _ask("Spin (scalar / fermion / vector / any)", _choice(("scalar", "fermion", "vector", "any")), "scalar")
        su2 = _ask("SU(2) representation dimension (1 = singlet, 2 = doublet, ...)", _positive_int, "1")
        su3 = _ask("SU(3) irrep (1, 3, 3bar, 6, 6bar, 8, 10, 10bar, or (p,q))", parse_su3, "1")
        hypercharge = _ask("U(1) hypercharge Y (exact fraction)", _hypercharge, "0")
        boson_kind, fermion_kind = "complex", "weyl"
        if spin in ("scalar", "vector", "any"):
            def boson_parser(text):
                kind = _choice(("complex", "real", "any"))(text)
                if kind == "real" and (hypercharge != 0 or su2 % 2 == 0 or su3[0] != su3[1]):
                    raise ValueError("a single real boson requires Y = 0, odd SU(2) dimension, and SU(3) labels p = q")
                return kind
            boson_kind = _ask("Boson field (complex / real / any)", boson_parser, "complex")
        if spin in ("fermion", "any"):
            def fermion_parser(text):
                kind = _choice(("weyl", "dirac", "majorana", "any"))(text)
                if kind == "majorana" and (hypercharge != 0 or su2 % 2 == 0 or su3[0] != su3[1]):
                    raise ValueError("a single Majorana fermion requires Y = 0, odd SU(2) dimension, and SU(3) labels p = q")
                return kind
            fermion_kind = _ask("Fermion field (Weyl / Dirac / Majorana / any)", fermion_parser, "weyl")
        fields.append(dict(name=name, spin=spin, su2=su2, su3=su3,
                           hypercharge=str(hypercharge), boson_kind=boson_kind,
                           fermion_kind=fermion_kind))
    print("\nRenormalizable operators have canonical dimension ≤ 4; exactly 4 excludes mass and cubic terms.")
    dimension = _ask("Operator dimensions (renormalizable / exactly4)", _choice(("renormalizable", "exactly4")), "renormalizable")
    generations = _ask("Number of Standard Model fermion generations", _positive_int, "3")
    return dict(fields=fields, max_dimension=4,
                exact_dimension=4 if dimension == "exactly4" else None,
                sm_generations=generations)



def show_pair_channels(rows, label=""):
    """Print every pair channel while retaining the complete returned catalogue."""
    print("Pair channels" + (" — " + label if label else ""))
    if not rows:
        print("None.")
    for row in rows:
        print(f"{row['pair']} | Lorentz {row['lorentz']} | SU(2) {row['su2']} | "
              f"SU(3) {row['su3']} | Y={row['hypercharge']} | copies={row['multiplicity']}")


def show_operators(rows, label=""):
    """Print IDs, dimensions and LaTeX source without importing a renderer."""
    print("Operators" + (" — " + label if label else ""))
    if not rows:
        print("None.")
    for row in rows:
        hc = " + h.c." if row['hc_relation'] == 'add Hermitian conjugate' else ""
        print(f"{row['id']} | d={row['dimension']} | {row['latex']}{hc}")


def show_catalogues(catalogues):
    """Plain terminal output; all structured metadata remains in the result."""
    for catalogue in catalogues:
        label = catalogue['label'] if len(catalogues) > 1 else ''
        show_pair_channels(catalogue['pairs'], label)
    for catalogue in catalogues:
        label = catalogue['label'] if len(catalogues) > 1 else ''
        show_operators(catalogue['operators'], label)
    if any(row['derivatives'] for catalogue in catalogues for row in catalogue['operators']):
        print('Derivative identities have not been removed.')
