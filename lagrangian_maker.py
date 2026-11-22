"""Public function for calculating and displaying a new-particle Lagrangian.

    from lagrangian_maker import make_lagrangian
    result = make_lagrangian(2, x, y)

Inputs are ordinary dictionaries. No notebook globals, prompts, widgets, or
result files are used. The function displays LaTeX and returns reusable results.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from lagrangian_engine import expand_models, pair_channels, enumerate_operators, model_label


@dataclass(repr=False)
class LagrangianResult:
    models: tuple
    catalogues: list
    sm_generations: int = 3

    def __repr__(self):
        count = sum(len(item["operators"]) for item in self.catalogues)
        return f"LagrangianResult({len(self.models)} scenario(s), {count} operator families)"

    def show(self):
        """Display pair channels first, followed by the operator equations."""
        from terminal_support import show_catalogues
        show_catalogues(self.catalogues)

    def select(self, operators="all", *, scenario=None, show=True):
        """Build L_SM plus selected operator IDs, or every row with 'all'.

        Example: lagrangian = result.select(["O0002", "O0005"])
        For multiple particle-type scenarios, pass the displayed scenario number.
        Each ID includes all its contractions/flavours with separate couplings.
        """
        from lagrangian_selection import select_operators
        return select_operators(self, operators, scenario=scenario, show=show)


def _field_input(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dictionary of particle properties.")
    required = {"spin", "su2", "su3", "hypercharge"}
    optional = {"boson_kind", "fermion_kind"}
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(sorted(missing))}.")
    if extra:
        raise ValueError(f"{name} has unknown properties: {', '.join(sorted(map(str, extra)))}.")
    field = dict(value, name=name)
    for key in ("spin", "boson_kind", "fermion_kind"):
        if key in field:
            field[key] = str(field[key]).strip().lower()
    if field.get("boson_kind", "complex") not in ("complex", "real", "any"):
        raise ValueError(f"{name}: boson_kind must be complex, real, or any.")
    if field.get("fermion_kind", "weyl") not in ("weyl", "dirac", "majorana", "any"):
        raise ValueError(f"{name}: fermion_kind must be weyl, dirac, majorana, or any.")
    return field


def make_lagrangian(n_particles, x, y=None, *, exact_dimension=None, sm_generations=3, show=True):
    """Calculate a model and immediately display its pair channels and operators.

    Example:
        x = dict(spin="scalar", su2=1, su3="1", hypercharge="0")
        y = dict(spin="fermion", su2=1, su3="1", hypercharge="0",
                 fermion_kind="majorana")
        result = make_lagrangian(2, x, y)

    Use make_lagrangian(1, x) for one particle. Spin may be "any", expanding
    scalar/fermion/vector choices at fixed gauge quantum numbers. Bosons default
    to complex and fermions to one left-handed Weyl field. Optional kinds:
    boson_kind="real"; fermion_kind="dirac" or "majorana". Setting either kind
    to "any" scans all types compatible with the fixed gauge quantum numbers.

    A Majorana multiplet uses one left-handed Weyl field and its conjugate,
    without an independent Dirac partner. It requires zero hypercharge, odd
    SU(2) dimension, and SU(3) Dynkin p=q. The default d<=4 includes Majorana
    mass terms (d=3); exact_dimension=4 excludes all lower-dimensional terms.

    Results use singlet-contraction notation. Derivative rows retain
    redundancies. Set show=False to calculate without notebook display.
    Each call is independent and does not modify x or y or write files.
    """
    if type(n_particles) is not int or n_particles not in (1, 2):
        raise ValueError("n_particles must be 1 or 2.")
    if n_particles == 2 and y is None:
        raise ValueError("Two particles require both x and y: make_lagrangian(2, x, y).")
    if n_particles == 1 and y is not None:
        raise ValueError("For one particle, call make_lagrangian(1, x) without y.")
    if exact_dimension is not None and (type(exact_dimension) is not int or exact_dimension != 4):
        raise ValueError("exact_dimension must be None (all d<=4) or 4 (only d=4).")
    if type(sm_generations) is not int or sm_generations < 1:
        raise ValueError("sm_generations must be a positive integer.")
    if type(show) is not bool:
        raise ValueError("show must be True or False.")

    fields = [_field_input(x, "X")]
    if n_particles == 2:
        fields.append(_field_input(y, "Y"))
    models = tuple(expand_models(fields))
    catalogues = []
    for index, model in enumerate(models, 1):
        catalogues.append(dict(index=index, label=f"Scenario {index}: {model_label(model)}",
                               pairs=pair_channels(model), operators=enumerate_operators(
                                   model, exact_dimension=exact_dimension, sm_generations=sm_generations)))
    result = LagrangianResult(models, catalogues, sm_generations)
    if show:
        result.show()
    return result
