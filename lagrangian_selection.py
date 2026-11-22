"""Select catalogue families and display a symbolic Lagrangian in the notebook."""
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field


def _indexed_tensor_expression(row):
    """Retain unresolved contractions explicitly, never erase their indices.

    I_k are collective spin/gauge/flavour indices. T denotes a symbolic
    invariant tensor, not a computed array. For self-conjugate content its
    contracted monomials are defined to form a Hermitian basis.
    """
    from lagrangian_engine import _latex_name

    factors, indices, derivatives = [], [], []
    for factor in row["factors"]:
        for _ in range(factor["count"]):
            index = "I_{" + str(len(indices) + 1) + "}"
            indices.append(index)
            name = factor["name"]
            if name.startswith("D("):
                mu = r"\mu_{" + str(len(derivatives) + 1) + "}"
                derivatives.append(mu)
                expression = r"\bigl(D_{" + mu + "} " + _latex_name(name[2:-1]) + r"\bigr)"
            else:
                expression = "{" + _latex_name(name) + "}"
            factors.append(expression + "^{" + index + "}")
    number = str(int(row["id"][1:]))
    label = number + (",a" if row["multiplicity"] > 1 else "")
    tensor = (r"\mathcal{T}^{(" + label + ")" + (r"\," + " ".join(derivatives) if derivatives else "")
              + "}_{" + " ".join(indices) + "}")
    return dict(expression=tensor + r"\," + " ".join(factors),
                self_conjugate=row["field_content"] == row["conjugate_content"],
                multiplicity=row["multiplicity"],
                coefficient_suffix="a" if row["multiplicity"] > 1 else "",
                index_ranges=(("a", row["multiplicity"]),) if row["multiplicity"] > 1 else (),
                style="tensor", tensor_components_computed=False,
                conventions="T is a symbolic invariant tensor with uncomputed components; "
                "I indices collect spin, gauge and flavour. For self-conjugate families the contracted "
                "tensor monomials define a Hermitian basis. No derivative identities are removed.")


def _coupled_term(coefficient, expression, self_conjugate):
    term = coefficient + r"\," + expression
    return term if self_conjugate else r"\left(" + term + r" + \mathrm{h.c.}\right)"


def _ids(operators, available):
    if isinstance(operators, str):
        if operators.strip().lower() == "all":
            return set(available)
        operators = [operators]
    if isinstance(operators, (bytes, Mapping)) or not isinstance(operators, Iterable):
        raise ValueError("Select operator IDs, e.g. ['O0002', 'O0005'], or 'all'.")
    values = list(operators)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Operator IDs must be strings, e.g. 'O0002'.")
    selected = {value.strip().upper() for value in values}
    unknown = selected - set(available)
    if unknown:
        raise ValueError("Unknown operator IDs in this scenario: " + ", ".join(sorted(unknown))
                         + ". Use IDs from the current result's operator list.")
    return selected


@dataclass(repr=False)
class SelectedLagrangian:
    scenario: int
    operators: tuple
    coefficients: tuple
    latex: str
    _source: object = field(repr=False)
    notation: tuple = ()

    @property
    def selected_ids(self):
        return tuple(row["id"] for row in self.operators)

    def __repr__(self):
        return f"SelectedLagrangian(scenario={self.scenario}, operators={self.selected_ids})"

    def add(self, operators, *, show=True):
        """Return a new Lagrangian with more IDs; existing selections stay intact."""
        available = {row["id"] for row in self._source.catalogues[self.scenario - 1]["operators"]}
        ids = set(self.selected_ids) | _ids(operators, available)
        return select_operators(self._source, ids, scenario=self.scenario, show=show)

    def show(self):
        """Display only the equation; coefficient metadata remains available."""
        print(self.latex)


def select_operators(result, operators="all", *, scenario=None, show=True):
    """Construct a symbolic sum without changing the catalogue or writing files."""
    from phenomenology_notation import format_operator

    if type(show) is not bool:
        raise ValueError("show must be True or False.")
    if scenario is None:
        if len(result.catalogues) != 1:
            raise ValueError("Multiple particle-type scenarios are available. Choose one with scenario=1, "
                             "scenario=2, etc.; operators from different models cannot be combined.")
        scenario = 1
    if type(scenario) is not int or not 1 <= scenario <= len(result.catalogues):
        raise ValueError(f"scenario must be an integer from 1 to {len(result.catalogues)}.")
    catalogue = result.catalogues[scenario - 1]
    selected = _ids(operators, {row["id"] for row in catalogue["operators"]})
    rows = tuple(deepcopy(row) for row in catalogue["operators"] if row["id"] in selected)
    coefficients, terms, notation = [], [], []
    for row in rows:
        number = str(int(row["id"][1:]))
        multiplicity = row["multiplicity"]
        formatted = format_operator(result.models[scenario - 1], row,
                                    sm_generations=getattr(result, "sm_generations", 3))
        if formatted is None:
            formatted = _indexed_tensor_expression(row)
        else:
            formatted = dict(formatted, style="standard")
        self_conjugate = row["field_content"] == row["conjugate_content"]
        if formatted["multiplicity"] != multiplicity or formatted["self_conjugate"] != self_conjugate:
            raise ValueError("Operator formatting must preserve contraction counts and conjugation: " + row["id"])
        components = formatted.get("components")
        suffix = "a" if components else formatted.get("coefficient_suffix", "")
        subscript = number + ("," + suffix if suffix else "")
        coefficient = "c_{" + subscript + "}"
        if components:
            if len(components) != multiplicity:
                raise ValueError("Explicit contractions do not match the counted multiplicity: " + row["id"])
            terms.extend(_coupled_term("c_{" + number + "," + str(index) + "}", expression, self_conjugate)
                         for index, expression in enumerate(components, 1))
        else:
            term = _coupled_term(coefficient, formatted["expression"], self_conjugate)
            ranges = formatted.get("index_ranges", ())
            ranges = tuple(ranges.items()) if isinstance(ranges, Mapping) else tuple(ranges)
            counted = 1
            for _, maximum in ranges:
                counted *= maximum
            if counted != multiplicity:
                raise ValueError("Summed coefficient indices do not match the counted multiplicity: " + row["id"])
            for index, maximum in reversed(ranges):
                term = r"\sum_{" + index + "=1}^{" + str(maximum) + "} " + term
            terms.append(term)
        coefficients.append(dict(operator_id=row["id"], kind="real" if self_conjugate else "complex",
                                 count=multiplicity, dimension=row["coefficient_dimension"], latex=coefficient))
        notation.append(dict(formatted, operator_id=row["id"]))
    latex = r"\mathcal{L} = \mathcal{L}_{\mathrm{SM}}"
    if terms:
        latex = (r"\begin{aligned}\mathcal{L} &= \mathcal{L}_{\mathrm{SM}}"
                 + "".join(r"\\ &\quad + " + term for term in terms) + r"\end{aligned}")
    selection = SelectedLagrangian(scenario, rows, tuple(coefficients), latex, result, tuple(notation))
    if show:
        selection.show()
    return selection
