"""Declarative Python inputs in plain TXT or embedded in a TeX document.

``read_tex(path)`` returns a :class:`TexInputs` without running external tools.
Its ``definition`` supports input discovery; ``build()`` validates the helper's
semantic bindings before passing numbers to ``ModelDefinition.build``.
``setup(path)`` reconciles inputs interactively and saves an atomic update without
creating a backup. Only the marked verbatim block is interpreted, and it is never exec'd.

Assignments accept Python literals and finite numeric arithmetic (+, -, *, /,
**) with previously declared numeric constants. Calls, imports, attributes,
indexing, comprehensions and arbitrary Python/TeX execution are not supported.
"""
from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from pprint import pformat
import shutil
import tempfile

from model_builder import ModelDefinition, _row_identity


BEGIN = "% LAGRANGIAN_INPUT_BEGIN"
END = "% LAGRANGIAN_INPUT_END"
BINDING_SCHEMA = "tex-input-bindings-v1"
OPERATOR_BINDING_SCHEMA = "operator-table-bindings-v1"
_SPECIAL = {"X", "Y", "n_particles", "selected_operators", "scenario",
            "exact_dimension", "sm_generations", "physical_inputs", "collider",
            "decays", "ranking", "explicit_decays", "output", "compiler_limits", "_input_bindings",
            "coefficients", "sm_inputs", "sm_scheme", "vacuum", "stable_particles",
            "vector_specifications", "model_name", "vevs", "higgs_mass", "observed_higgs_mass", "calculation", "_input_provenance", "operators", "_metadata"}
_MAX_BYTES = 2_000_000


class TexInputError(ValueError):
    """An input error, including the original source path and line."""


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        return False
    try:
        number = complex(value)
        return math.isfinite(number.real) and math.isfinite(number.imag)
    except (OverflowError, ValueError):
        return False


def _error(path, line, message):
    return TexInputError(f"{path}:{line}: {message}")


class _Evaluator:
    def __init__(self, path, offset, values):
        self.path, self.offset, self.values = path, offset, values

    def error(self, node, message):
        return _error(self.path, self.offset + getattr(node, "lineno", 1), message)

    def evaluate(self, node, depth=0):
        if depth > 100:
            raise self.error(node, "Input nesting is too deep.")
        visit = lambda child: self.evaluate(child, depth + 1)
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None or isinstance(value, (str, bool)):
                return value
            if _finite(value):
                return value
            raise self.error(node, "Expected a string, None, boolean or finite number.")
        if isinstance(node, ast.Name):
            if node.id not in self.values or not _finite(self.values[node.id]):
                raise self.error(node, f"{node.id!r} must name a previously declared numeric constant.")
            return self.values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [visit(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    raise self.error(node, "Dictionary unpacking is not supported.")
                key = visit(key_node)
                if isinstance(key, bool) or not isinstance(key, (str, int)):
                    raise self.error(key_node, "Dictionary keys must be strings or integers.")
                if key in result:
                    raise self.error(key_node, f"Duplicate dictionary key {key!r}.")
                result[key] = visit(value_node)
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if not _finite(value):
                raise self.error(node, "Unary arithmetic requires a finite number.")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = visit(node.left), visit(node.right)
            if not _finite(left) or not _finite(right):
                raise self.error(node, "Arithmetic operands must be finite numbers.")
            if isinstance(node.op, ast.Pow) and (isinstance(right, complex) or abs(right) > 1024):
                raise self.error(node, "Exponents must be real numbers between -1024 and 1024.")
            try:
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                elif isinstance(node.op, ast.Div):
                    value = left / right
                else:
                    value = left ** right
            except (ArithmeticError, ValueError) as exc:
                raise self.error(node, f"Invalid numeric expression: {exc}") from exc
            if not _finite(value):
                raise self.error(node, "Numeric expression must produce a finite value.")
            return value
        raise self.error(node, "Only literals and numeric arithmetic are allowed; Python code is not executed.")


@dataclass
class _Source:
    path: Path
    text: str
    block: str
    block_start: int
    block_end: int
    line_offset: int
    assignments: dict
    nodes: dict
    metadata_bytes: bytes | None = None
    metadata_path: Path | None = None
    operator_reference: str = ''

    def line(self, name="physical_inputs"):
        return self.line_offset + getattr(self.nodes.get(name), "lineno", 1)


def _parse(path, text=None, *, plain=None):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8") if text is None else text
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise _error(path, 1, "Input file exceeds the 2 MB limit.")
    lines = text.splitlines(keepends=True)
    if plain is None:
        plain = path.suffix.lower() == '.txt'
    if plain:
        first, last = 0, len(lines)
    else:
        starts = [i for i, line in enumerate(lines) if line.strip() == BEGIN]
        ends = [i for i, line in enumerate(lines) if line.strip() == END]
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            raise _error(path, 1, f"Use exactly one {BEGIN!r} / {END!r} block.")
        first, last = starts[0] + 1, ends[0]
        while first < last and not lines[first].strip():
            first += 1
        while last > first and not lines[last - 1].strip():
            last -= 1
        if first >= last or lines[first].strip() != r"\begin{verbatim}" or lines[last - 1].strip() != r"\end{verbatim}":
            raise _error(path, first + 1, "The input markers must enclose one verbatim environment.")
        first, last = first + 1, last - 1
    block = "".join(lines[first:last])
    try:
        tree = ast.parse(block, filename=str(path), mode="exec")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise _error(path, first + (getattr(exc, "lineno", None) or 1),
                     f"Invalid input syntax: {getattr(exc, 'msg', str(exc))}") from exc
    if sum(1 for _ in ast.walk(tree)) > 50000:
        raise _error(path, first + 1, "Too many input expressions.")
    values, nodes = {}, {}
    evaluator = _Evaluator(path, first, values)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise evaluator.error(node, "Use one name = value assignment per declaration.")
        name = node.targets[0].id
        if name in nodes:
            raise evaluator.error(node, f"Duplicate assignment to {name!r}.")
        if name.startswith("_") and name not in ("_input_bindings", "_input_provenance", "_metadata"):
            raise evaluator.error(node, "Names beginning with '_' are reserved for input metadata.")
        value = evaluator.evaluate(node.value)
        if name not in _SPECIAL and not _finite(value):
            raise evaluator.error(node, f"Unknown input {name!r}; additional names must be numeric constants.")
        values[name], nodes[name] = value, node
    source = _Source(path, text, block, sum(map(len, lines[:first])), sum(map(len, lines[:last])),
                     first, values, nodes)
    if '_metadata' in values:
        if any(name in values for name in ('_input_bindings', '_input_provenance')):
            raise _error(path, source.line('_metadata'),
                         'Use either companion metadata or inline input metadata, not both.')
        from model_metadata import load_metadata
        try:
            metadata, source.metadata_bytes, source.metadata_path = load_metadata(path, values['_metadata'])
        except (OSError, ValueError) as exc:
            raise _error(path, source.line('_metadata'), f'Cannot read model metadata: {exc}') from exc
        source.operator_reference = metadata.get('operator_reference', '')
        for name in ('_input_bindings', '_input_provenance'):
            if name in metadata:
                values[name] = metadata[name]
    return source


def _model_args(source):
    values = source.assignments
    if not isinstance(values.get("X"), dict):
        raise _error(source.path, source.line("X"), "X must be a particle dictionary.")
    y = values.get("Y")
    if y is not None and not isinstance(y, dict):
        raise _error(source.path, source.line("Y"), "Y must be a particle dictionary or None.")
    count = 1 if y is None else 2
    if "n_particles" in values and (type(values["n_particles"]) is not int or values["n_particles"] != count):
        raise _error(source.path, source.line("n_particles"), "n_particles must agree with the supplied X/Y dictionaries.")
    physical = values.get('physical_inputs', {})
    selected = values.get('selected_operators', 'all')
    if 'operators' in values:
        selected = [key for key, entry in operator_rows(source).items() if entry['enabled']]
    nested_limits = physical.get('compiler_limits') if isinstance(physical, dict) else None
    if 'compiler_limits' in values and nested_limits is not None and values['compiler_limits'] != nested_limits:
        raise _error(source.path, source.line('compiler_limits'),
                     'Conflicting top-level and physical_inputs compiler_limits; use one assignment.')
    from compiler_limits import normalize_limits
    try:
        budgets = normalize_limits(values.get('compiler_limits', nested_limits))
    except ValueError as exc:
        raise _error(source.path, source.line('compiler_limits' if 'compiler_limits' in values else 'physical_inputs'), str(exc)) from exc
    return dict(n_particles=count, x=values["X"], y=y,
                selected_operators=selected,
                scenario=values.get("scenario"), exact_dimension=values.get("exact_dimension"),
                sm_generations=values.get("sm_generations", 3), compiler_limits=budgets)


def _definition(source):
    try:
        return ModelDefinition(**_model_args(source))
    except TexInputError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise _error(source.path, source.line("X"), str(exc)) from exc


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _compiler_fingerprint(definition):
    """Conservative fallback until every backend publishes its basis identity."""
    names = {"model_builder.py", "lagrangian_engine.py", "component_invariants.py",
             "group_theory.py", "feynrules_export.py"}
    root = Path(__file__).resolve().parent
    # Dispatch overlaps intentionally preserve some old conventions. Hash the
    # complete compiler dependency set so a legacy supports_fields predicate
    # cannot omit a generic backend that actually handles this model.
    names.update(path.name for path in root.glob('*compiler.py') if not path.name.startswith('test_'))
    names.update(path.name for path in root.glob('generic_*.py'))
    names.update(('native_colour_basis.py', 'compiler_limits.py', 'joint_scalar_invariants.py', 'fixed_sm.py', 'sm_parameters.py'))
    return _hash({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in sorted(names) if (root / name).is_file()})


# The two published revisions and their local benchmark builds were reviewed
# together. The local exporters differ from their published counterparts only
# in error diagnostics. New Yukawa bases still change their individual hashes.
# A future compiler revision is gated again until its changes are reviewed.
_BENCHMARK_RELEASE_COMPILER = '3a7f49cf319954f05e008e8e6c024810ef0d40eed6592a2e2efa19059267ce45'
_REVIEWED_BINDING_COMPILERS = {
    '7938716385fbb1cf7aa95377f9241a0f424fcac9b7900c7f293f2ea4a8a5b7fd',  # preceding release
    'a15bda2685d72d40a99931ab988184c309a5c50780be866c73812f4754ef78b5',  # local preceding release
    '3ea030404748c216c13c6a8712844a472b66107aab3af7085e283af5b5fdef08',  # five measured local runs
}


def input_bindings(definition, *, recorded=None):
    """Describe coefficient meanings independently of display IDs and values.

    The complete field context is conservative: adding/changing a multiplet
    requires review even if one readable operator key happens to remain equal.
    Selection-only changes retain identities for unchanged operators. The
    convention/basis schema must change when compiler coefficient order changes.
    """
    fields = [dict(name=f.name, spin=f.spin, kind=f.kind, su2=f.su2,
                   su3=f.su3, hypercharge=str(f.hypercharge)) for f in definition.fields]
    context = dict(fields=fields, sm_generations=definition.result.sm_generations)
    convention = _compiler_fingerprint(definition)
    stored_compiler = recorded.get('compiler_fingerprint') if isinstance(recorded, dict) else None
    if (convention == _BENCHMARK_RELEASE_COMPILER and isinstance(stored_compiler, str)
            and stored_compiler in _REVIEWED_BINDING_COMPILERS):
        convention = stored_compiler
    metadata = {row["operator_id"]: row for row in definition.operator_inputs()}
    coefficients = {}
    for row in definition.lagrangian.operators:
        info = metadata[row["id"]]
        meaning = {key: info[key] for key in ("units", "multiplicity", "real", "add_hc", "invariant_labels", "kind")}
        for key in ("basis_fingerprint", "convention_fingerprint"):
            if key in info:
                meaning[key] = info[key]
        coefficients[info["key"]] = _hash(dict(schema=BINDING_SCHEMA, context=context,
                                              operator=_row_identity(row), meaning=meaning,
                                              compiler_fingerprint=convention))
    return dict(schema=BINDING_SCHEMA, model_signature=definition.signature,
                context=_hash(context), compiler_fingerprint=convention,
                operator_ids={row["id"]: definition.coefficient_key(row["id"])
                              for row in definition.lagrangian.operators},
                coefficients=coefficients)


def operator_rows(source):
    """Validate the operator-table structure, without interpreting dormant values."""
    values = source.assignments
    rows = values.get('operators')
    physical = values.get('physical_inputs', {})
    if ('selected_operators' in values or 'coefficients' in values
            or isinstance(physical, dict) and 'coefficients' in physical):
        raise _error(source.path, source.line('operators'),
                     'operators cannot be combined with selected_operators or a separate coefficients dictionary.')
    if not isinstance(rows, dict):
        raise _error(source.path, source.line('operators'), 'operators must be a dictionary keyed by readable operator names.')
    for key, entry in rows.items():
        if not isinstance(key, str) or not isinstance(entry, dict) or set(entry) != {'enabled', 'coefficient'}:
            raise _error(source.path, source.line('operators'),
                         f'Operator {key!r} must contain exactly enabled and coefficient.')
        if type(entry['enabled']) is not bool:
            raise _error(source.path, source.line('operators'), f'Operator {key!r}: enabled must be True or False.')
    return rows


def operator_inventory(definition):
    """View the full formal catalogue without repeating particle enumeration."""
    complete = copy(definition)
    complete.lagrangian = definition.result.select('all', scenario=definition.lagrangian.scenario, show=False)
    return complete


def operator_bindings(active_definition, inventory_definition=None, *, recorded=None):
    """Bind every catalogue row, overriding active rows with their actual basis.

    Compiler dispatch can depend on the selected operators. Dormant entries use
    the complete-catalogue convention until selected; a different active basis
    correctly requires review rather than silently reinterpreting a number.
    The aggregate selected signature is deliberately not a validation gate.
    """
    inventory = inventory_definition or operator_inventory(active_definition)
    complete = input_bindings(inventory, recorded=recorded)
    active = input_bindings(active_definition, recorded=recorded)
    if complete['context'] != active['context']:
        raise ValueError('Operator inventory and active definition describe different fields.')
    if not set(active['coefficients']).issubset(complete['coefficients']):
        raise ValueError('Operator inventory must include every active operator.')
    return dict(schema=OPERATOR_BINDING_SCHEMA, context=complete['context'],
                compiler_fingerprint=complete['compiler_fingerprint'],
                catalogue_fingerprint=_hash([_row_identity(row) for row in inventory.lagrangian.operators]),
                operator_ids=complete['operator_ids'],
                coefficients={**complete['coefficients'], **active['coefficients']})


def _check_operator_bindings(document):
    stored = document.assignments.get('_input_bindings', {})
    expected = input_bindings(document.definition, recorded=stored)
    valid = (isinstance(stored, dict) and stored.get('schema') == OPERATOR_BINDING_SCHEMA
             and all(stored.get(key) == expected[key] for key in ('context', 'compiler_fingerprint')))
    meanings = stored.get('coefficients', {}) if isinstance(stored, dict) else {}
    valid = valid and isinstance(meanings, dict)
    stale = [key for key, meaning in expected['coefficients'].items()
             if not valid or meanings.get(key) != meaning]
    if not valid or stale:
        suffix = ': ' + ', '.join(stale) if stale else ''
        raise _error(document.path, document._source.line('operators'),
                     'Missing or stale active operator bindings; run setup to review changed coefficient meanings' + suffix + '.')


@dataclass
class TexInputs:
    path: Path
    definition: ModelDefinition
    physical_inputs: dict
    collider: dict
    decays: dict
    ranking: dict
    output: str | None
    assignments: dict
    _source: _Source = field(repr=False)
    explicit_decays: dict = field(default_factory=dict)

    @property
    def values(self):
        """Alias for integrations that need all evaluated assignments."""
        return self.assignments

    @property
    def source(self):
        return self._source.text

    @property
    def provenance(self):
        """Input origins for reporting; never a numerical physics input."""
        from model_inputs_text import input_provenance
        return input_provenance(self)

    def build(self):
        """Validate meanings before binding the numbers to this catalogue."""
        if 'operators' in self.assignments:
            _check_operator_bindings(self)
        elif self.assignments.get("_input_bindings") != input_bindings(self.definition, recorded=self.assignments.get("_input_bindings")):
            raise _error(self.path, self._source.line("physical_inputs"),
                         "Missing or stale input bindings: run setup for this input file to review changed coefficient meanings.")
        values = deepcopy(self.physical_inputs)
        saved = values.get("model_signature")
        if saved is not None and saved != self.definition.signature:
            raise _error(self.path, self._source.line("physical_inputs"),
                         "Stale model_signature; run setup. Signatures are maintained automatically.")
        values["model_signature"] = self.definition.signature
        values['compiler_limits'] = self.definition.compiler_limits
        manual = values.get('feynrules')
        if isinstance(manual, dict) and isinstance(manual.get('model_files'), (list, tuple)):
            resolved = []
            for name in manual['model_files']:
                if not isinstance(name, str):
                    raise _error(self.path, self._source.line(), 'feynrules.model_files must list file paths.')
                path = Path(name).expanduser()
                resolved.append(str((path if path.is_absolute() else self.path.parent / path).resolve()))
            manual['model_files'] = resolved
        try:
            return self.definition.build(values)
        except (ValueError, TypeError) as exc:
            raise _error(self.path, self._source.line("physical_inputs"), str(exc)) from exc


def _document(source, definition=None):
    definition = definition or _definition(source)
    mappings = {}
    for name in ("physical_inputs", "collider", "decays", "ranking", "explicit_decays"):
        value = source.assignments.get(name, {})
        if not isinstance(value, dict):
            raise _error(source.path, source.line(name), f"{name} must be a dictionary.")
        mappings[name] = deepcopy(value)
    if 'operators' in source.assignments:
        rows = operator_rows(source)
        known = set(definition._ids)
        unknown = set(rows) - known
        if unknown:
            raise _error(source.path, source.line('operators'),
                         'Unknown readable operator keys: ' + ', '.join(sorted(unknown)) + '.')
        # Table bindings check each active meaning independently. A copied
        # legacy aggregate signature must not obstruct an ordinary toggle.
        mappings['physical_inputs'].pop('model_signature', None)
        mappings['physical_inputs']['coefficients'] = {
            key: deepcopy(entry['coefficient']) for key, entry in rows.items() if entry['enabled']}
    from model_inputs_text import normalize_inputs
    mappings = normalize_inputs(source, definition, mappings)
    output = source.assignments.get("output")
    if output is not None and (not isinstance(output, str) or not output.strip()):
        raise _error(source.path, source.line("output"), "output must be a nonempty directory string or None.")
    return TexInputs(source.path, definition, output=output, assignments=deepcopy(source.assignments),
                     _source=source, **mappings)


def read_tex(path):
    """Read the model/settings; incomplete numerical inputs are discoverable."""
    return _document(_parse(path, plain=False))


def read_inputs(path):
    """Read a declarative .txt/.tex file without executing any Python code."""
    suffix = Path(path).suffix.lower()
    if suffix not in ('.txt', '.tex'):
        raise _error(Path(path), 1, 'Declarative inputs require a .txt or .tex extension.')
    return read_tex(path) if suffix == '.tex' else _document(_parse(path))


def validate_inputs(path):
    document = read_inputs(path)
    document.build()
    return document


def discover_inputs(path):
    document = read_inputs(path)
    if 'operators' in document.assignments:
        return _discover_operator_table(document)
    return dict(required_inputs=document.definition.required_inputs(),
                operators=document.definition.operator_inputs(),
                bindings=input_bindings(document.definition, recorded=document.assignments.get('_input_bindings')), provenance=document.provenance)


def _discover_operator_table(document):
    """Describe switches without adopting any previously unreviewed values.

    ``bindings`` is the recorded metadata, never a freshly blessed replacement.
    Current active meanings are separate comparison data. Dormant rows display
    the full-catalogue convention, which may differ when a row is enabled.
    """
    active = {info['key']: info for info in document.definition.operator_inputs()}
    stored = document.assignments.get('_input_bindings', {})
    expected = input_bindings(document.definition, recorded=stored)
    stored = deepcopy(stored) if isinstance(stored, dict) and stored.get('schema') == OPERATOR_BINDING_SCHEMA else {}
    meanings = stored.get('coefficients', {})
    meanings = meanings if isinstance(meanings, dict) else {}
    same_context = all(stored.get(key) == expected[key] for key in ('context', 'compiler_fingerprint'))
    table = document.assignments['operators']
    rows = []
    for info in operator_inventory(document.definition).operator_inputs():
        key = info['key']
        entry = table.get(key, {})
        enabled = entry.get('enabled', False)
        meaning = meanings.get(key)
        status = ('verified' if same_context and meaning == expected['coefficients'].get(key) else
                  'unbound' if meaning is None else 'stale') if enabled else ('not_checked' if meaning else 'unbound')
        rows.append(dict(active.get(key, info), enabled=enabled, provided=key in table,
                         coefficient=deepcopy(entry.get('coefficient')), binding=meaning,
                         binding_status=status, convention_scope='active' if enabled else 'full_catalogue'))
    return dict(required_inputs=document.definition.required_inputs(), operators=rows,
                bindings=stored, expected_active_bindings=expected, provenance=document.provenance)


def validate_tex(path):
    """Validate parsed inputs and semantic bindings without external tools."""
    document = read_tex(path)
    document.build()
    return document


def discover_tex(path):
    """Return required constants and their conventions without a physics job."""
    document = read_tex(path)
    if 'operators' in document.assignments:
        return _discover_operator_table(document)
    return dict(required_inputs=document.definition.required_inputs(),
                operators=document.definition.operator_inputs(),
                bindings=input_bindings(document.definition, recorded=document.assignments.get('_input_bindings')))


def _node_spans(source):
    """AST byte columns are UTF-8 byte offsets, not Python string offsets."""
    lines = source.block.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def position(line, column):
        prefix = lines[line - 1].encode("utf-8")[:column].decode("utf-8")
        return offsets[line - 1] + len(prefix)

    return {name: (position(node.value.lineno, node.value.col_offset),
                   position(node.value.end_lineno, node.value.end_col_offset))
            for name, node in source.nodes.items()}


def _render_changes(source, changes):
    block = source.block
    spans = _node_spans(source)
    edits, additions = [], []
    lines = block.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def position(line, column):
        return offsets[line - 1] + len(lines[line - 1].encode('utf-8')[:column].decode('utf-8'))

    def render_value(node, value, indent=0):
        # Preserve named constants/mass expressions even when another entry in
        # the same dictionary must change. This never evaluates arbitrary code.
        previous = _Evaluator(source.path, source.line_offset, source.assignments).evaluate(node)
        if previous == value:
            return ast.get_source_segment(source.block, node)
        if isinstance(node, ast.Dict) and isinstance(value, dict):
            old_nodes = {_Evaluator(source.path, source.line_offset, source.assignments).evaluate(key): item
                         for key, item in zip(node.keys, node.values)}
            if list(old_nodes) == list(value):
                # Changing one coefficient must not remove the generated
                # contraction notes or the user's comments on its neighbours.
                start = position(node.lineno, node.col_offset)
                expression = ast.get_source_segment(source.block, node)
                replacements = [(position(child.lineno, child.col_offset) - start,
                                 position(child.end_lineno, child.end_col_offset) - start,
                                 render_value(child, value[key], indent + 4))
                                for key, child in old_nodes.items()]
                for first, last, replacement in sorted(replacements, reverse=True):
                    expression = expression[:first] + replacement + expression[last:]
                return expression
            pieces = []
            for key, item in value.items():
                expression = (render_value(old_nodes[key], item, indent + 4) if key in old_nodes
                              else pformat(item, sort_dicts=False, width=100))
                pieces.append(" " * (indent + 4) + repr(key) + ": " + expression + ",")
            return "{\n" + "\n".join(pieces) + "\n" + " " * indent + "}"
        return pformat(value, sort_dicts=False, width=100)

    for name, value in changes.items():
        replacement = (render_value(source.nodes[name].value, value) if name in source.nodes
                       else pformat(value, sort_dicts=False, width=100))
        if name in spans:
            start, end = spans[name]
            edits.append((start, end, replacement))
        else:
            additions.append(f"{name} = {replacement}\n")
    for start, end, replacement in sorted(edits, reverse=True):
        block = block[:start] + replacement + block[end:]
    if additions:
        block = block.rstrip("\r\n") + "\n\n" + "\n".join(additions)
    return source.text[:source.block_start] + block + source.text[source.block_end:]


def _prompt_value(label, source, input_fn, output_fn, *, default=None, has_default=False, validator=None):
    while True:
        suffix = f" [{default!r}]" if has_default else ""
        answer = input_fn(label + suffix + ": ").strip()
        if not answer and has_default:
            value = deepcopy(default)
        else:
            try:
                tree = ast.parse(answer, mode="eval")
                value = _Evaluator(source.path, source.line_offset, source.assignments).evaluate(tree.body)
            except (SyntaxError, TexInputError, ValueError) as exc:
                output_fn(str(exc))
                continue
        try:
            if validator:
                validator(value)
        except (ValueError, TypeError) as exc:
            output_fn(str(exc))
            continue
        return value


def _validate_coefficient(value, row):
    from feynrules_export import _number
    multiple = row["multiplicity"] != 1
    entries = value if multiple else [value]
    if not isinstance(entries, (list, tuple)) or len(entries) != row["multiplicity"]:
        raise ValueError(f"Enter {row['multiplicity']} coefficients in the displayed order.")
    for index, entry in enumerate(entries):
        _number(entry, f"{row['key']}[{index}]", real=row["real"])


def _write_atomic(source, updated):
    """Save without backups, checking for concurrent edits; return whether changed."""
    if updated == source.text:
        return False
    def unchanged():
        if source.path.read_text(encoding="utf-8") != source.text:
            raise _error(source.path, 1, "Input changed during setup; rerun setup to preserve your edits.")
        if source.metadata_path is not None:
            from model_metadata import load_metadata
            try:
                _, content, path = load_metadata(source.path, source.assignments['_metadata'])
            except (OSError, ValueError) as exc:
                raise _error(source.path, 1, 'Metadata changed during setup; original input was preserved.') from exc
            if path != source.metadata_path or content != source.metadata_bytes:
                raise _error(source.path, 1, 'Metadata changed during setup; original input was preserved.')
    unchanged()
    descriptor, name = tempfile.mkstemp(prefix="." + source.path.name + ".", suffix=".tmp", dir=source.path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        shutil.copymode(source.path, temporary)
        unchanged()
        os.replace(temporary, source.path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def setup(path, *, input_fn=input, output_fn=print):
    """Discover/reconcile constants, prompt for missing values, and save safely.

    No external tools run. Enter numeric literals/expressions or literal arrays
    at prompts. Existing coefficient expressions remain untouched if all their
    semantic identities and values remain valid. EOF/interrupt leaves the input
    unchanged. Unsupported physical sectors are reported, never declared solved.
    """
    if Path(path).suffix.lower() == '.txt':
        from model_inputs_text import setup_text
        return setup_text(path, input_fn=input_fn, output_fn=output_fn)
    source = _parse(path)
    if 'operators' in source.assignments:
        from model_inputs_text import setup_text
        return setup_text(path, input_fn=input_fn, output_fn=output_fn)
    changes = {}
    arguments = _model_args(source)
    if arguments["scenario"] is None:
        from lagrangian_engine import expand_models, model_label
        from lagrangian_maker import _field_input
        fields = [_field_input(arguments["x"], "X")]
        if arguments["y"] is not None:
            fields.append(_field_input(arguments["y"], "Y"))
        scenarios = tuple(expand_models(fields))
        if len(scenarios) > 1:
            for index, model in enumerate(scenarios, 1):
                output_fn(f"{index}: {model_label(model)}")

            def valid_scenario(value):
                if type(value) is not int or not 1 <= value <= len(scenarios):
                    raise ValueError(f"Choose an integer from 1 to {len(scenarios)}.")

            choice = _prompt_value("Scenario", source, input_fn, output_fn, validator=valid_scenario)
            arguments["scenario"] = choice
            source.assignments["scenario"] = choice
            changes["scenario"] = choice
    try:
        definition = ModelDefinition(**arguments)
    except (ValueError, TypeError) as exc:
        raise _error(source.path, source.line("X"), str(exc)) from exc
    document = _document(source, definition)
    expected = input_bindings(definition, recorded=source.assignments.get('_input_bindings'))
    old_bindings = source.assignments.get("_input_bindings")
    old_bindings = old_bindings if isinstance(old_bindings, dict) else {}
    old_coefficients = old_bindings.get("coefficients", {}) if old_bindings.get("schema") == BINDING_SCHEMA else {}
    old_coefficients = old_coefficients if isinstance(old_coefficients, dict) else {}
    template = definition.required_inputs()
    original = document.physical_inputs
    supplied = original.get("coefficients", {})
    if not isinstance(supplied, dict):
        raise _error(source.path, source.line(), "physical_inputs.coefficients must be a dictionary.")
    values = deepcopy(original)
    values.pop("model_signature", None)
    values.setdefault("model_name", template["model_name"])
    coefficients = {}
    for row in definition.operator_inputs():
        key = row["key"]
        value = supplied.get(key)
        output_fn(f"{key}: {row['units']}; {'real' if row['real'] else 'real or complex'}. {row['hint']}")
        output_fn(("Contraction: " if row["multiplicity"] == 1 else "Entry order: ")
                  + ", ".join(row["invariant_labels"]))
        matches = old_coefficients.get(key) == expected["coefficients"][key]
        if matches:
            try:
                _validate_coefficient(value, row)
                coefficients[key] = value
                output_fn(f"Keeping verified value: {value!r}")
                continue
            except (ValueError, TypeError):
                pass
        if key in supplied and not matches:
            output_fn(f"Previous unbound/changed value {value!r}; enter it explicitly to adopt it for this meaning.")
        coefficients[key] = _prompt_value(key, source, input_fn, output_fn,
                                         validator=lambda value, row=row: _validate_coefficient(value, row))
    values["coefficients"] = coefficients
    same_context = old_bindings.get("schema") == BINDING_SCHEMA and old_bindings.get("context") == expected["context"]
    from feynrules_export import _number
    existing_widths = original.get("widths", {})
    if not isinstance(existing_widths, dict):
        raise _error(source.path, source.line("physical_inputs"), "physical_inputs.widths must be a dictionary.")
    widths = {}
    for name, default in template["widths"].items():
        def valid_width(value, name=name):
            if _number(value, "widths." + name, real=True) < 0:
                raise ValueError("Initial widths must be nonnegative.")
        if same_context and name in existing_widths:
            try:
                valid_width(existing_widths[name])
                widths[name] = existing_widths[name]
                continue
            except (ValueError, TypeError):
                pass
        widths[name] = _prompt_value(f"widths.{name} [GeV]", source, input_fn, output_fn,
                                     default=default, has_default=True, validator=valid_width)
    values["widths"] = widths
    def valid_stable(value):
        if not isinstance(value, (list, tuple)) or any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in value):
            raise ValueError("Enter a list of physical particle names or integer PDG codes.")

    declarations = source.assignments.get("physical_inputs", {})
    missing_stability = "stable_particles" not in declarations and "stable_particles" not in source.assignments
    if missing_stability or not same_context and values["stable_particles"]:
        output_fn("Declare stable new particles explicitly; [] declares none. A calculated zero width alone does not establish stability.")
        values["stable_particles"] = _prompt_value("stable_particles (physical names/PDGs)", source,
                                                   input_fn, output_fn, default=[], has_default=True,
                                                   validator=valid_stable)
    try:
        valid_stable(values["stable_particles"])
    except ValueError:
        values["stable_particles"] = _prompt_value("stable_particles (physical names/PDGs)", source,
                                                   input_fn, output_fn, default=[], has_default=True,
                                                   validator=valid_stable)
    if "vector_specifications" in template:
        specifications = values.get("vector_specifications", {})
        specifications = specifications if isinstance(specifications, dict) else {}
        updated = {}
        for name in template["vector_specifications"]:
            def valid_vector(value):
                if not isinstance(value, dict) or not value:
                    raise ValueError("Enter a nonempty vector specification dictionary.")

            if same_context and isinstance(specifications.get(name), dict) and specifications[name]:
                updated[name] = specifications[name]
            else:
                output_fn(f"{name} requires a physical vector specification; {{'kind': 'proca'}} selects a massive effective matter vector.")
                updated[name] = _prompt_value("vector_specifications." + name, source, input_fn, output_fn,
                                              validator=valid_vector)
        values["vector_specifications"] = updated
    elif "vector_specifications" in values:
        del values["vector_specifications"]
    # Discovery diagnostics are derived, never manually edited or trusted input.
    if "unresolved" in values:
        values["unresolved"] = template.get("unresolved", {})
    for key, reason in template.get("unresolved", {}).items():
        output_fn(f"Export unresolved: {key}: {reason}")
    if values != original or missing_stability:
        nested = deepcopy(declarations)
        if missing_stability:
            nested["stable_particles"] = values["stable_particles"]
        nested.pop("model_signature", None)
        for key, value in values.items():
            if key not in original or value != original[key]:
                nested[key] = value
        changes["physical_inputs"] = nested
    if source.assignments.get("_input_bindings") != expected:
        changes["_input_bindings"] = expected
    if 'max_histories' in document.ranking:
        changes['ranking'] = {key: value for key, value in document.ranking.items() if key != 'max_histories'}
        output_fn('Removed deprecated ranking.max_histories; the complete history catalogue is streamed to disk.')
    updated_text = _render_changes(source, changes)
    updated_document = _document(_parse(source.path, updated_text))
    updated_document.build()
    changed = _write_atomic(source, updated_text)
    output_fn(f"Inputs ready: {source.path}" + ("" if changed else " (unchanged)"))
    return updated_document
