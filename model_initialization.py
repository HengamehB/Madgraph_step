"""Create a fresh, editable model from particle declarations without tools.

Only catalogue construction and the restricted input parser run here. An
existing destination is always replaced without creating a backup;
neither its old coefficients nor machine-specific configuration are reused.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import os
import re
import shlex
from pathlib import Path
from pprint import pformat
import stat
import tempfile
import textwrap

from coefficient_defaults import PROFILE, SM_DEFAULTS, _bare, _tokens, coefficient_default
from model_builder import ModelDefinition


DEFAULT_PARTICLES = Path(__file__).resolve().parent / 'input' / 'sample_particle_sextet.txt'


_PARTICLE_ASSIGNMENTS = {'X', 'Y', 'n_particles', 'scenario', 'exact_dimension',
                         'sm_generations', 'compiler_limits'}


@dataclass(frozen=True)
class InitializationResult:
    path: Path
    # Retained for API compatibility; initialization no longer creates backups.
    backup: Path | None
    operator_count: int
    enabled_count: int
    unresolved: dict
    replaced: bool = False


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    resolved: Path
    identity: tuple | None
    data: bytes | None


@dataclass(frozen=True)
class _RenderedModel:
    text: str
    metadata: dict
    reference: str
    operator_count: int
    enabled_count: int
    unresolved: dict


def _snapshot(path, *, missing=False):
    path = Path(path).expanduser().absolute()
    resolved = path.resolve()
    try:
        before = resolved.stat()
    except FileNotFoundError:
        if missing:
            return _Snapshot(path, resolved, None, None)
        raise
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f'Expected a regular file: {path}')
    data = resolved.read_bytes()
    after = resolved.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
    if identity(before) != identity(after) or path.resolve() != resolved:
        raise RuntimeError(f'File changed while being read: {path}; retry init.')
    return _Snapshot(path, resolved, identity(after), data)


def _unchanged(snapshot):
    try:
        current = _snapshot(snapshot.path, missing=snapshot.identity is None)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f'File changed during init: {snapshot.path}; retry init.') from exc
    if current != snapshot:
        raise RuntimeError(f'File changed during init: {snapshot.path}; retry init.')


def _different_files(source, destination):
    if source.resolved == destination.resolved or (source.identity and destination.identity
            and source.identity[:2] == destination.identity[:2]):
        raise ValueError('Particle source and model destination must be different files, including aliases.')


@contextmanager
def _destination_lock(path):
    # Retain the lock inode: removing it could let a third writer lock another
    # inode while a second writer still holds the original one.
    lock = path.with_name('.' + path.name + '.init.lock')
    with lock.open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another initializer is writing {path}; retry after it finishes.') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _commit(source, destination, content):
    """Check both read snapshots and commit only a fully written replacement."""
    path = destination.resolved
    if not path.parent.is_dir():
        raise FileNotFoundError(f'Model destination directory does not exist: {path.parent}')
    temporary = None
    with _destination_lock(path):
        _unchanged(source)
        _unchanged(destination)
        _different_files(source, destination)
        try:
            descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(content)
                if destination.identity:
                    os.fchmod(stream.fileno(), stat.S_IMODE(destination.identity[-1]))
                stream.flush()
                os.fsync(stream.fileno())
            _unchanged(source)
            _unchanged(destination)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _canonical_free_term(row, info, fields):
    """Select canonical diagonal norms/masses, excluding holomorphic mixing."""
    tokens = _tokens(row)
    names = {_bare(token) for token in tokens}
    if len(tokens) != 2 or len(names) != 1 or not info.get('kind'):
        return False
    field = fields.get(next(iter(names)))
    if field is None:
        return False
    if field.spin == 'fermion':
        if row['derivatives']:
            differentiated = [token[2:-1] for token in tokens if token.startswith('D(')]
            conjugates = [token[:-1] for token in tokens if token.endswith('†')]
            return 'kinetic' in info['kind'] and len(differentiated) == 1 and differentiated == conjugates
        if field.kind == 'dirac':
            return set(tokens) == {field.name + '_L', field.name + '_Rc'}
        return tokens == [field.name, field.name]
    undifferentiated = [token[2:-1] if token.startswith('D(') else token for token in tokens]
    norm = ([field.name, field.name] if field.kind == 'real'
            else [field.name, field.name + '†'])
    return (sorted(undifferentiated) == sorted(norm)
            and (row['derivatives'] == 0 or 'kinetic' in info['kind']))


def _particle(field):
    from lagrangian_engine import colour_display_label, parse_su3
    colour = colour_display_label(field.su3)
    try:
        parse_su3(colour)
    except ValueError:
        # Display labels for higher irreps include prose and are not input
        # syntax. Exact Dynkin pairs also distinguish equal-dimension irreps.
        colour = tuple(field.su3)
    value = dict(spin=field.spin, su2=field.su2, su3=colour,
                 hypercharge=str(field.hypercharge))
    value['fermion_kind' if field.spin == 'fermion' else 'boson_kind'] = field.kind
    return value


def _comment(value, *, indent=''):
    lines = []
    for line in str(value).splitlines() or ['']:
        lines.extend(textwrap.wrap(line, width=112 - len(indent), break_long_words=False,
                                   break_on_hyphens=False) or [''])
    return ''.join(indent + '# ' + line + '\n' for line in lines)


def _assignment(name, value, *, width=112):
    """Keep small settings together without long, one-item-per-line blocks."""
    prefix = name + ' = '
    literal = repr(value)
    if not isinstance(value, dict) or len(prefix + literal) <= width:
        return prefix + literal + '\n'
    lines, current = [prefix + '{'], '    '
    for key, entry in value.items():
        piece = repr(key) + ': ' + repr(entry) + ','
        if len(current) > 4 and len(current + ' ' + piece) > width:
            lines.append(current)
            current = '    '
        current += (' ' if len(current) > 4 else '') + piece
    lines.extend((current, '}'))
    return '\n'.join(lines) + '\n'


_GROUPS = (
    'Mass and kinetic terms',
    'Scalar mixing and tadpoles',
    'Cubic scalar interactions',
    'Higgs interactions',
    'Scalar quartic interactions',
    'Fermion interactions',
    'Derivative interactions',
    'Other interactions',
)


def _operator_group(row, info, fields):
    if _canonical_free_term(row, info, fields):
        return 0
    if row['derivatives']:
        return 6
    names = [_bare(token) for token in _tokens(row)]
    if 'H' in names:
        return 3
    if all(name in fields and fields[name].spin == 'scalar' for name in names):
        return 1 if len(names) <= 2 else 2 if len(names) == 3 else 4
    if any(name in ('Q', 'uc', 'dc', 'L', 'ec', 'u', 'd', 'e') or
           (name in fields and fields[name].spin == 'fermion') for name in names):
        return 5
    return 7


def format_operator_reference(rows):
    """Describe the reviewed bases for the human-readable metadata companion."""
    reference = ('# Operator conventions and suggested values\n'
                 '# This reference is managed with the generated model metadata.\n'
                 '# Only literals and finite arithmetic are parsed; the model file is never executed.\n'
                 '# Only canonical diagonal kinetic and mass terms start enabled. All interactions,\n'
                 '# inter-particle/chiral mixing and holomorphic mass additions start disabled.\n'
                 '# Suggestions are starting inputs, not fitted physics or a guarantee of export support.\n'
                 '# The fixed SM vacuum assumes a 125 GeV unmixed Higgs and zero new scalar VEVs.\n'
                 '# No potential minimization or global-stability test is performed.\n\n')
    for row, info, record in rows:
        key = info['key']
        reference += _comment(f"{info['operator_id']}: {key}; operator dimension {row['dimension']}; coefficient units: {info['units']}.")
        reference += _comment('Normalization: ' + info['hint'] + ' No implicit factorial unless stated.')
        reference += _comment('Hermitian conjugation: ' + row['hc_relation'] + '; coefficient is '
                              + ('real.' if info['real'] else 'real or complex.'))
        reference += _comment('Flavour/contraction entry order (' + str(info['multiplicity']) + '):')
        for index, label in enumerate(info['invariant_labels'], 1):
            reference += _comment(f'{index}. {label}')
        limits = [value for value in (info.get('obstruction'), row.get('notes')) if value]
        if not info.get('kind') and not info.get('obstruction'):
            limits.append('No implemented compiler lowering for this catalogue row.')
        reference += _comment('Limitations: ' + (' '.join(limits) if limits else 'No row-specific compiler limitation reported; full model validation remains required.'))
        reference += _comment('Suggested value: ' + record['reason']) + '\n'
    return reference


def _render(source, destination):
    """Finish all catalogue work and validate generated syntax before writing."""
    from tex_inputs import (_parse, _model_args, _definition, _document, _error,
                            operator_bindings, _validate_coefficient)
    from model_inputs_text import PROVENANCE_SCHEMA
    from pipeline_config import DECAY_DEFAULTS, RANKING_DEFAULTS, EXPLICIT_DECAY_DEFAULTS
    from madgraph_runner import DEFAULT_COLLIDER

    parsed = _parse(source.resolved, source.data.decode('utf-8'), plain=True)
    for name in parsed.assignments:
        if name not in _PARTICLE_ASSIGNMENTS:
            raise _error(parsed.path, parsed.line(name),
                         f'Particle declarations may contain only {", ".join(sorted(_PARTICLE_ASSIGNMENTS))}; found {name!r}.')
    original_args = _model_args(parsed)
    if original_args['scenario'] is None and any(
            isinstance(value, str) and value.strip().lower() == 'any'
            for name in ('X', 'Y') for value in (parsed.assignments.get(name) or {}).values()):
        raise _error(parsed.path, parsed.line('X'),
                     "Particle choices containing 'any' require an explicit scenario; init is noninteractive.")
    inventory = _definition(parsed)
    fields = {field.name: field for field in inventory.fields}
    operators = {row['id']: row for row in inventory.lagrangian.operators}
    infos = inventory.operator_inputs()
    enabled = [info['key'] for info in infos if _canonical_free_term(operators[info['operator_id']], info, fields)]
    particles = {field.name: _particle(field) for field in inventory.fields}
    args = dict(n_particles=len(fields), x=particles['X'], y=particles.get('Y'),
                scenario=1, exact_dimension=original_args['exact_dimension'],
                sm_generations=original_args['sm_generations'], compiler_limits=inventory.compiler_limits)
    active = ModelDefinition(**args, selected_operators=enabled)
    active_infos = {info['key']: info for info in active.operator_inputs()}
    bindings = operator_bindings(active, inventory)
    records, rows, unresolved = {}, {}, {}
    for old_info in infos:
        key = old_info['key']
        info = active_infos.get(key, old_info)
        chosen = key in active_infos
        record = coefficient_default(active if chosen else inventory, info)
        value = record['value']
        if value is not None:
            _validate_coefficient(value, info)
        elif chosen:
            unresolved[key] = record['reason']
        rows[key] = dict(enabled=chosen, coefficient=deepcopy(value))
        record.update(origin='default' if value is not None else 'unresolved',
                      binding=bindings['coefficients'][key], enabled=chosen)
        records[key] = record
    vectors = {name: None for name, field in fields.items() if field.spin == 'vector'}
    for name in vectors:
        unresolved['vector_specifications.' + name] = 'Declare the physical vector theory explicitly before validation.'

    standard_input = destination.resolved.parent == DEFAULT_PARTICLES.parent.resolve()
    model_argument = str(Path('input') / destination.resolved.name) if standard_input else str(destination.resolved)
    text = ('# Edit coefficients and switches below; init replaces this file without a backup.\n'
            '# Masses and mixings are derived. Disabled rows do not enter the Lagrangian.\n'
            '# Electroweak vacuum is fixed: unmixed 125 GeV Higgs, zero VEVs for new scalars.\n'
            '# Full operator conventions: ./run.sh inputs ' + shlex.quote(model_argument) + '\n')
    reference_rows = []
    if fields and all(field.spin == 'scalar' and field.kind == 'real' and field.su2 == 1
                      and tuple(field.su3) == (0, 0) and field.hypercharge == 0 for field in fields.values()):
        text += ('# These neutral singlets have no proton-production channel with only their free terms.\n'
                 '# Enable an SM portal interaction only if it belongs to the intended model.\n')
    text += '\n'
    for name, value in particles.items():
        text += _assignment(name, value)
    text += 'scenario = 1\n'
    text += 'sm_generations = ' + repr(args['sm_generations']) + '\n'
    if args['exact_dimension'] is not None:
        text += 'exact_dimension = ' + repr(args['exact_dimension']) + '\n'
    text += _assignment('compiler_limits', args['compiler_limits']) + '\n'
    grouped = {index: [] for index in range(len(_GROUPS))}
    for old_info in infos:
        key = old_info['key']
        info = active_infos.get(key, old_info)
        row, record = operators[info['operator_id']], records[key]
        reference_rows.append((row, info, record))
        annotation = info['units']
        if info['multiplicity'] > 1:
            annotation += f"; {info['multiplicity']} ordered entries: see reference"
        if rows[key]['enabled'] and rows[key]['coefficient'] is None:
            annotation += '; REQUIRED INPUT UNRESOLVED'
        elif info.get('obstruction') or not info.get('kind'):
            annotation += '; limitations: see reference'
        elif rows[key]['coefficient'] is None:
            annotation += '; unresolved: see reference'
        grouped[_operator_group(row, info, fields)].append(
            '    ' + repr(key) + ': ' + pformat(rows[key], sort_dicts=False, width=1000) + ',  # ' + annotation + '\n')
    text += 'operators = {\n'
    for index, entries in grouped.items():
        if entries:
            text += '    # ' + _GROUPS[index] + '\n' + ''.join(entries) + '\n'
    text += '}\n\n'
    text += ('# Fixed SM vacuum and particle stability\n'
             '# Incompatible tadpoles or Higgs mixing must be removed; coefficients are never silently changed.\n')
    settings = dict(model_name='SM_' + '_'.join(fields), sm_inputs=SM_DEFAULTS,
                    stable_particles=[])
    if vectors:
        text += '# Vector assumptions are physical inputs; None is intentionally unresolved.\n'
        settings['vector_specifications'] = vectors
    for name, value in settings.items():
        text += _assignment(name, value)
    text += '\n# Collider, decays and saved results\n'
    for name, value in dict(collider={key: value for key, value in DEFAULT_COLLIDER.items() if key != 'parton_pdgs'},
                            decays=DECAY_DEFAULTS, ranking=RANKING_DEFAULTS,
                            explicit_decays=EXPLICIT_DECAY_DEFAULTS,
                            output='../output' if standard_input else 'output').items():
        text += _assignment(name, value)
    provenance = dict(schema=PROVENANCE_SCHEMA, profile=PROFILE, coefficients=records, unresolved=unresolved)
    metadata = dict(_input_bindings=bindings, _input_provenance=provenance)
    generated = _parse(destination.resolved, text, plain=True)
    generated.assignments.update(metadata)
    document = _document(generated, active)
    if not unresolved:
        document.build()
    return _RenderedModel(text, metadata, format_operator_reference(reference_rows), len(rows), len(enabled), unresolved)


def _paired_model_path(source):
    """Retain the user's stem when it contains a named particle input."""
    stem = source.stem
    paired = re.sub(r'(?<![a-z0-9])particles?(?![a-z0-9])', 'model', stem,
                    count=1, flags=re.IGNORECASE)
    if paired == stem or stem.lower() in {'particle', 'particles'}:
        return source.with_name('my_model.txt')
    return source.with_name(paired + '.txt')


def initialize_model(particles_path=None, model_path=None):
    """Generate a fresh operator template next to the particle file, or at model_path.

    The bundled sample_particle_sextet.txt pairs with sample_model_sextet.txt.
    Every invocation starts from the declarations and replaces its destination.
    """
    source = _snapshot(DEFAULT_PARTICLES if particles_path is None else particles_path)
    destination = _snapshot(model_path if model_path is not None else _paired_model_path(source.path), missing=True)
    if destination.resolved.suffix.lower() != '.txt':
        raise ValueError('The generated model destination must have a .txt extension.')
    _different_files(source, destination)
    from model_metadata import publish_metadata
    rendered = _render(source, destination)
    _unchanged(source)
    _unchanged(destination)
    reference = publish_metadata(destination.resolved, rendered.metadata, reference=rendered.reference)
    text = rendered.text + ('\n# Managed operator details; keep the companion folder with this file.\n'
                            '_metadata = ' + repr(reference) + '\n')
    _commit(source, destination, text.encode('utf-8'))
    return InitializationResult(destination.resolved, None, rendered.operator_count,
                                rendered.enabled_count, rendered.unresolved, replaced=destination.identity is not None)
