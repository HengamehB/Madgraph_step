"""Versioned, checksummed JSON checkpoints; no cached code or executable loading."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from run_paths import contained_path
import csv
import fcntl
import hashlib
import json
import math
import os
import tempfile

SCHEMA = 'madgraph-step-artifact-v1'


@dataclass(frozen=True)
class JsonArray:
    """One-pass JSON array; native rows have schema-validated string keys.

    Native mode lets the C encoder traverse ordinary containers directly.
    General callers retain ``public``'s stringified-mapping-key convention.
    """
    values: object
    native: bool = False


def _history_rows(value):
    return type(value).__name__ == 'SQLiteHistoryRows' and type(value).__module__ == 'history_store'


def iter_json(value, *, indent=None, sort_keys=True, _depth=0):
    """Yield JSON fragments with memory bounded by one row and nesting depth.

    Unlike ``public`` this never copies a dataclass's row collection. Scalar
    encoding still uses Python's JSON encoder, including its finite-number
    checks and correct Unicode/string escaping.
    """
    if isinstance(value, Path):
        value = str(value)
    elif isinstance(value, Fraction):
        value = str(value)
    elif isinstance(value, complex):
        value = {'real': value.real, 'imag': value.imag}
    elif is_dataclass(value) and not isinstance(value, JsonArray):
        value = {f.name: getattr(value, f.name) for f in fields(value) if not f.name.startswith('_')}
    if isinstance(value, dict):
        entries = {str(key): item for key, item in value.items()}
        if len(entries) != len(value):
            raise ValueError('JSON mapping has colliding stringified keys')
        keys = sorted(entries) if sort_keys else entries
        yield '{'
        first = True
        for key in keys:
            if not first:
                yield ','
            if indent is not None:
                yield '\n' + ' ' * (indent * (_depth + 1))
            yield json.dumps(key, ensure_ascii=False)
            yield ': ' if indent is not None else ':'
            yield from iter_json(entries[key], indent=indent, sort_keys=sort_keys, _depth=_depth + 1)
            first = False
        if not first and indent is not None:
            yield '\n' + ' ' * (indent * _depth)
        yield '}'
    elif isinstance(value, (JsonArray, list, tuple, set, frozenset)) or _history_rows(value):
        iterable = value.values if isinstance(value, JsonArray) else value
        bounded_rows = isinstance(value, JsonArray) or _history_rows(value)
        native_rows = _history_rows(value) or isinstance(value, JsonArray) and value.native
        yield '['
        first = True
        for item in iterable:
            if not first:
                yield ','
            if indent is not None:
                yield '\n' + ' ' * (indent * (_depth + 1))
            if bounded_rows:
                # An indexed row/node is already bounded. Use the C encoder on
                # that one record, not on the collection of millions of rows.
                yield json.dumps(item if native_rows else public(item), default=_native_default,
                                 ensure_ascii=False, sort_keys=sort_keys,
                                 allow_nan=False, separators=(',', ':'))
            else:
                yield from iter_json(item, indent=indent, sort_keys=sort_keys, _depth=_depth + 1)
            first = False
        if not first and indent is not None:
            yield '\n' + ' ' * (indent * _depth)
        yield ']'
    elif hasattr(value, 'tolist'):
        # Numerical spectrum arrays are small, but keep larger arrays streamed
        # as well. NumPy scalars have ndim=0 and need scalar conversion.
        converted = JsonArray(value) if getattr(value, 'ndim', 0) else value.tolist()
        yield from iter_json(converted, indent=indent, sort_keys=sort_keys, _depth=_depth)
    elif value is None or isinstance(value, (bool, int, float, str)):
        yield json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    else:
        raise TypeError(f'Cannot serialize {type(value).__name__}')


def public(value):
    """Readable, lossless numerical JSON (complex numbers use real/imag pairs)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, Fraction):
        return str(value)
    if is_dataclass(value):
        return {f.name: public(getattr(value, f.name)) for f in fields(value) if not f.name.startswith('_')}
    if isinstance(value, dict):
        return {str(k): public(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [public(v) for v in value]
    if hasattr(value, 'tolist'):
        return public(value.tolist())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _native_default(value):
    """Convert only exceptional values; leave nested native containers intact."""
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in fields(value) if not f.name.startswith('_')}
    if isinstance(value, (set, frozenset)):
        return list(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return public(value)


def _row_mapping(row):
    if is_dataclass(row):
        return {f.name: getattr(row, f.name) for f in fields(row) if not f.name.startswith('_')}
    if isinstance(row, dict):
        return row
    raise TypeError(f'CSV row must be a mapping or dataclass, not {type(row).__name__}')


def digest(value):
    result = hashlib.sha256()
    for chunk in iter_json(value):
        result.update(chunk.encode('utf-8'))
    return result.hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class _HashedWriter:
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.size_bytes = 0

    def write(self, text):
        written = self.stream.write(text)
        encoded = text[:written].encode('utf-8')
        self.hasher.update(encoded)
        self.size_bytes += len(encoded)
        return written

    @property
    def metadata(self):
        return dict(sha256=self.hasher.hexdigest(), size_bytes=self.size_bytes)


@contextmanager
def atomic_writer(path):
    """Write beside the destination; errors preserve its previous complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            writer = _HashedWriter(stream)
            yield writer
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_text(path, value):
    with atomic_writer(path) as stream:
        stream.write(value)
    return stream.metadata


def write_json(path, value):
    with atomic_writer(path) as stream:
        pending, size = [], 0
        for chunk in iter_json(value, indent=2):
            pending.append(chunk)
            size += len(chunk)
            if size >= 65536:
                stream.write(''.join(pending))
                pending, size = [], 0
        if pending:
            stream.write(''.join(pending))
        stream.write('\n')
    return stream.metadata


def write_csv(path, rows, columns=(), *, fixed=False, native=False):
    """Stream CSV; fixed schemas also support one-pass row generators.

    Legacy callers may request a key union from a replayable collection. Only
    the finite set of column names is retained during that extra pass. Native
    mode is for schemas whose nested mappings already have string keys.
    """
    columns = dict.fromkeys(columns)
    if not fixed:
        if iter(rows) is rows:
            raise TypeError('One-pass CSV rows require explicit columns and fixed=True')
        for row in rows:
            columns.update(dict.fromkeys(str(key) for key in _row_mapping(row)))
    with atomic_writer(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(columns), extrasaction='raise')
        writer.writeheader()
        for row in rows:
            values = {}
            for key, raw in _row_mapping(row).items():
                value = raw if native else public(raw)
                while native and not isinstance(value, (type(None), bool, int, float, str, tuple, list, dict)):
                    value = _native_default(value)
                if isinstance(value, (list, tuple, dict, set, frozenset)) or is_dataclass(value):
                    value = json.dumps(value, default=_native_default, ensure_ascii=False,
                                       sort_keys=True, allow_nan=False)
                elif isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('CSV numerical values must be finite')
                values[str(key)] = value
            writer.writerow(values)
    return stream.metadata


def _classes():
    from feynrules_export import UFOModel
    from madgraph_runner import ProductionChannel, ProductionChannels, DecayResults, CrossSectionResults
    from cascade_ranking import FinalStateRanking
    return {c.__name__: c for c in (UFOModel, ProductionChannel, ProductionChannels,
                                   DecayResults, CrossSectionResults, FinalStateRanking)}


def _relative(path, root):
    lexical_root = Path(os.path.abspath(root))
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    for candidate in (lexical_root, root):
        try:
            relative = path.relative_to(candidate)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f'Checkpoint artifact escapes its run directory: {path}') from None
    path = contained_path(root, relative)
    if not path.is_relative_to(root):
        raise ValueError(f'Checkpoint artifact escapes its run directory: {path}')
    return str(path.relative_to(root))


def _checked_directories(value, root):
    """Check newly added descendants before provenance code can scan them.

    Recorded artifact hashes cannot cover an injected extra file. In
    particular, UFO provenance scans all Python files, so every local link in
    a decoded directory must be contained before returning that directory.
    """
    directories = set()
    def visit(item):
        if isinstance(item, Path):
            path = contained_path(root, item.relative_to(root))
            if path.is_dir():
                directories.add(path)
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)
    visit(value)
    pending, seen = list(directories), set()
    while pending:
        directory = pending.pop()
        if directory in seen:
            continue
        seen.add(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                path = contained_path(root, Path(entry.path).relative_to(root))
                if path.is_dir():
                    pending.append(path)


def encode(value, root):
    if _history_rows(value):
        reference = dict(value.reference(root=root, checksum=True))
        return {'$history_rows': encode(reference, root)}
    if isinstance(value, Path):
        return {'$path': _relative(value, root)}
    if isinstance(value, complex):
        return {'$complex': [value.real, value.imag]}
    if is_dataclass(value):
        if type(value).__name__ not in _classes():
            raise TypeError(f'Unsupported checkpoint type: {type(value).__name__}')
        return {'$type': type(value).__name__, 'fields': {f.name: encode(getattr(value, f.name), root) for f in fields(value)}}
    if isinstance(value, dict):
        return {'$dict': [[encode(k, root), encode(v, root)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'$tuple': [encode(v, root) for v in value]}
    if isinstance(value, list):
        return [encode(v, root) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')


def decode(value, root):
    if isinstance(value, list):
        return [decode(v, root) for v in value]
    if not isinstance(value, dict):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError('Invalid checkpoint value')
    if set(value) == {'$path'}:
        try:
            return contained_path(root, value['$path'])
        except ValueError as exc:
            raise ValueError('Unsafe checkpoint path: ' + str(exc)) from exc
    if set(value) == {'$complex'}:
        return complex(*value['$complex'])
    if set(value) == {'$history_rows'}:
        from history_store import SQLiteHistoryRows
        reference = decode(value['$history_rows'], root)
        rows = SQLiteHistoryRows.from_reference(reference, root=root)
        current = rows.reference(root=root)
        if any(current.get(key) != reference.get(key) for key in ('schema', 'fingerprint', 'revision')):
            raise ValueError('History checkpoint source revision changed; refresh the ranking reference')
        return rows
    if set(value) == {'$dict'}:
        return {decode(k, root): decode(v, root) for k, v in value['$dict']}
    if set(value) == {'$tuple'}:
        return tuple(decode(v, root) for v in value['$tuple'])
    if set(value) == {'$type', 'fields'}:
        cls = _classes().get(value['$type'])
        if cls is None or set(value['fields']) != {f.name for f in fields(cls)}:
            raise ValueError('Unknown checkpoint type or changed field schema')
        return cls(**{k: decode(v, root) for k, v in value['fields'].items()})
    raise ValueError('Invalid checkpoint record')


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock_depth = 0
        self.cache_rejections = []

    @contextmanager
    def lock(self):
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with contained_path(self.root, '.run.lock').open('a+') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Another process is using this run: {self.root}') from None
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _record(self, stage, fingerprint):
        if not stage.replace('_', '').replace('-', '').isalnum() or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Invalid stage checkpoint key')
        return contained_path(self.root, Path('.checkpoints') / stage / (fingerprint + '.json'))

    def load(self, stage, fingerprint):
        path = self._record(stage, fingerprint)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if record['schema'] != SCHEMA or record['fingerprint'] != fingerprint or record['stage'] != stage:
                raise ValueError('Checkpoint schema/fingerprint mismatch')
            if digest(record['data']) != record['data_digest']:
                raise ValueError('Checkpoint data checksum mismatch')
            for relative, checksum in record['artifacts'].items():
                artifact = decode({'$path': relative}, self.root)
                if not artifact.is_file() or file_digest(artifact) != checksum:
                    raise ValueError(f'Artifact missing or changed: {relative}')
            value = decode(record['data'], self.root)
            _checked_directories(value, self.root)
            return value
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            self.cache_rejections.append({'stage': stage, 'reason': str(exc)})
            return None

    def save(self, stage, fingerprint, value, artifact_directory=None):
        data = encode(value, self.root)
        artifacts = {}
        if artifact_directory is not None:
            for path in sorted(Path(artifact_directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pkl', '.o', '.a', '.so', '.dylib'}:
                    artifacts[_relative(path, self.root)] = file_digest(path)
        write_json(self._record(stage, fingerprint), dict(schema=SCHEMA, stage=stage,
                   fingerprint=fingerprint, data=data, data_digest=digest(data), artifacts=artifacts))
