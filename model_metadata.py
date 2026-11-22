"""Immutable companion records for compact, editable model inputs.

Only semantic bindings, provenance and operator documentation are stored here;
these records never contain a previous model file. Publish a complete record
before atomically switching the model's reference to it. Keep the referenced
``.model_metadata`` record beside the model when moving or copying an input.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid


SCHEMA = 'madgraph-step-model-metadata-v1'
DIRECTORY = '.model_metadata'
MAX_BYTES = 8_000_000
MAX_DEPTH = 100
MAX_NODES = 250_000
_REFERENCE = re.compile(r'\.model_metadata/([0-9a-f]{64})\.json\Z')
_FIELDS = {'_input_bindings', '_input_provenance', 'operator_reference'}


class MetadataError(ValueError):
    """Missing, invalid, changed or unsafe companion model metadata."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


class _Budget:
    def __init__(self):
        self.nodes = 0

    def visit(self, depth):
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise MetadataError('Model metadata exceeds its nesting/node budget.')


def _encode(value, budget, depth=0):
    budget.visit(depth)
    child = lambda item: _encode(item, budget, depth + 1)
    if isinstance(value, dict):
        entries = []
        for key, item in value.items():
            if type(key) not in (str, int):
                raise MetadataError('Model metadata dictionary keys must be strings or integers.')
            entries.append([key, child(item)])
        return {'$dict': entries}
    if isinstance(value, tuple):
        return {'$tuple': [child(item) for item in value]}
    if isinstance(value, list):
        return [child(item) for item in value]
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise MetadataError('Model metadata numbers must be finite.')
        return {'$complex': [value.real, value.imag]}
    if type(value) is float and not math.isfinite(value):
        raise MetadataError('Model metadata numbers must be finite.')
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise MetadataError(f'Unsupported model metadata value: {type(value).__name__}.')


def _decode(value, budget, depth=0):
    budget.visit(depth)
    child = lambda item: _decode(item, budget, depth + 1)
    if isinstance(value, list):
        return [child(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {'$dict'} and isinstance(value['$dict'], list):
            result = {}
            for entry in value['$dict']:
                if not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) not in (str, int):
                    raise MetadataError('Invalid model metadata dictionary entry.')
                key, item = entry
                if key in result:
                    raise MetadataError(f'Duplicate model metadata dictionary key: {key!r}.')
                result[key] = child(item)
            return result
        if set(value) == {'$tuple'} and isinstance(value['$tuple'], list):
            return tuple(child(item) for item in value['$tuple'])
        if set(value) == {'$complex'} and isinstance(value['$complex'], list):
            parts = value['$complex']
            if len(parts) == 2 and all(type(part) in (int, float) for part in parts):
                try:
                    number = complex(*parts)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise MetadataError('Invalid complex model metadata number.') from exc
                if math.isfinite(number.real) and math.isfinite(number.imag):
                    return number
            raise MetadataError('Invalid complex model metadata number.')
        raise MetadataError('Invalid model metadata encoded value.')
    if type(value) is float and not math.isfinite(value):
        raise MetadataError('Model metadata numbers must be finite.')
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise MetadataError('Invalid model metadata literal.')


def _validate(metadata):
    if not isinstance(metadata, dict) or set(metadata) - _FIELDS:
        raise MetadataError('Model metadata must contain only bindings, provenance and operator_reference.')
    if not set(metadata) & {'_input_bindings', '_input_provenance'}:
        raise MetadataError('Model metadata must contain bindings or provenance.')
    for key in ('_input_bindings', '_input_provenance'):
        if key in metadata and not isinstance(metadata[key], dict):
            raise MetadataError(f'{key} must be a model metadata dictionary.')
    if 'operator_reference' in metadata and not isinstance(metadata['operator_reference'], str):
        raise MetadataError('operator_reference must be text.')
    # Binding semantics are checked against the current compiler by the input
    # reader. Do not silently reinterpret a known record from another schema.
    schemas = {'_input_bindings': {'tex-input-bindings-v1', 'operator-table-bindings-v1'},
               '_input_provenance': {'model-input-origins-v1'}}
    for key, supported in schemas.items():
        if key in metadata and metadata[key].get('schema') not in supported:
            raise MetadataError(f'Unknown {key} schema in model metadata.')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MetadataError(f'Duplicate JSON key in model metadata: {key!r}.')
        result[key] = value
    return result


def _constant(value):
    raise MetadataError(f'Nonfinite JSON constant in model metadata: {value}.')


def _record(raw, expected_digest):
    try:
        envelope = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
        if (not isinstance(envelope, dict) or set(envelope) != {'schema', 'sha256', 'metadata'}
                or envelope['schema'] != SCHEMA or envelope['sha256'] != expected_digest):
            raise MetadataError('Model metadata schema or content-address mismatch.')
        metadata = _decode(envelope['metadata'], _Budget())
        _validate(metadata)
        digest = hashlib.sha256(_canonical(envelope['metadata'])).hexdigest()
        if digest != expected_digest:
            raise MetadataError('Model metadata checksum mismatch; the companion record was changed.')
        return metadata
    except MetadataError:
        raise
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise MetadataError(f'Invalid model metadata JSON: {exc}.') from exc


def _root(model_path):
    return Path(model_path).expanduser().absolute().parent.resolve()


@contextmanager
def _directory(model_path, *, create=False):
    """Anchor access with directory descriptors and reject symlink components."""
    root = _root(model_path)
    root_fd = metadata_fd = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
        if create:
            try:
                os.mkdir(DIRECTORY, mode=0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileExistsError:
                pass
        metadata_fd = os.open(DIRECTORY, flags, dir_fd=root_fd)
        yield root, metadata_fd
        before = os.fstat(metadata_fd)
        after = os.stat(DIRECTORY, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise MetadataError('Model metadata directory changed during access.')
    except OSError as exc:
        raise MetadataError('Cannot safely access model metadata beside ' + str(model_path)
                            + '; keep the referenced .model_metadata folder with the input. ' + str(exc)) from exc
    finally:
        if metadata_fd is not None:
            os.close(metadata_fd)
        if root_fd is not None:
            os.close(root_fd)


def _read(metadata_fd, name, digest):
    content_identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_mode)
    identity = lambda item: (*content_identity(item), item.st_ctime_ns)
    retry_snapshot = None
    for attempt in range(2):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=metadata_fd)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise MetadataError('Model metadata must be a regular file.')
            if before.st_size > MAX_BYTES:
                raise MetadataError('Model metadata exceeds the 8 MB limit.')
            if retry_snapshot is not None and identity(before) != identity(retry_snapshot):
                raise MetadataError('Model metadata changed during access.')
            raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = os.stat(name, dir_fd=metadata_fd, follow_symlinks=False)
        if identity(before) == identity(after) == identity(current):
            break
        # A winning publisher removes its temporary hard link after exposing
        # the complete record. That changes ctime, but not its content. Retry
        # exactly once for this transition; all access/checksum checks remain.
        if (attempt == 0 and (before.st_nlink, after.st_nlink, current.st_nlink) in ((2, 2, 1), (2, 1, 1))
                and content_identity(before) == content_identity(after) == content_identity(current)):
            retry_snapshot = current
            continue
        raise MetadataError('Model metadata changed during access.')
    if len(raw) > MAX_BYTES:
        raise MetadataError('Model metadata exceeds the 8 MB limit.')
    return _record(raw, digest), raw


def load_metadata(model_path, reference):
    """Return ``(metadata, exact_bytes, path)`` for a verified relative reference.

    A reference is deliberately restricted to the package-managed companion
    directory. Absolute paths, traversal and symlinks are never followed.
    """
    match = _REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None:
        raise MetadataError('Use a relative .model_metadata/<SHA-256>.json metadata reference.')
    digest = match[1]
    name = digest + '.json'
    with _directory(model_path) as (root, metadata_fd):
        metadata, raw = _read(metadata_fd, name, digest)
    return metadata, raw, root / DIRECTORY / name


def publish_metadata(model_path, metadata, reference=''):
    """Publish immutable metadata and return its portable relative filename.

    ``reference`` is optional human-readable operator documentation. Publishing
    does not edit the model, overwrite an existing record or save old model text.
    A crash before the caller switches its model reference leaves only harmless,
    unreferenced metadata; the previous model continues to read its own record.
    """
    if not isinstance(metadata, dict) or not isinstance(reference, str):
        raise MetadataError('Model metadata must be a dictionary and reference documentation must be text.')
    metadata = dict(metadata)
    if reference:
        if metadata.get('operator_reference', reference) != reference:
            raise MetadataError('Conflicting operator_reference documentation.')
        metadata['operator_reference'] = reference
    _validate(metadata)
    encoded = _encode(metadata, _Budget())
    digest = hashlib.sha256(_canonical(encoded)).hexdigest()
    name = digest + '.json'
    raw = json.dumps(dict(schema=SCHEMA, sha256=digest, metadata=encoded), ensure_ascii=False,
                     sort_keys=True, indent=2, allow_nan=False).encode('utf-8') + b'\n'
    if len(raw) > MAX_BYTES:
        raise MetadataError('Model metadata exceeds the 8 MB limit.')
    with _directory(model_path, create=True) as (_, metadata_fd):
        temporary = '.publish-' + uuid.uuid4().hex + '.tmp'
        descriptor = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=metadata_fd)
            with os.fdopen(descriptor, 'wb') as stream:
                descriptor = None
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # link is an atomic no-replace publication. Concurrent equal
                # writers reuse one complete record; unequal/corrupt records fail.
                os.link(temporary, name, src_dir_fd=metadata_fd, dst_dir_fd=metadata_fd,
                        follow_symlinks=False)
                os.fsync(metadata_fd)
            except FileExistsError:
                existing, existing_raw = _read(metadata_fd, name, digest)
                if existing != metadata or existing_raw != raw:
                    raise MetadataError('Immutable model metadata content-address collision or changed record.')
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=metadata_fd)
            except FileNotFoundError:
                pass
    return DIRECTORY + '/' + name
