"""Contained run paths and lexical migration of saved history references.

The caller supplies the trusted run root. Saved paths and symlink targets are
checked before following them; migration never probes an old run directory.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import os
from pathlib import Path
import re
import stat


_HISTORY_DIRECTORY = re.compile(r'[0-9a-f]{20}\Z')
_MAX_SYMLINKS = 40


def _path(value):
    try:
        result = Path(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Run path must be a filesystem path') from exc
    if '..' in result.parts:
        raise ValueError('Run path must not contain parent traversal')
    return result


def contained_path(root, relative):
    """Return an absolute path inside ``root``, allowing missing descendants.

    ``root`` is a trusted directory selected by the caller, and may itself use
    a system alias such as /tmp. The supplied path must be relative. Descendant
    links are inspected locally and followed only within the canonical root;
    even a dangling link to an outside path is rejected without probing it.
    As with any returned Path, callers must prevent concurrent replacement of
    directories between validation and subsequent access.
    """
    relative = _path(relative)
    if relative.is_absolute():
        raise ValueError('Run path must be relative')
    anchor = Path(root).resolve()
    current = anchor
    pending = deque(relative.parts)
    links = 0
    while pending:
        part = pending.popleft()
        # Parent components can only originate in a locally inspected link.
        if part == '..':
            if current == anchor:
                raise ValueError('Symlink escapes the run directory')
            current = current.parent
            continue
        if part == '.':
            continue
        candidate = current / part
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            current = candidate
            continue
        if not stat.S_ISLNK(info.st_mode):
            current = candidate
            continue
        links += 1
        if links > _MAX_SYMLINKS:
            raise ValueError('Too many symlinks in run path')
        target = Path(os.readlink(candidate))
        if target.is_absolute():
            try:
                target = target.relative_to(anchor)
            except ValueError:
                raise ValueError('Symlink escapes the run directory') from None
            current = anchor
        # Process link components in order: resolving a link before a later
        # '..' can change which parent is meant. Never normalize them upfront.
        pending.extendleft(reversed(target.parts))
    return current


def portable_history_reference(reference, root, *, absolute=False, allow_legacy=True):
    """Copy a reference with a current, contained history database path.

    Stored references use a POSIX path relative to the run and ``path_base``
    equal to ``'run'``. ``absolute=True`` instead returns a contained Path for
    opening. Other reference fields, including file-stat provenance, remain
    unchanged; this helper does not establish database content integrity.

    Legacy absolute paths may be migrated only when their exact final four
    components are ``<run-name>/histories/<20 lower-case hex>/histories.sqlite``.
    Matching is lexical, with no reads, resolves, stats, or existence checks
    against the saved absolute path.
    """
    if not isinstance(reference, Mapping) or 'path' not in reference:
        raise ValueError('History reference must contain a path')
    if reference.get('path_base') not in (None, 'run'):
        raise ValueError('Unknown history reference path base')
    path = _path(reference['path'])
    lexical_root = Path(os.path.abspath(root))
    anchor = Path(root).resolve()
    relative = path
    if path.is_absolute():
        for candidate_root in (lexical_root, anchor):
            try:
                relative = path.relative_to(candidate_root)
                break
            except ValueError:
                continue
        else:
            tail = path.parts[-4:]
            if not (allow_legacy and len(tail) == 4
                    and tail[0] == lexical_root.name and tail[1] == 'histories'
                    and _HISTORY_DIRECTORY.fullmatch(tail[2])
                    and tail[3] == 'histories.sqlite'):
                raise ValueError('History reference is outside the run directory and cannot be migrated')
            relative = Path(*tail[1:])
    target = contained_path(anchor, relative)
    if target == anchor:
        raise ValueError('History database path must name a file inside the run directory')
    result = dict(reference)
    result['path'] = target if absolute else target.relative_to(anchor).as_posix()
    result['path_base'] = 'run'
    return result
