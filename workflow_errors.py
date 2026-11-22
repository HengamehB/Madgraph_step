"""Stable diagnostic categories shared by interactive and saved workflows."""
import errno
import sqlite3


class PhysicalInputError(ValueError):
    category = 'physical_input'


class CompilerError(ValueError):
    category = 'compiler_failure'


class ResourceLimitError(CompilerError):
    category = 'resource_limit'


class BackendUnsupportedError(ValueError):
    category = 'not_madgraph_supported'


def error_category(error):
    """Preserve explicit diagnoses through exception wrapping.

    An uncategorized external failure is never evidence that MadGraph cannot
    support the requested physics.
    """
    current, seen = error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        category = getattr(current, 'category', None)
        if category:
            return category
        if isinstance(current, OSError) and current.errno in (errno.ENOSPC, errno.EDQUOT):
            return 'resource_limit'
        if isinstance(current, sqlite3.Error) and getattr(current, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL:
            return 'resource_limit'
        current = current.__cause__
    if isinstance(error, (KeyboardInterrupt, EOFError)):
        return 'interrupted'
    if isinstance(error, (ValueError, TypeError)):
        return 'physical_input'
    return 'execution_failure'
