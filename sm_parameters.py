"""Apply numerical SM inputs consistently to generated FeynRules SM copies."""
from __future__ import annotations

import re


# These inputs are declared by particle classes, including repeated ghost and
# Goldstone declarations, rather than by ordinary M$Parameters entries.
CLASS_INPUTS = frozenset(('MZ', 'Me', 'MMU', 'MTA', 'MU', 'MC', 'MT',
                          'MD', 'MS', 'MB', 'WZ', 'WW', 'WT', 'WH'))
SM_EXTERNAL_INPUTS = CLASS_INPUTS | {
    'Gf', 'aEWM1', 'aS', 'cabi', 'ymdo', 'ymup', 'yms', 'ymc', 'ymb',
    'ymt', 'yme', 'ymm', 'ymtau',
}


def apply_sm_inputs(source, inputs, *, skip=()):
    """Return a generated SM source with all requested numerical inputs applied.

    ``inputs`` is the complete physical-input dictionary. ``skip`` is reserved
    for a compiler that supplies a parameter through its own transformed basis,
    such as the neutral-vector compiler's physical/bare Z declarations. The
    Only independent numerical SM inputs may be set.
    """
    from feynrules_export import ModelResolutionError, _wl

    for key, value in inputs.get('sm_inputs', {}).items():
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z$][A-Za-z0-9$]*', key):
            raise ModelResolutionError(f'Unknown SM input {key!r}')
        if key not in SM_EXTERNAL_INPUTS:
            raise ModelResolutionError(
                f'Unknown SM input {key!r}: only independent external SM inputs can be set.')
        if key in skip:
            continue
        if key in CLASS_INPUTS:
            pattern = rf'(\{{\s*{re.escape(key)}\s*,\s*)([-+0-9.*^eE]+)(\s*\}})'
            source, count = re.subn(
                pattern, lambda match: match[1] + _wl(value) + match[3], source)
            if not count:
                raise ModelResolutionError(
                    f'SM class parameter {key!r} was not found in the bundled model.')
            continue
        # Match a complete symbol and do not cross into a following parameter
        # declaration if the requested block has lost its Value entry.
        pattern = (rf'(?<![A-Za-z0-9_$])({re.escape(key)}\s*==\s*\{{'
                   rf'(?:(?!\b[A-Za-z$][A-Za-z0-9$]*\s*==).)*?'
                   rf'\bValue\s*->\s*)([^,\n]+)')
        source, count = re.subn(
            pattern, lambda match: match[1] + _wl(value), source, count=1, flags=re.S)
        if count != 1:
            raise ModelResolutionError(
                f'Unknown SM input {key!r}: parameter value was not found in the bundled model.')
    return source
