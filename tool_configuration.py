"""Discover host tools and save approved paths without changing physics inputs."""
from __future__ import annotations

import configparser
from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat
import tempfile

from portable_runtime import ROOT, discover_tools, normalize_tool_path

CONFIG_KEYS = {'wolfram_kernel': 'mathematica', 'madgraph': 'madgraph',
               'feynrules': 'feynrules', 'fortran': 'fortran', 'cxx': 'cxx',
               'make': 'make', 'bash': 'bash', 'perl': 'perl'}


def _runtime_values(parser, base):
    values = {}
    for runtime_key, config_key in CONFIG_KEYS.items():
        value = parser.get('tools', config_key, fallback='').strip()
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute() and (runtime_key in ('madgraph', 'feynrules') or '/' in value or chr(92) in value):
                value = str(base / path)
            values[runtime_key] = value
    return values


def _config_value(name, path, base):
    # Keep bundled/source paths relocatable when they live beside this config.
    if name in ('madgraph', 'feynrules'):
        try:
            return str(Path(path).relative_to(base))
        except ValueError:
            pass
    return str(path)


def _replace_tools(original, changes):
    """Replace only changed tool assignments, retaining other bytes as text."""
    newline = '\r\n' if '\r\n' in original else '\n'
    lines = original.splitlines(keepends=True)
    pending = dict(changes)
    result, in_tools, found = [], False, False
    def append_pending():
        if pending:
            if result and not result[-1].endswith(('\n', '\r')):
                result[-1] += newline
            result.extend(f'{key} = {value}{newline}' for key, value in pending.items())
            pending.clear()
    for line in lines:
        section = re.match(r'^\s*\[([^]]+)\]', line)
        if section:
            if in_tools:
                append_pending()
            in_tools = section.group(1) == 'tools'
            found = found or in_tools
        # Whitespace around an INI delimiter must not consume the line ending:
        # an empty "mathematica =\n" still needs its new value on that line.
        assignment = re.match(r'^([ \t]*)([A-Za-z_][A-Za-z_0-9]*)([ \t]*[=:][ \t]*)([^\r\n]*)(\r?\n)?$', line)
        if in_tools and assignment and assignment.group(2).lower() in pending:
            key = assignment.group(2).lower()
            line = assignment.group(1) + assignment.group(2) + assignment.group(3) + pending.pop(key) + (assignment.group(5) or '')
        result.append(line)
    if not found:
        if result and not result[-1].endswith(('\n', '\r')):
            result[-1] += newline
        result.append(newline + '[tools]' + newline)
    append_pending()
    return ''.join(result)


def _atomic_write(path, content, mode):
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _save_lock(path):
    # This package's numerical backend supports POSIX hosts (including WSL).
    import fcntl
    with (path.parent / ('.' + path.name + '.lock')).open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def configure_tools(config_path=None, *, input_fn=input, output_fn=print):
    """Discover, preview and explicitly approve an atomic [tools] update.

    Return {saved, path, paths, changes, backup, cancelled}; the retained
    compatibility field ``backup`` is always None. A negative answer,
    EOF or Ctrl-C changes no configuration. Enter ``edit`` at the confirmation
    to choose a numbered candidate or a custom path for each tool. No tool is
    installed or started here; run the separate environment check afterward.
    """
    path = Path(config_path or ROOT / '.config').expanduser().resolve()
    original = path.read_bytes() if path.exists() else None
    content = original.decode('utf-8') if original is not None else ''
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(content)
    except configparser.Error as exc:
        raise ValueError(f'Invalid tool configuration {path}; file unchanged: {exc}') from exc
    if parser.defaults():
        raise ValueError('Use explicit .config sections rather than DEFAULT values.')
    runtime = _runtime_values(parser, path.parent)
    discovery = discover_tools(runtime)
    chosen = {key: entry['selected'] or (entry['candidates'][0]['path'] if entry['candidates'] else None)
              for key, entry in discovery['tools'].items()}
    result = dict(saved=False, path=path, paths=chosen, changes={}, backup=None, cancelled=False)
    def preview():
        output_fn(f'Tool configuration: {path}')
        changes = {}
        for name, key in CONFIG_KEYS.items():
            selected = chosen[name]
            entry = discovery['tools'][name]
            label = selected or '(not found; unchanged)'
            if entry['reason']:
                label += '; ' + entry['reason']
            output_fn(f'  {key}: {label}')
            if selected:
                value = _config_value(name, selected, path.parent)
                if parser.get('tools', key, fallback='').strip() != value:
                    changes[key] = value
        result['changes'] = changes
        if changes:
            output_fn('Exact values to save in [tools]:')
            for key, value in changes.items():
                output_fn(f'  {key} = {value}')
        return changes
    def edit():
        for name, key in CONFIG_KEYS.items():
            candidates = discovery['tools'][name]['candidates']
            for number, candidate in enumerate(candidates, 1):
                output_fn(f'  {number}. {candidate["path"]} ({candidate["origin"]})')
            while True:
                value = input_fn(f'{key}: candidate number or path; Enter keeps {chosen[name] or "unset"}: ').strip()
                if not value:
                    break
                if value.isdecimal() and 1 <= int(value) <= len(candidates):
                    value = candidates[int(value) - 1]['path']
                resolved = normalize_tool_path(name, value, base=path.parent)
                if resolved:
                    if any(character in resolved for character in ('\n', '\r', '\0')):
                        output_fn('A tool path cannot contain a newline or NUL.')
                        continue
                    chosen[name] = resolved
                    break
                output_fn(f'Invalid {key} path; no change was made.')
    try:
        while True:
            changes = preview()
            missing = [name for name, value in chosen.items() if value is None]
            if not changes and not missing:
                output_fn('Tool paths are already current; nothing was saved.')
                return result
            if missing:
                output_fn('Missing tool paths: ' + ', '.join(missing) +
                          '. Enter edit to provide custom locations; saving leaves missing paths unchanged.')
            answer = input_fn('Save these paths? [y/N/edit]: ').strip().lower()
            if answer == 'edit':
                edit()
                continue
            if answer not in ('y', 'yes'):
                result['cancelled'] = True
                output_fn('Cancelled; configuration is unchanged.')
                return result
            if not changes:
                output_fn('No paths were changed. Run configure-tools again when the missing tools are available.')
                return result
            break
    except (EOFError, KeyboardInterrupt):
        result['cancelled'] = True
        output_fn('Cancelled; configuration is unchanged.')
        return result
    updated = _replace_tools(content, changes).encode('utf-8')
    # Validate the actual replacement before writing, including custom values.
    check = configparser.ConfigParser(interpolation=None)
    try:
        check.read_string(updated.decode('utf-8'))
    except configparser.Error as exc:
        raise RuntimeError(f'Generated tool configuration is invalid; original file unchanged: {exc}') from exc
    for key, value in changes.items():
        if check.get('tools', key) != value:
            raise ValueError(f'Cannot safely save [tools] {key}')
    if not path.parent.is_dir():
        raise FileNotFoundError(f'Configuration directory does not exist: {path.parent}')
    with _save_lock(path):
        current = path.read_bytes() if path.exists() else None
        if current != original:
            raise RuntimeError('Configuration changed during approval; rerun configure-tools to review the new file.')
        mode = stat.S_IMODE(path.stat().st_mode) if original is not None else 0o600
        _atomic_write(path, updated, mode)
    result['saved'] = True
    output_fn(f'Saved {path}')
    output_fn('Paths are saved. Run check to verify versions, compiler startup and the Mathematica license.')
    return result
