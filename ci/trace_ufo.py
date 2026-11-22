"""Capture bounded evidence from the original and repaired benchmark exports."""
from __future__ import annotations

import ast
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


# Published commit used for the three-platform run that exposed the regression.
ORIGINAL = 'a28f8d4680ebbc67374c7d20d2ebb1d1e881592a'
FR_SOURCES = ('FeynRulesPackage.m', 'Core/ExtractVertexTools.m',
              'Core/MassDiagonalization.m', 'Interfaces/AspergeInterface.m')
GUARD = 'PORTABLE_UFO_UNKNOWN_FIELD'
FIELD_FILE = 'unknown_vertex_fields.json'


def save_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def configure_original(repo, original, candidate):
    """Keep the field guard and generated physics identical; change only FR namespace code."""
    from pipeline_config import load_config

    fr = repo / 'vendor/FeynRules'
    guard_source = (fr / 'Interfaces/UFO/PYIntVertices.m').read_text()
    dispatch = (fr / 'Interfaces/UFO/PYIntMain.m').read_text()
    if (GUARD not in guard_source or FIELD_FILE not in guard_source
            or 'PYReportUnknownLegs[' not in dispatch):
        raise ValueError('Original export requires the instrumented UFO interface')
    runtime = dict(load_config(candidate.config, model_file=candidate.model, read_model=False).runtime)
    if Path(runtime['feynrules']).resolve() != fr.resolve():
        raise ValueError('Comparison requires the bundled candidate FeynRules')
    copy = original.root / 'FeynRules'
    shutil.copytree(fr, copy)
    for source in FR_SOURCES:
        # Read local Git objects only. The workflow fetches this history with checkout.
        reference = ORIGINAL + ':vendor/FeynRules/' + source
        result = subprocess.run(['git', 'cat-file', 'blob', reference], cwd=repo,
                                capture_output=True, check=True, timeout=15)
        (copy / source).write_bytes(result.stdout)
    # Search the original bad export for the signature visible in its saved log.
    # The candidate keeps fail-fast handling of the very first unknown field.
    main = copy / 'Interfaces/UFO/PYIntMain.m'
    main.write_bytes(b'PRIVATE`PY$TraceSixLegSignature = True;\n' + main.read_bytes())
    runtime['feynrules'] = str(copy)
    config = configparser.ConfigParser(interpolation=None)
    config['tools'] = {('mathematica' if key == 'wolfram_kernel' else key): value
                       for key, value in runtime.items()}
    original.config = str(original.root / 'original.config')
    with Path(original.config).open('w', encoding='utf-8') as stream:
        config.write(stream)
    # Preserve relative metadata paths and input-relative output semantics.
    original.model.write_bytes(candidate.model.read_bytes())
    metadata = candidate.metadata_path.relative_to(candidate.input)
    (original.input / metadata).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate.metadata_path, original.input / metadata)
    settings = load_config(original.config, model_file=original.model)
    candidate_settings = load_config(candidate.config, model_file=candidate.model)
    if (settings.output != original.output.resolve() or candidate_settings.output != candidate.output.resolve()
            or settings.output.is_relative_to(candidate_settings.output)
            or candidate_settings.output.is_relative_to(settings.output)
            or settings.runtime != runtime or Path(settings.runtime['feynrules']) != copy.resolve()
            or any(getattr(settings, key) != getattr(candidate_settings, key)
                   for key in ('collider', 'decays', 'ranking', 'explicit_decays'))):
        raise ValueError('Original and candidate configurations failed the isolation or physics settings check')


def redact_secrets(text):
    init = os.environ.get('WOLFRAMINIT', '')
    entitlement = re.search(r'(?:^|\s)-entitlement\s+(\S+)', init)
    for value in (os.environ.get('ENTITLEMENT'), os.environ.get('WOLFRAMSCRIPT_ENTITLEMENTID'),
                  entitlement.group(1) if entitlement else None):
        if value:
            text = text.replace(value, '[redacted]')
    return re.sub(r'(?i)(?:-entitlement\s+|(?:entitlement|password|token)\s*[=:]\s*)\S+',
                  '[redacted credential]', text)


_PATH_START = re.compile(r'(?<![\w\]\)])(?:/[a-zA-Z._~0-9-]|[A-Za-z]:[\\/]|\\\\|~/)|\w+://')


def field_text(text):
    if not isinstance(text, str) or len(text) > 2048:
        raise ValueError('Malformed field diagnostic text')
    text = redact_secrets(text)
    # A Wolfram field can contain a rational slash; a filesystem path has a
    # directory separator followed by a name. Omit a path-bearing field whole.
    if _PATH_START.search(text):
        return '[redacted path]'
    return text


def safe_reason(text, limit=420):
    text = redact_secrets(text)
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    text = ' '.join(text.split())
    # Tool diagnostics put absolute paths at the end of a clause. A path may
    # contain spaces, so clipping its first word would expose the rest.
    clauses = re.split(r'(: |; )', text)
    for i in range(0, len(clauses), 2):
        match = _PATH_START.search(clauses[i])
        if match:
            clauses[i] = clauses[i][:match.start()] + '[redacted path]'
    return ''.join(clauses)[:limit]


_UFO_MODULE = re.compile(r'Invalid generated UFO Python in (.+?\.py), (?=line \d+, column \d+)')
_SOURCE_LOCATION = re.compile(
    r'''line (\d{1,12}), column (\d{1,12}), source ('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")''')


def cli_failure(path, fallback):
    """Retain one useful failure and structured UFO locations, with no raw stdout."""
    with path.open('rb') as stream:
        offset = max(0, path.stat().st_size - 32768)
        stream.seek(offset)
        data = stream.read(32768)
    # Discard an incomplete first line. A token or a path might cross the boundary.
    if offset:
        data = data.partition(b'\n')[2]
    lines = data.decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        found = _UFO_MODULE.search(line)
        if not found:
            continue
        module = re.split(r'[\\/]', found.group(1))[-1]
        module = safe_reason(module, 80) if re.fullmatch(r'[A-Za-z_]\w*\.py', module) else 'UFO module'
        locations = []
        message = ''
        for match in _SOURCE_LOCATION.finditer(line, found.end()):
            try:
                source = ast.literal_eval(match.group(3))
            except (SyntaxError, ValueError):
                source = '<unreadable excerpt>'
            if not locations:
                message = safe_reason(line[match.end():].split('. Conversion diagnostic:', 1)[0].lstrip(': '), 150)
            locations.append(dict(line=int(match.group(1)), column=int(match.group(2)),
                                  source=safe_reason(source, 185)))
            if len(locations) == 2:
                break
        if locations:
            first = locations[0]
            reason = f"Invalid generated UFO Python in {module}, line {first['line']}, column {first['column']}"
            if message:
                reason += ': ' + message
            return {'reason': reason, 'module': module, 'locations': locations}
    # The CLI often prints a complete dependency checklist after a missing
    # tool. Keep its actual rejection even when a later check succeeded.
    failures = [line for line in lines if line.startswith('MISSING ')]
    for line in reversed(failures or lines):
        line = line.strip()
        if line and not line.startswith(('OK ', 'Checking tool files', 'Run files:', 'Traceback (')):
            return {'reason': safe_reason(line)}
    return {'reason': fallback}


def field_report(path):
    """Publish only data emitted by PYReportUnknownLegs, independently validate the evidence."""
    with path.open('rb') as stream:
        data = stream.read(256 * 1024 + 1)
    if len(data) > 256 * 1024:
        raise ValueError('Oversized field diagnostic')
    raw = json.loads(data)
    if raw['stage'] != 'MakeOutgoingParticles':
        raise ValueError('Unexpected field diagnostic stage')
    report = {'stage': raw['stage']}
    for key in ('vertex_index', 'number_of_legs', 'unclassified_outgoing_legs'):
        n = raw[key]
        if type(n) is not int or n < 1:
            raise ValueError('Invalid field diagnostic count')
        report[key] = n
    count = report['number_of_legs']
    if report['unclassified_outgoing_legs'] > count or not (1 <= len(raw['legs']) <= min(count, 24)):
        raise ValueError('Invalid field diagnostic leg count')
    rows = []
    positions = set()
    for item in raw['legs']:
        position = item['position']
        if type(position) is not int or not 1 <= position <= count or position in positions:
            raise ValueError('Invalid field diagnostic leg position')
        positions.add(position)
        row = {'position': position}
        for side in ('incoming', 'outgoing'):
            for suffix in ('leg_id', 'field', 'symbol_or_head'):
                key = side + '_' + suffix
                row[key] = field_text(item[key])
            key = side + '_spin_code'
            if type(item[key]) is not int:
                raise ValueError('Invalid spin code')
            row[key] = item[key]
        rows.append(row)
    if not any(row['outgoing_spin_code'] == 0 for row in rows):
        raise ValueError('No unclassified field in diagnostic sample')
    report['legs'] = rows
    level, first = raw['match_level'], raw['first_unclassified_vertex_index']
    if type(first) is not int or not 1 <= first <= report['vertex_index']:
        raise ValueError('Invalid first unclassified vertex ordinal')
    report.update(match_level=level, first_unclassified_vertex_index=first)
    count_keys = ('ordered_signature_match_count', 'six_leg_multiset_match_count', 'vertices_in_replay')
    if level == 'first_unclassified':
        if report['vertex_index'] != first or any(key in raw for key in count_keys):
            raise ValueError('Unexpected replay counts in the candidate field trace')
        return report
    if level not in ('exact_ordered_signature', 'six_leg_multiset', 'first_unclassified_fallback'):
        raise ValueError('Unknown replay match level')
    counts = [raw[key] for key in count_keys]
    if any(type(n) is not int for n in counts):
        raise ValueError('Invalid replay search counts')
    ordered, multiset, vertices = counts
    if not (0 <= ordered <= multiset <= vertices and report['vertex_index'] <= vertices):
        raise ValueError('Inconsistent replay search counts')
    expected = [0, 0, 0, 0, 1, 0]
    observed = [row['outgoing_spin_code'] for row in sorted(rows, key=lambda row: row['position'])]
    if level == 'exact_ordered_signature':
        valid = ordered > 0 and count == 6 and observed == expected
        candidates = ordered
    elif level == 'six_leg_multiset':
        valid = ordered == 0 and multiset > 0 and count == 6 and sorted(observed) == sorted(expected)
        candidates = multiset
    else:
        valid = multiset == 0 and first == report['vertex_index']
        candidates = 0
    if not valid:
        raise ValueError('Selected replay vertex disagrees with the recorded signature search')
    report.update(zip(count_keys, counts))
    report['matching_vertices_at_selected_level'] = candidates
    report['ambiguous_at_selected_level'] = candidates > 1
    return report


def log_events(path):
    """Return diagnostic identifiers only, never raw kernel or licensing messages."""
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        head = stream.read(65536)
        stream.seek(max(len(head), path.stat().st_size - 131072))
        tail = stream.read(131072)
    text = redact_secrets((head + b'\n' + tail).decode('utf-8', errors='replace'))
    pattern = r'\b(?:PORTABLE_[A-Z_]{1,80}|[A-Za-z$]{1,48}::[A-Za-z0-9]{1,32})\b'
    return sorted(set(re.findall(pattern, text)))[:96]


def export_evidence(benchmark):
    attempts = benchmark.output.glob('*/*/export/*/attempt-*')
    attempt = max(attempts, key=lambda item: item.stat().st_mtime_ns, default=None)
    commands = [item for item in benchmark.report['commands'] if item['action'] == 'ufo']
    status = commands[-1]['status'] if commands else 'not_started'
    events = log_events(attempt / 'feynrules.log') if attempt else []
    result = {'export_command': status, 'kernel_events': events}
    last_failure = next((item for item in reversed(benchmark.report['commands'])
                         if item.get('failure')), None)
    if last_failure:
        result['cli_failure'] = {'action': last_failure['action'], **last_failure['failure']}
    path = next(attempt.glob('*_UFO/' + FIELD_FILE), None) if attempt else None
    if path is not None:
        try:
            result['fields'] = field_report(path)
        except (ValueError, KeyError, TypeError, OSError):
            result['invalid_field_report'] = True
    if status == 'failed' and GUARD in events and 'fields' in result:
        result['observation'] = 'guard_reproduced_unknown_fields'
    elif status == 'passed' and 'PORTABLE_EXPORT_SUCCESS' in events and GUARD not in events and path is None:
        result['observation'] = 'export_passed_without_unknown_fields'
    else:
        result['observation'] = 'inconclusive'
    return result


def vertex_checks(ufo):
    """Inspect the saved UFO's actual vertices, independent of generation order."""
    def definitions(filename, constructor):
        path = ufo / filename
        if path.stat().st_size > 2 * 1024 * 1024:
            raise AssertionError('Benchmark UFO source exceeds the reference size bound')
        tree = ast.parse(path.read_bytes())
        return [node.value for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == constructor]

    vertices = definitions('vertices.py', 'Vertex')
    couplings = definitions('couplings.py', 'Coupling')
    topologies = set()
    for vertex in vertices:
        parts = next((kw.value for kw in vertex.keywords if kw.arg == 'particles'), None)
        if not isinstance(parts, ast.List) or not all(isinstance(p, ast.Attribute)
                and isinstance(p.value, ast.Name) and p.value.id == 'P' for p in parts.elts):
            raise AssertionError('Invalid particle list in a benchmark UFO vertex')
        topologies.add(tuple(sorted(p.attr for p in parts.elts)))
    representative = (('x1bar', 'u', 'u'), ('x1', 'u__tilde__', 'u__tilde__'),
                      ('y1bar', 'u', 'e__minus__'), ('y1', 'u__tilde__', 'e__plus__'),
                      ('x1bar', 'x1', 'g'), ('y1bar', 'y1', 'g'),
                      ('x1bar', 'x2', 'W__plus__'))
    return {'vertices': len(vertices), 'couplings': len(couplings),
            'max_vertex_legs': max(map(len, topologies), default=0),
            'reference_vertices': 191, 'reference_couplings': 136,
            'representative_vertices': {' '.join(t): tuple(sorted(t)) in topologies for t in representative}}
