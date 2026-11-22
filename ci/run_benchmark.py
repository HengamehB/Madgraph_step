#!/usr/bin/env python3
"""Exercise the published CLI from fresh text inputs, without altering examples.

fast: Python-only catalogue, binding, spectrum and compiler checks.
licensed: those checks plus fresh FeynRules export, MadWidth and one integration.
--trace-original-fields adds one export-only run with the original FeynRules
namespace definitions and the same field diagnostic, before the repaired run.
Summaries, bounded field diagnostics and the licensed ranking CSV are written
to BENCHMARK_ARTIFACTS. Raw logs, local paths and tool configuration stay temporary.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci import trace_ufo

FREE = {'X X†', 'Y Y†', 'D(X) D(X†)', 'D(Y) D(Y†)'}
INVENTORY = FREE | {
    'H H† X X†', 'H H† Y Y†', 'X X† Y Y†', 'X X†^2 Y†',
    'X^2 X†^2', 'X^2 Y^2', 'Y^2 Y†^2', 'L Q Y†', 'Q^2 X†',
    'Y ec uc', 'Q^2 Y', 'Y dc† uc†',
}
QQ = [.1, 0., 0., .1, 0., .1]
EU = [.1, 0., 0., 0., .1, 0., 0., 0., .1]
PORTAL_SHIFT = .1 / (2 * math.sqrt(2) * 1.16637e-5)
COEFFICIENTS = {
    'X X†': -1500**2 + PORTAL_SHIFT,
    'Y Y†': -1200**2 + PORTAL_SHIFT,
    'D(X) D(X†)': 1., 'D(Y) D(Y†)': 1.,
    'H H† X X†': [-.1, 0.], 'H H† Y Y†': -.1,
    'X X† Y Y†': [-.05 + .02 / 3, -.02],
    'X^2 X†^2': [0., 0., -.1, 0., 0.], 'Y^2 Y†^2': -.1,
    'Q^2 X†': QQ, 'Y ec uc': EU,
}
WIDTH_TARGETS = {
    'x1': .879654062, 'x2': .887444456, 'x3': .895234830, 'y1': .706487172,
}
# Printed PDF targets, identified by signed PDGs so particle names/order do
# not disguise a charge-conjugation or flavour-order error.
PARTIAL_TARGETS = {
    (9700001, (2, 2)): .2984155, (9700001, (4, 4)): .2984147,
    (9700001, (6, 6)): .2828239, (9700002, (1, 2)): .2832043,
    (9700002, (2, 3)): .0152112, (9700002, (5, 6)): .2906138,
    (9700003, (5, 5)): .2984038, (9700004, (2, 11)): .2387324,
    (9700004, (4, 13)): .2387319, (9700004, (6, 15)): .2290229,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def edit_values(path, *, assignments=None, operators=None):
    """Edit existing AST value spans; preserve every row and managed metadata.

    AST columns are UTF-8 byte offsets, including operator names containing †.
    This parses syntax to locate edits and never executes the input document.
    """
    data = path.read_bytes()
    nodes = {node.targets[0].id: node.value for node in ast.parse(data).body
             if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name)}
    offsets = [0]
    for line in data.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []

    def replace(node, value):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end, repr(value).encode('utf-8')))

    for name, value in (assignments or {}).items():
        require(name in nodes, 'Generated model lacks assignment ' + name)
        replace(nodes[name], value)
    if operators:
        table = nodes['operators']
        require(isinstance(table, ast.Dict), 'Generated operators must be a dictionary')
        rows = {ast.literal_eval(key): value for key, value in zip(table.keys, table.values)}
        for name, changes in operators.items():
            require(name in rows, 'Generated inventory lacks operator ' + name)
            row = rows[name]
            require(isinstance(row, ast.Dict), 'Invalid generated operator row')
            fields = {ast.literal_eval(key): value for key, value in zip(row.keys, row.values)}
            for field, value in changes.items():
                require(field in fields, 'Generated operator lacks ' + field)
                replace(fields[field], value)
    for start, end, value in sorted(edits, reverse=True):
        data = data[:start] + value + data[end:]
    path.write_bytes(data)


class Benchmark:
    def __init__(self, root, mode, report):
        root = root.resolve()
        self.root, self.mode, self.report = root, mode, report
        self.config = os.environ.get('BENCHMARK_CONFIG')
        self.input = root / 'inputs with spaces'
        self.input.mkdir()
        self.model = self.input / 'generated benchmark.txt'
        self.particles = self.input / 'particles.txt'
        self.output = root / 'saved results'
        self.metadata_path = None
        self.metadata_digest = None

    def cli(self, action, *extra, source=None):
        command = [str(ROOT / 'run.sh'), action, str(source or self.model), *extra]
        if self.config:
            command.extend(['--config', self.config])
        log = self.root / ('command-%03d.log' % len(self.report['commands']))
        started = time.monotonic()
        record = {'action': action, 'arguments': list(extra), 'status': 'started'}
        # --model contains a temporary path. Artifacts record only its basename.
        if '--model' in record['arguments']:
            index = record['arguments'].index('--model') + 1
            record['arguments'][index] = Path(record['arguments'][index]).name
        print('Benchmark:', action, *record['arguments'], flush=True)
        self.report['commands'].append(record)
        environment = {**os.environ, 'PYTHON': sys.executable, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        with log.open('wb') as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=1800 if self.mode == 'licensed' else 600)
            except subprocess.TimeoutExpired:
                record.update(status='timed_out', elapsed_seconds=round(time.monotonic() - started, 2))
                record['failure'] = trace_ufo.cli_failure(log, 'CLI timed out')
                print('CLI timeout:', json.dumps(record['failure'], ensure_ascii=False), file=sys.stderr, flush=True)
                raise RuntimeError('CLI ' + action + ' exceeded its benchmark timeout') from None
        record.update(status='passed' if result.returncode == 0 else 'failed',
                      exit_code=result.returncode, elapsed_seconds=round(time.monotonic() - started, 2))
        if result.returncode:
            record['diagnostic_identifiers'] = trace_ufo.log_events(log)
            record['failure'] = trace_ufo.cli_failure(log, f'CLI exited with status {result.returncode}')
            print('CLI exit', result.returncode, json.dumps(record['failure'], ensure_ascii=False),
                  file=sys.stderr, flush=True)
            raise AssertionError('CLI ' + action + ' failed: ' + record['failure']['reason'])
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            tail = stream.read().decode('utf-8', errors='replace')
        for line in reversed(tail.splitlines()):
            for prefix in ('Run files:', 'Validation and model files:'):
                if line.startswith(prefix):
                    run = Path(line[len(prefix):].strip()).resolve()
                    require(run.is_relative_to(self.output.resolve()), 'CLI output escaped isolated benchmark directory')
                    return run
        return None

    def document(self):
        from tex_inputs import read_inputs
        document = read_inputs(self.model)
        require(set(document.assignments['operators']) == INVENTORY, 'Generated operator inventory changed')
        if self.metadata_path is not None:
            require((self.model.parent / document.assignments['_metadata']).resolve() == self.metadata_path,
                    'Benchmark edit changed companion metadata reference')
            require(digest(self.metadata_path) == self.metadata_digest,
                    'Benchmark must not rewrite generated operator meanings')
        return document

    def validation(self, label, active, expected_masses=None):
        run = self.cli('validate')
        require(run is not None, 'Validation did not identify its saved run')
        document = self.document()
        from tex_inputs import discover_inputs
        discovery = discover_inputs(self.model)
        rows = {row['key']: row for row in discovery['operators']}
        require({key for key, row in document.assignments['operators'].items() if row['enabled']} == set(active),
                'Active operator selection changed')
        selected = read_json(run / 'selected_operators.json')
        selected_ids = {row['id'] for row in selected}
        expected_symbols = {int(operator_id[1:]) for operator_id in selected_ids}
        for filename in ('lagrangian_expression.tex', 'lagrangian.tex'):
            expression = (run / filename).read_text(encoding='utf-8')
            symbols = {int(i) for i in re.findall(r'c_\{(\d+)(?:,a)?\}', expression)}
            require(symbols == expected_symbols, filename + ' disagrees with the selected terms')
        require({document.definition.coefficient_key(identity) for identity in selected_ids} == set(active),
                'Saved Lagrangian does not match enabled operators')
        contractions = read_json(run / 'contractions.json')
        require(set(contractions['coverage']) == selected_ids, 'Operator-to-compiler coverage is incomplete')
        resolved = {row['operator_id']: row for row in contractions['operators']}
        for key in active:
            require(rows[key]['binding_status'] == 'verified', 'Stale generated operator binding for ' + key)
            if key not in ('Q^2 X†', 'Y ec uc'):
                continue
            info = rows[key]
            flavour_order = ([f'Y_QQ[{i},{j}]' for i, j in itertools.combinations_with_replacement(range(1, 4), 2)]
                             if key == 'Q^2 X†' else
                             [f'lambda_R[{i},{j}]' for i, j in itertools.product(range(1, 4), repeat=2)])
            require(list(info['invariant_labels']) == flavour_order, 'Yukawa coefficient order changed')
            normalization = info['hint']
            require(('One half times the full symmetric complex matrix' in normalization
                     and 'normalized sextet' in normalization and 'CKM[i,k]' in normalization)
                    if key == 'Q^2 X†' else
                    ('Row-major order' in normalization and 'right-handed' in normalization),
                    'Documented Yukawa normalization convention changed')
            op = resolved[info['operator_id']]
            require(op['kind'] == ('sextet_qq_yukawa' if key == 'Q^2 X†' else 'triplet_eu_yukawa'),
                    'Wrong coloured Yukawa lowering')
            require(op['add_hc'] and not info['real'], 'Complex Yukawa Hermitian conjugate is missing')
            values = [complex(value['real'], value['imag']) if isinstance(value, dict) else complex(value)
                      for value in op['coefficients']]
            require(values == document.assignments['operators'][key]['coefficient'],
                    'Compiler changed ordered Yukawa coefficients')
        spectrum = read_json(run / 'spectrum.json')
        if expected_masses:
            for name, value in expected_masses.items():
                require(math.isclose(spectrum['masses'][name], value, rel_tol=1e-11, abs_tol=1e-8),
                        'Unexpected derived mass for ' + name)
        states = {state['name']: state for state in spectrum['scalar_states']}
        expected_states = {'x1': (6, '4/3'), 'x2': (6, '1/3'),
                           'x3': (6, '-2/3'), 'y1': (3, '-1/3')}
        require(set(states) == set(expected_states), 'Physical scalar content changed')
        for name, (colour, charge) in expected_states.items():
            require(states[name]['rep'] == colour and states[name]['charge'] == charge
                    and not states[name]['real'], 'Scalar representation or conjugation changed')
        transform = spectrum['transformation']
        require(len(transform) == 4 and all(len(row) == 4 for row in transform),
                'Unexpected scalar mixing dimensions')
        for i, row in enumerate(transform):
            for j, value in enumerate(row):
                value = complex(value['real'], value['imag']) if isinstance(value, dict) else complex(value)
                require(abs(value - int(i == j)) < 1e-11, 'Unexpected kinetic normalization or scalar mixing')
        validation = read_json(run / 'validation.json')
        require(validation['status'] == 'physical_inputs_validated', 'Physical validation did not complete')
        vacuum = validation['vacuum']
        require(vacuum['potential_minimized'] is False and vacuum['assumptions']['higgs_mass_gev'] == 125.,
                'Fixed SM vacuum convention changed')
        require(all(value == 0 for value in vacuum['assumptions']['scalar_vevs'].values()),
                'Unexpected new-scalar vacuum expectation value')
        self.report['checks'].append({'name': label, 'status': 'passed', 'active_operators': len(active),
                                      'masses_gev': spectrum['masses']})
        return run

    def fast(self):
        self.particles.write_bytes((ROOT / 'input' / 'particles.txt').read_bytes())
        self.cli('init', '--model', str(self.model), source=self.particles)
        generated = self.document()
        self.metadata_path = (self.model.parent / generated.assignments['_metadata']).resolve()
        require(self.metadata_path.is_relative_to(self.input), 'Generated metadata escaped input directory')
        self.metadata_digest = digest(self.metadata_path)
        require({key for key, row in generated.assignments['operators'].items() if row['enabled']} == FREE,
                'Initializer enabled an interaction by default')
        edit_values(self.model, assignments={
            'model_name': 'CI_triplet_sextet', 'output': '../saved results',
            'sm_inputs': {'Gf': 1.16637e-5, 'MZ': 91.1876, 'aEWM1': 127.9},
            'stable_particles': [],
            'collider': {'energy_tev': 13.6, 'max_final': 2, 'new_particles_only': True,
                         'pdf': 'nn23lo1', 'seed': 1, 'accuracy': .01, 'dynamical_scale_choice': 3},
            'decays': {'max_body': 2, 'precision': .01, 'min_br': 1e-8},
            'explicit_decays': {'history_ids': [], 'bwcutoff': 15., 'cut_decays': False},
        })
        self.cli('inputs')
        self.validation('generated free model', FREE,
                        {'H': 125., 'X1': 200., 'X2': 200., 'X3': 200., 'Y1': 600.})
        scalar_terms = set(COEFFICIENTS) - {'Q^2 X†', 'Y ec uc'}
        edit_values(self.model, operators={key: {'enabled': key in scalar_terms, 'coefficient': value}
                                          for key, value in COEFFICIENTS.items()})
        masses = {'H': 125., 'X1': 1500., 'X2': 1500., 'X3': 1500., 'Y1': 1200.}
        for key in ('Q^2 X†', 'Y ec uc'):
            edit_values(self.model, operators={key: {'enabled': True}})
            self.validation(key + ' alone', scalar_terms | {key}, masses)
            edit_values(self.model, operators={key: {'enabled': False}})
            require(self.document().assignments['operators'][key]['coefficient'] == COEFFICIENTS[key],
                    'Disabling an interaction erased its coefficient')
        qq_complex, eu_complex = list(QQ), list(EU)
        qq_complex[1], eu_complex[3] = .09 + .04j, -.06 + .08j
        edit_values(self.model, operators={'Q^2 X†': {'enabled': True, 'coefficient': qq_complex},
                                          'Y ec uc': {'enabled': True, 'coefficient': eu_complex}})
        self.validation('complex off-diagonal Yukawas', COEFFICIENTS, masses)
        edit_values(self.model, operators={'Q^2 X†': {'coefficient': QQ}, 'Y ec uc': {'coefficient': EU}})
        run = self.validation('full diagonal benchmark', COEFFICIENTS, masses)
        self.report['scope'] = {'fresh_ufo_export': False, 'numerical_widths': False,
                                'production_integrations': 0, 'package_defaults_modified': False}
        return run

    def widths(self, run):
        totals = {row['name']: row for row in read_json(run / 'total_widths.json')}
        decays = read_json(run / 'decays.json')
        modes = {(row['pdg'], tuple(sorted(row['daughters']))): row for row in decays['modes']}
        comparisons = []
        for name, target in WIDTH_TARGETS.items():
            value = totals[name]['width_gev']
            require(abs(value / target - 1) <= .001, 'Two-body total width disagrees for ' + name)
            comparisons.append({'mode': name, 'actual_gev': value, 'reference_gev': target,
                                'relative_difference': value / target - 1})
        for key, target in PARTIAL_TARGETS.items():
            require(key in modes, 'Missing benchmark decay mode')
            row = modes[key]
            value = row['width_gev']
            require(abs(value / target - 1) <= .001, 'Two-body partial width disagrees with PDF')
            require(row.get('primary_diagrams', {}).get('definitions'), 'Missing MadWidth primary diagram definitions')
            conjugate = (-key[0], tuple(sorted(-value for value in key[1])))
            require(conjugate in modes and math.isclose(modes[conjugate]['width_gev'], value, rel_tol=1e-10),
                    'Charge-conjugate decay mismatch')
            comparisons.append({'mode': row['process'], 'actual_gev': value, 'reference_gev': target,
                                'relative_difference': value / target - 1})
        self.report['width_comparisons'] = comparisons
        return modes[(9700001, (2, 2))]['branching_ratio']

    def ranking(self, run, selected, sigma, error, uu_br):
        # Stream the full CSV, including rows hidden by display cutoffs.
        calculated = uncalculated = uu = 0
        uu_row = None
        previous = math.inf
        ids = hashlib.sha256()
        with (run / 'ranking.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                ids.update((row['history_id'] + '\n').encode())
                if row['production_id'] != selected:
                    require(row['rate_pb'] == '' and row['production_cross_section_pb'] == '',
                            'Uncalculated channel was assigned a numerical rate')
                    uncalculated += 1
                    continue
                require(uncalculated == 0, 'Calculated ranking rows appear after uncalculated rows')
                value, br = float(row['rate_pb']), float(row['branching_ratio'])
                require(math.isfinite(value) and value >= 0 and value <= previous,
                        'Saved numerical ranking is not sorted in descending rate')
                require(math.isclose(value, sigma * br, rel_tol=1e-12, abs_tol=1e-16),
                        'Ranking rate is not production cross section times cascade BR')
                require(math.isclose(float(row['error_pb']), error * br, rel_tol=1e-12, abs_tol=1e-16),
                        'Ranking uncertainty does not propagate the integration error')
                require(math.isclose(float(row['production_cross_section_pb']), sigma, rel_tol=1e-12),
                        'Ranking used the wrong production cross section')
                if json.loads(row['final_pdgs']) == [2, 2]:
                    require(math.isclose(br, uu_br, rel_tol=1e-12), 'Saved uu branching ratio differs from MadWidth')
                    uu += 1
                    uu_row = {'history_id': row['history_id'], 'branching_ratio': br,
                              'rate_pb': value, 'error_pb': float(row['error_pb'])}
                previous = value
                calculated += 1
        require(calculated > 0 and uncalculated > 0 and uu == 1, 'Missing selected or uncalculated histories')
        return {'calculated': calculated, 'uncalculated': uncalculated,
                'history_ids_sha256': ids.hexdigest(), 'x1_to_uu': uu_row}

    def licensed(self, run, record_export):
        try:
            self.cli('check')
            self.cli('ufo')
        finally:
            # Capture a failed export before its isolated working directory is removed.
            record_export(trace_ufo.export_evidence(self))
        validation = read_json(run / 'validation.json')
        require(validation['status'] == 'ufo_validated' and validation['external_export_validated'],
                'Fresh FeynRules/MadGraph UFO export did not pass validation')
        require(Path(validation['ufo']).resolve().is_relative_to(self.output.resolve()),
                'Benchmark reused a UFO outside its fresh temporary outputs')
        vertices = trace_ufo.vertex_checks(Path(validation['ufo']))
        self.report['ufo_vertices'] = vertices
        require(vertices['vertices'] == vertices['reference_vertices'] and vertices['max_vertex_legs'] == 4
                and all(vertices['representative_vertices'].values()),
                'Saved UFO vertex count, arity or coloured benchmark vertices disagree with the reference')
        self.cli('channels')
        channels = read_json(run / 'production_channels.json')
        selected = [row['id'] for row in channels if row['process'] == 'p p > x1']
        require(len(selected) == 1 and len(channels) > 1, 'Expected resonant production channel was not discovered')
        selected = selected[0]
        self.cli('decays')
        uu_br = self.widths(run)
        self.cli('histories', '--page-size', '1')
        # Discovery must retain null rates until a production integration runs.
        with (run / 'ranking.csv').open(newline='', encoding='utf-8') as stream:
            discovered = 0
            for row in csv.DictReader(stream):
                require(row['rate_pb'] == '', 'Discovery unexpectedly integrated production')
                discovered += 1
        require(discovered > 0, 'No histories discovered')
        self.cli('rates', '--selected', selected, '--page-size', '1')
        rates = read_json(run / 'cross_sections.json')['rates']
        require(len(rates) == 1 and rates[0]['id'] == selected, 'Benchmark integrated an unselected channel')
        rate = rates[0]
        sigma, error = rate['cross_section_pb'], rate['error_pb']
        require(math.isfinite(sigma) and sigma > 0 and math.isfinite(error) and error > 0,
                'Production integration returned invalid numbers')
        require(rate['target_met'], 'Production integration did not reach its configured accuracy')
        cards = Path(rate['run_directory']) / 'process' / 'Cards'
        require(cards.resolve().is_relative_to(self.output.resolve()), 'Numerical cards escaped temporary outputs')
        card_values = {}
        for line in (cards / 'run_card.dat').read_text(encoding='utf-8').splitlines():
            line = line.split('!', 1)[0].split('#', 1)[0]
            if '=' in line:
                value, key = line.split('=', 1)
                card_values[key.strip()] = value.strip().strip("'\"")
        def number(key):
            return float(card_values[key].lower().replace('d', 'e'))
        require(number('ebeam1') == 6800 and number('ebeam2') == 6800
                and number('dynamical_scale_choice') == 3 and number('scalefact') == 1
                and card_values['pdlabel'] == 'nn23lo1'
                and card_values['fixed_ren_scale'].lower() in {'false', 'f', '.false.'}
                and card_values['fixed_fac_scale'].lower() in {'false', 'f', '.false.'},
                'Generated run card does not use the reference PDF, energy and scale prescription')
        selected_settings = ('ebeam1', 'ebeam2', 'pdlabel', 'dynamical_scale_choice',
                             'scalefact', 'fixed_ren_scale', 'fixed_fac_scale', 'iseed',
                             'ptj', 'pta', 'ptl', 'etaj', 'etaa', 'etal',
                             'drjj', 'drjl', 'draa', 'draj', 'dral', 'drll')
        self.report['numerical_card'] = {key: card_values[key] for key in selected_settings}
        self.report['parameter_card_sha256'] = digest(cards / 'param_card.dat')
        reference, reference_error = 1.1691955, .004566327258890234
        pull = (sigma - reference) / math.hypot(error, reference_error)
        require(abs(pull) <= 3, 'Production rate differs by more than three combined integration errors')
        self.report['production_comparison'] = {'process': 'p p > x1', 'cross_section_pb': sigma,
            'error_pb': error, 'reference_pb': reference, 'reference_error_pb': reference_error,
            'combined_mc_pull': pull, 'reference_type': 'previous validated package integration',
            'energy_tev': 13.6, 'pdf': 'nn23lo1', 'dynamical_scale_choice': 3,
            'max_body': 2, 'width_relative_tolerance': .001}
        before = self.ranking(run, selected, sigma, error, uu_br)
        attempts = {str(path.relative_to(run)) for path in (run / 'rates').glob('*/*/attempt-*')}
        require(len(attempts) == 1, 'Unexpected integration attempt count before resume')
        self.cli('rates', '--selected', selected, '--page-size', '20')
        require(attempts == {str(path.relative_to(run)) for path in (run / 'rates').glob('*/*/attempt-*')},
                'Resume repeated a completed production integration')
        require(self.ranking(run, selected, sigma, error, uu_br) == before,
                'Resume or pagination changed numerical rows or topology IDs')
        manifest = read_json(run / 'manifest.json')
        require(manifest['requested_complete'] and not manifest['all_rates_complete'],
                'Selected calculation completion was reported as all-channel completion')
        require(manifest['discovery_complete'], 'History discovery is incomplete')
        self.report['ranking'] = before
        self.report['scope'].update(fresh_ufo_export=True, numerical_widths=True, production_integrations=1)
        self.report['checks'].append({'name': 'fresh licensed export, widths, rate, ranking and resume', 'status': 'passed'})


def trace_original(candidate, trace, target):
    original = None
    trace['original'] = {'baseline_commit': trace_ufo.ORIGINAL, 'status': 'started',
                         'restored_sources': list(trace_ufo.FR_SOURCES),
                         'model_sha256': digest(candidate.model),
                         'metadata_sha256': candidate.metadata_digest}
    trace_ufo.save_json(target, trace)
    print('Tracing original FeynRules export (one export, no integration)', flush=True)
    evidence = {'observation': 'inconclusive'}
    phase = 'prepare_isolated_source'
    try:
        root = candidate.root / 'original export sandbox'
        root.mkdir()
        original = Benchmark(root, 'licensed', {'commands': []})
        trace_ufo.configure_original(ROOT, original, candidate)
        trace['original']['isolated_config_verified'] = True
        phase = 'run_instrumented_export'
        try:
            original.cli('ufo')
        finally:
            evidence = trace_ufo.export_evidence(original)
    except Exception as exc:
        # An instrumented rejection is expected. Preserve any other obstruction
        # separately; it must never skip the independent repaired benchmark.
        trace['original'].update(exception_type=type(exc).__name__, exception_phase=phase)
        if phase == 'prepare_isolated_source':
            trace['original']['reason'] = ('Pinned original FeynRules Git blob unavailable'
                                           if isinstance(exc, subprocess.CalledProcessError)
                                           else trace_ufo.safe_reason(str(exc)))
    status = {'guard_reproduced_unknown_fields': 'reproduced',
              'export_passed_without_unknown_fields': 'not_reproduced',
              'inconclusive': 'blocked'}[evidence['observation']]
    if status == 'reproduced' and evidence['fields']['match_level'] == 'first_unclassified':
        status = 'blocked'
        trace['original']['trace_error'] = 'Original export did not activate the signature replay'
    if 'fields' in evidence and 'matching_vertices_at_selected_level' in evidence['fields']:
        trace['original']['reference_association'] = {
            'match_level': evidence['fields']['match_level'],
            'matching_vertices': evidence['fields']['matching_vertices_at_selected_level'],
            'historical_coupling_id_verified': False}
    trace['original'].update(evidence, status=status, commands=original.report['commands'] if original else [])
    trace_ufo.save_json(target, trace)
    print('Original export trace:', status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fast', 'licensed'), required=True)
    parser.add_argument('--trace-original-fields', action='store_true',
                        help='Compare the original instrumented export with the repaired licensed export')
    args = parser.parse_args()
    if args.trace_original_fields and args.mode != 'licensed':
        parser.error('--trace-original-fields requires --mode licensed')
    destination = Path(os.environ.get('BENCHMARK_ARTIFACTS', ROOT / 'ci-artifacts')).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'madgraph-step-ci-benchmark-v1', 'mode': args.mode, 'status': 'running',
              'platform': platform.system(), 'architecture': platform.machine(),
              'python': platform.python_version(), 'checks': [], 'commands': [],
              'limitations': ['Fast mode does not run Mathematica or establish numerical physics agreement.',
                             'Licensed mode checks one production channel, not all production rates.',
                             'Two-body tree-level widths do not establish complete physical decay widths.',
                             'Monte Carlo agreement does not include perturbative/PDF uncertainty.']}
    started = time.monotonic()
    code = 0
    trace = {'schema': 'madgraph-step-ufo-field-trace-v1',
             'original': {'status': 'pending' if args.trace_original_fields else 'not_requested'},
             'candidate': {'observation': 'not_started'}}
    trace_path = destination / 'field-trace.json'
    try:
        report['python_packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}
        with tempfile.TemporaryDirectory(prefix='madgraph release benchmark ') as temporary:
            benchmark = Benchmark(Path(temporary), args.mode, report)
            run = benchmark.fast()
            if args.mode == 'licensed':
                if args.trace_original_fields:
                    trace_original(benchmark, trace, trace_path)

                def record_export(evidence):
                    trace['candidate'] = evidence
                    trace_ufo.save_json(trace_path, trace)

                benchmark.licensed(run, record_export)
                shutil.copyfile(run / 'ranking.csv', destination / 'ranking.csv')
                report['candidate_physics'] = 'passed'
                require(trace['candidate']['observation'] == 'export_passed_without_unknown_fields',
                        'Candidate export diagnostics were inconclusive')
                if args.trace_original_fields:
                    require(trace['original']['status'] == 'reproduced',
                            'Candidate benchmark passed but the original unidentified field was not verified')
        report['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report['status'] = 'interrupted' if code == 130 else 'failed'
        # Only our controlled assertion messages are publishable. Third-party
        # exceptions/logs may contain machine paths or licensing information.
        report['failure'] = {'type': type(exc).__name__}
        if isinstance(exc, AssertionError):
            report['failure']['message'] = str(exc)
        print('Benchmark failed:', type(exc).__name__, flush=True)
        if isinstance(exc, AssertionError):
            print(str(exc), flush=True)
    finally:
        if args.mode == 'licensed':
            report['field_trace'] = {'file': trace_path.name, 'original': trace['original']['status'],
                                     'candidate': trace['candidate']['observation']}
            trace_ufo.save_json(trace_path, trace)
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        target = destination / ('benchmark-' + args.mode + '.json')
        trace_ufo.save_json(target, report)
    print('Benchmark', report['status'], '(' + args.mode + ')', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
