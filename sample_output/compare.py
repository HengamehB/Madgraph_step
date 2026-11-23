"""Compare a saved sextet example run with the numerical tables beside this file."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent
PROCESS = 'p p > x1'


def rows(folder, name):
    with (folder / (name + '.csv')).open(encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def positive(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Expected a positive finite {key}, found {row[key]!r}')
    return value


def one(items, label):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f'Expected one {label}, found {len(items)}')
    return items[0]


def check_close(label, actual, reference, tolerance):
    if not math.isfinite(actual) or abs(actual - reference) > tolerance:
        raise ValueError(f'{label}: got {actual:.8g}, expected {reference:.8g} ± {tolerance:.3g}')


def mode_key(row):
    return int(row['pdg']), tuple(sorted(json.loads(row['daughters'])))


def check_widths(run, name, key):
    expected = {key(row): row for row in rows(REFERENCE, name)}
    checked = set()
    for row in rows(run, name):
        k = key(row)
        if k in expected:
            if k in checked:
                raise ValueError(f'Duplicate {name} entry for {k}')
            ref = positive(expected[k], 'width_gev')
            check_close(f'{name} {k} (GeV)', positive(row, 'width_gev'), ref, .001 * ref)
            checked.add(k)
    if checked != expected.keys():
        raise ValueError(f'Missing reference widths in {name}: {expected.keys() - checked}')
    return len(checked)


def compare(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('requested_complete'):
        raise ValueError('The requested calculation is incomplete. Finish the sample run first.')
    expected_channels = {r['process'] for r in rows(REFERENCE, 'production_channels')}
    if {r['process'] for r in rows(run, 'production_channels')} != expected_channels:
        raise ValueError('The discovered production channels differ from the sextet example.')

    integrated = {r['process']: r for r in rows(run, 'cross_sections')}
    if PROCESS not in integrated:
        raise ValueError(f'No numerical cross section for {PROCESS}. Run rates with --selected C0004.')
    prod = integrated[PROCESS]
    ref_prod = one(rows(REFERENCE, 'cross_sections'), 'reference production channel')
    sigma, err = positive(prod, 'cross_section_pb'), positive(prod, 'error_pb')
    if err > .01 * sigma:
        raise ValueError('The production integration did not achieve the sample’s 1% precision.')
    check_close('Production cross section (pb)', sigma, positive(ref_prod, 'cross_section_pb'),
                3 * math.hypot(err, positive(ref_prod, 'error_pb')))
    totals = check_widths(run, 'total_widths', lambda row: int(row['pdg']))
    partials = check_widths(run, 'decays', mode_key)

    expected = {row['history_id']: row for row in rows(REFERENCE, 'ranking')}
    total = len(expected)
    count = extra = 0
    uu = None
    for row in rows(run, 'ranking'):
        hid = row['history_id']
        ref = expected.pop(hid, None)
        if ref is None or row['production_process'] != ref['production_process']:
            raise ValueError(f'Unknown, duplicate, or mismatched decay history {hid}')
        if ref['production_process'] != PROCESS:
            has_rate = row['rate_pb'] != ''
            if has_rate != (row['production_process'] in integrated):
                raise ValueError(f'{hid}: calculated rate and production coverage disagree')
            if not has_rate and any(row[k] for k in ('production_cross_section_pb', 'production_error_pb', 'error_pb')):
                raise ValueError(f'{hid}: an uncalculated history contains numerical production data')
            extra += int(has_rate)
            continue
        br, rate = positive(row, 'branching_ratio'), float(row['rate_pb'])
        check_close(f'{hid} production cross section (pb)', float(row['production_cross_section_pb']),
                    sigma, 1e-10 * sigma)
        ref_br = positive(ref, 'branching_ratio')
        check_close(f'{hid} cascade BR', br, ref_br, .001 * ref_br)
        check_close(f'{hid} cross section × BR (pb)', rate, sigma * br, 1e-10 * sigma * br)
        check_close(f'{hid} integration error (pb)', float(row['error_pb']), err * br, 1e-10 * err * br)
        if json.loads(ref['final_pdgs']) == [2, 2]:
            check_close('pp → x1 → u u rate (pb)', rate, positive(ref, 'rate_pb'),
                        3 * math.hypot(float(row['error_pb']), positive(ref, 'error_pb')))
            uu = (br, rate)
        count += 1
    if expected or uu is None:
        raise ValueError(f'Incomplete ranking: {len(expected)} reference histories missing.')
    print(f'PASS: {len(expected_channels)} production channels, {total:,} histories, {count} selected rates, '
          f'{totals} total widths and {partials} partial widths.')
    print(f'pp → x1 = {sigma:.8g} ± {err:.3g} pb, BR(x1 → u u) = {uu[0]:.8g}, '
          f'σ × BR = {uu[1]:.8g} pb.')
    if extra:
        print(f'This run also contains {extra:,} calculated histories from other production channels.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path, help='directory printed after Run files')
    try:
        compare(parser.parse_args().run_directory)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        sys.exit(f'Comparison failed: {exc}')
