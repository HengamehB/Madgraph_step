"""Compare a saved sextet example run with the numerical tables beside this file."""
import argparse
import csv
import json
import math
import sys
from collections import Counter
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


def indexed(folder, name, key):
    result = {}
    for row in rows(folder, name):
        k = key(row)
        if k in result:
            raise ValueError(f'Duplicate {name} entry for {k}')
        result[k] = row
    return result


def check_widths(run):
    totals = indexed(run, 'total_widths', lambda row: int(row['pdg']))
    modes = indexed(run, 'decays', mode_key)
    compact = True
    for name, actual, key in [('total_widths', totals, lambda row: int(row['pdg'])), ('decays', modes, mode_key)]:
        reference = indexed(REFERENCE, name, key)
        compact = compact and actual.keys() == reference.keys()
        for k, ref in reference.items():
            if k not in actual:
                raise ValueError(f'Missing {name} entry for {k}')
            ref_width = positive(ref, 'width_gev')
            check_close(f'{name} {k} (GeV)', positive(actual[k], 'width_gev'), ref_width, .001 * ref_width)
            if name == 'decays':
                if actual[k]['process'].split() != ref['process'].split():
                    raise ValueError(f'decays {k}: process label differs from the reference decay')
                ref_br = positive(ref, 'branching_ratio')
                check_close(f'decays {k} BR', float(actual[k]['branching_ratio']), ref_br, .001 * ref_br)
    branching = {}
    for key, mode in modes.items():
        # The package stores one total width for a particle and its antiparticle.
        parent, _ = key
        total = positive(totals[abs(parent)], 'width_gev')
        ratio = positive(mode, 'width_gev') / total
        br = positive(mode, 'branching_ratio')
        check_close(f'decays {key} partial / total width', br, ratio, 1e-10 * ratio)
        branching[key] = br
    return branching, compact


def cascade_br(nodes, branching):
    """Reconstruct a forest probability, including assignments to identical parents."""
    particles = Counter(pdg for pdg, _ in nodes)
    subtrees = Counter(json.dumps(node) for node in nodes)
    weight = (math.prod(math.factorial(n) for n in particles.values()) /
              math.prod(math.factorial(n) for n in subtrees.values()))
    for pdg, children in nodes:
        if children:
            key = pdg, tuple(sorted(daughter for daughter, _ in children))
            weight *= branching[key] * cascade_br(children, branching)
    return weight


def compare(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('requested_complete'):
        raise ValueError('The requested calculation is incomplete. Finish the sample run first.')
    channels = indexed(run, 'production_channels', lambda row: row['id'])
    reference_channels = indexed(REFERENCE, 'production_channels', lambda row: row['id'])
    if channels.keys() != reference_channels.keys() or any(
        row['process'] != reference_channels[k]['process'] or
        json.loads(row['final_pdgs']) != json.loads(reference_channels[k]['final_pdgs'])
        for k, row in channels.items()
    ):
        raise ValueError('The discovered production channels differ from the sextet example.')

    integrated = indexed(run, 'cross_sections', lambda row: row['process'])
    if PROCESS not in integrated:
        raise ValueError(f'No numerical cross section for {PROCESS}. Run rates with --selected C0004.')
    prod = integrated[PROCESS]
    ref_prod = one(rows(REFERENCE, 'cross_sections'), 'reference production channel')
    sigma, err = positive(prod, 'cross_section_pb'), positive(prod, 'error_pb')
    if err > .01 * sigma:
        raise ValueError('The production integration did not achieve the sample’s 1% precision.')
    check_close('Production cross section (pb)', sigma, positive(ref_prod, 'cross_section_pb'),
                3 * math.hypot(err, positive(ref_prod, 'error_pb')))
    branching, compact = check_widths(run)
    # The published numerical excerpt records ten primary modes. Fresh package
    # outputs publish all contributing modes in the decays table.
    published = json.loads((REFERENCE / 'manifest.json').read_text(encoding='utf-8'))
    excerpt = compact and manifest.get('source_ranking_sha256') == published['source_ranking_sha256']

    expected = {row['history_id']: row for row in rows(REFERENCE, 'ranking')}
    total = len(expected)
    count = extra = 0
    uu = None
    previous_rate = math.inf
    for row in rows(run, 'ranking'):
        hid = row['history_id']
        ref = expected.pop(hid, None)
        if ref is None:
            raise ValueError(f'Unknown or duplicate decay history {hid}')
        forest = json.loads(row['history_key'])
        for key in ('production_id', 'production_process', 'history', 'final_state'):
            if row[key] != ref[key]:
                raise ValueError(f'{hid}: {key} differs from the reference history')
        for key in ('production_final_pdgs', 'final_pdgs', 'history_key'):
            if json.loads(row[key]) != json.loads(ref[key]):
                raise ValueError(f'{hid}: {key} differs from the reference history')

        br = positive(row, 'branching_ratio')
        ref_br = positive(ref, 'branching_ratio')
        check_close(f'{hid} cascade BR', br, ref_br, .001 * ref_br)
        try:
            tree_br = cascade_br(forest, branching)
        except KeyError as exc:
            if not excerpt:
                raise ValueError(f'{hid}: missing primary decay {exc} in decays.csv') from exc
        else:
            check_close(f'{hid} cascade BR from primary decays', br, tree_br, 1e-10 * tree_br)

        process = row['production_process']
        if process not in integrated:
            if any(row[k] != '' for k in ('rate_pb', 'production_cross_section_pb', 'production_error_pb', 'error_pb')):
                raise ValueError(f'{hid}: calculated rate and production coverage disagree')
            previous_rate = -math.inf
            continue
        chan = integrated[process]
        if chan['id'] != row['production_id']:
            raise ValueError(f'{hid}: cross section channel ID differs from ranking')
        production = float(chan['cross_section_pb'])
        error = float(chan['error_pb'])
        if not all(math.isfinite(v) and v >= 0 for v in (production, error)):
            raise ValueError(f'{hid}: invalid production cross section or error')
        for key, target in [('production_cross_section_pb', production), ('production_error_pb', error),
                            ('rate_pb', production * br), ('error_pb', error * br)]:
            check_close(f'{hid} {key}', float(row[key]), target, 1e-10 * target)
        rate = float(row['rate_pb'])
        if rate > previous_rate:
            raise ValueError(f'{hid}: rankings out of order')
        previous_rate = rate
        if process != PROCESS:
            extra += 1
            continue
        if forest == [[9700001, [[2, []], [2, []]]]]:
            check_close('pp → x1 → u u rate (pb)', rate, positive(ref, 'rate_pb'),
                        3 * math.hypot(float(row['error_pb']), positive(ref, 'error_pb')))
            uu = (br, rate)
        count += 1
    if expected or uu is None:
        raise ValueError(f'Incomplete ranking: {len(expected)} reference histories missing.')
    print(f'PASS: {len(channels)} production channels, {total:,} histories, {count} selected rates, '
          '4 reference total widths and 10 reference partial widths.')
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
