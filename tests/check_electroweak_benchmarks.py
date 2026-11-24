"""Reproduce the singlet and inert-doublet comparisons from a saved local run.

Usage: python tests/check_electroweak_benchmarks.py "output/CASE/RUN"
"""
import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(run, name):
    with (run / (name + '.csv')).open(encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def one(items):
    item, = items
    return item


def compare(run):
    sys.path.insert(0, str(ROOT / 'vendor/MG5_aMC'))
    sys.path.insert(0, str(ROOT))
    from models import import_ufo
    from models.model_reader import ModelReader
    from pipeline_storage import atomic_text, write_json

    manifest = json.loads((run / 'manifest.json').read_text())
    case = manifest['model']
    if case not in {'phantom_portal', 'inert_doublet'}:
        raise ValueError(f'Unexpected model: {case}')
    source = one(run.glob('export/' + manifest['stage_fingerprints']['export'][:16] + '/attempt-*/*_UFO'))
    with tempfile.TemporaryDirectory(prefix='mfr_verify_') as temp:
        ufo = shutil.copytree(source, Path(temp) / 'model')
        model = ModelReader(import_ufo.import_full_model(str(ufo), prefix=''))
        model.set_parameters_and_couplings(str(source.parent / 'param_card.dat'))

    def vertex(*pdgs):
        interaction = one(it for it in model['interactions'] if
                          sorted(p.get_pdg_code() for p in it['particles']) == sorted(pdgs))
        return model['coupling_dict'][one(interaction['couplings'].values())]

    checks = []
    def check(observable, expected, measured, rtol=0.001):
        error = abs(measured - expected)
        checks.append(dict(observable=observable, expected=expected, measured=measured,
                           relative_difference=error/abs(expected) if expected else None,
                           passed=math.isclose(expected, measured, rel_tol=rtol, abs_tol=1e-12)))

    def partial(pdg, daughters):
        return one(row for row in records(run, 'decays') if int(row['pdg']) == pdg and
                   sorted(json.loads(row['daughters'])) == sorted(daughters))

    # Physical targets are independent of the exported mass and coupling values.
    gf, mz, inve = 1.16637e-5, 91.1876, 127.9
    v = (math.sqrt(2)*gf)**-0.5
    mw = math.sqrt(mz*mz/2 + math.sqrt(mz**4/4 - math.pi*mz*mz/(math.sqrt(2)*gf*inve)))
    pole = lambda pdg: model['parameter_dict'][model.get_particle(pdg)['mass']].real
    check('UFO Higgs mass [GeV]', 125., pole(25), 1e-8)
    modes = []
    if case == 'phantom_portal':
        source_note = ('Chosen Z2-odd real singlet at 50 GeV, rho=0.01 in L = -rho X² H†H/2. '
                       '[Silveira and Zee](https://doi.org/10.1016/0370-2693(85)90624-0), '
                       '[analytic Higgs width, Eq. 21](https://arxiv.org/pdf/1207.4930).')
        check('UFO X mass [GeV]', 50., pole(9000001), 1e-8)
        coup = vertex(25, 9000001, 9000001)
        check('Im(hXX coupling) [GeV]', -0.01*v, coup.imag)
        check('Re(hXX coupling) [GeV]', 0., coup.real)
        expected = 0.01**2*v**2*math.sqrt(1-4*50**2/125**2)/(32*math.pi*125)
        decay = partial(25, [9000001, 9000001])
        check('Higgs partial width to two singlets [GeV]', expected, float(decay['width_gev']))
        stable = one(row for row in records(run,'total_widths') if int(row['pdg']) == 9000001)
        check('Stable X total width [GeV]', 0., float(stable['width_gev']))
        modes.append({'process':decay['process'], 'branching_ratio':float(decay['branching_ratio'])})
        for row in records(run,'ranking'):
            if row['rate_pb'].strip():
                raise ValueError('Discovery-only singlet has a numerical rate')
        scope = 'Decay and discovery only. The single on-shell Higgs partial width is compared independently of production.'
    else:
        source_note = ('Chosen inert Z2 spectrum in the one-VEV solution C of '
                       '[Deshpande and Ma, Eq. 6](https://doi.org/10.1103/PhysRevD.18.2574). '
                       'In their potential λ2=0.2, λ3=2a, λ4=λ5=−a, with a=(220²−100²)/v².')
        for pdg, name, target in [(9300001,'charged',220.),(9300002,'light neutral',100.),(9300003,'heavy neutral',220.)]:
            check(f'UFO {name} mass [GeV]', target, pole(pdg), 1e-8)
        wp = vertex(-24,9300001,9300002)
        wm = vertex(24,-9300001,9300002)
        check('Magnitude of W cubic coupling', mw/v, abs(wp))
        check('Charge conjugate W vertex, absolute mismatch', 0., abs(wp.conjugate()-wm))
        check('Magnitude of Z cubic coupling', mz/v, abs(vertex(23,9300002,9300003)))
        for parent, boson, mvec in [(9300001,24,mw),(-9300001,-24,mw),(9300003,23,mz)]:
            decay = partial(parent, [9300002,boson])
            kallen = (220**2-(100+mvec)**2)*(220**2-(100-mvec)**2)
            expected = kallen**1.5/(16*math.pi*v*v*220**3)
            check(decay['process']+' partial width [GeV]', expected, float(decay['width_gev']))
            check(decay['process']+' BR', 1., float(decay['branching_ratio']))
            modes.append({'process':decay['process'],'conjugated':decay['conjugated']=='True'})
        scope = ('Selected positively charged plus heavy neutral pair. MadWidth supplies the positive-parent width; '
                 'the negative-parent row uses charge conjugation. Both charge vertices were checked in the UFO.')

    paths = {k:str(p.relative_to(run)) for k,p in {
        'Lagrangian':run/'lagrangian.tex','Spectrum':run/'spectrum.json', 'FeynRules source':source.parent/'feynrules_source',
        'FeynRules log':source.parent/'feynrules.log','UFO':source,
        'MadGraph import log':source.parent/'madgraph_import/madgraph.log',
        'Parameter card':source.parent/'param_card.dat','Decay card':one(run.glob('decays/*/*/param_card.dat')),
        'MadWidth log':one(run.glob('decays/*/*/madgraph.log')), 'Widths':run/'decays.csv',
        'Ranked CSV':run/'ranking.csv','Report':run/'report.html','Completion manifest':run/'manifest.json'}.items()}
    production = None
    if case == 'inert_doublet':
        prod = one(records(run,'cross_sections'))
        if sorted(json.loads(prod['final_pdgs'])) != [9300001,9300003]:
            raise ValueError('Wrong integrated production channel')
        sigma, integration_error = float(prod['cross_section_pb']), float(prod['error_pb'])
        if not (0<sigma and 0<integration_error<=.01*sigma and prod['target_met']=='True'):
            raise ValueError('Incomplete production integration')
        first=None
        count=0
        for row in records(run,'ranking'):
            if row['production_id'] != prod['id']:
                if row['rate_pb'].strip():
                    raise ValueError('Unexpected numerical channel')
                continue
            rate=float(row['rate_pb'])
            if not math.isclose(rate,sigma*float(row['branching_ratio']),rel_tol=1e-12):
                raise ValueError(f"Rate mismatch: {row['history_id']}")
            if first is None:
                first=dict(history=row['history'],rate_pb=rate)
            count+=1
        if count!=manifest['history_counts']['calculated'] or not count:
            raise ValueError('Incomplete ranking')
        production=dict(process=prod['process'],sigma_pb=sigma,error_pb=integration_error,
                        computed_histories=count,leading_history=first)
        paths['Run card']=str(one(run.glob('rates/**/*/Cards/run_card.dat')).relative_to(run))
        paths['Integration log']=str(one(run.glob('rates/**/madgraph.log')).relative_to(run))
    for name, path in paths.items():
        if not (run/path).exists(): raise FileNotFoundError(f'{name}: {path}')
    passed=all(c['passed'] for c in checks)
    result=dict(case=case,run_id=manifest['run_id'],passed=passed,source=source_note,scope=scope,
                tolerance='0.1% for nonzero widths and couplings',checks=checks,decays=modes,
                production=production,history_counts=manifest['history_counts'],artifacts=paths)
    write_json(run/'comparison.json', result)
    lines=[f'# {case.replace("_"," ").title()} benchmark', '', source_note,'',scope,'',
           '| Observable | Expected | Measured | Pass |','|---|---:|---:|:---:|']
    for c in checks:
        lines.append(f"| {c['observable']} | {c['expected']:.10g} | {c['measured']:.10g} | {'Yes' if c['passed'] else 'No'} |")
    if production:
        lines.extend(['',f"Production {production['process']} = {sigma:.9g} ± {integration_error:.3g} pb. "
                        f"{count} of {manifest['history_counts']['total']} histories have a rate."])
    links=' · '.join(f'[{key}](<{path}>)' for key,path in paths.items())
    atomic_text(run/'comparison.md','\n'.join([*lines,'',links,'']))
    print(('PASS' if passed else 'FAIL'),case,run)
    return passed


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    raise SystemExit(0 if compare(parser.parse_args().run_directory.resolve()) else 1)
