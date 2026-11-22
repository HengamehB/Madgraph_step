"""Automatic gauge-matter component models with checked physical bases.

SU(2) singlet complex scalars and Dirac fermions support colour 1, 3 and 3bar,
norm potentials and diagonal covariant kinetic/mass terms. A colour-singlet
Y=1/2 scalar doublet supports all catalogue potential and 3x3 Yukawa families,
including complex neutral mixing, at a stationary zero new-field VEV. Nonzero
Higgs-doublet kinetic mixing is diagnosed. A real Y=0 scalar triplet supports
the stationary zero-VEV sector. Neutral Proca norms require an explicit Proca
specification and a healthy kinetic combination. Other selected contractions
are rejected explicitly; coefficients follow the catalogue monomial convention.
"""
from collections import Counter
from dataclasses import dataclass
import math
from itertools import permutations

import numpy as np
import generic_colour_yukawa as yukawa


def _doublet(fields):
    return (len(fields) == 1 and fields[0].spin == 'scalar' and fields[0].kind == 'complex'
            and fields[0].su2 == 2 and fields[0].su3 == (0, 0) and fields[0].hypercharge == 1 / 2)


def _triplet(fields):
    return (len(fields) == 1 and fields[0].spin == 'scalar' and fields[0].kind == 'real'
            and fields[0].su2 == 3 and fields[0].su3 == (0, 0) and fields[0].hypercharge == 0)


def _supported(f):
    if f.su2 != 1:
        return False
    if f.spin == 'vector':
        return f.kind == 'real' and f.su3 == (0, 0) and f.hypercharge == 0
    return (f.su3 in ((0, 0), (1, 0), (0, 1)) and
            (f.spin == 'scalar' and f.kind == 'complex' or f.spin == 'fermion' and f.kind == 'dirac'))


def supports_fields(fields):
    return bool(fields) and (all(_supported(f) for f in fields) or _doublet(fields) or _triplet(fields))


def _counts(row):
    return Counter({part['name']: part['count'] for part in row['factors']})


def classify(row, fields):
    """Classify only contractions whose basis is explicitly fixed here."""
    if _doublet(fields):
        return _doublet_classify(row, fields[0].name)
    if all(f.spin == 'scalar' for f in fields) and yukawa.classify(row, fields) == yukawa.TRIPLET:
        return yukawa.TRIPLET, ''
    if _triplet(fields):
        counts, n = _counts(row), fields[0].name
        if row['multiplicity'] == 1:
            if counts == {f'D({n})': 2}:
                return 'triplet_kinetic', ''
            if counts == {'H': 1, 'H†': 1, n: 1}:
                return 'triplet_odd_portal', ''
            if counts in ({n: 2}, {n: 4}, {'H': 1, 'H†': 1, n: 2}):
                return 'triplet_potential', ''
        return None, 'unresolved real-triplet contraction'
    counts = _counts(row)
    for f in fields:
        n = f.name
        if f.spin == 'scalar' and counts == {f'D({n})': 1, f'D({n}†)': 1} and row['multiplicity'] == 1:
            return 'scalar_covariant_kinetic', ''
        if f.spin == 'fermion':
            if counts == {n + '_L': 1, n + '_Rc': 1} and row['multiplicity'] == 1:
                return 'dirac_mass', ''
            for chirality in ('_L', '_Rc'):
                token = n + chirality
                if counts == {f'D({token})': 1, token + '†': 1} and row['multiplicity'] == 1:
                    return 'dirac_covariant_kinetic', ''
        if f.spin == 'vector' and counts == {f'D({n})': 2} and row['multiplicity'] == 4:
            return 'proca_kinetic', ''
    if row['derivatives']:
        return None, 'derivative mixing/current contractions need a complete physical-basis model'
    remainder = Counter(counts)
    if remainder['H'] != remainder['H†']:
        return None, 'Higgs/flavour contraction is not a norm potential'
    remainder.pop('H', None)
    remainder.pop('H†', None)
    for f in fields:
        if f.spin == 'scalar':
            if remainder[f.name] != remainder[f.name + '†']:
                return None, 'non-norm scalar contractions require component mixing or an explicit invariant basis'
            remainder.pop(f.name, None)
            remainder.pop(f.name + '†', None)
        elif f.spin == 'vector':
            if remainder[f.name] % 2:
                return None, 'odd-vector interactions require additional Lorentz/current contractions'
            remainder.pop(f.name, None)
    if remainder or row['multiplicity'] != 1:
        return None, 'selected flavour/mixing or multiple-invariant operator requires an explicit physical-basis contraction'
    return 'norm_potential', ''


@dataclass
class GaugeMatterCompiler:
    fields: tuple
    operators: tuple
    normalization: dict
    physical_masses: dict
    vev: float
    higgs_mass: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {'H': self.higgs_mass, **self.physical_masses}

    def write_sources(self, model, source, fr):
        from feynrules_export import ModelResolutionError, CompilerResolutionError, _wl, _internal
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        marker = 'Mass            -> {MH,125}'
        if marker not in sm:
            raise CompilerResolutionError('SM Higgs mass declaration changed; review the FeynRules template')
        sm = sm.replace(marker, f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile = source / 'SM_gauge_matter.fr'
        smfile.write_text(sm)
        classes, particles, parameters, terms = [], [], [], []
        symbols, coefficient_symbols, raw_scalars = {}, {}, {}
        has_np_interactions = any(op.kind == 'norm_potential' and sum(count for _, count in op.factors) > 2
                                  and any(value != 0 for value in op.coefficients) for op in self.operators)
        for op in self.operators:
            if op.kind in yukawa.KINDS:
                continue  # The common flavour renderer declares its own coupling parameters.
            for index, value in enumerate(op.coefficients, 1):
                key = (op.operator_id, index)
                coefficient_symbols[key] = f'pmGM{op.operator_id[1:]}n{index}'
                interaction = value != 0 and op.kind == 'norm_potential' and sum(count for _, count in op.factors) > 2
                declaration = _internal(coefficient_symbols[key], value, order=interaction)
                higgs = sum(count for token, count in op.factors if token in ('H', 'H†'))
                if interaction and higgs:
                    declaration = declaration.replace('InteractionOrder -> {NP, 1}',
                        f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
                parameters.append(declaration)
        scalar_maps = {(f.name, 0): {(i, f.su3 == (0, 1)): 1} for i, f in enumerate(self.fields) if f.spin == 'scalar'}
        lepton_numbers, lepton_note = yukawa.lepton_labels(self.operators, scalar_maps)
        for index, f in enumerate(self.fields, 1):
            letter = {'scalar': 'S', 'fermion': 'F', 'vector': 'V'}[f.spin]
            symbol = f'pmGM{letter}{index}'
            symbols[f.name] = symbol
            colour = f.su3 != (0, 0)
            conjugated = f.su3 == (0, 1)
            charge = float(f.hypercharge) * (-1 if conjugated else 1)
            lepton = f', LeptonNumber -> {lepton_numbers[index-1]}' if index-1 in lepton_numbers else ''
            self_conjugate = f.spin == 'vector'
            physical_name = f.name.lower() + ('bar' if conjugated else '')
            anti_name = physical_name if self_conjugate else (f.name.lower() if conjugated else physical_name + 'bar')
            mass = self.physical_masses[f.name]
            width = model.inputs.get('widths', {}).get(f.name, 0.)
            pdg = 9200000 + index
            extra = 'Indices -> {Index[Colour]},' if colour else ''
            if not self_conjugate:
                extra += f' AntiParticleName -> {_wl(anti_name)},'
            classes.append(f'''{letter}[{200+index}] == {{ClassName -> {symbol}, SelfConjugate -> {str(self_conjugate)},
              {extra} Mass -> {{pmGMM{index},{_wl(mass)}}}, Width -> {{pmGMW{index},{_wl(width)}}},
              QuantumNumbers -> {{Y -> {_wl(charge)}, Q -> {_wl(charge)}{lepton}}},
              ParticleName -> {_wl(physical_name)}, PDG -> {pdg},
              PropagatorLabel -> {_wl(f.name)}, PropagatorType -> {'D' if f.spin == 'scalar' else 'Straight' if f.spin == 'fermion' else 'Sine'},
              PropagatorArrow -> {'None' if self_conjugate else 'Forward'}, FullName -> {_wl(f.name)}}}''')
            particles.append(dict(name=physical_name, antiname=anti_name, pdg=pdg,
                                  source=f.name, spin={'scalar': 1, 'fermion': 2, 'vector': 3}[f.spin],
                                  mass=mass, width=width))
            if f.spin == 'scalar':
                argument = '[cc]' if colour else ''
                terms.append(f'DC[{symbol}bar{argument},mu]*DC[{symbol}{argument},mu]')
                raw = symbol + ('bar' if conjugated else '') + ('[{index}]' if colour else '')
                raw_scalars[f.name, 0] = f'({raw}/Sqrt[pmGMZ{index}])'
            elif f.spin == 'fermion':
                # FR contracts the common colour index in a spinor chain.
                terms.extend((f'I*{symbol}bar.Ga[mu].DC[{symbol},mu]', f'-pmGMM{index}*{symbol}bar.{symbol}'))
            else:
                terms.append(f'-1/4*(del[{symbol}[nu],mu]-del[{symbol}[mu],nu])*(del[{symbol}[nu],mu]-del[{symbol}[mu],nu])')

        for op in self.operators:
            if op.kind in yukawa.KINDS:
                declarations, interactions = yukawa.render(op, raw_scalars)
                parameters.extend(declarations)
                terms.extend(interactions)
                has_np_interactions |= bool(interactions)
                continue
            if op.kind != 'norm_potential':
                continue
            counts = Counter(dict(op.factors))
            pieces, number = [], 0
            for _ in range(counts['H']):
                pieces.append('((vev+H)^2/2)')
            for f in self.fields:
                if f.spin == 'fermion':
                    continue
                repetitions = counts[f.name] if f.spin == 'scalar' else counts[f.name] // 2
                for _ in range(repetitions):
                    number += 1
                    symbol = symbols[f.name]
                    if f.spin == 'vector':
                        pieces.append(f'({symbol}[m{number}]*{symbol}[m{number}]/pmGMZ{self.fields.index(f)+1})')
                    else:
                        arg = f'[c{number}]' if f.su3 != (0, 0) else ''
                        pieces.append(f'({symbol}bar{arg}*{symbol}{arg}/pmGMZ{self.fields.index(f)+1})')
            terms.append(coefficient_symbols[(op.operator_id, 1)] + '*' + '*'.join(pieces))
        for index, f in enumerate(self.fields, 1):
            if f.spin != 'fermion':
                parameters.append(_internal(f'pmGMZ{index}', self.normalization[f.name]))
        extension = source / 'gauge_matter.fr'
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                            + lepton_note +
                            'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}' + (',{NP,1}' if has_np_interactions else '') + '};\n'
                            'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
                            'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
                            '(* Diagonal fields are canonically normalized. Dirac mass phases are removed by a chiral rephasing. *)\n'
                            '(* Proca derivative basis: d_mu V_nu d^mu V^nu, (d.V)^2, d_mu V_nu d^nu V^mu, epsilon dV dV. *)\n'
                            '(* The last term is a total derivative; the second and third agree up to integration by parts. *)\n'
                            'LGM := Block[{mu,nu,cc,c1,c2,c3,c4,m1,m2,m3,m4},\n'
                            'ExpandIndices[' + ' + '.join(terms) + ', FlavorExpand -> {SU2D,SU2W}]];\n')
        return [smfile, extension], 'LSM + LGM', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import (ModelResolutionError, CompilerResolutionError, PreparedModel, ResolvedOperator,
                                 _fields, _coefficients, _number)
    fields = tuple(_fields(lagrangian))
    if not supports_fields(fields):
        return None
    if getattr(lagrangian._source, 'sm_generations', 3) != 3:
        raise CompilerResolutionError('Automatic SM export requires all three SM generations')
    if _doublet(fields):
        return _prepare_doublet(lagrangian, inputs, runtime)
    if _triplet(fields):
        return _prepare_triplet(lagrangian, inputs, runtime)
    if any(inputs.get('vevs', {}).get(f.name, 0) != 0 for f in fields):
        raise CompilerResolutionError('Gauge-matter compiler requires zero new-field VEVs')
    coefficients = _coefficients(lagrangian, inputs)
    operators, issues = [], {}
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        if reason:
            issues[row['id']] = reason
        else:
            operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                                              coefficients[row['id']], add_hc=row['field_content'] != row['conjugate_content']))
    if issues:
        raise CompilerResolutionError('Unresolved selected gauge-matter operators:\n' + '\n'.join(f'{key}: {reason}' for key, reason in issues.items()))
    gf = _number(inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5), 'sm_inputs.Gf', real=True)
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    if gf <= 0 or mh <= 0:
        raise ModelResolutionError('Gf and Higgs mass must be positive')
    vev = (math.sqrt(2.) * gf) ** -.5
    normalization, masses = {}, {}
    for f in fields:
        n = f.name
        quadratic = portal = 0.
        kinetic = None
        chiral = {}
        dirac_mass = None
        for op in operators:
            counts = Counter(dict(op.factors))
            if op.kind == 'norm_potential':
                pair = Counter({n: 1, n + '†': 1}) if f.spin == 'scalar' else Counter({n: 2})
                if counts == pair:
                    quadratic += op.coefficients[0]
                if counts == pair + Counter({'H': 1, 'H†': 1}):
                    portal += op.coefficients[0]
            elif op.kind == 'scalar_covariant_kinetic' and counts == {f'D({n})': 1, f'D({n}†)': 1}:
                kinetic = op.coefficients[0]
            elif op.kind == 'proca_kinetic' and counts == {f'D({n})': 2}:
                a, b, c, d = op.coefficients
                if not math.isclose(b + c, -a, rel_tol=1e-10, abs_tol=1e-12):
                    raise ModelResolutionError(f'{n}: Proca requires c2+c3=-c1 in the documented four-term derivative basis')
                kinetic = -2 * a
            elif op.kind == 'dirac_covariant_kinetic':
                for suffix in ('_L', '_Rc'):
                    token = n + suffix
                    if counts == {f'D({token})': 1, token + '†': 1}:
                        chiral[suffix] = 2 * complex(op.coefficients[0]).real
            elif op.kind == 'dirac_mass' and counts == {n + '_L': 1, n + '_Rc': 1}:
                dirac_mass = complex(op.coefficients[0])
        if f.spin == 'fermion':
            if set(chiral) != {'_L', '_Rc'} or any(z <= 0 for z in chiral.values()):
                raise ModelResolutionError(f'{n}: both positive chiral kinetic terms must be selected; canonical row coefficients are 1/2 plus h.c.')
            if dirac_mass is None:
                raise ModelResolutionError(f'{n}: select and supply the Dirac mass operator')
            masses[n] = abs(dirac_mass) / math.sqrt(chiral['_L'] * chiral['_Rc'])
            normalization[n] = chiral
        else:
            if kinetic is None or kinetic <= 0:
                raise ModelResolutionError(f'{n}: select a positive scalar kinetic term or physical Proca kinetic combination')
            if f.spin == 'vector':
                specification = inputs.get('vector_specifications', {}).get(n, {})
                if not isinstance(specification, dict) or specification.get('kind') != 'proca':
                    raise ModelResolutionError(f'{n}: set vector_specifications[{n!r}]={{"kind":"proca"}} to declare a massive effective Proca model')
            squared = (-1 if f.spin == 'scalar' else 2) * (quadratic + portal * vev * vev / 2) / kinetic
            if squared <= 0:
                raise ModelResolutionError(f'{n}: physical squared mass must be positive at the zero-field vacuum')
            normalization[n], masses[n] = float(kinetic), math.sqrt(squared)
        width = _number(inputs.get('widths', {}).get(n, 0.), f'widths.{n}', real=True)
        if width < 0:
            raise ModelResolutionError('Widths cannot be negative')
        inputs.setdefault('widths', {})[n] = width
    compiler = GaugeMatterCompiler(fields, tuple(operators), normalization, masses, vev, float(mh))
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, {'masses': [mh]}, runtime, compiler=compiler)


def _doublet_pairings(counts):
    left = sorted(name[:-1] for name, count in counts.items() if name.endswith('†') for _ in range(count))
    right = sorted(name for name, count in counts.items() if not name.endswith('†') for _ in range(count))
    if len(left) != len(right):
        return ()
    return tuple(sorted({tuple(sorted(zip(left, ordering))) for ordering in permutations(right)}))


def _doublet_classify(row, name):
    counts = _counts(row)
    if row['derivatives']:
        if counts == {f'D({name})': 1, f'D({name}†)': 1} and row['multiplicity'] == 1:
            return 'doublet_kinetic', ''
        if counts == {'D(H)': 1, f'D({name}†)': 1} and row['multiplicity'] == 1:
            return 'doublet_kinetic_mixing', ''
        return None, 'unresolved scalar-doublet derivative structure'
    if set(counts) <= {'H', 'H†', name, name + '†'}:
        pairs = _doublet_pairings(counts)
        if len(pairs) == row['multiplicity']:
            return 'doublet_potential', ''
    for q, s, dagger in (('L', 'ec', True), ('Q', 'dc', True), ('Q', 'uc', False)):
        if counts == {q: 1, s: 1, name + ('†' if dagger else ''): 1} and row['multiplicity'] == 9:
            return 'doublet_yukawa_' + s, ''
    return None, 'unresolved scalar-doublet flavour/invariant structure'


def _padd(a, b):
    answer = dict(a)
    for powers, value in b.items():
        answer[powers] = answer.get(powers, 0) + value
    return answer


def _pmul(a, b):
    answer = {}
    for x, cx in a.items():
        for y, cy in b.items():
            powers = tuple(i + j for i, j in zip(x, y))
            answer[powers] = answer.get(powers, 0) + cx * cy
    return answer


@dataclass
class ScalarDoubletCompiler:
    fields: tuple
    operators: tuple
    basis: dict
    charged_mass: float
    normalization: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        name = self.fields[0].name
        return {'H': float(self.basis['masses'][0]), name + '+': self.charged_mass,
                name + '1': float(self.basis['masses'][1]), name + '2': float(self.basis['masses'][2])}

    def write_sources(self, model, source, fr):
        from feynrules_export import ModelResolutionError, CompilerResolutionError, _wl, _internal
        name = self.fields[0].name
        b = self.basis
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        fields = ('H', 'pmDN1', 'pmDN2')
        parameters = [_internal('pmDK', self.normalization), _internal('pmDBareH', b['higgs_mass_before_mixing'])]
        for i in range(3):
            parameters.extend(_internal(f'pmDR{i}{j}', b['transformation'][i, j]) for j in range(3))
        def component(i):
            return '(' + ' + '.join(f'pmDR{i}{j}*{fields[j]}' for j in range(3) if b['transformation'][i, j] != 0) + ')'
        marker = 'Phi[2] -> (vev + H + I G0)/Sqrt[2]'
        if marker not in sm or 'Value            -> MH^2/(2*vev^2)' not in sm:
            raise CompilerResolutionError('SM doublet template changed; review Higgs definitions')
        sm = sm.replace(marker, f'Phi[2] -> (vev + {component(0)} + I G0)/Sqrt[2]')
        sm = sm.replace('Value            -> MH^2/(2*vev^2)', 'Value            -> pmDBareH^2/(2*vev^2)')
        sm = sm.replace('Mass            -> {MH,125}', f'Mass            -> {{MH,{_wl(b["masses"][0])}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile = source / 'SM_doublet.fr'
        smfile.write_text(sm)
        width = model.inputs.get('widths', {}).get(name, 0.)
        classes, particles = [], []
        for index, (symbol, pname, charge, mass, is_real) in enumerate((
                ('pmDC', name.lower() + '+', 1, self.charged_mass, False),
                ('pmDN1', name.lower() + '1', 0, b['masses'][1], True),
                ('pmDN2', name.lower() + '2', 0, b['masses'][2], True)), 1):
            anti = pname if is_real else name.lower() + '-'
            extra = '' if is_real else f'AntiParticleName -> {_wl(anti)},'
            pdg = 9300000 + index
            classes.append(f'''S[{300+index}] == {{ClassName -> {symbol}, SelfConjugate -> {str(is_real)},
              {extra} QuantumNumbers -> {{Q -> {charge}}}, Mass -> {{pmDM{index},{_wl(mass)}}},
              Width -> {{pmDW{index},{_wl(width)}}}, ParticleName -> {_wl(pname)}, PDG -> {pdg},
              PropagatorLabel -> {_wl(pname)}, PropagatorType -> D,
              PropagatorArrow -> {'None' if is_real else 'Forward'}}}''')
            particles.append(dict(name=pname, antiname=anti, pdg=pdg, source=name, spin=1, mass=float(mass), width=width))
        classes.append('S[304] == {ClassName -> pmD, SelfConjugate -> False, Unphysical -> True, '
                       'Indices -> {Index[SU2D]}, QuantumNumbers -> {Y -> 1/2}, '
                       'Definitions -> {pmD[1] -> pmDC/Sqrt[pmDK], pmD[2] -> ('
                       + component(1) + '+I*' + component(2) + ')/Sqrt[2]}}')
        symbols = {'H': 'Phi', name: 'pmD'}
        terms = []
        has_np = False
        for op in self.operators:
            counts = Counter(dict(op.factors))
            expressions = []
            if op.kind == 'doublet_potential':
                for pairs in _doublet_pairings(counts):
                    expressions.append('*'.join(f'{symbols[left]}bar[i{index}]*{symbols[right]}[i{index}]'
                                                for index, (left, right) in enumerate(pairs, 1)))
            elif op.kind == 'doublet_kinetic':
                expressions = ['DC[pmDbar[ii],mu]*DC[pmD[ii],mu]']
            elif op.kind == 'doublet_kinetic_mixing':
                expressions = ['DC[pmDbar[ii],mu]*DC[Phi[ii],mu]']
            else:
                species = op.kind.removeprefix('doublet_yukawa_')
                for left in range(1, 4):
                    for right in range(1, 4):
                        if species == 'ec':
                            expression = f'LLbar[sp,ii,{left}].lR[sp,{right}]*pmD[ii]'
                        elif species == 'dc':
                            expression = f'QLbar[sp,ii,{left},cc].dR[sp,{right},cc]*pmD[ii]'
                        else:
                            expression = f'QLbar[sp,ii,{left},cc].uR[sp,{right},cc]*pmDbar[jj]*Eps[ii,jj]'
                        expressions.append('HC[' + expression + ']')
            for index, (coefficient, expression) in enumerate(zip(op.coefficients, expressions), 1):
                parameter = f'pmDO{op.operator_id[1:]}n{index}'
                interaction = coefficient != 0 and sum(counts.values()) > 2
                has_np |= interaction
                declaration = _internal(parameter, coefficient, order=interaction)
                higgs = counts['H'] + counts['H†']
                if interaction and higgs:
                    declaration = declaration.replace('InteractionOrder -> {NP, 1}',
                        f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
                parameters.append(declaration)
                if coefficient == 0:
                    terms.append(f'0 (* resolved {op.operator_id}, contraction {index} *)')
                else:
                    term = parameter + '*(' + expression + ')'
                    terms.append('(' + term + '+HC[' + term + '])' if op.add_hc else term)
        extension = source / 'scalar_doublet.fr'
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                            'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}' + (',{NP,1}' if has_np else '') + '};\n'
                            'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
                            'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
                            '(* Doublet potentials use sorted products of dagger-undagger bilinears; Yukawas use row-major Q/L then singlet generation order. *)\n'
                            'LDoublet := Block[{mu,ii,jj,sp,cc,i1,i2}, ExpandIndices['
                            + ' + '.join(terms) + ', FlavorExpand -> {SU2D,SU2W}]] /. {G0|GP|GPbar -> 0};\n')
        return [smfile, extension], 'LSM + LDoublet', particles


def _prepare_doublet(lagrangian, inputs, runtime):
    from feynrules_export import ModelResolutionError, CompilerResolutionError, PreparedModel, ResolvedOperator, _fields, _coefficients, _number
    fields = tuple(_fields(lagrangian))
    name = fields[0].name
    if inputs.get('vevs', {}).get(name, 0.) != 0:
        raise CompilerResolutionError('The doublet compiler requires zero new-doublet VEV; rotate to the Higgs basis first')
    coefficients = _coefficients(lagrangian, inputs)
    operators = []
    for row in lagrangian.operators:
        kind, reason = _doublet_classify(row, name)
        if reason:
            raise CompilerResolutionError(f'{row["id"]}: {reason}')
        operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                                          coefficients[row['id']], add_hc=row['field_content'] != row['conjugate_content']))
    gf = _number(inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5), 'sm_inputs.Gf', real=True)
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    width = _number(inputs.get('widths', {}).get(name, 0.), f'widths.{name}', real=True)
    if gf <= 0 or mh <= 0 or width < 0:
        raise ModelResolutionError('Positive Gf and Higgs mass, and nonnegative width are required')
    vev = (math.sqrt(2.) * gf) ** -.5
    zero = (0, 0, 0, 0)
    def variable(index, coefficient=1.):
        powers = [0] * 4
        powers[index] = 1
        return {tuple(powers): coefficient}
    higgs = [{}, {zero: vev / math.sqrt(2), **variable(0, 1 / math.sqrt(2))}]
    doublet = [variable(3, 1 / math.sqrt(2)), _padd(variable(1, 1 / math.sqrt(2)), variable(2, 1j / math.sqrt(2)))]
    components = {'H': higgs, name: doublet}
    potential, kinetic = {}, None
    for op in operators:
        if op.kind == 'doublet_kinetic':
            kinetic = op.coefficients[0]
        elif op.kind == 'doublet_kinetic_mixing':
            if op.coefficients[0] != 0:
                raise CompilerResolutionError('Nonzero Higgs–new-doublet kinetic mixing requires a full charged/Goldstone basis transformation')
        elif op.kind == 'doublet_potential':
            for pairs, coefficient in zip(_doublet_pairings(Counter(dict(op.factors))), op.coefficients):
                polynomial = {zero: coefficient}
                for left, right in pairs:
                    bilinear = {}
                    for a, c in zip(components[left], components[right]):
                        bilinear = _padd(bilinear, _pmul({powers: complex(value).conjugate() for powers, value in a.items()}, c))
                    polynomial = _pmul(polynomial, bilinear)
                if op.add_hc:
                    polynomial = _padd(polynomial, {powers: complex(value).conjugate() for powers, value in polynomial.items()})
                potential = _padd(potential, polynomial)
    if kinetic is None or kinetic <= 0:
        raise ModelResolutionError('Select a positive doublet kinetic coefficient (canonical value 1)')
    tadpole = np.zeros(4)
    hessian = np.zeros((4, 4))
    hessian[0, 0] = mh * mh
    for powers, coefficient in potential.items():
        if abs(complex(coefficient).imag) > 1e-9 * max(1., abs(coefficient)):
            raise CompilerResolutionError('Scalar potential failed its Hermiticity residual')
        coefficient = float(complex(coefficient).real)
        positions = [index for index, count in enumerate(powers) for _ in range(count)]
        if len(positions) == 1:
            tadpole[positions[0]] -= coefficient
        elif len(positions) == 2:
            i, j = positions
            hessian[i, j] -= coefficient
            hessian[j, i] -= coefficient
    if max(abs(tadpole)) > 1e-10 * max(1., np.max(abs(hessian)) * vev):
        raise ModelResolutionError('New-doublet zero-VEV vacuum is not stationary; relate the mixed quadratic and odd quartic coefficients consistently')
    inverse = np.diag([1., 1 / math.sqrt(kinetic), 1 / math.sqrt(kinetic)])
    eigenvalues, rotations = np.linalg.eigh(inverse @ hessian[:3, :3] @ inverse)
    charged_squared = hessian[3, 3] / kinetic
    if min(eigenvalues) <= 0 or charged_squared <= 0:
        raise ModelResolutionError('Doublet vacuum has a nonpositive physical squared mass')
    transform = inverse @ rotations
    normalized = transform / np.linalg.norm(transform, axis=0)
    order = max(permutations(range(3)), key=lambda p: sum(abs(normalized[i, p[i]]) for i in range(3)))
    transform, eigenvalues = transform[:, order], eigenvalues[list(order)]
    for i in range(3):
        if transform[i, i] < 0:
            transform[:, i] *= -1
    if not np.allclose(transform.T @ np.diag([1., kinetic, kinetic]) @ transform, np.eye(3), atol=1e-10):
        raise CompilerResolutionError('Doublet kinetic normalization residual failed')
    if not np.allclose(transform.T @ hessian[:3, :3] @ transform, np.diag(eigenvalues), atol=1e-7):
        raise CompilerResolutionError('Doublet mass diagonalization residual failed')
    basis = dict(transformation=transform, masses=np.sqrt(eigenvalues), higgs_mass_before_mixing=mh)
    inputs.setdefault('widths', {})[name] = width
    compiler = ScalarDoubletCompiler(fields, tuple(operators), basis, math.sqrt(charged_squared), float(kinetic))
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, basis, runtime, compiler=compiler)


@dataclass
class RealTripletCompiler:
    fields: tuple
    operators: tuple
    normalization: float
    mass: float
    higgs_mass: float
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        n = self.fields[0].name
        return {'H': self.higgs_mass, n + '+': self.mass, n + '0': self.mass}

    def write_sources(self, model, source, fr):
        from feynrules_export import ModelResolutionError, CompilerResolutionError, _internal, _wl
        n = self.fields[0].name
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        marker = 'Representations   -> {Ta,SU2D}'
        generator = 'Definitions       -> {Ta[a_,b_,c_]->PauliSigma[a,b,c]/2,'
        if marker not in sm or generator not in sm:
            raise CompilerResolutionError('SM SU2 declarations changed; review the adjoint representation')
        sm = sm.replace(marker, 'Representations   -> {{Ta,SU2D},{pmTGen,PMTriplet}}')
        sm = sm.replace(generator, 'Definitions       -> {pmTGen[a_,b_,c_] -> -I Eps[a,b,c], Ta[a_,b_,c_]->PauliSigma[a,b,c]/2,')
        # A distinct matter index avoids FR's special gauge-adjoint placeholder
        # AdjointRep[SU2L], which the stock SM does not define for scalar fields.
        sm = 'IndexRange[Index[PMTriplet]] = Unfold[Range[3]];\nIndexStyle[PMTriplet,tt];\n' + sm
        sm = sm.replace('Mass            -> {MH,125}', f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile = source / 'SM_triplet.fr'
        smfile.write_text(sm)
        parameters = [_internal('pmTK', self.normalization)]
        terms = []
        for op in self.operators:
            counts = Counter(dict(op.factors))
            coefficient = op.coefficients[0]
            parameter = 'pmTO' + op.operator_id[1:]
            interaction = coefficient != 0 and sum(counts.values()) > 2 and op.kind != 'triplet_kinetic'
            declaration = _internal(parameter, coefficient, order=interaction)
            higgs = counts['H'] + counts['H†']
            if interaction and higgs:
                declaration = declaration.replace('InteractionOrder -> {NP, 1}',
                    f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
            parameters.append(declaration)
            if coefficient == 0:
                terms.append('0')
                continue
            if op.kind == 'triplet_kinetic':
                expression = 'DC[pmT[ii],mu]*DC[pmT[ii],mu]'
            elif op.kind == 'triplet_odd_portal':
                expression = 'Phibar[ii]*Ta[aa,ii,jj]*Phi[jj]*pmT[aa]'
            elif counts == {n: 2}:
                expression = 'pmT[ii]*pmT[ii]'
            elif counts == {n: 4}:
                expression = 'pmT[ii]*pmT[ii]*pmT[jj]*pmT[jj]'
            else:
                expression = 'Phibar[jj]*Phi[jj]*pmT[ii]*pmT[ii]'
            terms.append(parameter + '*(' + expression + ')')
        width = model.inputs.get('widths', {}).get(n, 0.)
        classes, particles = [], []
        for index, (symbol, pname, conjugate, charge) in enumerate((('pmTC', n.lower() + '+', False, 1), ('pmTN', n.lower() + '0', True, 0)), 1):
            anti = pname if conjugate else n.lower() + '-'
            extra = '' if conjugate else f'AntiParticleName -> {_wl(anti)},'
            classes.append(f'''S[{400+index}] == {{ClassName -> {symbol}, SelfConjugate -> {str(conjugate)},
              {extra} QuantumNumbers -> {{Q -> {charge}}}, Mass -> {{pmTM{index},{_wl(self.mass)}}},
              Width -> {{pmTW{index},{_wl(width)}}}, ParticleName -> {_wl(pname)}, PDG -> {9400000+index},
              PropagatorLabel -> {_wl(pname)}, PropagatorType -> D,
              PropagatorArrow -> {'None' if conjugate else 'Forward'}}}''')
            particles.append(dict(name=pname, antiname=anti, pdg=9400000+index, source=n, spin=1, mass=self.mass, width=width))
        classes.append('S[403] == {ClassName -> pmT, SelfConjugate -> True, Unphysical -> True, '
                       'Indices -> {Index[PMTriplet]}, FlavorIndex -> PMTriplet, '
                       'Definitions -> {pmT[1] -> (pmTC+pmTCbar)/Sqrt[2*pmTK], '
                       'pmT[2] -> I*(pmTC-pmTCbar)/Sqrt[2*pmTK], pmT[3] -> pmTN/Sqrt[pmTK]}}')
        has_np = any(op.coefficients[0] != 0 and op.kind == 'triplet_potential' and sum(count for _, count in op.factors) > 2 for op in self.operators)
        extension = source / 'real_triplet.fr'
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
                            'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}' + (',{NP,1}' if has_np else '') + '};\n'
                            'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
                            'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
                            'LTriplet := Block[{mu,ii,jj,aa}, ExpandIndices[' + ' + '.join(terms)
                            + ', FlavorExpand -> {SU2D,SU2W,PMTriplet}]] /. {G0|GP|GPbar -> 0};\n')
        return [smfile, extension], 'LSM + LTriplet', particles


def _prepare_triplet(lagrangian, inputs, runtime):
    from feynrules_export import ModelResolutionError, CompilerResolutionError, PreparedModel, ResolvedOperator, _fields, _coefficients, _number
    fields = tuple(_fields(lagrangian))
    n = fields[0].name
    if inputs.get('vevs', {}).get(n, 0.) != 0:
        raise CompilerResolutionError('Real triplet requires zero VEV; nonzero VEV changes the electroweak gauge-boson mass basis')
    coefficients = _coefficients(lagrangian, inputs)
    operators, kinetic, quadratic, portal = [], None, 0., 0.
    for row in lagrangian.operators:
        kind, reason = classify(row, fields)
        if reason:
            raise CompilerResolutionError(f'{row["id"]}: {reason}')
        counts, value = _counts(row), coefficients[row['id']][0]
        if kind == 'triplet_kinetic':
            kinetic = 2 * value
        elif kind == 'triplet_odd_portal' and value != 0:
            raise ModelResolutionError('Nonzero triplet odd portal induces a triplet tadpole; the declared zero-VEV vacuum is not stationary')
        elif counts == {n: 2}:
            quadratic = value
        elif counts == {'H': 1, 'H†': 1, n: 2}:
            portal = value
        operators.append(ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']), coefficients[row['id']]))
    gf = _number(inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5), 'sm_inputs.Gf', real=True)
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    width = _number(inputs.get('widths', {}).get(n, 0.), f'widths.{n}', real=True)
    if gf <= 0 or mh <= 0 or width < 0 or kinetic is None or kinetic <= 0:
        raise ModelResolutionError('Positive kinetic normalization, Gf and Higgs mass, and nonnegative width are required')
    v = (math.sqrt(2) * gf) ** -.5
    squared = -2 * (quadratic + portal * v * v / 2) / kinetic
    if squared <= 0:
        raise ModelResolutionError('Triplet vacuum has a nonpositive physical squared mass')
    inputs.setdefault('widths', {})[n] = width
    compiler = RealTripletCompiler(fields, tuple(operators), float(kinetic), math.sqrt(squared), float(mh))
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, {'masses': [mh]}, runtime, compiler=compiler)
