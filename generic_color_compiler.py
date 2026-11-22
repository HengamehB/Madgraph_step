"""Native SU(3) tensor-network lowering for scalar and vectorlike matter.

Invariant bases are constructed from delta, T/T6, f/d, epsilon and K6 tensors,
symmetrized as commuting scalar polynomials, and checked against the exact
catalogue multiplicity. No component-colour vertices are emitted: UFOs retain
the native colour tensors understood by MadGraph. Coefficients multiply the
displayed ordered native basis. Kinetic and mass mixing is resolved exactly
within equal gauge representations at zero new-field VEVs, and the same
transformation is applied to every scalar interaction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, permutations, product
import hashlib
import json
import math

import numpy as np
import generic_colour_yukawa as yukawa
from workflow_errors import ResourceLimitError

BASIS_VERSION = 'native-colour-networks-hermitian-v1'
_REPS = {(0, 0): 1, (1, 0): 3, (0, 1): -3, (2, 0): 6, (0, 2): -6, (1, 1): 8}


def supports_fields(fields):
    return bool(fields) and all(f.su2 == 1 and f.su3 in _REPS and
        ((f.spin == 'scalar' and (f.kind == 'complex' or _REPS[f.su3] in (1, 8))) or
         (f.spin == 'fermion' and f.kind == 'dirac')) for f in fields)


def uses_generic(fields, operators=None):
    if not supports_fields(fields):
        return False
    if any(abs(_REPS[f.su3]) in (6, 8) for f in fields):
        return True
    # Preserve existing norm-model coefficient conventions and particle IDs.
    # The native engine also handles triplets, and becomes the complete-model
    # fallback when a selected scalar contraction is outside that legacy path.
    if (operators is not None and all(f.spin == 'scalar' for f in fields)
            and any(_REPS[f.su3] != 1 for f in fields)):
        from gauge_matter_compiler import classify as legacy_classify
        return any(legacy_classify(row, fields)[1] for row in operators)
    return False


def _anti(rep):
    return rep if rep in (1, 8) else -rep


def _index(rep):
    return {3: 'Colour', 6: 'Sextet', 8: 'Gluon'}[abs(rep)]


@lru_cache(maxsize=1)
def colour_tensors():
    """Gell-Mann normalization Tr(Ta Tb)=delta_ab/2 and orthonormal K6."""
    t = np.zeros((8, 3, 3), complex)
    for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        even, odd = ((0, 1), (3, 4), (5, 6))[a]
        t[even, i, j] = t[even, j, i] = .5
        t[odd, i, j], t[odd, j, i] = -.5j, .5j
    t[2] = np.diag([.5, -.5, 0.])
    t[7] = np.diag([1., 1., -2.]) / (2 * math.sqrt(3))
    f, d = np.zeros((8, 8, 8)), np.zeros((8, 8, 8))
    for a, b, c in product(range(8), repeat=3):
        f[a, b, c] = (-2j * np.trace((t[a] @ t[b] - t[b] @ t[a]) @ t[c])).real
        d[a, b, c] = (2 * np.trace((t[a] @ t[b] + t[b] @ t[a]) @ t[c])).real
    k = np.zeros((6, 3, 3))
    for a, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        k[a, i, j] = k[a, j, i] = 1. if i == j else 1 / math.sqrt(2)
    t6 = (np.einsum('Aij,aik,Bkj->aAB', k, t, k) +
          np.einsum('Aij,ajk,Bik->aAB', k, t, k))
    epsilon = np.zeros((3, 3, 3))
    for order in permutations(range(3)):
        epsilon[order] = (-1) ** sum(order[i] > order[j] for i in range(3) for j in range(i+1, 3))
    determinant = np.einsum('Aij,Bkl,Cmn,ikm,jln->ABC', k, k, k, epsilon, epsilon, optimize=True)
    return dict(T=t, T6=t6, f=f, dSUN=d, K6=k, K6bar=k,
                Eps=epsilon, EpsBar=epsilon, D6=determinant, D6bar=determinant)


@lru_cache(maxsize=1)
def _primitives():
    tensors = colour_tensors()
    specifications = [('T', (8, -3, 3)), ('T6', (8, -6, 6)),
                      ('f', (8, 8, 8)), ('dSUN', (8, 8, 8)),
                      ('K6', (6, -3, -3)), ('K6bar', (-6, 3, 3)),
                      ('Eps', (3, 3, 3)), ('EpsBar', (-3, -3, -3)),
                      ('D6', (6, 6, 6)), ('D6bar', (-6, -6, -6))]
    return tuple((name, reps, tensors[name]) for name, reps in specifications)


def _cubic_matches(reps, positions):
    for name, signature, _ in _primitives():
        for order in permutations(range(3)):
            if tuple(reps[i] for i in order) == signature:
                yield (name, tuple(positions[i] for i in order))


def _networks(reps):
    size = len(reps)
    result = set()
    if size == 0:
        result.add(())
    elif size == 2 and _anti(reps[0]) == reps[1]:
        result.add(((f'Delta{abs(reps[0])}', (0, 1)),))
    elif size == 3:
        result.update((item,) for item in _cubic_matches(reps, tuple(range(3))))
    elif size == 4:
        for left in combinations(range(4), 2):
            right = tuple(i for i in range(4) if i not in left)
            if left > right:
                continue
            if _anti(reps[left[0]]) == reps[left[1]] and _anti(reps[right[0]]) == reps[right[1]]:
                result.add(tuple(sorted(((f'Delta{abs(reps[left[0]])}', left),
                                         (f'Delta{abs(reps[right[0]])}', right)))))
            for internal in (3, -3, 6, -6, 8):
                first = tuple(_cubic_matches(tuple(reps[i] for i in left) + (internal,), left + (4,)))
                second = tuple(_cubic_matches(tuple(reps[i] for i in right) + (_anti(internal),), right + (4,)))
                result.update(tuple(sorted((a, b))) for a in first for b in second)
    # Deltas first retain the familiar quadratic/norm normalization.
    return tuple(sorted(result, key=lambda network: (sum(0 if n.startswith('Delta') else 5 if n.startswith('D6') else 1
                                                       for n, _ in network), repr(network))))


def _limits(limits):
    """Only actual native construction costs are bounded; never truncate a basis."""
    from compiler_limits import DEFAULTS, normalize_limits
    values = normalize_limits(vars(limits) if hasattr(limits, '__dataclass_fields__') else limits)
    return tuple(values.get(name, DEFAULTS[name]) for name in
                 ('max_colour_candidate_networks', 'max_power_monomials', 'max_output_terms'))


def _tensor(network, size):
    if not network:
        return np.array(1.)
    operands, labels = [], []
    for name, slots in network:
        operands.extend((np.eye(int(name[5:])) if name.startswith('Delta') else colour_tensors()[name], list(slots)))
        labels.extend(slots)
    operands.append(list(range(size)))
    return np.einsum(*operands, optimize=True)


def _conjugate(poly, real_names):
    def bar(name):
        return name if name in real_names else name[:-1] if name.endswith('†') else name + '†'
    return {tuple(sorted((bar(name), index) for name, index in monomial)): complex(value).conjugate()
            for monomial, value in poly.items()}


def _sum(a, b, factor=1, *, tolerance=0.):
    answer = dict(a)
    for key, value in b.items():
        answer[key] = answer.get(key, 0j) + factor * value
    return {key: value for key, value in answer.items() if abs(value) > tolerance}


@dataclass(frozen=True)
class NativeInvariant:
    network: tuple
    legs: tuple
    singlets: tuple
    higgs_pairs: int
    mode: str = 'plain'

    def render(self, symbols=None, normalization=None):
        symbols, normalization = symbols or {}, normalization or {}
        pieces = []
        counter = 10
        for name, slots in self.network:
            indices = [f'c{i+1}' for i in slots]
            if name.startswith('Delta'):
                kind = _index(int(name[5:]))
                pieces.append('IndexDelta[' + ','.join(f'Index[{kind},{i}]' for i in indices) + ']')
            elif name.startswith('D6'):
                k = 'K6bar' if name.endswith('bar') else 'K6'
                dummy = [f'c{i}' for i in range(counter, counter + 6)]
                counter += 6
                pieces.extend(f'{k}[{external},{dummy[2*i]},{dummy[2*i+1]}]' for i, external in enumerate(indices))
                pieces.extend(('Eps[' + ','.join(dummy[::2]) + ']', 'Eps[' + ','.join(dummy[1::2]) + ']'))
            else:
                # FR assigns epsilon/anti-epsilon by attached field orientation.
                pieces.append(name.replace('EpsBar', 'Eps') + '[' + ','.join(indices) + ']')
        for index, token in enumerate(self.legs):
            expression = symbols.get(token, token)
            expression = expression.replace('{index}', f'c{index+1}') if '{index}' in expression else expression + f'[c{index+1}]'
            z = normalization.get(token.removesuffix('†'), 1.)
            if z != 1:
                from feynrules_export import _wl
                expression = '(' + expression + '/Sqrt[' + _wl(z) + '])'
            pieces.append(expression)
        for token in self.singlets:
            expression = symbols.get(token, token)
            z = normalization.get(token.removesuffix('†'), 1.)
            if z != 1:
                from feynrules_export import _wl
                expression = '(' + expression + '/Sqrt[' + _wl(z) + '])'
            pieces.append(expression)
        pieces.extend(['((vev+H)^2/2)'] * self.higgs_pairs)
        base = '*'.join(pieces) or '1'
        if self.mode == 'real':
            return f'(({base})+HC[{base}])/2'
        if self.mode == 'imag':
            return f'(({base})-HC[{base}])/(2*I)'
        return base


def _potential_tokens(row, fields):
    names = {f.name: f for f in fields if f.spin == 'scalar'}
    counts = Counter({part['name']: part['count'] for part in row['factors']})
    if row['derivatives'] or counts['H'] != counts['H†']:
        raise ValueError('non-scalar/derivative or non-norm Higgs contraction')
    higgs = counts.pop('H', 0)
    counts.pop('H†', None)
    if any(token.removesuffix('†') not in names for token in counts):
        raise ValueError('fermion/Lorentz contraction needs an additional lowerer')
    legs, reps, singlets = [], [], []
    for token, count in sorted(counts.items()):
        rep = _REPS[names[token.removesuffix('†')].su3]
        if token.endswith('†'):
            rep = _anti(rep)
        if rep == 1:
            singlets.extend([token] * count)
        else:
            legs.extend([token] * count); reps.extend([rep] * count)
    return tuple(legs), tuple(reps), tuple(singlets), higgs


@lru_cache(maxsize=256)
def _basis(fields, factors, derivatives, multiplicity, hermitian, limits):
    row = dict(factors=[dict(name=n, count=c) for n, c in factors], derivatives=derivatives)
    legs, reps, singlets, higgs = _potential_tokens(row, fields)
    real_names = {f.name for f in fields if f.kind == 'real'}
    candidates = []
    max_networks, max_tensor, max_terms = limits
    networks = _networks(reps)
    if len(networks) > max_networks:
        raise ResourceLimitError(f'Native colour construction has {len(networks)} candidate networks; '
                                 f'max_colour_candidate_networks is {max_networks}. Increase compiler_limits explicitly.')
    entries = math.prod(abs(rep) for rep in reps)
    if entries > max_tensor:
        raise ResourceLimitError(f'Native colour tensor has {entries} entries; max_power_monomials is {max_tensor}. '
                                 'Increase compiler_limits explicitly.')
    output_terms, seen = 0, set()
    def add_candidate(invariant, polynomial):
        nonlocal output_terms
        # Permuting commuting equal-field legs produces the same polynomial
        # many times. Bound actual stored data, not these duplicate networks.
        key = tuple((m, round(z.real, 12), round(z.imag, 12)) for m, z in sorted(polynomial.items()))
        if key in seen:
            return
        seen.add(key)
        output_terms += len(polynomial)
        if output_terms > max_terms:
            raise ResourceLimitError(f'Native colour candidate expansion exceeds max_output_terms={max_terms}. '
                                     'Increase compiler_limits explicitly.')
        candidates.append((invariant, polynomial))
    for network in networks:
        tensor = _tensor(network, len(reps))
        polynomial = {}
        for indices in np.ndindex(tensor.shape):
            coefficient = complex(tensor[indices])
            if abs(coefficient) < 1e-12:
                continue
            monomial = tuple(sorted(tuple(zip(legs, indices)) + tuple((name, 0) for name in singlets)))
            polynomial[monomial] = polynomial.get(monomial, 0j) + coefficient
        invariant = NativeInvariant(network, legs, singlets, higgs)
        if hermitian:
            conjugated = _conjugate(polynomial, real_names)
            for mode, factor in (('real', 1), ('imag', -1)):
                # These are dimensionless group tensors, before applying any
                # physical input. Roundoff cleanup must stay at this layer.
                projected = _sum(polynomial, conjugated, factor, tolerance=1e-11)
                projected = {key: value / (2 if mode == 'real' else 2j) for key, value in projected.items()}
                if projected:
                    add_candidate(NativeInvariant(network, legs, singlets, higgs, mode), projected)
        elif polynomial:
            add_candidate(invariant, polynomial)
    monomials = sorted(set().union(*(p.keys() for _, p in candidates))) if candidates else []
    basis, vectors, polynomials = [], [], []
    for invariant, poly in candidates:
        vector = np.array([poly.get(m, 0j) for m in monomials])
        if hermitian:
            vector = np.concatenate((vector.real, vector.imag))
        residual = vector.copy()
        for previous in vectors:
            residual -= np.vdot(previous, residual) * previous
        norm = np.linalg.norm(residual)
        if norm > 1e-9 * max(1., np.linalg.norm(vector)):
            basis.append(invariant); polynomials.append(poly); vectors.append(residual / norm)
    if len(basis) != multiplicity:
        from feynrules_export import CompilerResolutionError
        raise CompilerResolutionError(f'native colour tensor span has {len(basis)} invariants, catalogue requires {multiplicity}; no incomplete projection is exported')
    return tuple(basis), tuple(polynomials)


def invariant_basis(row, fields, limits=None):
    return _basis(tuple(fields), tuple((p['name'], p['count']) for p in row['factors']),
                  row['derivatives'], row['multiplicity'], row['field_content'] == row['conjugate_content'], _limits(limits))


def basis_labels(row, fields, limits=None):
    return tuple(invariant.render() for invariant in invariant_basis(row, fields, limits)[0])


def basis_fingerprint(row, fields, limits=None):
    return hashlib.sha256(json.dumps(dict(version=BASIS_VERSION, expressions=basis_labels(row, fields, limits)),
                                     sort_keys=True).encode()).hexdigest()


def kinetic_basis(row, fields, limits=None):
    stripped = dict(row, derivatives=0, factors=[dict(name=p['name'][2:-1], count=p['count']) for p in row['factors']])
    return invariant_basis(stripped, fields, limits)


def classify(row, fields, limits=None):
    _limits(limits)  # Invalid user budgets are input errors, never compiler gaps.
    kind = yukawa.classify(row, fields)
    if kind:
        return kind, ''
    counts = Counter({part['name']: part['count'] for part in row['factors']})
    for f in fields:
        n = f.name
        if f.spin == 'scalar' and row['multiplicity'] == 1:
            kinetic = {f'D({n})': 2} if f.kind == 'real' else {f'D({n})': 1, f'D({n}†)': 1}
            if counts == kinetic:
                return 'colour_scalar_kinetic', ''
        if f.spin == 'fermion' and row['multiplicity'] == 1:
            if counts == {n + '_L': 1, n + '_Rc': 1}:
                return 'colour_dirac_mass', ''
            for chirality in ('_L', '_Rc'):
                token = n + chirality
                if counts == {f'D({token})': 1, token + '†': 1}:
                    return 'colour_dirac_kinetic', ''
    scalar_names = {f.name for f in fields if f.spin == 'scalar'}
    if (sum(counts.values()) == 2 and row['multiplicity'] == 1 and
        all(token.startswith('D(') and token.endswith(')') and token[2:-1].removesuffix('†') in scalar_names
            for token in counts)):
        try:
            kinetic_basis(row, fields, limits)
            return 'colour_scalar_kinetic', ''
        except ResourceLimitError:
            raise
        except ValueError as exc:
            return None, 'compiler gap: ' + str(exc)
    try:
        invariant_basis(row, fields, limits)
        return 'colour_scalar_potential', ''
    except ResourceLimitError:
        raise
    except ValueError as exc:
        return None, 'compiler gap: ' + str(exc)


@dataclass
class GenericColorCompiler:
    fields: tuple
    operators: tuple
    normalization: dict
    physical_masses: dict
    higgs_mass: float
    bases: dict
    scalar_states: tuple = ()
    scalar_maps: dict = field(default_factory=dict)
    scalar_transformation: object = None
    stock_sm_neutrinos: bool = True

    @property
    def masses(self):
        return {'H': self.higgs_mass, **self.physical_masses}

    def write_sources(self, model, source, fr):
        from feynrules_export import ModelResolutionError, _wl, _internal
        sm = (fr / 'Models' / 'SM' / 'SM.fr').read_text()
        sm = sm.replace('Mass            -> {MH,125}', f'Mass            -> {{MH,{_wl(self.higgs_mass)}}}')
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, model.inputs)
        smfile = source / 'SM_colour.fr'; smfile.write_text(sm)
        parameters, classes, particles, terms, symbols = [], [], [], [], {}
        descriptions = [dict(source=s.source, label=s.label, rep=s.rep, real=s.real, name=s.name,
                             charge=s.charge, spin='scalar', mass=s.mass) for s in self.scalar_states]
        lepton_numbers, lepton_note = yukawa.lepton_labels(
            self.operators, {(token, 0): mapping for token, mapping in self.scalar_maps.items()})
        for f in self.fields:
            if f.spin == 'fermion':
                rep = _REPS[f.su3]
                descriptions.append(dict(source=f.name, label=f.name, rep=rep, real=False,
                    name=f.name.lower() + ('bar' if rep < 0 else ''), charge=float(f.hypercharge) * (-1 if rep < 0 else 1),
                    spin=f.spin, mass=self.physical_masses[f.name]))
        for index, description in enumerate(descriptions, 1):
            rep = description['rep']
            colour = rep != 1
            real = description['real']
            symbol = f'pmGC{index}'
            name = description['name']
            anti = name if real else name.removesuffix('bar') if name.endswith('bar') else name + 'bar'
            charge = description['charge']
            lepton = f', LeptonNumber -> {lepton_numbers[index-1]}' if index-1 in lepton_numbers else ''
            letter = 'S' if description['spin'] == 'scalar' else 'F'
            extra = f'Indices -> {{Index[{_index(rep)}]}}, ' if colour else ''
            if not real:
                extra += f'AntiParticleName -> {_wl(anti)}, '
            mass, width, pdg = description['mass'], model.inputs['widths'][description['source']], 9500000 + index
            classes.append(f'''{letter}[{500+index}] == {{ClassName -> {symbol}, SelfConjugate -> {str(real)},
                {extra}Mass -> {{pmGCM{index},{_wl(mass)}}}, Width -> {{pmGCW{index},{_wl(width)}}},
                QuantumNumbers -> {{Y -> {_wl(charge)}, Q -> {_wl(charge)}{lepton}}},
                ParticleName -> {_wl(name)}, PDG -> {pdg}, PropagatorLabel -> {_wl(description['label'])},
                PropagatorType -> {'D' if letter == 'S' else 'Straight'}, PropagatorArrow -> {'None' if real else 'Forward'}}}''')
            particles.append(dict(name=name, antiname=anti, source=description['source'], spin=1 if letter == 'S' else 2,
                                  mass=mass, width=width, pdg=pdg, charge=float(charge), color=abs(rep)))
            if description['spin'] == 'scalar':
                argument = '[cc]' if colour else ''
                left = symbol if real else symbol + 'bar'
                terms.append(('1/2*' if real else '') + f'DC[{left}{argument},mu]*DC[{symbol}{argument},mu]')
            else:
                terms.extend((f'I*{symbol}bar.Ga[mu].DC[{symbol},mu]', f'-pmGCM{index}*{symbol}bar.{symbol}'))
        real_constants = {}
        def real_literal(value):
            value = float(value)
            if value == int(value):
                return str(int(value))
            # Keep I symbolic: FeynRules' field collector rejects machine
            # Complex atoms multiplying tensor-valued field expressions.
            if value not in real_constants:
                name = f'pmGCR{len(real_constants)+1}'
                parameters.append(_internal(name, value))
                real_constants[value] = name
            return real_constants[value]
        def literal(value):
            value = complex(value)
            if value.imag == 0:
                return real_literal(value.real)
            return '(' + real_literal(value.real) + '+I*(' + real_literal(value.imag) + '))'
        for token, mapping in self.scalar_maps.items():
            pieces = []
            for (index, bar), coefficient in mapping.items():
                state = self.scalar_states[index]
                symbol = f'pmGC{index+1}' + ('bar' if bar and not state.real else '')
                if state.rep != 1:
                    symbol += '[{index}]'
                pieces.append(f'({literal(coefficient)})*{symbol}')
            symbols[token] = '(' + '+'.join(pieces) + ')'
        has_np = False
        for op in self.operators:
            if op.kind in yukawa.KINDS:
                declarations, interactions = yukawa.render(op, {(t, 0): s for t, s in symbols.items()})
                parameters.extend(declarations)
                terms.extend(interactions)
                has_np |= bool(interactions)
                continue
            if op.kind != 'colour_scalar_potential':
                continue
            for index, (coefficient, invariant) in enumerate(zip(op.coefficients, self.bases[op.operator_id]), 1):
                if coefficient == 0:
                    continue
                parameter = f'pmGCO{op.operator_id[1:]}n{index}'
                interaction = sum(count for _, count in op.factors) > 2
                has_np |= interaction
                declaration = _internal(parameter, coefficient, order=interaction)
                higgs = sum(count for token, count in op.factors if token in ('H', 'H†'))
                if interaction and higgs:
                    declaration = declaration.replace('InteractionOrder -> {NP, 1}',
                        f'InteractionOrder -> {{{{NP, 1}}, {{QED, {higgs}}}}}')
                parameters.append(declaration)
                expression = invariant.render(symbols)
                term = parameter + '*(' + expression + ')'
                terms.append('(' + term + '+HC[' + term + '])' if op.add_hc else term)
        extension = source / 'generic_colour.fr'
        extension.write_text('M$ModelName = ' + _wl(model.model_name) + ';\n'
            + lepton_note +
            'IndexRange[Index[Sextet]] = NoUnfold[Range[6]];\nIndexStyle[Sextet,sx];\n'
            'AddGaugeRepresentation[{SU3C -> {T6,Sextet}}];\n'
            'M$InteractionOrderHierarchy = {{QCD,1},{QED,2}' + (',{NP,1}' if has_np else '') + '};\n'
            'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
            'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
            '(* ' + BASIS_VERSION + '; normalized native colour contractions. *)\n'
            'LColour := Block[{mu,cc,' + ','.join(f'c{i}' for i in range(1, 24)) + '}, ExpandIndices['
            + ' + '.join(terms) + ', FlavorExpand -> {SU2D,SU2W}]];\n')
        return [smfile, extension], 'LSM + LColour', particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import CompilerResolutionError, ModelResolutionError, PreparedModel, ResolvedOperator, _fields, _coefficients, _number
    fields = tuple(_fields(lagrangian))
    if not uses_generic(fields, lagrangian.operators):
        return None
    if any(_number(inputs.get('vevs', {}).get(f.name, 0.), 'vevs.' + f.name, real=True) != 0 for f in fields):
        raise CompilerResolutionError('Generic colour compiler requires a colour-preserving zero new-field VEV vacuum')
    if getattr(lagrangian._source, 'sm_generations', 3) != 3:
        raise CompilerResolutionError('Automatic SM export requires three SM generations')
    coefficients = _coefficients(lagrangian, inputs)
    operators, bases, quadratic, kinetic_polynomial, higgs_mixing = [], {}, {}, {}, {}
    gf = _number(inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5), 'Gf', real=True)
    mh = _number(inputs.get('higgs_mass', 125.), 'higgs_mass', real=True)
    if gf <= 0 or mh <= 0:
        raise ModelResolutionError('Gf and higgs_mass must be positive')
    vev2 = 1 / (math.sqrt(2.) * gf)
    for row in lagrangian.operators:
        kind, reason = classify(row, fields, inputs.get('compiler_limits'))
        values = coefficients[row['id']]
        if reason:
            if all(value == 0 for value in values):
                kind = 'zero_verified'
            else:
                raise CompilerResolutionError(f'{row["id"]} ({row["field_content"]}): {reason}')
        elif kind in yukawa.KINDS and not any(values):
            kind = 'zero_verified'
        op = ResolvedOperator(row['id'], kind, tuple((p['name'], p['count']) for p in row['factors']),
                              values, add_hc=row['field_content'] != row['conjugate_content'])
        operators.append(op)
        if kind == 'colour_scalar_kinetic':
            _, polynomials = kinetic_basis(row, fields, inputs.get('compiler_limits'))
            contribution = {m: values[0] * c for m, c in polynomials[0].items()}
            if op.add_hc:
                contribution = _sum(contribution, _conjugate(contribution, {f.name for f in fields if f.kind == 'real'}))
            kinetic_polynomial = _sum(kinetic_polynomial, contribution)
        if kind == 'colour_scalar_potential':
            basis, polynomials = invariant_basis(row, fields, inputs.get('compiler_limits'))
            bases[op.operator_id] = basis
            for coefficient, invariant, polynomial in zip(values, basis, polynomials):
                if coefficient == 0:
                    continue
                contribution = {m: coefficient * c * (vev2 / 2) ** invariant.higgs_pairs
                                for m, c in polynomial.items() if len(m) <= 2}
                mixing = {m: coefficient * c * (vev2 / 2) ** invariant.higgs_pairs *
                          (2 * invariant.higgs_pairs / math.sqrt(vev2))
                          for m, c in polynomial.items() if len(m) == 1 and invariant.higgs_pairs}
                if op.add_hc:
                    contribution = _sum(contribution, _conjugate(contribution, {f.name for f in fields if f.kind == 'real'}))
                    mixing = _sum(mixing, _conjugate(mixing, {f.name for f in fields if f.kind == 'real'}))
                quadratic = _sum(quadratic, contribution)
                higgs_mixing = _sum(higgs_mixing, mixing)
    normalization, masses = {}, {}
    for f in fields:
        n = f.name
        if f.spin == 'fermion':
            chiral, mass = {}, None
            for op in operators:
                counts = Counter(dict(op.factors))
                if op.kind == 'colour_dirac_mass' and counts == {n + '_L': 1, n + '_Rc': 1}:
                    mass = complex(op.coefficients[0])
                if op.kind == 'colour_dirac_kinetic':
                    for suffix in ('_L', '_Rc'):
                        token = n + suffix
                        if counts == {f'D({token})': 1, token + '†': 1}:
                            chiral[suffix] = 2 * complex(op.coefficients[0]).real
            if mass is None or set(chiral) != {'_L', '_Rc'} or min(chiral.values()) <= 0:
                raise ModelResolutionError(f'{n}: supply both positive chiral kinetic coefficients and a Dirac mass')
            masses[n] = abs(mass) / math.sqrt(chiral['_L'] * chiral['_Rc'])
            normalization[n] = 1.
            continue
    from native_colour_basis import resolve_scalars
    scalar_states, scalar_maps, transformation = resolve_scalars(fields, _REPS, kinetic_polynomial, quadratic)
    if higgs_mixing:
        raise CompilerResolutionError('compiler gap: Higgs-singlet mixing in a coloured model requires simultaneous SM Higgs interaction transformation')
    masses.update({state.label: state.mass for state in scalar_states})
    inputs.setdefault('widths', {})
    for f in fields:
        width = _number(inputs['widths'].get(f.name, 0.), 'widths.' + f.name, real=True)
        if width < 0:
            raise ModelResolutionError('Widths must be nonnegative')
        inputs['widths'][f.name] = width
    compiler = GenericColorCompiler(fields, tuple(operators), normalization, masses, float(mh), bases,
                                    scalar_states, scalar_maps, transformation)
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, None, runtime, compiler=compiler)
