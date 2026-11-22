"""Rank explicit decay histories using on-shell production times branching ratios.

The result is a narrow-width, parton-level ranking, not a sum of exclusive
signals. The decay tables must contain primary MadWidth modes: already counted
on-shell cascades must not also be inserted as independent multibody decays.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import html
import itertools
import math
import warnings
from types import SimpleNamespace


class CascadeError(ValueError):
    """A cascade cannot be assigned a defensible probability."""


class CascadeResourceLimit(CascadeError):
    category = 'resource_limit'


# These are terminal *parton-level* objects, not a claim of physical stability.
_SM_TERMINAL = frozenset((1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 21, 22))
_SM_HEAVY = frozenset((6, 23, 24, 25))


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _finite(value, label, *, nonnegative=True):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise CascadeError(f"{label} must be a finite number") from None
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise CascadeError(f"{label} must be finite and nonnegative")
    return result


@dataclass
class FinalStateRanking:
    """All calculated histories; display filtering never changes probabilities."""

    rows: object
    coverage: dict[str, float]
    omitted_probability: dict[str, float]
    notices: tuple[str, ...]
    uncomputed_production_ids: tuple[str, ...] = ()
    calculation_limits: dict = field(default_factory=dict)

    @property
    def per_production_coverage(self):
        return self.coverage

    def __repr__(self):
        return f"FinalStateRanking({len(self.rows)} decay histories; rates in pb, narrow-width approximation)"

    def refresh(self):
        if hasattr(self.rows, 'summary'):
            for key,value in self.rows.summary().items():
                setattr(self,key,value)
        return self

    def visible_rows(self, min_relative_rate=1e-4, show_all=False, limit=100, cursor=None):
        """Return one bounded page; complete iteration is available on rows."""
        cutoff = _finite(min_relative_rate, 'min_relative_rate')
        if cutoff > 1:
            raise CascadeError('min_relative_rate must lie between 0 and 1')
        if hasattr(self.rows, 'page'):
            return self.rows.page(limit=limit,cursor=cursor,min_relative_rate=0 if show_all else cutoff)['rows']
        maximum = max((row['rate_pb'] for row in self.rows if row['rate_pb'] is not None),default=0)
        return tuple(itertools.islice((r for r in self.rows if show_all or r['rate_pb'] is None or r['rate_pb'] >= cutoff*maximum),limit))

    def show(self, min_relative_rate=1e-4, show_all=False):
        from madgraph_runner import _table
        selected = self.visible_rows(min_relative_rate, show_all)
        headings = ('Production', 'Decay history', 'Final state', 'BR product', 'σ × BR [pb]', 'MC error [pb]', 'Flags')
        def number(value):
            return 'not calculated' if value is None else f'{value:.6g}'
        cells = ((row['production_process'], row['history'], row['final_state'],
                  f"{row['branching_ratio']:.5g}", number(row['rate_pb']),
                  number(row['error_pb']), '; '.join(row['flags'])) for row in selected)
        _table(headings, cells)
        print('Tree-level narrow-width rates; loop-induced channels and decay acceptance excluded. Histories can overlap; do not sum rows.')
        if len(selected) != len(self.rows):
            print(f'{len(selected)}/{len(self.rows)} histories shown; rows.page(cursor=...) advances pages; rows iterates every saved history.')
        missing = {key: value for key, value in self.omitted_probability.items() if value is not None and value > 1e-8}
        if missing:
            print('Uncovered decay probability: ' + ', '.join(f'{key}: {value:.3g}' for key, value in missing.items()))
        if self.uncomputed_production_ids:
            print(f'{len(self.uncomputed_production_ids)} production channels have no cross-section calculation.')
        if any(row['flags'] for row in selected):
            print('Flags identify possible failures of the narrow-width approximation; see .notices.')
        return self


def _prepare_context(channels, decays, cross_sections, stable_particles=(), *,
                      min_probability=0.0, max_depth=12, max_histories=100000,
                      min_relative_rate=1e-4, max_width_mass_ratio=0.01,
                      threshold_widths=5.0, show=True, validate_only=False):
    """Expand new particles and t/W/Z/h; rank distinct histories by σ × BR.

    ``stable_particles`` accepts model names or signed PDG IDs. A nonzero
    calculated width contradicts this declaration. No calculated decay is not
    evidence of stability. Quarks except top, gluons, photons, charged leptons
    including tau, and neutrinos terminate this parton-level calculation.

    ``min_probability`` is an optional calculation truncation applied after
    identical-parent assignments have been combined. Its omitted probability
    is retained in the coverage report. ``min_relative_rate`` affects display
    only. Depth/resource limits raise an error instead of silently truncating.
    ``validate_only`` checks the reachable decay graph without enumerating
    combinatorial histories and returns None, for production preflight.
    """
    minimum = _finite(min_probability, 'min_probability')
    if minimum > 1:
        raise CascadeError('min_probability must lie between 0 and 1')
    ratio_limit = _finite(max_width_mass_ratio, 'max_width_mass_ratio')
    threshold = _finite(threshold_widths, 'threshold_widths')
    if ratio_limit == 0:
        raise CascadeError('max_width_mass_ratio must be positive')
    for label, number in (('max_depth', max_depth), ('max_histories', max_histories)):
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise CascadeError(f'{label} must be a positive integer')

    for label in ('model_digest', 'benchmark_digest'):
        values = {_get(item, label) for item in (channels, decays, cross_sections) if _get(item, label)}
        if len(values) > 1:
            raise CascadeError(f'{label} differs between production, decay and cross-section results; recalculate at one benchmark')
    if _get(decays, 'width_digest') and _get(cross_sections, 'width_digest') and _get(decays, 'width_digest') != _get(cross_sections, 'width_digest'):
        raise CascadeError('Cross sections used different decay widths; recalculate cross sections with these decays')

    particles = {}
    for item in (channels, decays, cross_sections):
        for pdg, details in (_get(item, 'particles', {}) or {}).items():
            pdg = int(pdg)
            if pdg in particles:
                for key in ('name', 'mass_gev', 'anti_pdg'):
                    if key in details and key in particles[pdg] and details[key] != particles[pdg][key]:
                        raise CascadeError(f'Particle metadata disagrees for PDG {pdg}: {key}')
            particles.setdefault(pdg, {}).update(details)
    if not particles:
        raise CascadeError('Particle metadata is required; recompute channels and decays with the current runner')

    def info(pdg):
        if pdg not in particles:
            raise CascadeError(f'Missing particle metadata for PDG {pdg}')
        return particles[pdg]

    def name(pdg):
        return str(info(pdg).get('name', pdg))

    def conjugate(pdg):
        partner = info(pdg).get('anti_pdg')
        if partner is None:
            raise CascadeError(f'Missing antiparticle PDG for {name(pdg)}')
        info(int(partner))
        return int(partner)

    names = {}
    for pdg, details in particles.items():
        names.setdefault(str(details.get('name', pdg)).lower(), set()).add(pdg)
    if isinstance(stable_particles, (str, int)):
        stable_particles = (stable_particles,)
    stable = set()
    for value in stable_particles:
        if isinstance(value, int) and not isinstance(value, bool):
            info(value)
            stable.add(value)
        else:
            candidates = names.get(str(value).lower(), set())
            if len(candidates) != 1:
                raise CascadeError(f'Unknown or ambiguous stable particle {value!r}; use a signed PDG ID')
            stable.update(candidates)
    # Stability is a property of a particle and its charge conjugate.
    stable.update(conjugate(pdg) for pdg in tuple(stable))

    totals = {int(pdg): _finite(value, f'total width of PDG {pdg}') for pdg, value in (_get(decays, 'total_widths', {}) or {}).items()}
    mode_source = _get(decays, 'modes', None)
    if mode_source is None:
        raise CascadeError('Full primary decay tables (.modes) are required; two-body tables alone cannot rank cascades')
    modes = {}
    for row in mode_source:
        pdg = int(row['pdg'])
        info(pdg)
        daughters = tuple(sorted(int(p) for p in row['daughters']))
        if len(daughters) < 2:
            raise CascadeError(f'Decay of {name(pdg)} has fewer than two daughters')
        for daughter in daughters:
            info(daughter)
        probability = _finite(row['branching_ratio'], f'branching ratio of {name(pdg)}')
        if probability > 1 + 1e-6:
            raise CascadeError(f'Branching ratio exceeds one for {name(pdg)}')
        if any(previous[0] == daughters for previous in modes.get(pdg, ())):
            raise CascadeError(f'Duplicate primary decay channel for {name(pdg)}; combine its amplitudes in MadGraph first')
        modes.setdefault(pdg, []).append((daughters, probability))
    for pdg, entries in modes.items():
        if math.fsum(probability for _, probability in entries) > 1 + 1e-6:
            raise CascadeError(f'Branching ratios sum to more than one for {name(pdg)}')

    def width(pdg):
        return totals.get(pdg, totals.get(conjugate(pdg)))

    for pdg in stable:
        if (width(pdg) or 0) > 0 or any(probability > 0 for _, probability in modes.get(pdg, ())):
            raise CascadeError(f'{name(pdg)} was declared stable but has a nonzero calculated decay; change the physical inputs and recalculate')

    def primary(pdg):
        if pdg in modes:
            return modes[pdg]
        partner = conjugate(pdg)
        if partner in modes:
            return [(tuple(sorted(conjugate(d) for d in daughters)), probability) for daughters, probability in modes[partner]]
        return []

    def terminal(pdg):
        return pdg in stable or (not info(pdg).get('is_new', abs(pdg) not in _SM_TERMINAL | _SM_HEAVY) and abs(pdg) in _SM_TERMINAL)

    def mass(pdg):
        value = info(pdg).get('mass_gev')
        return None if value is None else abs(_finite(value, f'mass of {name(pdg)}', nonnegative=False))

    def decay_choices(pdg, ancestors):
        if pdg in ancestors:
            raise CascadeError(f'Cyclic primary decay table involving {name(pdg)}; on-shell cascades must decrease mass')
        if len(ancestors) >= max_depth:
            raise CascadeResourceLimit(f'max_depth={max_depth} reached at {name(pdg)}; increase the explicit calculation limit')
        total = width(pdg)
        if total is None:
            raise CascadeError(f'No calculated total width for {name(pdg)}; compute its decays or explicitly declare a physically stable particle')
        if total == 0:
            raise CascadeError(f'Zero calculated width for {name(pdg)} does not establish stability; explicitly declare it stable only if physically justified')
        choices = [(d, p) for d, p in primary(pdg) if p > 0]
        if not choices:
            raise CascadeError(f'No usable primary decay modes for unstable {name(pdg)}; a missing or truncated table is not stability')
        for daughters, _ in choices:
            parent_mass = mass(pdg)
            daughter_masses = [mass(d) for d in daughters]
            if parent_mass is not None and all(m is not None for m in daughter_masses):
                if sum(daughter_masses) > parent_mass + 1e-8 * max(1.0, parent_mass):
                    raise CascadeError(f'Kinematically closed primary decay of {name(pdg)} has positive BR; recalculate widths at this benchmark')
        return choices

    channel_items = _get(channels, 'channels', channels)
    channel_rows = [dict(id=str(_get(c, 'id')), process=_get(c, 'process'),
                         final_pdgs=tuple(sorted(int(p) for p in _get(c, 'final_pdgs')))) for c in channel_items]
    if len({c['id'] for c in channel_rows}) != len(channel_rows):
        raise CascadeError('Duplicate production channel ID')
    if len({c['final_pdgs'] for c in channel_rows}) != len(channel_rows):
        raise CascadeError('Duplicate physical production channel')
    channel_map = {c['id']:c for c in channel_rows}
    seen_rates = set()
    for rate in _get(cross_sections, 'rates', ()):
        key = str(rate['id'])
        if key not in channel_map:
            raise CascadeError(f'Cross section refers to unknown channel {key}')
        if key in seen_rates:
            raise CascadeError(f'Duplicate cross-section result for channel {key}')
        seen_rates.add(key)
        if rate.get('status') in ('error','failed','timeout','cancelled'):
            raise CascadeError(f'Cross-section calculation {key} did not succeed')
        if 'final_pdgs' in rate and tuple(sorted(rate['final_pdgs'])) != channel_map[key]['final_pdgs']:
            raise CascadeError('Cross-section final state disagrees with channel')
        if rate.get('width_digest') and rate['width_digest'] != _get(decays,'width_digest'):
            raise CascadeError('Cross sections used different decay widths')

    @lru_cache(maxsize=4096)
    def visit(pdg, ancestors=()):
        if terminal(pdg):
            return
        for daughters,_ in decay_choices(pdg,ancestors):
            for child in set(daughters):
                visit(child,ancestors+(pdg,))
    for channel in channel_rows:
        for pdg in set(channel['final_pdgs']):
            visit(pdg)
    if validate_only:
        return None

    notices = ['Tree-level narrow-width approximation; no decay-product acceptance or detector efficiency.',
               'Loop-induced production and decay channels are excluded, including gluon fusion and Higgs loop decays in the renormalizable SM template.',
               'MC errors propagate production integration only; width, scale, PDF and approximation errors are not included.',
               'Different production channels and intermediate histories can overlap or interfere; rows are not an inclusive total.',
               'Coverage measures supplied decay-table probabilities, not unknown loop modes or the total physical branching fraction beyond the width approximation.',
               'Quarks, gluons and tau are terminal parton-level objects; this is not hadronization or tau-decay simulation.']
    from madgraph_runner import scale_diagnostics
    notices.append(scale_diagnostics(_get(channels, 'collider', {}) or {},
                                     has_decay_chain=True)['note'])
    notices.append('Explicit decay-chain comparisons can differ through scale choices, finite resonance windows, off-shell kinematics, cuts, spin correlations or interference within retained diagrams, and numerical uncertainty; a discrepancy is not solely a resonance-window effect.')
    approximation = _get(decays, 'approximation')
    if approximation:
        notices.append(str(approximation))
    notices.extend(str(v) for v in (_get(decays, 'coverage_notices', ()) or ()))

    overlap_pairs = set()
    resonance_pdgs = [p for p in particles if not terminal(p) and (width(p) or 0) > 0 and mass(p)]
    for first, second in itertools.combinations(resonance_pdgs, 2):
        if first == conjugate(second):
            continue
        attributes = ('spin', 'color', 'charge')
        if not all(info(first).get(key) is not None and info(first).get(key) == info(second).get(key) for key in attributes):
            continue
        if abs(mass(first) - mass(second)) <= threshold * (width(first) + width(second)):
            overlap_pairs.add(tuple(sorted((first, second))))

    def node_flags(pdg, daughters):
        flags = set()
        value, total = mass(pdg), width(pdg)
        if value is None or value == 0:
            flags.add(f'{name(pdg)}: mass unavailable for narrow-width check')
        else:
            if total / value > ratio_limit:
                flags.add(f'{name(pdg)}: Γ/m > {ratio_limit:g}')
            daughter_masses = [mass(child) for child in daughters]
            if any(item is None for item in daughter_masses):
                flags.add(f'{name(pdg)}: threshold check unavailable')
            elif value - sum(daughter_masses) <= threshold * total:
                flags.add(f'{name(pdg)}: near decay threshold')
        for pair in overlap_pairs:
            if pdg in pair:
                flags.add('Possible overlapping resonances: ' + '/'.join(name(p) for p in pair))
        return flags

    identity = {p:{k:v for k,v in item.items() if k in ('pdg','name','anti_pdg','spin','color','charge','is_new','basis_id')}
                for p,item in particles.items()}
    return dict(particles=particles,channels=channel_rows,particle_identity=identity,total_widths=totals,
                mode_records=list(mode_source),stable=sorted(stable),terminal=terminal,
                decay_choices=decay_choices,node_flags=node_flags,notices=notices,
                limits=dict(min_probability=minimum,max_depth=max_depth,max_width_mass_ratio=ratio_limit,threshold_widths=threshold),
                provenance={k:_get(decays,k) for k in ('model_digest','benchmark_digest','width_digest')})


def rank_final_states(channels, decays, cross_sections=None, stable_particles=(), *,
                      min_probability=0.0, max_depth=12, max_histories=None,
                      min_relative_rate=1e-4, max_width_mass_ratio=0.01,
                      threshold_widths=5.0, show=True, validate_only=False,
                      storage_path=None, discovery=True, batch_size=1000,
                      memory_mb=64, max_render_nodes=10000):
    """Discover all histories with nullable numerical rates and bounded memory.

    max_histories is accepted only for migration and no longer truncates work.
    The SQLite catalogue is resumable; display uses bounded, revision-bound pages.
    """
    from history_store import build_history_store
    if max_histories is not None:
        if type(max_histories) is not int or max_histories < 1:
            raise CascadeError('max_histories must be a positive integer when supplied')
        warnings.warn('max_histories is deprecated and ignored; history expansion streams to disk', DeprecationWarning, stacklevel=2)
    if not 0 <= _finite(min_relative_rate,'min_relative_rate') <= 1:
        raise CascadeError('min_relative_rate must lie between 0 and 1')
    context = _prepare_context(channels,decays,cross_sections,stable_particles,
                min_probability=min_probability,max_depth=max_depth,max_histories=1,
                min_relative_rate=min_relative_rate,max_width_mass_ratio=max_width_mass_ratio,
                threshold_widths=threshold_widths,show=False,validate_only=validate_only)
    if validate_only:
        return None
    if not discovery:
        selected = {str(rate['id']) for rate in _get(cross_sections,'rates',())}
        context['channels'] = [c for c in context['channels'] if c['id'] in selected]
    def wrap(rows):
        return FinalStateRanking(rows,**rows.summary())
    try:
        rows = build_history_store(context,storage_path,batch_size=batch_size,memory_mb=memory_mb,max_render_nodes=max_render_nodes)
        if cross_sections is not None:
            rows.update_rates(cross_sections)
        result = wrap(rows)
    except BaseException as error:
        if getattr(error,'partial_rows',None) is not None:
            error.partial_ranking = wrap(error.partial_rows)
        raise
    if show:
        result.show(min_relative_rate=min_relative_rate)
    return result
