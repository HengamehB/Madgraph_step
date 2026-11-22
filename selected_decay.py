"""Selected decay-history integrations with verified MadWidth primary diagrams.

The factorized ranking and an explicit matrix-element integration are separate
predictions. No branching ratio or identical-parent factor multiplies the latter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path


class PrimaryDiagramError(ValueError):
    category = 'compiler_failure'


def canonical_forest(forest):
    def node(value):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('A history node must contain its PDG and child nodes')
        pdg, children = value
        if type(pdg) is not int or not pdg:
            raise ValueError('History PDGs must be nonzero integers')
        return pdg, tuple(sorted(node(child) for child in children))
    return tuple(sorted(node(tree) for tree in forest))


def primary_keys(forest):
    answer = set()
    def visit(tree):
        pdg, children = tree
        if children:
            answer.add((pdg, tuple(sorted(child[0] for child in children))))
            for child in children:
                visit(child)
    for tree in forest:
        visit(tree)
    return answer


def render_history(forest, names, orders=''):
    """One clause per decaying occurrence, including identical parents."""
    forest = canonical_forest(forest)
    def decay(tree):
        parent, children = tree
        text = names[parent] + ' > ' + ' '.join(names[child[0]] for child in children)
        if orders:
            text += ' ' + orders
        nested = [decay(child) for child in children if child[1]]
        return '(' + text + (', ' + ', '.join(nested) if nested else '') + ')'
    core = 'p p > ' + ' '.join(names[tree[0]] for tree in forest)
    if orders:
        core += ' ' + orders
    return core + ''.join(', ' + decay(tree) for tree in forest if tree[1])


def diagram_definition(diagram, parent, model, *, conjugate=False):
    """Canonical rooted tree, independent of numbering/identical-leg order.

    Interaction IDs and internal PDGs are retained. The artificial identity
    vertex used only by MG decay chains is removed before matching MadWidth.
    """
    vertices = list(diagram['vertices'])
    root_number = 1
    if vertices and vertices[-1]['id'] == 0:
        identity = vertices.pop()
        others = [leg['number'] for leg in identity['legs'] if leg['number'] != 1]
        if len(others) != 1:
            raise PrimaryDiagramError('Unrecognized MadGraph decay identity vertex')
        root_number = others[0]
    def anti(pdg):
        return model.get_particle(pdg).get_anti_pdg_code()
    interaction_map = {}
    if conjugate:
        signatures = defaultdict(list)
        for interaction in model['interactions']:
            signature = (tuple(sorted(p.get_pdg_code() for p in interaction['particles'])),
                         tuple(sorted(interaction['orders'].items())))
            signatures[signature].append(interaction['id'])
        for interaction in model['interactions']:
            signature = (tuple(sorted(p.get_anti_pdg_code() for p in interaction['particles'])),
                         tuple(sorted(interaction['orders'].items())))
            matches = signatures[signature]
            if len(matches) == 1:
                interaction_map[interaction['id']] = matches[0]
    # MG numbers an internal current with one of its descendant external
    # numbers. Follow its ordered vertex reductions; numbers are not globally
    # unique graph-edge labels.
    currents = {}
    for vertex in vertices:
        interaction = vertex['id']
        if conjugate:
            if interaction not in interaction_map:
                raise PrimaryDiagramError('Cannot uniquely conjugate a primary-decay interaction')
            interaction = interaction_map[interaction]
        children = []
        for leg in vertex['legs'][:-1]:
            number, pdg = leg['number'], leg['id']
            if conjugate:
                pdg = anti(pdg)
            branch = (('internal', pdg, currents.pop(number)) if number in currents
                      else ('external', pdg))
            children.append(branch)
        currents[vertex['legs'][-1]['number']] = interaction, tuple(sorted(children))
    if len(currents) != 1 or root_number not in currents:
        raise PrimaryDiagramError('Disconnected primary-decay diagram')
    tree = currents[root_number]
    return (anti(parent) if conjugate else parent), tree


def _encoded(definition):
    return json.dumps(definition, separators=(',', ':'))


def _definitions(amplitude, model, *, conjugate=False):
    parent = amplitude['process'].get_initial_ids()[0]
    return {_encoded(diagram_definition(diagram, parent, model, conjugate=conjugate))
            for diagram in amplitude['diagrams']}


def primary_provenance(madgraph_root, settings):
    root = Path(madgraph_root)
    files = ('VERSION', 'mg5decay/decay_objects.py', 'madgraph/core/base_objects.py',
             'madgraph/core/diagram_generation.py', 'madgraph/interface/madgraph_interface.py',
             'models/model_reader.py', 'models/import_ufo.py')
    return dict(schema='madwidth-primary-provenance-v1', settings=dict(settings),
                sources={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                matcher=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def capture_primary_diagrams(command, modes, provenance):
    """Attach available exact MadWidth diagrams; reconstruct missing ones later."""
    model = getattr(command, '_curr_decaymodel', None)
    for mode in modes:
        mode['primary_provenance'] = provenance
        if model is None:
            continue
        parent = int(mode['pdg'])
        daughters = tuple(sorted(mode['daughters']))
        positive = parent > 0 or model.get_particle(parent)['self_antipart']
        if not positive:
            daughters = tuple(sorted(model.get_particle(p).get_anti_pdg_code() for p in daughters))
        particle = model.get_particle(abs(parent))
        for amp in particle.get_amplitudes(len(daughters)) or ():
            if tuple(sorted(amp['process'].get_final_ids())) == daughters:
                mode['primary_diagrams'] = dict(schema='madwidth-primary-diagrams-v1',
                    parent=abs(parent), daughters=daughters,
                    definitions=sorted(_definitions(amp, model)))
                break


def restrict_primary_amplitudes(command, card, forest, modes, provenance=None):
    """Reconstruct/verify primary diagrams and filter BEFORE HELAS generation."""
    requested = primary_keys(forest)
    if not requested:
        return []
    from madgraph.core import base_objects
    from mg5decay.decay_objects import DecayModel
    generated = command._curr_amps
    model = command._curr_model
    # DecayModel converts its input's particles in place. Keep those special
    # objects out of the production model: HELAS deep-copies its ordinary
    # particle list when assembling identical-parent decay chains.
    decaymodel = DecayModel(copy.deepcopy(model), True)
    decaymodel.read_param_card(str(card))
    recorded = {(int(mode['pdg']), tuple(sorted(mode['daughters']))): mode for mode in modes}
    allowed = {}
    for parent, daughters in sorted(requested):
        if (parent, daughters) not in recorded:
            raise PrimaryDiagramError(f'History selects a mode absent from its decay table: {parent} > {daughters}')
        if provenance is not None and recorded[parent, daughters].get('primary_provenance') != provenance:
            raise PrimaryDiagramError('MadWidth primary-diagram provenance is missing or changed; recompute decays before selecting this history')
        physical = model.get_particle(parent)
        conjugate = parent < 0 and not physical['self_antipart']
        target = tuple(sorted(model.get_particle(p).get_anti_pdg_code() for p in daughters)) if conjugate else daughters
        particle = decaymodel.get_particle(abs(parent))
        existing = particle.get_amplitudes(len(target))
        if not existing:
            decaymodel.find_channels(particle, len(target), min_br=0)
        matching = [amp for amp in particle.get_amplitudes(len(target)) or ()
                    if tuple(sorted(amp['process'].get_final_ids())) == target]
        if len(matching) != 1:
            raise PrimaryDiagramError(f'MadWidth cannot reconstruct exactly one primary mode {parent} > {daughters}')
        definitions = _definitions(matching[0], decaymodel)
        saved = recorded[parent, daughters].get('primary_diagrams')
        if saved and set(saved['definitions']) != definitions:
            raise PrimaryDiagramError(f'MadWidth primary diagrams changed for {parent} > {daughters}')
        allowed[parent, daughters] = definitions
    reports = []
    seen = set()
    def visit(chain, is_decay=False):
        for amp in chain.get('amplitudes'):
            if not is_decay:
                continue
            process = amp['process']
            parent = process.get_initial_ids()[0]
            daughters = tuple(sorted(process.get_final_ids()))
            key = parent, daughters
            if key not in allowed:
                raise PrimaryDiagramError(f'Unexpected generated primary subamplitude: {key}')
            conjugate = parent < 0 and not model.get_particle(parent)['self_antipart']
            tagged = [(diagram, _encoded(diagram_definition(diagram, parent, model, conjugate=conjugate)))
                      for diagram in amp['diagrams']]
            generated_definitions = {tag for _, tag in tagged}
            missing = allowed[key] - generated_definitions
            if missing:
                raise PrimaryDiagramError(f'Generated decay subamplitude misses {len(missing)} MadWidth primary diagrams for {key}')
            kept = [diagram for diagram, tag in tagged if tag in allowed[key]]
            if not kept:
                raise PrimaryDiagramError(f'No matched primary diagrams remain for {key}')
            amp.set('diagrams', base_objects.DiagramList(kept))
            seen.add(key)
            reports.append(dict(parent=parent, daughters=daughters, generated=len(tagged),
                                retained=len(kept), excluded=len(tagged)-len(kept),
                                definitions=sorted(allowed[key]), verified=True))
        for child in chain.get('decay_chains'):
            visit(child, True)
    for chain in generated:
        visit(chain)
    if seen != requested:
        raise PrimaryDiagramError('Generated decay chains do not match every selected primary mode')
    return reports


def integrate_selected_history(channels, decays, history_row, *, runtime=None,
        output_dir=None, cut_decays=False, bwcutoff=15.0, accuracy=None, timeout=7200):
    """Integrate only this canonical history; do not multiply its result by BRs."""
    from madgraph_runner import _run_job, validate_provenance, _collider, scale_diagnostics
    validate_provenance(channels, decays)
    if type(cut_decays) is not bool:
        raise ValueError('cut_decays must be a boolean')
    if isinstance(bwcutoff, bool) or not math.isfinite(float(bwcutoff)) or bwcutoff <= 0:
        raise ValueError('bwcutoff must be positive and finite')
    forest = canonical_forest(history_row['history_key'])
    channel = next((c for c in channels if c.id == history_row['production_id']), None)
    if channel is None or Counter(t[0] for t in forest) != Counter(channel.final_pdgs):
        raise ValueError('Selected history does not belong to its production channel')
    modes = {(int(m['pdg']), tuple(sorted(m['daughters']))): m for m in decays.modes}
    for key in primary_keys(forest):
        if key not in modes:
            raise ValueError(f'Selected history contains an unknown primary decay: {key}')
        if not modes[key].get('primary_provenance'):
            raise PrimaryDiagramError('Legacy decay tables lack primary-diagram provenance; recompute decays before selected integration')
    names = {int(p): info['name'] for p, info in channels.particles.items()}
    process = render_history(forest, names)
    collider = dict(channels.collider)
    if accuracy is not None:
        collider['accuracy'] = accuracy
    collider = _collider(collider)
    history_id = history_row.get('history_id') or history_row.get('id') or ('H' + hashlib.sha256(
        _encoded((tuple(sorted(channel.final_pdgs)), forest)).encode()).hexdigest()[:20])
    result, run = _run_job(dict(action='selected_history', ufo=str(channels.ufo),
        parameter_card=str(decays.parameter_card), parameters={}, collider=collider,
        process=process, history_key=forest, primary_modes=list(modes.values()),
        decay_settings=decays.settings,
        decay_chain_settings=dict(cut_decays=cut_decays, bwcutoff=float(bwcutoff))),
        runtime or channels.runtime, output_dir, timeout=timeout)
    validate_provenance(channels, decays)
    result.update(schema='selected-history-integration-v1', method='madgraph_decay_chain',
        history_id=history_id, history_key=forest, production_id=channel.id,
        production_process=channel.process, process_definition=process,
        factorized_reference_pb=history_row.get('rate_pb'), status='computed',
        collider=collider, decay_chain_settings=dict(cut_decays=cut_decays, bwcutoff=float(bwcutoff)),
        scale_diagnostics=scale_diagnostics(collider, has_decay_chain=bool(primary_keys(forest))),
        comparison_notice=('An explicit chain and a factorized rate can differ through scale evaluation, '
            'finite resonance windows, off-shell kinematics, cuts, spin correlations/interference within '
            'the retained diagrams, and numerical integration. A difference is not automatically a Breit-Wigner-window effect.'),
        model_digest=channels.model_digest, benchmark_digest=channels.benchmark_digest,
        width_digest=decays.width_digest, run_directory=str(run))
    (run / 'selected_history_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
