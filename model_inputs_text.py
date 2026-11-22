"""Plain-text model inputs and coefficient setup, without executing Python.

The shared parser lives in tex_inputs for compatibility. This module only
normalizes declarative values and reconciles their semantic bindings.
"""
from copy import deepcopy
from pathlib import Path

from coefficient_defaults import PROFILE, coefficient_default

PROVENANCE_SCHEMA = 'model-input-origins-v1'
FOOTER_BEGIN = '# INPUT_METADATA_BEGIN (managed by setup; keep with this file)'
FOOTER_END = '# INPUT_METADATA_END'
_PHYSICAL = ('coefficients', 'sm_inputs', 'sm_scheme', 'vacuum', 'stable_particles',
             'vector_specifications', 'model_name', 'observed_higgs_mass', 'higgs_mass', 'vevs')
_SETTINGS = ('collider', 'decays', 'ranking', 'explicit_decays')


def normalize_inputs(source, definition, mappings):
    from tex_inputs import _error
    values = source.assignments
    physical = mappings['physical_inputs']
    for name in _PHYSICAL:
        if name in values:
            if name in physical and physical[name] != values[name]:
                raise _error(source.path, source.line(name), f'Conflicting {name} and physical_inputs.{name}; use one value.')
            physical[name] = deepcopy(values[name])
    from fixed_sm import normalize_fixed_sm
    # Both supported text formats share the same numerical contract.
    try:
        physical = normalize_fixed_sm(physical, definition.fields)
    except ValueError as exc:
        key = next((name for name in ('sm_scheme', 'vacuum', 'vevs', 'higgs_mass',
                                      'observed_higgs_mass', 'sm_inputs') if name in values and name in str(exc)),
                   'physical_inputs')
        raise _error(source.path, source.line(key), str(exc)) from exc
    physical.setdefault('model_name', 'SM_' + '_'.join(field.name for field in definition.fields))
    physical.setdefault('stable_particles', [])
    mappings['physical_inputs'] = physical
    calculation = values.get('calculation', {})
    if not isinstance(calculation, dict) or set(calculation) - set(_SETTINGS):
        raise _error(source.path, source.line('calculation'), 'calculation may contain only collider, decays, ranking and explicit_decays dictionaries.')
    for name, value in calculation.items():
        if not isinstance(value, dict):
            raise _error(source.path, source.line('calculation'), f'calculation.{name} must be a dictionary.')
        if name in values and mappings[name] != value:
            raise _error(source.path, source.line(name), f'Conflicting {name} and calculation.{name}; use one value.')
        mappings[name] = deepcopy(value)
    return mappings


def input_provenance(document):
    """Derive current origins; edited numbers cannot retain an automatic label."""
    from tex_inputs import input_bindings, operator_inventory
    stored = document.assignments.get('_input_provenance', {})
    stored = stored if isinstance(stored, dict) and stored.get('schema') == PROVENANCE_SCHEMA else {}
    previous = stored.get('coefficients', {})
    previous = previous if isinstance(previous, dict) else {}
    bindings = input_bindings(document.definition)
    records = {}
    table = document.assignments.get('operators')
    active_info = {info['key']: info for info in document.definition.operator_inputs()}
    keys = table if table is not None else active_info
    saved_bindings = document.assignments.get('_input_bindings', {})
    saved_bindings = saved_bindings.get('coefficients', {}) if isinstance(saved_bindings, dict) else {}
    saved_bindings = saved_bindings if isinstance(saved_bindings, dict) else {}
    formal_rows = {document.definition.coefficient_key(row['id']): row
                   for row in operator_inventory(document.definition).lagrangian.operators} if table is not None else {}
    for key in keys:
        enabled = table[key]['enabled'] if table is not None else True
        value = table[key]['coefficient'] if table is not None else document.physical_inputs.get('coefficients', {}).get(key)
        # Dormant rows retain their last reviewed convention. Merely viewing a
        # model must not rebind their saved numbers to today's compiler basis.
        meaning = bindings['coefficients'][key] if enabled else saved_bindings.get(key)
        if key in active_info:
            units = active_info[key]['units']
        else:
            row = formal_rows[key]
            dimension = row['coefficient_dimension']
            units = 'dimensionless' if dimension == 0 else 'GeV' if dimension == 1 else f'GeV^{dimension}'
        old = previous.get(key, {})
        old = old if isinstance(old, dict) else {}
        matches = old.get('binding') == meaning and old.get('value') == value
        record = deepcopy(old) if matches else dict(origin='explicit' if value is not None else 'unresolved')
        record.update(value=deepcopy(value), binding=meaning, units=units)
        if table is not None:
            record['enabled'] = enabled
        records[key] = record
    raw = document.assignments.get('physical_inputs', {})
    sm = document.assignments.get('sm_inputs', raw.get('sm_inputs', {}))
    defaults = {
        'stable_particles': dict(origin='explicit' if 'stable_particles' in document.assignments or 'stable_particles' in raw else 'default',
                                 value=deepcopy(document.physical_inputs['stable_particles'])),
        'sm_inputs': {key: dict(origin='explicit' if key in sm else 'default', value=value)
                      for key, value in document.physical_inputs['sm_inputs'].items()},
    }
    return dict(schema=PROVENANCE_SCHEMA, profile=PROFILE, coefficients=records, defaults=defaults)


def _render(source, changes, metadata):
    """Keep the binding/provenance footer self-contained and outside user values."""
    from tex_inputs import _parse, _render_changes
    from pprint import pformat
    if '_metadata' in source.assignments:
        from model_metadata import publish_metadata
        reference = publish_metadata(source.path, metadata, reference=source.operator_reference)
        return _render_changes(source, {**changes, '_metadata': reference})
    text = _render_changes(source, changes)
    parsed = _parse(source.path, text)
    lines = parsed.block.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == FOOTER_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.strip() == FOOTER_END]
    if starts or ends:
        if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
            from tex_inputs import _error
            raise _error(source.path, 1, 'Malformed managed input metadata footer.')
        # Only metadata/comments may be inside the managed footer. Never erase
        # a user coefficient declaration accidentally placed between markers.
        for name, node in parsed.nodes.items():
            if starts[0] < node.lineno - 1 < ends[0] and name not in metadata:
                from tex_inputs import _error
                raise _error(source.path, node.lineno, 'Move editable declarations outside the managed metadata footer.')
    removed = set(range(starts[0], ends[0] + 1)) if starts else set()
    for name in metadata:
        node = parsed.nodes.get(name)
        if node is not None:
            removed.update(range(node.lineno - 1, node.end_lineno))
    text = ''.join(line for i, line in enumerate(lines) if i not in removed).rstrip() + '\n\n'
    block = text + FOOTER_BEGIN + '\n' + ''.join(name + ' = ' + pformat(value, sort_dicts=False, width=100) + '\n'
                                                    for name, value in metadata.items()) + FOOTER_END + '\n'
    return parsed.text[:parsed.block_start] + block + parsed.text[parsed.block_end:]


def setup_text(path, *, input_fn=input, output_fn=print):
    from tex_inputs import (_parse, _model_args, _document, _error, _prompt_value, _validate_coefficient,
                            input_bindings, operator_bindings, _write_atomic, BINDING_SCHEMA,
                            OPERATOR_BINDING_SCHEMA)
    from model_builder import ModelDefinition
    source = _parse(path)
    args, changes = _model_args(source), {}
    if args['scenario'] is None:
        from lagrangian_engine import expand_models, model_label
        from lagrangian_maker import _field_input
        fields = [_field_input(args['x'], 'X')]
        if args['y'] is not None:
            fields.append(_field_input(args['y'], 'Y'))
        scenarios = tuple(expand_models(fields))
        if len(scenarios) > 1:
            for index, fields in enumerate(scenarios, 1):
                output_fn(f'{index}: {model_label(fields)}')
            def valid(value):
                if type(value) is not int or not 1 <= value <= len(scenarios):
                    raise ValueError('Choose one displayed scenario number.')
            args['scenario'] = _prompt_value('Scenario', source, input_fn, output_fn, validator=valid)
            changes['scenario'] = args['scenario']
    definition = ModelDefinition(**args)
    document = _document(source, definition)
    table = deepcopy(source.assignments.get('operators'))
    bindings = operator_bindings(definition) if table is not None else input_bindings(definition)
    old = source.assignments.get('_input_bindings', {})
    schema = OPERATOR_BINDING_SCHEMA if table is not None else BINDING_SCHEMA
    old = old if isinstance(old, dict) and old.get('schema') == schema else {}
    old_coefficients = old.get('coefficients', {})
    old_coefficients = old_coefficients if isinstance(old_coefficients, dict) else {}
    previous = source.assignments.get('_input_provenance', {})
    previous = previous.get('coefficients', {}) if isinstance(previous, dict) and previous.get('schema') == PROVENANCE_SCHEMA else {}
    original = document.physical_inputs
    supplied = original.get('coefficients', {})
    if not isinstance(supplied, dict):
        raise _error(source.path, source.line('coefficients'), 'coefficients must be a dictionary.')
    coefficients, records, unresolved = {}, {}, {}
    if table is not None:
        provenance = input_provenance(document)['coefficients']
        for key, entry in table.items():
            if not entry['enabled']:
                # Do not validate, regenerate, or silently adopt a dormant
                # coefficient. Its previous meaning must survive setup too.
                records[key] = provenance[key]
                if key in old_coefficients:
                    bindings['coefficients'][key] = old_coefficients[key]
                else:
                    bindings['coefficients'].pop(key, None)
    for info in definition.operator_inputs():
        key = info['key']
        meaning = bindings['coefficients'][key]
        value = supplied.get(key)
        record = previous.get(key, {}) if isinstance(previous, dict) else {}
        record = record if isinstance(record, dict) else {}
        old_meaning = old_coefficients.get(key)
        valid = False
        try:
            _validate_coefficient(value, info)
            valid = True
        except (ValueError, TypeError):
            pass
        output_fn(f"{key}: {info['units']}; {'real' if info['real'] else 'real or complex'}. {info['hint']}")
        output_fn('Entry order: ' + ', '.join(info['invariant_labels']))
        if valid and old_meaning == meaning:
            if record.get('binding') != meaning or record.get('value') != value:
                record = dict(origin='explicit')
            else:
                record = deepcopy(record)
            output_fn(f'Keeping verified value: {value!r}')
        else:
            generated = record.get('origin') == 'default' and record.get('binding') == old_meaning and record.get('value') == value
            if key in supplied and value is not None and not generated:
                output_fn(f'Unbound or changed meaning for {key}; enter the value explicitly to adopt it.')
                value = _prompt_value(key, source, input_fn, output_fn,
                                      validator=lambda value, info=info: _validate_coefficient(value, info))
                record = dict(origin='explicit')
            else:
                record = coefficient_default(definition, info)
                value = record['value']
                record['origin'] = 'default' if value is not None else 'unresolved'
                if value is None:
                    unresolved[key] = record['reason']
                    output_fn('Unresolved: ' + record['reason'])
                else:
                    _validate_coefficient(value, info)
                    output_fn(f"Generated {value!r}: {record['reason']}")
        coefficients[key] = value
        record.update(binding=meaning, value=deepcopy(value), units=info['units'])
        if table is not None:
            table[key]['coefficient'] = deepcopy(value)
            record['enabled'] = True
        records[key] = record
    # Keep the input file’s original coefficient layout during setup.
    if table is not None:
        changes['operators'] = table
        if 'model_signature' in source.assignments.get('physical_inputs', {}):
            nested = deepcopy(source.assignments['physical_inputs'])
            nested.pop('model_signature')
            changes['physical_inputs'] = nested
    elif 'coefficients' in source.assignments or 'physical_inputs' not in source.assignments:
        changes['coefficients'] = coefficients
    else:
        nested = deepcopy(source.assignments['physical_inputs'])
        nested.pop('model_signature', None)
        nested['coefficients'] = coefficients
        changes['physical_inputs'] = nested
    if 'max_histories' in document.ranking:
        ranking = {key: value for key, value in document.ranking.items() if key != 'max_histories'}
        if 'ranking' in source.assignments:
            changes['ranking'] = ranking
        else:
            changes['calculation'] = {**source.assignments.get('calculation', {}), 'ranking': ranking}
    for field in definition.fields:
        if field.spin == 'vector' and not original.get('vector_specifications', {}).get(field.name):
            unresolved['vector_specifications.' + field.name] = 'Declare the physical vector theory explicitly, e.g. a Proca effective matter vector.'
    metadata = dict(_input_bindings=bindings, _input_provenance=dict(schema=PROVENANCE_SCHEMA,
                    profile=PROFILE, coefficients=records, unresolved=unresolved))
    if (table is not None and source.operator_reference
            and (bindings != old or changes.get('scenario') is not None)):
        from model_initialization import format_operator_reference
        from tex_inputs import operator_inventory
        inventory = operator_inventory(definition)
        active = {info['key']: info for info in definition.operator_inputs()}
        reference_rows = []
        for row, formal in zip(inventory.lagrangian.operators, inventory.operator_inputs(), strict=True):
            key = formal['key']
            info = active.get(key, formal)
            defaults = coefficient_default(definition if key in active else inventory, info)
            reference_rows.append((row, info, defaults))
        source.operator_reference = format_operator_reference(reference_rows)
    text = _render(source, changes, metadata)
    updated = _document(_parse(source.path, text), definition)
    # Build validates only the declarative/backend contract. No vacuum or tools.
    if not unresolved:
        updated.build()
    changed = _write_atomic(source, text)
    output_fn(('Inputs need explicit values/specifications: ' if unresolved else 'Coefficient inputs ready: ')
              + str(source.path) + ('' if changed else ' (unchanged)'))
    return updated


def show_inputs(document, output_fn=print):
    """Read-only coefficient discovery using the same defaults as TXT setup."""
    from pprint import pformat
    output_fn('# Coefficients multiply Lagrangian operators in the normalization shown for each row.')
    output_fn('# Generated values are starting choices; spectrum checks run automatically during export.')
    output_fn('# Masses and mixing are derived. No independent VEV or width seeds are requested.')
    supplied = document.physical_inputs.get('coefficients', {})
    values = {}
    table = document.assignments.get('operators')
    infos = document.definition.operator_inputs()
    if table is not None:
        from tex_inputs import _discover_operator_table, operator_inventory
        infos = _discover_operator_table(document)['operators']
        inventory = operator_inventory(document.definition)
        formal = {document.definition.coefficient_key(row['id']): row
                  for row in inventory.lagrangian.operators}
    for info in infos:
        default = coefficient_default(inventory if table is not None and not info['enabled'] else document.definition, info)
        key = info['key']
        provided = table[key]['coefficient'] if table is not None and key in table else supplied.get(key, default['value'])
        values[key] = provided
        output_fn(f"# {key!r}: {info['units']}; {'real' if info['real'] else 'real or complex; h.c. included'}. {info['hint']}")
        output_fn('# Ordered contractions: ' + ', '.join(info['invariant_labels']))
        if table is not None:
            output_fn('# ' + ('Enabled; active basis.' if info['enabled'] else 'Disabled; full-catalogue basis, checked again when enabled.'))
            output_fn('# Hermitian conjugation: ' + formal[key]['hc_relation'])
            limitations = [part for part in (info.get('obstruction'), formal[key].get('notes')) if part]
            if not info.get('kind') and not info.get('obstruction'):
                limitations.append('No implemented compiler lowering for this catalogue row.')
            if limitations:
                output_fn('# Limitations: ' + ' '.join(limitations))
            output_fn('# Saved coefficient: ' + repr(provided))
        else:
            output_fn('# ' + ('Provided value.' if key in supplied else 'Generated default: ' + default['reason']))
    if table is not None:
        output_fn('operators = ' + pformat(table, sort_dicts=False, width=100))
    else:
        output_fn('coefficients = ' + pformat(values, sort_dicts=False, width=100))
    output_fn('# Fixed electroweak vacuum: unmixed 125 GeV Higgs, zero new scalar VEVs.')
    output_fn('sm_inputs = ' + pformat(document.physical_inputs['sm_inputs'], sort_dicts=False))
    output_fn('stable_particles = ' + repr(document.physical_inputs.get('stable_particles', [])))
    if any(field.spin == 'vector' for field in document.definition.fields):
        output_fn('vector_specifications = ' + pformat(document.physical_inputs.get('vector_specifications',
                    {field.name: None for field in document.definition.fields if field.spin == 'vector'})))
