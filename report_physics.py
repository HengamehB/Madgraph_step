"""Model, production and decay tables from the published run artifacts."""
import heapq
import json

from report_data import read_json, record, table_rows
from report_markup import esc, measure, number, percent, section, table


def value(data):
    if data is None:
        return 'Unavailable'
    if isinstance(data, (dict, list, tuple)):
        return esc(json.dumps(data, ensure_ascii=False))
    return esc(data)


def process(label):
    return esc(str(label or '—').replace('>', '→'))


def model_section(root, manifest):
    spectrum = read_json(root, 'spectrum.json', {})
    inputs = read_json(root, 'resolved_inputs.json', {})
    operators = read_json(root, 'selected_operators.json', [])
    states = {s['label']:s for s in spectrum.get('scalar_states', []) if s.get('label')}
    masses = [(esc(label), number(mass), value(states.get(label, {}).get('charge', '—')))
              for label, mass in (spectrum.get('masses') or {}).items()]
    body = '<h3>Mass spectrum</h3>'
    body += table(('Mass eigenstate','Mass (GeV)','Charge (e)'), masses) if masses else '<p>No mass spectrum is recorded for this stage.</p>'
    transform = spectrum.get('transformation')
    if transform:
        body += '<details><summary>Mixing transformation and bases</summary><pre>' + esc(json.dumps({key:spectrum[key] for key in ('raw_field_basis','scalar_basis','transformation') if key in spectrum},ensure_ascii=False,indent=2)) + '</pre></details>'
    body += '<h3>Enabled operator coefficients</h3>'
    couplings = []
    for op in operators:
        oid = op['id']
        power = op.get('coefficient_dimension')
        unit = '—' if power is None else 'Dimensionless' if power == 0 else 'GeV' if power == 1 else 'GeV<sup>' + esc(power) + '</sup>'
        label = esc(op.get('field_content') or op.get('operator') or oid)
        convention = ' · '.join(str(item) for item in (oid,op.get('hc_relation'),op.get('coupling_parameters')) if item)
        couplings.append(('<details><summary>' + label + '</summary>' + esc(convention) + '</details>', value((inputs.get('coefficients') or {}).get(oid)), unit))
    body += table(('Operator','Coefficient','Units'),couplings) if couplings else '<p>No active operator list was saved.</p>'
    return section('model','Model and spectrum',body)


def production_section(root, manifest):
    if not record(manifest,'production_channels'):
        return section('production','Production','<p>Channel discovery results are unavailable.</p>')
    count = 0
    def calculated():
        nonlocal count
        for row in table_rows(root, manifest, 'cross_sections'):
            sigma = row.get('cross_section_pb')
            if sigma not in ('', None):
                count += 1
                yield float(sigma), row
    leading = heapq.nlargest(25, calculated(), key=lambda item:item[0])
    processes = {r['id']:None for _,r in leading}
    missing = set(manifest.get('uncomputed_production_ids') or [])
    pending = []
    total = 0
    for row in table_rows(root, manifest, 'production_channels'):
        total += 1
        cid = row['id']
        if cid in processes:
            processes[cid] = row.get('process')
        if len(pending) < 10 and (cid in missing or not count):
            pending.append((esc(cid), process(row.get('process')), 'Not calculated'))
    rows = [(esc(r['id']),process(r.get('process') or processes[r['id']]),measure(sigma,r.get('error_pb'))) for sigma,r in leading]
    body = f'<p><strong>{count:,} of {total:,}</strong> production channels have a numerical cross section.</p>'
    if count > len(rows) or total - count > len(pending):
        body += '<p>Showing up to 25 leading calculated channels and 10 awaiting integration. The saved tables include the full list.</p>'
    if rows or pending:
        body += table(('Channel','Production process','Cross section ± integration error (pb)'),rows + pending)
    return section('production','Production',body)


def decay_section(root, manifest):
    if not record(manifest, 'total_widths'):
        return section('decays','Widths and branching fractions','<p>Widths are unavailable for this stage.</p>')
    top = {}
    for index, row in enumerate(table_rows(root, manifest, 'decays')):
        if row.get('branching_ratio') in ('',None):
            continue
        item = (float(row['branching_ratio']),-index,row.get('process'))
        heap = top.setdefault(row['pdg'], [])
        if len(heap) < 5:
            heapq.heappush(heap,item)
        else:
            heapq.heappushpop(heap,item)
    rows = []
    for row in table_rows(root, manifest, 'total_widths'):
        modes = '<br>'.join(process(label) + ' · ' + percent(br) for br,_,label in sorted(top.get(row['pdg'],()),reverse=True))
        omitted = row.get('omitted_probability')
        rows.append((esc(row.get('name') or row['pdg']),number(row.get('width_gev')),modes or 'No reported mode',percent(omitted) if omitted not in ('',None) else 'Unknown'))
    body = '<p>Up to five leading direct decays per parent. Each branching fraction uses that parent’s calculated total width.</p>'
    body += table(('Parent','Width (GeV)','Leading modes · branching fraction','Reported omitted probability'),rows) if rows else '<p>No widths were recorded.</p>'
    body += '<p class="note">Omitted contributions can be uncertain. A zero calculated width alone does not establish stability.</p>'
    return section('decays','Widths and branching fractions',body)


def methods_section(manifest, page):
    settings = manifest.get('settings') or {}
    body = '<p>Tree-level production excludes loop-induced processes. The ranked rates use the narrow-width approximation. Distinct histories can overlap, so adding them can double-count a final state.</p>'
    collider = settings.get('collider') or {}
    if str(collider.get('dynamical_scale_choice')) == '3':
        body += '<p>Scale choice 3 depends on external momenta. Production and explicit decay-chain integrations can use different numerical scales.</p>'
    if manifest.get('vacuum'):
        body += '<p>The assumed electroweak vacuum has a 125 GeV Higgs and zero new-scalar VEVs.</p>'
    notes = list(page.get('notices') or ()) + list((manifest.get('vacuum') or {}).get('warnings') or ())
    if notes:
        body += '<ul>' + ''.join('<li>' + esc(note) + '</li>' for note in dict.fromkeys(notes)) + '</ul>'
    if page.get('coverage'):
        body += '<details><summary>Cascade probability coverage by production channel</summary>' + table(('Channel','Covered','Omitted'),[(esc(k),percent(v[0]),percent(v[1]) if v[1] is not None else 'Unknown') for k,v in page['coverage'].items()]) + '</details>'
    for key,title in [('collider','Collider, scales and cuts'),('decays','Decay settings'),('ranking','Ranking settings'),('explicit_decays','Explicit integrations')]:
        group = settings.get(key) or {}
        if group:
            body += '<details><summary>' + title + '</summary>' + table(('Setting','Value'),[(esc(k),value(v)) for k,v in group.items()]) + '</details>'
    return section('methods','Methods and coverage',body)
