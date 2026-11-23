"""Build the offline HTML report from a published run. No physics tools are needed."""
import csv
from pathlib import Path

from pipeline_storage import RunStore, atomic_text
from report_data import href, load_saved_page, read_json, record
from report_markup import document, esc, measure, percent, section, table
from report_physics import model_section, production_section, decay_section, methods_section, process
from run_paths import contained_path


def link(root, path, label, download=False):
    try:
        url = href(root, path)
    except (ValueError, OSError):
        url = None
    if not url:
        return ''
    attrs = ' class="download" download="ranking.csv"' if download else ''
    return '<a href="' + esc(url) + '"' + attrs + '>' + esc(label) + '</a>'


FILTERS = '''<div class="filters" hidden id="filters"><label>Exact terminal state <select id="exact"><option value="">All displayed states</option></select></label><label>Parton-level signature <select id="signature"><option value="">All displayed signatures</option><option value="jets">Only quarks or gluons</option><option value="leptons">Contains e or μ</option><option value="nu">Contains a neutrino</option></select></label></div>'''
SCRIPT = '''<script>(()=>{const controls=document.querySelector('#filters');if(!controls)return;
const rows=Array.from(document.querySelectorAll('#ranked tbody tr'));const exact=document.querySelector('#exact'),sig=document.querySelector('#signature');
const states=new Map();for(const r of rows){const cell=r.querySelector('[data-state]');if(cell)states.set(cell.dataset.state,cell.textContent)}
for(const [key,label] of states){const opt=document.createElement('option');opt.value=key;opt.textContent=label;exact.append(opt)}
function apply(){let count=0;for(const r of rows){const cell=r.querySelector('[data-state]');r.hidden=Boolean((exact.value&&cell?.dataset.state!==exact.value)||(sig.value&&!(cell?.dataset.tags||'').split(' ').includes(sig.value)));if(!r.hidden)count++}document.querySelector('#visible-count').textContent=count+' of '+rows.length+' displayed histories match.'}
exact.addEventListener('change',apply);sig.addEventListener('change',apply);controls.hidden=false;})();</script>'''


def terminal(row):
    label = esc(row.get('final_state') or 'Unavailable')
    codes = row.get('final_pdgs')
    if not codes:
        return label
    key = ','.join(str(int(p)) for p in sorted(codes))
    kinds = {abs(int(p)) for p in codes}
    tags = []
    if kinds <= {1,2,3,4,5,6,21}:
        tags.append('jets')
    if kinds & {11,13}:
        tags.append('leptons')
    if kinds & {12,14,16}:
        tags.append('nu')
    return '<span data-state="' + esc(key) + '" data-tags="' + ' '.join(tags) + '">' + label + '</span>'


def ranked(page, download=''):
    rows = page['rows']
    text = download
    total = page.get('total')
    calculated = page.get('calculated')
    if total is not None:
        text += f'<p>{total:,} saved histories' + (f' · {calculated:,} with calculated rates' if calculated is not None else '') + '.</p>'
    if not rows:
        return section('ranked','Ranked histories',text + '<p>No ranked rows are available for this stage.</p>')
    text += f'<p>{len(rows):,} histories on this page. The final-state filters apply only to these rows.</p>'
    text += FILTERS + '<p id="visible-count" aria-live="polite" class="note"></p>'
    result = []
    for row in rows:
        detail = '<details><summary>History ID and flags</summary><code>' + esc(row.get('history_id','')) + '</code>'
        flags = row.get('flags') or []
        if flags:
            detail += '<ul>' + ''.join('<li>' + esc(flag) + '</li>' for flag in flags) + '</ul>'
        detail += '</details>'
        if flags:
            detail = '<small>' + esc(flags[0]) + '</small>' + detail
        result.append((process(row.get('production_process')),process(row.get('history')) + detail,terminal(row),measure(row.get('production_cross_section_pb'),row.get('production_error_pb')),percent(row.get('branching_ratio')),'<strong>' + measure(row.get('rate_pb'),row.get('error_pb')) + '</strong>'))
    text += table(('Production','Decay history','Terminal state','σ ± MC error (pb)','Cascade BR','σ × BR ± MC error (pb)'),result)
    if page.get('next_command'):
        text += '<p>Render the next history page</p><pre>' + esc(page['next_command']) + '</pre>'
    if page.get('more'):
        text += '<p>The preview ends here. The complete saved ranking lists the remaining histories.</p>'
    return section('ranked','Ranked histories',text) + SCRIPT


def downloads(root, manifest):
    items = []
    names = {'ranking':'Ranked histories','production_channels':'Discovered production channels','cross_sections':'Production cross sections','decays':'Decay modes','total_widths':'Total widths'}
    for name, formats in (manifest.get('current_tables') or {}).items():
        links = [link(root, info.get('path'), fmt.upper()) for fmt,info in formats.items()]
        if any(links):
            items.append(esc(names.get(name,name)) + ' · ' + ' · '.join(filter(None,links)))
    for filename,label in [('manifest.json','Completion manifest'),('resolved_inputs.json','Resolved model inputs'),('spectrum.json','Spectrum and mixing matrices'),('lagrangian.tex','Generated Lagrangian'),('validation.json','Validation record'),('operator_inputs.json','Coefficient conventions and ordering')]:
        item = link(root,filename,label)
        if item:
            items.append(item)
    if manifest.get('vacuum'):
        items.append(link(root,'vacuum_assumptions.json','Electroweak vacuum assumptions'))
    nodes = ((manifest.get('history_exports') or {}).get('files') or {}).get('history_nodes.json') or {}
    if nodes:
        items.append(link(root,nodes.get('path'),'Complete decay history nodes'))
    if manifest.get('explicit_results_file'):
        items.append(link(root,manifest['explicit_results_file'],'Explicit decay-chain integration results'))
    return section('files','Run files','<ul>' + ''.join('<li>' + item + '</li>' for item in items if item) + '</ul>')


def calculated_histories(manifest):
    return (manifest.get('history_counts') or {}).get('calculated')


def write_report(root, manifest, limit=100, *, page=None):
    root = Path(root).resolve()
    if page is None:
        page = load_saved_page(root,manifest,limit)
    issues = list(page.get('problems', ()))
    parts = []
    for function,anchor,title in [(model_section,'model','Model and spectrum'),(production_section,'production','Production'),(decay_section,'decays','Widths and branching fractions')]:
        try:
            parts.append(function(root,manifest))
        except (ValueError, OSError, KeyError, TypeError, csv.Error) as exc:
            parts.append(section(anchor,title,'<p class="attention">Saved data unavailable. ' + esc(exc) + '</p>'))
    if manifest.get('report_notices'):
        page['notices'] = list(page.get('notices') or ()) + list(manifest['report_notices'])
    elif record(manifest,'decays','json').get('path'):
        try:
            decays = read_json(root,record(manifest,'decays','json')['path'],{})
            page.setdefault('notices', []).extend(decays.get('coverage_notices') or [])
        except (OSError, ValueError, TypeError) as exc:
            issues.append('Decay notices unavailable: ' + str(exc))
    title = manifest.get('model') or 'Model run'
    energy = ((manifest.get('settings') or {}).get('collider') or {}).get('energy_tev')
    status = ('All production rates calculated' if manifest.get('all_rates_complete') else 'Requested selection complete · partial production coverage' if manifest.get('requested_complete') else 'Run ' + str(manifest.get('status','incomplete')).replace('_',' '))
    if manifest.get('status') == 'histories_complete' and not calculated_histories(manifest):
        status = 'History discovery complete · production rates pending'
    head = '<header><div class="meta">Madgraph_step · Physics report</div><h1>' + esc(title) + '</h1><p>' + esc(status) + (' · pp at ' + esc(energy) + ' TeV' if energy is not None else '') + '</p></header>'
    head += '<nav aria-label="Report sections">' + ''.join('<a href="#' + key + '">' + label + '</a>' for key,label in [('model','Model'),('production','Production'),('decays','Decays'),('ranked','Ranked rates'),('methods','Methods'),('files','Downloads')]) + '</nav>'
    for failure in manifest.get('failures') or []:
        issues.append(' · '.join(str(failure[k]) for k in ('stage','channel','category','message') if failure.get(k)))
    if issues:
        head += '<div class="attention" role="status"><strong>Run notes</strong><ul>' + ''.join('<li>' + esc(item) + '</li>' for item in issues) + '</ul></div>'
    download = link(root,record(manifest,'ranking').get('path'),'Download ranked rates (CSV)',True)
    body = head + ''.join(parts) + ranked(page,download) + methods_section(manifest,page) + downloads(root,manifest)
    destination = contained_path(root,'report.html')
    atomic_text(destination,document(title + ' · Physics report',body))
    return destination


def refresh_report(root, limit=100):
    root = Path(root).expanduser().resolve()
    if not contained_path(root,'manifest.json').is_file():
        raise ValueError(f'No saved run manifest in {root}')
    with RunStore(root).lock():
        manifest = read_json(root,'manifest.json')
        if not isinstance(manifest,dict) or manifest.get('schema') != 'madgraph-step-run-v1':
            raise ValueError('Unrecognized saved run manifest')
        return write_report(root,manifest,limit)
