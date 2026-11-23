"""Complete machine-readable tables and a standalone human-readable run report."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import fields, is_dataclass
import html
import itertools
import json
import os
from pathlib import Path
import shlex
import uuid

from pipeline_storage import JsonArray, atomic_text, public, write_json, write_csv, file_digest, digest
from workflow_errors import error_category


def _vacuum_html(assumptions):
    if assumptions is None:
        return ''
    return ('<h2>Electroweak vacuum</h2><p>The model assumes an unmixed 125 GeV Higgs '
            'and zero VEVs for all new scalars. Potential minimization and global stability analysis '
            'were skipped.</p>')


def _table(rows, keys):
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        return '<p>No rows available.</p>'
    def cell(value):
        if value is None:
            return ''
        if isinstance(value, (list, tuple)):
            value = '; '.join(map(str, value))
        return html.escape(str(value))
    return ('<div class="table"><table><thead><tr>' + ''.join('<th>' + cell(key) + '</th>' for key in keys)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell(row.get(key, '')) + '</td>' for key in keys) + '</tr>' for row in itertools.chain((first,), iterator))
            + '</tbody></table></div>')


RANKING_COLUMNS = ('history_id', 'rate_status', 'production_id', 'production_process',
    'production_final_pdgs', 'final_pdgs', 'final_state', 'history', 'history_key',
    'branching_ratio', 'production_cross_section_pb', 'production_error_pb', 'rate_pb',
    'error_pb', 'flags', 'history_ref', 'render_truncated')
_DISPLAY_COLUMNS = ('history_id', 'rate_status', 'production_id', 'production_process',
    'history', 'final_state', 'production_cross_section_pb', 'production_error_pb',
    'branching_ratio', 'rate_pb', 'error_pb', 'flags')
_TERMINAL_EXPORT = frozenset(('histories_complete', 'complete', 'partial', 'export'))


def _metadata(ranking):
    names = (f.name for f in fields(ranking)) if is_dataclass(ranking) else (
        'coverage', 'omitted_probability', 'notices', 'uncomputed_production_ids', 'calculation_limits')
    return {name: getattr(ranking, name) for name in names
            if name != 'rows' and not name.startswith('_')}


def _reference(ranking, root=None, *, checksum=False):
    rows = ranking.rows
    if hasattr(rows, 'reference'):
        if type(rows).__name__ == 'SQLiteHistoryRows':
            return public(rows.reference(root=root, checksum=checksum))
        return public(rows.reference())
    return dict(schema='in-memory-histories-v1', fingerprint=digest(rows), revision=0)


def _reference_identity(reference):
    """A copied, checksum-verified snapshot keeps its export identity."""
    return {key: value for key, value in reference.items() if key != 'file_stat'}


def _counts(ranking):
    if hasattr(ranking.rows, 'counts'):
        return ranking.rows.counts()
    return dict(total=len(ranking.rows), calculated=sum(row.get('rate_pb') is not None for row in ranking.rows),
                discovery_complete=not ranking.calculation_limits.get('incomplete', False), revision=0)


def _normalized_rows(ranking):
    rows = ranking.rows.iter_rows() if hasattr(ranking.rows, 'iter_rows') else iter(ranking.rows)
    for row in rows:
        unknown = set(row) - set(RANKING_COLUMNS)
        if unknown:
            raise ValueError('Unrecognized ranking export fields: ' + ', '.join(sorted(unknown)))
        yield {key: row.get(key) for key in RANKING_COLUMNS}


def _progress_rows(rows, progress):
    """Count an existing streaming pass without retaining its rows."""
    count = 0
    for row in rows:
        yield row
        count += 1
        if count % 1000 == 0:
            progress.update(count)
    progress.update(count)


def _file_stat(path):
    stat = path.stat()
    return dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _safe_export_path(root, relative):
    from run_paths import contained_path
    return contained_path(root, relative)


def _existing_export(root, identity):
    path = _safe_export_path(root, 'ranking_exports.json')
    try:
        value = json.loads(path.read_text())
        if (value.get('schema') != 'history-exports-v1' or value.get('identity') != identity
                or value.get('complete') is not True
                or set(value['files']) != {'ranking.json', 'ranking.csv', 'history_nodes.json'}):
            return None
        for record in value['files'].values():
            if not _verified_export_file(root, record):
                return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _verified_export_file(root, record):
    try:
        file = _safe_export_path(root, record['path'])
        stat = _file_stat(file)
        return (stat['size_bytes'] == record['size_bytes']
                and (stat == record.get('stat') or file_digest(file) == record['sha256']))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _link_exports(root, result):
    for name, record in result['files'].items():
        source = _safe_export_path(root, record['path'])
        _safe_export_path(root, name)
        # Validate any link, then atomically replace the compatibility name
        # itself. Replacing its resolved target could overwrite another file.
        destination = root / name
        if destination.exists() and os.path.samefile(source, destination):
            continue
        temporary = root / ('.' + name + '.link-' + uuid.uuid4().hex)
        try:
            os.link(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    for record in result['files'].values():
        record['stat'] = _file_stat(_safe_export_path(root, record['path']))


def export_ranking(project):
    """Commit complete row/node exports without copying the indexed catalogue.

    Bundle files are immutable after their commit. Root compatibility names are
    hardlinks; the manifest points directly to the verified bundle, so an error
    replacing a compatibility link cannot advertise a half-written file pair.
    """
    from workflow_progress import stage
    root, ranking = Path(project.run_directory).resolve(), project.ranking
    with stage('Checking history snapshot for export'):
        reference, metadata = _reference(ranking, root, checksum=True), _metadata(ranking)
    if hasattr(project, '_retain_history_reference'):
        project._retain_history_reference(reference)
    identity = digest(dict(reference=_reference_identity(reference), metadata=metadata, columns=RANKING_COLUMNS))
    with stage('Checking saved history exports'):
        existing = _existing_export(root, identity)
    if existing is not None:
        _link_exports(root, existing)
        write_json(root / 'ranking_exports.json', existing)
        return existing
    write_json(root / 'export_state.json', dict(status='exporting', complete=False,
                                               identity=identity, source=reference))
    folder = _safe_export_path(root, Path('exports') / identity[:24])
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    progress_path = _safe_export_path(root, folder.relative_to(root) / 'progress.json')
    try:
        progress = json.loads(progress_path.read_text())
        if (progress.get('identity') == identity and progress.get('source') == reference
                and isinstance(progress.get('files'), dict)):
            files = {name: record for name, record in progress['files'].items()
                     if name in {'ranking.json', 'ranking.csv', 'history_nodes.json'}}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    total = _counts(ranking)['total']
    specs = (
        ('ranking.json', total, lambda path, progress: write_json(path, {**metadata, 'rows': JsonArray(
            _progress_rows(_normalized_rows(ranking), progress), native=True)})),
        ('ranking.csv', total, lambda path, progress: write_csv(path,
            _progress_rows(_normalized_rows(ranking), progress), RANKING_COLUMNS, fixed=True, native=True)),
        ('history_nodes.json', None, lambda path, progress: write_json(path, dict(source=reference, nodes=JsonArray(
            _progress_rows(ranking.rows.iter_nodes() if hasattr(ranking.rows, 'iter_nodes') else (), progress), native=True)))),
    )
    for name, total, writer in specs:
        path = _safe_export_path(root, folder.relative_to(root) / name)
        relative = str(path.relative_to(root))
        previous = files.get(name)
        with stage('Checking saved ' + name):
            reusable = (isinstance(previous, dict) and previous.get('path') == relative
                        and _verified_export_file(root, previous))
        if not reusable:
            with stage('Saving ' + name, total=total, unit='rows') as progress:
                files[name] = dict(path=relative, **writer(path, progress), stat=_file_stat(path))
            write_json(progress_path, dict(identity=identity, source=reference, files=files))
    if _reference(ranking, root) != reference:
        raise RuntimeError('History source revision changed during export; retry the snapshot')
    result = dict(schema='history-exports-v1', identity=identity, source=reference,
                  counts=_counts(ranking), files=files, complete=True)
    write_json(folder / 'manifest.json', result)
    _link_exports(root, result)
    write_json(root / 'ranking_exports.json', result)
    write_json(root / 'export_state.json', dict(status='complete', complete=True,
                                               identity=identity, source=reference))
    return result


def _page(ranking, limit=100, min_relative_rate=0.):
    if hasattr(ranking.rows, 'page'):
        return ranking.rows.page(limit=limit, min_relative_rate=min_relative_rate, include_uncalculated=True)
    maximum = max((row.get('rate_pb') or 0. for row in ranking.rows), default=0.)
    selected = (row for row in ranking.rows if row.get('rate_pb') is None
                or row['rate_pb'] >= min_relative_rate * maximum)
    return dict(rows=list(itertools.islice(selected, limit)), next_cursor=None, revision=0, total=len(ranking.rows))


def _next_command(project, page):
    if not page.get('next_cursor'):
        return ''
    command = [str(Path(__file__).resolve().parent / 'run.sh'), 'histories',
               str(project.config.model_file), '--cursor', str(page['next_cursor']),
               '--page-size', str(page.get('page_size', len(page['rows']) or 100)), '--report-page']
    if getattr(project.config, 'output', None) is not None:
        command.extend(('--output', str(project.config.output)))
    config_path = getattr(project.config, 'path', None)
    if config_path is not None and Path(config_path).is_file():
        command.extend(('--config', str(config_path)))
    return shlex.join(command)


def _page_html(project, page):
    command = _next_command(project, page)
    return (f'<p>Source revision {html.escape(str(page["revision"]))}; '
            f'{len(page["rows"])} rows shown from {page["total"]} histories. '
            'Uncalculated rates remain empty; they are not zero.</p>'
            + _table(page['rows'], _DISPLAY_COLUMNS)
            + ('<p>Render the next page with:</p><pre>' + html.escape(command) + '</pre>' if command else '')
            + '<p>Large histories use compact DAG references. The complete node data are in history_nodes.json.</p>')


def write_history_page(project, page):
    """Render an already bounded, revision-bound SQL page; never scan all rows."""
    if not isinstance(page.get('rows'), (list, tuple)) or len(page['rows']) > 1000:
        raise ValueError('History pages must contain at most 1000 rows')
    current = _reference(project.ranking, project.run_directory)
    if current.get('revision') != page.get('revision'):
        raise ValueError('History page source revision is stale; request a fresh first page')
    if page.get('fingerprint') is not None and page['fingerprint'] != current.get('fingerprint'):
        raise ValueError('History page source fingerprint differs; request a fresh first page')
    identity = digest(dict(revision=page['revision'], rows=[row.get('history_id') for row in page['rows']],
                           next_cursor=page.get('next_cursor')))
    destination = _safe_export_path(project.run_directory, Path('history-pages') / (identity[:24] + '.html'))
    atomic_text(destination, '<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>Ranked history page</title><style>body{font:16px system-ui;margin:2em}'
                'table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.4em}</style>'
                '<h1>Ranked histories</h1>'
                + _vacuum_html(getattr(project, 'vacuum_assumptions', None))
                + _page_html(project, page) + '</html>\n')
    return destination


def publish(project, *, status, selected='all', error=None):
    root = Path(project.run_directory).resolve()
    channels, decays, rates, ranking = project._channels, project._decays, project.cross_sections, project.ranking
    tables = []
    exported = None
    history_reference = None
    history_counts = None
    # A new attempt may fail before producing replacements. Retain prior tables
    # in a separate snapshot, so the current manifest never accompanies stale
    # top-level ranking/rate files from a previous request.
    absent = []
    for result, names in ((channels, ('production_channels',)),
                          (decays, ('decays', 'total_widths')),
                          (rates, ('cross_sections',)), (ranking, ('ranking',))):
        if result is None:
            for name in names:
                for suffix in ('.json', '.csv'):
                    _safe_export_path(root, name + suffix)
                    # Archive the compatibility name, not the target of an
                    # internal symlink that may also name a physical artifact.
                    absent.append(root / (name + suffix))
    old_files = [path for path in absent if path.exists()]
    if old_files:
        # Renaming a compatibility hardlink changes the bundle inode's ctime
        # on macOS. Record only this known metadata operation so a fresh page
        # request need not rehash gigabytes. A preexisting stat mismatch is
        # deliberately left untouched for the checksum validator to inspect.
        try:
            previous_export = json.loads(_safe_export_path(root, 'ranking_exports.json').read_text())
        except (OSError, ValueError):
            previous_export = {}
        records = previous_export.get('files', {}) if isinstance(previous_export, dict) else {}
        if not isinstance(records, dict):
            records = {}
        touched_export = False
        archive = _safe_export_path(root, Path('previous_results') / uuid.uuid4().hex[:12])
        archive.mkdir(parents=True, exist_ok=False)
        for path in old_files:
            record = records.get(path.name)
            trusted_source = None
            if isinstance(record, dict):
                try:
                    source = _safe_export_path(root, record['path'])
                    if os.path.samefile(source, path) and _file_stat(source) == record.get('stat'):
                        trusted_source = source
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            path.replace(archive / path.name)
            if trusted_source is not None:
                record['stat'] = _file_stat(trusted_source)
                touched_export = True
        if touched_export:
            write_json(root / 'ranking_exports.json', previous_export)
    if channels is not None:
        write_json(root / 'production_channels.json', JsonArray(iter(channels)))
        write_csv(root / 'production_channels.csv', channels, ('id', 'process', 'final_pdgs', 'diagrams'), fixed=True)
        tables.append(('Production channels', 'production_channels'))
    if decays is not None:
        write_json(root / 'decays.json', decays)
        write_csv(root / 'decays.csv', decays.modes, ('pdg', 'process', 'daughters', 'width_gev', 'branching_ratio'))
        write_json(root / 'total_widths.json', decays.total_width_rows())
        write_csv(root / 'total_widths.csv', decays.total_width_rows(), ('pdg', 'name', 'width_gev', 'omitted_probability'))
        tables.extend((('Decay modes', 'decays'), ('Total widths', 'total_widths')))
    if rates is not None:
        write_json(root / 'cross_sections.json', rates)
        write_csv(root / 'cross_sections.csv', rates.rates, ('id', 'process', 'cross_section_pb', 'error_pb', 'target_met'))
        tables.append(('Cross sections', 'cross_sections'))
    if ranking is not None:
        history_reference, history_counts = _reference(ranking, root), _counts(ranking)
        identity = digest(dict(reference=_reference_identity(history_reference), metadata=_metadata(ranking), columns=RANKING_COLUMNS))
        if status in _TERMINAL_EXPORT:
            project.export_complete = False
            exported = export_ranking(project)
            history_reference = _reference(ranking, root)
        else:
            exported = _existing_export(root, identity)
        project.export_complete = exported is not None
        if exported is not None:
            tables.append(('All ranked histories', 'ranking'))
    else:
        project.export_complete = False
    if hasattr(project, '_update_completion'):
        project._update_completion()
    explicit = getattr(project, 'explicit_decay_results', None)
    if explicit is not None:
        write_json(root / 'explicit_decay_results.json', explicit)
    failures = list(project.failures)
    if error is not None and not any(row.get('message') == str(error) for row in failures):
        failures.append(dict(stage='pipeline', type=type(error).__name__, category=error_category(error), message=str(error)))
    all_ids = {c.id for c in channels} if channels is not None else set()
    calculated = {row['id'] for row in rates.rates} if rates is not None else set()
    current_tables = {name: {suffix[1:]: dict(path=name + suffix, sha256=file_digest(root / (name + suffix)))
                            for suffix in ('.json', '.csv')} for _, name in tables if name != 'ranking'}
    if exported is not None:
        current_tables['ranking'] = {kind: exported['files']['ranking.' + kind] for kind in ('json', 'csv')}
    vacuum = getattr(project, 'vacuum_assumptions', None)
    manifest = dict(schema='madgraph-step-run-v1', status=status, complete=project.complete,
                    requested_complete=project.requested_complete, updated_at=datetime.now(timezone.utc).isoformat(),
                    model=project.inputs.get('model_name'), run_id=root.name, input_file=str(project.config.model_file),
                    requested_scope=selected, physics_fingerprint=project._signature,
                    stage_fingerprints=project._stage_keys, failures=failures,
                    vacuum_status=(vacuum or {}).get('status'), vacuum=vacuum,
                    uncomputed_production_ids=sorted(all_ids-calculated),
                    cache_rejections=project._store.cache_rejections,
                    settings=dict(collider=project.config.collider, decays=project.config.decays,
                                  ranking=project.config.ranking, explicit_decays=getattr(project.config, 'explicit_decays', {})),
                    artifact_schema='madgraph-step-artifact-v1',
                    current_tables=current_tables,
                    history_store=history_reference, history_counts=history_counts,
                    history_exports=exported, export_complete=project.export_complete,
                    discovery_complete=getattr(project, 'discovery_complete', bool(history_counts and history_counts.get('discovery_complete'))),
                    requested_rates_complete=getattr(project, 'requested_rates_complete', project.requested_complete),
                    all_rates_complete=getattr(project, 'all_rates_complete', project.complete),
                    selected_integrations_complete=getattr(project, 'selected_integrations_complete', False),
                    previous_results='previous_results/' if _safe_export_path(root, 'previous_results').is_dir() else None)
    from report_view import write_report
    if explicit is not None:
        manifest['explicit_results_file'] = 'explicit_decay_results.json'
    if decays is not None:
        manifest['report_notices'] = list(decays.coverage_notices)
    write_json(root / 'manifest.json', manifest)
    live_page = None
    if ranking is not None and hasattr(ranking.rows, 'page'):
        from history_store import SQLiteHistoryRows
        if not isinstance(ranking.rows, SQLiteHistoryRows):
            # Third-party page providers can expose live rows before an export exists.
            live_page = _page(ranking, min_relative_rate=project.config.ranking.get('min_relative_rate', 0.))
            live_page['next_command'] = _next_command(project, live_page)
            live_page['calculated'] = (history_counts or {}).get('calculated')
            live_page['notices'] = list(ranking.notices)
            live_page['coverage'] = {key:(v,ranking.omitted_probability.get(key)) for key,v in ranking.coverage.items()}
    write_report(root, manifest, page=live_page)
    return manifest
