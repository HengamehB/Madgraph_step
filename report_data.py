"""Read a bounded report preview from artifacts in the selected run directory."""
from contextlib import closing
import csv
from itertools import islice
import json
import math
from pathlib import Path
import sqlite3
from urllib.parse import quote

from run_paths import contained_path, portable_history_reference


def read_json(root, path, default=None):
    target = contained_path(root, path)
    if not target.is_file():
        return default
    with target.open(encoding='utf-8') as stream:
        return json.load(stream)


def href(root, path):
    if not path:
        return None
    target = contained_path(root, path)
    if target.is_file():
        return quote(target.relative_to(Path(root).resolve()).as_posix(), safe='/')
    return None


def record(manifest, name, kind='csv'):
    return (manifest.get('current_tables') or {}).get(name, {}).get(kind) or {}


def table_rows(root, manifest, name):
    artifact = record(manifest, name)
    if not artifact.get('path'):
        return
    path = contained_path(root, artifact['path'])
    if artifact.get('size_bytes') is not None and path.stat().st_size != artifact['size_bytes']:
        raise ValueError(f'Saved {name} table differs in size from its manifest')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if name == 'ranking' and not {'history_id','rate_pb','branching_ratio','final_state'}.issubset(reader.fieldnames or ()):
            raise ValueError('The current ranking CSV lacks its required columns')
        yield from reader


def history_row(raw):
    result = dict(raw)
    for key in ('rate_pb','error_pb','production_cross_section_pb','production_error_pb','branching_ratio'):
        value = raw.get(key)
        result[key] = None if value in ('', None) else float(value)
        if result[key] is not None and not math.isfinite(result[key]):
            raise ValueError(f'Non-finite value in ranking column {key}')
    for key in ('final_pdgs','flags','history_ref'):
        result[key] = json.loads(raw[key]) if raw.get(key) else None
    return result


def index_page(root, ref, page, limit, need_rows):
    from history_store import SCHEMA, SQLiteHistoryRows, _connect, _get
    path = portable_history_reference(ref, root, absolute=True)['path']
    stat = path.stat()
    saved = ref.get('file_stat') or {}
    if (stat.st_size, stat.st_mtime_ns) != (saved.get('size_bytes'), saved.get('mtime_ns')):
        raise ValueError('History file metadata differs from the saved snapshot')
    rows = SQLiteHistoryRows(path)
    with closing(_connect(path, True, rows.memory_mb)) as db:
        db.execute('BEGIN')
        if (ref.get('schema') != SCHEMA or _get(db, 'schema') != SCHEMA or
                _get(db, 'fingerprint') != ref.get('fingerprint') or _get(db, 'revision') != ref.get('revision')):
            raise ValueError('Saved history revision or fingerprint differs from the manifest')
        page['notices'] = _get(db, 'notices', [])
        for item in db.execute('SELECT display_id,coverage,complete FROM productions ORDER BY display_id'):
            value = item['coverage']
            page['coverage'][item['display_id']] = (value, max(0., 1. - value) if item['complete'] else None)
        if not need_rows:
            return
        query = rows._SELECT + ' ORDER BY h.rate DESC,h.history_id LIMIT ?'
        plan = [row[3].upper() for row in db.execute('EXPLAIN QUERY PLAN ' + query, (limit + 1,))]
        if not any('HISTORY_RANK_COVERING' in line for line in plan) or any('TEMP B-TREE' in line for line in plan):
            raise ValueError('The saved history ranking index is not ready')
        chunk = db.execute(query, (limit + 1,)).fetchall()
        decode = rows._decoder(db)
        page.update(rows=[decode(row) for row in chunk[:limit]], more=len(chunk) > limit, source='history index')


def load_saved_page(root, manifest, limit=100):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('Page size must be between 1 and 1000')
    counts = manifest.get('history_counts') or {}
    page = dict(rows=[], total=counts.get('total'), calculated=counts.get('calculated'), more=False,
                notices=[], coverage={}, source=None, problems=[])
    if record(manifest, 'ranking'):
        previous = csv.field_size_limit()
        try:
            csv.field_size_limit(max(previous, 16 * 1024 * 1024))
            with closing(table_rows(root, manifest, 'ranking')) as rows:
                prefix = [history_row(row) for row in islice(rows, limit + 1)]
            page.update(rows=prefix[:limit], more=len(prefix) > limit, source='ranking CSV')
        except (OSError, csv.Error, ValueError) as exc:
            page['problems'].append(f'Ranking CSV unavailable: {exc}')
        finally:
            csv.field_size_limit(previous)
    reference = manifest.get('history_store') or {}
    if reference.get('path'):
        try:
            index_page(root, reference, page, limit, not page['source'])
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            page['problems'].append(f'History index unavailable: {exc}')
    return page
