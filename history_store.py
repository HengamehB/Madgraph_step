"""Disk-backed, resumable decay forests. No history-sized Python collection.

Only primary modes, one combination, and bounded node/page caches are resident.
Repeated subtrees are stored as DAG nodes; rates are attached independently.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import base64
import errno
import hashlib
import itertools
import json
import math
import sqlite3
import tempfile

from workflow_progress import format_count, message, stage

SCHEMA = 'madgraph-history-store-v2'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _connect(path, readonly=False, memory_mb=64):
    # SQLite can open journals even when the main database path was checked.
    # Inspect these local links before handing any path to the SQLite library.
    from run_paths import contained_path
    path = Path(path)
    for suffix in ('-journal', '-wal', '-shm'):
        contained_path(path.parent, path.name + suffix)
    db = sqlite3.connect(Path(path).resolve().as_uri() + ('?mode=ro' if readonly else '?mode=rwc'), uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA mmap_size=0')
    db.execute(f'PRAGMA cache_size=-{max(256, int(memory_mb) * 1024 // 4)}')
    if not readonly:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
    return db


def _put(db, key, value):
    encoded = _json(value)
    old = db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    if old is None or old[0] != encoded:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, encoded))


def _file_stat(path):
    stat = Path(path).stat()
    return dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)


def _get(db, key, default=None):
    row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def _revision(db):
    revision = _get(db, 'revision', 0) + 1
    _put(db, 'revision', revision)
    return revision


def replacement_indices(n, count, start=0):
    """Seekable combinations-with-replacement without tuple(range(n))."""
    total = math.comb(n + count - 1, count) if n else 0
    if not 0 <= start < total:
        return
    rank, minimum, remaining, indices = start, 0, count, []
    while remaining:
        base = math.comb(n - minimum + remaining - 1, remaining)
        lo, hi = minimum, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            skipped = base - math.comb(n - mid + remaining - 1, remaining)
            if skipped <= rank:
                lo = mid
            else:
                hi = mid - 1
        rank -= base - math.comb(n - lo + remaining - 1, remaining)
        indices.append(lo)
        minimum, remaining = lo, remaining - 1
    while True:
        yield tuple(indices)
        for i in range(count - 1, -1, -1):
            if indices[i] < n - 1:
                value = indices[i] + 1
                indices[i:] = [value] * (count - i)
                break
        else:
            return


class HistoryStoreError(RuntimeError):
    category = 'resource_limit'


class SQLiteHistoryRows:
    """Replayable row sequence; materialization is bounded to one page/tree."""

    def __init__(self, path, *, owner=None):
        self.path = Path(path).resolve()
        self._owner = owner
        self._checksum_cache = None
        with closing(_connect(self.path, True)) as db:
            if _get(db, 'schema') != SCHEMA:
                raise ValueError('Unknown history database schema')
            self.fingerprint = _get(db, 'fingerprint')
            self.memory_mb = _get(db, 'memory_mb', 64)
            self.max_render_nodes = _get(db, 'max_render_nodes', 10000)

    def __len__(self):
        return self.counts()['total']

    def close(self):
        owner, self._owner = self._owner, None
        if owner is not None:
            owner.cleanup()

    def __del__(self):
        if getattr(self, '_owner', None) is not None:
            self.close()

    def counts(self):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            row = db.execute("SELECT COALESCE(SUM(row_count),0),COALESCE(SUM(CASE WHEN status='computed' THEN row_count ELSE 0 END),0),COALESCE(MIN(complete),1) FROM productions").fetchone()
            return dict(total=row[0], calculated=row[1], discovery_complete=bool(row[2]) and bool(_get(db, 'discovery_complete', False)),
                        revision=_get(db, 'revision'), fingerprint=self.fingerprint)

    def reference(self, *, root=None, checksum=False):
        """Describe a snapshot; terminal callers may request a streamed checksum."""
        from run_paths import contained_path, portable_history_reference
        from pipeline_storage import file_digest
        stat = _file_stat(self.path)
        revision = self.counts()['revision']
        if self._checksum_cache is None or self._checksum_cache[0] != stat:
            self._checksum_cache = None
            try:
                sidecar = contained_path(self.path.parent, 'history_reference.json')
                previous = json.loads(sidecar.read_text())
                if (isinstance(previous, dict) and previous.get('file_stat') == stat and previous.get('schema') == SCHEMA
                        and previous.get('fingerprint') == self.fingerprint
                        and previous.get('revision') == revision and previous.get('sha256')):
                    self._checksum_cache = (stat, previous['sha256'])
            except (OSError, ValueError, TypeError):
                pass
        if checksum and self._checksum_cache is None:
            value = file_digest(self.path)
            if _file_stat(self.path) != stat:
                raise RuntimeError('History database changed while checksumming its snapshot')
            self._checksum_cache = (stat, value)
        result = dict(path=self.path, schema=SCHEMA, fingerprint=self.fingerprint,
                      revision=revision, file_stat=stat)
        if self._checksum_cache is not None:
            result['sha256'] = self._checksum_cache[1]
        return portable_history_reference(result, root) if root is not None else result

    @classmethod
    def from_reference(cls, reference, *, root=None):
        if root is not None:
            from run_paths import portable_history_reference
            reference = portable_history_reference(reference, root, absolute=True)
        result = cls(reference['path'])
        if reference.get('schema') != SCHEMA or result.fingerprint != reference['fingerprint']:
            raise ValueError('History checkpoint fingerprint differs')
        if result.counts()['revision'] != reference['revision']:
            raise ValueError('History checkpoint revision differs; refresh the catalogue')
        current_stat = _file_stat(result.path)
        if current_stat != reference.get('file_stat'):
            from pipeline_storage import file_digest
            if not reference.get('sha256') or file_digest(result.path) != reference['sha256']:
                raise ValueError('History database changed outside its recorded revision; refresh the catalogue')
            if _file_stat(result.path) != current_stat:
                raise ValueError('History database changed during snapshot verification')
            result._checksum_cache = (current_stat, reference['sha256'])
        return result

    def summary(self):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            coverage, omitted, missing = {}, {}, []
            incomplete = []
            for row in db.execute('SELECT display_id,coverage,status,complete FROM productions ORDER BY display_id'):
                coverage[row[0]] = row[1]
                # An interrupted prefix cannot measure omitted physical probability.
                omitted[row[0]] = max(0., 1. - row[1]) if row[3] else None
                if not row[3]:
                    incomplete.append(row[0])
                if row[2] != 'computed':
                    missing.append(row[0])
            return dict(coverage=coverage, omitted_probability=omitted,
                        uncomputed_production_ids=tuple(missing),
                        notices=tuple(_get(db, 'notices', [])),
                        calculation_limits=dict(_get(db, 'limits', {}),
                                                incomplete=not self.counts()['discovery_complete'],
                                                discovery_incomplete_production_ids=incomplete))

    def production_statuses(self):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            return {row['display_id']: dict(status=row['status'],
                    error=json.loads(row['failure']) if row['failure'] else None)
                    for row in db.execute('SELECT display_id,status,failure FROM productions')}

    def iter_nodes(self):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            for row in db.execute('SELECT * FROM nodes ORDER BY node_id'):
                yield dict(node_id=row['node_id'], key=row['node_key'], pdg=row['pdg'],
                           children=json.loads(row['children']), branching_ratio=row['probability'],
                           final_counts=json.loads(row['final_counts']), flags=json.loads(row['flags']))

    def _decoder(self, db):
        particles = _get(db, 'particles')
        render_limit = _get(db, 'max_render_nodes', 10000)
        def name(pdg):
            return particles[str(pdg)].get('name', str(pdg))

        @lru_cache(maxsize=max(16, min(4096, self.memory_mb * 16)))
        def node(node_id):
            row = db.execute('SELECT * FROM nodes WHERE node_id=?', (node_id,)).fetchone()
            if row is None:
                raise ValueError('History references a missing DAG node')
            return dict(pdg=row['pdg'], children=json.loads(row['children']),
                        final_counts=json.loads(row['final_counts']), flags=json.loads(row['flags']))

        # Millions of forests reuse a small number of subtrees. Cache rendered
        # trees with a node-count budget, not an unbounded entry count or the
        # product of max_render_nodes and the number of histories.
        tree_cache, cache_nodes = OrderedDict(), 0
        cache_budget = max(1, self.memory_mb * 128)
        def rendered(root):
            nonlocal cache_nodes
            if root in tree_cache:
                value = tree_cache.pop(root)
                tree_cache[root] = value
                return value
            item = node(root)
            children, size = [], 1
            for child in item['children']:
                value = rendered(child)
                size += value[2]
                if size > render_limit:
                    raise OverflowError
                children.append(value)
            children.sort(key=lambda value: value[0])
            tree = (item['pdg'],tuple(value[0] for value in children))
            label = name(item['pdg'])
            text = label if not children else label + ' → (' + ' '.join(value[1] for value in children) + ')'
            value = tree,text,size
            if size <= cache_budget:
                while tree_cache and cache_nodes + size > cache_budget:
                    cache_nodes -= tree_cache.popitem(last=False)[1][2]
                tree_cache[root] = value
                cache_nodes += size
            return value

        def decode(row):
            roots = json.loads(row['roots'])
            final, flags = Counter(), set()
            for root in roots:
                item = node(root)
                final.update({int(p): n for p, n in item['final_counts']})
                flags.update(item['flags'])
            flags.update(json.loads(row['production_flags']))
            if row['target_met'] == 0:
                flags.add('Production MC precision target not met')
            truncated = False
            try:
                trees, size = [], 0
                for root in roots:
                    value = rendered(root)
                    size += value[2]
                    if size > render_limit:
                        raise OverflowError
                    trees.append(value)
                trees.sort(key=lambda value: value[0])
                forest = tuple(value[0] for value in trees)
                history = '; '.join(value[1] for value in trees)
            except OverflowError:
                forest, truncated = None, True
                history = 'Full decay DAG: ' + ', '.join(map(str, roots)) + ' (see history_nodes.json)'
            # A terminal multiset is compact even if the full tree is huge.
            total_leaves = sum(final.values())
            final_pdgs = tuple(sorted(final.elements())) if total_leaves <= render_limit else None
            final_state = (' '.join(name(p) for p in final_pdgs) if final_pdgs is not None else
                           ' '.join(f'{name(p)}×{n}' for p, n in sorted(final.items())))
            return dict(history_id=row['history_id'], production_id=row['display_id'],
                        production_process=row['process'], production_final_pdgs=tuple(json.loads(row['final_pdgs'])),
                        final_pdgs=final_pdgs, final_state=final_state, history=history, history_key=forest,
                        history_ref=roots, render_truncated=truncated,
                        branching_ratio=row['probability'], production_cross_section_pb=row['sigma'],
                        production_error_pb=row['sigma_error'], rate_pb=row['rate'],
                        error_pb=None if row['sigma_error'] is None else row['sigma_error'] * row['probability'],
                        rate_status=row['status'], flags=tuple(sorted(flags)))
        return decode

    _SELECT = '''SELECT h.*, p.display_id,p.process,p.final_pdgs,p.sigma,p.sigma_error,p.status,p.target_met,
                      p.flags AS production_flags FROM histories h JOIN productions p ON p.pid=h.pid'''

    def iter_rows(self):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            db.execute('BEGIN')
            decode = self._decoder(db)
            for row in db.execute(self._SELECT + ' ORDER BY h.rate DESC,h.history_id'):
                yield decode(row)

    def __iter__(self):
        return self.iter_rows()

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if len(range(start, stop, step)) > 1000:
                raise ValueError('Use iter_rows() or page() instead of materializing more than 1000 histories')
            return tuple(itertools.islice(self.iter_rows(), start, stop, step))
        if not isinstance(index, int):
            raise TypeError('Use get(history_id) to select a history by ID')
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            row = db.execute(self._SELECT + ' ORDER BY h.rate DESC,h.history_id LIMIT 1 OFFSET ?', (index,)).fetchone()
            if row is None:
                raise IndexError(index)
            return self._decoder(db)(row)

    def get(self, history_id):
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            row = db.execute(self._SELECT + ' WHERE h.history_id=?', (history_id,)).fetchone()
            if row is None:
                raise KeyError(f'History is absent from this calculation: {history_id}')
            return self._decoder(db)(row)

    def page(self, limit=100, cursor=None, min_relative_rate=0., include_uncalculated=True):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Page size must be between 1 and 1000')
        if not 0 <= min_relative_rate <= 1:
            raise ValueError('Relative cutoff must be between zero and one')
        with closing(_connect(self.path, True, self.memory_mb)) as db:
            db.execute('BEGIN')
            revision = _get(db, 'revision')
            file_stat = _file_stat(self.path)
            filters = [float(min_relative_rate), bool(include_uncalculated)]
            clauses, args = [], []
            if min_relative_rate:
                maximum = db.execute('SELECT MAX(rate) FROM histories').fetchone()[0]
                if maximum is not None:
                    clauses.append('(h.rate>=?' + (' OR h.rate IS NULL)' if include_uncalculated else ')'))
                    args.append(maximum * min_relative_rate)
            if not include_uncalculated:
                clauses.append('h.rate IS NOT NULL')
            if cursor:
                try:
                    token = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                except (ValueError, TypeError, UnicodeError):
                    raise ValueError('Invalid history cursor') from None
                if not isinstance(token, list) or len(token) != 6 or token[:4] != [self.fingerprint, revision, filters, file_stat]:
                    raise ValueError('History cursor belongs to another revision or filter; start a new page')
                rate, history_id = token[4:]
                if rate is None:
                    clauses.append('(h.rate IS NULL AND h.history_id>?)')
                    args.append(history_id)
                else:
                    clauses.append('(h.rate<? OR (h.rate=? AND h.history_id>?) OR h.rate IS NULL)')
                    args.extend((rate, rate, history_id))
            query = self._SELECT + (' WHERE ' + ' AND '.join(clauses) if clauses else '') + ' ORDER BY h.rate DESC,h.history_id LIMIT ?'
            records = db.execute(query, (*args, limit + 1)).fetchmany(limit + 1)
            more = len(records) > limit
            records = records[:limit]
            decode = self._decoder(db)
            next_cursor = None
            if more:
                last = records[-1]
                token = [self.fingerprint, revision, filters, file_stat, last['rate'], last['history_id']]
                next_cursor = base64.urlsafe_b64encode(_json(token).encode()).decode()
            return dict(rows=[decode(row) for row in records], next_cursor=next_cursor,
                        revision=revision, total=db.execute('SELECT SUM(row_count) FROM productions').fetchone()[0] or 0,
                        fingerprint=self.fingerprint, page_size=limit, cursor=cursor,
                        min_relative_rate=min_relative_rate, include_uncalculated=include_uncalculated)

    def update_rates(self, results=None, statuses=None):
        rates = (results.get('rates', ()) if isinstance(results, dict) else
                 getattr(results, 'rates', results)) if results is not None else ()
        with closing(_connect(self.path, False, self.memory_mb)) as db, db:
            changed = False
            for rate in rates:
                production = db.execute('SELECT * FROM productions WHERE display_id=?', (str(rate['id']),)).fetchone()
                if production is None:
                    raise ValueError('Cross section refers to an unknown production channel')
                if 'final_pdgs' in rate and sorted(rate['final_pdgs']) != json.loads(production['final_pdgs']):
                    raise ValueError('Cross-section final state disagrees with production')
                sigma, error = rate.get('cross_section_pb'), rate.get('error_pb')
                if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0 for v in (sigma,error)):
                    raise ValueError('Calculated cross section and error must be finite nonnegative numbers')
                target = int(rate.get('target_met', True))
                if (production['sigma'], production['sigma_error'], production['target_met'], production['status']) == (sigma,error,target,'computed'):
                    continue
                db.execute("UPDATE productions SET sigma=?,sigma_error=?,target_met=?,status='computed',failure=NULL WHERE pid=?", (sigma,error,target,production['pid']))
                db.execute('UPDATE histories SET rate=probability*? WHERE pid=?', (sigma,production['pid']))
                changed = True
            for display_id, record in (statuses or {}).items():
                status = record['status']
                if status == 'uncomputed':
                    status = 'not_calculated'
                if status not in ('not_calculated','running','failed'):
                    raise ValueError('Invalid production calculation status')
                production = db.execute('SELECT * FROM productions WHERE display_id=?', (str(display_id),)).fetchone()
                if production is None:
                    raise ValueError('Status refers to an unknown production channel')
                failure = _json(record['error']) if record.get('error') is not None else None
                if production['status'] == status and production['failure'] == failure and production['sigma'] is None:
                    continue
                db.execute('UPDATE productions SET sigma=NULL,sigma_error=NULL,target_met=NULL,status=?,failure=? WHERE pid=?', (status,failure,production['pid']))
                db.execute('UPDATE histories SET rate=NULL WHERE pid=?', (production['pid'],))
                changed = True
            if changed:
                _revision(db)
        return self.counts()


def estimate_history_combinations(context):
    """Count candidate forests without constructing their nodes or row products.

    This uses the same primary modes and identical-parent combinations as the
    writer. A positive min_probability can discard rows later, so these counts
    describe combinations visited, not a guarantee of the number saved.
    """
    particle_choices = {}

    def combinations(pdgs, ancestors=()):
        total = 1
        for pdg, multiplicity in Counter(pdgs).items():
            choices = count(pdg, ancestors)
            total *= math.comb(choices + multiplicity - 1, multiplicity) if choices else 0
        return total

    @lru_cache(maxsize=4096)
    def count(pdg, ancestors=()):
        if context['terminal'](pdg):
            result = 1
        else:
            result = sum(combinations(daughters, ancestors + (pdg,))
                         for daughters, _ in context['decay_choices'](pdg, ancestors))
        particle_choices[pdg] = result
        return result

    productions = {channel['id']: combinations(channel['final_pdgs'])
                   for channel in context['channels']}
    return dict(total_combinations=sum(productions.values()), productions=productions,
                particle_choices=particle_choices)


def build_history_store(context, path=None, *, batch_size=1000, memory_mb=64, max_render_nodes=10000):
    """Build or resume canonical subtree choices and production forests."""
    for label, number in (('batch_size',batch_size),('memory_mb',memory_mb),('max_render_nodes',max_render_nodes)):
        if type(number) is not int or number < 1:
            raise ValueError(f'{label} must be a positive integer')
    with stage('Counting decay-history combinations'):
        expected = estimate_history_combinations(context)
    message(f"Decay-history discovery: {format_count(expected['total_combinations'])} candidate combinations "
            f"across {format_count(len(context['channels']))} production channels; no automatic truncation")
    if context['limits']['min_probability'] > 0:
        message('The configured min_probability filter can reduce the number of saved histories')
    for channel in context['channels']:
        message(f"  {channel['id']} {channel['process']}: "
                f"{format_count(expected['productions'][channel['id']])} candidate combinations")
    owner = None
    if path is None:
        owner = tempfile.TemporaryDirectory(prefix='madgraph_histories_')
        path = Path(owner.name) / 'histories.sqlite'
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    particles, channels = context['particles'], context['channels']
    def production_identity(finals):
        return _hash(('pp',[(p,context['particle_identity'][p]) for p in sorted(finals)]))
    payload = {key: context[key] for key in ('particles','mode_records','total_widths','stable','limits','provenance')}
    payload['productions'] = sorted(tuple(sorted(c['final_pdgs'])) for c in channels)
    fingerprint = _hash(dict(schema=SCHEMA, inputs=payload))
    db = _connect(path, memory_mb=memory_mb)
    try:
        has_metadata = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
        if has_metadata and _get(db, 'schema') not in (None, SCHEMA):
            raise ValueError('History database schema changed; use a new calculation path')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS nodes(node_id INTEGER PRIMARY KEY,node_key TEXT UNIQUE,pdg INTEGER,children TEXT,probability REAL,final_counts TEXT,flags TEXT,cuts TEXT);
        CREATE TABLE IF NOT EXISTS choices(pdg INTEGER,ordinal INTEGER,node_id INTEGER REFERENCES nodes,PRIMARY KEY(pdg,ordinal)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS expansions(pdg INTEGER PRIMARY KEY,next_ordinal INTEGER DEFAULT 0,complete INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS productions(pid INTEGER PRIMARY KEY,identity TEXT UNIQUE,display_id TEXT UNIQUE,process TEXT,final_pdgs TEXT,next_ordinal INTEGER DEFAULT 0,row_count INTEGER DEFAULT 0,complete INTEGER DEFAULT 0,coverage REAL DEFAULT 0,sigma REAL,sigma_error REAL,status TEXT DEFAULT 'not_calculated',target_met INTEGER,flags TEXT DEFAULT '[]',failure TEXT);
        CREATE TABLE IF NOT EXISTS histories(history_id TEXT NOT NULL,pid INTEGER REFERENCES productions,ordinal INTEGER,roots TEXT,probability REAL,rate REAL,PRIMARY KEY(pid,ordinal)) WITHOUT ROWID;
        ''')
        existing = _get(db, 'fingerprint')
        if existing and existing != fingerprint:
            raise ValueError('History database input fingerprint changed; use a new calculation path')
        with db:
            view_changed = existing is not None and (_get(db, 'max_render_nodes') != max_render_nodes or _get(db,'memory_mb') != memory_mb)
            _put(db, 'schema', SCHEMA)
            _put(db, 'fingerprint', fingerprint)
            _put(db, 'particles', particles)
            _put(db, 'limits', context['limits'])
            _put(db, 'notices', context['notices'])
            _put(db, 'memory_mb', memory_mb)
            _put(db, 'max_render_nodes', max_render_nodes)
            if existing is None:
                _put(db, 'revision', 0)
                _put(db, 'discovery_complete', False)
            elif view_changed:
                _revision(db)
            # Free all display labels together so harmless C-label permutations
            # cannot collide with a label still attached to another production.
            mappings = {c['id']: production_identity(c['final_pdgs']) for c in channels}
            renamed = any(row['display_id'] not in mappings or mappings[row['display_id']] != row['identity']
                          for row in db.execute('SELECT display_id,identity FROM productions'))
            if renamed:
                db.execute("UPDATE productions SET display_id='@' || identity")
            for channel in channels:
                finals = tuple(sorted(channel['final_pdgs']))
                identity = production_identity(finals)
                previous = db.execute('SELECT * FROM productions WHERE identity=?', (identity,)).fetchone()
                if previous is None:
                    db.execute('INSERT INTO productions(identity,display_id,process,final_pdgs) VALUES (?,?,?,?)', (identity,channel['id'],channel['process'],_json(finals)))
                elif previous['display_id'] != channel['id'] or previous['process'] != channel['process']:
                    db.execute('UPDATE productions SET display_id=?,process=? WHERE identity=?', (channel['id'],channel['process'],identity))
                    _revision(db)

        maximum = max((len(c['final_pdgs']) for c in channels),default=0)
        all_finals = {tuple(sorted(c['final_pdgs'])):c['id'] for c in channels}
        @lru_cache(maxsize=max(16,min(4096,memory_mb * 16)))
        def node_data(node_id):
            row = db.execute('SELECT * FROM nodes WHERE node_id=?', (node_id,)).fetchone()
            return dict(key=row['node_key'], pdg=row['pdg'], probability=row['probability'],
                        final=Counter({int(p):n for p,n in json.loads(row['final_counts'])}),
                        flags=set(json.loads(row['flags'])), cuts={tuple(c) for c in json.loads(row['cuts'])})

        def insert_node(pdg, children, probability):
            items = [node_data(c) for c in children]
            children = tuple(c for _,c in sorted((node_data(c)['key'],c) for c in children))
            key = _hash((context['particle_identity'][pdg],pdg,[node_data(c)['key'] for c in children]))
            final, flags = Counter(), set()
            cuts = {(pdg,)}
            frontier = {()}
            for item in items:
                final.update(item['final'])
                flags.update(item['flags'])
                frontier = {tuple(sorted(a+b)) for a in frontier for b in item['cuts'] if len(a)+len(b)<=maximum}
            if not children:
                final[pdg] += 1
            else:
                flags.update(context['node_flags'](pdg,[item['pdg'] for item in items]))
                cuts.update(frontier)
            db.execute('INSERT OR IGNORE INTO nodes(node_key,pdg,children,probability,final_counts,flags,cuts) VALUES (?,?,?,?,?,?,?)',
                       (key,pdg,_json(children),probability,_json(sorted(final.items())),_json(sorted(flags)),_json(sorted(cuts))))
            return db.execute('SELECT node_id FROM nodes WHERE node_key=?', (key,)).fetchone()[0]

        @lru_cache(maxsize=max(16,min(4096,memory_mb * 16)))
        def choice(pdg, ordinal):
            return db.execute('SELECT node_id FROM choices WHERE pdg=? AND ordinal=?',(pdg,ordinal)).fetchone()[0]

        def combine(pdgs, start=0):
            groups = []
            for pdg,count in sorted(Counter(pdgs).items()):
                n = db.execute('SELECT next_ordinal FROM expansions WHERE pdg=?', (pdg,)).fetchone()[0]
                groups.append((pdg,count,n,math.comb(n+count-1,count) if n else 0))
            suffix = [1] * (len(groups)+1)
            for index in range(len(groups)-1,-1,-1):
                suffix[index] = suffix[index+1] * groups[index][3]
            def walk(index, offset, roots, probability):
                if index == len(groups):
                    yield roots,probability
                    return
                pdg,count,n,size = groups[index]
                if not size or not suffix[index+1]:
                    return
                first,remainder = divmod(offset,suffix[index+1])
                for indices in replacement_indices(n,count,first):
                    children, weight = [], 1.
                    repetitions = Counter(indices)
                    factor = math.factorial(count)
                    for ordinal, number in repetitions.items():
                        node_id = choice(pdg, ordinal)
                        children.extend([node_id]*number)
                        weight *= node_data(node_id)['probability'] ** number
                        factor //= math.factorial(number)
                    yield from walk(index+1,remainder,roots+tuple(children),probability*weight*factor)
                    remainder = 0
            return suffix[0],walk(0,start,(),1.)

        def checkpoint():
            _revision(db)
            db.commit()

        def ensure(pdg, ancestors=()):
            previous = db.execute('SELECT * FROM expansions WHERE pdg=?',(pdg,)).fetchone()
            if previous and previous['complete']:
                return
            if context['terminal'](pdg):
                with db:
                    root = insert_node(pdg,(),1.)
                    db.execute('INSERT OR REPLACE INTO choices VALUES (?,?,?)',(pdg,0,root))
                    db.execute('INSERT OR REPLACE INTO expansions VALUES (?,1,1)',(pdg,))
                    _revision(db)
                return
            modes = sorted(context['decay_choices'](pdg,ancestors))
            for daughters,_ in modes:
                for child in sorted(set(daughters)):
                    ensure(child,ancestors+(pdg,))
            db.execute('INSERT OR IGNORE INTO expansions(pdg) VALUES (?)',(pdg,))
            start = db.execute('SELECT next_ordinal FROM expansions WHERE pdg=?',(pdg,)).fetchone()[0]
            offset, ordinal = 0,start
            name = particles[pdg].get('name', str(pdg))
            with stage(f'Expanding decays of {name}', total=expected['particle_choices'][pdg],
                       completed=start, unit='choices') as progress:
                for daughters,branch in modes:
                    total,_ = combine(daughters)
                    if offset+total <= start:
                        offset += total
                        continue
                    _,combinations = combine(daughters,max(0,start-offset))
                    for children,weight in combinations:
                        root = insert_node(pdg,children,branch*weight)
                        db.execute('INSERT INTO choices VALUES (?,?,?)',(pdg,ordinal,root))
                        ordinal += 1
                        db.execute('UPDATE expansions SET next_ordinal=? WHERE pdg=?',(ordinal,pdg))
                        if ordinal % batch_size == 0:
                            checkpoint()
                            progress.update(ordinal)
                    offset += total
                db.execute('UPDATE expansions SET complete=1 WHERE pdg=?',(pdg,))
                checkpoint()
                progress.update(ordinal)

        for channel in channels:
            for pdg in sorted(set(channel['final_pdgs'])):
                ensure(pdg)
            production = db.execute('SELECT * FROM productions WHERE display_id=?',(channel['id'],)).fetchone()
            if production['complete']:
                continue
            pid, ordinal, count = production['pid'],production['next_ordinal'],production['row_count']
            probability_sum, correction = production['coverage'],0.
            total,combinations = combine(channel['final_pdgs'],ordinal)
            with stage(f"Building histories for {channel['id']} {channel['process']}", total=total,
                       completed=ordinal, unit='combinations') as progress:
                for roots,probability in combinations:
                    ordinal += 1
                    if probability >= context['limits']['min_probability']:
                        roots = tuple(c for _,c in sorted((node_data(c)['key'],c) for c in roots))
                        history_id = 'H'+_hash((production['identity'],[node_data(c)['key'] for c in roots]))
                        rate = None if production['sigma'] is None else production['sigma']*probability
                        db.execute('INSERT INTO histories VALUES (?,?,?,?,?,?)',(history_id,pid,ordinal-1,_json(roots),probability,rate))
                        count += 1
                        y = probability-correction
                        updated = probability_sum+y
                        correction = (updated-probability_sum)-y
                        probability_sum = updated
                    if ordinal % batch_size == 0:
                        db.execute('UPDATE productions SET next_ordinal=?,row_count=?,coverage=? WHERE pid=?',(ordinal,count,probability_sum,pid))
                        checkpoint()
                        progress.update(ordinal, detail=f'{format_count(count)} saved')
                if probability_sum > 1+1e-6:
                    raise ValueError('Cascade probability exceeds one')
                db.execute('UPDATE productions SET next_ordinal=?,row_count=?,coverage=?,complete=1 WHERE pid=?',(ordinal,count,probability_sum,pid))
                checkpoint()
                progress.update(ordinal, detail=f'{format_count(count)} saved')

        # Append discovery records in physical production/ordinal order. Building
        # random SHA/order indexes per insertion thrashes the bounded page cache
        # on million-row catalogues. SQLite's external index sort uses temp files
        # and builds them once after discovery (or retries after interruption).
        # Partial catalogues remain queryable through a file-backed SQL sort.
        indexes = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        if not {'history_id_lookup','history_rank_covering'}.issubset(indexes):
            saved = db.execute('SELECT COALESCE(SUM(row_count),0) FROM productions').fetchone()[0]
            with stage(f'Indexing {format_count(saved)} saved histories', total=2, unit='indexes') as progress:
                db.execute('BEGIN')
                progress.update(0, detail='history ID lookup')
                db.execute('CREATE UNIQUE INDEX IF NOT EXISTS history_id_lookup ON histories(history_id)')
                # Cover the row payload too: ordered exports should read index pages
                # sequentially instead of making millions of random table lookups.
                progress.update(1, detail='ranking and export order')
                db.execute('CREATE INDEX IF NOT EXISTS history_rank_covering ON histories(rate DESC,history_id,roots,probability)')
                db.execute('DROP INDEX IF EXISTS history_rank')
                checkpoint()
                progress.update(2)

        # Production overlap can be decided from compact possible cuts, without
        # expanding the millions of combined history forests.
        links = {c['id']:set() for c in channels}
        @lru_cache(maxsize=max(16,min(4096,memory_mb*16)))
        def possible_cuts(pdg, budget):
            answer = {(pdg,)}
            if budget<2 or context['terminal'](pdg):
                return answer
            for daughters,_ in context['decay_choices'](pdg,()):
                if len(daughters)>budget:
                    continue
                partial = {()}
                for child in daughters:
                    partial = {tuple(sorted(a+b)) for a in partial for b in possible_cuts(child,budget-len(daughters)+1) if len(a)+len(b)<=budget}
                answer.update(partial)
            return answer
        for channel in channels:
            produced = channel['final_pdgs']
            frontier = {()}
            for pdg in produced:
                options = possible_cuts(pdg,maximum-len(produced)+1)
                frontier = {tuple(sorted(a+b)) for a in frontier for b in options if len(a)+len(b)<=maximum}
            for final in frontier:
                other = all_finals.get(final)
                if other and other != channel['id']:
                    links[channel['id']].add(other)
                    links[other].add(channel['id'])
        for display_id,others in links.items():
            flags = ['Possible production overlap: '+', '.join(sorted(others))] if others else []
            if db.execute('SELECT flags FROM productions WHERE display_id=?',(display_id,)).fetchone()[0] != _json(flags):
                db.execute('UPDATE productions SET flags=? WHERE display_id=?',(_json(flags),display_id))
                _revision(db)
        if not _get(db, 'discovery_complete', False):
            _put(db, 'discovery_complete', True)
            _revision(db)
        db.commit()
    except BaseException as error:
        db.rollback()
        if (isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)) or (
                isinstance(error, sqlite3.Error) and getattr(error, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL):
            error.category = 'resource_limit'
        try:
            error.partial_rows = SQLiteHistoryRows(path,owner=owner)
        except Exception:
            pass
        raise
    finally:
        db.close()
    return SQLiteHistoryRows(path,owner=owner)
