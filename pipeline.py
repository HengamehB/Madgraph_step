"""One durable, file-driven model workflow shared by the CLI and Python API."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
import re
import importlib.metadata
import sys
import types
import uuid

from pipeline_config import load_config, PipelineConfig
from pipeline_storage import RunStore, atomic_text, digest, file_digest, public, write_json
from workflow_errors import error_category
from run_paths import contained_path, portable_history_reference

ROOT = Path(__file__).resolve().parent


def _call_input_function(path, function_name):
    """Execute explicitly selected legacy Python inputs without stale bytecode."""
    path = Path(path).resolve()
    module = types.ModuleType('madgraph_user_inputs')
    module.__file__ = str(path)
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(path.parent))
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise ValueError(f'{path} must define {function_name}()')
        value = function()
    finally:
        sys.path[:] = old_path
    return value


def load_model_definition(path):
    """Discover operators even when the numerical inputs are incomplete."""
    from model_builder import ModelDefinition
    if Path(path).suffix.lower() in {'.txt', '.tex'}:
        from tex_inputs import read_inputs
        return read_inputs(path).definition
    if Path(path).suffix.lower() != '.py':
        raise ValueError('Model input must be .txt, .tex, or an explicitly selected legacy .py file')
    value = _call_input_function(path, 'define_model')
    if not isinstance(value, ModelDefinition):
        raise ValueError('define_model() must return a ModelDefinition')
    return value


def load_model(path):
    """Read a restricted TeX input block or an explicitly selected Python file."""
    if Path(path).suffix.lower() in {'.txt', '.tex'}:
        from tex_inputs import read_inputs
        return read_inputs(path).build()
    if Path(path).suffix.lower() != '.py':
        raise ValueError('Model input must be .txt, .tex, or an explicitly selected legacy .py file')
    value = _call_input_function(path, 'build_model')
    if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[1], dict):
        raise ValueError('build_model() must return (selected_lagrangian, physical_inputs_dict)')
    if not hasattr(value[0], 'selected_ids'):
        raise ValueError('build_model() must return a result.select(...) Lagrangian')
    from feynrules_export import _fields
    from fixed_sm import normalize_fixed_sm
    lagrangian, inputs = value
    return lagrangian, normalize_fixed_sm(inputs, _fields(lagrangian))


def _safe_name(value):
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')
    return name[:100] or 'model'


class Pipeline:
    """Resume verified stage artifacts; always build fresh numerical executables.

    A physics benchmark has one stable run directory. Collider, decay, tool and
    ranking changes receive independent stage fingerprints, allowing display
    edits and resumed invocations to reuse completed integrations safely.
    """
    def __init__(self, config=None, *, model_file=None, output=None):
        self._config_object = isinstance(config, PipelineConfig)
        overrides = (config.input_overrides if self._config_object else None)
        self._model_override = model_file if model_file is not None else (
            overrides.get('model_file') if overrides is not None else config.model_file if self._config_object else None)
        self._output_override = output if output is not None else (
            overrides.get('output') if overrides is not None else config.output if self._config_object else None)
        self.config = config if self._config_object else load_config(config, model_file=model_file, output=output)
        self._config_digest = self._read_config()
        self._input_source = b''
        self._input_metadata = None
        self._signature = None
        self._stage_keys = {}
        self._memory_keys = {}
        self._tool_digest_cache = {}
        self._settings_snapshot = None
        self._active_settings = None
        self.lagrangian = self.inputs = self._prepared = None
        self._ufo = self._channels = self._decays = None
        self.cross_sections = self.ranking = None
        self.explicit_decay_results = []
        self.run_directory = self._store = None
        self.failures = []
        self._requested_scope = 'all'
        self.complete = self.requested_complete = False
        self.discovery_complete = self.requested_rates_complete = self.all_rates_complete = False
        self.selected_integrations_complete = self.export_complete = False
        self._refresh()

    def _read_config(self):
        return self.config.path.read_bytes() if self.config.path.is_file() else b''

    def _code_fingerprint(self, *names):
        return digest({name: file_digest(ROOT / name) if (ROOT / name).exists() else None for name in names})

    def _compiler_fingerprint(self):
        names = ['feynrules_export.py', 'model_builder.py', 'lagrangian_engine.py', 'lagrangian_maker.py',
                 'lagrangian_selection.py', 'group_theory.py', 'component_invariants.py', 'phenomenology_notation.py']
        names += [p.name for p in ROOT.glob('*compiler.py') if not p.name.startswith('test_')]
        names += [p.name for p in ROOT.glob('generic_*.py')]
        names += ['compiler_limits.py', 'native_colour_basis.py', 'joint_scalar_invariants.py',
                  'fixed_sm.py', 'sm_parameters.py']
        versions = {}
        for name in ('numpy', 'scipy'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return digest({'source': self._code_fingerprint(*sorted(set(names))), 'libraries': versions,
                       'python': sys.version.split()[0]})

    def _tool_fingerprint(self, *keys):
        from portable_runtime import _paths, tool_source_files
        from workflow_progress import stage
        paths = _paths(self.config.runtime)
        records = {}
        for key in keys:
            value = paths.get(key)
            if not value:
                records[key] = None
                continue
            path = Path(value)
            if not path.exists():
                records[key] = {'missing': True}
                continue
            with stage('Checking tool files: ' + key):
                is_directory = path.is_dir()
                files = tool_source_files(key, path) if is_directory else [path]
                entries = {}
                for file in sorted(files):
                    stat = file.stat()
                    cache_key = (str(file), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                    if cache_key not in self._tool_digest_cache:
                        self._tool_digest_cache[cache_key] = file_digest(file)
                    entries[str(file.relative_to(path)) if is_directory else '@executable'] = self._tool_digest_cache[cache_key]
            # Tool content, not its installation directory, determines physics.
            records[key] = {'content': entries}
        return digest(records)

    def _refresh(self):
        raw = self._read_config()
        config_changed = raw != self._config_digest
        # A config edit may point to a replacement after the old file was moved.
        # Resolve that edit before attempting to read the obsolete input path.
        source = None if config_changed else self.config.model_file.read_bytes()
        def metadata_bytes(content):
            if self.config.model_file.suffix.lower() not in {'.txt', '.tex'} or b'_metadata' not in content:
                return None
            from tex_inputs import _parse
            try:
                return _parse(self.config.model_file, content.decode('utf-8')).metadata_bytes
            except Exception as exc:
                self._input_failure(content, exc)
                raise
        metadata = None if source is None else metadata_bytes(source)
        changed = source != self._input_source or metadata != self._input_metadata
        if config_changed or (self.inputs is not None and self.config.model_file.suffix.lower() in {'.txt', '.tex'} and changed):
            # PipelineConfig is a snapshot: preserve its chosen input/output when
            # refreshing a .tex file supplied through the CLI.
            self.config = load_config(self.config.path if self.config.path.exists() else None,
                                      model_file=self._model_override,
                                      output=self._output_override)
            self._config_digest = raw
            source = self.config.model_file.read_bytes()
            metadata = metadata_bytes(source)
            changed = source != self._input_source or metadata != self._input_metadata
        if self.inputs is None or changed:
            if self._store is not None and self._store._lock_depth and self.inputs is not None:
                raise RuntimeError('Physical input file changed during the active run; rerun with the saved inputs')
            try:
                self.lagrangian, self.inputs = load_model(self.config.model_file)
            except Exception as exc:
                self._input_failure(source, exc)
                raise
            self._input_source = source
            self._input_metadata = metadata
        else:
            # Python callers can edit project.inputs between stages. Apply the
            # EW rules to those edits before choosing a cache key.
            from feynrules_export import _fields
            from fixed_sm import normalize_fixed_sm
            normalized = normalize_fixed_sm(self.inputs, _fields(self.lagrangian))
            self.inputs.clear()
            self.inputs.update(normalized)
        self._bind_physics()
        current = deepcopy(dict(runtime=self.config.runtime, collider=self.config.collider,
                                decays=self.config.decays, ranking=self.config.ranking,
                                explicit_decays=self.config.explicit_decays))
        if self._settings_snapshot is not None:
            for name, attr in (('runtime', '_ufo'), ('collider', '_channels'), ('decays', '_decays')):
                if current[name] != self._settings_snapshot[name]:
                    stage = {'runtime': 'export', 'collider': 'channels', 'decays': 'decays'}[name]
                    setattr(self, attr, None)
                    self._memory_keys.pop(stage, None)
                    self._stage_keys.pop(stage, None)
                    self._invalidate_after(stage)
            if current['ranking'] != self._settings_snapshot['ranking']:
                self.ranking = None
                self._stage_keys.pop('ranking', None)
                self.complete = self.requested_complete = False
                self.discovery_complete = self.export_complete = False
            if current['explicit_decays'] != self._settings_snapshot['explicit_decays']:
                self.explicit_decay_results = []
                self.selected_integrations_complete = False
                self.complete = self.requested_complete = self.export_complete = False
        self._settings_snapshot = current

    def _bind_physics(self):
        payload = dict(inputs=self.inputs, selected=self.lagrangian.operators,
                       fields=repr(self.lagrangian._source.models[self.lagrangian.scenario - 1]),
                       compiler=self._compiler_fingerprint())
        manual = self.inputs.get('feynrules')
        if isinstance(manual, dict):
            # Trusted legacy FeynRules sources are physics input too. A path
            # remaining unchanged does not prove its model declarations did.
            model_files = manual.get('model_files', ())
            if isinstance(model_files, (list, tuple)):
                payload['explicit_model_sources'] = {
                    str(Path(name).expanduser().resolve()): file_digest(Path(name).expanduser())
                    if Path(name).expanduser().is_file() else None for name in model_files if isinstance(name, str)}
        signature = digest(payload)
        destination = self.config.output / _safe_name(self.inputs.get('model_name', self.config.model_file.stem)) / signature[:20]
        if signature != self._signature or destination != self.run_directory:
            if self._store is not None and self._store._lock_depth:
                raise RuntimeError('Physical inputs changed during the active run; rerun with the saved inputs')
            self._signature = signature
            self._ufo = self._channels = self._decays = self._prepared = None
            self.cross_sections = self.ranking = None
            self.explicit_decay_results = []
            self._stage_keys, self._memory_keys = {}, {}
            self.failures = []
            self.complete = self.requested_complete = False
            self.discovery_complete = self.requested_rates_complete = self.all_rates_complete = False
            self.selected_integrations_complete = self.export_complete = False
            self.run_directory = destination
            self._store = RunStore(destination)

    @property
    def vacuum_assumptions(self):
        """Report the electroweak expansion point for a successfully loaded model."""
        from fixed_sm import fixed_sm_report
        return fixed_sm_report(self.inputs) if self.lagrangian is not None else None

    def _input_failure(self, source, error):
        self.lagrangian = self._prepared = self._ufo = self._channels = self._decays = None
        self.cross_sections = self.ranking = None
        self.explicit_decay_results = []
        self._stage_keys, self._memory_keys = {}, {}
        self.failures = []
        self.complete = self.requested_complete = False
        self.discovery_complete = self.requested_rates_complete = self.all_rates_complete = False
        self.selected_integrations_complete = self.export_complete = False
        self.inputs = {'model_name': self.config.model_file.stem}
        self._signature = digest({'source': source.decode(errors='replace')})
        self.run_directory = self.config.output / _safe_name(self.config.model_file.stem) / ('invalid-' + self._signature[:20])
        self._store = RunStore(self.run_directory)
        try:
            with self._store.lock():
                atomic_text(self.run_directory / ('input' + self.config.model_file.suffix), source.decode(errors='replace'))
                write_json(self.run_directory / 'validation.json', {'status': 'failed', 'type': type(error).__name__, 'category': error_category(error), 'message': str(error)})
                self._publish('failed', error=error)
        except Exception as report_error:
            error.add_note('Failure report could not be written: ' + str(report_error))
        finally:
            # A caller may restore exactly the last valid source bytes.
            # Force a reload rather than keeping the temporary failure label.
            self.inputs = None
            self._input_source = b''
            self._input_metadata = None

    def _snapshot(self):
        if self._input_metadata is not None:
            # Preserve the immutable companion with the exact editable source.
            # It remains relative to input.txt even after the run is relocated.
            from tex_inputs import _parse
            from model_metadata import load_metadata, MetadataError
            source = _parse(self.config.model_file, self._input_source.decode())
            if source.metadata_bytes != self._input_metadata:
                raise RuntimeError('Model metadata changed during the active run; rerun with verified inputs.')
            from run_paths import contained_path
            reference = source.assignments['_metadata']
            target = contained_path(self.run_directory, reference)
            literal = self.run_directory / reference
            if literal.is_symlink() or literal.parent.is_symlink():
                raise ValueError('Saved model metadata must not use symlinks.')
            content = None
            if target.exists():
                if not target.is_file():
                    raise ValueError('Saved model metadata must be a regular file.')
                try:
                    _, content, _ = load_metadata(self.run_directory / 'input.txt', reference)
                except MetadataError:
                    pass  # Rebuild a damaged derived copy from the verified source.
            if content != self._input_metadata:
                atomic_text(target, self._input_metadata.decode('utf-8'))
        atomic_text(self.run_directory / ('input' + self.config.model_file.suffix), self._input_source.decode())
        write_json(self.run_directory / 'resolved_inputs.json', self.inputs)
        write_json(self.run_directory / 'vacuum_assumptions.json', self.vacuum_assumptions)
        if self.config.model_file.suffix.lower() in {'.txt', '.tex'}:
            from tex_inputs import read_inputs
            source = read_inputs(self.config.model_file)
            write_json(self.run_directory / 'input_resolution.json', getattr(source, 'provenance', {}))
            # The editable operator inventory is source/provenance, not part
            # of the active physical identity. Keep disabled values reviewable
            # while retaining the existing verified numerical artifacts.
            write_json(self.run_directory / 'operator_inputs.json', dict(
                schema='operator-input-state-v1', operators=source.assignments.get('operators'),
                interpretation='Only enabled entries enter the selected Lagrangian and numerical physics.'))
        write_json(self.run_directory / 'operator_catalogue.json', self.lagrangian._source.catalogues)
        write_json(self.run_directory / 'selected_operators.json', self.lagrangian.operators)
        expression = self.lagrangian.latex
        atomic_text(self.run_directory / 'lagrangian_expression.tex', expression + '\n')
        equation = expression.replace(r'\begin{aligned}', r'\begin{align*}').replace(r'\end{aligned}', r'\end{align*}')
        if r'\begin{align*}' not in equation:
            equation = r'\[' + equation + r'\]'
        document = ('\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n'
                    '\\usepackage[margin=1in]{geometry}\n\\allowdisplaybreaks\n'
                    '\\begin{document}\n' + equation + '\n\\end{document}\n')
        atomic_text(self.run_directory / 'lagrangian.tex', document)
        write_json(self.run_directory / 'settings.json', dict(collider=self.config.collider, decays=self.config.decays,
                   ranking=self.config.ranking, explicit_decays=self.config.explicit_decays, runtime=self.config.runtime))

    @contextmanager
    def _session(self):
        self._refresh()
        with self._store.lock():
            current = digest({'settings': self._settings_snapshot,
                              'runner': self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py', 'selected_decay.py')})
            outer = self._store._lock_depth == 1
            if not outer and self._active_settings != current:
                raise RuntimeError('Calculation settings or runner changed during the active run; rerun')
            if outer:
                self._active_settings = current
            try:
                self._snapshot()
                yield
            except BaseException as exc:
                if outer:
                    self.complete = self.requested_complete = False
                    self.export_complete = False
                    try:
                        self._publish('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=exc)
                    except Exception as report_error:
                        exc.add_note('Failure report could not be written: ' + str(report_error))
                raise
            finally:
                if outer:
                    self._active_settings = None

    def _key(self, stage, **payload):
        key = digest({'schema': 'pipeline-stage-v1', 'stage': stage, 'physics': self._signature, **payload})
        self._stage_keys[stage] = key
        return key

    def _attempt(self, stage, key):
        return contained_path(self.run_directory, Path(stage) / key[:16] / ('attempt-' + uuid.uuid4().hex[:10]))

    def model(self):
        self._refresh()
        return self.lagrangian, self.inputs

    def validate(self):
        """Check the inputs and construct the spectrum at the SM electroweak vacuum."""
        from feynrules_export import prepare_model
        with self._session():
            try:
                if self._prepared is None:
                    self._prepared = prepare_model(self.lagrangian, self.inputs, runtime=self.config.runtime)
                prepared = self._prepared
                prepared.runtime = deepcopy(self.config.runtime)
                compiler = prepared.compiler
                bases = {'scalar_basis': prepared.basis}
                if compiler is not None:
                    for key, value in vars(compiler).items():
                        if 'basis' in key or key in ('normalization', 'normalizations', 'mixing', 'transformation', 'particles', 'coordinates',
                                                     'scalar_transformation', 'scalar_maps', 'scalar_states'):
                            bases[key] = value
                write_json(self.run_directory / 'spectrum.json', dict(masses=prepared.masses, **bases))
                write_json(self.run_directory / 'contractions.json', dict(operators=prepared.operators, coverage=prepared.coverage,
                           notation=self.lagrangian.notation))
                write_json(self.run_directory / 'validation.json', dict(status='physical_inputs_validated',
                           external_export_validated=self._ufo is not None, coverage=prepared.coverage,
                           vacuum=self.vacuum_assumptions))
                self._publish('validated')
                return prepared
            except Exception as exc:
                try:
                    write_json(self.run_directory / 'validation.json', dict(status='failed', type=type(exc).__name__, category=error_category(exc), message=str(exc),
                               vacuum=self.vacuum_assumptions))
                    self._publish('failed', error=exc)
                except Exception as report_error:
                    exc.add_note('Failure report could not be written: ' + str(report_error))
                raise

    def _invalidate_after(self, stage):
        affected = {'export': ('channels', 'decays', 'rates', 'ranking'),
                    'channels': ('rates', 'ranking'), 'decays': ('rates', 'ranking')}[stage]
        for name in affected:
            setattr(self, {'channels': '_channels', 'decays': '_decays', 'rates': 'cross_sections'}.get(name, name), None)
            self._memory_keys.pop(name, None)
            self._stage_keys.pop(name, None)
        self.complete = self.requested_complete = False
        self.discovery_complete = self.requested_rates_complete = self.all_rates_complete = False
        self.selected_integrations_complete = self.export_complete = False
        self.explicit_decay_results = []

    def export_ufo(self):
        from madgraph_runner import _artifact
        from workflow_progress import stage
        with self._session():
            key = self._key('export', tools=self._tool_fingerprint('feynrules', 'wolfram_kernel', 'madgraph', 'python'),
                            runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py'))
            if self._ufo is None or self._memory_keys.get('export') != key:
                self._invalidate_after('export')
                self._ufo = self._store.load('export', key)
                if self._ufo is None:
                    prepared = self.validate()
                    attempt = self._attempt('export', key)
                    with stage('Preparing and exporting UFO with FeynRules', log=attempt / 'feynrules.log'):
                        self._ufo = prepared.export_ufo(attempt)
                    self._store.save('export', key, self._ufo, attempt)
                self._memory_keys['export'] = key
            _artifact(self._ufo)
            # The cached UFO verifies its own source and numerical artifacts.
            # Rebuild the inexpensive physical metadata as well: top-level
            # spectrum/contraction files may have been removed or edited since
            # the last process, and a complete report must not link stale data.
            prepared = self.validate()
            write_json(self.run_directory / 'validation.json', dict(status='ufo_validated',
                       external_export_validated=True,
                       coverage=prepared.coverage,
                       vacuum=self.vacuum_assumptions,
                       model_digest=self._ufo.saved_model_digest, parameter_card_digest=self._ufo.saved_parameter_card_digest,
                       ufo=str(self._ufo.path), parameter_card=str(self._ufo.parameter_card)))
            self._publish('exported')
            return self._ufo

    def _export_identity(self, ufo):
        # A repaired export may have a new path or numerical content; neither
        # old channel snapshots nor old width cards may follow it implicitly.
        if not hasattr(ufo, 'path'):
            return None
        relative = Path(ufo.path).relative_to(self.run_directory)
        contained_path(self.run_directory, relative)
        return dict(path=relative.as_posix(), model=getattr(ufo, 'saved_model_digest', ''),
                    card=getattr(ufo, 'saved_parameter_card_digest', ''))

    def channels(self, *, show=True):
        from madgraph_runner import list_production_channels, validate_provenance
        with self._session():
            ufo = self.export_ufo()
            key = self._key('channels', export=self._stage_keys.get('export'),
                            artifact=self._export_identity(ufo), collider=self.config.collider,
                            runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py'),
                            tools=self._tool_fingerprint('madgraph', 'python'))
            if self._channels is None or ('channels' in self._memory_keys and self._memory_keys['channels'] != key):
                self._invalidate_after('channels')
                self._channels = self._store.load('channels', key)
                if self._channels is None:
                    attempt = self._attempt('channels', key)
                    self._channels = list_production_channels(ufo, self.config.collider, self.config.runtime,
                                                              output_dir=attempt, show=False)
                    self._store.save('channels', key, self._channels, attempt)
                self._memory_keys['channels'] = key
            validate_provenance(self._channels)
            self._channels.runtime = deepcopy(self.config.runtime)
            self._publish('channels_complete')
            if show:
                self._channels.show()
                notice = self.production_notice()
                if notice:
                    print(notice)
            return self._channels

    def production_notice(self):
        """Explain an empty completed discovery without inventing a rate."""
        if self._channels is None or len(self._channels):
            return None
        notice = ('No tree-level production channels containing new particles were found '
                  'for the configured beams, energy and final-state multiplicity. '
                  'Discovery completed; there are no production rates or decay histories to rank.')
        if self.inputs.get('feynrules'):
            return notice
        from feynrules_export import _fields, _coefficients
        fields = tuple(_fields(self.lagrangian))
        if not fields or not all(f.spin == 'scalar' and f.kind == 'real' and f.su2 == 1
                and tuple(f.su3) == (0, 0) and f.hypercharge == 0 for f in fields):
            return notice
        names = {f.name for f in fields}
        values = _coefficients(self.lagrangian, self.inputs)
        for row in self.lagrangian.operators:
            if not any(value != 0 for value in values[row['id']]):
                continue
            factors = {item['name'][2:-1] if item['name'].startswith('D(') else item['name']
                       for item in row['factors']}
            if not factors <= names:
                return notice
        return (notice + ' The selected neutral gauge-singlet scalars have no nonzero '
                'interaction with Standard Model fields. Their masses and self interactions '
                'alone cannot produce them from proton constituents. Enable a portal interaction '
                'only if it belongs to the intended model; coefficients have not been changed.')

    def _add_production_notice(self):
        notice = self.production_notice()
        if notice and self.ranking is not None and notice not in self.ranking.notices:
            self.ranking.notices = tuple(self.ranking.notices) + (notice,)

    def decays(self, *, show=True):
        from madgraph_runner import compute_decays, validate_provenance
        with self._session():
            ufo = self.export_ufo()
            key = self._key('decays', export=self._stage_keys.get('export'),
                            artifact=self._export_identity(ufo), settings=self.config.decays,
                            runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py', 'selected_decay.py'),
                            tools=self._tool_fingerprint('madgraph', 'python', 'fortran', 'cxx', 'make'))
            if self._decays is None or ('decays' in self._memory_keys and self._memory_keys['decays'] != key):
                self._invalidate_after('decays')
                self._decays = self._store.load('decays', key)
                if self._decays is None:
                    attempt = self._attempt('decays', key)
                    self._decays = compute_decays(ufo, runtime=self.config.runtime, output_dir=attempt,
                                                 show=False, **self.config.decays)
                    self._store.save('decays', key, self._decays, attempt)
                self._memory_keys['decays'] = key
            validate_provenance(self._decays, self._decays)
            self._publish('decays_complete')
            if show:
                self._decays.show()
            return self._decays

    def _publish(self, status, selected=None, error=None):
        from pipeline_reports import publish
        self._add_production_notice()
        if selected is not None:
            self._requested_scope = selected
        return publish(self, status=status, selected=self._requested_scope, error=error)

    def _update_completion(self):
        self.requested_complete = bool(self.discovery_complete and self.requested_rates_complete
                                       and self.selected_integrations_complete and self.export_complete)
        self.complete = bool(self.discovery_complete and self.all_rates_complete
                             and self.selected_integrations_complete and self.export_complete)

    @staticmethod
    def _production_identity(channel, channels):
        return dict(process=channel.process, final_pdgs=sorted(channel.final_pdgs),
                    incoming_partons=sorted(channels.collider.get('parton_pdgs', ())))

    def _rate_key(self, channel, channels, base):
        identity = self._production_identity(channel, channels)
        return 'rate-' + digest(identity), digest({'numerical_inputs': base, 'production': identity})

    def _numerical_key(self, channels, decays):
        settings = {key: value for key, value in channels.collider.items()
                    if key not in {'max_tree_signatures', 'new_particles_only'}}
        # Selecting which external channels to discover does not alter an
        # individual channel's matrix element or numerical integration.
        # max_final remains because the worker derives coupling-order caps from it.
        return self._key('rates', model=channels.model_digest, benchmark=channels.benchmark_digest,
                         widths=decays.width_digest, collider=settings,
                         tools=self._tool_fingerprint('madgraph', 'python', 'fortran', 'cxx', 'make'),
                         runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py'))

    @staticmethod
    def _cross_sections(channels, decays, values):
        from madgraph_runner import CrossSectionResults
        return CrossSectionResults(tuple(values), dict(channels.collider), channels.particles,
                                   channels.model_digest, channels.benchmark_digest, decays.width_digest)

    def _hydrate_rates(self, channels, decays, numerical_key):
        """Verify small numerical checkpoints before exposing saved rates."""
        from madgraph_runner import validate_provenance
        import json
        valid, statuses = {}, {}
        old = self.ranking.rows.production_statuses() if hasattr(self.ranking.rows, 'production_statuses') else {}
        context_path = self.ranking.rows.path.with_name('rate_context.json')
        try:
            same_context = json.loads(context_path.read_text())['numerical_key'] == numerical_key
        except (OSError, ValueError, KeyError):
            same_context = False
        for channel in channels:
            stage, key = self._rate_key(channel, channels, numerical_key)
            answer = self._store.load(stage, key)
            if answer is not None and (len(answer.rates) != 1 or
                    answer.model_digest != channels.model_digest or
                    answer.benchmark_digest != channels.benchmark_digest or
                    answer.width_digest != decays.width_digest):
                answer = None
            if answer is None:
                previous = old.get(channel.id, {})
                if same_context and previous.get('status') == 'running':
                    statuses[channel.id] = {'status': 'failed', 'error': dict(stage='rates', channel=channel.id,
                        type='InterruptedError', category='execution_failure',
                        message='Previous integration ended without a committed result; rerun to retry.')}
                elif not same_context or previous.get('status') not in {'failed', 'not_calculated'}:
                    statuses[channel.id] = {'status': 'not_calculated'}
                continue
            row = dict(answer.rates[0])
            # Only the display alias changes; the cache key checked topology.
            row.update(id=channel.id, process=channel.process, final_pdgs=channel.final_pdgs)
            valid[channel.id] = row
        self.cross_sections = self._cross_sections(channels, decays, valid.values())
        validate_provenance(channels, decays, self.cross_sections)
        self.ranking.rows.update_rates(self.cross_sections, statuses=statuses)
        write_json(context_path, {'numerical_key': numerical_key})
        self.ranking.refresh()
        self.all_rates_complete = len(valid) == len(channels)
        requested = self._selected_channels(channels, self._requested_scope)
        self.requested_rates_complete = all(channel.id in valid for channel in requested)
        self.failures = [failure for failure in self.failures if failure.get('stage') != 'rates']
        for channel_id, previous in old.items():
            if same_context and channel_id not in valid and channel_id not in statuses and previous.get('status') == 'failed':
                failure = previous.get('error') or {'message': 'Previous integration failed'}
                self.failures.append(dict(failure, stage='rates', channel=channel_id))
        self.failures.extend(record['error'] for record in statuses.values() if record['status'] == 'failed')
        return valid

    def _history_key(self, channels, decays):
        settings = {key: value for key, value in self.config.ranking.items()
                    if key not in {'min_relative_rate', 'batch_size', 'memory_mb', 'max_render_nodes', 'max_histories'}}
        return self._key('histories', model=channels.model_digest,
                         benchmark=channels.benchmark_digest, widths=decays.width_digest,
                         production=[self._production_identity(c, channels) for c in channels],
                         stable_particles=self.inputs.get('stable_particles', ()), settings=settings,
                         code=self._code_fingerprint('cascade_ranking.py', 'history_store.py'))

    def _retain_history_reference(self, reference):
        """Keep the last trusted SQL snapshot when progress hides old tables."""
        import json
        if not isinstance(reference, dict) or not reference.get('path'):
            return
        reference = portable_history_reference(reference, self.run_directory)
        path = portable_history_reference(reference, self.run_directory, absolute=True)['path']
        sidecar = contained_path(self.run_directory, path.with_name('history_reference.json').relative_to(self.run_directory))
        try:
            previous = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            previous = None
        if (isinstance(previous, dict) and previous.get('fingerprint') == reference.get('fingerprint')
                and previous.get('revision', -1) > reference.get('revision', -1)):
            return
        if (isinstance(previous, dict) and not reference.get('sha256')
                and previous.get('file_stat') == reference.get('file_stat')
                and previous.get('revision') == reference.get('revision')
                and previous.get('fingerprint') == reference.get('fingerprint')
                and previous.get('sha256')):
            reference['sha256'] = previous['sha256']
        if public(reference) != previous:
            write_json(sidecar, reference)

    def _capture_history_reference(self):
        import json
        try:
            manifest = json.loads(contained_path(self.run_directory, 'manifest.json').read_text())
        except (OSError, ValueError):
            return
        self._retain_history_reference(manifest.get('history_store'))

    def _verify_history_cache(self, path):
        """Rebuild untrusted derived rows while preserving verified numerics."""
        import json
        import sqlite3
        from history_store import SQLiteHistoryRows
        path = contained_path(self.run_directory, path.relative_to(self.run_directory))
        sqlite_sidecars = [contained_path(self.run_directory, path.with_name(path.name + suffix).relative_to(self.run_directory))
                           for suffix in ('-journal', '-wal', '-shm')]
        if not path.exists():
            return
        sidecar = contained_path(self.run_directory, path.with_name('history_reference.json').relative_to(self.run_directory))
        try:
            previous = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            previous = None
        reason = None
        if not isinstance(previous, dict) or not previous.get('file_stat'):
            reason = 'History database has no trusted file-stat reference'
        else:
            try:
                current = SQLiteHistoryRows(path).reference(root=self.run_directory)
            except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                reason = 'History database cannot be verified: ' + str(exc)
            else:
                previous = portable_history_reference(previous, self.run_directory)
                if (previous['path'] != current['path']
                        or previous['fingerprint'] != current['fingerprint']):
                    raise ValueError('History database source differs from its saved reference')
                # A newer committed revision can be a legitimate interrupted
                # discovery/rate update that did not reach report publication.
                if (current['revision'] <= previous['revision']
                        and current['file_stat'] != previous['file_stat']):
                    try:
                        verified = SQLiteHistoryRows.from_reference(previous, root=self.run_directory)
                    except (OSError, ValueError, sqlite3.DatabaseError):
                        reason = 'History database changed without a new committed revision or matching checksum'
                    else:
                        self._retain_history_reference(verified.reference(root=self.run_directory))
        if reason is None:
            return
        quarantine = path.parent / ('quarantine-' + uuid.uuid4().hex[:10])
        quarantine.mkdir()
        for candidate in (path, sidecar, *sqlite_sidecars):
            if candidate.exists():
                candidate.replace(quarantine / candidate.name)
        write_json(quarantine / 'reason.json', dict(reason=reason, previous_reference=previous))
        self._store.cache_rejections.append(dict(stage='histories', reason=reason,
                                                quarantine=str(quarantine)))

    def saved_histories(self):
        """Open a verified saved catalogue without starting or repairing jobs.

        Cursor navigation is intentionally stricter than discovery: a missing
        checkpoint or changed calculation requires an explicit fresh request.
        """
        import json
        from cascade_ranking import FinalStateRanking
        from history_store import SQLiteHistoryRows
        from madgraph_runner import validate_provenance
        self._refresh()
        message = 'Saved history catalogue is stale or unavailable; run histories without --cursor to refresh it.'
        with self._store.lock():
            try:
                manifest = json.loads(contained_path(self.run_directory, 'manifest.json').read_text())
                if manifest['physics_fingerprint'] != self._signature:
                    raise ValueError(message)
                for name in ('collider', 'decays'):
                    if public(getattr(self.config, name)) != manifest['settings'][name]:
                        raise ValueError(message)
                for name in ('memory_mb', 'max_render_nodes'):
                    if self.config.ranking[name] != manifest['settings']['ranking'][name]:
                        raise ValueError(message)
                keys = manifest['stage_fingerprints']
                saved = {}
                for stage in ('export', 'channels', 'decays'):
                    saved[stage] = self._store.load(stage, keys[stage])
                    if saved[stage] is None:
                        raise ValueError(message)
                channels, decays, ufo = saved['channels'], saved['decays'], saved['export']
                expected_export = self._key('export', tools=self._tool_fingerprint('feynrules', 'wolfram_kernel', 'madgraph', 'python'),
                                           runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py'))
                expected_decays = self._key('decays', export=expected_export,
                    artifact=self._export_identity(ufo), settings=self.config.decays,
                    runner=self._code_fingerprint('madgraph_runner.py', 'portable_runtime.py', 'selected_decay.py'),
                    tools=self._tool_fingerprint('madgraph', 'python', 'fortran', 'cxx', 'make'))
                numerical = self._numerical_key(channels, decays)
                if (keys['export'] != expected_export or keys['decays'] != expected_decays
                        or keys['histories'] != self._history_key(channels, decays)
                        or keys['rates'] != numerical):
                    raise ValueError(message)
                reference = manifest['history_store']
                rows = SQLiteHistoryRows.from_reference(reference, root=self.run_directory)
                channels.runtime = deepcopy(self.config.runtime)
                values = []
                statuses = rows.production_statuses()
                for channel in channels:
                    if statuses.get(channel.id, {}).get('status') != 'computed':
                        continue
                    stage, key = self._rate_key(channel, channels, numerical)
                    result = self._store.load(stage, key)
                    if result is None or len(result.rates) != 1:
                        raise ValueError(message)
                    validate_provenance(channels, decays, result)
                    row = dict(result.rates[0])
                    row.update(id=channel.id, process=channel.process, final_pdgs=channel.final_pdgs)
                    values.append(row)
                cross_sections = self._cross_sections(channels, decays, values)
                validate_provenance(channels, decays, cross_sections)
            except (OSError, KeyError, TypeError) as exc:
                raise ValueError(message) from exc
            self._channels, self._decays, self._ufo = channels, decays, ufo
            self._stage_keys = dict(keys)
            self.cross_sections = cross_sections
            self.ranking = FinalStateRanking(rows, {}, {}, ()).refresh()
            self.failures = list(manifest.get('failures', ()))
            self._requested_scope = manifest.get('requested_scope', 'all')
            for name in ('discovery_complete', 'requested_rates_complete', 'all_rates_complete',
                         'selected_integrations_complete', 'export_complete'):
                setattr(self, name, bool(manifest.get(name, False)))
            if self.export_complete:
                from pipeline_reports import _existing_export, _reference_identity
                exported = manifest.get('history_exports') or {}
                source = portable_history_reference(exported['source'], self.run_directory) if exported.get('source') else {}
                self.export_complete = bool(_reference_identity(source) == _reference_identity(public(rows.reference(root=self.run_directory)))
                    and _existing_export(self.run_directory, exported.get('identity')) is not None)
            self._hydrate_explicit()
            self._update_completion()
            self._add_production_notice()
            return self.ranking

    def histories(self, *, show=True):
        """Discover every history and hydrate saved rates without integrating."""
        from cascade_ranking import rank_final_states
        with self._session():
            self._capture_history_reference()
            self.export_complete = False
            self._publish('discovering')
            decays = self.decays(show=False)
            decay_key = self._stage_keys['decays']
            channels = self.channels(show=False)
            if self._stage_keys.get('decays') != decay_key:
                raise RuntimeError('Exported model or tools changed during decay/production preparation; rerun')
            key = self._history_key(channels, decays)
            path = self.run_directory / 'histories' / key[:20] / 'histories.sqlite'
            from workflow_progress import stage
            with stage('Checking saved decay histories'):
                self._verify_history_cache(path)
            try:
                self.ranking = rank_final_states(channels, decays, None,
                    stable_particles=self.inputs.get('stable_particles', ()), storage_path=path,
                    discovery=True, show=False, **self.config.ranking)
            except BaseException as exc:
                self.ranking = getattr(exc, 'partial_ranking', None)
                self.discovery_complete = False
                if self.ranking is not None:
                    try:
                        self._retain_history_reference(self.ranking.rows.reference())
                    except Exception as reference_error:
                        exc.add_note('History reference could not be written: ' + str(reference_error))
                raise
            self.discovery_complete = bool(self.ranking.rows.counts()['discovery_complete'])
            self._hydrate_rates(channels, decays, self._numerical_key(channels, decays))
            self._hydrate_explicit()
            self._retain_history_reference(self.ranking.rows.reference())
            self._publish('histories_complete')
            if show:
                self.ranking.show(min_relative_rate=self.config.ranking['min_relative_rate'])
            return self.ranking

    @staticmethod
    def _selected_channels(channels, selected):
        if selected == 'all':
            return list(channels)
        ids = {selected} if isinstance(selected, str) else set(selected)
        unknown = ids - {channel.id for channel in channels}
        if unknown:
            raise ValueError('Unknown channel IDs: ' + ', '.join(sorted(unknown)))
        return [channel for channel in channels if channel.id in ids]

    def rates(self, selected='all', *, show=True):
        """Finish discovery first, then checkpoint independent integrations."""
        from madgraph_runner import compute_cross_sections, validate_provenance
        with self._session():
            self.failures = []
            self.requested_rates_complete = self.export_complete = False
            self.complete = self.requested_complete = False
            self._requested_scope = selected
            self.histories(show=False)
            channels, decays = self._channels, self._decays
            chosen = self._selected_channels(channels, selected)
            base = self._stage_keys['rates']
            values = {row['id']: row for row in self.cross_sections.rates}
            for channel in chosen:
                if channel.id in values:
                    continue
                self.failures = [failure for failure in self.failures
                                 if not (failure.get('stage') == 'rates' and failure.get('channel') == channel.id)]
                stage, key = self._rate_key(channel, channels, base)
                self.ranking.rows.update_rates(None, statuses={channel.id: {'status': 'running'}})
                self._retain_history_reference(self.ranking.rows.reference())
                self.export_complete = False
                self._publish('integrating', selected)
                attempt = self._attempt('rates/' + channel.id, key)
                try:
                    answer = compute_cross_sections(channels, selected=[channel.id], widths=decays,
                                runtime=self.config.runtime, output_dir=attempt, show=False)
                    if len(answer.rates) != 1 or answer.rates[0]['id'] != channel.id:
                        raise RuntimeError('Integration returned an unexpected channel result')
                    validate_provenance(channels, decays, answer)
                except Exception as exc:
                    import errno
                    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}:
                        raise
                    failure = dict(stage='rates', channel=channel.id, type=type(exc).__name__,
                                   category=error_category(exc), message=str(exc))
                    self.failures.append(failure)
                    self.ranking.rows.update_rates(None, statuses={channel.id: {'status': 'failed', 'error': failure}})
                else:
                    # Resume repairs a crash between numerical and SQL commits.
                    self._store.save(stage, key, answer, attempt)
                    values[channel.id] = dict(answer.rates[0])
                    self.cross_sections = self._cross_sections(channels, decays, values.values())
                    self.ranking.rows.update_rates(answer)
                self.ranking.refresh()
                self._retain_history_reference(self.ranking.rows.reference())
                self._publish('integrating', selected)
            self.cross_sections = self._cross_sections(channels, decays, values.values())
            validate_provenance(channels, decays, self.cross_sections)
            self.ranking.refresh()
            self.requested_rates_complete = all(channel.id in values for channel in chosen)
            self.all_rates_complete = len(values) == len(channels)
            self._publish('complete' if self.all_rates_complete and self.selected_integrations_complete and not self.failures else 'partial', selected)
            if show:
                self.ranking.show(min_relative_rate=self.config.ranking['min_relative_rate'])
            return self.ranking

    def explicit_decays(self, *, show=True):
        """Integrate selected chains separately from factorized rates."""
        from selected_decay import integrate_selected_history
        with self._session():
            self.histories(show=False)
            channels, decays = self._channels, self._decays
            settings = self.config.explicit_decays
            ids = settings.get('history_ids', [])
            self.failures = [failure for failure in self.failures if failure.get('stage') != 'explicit_decays']
            self.explicit_decay_results = []
            self.selected_integrations_complete = False
            base = self._explicit_key_base()
            # Validate the full requested selection before starting a job.
            for history_id in ids:
                self._history_by_id(history_id, required=True)
            for history_id in ids:
                history = self._history_by_id(history_id, required=True)
                if history['history_key'] is None:
                    from workflow_errors import ResourceLimitError
                    raise ResourceLimitError('Selected history exceeds ranking.max_render_nodes; increase that explicit limit to integrate its full tree.')
                key = digest({'inputs': base, 'history_id': history_id, 'history_key': history['history_key']})
                stage = 'explicit-' + digest(history_id)
                result = self._store.load(stage, key)
                if result is None:
                    attempt = self._attempt('explicit_decays/' + history_id, key)
                    status_stage = 'explicit-status-' + digest(history_id)
                    self._store.save(status_stage, key, dict(history_id=history_id, status='running',
                        method='madgraph_decay_chain', cross_section_pb=None, error_pb=None))
                    self.export_complete = False
                    self._publish('selected_integrating')
                    try:
                        result = integrate_selected_history(channels, decays, history,
                            runtime=self.config.runtime, output_dir=attempt,
                            bwcutoff=settings['bwcutoff'], cut_decays=settings['cut_decays'])
                    except Exception as exc:
                        import errno
                        if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}:
                            raise
                        failure = dict(stage='explicit_decays', history_id=history_id,
                                       type=type(exc).__name__, category=error_category(exc), message=str(exc))
                        self.failures.append(failure)
                        failed = dict(history_id=history_id, status='failed', method='madgraph_decay_chain',
                                      cross_section_pb=None, error_pb=None, failure=failure)
                        self._store.save(status_stage, key, failed)
                        self.explicit_decay_results.append(failed)
                        continue
                    self._store.save(stage, key, result, attempt)
                result = dict(result)
                result.update(production_id=history['production_id'],
                              production_process=history['production_process'],
                              factorized_reference_pb=history.get('rate_pb'))
                self.explicit_decay_results.append(result)
            self.selected_integrations_complete = all(row.get('status') == 'computed' for row in self.explicit_decay_results) and len(self.explicit_decay_results) == len(ids)
            self._publish('complete' if self.all_rates_complete and self.selected_integrations_complete else 'partial')
            if show:
                for row in self.explicit_decay_results:
                    print(row['history_id'], row.get('cross_section_pb'), '+/-', row.get('error_pb'), 'pb', row['status'])
            return self.explicit_decay_results

    def _history_by_id(self, history_id, *, required=False):
        try:
            return self.ranking.rows.get(history_id)
        except KeyError:
            if required:
                raise ValueError('Unknown history ID: ' + history_id) from None
            return None

    def _explicit_key_base(self):
        settings = self.config.explicit_decays
        return self._key('explicit_decays', numerical_inputs=self._stage_keys['rates'],
                         settings={key: value for key, value in settings.items() if key != 'history_ids'},
                         code=self._code_fingerprint('selected_decay.py', 'madgraph_runner.py'))

    def _hydrate_explicit(self):
        """Browsing retains verified selected integrations without running them."""
        ids = self.config.explicit_decays.get('history_ids', [])
        self.explicit_decay_results = []
        self.failures = [failure for failure in self.failures if failure.get('stage') != 'explicit_decays']
        base = self._explicit_key_base()
        for history_id in ids:
            history = self._history_by_id(history_id)
            result = None
            if history is not None:
                key = digest({'inputs': base, 'history_id': history_id, 'history_key': history['history_key']})
                result = self._store.load('explicit-' + digest(history_id), key)
                if result is None:
                    result = self._store.load('explicit-status-' + digest(history_id), key)
            if result is not None:
                result = dict(result)
                result.update(production_id=history['production_id'],
                              production_process=history['production_process'],
                              factorized_reference_pb=history.get('rate_pb'))
                if result['status'] == 'running':
                    failure = dict(stage='explicit_decays', history_id=history_id,
                        type='InterruptedError', category='execution_failure',
                        message='Previous explicit integration ended without a committed result; rerun to retry.')
                    result.update(status='failed', cross_section_pb=None, error_pb=None, failure=failure)
                if result['status'] == 'failed' and result.get('failure'):
                    self.failures.append(result['failure'])
            else:
                result = dict(history_id=history_id, status='not_calculated', method='madgraph_decay_chain',
                              cross_section_pb=None, error_pb=None, factorized_reference_pb=None)
            self.explicit_decay_results.append(result)
        self.selected_integrations_complete = all(row['status'] == 'computed' for row in self.explicit_decay_results)

    def run(self, selected='all', *, show=False):
        """Discover, integrate requested production, then selected decay chains."""
        with self._session():
            self.rates(selected, show=show)
            if self.config.explicit_decays.get('history_ids'):
                self.explicit_decays(show=show)
            return self.ranking
