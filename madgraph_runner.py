"""Portable tree-level MadGraph calculations, with no extra Python packages.

Each calculation uses a fresh build in a temporary directory and preserves its
cards and logs under ``runs``. Generated executables are never reused on another
host. The worker blocks network connections and uses the bundled PDF grid.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence
import ast
import hashlib
import html
import itertools
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent


class ArtifactChangedError(ValueError):
    """Inputs changed during execution, rather than failing a physics check."""
    category = 'execution_failure'


DEFAULT_COLLIDER = {
    "energy_tev": 13.6, "max_final": 3, "max_tree_signatures": 250000,
    "new_particles_only": False,
    "parton_pdgs": (21, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5),
    "pdf": "nn23lo1", "seed": 1, "accuracy": 0.01,
    "ptj": 20.0, "pta": 20.0, "ptl": 10.0,
    "etaj": 5.0, "etaa": 2.5, "etal": 2.5,
    "drjj": 0.4, "drjl": 0.4, "draa": 0.4,
    "draj": 0.4, "dral": 0.4, "drll": 0.4,
    "dynamical_scale_choice": 3,
}


class MadGraphError(RuntimeError):
    """A failed calculation; the message includes its retained log."""


class ProductionResourceLimit(RuntimeError):
    category = 'resource_limit'


def _table(headers, rows):
    """Stream a readable text table from Python scripts and the terminal."""
    import csv
    writer = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    writer.writerow(headers)
    writer.writerows(rows)


@dataclass(frozen=True)
class ProductionChannel:
    id: str
    process: str
    final_pdgs: tuple[int, ...]
    diagrams: int


@dataclass
class ProductionChannels(Sequence):
    channels: tuple[ProductionChannel, ...]
    ufo: Path
    new_particles: tuple[str, ...]
    collider: dict
    parameter_card: Path
    model_digest: str
    run_directory: Path
    runtime: dict | None = field(default=None, repr=False)
    particles: dict = field(default_factory=dict)
    benchmark_digest: str = ''
    source_parameter_card: Path | None = field(default=None, repr=False)
    source_benchmark_digest: str = field(default='', repr=False)
    source_width_digest: str = field(default='', repr=False)

    def __getitem__(self, item):
        return self.channels[item]

    def __len__(self):
        return len(self.channels)

    def __repr__(self):
        return f"ProductionChannels({len(self)} channels; tree-level pp, {self.collider['energy_tev']:g} TeV)"

    def show(self, limit=20):
        rows = self.channels if limit is None else self.channels[:limit]
        _table(("ID", "Process"), [(c.id, c.process) for c in rows])
        if limit is not None and len(self) > limit:
            print(f"{len(self)} channels; show(limit=None) displays all.")
        print("Separate Born predictions; do not sum different multiplicities.")
        return self


@dataclass
class DecayResults:
    two_body: tuple[dict, ...]
    total_widths: dict[int, float]
    parameter_card: Path
    run_directory: Path
    model_digest: str
    approximation: str = "Tree-level MadWidth, adaptive through four-body decays; 1% target, not a guaranteed total-width accuracy."
    modes: tuple[dict, ...] = ()
    particles: dict = field(default_factory=dict)
    benchmark_digest: str = ''
    width_digest: str = ''
    source_parameter_card: Path | None = field(default=None, repr=False)
    source_benchmark_digest: str = field(default='', repr=False)
    ufo: Path | None = field(default=None, repr=False)
    settings: dict = field(default_factory=dict)
    omitted_probability: dict = field(default_factory=dict)
    coverage_notices: tuple[str, ...] = ()
    source_width_digest: str = field(default='', repr=False)

    def __repr__(self):
        return f"DecayResults({len(self.two_body)} two-body channels; estimated total widths in GeV)"

    def total_width_rows(self):
        """Estimated total widths, with one row per particle/conjugate pair."""
        rows, seen = [], set()
        for pdg in sorted(self.total_widths, key=lambda p: (abs(p), p < 0)):
            metadata = self.particles.get(pdg, {})
            conjugate = metadata.get('anti_pdg', -pdg)
            key = tuple(sorted({pdg, conjugate}))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(pdg=pdg, name=metadata.get('name', str(pdg)),
                             width_gev=self.total_widths[pdg],
                             omitted_probability=self.omitted_probability.get(pdg, 0.0)))
        return rows

    def show(self):
        totals = self.total_width_rows()
        has_omissions = any(row['omitted_probability'] for row in totals)
        headers = ('Particle', 'Estimated total width [GeV]') + (('Unreported BR',) if has_omissions else ())
        _table(headers, [(row['name'], f"{row['width_gev']:.6g}") +
                         ((f"{row['omitted_probability']:.3g}",) if has_omissions else ()) for row in totals])
        if self.two_body:
            _table(("Two-body decay", "Partial width [GeV]", "BR"), [(r['process'], f"{r['width_gev']:.6g}", f"{r['branching_ratio']:.6g}") for r in self.two_body])
        print(self.approximation)
        for notice in dict.fromkeys(self.coverage_notices):
            print(notice)
        if any(v == 0 for v in self.total_widths.values()):
            print("Zero calculated width does not establish stability beyond this approximation.")
        return self



@dataclass
class CrossSectionResults:
    rates: tuple[dict, ...]
    collider: dict
    particles: dict = field(default_factory=dict)
    model_digest: str = ''
    benchmark_digest: str = ''
    width_digest: str = ''

    def __repr__(self):
        return f"CrossSectionResults({len(self.rates)} separate tree-level predictions in pb)"

    def show(self):
        _table(("ID", "Process", "σ [pb]", "MC error [pb]", "Target met"), [(r['id'], r['process'], f"{r['cross_section_pb']:.6g}", f"{r['error_pb']:.3g}", r['target_met']) for r in self.rates])
        return self


def _digest(path):
    digest = hashlib.sha256()
    root = Path(path)
    for file in sorted([*root.rglob("*.py"), *root.glob('restrict*.dat')]):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _card_digest(path, *, widths=False):
    """Hash numerical SLHA content independently of formatting and comments.

    The benchmark excludes DECAY blocks so that computing widths does not
    pretend to change masses/couplings. Widths receive a separate fingerprint.
    """
    section = None
    records = []
    def canonical(token):
        try:
            return str(Decimal(token.replace('D', 'e').replace('d', 'e')).normalize())
        except InvalidOperation:
            return token.lower()
    for raw in Path(path).read_text().splitlines():
        bits = raw.split('#', 1)[0].split()
        if not bits:
            continue
        keyword = bits[0].upper()
        if keyword in ('BLOCK', 'DECAY'):
            section = keyword
            header = tuple(canonical(t) for t in bits)
            if (section == 'DECAY') == widths:
                records.append(header)
        elif section and (section == 'DECAY') == widths:
            records.append(header + tuple(canonical(t) for t in bits))
    return hashlib.sha256(repr(sorted(records)).encode()).hexdigest()


def _full_card_digest(path):
    """Fingerprint all numerical SLHA inputs, including widths and decay rows."""
    return hashlib.sha256((_card_digest(path) + ':' + _card_digest(path, widths=True)).encode()).hexdigest()


def validate_provenance(channels, decays=None, rates=None):
    """Reject edited or mixed benchmark artifacts before further calculations."""
    if _digest(channels.ufo) != channels.model_digest:
        raise ValueError("The UFO changed; list channels and recompute results")
    if _card_digest(channels.parameter_card) != channels.benchmark_digest:
        raise ValueError("The channel parameter card changed; list channels again")
    for result in (channels, decays):
        if result is None:
            continue
        source = result.source_parameter_card
        if source and _card_digest(source) != result.source_benchmark_digest:
            raise ValueError("The source benchmark changed; prepare and recompute results")
        if source and _card_digest(source, widths=True) != result.source_width_digest:
            raise ValueError("The source widths changed; prepare and recompute results")
    if decays is not None:
        if decays.model_digest != channels.model_digest or decays.benchmark_digest != channels.benchmark_digest:
            raise ValueError("Decay results and production channels use different benchmarks")
        if _card_digest(decays.parameter_card) != decays.benchmark_digest or _card_digest(decays.parameter_card, widths=True) != decays.width_digest:
            raise ValueError("The decay parameter card changed; recompute decays")
    if rates is not None:
        if rates.model_digest != channels.model_digest or rates.benchmark_digest != channels.benchmark_digest:
            raise ValueError("Cross sections and channels use different benchmarks")
        expected = decays.width_digest if decays else _card_digest(channels.parameter_card, widths=True)
        if rates.width_digest != expected:
            raise ValueError("Cross sections used different propagator widths; recompute rates")
        if rates.collider != channels.collider:
            raise ValueError("Collider settings changed; recompute rates")
    return True


def _artifact(ufo, new_particles=None, *, allow_unsealed=False):
    path = Path(getattr(ufo, "path", ufo)).expanduser().resolve()
    if not (path / "particles.py").is_file():
        raise ValueError(f"Not a UFO model directory: {path}")
    names = new_particles or getattr(ufo, "new_particles", None)
    if not names and hasattr(ufo, "particles"):
        names = [name for p in ufo.particles for name in (p['name'], p.get('antiname', p['name']))]
    if isinstance(names, str):
        names = [names]
    card = getattr(ufo, "parameter_card", None)
    if card is None and (path / "param_card.dat").is_file():
        card = path / "param_card.dat"
    card = Path(card).resolve() if card else None
    if getattr(ufo, 'benchmark_only', False):
        saved_model = getattr(ufo, 'saved_model_digest', '')
        saved_card = getattr(ufo, 'saved_parameter_card_digest', '')
        if (saved_model or saved_card) and getattr(ufo, 'parameter_values', {}):
            raise ValueError('Sealed exported benchmarks cannot override parameter_values; change physical inputs and re-export before calculating')
        if not saved_model and not saved_card and allow_unsealed:
            pass  # Initial export acceptance seals the completed card afterward.
        elif not saved_model or not saved_card or card is None:
            raise ValueError('This exported benchmark has no complete saved fingerprint; re-export before calculating')
        elif _digest(path) != saved_model or _full_card_digest(card) != saved_card:
            raise ValueError('The exported UFO or its parameter card changed after export; prepare physical inputs and re-export before calculating')
    parameters = getattr(ufo, "parameter_values", {})
    return path, tuple(dict.fromkeys(names or ())), card, parameters


def _collider(values=None):
    supplied = dict(values or {})
    unknown = supplied.keys() - DEFAULT_COLLIDER.keys()
    if unknown:
        raise ValueError(f"Unknown collider settings: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_COLLIDER, **supplied}
    if type(result['new_particles_only']) is not bool:
        raise ValueError('new_particles_only must be True or False')
    for key in ('max_final', 'max_tree_signatures'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    numeric_keys = ('energy_tev', 'accuracy', 'ptj', 'pta', 'ptl', 'etaj', 'etaa', 'etal',
                    'drjj', 'drjl', 'draa', 'draj', 'dral', 'drll')
    for key in numeric_keys:
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"{key} must be a finite number")
    if result['energy_tev'] <= 0:
        raise ValueError("energy_tev must be positive")
    if not 0 < result['accuracy'] < 1:
        raise ValueError("accuracy must be between 0 and 1")
    if isinstance(result['seed'], bool) or not isinstance(result['seed'], int) or not 1 <= result['seed'] < 30081 * 30081:
        raise ValueError("seed must be an integer from 1 through 904862560")
    if type(result['dynamical_scale_choice']) is not int or result['dynamical_scale_choice'] not in (-1, 1, 2, 3, 4):
        raise ValueError("dynamical_scale_choice must be -1, 1, 2, 3 or 4")
    for key in ('ptj', 'pta', 'ptl', 'etaj', 'etaa', 'etal', 'drjj', 'drjl', 'draa', 'draj', 'dral', 'drll'):
        if not math.isfinite(float(result[key])) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if result['pdf'] not in ('nn23lo1', 'cteq6l1'):
        raise ValueError("Use a bundled PDF: nn23lo1 or cteq6l1; external PDF installation is disabled")
    result['parton_pdgs'] = tuple(dict.fromkeys(result['parton_pdgs']))
    return result


def scale_diagnostics(collider, *, has_decay_chain=False):
    """Describe the actual LO scale prescription without changing it.

    The bundled setscales.f evaluates choices 1--3 using final external
    momenta. Resolving a resonance into daughters changes those momenta even
    when two run cards contain the same scale-choice integer.
    """
    choice = collider.get('dynamical_scale_choice', DEFAULT_COLLIDER['dynamical_scale_choice'])
    definitions = {
        -1: 'MadGraph clustering prescription applied to external states',
        1: 'sum of final external transverse energies',
        2: 'sum of final external transverse masses',
        3: 'one half of the sum of final external transverse masses',
        4: 'sqrt(s_hat), computed from the incoming parton momenta',
    }
    common_incoming = choice == 4
    note = ('Choice 4 uses the same function of incoming parton momenta for production and decay-chain calculations. '
            'It does not imply identical sampled phase-space points or numerical rates.' if common_incoming else
            'Equal dynamical_scale_choice integers do not ensure equal numerical renormalization or factorization scales: '
            'production uses external parents while a decay chain uses their external daughters. '
            'This can change PDFs and running couplings independently of the resonance window.')
    return dict(schema='madgraph-lo-scale-diagnostics-v1', dynamical_scale_choice=choice,
                definition=definitions[choice], fixed_ren_scale=False, fixed_fac_scale=False,
                scalefact=1.0, has_decay_chain=bool(has_decay_chain),
                common_incoming_prescription=common_incoming, note=note)


def _run_job(job, runtime=None, output_dir=None, timeout=7200, *, resolved_paths=None):
    from portable_runtime import environment, resolve_runtime
    from workflow_progress import stage as progress_stage
    from functools import partial
    with progress_stage('Checking MadGraph tools'):
        paths = resolved_paths if resolved_paths is not None else resolve_runtime(runtime, scope='madgraph')
    mg = Path(paths['madgraph'])
    if not (mg / 'madgraph').is_dir():
        raise MadGraphError(f"MadGraph source directory is missing: {mg}")
    run = Path(output_dir or ROOT / 'runs' / (job['action'] + '_' + uuid.uuid4().hex[:10])).resolve()
    run.mkdir(parents=True, exist_ok=False)
    # Retain the reproducible request before rewriting paths for the disposable
    # build tree. Never persist the caller's environment or executable objects.
    from pipeline_storage import write_json
    write_json(run / 'command.json', dict(schema='madgraph-step-command-v1',
               argv=[str(paths.get('python') or sys.executable), str(Path(__file__).resolve()), '--worker'],
               job=job, tools=paths))
    log = run / 'madgraph.log'
    result = None
    temporary_root = tempfile.gettempdir()
    if ' ' in temporary_root and Path('/tmp').is_dir():
        temporary_root = '/tmp'
    with tempfile.TemporaryDirectory(prefix='mgstep_', dir=temporary_root) as temporary:
        stage = Path(temporary)
        work = stage / 'work'
        work.mkdir()
        # MG writes caches/configuration beside its sources. Always use a fresh
        # scratch copy, leaving even a caller's external installation untouched.
        with progress_stage('Preparing MadGraph scratch files'):
            shutil.copytree(mg, stage / 'mg5', ignore=partial(_source_copy_ignore, source_root=mg))
        mg = stage / 'mg5'
        if job['action'] in ('decays', 'selected_history'):
            _retain_small_colored_widths(mg)
        shutil.copytree(job.pop('ufo'), stage / 'model', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pkl'))
        card = job.pop('parameter_card', None)
        if card:
            shutil.copy2(card, work / 'input_param_card.dat')
        job.update(madgraph=str(mg), work=str(work), model=str(stage / 'model'), paths={k: str(v) if v is not None else None for k, v in paths.items()})
        try:
            action_labels = {'validate': 'Checking UFO in MadGraph',
                             'list': 'Finding tree-level production channels',
                             'decays': 'Calculating decay widths',
                             'selected_history': 'Integrating selected decay history',
                             'cross_section': 'Integrating production channel'}
            with progress_stage(action_labels.get(job['action'], 'Running MadGraph'), log=log), log.open('w') as stream:
                proc = subprocess.Popen([str(paths.get('python') or sys.executable), str(Path(__file__).resolve()), '--worker'], stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, cwd=stage, env=environment(runtime), text=True, start_new_session=True)
                try:
                    proc.communicate(json.dumps(job), timeout=timeout)
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                    raise
            outcome = work / '.result'
            if outcome.is_file():
                result = json.loads(outcome.read_text())
                outcome.unlink()
            if proc.returncode or result is None:
                failure_path = work / 'worker_error.json'
                if failure_path.is_file():
                    failure = json.loads(failure_path.read_text())
                    error = MadGraphError(f"{failure['message']}. See {log}")
                    error.category = failure.get('category', 'execution_failure')
                    raise error
                raise MadGraphError(f"MadGraph calculation failed. See {log}")
        finally:
            # MG creates optional links to loop libraries even for a pure tree
            # process; absent optional targets must not discard a valid result.
            shutil.copytree(work, run, dirs_exist_ok=True, ignore=_copy_ignore)
    return result, run


def _retain_small_colored_widths(madgraph):
    """Remove two known MadGraph 3.x width suppressions in a scratch copy.

    A partonic partial width below the QCD scale is not zero. Keep the upstream
    warning and the calculated value; applicability is reported to the caller.
    Fail closed if either known source construct changes in another version.
    """
    source = Path(madgraph) / 'madgraph' / 'interface' / 'madgraph_interface.py'
    text = source.read_text()
    tree = ast.parse(text)
    guards = {
        ast.dump(ast.parse("0 < value < 0.1 and particle['color'] !=1", mode='eval').body):
            ast.dump(ast.parse('value = 0').body[0]),
        ast.dump(ast.parse("0 < BR.value * width <0.1 and particle['color'] !=1", mode='eval').body):
            ast.dump(ast.parse('continue').body[0]),
    }
    edits, found = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        guard = ast.dump(node.test)
        if guard not in guards:
            continue
        if guard in found or len(node.body) != 2 or node.orelse:
            raise MadGraphError('Unsupported MadGraph colored-width guard; source compatibility review required')
        warning, suppression = node.body
        warning_source = ast.get_source_segment(text, warning)
        if (not isinstance(warning, ast.Expr) or not isinstance(warning.value, ast.Call)
                or ast.dump(warning.value.func) != ast.dump(ast.parse('logger.warning', mode='eval').body)
                or 'lower than QCD scale:' not in warning_source
                or 'Set it to zero.' not in warning_source
                or ast.dump(suppression) != guards[guard]):
            raise MadGraphError('Unsupported MadGraph colored-width suppression; source compatibility review required')
        found.add(guard)
        edits.extend([(warning.lineno-1, warning.end_lineno, 'warning'),
                      (suppression.lineno-1, suppression.end_lineno, 'remove')])
    if len(found) != 2:
        raise MadGraphError('MadGraph does not contain the two validated colored-width guards; source compatibility review required')
    lines = text.splitlines(keepends=True)
    for start, end, kind in sorted(edits, reverse=True):
        lines[start:end] = ([] if kind == 'remove' else
                            [line.replace('Set it to zero.', 'Retained perturbative width; hadronization may matter.') for line in lines[start:end]])
    updated = ''.join(lines)
    compile(updated, str(source), 'exec')
    source.write_text(updated)


def _copy_ignore(directory, names):
    """Skip broken optional links while retaining valid relative source links.

    shutil's ignore_dangling_symlinks option tests relative link targets against
    the working directory, which can incorrectly omit MadGraph's eepdf.inc.
    """
    skipped = set(shutil.ignore_patterns('__pycache__', '*.pyc', '*.pkl', '.git')(directory, names))
    for name in names:
        path = Path(directory) / name
        if path.is_symlink() and not path.exists():
            skipped.add(name)
    return skipped


def _source_copy_ignore(directory, names, *, source_root=None):
    """Build from source even when an external installation contains binaries."""
    skipped = _copy_ignore(directory, names)
    if source_root is not None:
        from portable_runtime import source_tree_exclusions
        skipped.update(source_tree_exclusions('madgraph', source_root, directory, names))
    native_headers = {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
                      b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'}
    for name in names:
        path = Path(directory) / name
        if name in skipped or not path.is_file():
            continue
        if path.suffix.lower() in {'.o', '.obj', '.a', '.so', '.dylib', '.dll', '.exe', '.mod', '.smod'} or '.so.' in name:
            skipped.add(name)
            continue
        with path.open('rb') as stream:
            header = stream.read(4)
        if header in native_headers or header[:2] == b'MZ':
            skipped.add(name)
    return skipped


def validate_ufo(ufo, runtime=None, output_dir=None):
    """Import a UFO in MadGraph and evaluate every parameter and coupling.

    This fresh, offline worker catches dependencies that a plain Python import
    cannot evaluate. No amplitudes are generated. Return physical metadata and
    the completed numerical parameter card alongside the retained import log.
    """
    path, names, card, parameters = _artifact(ufo, allow_unsealed=True)
    model_digest, source_digest = _digest(path), _card_digest(card) if card else ''
    source_width_digest = _card_digest(card, widths=True) if card else ''
    result, run = _run_job(dict(action='validate', ufo=str(path), new_particles=names,
                               parameter_card=str(card) if card else None,
                               parameters=parameters), runtime, output_dir)
    if _digest(path) != model_digest or (card and (_card_digest(card) != source_digest or _card_digest(card, widths=True) != source_width_digest)):
        raise ArtifactChangedError('The source UFO or benchmark changed during validation; rerun')
    return dict(particles={int(k): v for k, v in result['particles'].items()},
                parameter_count=result['parameter_count'], coupling_count=result['coupling_count'],
                parameter_card=run / 'param_card.dat', run_directory=run,
                model_digest=model_digest, benchmark_digest=_card_digest(run / 'param_card.dat'))


def list_production_channels(ufo, collider=None, runtime=None, *, new_particles=None, output_dir=None, show=True):
    """List connected tree channels with at least one specified new particle.

    A plain UFO path also needs ``new_particles=['x', 'y']``. The exported model
    artifact supplies these names automatically. Initial partons are summed for
    each exact final state, including all coupling orders in the model.
    ``collider={'new_particles_only': True}`` excludes associated SM final
    particles before diagram generation, while retaining internal SM exchanges.
    """
    path, names, card, parameters = _artifact(ufo, new_particles)
    if not names:
        raise ValueError("Specify new_particles when passing a UFO directory instead of an exported model")
    settings = _collider(collider)
    model_digest, source_digest = _digest(path), _card_digest(card) if card else ''
    source_width_digest = _card_digest(card, widths=True) if card else ''
    result, run = _run_job(dict(action='list', ufo=str(path), new_particles=names, parameter_card=str(card) if card else None, parameters=parameters, collider=settings), runtime, output_dir)
    if _digest(path) != model_digest or (card and (_card_digest(card) != source_digest or _card_digest(card, widths=True) != source_width_digest)):
        raise ArtifactChangedError('The source UFO or benchmark changed during channel generation; rerun')
    channels = ProductionChannels(tuple(ProductionChannel(f'C{i:04d}', row['process'], tuple(row['final_pdgs']), row['diagrams']) for i, row in enumerate(result['channels'], 1)), path, names, settings, run / 'param_card.dat', model_digest, run, runtime,
                                  {int(k): v for k, v in result['particles'].items()}, _card_digest(run / 'param_card.dat'), card, source_digest, source_width_digest)
    if show:
        channels.show()
    return channels


def compute_decays(ufo, inputs=None, runtime=None, *, new_particles=None, output_dir=None, max_body=4, precision=0.01, min_br=1e-8, show=True):
    """Compute two-body partial widths and adaptive total widths up to four bodies.

    ``inputs`` optionally contains ``parameter_card`` and/or ``parameters`` (UFO
    external parameter names). Other keys are rejected rather than ignored.
    The exported artifact already carries its chosen numerical benchmark.
    ``min_br`` filters reported modes only; the full computed SLHA table and
    total widths remain unchanged. Adaptive/body truncation is separate.
    """
    path, names, card, parameters = _artifact(ufo, new_particles)
    if not names:
        raise ValueError("Specify new_particles for a plain UFO directory")
    inputs = dict(inputs or {})
    if inputs.keys() - {'parameter_card', 'parameters'}:
        raise ValueError("Decay inputs accept only parameter_card and parameters; prepare physical inputs before UFO export")
    if getattr(ufo, 'benchmark_only', False) and inputs:
        raise ValueError("This UFO is tied to its prepared benchmark. Change physical inputs and export again; independent parameter/card overrides could invalidate its field transformations.")
    card = inputs.get('parameter_card', card)
    card = Path(card).expanduser().resolve() if card else None
    parameters = {**parameters, **inputs.get('parameters', {})}
    if max_body not in (2, 3, 4) or isinstance(max_body, bool):
        raise ValueError('max_body must be 2, 3, or 4')
    if not 0 < precision < 1 or not 0 < min_br < 1:
        raise ValueError('precision and min_br must lie strictly between 0 and 1')
    settings = dict(max_body=max_body, precision=precision, min_br=min_br)
    model_digest, source_digest = _digest(path), _card_digest(card) if card else ''
    source_width_digest = _card_digest(card, widths=True) if card else ''
    result, run = _run_job(dict(action='decays', ufo=str(path), new_particles=names, parameter_card=str(card) if card else None, parameters=parameters, decay_settings=settings), runtime, output_dir)
    if _digest(path) != model_digest or (card and (_card_digest(card) != source_digest or _card_digest(card, widths=True) != source_width_digest)):
        raise ArtifactChangedError('The source UFO or benchmark changed during decay calculation; rerun')
    decay = DecayResults(tuple(result['two_body']), {int(k): v for k, v in result['total_widths'].items()}, run / 'param_card.dat', run, model_digest,
                         f"Tree-level MadWidth through {max_body}-body decays; {100*precision:g}% adaptive target. Reported modes require BR ≥ {min_br:g}, without renormalizing their BRs or total widths. Unknown adaptive/body tails remain; total-width accuracy is not guaranteed.",
                         tuple(result['modes']), {int(k): v for k, v in result['particles'].items()}, _card_digest(run / 'param_card.dat'), _card_digest(run / 'param_card.dat', widths=True),
                         Path(card) if card else None, source_digest, path, settings,
                         {int(k): v for k, v in result['omitted_probability'].items()}, tuple(result['coverage_notices']), source_width_digest)
    if show:
        decay.show()
    return decay


def compute_cross_sections(channels, selected='all', widths=None, runtime=None, *, output_dir=None, show=True):
    """Integrate selected channel IDs; each prediction has its own MC error.

    Pass ``widths=compute_decays(ufo)`` to use computed propagator widths. Without
    this argument, use the numerical widths in the model's parameter card.
    """
    if not isinstance(channels, ProductionChannels):
        raise TypeError("channels must be returned by list_production_channels")
    validate_provenance(channels, widths)
    if selected == 'all':
        chosen = list(channels)
    else:
        ids = {selected} if isinstance(selected, str) else set(selected)
        unknown = ids - {c.id for c in channels}
        if unknown:
            raise ValueError(f"Unknown channel IDs: {', '.join(sorted(unknown))}")
        chosen = [c for c in channels if c.id in ids]
    rates = []
    collider_snapshot = dict(channels.collider)
    from portable_runtime import resolve_runtime
    paths = resolve_runtime(runtime or channels.runtime, scope='madgraph') if chosen else None
    for channel in chosen:
        validate_provenance(channels, widths)
        if channels.collider != collider_snapshot:
            raise ArtifactChangedError('Collider settings changed during the calculation; rerun')
        destination = Path(output_dir) / channel.id if output_dir else None
        result, run = _run_job(dict(action='cross_section', ufo=str(channels.ufo), parameter_card=str(widths.parameter_card if widths else channels.parameter_card), parameters={}, collider=channels.collider, process=channel.process), runtime or channels.runtime, destination, resolved_paths=paths)
        validate_provenance(channels, widths)
        if channels.collider != collider_snapshot:
            raise ArtifactChangedError('Collider settings changed during the calculation; rerun')
        result.update(id=channel.id, process=channel.process, final_pdgs=channel.final_pdgs,
                      run_directory=run, status='computed', scale_diagnostics=scale_diagnostics(channels.collider))
        rates.append(result)
    answer = CrossSectionResults(tuple(rates), dict(channels.collider), channels.particles, channels.model_digest, channels.benchmark_digest,
                                 widths.width_digest if widths else _card_digest(channels.parameter_card, widths=True))
    if show:
        answer.show()
    return answer


def _tree_signatures(vertices, anti, max_legs=5, max_signatures=250000):
    """All connected tree external multisets up to max_legs, for vertices ≥3.

    Glue one elementary vertex at a time. An n-leg tree with vertices of valence
    at least three uses at most n-2 vertices; recursive growth therefore ends.
    This is a conservative structural prefilter, not a matrix-element engine.
    """
    elementary = {tuple(sorted(v)) for v in vertices if 3 <= len(v) <= max_legs}
    if len(elementary) > max_signatures:
        raise ProductionResourceLimit(f'Production tree enumeration exceeds collider.max_tree_signatures={max_signatures}')
    total = len(elementary)
    by_size = defaultdict(set)
    for vertex in elementary:
        by_size[len(vertex)].add(vertex)
    for size in range(4, max_legs + 1):
        for left_size in range(3, size):
            right_size = size - left_size + 2
            for left in tuple(by_size[left_size]):
                for right in (v for v in elementary if len(v) == right_size):
                    for particle in set(left):
                        conjugate = anti[particle]
                        if conjugate in right:
                            count = Counter(left) + Counter(right)
                            count[particle] -= 1
                            count[conjugate] -= 1
                            signature = tuple(sorted(count.elements()))
                            if signature not in by_size[size]:
                                if total >= max_signatures:
                                    raise ProductionResourceLimit(f'Production tree enumeration exceeds collider.max_tree_signatures={max_signatures} at {size} external legs')
                                by_size[size].add(signature)
                                total += 1
    return set().union(*by_size.values()) if by_size else set()


def _candidate_finals(vertices, anti, partons, new_pdgs, max_final=3, max_tree_signatures=250000,
                      *, new_particles_only=False):
    # This restriction concerns outgoing production particles only. Retain all
    # vertices when building connected trees, including internal SM exchanges.
    new = set(new_pdgs)
    if new_particles_only:
        new |= {anti[p] for p in new}
    finals = set()
    for external in _tree_signatures(vertices, anti, max_final + 2, max_tree_signatures):
        for initial in itertools.combinations_with_replacement(partons, 2):
            count = Counter(external)
            incoming = Counter(anti[p] for p in initial)
            if any(count[p] < n for p, n in incoming.items()):
                continue
            rest = count - incoming
            if set(rest) & new and (not new_particles_only or set(rest) <= new):
                final = tuple(sorted(rest.elements()))
                if 1 <= len(final) <= max_final:
                    finals.add(final)
    return sorted(finals, key=lambda f: (len(f), f))


def _read_decays(path):
    """Parse standard SLHA DECAY blocks, keeping both partial and total widths."""
    answer = {}
    current = None
    for raw in Path(path).read_text().splitlines():
        bits = raw.split('#', 1)[0].split()
        if not bits:
            continue
        if bits[0].upper() == 'DECAY':
            current = int(bits[1])
            answer[current] = {'total': float(bits[2].replace('D', 'e').replace('d', 'e')), 'channels': []}
        elif bits[0].upper() == 'BLOCK':
            current = None
        elif current is not None:
            br, number = float(bits[0].replace('D', 'e').replace('d', 'e')), int(bits[1])
            daughters = tuple(int(p) for p in bits[2:])
            if len(daughters) != number:
                raise ValueError(f"Malformed decay table in {path}")
            answer[current]['channels'].append((daughters, br * answer[current]['total']))
    return answer


def _reported_decays(totals, parents, new_pdgs, physical, anti, metadata, min_br):
    """Filter reporting without modifying total widths or renormalizing BRs."""
    modes, widths, omitted, notices = [], {}, {}, []
    for pdg in sorted(parents):
        total = totals[pdg]['total']
        signed_parents = [pdg] + ([anti[pdg]] if anti[pdg] != pdg else [])
        for parent in signed_parents:
            widths[parent], omitted[parent] = total, 0.0
        for daughters, width in totals[pdg]['channels']:
            br = width / total if total else 0.0
            if abs(metadata[pdg]['color']) != 1 and 0 < width < 0.1:
                notices.append(f"{physical[pdg]} > {' '.join(physical[p] for p in daughters)}: retained perturbative partial width {width:g} GeV below the QCD scale; hadronization may matter.")
            for parent in signed_parents:
                if br < min_br:
                    omitted[parent] += br
                    continue
                children = daughters if parent == pdg else tuple(anti[p] for p in daughters)
                modes.append(dict(pdg=parent, daughters=children,
                                  process=physical[parent] + ' > ' + ' '.join(physical[p] for p in children),
                                  width_gev=width, branching_ratio=br, conjugated=parent != pdg))
    two_body = [row for row in modes if row['pdg'] in new_pdgs and len(row['daughters']) == 2]
    if any(omitted.values()):
        notices.append('Reported decay rows omit modes below min_br; omitted_probability records their summed BR for each parent. The SLHA card and total widths retain those modes.')
    notices.append('Adaptive precision and the maximum body multiplicity can leave uncomputed contributions. Their probability is unknown; reported BRs use the computed total, not a guaranteed physical total.')
    return dict(two_body=two_body, total_widths=widths, modes=modes, particles=metadata,
                omitted_probability=omitted, coverage_notices=notices)


def _worker(job):
    # Deny network access even if an upstream optional feature tries to use it.
    import socket
    def deny_network(*args, **kwargs):
        raise RuntimeError("Network access is disabled for portable MadGraph calculations")
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    os.environ.update(LC_ALL='C', LANG='C', HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9')
    sys.path.insert(0, job['madgraph'])
    import logging
    logging.basicConfig(level=logging.INFO)
    from madgraph.interface.master_interface import MasterCmd
    from models.check_param_card import ParamCard
    work = Path(job['work'])
    os.chdir(work)
    command = MasterCmd()
    options = dict(automatic_html_opening=False, automatic_update=0, notification_center=False, run_mode=0, nb_core=1, crash_on_error=True, fortran_compiler=job['paths'].get('fortran'), cpp_compiler=job['paths'].get('cxx'))
    command.options.update(options)
    command.exec_cmd(f"import model {job['model']} -modelname", printcmd=False)
    model = command._curr_model
    card = work / 'param_card.dat'
    if (work / 'input_param_card.dat').is_file():
        shutil.copy2(work / 'input_param_card.dat', card)
    else:
        model.write_param_card(str(card))
    if job.get('parameters'):
        parsed = ParamCard(str(card))
        externals = {p.name: p for p in model['parameters'].get(('external',), [])}
        for name, value in job['parameters'].items():
            if name not in externals:
                raise ValueError(f"Unknown external UFO parameter: {name}")
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"Non-finite external parameter: {name}")
            parameter = externals[name]
            parsed[parameter.lhablock.lower()].get(tuple(parameter.lhacode)).value = float(value)
        parsed.write(str(card))
    physical = {}
    anti = {}
    for particle in model['particles']:
        if particle['spin'] < 1 or particle.get('ghost') or particle.get('goldstone'):
            continue
        pdg = particle.get_pdg_code()
        physical[pdg] = particle['name']
        antipdg = pdg if particle['self_antipart'] else -pdg
        physical[antipdg] = particle['antiname']
        anti[pdg], anti[antipdg] = antipdg, pdg
    name_to_pdg = {name: pdg for pdg, name in physical.items()}
    requested = job.get('new_particles', ())
    if any(name not in name_to_pdg for name in requested):
        raise ValueError(f"Unknown new particle names; available: {', '.join(sorted(name_to_pdg))}")
    new = {name_to_pdg[name] for name in requested}
    new |= {anti[p] for p in new}
    from models.model_reader import ModelReader
    reader = ModelReader(model)
    numeric = reader.set_parameters_and_couplings(str(card))
    for kind in ('parameter_dict', 'coupling_dict'):
        for name, value in reader.get(kind).items():
            number = complex(value)
            if not math.isfinite(number.real) or not math.isfinite(number.imag):
                raise ValueError(f"Non-finite numerical UFO {kind}: {name}={value}")
    metadata = {}
    for pdg, name in physical.items():
        particle = model.get_particle(pdg)
        mass_name = str(particle['mass'])
        mass = abs(complex(numeric[mass_name])) if mass_name in numeric else abs(float(mass_name)) if mass_name.lower() != 'zero' else 0.0
        metadata[pdg] = dict(name=name, antiname=physical[anti[pdg]], aliases=[name], anti_pdg=anti[pdg], pdg=pdg, mass_gev=mass,
                             spin=particle['spin'], color=particle.get_color(), charge=particle.get_charge(), is_new=pdg in new)
    # FeynRules' UFO writer omits dependent masses (notably MW), whereas
    # MadEvent's Fortran card reader needs a MASS entry for every particle.
    # Fill these from the evaluated UFO, never from a guessed physical value.
    completed = ParamCard(str(card))
    for pdg in sorted({abs(p) for p in physical}):
        if 'mass' not in completed or (pdg,) not in completed['mass'].param_dict:
            completed.add_param('mass', [pdg], metadata[pdg]['mass_gev'], 'derived from UFO')
        if 'decay' not in completed or (pdg,) not in completed['decay'].param_dict:
            width_name = str(model.get_particle(pdg)['width'])
            width = float(complex(numeric.get(width_name, 0.0)).real)
            completed.add_param('decay', [pdg], width, 'from UFO')
    completed.write(str(card))
    if job['action'] == 'validate':
        if any(len(v['particles']) < 3 for v in model['interactions']
               if all(p.get_pdg_code() in physical for p in v['particles'])):
            raise ValueError("UFO contains two-point insertions; normalize and diagonalize the physical model before export")
        for pdg in new:
            if str(model.get_particle(pdg)['width']).strip().upper() in ('ZERO', '0', '0.0'):
                raise ValueError(f"New particle {physical[pdg]} has literal-zero UFO width; use an external DECAY parameter so MadWidth can calculate its decays")
        return dict(particles=metadata, parameter_count=len(reader.get('parameter_dict')),
                    coupling_count=len(reader.get('coupling_dict')))
    # Enumerate all independent coupling orders explicitly, overriding MG's
    # default preference for low weighted order (QCD versus electroweak).
    order_names = sorted({key for vertex in model['interactions'] for key in vertex['orders']})
    vertex_bound = job.get('collider', {}).get('max_final', 3)
    if job['action'] == 'selected_history':
        from selected_decay import canonical_forest, primary_keys
        forest = canonical_forest(job['history_key'])
        # A trivalent tree has at most N_out vertices for 2->N_out, and
        # N_out-1 for a decay. Production bounds must not clip a longer decay.
        vertex_bound = max([len(forest)] + [len(daughters) - 1
                            for _, daughters in primary_keys(forest)])
    order_bound = max([abs(v) for vertex in model['interactions'] for v in vertex['orders'].values()] or [1]) * vertex_bound
    orders = ' '.join(f'{name}<={order_bound}' for name in order_names)
    if job['action'] == 'list':
        partons = job['collider']['parton_pdgs']
        if set(partons) - physical.keys():
            raise ValueError("The UFO does not contain all requested proton partons")
        command.exec_cmd('define p = ' + ' '.join(physical[p] for p in partons))
        vertices = [tuple(p.get_pdg_code() for p in v['particles']) for v in model['interactions'] if all(p.get_pdg_code() in physical for p in v['particles'])]
        if any(len(v) < 3 for v in vertices):
            raise ValueError("UFO contains two-point insertions; normalize and diagonalize the physical model before production")
        candidates = _candidate_finals(vertices, anti, partons, new, job['collider']['max_final'], job['collider']['max_tree_signatures'],
                                        new_particles_only=job['collider'].get('new_particles_only', False))
        channels = []
        from madgraph.core.diagram_generation import NoDiagramException
        for final in candidates:
            process = 'p p > ' + ' '.join(physical[p] for p in final)
            try:
                # A failed generation must never leave a previous channel's
                # amplitudes available for inspection.
                command._curr_amps = []
                command.exec_cmd('generate ' + process + ' ' + orders, printcmd=False)
            except NoDiagramException:
                continue
            diagrams = sum(len(a['diagrams']) for a in command._curr_amps)
            if diagrams:
                channels.append(dict(process=process, final_pdgs=final, diagrams=diagrams))
        return dict(channels=channels, particles=metadata)
    if job['action'] == 'decays':
        pdgs = sorted({abs(p) for p in new})
        for pdg in pdgs:
            width_parameter = str(model.get_particle(pdg)['width']).strip().upper()
            if width_parameter in ('ZERO', '0', '0.0'):
                raise ValueError(f"New particle {physical[pdg]} has literal-zero UFO width. MadWidth treats it as stable and skips its decays. Re-export with an external DECAY width parameter, whose initial value may be zero.")
        settings = job['decay_settings']
        unstable = set(pdgs) | ({6, 23, 24, 25} & physical.keys())
        parents = ' '.join(map(str, sorted(unstable)))
        # MG's min_br drops amplitudes BEFORE normalizing its total width. A
        # literal zero selects an upstream default, so use a tiny positive value.
        # The user's threshold is applied only to our reported rows below.
        common = f" --precision_channel={settings['precision']} --min_br=1e-300 --path=param_card.dat"
        command.exec_cmd('compute_widths ' + parents + ' --body_decay=2' + common + ' --output=two_body_param_card.dat')
        level = 2 if settings['max_body'] == 2 else settings['max_body'] + settings['precision']
        if settings['max_body'] == 2:
            shutil.copy2(work / 'two_body_param_card.dat', work / 'total_param_card.dat')
        else:
            command.exec_cmd('compute_widths ' + parents + f' --body_decay={level}' + common + ' --output=total_param_card.dat')
        totals = _read_decays(work / 'total_param_card.dat')
        shutil.copy2(work / 'total_param_card.dat', card)
        result = _reported_decays(totals, unstable, set(pdgs), physical, anti, metadata, settings['min_br'])
        from selected_decay import capture_primary_diagrams, primary_provenance
        capture_primary_diagrams(command, result['modes'], primary_provenance(job['madgraph'], settings))
        return result
    if job['action'] in ('cross_section', 'selected_history'):
        settings = job['collider']
        command.exec_cmd('define p = ' + ' '.join(physical[p] for p in settings['parton_pdgs']))
        generated_process = job['process'] + ' ' + orders
        if job['action'] == 'selected_history':
            from selected_decay import canonical_forest, render_history
            generated_process = render_history(canonical_forest(job['history_key']), physical, orders)
        command.exec_cmd('generate ' + generated_process)
        primary_validation = None
        if job['action'] == 'selected_history':
            from selected_decay import canonical_forest, restrict_primary_amplitudes, primary_provenance
            primary_validation = restrict_primary_amplitudes(command, card,
                canonical_forest(job['history_key']), job['primary_modes'],
                primary_provenance(job['madgraph'], job['decay_settings']))
            (work / 'primary_diagrams.json').write_text(json.dumps(primary_validation, indent=2))
        command.exec_cmd('output madevent process -f')
        process = work / 'process'
        shutil.copy2(card, process / 'Cards' / 'param_card.dat')
        from madgraph.various.banner import RunCardLO
        run_card = RunCardLO(str(process / 'Cards' / 'run_card.dat'))
        run_values = dict(lpp1=1, lpp2=1, ebeam1=settings['energy_tev'] * 500, ebeam2=settings['energy_tev'] * 500, pdlabel=settings['pdf'], iseed=settings['seed'], use_syst=False, systematics_program='none', fixed_ren_scale=False, fixed_fac_scale=False, scalefact=1.0, nevents=0)
        for key in ('ptj', 'pta', 'ptl', 'etaj', 'etaa', 'etal', 'drjj', 'drjl', 'draa', 'draj', 'dral', 'drll', 'dynamical_scale_choice'):
            run_values[key] = settings[key]
        if job['action'] == 'selected_history':
            run_values.update(job['decay_chain_settings'])
        for key, value in run_values.items():
            run_card[key] = value
        run_card.write(str(process / 'Cards' / 'run_card.dat'))
        from madgraph.interface.madevent_interface import MadEventCmd
        event = MadEventCmd(str(process), options=command.options)
        event.force = True
        event.exec_cmd(f"survey run_01 --accuracy={settings['accuracy']} --points=1000 --iterations=5")
        if event.results.current['cross']:
            event.exec_cmd(f"refine {settings['accuracy']}")
        value, error = float(event.results.current['cross']), float(event.results.current['error'])
        if not math.isfinite(value) or not math.isfinite(error) or value < 0 or error < 0:
            raise ValueError('MadGraph returned an invalid cross section or integration error')
        (work / 'build_host.txt').write_text(f"{platform.system()} {platform.machine()}\nPython {sys.version}\nFortran {job['paths'].get('fortran')}\n")
        result = dict(cross_section_pb=value, error_pb=error, target_met=bool(value and error / abs(value) <= settings['accuracy']), width_source='parameter_card.dat', units='pb',
                      scale_diagnostics=scale_diagnostics(settings, has_decay_chain=bool(primary_validation)))
        if primary_validation is not None:
            result['primary_diagram_validation'] = primary_validation
            result['generated_process'] = generated_process
        return result
    raise ValueError(f"Unknown job action: {job['action']}")


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    payload = json.load(sys.stdin)
    try:
        value = _worker(payload)
    except Exception as exc:
        (Path(payload['work']) / 'worker_error.json').write_text(json.dumps(
            dict(type=type(exc).__name__, message=str(exc),
                 category=getattr(exc, 'category', 'execution_failure'))))
        raise
    (Path(payload['work']) / '.result').write_text(json.dumps(value))
