"""Read portable INI settings without installing tools or changing the host."""
from dataclasses import dataclass, field
from pathlib import Path
import configparser
import math

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    runtime: dict
    model_file: Path
    output: Path
    collider: dict
    decays: dict
    ranking: dict
    explicit_decays: dict = field(default_factory=dict)
    # None means this object was constructed by a legacy caller rather than
    # load_config. File-derived choices must remain refreshable in a live API.
    input_overrides: dict | None = field(default=None, compare=False, repr=False)


DECAY_DEFAULTS = {"max_body": 4, "precision": .01, "min_br": 1e-8}
RANKING_DEFAULTS = {"min_relative_rate": 1e-4, "max_width_mass_ratio": .01,
                    "threshold_widths": 5., "max_depth": 12,
                    "batch_size": 1000, "memory_mb": 64, "max_render_nodes": 10000}
EXPLICIT_DECAY_DEFAULTS = {"history_ids": [], "bwcutoff": 15., "cut_decays": False}


def explicit_decay_settings(supplied=None):
    """Selected matrix-element integrations are separate from factorized rates."""
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict):
        raise ValueError("explicit_decays must be a dictionary")
    unknown = set(supplied) - set(EXPLICIT_DECAY_DEFAULTS)
    if unknown:
        raise ValueError("Unknown explicit_decays keys: " + ", ".join(sorted(unknown)))
    values = {**EXPLICIT_DECAY_DEFAULTS, **supplied}
    ids = values['history_ids']
    if not isinstance(ids, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("explicit_decays.history_ids must list nonempty history IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("explicit_decays.history_ids contains duplicate IDs")
    values['history_ids'] = list(ids)
    if not isinstance(values['cut_decays'], bool):
        raise ValueError("explicit_decays.cut_decays must be True or False")
    try:
        if isinstance(values['bwcutoff'], bool):
            raise ValueError()
        values['bwcutoff'] = float(values['bwcutoff'])
        if not math.isfinite(values['bwcutoff']) or values['bwcutoff'] <= 0:
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("explicit_decays.bwcutoff must be finite and positive") from exc
    return values


def _numbers(section, supplied, defaults, integer_keys=()):
    if not isinstance(supplied, dict):
        raise ValueError(f"{section} must be a dictionary")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {section} keys: {', '.join(sorted(unknown))}")
    values = dict(defaults)
    for key, raw in supplied.items():
        try:
            if isinstance(raw, bool):
                raise ValueError()
            value = int(raw) if key in integer_keys else float(raw)
            if key in integer_keys and float(raw) != value:
                raise ValueError()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"[{section}] {key} must be {'an integer' if key in integer_keys else 'numeric'}") from exc
        if not math.isfinite(value):
            raise ValueError(f"[{section}] {key} must be finite")
        values[key] = value
    return values


def calculation_settings(collider=None, decays=None, ranking=None):
    """Validate file and API settings through one shared set of bounds."""
    from madgraph_runner import _collider
    for label, supplied in (("collider", collider), ("decays", decays), ("ranking", ranking)):
        if supplied is not None and not isinstance(supplied, dict):
            raise ValueError(f"{label} must be a dictionary")
    collider = _collider(collider)
    decays = _numbers("decays", {} if decays is None else decays, DECAY_DEFAULTS, {"max_body"})
    if decays["max_body"] not in (2, 3, 4) or not 0 < decays["precision"] < 1 or not 0 < decays["min_br"] < 1:
        raise ValueError("Decay max_body must be 2–4; precision and min_br must be between zero and one.")
    # Old inputs remain readable, but a row count is no longer a truncation
    # policy. The disk-backed catalogue bounds working memory instead.
    supplied_ranking = {} if ranking is None else dict(ranking)
    legacy = supplied_ranking.pop('max_histories', None)
    ranking = _numbers("ranking", supplied_ranking, RANKING_DEFAULTS,
                       {"max_depth", "batch_size", "memory_mb", "max_render_nodes"})
    if legacy is not None:
        legacy = _numbers('ranking', {'max_histories': legacy}, {'max_histories': 100000}, {'max_histories'})['max_histories']
        if legacy <= 0:
            raise ValueError("ranking.max_histories must be positive when present")
        import warnings
        warnings.warn("ranking.max_histories is deprecated and ignored; histories are streamed to disk.",
                      DeprecationWarning, stacklevel=2)
    if not 0 <= ranking["min_relative_rate"] <= 1 or any(ranking[k] <= 0 for k in ranking if k != "min_relative_rate"):
        raise ValueError("Invalid ranking bounds: relative cutoff must be 0–1; other bounds must be positive.")
    return collider, decays, ranking


def load_config(path=None, *, model_file=None, output=None, read_model=True):
    from madgraph_runner import DEFAULT_COLLIDER, _collider
    explicit_path = path is not None
    path = Path(path or ROOT / ".config").expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path) and explicit_path:
        raise FileNotFoundError(f"Configuration not found: {path}")
    allowed = {
        "tools": {"mathematica", "madgraph", "feynrules", "fortran", "cxx", "make", "bash", "perl"},
        "project": {"model", "output"},
        "collider": set(DEFAULT_COLLIDER) - {"parton_pdgs"},
        "decays": {"max_body", "precision", "min_br"},
        "ranking": set(RANKING_DEFAULTS) | {"max_histories"},
        "explicit_decays": set(EXPLICIT_DECAY_DEFAULTS),
    }
    for section in parser:
        if section == "DEFAULT":
            if parser.defaults():
                raise ValueError("Use explicit .config sections rather than DEFAULT values.")
            continue
        if section not in allowed:
            raise ValueError(f"Unknown .config section: {section}")
        unknown = set(parser[section]) - allowed[section]
        if unknown:
            raise ValueError(f"Unknown .config keys in [{section}]: {', '.join(sorted(unknown))}")

    def file_path(value):
        result = Path(value).expanduser()
        return (result if result.is_absolute() else path.parent / result).resolve()

    runtime = {}
    for key, value in parser.items("tools") if parser.has_section("tools") else ():
        value = value.strip()
        if not value:
            continue
        destination = "wolfram_kernel" if key == "mathematica" else key
        runtime[destination] = str(file_path(value)) if key in {"madgraph", "feynrules"} or "/" in value or "\\" in value else value
    # A missing bundle is not a configured override: allow host discovery to
    # find a PATH installation. Explicit stale user paths remain errors.
    from portable_runtime import normalize_tool_path
    for tool, folder in (("madgraph", "MG5_aMC"), ("feynrules", "FeynRules")):
        if tool not in runtime:
            bundled = normalize_tool_path(tool, ROOT / "vendor" / folder)
            if bundled:
                runtime[tool] = bundled

    collider = {}
    for key, value in parser.items("collider") if parser.has_section("collider") else ():
        if key == 'new_particles_only':
            try:
                collider[key] = parser.getboolean('collider', key)
            except ValueError as exc:
                raise ValueError('[collider] new_particles_only must be a boolean (true/false, yes/no, on/off or 1/0)') from exc
        else:
            collider[key] = value if key == "pdf" else int(value) if key in {"max_final", "max_tree_signatures", "seed", "dynamical_scale_choice"} else float(value)
    decays = dict(parser.items("decays")) if parser.has_section("decays") else {}
    ranking = dict(parser.items("ranking")) if parser.has_section("ranking") else {}
    explicit_decays = {}
    if parser.has_section('explicit_decays'):
        for key, value in parser.items('explicit_decays'):
            if key == 'history_ids':
                explicit_decays[key] = [item.strip() for item in value.split(',') if item.strip()]
            elif key == 'cut_decays':
                explicit_decays[key] = parser.getboolean('explicit_decays', key)
            else:
                explicit_decays[key] = value
    configured_model = parser.get("project", "model", fallback=None)
    if model_file is not None:
        model = Path(model_file).expanduser().resolve()
    elif configured_model is not None:
        model = file_path(configured_model)
    else:
        sample = Path("input/sample_model_sextet.txt")
        model = file_path(sample)
        if not model.is_file():
            legacy = file_path("input/my_model.txt")
            model = legacy if legacy.is_file() else ROOT / sample
    destination = file_path(parser.get("project", "output", fallback="runs"))
    if read_model and model.suffix.lower() in {".txt", ".tex"}:
        from tex_inputs import read_inputs
        source = read_inputs(model)
        # A declarative document owns ALL physics settings, including defaults. Legacy
        # INI physics settings must not silently alter the saved benchmark.
        collider, decays, ranking = source.collider, source.decays, source.ranking
        explicit_decays = getattr(source, 'explicit_decays', {})
        destination = Path(source.output or "output").expanduser()
        if not destination.is_absolute():
            destination = model.parent / destination
    if output is not None:
        destination = Path(output).expanduser()  # Explicit CLI/API paths use cwd.
    collider, decays, ranking = calculation_settings(collider, decays, ranking)
    return PipelineConfig(path, runtime, model, destination.resolve(), collider, decays, ranking,
                          explicit_decay_settings(explicit_decays),
                          dict(model_file=model if model_file is not None else None,
                               output=destination.resolve() if output is not None else None))
