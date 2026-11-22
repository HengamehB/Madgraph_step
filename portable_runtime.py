"""Host discovery for the self-contained Madgraph_step source distribution.

This module never installs software, activates conda, or downloads resources.
Paths in returned reports describe the current host, not saved configuration.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FEYNRULES_DIR = ROOT / "vendor" / "FeynRules"
MADGRAPH_DIR = ROOT / "vendor" / "MG5_aMC"
_TOOLS = {"wolfram_kernel": ("WolframKernel", "MathKernel", "math"),
          "fortran": ("gfortran",), "cxx": ("c++", "g++", "clang++"),
          "make": ("make",), "bash": ("bash",), "perl": ("perl",)}
_PACKAGES = ("numpy", "scipy")
_SCOPES = {
    "full": set(_TOOLS) | set(_PACKAGES) | {"feynrules", "madgraph"},
    "madgraph": {"numpy", "scipy", "fortran", "cxx", "make", "bash", "perl", "madgraph"},
    "feynrules": {"numpy", "scipy", "wolfram_kernel", "feynrules"},
}

_SOURCE_FILES = {
    'feynrules': ('FeynRules.m', 'FeynRulesPackage.m', 'Models/SM/SM.fr', 'Interfaces/UFO/PYIntMain.m'),
    'madgraph': ('bin/mg5_aMC', 'VERSION', 'madgraph/interface/master_interface.py',
                 'Template/LO/lib/Pdfdata/NNPDF23_lo_as_0130_qed_mem0.grid'),
}


def source_tree_exclusions(tool, root, directory, names):
    """Prune non-runtime trees before reading them, in checks and scratch copies.

    Only MadGraph's own top-level tests are omitted. A nested directory with
    the same name may contain runtime assets and must remain available.
    """
    skipped = set(names) & {'__pycache__', '.git'}
    if tool == 'madgraph' and Path(directory) == Path(root):
        skipped.update(set(names) & {'tests'})
    return skipped


def tool_source_files(tool, root):
    """Yield runtime source/data paths without entering unused tool tests."""
    root = Path(root)
    suffixes = {'.py', '.m', '.wl', '.fr', '.f', '.for', '.f90', '.f95',
                '.c', '.cc', '.cpp', '.h', '.hpp', '.inc', '.dat', '.grid',
                '.tbl', '.pds', '.txt', '.mk', '.cmake', '.sh', '.pl'}
    filenames = {'version', 'mg5_amc', 'makefile', 'gnumakefile',
                 'cmakelists.txt', 'configure', 'make_opts'}
    def fail(error):
        raise error
    for directory, directories, files in os.walk(root, onerror=fail):
        skipped = source_tree_exclusions(tool, root, directory, directories)
        directories[:] = [name for name in directories if name not in skipped]
        for name in files:
            path = Path(directory) / name
            if (path.suffix.lower() in suffixes or name.lower() in filenames) and path.is_file():
                yield path


def _wolfram_candidates(*, application_roots=None, linux_roots=None):
    """List conventional installs without starting a kernel or searching the disk.

    PATH and explicit configuration still take precedence. Both system and
    per-user macOS applications are supported, including Wolfram Engine.
    Optional roots keep discovery testable without depending on the CI host.
    """
    if application_roots is None:
        application_roots = (Path('/Applications'), Path.home() / 'Applications')
    if linux_roots is None:
        linux_roots = (Path('/usr/local/Wolfram'), Path('/opt/Wolfram'))
    candidates = []
    for root in application_roots:
        for app in ('Mathematica.app', 'Wolfram.app', 'Wolfram Engine.app',
                    'WolframEngine.app', 'Engine.app'):
            for kernel in ('WolframKernel', 'MathKernel'):
                candidates.append(Path(root) / app / 'Contents' / 'MacOS' / kernel)
    for root in linux_roots:
        for product in ('Mathematica', 'Wolfram', 'WolframEngine'):
            parent = Path(root) / product
            if not parent.is_dir():
                continue
            # A versionless installation or symlink is valid too. Versioned
            # installations are ordered numerically, so 14.10 precedes 14.9.
            for kernel in ('WolframKernel', 'MathKernel'):
                candidates.append(parent / 'Executables' / kernel)
                versions = parent.glob('*/Executables/' + kernel)
                candidates.extend(sorted(versions, key=lambda path: tuple(
                    (1, int(part)) if part.isdecimal() else (0, part.lower())
                    for part in re.split(r'(\d+)', path.parents[1].name)), reverse=True))
    return candidates


def normalize_tool_path(name, value, *, base=None):
    """Resolve a tool candidate without running it; invalid candidates return None.

    Source distributions must contain the files and versions required by this
    package. Kernel licensing and actual compiler startup are checked by check.
    """
    if name not in set(_TOOLS) | set(_SOURCE_FILES):
        raise ValueError(f'Unknown tool: {name}')
    if value is None or not str(value).strip():
        return None
    value = os.fspath(value)
    if any(character in value for character in ('\n', '\r', '\0')):
        return None
    if name in _TOOLS:
        if base is not None and not Path(value).expanduser().is_absolute() and ('/' in value or '\\' in value):
            value = str(Path(base) / Path(value).expanduser())
        return _executable(value)
    if name == 'madgraph' and value == 'mg5_aMC':
        value = shutil.which(value) or value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base or ROOT) / path
    if name == 'madgraph' and path.name == 'mg5_aMC':
        path = path.parent.parent
    elif name == 'feynrules' and path.name == 'FeynRules.m':
        path = path.parent
    path = path.resolve()
    if any(not (path / item).is_file() for item in _SOURCE_FILES[name]):
        return None
    version = (_version(path / 'VERSION', r'version\s*=\s*([^\s]+)') if name == 'madgraph'
               else _version(path / 'FeynRulesPackage.m', r'FR\$VersionNumber\s*=\s*"([^"]+)"'))
    return str(path) if version.startswith('3.' if name == 'madgraph' else '2.') else None


def discover_tools(runtime=None):
    """List valid configured, bundled, PATH and conventional-location candidates.

    Discovery is read-only: it neither runs the kernel nor edits configuration.
    An explicit invalid path remains invalid; alternatives are proposals only.
    """
    runtime = dict(runtime or {})
    unknown = set(runtime) - set(_TOOLS) - {'python', 'madgraph', 'feynrules'}
    if unknown:
        raise ValueError('Unknown runtime keys: ' + ', '.join(sorted(unknown)))
    report = {}
    for name in ('wolfram_kernel', 'madgraph', 'feynrules', 'fortran', 'cxx', 'make', 'bash', 'perl'):
        candidates, seen = [], set()
        def add(value, origin):
            path = normalize_tool_path(name, value)
            if path and Path(path).resolve() not in seen:
                seen.add(Path(path).resolve())
                candidates.append({'path': path, 'origin': origin})
        if name in runtime:
            add(runtime[name], 'configured')
        if name in _SOURCE_FILES:
            add(FEYNRULES_DIR if name == 'feynrules' else MADGRAPH_DIR, 'bundled')
            if name == 'madgraph':
                located = shutil.which('mg5_aMC')
                if located:
                    add(located, 'PATH')
        else:
            for executable in _TOOLS[name]:
                add(executable, 'PATH')
            if name == 'wolfram_kernel':
                for candidate in _wolfram_candidates():
                    add(candidate, 'standard location')
        selected = (normalize_tool_path(name, runtime[name]) if name in runtime else
                    candidates[0]['path'] if candidates else None)
        report[name] = {'configured': runtime.get(name), 'selected': selected,
                        'candidates': candidates,
                        'reason': ('configured path is invalid' if name in runtime and selected is None
                                   else 'no valid candidate found' if selected is None else None)}
    return {'tools': report, 'paths': {name: entry['selected'] for name, entry in report.items()}}


def _executable(value):
    if value is None:
        return None
    value = os.fspath(value)
    candidate = shutil.which(value)
    if candidate:
        return str(Path(candidate).absolute())
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.absolute())
    return None


def _paths(runtime=None):
    runtime = dict(runtime or {})
    allowed = set(_TOOLS) | {"python", "feynrules", "madgraph"}
    unknown = set(runtime) - allowed
    if unknown:
        raise ValueError("Unknown runtime keys: " + ", ".join(sorted(unknown)))
    paths = {}
    for name, candidates in _TOOLS.items():
        if name in runtime:
            paths[name] = _executable(runtime[name])
        else:
            paths[name] = next((p for candidate in candidates
                                if (p := _executable(candidate))), None)
    if "wolfram_kernel" not in runtime and not paths["wolfram_kernel"]:
        paths["wolfram_kernel"] = next((p for candidate in _wolfram_candidates()
                                        if (p := _executable(candidate))), None)
    paths["python"] = str(Path(sys.executable).absolute())
    if "python" in runtime and Path(runtime["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("python must be the active Python interpreter")
    for key, default in (("feynrules", FEYNRULES_DIR), ("madgraph", MADGRAPH_DIR)):
        path = Path(runtime.get(key, default)).expanduser()
        if key == 'madgraph' and key not in runtime and normalize_tool_path(key, path) is None:
            discovered = shutil.which('mg5_aMC')
            if discovered and normalize_tool_path(key, discovered):
                path = Path(normalize_tool_path(key, discovered))
        if not path.is_absolute():
            path = ROOT / path
        if key == "madgraph" and path.name == "mg5_aMC":
            path = path.parent.parent
        elif key == "feynrules" and path.name == "FeynRules.m":
            path = path.parent
        paths[key] = str(path.resolve())
    return paths


def environment(runtime=None):
    """Subprocess environment: active tools, portable locale, no Python network.

    Python child processes load our tiny sitecustomize guard. Loopback remains
    available for local processes; remote sockets are denied.
    MadGraph's own update/browser settings are also disabled in its bundled card.
    """
    paths = _paths(runtime)
    env = os.environ.copy()
    env.update(LC_ALL="C", LANG="C", LC_CTYPE="C", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", MADGRAPH_STEP_OFFLINE="1")
    bins = [str(Path(sys.executable).parent)]
    bins += [str(Path(paths[key]).parent) for key in _TOOLS if paths[key]]
    env["PATH"] = os.pathsep.join(dict.fromkeys(bins + env.get("PATH", "").split(os.pathsep)))
    guard = str(ROOT / "vendor" / "offline_guard")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (guard, env.get("PYTHONPATH", ""))))
    if paths["fortran"]:
        env["FC"] = env["F77"] = env["F90"] = paths["fortran"]
    if paths["cxx"]:
        env["CXX"] = paths["cxx"]
    return env


def _probe(command, env, timeout=15):
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env,
                                timeout=timeout, cwd=ROOT)
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, output
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def check_environment(runtime=None, *, scope="full"):
    """Return {ok, checks, paths}; check installed tools without installing any.

    ``runtime`` may override wolfram_kernel, fortran, cxx, make, bash, and perl.
    ``scope`` is full (the default), madgraph, or feynrules. Only dependencies
    required by that stage are checked. Full and FeynRules checks start the
    actual Mathematica kernel to detect licensing/startup errors; MadGraph
    checks do not start it. No terminal scope requires notebook packages.
    """
    if scope not in _SCOPES:
        raise ValueError("Unknown runtime scope: " + str(scope))
    required = _SCOPES[scope]
    paths = _paths(runtime)
    env = environment(runtime)
    checks = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    record("platform", os.name != "nt", platform.platform() if os.name != "nt"
           else "Use Linux under WSL; native Windows is not a supported MadGraph host.")
    record("python", sys.version_info[:2] == (3, 11),
           f"Python {platform.python_version()} at {sys.executable}; Python 3.11 is required.")
    for package in _PACKAGES:
        if package not in required:
            continue
        try:
            version = importlib.metadata.version(package)
            found = importlib.util.find_spec(package) is not None
            record(package, found, version)
        except (ImportError, importlib.metadata.PackageNotFoundError, ValueError) as exc:
            record(package, False, f"Missing from the active Python environment: {exc}")
    for name in _TOOLS:
        if name not in required:
            continue
        executable = paths[name]
        if not executable:
            record(name, False, f"Executable not found; provide runtime['{name}'].")
            continue
        if name == "wolfram_kernel":
            ok, output = _probe([executable, "-noinit", "-noprompt", "-run",
                                 'Print["MADGRAPH_STEP_KERNEL=" <> $Version]; Quit[]'], env, 45)
            marker = next((line for line in output.splitlines()
                           if "MADGRAPH_STEP_KERNEL=" in line), "")
            record(name, ok and bool(marker), marker.replace("MADGRAPH_STEP_KERNEL=", "")
                   or output[-1000:] or "Kernel did not return a version; check its license.")
        else:
            ok, output = _probe([executable, "-v" if name == "perl" else "--version"], env)
            record(name, ok, next((line for line in output.splitlines() if line.strip()),
                                  "No version output"))
    if "feynrules" in required:
        fr = Path(paths["feynrules"])
        required_fr = _SOURCE_FILES['feynrules']
        missing = [name for name in required_fr if not (fr / name).is_file()]
        fr_version = _version(fr / "FeynRulesPackage.m", r'FR\$VersionNumber\s*=\s*"([^"]+)"')
        record("feynrules", not missing and fr_version.startswith("2."),
               f"FeynRules {fr_version} at {fr}" if not missing and fr_version.startswith("2.")
               else "FeynRules 2.x with the UFO interface is required. Missing: " + ", ".join(missing))
    if "madgraph" in required:
        mg = Path(paths["madgraph"])
        required_mg = _SOURCE_FILES['madgraph']
        missing = [name for name in required_mg if not (mg / name).is_file()]
        mg_version = _version(mg / "VERSION", r'version\s*=\s*([^\s]+)')
        record("madgraph", not missing and mg_version.startswith("3."),
               f"MadGraph {mg_version} at {mg}, with nn23lo1 PDF" if not missing and mg_version.startswith("3.")
               else "MadGraph 3.x with its LO source and built-in PDFs is required. Missing: " + ", ".join(missing))
    return {"ok": all(item["ok"] for item in checks), "checks": checks, "paths": paths, "scope": scope}


def _version(file, pattern):
    if not file.is_file():
        return "unknown"
    match = re.search(pattern, file.read_text(errors="replace"))
    return match.group(1) if match else "unknown"


def resolve_runtime(runtime=None, *, scope="full"):
    """Return verified host paths or a concise diagnostic; never install tools."""
    report = check_environment(runtime, scope=scope)
    if not report["ok"]:
        problems = [f"{item['name']}: {item['detail']}" for item in report["checks"] if not item["ok"]]
        raise RuntimeError("Environment is not ready:\n" + "\n".join(problems))
    return report["paths"]


def host_fingerprint(runtime=None):
    """Build identity; generated processes must be regenerated when this changes.

    Including the project path also invalidates absolute makefile paths after a
    move. Do not delete source/vendor data when comparing this identity.
    """
    paths = _paths(runtime)
    env = environment(runtime)
    compilers = {}
    for key in ("fortran", "cxx", "make"):
        path = paths[key]
        ok, version = _probe([path, "--version"], env) if path else (False, "missing")
        compilers[key] = {"path": path, "version": version.splitlines()[0] if version else "", "ok": ok}
    return {"system": platform.system(), "machine": platform.machine(),
            "release": platform.release(), "python": platform.python_version(),
            "project_root": str(ROOT), "madgraph": _version(Path(paths["madgraph"]) / "VERSION", r'version\s*=\s*([^\s]+)'), "compilers": compilers}


if __name__ == "__main__":
    report = check_environment()
    for check in report["checks"]:
        print(f"{'OK' if check['ok'] else 'MISSING'} {check['name']}: {check['detail']}")
    raise SystemExit(0 if report["ok"] else 1)
