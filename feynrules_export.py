"""Resolve selected operators and export a benchmark UFO using bundled FeynRules.

Automatic compilers construct invariant contractions and physical mass bases
for scalar, fermion and effective-vector sectors, retaining established input
conventions where specialized compilers already exist. See REFERENCE.md for
validated coverage and remaining implementation gaps. Unresolved nonzero
contractions block export with their IDs; exact-zero terms retain explicit
coverage records. A display formula is never treated as an executable
contraction. The advanced explicit-model hook is separate from automatic coverage.

Numbers use GeV units.  Coefficients multiply precisely the monomials printed in
the catalogue: e.g. c X^4 has no implicit 1/4! and c (d X)^2 has no implicit 1/2.
Automatic UFOs describe one benchmark point. Re-export after changing inputs;
their computed masses and mixing cannot safely be scanned independently.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations, product
import ast
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from workflow_errors import PhysicalInputError, CompilerError


ROOT = Path(__file__).resolve().parent


class ModelResolutionError(PhysicalInputError):
    """Selected operators or a numerical vacuum cannot be exported safely."""


class BackendCapabilityError(ModelResolutionError):
    """A representation is outside the pinned FeynRules/MadGraph backend."""
    category = 'not_madgraph_supported'


class CompilerResolutionError(ModelResolutionError, CompilerError):
    """Our compiler or exporter failed to lower a backend-supported model."""
    category = 'compiler_failure'


def _check_unknown_ufo_fields(ufo, log, log_text):
    """Report the first rejected field even when the kernel exits via Abort[]."""
    tag = 'PORTABLE_UFO_UNKNOWN_FIELD'
    trace = ufo / 'unknown_vertex_fields.json'
    if tag not in log_text and not trace.exists():
        return
    try:
        with trace.open('rb') as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError('field trace exceeds 1 MiB')
        report = json.loads(data)
        leg = next(row for row in report['legs'] if row['outgoing_spin_code'] == 0)
        detail = (f"vertex {report['vertex_index']}, leg {leg['position']}: "
                  f"{str(leg['incoming_field'])[:180]} -> {str(leg['outgoing_field'])[:180]}")
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        # Export itself can fail (for example on a full disk). Preserve the
        # leg that FeynRules printed even if the JSON could not be recovered.
        marker = re.search(r'PORTABLE_UFO_UNKNOWN_FIELD[^\n]{0,500}', log_text)
        detail = marker.group(0)[len(tag):].strip() if marker else 'no logged field'
        detail += f' (unreadable field trace: {exc})'
    raise CompilerResolutionError(f'{tag} {detail}. Field trace: {trace}. FeynRules log: {log}.')


def backend_preflight(fields):
    """Reject true generator limits before FeynRules can discard particles.

    Both the bundled UFO writer's GenInt$Colors and MadGraph's color-flow
    exporter support exactly these SU(3) irreducible representations. This is
    deliberately independent of the Python compiler's current coverage.
    """
    supported = {(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)}
    for field in fields:
        if tuple(field.su3) not in supported:
            p, q = field.su3
            dimension = (p + 1) * (q + 1) * (p + q + 2) // 2
            raise BackendCapabilityError(
                f"Not MadGraph supported: {field.name}: SU(3) ({p},{q}), dimension {dimension}, is outside the "
                "bundled FeynRules UFO and MadGraph color backend (1, 3, 3bar, 6, 6bar, 8). "
                "Extending the model compiler alone cannot enable this representation.")


def _verify_exported_particles(expected, actual):
    """Verify all requested physical states and their charge conjugates survived."""
    problems = []
    for particle in expected:
        pdg = int(particle['pdg'])
        record = actual.get(pdg)
        if record is None:
            problems.append(f"missing {particle['name']} (PDG {pdg})")
            continue
        for key, desired in (('name', particle['name']), ('spin', particle['spin'])):
            if record.get(key) != desired:
                problems.append(f"{particle['name']}: exported {key}={record.get(key)!r}, expected {desired!r}")
        for key in ('charge', 'color'):
            if key in particle and not math.isclose(float(record.get(key, float('nan'))),
                                                     float(particle[key]), rel_tol=1e-10, abs_tol=1e-10):
                problems.append(f"{particle['name']}: exported {key} differs from the resolved representation")
        anti = particle.get('antiname', particle['name'])
        anti_pdg = pdg if anti == particle['name'] else -pdg
        if actual.get(anti_pdg, {}).get('name') != anti:
            problems.append(f"missing or incorrect conjugate {anti} (PDG {anti_pdg})")
        if not math.isclose(float(record.get('mass_gev', -1)), float(particle['mass']), rel_tol=1e-7,
                            abs_tol=0. if float(particle['mass']) else 1e-12):
            problems.append(f"{particle['name']}: exported mass differs from the resolved spectrum")
    if problems:
        raise CompilerResolutionError('UFO physical-particle coverage failed: ' + '; '.join(problems))


@dataclass(frozen=True)
class ResolvedOperator:
    operator_id: str
    kind: str
    factors: tuple
    coefficients: tuple
    expressions: tuple = ()
    add_hc: bool = False


@dataclass
class UFOModel:
    path: Path
    model_name: str
    particles: tuple
    masses: dict
    parameter_values: dict = field(default_factory=dict)
    parameter_card: Path | None = None
    source_directory: Path | None = None
    selected_ids: tuple = ()
    benchmark_only: bool = True
    saved_model_digest: str = ''
    saved_parameter_card_digest: str = ''

    @property
    def new_particles(self):
        return tuple(dict.fromkeys(name for p in self.particles
                                   for name in (p["name"], p.get("antiname", p["name"]))))

    def __fspath__(self):
        return str(self.path)

    def __repr__(self):
        return f"UFOModel({self.model_name!r}, particles={self.new_particles!r})"


def _fields(lagrangian):
    if not hasattr(lagrangian, "operators") or not hasattr(lagrangian, "_source"):
        raise TypeError("Use result.select(...) to choose one concrete Lagrangian first.")
    return lagrangian._source.models[lagrangian.scenario - 1]


def _automatic_field(field):
    return (field.spin == "scalar" and field.kind == "real" and field.su2 == 1
            and field.su3 == (0, 0) and field.hypercharge == 0)


def _classify(row, fields):
    names = {f.name for f in fields}
    counts = Counter({factor["name"]: factor["count"] for factor in row["factors"]})
    if row["multiplicity"] != 1:
        return None, "explicit invariant/flavour contractions are required"
    if row["derivatives"]:
        bare = [name[2:-1] for name, count in counts.items()
                for _ in range(count) if name.startswith("D(") and name.endswith(")")]
        if (sum(counts.values()) == 2 and len(bare) == 2 and set(bare) <= names
                and all(_automatic_field(f) for f in fields if f.name in bare)):
            return "kinetic", ""
        return None, "derivative/Lorentz contractions and integration-by-parts relations are unresolved"
    if not set(counts) <= names | {"H", "H†"}:
        return None, "fermion, quark, lepton or gauge-field contractions require an explicit FeynRules expression"
    if any(not _automatic_field(f) for f in fields if f.name in counts):
        return None, "this field representation/type requires explicit component fields and contractions"
    if counts["H"] != counts["H†"]:
        return None, "the Higgs indices are not a product of H-dagger H"
    return "potential", ""


def required_inputs(lagrangian, *, compiler_limits=None):
    """Return an editable dictionary. None means a value must be supplied.

    The complete selected coefficient list is retained, including zero-valued
    terms. Unsupported sectors are reported in ``unresolved``.
    """
    fields = _fields(lagrangian)
    from neutral_fermion_compiler import supports_fields, classify as neutral_classify
    from gauge_matter_compiler import supports_fields as gauge_supports, classify as gauge_classify
    from vector_current_compiler import supports_fields as vector_supports, classify as vector_classify
    from generic_scalar_compiler import uses_generic, classify as generic_classify
    from generic_color_compiler import uses_generic as colour_uses_generic, classify as colour_classify
    from generic_joint_scalar_compiler import uses_generic as joint_uses_generic, classify as joint_classify
    from generic_fermion_compiler import uses_generic as fermion_uses_generic, classify as fermion_classify
    from generic_majorana_compiler import uses_generic as majorana_uses_generic, classify as majorana_classify
    from mixed_scalar_fermion_compiler import uses_generic as mixed_uses_generic, classify as mixed_classify
    from generic_vector_compiler import uses_generic as generic_vector_uses, classify as generic_vector_classify
    neutral = supports_fields(fields)
    gauge = gauge_supports(fields)
    vector = vector_supports(fields)
    generic = uses_generic(fields)
    colour = colour_uses_generic(fields, lagrangian.operators)
    joint = joint_uses_generic(fields)
    fermion = fermion_uses_generic(fields)
    majorana = majorana_uses_generic(fields)
    mixed = mixed_uses_generic(fields)
    generic_vector = generic_vector_uses(fields)
    unresolved = {}
    for row in lagrangian.operators:
        _, reason = _classify(row, fields)
        if generic:
            _, reason = generic_classify(row, fields)
        elif generic_vector:
            _, reason = generic_vector_classify(row, fields)
        elif fermion:
            _, reason = fermion_classify(row, fields)
        elif majorana:
            _, reason = majorana_classify(row, fields)
        elif mixed:
            _, reason = mixed_classify(row, fields)
        elif joint:
            _, reason = joint_classify(row, fields, limits=compiler_limits)
        elif colour:
            _, reason = colour_classify(row, fields, limits=compiler_limits)
        if reason and neutral:
            _, reason = neutral_classify(row, fields)
        if reason and gauge:
            _, reason = gauge_classify(row, fields)
        if reason and vector:
            _, reason = vector_classify(row, fields)
        if reason:
            unresolved[row["id"]] = reason
    for f in fields:
        if f.spin == "vector":
            if not vector and not gauge and not generic_vector:
                unresolved[f.name] = "automatic vector component/gauge-model resolution is unavailable for this sector"
        elif not _automatic_field(f) and not neutral and not gauge and not vector and not generic and not colour and not joint and not fermion and not majorana and not mixed:
            unresolved.setdefault(f.name, "automatic component-field/mass resolution is unavailable for this field type")
    result = {
        "model_name": "SM_" + "_".join(f.name for f in fields),
        "coefficients": {row["id"]: None if row["multiplicity"] == 1
                         else [None] * row["multiplicity"] for row in lagrangian.operators},
        "vevs": {f.name: 0.0 for f in fields if f.spin == "scalar"},
        "widths": {f.name: 0.0 for f in fields},
        "higgs_mass": 125.0,
        "sm_inputs": {"Gf": 1.16637e-5},
        "unresolved": unresolved,
    }
    if any(f.spin == "vector" for f in fields):
        result["vector_specifications"] = {f.name: None for f in fields if f.spin == "vector"}
    return result


def _number(value, label, *, real=False):
    if value is None or isinstance(value, (bool, str)):
        raise ModelResolutionError(f"Supply a finite numeric value for {label}.")
    try:
        result = complex(value)
    except (ValueError, TypeError) as exc:
        raise ModelResolutionError(f"Supply a finite numeric value for {label}.") from exc
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ModelResolutionError(f"{label} must be finite.")
    if real and result.imag != 0:
        raise ModelResolutionError(f"{label} must be real for a Hermitian operator.")
    return result.real if result.imag == 0 else result


def _coefficients(lagrangian, inputs):
    supplied = inputs.get("coefficients", {})
    required = set(lagrangian.selected_ids)
    if set(supplied) != required:
        missing, extra = required - set(supplied), set(supplied) - required
        raise ModelResolutionError(f"Coefficient IDs must match the selected operators. Missing={sorted(missing)}, extra={sorted(extra)}.")
    values = {}
    for row in lagrangian.operators:
        value = supplied[row["id"]]
        entries = [value] if row["multiplicity"] == 1 else value
        if not isinstance(entries, (list, tuple)) or len(entries) != row["multiplicity"]:
            raise ModelResolutionError(f"{row['id']} needs {row['multiplicity']} coefficients, one per explicit invariant.")
        values[row["id"]] = tuple(_number(x, f"{row['id']}[{i}]",
                                          real=row["field_content"] == row["conjugate_content"])
                                  for i, x in enumerate(entries, 1))
    return values


def _wl(value):
    """Wolfram scalar literal, avoiding machine scientific-notation syntax."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    value = complex(value)
    if value.imag:
        return f"({_wl(value.real)} + I*({_wl(value.imag)}))"
    if value.real == 0:
        return "0"
    if value.real == int(value.real):
        return str(int(value.real))
    return repr(float(value.real)).replace("e", "*^")


def _poly_add(poly, powers, value):
    powers = tuple(powers)
    poly[powers] = poly.get(powers, 0.0) + value


def _scalar_basis(fields, operators, inputs):
    """Canonicalize a positive kinetic form and diagonalize the real Hessian."""
    n = len(fields) + 1
    names = ["H"] + [f.name for f in fields]
    indices = {name: i for i, name in enumerate(names)}
    gf = _number(inputs.get("sm_inputs", {}).get("Gf", 1.16637e-5), "sm_inputs.Gf", real=True)
    mh = _number(inputs.get("higgs_mass", 125.0), "higgs_mass", real=True)
    if gf <= 0 or mh <= 0:
        raise ModelResolutionError("Gf and the unmixed Higgs mass must be positive.")
    v = (math.sqrt(2.0) * gf) ** -0.5
    vevs = {f.name: 0.0 for f in fields}
    kinetic = np.zeros((n, n))
    kinetic[0, 0] = 1.0
    potential = {}
    tadpole_scales = np.zeros(n)
    for operator in operators:
        c = operator.coefficients[0]
        counts = dict(operator.factors)
        if operator.kind == "kinetic":
            ids = [indices[name[2:-1]] for name, count in counts.items() for _ in range(count)]
            if ids[0] == ids[1]:
                kinetic[ids[0], ids[0]] += 2 * c
            else:
                kinetic[ids[0], ids[1]] += c
                kinetic[ids[1], ids[0]] += c
        else:
            higgs_pairs = counts.get("H", 0)
            full_powers = [2 * higgs_pairs] + [counts.get(f.name, 0) for f in fields]
            shifts = [v] + [vevs[f.name] for f in fields]
            for powers in product(*(range(power + 1) for power in full_powers)):
                value = c / 2 ** higgs_pairs
                for total, degree, shift in zip(full_powers, powers, shifts):
                    value *= math.comb(total, degree) * shift ** (total - degree)
                _poly_add(potential, powers, value)
                if sum(powers) == 1:
                    tadpole_scales[powers.index(1)] += abs(value)
    tadpoles = np.zeros(n)
    hessian = np.zeros((n, n))
    hessian[0, 0] = mh * mh
    for powers, coefficient in potential.items():
        ids = [i for i, power in enumerate(powers) for _ in range(power)]
        if len(ids) == 1:
            tadpoles[ids[0]] -= coefficient
        elif len(ids) == 2:
            i, j = ids
            hessian[i, j] -= coefficient
            hessian[j, i] -= coefficient
    # With only the SM Higgs carrying a VEV, the SM tadpole vanishes at v.
    # Any additional linear term in the expansion must cancel on its own.
    mu_h_squared = mh * mh / 2
    diagonal = kinetic.diagonal()
    if np.any(diagonal <= 0):
        raise ModelResolutionError("The selected kinetic matrix must be positive definite. Select and supply each field's kinetic operator (canonical real scalar coefficient 1/2).")
    field_scale = np.sqrt(diagonal)
    normalized_kinetic = (kinetic / field_scale[:, None]) / field_scale[None, :]
    normalized_hessian = (hessian / field_scale[:, None]) / field_scale[None, :]
    # Check each vacuum equation against the terms that actually enter it.
    # A large unrelated mass must not hide a nonzero tadpole, and changing a
    # field's input normalization must not change this cancellation test.
    canonical_tadpoles = tadpoles / field_scale
    canonical_tadpole_scales = tadpole_scales / field_scale
    bad_tadpoles = abs(canonical_tadpoles) > 1e-10 * canonical_tadpole_scales
    if np.any(bad_tadpoles):
        bad = ", ".join(f"{names[i]}: {t:.6g} GeV^3" for i, t in enumerate(tadpoles) if bad_tadpoles[i])
        raise ModelResolutionError("The specified vacuum is not stationary (potential derivatives: " + bad
                                   + "). Supply the selected tadpole coefficients consistently; no counterterm is added automatically.")
    eigk, veck = np.linalg.eigh(normalized_kinetic)
    if eigk.min() <= 0:
        raise ModelResolutionError("The selected kinetic matrix must be positive definite. Select and supply each field's kinetic operator (canonical real scalar coefficient 1/2).")
    invroot = (veck / np.sqrt(eigk)) @ veck.T
    values, rotations = np.linalg.eigh(invroot @ normalized_hessian @ invroot)
    canonical_hessian = invroot @ normalized_hessian @ invroot
    # An unrelated heavy state must not hide a light tachyon. Bound each
    # Rayleigh quotient's own cancellation error instead of using max(m^2).
    mode_scales = np.einsum('ij,ik,jk->k', abs(canonical_hessian), abs(rotations), abs(rotations))
    if np.any(values < -64*np.finfo(float).eps*mode_scales):
        raise ModelResolutionError("The scalar mass matrix contains a tachyon at the supplied vacuum.")
    # Keep the mass state most associated with each original field's label.
    canonical_transform = invroot @ rotations
    transform = canonical_transform / field_scale[:, None]
    normalized = canonical_transform / np.linalg.norm(canonical_transform, axis=0)
    orders = list(permutations(range(n)))
    order = max(orders, key=lambda p: sum(abs(normalized[i, p[i]]) for i in range(n)))
    transform = transform[:, order]
    values = np.maximum(values[list(order)], 0.0)
    for j in range(n):
        if transform[j, j] < 0:
            transform[:, j] *= -1
    if not np.allclose(transform.T @ kinetic @ transform, np.eye(n), rtol=1e-10, atol=1e-10):
        raise CompilerResolutionError("Kinetic normalization failed its residual check.")
    mass_scale = float(abs(values).max())
    residual = transform.T @ hessian @ transform - np.diag(values)
    if (mass_scale and np.max(abs(residual / mass_scale)) > 1e-10
            or not mass_scale and np.any(residual != 0)):
        raise CompilerResolutionError("Mass diagonalization failed its residual check.")
    return {"names": names, "kinetic": kinetic, "hessian": hessian,
            "transformation": transform, "masses": np.sqrt(values), "vev": v,
            "higgs_mass_before_mixing": mh, "gf": gf, "vevs": vevs,
            "mu_h_squared": mu_h_squared}


@dataclass(repr=False)
class PreparedModel:
    lagrangian: object
    inputs: dict
    operators: tuple
    fields: tuple
    basis: dict | None
    runtime: dict | None = None
    manual: dict | None = None
    compiler: object | None = None

    @property
    def model_name(self):
        return self.inputs["model_name"]

    @property
    def masses(self):
        if self.compiler is not None:
            return self.compiler.masses
        if self.basis is None:
            return {p["source"]: p["mass"] for p in self.manual["particles"]}
        return dict(zip(self.basis["names"], self.basis["masses"].tolist()))

    @property
    def coverage(self):
        state = "user-specified explicit contraction" if self.manual else "resolved and numerically normalized"
        return {op.operator_id: ('zero_verified: complete coefficient vector is explicitly zero'
                                 if op.kind == 'zero_verified' else state) for op in self.operators}

    def __repr__(self):
        return f"PreparedModel({self.model_name!r}, masses={self.masses!r})"

    def export_ufo(self, output=None, *, timeout=1800, analytic_decays=True):
        """Run local Mathematica/FeynRules and return a UFOModel artifact.

        Output is a new directory; an existing model is never overwritten. All
        generated source and logs are alongside the UFO for later inspection.
        No network access or package installation is performed.
        """
        from portable_runtime import resolve_runtime, environment
        from compiler_limits import DEFAULTS, normalize_limits
        from fixed_sm import check_fixed_sm_model, normalize_fixed_sm

        # A trusted Python caller may supply a PreparedModel from a component
        # compiler directly. Enforce the same assumptions at the export boundary.
        backend_preflight(self.fields)
        self.inputs = normalize_fixed_sm(self.inputs, self.fields)
        check_fixed_sm_model(self)
        construction_limits = normalize_limits(self.inputs.get('compiler_limits'))
        wolfram_limits = {name: construction_limits.get(name, DEFAULTS[name])
                          for name in ('max_wolfram_recursion', 'max_wolfram_iterations')}

        if type(analytic_decays) is not bool:
            raise ValueError("analytic_decays must be True or False.")
        requested_analytic_decays = analytic_decays
        decay_policy_reason = None
        if any(field.su3 in {(2, 0), (0, 2)} for field in self.fields):
            # FeynRules 2.3.49 Decay.m sums Colour/Gluon indices only. For
            # sextets it can emit literal zero for an open, nonzero decay;
            # MadWidth would trust that zero instead of evaluating vertices.
            analytic_decays = False
            decay_policy_reason = ('The bundled FeynRules analytic-width colour sums do not support sextets. '
                                   'MadWidth must calculate these widths from the exported matrix elements.')

        paths = resolve_runtime(self.runtime)
        kernel = paths.get("wolfram_kernel")
        fr = Path(paths["feynrules"])
        if not kernel or not Path(kernel).is_file():
            raise RuntimeError("A working Mathematica kernel is required; set runtime['wolfram_kernel'].")
        if not (fr / "FeynRules.m").is_file():
            raise RuntimeError("Bundled FeynRules is missing; check_environment() reports the expected path.")
        output = Path(output) if output is not None else ROOT / "runs" / self.model_name
        output = output.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Choose a fresh export directory; refusing to overwrite {output}.")
        output.mkdir(parents=True, exist_ok=True)
        from pipeline_storage import write_json
        write_json(output / 'export_policy.json', dict(
            requested_analytic_decays=requested_analytic_decays,
            effective_analytic_decays=analytic_decays, reason=decay_policy_reason,
            effective_wolfram_limits=wolfram_limits))
        source = output / "feynrules_source"
        source.mkdir()
        ufo = output / (self.model_name + "_UFO")
        if self.compiler is not None:
            model_files, expression, particles = self.compiler.write_sources(self, source, fr)
        else:
            model_files, expression, particles = self._automatic_sources(source, fr)
        driver = source / "export.wl"
        # The bundled SM represents each massless, left-handed neutrino by a
        # projected four-component Dirac field. FeynRules' generic normalizer
        # expects both chiralities and otherwise falsely rejects the stock SM.
        # Double only those terms in the CHECK expression, never in LPortable.
        kinetic_check = "LPortable"
        if (self.compiler is None
                or getattr(self.compiler, "stock_sm_neutrinos", False)):
            kinetic_check += (" + GetKineticTerms[LPortable, FlavorExpand -> True, "
                              "SelectParticles -> {{ve,vebar},{vm,vmbar},{vt,vtbar}}]")
        lines = [
            '$HistoryLength = 0;',
            f'$FeynRulesPath = {_wl(str(fr))};',
            'SetDirectory[$FeynRulesPath];',
            'FR$Parallel = False;',
            'Get[FileNameJoin[{$FeynRulesPath, "FeynRules.m"}]];',
            f'$RecursionLimit = {wolfram_limits["max_wolfram_recursion"]};',
            f'$IterationLimit = {wolfram_limits["max_wolfram_iterations"]};',
            'LoadModel[' + ', '.join(_wl(str(p)) for p in model_files) + '];',
            'FeynmanGauge = False;',
            f'LPortable = {expression};',
            'If[CheckHermiticity[LPortable] =!= {}, Print["PORTABLE_HERMITICITY_FAILED"]; Quit[12]];',
            '(* Check-only adjustment for the stock SM chiral neutrinos; exported LPortable is unchanged. *)',
            f'If[CheckKineticTermNormalisation[{kinetic_check}] =!= True, Print["PORTABLE_KINETIC_CHECK_FAILED"]; Quit[13]];',
            'If[CheckDiagonalMassTerms[LPortable] =!= True, Print["PORTABLE_MASS_CHECK_FAILED"]; Quit[14]];',
            'portableSpectrum = GetMassSpectrum[LPortable];',
            'If[Head[portableSpectrum] =!= TableForm, Print["PORTABLE_SPECTRUM_CHECK_FAILED"]; Quit[16]];',
            'portableBadMasses = Select[Rest[First[portableSpectrum]], '
            'Not[TrueQ[Abs[#[[3]] - NumericalValue[Mass[#[[1]]]]] <= '
            '10^-6 Max[1, Abs[NumericalValue[Mass[#[[1]]]]]]]] &];',
            'If[portableBadMasses =!= {}, Print["PORTABLE_DECLARED_MASSES_MISMATCH: ", portableBadMasses]; Quit[17]];',
            f'WriteUFO[LPortable, Output -> {_wl(str(ufo))}, AddDecays -> {str(analytic_decays)}, RemoveGhosts -> True];',
            f'If[!FileExistsQ[{_wl(str(ufo / "vertices.py"))}], Print["PORTABLE_EXPORT_FAILED"]; Quit[15]];',
            'Print["PORTABLE_EXPORT_SUCCESS"];',
            'Quit[0];',
        ]
        # Parse the package load before parsing any FeynRules symbols: wrapping
        # the entire script in one held expression would create Global` names
        # before FeynRules` is put on $ContextPath.
        driver.write_text("\n".join(lines[:5]) + "\nCheckAbort[\n"
                          + "\n".join(lines[5:]) + "\n, Quit[18]];\n")
        log = output / "feynrules.log"
        try:
            with log.open("w") as handle:
                result = subprocess.run([str(kernel), "-noinit", "-noprompt", "-script", str(driver)],
                                        cwd=source, env=environment(self.runtime), stdout=handle,
                                        stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FeynRules timed out; source and log retained in {output}.") from exc
        log_text = log.read_text(errors='replace')
        _check_unknown_ufo_fields(ufo, log, log_text)
        # Wolfram can return exit status zero after aborting evaluation at a
        # resource limit. Preserve its actual diagnostic instead of treating
        # the missing export marker as a physics or generic execution failure.
        resource_message = re.search(
            r'(?m)^\s*(?:\$RecursionLimit::reclim|\$IterationLimit::itlim|General::nomem)\s*:[^\n]*',
            log_text)
        if resource_message is None:
            resource_message = re.search(r'SystemException\["MemoryAllocationFailure"[^\n]*', log_text)
        if resource_message is not None:
            from workflow_errors import ResourceLimitError
            context = 'FeynRules symbolic model construction/export'
            if 'Checking for hermiticity' in log_text and 'Starting Feynman rule calculation.' not in log_text:
                context = 'FeynRules hermiticity construction'
            setting = ('max_wolfram_recursion' if '$RecursionLimit::' in resource_message.group(0)
                       else 'max_wolfram_iterations' if '$IterationLimit::' in resource_message.group(0) else None)
            configured = f' Configured compiler_limits.{setting}={wolfram_limits[setting]}.' if setting else ''
            raise ResourceLimitError(f'{context} exceeded a Wolfram resource limit: '
                                     f'{resource_message.group(0).strip()}{configured} Source and full log retained at {log}.')
        if any(marker in log_text for marker in ('PORTABLE_HERMITICITY_FAILED', 'PORTABLE_KINETIC_CHECK_FAILED',
                'PORTABLE_MASS_CHECK_FAILED', 'PORTABLE_SPECTRUM_CHECK_FAILED', 'PORTABLE_DECLARED_MASSES_MISMATCH',
                'PORTABLE_EXPORT_FAILED')):
            raise CompilerResolutionError(f'Generated FeynRules model failed its physical/export checks; inspect {log}.')
        if result.returncode or not (ufo / "vertices.py").is_file() or "PORTABLE_EXPORT_SUCCESS" not in log_text:
            raise RuntimeError(f"FeynRules export failed; inspect {log} (exit {result.returncode}).")
        _python3_ufo(ufo)
        _ensure_variable_widths(ufo, particles)
        card = _write_parameter_card(ufo, output, paths.get("python") or sys.executable,
                                     environment(self.runtime))
        masses = {p["pdg"]: p["mass"] for p in particles}
        if self.basis is not None:
            masses[25] = float(self.basis["masses"][0])
        artifact = UFOModel(ufo, self.model_name, tuple(particles), masses,
                            parameter_card=card, source_directory=source,
                            selected_ids=self.lagrangian.selected_ids)
        # Import and evaluate through the actual pinned MadGraph loader before
        # reporting success: Python syntax/import checks alone cannot detect
        # all UFO naming, mass, or two-point interaction inconsistencies.
        from madgraph_runner import validate_ufo, _full_card_digest
        validation = validate_ufo(artifact, runtime=self.runtime, output_dir=output / 'madgraph_import')
        _verify_exported_particles(particles, validation['particles'])
        artifact.parameter_card = validation['parameter_card']
        artifact.saved_model_digest = validation['model_digest']
        artifact.saved_parameter_card_digest = _full_card_digest(artifact.parameter_card)
        return artifact

    def _automatic_sources(self, source, fr):
        # Modify a generated SM copy. The vendor model is never changed. Its
        # Higgs doublet definition feeds all SM gauge and quark/lepton Yukawas.
        sm = (fr / "Models" / "SM" / "SM.fr").read_text()
        b = self.basis
        symbols = ["H"] + [f"pmS{i}" for i in range(1, len(self.fields) + 1)]
        mix_names = [[f"pmR{i}{j}" for j in range(len(symbols))] for i in range(len(symbols))]
        def field_expression(i):
            terms = [f"{mix_names[i][j]}*{symbols[j]}" for j in range(len(symbols))
                     if b["transformation"][i, j] != 0]
            return "(" + " + ".join(terms) + ")"
        original = "Phi[2] -> (vev + H + I G0)/Sqrt[2]"
        if original not in sm or "Value            -> MH^2/(2*vev^2)" not in sm:
            raise CompilerResolutionError("Bundled SM template changed; its Higgs definitions must be reviewed before export.")
        sm = sm.replace(original, f"Phi[2] -> (vev + {field_expression(0)} + I G0)/Sqrt[2]")
        sm = sm.replace("Value            -> MH^2/(2*vev^2)", "Value            -> pmBareH^2/(2*vev^2)")
        if "+ muH^2 Phibar[ii] Phi[ii]" not in sm:
            raise CompilerResolutionError("Bundled SM Higgs potential changed; review its vacuum condition.")
        sm = sm.replace("+ muH^2 Phibar[ii] Phi[ii]", "+ pmMuH2 Phibar[ii] Phi[ii]")
        sm = sm.replace("Mass            -> {MH,125}", f"Mass            -> {{MH,{_wl(b['masses'][0])}}}")
        from sm_parameters import apply_sm_inputs
        sm = apply_sm_inputs(sm, self.inputs)
        smfile = source / "SM_portable.fr"
        smfile.write_text(sm)
        parameters = [_internal("pmBareH", b["higgs_mass_before_mixing"]),
                      _internal("pmMuH2", b["mu_h_squared"])]
        for i, row in enumerate(b["transformation"]):
            parameters.extend(_internal(mix_names[i][j], value) for j, value in enumerate(row))
        classes, particles = [], []
        for i, f in enumerate(self.fields, 1):
            name = f.name.lower()
            pdg = 9000000 + i
            width = _number(self.inputs["widths"][f.name], f"widths.{f.name}", real=True)
            if width < 0:
                raise ModelResolutionError("Decay widths cannot be negative.")
            mass = float(b["masses"][i])
            classes.append(f'''S[{100+i}] == {{ClassName -> {symbols[i]}, SelfConjugate -> True,
                Mass -> {{pmM{i}, {_wl(mass)}}}, Width -> {{pmW{i}, {_wl(width)}}},
                ParticleName -> {_wl(name)}, PDG -> {pdg},
                PropagatorLabel -> {_wl(f.name)}, PropagatorType -> D,
                PropagatorArrow -> None, FullName -> {_wl(f.name + " mass eigenstate")}}}''')
            particles.append(dict(name=name, antiname=name, pdg=pdg, source=f.name,
                                  spin=1, mass=mass, width=width))
        terms = []
        for op in self.operators:
            symbol = "pmC" + op.operator_id[1:]
            # The bundled SM assigns QED=-1 to vev. Retain the Higgs-field
            # count in the coefficient so each replacement H -> vev lowers
            # the vertex order by one instead of creating negative orders.
            higgs = sum(count for name, count in op.factors if name in ("H", "H†"))
            parameters.append(_internal(symbol, op.coefficients[0], order=True, qed_order=higgs))
            # An explicitly assigned zero is a resolved zero term, not a
            # missing operator. Keep its parameter/provenance in the source.
            if op.coefficients[0] == 0:
                terms.append(f"0 (* selected {op.operator_id}: coefficient is zero *)")
                continue
            factors = []
            for name, count in op.factors:
                if name == "H†":
                    continue
                if name == "H":
                    factor = f"((vev + {field_expression(0)})^2/2)"
                elif name.startswith("D("):
                    i = b["names"].index(name[2:-1])
                    factor = "(" + " + ".join(f"{mix_names[i][j]}*del[{symbols[j]},mu]"
                                               for j in range(len(symbols)) if b["transformation"][i, j] != 0) + ")"
                else:
                    factor = field_expression(b["names"].index(name))
                factors.append(f"({factor})^{count}" if count != 1 else factor)
            terms.append(symbol + "*" + "*".join(factors))
        extension = source / "extension.fr"
        extension.write_text(
            'M$ModelName = ' + _wl(self.model_name) + ';\n'
            'M$Information = {Authors -> {"Lagrangian maker"}, Version -> "1.0"};\n'
            'M$InteractionOrderHierarchy = {{QCD, 1}, {QED, 2}, {NP, 1}};\n'
            'M$ClassesDescription = {\n' + ',\n'.join(classes) + '\n};\n'
            'M$Parameters = {\n' + ',\n'.join(parameters) + '\n};\n'
            'LNew := Block[{mu}, ' + (' + '.join(terms) or '0') + '];\n')
        return [smfile, extension], "LSM + LNew", particles


def _internal(name, value, *, order=False, qed_order=0):
    complex_flag = "True" if complex(value).imag else "False"
    orders = ['{NP, 1}'] if order else []
    if qed_order:
        orders.append(f'{{QED, {qed_order}}}')
    interaction_order = ('InteractionOrder -> '
                         + (orders[0] if len(orders) == 1 else '{' + ', '.join(orders) + '}')
                         + ', ') if orders else ''
    return (f'{name} == {{ParameterType -> Internal, Value -> {_wl(value)}, '
            f'ComplexParameter -> {complex_flag}, '
            + interaction_order
            + f'Description -> {_wl(name)}}}')


def _ufo_syntax_location(number, column, line):
    column = column or 1
    excerpt = line.rstrip('\r\n')[max(0, column - 81):column + 100].strip()
    return f'line {number}, column {column}, source {excerpt!r}'


def _python3_ufo(ufo):
    """Convert the pinned UFO template's remaining Python-2 raise statements.

    Python 3.11 includes lib2to3, so this needs no extra package. Imports retain
    the UFO convention; applying the import fixer would break standalone cards.
    Every generated file is parsed again, making syntax failures explicit.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lib2to3.refactor import RefactoringTool
        from lib2to3.pgen2.parse import ParseError
        from lib2to3.pgen2.tokenize import TokenError
    fixer = None
    for path in Path(ufo).glob("*.py"):
        text = path.read_text()
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as original:
            if fixer is None:
                fixer = RefactoringTool(["lib2to3.fixes.fix_raise", "lib2to3.fixes.fix_except",
                                         "lib2to3.fixes.fix_print", "lib2to3.fixes.fix_exec"])
            try:
                text = str(fixer.refactor_string(text, str(path)))
                ast.parse(text, filename=str(path))
            except (ParseError, TokenError, SyntaxError) as exc:
                # lib2to3 often omits the filename. Keep the original Python
                # diagnostic and the source at its offending column, including
                # when the offending line is a long generated vertex.
                location = _ufo_syntax_location(original.lineno, original.offset, original.text or '')
                if isinstance(exc, SyntaxError):
                    second_location = _ufo_syntax_location(exc.lineno, exc.offset, exc.text or '')
                else:
                    # A legacy print/raise can precede a separate broken
                    # vertex. lib2to3's parser reports zero-based columns.
                    number, offset = exc.context[1] if isinstance(exc, ParseError) else exc.args[1]
                    with path.open() as source:
                        line = next((line for i, line in enumerate(source, 1) if i == number), '')
                    second_location = _ufo_syntax_location(number, offset + 1, line)
                extra = f' ({second_location})' if second_location != location else ''
                raise CompilerResolutionError(
                    f'Invalid generated UFO Python in {path}, {location}: {original.msg}. '
                    f'Conversion diagnostic: {str(exc)[:240]}{extra}'
                ) from None
            path.write_text(text)


def _ensure_variable_widths(ufo, particles):
    """Keep zero initial widths as external DECAY parameters for MadWidth.

    FeynRules replaces zero-valued widths by Param.ZERO, which MadGraph treats
    as an assertion of stability and excludes from width calculation. A zero
    initial guess is not such an assertion. Only new-particle width references
    are repaired; masses, couplings and analytic decay expressions are intact.
    """
    path = Path(ufo) / "particles.py"
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    metadata = {abs(int(p['pdg'])): p for p in particles}
    changes, parameters = [], []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Particle':
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        pdg = abs(int(ast.literal_eval(kwargs['pdg_code'])))
        width = kwargs.get('width')
        if pdg not in metadata or not isinstance(width, ast.Attribute) or width.attr != 'ZERO':
            continue
        name = f'PMWidth{pdg}'
        changes.append((starts[width.lineno-1]+width.col_offset,
                        starts[width.end_lineno-1]+width.end_col_offset, 'Param.'+name))
        value = float(metadata[pdg].get('width', 0.))
        parameters.append(f"\n{name} = Parameter(name={name!r}, nature='external', type='real', "
                          f"value={value!r}, texname={name!r}, lhablock='DECAY', lhacode=[{pdg}])\n")
    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    if changes:
        path.write_text(source)
        parameter_file = Path(ufo) / 'parameters.py'
        parameter_file.write_text(parameter_file.read_text() + '\n# Variable initial widths for MadWidth.\n' + ''.join(parameters))


def _write_parameter_card(ufo, output, python, env):
    # Import in a fresh interpreter: UFO modules use unqualified imports and
    # must not collide with a previously loaded model in this notebook kernel.
    card = output / "param_card.dat"
    # The pinned FR template writes only six digits after the decimal. This
    # changes mixed eigenmasses on round-trip and can spoil narrow splittings;
    # retain binary64 precision in this generated model's card writer.
    writer = Path(ufo) / 'write_param_card.py'
    writer.write_text(writer.read_text().replace('%e', '%.16e').replace('%f', '%.16e'))
    script = ("import sys, importlib, cmath, math\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[1])\n"
              "for name in ('particles', 'parameters', 'coupling_orders', 'couplings', 'lorentz', 'vertices'):\n"
              "    importlib.import_module(name)\n"
              "if Path('decays.py').is_file(): importlib.import_module('decays')\n"
              "import object_library, function_library\n"
              "namespace = dict(vars(function_library), cmath=cmath, complex=complex, abs=abs)\n"
              "pending = []\n"
              "for parameter in object_library.all_parameters:\n"
              "    if parameter.nature == 'external': namespace[parameter.name] = parameter.value\n"
              "    else: pending.append(parameter)\n"
              "while pending:\n"
              "    remaining = []\n"
              "    for parameter in pending:\n"
              "        try: namespace[parameter.name] = eval(str(parameter.value), namespace)\n"
              "        except NameError: remaining.append(parameter)\n"
              "    if len(remaining) == len(pending):\n"
              "        raise RuntimeError('Unresolved UFO parameter dependencies: ' + ', '.join(p.name+'='+str(p.value) for p in remaining))\n"
              "    pending = remaining\n"
              "for coupling in object_library.all_couplings:\n"
              "    values = coupling.value.values() if isinstance(coupling.value, dict) else [coupling.value]\n"
              "    for expression in values:\n"
              "        value = complex(eval(str(expression), namespace))\n"
              "        if not math.isfinite(value.real) or not math.isfinite(value.imag):\n"
              "            raise RuntimeError('Nonfinite UFO coupling: '+coupling.name)\n"
              "import write_param_card\nwrite_param_card.ParamCardWriter(sys.argv[2])\n")
    run = subprocess.run([str(python), "-c", script, str(ufo), str(card)],
                         cwd=ufo, env=env, capture_output=True, text=True)
    if run.returncode or not card.is_file():
        (output / "parameter_card.log").write_text(run.stdout + run.stderr)
        raise CompilerResolutionError(f"UFO numerical validation or parameter-card generation failed; inspect {output / 'parameter_card.log'}.")
    return card


def prepare_model(lagrangian, inputs, runtime=None):
    """Compile the spectrum about the fixed SM electroweak vacuum."""
    from fixed_sm import check_fixed_sm_model, normalize_fixed_sm
    fields = _fields(lagrangian)
    backend_preflight(fields)
    values = normalize_fixed_sm(inputs, fields)
    prepared = _prepare_model(lagrangian, values, fields, runtime)
    return check_fixed_sm_model(prepared)


def _prepare_model(lagrangian, inputs, fields, runtime):
    """Lower selected operators into normalized component fields."""
    name = inputs.get("model_name", "SM_" + "_".join(f.name for f in fields))
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ModelResolutionError("model_name must contain only letters, numbers and underscores, starting with a letter.")
    inputs["model_name"] = name
    # Every selected row is shape-validated. Compiler paths may account for an
    # exact-zero coefficient vector without constructing its vanishing vertex.
    manual = inputs.get("feynrules")
    if not manual:
        if getattr(lagrangian._source, 'sm_generations', 3) != 3:
            raise CompilerResolutionError('Automatic physical SM export currently requires all three SM generations; other generation counts need a generalized SM template.')
        inputs.setdefault('widths', {})
        for field in fields:
            width = _number(inputs['widths'].get(field.name, 0.), f'widths.{field.name}', real=True)
            if width < 0:
                raise ModelResolutionError(f'widths.{field.name} must be nonnegative.')
            inputs['widths'][field.name] = width
        from generic_scalar_compiler import uses_generic, prepare as prepare_generic
        if uses_generic(fields):
            return prepare_generic(lagrangian, inputs, runtime)
        from generic_vector_compiler import uses_generic as generic_vector_uses, prepare as prepare_generic_vector
        if generic_vector_uses(fields):
            return prepare_generic_vector(lagrangian, inputs, runtime)
        from generic_fermion_compiler import uses_generic as fermion_uses_generic, prepare as prepare_fermion
        if fermion_uses_generic(fields):
            return prepare_fermion(lagrangian, inputs, runtime)
        from generic_majorana_compiler import uses_generic as majorana_uses_generic, prepare as prepare_majorana
        if majorana_uses_generic(fields):
            return prepare_majorana(lagrangian, inputs, runtime)
        from mixed_scalar_fermion_compiler import uses_generic as mixed_uses_generic, prepare as prepare_mixed
        if mixed_uses_generic(fields):
            return prepare_mixed(lagrangian, inputs, runtime)
        from generic_joint_scalar_compiler import uses_generic as joint_uses_generic, prepare as prepare_joint
        if joint_uses_generic(fields):
            return prepare_joint(lagrangian, inputs, runtime)
        from generic_color_compiler import uses_generic as colour_uses_generic, prepare as prepare_colour
        if colour_uses_generic(fields, lagrangian.operators):
            return prepare_colour(lagrangian, inputs, runtime)
        from neutral_fermion_compiler import prepare as prepare_neutral
        neutral_model = prepare_neutral(lagrangian, inputs, runtime)
        if neutral_model is not None:
            return neutral_model
        from vector_current_compiler import prepare as prepare_vector
        vector_model = prepare_vector(lagrangian, inputs, runtime)
        if vector_model is not None:
            return vector_model
        from gauge_matter_compiler import prepare as prepare_gauge
        gauge_model = prepare_gauge(lagrangian, inputs, runtime)
        if gauge_model is not None:
            return gauge_model
    unresolved = {}
    kinds = {}
    for row in lagrangian.operators:
        kind, reason = _classify(row, fields)
        kinds[row["id"]] = kind
        if reason:
            unresolved[row["id"]] = reason
    if not manual:
        for f in fields:
            if not _automatic_field(f):
                unresolved[f.name] = ("vector kinetic structure and a physical gauge/effective-model specification are required"
                                      if f.spin == "vector" else "explicit component fields and physical mass basis are required")
        if unresolved:
            raise CompilerResolutionError("Unresolved selected model:\n" + "\n".join(f"  {key}: {reason}" for key, reason in unresolved.items()))
    coefficients = _coefficients(lagrangian, inputs)
    operators = []
    if manual:
        if manual.get("physical_basis") is not True or not manual.get("model_files"):
            raise ModelResolutionError("Explicit FeynRules models require model_files and physical_basis=True after resolving symmetry breaking and mixing.")
        mapping = manual.get("operator_map", {})
        if set(mapping) != set(lagrangian.selected_ids):
            raise ModelResolutionError("feynrules.operator_map must contain exactly every selected operator ID.")
        if any(f.spin == "vector" and not manual.get("vector_specifications", {}).get(f.name) for f in fields):
            raise ModelResolutionError("Each vector needs an explicit vector_specifications entry describing its kinetic and gauge/effective-model meaning.")
        particles = manual.get("particles", [])
        required_keys = {"name", "pdg", "source", "mass", "width", "spin"}
        if not particles or any(not required_keys <= set(p) for p in particles):
            raise ModelResolutionError("Explicit FeynRules particles need name, pdg, source, mass, width and spin.")
        if not {f.name for f in fields} <= {p["source"] for p in particles}:
            raise ModelResolutionError("Every new multiplet needs physical component particles in feynrules.particles.")
        for row in lagrangian.operators:
            expressions = mapping[row["id"]]
            if isinstance(expressions, str):
                expressions = [expressions]
            if (not isinstance(expressions, (list, tuple)) or len(expressions) != row["multiplicity"]
                    or any(not isinstance(x, str) or not x.strip() for x in expressions)):
                raise ModelResolutionError(f"{row['id']} requires exactly {row['multiplicity']} explicit contraction expressions.")
            operators.append(ResolvedOperator(row["id"], "explicit", tuple((f["name"], f["count"]) for f in row["factors"]),
                                              coefficients[row["id"]], tuple(expressions),
                                              row["field_content"] != row["conjugate_content"]))
        return PreparedModel(lagrangian, inputs, tuple(operators), fields, None, runtime, manual)
    inputs.setdefault("widths", {})
    for f in fields:
        inputs["widths"].setdefault(f.name, 0.0)
    for row in lagrangian.operators:
        operators.append(ResolvedOperator(row["id"], kinds[row["id"]],
                                          tuple((f["name"], f["count"]) for f in row["factors"]), coefficients[row["id"]]))
    basis = _scalar_basis(fields, operators, inputs)
    return PreparedModel(lagrangian, inputs, tuple(operators), fields, basis, runtime)


def scalar_portal_example(lagrangian, *, masses=None, portal=-0.05, quartic=-0.01):
    """An explicitly labelled benchmark dictionary for real scalar singlets.

    Select 'all' first. All odd and mixed-singlet terms are zero. Each field has
    c(dX)^2=1/2, c(H-dagger H)X^2=portal, cX^4=quartic; its quadratic coefficient
    includes the Higgs-VEV shift so the requested physical mass is obtained.
    This is a benchmark, not a replacement for the user's chosen interactions.
    """
    fields = _fields(lagrangian)
    if not all(_automatic_field(f) for f in fields):
        raise ModelResolutionError("scalar_portal_example requires only real neutral scalar singlets.")
    values = required_inputs(lagrangian)
    masses = masses or {f.name: 600.0 if i == 0 else 200.0 for i, f in enumerate(fields)}
    if set(masses) != {f.name for f in fields}:
        raise ModelResolutionError("Supply one positive mass per scalar field.")
    v = (math.sqrt(2.0) * values["sm_inputs"]["Gf"]) ** -0.5
    for row in lagrangian.operators:
        counts = {f["name"]: f["count"] for f in row["factors"]}
        value = 0.0
        for f in fields:
            m = _number(masses[f.name], f"masses.{f.name}", real=True)
            if m <= 0:
                raise ModelResolutionError("Example masses must be positive.")
            if counts == {f"D({f.name})": 2}:
                value = 0.5
            elif counts == {f.name: 2}:
                value = -0.5 * m * m - portal * v * v / 2
            elif counts == {"H": 1, "H†": 1, f.name: 2}:
                value = portal
            elif counts == {f.name: 4}:
                value = quartic
        values["coefficients"][row["id"]] = value
    # Catch incomplete selections rather than silently invent omitted terms.
    trial = prepare_model(lagrangian, values)
    if any(not math.isclose(trial.masses[f.name], masses[f.name], rel_tol=1e-9) for f in fields):
        raise ModelResolutionError("The example requires the quadratic, kinetic and Higgs-portal terms for every scalar; select 'all' first.")
    return values
