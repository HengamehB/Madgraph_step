"""Spin-independent, inspectable inputs for a selected operator catalogue.

This layer binds numerical coefficients to the actual model, rather than to
example-specific notions of a portal or quartic. It does not run Mathematica,
resolve a vacuum, or claim that every catalogue representation can be exported.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from math import isqrt
from pprint import pformat
import re

import generic_colour_yukawa as colour_yukawa
from lagrangian_maker import make_lagrangian
from feynrules_export import ModelResolutionError, _number, required_inputs


_SCHEMA = "operator-inputs-v1"


def _row_identity(row):
    """Stable physical metadata, deliberately excluding display row numbers."""
    return {key: row[key] for key in (
        "field_content", "conjugate_content", "dimension", "coefficient_dimension",
        "derivatives", "multiplicity", "factors")}


def _digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _keys(rows):
    counts = Counter(row["field_content"] for row in rows)
    return {row["id"]: (row["field_content"] if counts[row["field_content"]] == 1
                        else row["field_content"] + " [" + _digest(_row_identity(row))[:12] + "]")
            for row in rows}


def _kind(row, fields, compiler_limits=None, operators=None):
    from feynrules_export import _classify
    from neutral_fermion_compiler import supports_fields as neutral_supports, classify as neutral_classify
    from gauge_matter_compiler import supports_fields as gauge_supports, classify as gauge_classify
    from vector_current_compiler import supports_fields as vector_supports, classify as vector_classify
    from generic_scalar_compiler import uses_generic, classify as generic_classify
    from generic_color_compiler import uses_generic as colour_uses_generic, classify as colour_classify
    from generic_joint_scalar_compiler import uses_generic as joint_uses_generic, classify as joint_classify
    from generic_fermion_compiler import uses_generic as fermion_uses_generic, classify as fermion_classify
    from generic_majorana_compiler import uses_generic as majorana_uses_generic, classify as majorana_classify
    from mixed_scalar_fermion_compiler import uses_generic as mixed_uses_generic, classify as mixed_classify
    from generic_vector_compiler import uses_generic as vector_uses_generic, classify as generic_vector_classify
    if vector_uses_generic(fields):
        return generic_vector_classify(row, fields)
    if mixed_uses_generic(fields):
        return mixed_classify(row, fields)
    if uses_generic(fields):
        return generic_classify(row, fields)
    if fermion_uses_generic(fields):
        return fermion_classify(row, fields)
    if majorana_uses_generic(fields):
        return majorana_classify(row, fields)
    if joint_uses_generic(fields):
        return joint_classify(row, fields, limits=compiler_limits)
    if colour_uses_generic(fields, operators):
        return colour_classify(row, fields, limits=compiler_limits)

    # Match the compiler dispatch order, retaining the scalar sub-sector of a
    # mixed scalar/neutral-fermion model when the neutral classifier defers.
    classifiers = []
    if neutral_supports(fields):
        classifiers.append(neutral_classify)
    if vector_supports(fields):
        classifiers.append(vector_classify)
    if gauge_supports(fields):
        classifiers.append(gauge_classify)
    classifiers.append(_classify)
    reason = ""
    for classifier in classifiers:
        kind, reason = classifier(row, fields)
        if kind:
            return kind, ""
    return None, reason


def _hints(row, fields, compiler_limits=None, operators=None):
    kind, reason = _kind(row, fields, compiler_limits, operators)
    counts = {part["name"]: part["count"] for part in row["factors"]}
    names = {field.name: field for field in fields}
    labels = tuple(f"invariant {i}" for i in range(1, row["multiplicity"] + 1))
    hint = "Coefficient multiplies the stated monomial; no implicit factorial."
    if kind in colour_yukawa.KINDS:
        labels = colour_yukawa.basis_labels(row, fields)
        hint = colour_yukawa.normalization(kind)
    elif kind in ('generic_scalar_potential', 'generic_scalar_kinetic'):
        from generic_scalar_compiler import basis_labels, BASIS_VERSION
        labels = basis_labels(row, fields, limits=compiler_limits)
        hint = (f"Explicit physical-component basis {BASIS_VERSION}; h is the neutral Higgs radial component before its VEV shift. "
                "The displayed ordered invariant polynomials define coefficient normalization; do not reuse coefficients from a different basis.")
    elif kind in ('joint_scalar_potential', 'joint_scalar_kinetic'):
        from generic_joint_scalar_compiler import basis_labels, BASIS_VERSION
        labels = basis_labels(row, fields, limits=compiler_limits)
        hint = (f"Joint weak/SU(3) invariant basis {BASIS_VERSION}; Bose symmetry is applied to the combined tensor. "
                "Coefficients multiply the displayed contractions, with no implicit factorial; weak components are ordered by descending T3.")
    elif kind == 'colour_scalar_potential':
        from generic_color_compiler import basis_labels, BASIS_VERSION
        labels = basis_labels(row, fields, limits=compiler_limits)
        hint = f"Ordered native SU(3) tensor basis {BASIS_VERSION}; coefficients multiply the displayed contractions."
    elif kind == 'colour_scalar_kinetic':
        hint = ('Native-colour covariant bilinear; diagonal norm coefficients are 1 for complex and 1/2 for real scalars. '
                'Off-diagonal and holomorphic terms contribute to the full kinetic matrix, including the stated Hermitian conjugate.')
    elif kind == 'colour_dirac_kinetic':
        hint = "Weyl covariant kinetic term including its Hermitian conjugate; canonical coefficient is 1/2."
    elif kind == 'colour_dirac_mass':
        hint = "Weyl-pair Dirac mass plus Hermitian conjugate; coefficient is -m before normalization."
    elif kind and kind.startswith('generic_scalar_yukawa_'):
        from generic_scalar_compiler import basis_labels
        labels = basis_labels(row, fields)
        hint = "Catalogue Weyl Yukawa plus h.c.; row-major doublet-generation, charge-conjugate-singlet-generation order."
    elif kind == 'generic_scalar_lepton_yukawa':
        from generic_scalar_yukawa import basis_labels
        labels = basis_labels(row, fields, limits=compiler_limits)
        hint = ('Explicit Lorentz/weak/flavour tensor basis. B(L_i,L_j)=HC[LLbar_i ProjP CC(LL_j)]; '
                'B(ec_i,ec_j)=lRbar_i ProjM CC(lR_j); the conjugate bilinear is its HC. No implicit factorial.')
    elif kind in ('generic_dirac_mass','generic_dirac_kinetic'):
        from generic_fermion_compiler import basis_labels
        labels = basis_labels(row, fields)
        hint = "Gauge identity contraction with complete chiral kinetic normalization and complex Dirac mass diagonalization."
    elif kind in ('generic_majorana_mass', 'generic_majorana_kinetic'):
        from generic_majorana_compiler import basis_labels
        labels = basis_labels(row, fields)
        hint = ('Real-representation invariant metric and Hermitian kinetic normalization followed by complex symmetric Takagi diagonalization; '
                'each mass eigenmultiplet contains one neutral Majorana component and charged Dirac pairs.')
    elif kind == 'mixed_scalar_dirac_yukawa':
        from mixed_scalar_fermion_compiler import basis_labels
        labels = basis_labels(row, fields)
        hint = 'The singlet VEV contributes to the Dirac mass; chiral normalization and phase removal transform every scalar and pseudoscalar interaction.'
    elif kind and kind.startswith('generic_vector_'):
        from generic_vector_compiler import basis_labels
        labels = basis_labels(row, fields, limits=compiler_limits)
        hint = ('Explicit massive matter-vector contractions. The full covariant derivative expressions are retained; '
                'quadratic ghost freedom requires the c1+c2+c3 matrix to vanish, with a positive physical kinetic form.')
    if kind in ("weyl_kinetic", "dirac_covariant_kinetic"):
        bare = next(name[2:-1] for name in counts if name.startswith("D("))
        hint = ("Includes the Hermitian conjugate; canonical diagonal coefficient is 1/2."
                if counts.get(bare + "†") else
                "Off-diagonal Weyl kinetic mixing, including the Hermitian conjugate.")
    elif kind in ("scalar_covariant_kinetic", "doublet_kinetic"):
        hint = "Covariant scalar kinetic term; canonical coefficient is 1."
    elif kind == "triplet_kinetic" or kind == "kinetic" and row["multiplicity"] == 1:
        hint = ("Real scalar kinetic term; canonical diagonal coefficient is 1/2."
                if len(counts) == 1 else "Mixed real-scalar kinetic coefficient; no implicit factor.")
    elif kind in ("weyl_mass", "dirac_mass"):
        hint = ("Weyl-pair mass plus h.c.; identical-pair coefficient is -m/2; "
                "distinct-pair coefficient is -m before normalization and mixing.")
    elif kind == "neutrino_portal":
        labels = tuple(("electron", "muon", "tau")[i] if i < 3 else f"lepton {i+1}"
                       for i in range(row["multiplicity"]))
        hint = "Neutral-fermion Yukawa plus h.c.; lepton flavour order " + ", ".join(labels) + "."
    elif kind and kind.startswith("current_"):
        generations = isqrt(row["multiplicity"])
        labels = (tuple(f"{i}{i}" for i in range(1, generations+1))
                  + tuple(f"{part}{i}{j}" for i in range(1, generations+1)
                          for j in range(i+1, generations+1) for part in ("Re", "Im")))
        hint = ("Hermitian current basis: Reij = Jij+Jji, Imij = i(Jij-Jji); "
                "uc/dc/ec denote charge-conjugate Weyl fields.")
    elif kind in ("proca_kinetic", "kinetic") and row["multiplicity"] == 4:
        labels = ("d_mu V_nu d^mu V^nu", "(div V)^2", "d_mu V_nu d^nu V^mu", "epsilon dV dV")
        hint = "Proca kinetic basis; canonical choice [-0.5, 0, 0.5, 0]; require c2+c3=-c1."
    elif kind == "cubic_derivative":
        labels = ("V^2 div V", "V_mu V_nu d^mu V^nu")
        hint = "After integration by parts the effective coefficient is c1-c2/2."
    elif kind == "higgs_current":
        hint = "i V_mu H† D^mu H + h.c.; its imaginary part is tied to the Higgs-divergence coefficient."
    elif kind == "higgs_divergence":
        hint = "(div V)(H†H); require c_div+Im(c_Hcurrent)=0 in the supported Proca sector."
    elif kind == "hypercharge_mixing":
        hint = "(B_mu_nu V^mu_nu+i dualB_mu_nu V^mu_nu)/2 + h.c.; imaginary part is a boundary term."
    elif kind and kind.startswith("doublet_yukawa_"):
        generations = isqrt(row["multiplicity"])
        labels = tuple(f"{i}{j}" for i in range(1, generations+1) for j in range(1, generations+1))
        hint = ("Catalogue Weyl Yukawa plus h.c.; row-major doublet-generation, "
                "charge-conjugate-singlet-generation order; not a direct four-component Yukawa coefficient.")
    elif kind == "doublet_potential":
        from gauge_matter_compiler import _doublet_pairings
        labels = tuple(" ".join(f"({left}† {right})" for left, right in pairs)
                       for pairs in _doublet_pairings(Counter(counts)))
        hint = "Ordered products of SU(2) doublet inner products; no implicit factorial."
    elif kind == "potential" and len(counts) == 1:
        name, power = next(iter(counts.items()))
        if name in names and power == 2:
            hint = "Real-scalar quadratic coefficient is -m^2/2 only before VEV shifts and mixing."
    if reason:
        hint = "Automatic export unresolved: " + reason + ". No contraction ordering is assumed."
    return kind, reason, labels, hint


class ModelDefinition:
    """One concrete model and its selected operators, independent of spin.

    ``selected_operators`` accepts ``'all'``, current catalogue IDs, or readable
    keys printed by :meth:`show_inputs`. When ``any`` gives multiple scenarios,
    choose one explicitly. Numerical inputs must carry the literal signature
    printed in the template; regenerating it is an intentional model change.
    """

    def __init__(self, n_particles, x, y=None, *, selected_operators="all", scenario=None,
                 exact_dimension=None, sm_generations=3, compiler_limits=None):
        from compiler_limits import normalize_limits
        self.compiler_limits = normalize_limits(compiler_limits)
        self.result = make_lagrangian(n_particles, x, y, exact_dimension=exact_dimension,
                                     sm_generations=sm_generations, show=False)
        complete = self.result.select("all", scenario=scenario, show=False)
        self._keys = _keys(complete.operators)
        self._ids = {key: operator_id for operator_id, key in self._keys.items()}
        if isinstance(selected_operators, str) and selected_operators.lower().strip() == "all":
            selected = "all"
        else:
            requested = ([selected_operators] if isinstance(selected_operators, str)
                         else selected_operators)
            if isinstance(requested, Mapping) or not hasattr(requested, "__iter__"):
                raise ModelResolutionError("selected_operators must be 'all' or a list of operator keys/IDs.")
            selected = [self._resolve_key(key) for key in requested]
        self.lagrangian = self.result.select(selected, scenario=complete.scenario, show=False)
        self.fields = self.result.models[complete.scenario - 1]
        identity = dict(schema=_SCHEMA, sm_generations=sm_generations,
                        fields=[dict(name=f.name, spin=f.spin, kind=f.kind, su2=f.su2,
                                     su3=f.su3, hypercharge=str(f.hypercharge)) for f in self.fields],
                        operators=sorted((_row_identity(row) for row in self.lagrangian.operators),
                                         key=lambda row: json.dumps(row, sort_keys=True)))
        from generic_scalar_compiler import uses_generic, BASIS_VERSION
        if uses_generic(self.fields):
            identity['component_basis'] = BASIS_VERSION
        from generic_color_compiler import uses_generic as colour_uses_generic, BASIS_VERSION as COLOUR_BASIS_VERSION
        if colour_uses_generic(self.fields, self.lagrangian.operators):
            identity['colour_basis'] = COLOUR_BASIS_VERSION
        from generic_joint_scalar_compiler import uses_generic as joint_uses_generic, BASIS_VERSION as JOINT_BASIS_VERSION
        if joint_uses_generic(self.fields):
            identity['joint_scalar_basis'] = JOINT_BASIS_VERSION
        yukawa_bases = {self._keys[row['id']]: colour_yukawa.basis_fingerprint(row, self.fields)
                        for row in self.lagrangian.operators
                        if colour_yukawa.classify(row, self.fields)}
        if yukawa_bases:
            identity['coloured_yukawa_bases'] = yukawa_bases
        from generic_fermion_compiler import uses_generic as fermion_uses_generic, BASIS_VERSION as FERMION_BASIS_VERSION
        if fermion_uses_generic(self.fields):
            identity['fermion_basis'] = FERMION_BASIS_VERSION
        from generic_majorana_compiler import uses_generic as majorana_uses_generic, BASIS_VERSION as MAJORANA_BASIS_VERSION
        if majorana_uses_generic(self.fields):
            identity['majorana_basis'] = MAJORANA_BASIS_VERSION
        from mixed_scalar_fermion_compiler import uses_generic as mixed_uses_generic, BASIS_VERSION as MIXED_BASIS_VERSION
        if mixed_uses_generic(self.fields):
            identity['mixed_scalar_fermion_basis'] = MIXED_BASIS_VERSION
        from generic_vector_compiler import uses_generic as vector_uses_generic, BASIS_VERSION as VECTOR_BASIS_VERSION
        if vector_uses_generic(self.fields):
            identity['generic_vector_basis'] = VECTOR_BASIS_VERSION
        self.signature = _SCHEMA + ":" + _digest(identity)

    def _resolve_key(self, key):
        if isinstance(key, str):
            if key in self._ids:
                return self._ids[key]
            if key.upper() in self._keys:
                return key.upper()
        raise ModelResolutionError(f"Unknown operator {key!r}; run make inputs (or define_model().show_inputs()).")

    def coefficient_key(self, operator_id):
        """Readable coefficient key for a current catalogue operator ID."""
        return self._keys[self._resolve_key(operator_id)]

    def operator_inputs(self):
        """Return selected coefficient labels, units, multiplicities and hints."""
        rows = []
        for row in self.lagrangian.operators:
            kind, obstruction, labels, hint = _hints(row, self.fields, self.compiler_limits, self.lagrangian.operators)
            dimension = row["coefficient_dimension"]
            rows.append(dict(key=self._keys[row["id"]], operator_id=row["id"],
                             label=row["field_content"], dimension=dimension,
                             units="dimensionless" if dimension == 0 else "GeV" if dimension == 1 else f"GeV^{dimension}",
                             multiplicity=row["multiplicity"],
                             real=row["field_content"] == row["conjugate_content"],
                             add_hc=row["field_content"] != row["conjugate_content"],
                             invariant_labels=labels, kind=kind, obstruction=obstruction, hint=hint))
            if kind in colour_yukawa.KINDS:
                rows[-1]['basis_fingerprint'] = colour_yukawa.basis_fingerprint(row, self.fields)
            elif kind in ('generic_scalar_potential', 'generic_scalar_kinetic'):
                from generic_scalar_compiler import invariant_basis, BASIS_VERSION
                basis = invariant_basis(row, self.fields, limits=self.compiler_limits)
                rows[-1]['basis_fingerprint'] = _digest(dict(version=BASIS_VERSION,
                    invariants=[[(list(m), round(z.real, 10), round(z.imag, 10))
                                 for m,z in sorted(p.items())] for p in basis]))
            elif kind in ('joint_scalar_potential', 'joint_scalar_kinetic'):
                from generic_joint_scalar_compiler import basis_fingerprint
                rows[-1]['basis_fingerprint'] = basis_fingerprint(row, self.fields, limits=self.compiler_limits)
            elif kind == 'colour_scalar_potential':
                from generic_color_compiler import basis_fingerprint
                rows[-1]['basis_fingerprint'] = basis_fingerprint(row, self.fields, limits=self.compiler_limits)
            elif kind == 'generic_scalar_lepton_yukawa':
                from generic_scalar_yukawa import basis_fingerprint
                rows[-1]['basis_fingerprint'] = basis_fingerprint(row, self.fields, limits=self.compiler_limits)
            elif kind and kind.startswith('generic_vector_'):
                from generic_vector_compiler import basis_fingerprint
                rows[-1]['basis_fingerprint'] = basis_fingerprint(row, self.fields, limits=self.compiler_limits)
        return rows

    def required_inputs(self):
        """Editable template. Every operator coefficient is deliberately None.

        All new scalar VEVs are zero. Widths are initial seeds, and masses
        follow the quadratic operators in the fixed electroweak vacuum.
        """
        template = required_inputs(self.lagrangian, compiler_limits=self.compiler_limits)
        template["coefficients"] = {self._keys[key]: value
                                    for key, value in template["coefficients"].items()}
        template["unresolved"] = {self._keys.get(key, key): reason
                                  for key, reason in template["unresolved"].items()}
        if self.result.sm_generations != 3:
            template["unresolved"]["SM"] = "Automatic physical SM export requires exactly three SM generations."
        from vector_current_compiler import supports_fields
        if supports_fields(self.fields):
            for field in self.fields:
                template["unresolved"].pop(field.name, None)
        return dict(model_signature=self.signature, **template, stable_particles=[], compiler_limits=dict(self.compiler_limits))

    def show_inputs(self):
        """Print a copyable physical_inputs dictionary without building a UFO."""
        print("# Coefficients multiply Lagrangian operators, not potential terms.")
        print("# Fill every None; zero deliberately disables a selected interaction.")
        print("# Masses are derived; widths=0 are seeds, not a stability declaration.")
        print("# The electroweak vacuum fixes the Higgs at 125 GeV and every new scalar VEV at zero.")
        for row in self.operator_inputs():
            reality = "real" if row["real"] else "real or complex; h.c. included"
            count = ("; contraction " if row["multiplicity"] == 1 else "; entries ") + ", ".join(row["invariant_labels"])
            print(f"# {row['key']!r}: {row['units']}; {reality}{count}. {row['hint']}")
        if any(field.spin == "vector" for field in self.fields):
            print("# Set vector_specifications for every vector; {'kind': 'proca'} selects a massive effective matter vector.")
        editable = {key: value for key, value in self.required_inputs().items()
                    if key not in {"vevs", "higgs_mass"}}
        print("physical_inputs = " + pformat(editable, sort_dicts=False, width=100))

    def build(self, physical_inputs):
        """Validate saved numbers and return (selected Lagrangian, backend inputs).

        This performs no external calculation. Physical consistency, supported
        contractions, kinetic normalization and vacuum checks follow at export.
        """
        if not isinstance(physical_inputs, Mapping):
            raise ModelResolutionError("physical_inputs must be a dictionary; generate it with make inputs "
                                       "(or define_model().show_inputs()).")
        if physical_inputs.get("model_signature") != self.signature:
            raise ModelResolutionError("physical_inputs has a missing or stale model_signature: the particles, "
                                       "representations or selected operators changed. Run make inputs "
                                       "(or define_model().show_inputs()) for a fresh template and review every coefficient.")
        template = self.required_inputs()
        unknown = set(physical_inputs) - set(template) - {"feynrules", "sm_scheme", "observed_higgs_mass", "vacuum"}
        if unknown:
            raise ModelResolutionError("Unknown physical input keys: " + ", ".join(sorted(map(str, unknown)))
                                       + ". Masses and kinetic normalization belong in operator coefficients.")
        from fixed_sm import normalize_fixed_sm
        inputs = normalize_fixed_sm(physical_inputs, self.fields)
        from compiler_limits import normalize_limits
        supplied_limits = normalize_limits(inputs.get('compiler_limits', self.compiler_limits))
        if self.compiler_limits and supplied_limits != self.compiler_limits:
            raise ModelResolutionError('compiler_limits differ from the budgets used to discover the coefficient basis')
        inputs['compiler_limits'] = supplied_limits
        supplied = inputs.get("coefficients")
        if not isinstance(supplied, Mapping):
            raise ModelResolutionError("physical_inputs['coefficients'] must contain every selected operator key.")
        coefficients, extra, duplicate = {}, [], []
        selected = set(self.lagrangian.selected_ids)
        for key, value in supplied.items():
            try:
                operator_id = self._resolve_key(key)
            except ModelResolutionError:
                extra.append(str(key))
                continue
            if operator_id not in selected:
                extra.append(str(key))
            elif operator_id in coefficients:
                duplicate.append(self._keys[operator_id])
            else:
                coefficients[operator_id] = value
        missing = [self._keys[key] for key in self.lagrangian.selected_ids if key not in coefficients]
        if missing or extra or duplicate:
            raise ModelResolutionError(f"Coefficient keys must match the selected model: missing={missing}, "
                                       f"extra={extra}, duplicate={duplicate}.")
        errors = []
        for row in self.lagrangian.operators:
            key = self._keys[row["id"]]
            value = coefficients[row["id"]]
            multiple = row["multiplicity"] != 1
            entries = value if multiple else [value]
            if not isinstance(entries, (list, tuple)) or len(entries) != row["multiplicity"]:
                errors.append(f"{key!r} needs {row['multiplicity']} coefficients in its displayed invariant order")
                continue
            normalized = []
            for i, entry in enumerate(entries):
                try:
                    normalized.append(_number(entry, f"{key!r}" + (f"[{i}]" if multiple else ""),
                                              real=row["field_content"] == row["conjugate_content"]))
                except ModelResolutionError as exc:
                    errors.append(str(exc))
            if len(normalized) == len(entries):
                coefficients[row["id"]] = normalized if multiple else normalized[0]
        if errors:
            raise ModelResolutionError("Incomplete or invalid coefficients:\n  " + "\n  ".join(errors))
        inputs["coefficients"] = coefficients
        widths = inputs.get("widths", template["widths"])
        if not isinstance(widths, Mapping):
            raise ModelResolutionError("widths must be a dictionary.")
        extras = set(widths) - set(template["widths"])
        if extras:
            raise ModelResolutionError(f"Unknown widths fields: {sorted(map(str, extras))}.")
        inputs["widths"] = {}
        for name, value in widths.items():
            width = _number(value, f"widths.{name}", real=True)
            if width < 0:
                raise ModelResolutionError(f"widths.{name} must be nonnegative.")
            inputs["widths"][name] = width
        name = inputs.get("model_name", template["model_name"])
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ModelResolutionError("model_name must start with a letter and contain only letters, numbers and underscores.")
        inputs["model_name"] = name
        stable = inputs.get("stable_particles", [])
        if not isinstance(stable, (list, tuple)) or any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in stable):
            raise ModelResolutionError("stable_particles must list physical UFO particle names or integer PDG codes.")
        inputs["stable_particles"] = list(stable)
        vectors = {field.name for field in self.fields if field.spin == "vector"}
        if vectors and not inputs.get("feynrules"):
            specifications = inputs.get("vector_specifications")
            if not isinstance(specifications, Mapping) or set(specifications) != vectors:
                raise ModelResolutionError("vector_specifications must describe exactly these vector fields: "
                                           + ", ".join(sorted(vectors)) + ".")
            for key, specification in specifications.items():
                if not isinstance(specification, Mapping) or not specification:
                    raise ModelResolutionError(f"Supply vector_specifications.{key}; for a neutral massive effective "
                                               "vector use {'kind': 'proca'} only when that physical theory is intended.")
        # The manual hook is an advanced option, but its mappings can use the
        # same readable keys as automatic coefficients.
        if inputs.get("feynrules"):
            manual = inputs["feynrules"]
            if not isinstance(manual, Mapping):
                raise ModelResolutionError("feynrules must be an explicit-model dictionary.")
            if "operator_map" in manual:
                converted = {}
                for key, expression in manual["operator_map"].items():
                    operator_id = self._resolve_key(key)
                    if operator_id in converted:
                        raise ModelResolutionError(f"Duplicate feynrules.operator_map entry for {self._keys[operator_id]!r}.")
                    converted[operator_id] = expression
                manual["operator_map"] = converted
        return self.lagrangian, inputs
