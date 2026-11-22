"""Automatic component and mass-basis compiler for neutral singlet fermions.

All fermion inputs are first represented as left-handed Weyl fields. A Dirac
input contributes its two independent left-handed fields. General complex mass
matrices are resolved into Majorana mass eigenstates, including the three SM
neutrinos when neutrino portal terms are selected. No tensor strings are needed.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import math
import re

import numpy as np
from scipy.linalg import null_space


def supports_fields(fields):
    return (any(f.spin == "fermion" for f in fields)
            and all(f.su2 == 1 and f.su3 == (0, 0) and f.hypercharge == 0
                    and (f.spin == "fermion" or f.spin == "scalar" and f.kind == "real")
                    for f in fields))


def tokens_for(fields):
    tokens = []
    for f in fields:
        if f.spin != "fermion":
            continue
        tokens.extend([f.name + "_L", f.name + "_Rc"] if f.kind == "dirac" else [f.name])
    return tokens


def classify(row, fields):
    tokens = set(tokens_for(fields))
    scalars = {f.name for f in fields if f.spin == "scalar"}
    counts = Counter({f["name"]: f["count"] for f in row["factors"]})
    expanded = list(counts.elements())
    if row["derivatives"]:
        differentiated = [n for n in expanded if n.startswith("D(")]
        other = [n for n in expanded if not n.startswith("D(")]
        if (len(differentiated) == len(other) == 1
                and differentiated[0][2:-1] in tokens
                and other[0].endswith("†") and other[0][:-1] in tokens
                and row["multiplicity"] == 1):
            return "weyl_kinetic", ""
        return None, "unresolved derivative contraction"
    if (counts["H"] == 1 and counts["L"] == 1 and len(expanded) == 3
            and sum(counts[t] for t in tokens) == 1 and row["multiplicity"] == 3):
        return "neutrino_portal", ""
    fermions = [n for n in expanded if n in tokens]
    others = [n for n in expanded if n not in tokens]
    if (len(fermions) == 2 and len(others) <= 1 and set(others) <= scalars
            and row["multiplicity"] == 1):
        return "weyl_mass" if not others else "weyl_yukawa", ""
    return None, "explicit fermion contraction is not resolved by the neutral-singlet compiler"


def takagi(matrix):
    """Return U,m with U.T @ M @ U=diag(m), including exact zero modes.

    The real symmetric doubled problem avoids SVD phase ambiguities at a
    degenerate Dirac pair. Both congruence and unitarity are checked afterwards.
    """
    matrix = np.asarray(matrix, dtype=complex)
    n = len(matrix)
    scale = float(np.max(abs(matrix), initial=0.))
    if not scale:
        return np.eye(n, dtype=complex), np.zeros(n)
    matrix = matrix / scale
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Takagi decomposition requires a symmetric mass matrix.")
    norm = np.linalg.norm(matrix, 2)
    tol = np.finfo(float).eps * max(n, 1) * norm * 8
    doubled = np.block([[matrix.real, -matrix.imag], [-matrix.imag, -matrix.real]])
    values, vectors = np.linalg.eigh(doubled)
    keep = values > tol
    positive = values[keep]
    u = vectors[:n, keep] + 1j * vectors[n:, keep]
    zeros = null_space(matrix, rcond=tol / norm)
    if len(positive) + zeros.shape[1] != n:
        raise ValueError("Mass spectrum is too ill-conditioned for stable Takagi rank determination.")
    # The doubled eigensolver controls absolute residuals. For a light state
    # next to exact zero modes, its eigenvector may consequently contain a
    # larger relative null-space component. Project that component away and
    # restore complex orthonormality; the congruence check below still rejects
    # any correction that would change a resolved physical mass matrix.
    if len(positive):
        if zeros.shape[1]:
            u -= zeros @ (zeros.conj().T @ u)
        gram_values, gram_vectors = np.linalg.eigh(u.conj().T @ u)
        if gram_values.min() <= 0:
            raise ValueError("Takagi eigenspace is numerically rank deficient.")
        u = u @ ((gram_vectors / np.sqrt(gram_values)) @ gram_vectors.conj().T)
    u = np.column_stack((zeros, u))
    masses = np.concatenate((np.zeros(zeros.shape[1]), positive))
    if not np.allclose(u.conj().T @ u, np.eye(n), atol=2e-9, rtol=2e-9):
        raise ValueError("Takagi unitarity residual exceeds tolerance.")
    if not np.allclose(u.T @ matrix @ u, np.diag(masses), atol=tol * 20, rtol=2e-9):
        raise ValueError("Takagi mass residual exceeds tolerance.")
    return u, masses * scale


def _basis(fields, operators, v, scalar_vevs=None):
    from feynrules_export import ModelResolutionError, CompilerResolutionError

    tokens = ["nu_e", "nu_mu", "nu_tau"] + tokens_for(fields)
    positions = {name: i for i, name in enumerate(tokens)}
    n = len(tokens)
    k = np.zeros((n, n), complex)
    k[:3, :3] = np.eye(3)
    m = np.zeros_like(k)
    for op in operators:
        counts = Counter(dict(op.factors))
        factors = list(counts.elements())
        if op.kind == "weyl_kinetic":
            bare = next(t[2:-1] for t in factors if t.startswith("D("))
            dagger = next(t[:-1] for t in factors if t.endswith("†"))
            i, j = positions[bare], positions[dagger]
            c = op.coefficients[0]
            k[j, i] += c
            k[i, j] += np.conjugate(c)
        elif op.kind in ("weyl_mass", "weyl_yukawa"):
            i, j = [positions[t] for t in factors if t in positions]
            value = op.coefficients[0]
            if op.kind == "weyl_yukawa":
                scalar = next(t for t in factors if t not in positions)
                value *= (scalar_vevs or {}).get(scalar, 0.0)
            m[i, j] -= value
            m[j, i] -= value
        elif op.kind == "neutrino_portal":
            j = positions[next(t for t in factors if t in positions)]
            for i, c in enumerate(op.coefficients):
                m[i, j] -= c * v / math.sqrt(2)
                m[j, i] -= c * v / math.sqrt(2)
    diagonal = k.diagonal().real
    if np.any(diagonal <= 0):
        raise ModelResolutionError("Fermion kinetic form is not positive definite; each canonical Weyl kinetic row plus h.c. has coefficient 1/2.")
    field_scale = np.sqrt(diagonal)
    normalized_k = (k / field_scale[:, None]) / field_scale[None, :]
    normalized_m = (m / field_scale[:, None]) / field_scale[None, :]
    eigen, vectors = np.linalg.eigh(normalized_k)
    if eigen.min() <= 0:
        raise ModelResolutionError("Fermion kinetic form is not positive definite; each canonical Weyl kinetic row plus h.c. has coefficient 1/2.")
    inverse = (vectors / np.sqrt(eigen)) @ vectors.conj().T
    try:
        rotation, masses = takagi(inverse.T @ normalized_m @ inverse)
    except ValueError as exc:
        raise CompilerResolutionError(str(exc)) from exc
    canonical_transform = inverse @ rotation
    transform = canonical_transform / field_scale[:, None]
    normalized = canonical_transform / np.linalg.norm(canonical_transform, axis=0)
    order = max(permutations(range(n)), key=lambda p: sum(abs(normalized[i, p[i]]) for i in range(n)))
    transform = transform[:, order]
    masses = masses[list(order)]
    for i in range(n):
        value = transform[i, i]
        if value.real < 0 or value.real == 0 and value.imag < 0:
            transform[:, i] *= -1
    if not np.allclose(transform.conj().T @ k @ transform, np.eye(n), atol=2e-9, rtol=2e-9):
        raise CompilerResolutionError("Neutral fermion kinetic normalization failed its residual check.")
    return dict(tokens=tokens, kinetic=k, mass_matrix=m, transformation=transform, masses=masses)


@dataclass
class NeutralFermionCompiler:
    scalar_basis: dict
    fermion_basis: dict
    scalar_fields: tuple
    scalar_operators: tuple
    fermion_operators: tuple
    all_fields: tuple

    @property
    def masses(self):
        names = self.scalar_basis["names"]
        answer = dict(zip(names, self.scalar_basis["masses"].tolist()))
        for token, mass in zip(self.fermion_basis["tokens"], self.fermion_basis["masses"]):
            answer[token] = float(mass)
        return answer

    def write_sources(self, model, source, fr):
        from feynrules_export import PreparedModel, CompilerResolutionError, _internal, _wl

        scalar_model = PreparedModel(model.lagrangian, model.inputs, self.scalar_operators,
                                     self.scalar_fields, self.scalar_basis, model.runtime)
        files, _, scalar_particles = scalar_model._automatic_sources(source, fr)
        smfile, extension = files
        sm = smfile.read_text()
        b = self.fermion_basis
        n = len(b["tokens"])
        symbols = ["ve", "vm", "vt"] + [f"pmN{i}" for i in range(4, n + 1)]
        by_token = {}
        names, sources, widths = ["ve", "vm", "vt"], ["nu_e", "nu_mu", "nu_tau"], [0, 0, 0]
        for f in self.all_fields:
            if f.spin != "fermion":
                continue
            ts = [f.name + "_L", f.name + "_Rc"] if f.kind == "dirac" else [f.name]
            for index, token in enumerate(ts, 1):
                by_token[token] = f.name
                names.append(f.name.lower() + (str(index) if len(ts) == 2 else ""))
                sources.append(f.name)
                widths.append(model.inputs.get("widths", {}).get(f.name, 0.0))
        pdgs = [12, 14, 16] + [9100000 + i for i in range(1, n - 2)]
        masses = [f"{{pmNM{i+1},{_wl(mass)}}}" for i, mass in enumerate(b["masses"])]
        width_entries = ["0", "0", "0"] + [f"{{pmNW{i+1},{_wl(widths[i])}}}" for i in range(3, n)]
        declaration = f'''F[1] == {{ClassName -> vl, ClassMembers -> {{{','.join(symbols)}}},
          Indices -> {{Index[PMNeutrino]}}, FlavorIndex -> PMNeutrino,
          SelfConjugate -> True, Mass -> {{pmMnu,{','.join(masses)}}},
          Width -> {{{','.join(width_entries)}}},
          ParticleName -> {{{','.join(_wl(x) for x in names)}}},
          PDG -> {{{','.join(map(str,pdgs))}}},
          PropagatorLabel -> {{{','.join(_wl(x) for x in ['N']+names)}}},
          PropagatorType -> Straight, PropagatorArrow -> None,
          FullName -> {{{','.join(_wl(x) for x in sources)}}}
        }},\n  '''
        sm, count = re.subn(r"F\[1\]\s*==\s*\{.*?(?=F\[2\]\s*==)", declaration,
                            sm, count=1, flags=re.S)
        if count != 1:
            raise CompilerResolutionError("Cannot identify SM neutrino class for automatic neutral-fermion mixing.")
        old = "ProjM[sp1,sp2] vl[sp2,ff]"
        new = ("Sum[pmNU[ff,nn] ProjM[sp1,sp2] vl[sp2,nn], {nn,1," + str(n) + "}]")
        if old not in sm:
            raise CompilerResolutionError("SM lepton-doublet definition changed; neutrino mixing needs review.")
        sm = sm.replace(old, new)
        # A general Majorana mass matrix breaks lepton number. Retaining the
        # stock conserved-QN tag on charged leptons would cause spurious export
        # warnings for every legitimate charged-current vertex.
        sm = sm.replace(", LeptonNumber -> 1", "")
        # Declare the enlarged index before LoadModel reads particle classes.
        sm = f"IndexRange[Index[PMNeutrino]] = Range[{n}];\nIndexStyle[PMNeutrino,n];\n" + sm
        smfile.write_text(sm)
        parameters = []
        definitions, values = [], []
        for i in range(3):
            for j in range(n):
                value = b['transformation'][i,j]
                (definitions if value == 0 else values).append(f"pmNU[{i+1},{j+1}] -> {_wl(value)}")
        parameters.append("pmNU == {ParameterType -> Internal, ComplexParameter -> True, "
                          "Indices -> {Index[Generation],Index[PMNeutrino]}, Definitions -> {"
                          + ",".join(definitions) + '}, Value -> {' + ",".join(values)
                          + '}, Description -> "Active neutrino mixing"}')
        transform = b["transformation"]
        coefficient_symbols = {}
        def mixing_coefficient(value):
            key = complex(value)
            if key not in coefficient_symbols:
                symbol = 'pmMix' + str(len(coefficient_symbols) + 1)
                coefficient_symbols[key] = symbol
                parameters.append(_internal(symbol, value))
            return coefficient_symbols[key]
        def bilinear(i, j):
            terms = []
            for a in range(n):
                for c in range(n):
                    value = transform[i, a] * transform[j, c]
                    if value != 0:
                        terms.append(f"{mixing_coefficient(value)}*{symbols[a]}bar.ProjM.{symbols[c]}")
            return "(" + " + ".join(terms) + ")"
        def scalar_expression(name):
            i = self.scalar_basis["names"].index(name)
            scalar_symbols = ["H"] + [f"pmS{k}" for k in range(1, len(self.scalar_fields) + 1)]
            terms = [f"pmR{i}{j}*{scalar_symbols[j]}" for j in range(len(scalar_symbols))
                     if self.scalar_basis["transformation"][i,j] != 0]
            return "(" + " + ".join(terms) + ")"
        terms = []
        position = {token: i for i, token in enumerate(b["tokens"])}
        for op in self.fermion_operators:
            factors = list(Counter(dict(op.factors)).elements())
            if op.kind == "neutrino_portal":
                j = position[next(t for t in factors if t in position)]
                expressions = [f"(vev + {scalar_expression('H')})/Sqrt[2]*{bilinear(i,j)}" for i in range(3)]
            elif op.kind in ("weyl_mass", "weyl_yukawa"):
                i, j = [position[t] for t in factors if t in position]
                expressions = [bilinear(i,j)]
                if op.kind == "weyl_yukawa":
                    scalar = next(t for t in factors if t not in position)
                    expressions = [scalar_expression(scalar) + "*" + expressions[0]]
            else:
                # Integration by parts puts all derivative rows in the
                # Hermitian K form used above. T† K T = 1 was checked before
                # reaching this step, so write the canonical result directly.
                # FR's diagonal checker does not handle left-derivative IBP
                # when complex Majorana phases are left in the expression.
                symbol = f"pmF{op.operator_id[1:]}n1"
                parameters.append(_internal(symbol, op.coefficients[0], order=True))
                terms.append(f"0 (* {op.operator_id} is included in the normalized kinetic matrix; coefficient {symbol} *)")
                continue
            for index, (value, expression) in enumerate(zip(op.coefficients, expressions), 1):
                symbol = f"pmF{op.operator_id[1:]}n{index}"
                higgs = sum(count for name, count in op.factors if name in ("H", "H†"))
                parameters.append(_internal(symbol, value, order=True, qed_order=higgs))
                if value == 0:
                    terms.append(f"0 (* resolved zero {op.operator_id}, component {index} *)")
                else:
                    term = f"{symbol}*({expression})"
                    terms.append(f"({term} + HC[{term}])")
        # A second model file merges its parameter declarations with SM/scalars.
        fermionfile = source / "neutral_fermions.fr"
        species = "|".join(name for s in symbols for name in (s,s+"bar"))
        canonical = " + ".join(f"I/2*{s}bar.Ga[mu].del[{s},mu]" for s in symbols)
        fermionfile.write_text('M$ModelName = ' + _wl(model.model_name) + ';\nM$Parameters = {\n'
                               + ',\n'.join(parameters) + '\n};\nLNeutral := '
                               + (' + '.join(terms) or '0') + ';\n'
                               'LActiveKinetic := Plus @@ Select[List @@ Expand[GetKineticTerms[LSM, FlavorExpand -> True]], '
                               f'!FreeQ[#, {species}] &];\n'
                               f'LCanonicalNeutral := Block[{{mu}}, {canonical}];\n')
        particles = list(scalar_particles)
        particles.extend(dict(name=names[i], antiname=names[i], pdg=pdgs[i], source=sources[i],
                              spin=2, mass=float(b["masses"][i]), width=widths[i]) for i in range(3,n))
        return files + [fermionfile], "LSM + LNew + LNeutral - LActiveKinetic + LCanonicalNeutral", particles


def prepare(lagrangian, inputs, runtime=None):
    from feynrules_export import (PreparedModel, ResolvedOperator, ModelResolutionError, CompilerResolutionError,
                                 _fields, _coefficients, _classify, _scalar_basis)

    fields = _fields(lagrangian)
    if not supports_fields(fields):
        return None
    if getattr(lagrangian._source, "sm_generations", 3) != 3:
        raise CompilerResolutionError("Automatic physical SM export requires three SM generations.")
    coefficients = _coefficients(lagrangian, inputs)
    scalar_fields = tuple(f for f in fields if f.spin == "scalar")
    scalar_ops, fermion_ops, unresolved = [], [], {}
    for row in lagrangian.operators:
        kind, reason = _classify(row, scalar_fields)
        destination = scalar_ops
        if reason:
            kind, reason = classify(row, fields)
            destination = fermion_ops
        if reason:
            unresolved[row["id"]] = reason
            continue
        destination.append(ResolvedOperator(row["id"], kind,
                           tuple((f["name"],f["count"]) for f in row["factors"]), coefficients[row["id"]]))
    if unresolved:
        raise CompilerResolutionError("Unresolved selected operators:\n" + "\n".join(f"{k}: {v}" for k,v in unresolved.items()))
    scalar_basis = _scalar_basis(scalar_fields, scalar_ops, inputs)
    fermion_basis = _basis(fields, fermion_ops, scalar_basis["vev"], scalar_basis["vevs"])
    compiler = NeutralFermionCompiler(scalar_basis, fermion_basis, scalar_fields, tuple(scalar_ops), tuple(fermion_ops), fields)
    inputs.setdefault("widths", {})
    for f in fields:
        inputs["widths"].setdefault(f.name, 0.0)
    return PreparedModel(lagrangian, inputs, tuple(scalar_ops + fermion_ops), fields,
                         scalar_basis, runtime, compiler=compiler)
