"""Run the affected vendored Wolfram rules on Mathics, with a failing control.

This isolated CI probe neither loads a complete model nor writes a UFO.
The calling Python environment must contain Mathics3. Run in a git checkout.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path("vendor/FeynRules")
BASELINE = "a28f8d4680ebbc67374c7d20d2ebb1d1e881592a"


def section(source, start, end):
    first = source.index(start)
    return source[first : source.index(end, first + len(start))]


def run_probe(revision):
    def read(name):
        path = VENDOR / name
        if revision is None:
            return (ROOT / path).read_text(encoding="utf-8")
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path.as_posix()}"], cwd=ROOT, encoding="utf-8"
        )

    contraction = read("Core/ExtractVertexTools.m")
    mixing = read("Core/MassDiagonalization.m")
    classifier = read("Interfaces/UFO/PYIntParticles.m")
    declaration, = re.findall(
        r"^(?:FeynRules`)?MatrixSymbol::usage[^\n]+;$",
        read("FeynRulesPackage.m"),
        flags=re.MULTILINE,
    )
    rules = [
        section(contraction, "OperatorChain[x___, a_, y___]", "ToOperatorChain /:"),
        section(contraction, "(*Commutation relations*)", "(*Creation operator list (CreaList)*)"),
        section(mixing, "(*Getting back a matrix symbol*)", "(*Getting back a mass matrix*)"),
        section(classifier, "ParticleToSpin[particle_] :=", "];") + "];",
    ]
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        (directory / "usage.m").write_text(declaration + "\n", encoding="utf-8")
        (directory / "rules.m").write_text("\n\n".join(rules), encoding="utf-8")
        (directory / "model.m").write_text(MODEL, encoding="utf-8")
        (directory / "checks.m").write_text(CHECKS, encoding="utf-8")
        (directory / "driver.m").write_text(DRIVER, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "mathics", "--quiet", "--file", "driver.m"],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            timeout=90,
        )
    print(f"\n--- {revision or 'candidate working tree'} ---", flush=True)
    print(result.stdout, flush=True)
    result.check_returncode()
    observed = re.findall(r"^FRPROBE (\w+) (True|False)$", result.stdout, re.MULTILINE)
    expected = {
        "systemSymbolsProtectedAndUnchanged": True,
        "ufoClassifier": True,
        "sameScalarContraction": revision is None,
        "differentScalarContraction": revision is None,
        "publicMixingAccessor": revision is None,
    }
    actual = {name: truth == "True" for name, truth in observed}
    if len(observed) != len(expected) or actual != expected:
        raise SystemExit(f"Unexpected results. Expected {expected}, found {actual}.")


# Protect the two pre-existing system names even on an evaluator which has
# never implemented them. Save and compare their definitions independently.
# Get reads separate files so their names resolve only after BeginPackage/Begin.
DRIVER = '''
Protect[System`Commutator, System`MatrixSymbol];
Global`before = {DownValues[System`Commutator], DownValues[System`MatrixSymbol]};
BeginPackage["FeynRules`"];
Get[FileNameJoin[{Directory[], "usage.m"}]];
Begin["PRIVATE`"];
Get[FileNameJoin[{Directory[], "model.m"}]];
Get[FileNameJoin[{Directory[], "rules.m"}]];
Get[FileNameJoin[{Directory[], "checks.m"}]];
End[];
EndPackage[];
'''

# The fixture supplies a scalar species delta and the model registry predicates.
# OperatorChain, its contraction and wavefunction rules, the public mixing
# accessor and the UFO classifier are taken verbatim from each revision.
MODEL = '''
numQ[p_] := NumberQ[p];
ScalarFieldQ[p_] := MemberQ[{probeH, probeY}, p];
DiracFieldQ[p_] := p === probeQ;
VectorFieldQ[p_] := p === probeG;
FieldQ[p_] := ScalarFieldQ[p] || DiracFieldQ[p] || VectorFieldQ[p];
KronDelta[a_, b_, {}] := If[a === b, 1, 0];
MakeIdenticalFermions[p_] := p;
FR$MixingRules[Mix[probeMix]] = {MixingMatrix -> {probeEven, probeOdd}};
'''

CHECKS = '''
assert[name_, value_] := Print["FRPROBE ", name, " ", If[TrueQ[value], "True", "False"]];
assert["systemSymbolsProtectedAndUnchanged",
  MemberQ[Attributes[System`Commutator], Protected] && MemberQ[Attributes[System`MatrixSymbol], Protected] &&
  Global`before === {DownValues[System`Commutator], DownValues[System`MatrixSymbol]}];
assert["ufoClassifier", (ParticleToSpin /@ {probeH, probeY, probeQ, probeG, unknownField}) === {1, 1, 2, 3, 0}];
same = OperatorChain[probeH, crea[probeH, {}, 1]];
different = OperatorChain[probeH, crea[probeY, {}, 1]];
Print["Scalar contractions (same and different species): ", InputForm[{same, different}]];
assert["sameScalarContraction", same === uwave[probeH, 1]];
assert["differentScalarContraction", different === 0];
assert["publicMixingAccessor", {FeynRules`MatrixSymbol[probeMix, "S"], FeynRules`MatrixSymbol[probeMix, "PS"]} === {probeEven, probeOdd}];
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    run_probe(args.baseline)
    run_probe(None)
