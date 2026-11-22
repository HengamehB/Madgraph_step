# Madgraph_step

Choose new particle representations, set their operator coefficients, and rank tree-level proton production and decay histories by cross section times branching probability.

## First use

Use Python 3.11 with NumPy and SciPy (see [requirements.txt](requirements.txt)), a locally activated Mathematica or Wolfram Engine kernel, GNU Fortran, a C++ compiler, make, Bash, and Perl. FeynRules 2.3.49, MadGraph 3.7.3, and the benchmark PDF grid are included. The package runs in your existing environment. Jupyter is optional and unnecessary for these commands.

From the repository directory:

```bash
./run.sh configure-tools
./run.sh check
```

Review the detected paths. Choose `edit` to enter a custom kernel executable, then `y` to save paths in your local `.config`. The kernel check must succeed with your own license. Each computer maintains its own configuration.

The supplied model is already set up for the coloured-scalar benchmark. To try it, run:

```bash
./run.sh validate input/my_model.txt
./run.sh channels input/my_model.txt
```

For a new model, edit `input/particles.txt`, then generate an editable model. This replaces the destination and discards earlier coefficient edits. Use `--model` for a different name:

```bash
./run.sh init input/particles.txt --model "input/my scan.txt"
# Edit operator switches, coefficients, stability and calculation settings.
./run.sh validate "input/my scan.txt"
./run.sh channels "input/my scan.txt"
```

Keep the generated model with its hidden `input/.model_metadata/` companion. To calculate one production channel, use its ID from `channels`; for example, `./run.sh run input/my_model.txt --selected C0004` for the supplied benchmark. Omitting `--selected` integrates all enumerated production channels. Progress and intermediate files are saved. Use the same command to resume.

## Results

Open `output/<model-name>/<run-id>/report.html` or `ranking.csv`. The CSV gives each production channel and complete decay history, its production cross section, branching probability, and their product in pb. Rows without an integrated production rate remain uncalculated. The manifest distinguishes completion of a selected run from completion of all channels.

The supplied benchmark uses 13.6 TeV, two-body decays, dynamical scale choice 3, and production of at most two new particles. The production multiplicity does not bound the cascade length. Rates are tree-level narrow-width estimates. Higher-body decay contributions may be missing and distinct overlapping histories must not be added as exclusive rates. [Step-by-step guide (PDF)](user_guide/Madgraph_step_User_Guide.pdf) · [Editable guide](user_guide/Madgraph_step_User_Guide.tex)

## Testing new machines

GitHub Actions runs the regression suite and validates newly generated models on Ubuntu, macOS, and Windows Server using WSL2. The **Licensed platform benchmarks** workflow is launched manually. It runs the complete input-to-CSV benchmark once per platform on Ubuntu, macOS and Windows Server using WSL2. A WSL2 result covers the Linux kernel and Linux tools inside Windows.

Only that hosted workflow uses the repository's `WOLFRAMSCRIPT_ENTITLEMENTID` Actions secret for Wolfram licensing. Ordinary users run with their local license. Each licensed run generates its own UFO, calculates two-body widths, integrates one selected channel, verifies the ranked CSV and checks resume. See [Actions runs](../../actions) for platform results and downloadable summaries.

Pushing a branch under `single-runner/` runs the portable checks on Ubuntu alone. The Ubuntu job compares the original and patched FeynRules contractions and mixing lookup using Mathics in a separate test environment. This check uses no Wolfram entitlement.

The repository owner can manually dispatch **Licensed platform benchmarks** on `single-runner/wolfram15-ufo` to trace the original export and test the repair on one Ubuntu runner. Both exports use the same coloured-scalar coefficients. The job saves the first unclassified vertex's incoming and outgoing fields in `field-trace.json`. The repaired model also goes through MadGraph import, two-body widths and one production integration. This manual job uses Wolfram Service Credits.
