# Madgraph_step

Describe new particles and their operator coefficients in a text file. Madgraph_step constructs the Lagrangian, derives masses and mixing, exports a UFO with FeynRules, and uses MadGraph to rank production and decay histories by cross section × branching probability. Start with the included coloured-scalar model to check your installation.

## 1. Prepare your platform

You need **Python 3.11**, a locally licensed **Mathematica or Wolfram Engine** kernel, GNU Fortran, a C++ compiler, Make, Bash and Perl. The repository includes the patched FeynRules 2.3.49, MadGraph5_aMC@NLO 3.7.3, and the nn23lo1 parton distribution grid used by the example. The terminal workflow requires NumPy and SciPy. Wolfram Service Credits are used only for the project's licensed GitHub tests. Your runs use your own kernel and license. [Bundled software and provenance](vendor/PROVENANCE.txt)

**macOS** — Install the Xcode Command Line Tools if needed. With Homebrew installed, install GNU Fortran through its GCC package. Install and activate Mathematica or Wolfram Engine for macOS. [Homebrew](https://brew.sh/) · [Wolfram Engine](https://www.wolfram.com/engine/)

```bash
xcode-select --install
brew install gcc
```

**Ubuntu Linux** — Install the build tools, then install and activate Mathematica or Wolfram Engine for Linux. The hosted Linux benchmark used Ubuntu 24.04. [Wolfram Engine](https://www.wolfram.com/engine/)

```bash
sudo apt update
sudo apt install git build-essential gfortran perl
```

**Windows with WSL2** — In an administrator PowerShell, run the commands below and restart if prompted. Confirm that Ubuntu shows version `2` in the list. Then open the **Ubuntu terminal**, run the Ubuntu installation steps above, and perform the rest of this guide there. Clone into your Linux home directory. Use a Linux Python and a licensed Linux Wolfram kernel inside WSL2. This is the Windows configuration tested on GitHub. [Microsoft WSL installation guide](https://learn.microsoft.com/windows/wsl/install)

```powershell
wsl --install -d Ubuntu
wsl --list --verbose
```

## 2. Download the release and set up Python

From your macOS or Linux terminal, or your Ubuntu terminal on Windows, clone the tagged version. Access to a private repository requires a GitHub account with permission. You can also sign in on GitHub, download the release's **Source code (zip)**, extract it, and open a terminal in the extracted folder. [Release v0.1.0](https://github.com/HengamehB/Madgraph_step/releases/tag/v0.1.0)

```bash
git clone --branch v0.1.0 https://github.com/HengamehB/Madgraph_step.git
cd Madgraph_step
```

If you already have Python 3.11, create an isolated environment in the package directory.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If you use Conda, you can create the environment with these two commands instead. This also provides Python 3.11 on systems with a different default Python. If you need Conda, install Miniforge for the operating system and architecture in which you will run the package. [Miniforge installation](https://github.com/conda-forge/miniforge#install)

```bash
conda create -n madgraph-step python=3.11 pip
conda activate madgraph-step
```

Install the Python dependencies into the activated environment, configure tool paths, and check the installation.

```bash
python --version
python -m pip install -r requirements.txt
./run.sh configure-tools
./run.sh check
```

The first line should show `Python 3.11.x`. At `Save these paths? [y/N/edit]`, enter `y` if the detected paths are correct. Enter `edit` to supply a full executable path. Keep the bundled FeynRules selected to use the Wolfram 15 repair. The `check` command starts the actual kernel to verify its license and reports each tool as `OK` or `MISSING`. Reactivate the same Python environment in future terminal sessions.

If the kernel is missing, point `mathematica` under `[tools]` in the package's `.config` to the full executable. On macOS one common path is `/Applications/Mathematica.app/Contents/MacOS/WolframKernel`. On Linux or WSL2, use the path from your Linux installation. Enter paths without surrounding quotes in this file. Run `./run.sh check` again after correcting a path or activating the license.

## 3. Run the supplied example

**Run the supplied model directly.** The release includes its companion metadata. Leave the hidden `input/.model_metadata/` directory in place. The model describes a 1,500 GeV colour sextet and a 1,200 GeV colour triplet. The example uses 13.6 TeV proton collisions, the `nn23lo1` PDF, dynamical scale choice 3, and two-body decays. This command calculates widths, integrates the single `p p > x1` production channel, and saves the ranked histories. The first run also builds the UFO and numerical code. [Example model](input/my_model.txt)

```bash
./run.sh rates input/my_model.txt --selected C0004 --page-size 5
```

For this model, `C0004` is `p p > x1`. In the reference run, the corresponding CSV entry for `x1 → u u` had the following values. The integration errors are Monte Carlo uncertainties.

| Quantity | Reference result |
| --- | ---: |
| Production cross section | 1.1691955 ± 0.0045663 pb |
| Branching fraction for `x1 → u u` | 0.3392486 |
| Ranked rate for `p p → x1 → u u` | 0.39664794 ± 0.00154912 pb |

The Ubuntu, macOS and Windows WSL2 benchmark runs each exported 191 vertices and 136 couplings and matched four total and ten partial two-body widths within 0.1% of their reference values. Their ranked tables had 30 calculated histories and 1,895 histories with blank rates because only one production channel was integrated. [Ubuntu results](https://github.com/HengamehB/Madgraph_step/actions/runs/45078417909) · [macOS and Windows WSL2 results and CSV downloads](https://github.com/HengamehB/Madgraph_step/actions/runs/45081434860)

## 4. Find your results

Open the directory printed after **Run files**. With `output = '../output'` in a model under `input/`, each run is saved in `output/<model_name>/<run-id>/`.

| File | What it contains |
| --- | --- |
| `report.html` | Run summary, physics notices, and links to the tables |
| `ranking.csv` | All saved histories, cross sections, cascade branching probabilities, rates in pb, uncertainties, and flags |
| `production_channels.csv`, `cross_sections.csv` | Discovered channel IDs and numerical results for integrated channels |
| `total_widths.csv`, `decays.csv` | Calculated widths and branching fractions |
| `manifest.json` | `requested_complete` for the requested selection and `all_rates_complete` for all discovered production channels |

`--page-size 5` controls terminal display. Every history is still saved. Press **Ctrl+C** to stop and repeat the same command to resume verified work. A different page size reuses the integration. A blank rate means the channel still needs a numerical calculation.

## 5. Build your own model

Edit `input/particles.txt`, then give your model its own filename. The following command generates the catalogue and coefficient template. It overwrites its chosen destination on every invocation, without a backup.

```bash
./run.sh init input/particles.txt --model "input/my scan.txt"
```

Edit the switches, coefficients, stability declarations and collider settings in `input/my scan.txt`. Keep its generated `_metadata` reference and companion `.model_metadata/` directory together when moving the model. Then run these steps from the package directory.

```bash
./run.sh inputs "input/my scan.txt"
./run.sh validate "input/my scan.txt"
./run.sh channels "input/my scan.txt"
```

`inputs` explains coefficient ordering and normalization. `validate` derives the spectrum under the fixed SM vacuum assumption of a 125 GeV Higgs and zero new-scalar VEVs. `channels` lists the channel IDs for this particular model. Pass an ID from that list to `rates --selected`, as in the example above. For histories before production integration use `./run.sh histories "input/my scan.txt" --page-size 5`. To integrate **all** discovered channels and run any configured explicit decay chains, use `./run.sh run "input/my scan.txt"`.

The model's `output` path is relative to the model file. To override it for a command, add `--output "scan results"` from the directory where you want the new results folder. The [terminal user guide](user_guide/Madgraph_step_User_Guide.pdf) has further examples.

### Interpreting a ranking

Each row is a tree-level, narrow-width estimate for one production channel and complete decay history. Production multiplicity limits apply before the cascade decays. The supplied example uses two-body widths. The general adaptive decay defaults extend through four bodies with a 1% target. A full stored table can still omit physical decay contributions or loop-induced production. Inspect the omitted-probability and narrow-width flags, and keep overlapping histories separate. Dynamical scales can depend on the external particles, so a production-only and an explicit decayed process can use different numerical scales even with the same scale-choice number.
