# Madgraph_step

Madgraph_step takes particle representations and operator coefficients, builds the Lagrangian and UFO, then uses MadGraph to rank production and decay histories by **cross section × branching fraction**. The supplied example pairs a 1,500 GeV colour sextet with a 1,200 GeV colour triplet.

## 1. Set up

Clone the repository, or extract the [source ZIP](https://github.com/HengamehB/Madgraph_step/archive/refs/heads/main.zip) and open a terminal in the extracted folder.

```bash
git clone https://github.com/HengamehB/Madgraph_step.git
cd Madgraph_step
```

Activate a Python 3.11 environment. If you need a new one, Conda can provide Python and its dependencies in one step:

```bash
conda create -n madgraph-step -c conda-forge python=3.11 numpy scipy
conda activate madgraph-step
```

In an existing Python 3.11 environment, `python -m pip install -r requirements.txt` installs any missing Python dependencies. FeynRules, MadGraph5_aMC@NLO and the example PDF grid are bundled.

**Already have a working Mathematica or Wolfram Engine installation?** Use its licensed kernel and go straight to the check below. On Windows, run the package in Ubuntu under WSL2 with a Linux Wolfram kernel available there.

```bash
./run.sh check
```

If the check passes, continue to step 2. If a tool is missing, install only what the check identifies:

| Platform | Tools for compiling MadGraph processes |
| --- | --- |
| macOS | Xcode Command Line Tools (`xcode-select --install`) supply C++ and Make. You also need GNU Fortran (`gfortran`). |
| Ubuntu | `sudo apt install build-essential gfortran perl` supplies the build tools. |
| Windows | If needed, run `wsl --install -d Ubuntu` in administrator PowerShell. Open Ubuntu and follow the Linux steps there. [WSL installation guide](https://learn.microsoft.com/windows/wsl/install) |

If you need a kernel, install and activate Mathematica or [Wolfram Engine](https://www.wolfram.com/engine/) for the operating system where you run the package. If a tool is installed but detection missed it, run `./run.sh configure-tools`, enter `edit` to supply its path or `y` to save the detected paths, then repeat `./run.sh check`.

## 2. Run the sextet example

The particle file declares the representations. The prepared model already contains the operator coefficients, two-body decay settings and 13.6 TeV proton collider settings. Run the model directly; `init` is for generating a fresh coefficient template. [Particle definitions](input/sample_particle_sextet.txt) · [Prepared model](input/sample_model_sextet.txt)

```bash
./run.sh rates input/sample_model_sextet.txt --selected C0004 --page-size 5
```

This computes the widths, integrates `p p > x1` (channel `C0004` for this model), and saves the ranked table. In a fresh run, **30 of 1,925 histories** have numerical rates. The other histories come from the 13 production channels outside this selection. The page size limits the terminal preview. After Ctrl+C, repeat the command to resume verified work.

## 3. Check your result

The **Run files** line prints the directory for this calculation, under `output/triplet_sextet_particles/<run-id>/`. Open `report.html` in a browser or `ranking.csv` for every history. For the direct decay `x1 → u u`, the reference values are:

| Production cross section | Branching fraction | Cross section × BR |
| ---: | ---: | ---: |
| 1.1691955 ± 0.0045663 pb | 0.3392486 | 0.39664794 ± 0.00154912 pb |

From the package directory, substitute the run directory printed on your machine:

```bash
python sample_output/compare.py "output/triplet_sextet_particles/<run-id>"
```

The comparison checks the discovered channels, decay histories, four total widths and ten partial widths. It allows for Monte Carlo integration uncertainty. The reference HTML and CSVs are in [sample_output/](sample_output/).

## 4. Use your own filenames

To start from new particle definitions, copy the particle file, edit its `X` and `Y`, and generate a model. A name such as `scan_particle_sextet.txt` generates `scan_model_sextet.txt` beside it. Use `--model` to choose any other destination filename. **Each `init` replaces its destination**, so give your own model a separate name.

```bash
cp input/sample_particle_sextet.txt "input/my particles.txt"
# Edit "input/my particles.txt", then generate its coefficient table.
./run.sh init "input/my particles.txt" --model "input/my model.txt"
```

Edit the operator switches, coefficients and stability declarations in `input/my model.txt`. Keep its `_metadata` line and the companion `input/.model_metadata/` folder. Use the same model path for subsequent commands:

```bash
./run.sh channels "input/my model.txt"
./run.sh run "input/my model.txt"
```

`run` integrates all discovered production channels by default. To select one, use `rates --selected` with an ID from **your** channel list. For a scan that changes only coefficients, you can copy the prepared model under another name in the same directory and edit the copy directly. The [user guide](user_guide/Madgraph_step_User_Guide.pdf) covers the commands, output files and physics limits.
