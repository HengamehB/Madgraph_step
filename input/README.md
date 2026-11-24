# Run a supplied model

After [setting up MadFieldRunner](../README.md), open a terminal in the package directory with your Python environment active. Each row links a particle declaration, its ready-to-run model with numerical coefficients, and the measured output. Run the model file directly. Use `./run.sh init` when you want to generate a fresh coefficient template.

| Example | Particles | Model | Measured output |
|---|---|---|---|
| Triplet and sextet, 1,200 and 1,500 GeV | [Particles](sample_particle_sextet.txt) | [Model](sample_model_sextet.txt) | [Report](../output/triplet_sextet/report.html) |
| Silveira–Zee, Higgs invisible width | [Particles](phantom_portal_particles.txt) | [Model](phantom_portal_model.txt) | [Comparison](../output/phantom_portal/comparison.md) |
| Deshpande–Ma, inert-doublet mass splitting | [Particles](inert_doublet_particles.txt) | [Model](inert_doublet_model.txt) | [Comparison](../output/inert_doublet/comparison.md) |
| Buchmüller–Rückl–Wyler, left-handed leptoquark | [Particles](brw_leptoquark_particles.txt) | [Model](brw_leptoquark_model.txt) | [Comparison](../output/brw_leptoquark/comparison.md) |
| Han–Lewis–McElmurry, identical-quark diquark coupling | [Particles](hlm_diquark_particles.txt) | [Model](hlm_diquark_model.txt) | [Comparison](../output/hlm_diquark/comparison.md) |
| Manohar–Wise, colour-octet doublet | [Particles](mw_colour_octet_particles.txt) | [Model](mw_colour_octet_model.txt) | [Comparison](../output/mw_colour_octet/comparison.md) |

Choose the command for the example you want to reproduce. Each `rates` command integrates one production channel. The singlet's `histories` command computes the Higgs width and discovers histories.

```bash
./run.sh rates input/sample_model_sextet.txt --selected C0004 --page-size 5
./run.sh histories input/phantom_portal_model.txt --page-size 1
./run.sh rates input/inert_doublet_model.txt --selected C0005 --page-size 1
./run.sh rates input/brw_leptoquark_model.txt --selected C0001 --page-size 1
./run.sh rates input/hlm_diquark_model.txt --selected C0004 --page-size 1
./run.sh rates input/mw_colour_octet_model.txt --selected C0005 --page-size 1
```

Your results go to `output/<model_name>/<run-id>/`. The original triplet–sextet model uses `triplet_sextet_particles` as its model name. Each of the other five uses its case name. Open `report.html` for the report, `ranking.csv` for the histories, or `decays.csv` for partial widths and branching fractions. Blank rates identify channels without a numerical integration.

The supplied production runs use 13.6 TeV proton beams and tree-level amplitudes with at most two outgoing new particles. Decays are restricted to two bodies at each step. If you edit the physics, use `./run.sh channels input/<model>.txt` to look up the appropriate channel ID before running `rates`. Keep `.model_metadata/` with the model files when copying them to another directory.
