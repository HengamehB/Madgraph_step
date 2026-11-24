# Manohar–Wise colour-octet weak doublet

**PASS.** The exported model reproduces the three scalar masses and the finite-mass, tree-level partial widths below. The reference is [Manohar and Wise, Eqs. (13)–(15)](https://arxiv.org/pdf/hep-ph/0606172#page=4). The 750 GeV mass parameter and (4, 1, 1) portal couplings combine choices discussed in the paper. This benchmark takes real ηU = 1 and ηD = 0.

| Observable | Analytic (GeV) | Local result (GeV) | Relative difference |
| --- | ---: | ---: | ---: |
| X1 | 789.38239696 | 789.38239696 | 1.11e-16 |
| X2r | 817.67536046 | 817.67536046 | 0 |
| X2i | 779.72330123 | 779.72330123 | 0 |
| charged octet to t bbar | 6.95253114 | 6.95253113 | 1.93e-09 |
| CP-even octet to t tbar | 5.92686334 | 5.92686431 | 1.64e-07 |
| CP-odd octet to t tbar | 6.79318533 | 6.79318563 | 4.44e-08 |

The formulas use the benchmark vacuum expectation value 246.22056907 GeV, top mass 172 GeV and bottom mass 4.7 GeV. The charged width includes finite-bottom-mass kinematics. The neutral CP-even and CP-odd widths scale as β³ and β, respectively. The width tolerance is 0.1%.

The inclusive, tree-level production cross section for **pp → S⁻S⁺** at 13.6 TeV is **0.47549019 ± 0.00281 pb** (C0005). This is a local prediction with the shipped PDF, scale and cuts. The paper’s gluon-only cross section describes a different production selection.

The ranking contains 81 calculated rows among 7,049 histories. Every calculated rate and integration uncertainty equals the corresponding production quantity times the complete cascade branching fraction. The other rates are blank.

The exported octet–quark vertices carry a fundamental SU(3) generator. The neutral CP-even and CP-odd couplings use the scalar and pseudoscalar spin structures. FeynRules verified Hermiticity, kinetic normalization and the mass spectrum before the fresh MadGraph import.

## Run files

- [Generated Lagrangian](lagrangian_expression.tex)
- [Contraction conventions](contractions.json)
- [Spectrum and mixing](spectrum.json)
- [Export validation](validation.json)
- [FeynRules extension](export/6d852e1a12004968/attempt-fe106aa690/feynrules_source/generic_joint_scalar.fr)
- [FeynRules log](export/6d852e1a12004968/attempt-fe106aa690/feynrules.log)
- [UFO vertices](export/6d852e1a12004968/attempt-fe106aa690/mw_colour_octet_UFO/vertices.py)
- [UFO Lorentz structures](export/6d852e1a12004968/attempt-fe106aa690/mw_colour_octet_UFO/lorentz.py)
- [MadGraph import](export/6d852e1a12004968/attempt-fe106aa690/madgraph_import/madgraph.log)
- [Width card](decays/3497f9a5d17ecade/attempt-21dc6933a3/two_body_param_card.dat)
- [MadWidth log](decays/3497f9a5d17ecade/attempt-21dc6933a3/madgraph.log)
- [Integration parameter card](rates/C0005/7bcae7a375744bb0/attempt-55c9eb325a/C0005/process/Cards/param_card.dat)
- [Integration run card](rates/C0005/7bcae7a375744bb0/attempt-55c9eb325a/C0005/process/Cards/run_card.dat)
- [Integration log](rates/C0005/7bcae7a375744bb0/attempt-55c9eb325a/C0005/madgraph.log)
- [Ranked CSV](ranking.csv)
- [Report](report.html)
- [Completion manifest](manifest.json)
