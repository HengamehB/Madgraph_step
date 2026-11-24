# HLM diquark benchmark

Passed with the 1 TeV weak-triplet colour sextet. Only the charge +4/3 `pp → x1 → uu` channel was integrated. All 42 histories were retained, including 41 without a numerical rate.

| Observable | Independent prediction | Local result |
|---|---:|---:|
| Charge +4/3 mass | 1000 GeV | 1000 GeV |
| Γ(x1 → uu) | 0.79577472 GeV | 0.79577000 GeV |
| BR(x1 → uu) | 1 | 1 |
| σ(pp → x1) | 16.61221352 pb | 16.66900000 ± 0.05967000 pb |
| σ × BR | 16.61221352 pb | 16.66900000 ± 0.05967000 pb |

The UFO contains both conjugate uu vertices, with K6/K6Bar colour tensors and the two conjugate chiral projectors. The coefficient has magnitude √2 × 0.2 = 2√2 × 0.1, matching Eq. (3.2) of [Han, Lewis and McElmurry](https://arxiv.org/pdf/0909.2666#page=5). The independent width includes the identical-fermion symmetry factor and the finite up-quark mass from the exported card. The PDF integral evaluates Eq. (3.4) in Python at 13.6 TeV using the nn23lo1 member-zero grid, six sextet colours, three quark colours and factorization scale 500 GeV for the on-shell 2→1 process. The production difference is +0.952 integration standard errors. The paper's 14 TeV CTEQ6L table uses a different setup.

[Lagrangian](lagrangian.tex) · [spectrum](spectrum.json) · [UFO vertices](export/467f68f13bc8a123/attempt-8045794503/hlm_diquark_UFO/vertices.py) · [UFO couplings](export/467f68f13bc8a123/attempt-8045794503/hlm_diquark_UFO/couplings.py) · [FeynRules log](export/467f68f13bc8a123/attempt-8045794503/feynrules.log) · [fresh MadGraph import](export/467f68f13bc8a123/attempt-8045794503/madgraph_import/madgraph.log) · [width parameter card](decays/86c6fc99494cf0fb/attempt-2363ff3cef/total_param_card.dat) · [MadWidth log](decays/86c6fc99494cf0fb/attempt-2363ff3cef/madgraph.log) · [integration run card](rates/C0004/159cd6cccb2d8291/attempt-eb6f630908/C0004/process/Cards/run_card.dat) · [integration log](rates/C0004/159cd6cccb2d8291/attempt-eb6f630908/C0004/madgraph.log) · [ranked CSV](ranking.csv) · [convolution source](convolution.py)

Run the independent integral from the package directory with your Python environment active.

```bash
python output/hlm_diquark/convolution.py vendor/MG5_aMC/Template/LO/lib/Pdfdata/NNPDF23_lo_as_0130_qed_mem0.grid
```
