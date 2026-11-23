# Sample output · coloured triplet and sextet

Open **report.html** in your browser after extracting the [main ZIP](https://github.com/HengamehB/Madgraph_step/archive/refs/heads/main.zip). Keep the sample_output folder together so the report can open its CSV tables. GitHub's file view shows the HTML source.

[HTML report](report.html) · [Full ranked CSV](ranking.csv) · [Production cross section](cross_sections.csv) · [Decay widths and branching fractions](decays.csv) · [Calculation settings](manifest.json)

The [macOS benchmark](https://github.com/HengamehB/Madgraph_step/actions/runs/45081434860) used 13.6 TeV proton collisions, two-body decays and one integrated production channel. The original ranked CSV contains **1,925 decay histories**. **30** have numerical rates. The other **1,895** have blank cross sections and rates. The model coefficients correspond to [the shipped input](../input/my_model.txt).

| Selected process | Production cross section (pb) | Branching fraction | Cross section × BR (pb) |
| --- | ---: | ---: | ---: |
| pp → x1 → u u | 1.1691955 ± 0.00457 | 0.3392486 | 0.39664794 ± 0.00155 |

The width tables preserve the four measured totals and 10 partial widths recorded by the benchmark. Its integration error propagates into the quoted rate. Overlapping decay histories cannot be added to form an inclusive prediction.
