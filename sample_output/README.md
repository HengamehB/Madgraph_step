# Sextet example results

These are the reference results for [sample_model_sextet.txt](../input/sample_model_sextet.txt). From the package directory, reproduce them with:

```bash
./run.sh rates input/sample_model_sextet.txt --selected C0004 --page-size 5
```

Open `report.html` locally for the formatted reference. The [ranked CSV](ranking.csv) contains 1,925 decay histories. Of the 14 discovered production channels, only `p p > x1` was integrated, giving 30 numerical rates. Blank rates indicate uncalculated channels.

| Process | Production cross section | Cascade branching fraction | Cross section × BR |
| --- | ---: | ---: | ---: |
| pp → x1 → u u | 1.1691955 ± 0.0045663 pb | 0.3392486 | 0.39664794 ± 0.00154912 pb |

To compare your output, replace `<run-id>` with the directory printed after **Run files**:

```bash
python sample_output/compare.py "output/triplet_sextet_particles/<run-id>"
```

The checker compares the selected histories, the four [total widths](total_widths.csv) and ten recorded [partial widths](decays.csv). It uses a 0.1% relative tolerance for widths and branching fractions, and three combined Monte Carlo standard errors for the production cross section and the direct `u u` rate. It also reports how many histories already have rates from additional channels you integrated in the same run directory. Full calculation settings are in the [manifest](manifest.json).
