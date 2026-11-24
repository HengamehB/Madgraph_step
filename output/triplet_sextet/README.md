# Sextet example results

These are the reference results for [sample_model_sextet.txt](../../input/sample_model_sextet.txt). From the package directory, reproduce them with:

```bash
./run.sh rates input/sample_model_sextet.txt --selected C0004 --page-size 5
```

Open `report.html` locally for the formatted reference. The [ranked CSV](ranking.csv) contains 1,925 decay histories. Of the 14 discovered production channels, only `p p > x1` was integrated, giving 30 numerical rates. Blank rates indicate uncalculated channels.

| Process | Production cross section | Cascade branching fraction | Cross section × BR |
| --- | ---: | ---: | ---: |
| pp → x1 → u u | 1.1691955 ± 0.0045663 pb | 0.3392486 | 0.39664794 ± 0.00154912 pb |

From the package directory, replace the entire path between the quotes below with the full directory printed after **Run files**. Keep the quotes so paths containing spaces work.

```bash
python output/triplet_sextet/compare.py "path/to/saved run"
```

The checker compares the selected histories, the four [total widths](total_widths.csv) and ten recorded [partial widths](decays.csv). It uses a 0.1% relative tolerance for widths and branching fractions, and three combined Monte Carlo standard errors for the production cross section and the direct `u u` rate. It also reports how many histories already have rates from additional channels you integrated in the same run directory. The [reference manifest](manifest.json) records the channel selection and completion status. The [prepared model](../../input/sample_model_sextet.txt) specifies the input coefficients, seed, and calculation settings.
