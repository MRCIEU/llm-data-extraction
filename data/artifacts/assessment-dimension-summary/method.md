# Assessment dimension summary prototype

For each model and dimension, the point estimate is the sum of positive, nonmissing processed scores across five question groups and both observed reviewers divided by the eligible cell count.
This available-cell weighting preserves the existing score eligibility rule and gives greater weight to studies, reviewers and groups with more eligible cells.
The original reviewer halves are tagged before matching the complete 100-study by eight-model by two-reviewer grid.
The 16 audited Q5 Detail exclusions remain missing, and processed completeness scores are not transformed again.

Intervals are 95% percentile study-cluster bootstrap intervals from 10,000 replicates, using NumPy Generator(PCG64), seed 20261010, and numpy.quantile(method='linear').
Each replicate draws 100 study identifiers with replacement from the sorted study list, with the same draws used for all 24 targets.
Each selected study retains both reviewers, all question cells and their missingness, including studies contributing no eligible cells to a target.
Repeated draws retain their multiplicities and each replicate recomputes the sum/count ratio.
Displayed points are the observed estimates, not bootstrap averages.
The CSV reports eligible cells, contributing studies, undefined replicate counts and interval status.
If any replicate for a target is undefined, its interval is withheld without dropping replicates and figure export fails explicitly after saving the diagnostic summary.
Equal endpoints are reported as degenerate intervals.

The marginal intervals describe study-sampling variation conditional on the two observed reviewers, fixed extraction outputs and the existing missing-score rule.
They do not measure variation across reviewers or repeated model runs, correct assessment bias, or resolve missingness assumptions.
Interval overlap is not a formal paired model comparison.
The display uses the fixed descending pooled overall-score order from the heatmap; this order is separate from the unchanged alphabetical statistical target-array order and is not a formal model comparison.
Manuscript integration and final caption/page checks are handled separately.

## Compact and combined figures

The compact uncertainty panel is 6.175 by 2.2 inches with text at least 8 points.
The right-hand Mean columns display the 24 observed means to two decimal places, aligned with their model rows; interval endpoints are shown graphically without numeric labels.
The combined figure is 6.175 by 5.77 inches: panel A embeds the existing heatmap at its full width and original content scale, while panel B displays the same 24 estimates and intervals in the same descending pooled overall-score order as panel A.
Only 90 verified all-white bottom rows are cropped from the heatmap display, preserving every original nonwhite pixel and 34 blank rows below its content.
Panel A retains descending pooled overall-score order, all cell annotations and the three original dimension palettes.
The combined PDF contains the 300 DPI heatmap raster with only its blank bottom margin cropped, and vector uncertainty marks/text, so it is not fully vector.
The heatmap input hash, physical geometry and final output hashes are recorded in metadata.json.
Use the assessment-combined recipe to render the approved CSV without rerunning bootstrap or rewriting the CSV.

## Reproduction

```sh
just --no-dotenv -f justfile-processing assessment-dimension-summary --input-file data/artifacts/assessment-results/assessment-results-numeric.csv --audit-file data/artifacts/assessment-results/assessment-score-exclusions.csv --output-dir data/artifacts/assessment-dimension-summary --replicates 10000 --seed 20261010 --confidence 0.95
```

Input and script hashes, software versions, study order and output filenames are recorded in metadata.json.
