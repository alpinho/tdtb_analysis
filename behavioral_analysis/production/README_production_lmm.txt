# Production LMM -- model and outputs

Reference for the linear mixed model (LMM) fitted in `production_lmm.py`
(function `mixed_ancova_tables`) and for every column of the three TSV
files it writes. This version adds estimated marginal means (EMMs) and a
batch / latency-correction run configuration.

## 1. What is being modelled

The dependent variable is `signed_asynchrony` (SA): the reproduction
error on the Production task, defined in the manuscript as
`SA = (RT - S) / S`, where RT is the produced interval and S is the
Standard. SA is signed (positive = pressed late, negative = pressed
early) and normalised by S (dimensionless, comparable across standards).

The model is fitted on **subject-level values**, not raw trials. Before
fitting, `ffx_dvar` collapses across sessions to one value per
`condition x modality x standard x subject` cell, using one of two
estimators:

- `mean` -- the subject's mean SA in that cell. Captures systematic
  **bias / accuracy** (early vs late).
- `std` -- the subject's SD of SA across sessions in that cell. Captures
  **variability / consistency**.

`mean` and `std` are analysed separately, so each produces its own set of
output files.

## 2. Predictors

| Predictor    | Type                     | Coding / centring                                            |
|--------------|--------------------------|--------------------------------------------------------------|
| `condition`  | categorical, 2 levels    | treatment-coded, reference = **beat** (terms are interval - beat) |
| `modality`   | categorical, 2 levels    | treatment-coded, reference = **auditory** (terms are visual - auditory) |
| `standard`   | continuous (ms)          | mean-centred as `standard_c = standard - mean(standard)`     |

Because `standard` is centred, the intercept and the categorical main
effects are evaluated **at the mean standard**, not at standard = 0.

## 3. The three models

All three are fitted on the same data, by maximum likelihood
(`reml=False`), each with a per-subject random intercept **and** a random
slope for `standard_c`:

| `model` label   | Fixed-effects formula                                       | Data used        |
|-----------------|-------------------------------------------------------------|------------------|
| `2way`          | `signed_asynchrony ~ C(condition) * C(modality) * standard_c` | all rows         |
| `auditory_1way` | `signed_asynchrony ~ C(condition) * standard_c`             | auditory rows    |
| `visual_1way`   | `signed_asynchrony ~ C(condition) * standard_c`             | visual rows      |

`*` expands to main effects plus all interactions. The `2way` model is
the omnibus test of Condition, Modality, Standard and their interactions;
the two `1way` models decompose Condition x Standard **within** each
modality.

### Random effects

`re_formula = '~standard_c'` gives each subject their own **baseline** SA
(random intercept) and their own **sensitivity to standard** (random
slope). This is the repeated-measures part of the model. If the
random-slope model fails to converge, `_fit_mixedlm` retries with a
different optimiser and, as a last resort, drops the random slope
(intercept only). The `random_effects` column of the model-info file
always reports what was **actually** fitted.

### Fixed-effect terms, interpreted (`2way` model)

With both references (beat, auditory) and `standard_c` centred:

| Term                                             | Meaning                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------|
| `Intercept`                                      | predicted SA for beat, auditory, at the mean standard                   |
| `C(condition)[T.interval]`                       | interval - beat, for auditory, at mean standard                         |
| `C(modality)[T.visual]`                          | visual - auditory, for beat, at mean standard                           |
| `standard_c`                                     | slope of SA on standard (SA units per ms), beat, auditory              |
| `...[T.interval]:standard_c`                     | how the standard slope changes from beat to interval                    |
| `...[T.visual]:standard_c`                       | how the standard slope changes from auditory to visual                  |
| two- and three-way `:` terms                     | how one effect is modulated by the other factor(s)                      |

## 4. Estimated marginal means (EMMs)

An EMM is a **model-predicted cell mean**: the fitted SA for one
`modality x condition` cell, with `standard_c` held at 0 (i.e. at the
mean standard). It is the model-based counterpart of the raw cell mean,
read straight off the fixed effects.

For a cell, the code builds the model's design row `L` for that cell
(from the stored patsy design, so it aligns with the fitted parameters
whatever the coding), and computes:

- estimate = `L @ beta` (linear combination of the fixed effects),
- se = `sqrt(L @ V @ L)`, with `V` the fixed-effect covariance
  (`result.cov_params()` restricted to the fixed effects),
- Wald z = estimate / se, two-sided p, and a 95% Wald CI.

Using the full covariance `V` gives the **correct** SE for cells built
from several coefficients (e.g. the interval cells combine the intercept,
the condition term and, in the 2way model, an interaction); this is not
the sum of the individual coefficient SEs. Inference is Wald z, matching
the coefficient block.

Which cells are reported (`_emm_cells`):

- `2way`: all four cells (auditory/visual x beat/interval), from the
  pooled fit.
- `auditory_1way`: the two auditory cells (beat, interval), from the
  auditory-only fit.
- `visual_1way`: the two visual cells, from the visual-only fit.

Note: each modality's cells therefore appear **twice** across models --
once from `2way` (variance pooled across modalities) and once from the
matching `1way` fit (that modality's data only). The estimates and SEs
can differ slightly between the two; this is expected, not a bug.

## 5. Output files

Per estimator and session tag, three files are written to the `tables`
folder:

```
lmm_<estimator>_<sesstag>_results.tsv     coefficients + omnibus tests
lmm_<estimator>_<sesstag>_emm.tsv         estimated marginal means
lmm_<estimator>_<sesstag>_modelinfo.tsv   fit diagnostics
```

`<estimator>` is `mean` or `std`; `<sesstag>` is a session key
(`allses`, `behavses`, `ses-01`, ...). The JASP wide-format files
(`wide_df_production_*`) are written separately to the `jasp` folder.

### 5a. `..._results.tsv`

One long table holding all three models. Each row is tagged by `model`
(which model) and `block` (which kind of row).

| Column      | `block = coefficient`                          | `block = omnibus`                              |
|-------------|------------------------------------------------|------------------------------------------------|
| `model`     | `2way` / `auditory_1way` / `visual_1way`       | same                                           |
| `block`     | `coefficient` (one row per model parameter)    | `omnibus` (one row per grouped term)           |
| `term`      | parameter name (e.g. `C(condition)[T.interval]`) | factor/term name (e.g. `C(condition)`)       |
| `estimate`  | fixed-effect coefficient (SA units)            | *(empty)*                                       |
| `se`        | standard error of `estimate`                   | *(empty)*                                       |
| `statistic` | Wald **z** ( = `estimate / se`)                | Wald **chi-square** for the term               |
| `df`        | *(empty)*                                      | degrees of freedom (number of coefficients tested) |
| `p`         | two-sided Wald p-value                          | chi-square p-value                             |
| `ci_low`    | lower 95% Wald CI for `estimate`               | *(empty)*                                       |
| `ci_high`   | upper 95% Wald CI for `estimate`               | *(empty)*                                       |

Notes:

- **coefficient** rows quantify the size and direction of a specific
  contrast. **omnibus** rows are the mixed-model analogue of an ANCOVA
  main effect or interaction: they test whether all coefficients making
  up a term are jointly zero. For a 1-df term the two agree exactly
  (chi-square = z^2).
- The statistic is a **z / chi-square**, not t / F: statsmodels MixedLM
  uses a large-sample normal approximation, so there are no residual
  degrees of freedom, and `df` refers only to the number of constraints
  in the omnibus test.

### 5b. `..._emm.tsv`

One row per reported cell, across all three models. The schema differs
from the results file: it carries `modality` and `condition` columns
instead of `term`.

| Column      | Meaning                                                              |
|-------------|---------------------------------------------------------------------|
| `model`     | `2way` / `auditory_1way` / `visual_1way`                            |
| `block`     | always `emm`                                                         |
| `modality`  | `auditory` / `visual` (the cell's modality)                         |
| `condition` | `beat` / `interval` (the cell's condition)                          |
| `estimate`  | predicted SA for the cell, at the mean standard                     |
| `se`        | Wald standard error of the cell mean ( = `sqrt(L V L)`)            |
| `statistic` | Wald **z** ( = `estimate / se`)                                     |
| `df`        | *(empty)*                                                            |
| `p`         | two-sided Wald p-value (cell mean vs 0)                              |
| `ci_low`    | lower 95% Wald CI                                                    |
| `ci_high`   | upper 95% Wald CI                                                    |

The `p` here tests the cell mean against 0 (i.e. against SA = 0, the
"RT = Standard" line). To compare cells to each other, read the
corresponding contrast in the coefficient block of the results file.

### 5c. `..._modelinfo.tsv`

One row per model; fit metadata and diagnostics.

| Column           | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `model`          | `2way` / `auditory_1way` / `visual_1way`                                |
| `formula`        | the fixed-effects formula that was fitted                               |
| `random_effects` | random-effects structure **actually** fitted (`intercept + standard_c`, or `intercept` if the slope was dropped) |
| `n_obs`          | number of rows (subject x cells) entering the model                     |
| `n_subjects`     | number of subjects (random-effect groups)                               |
| `converged`      | optimiser convergence flag; if `False`, treat estimates with caution    |
| `log_likelihood` | maximum-likelihood log-likelihood                                       |
| `aic`            | Akaike information criterion (lower = better)                            |
| `bic`            | Bayesian information criterion (lower = better)                         |

Because all models are fitted by ML, `aic` / `bic` / `log_likelihood`
are comparable across models fitted on the same DV and data. Always
check `converged` before trusting a row of the results or EMM files.

## 6. Inputs, batches and the output tree

The run block (`if __name__ == "__main__"`) loops over two nested
selectors:

- **Batch** (`BATCHES_TO_RUN`): `first` or `second`. The first batch
  (Expyriment) and second batch (PsychoPy) have their own subject lists
  and session tags (`fb_*` / `sb_*` dictionaries).
- **Input type** (`INPUT_TYPES_TO_RUN`): `latency_corrected` or
  `uncorrected`. This selects which input dataframe is read and where the
  outputs go.

Input dataframes live in `production_results/dataframes/` and are named

```
df_production_<fb|sb>_<a_b_c>_<sesstag>.tsv
```

where `fb`/`sb` is the batch and `a_b_c` records, in ms, the latency
components removed from the CPU-logged RT when that dataframe was built
upstream:

- `a` is the physical-onset latency (133 for the first batch / Expyriment,
  63 for the second / PsychoPy),
- `b` and `c` are the response-side latency components (button-press and
  the remaining term; confirm the exact mapping against the dataframe-
  generation script),
- `0_0_0` is the **uncorrected** RT (no latency removed).

Outputs mirror this under `production_results/lmm/`:

```
production_results/lmm/<batch_folder>/{jasp,plots,tables}/
```

with `<batch_folder>` one of `first_batch_0_0_0`, `first_batch_133_35_20`,
`second_batch_0_0_0`, `second_batch_63_35_20`. So the batch and the
latency-correction choice are both encoded in the path, keeping the two
pipelines fully separate on disk.

The two selector lists are meant to be edited by hand (comment out
entries) to run one combination at a time. As written, the script runs
both batches with the `uncorrected` input only.

A note on why the latency choice matters for this task specifically:
because SA is normalised (`SA = (RT - S)/S`), removing a fixed latency
`L` in ms enters SA as `-L/S`, i.e. a term that varies hyperbolically
with the standard and (since the onset latency differs by modality) by
modality. For the Production task, where the produced interval is bounded
by the marking event and the participant's own feedback (both rendered
through the same output path, so the onset latency cancels and the
button latency sits inside the closed sensorimotor loop), the
`uncorrected` RT is the appropriate measure, and the correction would
inject that `-L/S` artefact onto the very predictors under test. For
open-loop reaction tasks (e.g. NTFD) the opposite holds and the
correction is appropriate.

## 7. Plots

`plot_ancova` writes two group-level figures per (estimator, session) to
the `plots` folder: mean SA across standards (`mean_lmm_production_*`) and
SD of SA across standards (`std_lmm_production_*`), each with auditory and
visual panels sharing a common y-range and a reference line at SA = 0
(`RT = Standard`).

## 8. What changed from the previous version

- **EMMs added.** A third output file, `..._emm.tsv`, now reports each
  `modality x condition` cell mean at the mean standard, with a Wald SE,
  z, p and 95% CI. Output is now three files per (estimator, session)
  instead of two.
- **Batch / latency run configuration.** Inputs and outputs are now
  organised by batch (`first` / `second`) and input type
  (`latency_corrected` / `uncorrected`), with the choice encoded in both
  the input filename (`a_b_c`) and the output folder. `INPUT_TYPES_TO_RUN`
  and `BATCHES_TO_RUN` toggle which combinations run.
- **Unchanged.** The three-model structure, the random-effects
  specification, the coefficient and omnibus blocks of `..._results.tsv`,
  and `..._modelinfo.tsv` are as before.
