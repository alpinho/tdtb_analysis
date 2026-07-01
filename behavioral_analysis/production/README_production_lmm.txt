# Production LMM -- model and outputs

Reference for the linear mixed model (LMM) fitted in `production_lmm.py`
(function `mixed_ancova_tables`) and for every column of the two TSV
files it writes.

## 1. What is being modelled

The dependent variable is `signed_asynchrony` (ms): how far a produced
event falls before (negative) or after (positive) its target.

The model is fitted on **subject-level values**, not raw trials. Before
fitting, `ffx_dvar` collapses across sessions to one value per
`condition x modality x standard x subject` cell, using one of two
estimators:

- `mean` -- the subject's mean asynchrony in that cell. Captures
  systematic **bias / accuracy** (early vs late).
- `std` -- the subject's SD of asynchrony across sessions in that cell.
  Captures **variability / consistency**.

`mean` and `std` are analysed separately, so each produces its own pair
of output files.

## 2. Predictors

| Predictor    | Type                     | Coding / centring                                            |
|--------------|--------------------------|--------------------------------------------------------------|
| `condition`  | categorical, 2 levels    | treatment-coded, reference = **beat** (so terms are interval - beat) |
| `modality`   | categorical, 2 levels    | treatment-coded, reference = **auditory** (so terms are visual - auditory) |
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
modality (useful for interpreting a Modality interaction).

### Random effects

`re_formula = '~standard_c'` gives each subject their own **baseline**
asynchrony (random intercept) and their own **sensitivity to standard**
(random slope). This is the repeated-measures part of the model: it
accounts for the fact that all cells come from the same set of subjects.

If the random-slope model fails to converge, `_fit_mixedlm` retries with
a different optimiser and, as a last resort, drops the random slope
(intercept only). The `random_effects` column of the model-info file
always reports what was **actually** fitted.

### Fixed-effect terms, interpreted (`2way` model)

With both references (beat, auditory) and `standard_c` centred:

| Term                                             | Meaning                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------|
| `Intercept`                                      | predicted asynchrony for beat, auditory, at the mean standard           |
| `C(condition)[T.interval]`                       | interval - beat, for auditory, at mean standard                         |
| `C(modality)[T.visual]`                          | visual - auditory, for beat, at mean standard                           |
| `standard_c`                                     | slope on standard (ms of asynchrony per ms of standard), beat, auditory |
| `...[T.interval]:standard_c`                     | how the standard slope changes from beat to interval                    |
| `...[T.visual]:standard_c`                       | how the standard slope changes from auditory to visual                  |
| two- and three-way `:` terms                     | how one effect is modulated by the other factor(s)                      |

## 4. Output files

Per estimator and session tag, two files are written to the `tables`
folder:

```
lmm_<estimator>_<sesstag>_results.tsv     <- coefficients + omnibus tests
lmm_<estimator>_<sesstag>_modelinfo.tsv   <- fit diagnostics
```

`<estimator>` is `mean` or `std`; `<sesstag>` is a session key
(`allses`, `behavses`, `ses-01`, ...). The JASP wide-format files
(`wide_df_production_*`) are unchanged and written separately.

### 4a. `..._results.tsv`

One long table holding all three models. Each row is tagged by `model`
(which model) and `block` (which kind of row).

| Column      | `block = coefficient`                          | `block = omnibus`                              |
|-------------|------------------------------------------------|------------------------------------------------|
| `model`     | `2way` / `auditory_1way` / `visual_1way`       | same                                           |
| `block`     | `coefficient` (one row per model parameter)    | `omnibus` (one row per grouped term)           |
| `term`      | parameter name (e.g. `C(condition)[T.interval]`) | factor/term name (e.g. `C(condition)`)       |
| `estimate`  | fixed-effect coefficient (units of the DV)     | *(empty)*                                       |
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
  degrees of freedom and `df` refers only to the number of constraints
  in the omnibus test.
- Report the **omnibus** rows for "is there an effect of X" questions and
  the **coefficient** rows for "how large, in which direction".

### 4b. `..._modelinfo.tsv`

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
check `converged` before trusting a row of the results file.

## 5. What changed from the previous version

Previously each model wrote three files (`_fixed_effects`, `_wald_terms`,
`_model_info`), i.e. nine files per estimator and session. These are now
consolidated into the two files above: the statistical results you cite
(`_results.tsv`) and the diagnostics you check (`_modelinfo.tsv`). The
`model` and `block` columns replace the per-model, per-table filenames.
