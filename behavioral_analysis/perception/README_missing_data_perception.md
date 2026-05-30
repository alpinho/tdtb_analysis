# Missing-Data Handling in `perception_analysis.py`

## Overview

The perception analysis pipeline estimates psychometric functions from the
proportion of valid “longer” responses at each comparison level. The relevant
analysis units are:

* subject,
* session,
* modality,
* condition,
* standard duration.

For each standard duration, the script estimates two psychometric parameters:

* **PSE (Point of Subjective Equality)**  
  Horizontal displacement of the psychometric function. A PSE of `0` indicates
  no perceptual bias.

* **DL (Discrimination Limen)**  
  Sensitivity/discrimination parameter related to the slope of the
  psychometric function. Smaller DL values indicate steeper psychometric
  functions and better discrimination performance.

All comparison levels are expressed as proportional deviations from the
standard duration:

```text
(comparison - standard) / standard
```

Therefore:

```text
0.20 = +20%
-0.20 = -20%
```

---

# Current analysis settings

The current script uses the following missing-data and fit-validity settings:

```python
FIT_MAX_ABS_PSE = 0.50
FIT_MAX_DL = 0.50
FIT_MIN_DL = 0.0
MIN_VALID_STANDARDS = 1
SESSION_AVERAGE_MULTISESSION_ANOVA = True
SHOW_GROUP_FIT_ESTIMATES = True
```

`SESSION_AVERAGE_MULTISESSION_ANOVA = True` is important: for multi-session
summaries, the script does **not** pool raw trials across sessions before
fitting the ANOVA input. Instead, it estimates session-level psychometric
parameters first and then averages those estimates within subject.

---

# Fit-validity parameters

## `FIT_MAX_ABS_PSE`

Maximum allowed absolute PSE value.

With the current setting:

```python
FIT_MAX_ABS_PSE = 0.50
```

psychometric fits with:

```text
|PSE| > 50%
```

are rejected.

This prevents accepting fits whose estimated perceptual bias falls very far
outside the tested comparison range.

---

## `FIT_MAX_DL`

Maximum allowed DL value.

With the current setting:

```python
FIT_MAX_DL = 0.50
```

psychometric fits with:

```text
DL > 50%
```

are rejected.

Large DL values correspond to very shallow psychometric functions, usually
indicating unstable or poorly constrained fits.

---

## `FIT_MIN_DL`

Minimum allowed DL value.

With the current setting:

```python
FIT_MIN_DL = 0.0
```

DL must be strictly positive.

DL behaves as a scale/sensitivity parameter and therefore cannot meaningfully
be zero or negative.

---

## `MIN_VALID_STANDARDS`

Minimum number of valid standard-level DL estimates required for a
subject/modality/condition cell to enter the repeated-measures ANOVA.

With the current setting:

```python
MIN_VALID_STANDARDS = 1
```

at least one valid standard-level DL estimate is required for that
subject/modality/condition cell.

---

# Handling of missing and invalid data

## Step 1 — Trial-level response handling

For each subject, session, modality, condition, and standard, the script builds
a psychometric curve across comparison levels.

At each comparison level, it computes:

```text
number of “longer” responses / number of valid responses
```

Trials with missing responses do not contribute to the numerator or denominator
of the corresponding comparison-level response proportion.

If a comparison-level cell has no valid responses, its response proportion is
undefined and becomes `NaN`.

---

## Step 2 — Beat/interval grid checks

The script requires Beat and Interval trials to have compatible standard and
comparison grids before estimating the psychometric frequencies.

For multi-session summaries, a subject/session/modality cell is used only when
it contains both Beat and Interval trials. If a given session is incomplete for
that subject/modality, that session is skipped for the multi-session average.

If no valid session remains for a subject/modality/condition combination, the
corresponding PSE and DL values are set to `NaN`.

---

## Step 3 — Psychometric fitting

For each valid standard-level psychometric curve, the script fits the selected
estimator:

```python
mle_expit
```

or:

```python
mle_cdf
```

The ANOVA input is based on `mle_expit`; `mle_cdf` is retained for descriptive
psychometric/PSE plotting.

If:

* the optimization fails,
* the estimates are non-finite,
* `|PSE| > FIT_MAX_ABS_PSE`,
* `DL > FIT_MAX_DL`,
* or `DL <= FIT_MIN_DL`,

then:

```python
PSE = NaN
DL = NaN
```

for that standard.

No imputation occurs at this stage.

---

## Step 4 — Multi-session summaries

For single-session tags such as:

```text
ses-01, ses-02, ses-03, ses-04, ses-05
```

the script estimates psychometric parameters directly from that session.

For multi-session tags such as:

```text
allses, behavses, imgses, behav12, behav13, behav23
```

the current script uses session-level averaging:

```text
fit each available session separately
→ average PSE/DL across sessions within subject
→ enter the subject-level averages into the ANOVA
```

This avoids refitting a nonlinear psychometric function after pooling raw
trials across sessions. Pooling raw trials can produce DL/PSE values that are
not equivalent to the average of the session-specific estimates.

Sessions with incomplete Beat/Interval data are skipped for that
subject/modality. They are not treated as valid zeroes and are not imputed.

---

## Step 5 — Outlier rejection

After fitting, additional outlier rejection is performed separately for each
combination of:

* standard,
* modality,
* condition.

Outliers are detected using an interquartile-range (IQR) rule:

```text
upper threshold = Q3 + 1.7 × IQR
lower threshold = Q1 − 1.7 × IQR
```

Values outside these thresholds are converted to:

```python
NaN
```

This procedure is applied independently to:

* PSE,
* DL.

The raw values are retained in:

```text
PSE_raw
DL_raw
```

and the cleaned values are stored in:

```text
PSE
DL
```

No imputation is performed after outlier rejection.

---

## Step 6 — Averaging across standards for the ANOVA

For each subject, modality, and condition, the script computes the mean DL
across standards.

Invalid or missing standards are ignored when computing the mean. For example:

```text
[0.04, 0.05, NaN, 0.03, NaN]
```

becomes:

```text
mean([0.04, 0.05, 0.03])
```

Before a cell is retained, the script counts the number of valid standard-level
DL estimates:

```text
ValidStandards = count(non-NaN DL values)
```

If:

```text
ValidStandards < MIN_VALID_STANDARDS
```

then the subject/modality/condition cell is excluded from the ANOVA input.

The excluded cells are saved to:

```text
anovas/twoway/twoway_excluded_cells_<sesstag>.tsv
```

---

## Step 7 — Subject exclusion from repeated-measures ANOVA

The two-way repeated-measures ANOVA requires a complete within-subject design:

```text
Modality × Condition
```

Therefore, each subject must have all four valid combinations:

* auditory beat,
* auditory interval,
* visual beat,
* visual interval.

Subjects missing any of these four combinations after fitting, cleaning, and
minimum-validity filtering are excluded from the ANOVA.

The excluded subjects are saved to:

```text
anovas/twoway/twoway_excluded_<sesstag>.tsv
```

---

# Group psychometric plots

For single-session tags, group psychometric plots are computed from the selected
session.

For multi-session tags, the group psychometric plots now follow the same
session-averaging principle as the ANOVA:

```text
compute response frequencies separately per session
→ average response frequencies across sessions within subject
→ average across subjects
→ fit/plot the group psychometric curve
```

This means the group plots no longer summarize multi-session tags by pooling
all raw trials across sessions first.

Group-level PSE and DL values are displayed on the group psychometric plots
when:

```python
SHOW_GROUP_FIT_ESTIMATES = True
```

These group-level estimates are descriptive and are not the direct input to the
ANOVA. The ANOVA uses subject-level DL values.

---

# Post-hoc tests

The repeated-measures ANOVA is run on DL values with within-subject factors:

```text
Modality × Condition
```

The saved post-hoc file:

```text
anovas/twoway/twoway_posthoc_<sesstag>.tsv
```

contains the Beat vs Interval comparisons within each modality. These are the
same comparisons shown by the within-subplot annotations in the ANOVA boxplot.

---

# Important conceptual consequence

Missing data are never replaced by group medians or other imputed values.

Instead:

* invalid trials lead to undefined response proportions when no valid response
  exists for a comparison level,
* invalid standard-level fits become `NaN`,
* incomplete sessions are skipped during multi-session averaging,
* session-level estimates are averaged within subject for multi-session tags,
* invalid standards are ignored when computing subject-level mean DL,
* subject/modality/condition cells are excluded if they do not meet
  `MIN_VALID_STANDARDS`,
* subjects are excluded from the repeated-measures ANOVA only if the complete
  Modality × Condition design is not available after these steps.

This avoids artificially reducing variability through imputation while allowing
partially valid psychometric data to contribute whenever enough valid
within-subject information remains.
