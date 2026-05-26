# Missing-Data Handling in `perception_analysis.py`

## Overview

The perception analysis pipeline estimates psychometric functions separately for each:

* subject,
* modality,
* condition,
* standard duration.

For each standard duration, the script computes a psychometric curve across multiple comparison levels and estimates two parameters:

* **PSE (Point of Subjective Equality)**
  Horizontal displacement of the psychometric function. A PSE of `0` indicates no perceptual bias.

* **DL (Discrimination Limen)**
  Sensitivity/discrimination parameter related to the slope of the psychometric function. Smaller DL values indicate steeper psychometric functions and better discrimination performance.

All comparisons are expressed as proportional deviations from the standard duration:

```text
(comparison - standard) / standard
```

Therefore:

```text
0.20 = +20%
-0.20 = -20%
```

---

# Fit-validity parameters

The following parameters determine whether a psychometric fit is considered valid:

```python
FIT_MAX_ABS_PSE
FIT_MAX_DL
FIT_MIN_DL
MIN_VALID_STANDARDS
```

## `FIT_MAX_ABS_PSE`

Maximum allowed absolute PSE value.

Example:

```python
FIT_MAX_ABS_PSE = 0.20
```

means that psychometric fits with:

```text
|PSE| > 20%
```

are rejected.

This prevents accepting fits whose estimated perceptual bias falls far outside the range of tested comparison values.

---

## `FIT_MAX_DL`

Maximum allowed DL value.

Example:

```python
FIT_MAX_DL = 0.20
```

means that psychometric fits with:

```text
DL > 20%
```

are rejected.

Large DL values correspond to very shallow psychometric functions, usually indicating unstable or poorly constrained fits.

---

## `FIT_MIN_DL`

Minimum allowed DL value.

Example:

```python
FIT_MIN_DL = 0.0
```

requires DL to be strictly positive.

DL behaves as a scale/sensitivity parameter and therefore cannot meaningfully be zero or negative.

---

## `MIN_VALID_STANDARDS`

Minimum number of valid standard-level fits required to compute the subject-level mean DL for a given modality and condition.

Example:

```python
MIN_VALID_STANDARDS = 3
```

means that at least 3 standards must have valid DL estimates for the corresponding subject/modality/condition combination.

---

# Handling of missing and invalid data

## Step 1 — Psychometric fitting

For each standard duration, the psychometric function is constructed across comparison levels.

For each comparison level, the script computes:

```text
number of “longer” responses /
number of valid responses
```

These proportions define the psychometric curve used to estimate PSE and DL.

If:

* the optimization fails,
* the estimates are non-finite,
* or the estimated parameters violate the fit-validity criteria,

then:

```python
PSE = NaN
DL = NaN
```

for that standard.

No imputation occurs at this stage.

---

## Step 2 — Outlier rejection

After fitting, additional outlier rejection is performed separately for each combination of:

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

No imputation is performed after outlier rejection.

---

## Step 3 — Averaging across standards

For each subject, modality, and condition, the script computes the mean DL across standards using:

```python
np.nanmean()
```

Therefore:

* valid standards contribute to the mean,
* invalid or missing standards are ignored.

Example:

```text
[0.04, 0.05, NaN, 0.03, NaN]
```

becomes:

```text
mean([0.04, 0.05, 0.03])
```

---

## Step 4 — Minimum-validity requirement

Before computing the subject-level mean DL, the script counts the number of valid standard-level DL estimates.

If:

```text
number of valid standards < MIN_VALID_STANDARDS
```

then the subject-level mean DL becomes:

```python
NaN
```

for that modality and condition.

---

## Step 5 — Subject exclusion from repeated-measures ANOVA

The repeated-measures ANOVA requires a complete within-subject design.

Therefore, subjects are excluded if they do not contain all required:

* modality,
* condition

combinations after the previous cleaning steps.

For the current two-way repeated-measures ANOVA:

```text
Modality × Condition
```

each subject must contain four valid combinations:

* auditory beat,
* auditory interval,
* visual beat,
* visual interval.

Subjects missing any of these combinations are excluded from the ANOVA.

---

# Important conceptual consequence

Missing data are never replaced by group medians or other imputed values.

Instead:

* invalid standards are ignored,
* subject-level means are computed from the remaining valid standards,
* subjects are excluded only if insufficient valid information remains.

This avoids artificially reducing variability through imputation while still allowing partially valid psychometric data to contribute to the analysis.

