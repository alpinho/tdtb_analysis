# Missing-Data and Outlier Handling in `production_ancova.py`

## Overview

The production ANCOVA pipeline analyzes signed asynchrony from the
Production task. The dependent variable is:

```text
signed_asynchrony
```

This value is analyzed separately for combinations of:

* subject,
* modality,
* condition,
* standard duration.

The script produces two subject-level summaries:

* **Mean signed asynchrony**
  Average signed asynchrony for each subject, modality, condition, and
  standard.

* **SD signed asynchrony**
  Within-subject standard deviation of signed asynchrony for each subject,
  modality, condition, and standard.

These summaries are then used for:

* wide-format JASP tables,
* group-level plotting,
* linear mixed-model ANCOVA-style analyses.

---

# Input filtering

For each batch, input type, and session tag, the script loads one dataframe:

```text
df_production_<batch>_<latency_parameters>_<session_tag>.tsv
```

The dataframe is filtered in two steps.

## Step 1 — Subject filtering

Only subjects listed for the corresponding session tag are retained.

For example:

* behavioral-session tags use the good behavioral subject list,
* imaging-session tags use the imaging subject list,
* second-batch analyses use the second-batch subject list.

Subjects outside the relevant list are excluded before any dependent-variable
summaries are computed.

## Step 2 — Session filtering

Rows are then filtered according to the session list associated with the tag.

Examples:

```python
'behavses' -> [1, 2, 3]
'imgses'   -> [4, 5]
'allses'   -> [1, 2, 3, 4, 5]
'ses-04'   -> [4]
```

This means that multi-session tags pool the rows from all selected sessions
before subject-level summaries are computed.

---

# Handling of missing data

## Step 1 — Trial/row-level missing values

Rows with missing signed asynchrony are removed before the subject-level
summaries are computed:

```python
df = df.dropna(subset=['signed_asynchrony'])
```

Therefore, rows with `NaN` in `signed_asynchrony` do not contribute to:

* subject-level mean signed asynchrony,
* subject-level SD signed asynchrony,
* group plots,
* mixed-model tables,
* JASP-format tables.

No imputation is performed at this stage.

---

## Step 2 — Subject-level summaries

After removing rows with missing signed asynchrony, the script computes
subject-level summaries by grouping over:

```text
condition × modality × standard × subject
```

For the mean analysis:

```python
df.groupby([...]).mean()
```

For the SD analysis:

```python
df.groupby([...]).std()
```

Thus, each retained subject contributes one value per available:

```text
condition × modality × standard
```

cell.

For multi-session tags such as `behavses`, `imgses`, or `allses`, the current
script pools all selected sessions first and then computes the summary. It does
not first compute session-specific summaries and then average across sessions.

---

## Step 3 — SD-specific missing values

The SD analysis can introduce additional missing values.

For example, if a given subject has only one retained observation in a specific
condition, modality, and standard cell, the standard deviation may be undefined
and become `NaN`.

Those missing SD values are not imputed.

They are excluded later by the mixed-model function, which applies another
`dropna()` step to the model input.

---

## Step 4 — Mixed-model missing-data handling

Before fitting the linear mixed models, the script keeps only these columns:

```text
subject, condition, modality, standard, signed_asynchrony
```

It then removes rows with missing values in the model dataframe:

```python
mdf = df[cols].dropna().copy()
```

The mixed models are therefore fitted using all available complete rows in the
subject-level summary dataframe.

The script does not require every subject to have a fully complete
`Condition × Modality × Standard` grid before fitting the mixed model. The LMM
can use unbalanced data, provided that the relevant rows are present and
non-missing.

---

# Handling of outliers

The current `production_ancova.py` script does **not** perform explicit outlier
rejection.

In particular, it does not apply:

* an IQR rule,
* z-score thresholding,
* winsorization,
* trimming,
* subject-level exclusion based on extreme signed asynchrony,
* condition-level exclusion based on extreme means or SDs.

Therefore, any non-missing `signed_asynchrony` value that survives the upstream
production dataframe creation step contributes to the summaries and models.

If outlier rejection is required, it needs to be performed either:

* upstream, during construction of the production dataframe, or
* explicitly added to this ANCOVA script before the subject-level summaries are
  computed.

---

# Group-level plotting

For plotting, the script first computes subject-level summaries and then
averages across subjects by:

```text
condition × modality × standard
```

The plotted values are therefore group summaries of the subject-level mean or
SD signed asynchrony.

No extra missing-data imputation or outlier rejection is applied specifically
for plotting.

One practical implication is that the plot assumes that each condition and
modality has valid data across the standard values being plotted. If a condition
or modality is completely missing after filtering, plotting can fail or produce
empty curves.

---

# JASP wide-format files

The script also exports wide-format dataframes for JASP. These are created by
pivoting the subject-level summary dataframe with:

```text
index = subject × standard
columns = modality × condition
values = signed_asynchrony
```

If a subject-standard row lacks one or more modality-condition cells, the
corresponding wide-format cell remains missing.

The script does not exclude incomplete rows from the JASP files. Any additional
complete-case filtering for repeated-measures analyses in JASP would need to be
handled inside JASP or in a separate preprocessing step.

---

# Important conceptual consequence

Missing data are handled by deletion, not imputation.

The current production ANCOVA pipeline uses this sequence:

```text
remove rows with missing signed_asynchrony
→ compute subject-level mean/SD summaries
→ remove remaining missing model rows before LMM fitting
→ fit mixed models using available complete rows
```

Unlike the perception analysis, this production script does not estimate
psychometric parameters, does not apply fit-validity thresholds, and does not
perform explicit outlier rejection.

Therefore, the missing-data and outlier logic is simpler:

* missing trial/row values are excluded,
* undefined summary values are excluded from model fitting,
* outliers are not explicitly removed,
* the LMM analyses can use unbalanced subject-level data.

---

# Suggested reporting text

For the production ANCOVA analyses, rows with missing signed asynchrony values
were excluded before computing subject-level summaries. Mean and standard
deviation signed asynchrony were then computed for each subject, modality,
condition, and standard duration. Linear mixed-model ANCOVA-style analyses were
fit to the resulting subject-level summaries after removing remaining rows with
missing model values. No additional outlier rejection or imputation was applied
in the ANCOVA script.
