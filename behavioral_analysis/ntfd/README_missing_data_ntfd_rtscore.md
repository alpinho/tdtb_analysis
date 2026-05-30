# Missing-Data and Outlier Handling in `ntfd_rtscore.py`

## Overview

The NTFD reaction-time and score analysis reads precomputed NTFD
dataframes, applies batch-specific latency corrections, computes
subject-level summaries, and generates paired-condition boxplots for:

- reaction time,
- success score.

The analysis is performed separately for each batch and session tag. In the
cleaned version of the script, first- and second-batch inputs are controlled by
batch dictionaries, and reaction times are always latency-corrected.

---

# Input filtering

For each batch and session tag, the script reads:

```text
df_ntfd_<tag>.tsv
```

from the corresponding batch-specific `dataframes` folder.

The dataframe is then filtered in two steps:

```python
df_subfiltered = db[db['subject'].isin(subjects_dic[key])]
df = df_subfiltered[df_subfiltered['session'].isin(sessions_list)]
```

Therefore, even if the input dataframe already contains a selected subject
set, `ntfd_rtscore.py` filters it again using the subject list defined for the
current analysis tag.

This means that the analyzed sample is determined by:

1. the subjects included when the dataframe was originally created, and
2. the subjects retained by `ntfd_rtscore.py`.

The effective sample is the intersection of both.

---

# Batch-specific latency correction

Reaction times are always corrected in this script.

For the first batch, the script uses:

```python
audio_latency = 133
visual_latency = 35
button_press = 20
```

For the second batch, the script uses:

```python
audio_latency = 63
visual_latency = 35
button_press = 20
```

The correction is applied as:

```python
df.loc[df['modality'] == 'auditory', 'reaction_time'] -= (
    audio_latency + button_press)

df.loc[df['modality'] == 'visual', 'reaction_time'] -= (
    visual_latency + button_press)
```

So the dependent variable analyzed and plotted is the latency-corrected
reaction time.

---

# Missing-data handling

## Step 1 — Missing reaction times

Rows with missing reaction time are removed before any subject-level averaging:

```python
df = df.dropna(subset=['reaction_time'])
```

These rows usually correspond to trials with no valid response.

No reaction-time imputation is performed.

Missing reaction times are not replaced by:

- subject means,
- condition means,
- group means,
- medians,
- interpolated values,
- model-predicted values.

They are simply excluded before averaging.

---

## Step 2 — Dropping the answer column

After missing reaction times are removed, the response-key column is dropped:

```python
df = df.drop(columns=['answer'])
```

This does not remove additional rows. It only removes a column that is no
longer needed for the reaction-time and score summaries.

---

## Step 3 — Subject-level fixed-effects summaries

The function `ffx_dvar()` computes subject-level means after dropping the
session column:

```python
df_ffx = df.drop(['session'], axis=1)

df_ffx = df_ffx.groupby([
    'condition', 'modality', 'subject']).mean(
        numeric_only=True).reset_index()
```

This produces one value per:

```text
subject × modality × condition
```

for each numeric dependent variable, including:

- `reaction_time`,
- `score`.

For multi-session tags, this means that valid trials are averaged across the
included sessions after the filtering step.

---

## Step 4 — Minimum number of subjects

After subject-level averaging, the script checks the number of subjects:

```python
n_subjects = df_ffx['subject'].nunique()
```

If fewer than two subjects remain, plotting and paired tests are skipped:

```python
if n_subjects < 2:
    continue
```

This prevents paired t-tests from being attempted with an insufficient sample.

---

# Paired-test handling

The cleaned version aligns subjects before computing paired comparisons. This
is important because paired t-tests require that both arrays contain values
from the same subjects in the same order.

The intended paired comparisons are:

For reaction time:

```text
auditory beat vs auditory interval
visual beat vs visual interval
```

If the random condition is present, additional comparisons are run:

```text
auditory beat vs auditory random
auditory interval vs auditory random
visual beat vs visual random
visual interval vs visual random
```

The same comparison structure is applied to success scores.

If a subject is missing one member of a pair, that subject should not
contribute to that specific paired comparison.

---

# Random-condition handling

Older versions used a hard-coded list such as:

```python
NON_RANDOM_TAGS = ['ses-04', 'ses-05']
```

This meant that those session tags were forced to be treated as if they had no
random condition.

The cleaned version removes this hard-coded rule. Instead, whether the random
condition is included is determined from the dataframe itself.

Conceptually:

```text
if random condition exists in the analyzed dataframe:
    plot/test Beat, Interval, and Random
else:
    plot/test Beat and Interval only
```

This avoids mismatches between the script logic and the actual contents of the
data.

---

# Outlier handling

`ntfd_rtscore.py` does not perform explicit outlier rejection.

There is no IQR rule, z-score threshold, winsorization, trimming, or
participant-level exclusion based on extreme reaction times or scores in this
script.

Specifically, the script does not remove trials based on:

- very fast reaction times,
- very slow reaction times,
- reaction-time distance from the subject mean,
- reaction-time distance from the group mean,
- score outliers.

The only row-level exclusion applied in this script is removal of missing
reaction-time values.

If outlier rejection is needed, it would need to be added explicitly before
`ffx_dvar()` computes the subject-level means.

---

# Plotting and missing data

The plotting function receives subject-level values for each condition and
modality.

For the y-axis limits, the script uses only finite values:

```python
data = np.asarray(data, dtype=float)
data = data[np.isfinite(data)]
```

This affects only plotting limits. It does not impute or alter the data.

If no finite values are available, a default y-axis range is used.

---

# Summary

The NTFD reaction-time/score script handles missing data by exclusion, not
imputation.

In brief:

```text
raw dataframe
→ filter subjects according to current input dictionary
→ filter sessions according to current session tag
→ drop rows with missing reaction_time
→ apply latency correction
→ average valid rows within subject × modality × condition
→ run paired tests and plot group summaries
```

There is currently no explicit outlier rejection.

The most important consequences are:

- missing reaction-time trials do not contribute to the subject mean;
- subjects with incomplete paired data should be excluded pairwise from the
  relevant paired comparison;
- multi-session tags average available valid trials across the included
  sessions;
- random-condition handling is based on whether random trials are present in
  the analyzed dataframe;
- no missing values are replaced by imputed values.
