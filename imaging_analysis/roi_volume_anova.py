"""
Run ROI-wise ANOVAs for Music-SDTB.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: October 2024
Last update: May 2026

Compatibility: Python 3.10.14

Usage:
  python roi_anova_msdtb.py <n_rois> <encoding_type>
  <n_rois> in {2, 4, 6, 8, 10}
  <encoding_type> in {bothmod, auditory, visual}

Note: The encoding type pertain to the modality used to define the 
      ROIs. To define what tasks were used to define the ROIs, 
      please edit the variable `task_roidef_id` in Input section below.
"""

import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.stats import f as f_dist
from statsmodels.stats.anova import AnovaRM


# =========================== HELPERS =============================== #

def dataframe(rdata, hemis, tasks, contrasts, n_subj, outpath):
    """
    Build long dataframe from PSC array.

    Input array shape: (hemisphere, tasks, contrasts, subjects)
    Output columns:
      PSC, Subject, Contrast, Category, Modality, Task, Hemisphere
    """
    if isinstance(rdata, str):
        data = np.load(rdata)
    else:
        data = rdata

    subjects = [f"sub-{s:02d}" for s in n_subj]
    category = [c.split(" ", 1)[1] for c in contrasts]
    modality = [c.split(" ", 1)[0] for c in contrasts]

    subjects_col = np.tile(
        subjects, data.shape[2] * data.shape[1] * data.shape[0]
    )
    contrasts_rep = np.repeat(contrasts, len(subjects))
    contrasts_col = np.tile(contrasts_rep, data.shape[1] * data.shape[0])
    category_rep = np.repeat(category, len(subjects))
    category_col = np.tile(category_rep, data.shape[1] * data.shape[0])
    modality_rep = np.repeat(modality, len(subjects))
    modality_col = np.tile(modality_rep, data.shape[1] * data.shape[0])
    tasks_rep = np.repeat(tasks, len(modality_rep))
    tasks_col = np.tile(tasks_rep, data.shape[0])
    hem_col = np.repeat(hemis, len(tasks_rep))
    data_col = np.ravel(data)

    table = np.vstack((
        data_col, subjects_col, contrasts_col, category_col,
        modality_col, tasks_col, hem_col
    )).T

    df = pd.DataFrame(
        table,
        columns=[
            'PSC', 'Subject', 'Contrast', 'Category',
            'Modality', 'Task', 'Hemisphere'
        ]
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False, sep='\t')
    return df


def pval_label_converter(pvalues):
    """Convert p-values to star labels."""
    out = []
    for p in pvalues:
        if p <= .0001:
            out.append('****')
        elif p <= .001:
            out.append('***')
        elif p <= .01:
            out.append('**')
        elif p <= .05:
            out.append('*')
        else:
            out.append('ns')
    return out


def _assert_no_nan_psc(df, where=''):
    """Fail if PSC contains NaNs."""
    assert df['PSC'].notna().all(), f'NaN PSC values found {where}'


def _assert_unique_cells(df, keys, where=''):
    """Fail if repeated-measures cells are duplicated."""
    assert not df.duplicated(keys).any(), (
        f'Duplicate rows per RM cell {where} (keys={keys})'
    )


def _assert_expected_levels(df, col, expected, where=''):
    """Fail if factor levels are missing or unexpected."""
    got = set(df[col].unique())
    exp = set(expected)
    missing = exp - got
    extra = got - exp
    assert not missing, (
        f'Missing levels in {col} {where}: {sorted(missing)}'
    )
    assert not extra, (
        f'Unexpected levels in {col} {where}: {sorted(extra)}'
    )


def _assert_complete_within(df, subject, within_cols, where=''):
    """Fail if any subject lacks full within-factor coverage."""
    n_levels = [df[c].nunique() for c in within_cols]
    expected_cells = int(np.prod(n_levels))
    counts = df.groupby(subject)[within_cols].size()
    bad = counts[counts != expected_cells]
    assert bad.empty, (
        f'Incomplete within-subject cells {where}. '
        f'Expected {expected_cells} rows/subject, got {bad.to_dict()}'
    )


def _gg_epsilon_from_cov(cov: np.ndarray) -> float:
    """Compute Greenhouse-Geisser epsilon from a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        k x k covariance matrix of repeated-measures cells.

    Returns
    -------
    float
        Epsilon in [1/(k-1), 1]. For k < 3, returns 1.0.
    """
    cov = np.asarray(cov, dtype=float)
    k = cov.shape[0]
    if k < 3:
        return 1.0

    tr_s = float(np.trace(cov))
    tr_s2 = float(np.trace(cov @ cov))
    if (tr_s2 <= 0.0) or (not np.isfinite(tr_s)) or (not np.isfinite(tr_s2)):
        return 1.0

    eps = (tr_s ** 2) / ((k - 1) * tr_s2)
    lb = 1.0 / (k - 1)
    return float(np.clip(eps, lb, 1.0))


def _wide_cells(df: pd.DataFrame, subject_col: str, within_cols: list,
                dv_col: str, average_over: list | None = None) -> pd.DataFrame:
    """Build a subject x cell wide matrix for GG epsilon.

    If `average_over` is not None, the dependent variable is averaged
    over the remaining within-factors before pivoting.

    Rows (subjects) with any missing cells are dropped, since GG
    epsilon requires complete within-subject observations.
    """
    tmp = df.copy()
    if average_over:
        tmp = (
            tmp.groupby([subject_col] + within_cols, as_index=False)[dv_col]
            .mean()
        )

    wide = tmp.pivot_table(
        index=subject_col,
        columns=within_cols,
        values=dv_col,
        aggfunc='mean'
    )
    return wide.dropna(axis=0, how='any')


def _ng2_threeway_within(df: pd.DataFrame,
                         factors: list[str],
                         subject: str = 'Subject',
                         dv: str = 'PSC') -> dict[str, float]:
    """Compute generalized eta-squared (ng2) for 3-way within-subjects.

    Generalized eta-squared for an effect E is:
      ng2 = SS_E / (SS_E + SS_S + SS_{S×E})

    where SS_S is the subject (between-subject) sum of squares and
    SS_{S×E} is the subject-by-effect interaction sum of squares.
    This implementation assumes a complete, balanced within-subjects
    design with one observation per subject × cell (or averaged first).
    """
    if len(factors) != 3:
        raise ValueError('Expected exactly 3 within-subject factors.')

    wide = df.pivot_table(
        index=subject,
        columns=factors,
        values=dv,
        aggfunc='mean'
    ).dropna(axis=0, how='any')

    # Ensure deterministic ordering for reshape.
    wide = wide.sort_index(axis=1)

    levs = [list(wide.columns.levels[i]) for i in range(3)]
    la, lb, lc = [len(x) for x in levs]
    s = wide.shape[0]

    y = wide.to_numpy().reshape(s, la, lb, lc)

    gm = float(np.mean(y))

    # Subject mean (across all cells).
    ms = np.mean(y, axis=(1, 2, 3))  # (S,)
    ss_s = la * lb * lc * np.sum((ms - gm) ** 2)

    # Grand-averaged means (across subjects).
    m_abc = np.mean(y, axis=0)  # (A,B,C)
    m_a = np.mean(m_abc, axis=(1, 2))  # (A,)
    m_b = np.mean(m_abc, axis=(0, 2))  # (B,)
    m_c = np.mean(m_abc, axis=(0, 1))  # (C,)

    m_ab = np.mean(m_abc, axis=2)  # (A,B)
    m_ac = np.mean(m_abc, axis=1)  # (A,C)
    m_bc = np.mean(m_abc, axis=0)  # (B,C)

    # Effect SS (within-subjects).
    ss_a = s * lb * lc * np.sum((m_a - gm) ** 2)
    ss_b = s * la * lc * np.sum((m_b - gm) ** 2)
    ss_c = s * la * lb * np.sum((m_c - gm) ** 2)

    ab = m_ab - m_a[:, None] - m_b[None, :] + gm
    ac = m_ac - m_a[:, None] - m_c[None, :] + gm
    bc = m_bc - m_b[:, None] - m_c[None, :] + gm

    ss_ab = s * lc * np.sum(ab ** 2)
    ss_ac = s * lb * np.sum(ac ** 2)
    ss_bc = s * la * np.sum(bc ** 2)

    abc = (
        m_abc
        - m_ab[:, :, None]
        - m_ac[:, None, :]
        - m_bc[None, :, :]
        + m_a[:, None, None]
        + m_b[None, :, None]
        + m_c[None, None, :]
        - gm
    )
    ss_abc = s * np.sum(abc ** 2)

    # Subject-by-effect SS (error terms).
    m_sabc = y  # (S,A,B,C)
    m_sa = np.mean(y, axis=(2, 3))  # (S,A)
    m_sb = np.mean(y, axis=(1, 3))  # (S,B)
    m_sc = np.mean(y, axis=(1, 2))  # (S,C)

    m_sab = np.mean(y, axis=3)  # (S,A,B)
    m_sac = np.mean(y, axis=2)  # (S,A,C)
    m_sbc = np.mean(y, axis=1)  # (S,B,C)

    sa = m_sa - ms[:, None] - m_a[None, :] + gm
    sb = m_sb - ms[:, None] - m_b[None, :] + gm
    sc = m_sc - ms[:, None] - m_c[None, :] + gm

    ss_sa = lb * lc * np.sum(sa ** 2)
    ss_sb = la * lc * np.sum(sb ** 2)
    ss_sc = la * lb * np.sum(sc ** 2)

    sab = (
        m_sab
        - m_sa[:, :, None]
        - m_sb[:, None, :]
        - m_ab[None, :, :]
        + ms[:, None, None]
        + m_a[None, :, None]
        + m_b[None, None, :]
        - gm
    )
    sac = (
        m_sac
        - m_sa[:, :, None]
        - m_sc[:, None, :]
        - m_ac[None, :, :]
        + ms[:, None, None]
        + m_a[None, :, None]
        + m_c[None, None, :]
        - gm
    )
    sbc = (
        m_sbc
        - m_sb[:, :, None]
        - m_sc[:, None, :]
        - m_bc[None, :, :]
        + ms[:, None, None]
        + m_b[None, :, None]
        + m_c[None, None, :]
        - gm
    )

    ss_sab = lc * np.sum(sab ** 2)
    ss_sac = lb * np.sum(sac ** 2)
    ss_sbc = la * np.sum(sbc ** 2)

    sabc = (
        m_sabc
        - m_sab[:, :, :, None]
        - m_sac[:, :, None, :]
        - m_sbc[:, None, :, :]
        - m_abc[None, :, :, :]
        + m_sa[:, :, None, None]
        + m_sb[:, None, :, None]
        + m_sc[:, None, None, :]
        + m_ab[None, :, :, None]
        + m_ac[None, :, None, :]
        + m_bc[None, None, :, :]
        - ms[:, None, None, None]
        - m_a[None, :, None, None]
        - m_b[None, None, :, None]
        - m_c[None, None, None, :]
        + gm
    )
    ss_sabc = np.sum(sabc ** 2)

    fac_a, fac_b, fac_c = factors
    out = {
        fac_a: float(ss_a / (ss_a + ss_s + ss_sa)),
        fac_b: float(ss_b / (ss_b + ss_s + ss_sb)),
        fac_c: float(ss_c / (ss_c + ss_s + ss_sc)),
        f'{fac_a}:{fac_b}': float(ss_ab / (ss_ab + ss_s + ss_sab)),
        f'{fac_a}:{fac_c}': float(ss_ac / (ss_ac + ss_s + ss_sac)),
        f'{fac_b}:{fac_c}': float(ss_bc / (ss_bc + ss_s + ss_sbc)),
        f'{fac_a}:{fac_b}:{fac_c}': float(ss_abc / (ss_abc + ss_s + ss_sabc)),
    }
    return out


def preprocess_rand_ntfd_nonrandom(df, aggregation_type='mean'):
    """
    Aggregate 'Beat' and 'Interval' categories into a unified 'Non-Random' category
    for rand_ntfd_pairs analysis.

    This function relabels rows where 'Category' is 'Beat' or 'Interval' as
    'Non-Random', then aggregates the PSC (Percent Signal Change) values across
    the new category groupings using the specified aggregation method.

    Parameters
    ----------
    df : str or pd.DataFrame
        Input data. If a string, interpreted as a file path to a
        tab-separated CSV file which will be loaded automatically.
        If a DataFrame, used directly.
    aggregation_type : str, optional
        Aggregation function to apply to PSC values when grouping.
        Accepts any valid pandas aggregation string (e.g., 'sum', 'mean').
        Default is 'mean'.

    Returns
    -------
    pd.DataFrame
        A grouped DataFrame with one row per unique combination of
        ['Subject', 'Modality', 'Category', 'Task', 'Hemisphere']
        (plus 'ROI' if present), where 'Beat' and 'Interval' categories
        have been merged into 'Non-Random' and PSC values aggregated
        accordingly.

    Notes
    -----
    - The original DataFrame is not modified; a copy is made internally.
    - 'PSC' column is coerced to numeric; non-parseable values become NaN.
    - If the input DataFrame contains an 'ROI' column, it is included
      as an additional grouping key.

    Examples
    --------
    >>> result = preprocess_rand_ntfd_nonrandom(df, aggregation_type='mean')
    >>> result = preprocess_rand_ntfd_nonrandom('data/rand_ntfd.csv', aggregation_type='sum')
    """

    if isinstance(df, str):
        df = pd.read_csv(df, sep='	')

    db = df.copy()
    db['PSC'] = pd.to_numeric(db['PSC'], errors='coerce')

    nonrandom = db['Category'].isin(['Beat', 'Interval'])
    db.loc[nonrandom, 'Category'] = 'Non-Random'

    group_cols = ['Subject', 'Modality', 'Category', 'Task', 'Hemisphere']
    if 'ROI' in db.columns:
        group_cols.append('ROI')

    db = db.groupby(group_cols, as_index=False).agg({'PSC': aggregation_type})

    return db


# ================== ANOVAS FUNCTIONS PER ROI ======================= #

def oneway_rm_cat(df, tasks_dic, out_dir, prefix, roi,
                  hems=('lh', 'rh', 'bh'),
                  modalities=('Auditory', 'Visual')):
    """1-way RM-ANOVA on Category within Modality."""
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df.drop(columns=['Contrast'])
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')
    os.makedirs(out_dir, exist_ok=True)

    for ttag, task in zip(tasks_dic.keys(), tasks_dic.values()):
        for mod in modalities:
            for hem in hems:
                db = df[
                    (df.Task == task) &
                    (df.Modality == mod) &
                    (df.Hemisphere == hem)
                ].copy()

                anova = pg.rm_anova(
                    data=db, dv='PSC', within='Category',
                    subject='Subject', detailed=True, effsize='ng2'
                )
                post = pg.pairwise_tests(
                    data=db, dv='PSC', within='Category',
                    subject='Subject', return_desc=True,
                    padjust='holm', effsize='cohen'
                )

                base = f"{prefix}_{roi}_{hem}_1w-{ttag}_{mod.lower()}_"
                anova.to_csv(
                    os.path.join(out_dir, base + 'anova.tsv'),
                    sep='\t', index=False
                )
                post.to_csv(
                    os.path.join(out_dir, base + 'posthoc.tsv'),
                    sep='\t', index=False
                )


def twoway_rm_modcat(df, tasks_dic, out_dir, prefix, roi,
                     alternative='two-sided',
                     hems=('lh', 'rh', 'bh')):
    """2-way RM-ANOVA per task: Modality x Category."""
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    if 'Contrast' in df.columns:
        df = df.drop(columns=['Contrast'])
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')
    os.makedirs(out_dir, exist_ok=True)

    for ttag, task in zip(tasks_dic.keys(), tasks_dic.values()):
        for hem in hems:
            db = df[(df.Task == task) & (df.Hemisphere == hem)].copy()

            _assert_no_nan_psc(db, where=f'2w_task {task} {hem}')
            _assert_unique_cells(
                db, ['Subject', 'Modality', 'Category'],
                where=f'2w_task {task} {hem}'
            )
            _assert_complete_within(
                db, 'Subject', ['Modality', 'Category'],
                where=f'2w_task {task} {hem}'
            )

            anova = pg.rm_anova(
                data=db, dv='PSC', within=['Modality', 'Category'],
                subject='Subject', detailed=True, effsize='ng2'
            )
            post1 = pg.pairwise_tests(
                data=db, dv='PSC', within=['Modality', 'Category'],
                subject='Subject', alternative=alternative,
                return_desc=True, padjust='holm',
                effsize='cohen'
            )
            post2 = pg.pairwise_tests(
                data=db, dv='PSC', within=['Category', 'Modality'],
                subject='Subject', alternative=alternative,
                return_desc=True, padjust='holm',
                effsize='cohen'
            )

            base = f"{prefix}_{roi}_{hem}_2w-{ttag}_"
            anova.to_csv(
                os.path.join(out_dir, base + 'anova.tsv'),
                sep='\t', index=False
            )
            post1.to_csv(
                os.path.join(out_dir, base + 'modcat_posthoc.tsv'),
                sep='\t', index=False
            )
            post2.to_csv(
                os.path.join(out_dir, base + 'catmod_posthoc.tsv'),
                sep='\t', index=False
            )


def twoway_rm_modcat_taskavg(df, out_dir, prefix, roi,
                             hems=('lh', 'rh', 'bh')):
    """
    2-way RM-ANOVA across tasks: Modality x Category.

    It takes all tasks (but allmain_tasks) and average them.
    This was to check whether it delivers the same results as
    allmain_tasks in 'twoway_rm_modcat' function,
    where we do the ffx of all tasks and extract the psc's (and it does).
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df[df.Task != 'All Tasks'].copy()
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')
    os.makedirs(out_dir, exist_ok=True)

    for hem in hems:
        db = df[df.Hemisphere == hem].copy()

        exp_n_tasks = db['Task'].nunique()
        counts = db.groupby(
            ['Subject', 'Category', 'Modality']
        )['Task'].nunique()
        assert (counts == exp_n_tasks).all(), (
            'Unequal task coverage before task-averaging in '
            'twoway_rmanova_gtasks.'
        )

        db = db.drop(columns=['Hemisphere', 'Task', 'Contrast'])
        db = db.groupby(
            ['Category', 'Modality', 'Subject'],
            as_index=False
        ).agg({'PSC': 'mean'})

        _assert_no_nan_psc(db, where=f'2w_gtasks {hem}')
        _assert_unique_cells(
            db, ['Subject', 'Category', 'Modality'],
            where=f'2w_gtasks {hem}'
        )
        _assert_complete_within(
            db, 'Subject', ['Category', 'Modality'],
            where=f'2w_gtasks {hem}'
        )

        anova = pg.rm_anova(
            data=db, dv='PSC', within=['Modality', 'Category'],
            subject='Subject', detailed=True, effsize='ng2'
        )
        post = pg.pairwise_tests(
            data=db, dv='PSC', within=['Category', 'Modality'],
            subject='Subject', return_desc=True,
            padjust='holm', effsize='cohen'
        )

        base = f"{prefix}_{roi}_{hem}_2w-taskavg_"
        anova.to_csv(
            os.path.join(out_dir, base + 'anova.tsv'),
            sep='\t', index=False
        )
        post.to_csv(
            os.path.join(out_dir, base + 'posthoc.tsv'),
            sep='\t', index=False
        )


def twoway_rm_modtask(
    df,
    out_dir,
    prefix,
    roi,
    alternative="two-sided",
    hems=("lh", "rh", "bh"),
):
    """
    2-way RM-ANOVA PER ROI: Modality x Task.

    - Drops "All Tasks"
    - If Category exists, averages PSC across Category first
      so the design is Subject x Modality x Task (per hemisphere).
    - Writes ANOVA + posthoc TSVs per hemisphere.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t")

    if "Contrast" in df.columns:
        df = df.drop(columns=["Contrast"])

    df["PSC"] = pd.to_numeric(df["PSC"], errors="coerce")

    df = df[df.Task != "All Tasks"].copy()

    if "Category" in df.columns:
        df = (
            df.groupby(
                ["Subject", "Task", "Modality", "Hemisphere"],
                as_index=False,
            )
            .agg({"PSC": "mean"})
        )

    expected_tasks = ["Production", "Perception", "NTFD"]
    if "NTFD Random" in df["Task"].unique():
        expected_tasks.append("NTFD Random")

    os.makedirs(out_dir, exist_ok=True)

    for hem in hems:
        db = df[df.Hemisphere == hem].copy()

        _assert_no_nan_psc(db, where=f"2w_modtask {roi} {hem}")
        _assert_unique_cells(
            db,
            ["Subject", "Task", "Modality", "Hemisphere"],
            where=f"2w_modtask {roi} {hem}",
        )
        _assert_expected_levels(
            db,
            "Task",
            expected_tasks,
            where=f"2w_modtask {roi} {hem}",
        )
        _assert_expected_levels(
            db,
            "Modality",
            ["Auditory", "Visual"],
            where=f"2w_modtask {roi} {hem}",
        )
        _assert_complete_within(
            db,
            "Subject",
            ["Task", "Modality"],
            where=f"2w_modtask {roi} {hem}",
        )

        anova = pg.rm_anova(
            data=db,
            dv="PSC",
            within=["Task", "Modality"],
            subject="Subject",
            detailed=True,
            effsize="ng2",
        )

        post1 = pg.pairwise_tests(
            data=db,
            dv="PSC",
            within=["Modality", "Task"],
            subject="Subject",
            alternative=alternative,
            return_desc=True,
            padjust="holm",
            effsize="cohen",
        )

        post2 = pg.pairwise_tests(
            data=db,
            dv="PSC",
            within=["Task", "Modality"],
            subject="Subject",
            alternative=alternative,
            return_desc=True,
            padjust="holm",
            effsize="cohen",
        )

        base = f"{prefix}_{roi}_{hem}_2w-"
        anova.to_csv(
            os.path.join(out_dir, base + "modtask_anova.tsv"),
            sep="\t",
            index=False,
        )
        post1.to_csv(
            os.path.join(out_dir, base + "modtask_posthoc.tsv"),
            sep="\t",
            index=False,
        )
        post2.to_csv(
            os.path.join(out_dir, base + "modtask_posthoc.tsv"),
            sep="\t",
            index=False,
        )


# ================= ANOVAS FUNCTIONS ACROSS ROIS ==================== #

def twoway_rm_roicat(df, tasks_dic, out_dir, prefix,
                     alternative='two-sided',
                     modality=None,
                     hems=('lh', 'rh', 'bh')):
    """2-way RM-ANOVA per task: ROI x Category."""
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df.drop(columns=['Contrast'])
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')

    if modality is None:
        if 'Modality' in df.columns:
            df = df.drop(columns=['Modality'])
            df = df.groupby(
                ['Category', 'Task', 'Subject', 'ROI', 'Hemisphere'],
                as_index=False
            ).agg({'PSC': 'mean'})
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory'].drop(columns=['Modality'])
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual'].drop(columns=['Modality'])

    os.makedirs(out_dir, exist_ok=True)

    for ttag, task in zip(tasks_dic.keys(), tasks_dic.values()):
        for hem in hems:
            db = df[
                (df.Task == task) & (df.Hemisphere == hem)
            ].copy()

            _assert_no_nan_psc(db, where=f'catroi {task} {hem}')
            _assert_unique_cells(
                db, ['Subject', 'ROI', 'Category', 'Task', 'Hemisphere'],
                where=f'catroi {task} {hem}'
            )
            _assert_complete_within(
                db, 'Subject', ['ROI', 'Category'],
                where=f'catroi {task} {hem}'
            )

            anova = pg.rm_anova(
                data=db, dv='PSC', within=['ROI', 'Category'],
                subject='Subject', detailed=True, effsize='ng2'
            )
            post = pg.pairwise_tests(
                data=db, dv='PSC', within=['ROI', 'Category'],
                subject='Subject', alternative=alternative,
                return_desc=True, padjust='holm', effsize='cohen'
            )

            base = f"{prefix}_{hem}_2w-{ttag}_"
            anova.to_csv(
                os.path.join(out_dir, base + 'anova.tsv'),
                sep='\t', index=False
            )
            post.to_csv(
                os.path.join(out_dir, base + 'posthoc.tsv'),
                sep='\t', index=False
            )


def twoway_rm_roitask(df, out_dir, prefix,
                      alternative='two-sided',
                      modality=None,
                      hems=('lh', 'rh', 'bh')):
    """2-way RM-ANOVA: ROI x Task (Category dropped)."""
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df.drop(columns=['Contrast'])
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')

    if modality is None:
        if 'Modality' in df.columns:
            df = df.drop(columns=['Modality'])
            df = df.groupby(
                ['Category', 'Task', 'Subject', 'ROI', 'Hemisphere'],
                as_index=False
            ).agg({'PSC': 'mean'})
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory'].drop(columns=['Modality'])
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual'].drop(columns=['Modality'])

    if 'Category' in df.columns:
        df = df.groupby(
            ['Task', 'Subject', 'ROI', 'Hemisphere'],
            as_index=False
        ).agg({'PSC': 'mean'})

    df = df[df.Task != 'All Tasks'].copy()
    os.makedirs(out_dir, exist_ok=True)

    for hem in hems:
        db = df[df.Hemisphere == hem].copy()
        _assert_no_nan_psc(db, where=f'timingroi {hem}')
        _assert_unique_cells(
            db, ['Subject', 'ROI', 'Task', 'Hemisphere'],
            where=f'timingroi {hem}'
        )
        _assert_expected_levels(
            db, 'Task', ['Production', 'Perception', 'NTFD'],
            where=f'timingroi {hem}'
        )
        _assert_complete_within(
            db, 'Subject', ['ROI', 'Task'],
            where=f'timingroi {hem}'
        )

        anova = pg.rm_anova(
            data=db, dv='PSC', within=['ROI', 'Task'],
            subject='Subject', detailed=True, effsize='ng2'
        )
        post = pg.pairwise_tests(
            data=db, dv='PSC', within=['ROI', 'Task'],
            subject='Subject', alternative=alternative,
            return_desc=True, padjust='holm', effsize='cohen'
        )

        base = f"{prefix}_{hem}_2w_"
        anova.to_csv(
            os.path.join(out_dir, base + 'anova.tsv'),
            sep='\t', index=False
        )
        post.to_csv(
            os.path.join(out_dir, base + 'posthoc.tsv'),
            sep='\t', index=False
        )


def threeway_rm_catmodtask(df, out_dir, prefix, roi, hems=('lh', 'rh', 'bh')):
    """3-way RM-ANOVA: Category x Modality x Task."""
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df[df.Task != 'All Tasks'].copy()
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')
    os.makedirs(out_dir, exist_ok=True)

    for hem in hems:
        db = df[df.Hemisphere == hem].copy()

        _assert_no_nan_psc(db, where=f'3way {hem}')
        _assert_unique_cells(
            db, ['Subject', 'Category', 'Modality', 'Task'],
            where=f'3way {hem}'
        )
        _assert_complete_within(
            db, 'Subject', ['Category', 'Modality', 'Task'],
            where=f'3way {hem}'
        )

        model = AnovaRM(
            data=db, depvar='PSC', subject='Subject',
            within=['Category', 'Modality', 'Task']
        )
        res = model.fit()

        aov = res.anova_table.copy()

        effects = {
            'Category': (['Category'], ['Modality', 'Task']),
            'Modality': (['Modality'], ['Category', 'Task']),
            'Task': (['Task'], ['Category', 'Modality']),
            'Category:Modality': (['Category', 'Modality'], ['Task']),
            'Category:Task': (['Category', 'Task'], ['Modality']),
            'Modality:Task': (['Modality', 'Task'], ['Category']),
            'Category:Modality:Task': (['Category', 'Modality', 'Task'], None),
        }

        eps_vals = []
        pgg_vals = []

        for eff in aov.index:
            eff_name = str(eff)
            if eff_name not in effects:
                eps_vals.append(np.nan)
                pgg_vals.append(np.nan)
                continue

            within_cols, avg_over = effects[eff_name]
            wide = _wide_cells(
                db, subject_col='Subject', within_cols=within_cols,
                dv_col='PSC', average_over=avg_over
            )
            cov = np.cov(wide.values, rowvar=False, ddof=1)
            eps = _gg_epsilon_from_cov(cov)

            f_val = float(aov.loc[eff, 'F Value'])
            df1 = float(aov.loc[eff, 'Num DF'])
            df2 = float(aov.loc[eff, 'Den DF'])
            p_gg = float(f_dist.sf(f_val, eps * df1, eps * df2))

            eps_vals.append(eps)
            pgg_vals.append(p_gg)

        aov['eps_GG'] = eps_vals
        aov['p_GG'] = pgg_vals

        ng2_map = _ng2_threeway_within(
            db, factors=['Category', 'Modality', 'Task'],
            subject='Subject', dv='PSC'
        )
        aov['ng2'] = [ng2_map.get(str(eff), np.nan) for eff in aov.index]

        ph_cat = pg.pairwise_tests(
            data=db, dv='PSC', within='Category', subject='Subject',
            padjust='holm', effsize='cohen', return_desc=True
        )
        ph_mod = pg.pairwise_tests(
            data=db, dv='PSC', within='Modality', subject='Subject',
            padjust='holm', effsize='cohen', return_desc=True
        )
        ph_task = pg.pairwise_tests(
            data=db, dv='PSC', within='Task', subject='Subject',
            padjust='holm', effsize='cohen', return_desc=True
        )

        base = f"{prefix}_{roi}_{hem}_3w_"
        aov.to_csv(
            os.path.join(out_dir, base + 'anova.tsv'), sep='\t'
        )
        ph_cat.to_csv(
            os.path.join(out_dir, base + 'posthoc_category.tsv'),
            index=False, header=False, sep='\t'
        )
        ph_mod.to_csv(
            os.path.join(out_dir, base + 'posthoc_modality.tsv'),
            index=False, header=False, sep='\t'
        )
        ph_task.to_csv(
            os.path.join(out_dir, base + 'posthoc_task.tsv'),
            index=False, header=False, sep='\t'
        )


def threeway_rm_roimodtask(df, output_dir, prefix, hems=['lh', 'rh', 'bh']):
    """
    3-way RM-ANOVA (ROI x Modality x Task) via statsmodels.AnovaRM,
    then Holm-corrected paired t-tests:
     • mains: ROI, Task, Modality
     • ROI x Modality: only Aud vs Vis within each ROI
     • ROI x Task:     only each Task-pair within each ROI
     • Modality x Task: only each Task-pair within each Modality
     • 3-way: only each Task-pair within each (ROI, Modality) cell
    All posthocs in one TSV per hemisphere, with the same columns
    as in the 2-way posthoc files.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # drop “All Tasks” and coerce
    df = df.loc[df.Task != 'All Tasks'].copy()
    df['PSC'] = pd.to_numeric(df['PSC'])

    for hem in hems:
        sub = df.loc[df.Hemisphere == hem]
        agg = (
            sub
            .groupby(['Subject', 'ROI', 'Modality', 'Task'], as_index=False)
            ['PSC']
            .mean()
        )

        _assert_no_nan_psc(agg, where=f'3way_timing {hem}')
        _assert_unique_cells(
            agg, ['Subject', 'ROI', 'Modality', 'Task'],
            where=f'3way_timing {hem}'
        )
        _assert_complete_within(
            agg, 'Subject', ['ROI', 'Modality', 'Task'],
            where=f'3way_timing {hem}'
        )

        # 1) omnibus 3-way ANOVA
        model = AnovaRM(
            agg, depvar='PSC', subject='Subject',
            within=['ROI', 'Modality', 'Task']
        )
        res3 = model.fit()

        aov = res3.anova_table.copy()

        effects = {
            'ROI': (['ROI'], ['Modality', 'Task']),
            'Modality': (['Modality'], ['ROI', 'Task']),
            'Task': (['Task'], ['ROI', 'Modality']),
            'ROI:Modality': (['ROI', 'Modality'], ['Task']),
            'ROI:Task': (['ROI', 'Task'], ['Modality']),
            'Modality:Task': (['Modality', 'Task'], ['ROI']),
            'ROI:Modality:Task': (['ROI', 'Modality', 'Task'], None),
        }

        eps_vals = []
        pgg_vals = []

        for eff in aov.index:
            eff_name = str(eff)
            if eff_name not in effects:
                eps_vals.append(np.nan)
                pgg_vals.append(np.nan)
                continue

            within_cols, avg_over = effects[eff_name]
            wide = _wide_cells(
                agg, subject_col='Subject', within_cols=within_cols,
                dv_col='PSC', average_over=avg_over
            )
            cov = np.cov(wide.values, rowvar=False, ddof=1)
            eps = _gg_epsilon_from_cov(cov)

            f_val = float(aov.loc[eff, 'F Value'])
            df1 = float(aov.loc[eff, 'Num DF'])
            df2 = float(aov.loc[eff, 'Den DF'])
            p_gg = float(f_dist.sf(f_val, eps * df1, eps * df2))

            eps_vals.append(eps)
            pgg_vals.append(p_gg)

        aov['eps_GG'] = eps_vals
        aov['p_GG'] = pgg_vals

        ng2_map = _ng2_threeway_within(
            agg, factors=['ROI', 'Modality', 'Task'],
            subject='Subject', dv='PSC'
        )
        aov['ng2'] = [ng2_map.get(str(eff), np.nan) for eff in aov.index]

        os.makedirs(output_dir, exist_ok=True)
        base = f"{prefix}_{hem}_3way"

        # save ANOVA
        aov.to_csv(
            os.path.join(output_dir, base + '_anova.tsv'),
            sep='\t'
        )

        # 2) post-hocs
        rows = []

        # — mains —
        for factor in ['ROI', 'Modality', 'Task']:
            ph = pg.pairwise_tests(
                data=agg, dv='PSC', within=factor, subject='Subject',
                padjust='holm', effsize='cohen', return_desc=True
            )
            if 'Contrast' in ph.columns:
                ph['Contrast'] = factor
            else:
                ph.insert(0, 'Contrast', factor)
            rows.append(ph)

        # — ROI × Modality (Aud vs Vis within each ROI) —
        for roi in agg['ROI'].unique():
            sub_roi = agg.loc[agg.ROI == roi]
            ph = pg.pairwise_tests(
                data=sub_roi, dv='PSC', within='Modality',
                subject='Subject', padjust='holm', effsize='cohen',
                return_desc=True
            )
            if 'Contrast' in ph.columns:
                ph['Contrast'] = 'ROI:Modality'
            else:
                ph.insert(0, 'Contrast', 'ROI:Modality')
            ph.insert(1, 'ROI', roi)
            rows.append(ph)

        # — ROI × Task (all 3 Task pairs within each ROI) —
        for roi in agg['ROI'].unique():
            sub_roi = agg.loc[agg.ROI == roi]
            ph = pg.pairwise_tests(
                data=sub_roi, dv='PSC', within='Task',
                subject='Subject', padjust='holm', effsize='cohen',
                return_desc=True
            )
            if 'Contrast' in ph.columns:
                ph['Contrast'] = 'ROI:Task'
            else:
                ph.insert(0, 'Contrast', 'ROI:Task')
            ph.insert(1, 'ROI', roi)
            rows.append(ph)

        # — Modality × Task (all 3 Task pairs within each Modality) —
        for mod in agg['Modality'].unique():
            sub_mod = agg.loc[agg.Modality == mod]
            ph = pg.pairwise_tests(
                data=sub_mod, dv='PSC', within='Task',
                subject='Subject', padjust='holm', effsize='cohen',
                return_desc=True
            )
            if 'Contrast' in ph.columns:
                ph['Contrast'] = 'Modality:Task'
            else:
                ph.insert(0, 'Contrast', 'Modality:Task')
            ph.insert(1, 'Modality', mod)
            rows.append(ph)

        # — 3-way ROI × Modality × Task —
        #    (only Task-pairs within each (ROI,Modality) cell)
        for roi in agg['ROI'].unique():
            for mod in agg['Modality'].unique():
                sub_cell = agg[(agg.ROI == roi) & (agg.Modality == mod)]
                ph = pg.pairwise_tests(
                    data=sub_cell, dv='PSC', within='Task',
                    subject='Subject', padjust='holm', effsize='cohen',
                    return_desc=True
                )
                if 'Contrast' in ph.columns:
                    ph['Contrast'] = 'ROI:Modality:Task'
                else:
                    ph.insert(0, 'Contrast', 'ROI:Modality:Task')
                ph.insert(1, 'ROI', roi)
                ph.insert(2, 'Modality', mod)
                rows.append(ph)

        # concat & save
        posthoc_all = pd.concat(rows, ignore_index=True, sort=False)

        # **Reorder columns:** Contrast, ROI, Modality, then the rest
        cols = list(posthoc_all.columns)
        front = ['Contrast', 'ROI', 'Modality']
        rest = [c for c in cols if c not in front]
        posthoc_all = posthoc_all[front + rest]

        posthoc_all.to_csv(
            os.path.join(output_dir, base + '_posthoc.tsv'),
            sep='\t', index=False
        )


# =========================== INPUTS ================================ #

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
    29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
]

# Which task's encoding maps defined the ROIs used for extraction?
# Valid options ONLY: 'allmain_tasks' or 'rand_ntfd'.
# Note: the second sys argument when running the script from command 
# line is used to set the sensory modality used to definied the ROIs.
task_roidef_id = 'allmain_tasks'  # or 'rand_ntfd'

# What task(s) used for the roi extraction and what ANOVA to do?
# 'main_tasks' | 'rand_ntfd_pairs' | 'rand_ntfd_nonrandom' |
# 'all_tasks'
# Modes via `folder_name`:
#   • 'main_tasks'
#   • 'rand_ntfd_pairs'      -> Category: Beat, Interval, Random
#   • 'rand_ntfd_nonrandom'  -> Category: Non-Random, Random
#   • 'all_tasks'            -> main_tasks + rand_ntfd_pairs
folder_name = 'rand_ntfd_nonrandom'

tags = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a',
    'a4g', 'a3g', 'a2g', 'a1g', 'g'
]
weights_list = [
    (1., 0.), (.9, .1), (.8, .2), (.7, .3), (.6, .4),
    (.5, .5),
    (.4, .6), (.3, .7), (.2, .8), (.1, .9), (0., 1.)
]

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')
hmat_dir = os.path.join(atlases_dir, 'hmat_atlas')

model = 'rwls'
masking = 'wb'
hrf_cutoff = 'hrf128'

roi_dir = os.path.join(
    working_dir,
    'roi_analyses_' + model + '_' + hrf_cutoff + '_' + masking +
    '_puncorr_unsmoothed'
)

# ----------------- Full contrast dictionaries (kept) --------------- #

ALL_CONTRASTS_MAIN = {
    1: 'Encoding',
    2: 'Auditory Encoding',
    3: 'Visual Encoding',
    4: 'Auditory vs Visual Encoding',
    5: 'Visual vs Auditory Encoding',
    6: 'Beat',
    7: 'Interval',
    8: 'Beat vs Interval',
    9: 'Interval vs Beat',
    10: 'Auditory Beat',
    11: 'Auditory Interval',
    12: 'Auditory Beat vs Auditory Interval',
    13: 'Auditory Interval vs Auditory Beat',
    14: 'Visual Beat',
    15: 'Visual Interval',
    16: 'Visual Beat vs Visual Interval',
    17: 'Visual Interval vs Visual Beat',
    18: 'Decision'
}

ALL_CONTRASTS_RAND = {
    1: 'Encoding',
    2: 'Auditory Encoding',
    3: 'Visual Encoding',
    4: 'Auditory vs Visual Encoding',
    5: 'Visual vs Auditory Encoding',
    6: 'Beat',
    7: 'Interval',
    8: 'Non-Random',
    9: 'Random',
    10: 'Beat vs Interval',
    11: 'Interval vs Beat',
    12: 'Beat vs Random',
    13: 'Random vs Beat',
    14: 'Interval vs Random',
    15: 'Random vs Interval',
    16: 'Non-Random vs Random',
    17: 'Random vs Non-Random',
    18: 'Auditory Beat',
    19: 'Auditory Interval',
    20: 'Auditory Non-Random',
    21: 'Auditory Random',
    22: 'Auditory Beat vs Auditory Interval',
    23: 'Auditory Interval vs Auditory Beat',
    24: 'Auditory Beat vs Auditory Random',
    25: 'Auditory Random vs Auditory Beat',
    26: 'Auditory Interval vs Auditory Random',
    27: 'Auditory Random vs Auditory Interval',
    28: 'Auditory Non-Random vs Auditory Random',
    29: 'Auditory Random vs Auditory Non-Random',
    30: 'Visual Beat',
    31: 'Visual Interval',
    32: 'Visual Non-Random',
    33: 'Visual Random',
    34: 'Visual Beat vs Visual Interval',
    35: 'Visual Interval vs Visual Beat',
    36: 'Visual Beat vs Visual Random',
    37: 'Visual Random vs Visual Beat',
    38: 'Visual Interval vs Visual Random',
    39: 'Visual Random vs Visual Interval',
    40: 'Visual Non-Random vs Visual Random',
    41: 'Visual Random vs Visual Non-Random',
    42: 'Decision'
}

# ------------------ Tasks and selected_contrasts ------------------- #

if folder_name == 'main_tasks':
    tasks = {
        'prod': 'Production',
        'percep': 'Perception',
        'ntfd': 'NTFD',
        'allmain_tasks': 'All Tasks'
    }
    selected_contrasts = {
        10: 'Auditory Beat',
        11: 'Auditory Interval',
        14: 'Visual Beat',
        15: 'Visual Interval'
    }
elif folder_name == 'rand_ntfd_pairs':
    tasks = {'rand_ntfd': 'NTFD Random'}
    selected_contrasts = {
        18: 'Auditory Beat',
        19: 'Auditory Interval',
        21: 'Auditory Random',
        30: 'Visual Beat',
        31: 'Visual Interval',
        33: 'Visual Random'
    }
elif folder_name == 'all_tasks':
    tasks = {
        'prod': 'Production',
        'percep': 'Perception',
        'ntfd': 'NTFD',
        'rand_ntfd': 'NTFD Random'
    }
    selected_contrasts = None
else:
    assert folder_name == 'rand_ntfd_nonrandom'
    tasks = {'rand_ntfd': 'NTFD Random'}
    selected_contrasts = {
        20: 'Auditory Non-Random',
        21: 'Auditory Random',
        32: 'Visual Non-Random',
        33: 'Visual Random'
    }

# ###### ROI sets ######
atlas_dirnames10 = [
    fsl_dir, ntk_dir, ntk_dir, ntk_dir, hmat_dir, hmat_dir, hmat_dir,
    hmat_dir, fsl_dir, fsl_dir
]
atlas_names10 = [
    'hos', 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
    'hmat', 'hmat', 'hmat', 'hmat', 'hos', 'hos'
]
region_names10 = [
    'dorsal_striatum', 'cerebellum', 'cerebellum', 'cerebellum',
    'motor_area', 'motor_area', 'motor_area', 'motor_area',
    'heschl_gyrus', 'occipital_lobe'
]
roi_names10 = [
    'dstr', 'cereb-s', 'cereb-i', 'cereb',
    'pmd', 'pmv', 'sma', 'presma',
    'heschl', 'occipital'
]

# #######################

atlas_dirnames8 = [
    fsl_dir, ntk_dir, hmat_dir, hmat_dir, hmat_dir, hmat_dir,
    fsl_dir, fsl_dir
]
atlas_names8 = [
    'hos', 'ntk_symmni128', 'hmat', 'hmat', 'hmat', 'hmat',
    'hos', 'hos'
]
region_names8 = [
    'dorsal_striatum', 'cerebellum',
    'motor_area', 'motor_area', 'motor_area', 'motor_area',
    'heschl_gyrus', 'occipital_lobe'
]
roi_names8 = [
    'dstr', 'cereb', 'pmd', 'pmv', 'sma', 'presma',
    'heschl', 'occipital'
]

# #######################

atlas_dirnames6 = [
    fsl_dir, ntk_dir, hmat_dir, hmat_dir, hmat_dir, hmat_dir,
    fsl_dir, fsl_dir
]
atlas_names6 = [
    'hmat', 'hmat', 'hmat', 'hmat', 'hos', 'hos'
]
region_names6 = [
    'motor_area', 'motor_area', 'motor_area', 'motor_area',
    'heschl_gyrus', 'occipital_lobe'
]
roi_names6 = [
    'pmd', 'pmv', 'sma', 'presma',
    'heschl', 'occipital'
]

# #######################

atlas_dirnames4 = [hmat_dir, hmat_dir, hmat_dir, hmat_dir]
atlas_names4 = ['hmat', 'hmat', 'hmat', 'hmat']
region_names4 = ['motor_area', 'motor_area', 'motor_area', 'motor_area']
roi_names4 = ['pmd', 'pmv', 'sma', 'presma']

# #######################

atlas_dirnames3 = [fsl_dir, ntk_dir, hmat_dir]
atlas_names3 = ['hos', 'ntk_symmni128', 'hmat']
region_names3 = ['dorsal_striatum', 'cerebellum', 'motor_area']
roi_names3 = ['dstr', 'cereb', 'sma']

# #######################

# atlas_dirnames2 = [fsl_dir, ntk_dir]
# atlas_names2 = ['hos', 'ntk_symmni128']
# region_names2 = ['dorsal_striatum', 'cerebellum']
# roi_names2 = ['dstr', 'cereb']

# atlas_dirnames2 = [fsl_dir, fsl_dir]
# atlas_names2 = ['hos', 'hos']
# region_names2 = ['heschl_gyrus', 'occipital_lobe']
# roi_names2 = ['heschl', 'occipital']

# atlas_dirnames2 = [hmat_dir, hmat_dir]
# atlas_names2 = ['hmat', 'hmat']
# region_names2 = ['motor_area', 'motor_area']
# roi_names2 = ['presma', 'sma']

# atlas_dirnames2 = [fsl_dir, hmat_dir]
# atlas_names2 = ['hos', 'hmat']
# region_names2 = ['dorsal_striatum', 'motor_area']
# roi_names2 = ['dstr', 'sma']

atlas_dirnames2 = [ntk_dir, hmat_dir]
atlas_names2 = ['ntk_symmni128', 'hmat']
region_names2 = ['cerebellum', 'motor_area']
roi_names2 = ['cereb', 'sma']


# ============================= RUN ================================= #

if __name__ == '__main__':

    assert len(sys.argv) > 2, (
        "Usage: python roi_anova_msdtb.py <n_rois> <encoding_type>\n"
        "  n_rois: 2|4|8|10\n"
        "  encoding_type: bothmod|auditory|visual"
    )

    n_rois = int(sys.argv[1])
    if n_rois == 10:
        atlas_dirnames = atlas_dirnames10
        atlas_names = atlas_names10
        region_names = region_names10
        roi_names = roi_names10
    elif n_rois == 8:
        atlas_dirnames = atlas_dirnames8
        atlas_names = atlas_names8
        region_names = region_names8
        roi_names = roi_names8
    elif n_rois == 6:
        atlas_dirnames = atlas_dirnames6
        atlas_names = atlas_names6
        region_names = region_names6
        roi_names = roi_names6
    elif n_rois == 4:
        atlas_dirnames = atlas_dirnames4
        atlas_names = atlas_names4
        region_names = region_names4
        roi_names = roi_names4
    elif n_rois == 3:
        atlas_dirnames = atlas_dirnames3
        atlas_names = atlas_names3
        region_names = region_names3
        roi_names = roi_names3
    elif n_rois == 2:
        atlas_dirnames = atlas_dirnames2
        atlas_names = atlas_names2
        region_names = region_names2
        roi_names = roi_names2
    else:
        raise ValueError("n_rois must be one of {2, 3, 4, 6, 8, 10}.")

    encoding_type = sys.argv[2]
    msdtb_dir = os.path.join(
        roi_dir, f"{encoding_type}_{task_roidef_id}", folder_name
    )

    if folder_name != 'all_tasks':
        keys = list(selected_contrasts.keys())
        if encoding_type == 'bothmod':
            filtered_contrasts = selected_contrasts
        elif encoding_type == 'auditory':
            filtered_contrasts = {
                k: v for k, v in selected_contrasts.items()
                if v.startswith('Auditory ')
            }
        elif encoding_type == 'visual':
            filtered_contrasts = {
                k: v for k, v in selected_contrasts.items()
                if v.startswith('Visual ')
            }
        else:
            raise ValueError(
                "encoding_type must be 'bothmod', 'auditory', or 'visual'."
            )
    else:
        filtered_contrasts = None

    for tag, wpair in zip(tags, weights_list):

        # ##################### PER-ROI ANOVAS ###################### #

        dfrois = pd.DataFrame()

        for adir, aname, rname, rlab in zip(
            atlas_dirnames, atlas_names, region_names, roi_names
        ):
            
            # ###### Preparing the dataframe for the current ROI ######
            if rname == 'dorsal_striatum':
                outdir = os.path.join(msdtb_dir, rname, aname)
            else:
                outdir = os.path.join(msdtb_dir, rname, aname, rlab)

            anovas_dir = os.path.join(outdir, 'anovas')
            df_path = os.path.join(anovas_dir, f"{tag}_{rlab}_df.tsv")
            os.makedirs(anovas_dir, exist_ok=True)

            if folder_name == 'all_tasks':
                main_dir = os.path.join(
                    roi_dir, f"{encoding_type}_{task_roidef_id}", 'main_tasks'
                )
                rand_dir = os.path.join(
                    roi_dir,
                    f"{encoding_type}_{task_roidef_id}",
                    'rand_ntfd_pairs',
                )

                if rname == 'dorsal_striatum':
                    outdir_main = os.path.join(main_dir, rname, aname)
                    outdir_rand = os.path.join(rand_dir, rname, aname)
                else:
                    outdir_main = os.path.join(main_dir, rname, aname, rlab)
                    outdir_rand = os.path.join(rand_dir, rname, aname, rlab)

                df_path_main = os.path.join(
                    outdir_main, 'anovas', f"{tag}_{rlab}_df.tsv"
                )
                df_path_rand = os.path.join(
                    outdir_rand, 'anovas', f"{tag}_{rlab}_df.tsv"
                )

                dfroi_main = pd.read_csv(df_path_main, sep='\t')
                dfroi_rand = pd.read_csv(df_path_rand, sep='\t')
                dfroi_rand = dfroi_rand.copy()
                dfroi_rand['Task'] = dfroi_rand['Task'].replace({
                    'NTFD_Random': 'NTFD Random',
                    'NTFD-Random': 'NTFD Random',
                })
                dfroi = pd.concat(
                    [dfroi_main, dfroi_rand], ignore_index=True, axis=0
                )
                dfroi.to_csv(df_path, index=False, sep='\t')
            else:
                rois_path = os.path.join(
                    outdir, 'rois_extraction', f"{tag}_{rlab}_psc.npy"
                )
                dfroi = dataframe(
                    rois_path, ['lh', 'rh', 'bh'], list(tasks.values()),
                    list(filtered_contrasts.values()), SUBJECTS, df_path
                )

            dfroi['ROI'] = np.repeat(rlab, len(dfroi.index))
            dfrois = pd.concat([dfrois, dfroi], ignore_index=True)

            # #### Per-ROI analyses (and posthocs written to TSVs) ####
            if n_rois in [4, 6, 10] and encoding_type == 'bothmod':
                if folder_name == 'main_tasks':

                    # ######### One-way ANOVA #########
                    ow_dir = os.path.join(anovas_dir, '1way-anova')
                    if encoding_type == 'bothmod':
                        oneway_rm_cat(
                            df_path, tasks, ow_dir, tag, rlab
                        )
                    elif encoding_type == 'auditory':
                        oneway_rm_cat(
                            df_path, tasks, ow_dir, tag, rlab,
                            modalities=('Auditory',)
                        )
                    else:
                        oneway_rm_cat(
                            df_path, tasks, ow_dir, tag, rlab,
                            modalities=('Visual',)
                        )

                    # ######### Two-way ANOVA #########
                    t2_dir = os.path.join(anovas_dir, '2way-anova_modcat')
                    twoway_rm_modcat(
                        df_path, tasks, t2_dir, tag, rlab
                    )
                    gt_dir = os.path.join(
                        anovas_dir, '2way-anova_modcat_tasks-averaged'
                    )
                    twoway_rm_modcat_taskavg(df_path, gt_dir, tag, rlab)

                    # ######### Three-way ANOVA #########
                    three_dir = os.path.join(
                        anovas_dir, '3way-anova_catmodtask'
                    )
                    threeway_rm_catmodtask(df_path, three_dir, tag, rlab)

                if folder_name in ('all_tasks', 'rand_ntfd_pairs',
                                   'rand_ntfd_nonrandom'):

                    # ######### Two-way ANOVA #########
                    t2_dir = os.path.join(anovas_dir, '2way-anova_modcat')
                    twoway_rm_modcat(
                        df_path, tasks, t2_dir, tag, rlab
                    )

                    if folder_name == 'rand_ntfd_pairs':
                        df_t2 = preprocess_rand_ntfd_nonrandom(
                            dfroi, aggregation_type='mean')
                        t2_nr_dir = os.path.join(
                            anovas_dir, '2way-anova_modcat_nonrandom-mean'
                        )
                        twoway_rm_modcat(
                            df_t2, tasks, t2_nr_dir, tag, rlab
                        )

                if folder_name in ('main_tasks', 'all_tasks'):

                    # ######### Two-way ANOVA #########
                    tm_dir = os.path.join(anovas_dir, '2way-anova_modtask')
                    twoway_rm_modtask(df_path, tm_dir, tag, rlab)

        # Save concatenated ROI dataframe
        df_dir = os.path.join(msdtb_dir, 'df_rois_volume')

        os.makedirs(df_dir, exist_ok=True)
        dfrois.to_csv(
            os.path.join(df_dir, f"dfrois_{tag}_{n_rois}-rois.tsv"),
            sep='\t', index=False
        )

        # Open dfrois from saved TSV
        dfrois = pd.read_csv(
            os.path.join(df_dir, f"dfrois_{tag}_{n_rois}-rois.tsv"),
            sep='\t'
        )

        # #################### MULTI-ROI ANOVAS ##################### #
        # Multi-ROI analyses + posthoc plots
        if n_rois in (2, 3, 4, 6, 8):

            # both modalities
            cat_dir = os.path.join(
                msdtb_dir, f"2way-anova_vol_roicat_{n_rois}rois"
            )
            twoway_rm_roicat(dfrois, tasks, cat_dir, tag, modality=None)

            # auditory
            cat_dir_a = os.path.join(
                msdtb_dir, f"2way-anova_vol_roicat_{n_rois}rois_auditory"
            )
            twoway_rm_roicat(
                dfrois, tasks, cat_dir_a, tag, modality='auditory'
            )

            # visual
            cat_dir_v = os.path.join(
                msdtb_dir, f"2way-anova_vol_roicat_{n_rois}rois_visual"
            )
            twoway_rm_roicat(
                dfrois, tasks, cat_dir_v, tag, modality='visual'
            )

            # timing-ROI (main_tasks only) + posthoc plots
            if folder_name == 'main_tasks':
                t_dir = os.path.join(
                    msdtb_dir, f"2way-anova_vol_roitask_{n_rois}rois"
                )
                twoway_rm_roitask(
                    dfrois, t_dir, tag, modality=None
                )

                t_dir_a = os.path.join(
                    msdtb_dir,
                    f"2way-anova_vol_roitask_{n_rois}rois_auditory"
                )
                twoway_rm_roitask(
                    dfrois, t_dir_a, tag, modality='auditory'
                )

                t_dir_v = os.path.join(
                    msdtb_dir,
                    f"2way-anova_vol_roitask_{n_rois}rois_visual"
                )
                twoway_rm_roitask(
                    dfrois, t_dir_v, tag, modality='visual'
                )

                t_three_dir = os.path.join(
                    msdtb_dir, f"3way-anova_vol_roimodtask_{n_rois}rois"
                )
                threeway_rm_roimodtask(dfrois, t_three_dir, tag)