"""
Linear mixed model (LMM) analyses of behavioral data of Production Tasks
of the TDTB project

Runs the same analysis pipeline on two dependent variables, both read
from the trial-level dataframes written by production_df.py and both
expressed as a fraction of the standard interval:

    signed_asynchrony    A = (R - S) / S, the constant error, negative
                         for anticipations and positive for lags, so
                         the two cancel when averaged.
    absolute_asynchrony  |A|, the magnitude of the error irrespective
                         of direction.

Each dependent variable writes into its own subtree, so the two
analyses never overwrite each other:

    lmm/<batch_folder>/<dependent_variable>/jasp/
    lmm/<batch_folder>/<dependent_variable>/plots/
    lmm/<batch_folder>/<dependent_variable>/tables/

For each dependent variable and session tag, two estimators are formed
per subject, condition, modality and standard: the mean over trials and
the standard deviation over trials. Each is then passed through the wide
export for JASP, the three mixed models (2way, auditory_1way,
visual_1way) and the group-level plot across standards.

Two things to keep in mind when reading the absolute results. The
absolute measure mixes bias and variability, since a subject with no
systematic bias still has a positive mean |A| that grows with their
inconsistency; a condition effect there can therefore come from a
difference in variability alone, and the signed mean and its standard
deviation are what separate the two. And |A| is bounded below by zero
and right-skewed at the trial level, although the models are fitted on
subject-level means over roughly thirty trials, which are far closer to
normal than the raw trials are.

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: September 2026

Compatibility: Python 3.10.14
"""

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from scipy import stats
from matplotlib import pyplot as plt

# %%
# ======================== MAIN FUNCTIONS ==============================


def ffx_dvar(df, estimator='mean'):
    """Subject-level dependent variable for the LMM analyses."""
    # Fixed Effects within subjects
    df_ffx = df.drop(['session'], axis=1)
    if estimator == 'mean':
        df_ffx = df_ffx.groupby([
            'condition', 'modality',
            'standard', 'subject']).mean().reset_index()
    else:
        assert estimator == 'std'
        df_ffx = df_ffx.groupby([
            'condition', 'modality',
            'standard', 'subject']).std().reset_index()

    return df_ffx


def group_dvar(df_ffx, dvar, estimator='mean'):
    """Group-level dependent variable for plotting.

    ``dvar`` names the column to summarise, either 'signed_asynchrony'
    or 'absolute_asynchrony'. Returns a nested list ordered as
    [[auditory beat, auditory interval], [visual beat, visual
    interval]], each entry holding one value per standard in ascending
    order, which is the layout plot_ancova expects.
    """
    # Group effect for plotting
    df_group = df_ffx.drop(['subject'], axis=1)
    if estimator == 'mean':
        df_group = df_group.groupby([
            'condition', 'modality', 'standard']).mean().reset_index()
    else:
        assert estimator == 'std'
        df_group = df_group.groupby([
            'condition', 'modality', 'standard']).std().reset_index()

    def cell(modality, condition):
        mask = ((df_group['modality'] == modality) &
                (df_group['condition'] == condition))
        return df_group[mask][dvar].values.tolist()

    group_async = [[cell('auditory', 'beat'), cell('auditory', 'interval')],
                   [cell('visual', 'beat'), cell('visual', 'interval')]]

    return group_async


def wide_dataframe(df, dvar, output_folder, estimator_id, sesstag):
    """
    Convert dataframe to the wide format for ANCOVA in JASP (kept as an
    independent cross-check of the LMM).

    ``dvar`` names the column to pivot, either 'signed_asynchrony' or
    'absolute_asynchrony'.
    """

    wdf = pd.pivot(df, values=dvar,
                   index=['subject', 'standard'],
                   columns=['modality', 'condition'])

    # Flatten the MultiIndex and capitalize the column names
    wdf.columns = [' '.join(col).strip().title()
                   for col in wdf.columns.values]

    # Reset the index so that Standard and Subject are regular columns
    wdf.reset_index(inplace=True)

    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save dataframe
    wdf_outpath = os.path.join(
        output_folder, 'wide_df_production_' + estimator_id + '_' + sesstag +
        '.tsv')
    wdf.to_csv(wdf_outpath, index=False, sep='\t')


def _fit_mixedlm(formula, df, group_col, re_formula=None):
    """Fit a mixed model, with conservative fallback options."""
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df[group_col],
        re_formula=re_formula,
    )

    try:
        return model.fit(reml=False, method='lbfgs')
    except Exception:
        try:
            return model.fit(reml=False, method='powell', maxiter=1000)
        except Exception:
            if re_formula is None:
                raise

            model = smf.mixedlm(
                formula,
                data=df,
                groups=df[group_col],
            )
            return model.fit(reml=False, method='lbfgs')


def _coefficient_rows(result, model_label):
    """Tidy fixed-effect coefficients, one row per model term.

    'statistic' is a Wald z value (statsmodels MixedLM uses a normal
    approximation, not a t/F); 'p' and the CI are the matching
    two-sided Wald quantities.
    """
    ci = result.conf_int()
    rows = []
    for name in result.fe_params.index:
        rows.append({
            'model': model_label,
            'block': 'coefficient',
            'term': name,
            'estimate': result.fe_params[name],
            'se': result.bse[name],
            'statistic': result.tvalues[name],   # Wald z
            'df': np.nan,
            'p': result.pvalues[name],
            'ci_low': ci.loc[name, 0],
            'ci_high': ci.loc[name, 1],
        })
    return pd.DataFrame(rows)


def _omnibus_rows(result, model_label):
    """Tidy Wald tests for grouped model terms, one row per factor/term.

    Each row tests the joint null that all coefficients belonging to a
    term are zero -- the mixed-model analogue of an ANCOVA main effect
    or interaction. The statistic is a Wald chi-square on 'df' degrees
    of freedom.
    """
    try:
        table = result.wald_test_terms(scalar=True).table
    except Exception:
        return pd.DataFrame()

    table = table.reset_index().rename(columns={'index': 'term'})
    rows = []
    for _, r in table.iterrows():
        rows.append({
            'model': model_label,
            'block': 'omnibus',
            'term': r['term'],
            'estimate': np.nan,
            'se': np.nan,
            'statistic': r['statistic'],         # Wald chi-square
            'df': r['df_constraint'],
            'p': r['pvalue'],
            'ci_low': np.nan,
            'ci_high': np.nan,
        })
    return pd.DataFrame(rows)


def _re_structure(result):
    """Human-readable random-effects structure actually fitted.

    Reads the effects that were estimated (not merely requested), so a
    fallback that drops the random slope is reported truthfully.
    """
    names = list(result.model.data.exog_re_names)
    pretty = ['intercept' if n == 'Group' else n for n in names]
    return ' + '.join(pretty)


def _info_row(result, model_label, formula, n_obs, n_subjects):
    """One-row model summary / fit diagnostics."""
    return pd.DataFrame([{
        'model': model_label,
        'formula': formula,
        'random_effects': _re_structure(result),
        'n_obs': n_obs,
        'n_subjects': n_subjects,
        'converged': result.converged,
        'log_likelihood': result.llf,
        'aic': result.aic,
        'bic': result.bic,
    }])


def _emm_cells(model_label):
    """Cells for which to compute estimated marginal means, per model.

    Returns one entry per cell: the modality/condition labels used for
    reporting, plus a 'data' dict holding exactly the predictors the
    model's formula references, with standard_c fixed at 0 (the mean
    standard). The 2way model varies both factors; each 1way model fixes
    its modality and varies condition.
    """
    conditions = ['beat', 'interval']
    if model_label == '2way':
        return [
            {'modality': mod, 'condition': cond,
             'data': {'condition': cond, 'modality': mod, 'standard_c': 0.0}}
            for mod in ['auditory', 'visual']
            for cond in conditions
        ]
    modality = 'auditory' if model_label == 'auditory_1way' else 'visual'
    return [
        {'modality': modality, 'condition': cond,
         'data': {'condition': cond, 'standard_c': 0.0}}
        for cond in conditions
    ]


def _emm_rows(result, model_label, cells):
    """Estimated marginal means (EMMs) at the mean standard, one row per cell.

    Each cell mean is the linear combination L @ beta of the fixed
    effects, with L the model's design row for that cell evaluated at
    standard_c = 0. Its standard error is sqrt(L @ V @ L), where V is the
    fixed-effect covariance (result.cov_params restricted to the fixed
    effects). This gives the correct SE for cells that combine several
    coefficients (e.g. the interval cells), which is not the sum of the
    coefficient SEs. Inference is Wald z, to match the coefficient block.

    L is built from the model's stored patsy design so it always aligns
    with the fitted parameters, regardless of the coding used.
    """
    design_info = result.model.data.design_info
    fe_names = list(result.fe_params.index)
    beta = result.fe_params.values

    cov = result.cov_params()
    if hasattr(cov, 'loc'):
        cov_fe = cov.loc[fe_names, fe_names].values
    else:
        cov_fe = np.asarray(cov)[:len(fe_names), :len(fe_names)]

    z_crit = stats.norm.ppf(0.975)
    rows = []
    for cell in cells:
        design = patsy.build_design_matrices(
            [design_info],
            pd.DataFrame([cell['data']]),
            return_type='matrix',
        )[0]
        contrast = np.asarray(design).ravel()

        estimate = float(contrast @ beta)
        se = float(np.sqrt(contrast @ cov_fe @ contrast))
        z = estimate / se
        rows.append({
            'model': model_label,
            'block': 'emm',
            'modality': cell['modality'],
            'condition': cell['condition'],
            'estimate': estimate,
            'se': se,
            'statistic': z,               # Wald z
            'df': np.nan,
            'p': 2 * stats.norm.sf(abs(z)),
            'ci_low': estimate - z_crit * se,
            'ci_high': estimate + z_crit * se,
        })
    return pd.DataFrame(rows)


def mixed_ancova_tables(df, dvar, output_folder, estimator_id, sesstag):
    """Fit the linear mixed models (LMMs) and save two consolidated TSV
    files.

    ``dvar`` names the dependent variable, either 'signed_asynchrony'
    or 'absolute_asynchrony'. Both enter the models the same way, as a
    subject-level mean or standard deviation over trials.

    Standard is treated as a continuous, mean-centred within-subject
    predictor (standard_c). Three models are fitted on the same data:

      * 2way          : condition * modality * standard_c (all rows)
      * auditory_1way : condition * standard_c (auditory rows only)
      * visual_1way   : condition * standard_c (visual rows only)

    Each model carries a per-subject random intercept and random slope
    for standard_c. Three files are written per (estimator, session):

      lmm_<estimator>_<sesstag>_results.tsv    coefficients + omnibus
      lmm_<estimator>_<sesstag>_emm.tsv        estimated marginal means
      lmm_<estimator>_<sesstag>_modelinfo.tsv  fit diagnostics

    Every row in the results file is tagged by 'model' (which of the
    three) and 'block' ('coefficient' or 'omnibus'). The EMM file reports
    each modality x condition cell mean at the mean standard, with its
    Wald standard error, z, p and 95% CI, tagged by 'model' and
    block='emm'. Cells from the 2way model use the pooled fit; cells from
    the auditory_1way and visual_1way models use each modality's own fit.
    """
    cols = [
        'subject',
        'condition',
        'modality',
        'standard',
        dvar,
    ]
    mdf = df[cols].dropna().copy()

    mdf['subject'] = mdf['subject'].astype(str)
    mdf['condition'] = pd.Categorical(
        mdf['condition'],
        categories=['beat', 'interval'],
        ordered=True,
    )
    mdf['modality'] = pd.Categorical(
        mdf['modality'],
        categories=['auditory', 'visual'],
        ordered=True,
    )
    mdf['standard_c'] = mdf['standard'] - mdf['standard'].mean()

    re_formula = '~standard_c'

    # (label, formula, data) for each model fitted on the same DV.
    specs = [
        (
            '2way',
            dvar + ' ~ C(condition) * C(modality) * standard_c',
            mdf,
        ),
        (
            'auditory_1way',
            dvar + ' ~ C(condition) * standard_c',
            mdf[mdf['modality'] == 'auditory'].copy(),
        ),
        (
            'visual_1way',
            dvar + ' ~ C(condition) * standard_c',
            mdf[mdf['modality'] == 'visual'].copy(),
        ),
    ]

    results_parts = []
    emm_parts = []
    info_parts = []
    for label, formula, data in specs:
        result = _fit_mixedlm(
            formula,
            data,
            'subject',
            re_formula=re_formula,
        )
        results_parts.append(_coefficient_rows(result, label))
        results_parts.append(_omnibus_rows(result, label))
        emm_parts.append(_emm_rows(result, label, _emm_cells(label)))
        info_parts.append(
            _info_row(
                result,
                label,
                formula,
                len(data),
                data['subject'].nunique(),
            )
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    stem = 'lmm_' + estimator_id + '_' + sesstag

    results = pd.concat(results_parts, ignore_index=True)
    results.to_csv(
        os.path.join(output_folder, stem + '_results.tsv'),
        index=False,
        sep='\t',
        na_rep='',
    )

    emm = pd.concat(emm_parts, ignore_index=True)
    emm.to_csv(
        os.path.join(output_folder, stem + '_emm.tsv'),
        index=False,
        sep='\t',
        na_rep='',
    )

    info = pd.concat(info_parts, ignore_index=True)
    info.to_csv(
        os.path.join(output_folder, stem + '_modelinfo.tsv'),
        index=False,
        sep='\t',
    )


def plot_ancova(x, y, yaxis_name, yname_pos, title,
                output_folder, fname, y_values=None, legend_loc='lower left',
                hline_legend=None, hline_label_pad=10):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between
    #          subplots
    # hspace # the amount of height reserved for white space between
             # subplots
    plt.subplots_adjust(left=.095, bottom=.15, right=.98, wspace=.175)

    # --- Automatic y-limits: audio and visual share the same range --------
    # Flatten all data values across both modalities and all conditions
    all_vals = [
        v for modality in y for condition in modality for v in condition
    ]
    data_min, data_max = min(all_vals), max(all_vals)
    data_range = data_max - data_min if data_max != data_min else 1.0
    pad = data_range * 0.15
    y_min_auto = data_min - pad
    y_max_auto = data_max + pad

    if y_values is None:
        # Generate ~5 evenly-spaced, rounded tick values spanning the data
        raw_ticks = np.linspace(y_min_auto, y_max_auto, 5)
        # Round to 2 significant decimal places for clean labels
        tick_magnitude = 10 ** np.floor(np.log10(abs(data_range) + 1e-12))
        decimals = max(0, int(-np.floor(np.log10(tick_magnitude + 1e-12))))
        y_values = np.round(raw_ticks, max(decimals, 2))
        y_lim = (y_min_auto, y_max_auto)
    else:
        # Honour explicit ticks but still derive shared limits from data
        y_lim = (y_min_auto, y_max_auto)
    # ----------------------------------------------------------------------

    colors = ['tab:blue', 'tab:orange']
    legend_labels = ['Beat', 'Interval']

    for m, modality_y in enumerate(y):
        for c, condition_y in enumerate(modality_y):
            # Linear fit
            a, b = np.polyfit(x, condition_y, deg=1)
            y_est = a * x + b
            # y_err = x.std() * \
            #     np.sqrt(1/len(x) +
            #             (x - x.mean())**2 /
            #             np.sum((x - x.mean())**2))

            # Plot the linear fit
            ax[m].plot(x, y_est, '-', color=colors[c], linewidth=12,
                       label=legend_labels[c], alpha=.5)
            # ax[0].fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            ax[m].plot(x, condition_y, 'bo', color=colors[c], markersize=16,
                       alpha=.5)
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_labels = [str(xl) for xl in x]
            ax[m].set_xticks(x, x_labels, fontsize=24)
            # Set limits of y-axis (shared across both subplots)
            ax[m].set_ylim(y_lim)
            y_labels = [str(yl) for yl in y_values]
            ax[m].set_yticks(y_values, y_labels, fontsize=24)
            # Add horizontal dashed line at y = 0
            if hline_legend:
                ax[m].axhline(0., linestyle='--', color='grey',
                              linewidth=12, alpha=.5)

        # Add legend
        if m == 0:
            ax[m].set_title('Audio', weight='bold', pad=0, fontsize=40)
            ax[m].legend(loc=legend_loc, frameon=False, prop={'size': 24})
        else:
            assert m == 1
            ax[m].set_title('Visual', weight='bold', pad=0, fontsize=40)

        # Name of x-axis
        fig.text(.465, .025, 'Standards (ms)', fontsize=30)
        # Name of y-axis
        fig.text(.005, yname_pos, yaxis_name, fontsize=30, rotation=90)

        # Label horizontal dashed line using y = 0 data coordinates.
        # The vertical offset is in points, so it does not depend on ylim.
        if hline_legend:
            ax[m].annotate(
                hline_legend,
                xy=(0.5, 0.),
                xycoords=('axes fraction', 'data'),
                xytext=(0, hline_label_pad),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=24,
                color='dimgrey',
            )

    # if fname[:4] == 'mean' and fname[-6:] == 'allses':
    #     fig.text(.125, .2, r'$p_{Condition*Standard}=2\mathrm{e}{-3}$',
    #              fontsize=24)
    #     fig.text(.6, .2, r'$p_{Condition*Standard}<1\mathrm{e}{-3}$',
    #              fontsize=24)
    # elif fname[:3] == 'std' and fname[-6:] == 'allses':
    #     fig.text(.125, .2, r'$p_{Condition*Standard}: \mathrm{n.s.}$',
    #              fontsize=24)
    #     fig.text(.6, .2, r'$p_{Condition*Standard}: \mathrm{n.s.}$',
    #              fontsize=24)
    # else:
    #     pass

    # Title
    # plt.suptitle(title, x=.5, y=.98, size=24, linespacing=.75)

    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save figure
    plt.savefig(
        os.path.join(output_folder, fname + '.png'),
        dpi=300,
        bbox_inches='tight')


# %%
# ========================== INPUTS ===================================

# *********************** First Batch *********************************
# Expyriment / Implicit

# All subjects
ALL_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                 41, 42, 43, 44, 45, 46, 47]

# Img subjects only (without pilot)
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# *********************** Second Batch ********************************
# Psychopy / Implicit

# All subjects
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                   62, 63]

# First Session completed
GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61,
                    62, 63]

# Second Session completed
SB2_SUBJECTS = [48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63]

# *********************** Third Batch *********************************
# Psychopy / Explicit

# All subjects
ALL_TB_SUBJECTS = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

# First Session completed
GOOD_TB_SUBJECTS = [64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75]

# Second Session completed
TB2_SUBJECTS = [65, 66, 68, 72, 73, 75]

# #####################################################################

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results')
DATAFRAMES_FOLDER = os.path.join(RESULTS_FOLDER, 'dataframes')
LMM_FOLDER = os.path.join(RESULTS_FOLDER, 'lmm')

# #### First Batch ####

fb_sessions_dic = {
    'allses': 'All Sessions',
    'behavses': 'Behavioral Sessions',
    'imgses': 'Imaging Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
    'ses-03': 'Session 3',
    'ses-04': 'Session 4',
    'ses-05': 'Session 5',
    'behav12': 'Behavioral Sessions 1 and 2',
    'behav13': 'Behavioral Sessions 1 and 3',
    'behav23': 'Behavioral Sessions 2 and 3',
}

fb_subjects_dic = {
    'allses': GOOD_SUBJECTS,
    'behavses': GOOD_SUBJECTS,
    'imgses': IMG_SUBJECTS,
    'ses-01': GOOD_SUBJECTS,
    'ses-02': GOOD_SUBJECTS,
    'ses-03': GOOD_SUBJECTS,
    'ses-04': IMG_SUBJECTS,
    'ses-05': IMG_SUBJECTS,
    'behav12': GOOD_SUBJECTS,
    'behav13': GOOD_SUBJECTS,
    'behav23': GOOD_SUBJECTS,
}

# #### Second Batch ####

sb_sessions_dic = {
    'behavses': 'Behavioral Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
}

sb_subjects_dic = {
    'behavses': GOOD_SB_SUBJECTS,
    'ses-01': GOOD_SB_SUBJECTS,
    'ses-02': SB2_SUBJECTS,
}

# #### Third Batch ####

tb_sessions_dic = {
    'behavses': 'Behavioral Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
}

tb_subjects_dic = {
    'behavses': GOOD_TB_SUBJECTS,
    'ses-01':   GOOD_TB_SUBJECTS,
    'ses-02':   TB2_SUBJECTS,
}

# #### Map tag -> integer session list ####

sessions_list_dic = {
    'allses': [1, 2, 3, 4, 5],
    'behavses': [1, 2, 3],
    'imgses': [4, 5],
    'ses-01': [1],
    'ses-02': [2],
    'ses-03': [3],
    'ses-04': [4],
    'ses-05': [5],
    'behav12': [1, 2],
    'behav13': [1, 3],
    'behav23': [2, 3],
}

# #### Dataframe/output configurations ####

fb_inputs_dic = {
    'latency_corrected': {
        'db_fname': 'df_production_fb_133_35_20',
        'batch_folder': 'first_batch_133_35_20',
    },
    'uncorrected': {
        'db_fname': 'df_production_fb_0_0_0',
        'batch_folder': 'first_batch_0_0_0',
    },
}

sb_inputs_dic = {
    'latency_corrected': {
        'db_fname': 'df_production_sb_63_35_20',
        'batch_folder': 'second_batch_63_35_20',
    },
    'uncorrected': {
        'db_fname': 'df_production_sb_0_0_0',
        'batch_folder': 'second_batch_0_0_0',
    },
}

tb_inputs_dic = {
    'latency_corrected': {
        'db_fname': 'df_production_tb_63_35_20',
        'batch_folder': 'third_batch_63_35_20',
    },
    'uncorrected': {
        'db_fname': 'df_production_tb_0_0_0',
        'batch_folder': 'third_batch_0_0_0',
    },
}

batch_dic = {
    'first': {
        'sessions': fb_sessions_dic,
        'subjects': fb_subjects_dic,
        'inputs': fb_inputs_dic,
    },
    'second': {
        'sessions': sb_sessions_dic,
        'subjects': sb_subjects_dic,
        'inputs': sb_inputs_dic,
    },
    'third': {
        'sessions': tb_sessions_dic,
        'subjects': tb_subjects_dic,
        'inputs': tb_inputs_dic,
    }
}

# Keep these lists explicit so each input/output type can be run one at a
# time by commenting out any entry if needed.
BATCHES_TO_RUN = ['first', 'second', 'third']
# BATCHES_TO_RUN = ['first', 'second']
# BATCHES_TO_RUN = ['second', 'third']
# BATCHES_TO_RUN = ['second']
# BATCHES_TO_RUN = ['third']

# INPUT_TYPES_TO_RUN = ['latency_corrected', 'uncorrected']
INPUT_TYPES_TO_RUN = ['uncorrected']

# #### Dependent variables ####
# Each dependent variable gets its own subfolder inside the batch folder,
# holding its own jasp/, plots/ and tables/ directories, so the two
# analyses never overwrite each other.
#
#   label   -- used in the axis names and titles of the plots.
#   hline   -- draw the dashed line at zero. Meaningful for the signed
#              measure, where zero separates anticipation from lag. The
#              absolute measure is bounded below by zero, so the line
#              would sit on the axis and mark a floor rather than a
#              crossing point.
dvars_dic = {
    'signed_asynchrony': {
        'label': 'Signed Asynchrony',
        'hline': True,
    },
    'absolute_asynchrony': {
        'label': 'Absolute Asynchrony',
        'hline': False,
    },
}

# DVARS_TO_RUN = ['signed_asynchrony']
# DVARS_TO_RUN = ['absolute_asynchrony']
DVARS_TO_RUN = ['signed_asynchrony', 'absolute_asynchrony']

# %%
# ============================ RUN ====================================

if __name__ == "__main__":

    for batch_tag in BATCHES_TO_RUN:
        batch_info = batch_dic[batch_tag]
        sessions_dic = batch_info['sessions']
        subjects_dic = batch_info['subjects']
        inputs_dic = batch_info['inputs']

        for input_type in INPUT_TYPES_TO_RUN:
            input_info = inputs_dic[input_type]
            db_fname = input_info['db_fname']
            batch_folder = input_info['batch_folder']

            print('\n' + '=' * 60)
            print(f'Batch: {batch_tag}  |  Input: {input_type}')
            print(f'Dataframe prefix: {db_fname}')
            print(f'Output folder: {batch_folder}')
            print('=' * 60)

            for key, value in sessions_dic.items():
                sessions_list = sessions_list_dic[key]

                print(f'\nSession tag: {key}  |  {value}')

                # Open dataframe.
                db_path = os.path.join(
                    DATAFRAMES_FOLDER, f'{db_fname}_{key}.tsv')
                db = pd.read_csv(db_path, sep='\t')

                # Filter Dataframe according to list of subjects.
                df_subfiltered = db[db['subject'].isin(subjects_dic[key])]

                # Filter Dataframe according to list of sessions.
                df_sessions = df_subfiltered[
                    df_subfiltered['session'].isin(sessions_list)]

                for dvar in DVARS_TO_RUN:
                    dvar_info = dvars_dic[dvar]
                    dvar_label = dvar_info['label']

                    if dvar not in df_sessions.columns:
                        print(f'  Skipping {dvar}: column absent from '
                              f'{os.path.basename(db_path)}. Regenerate the '
                              'dataframe with production_df.py.')
                        continue

                    print(f'  Dependent variable: {dvar}')

                    # Each dependent variable writes into its own subtree.
                    dvar_folder = os.path.join(
                        LMM_FOLDER, batch_folder, dvar)
                    jasp_folder = os.path.join(dvar_folder, 'jasp')
                    plots_folder = os.path.join(dvar_folder, 'plots')
                    tables_folder = os.path.join(dvar_folder, 'tables')

                    # Remove rows with 'NaN' entries. Both asynchronies
                    # are NaN on the same trials, since one is the
                    # absolute value of the other, but the drop is done
                    # per variable so that each analysis is explicit
                    # about the rows it used.
                    df = df_sessions.dropna(subset=[dvar])

                    # Extract covariate.
                    standards = np.unique(df['standard'])

                    # Extract dependent variable.
                    db_ffx_mean = ffx_dvar(df, estimator='mean')
                    db_ffx_std = ffx_dvar(df, estimator='std')
                    mean_async = group_dvar(
                        db_ffx_mean, dvar, estimator='mean')
                    std_async = group_dvar(
                        db_ffx_std, dvar, estimator='mean')

                    # Convert dataframe in the wide format for JASP.
                    wide_dataframe(
                        db_ffx_mean, dvar, jasp_folder, 'mean', key)
                    wide_dataframe(
                        db_ffx_std, dvar, jasp_folder, 'std', key)

                    # Fit the LMMs and save tables.
                    mixed_ancova_tables(
                        db_ffx_mean, dvar, tables_folder, 'mean', key)
                    mixed_ancova_tables(
                        db_ffx_std, dvar, tables_folder, 'std', key)

                    # Plot group-level results across standards.
                    plot_ancova(
                        standards, mean_async,
                        'Mean of ' + dvar_label, .165,
                        'Mean of ' + dvar_label + ' for every Standard: ' +
                        value,
                        plots_folder, 'mean_lmm_production_' + key,
                        hline_legend=(r'$RT=Standard$'
                                      if dvar_info['hline'] else None),
                        legend_loc='upper right')

                    plot_ancova(
                        standards, std_async,
                        'SD of ' + dvar_label, .165,
                        'Standard Deviation (SD) of ' + dvar_label +
                        ' for every Standard: ' + value,
                        plots_folder, 'std_lmm_production_' + key,
                        legend_loc='upper right')