"""
ANCOVA analyses of behavioral data of Production Tasks of the
 Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: June 2026

Compatibility: Python 3.10.14
"""

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt

# %%
# ======================== MAIN FUNCTIONS ==============================


def ffx_dvar(df, estimator='mean'):
    """Subject-level dependent variable for ANCOVA analyses."""
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


def group_dvar(df_ffx, estimator='mean'):
    """Group-level dependent variable for plotting."""
    # Group effect for plotting
    df_group = df_ffx.drop(['subject'], axis=1)
    if estimator == 'mean':
        df_group = df_group.groupby([
            'condition', 'modality', 'standard']).mean().reset_index()
    else:
        assert estimator == 'std'
        df_group = df_group.groupby([
            'condition', 'modality', 'standard']).std().reset_index()

    async_ab = df_group[df_group.modality=='auditory'][
        df_group.condition=='beat'].signed_asynchrony.values
    async_ai = df_group[df_group.modality=='auditory'][
        df_group.condition=='interval'].signed_asynchrony.values
    async_vb = df_group[df_group.modality=='visual'][
        df_group.condition=='beat'].signed_asynchrony.values
    async_vi = df_group[df_group.modality=='visual'][
        df_group.condition=='interval'].signed_asynchrony.values

    group_async = [[async_ab.tolist()] + [async_ai.tolist()]] + \
                   [[async_vb.tolist()] + [async_vi.tolist()]]

    return group_async


def wide_dataframe(df, output_folder, estimator_id, sesstag):
    """
    Convert dataframe in the wide format for ancova analyses with JASP.
    """

    wdf = pd.pivot(df, values='signed_asynchrony',
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


def _fixed_effects_table(result):
    """Return a fixed-effect summary table."""
    ci = result.conf_int()
    fe_names = result.fe_params.index

    rows = []
    for name in fe_names:
        rows.append({
            'term': name,
            'estimate': result.fe_params[name],
            'se': result.bse[name],
            'z': result.tvalues[name],
            'p': result.pvalues[name],
            'ci_low': ci.loc[name, 0],
            'ci_high': ci.loc[name, 1],
        })

    return pd.DataFrame(rows)


def _wald_terms_table(result):
    """Return Wald tests for model terms when available."""
    try:
        table = result.wald_test_terms().table
    except Exception:
        return pd.DataFrame()

    table = table.reset_index().rename(columns={'index': 'term'})
    return table


def _model_info_table(result, model_name, formula, re_formula, n_obs,
                      n_subjects):
    """Return basic model information."""
    return pd.DataFrame([{
        'model': model_name,
        'formula': formula,
        'random_effects': re_formula if re_formula else '1',
        'n_obs': n_obs,
        'n_subjects': n_subjects,
        'converged': result.converged,
        'log_likelihood': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        'scale': result.scale,
    }])


def _save_mixedlm_outputs(result, output_folder, prefix, formula,
                          re_formula, n_obs, n_subjects):
    """Save fixed-effect, Wald-term, and model-info tables."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fixed = _fixed_effects_table(result)
    fixed.to_csv(
        os.path.join(output_folder, prefix + '_fixed_effects.tsv'),
        index=False,
        sep='\t',
    )

    terms = _wald_terms_table(result)
    if not terms.empty:
        terms.to_csv(
            os.path.join(output_folder, prefix + '_wald_terms.tsv'),
            index=False,
            sep='\t',
        )

    info = _model_info_table(
        result,
        prefix,
        formula,
        re_formula,
        n_obs,
        n_subjects,
    )
    info.to_csv(
        os.path.join(output_folder, prefix + '_model_info.tsv'),
        index=False,
        sep='\t',
    )


def mixed_ancova_tables(df, output_folder, estimator_id, sesstag):
    """Fit Linear Mixed Model (LMM) ANCOVA-style models and save model 
       tables.

    Standard is treated as a continuous within-subject predictor. The
    two-way model tests Condition, Modality, Standard, and their
    interactions. The one-way models test Condition and Standard
    separately within each modality.
    """
    cols = [
        'subject',
        'condition',
        'modality',
        'standard',
        'signed_asynchrony',
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

    formula = (
        'signed_asynchrony ~ '
        'C(condition) * C(modality) * standard_c'
    )
    re_formula = '~standard_c'
    result = _fit_mixedlm(
        formula,
        mdf,
        'subject',
        re_formula=re_formula,
    )

    prefix = 'lmm_' + estimator_id + '_' + sesstag + '_2way'
    _save_mixedlm_outputs(
        result,
        output_folder,
        prefix,
        formula,
        re_formula,
        len(mdf),
        mdf['subject'].nunique(),
    )

    for modality in ['auditory', 'visual']:
        sdf = mdf[mdf['modality'] == modality].copy()
        formula = 'signed_asynchrony ~ C(condition) * standard_c'
        result = _fit_mixedlm(
            formula,
            sdf,
            'subject',
            re_formula=re_formula,
        )

        prefix = (
            'lmm_' + estimator_id + '_' + sesstag + '_' +
            modality + '_1way'
        )
        _save_mixedlm_outputs(
            result,
            output_folder,
            prefix,
            formula,
            re_formula,
            len(sdf),
            sdf['subject'].nunique(),
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

# ##################### Subjects' lists ###############################
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

# Subjects who did all behavioral sessions with the random condition...
# ... in the NTFD task and img sessions
BEHAVIMG_RAND_SUBJECTS = [16, 18, 20, 21, 22, 23, 26, 28, 29, 32, 34, 35, 38,
                          39, 40, 41, 42, 43, 44, 45, 46, 47]

# Second batch
SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54]

# #####################################################################

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results')
DATAFRAMES_FOLDER = os.path.join(RESULTS_FOLDER, 'dataframes')
ANCOVA_FOLDER = os.path.join(RESULTS_FOLDER, 'ancova')

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
    'ses-01': 'Session 1',
}

sb_subjects_dic = {
    'ses-01': SB_SUBJECTS,
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

# Keep these lists explicit so each input/output type can be run one at a
# time by commenting out any entry if needed.
# BATCHES_TO_RUN = ['first', 'second']
BATCHES_TO_RUN = ['second']

INPUT_TYPES_TO_RUN = ['latency_corrected', 'uncorrected']

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
}

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

            jasp_folder = os.path.join(
                ANCOVA_FOLDER, batch_folder, 'jasp')
            plots_folder = os.path.join(
                ANCOVA_FOLDER, batch_folder, 'plots')
            tables_folder = os.path.join(
                ANCOVA_FOLDER, batch_folder, 'tables')

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
                df = df_subfiltered[
                    df_subfiltered['session'].isin(sessions_list)]

                # Remove rows with 'NaN' entries.
                df = df.dropna(subset=['signed_asynchrony'])

                # Extract covariate.
                standards = np.unique(df['standard'])

                # Extract dependent variable.
                db_ffx_mean = ffx_dvar(df, estimator='mean')
                db_ffx_std = ffx_dvar(df, estimator='std')
                mean_async = group_dvar(db_ffx_mean, estimator='mean')
                std_async = group_dvar(db_ffx_std, estimator='mean')

                # Convert dataframe in the wide format for JASP.
                wide_dataframe(db_ffx_mean, jasp_folder, 'mean', key)
                wide_dataframe(db_ffx_std, jasp_folder, 'std', key)

                # Fit LMM ANCOVA-style models and save tables.
                mixed_ancova_tables(
                    db_ffx_mean, tables_folder, 'mean', key)
                mixed_ancova_tables(
                    db_ffx_std, tables_folder, 'std', key)

                # Plot ANCOVA.
                plot_ancova(
                    standards, mean_async,
                    'Mean of Signed Asynchrony', .165,
                    'Mean of Signed Asynchrony for every Standard: ' +
                    value,
                    plots_folder, 'mean_ancova_production_' + key,
                    hline_legend=r'$RT=Standard$',
                    legend_loc='upper right')

                plot_ancova(
                    standards, std_async,
                    'SD of Signed Asynchrony', .165,
                    'Standard Deviation (SD) of Signed Asynchrony '
                    'for every Standard: ' + value,
                    plots_folder, 'std_ancova_production_' + key,
                    legend_loc='upper right')