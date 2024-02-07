"""
Analysis of behavioral data for the Production Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: February 2023
Last update: February 2024

Compatibility: Python 3.10.4
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from statannotations.Annotator import Annotator
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# setting path
sys.path.append('../')
# importing
from utils import parse_logfile, customize_vplot, change_width, ffx

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# %%
# ======================== MAIN FUNCTIONS ==============================


def production_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            condition = datum[4]
            theoretical_isi1 = int(datum[8])
            real_isi1 = int(datum[9])
            if data[dt+8][5] == 'feedback' and data[dt+8][11] in ['o', 'b']:
                rt = int(data[dt+7][7]) + int(data[dt+8][10])
            elif data[dt+8][5] == 'feedback' and data[dt+8][10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, real_isi1, rt])

    return trials


def filter_trialtype(trs, category):
    beat = [tr[1:] for tr in trs if tr[0][:4] == 'beat']
    interval = [tr[1:] for tr in trs if tr[0][:8] == 'interval']
    random = [tr[1:] for tr in trs if tr[0][:6] == 'random']

    if category in ['production', 'ntfd']:
        beat = [list(map(int, b)) if ~np.any(np.isnan(b)) else b
                for b in beat]
        interval = [list(map(int, i)) if ~np.any(np.isnan(i)) else i
                    for i in interval]
        if random:
            random = [list(map(int, r)) if ~np.any(np.isnan(r)) else r
                      for r in random]
    else:
        assert category == 'perception'
        beat = [[int(b[0]), int(b[1]), b[2]] for b in beat]
        interval = [[int(i[0]), int(i[1]), i[2]] for i in interval]

    return beat, interval, random


def symlog_transform(arr, shift):
    """About this function, consult:
    https://pythonmatplotlibtips.blogspot.com/2018/11/x-symlog-with-shift.html
    """
    logv = np.abs(arr)*(10.**shift)
    logv[np.where(logv < 1.)] = 1.
    logv = np.sign(arr)*np.log10(logv)

    return logv


def individual_production_isi_sync(
        subjects, sesstypes, this_dir, output_folder, sync_type, n_trials,
        flatten=True, tasks=['Auditory Production', 'Visual Production']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            data = parse_logfile(logfiles_dir, subject, sesstypes, task,
                                 n_trials)
            trials = production_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'production')

            # ############# Asynchronies per ISI #######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            ss_isi_beat = []
            as_isi_beat = []
            for i in isi1s:
                ss_beat = []
                as_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            ssb = round((beat_trial[2] - beat_trial[1]) / \
                                        beat_trial[1], 2)
                            asb = abs(ssb)
                        else:
                            ssb = np.nan
                            asb = np.nan
                        ss_beat.append(ssb)
                        as_beat.append(asb)
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(ss_beat)):
                    miss_sbval = np.nanmedian(ss_beat)
                    ss_beat = np.where(np.isnan(ss_beat), miss_sbval,
                                       ss_beat).tolist()
                if np.any(np.isnan(as_beat)):
                    miss_abval = np.nanmedian(as_beat)
                    as_beat = np.where(np.isnan(as_beat), miss_abval,
                                       as_beat).tolist()
                # Append isi array
                ss_isi_beat.append(ss_beat)
                as_isi_beat.append(as_beat)

            ss_isi_interval = []
            as_isi_interval = []
            for i in isi1s:
                ss_interval = []
                as_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == i:
                        if ~np.any(np.isnan(interval_trial)):
                            ssi = round((interval_trial[2] - \
                                         interval_trial[1]) / \
                                        interval_trial[1], 2)
                            asi = abs(ssi)
                        else:
                            ssi = np.nan
                            asi = np.nan
                        ss_interval.append(ssi)
                        as_interval.append(asi)
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(ss_interval)):
                    miss_sival = np.nanmedian(ss_interval)
                    ss_interval = np.where(np.isnan(ss_interval), miss_sival,
                                           ss_interval).tolist()
                if np.any(np.isnan(as_interval)):
                    miss_aival = np.nanmedian(as_interval)
                    as_interval = np.where(np.isnan(as_interval), miss_aival,
                                           as_interval).tolist()
                # Append isi array
                ss_isi_interval.append(ss_interval)
                as_isi_interval.append(as_interval)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 120))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .97 - s*.02, .3, .016])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            # Transform in Symlog
            logbeat = []
            loginterval = []
            shift = 2
            # Convert x value to symlog scale with shift=2
            if sync_type == 'signed':
                for lsbeat in ss_isi_beat:
                    logv = symlog_transform(lsbeat, shift)
                    logbeat.append(logv)
                for lsint in ss_isi_interval:
                    logv = symlog_transform(lsint, shift)
                    loginterval.append(logv)
            else:
                assert sync_type == 'absolute'
                for lsbeat in as_isi_beat:
                    logv = symlog_transform(lsbeat, shift)
                    logbeat.append(logv)
                for lsint in as_isi_interval:
                    logv = symlog_transform(lsint, shift)
                    loginterval.append(logv)

            beat = ax.boxplot(logbeat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(loginterval,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. + width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)
            # Overplot the mean, with horizontal alignment
            # in the center of each box
            for j in np.arange(len(x)):
                medbeat = beat['medians'][j]
                medinterval = interval['medians'][j]
                ax.plot(np.average(medbeat.get_xdata()),
                        np.average(ss_isi_beat[j]),
                        color='w', marker='*', markeredgecolor='k')
                ax.plot(np.average(medinterval.get_xdata()),
                        np.average(ss_isi_interval[j]),
                        color='w', marker='*', markeredgecolor='k')

            # Fill boxes with colors
            colors1 = ['tab:blue', 'lightblue']
            colors2 = ['purple', 'thistle']
            for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
                if sync_type == 'signed':
                    patch1.set_facecolor(colors1[0])
                    patch2.set_facecolor(colors1[1])
                else:
                    assert sync_type == 'absolute'
                    patch1.set_facecolor(colors2[0])
                    patch2.set_facecolor(colors2[1])

            # x-label at the bottom
            if s == len(subjects) - 1:
                fig.text(.5, .005, ' ISIs (ms)', size=18)

            # x-tick labels with the standards
            ax.set_xticks(x*2., x_labels)

            if sync_type == 'signed':
                plt.ylim([-3., 3.])
                if (t % 2) == 0:
                    ax.set_ylabel('SymLog10(Asynchrony)')
            else:
                assert sync_type == 'absolute'
                plt.ylim([-.3, 3.])
                if (t % 2) == 0:
                    ax.set_ylabel('Log10(Asynchrony)')

            if s == 0:
                ax.set_title(task, pad=30, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'best',
                              prop={'size': 8})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right', prop={'size': 8})
                    fig.text(.26, 0.9825, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.9825, ' Mean', color='black',
                             weight='roman', size='x-small')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate all data
            if task == 'Auditory Production' and sync_type == 'signed':
                allsub_beat_audio.append(ss_isi_beat)
                allsub_interval_audio.append(ss_isi_interval)
            elif task == 'Visual Production' and sync_type == 'signed':
                allsub_beat_visual.append(ss_isi_beat)
                allsub_interval_visual.append(ss_isi_interval)
            elif task == 'Auditory Production' and sync_type == 'absolute':
                allsub_beat_audio.append(as_isi_beat)
                allsub_interval_audio.append(as_isi_interval)
            else:
                assert task == 'Visual Production' and sync_type == 'absolute'
                allsub_beat_visual.append(as_isi_beat)
                allsub_interval_visual.append(as_isi_interval)

        fig.text(.07, .9765 - s * .02, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    if sync_type == 'signed':
        plt.suptitle(
            'Individual Signed Asynchrony for the Production tasks',
            x=.5, y=.9975, size=14, linespacing=.75)
    else:
        assert sync_type == 'absolute'
        plt.suptitle(
            'Individual Absolute Asynchrony for the Production tasks',
            x=.5, y=.9975, size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(
        this_dir, output_folder,
        'production_individual_isi_' + sync_type + '_asynch.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio)
        allsub_interval_audio = np.ravel(allsub_interval_audio)
        allsub_beat_visual = np.ravel(allsub_beat_visual)
        allsub_interval_visual = np.ravel(allsub_interval_visual)

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def individual_production_isi_rts(
        subjects, sesstypes, this_dir, output_folder, n_trials, flatten=True,
        tasks = ['Auditory Production', 'Visual Production']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstypes, task,
                                 n_trials)
            trials = production_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'production')

            # Filter necessary data
            beat_trials = [np.delete(trial, 1).tolist()
                           for trial in beat_trials]
            interval_trials = [np.delete(trial, 1).tolist()
                               for trial in interval_trials]

            # ############## Extract RT's per ISI ###################### 
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            rts_beat.append(beat_trial[1])
                        else:
                            rts_beat.append(np.nan)
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(rts_beat)):
                    miss_bval = np.nanmedian(rts_beat)
                    rts_beat = np.where(np.isnan(rts_beat), miss_bval,
                                        rts_beat).tolist()
                # Append isi array
                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        if ~np.any(np.isnan(interval_trial)):
                            rts_interval.append(interval_trial[1])
                        else:
                            rts_interval.append(np.nan)
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(rts_interval)):
                    miss_ival = np.nanmedian(rts_interval)
                    rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                                            rts_interval).tolist()
                # Append isi array
                rt_isi1_grouped_interval.append(rts_interval)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 120))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .97 - s*.02, .3, .016])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            # Transform in the LogSpace
            logbeat = [np.log10(i) for i in rt_isi1_grouped_beat]
            loginterval = [np.log10(j) for j in rt_isi1_grouped_interval]

            beat = ax.boxplot(logbeat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(loginterval,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. + width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)

            # Overplot the mean, with horizontal alignment
            # in the center of each box
            for j in np.arange(len(x)):
                medbeat = beat['medians'][j]
                medinterval = interval['medians'][j]
                ax.plot(np.average(medbeat.get_xdata()),
                        np.average(rt_isi1_grouped_beat[j]),
                        color='w', marker='*', markeredgecolor='k')
                ax.plot(np.average(medinterval.get_xdata()),
                        np.average(rt_isi1_grouped_interval[j]),
                        color='w', marker='*', markeredgecolor='k')

            # Fill boxes with colors
            colors = ['b', 'y']
            for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
                patch1.set_facecolor(colors[0])
                patch2.set_facecolor(colors[1])

            if s == len(subjects) - 1:
                fig.text(.5, .005, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)
            plt.ylim([2., 3.35])

            if (t % 2) == 0:
                ax.set_ylabel('Log10(Response Time)')

            if s == 0:
                ax.set_title(task, pad=30, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'best',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.26, 0.9825, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.9825, ' Mean', color='black',
                             weight='roman', size='x-small')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data
            if task == 'Auditory Production':
                allsub_beat_audio.append(rt_isi1_grouped_beat)
                allsub_interval_audio.append(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual Production'
                allsub_beat_visual.append(rt_isi1_grouped_beat)
                allsub_interval_visual.append(rt_isi1_grouped_interval)

        fig.text(.07, .9765 - s * .02, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    plt.suptitle(
        'Individual Response Time for the Production tasks', x=.5, y=.9975,
        size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_folder,
                             'production_individual_isi_responsetime.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def ginput_resize(audio_beat, audio_interval, visual_beat, visual_interval):
    # Inputs shape (n_subjects, n_isi, n_trials)

    # Resize numpy arrays when there is less trials per isi because the participant
    # only did the behavioral sessions
    nt_max = np.ravel([[np.array(isi).shape[0] for isi in isis]
                       for isis in audio_beat]).max()

    rsi_audio_beat = [
        [np.append(
            ab_trials, np.repeat(np.nan, nt_max - len(ab_trials))).tolist()
         if len(ab_trials) < nt_max else ab_trials for ab_trials in ab_isis]
        for ab_isis in audio_beat]
    rsi_audio_interval = [
        [np.append(
            ai_trials, np.repeat(np.nan, nt_max - len(ai_trials))).tolist()
         if len(ai_trials) < nt_max else ai_trials for ai_trials in ai_isis]
        for ai_isis in audio_interval]
    rsi_visual_beat = [
        [np.append(
            vb_trials, np.repeat(np.nan, nt_max - len(vb_trials))).tolist()
         if len(vb_trials) < nt_max else vb_trials for vb_trials in vb_isis]
        for vb_isis in visual_beat]
    rsi_visual_interval = [
        [np.append(
            vi_trials, np.repeat(np.nan, nt_max - len(vi_trials))).tolist()
         if len(vi_trials) < nt_max else vi_trials for vi_trials in vi_isis]
        for vi_isis in visual_interval]

    return (rsi_audio_beat, rsi_audio_interval, rsi_visual_beat,
            rsi_visual_interval)


def ginput_reshape(audio_beat, audio_interval, visual_beat, visual_interval):
    # Reshape (n_subjects, n_isi, n_trials) --> (n_isi, n_subjects*n_trials)

    # Swap n_subjects with n_isi
    s_audio_beat = np.swapaxes(audio_beat, 0, 1)
    s_audio_interval = np.swapaxes(audio_interval, 0, 1)
    s_visual_beat = np.swapaxes(visual_beat, 0, 1)
    s_visual_interval = np.swapaxes(visual_interval, 0, 1)

    # Reshape
    rsh_audio_beat = np.reshape(
        s_audio_beat,
        (s_audio_beat.shape[0],
         s_audio_beat.shape[1]*s_audio_beat.shape[2]))

    rsh_audio_interval = np.reshape(
        s_audio_interval,
        (s_audio_interval.shape[0],
         s_audio_interval.shape[1]*s_audio_interval.shape[2]))

    rsh_visual_beat = np.reshape(
        s_visual_beat,
        (s_visual_beat.shape[0],
         s_visual_beat.shape[1]*s_visual_beat.shape[2]))

    rsh_visual_interval = np.reshape(
        s_visual_interval,
        (s_visual_interval.shape[0],
         s_visual_interval.shape[1]*s_visual_interval.shape[2]))

    return (rsh_audio_beat, rsh_audio_interval, rsh_visual_beat,
            rsh_visual_interval)


def plot_violin(audio_beat, audio_interval,
                visual_beat, visual_interval,
                isi1s, ylim_b, ylim_t, y_label,
                title, output_folder, fname):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.1, right=.98, bottom=.15, wspace=.075)

    for i, (isi_audio_beat, isi_audio_interval) in enumerate(
            zip(audio_beat, audio_interval)):
        pos_ab = [i*2 - .4]
        pos_ai = [i*2 + .4]

        # Remove NaNs from data if they exist
        isi_audio_beat = isi_audio_beat[~np.isnan(isi_audio_beat)]
        isi_audio_interval = isi_audio_interval[~np.isnan(isi_audio_interval)]

        v1_ab = ax1.violinplot(isi_audio_beat, pos_ab, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        v1_ai = ax1.violinplot(isi_audio_interval, pos_ai, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        customize_vplot(isi_audio_beat, ax1, pos_ab)
        customize_vplot(isi_audio_interval, ax1, pos_ai)

        for vab in v1_ab['bodies']:
            vab.set_facecolor('tab:blue')
            vab.set_edgecolor('black')
            vab.set_alpha(1)

        for vai in v1_ai['bodies']:
            vai.set_facecolor('tab:orange')
            vai.set_edgecolor('black')
            vai.set_alpha(1)

        labels = []
        cb = vab.get_facecolor()
        ci = vai.get_facecolor()
        labels.append((mpatches.Patch(color=cb), 'Beat'))
        labels.append((mpatches.Patch(color=ci), 'Interval'))

        v1_ab['cmaxes'].set_color('black')
        v1_ab['cmins'].set_color('black')
        v1_ab['cbars'].set_color('black')
        v1_ab['cmeans'].set_color('black')

        v1_ai['cmaxes'].set_color('black')
        v1_ai['cmins'].set_color('black')
        v1_ai['cbars'].set_color('black')
        v1_ai['cmeans'].set_color('black')

    for j, (isi_visual_beat, isi_visual_interval) in enumerate(
            zip(visual_beat, visual_interval)):
        pos_vb = [j*2 - .4]
        pos_vi = [j*2 + .4]

        # Remove NaNs from data if they exist
        isi_visual_beat = isi_visual_beat[~np.isnan(isi_visual_beat)]
        isi_visual_interval = isi_visual_interval[~np.isnan(
            isi_visual_interval)]

        v2_ab = ax2.violinplot(isi_visual_beat, pos_vb, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        v2_ai = ax2.violinplot(isi_visual_interval, pos_vi, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        customize_vplot(isi_visual_beat, ax2, pos_vb)
        customize_vplot(isi_visual_interval, ax2, pos_vi)

        for vab in v2_ab['bodies']:
            vab.set_facecolor('tab:blue')
            vab.set_edgecolor('black')
            vab.set_alpha(1)

        for vai in v2_ai['bodies']:
            vai.set_facecolor('tab:orange')
            vai.set_edgecolor('black')
            vai.set_alpha(1)

        v2_ab['cmaxes'].set_color('black')
        v2_ab['cmins'].set_color('black')
        v2_ab['cbars'].set_color('black')
        v2_ab['cmeans'].set_color('black')

        v2_ai['cmaxes'].set_color('black')
        v2_ai['cmins'].set_color('black')
        v2_ai['cbars'].set_color('black')
        v2_ai['cmeans'].set_color('black')

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Label x-axis
    x_labels = [str(standard) for standard in standards]
    pos = np.arange(len(standards))*2
    ax1.set_xticks(pos, x_labels)
    ax2.set_xticks(pos, x_labels)

    # Set limits of y-axis
    ax1.set_ylim(bottom=ylim_b, top=ylim_t)
    ax2.set_ylim(bottom=ylim_b, top=ylim_t)
    # Set y label
    ax1.set_ylabel(y_label, labelpad=.5)
    # Remove y frame, labels and spines of second plot
    ax2.spines['left'].set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    # Title of each plot
    ax1.set_title('Auditory Conditions', fontweight='semibold', size=10,
                  y=-.175)
    ax2.set_title('Visual Conditions', fontweight='semibold', size=10,
                  y=-.175)

    # Add legend
    ax1.legend(*zip(*labels), loc='best', frameon=False)
    fig.text(.75, 0.84, 'white circle: median', size=8)
    fig.text(.75, 0.8, 'hline: mean', size=8)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))


def dataframe(sync_audio_beat, sync_audio_interval, sync_visual_beat,
              sync_visual_interval, stand_numbers, output_dir):
    # Inputs shape (n_subjects, n_isi, n_trials)

    # Compute mean of trials synchronies for each standard and subject
    sync_audio_beat = np.nanmean(sync_audio_beat, axis=2)
    sync_audio_interval = np.nanmean(sync_audio_interval, axis=2)
    sync_visual_beat = np.nanmean(sync_visual_beat, axis=2)
    sync_visual_interval = np.nanmean(sync_visual_interval, axis=2)

    conditions_names = np.array(['beat', 'interval'])
    modalities_names = np.array(['audio', 'visual'])

    # Flatten the synchronies arrays
    sync_audio_beat_flatten = np.ravel(sync_audio_beat)
    sync_audio_interval_flatten = np.ravel(sync_audio_interval)
    sync_visual_beat_flatten = np.ravel(sync_visual_beat)
    sync_visual_interval_flatten = np.ravel(sync_visual_interval)

    # Stack synchronies in one single array
    asynchronies = np.hstack((sync_audio_beat_flatten,
                              sync_audio_interval_flatten,
                              sync_visual_beat_flatten,
                              sync_visual_interval_flatten))

    # ## Standards column
    standards_allsubjects = np.tile(stand_numbers, sync_audio_beat.shape[0])
    standards = np.tile(standards_allsubjects,
                        conditions_names.shape[0] * modalities_names.shape[0])

    # ## Subjects column
    itag = ['sub-%02d' % s for s in SUBJECTS]
    stand_allsubjects = np.repeat(itag, stand_numbers.shape[0])
    subjects = np.tile(
        stand_allsubjects,
        conditions_names.shape[0] * modalities_names.shape[0])

    # ## Modality column
    modalities_stack = np.repeat(modalities_names, conditions_names.shape[0])
    modalities = np.repeat(modalities_stack, stand_allsubjects.shape[0])

    # ## Conditions column
    conditions_stack = np.tile(conditions_names, modalities_names.shape[0])
    conditions = np.repeat(conditions_stack, stand_allsubjects.shape[0])

    # ## Build tables and dataframes
    table = np.vstack((asynchronies, standards,
                       subjects, modalities, conditions)).T

    df = pd.DataFrame(table, columns=['Asynchronies', 'Standard', 'Subject',
                                      'Modality', 'Condition'])
    df['Asynchronies'] = df['Asynchronies'].apply(pd.to_numeric)

    output_folder = os.path.join(output_dir, 'anovas')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save dataframe
    outpath = os.path.join(output_folder, 'df_asynchronies.tsv')
    df.to_csv(outpath, index=False, sep='\t')

    return df


def threeway_repanova(df, output_dir):
    # Create AnovaRM object
    model = AnovaRM(data=df, depvar='Asynchronies', subject='Subject',
                    within=['Modality', 'Condition', 'Standard'])

    # Run the 3-way repeated measures ANOVA
    results = model.fit()

    # Create output_folder, if it does not exist
    output_folder = os.path.join(output_dir, 'anovas/threeway')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save ANOVA results in a TSV file
    results.anova_table.to_csv(os.path.join(output_folder,
                                            'threeway_anova_results.tsv'),
                               sep='\t')

    # Perform pairwise Tukey HSD tests
    posthoc_modality = pairwise_tukeyhsd(df['Asynchronies'], df['Modality'],
                                         alpha=0.05)
    posthoc_condition = pairwise_tukeyhsd(df['Asynchronies'], df['Condition'],
                                          alpha=0.05)
    posthoc_standard = pairwise_tukeyhsd(df['Asynchronies'], df['Standard'],
                                         alpha=0.05)

    with open(os.path.join(output_folder, 'posthoc_modality.tsv'), 'w') as fm:
        fm.write(posthoc_modality.summary().as_csv(sep='\t'))

    with open(os.path.join(output_folder, 'posthoc_condition.tsv'), 'w') as fc:
        fc.write(posthoc_condition.summary().as_csv(sep='\t'))

    with open(os.path.join(output_folder, 'posthoc_standard.tsv'), 'w') as fs:
        fs.write(posthoc_standard.summary().as_csv(sep='\t'))

    # Plot
    modalities = np.unique(df.Modality).tolist()
    conditions = np.unique(df.Condition).tolist()
    standards = np.unique(df.Standard).tolist()

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            fig = plt.figure(figsize=(8, 4))

        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.1 + m*.45, .15, .43, .75])

        x_labels = [str(st) for st in standards]
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars

        asynch_beat = [df[df.Modality==modality][df.Condition=='beat'][
            df.Standard==st].Asynchronies.values.tolist() for st in standards]

        asynch_interval = [df[df.Modality==modality][df.Condition=='interval'][
            df.Standard==st].Asynchronies.values.tolist() for st in standards]

        beat = ax.boxplot(asynch_beat,
                          bootstrap=100,
                          positions=np.arange(len(x))*2. - width,
                          widths=0.6,
                          flierprops={'marker': '', 'markersize': 5},
                          patch_artist=True)
        interval = ax.boxplot(asynch_interval,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. + width,
                              widths=0.6,
                              flierprops={'marker': '', 'markersize': 5},
                              patch_artist=True)

        # Fill boxes with colors
        colors = ['b', 'y']
        for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
            patch1.set_facecolor(colors[0])
            patch2.set_facecolor(colors[1])

        # Set ticks labels in x-axis
        ax.set_xticks(x*2., x_labels)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # For the first plot,
        if m == 0:
            # Place legend
            ax.legend([beat["boxes"][0], interval["boxes"][0]],
                      ['Beat', 'Interval'],
                      loc='upper right', frameon=False,
                      prop={'size': 12})
            # Title of each plot
            ax.set_title('Auditory Conditions', fontweight='semibold',
                         size=10, y=.95)
            # Set name for y-axis
            ax.set_ylabel('Group Signed-Asynchronies')
        # For the second plot
        else:
            # ... remove y frame on the left
            ax.spines['left'].set_visible(False)
            # ... remove labels and ticks
            ax.axes.get_yaxis().set_visible(False)
            # Title of each plot
            ax.set_title('Visual Conditions', fontweight='semibold', size=10,
                         y=.95)

        # Set limits of ticks in y axis
        plt.ylim([-.3, .45])

        # Set name for x-axis
        fig.text(.45, .025, 'Standards (ms)', size=12)

    # Title
    plt.suptitle('Descriptive Stats of Group Signed-Asynchronies ' +
                 'for 3-way Repeated Measures ANOVA',
                 x=.5, y=.98, size=12, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(output_folder, 'threeway_boxplot.pdf'))


def plot_pttest_isi(audio_beat, audio_interval, visual_beat, visual_interval,
                    pval_audio, pval_visual,
                    isi1s, y, ylim_b, ylim_t, yshift,
                    title, output_folder, fname):

    # Concatenate data
    data_audio = [np.append(audio_beat[j], audio_interval[j]).tolist()
                  for j in np.arange(len(audio_beat))]
    data_visual = [np.append(visual_beat[j], visual_interval[j]).tolist()
                   for j in np.arange(len(visual_beat))]

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.12, right=.99, bottom=.15, wspace=.075)

    # Prepare the data
    x = 'Standard'
    z = 'Conditions'
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            n_isi = np.array(data_audio).shape[0]
            n_repeat = np.array(data_audio).shape[1]
            standard = [np.repeat(str(isi1), n_repeat) for isi1 in isi1s]
            conditions = [
                np.append(np.repeat('Beat', n_repeat / 2),
                np.repeat('Interval', n_repeat / 2)).tolist()
                for j in np.arange(n_isi)]
            data_list = data_audio
            pvalue = pval_audio
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            n_isi = np.array(data_visual).shape[0]
            n_repeat = np.array(data_visual).shape[1]
            standard = [np.repeat(str(isi1), n_repeat) for isi1 in isi1s]
            conditions = [
                np.append(np.repeat('Beat', n_repeat / 2),
                np.repeat('Interval', n_repeat / 2)).tolist()
                for j in np.arange(n_isi)]
            data_list = data_visual
            pvalue = pval_visual
            x_label = 'Visual Conditions'
        d = {x: np.ravel(standard),
             y: np.ravel(data_list),
             z: np.ravel(conditions)}
        df = pd.DataFrame(data=d)

        # Create bar plot
        sns.barplot(ax=ax[m],
            x=x,
            y=y,
            hue=z,
            data=df,
            estimator=np.mean,
            ci=95, # 1.96 * standard error (95% confidence interval)
            errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

        # Annotate
        pairs = tuple([[(str(isi1), 'Beat'), (str(isi1), 'Interval')]
                       for isi1 in isi1s])
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y, hue=z)
        annotator.configure(test=None,
                            text_format="star", # text_format="simple"
                            # test_short_name="pttest", # if former is "simple"
                            fontsize=10.)

        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        # Remove frame of legend
        ax[m].legend(frameon=False)

        # Change position of legend
        sns.move_legend(ax[m], "upper right")

        # For the second (right) plot, ...
        if m ==1:
            # ... remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # ... remove y frame
            ax[m].spines['left'].set_visible(False)
        else:
            # ... remove legend
            ax[m].legend([],[], frameon=False)

        # Change x label
        ax[m].set_xlabel(x_label, fontweight='semibold', labelpad=20)

        # Display means rounded to two decimals on the top
        # for p in ax[m].patches:
        #     ax[m].text(p.get_x() + p.get_width()/2.,
        #                p.get_height() + np.sign(p.get_height()) * yshift,
        #                '{:.2e}'.format(p.get_height()), fontsize=2.5,
        #                fontweight='bold', color='black', ha='center',
        #                va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .4)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # Common x-label
    fig.text(.555, .055, 'Standards (ms)', ha='center', fontsize=10)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))


def plot_pttest(data_audio, data_visual,
                pval_audio_bi, pval_audio_br, pval_audio_ir,
                pval_visual_bi, pval_visual_br, pval_visual_ir,
                y, ylim_b, ylim_t, yshift, title, this_dir, output_folder,
                fname):

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25)

    # Define subplot of bar charts and its position in the fig
    # plt.axes([left, bottom, width, height])
    # ax = plt.axes([.225, .145, .65, .65])

    # Prepare the data
    x = 'Conditions'
    pval_audio = [pval_audio_bi, pval_audio_br, pval_audio_ir]
    pval_visual = [pval_visual_bi, pval_visual_br, pval_visual_ir]
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            data_list = data_audio
            pvalue = pval_audio
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            data_list = data_visual
            pvalue = pval_visual
            x_label = 'Visual Conditions'
        conditions = np.repeat('Beat', len(data_list) / 3).tolist() + \
            np.repeat('Interval', len(data_list) / 3).tolist() + \
            np.repeat('Random', len(data_list) / 3).tolist()
        d = {x: conditions, y: data_list}
        df = pd.DataFrame(data=d)

        # Create bar plot
        sns.barplot(ax=ax[m],
            x=x,
            y=y,
            data=df,
            estimator=np.mean,
            ci=95, # 1.96 * standard error (95% confidence interval)
            errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

        # Annotate
        pairs = [('Beat', 'Interval'), ('Beat', 'Random'), ('Interval', 'Random')]
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None,
                            text_format="star",
                            # test_short_name="pttest",
                            fontsize=10.)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m ==1:
            # Remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # Remove y frame
            ax[m].spines['left'].set_visible(False)
            # Change x label
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=20)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=20)

        # Display means rounded to two decimals on the top
        # ax.bar_label(ax.containers[0], padding=-50)
        # for p in ax[m].patches:
        #     ax[m].text(p.get_x() + p.get_width()/2., p.get_height() + yshift,
        #                '{:.2e}'.format(p.get_height()), fontsize=7.,
        #                color='black', ha='center', va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .7)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # Common x-label
    fig.text(.53, .055, 'Standards (ms)', ha='center', fontsize=10)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_folder, fname + '.pdf'))


def plotfit_production(x, y, y_values, yaxis_name, yname_pos, title,
                       output_folder, fname, legend_loc='lower left',
                       hline_legend=None, hline_yloc=[.4275, .435]):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.095, bottom=.11, right=.98, wspace=.175)

    colors = ['tab:blue', 'tab:orange']
    legend_labels = ['Beat', 'Interval']

    for m, modality_y in enumerate(y):
        for c, condition_y in enumerate(modality_y):
            # Linear fit
            a, b = np.polyfit(x, condition_y, deg=1)
            y_est = a * x + b
            # y_err = x.std() * \
            #     np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))

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
            # Set limits of y-axis
            y_labels = [str(yl) for yl in y_values]
            ax[m].set_yticks(y_values, y_labels, fontsize=24)
            # Add horizontal dashed line at y = 0.5
            if hline_legend:
                ax[m].axhline(0., linestyle='--', color='grey', linewidth=12,
                              alpha=.5)

        # Add legend
        if m == 0:
            ax[m].set_title('Auditory Production', weight='bold', pad=0,
                            fontsize=24)
            ax[m].legend(loc=legend_loc, frameon=False, prop={'size': 24})
        else:
            assert m == 1
            ax[m].set_title('Visual Production', weight='bold', pad=0,
                            fontsize=24)

        # Name of x-axis
        fig.text(.465, .018, 'Standards (ms)', fontsize=24)
        # Name of y-axis
        fig.text(.005, yname_pos, yaxis_name, fontsize=26, rotation=90)
        # Legends for horizontal dashed lines
        if hline_legend:
            fig.text(.355, hline_yloc[0], hline_legend, fontsize=24,
                     color='dimgrey')
            fig.text(.825, hline_yloc[1], hline_legend, fontsize=24,
                     color='dimgrey')

    # Title
    plt.suptitle(title, x=.5, y=.98, size=24, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))


def production_ancova(dependent_var, covariate, output_dir, dfname, resname,
                      modality='audio'):
    # ## Create columns of dataframe
    # Dependent var
    mean_flatten = np.ravel(dependent_var)
    # Create columns of independent (categorical) var,
    # i.e. Condition and Modality
    standval = np.tile(covariate,
                       dependent_var.shape[1] * dependent_var.shape[0])
    condtag = np.repeat(['beat', 'interval'], len(covariate))
    condval = np.tile(condtag, dependent_var.shape[0])
    modval = np.repeat(['audio', 'visual'], len(condtag))

    # Build DataFrame
    table = np.vstack((mean_flatten, standval, condval, modval)).T

    df = pd.DataFrame(
        table, columns=['Mean Error', 'Standard', 'Condition', 'Modality'])
    df['Mean Error'] = df['Mean Error'].apply(pd.to_numeric)
    df['Standard'] = df['Standard'].apply(pd.to_numeric)

    df_modality = df[df.Modality == modality]

    aoc_modality = pg.ancova(data=df_modality, dv='Mean Error',
                             covar='Standard', between='Condition')

    # Save dataframe and ANCOVA's results
    output_folder = os.path.join(output_dir, 'ancova')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save dataframe
    df_outpath = os.path.join(output_folder, dfname + '_' + modality + '.tsv')
    df.to_csv(df_outpath, index=False, sep='\t')

    # Save ANCOVA results
    res_outpath = os.path.join(output_folder, resname + '_' + modality + '.tsv')
    aoc_modality.to_csv(res_outpath, index=False, sep='\t')

    return aoc_modality


# %%
# =========================== INPUTS ===================================

# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
#             40, 41, 42, 43, 44, 45, 46, 47]
SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
            44, 45, 46, 47]

# TASKS = ['Visual Production']

SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSTYPES = ['imaging_session']

N_TRIALS = 30

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    # ############### PRODUCTION SYNCHRONIES ###########################

    # ### Individual analysis per standard --- box plots
    ssync_audio_beat, ssync_audio_interval, ssync_visual_beat, \
        ssync_visual_interval, standards = individual_production_isi_sync(
            SUBJECTS, SESSTYPES, MAIN_DIR, RESULTS_FOLDER, 'signed',
            N_TRIALS, flatten=False)

    async_audio_beat, async_audio_interval, async_visual_beat, \
        async_visual_interval, standards = individual_production_isi_sync(
            SUBJECTS, SESSTYPES, MAIN_DIR, RESULTS_FOLDER, 'absolute',
            N_TRIALS, flatten=False)

    # ## Compute mean of asynchronies across trials per subject
    # ## for every standard (fixed-effects)

    # Signed asynchronies
    ffx_ssync_audio_beat, ffx_ssync_audio_interval, ffx_ssync_visual_beat, \
        ffx_ssync_visual_interval = ffx(ssync_audio_beat,
                                        ssync_audio_interval,
                                        ssync_visual_beat,
                                        ssync_visual_interval)

    # Absolute asynchronies
    ffx_async_audio_beat, ffx_async_audio_interval, ffx_async_visual_beat, \
        ffx_async_visual_interval = ffx(async_audio_beat,
                                        async_audio_interval,
                                        async_visual_beat,
                                        async_visual_interval)

    # ### Group Analyses per standard --- bar plots + paired t-test

    # Signed asynchronies
    _, pssync_audio = stats.ttest_rel(
        ffx_ssync_audio_beat, ffx_ssync_audio_interval,
        axis=1, alternative='two-sided')

    _, pssync_visual = stats.ttest_rel(
        ffx_ssync_visual_beat, ffx_ssync_visual_interval,
        axis=1, alternative='two-sided')

    ssync_title = 'Group Mean of Signed Asynchrony for the Production tasks'
    ssync_f = 'paired-ttest_signed_asynch'
    plot_pttest_isi(ffx_ssync_audio_beat, ffx_ssync_audio_interval,
                    ffx_ssync_visual_beat, ffx_ssync_visual_interval,
                    pssync_audio, pssync_visual,
                    standards, 'Signed Asynchrony', -.125, .3, -.039,
                    ssync_title, RESULTS_FOLDER, ssync_f)

    # Absolute asynchronies
    _, pasync_audio = stats.ttest_rel(
        ffx_async_audio_beat, ffx_async_audio_interval,
        axis=1, alternative='two-sided')

    _, pasync_visual = stats.ttest_rel(
        ffx_async_visual_beat, ffx_async_visual_interval,
        axis=1, alternative='two-sided')

    async_title = 'Group Mean of Absolute Asynchrony for the Production tasks'
    async_f = 'paired-ttest_absolute_asynch'
    plot_pttest_isi(ffx_async_audio_beat, ffx_async_audio_interval,
                    ffx_async_visual_beat, ffx_async_visual_interval,
                    pasync_audio, pasync_visual,
                    standards, 'Absolute Asynchrony', -0., .3, -.04,
                    async_title, RESULTS_FOLDER, async_f)

    # ### Group Analyses per standard --- violin plots

    # ## Resize&Reshape

    # Signed asynchronies
    rsized_ssync_audio_beat, rsized_ssync_audio_interval, \
        rsized_ssync_visual_beat, rsized_ssync_visual_interval = \
            ginput_resize(ssync_audio_beat, ssync_audio_interval,
                          ssync_visual_beat, ssync_visual_interval)
    rs_ssync_audio_beat, rs_ssync_audio_interval, \
        rs_ssync_visual_beat, rs_ssync_visual_interval = \
            ginput_reshape(rsized_ssync_audio_beat,
                           rsized_ssync_audio_interval,
                           rsized_ssync_visual_beat,
                           rsized_ssync_visual_interval)

    # Absolute asynchronies
    rsized_async_audio_beat, rsized_async_audio_interval, \
        rsized_async_visual_beat, rsized_async_visual_interval = \
            ginput_resize(async_audio_beat, async_audio_interval,
                          async_visual_beat, async_visual_interval)
    rs_async_audio_beat, rs_async_audio_interval, \
        rs_async_visual_beat, rs_async_visual_interval = \
            ginput_reshape(rsized_async_audio_beat,
                           rsized_async_audio_interval,
                           rsized_async_visual_beat,
                           rsized_async_visual_interval)

    # Signed asynchronies
    plot_violin(
        rs_ssync_audio_beat, rs_ssync_audio_interval,
        rs_ssync_visual_beat, rs_ssync_visual_interval,
        standards, -1., 4., 'Asynchrony',
        'Group Distribution of Signed-Asynchrony for the Production Tasks',
        RESULTS_FOLDER,
        'production_groupviolin_signed_asynch')

    # Absolute asynchronies
    plot_violin(
        rs_async_audio_beat, rs_async_audio_interval,
        rs_async_visual_beat, rs_async_visual_interval,
        standards, -.05, 4., 'Asynchrony',
        'Group Distribution of Absolute-Asynchrony for the Production Tasks',
        RESULTS_FOLDER,
        'production_groupviolin_absolute_asynch')

    # ### Regression of mean and std errors for signed asychronies ###

    # Compute Group Mean plus Std of Error and stack
    mean_ffx_ssync_ab = np.mean(ffx_ssync_audio_beat, axis=1).tolist()
    mean_ffx_ssync_ai = np.mean(ffx_ssync_audio_interval, axis=1).tolist()
    mean_ffx_ssync_vb = np.mean(ffx_ssync_visual_beat, axis=1).tolist()
    mean_ffx_ssync_vi = np.mean(ffx_ssync_visual_interval, axis=1).tolist()

    std_ffx_ssync_ab = np.std(ffx_ssync_audio_beat, axis=1).tolist()
    std_ffx_ssync_ai = np.std(ffx_ssync_audio_interval, axis=1).tolist()
    std_ffx_ssync_vb = np.std(ffx_ssync_visual_beat, axis=1).tolist()
    std_ffx_ssync_vi = np.std(ffx_ssync_visual_interval, axis=1).tolist()

    # Plot
    mean_ffx_ssync_data = [[mean_ffx_ssync_ab] + [mean_ffx_ssync_ai]] + \
        [[mean_ffx_ssync_vb] + [mean_ffx_ssync_vi]]
    mean_ffx_ssync_std = [[std_ffx_ssync_ab] + [std_ffx_ssync_ai]] + \
        [[std_ffx_ssync_vb] + [std_ffx_ssync_vi]]

    plotfit_production(
        standards, mean_ffx_ssync_data, np.around(np.arange(-.1, .2, .05), 2),
        'Mean of Signed Asynchrony', .225,
        'Mean of Signed Asynchrony for every Standard',
        RESULTS_FOLDER, 'mean_ssynch_fit_production',
        hline_legend=r'$RT=Standard$', hline_yloc=[.41, .41])
    plotfit_production(
        standards, mean_ffx_ssync_std, np.around(np.arange(.06, .14, .02), 3),
        'SD of Signed Asynchrony', .225,
        'Standard Deviation (SD) of of Signed Asynchrony ' + \
        'for every Standard', RESULTS_FOLDER,
        'std_ssynch_fit_production', legend_loc='upper left')

    # Compute three-way ANOVA for signed asychronies
    db = dataframe(rsized_ssync_audio_beat, rsized_ssync_audio_interval,
                   rsized_ssync_visual_beat, rsized_ssync_visual_interval,
                   standards, RESULTS_FOLDER)
    threeway_repanova(db, RESULTS_FOLDER)


    # # # # # ############## PRODUCTION RESPONSE TIME ########################

    # ### Individual analysis per standard --- box plots ###
    rtsprod_audio_beat, rtsprod_audio_interval, rtsprod_visual_beat, \
        rtsprod_visual_interval, standards = individual_production_isi_rts(
            SUBJECTS, SESSTYPES, MAIN_DIR
            , RESULTS_FOLDER, N_TRIALS,
            flatten=False)

    # ### Group Analyses per standard --- bar plots + paired t-test ###
    # Compute mean of response time across trials per subject
    # for every standard
    ffx_rtsprod_audio_beat, ffx_rtsprod_audio_interval, \
        ffx_rtsprod_visual_beat, ffx_rtsprod_visual_interval = ffx(
            rtsprod_audio_beat, rtsprod_audio_interval,
            rtsprod_visual_beat, rtsprod_visual_interval)

    # Compute Stats
    _, prtprod_audio = stats.ttest_rel(
        ffx_rtsprod_audio_beat, ffx_rtsprod_audio_interval,
        axis=1, alternative='two-sided')

    _, prtprod_visual = stats.ttest_rel(
        ffx_rtsprod_visual_beat, ffx_rtsprod_visual_interval,
        axis=1, alternative='two-sided')

    # Plot
    rtprod_title = 'Group Mean of Response Time for the Production tasks'
    rtprod_f = 'paired-ttest_responsetime_production'
    plot_pttest_isi(ffx_rtsprod_audio_beat, ffx_rtsprod_audio_interval,
                    ffx_rtsprod_visual_beat, ffx_rtsprod_visual_interval,
                    prtprod_audio, prtprod_visual,
                    standards, 'Response Time (ms)', 0., 900., -100.,
                    rtprod_title, RESULTS_FOLDER, rtprod_f)

    # ### Group Analyses per standard --- violin plots ###
    # Resize
    rsized_rtsprod_audio_beat, rsized_rtsprod_audio_interval, \
        rsized_rtsprod_visual_beat, rsized_rtsprod_visual_interval = \
            ginput_resize(rtsprod_audio_beat, rtsprod_audio_interval,
                          rtsprod_visual_beat, rtsprod_visual_interval)
    # Reshape
    rs_rtsprod_audio_beat, rs_rtsprod_audio_interval, \
        rs_rtsprod_visual_beat, rs_rtsprod_visual_interval = \
            ginput_reshape(rsized_rtsprod_audio_beat,
                           rsized_rtsprod_audio_interval,
                           rsized_rtsprod_visual_beat,
                           rsized_rtsprod_visual_interval)

    # Plot
    plot_violin(
        rs_rtsprod_audio_beat, rs_rtsprod_audio_interval,
        rs_rtsprod_visual_beat, rs_rtsprod_visual_interval,
        standards, 0., 2250., 'Response Time (ms)',
        'Group Distribution of Response Time for the Production Tasks',
        RESULTS_FOLDER,
        'production_groupviolin_responsetime')

    # ### Regression of mean and std errors ###

    # Signed synch
    error_rtsprod_audio_beat = [
        [(ab - standards[s]).tolist() for s, ab in enumerate(rts_ab)]
        for rts_ab in rtsprod_audio_beat]
    error_rtsprod_audio_interval = [
        [(ai - standards[s]).tolist() for s, ai in enumerate(rts_ai)]
        for rts_ai in rtsprod_audio_interval]
    error_rtsprod_visual_beat = [
        [(vb - standards[s]).tolist() for s, vb in enumerate(rts_vb)]
        for rts_vb in rtsprod_visual_beat]
    error_rtsprod_visual_interval = [
        [(vi - standards[s]).tolist() for s, vi in enumerate(rts_vi)]
        for rts_vi in rtsprod_visual_interval]

    # Absolute asynch
    abs_error_rtsprod_audio_beat = [
        [[np.absolute(z_ab).tolist() for z_ab in y_ab] for y_ab in x_ab]
        for x_ab in error_rtsprod_audio_beat]
    abs_error_rtsprod_audio_interval = [
        [[np.absolute(z_ai).tolist() for z_ai in y_ai] for y_ai in x_ai]
        for x_ai in error_rtsprod_audio_interval]
    abs_error_rtsprod_visual_beat = [
        [[np.absolute(z_vb).tolist() for z_vb in y_vb] for y_vb in x_vb]
        for x_vb in error_rtsprod_visual_beat]
    abs_error_rtsprod_visual_interval = [
        [[np.absolute(z_vi).tolist() for z_vi in y_vi] for y_vi in x_vi]
        for x_vi in error_rtsprod_visual_interval]

    # Fixed-effects
    ffxerr_rtsprod_audio_beat, ffxerr_rtsprod_audio_interval, \
        ffxerr_rtsprod_visual_beat, ffxerr_rtsprod_visual_interval = ffx(
            error_rtsprod_audio_beat, error_rtsprod_audio_interval,
            error_rtsprod_visual_beat, error_rtsprod_visual_interval)

    ffxabserr_rtsprod_audio_beat, ffxabserr_rtsprod_audio_interval, \
        ffxabserr_rtsprod_visual_beat, ffxabserr_rtsprod_visual_interval = ffx(
            abs_error_rtsprod_audio_beat, abs_error_rtsprod_audio_interval,
            abs_error_rtsprod_visual_beat, abs_error_rtsprod_visual_interval)

    # Compute Group Mean plus Std of Error and stack
    mean_ffx_ab = np.mean(ffxerr_rtsprod_audio_beat, axis=1).tolist()
    mean_ffx_ai = np.mean(ffxerr_rtsprod_audio_interval, axis=1).tolist()
    mean_ffx_vb = np.mean(ffxerr_rtsprod_visual_beat, axis=1).tolist()
    mean_ffx_vi = np.mean(ffxerr_rtsprod_visual_interval, axis=1).tolist()

    mean_abs_ffx_ab = np.mean(ffxabserr_rtsprod_audio_beat, axis=1).tolist()
    mean_abs_ffx_ai = np.mean(ffxabserr_rtsprod_audio_interval, axis=1).tolist()
    mean_abs_ffx_vb = np.mean(ffxabserr_rtsprod_visual_beat, axis=1).tolist()
    mean_abs_ffx_vi = np.mean(ffxabserr_rtsprod_visual_interval,
                              axis=1).tolist()

    std_ffx_ab = np.std(ffxerr_rtsprod_audio_beat, axis=1).tolist()
    std_ffx_ai = np.std(ffxerr_rtsprod_audio_interval, axis=1).tolist()
    std_ffx_vb = np.std(ffxerr_rtsprod_visual_beat, axis=1).tolist()
    std_ffx_vi = np.std(ffxerr_rtsprod_visual_interval, axis=1).tolist()

    std_abs_ffx_ab = np.std(ffxabserr_rtsprod_audio_beat, axis=1).tolist()
    std_abs_ffx_ai = np.std(ffxabserr_rtsprod_audio_interval, axis=1).tolist()
    std_abs_ffx_vb = np.std(ffxabserr_rtsprod_visual_beat, axis=1).tolist()
    std_abs_ffx_vi = np.std(ffxabserr_rtsprod_visual_interval, axis=1).tolist()

    # Plot
    mean_ffx_data = [
        [mean_ffx_ab] + [mean_ffx_ai]] + [[mean_ffx_vb] + [mean_ffx_vi]]
    mean_ffx_std = [[std_ffx_ab] + [std_ffx_ai]] + [[std_ffx_vb] + [std_ffx_vi]]

    mean_abs_ffx_data = [[mean_abs_ffx_ab] + [mean_abs_ffx_ai]] + \
        [[mean_abs_ffx_vb] + [mean_abs_ffx_vi]]
    mean_abs_ffx_std = [[std_abs_ffx_ab] + [std_abs_ffx_ai]] + \
        [[std_abs_ffx_vb] + [std_abs_ffx_vi]]

    plotfit_production(
        standards, mean_ffx_data, np.linspace(-60, 90, 6),
        'RT-Difference Mean (ms)', .225,
        'Mean of Response-Time (RT) Difference for every Standard',
        RESULTS_FOLDER, 'mean-err_production',
        hline_legend=r'$RT=Standard$')
    plotfit_production(
        standards, mean_ffx_std, np.linspace(30, 70, 6),
        'RT-Difference SD (ms)', .225,
        'Standard Deviation (SD) of Response-Time (RT) Difference ' + \
        'for every Standard', RESULTS_FOLDER,
        'std-err_production')

    plotfit_production(
        standards, mean_abs_ffx_data, np.linspace(-60, 140, 6),
        'Absolute RT-Difference Mean (ms)', .125,
        'Mean of Absolute Response-Time (RT) Difference for every Standard',
        RESULTS_FOLDER, 'mean-abserr_production',
        hline_legend=r'$RT=Standard$')
    plotfit_production(
        standards, mean_abs_ffx_std, np.linspace(30, 70, 6),
        'Absolute RT-Difference SD (ms)', .125,
        'Standard Deviation (SD) of Absolute Response-Time (RT) Difference' + \
        ' for every Standard', RESULTS_FOLDER,
        'std-abserr_production')

    # Compute ANCOVAs
    # Stack multidimensional numpy array to produce a dataframe
    mean_ffx_data = np.array(mean_ffx_data)
    mean_ffx_std = np.array(mean_ffx_std)

    aoc_mean_audio = production_ancova(mean_ffx_data, standards,
                                       RESULTS_FOLDER,
                                       'df_ancova_mean-rtprod',
                                       'res_ancova_mean-rtprod',
                                       modality='audio')
    aoc_mean_visual = production_ancova(mean_ffx_data, standards,
                                        RESULTS_FOLDER,
                                        'df_ancova_mean-rtprod',
                                        'res_ancova_mean-rtprod',
                                        modality='visual')

    aoc_std_audio = production_ancova(mean_ffx_std, standards,
                                      RESULTS_FOLDER,
                                      'df_ancova_sd-rtprod',
                                      'res_ancova_sd-rtprod',
                                      modality='audio')
    aoc_std_visual = production_ancova(mean_ffx_std, standards,
                                       RESULTS_FOLDER,
                                       'df_ancova_sd-rtprod',
                                       'res_ancova_sd-rtprod',
                                       modality='visual')
