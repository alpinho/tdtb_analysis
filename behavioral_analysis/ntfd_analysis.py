"""
Analysis of behavioral data for the NTFD Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: February 2023
Last update: August 2023

Compatibility: Python 3.10.4
"""

import sys
import os

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn as sns

import warnings

from scipy import stats, optimize, special
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from statannotations.Annotator import Annotator

# setting path
sys.path.append('../')
# importing
from utils import parse_logfile, customize_vplot, change_width

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# %%
# ======================== MAIN FUNCTIONS ==============================


def ntfd_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            condition = datum[4]
            theoretical_isi1 = int(data[dt-2][8])
            if datum[11] in ['o', 'p', 'b', 'y']:
                rt = int(data[dt-1][7]) + int(datum[10])
            elif datum[10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, rt])

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


def success_trialtype_filter(data):
    trial_beat = [dt for dt in data if dt[4][:4] == 'beat']
    trial_interval = [dt for  dt in data if dt[4][:8] == 'interval']
    trial_random = [dt for dt in data if dt[4][:6] == 'random']

    return trial_beat, trial_interval, trial_random


def success(data, subject):
    if subject == 4:
        high_cir = ['o', 'y']
        low_tri = ['p', 'b']
    else:
        high_cir = ['o', 'b']
        low_tri = ['p', 'y']
    scores = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            answer = datum[11]
            dstimulus = data[dt-1][5]
            if answer in high_cir and dstimulus in ['beep_880hz', 'circle']:
                scores.append(1)
            elif answer in low_tri and dstimulus in ['beep_220hz', 'triangle']:
                scores.append(1)
            elif answer == 'None':
                scores.append(np.nan)
            else:
                scores.append(0)

    # Replace missing values (nan's) by median of the all sample
    if np.any(np.isnan(scores)):
        missval = np.nanmedian(scores)
        scores = np.where(np.isnan(scores), missval, scores).tolist()

    return round(np.sum(scores)/len(scores), 3)


def resize_arrays(arr):
    """
    Resize numpy arrays when there is less trials per isi because
    the participant only did the behavioral sessions
    """
    maxlength = np.amax([np.array(arr0).shape[0] for arr0 in arr])
    new_arr = [
        np.append(arr0, np.repeat(np.nan, maxlength - len(arr0))).tolist()
        if len(arr0) < maxlength else arr0 for arr0 in arr]

    return new_arr

def individual_ntfd_rts(subjects, this_dir, output_dir, sesstype, n_trials,
                        flatten=True,
                        tasks=['Auditory No-Temporal Feature Discrimination',
                               'Visual No-Temporal Feature Discrimination']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_random_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    allsub_random_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials)

            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)
            beat_trials, interval_trials, random_trials = \
                filter_trialtype(trials, 'ntfd')

            # ############## Extract RT's ######################
            beat_trials = np.array([bt[1] for bt in beat_trials])
            interval_trials = np.array([it[1] for it in interval_trials])
            random_trials = np.array([rt[1] for rt in random_trials])

            # Replace missing values (nan's) by median of the all sample
            if np.any(np.isnan(beat_trials)):
                miss_bval = np.nanmedian(beat_trials)
                beat_trials = np.where(np.isnan(beat_trials),
                                       miss_bval, beat_trials)

            if np.any(np.isnan(interval_trials)):
                miss_ival = np.nanmedian(interval_trials)
                interval_trials = np.where(np.isnan(interval_trials),
                                           miss_ival, interval_trials)

            if np.any(np.isnan(random_trials)):
                miss_rval = np.nanmedian(random_trials)
                random_trials = np.where(np.isnan(random_trials),
                                         miss_rval, random_trials)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 100))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .965 - s*.025, .31, .02])

            labels = ['beat', 'interval', 'random']
            x = [.2, .4, .6]  # the label locations
            width = .175  # the width of the bars
            ntfd_plt = ax.bar(x,
                              [round(beat_trials.mean(0), 2),
                               round(interval_trials.mean(0), 2),
                               round(random_trials.mean(0), 2)],
                              width=width, alpha=.5,
                              color=['tab:blue', 'tab:orange', 'green'],
                              yerr=[round(beat_trials.std(0), 2),
                                    round(interval_trials.std(0), 2),
                                    round(random_trials.std(0), 2)],
                              error_kw=dict(capsize=2), label=labels)
            # Add means values on the top of the bar
            ax.bar_label(ntfd_plt, label_type='center')
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 850.])

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', weight='bold', pad=12)
                    fig.text(.25, .98, 'Error bars: SD', fontsize=12)
                else:
                    assert t ==1
                    ax.set_title('Visual NTFD', weight='bold', pad=12)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data to compute the paired sample t-test
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(beat_trials.tolist())
                allsub_interval_audio.append(interval_trials.tolist())
                allsub_random_audio.append(random_trials.tolist())
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(beat_trials.tolist())
                allsub_interval_visual.append(interval_trials.tolist())
                allsub_random_visual.append(random_trials.tolist())

        fig.text(.07, .975 - s*.025, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.1675, .426, 'Reaction Time (ms)', ha='center', fontsize=12,
             rotation = 90)

    # Title
    plt.suptitle('Individual Mean and Standard Deviation of Reaction Time ' + \
                 'for the NTFD tasks', x=.5, y=.995, size=14, linespacing=.75)
    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir, 'ntfd_individual_rt.pdf'))

    # Resize outputs
    allsub_beat_audio = resize_arrays(allsub_beat_audio)
    allsub_interval_audio = resize_arrays(allsub_interval_audio)
    allsub_random_audio = resize_arrays(allsub_random_audio)
    allsub_beat_visual = resize_arrays(allsub_beat_visual)
    allsub_interval_visual = resize_arrays(allsub_interval_visual)
    allsub_random_visual = resize_arrays(allsub_random_visual)

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_random_audio = np.ravel(allsub_random_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()
        allsub_random_visual = np.ravel(allsub_random_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_random_audio,
            allsub_beat_visual, allsub_interval_visual, allsub_random_visual)


def individual_ntfd_isi_rts(
        subjects, this_dir, output_dir, sesstype, n_trials, n_isi_trials,
        flatten=True, tasks=['Auditory No-Temporal Feature Discrimination',
                             'Visual No-Temporal Feature Discrimination']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials, 'ntfd')

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
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(rts_beat)):
                    miss_bval = np.nanmedian(rts_beat)
                    rts_beat = np.where(np.isnan(rts_beat), miss_bval,
                                        rts_beat).tolist()
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
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(rts_interval)):
                    miss_ival = np.nanmedian(rts_interval)
                    rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                                            rts_interval).tolist()
                rt_isi1_grouped_interval.append(rts_interval)

            # ################## Plotting set 1 ########################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 160))

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
                ax.set_ylabel('Log10(Reaction Time)')

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', pad=30, weight='bold')
                    ax.legend(frameon=False, loc = 'upper left',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.26, 0.9825, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.9825, ' Mean', color='black',
                             weight='roman', size='x-small')
                else:
                    assert t == 1
                    ax.set_title('Visual NTFD', pad=30, weight='bold')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data
            diff = n_isi_trials - np.array(rt_isi1_grouped_interval).shape[1]
            if diff != 0:
                # Add missing data for subjects who have less data because of
                # the introduction of the random condition
                rt_isi1_grouped_beat = [
                    np.append(rb, np.repeat(np.median(rb), diff)).tolist()
                    for rb in rt_isi1_grouped_beat]
                rt_isi1_grouped_interval = [
                    np.append(ri, np.repeat(np.median(ri), diff)).tolist()
                    for ri in rt_isi1_grouped_interval]
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(rt_isi1_grouped_beat)
                allsub_interval_audio.append(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(rt_isi1_grouped_beat)
                allsub_interval_visual.append(rt_isi1_grouped_interval)

        fig.text(.07, .9775 - s * .02, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    plt.suptitle(
        'Individual Reaction Time for the NTFD tasks',
        x=.5, y=.9975, size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir,
                             'ntfd_individual_isi_rt.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def individual_ntfd_sucessrate(
        subjects, this_dir, output_dir, sesstype, n_trials, random=False,
        flatten=True, tasks=['Auditory No-Temporal Feature Discrimination',
                             'Visual No-Temporal Feature Discrimination']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_success_rate_audio_beat = []
    allsub_success_rate_audio_interval = []
    allsub_success_rate_audio_random  = []
    allsub_success_rate_visual_beat = []
    allsub_success_rate_visual_interval = []
    allsub_success_rate_visual_random  = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            ft_beat, ft_interval, ft_random = success_trialtype_filter(data)
            success_rate_beat = success(ft_beat, subject)
            success_rate_interval = success(ft_interval, subject)
            success_rate_random = success(ft_random, subject)

            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_success_rate_audio_beat.append(
                    success_rate_beat)
                allsub_success_rate_audio_interval.append(
                    success_rate_interval)
                allsub_success_rate_audio_random.append(
                    success_rate_random)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_success_rate_visual_beat.append(
                    success_rate_beat)
                allsub_success_rate_visual_interval.append(
                    success_rate_interval)
                allsub_success_rate_visual_random.append(
                    success_rate_random)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                if random:
                    fig = plt.figure(figsize=(8, 100))
                else:
                    fig = plt.figure(figsize=(8, 100))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            if random:
                ax = plt.axes([.235 + t*.42, .965 - s*.025, .31, .02])
                labels = ['beat', 'interval', 'random']
                x = [.2, .4, .6]  # the label locations
            else:
                ax = plt.axes([.235 + t*.42, .97 - s*.02, .31, .015])
                labels = ['beat', 'interval']
                x = [.2, .6]  # the label locations

            width = .175  # the width of the bars
            if random:
                ntfd_plt = ax.bar(x,
                                  [round(np.mean(success_rate_beat), 2),
                                   round(np.mean(success_rate_interval), 2),
                                   round(np.mean(success_rate_random), 2)],
                                  width=width, alpha=.5,
                                  color=['tab:blue', 'tab:orange', 'green'])
            else:
                ntfd_plt = ax.bar(x,
                                  [round(np.mean(success_rate_beat), 2),
                                   round(np.mean(success_rate_interval), 2)],
                                  width=width, alpha=.5,
                                  color=['tab:blue', 'tab:orange'])
            # Add means values on the top of the bar
            ax.bar_label(ntfd_plt, label_type='center')
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 1.])

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', weight='bold', pad=12)
                else:
                    assert t ==1
                    ax.set_title('Visual NTFD', weight='bold', pad=12)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if random:
            fig.text(.07, .975 - s*.025, 'Subject %d' % subject, ha='center',
                     fontsize=12, weight='bold')
        else:
            fig.text(.07, .9775 - s*.02, 'Subject %d' % subject, ha='center',
                     fontsize=12, weight='bold')
    fig.text(.1675, .426, 'Success Rate', ha='center', fontsize=12,
             rotation = 90)

    # Title
    plt.suptitle('Individual Mean of Success Rate for the NTFD tasks',
                 x=.5, y=.995, size=14, linespacing=.75)
    # plt.show()

    # Save figure
    if random:
        plt.savefig(os.path.join(this_dir, output_dir,
                                 'ntfd_individual_success_rate_random.pdf'))
    else:
        plt.savefig(os.path.join(this_dir, output_dir,
                                 'ntfd_individual_success_rate_norandom.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_success_rate_audio_beat = np.ravel(
            allsub_success_rate_audio_beat).tolist()
        allsub_success_rate_audio_interval = np.ravel(
            allsub_success_rate_audio_interval).tolist()
        allsub_success_rate_audio_random = np.ravel(
            allsub_success_rate_audio_random).tolist()
        allsub_success_rate_visual_beat = np.ravel(
            allsub_success_rate_visual_beat).tolist()
        allsub_success_rate_visual_interval = np.ravel(
            allsub_success_rate_visual_interval).tolist()
        allsub_success_rate_visual_random = np.ravel(
            allsub_success_rate_visual_random).tolist()

    return (allsub_success_rate_audio_beat,
            allsub_success_rate_audio_interval,
            allsub_success_rate_audio_random,
            allsub_success_rate_visual_beat,
            allsub_success_rate_visual_interval,
            allsub_success_rate_visual_random)


def ginput_reshape(audio_beat, audio_interval, visual_beat, visual_interval):
    # Reshape (n_subjects, n_isi, n_trials) --> (n_isi, n_subjects*n_trials)

    s_audio_beat = np.swapaxes(audio_beat, 0, 1)
    s_audio_interval = np.swapaxes(audio_interval, 0, 1)
    s_visual_beat = np.swapaxes(visual_beat, 0, 1)
    s_visual_interval = np.swapaxes(visual_interval, 0, 1)

    rs_audio_beat = np.reshape(
        s_audio_beat,
        (s_audio_beat.shape[0],
         s_audio_beat.shape[1]*s_audio_beat.shape[2]))

    rs_audio_interval = np.reshape(
        s_audio_interval,
        (s_audio_interval.shape[0],
         s_audio_interval.shape[1]*s_audio_interval.shape[2]))

    rs_visual_beat = np.reshape(
        s_visual_beat,
        (s_visual_beat.shape[0],
         s_visual_beat.shape[1]*s_visual_beat.shape[2]))

    rs_visual_interval = np.reshape(
        s_visual_interval,
        (s_visual_interval.shape[0],
         s_visual_interval.shape[1]*s_visual_interval.shape[2]))

    return rs_audio_beat, rs_audio_interval, rs_visual_beat, rs_visual_interval


def ffx(audio_beat, audio_interval, visual_beat, visual_interval):
    # Inputs shape (n_subjects, n_isi, n_trials)
    # Computes mean of elements in the third dimension
    # Swaps dimensions and returns array w/ shape (n_isi, n_subjects)

    mean_audio_beat = np.array(audio_beat).mean(2)
    mean_audio_interval = np.array(audio_interval).mean(2)
    mean_visual_beat = np.array(visual_beat).mean(2)
    mean_visual_interval = np.array(visual_interval).mean(2)

    ffx_audio_beat = np.swapaxes(mean_audio_beat, 0, 1)
    ffx_audio_interval = np.swapaxes(mean_audio_interval, 0, 1)
    ffx_visual_beat = np.swapaxes(mean_visual_beat, 0, 1)
    ffx_visual_interval = np.swapaxes(mean_visual_interval, 0, 1)

    return (ffx_audio_beat, ffx_audio_interval, ffx_visual_beat,
            ffx_visual_interval)


def plot_violin(audio_beat, audio_interval,
                visual_beat, visual_interval,
                isi1s, ylim_b, ylim_t, y_label,
                title, this_dir, output_dir, fname):

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
    plt.savefig(os.path.join(this_dir, output_dir, fname + '.pdf'))


def plot_pttest_isi(audio_beat, audio_interval, visual_beat, visual_interval,
                    pval_audio, pval_visual,
                    isi1s, y, ylim_b, ylim_t, yshift,
                    title, this_dir, output_dir, fname):

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
    plt.savefig(os.path.join(this_dir, output_dir, fname + '.pdf'))


def plot_pttest(data_audio, data_visual,
                pval_audio_bi, pval_audio_br, pval_audio_ir,
                pval_visual_bi, pval_visual_br, pval_visual_ir,
                y, ylim_b, ylim_t, yshift, title, this_dir, output_dir, fname):

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
                             labelpad=15)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=15)

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

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir, fname + '.pdf'))


def group_successrate_norand(
        israte_audio_beat, israte_audio_interval,
        israte_visual_beat, israte_visual_interval,
        p_audio_bi, p_visual_bi, this_dir, output_dir):

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25)

    # Prepare the data
    x = 'Conditions'
    y = 'Success Rate'
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            data_list = israte_audio_beat + israte_audio_interval
            pvalue = p_audio_bi
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            data_list = israte_visual_beat + israte_visual_interval
            pvalue = p_visual_bi
            x_label = 'Visual Conditions'
        conditions = np.repeat('Beat', len(data_list) / 2).tolist() + \
            np.repeat('Interval', len(data_list) / 2).tolist()
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
        pairs = [('Beat', 'Interval')]
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None,
                            # text_format="star",
                            text_format="simple",
                            test_short_name="pttest",
                            fontsize=10.)
        annotator.set_pvalues([pvalue])
        annotator.annotate()

        # Set limits of y-axis
        # ax[m].set_ylim(bottom=0., top=1.)

        if m ==1:
            # Remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # Remove y frame
            ax[m].spines['left'].set_visible(False)
            # Change x label
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=15)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=15)

        # Display means rounded to two decimals on the top
        # ax.bar_label(ax.containers[0], padding=-50)
        yshift = -.1
        for p in ax[m].patches:
            ax[m].text(p.get_x() + p.get_width()/2., p.get_height() + yshift,
                       '{:.2e}'.format(p.get_height()), fontsize=7.,
                       color='black', ha='center', va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .7)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle('Group Mean of Success Rate for the NTFD Tasks', size=10,
                 linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir,
                             'ntfd_group_success_rate_norandom.pdf'))


def group_successrate_rand(
        israte_audio_beat, israte_audio_interval, israte_audio_random,
        israte_visual_beat, israte_visual_interval, israte_visual_random,
        p_audio_bi, p_audio_br, p_audio_ir,
        p_visual_bi, p_visual_br, p_visual_ir,
        this_dir, output_dir):

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25)

    # Prepare the data
    x = 'Conditions'
    y = 'Success Rate'
    p_audio = [p_audio_bi, p_audio_br, p_audio_ir]
    p_visual = [p_visual_bi, p_visual_br, p_visual_ir]
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            data_list = (israte_audio_beat + israte_audio_interval + \
                         israte_audio_random)
            pvalue = p_audio
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            data_list = (israte_visual_beat + israte_visual_interval + \
                         israte_visual_random)
            pvalue = p_visual
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
        pairs = [('Beat', 'Interval'), ('Beat', 'Random'),
                 ('Interval', 'Random')]
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None,
                            # text_format="star",
                            text_format="simple",
                            test_short_name="pttest",
                            fontsize=10.)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        # ax[m].set_ylim(bottom=0., top=1.)

        if m ==1:
            # Remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # Remove y frame
            ax[m].spines['left'].set_visible(False)
            # Change x label
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=15)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=15)
            # Set y labels
            ax[m].set_yticks([0., .2, .4, .6, .8, 1.],
                             ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        # Display means rounded to two decimals on the top
        # ax.bar_label(ax.containers[0], padding=-50)
        yshift = -.1
        for p in ax[m].patches:
            ax[m].text(p.get_x() + p.get_width()/2., p.get_height() + yshift,
                       '{:.2e}'.format(p.get_height()), fontsize=7.,
                       color='black', ha='center', va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .7)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)


    # Title
    plt.suptitle('Group Mean of Success Rate for the NTFD Tasks', size=10,
                 linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir,
                             'ntfd_group_success_rate_random.pdf'))


# %%
# =========================== INPUTS ===================================

# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
#             40, 41, 42, 43, 44, 45, 46, 47]
SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47]

# This set of subjects are those that for the behavioral experiments did
# the NTFD with the Random Condition
# RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
#                  47]
RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                 47]

# TASKS = ['Auditory No-Temporal Feature Discrimination',
#          'Visual No-Temporal Feature Discrimination']

SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSTYPES = ['behavioral_session']

PLOTS_FOLDER = 'ntfd_results'

# Total number of trials per run
N_TRIALS = 30
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of all behavioral sessions
N_ISI_TRIALS_BEHAV = 36 # (3*4*3)
# Total number of trials per isi per condition across all runs of all ...
# ... imaging sessions
N_ISI_TRIALS_IMG = 16 # (3*2*2 + 2*2)

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

if SESSTYPES == ['behavioral_session', 'imaging_session']:
    N_ISI_TRIALS = N_ISI_TRIALS_BEHAV + N_ISI_TRIALS_IMG
elif SESSTYPES == ['behavioral_session']:
    N_ISI_TRIALS = N_ISI_TRIALS_BEHAV
else:
    assert SESSTYPES == ['imaging_session']
    N_ISI_TRIALS = N_ISI_TRIALS_IMG

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(os.path.join(MAIN_DIR, PLOTS_FOLDER)):
        os.makedirs(os.path.join(MAIN_DIR, PLOTS_FOLDER))

    # ################### NTFD RT'S ####################################

    # ### Individual analysis merging all standards --- bar plots
    m_rtsntfd_audio_beat, m_rtsntfd_audio_interval, m_rtsntfd_audio_random, \
        m_rtsntfd_visual_beat, m_rtsntfd_visual_interval, \
        m_rtsntfd_visual_random = individual_ntfd_rts(
            RAND_SUBJECTS, MAIN_DIR, PLOTS_FOLDER, SESSTYPES, N_TRIALS,
            flatten=False)

    # Compute ffx
    m_rtsntfd_audio_beat = np.nanmean(
        m_rtsntfd_audio_beat, axis=1).tolist()
    m_rtsntfd_audio_interval = np.nanmean(
        m_rtsntfd_audio_interval, axis=1).tolist()
    m_rtsntfd_audio_random = np.nanmean(
        m_rtsntfd_audio_random, axis=1).tolist()
    m_rtsntfd_visual_beat = np.nanmean(
        m_rtsntfd_visual_beat, axis=1).tolist()
    m_rtsntfd_visual_interval = np.nanmean(
        m_rtsntfd_visual_interval, axis=1).tolist()
    m_rtsntfd_visual_random = np.nanmean(
        m_rtsntfd_visual_random, axis=1).tolist()

    # Concatenate
    m_rtsntfd_audio = m_rtsntfd_audio_beat + m_rtsntfd_audio_interval + \
        m_rtsntfd_audio_random
    m_rtsntfd_visual = m_rtsntfd_visual_beat + m_rtsntfd_visual_interval + \
        m_rtsntfd_visual_random

    # Compute stats
    _, pntfd_audio_bi = stats.ttest_rel(
        m_rtsntfd_audio_beat, m_rtsntfd_audio_interval,
        axis=0, alternative='two-sided')

    _, pntfd_audio_br = stats.ttest_rel(
        m_rtsntfd_audio_beat, m_rtsntfd_audio_random,
        axis=0, alternative='two-sided')

    _, pntfd_audio_ir = stats.ttest_rel(
        m_rtsntfd_audio_interval, m_rtsntfd_audio_random,
        axis=0, alternative='two-sided')

    _, pntfd_visual_bi = stats.ttest_rel(
        m_rtsntfd_visual_beat, m_rtsntfd_visual_interval,
        axis=0, alternative='two-sided')

    _, pntfd_visual_br = stats.ttest_rel(
        m_rtsntfd_visual_beat, m_rtsntfd_visual_random,
        axis=0, alternative='two-sided')

    _, pntfd_visual_ir = stats.ttest_rel(
        m_rtsntfd_visual_interval, m_rtsntfd_visual_random,
        axis=0, alternative='two-sided')

    ntfd_title = 'Group Mean of Reaction Time for the NTFD tasks'
    ntfd_f = 'paired-ttest_merged-rt_ntfd'
    plot_pttest(m_rtsntfd_audio, m_rtsntfd_visual,
                pntfd_audio_bi, pntfd_audio_br, pntfd_audio_ir,
                pntfd_visual_bi, pntfd_visual_br, pntfd_visual_ir,
                'Reaction Time (ms)', 0., 750., -100.,
                ntfd_title, MAIN_DIR, PLOTS_FOLDER, ntfd_f)

    # ##################################################################
    # ### Individual analysis per standards --- box plots
    rtsntfd_audio_beat, rtsntfd_audio_interval, rtsntfd_visual_beat, \
        rtsntfd_visual_interval, standards = individual_ntfd_isi_rts(
            SUBJECTS, MAIN_DIR, PLOTS_FOLDER, SESSTYPES, N_TRIALS,
            N_ISI_TRIALS, flatten=False)

    # ## Compute mean of reaction time across trials per subject
    # ## for every standard (fixed-effects)
    ffx_rtsntfd_audio_beat, ffx_rtsntfd_audio_interval, \
        ffx_rtsntfd_visual_beat, ffx_rtsntfd_visual_interval = ffx(
            rtsntfd_audio_beat, rtsntfd_audio_interval,
            rtsntfd_visual_beat, rtsntfd_visual_interval)

    # ### Group Analyses per standard --- bar plots + paired t-test
    _, prtntfd_audio = stats.ttest_rel(
        ffx_rtsntfd_audio_beat, ffx_rtsntfd_audio_interval,
        axis=1, alternative='two-sided')

    _, prtntfd_visual = stats.ttest_rel(
        ffx_rtsntfd_visual_beat, ffx_rtsntfd_visual_interval,
        axis=1, alternative='two-sided')

    rtntfd_title = 'Group Mean of Reaction Time for the NTFD tasks'
    rtntfd_f = 'paired-ttest_rt_ntfd'
    plot_pttest_isi(ffx_rtsntfd_audio_beat, ffx_rtsntfd_audio_interval,
                    ffx_rtsntfd_visual_beat, ffx_rtsntfd_visual_interval,
                    prtntfd_audio, prtntfd_visual,
                    standards, 'Reaction Time (ms)', 0., 650., -100.,
                    rtntfd_title, MAIN_DIR, PLOTS_FOLDER, rtntfd_f)

    # ## Reshape
    rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval, \
        rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval = ginput_reshape(
            rtsntfd_audio_beat, rtsntfd_audio_interval,
            rtsntfd_visual_beat, rtsntfd_visual_interval)

    # ### Group Analyses per standard --- violin plots
    plot_violin(
        rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval,
        rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval,
        standards, 0., 2250., 'Reaction Time (ms)',
        'Group Distribution of Reaction Time for the NTFD Tasks',
        MAIN_DIR, PLOTS_FOLDER,
        'ntfd_groupviolin_rt')

    # ##################################################################
    # ### Analysis for success rate

    # Individual analysis
    individual_norand_srate_audio_beat, \
        individual_norand_srate_audio_interval, _, \
        individual_norand_srate_visual_beat, \
        individual_norand_srate_visual_interval, _ = \
            individual_ntfd_sucessrate(SUBJECTS, MAIN_DIR, PLOTS_FOLDER,
                                       SESSTYPES, N_TRIALS, flatten=False)

    individual_rand_srate_audio_beat, \
        individual_rand_srate_audio_interval, \
        individual_rand_srate_audio_random, \
        individual_rand_srate_visual_beat, \
        individual_rand_srate_visual_interval, \
        individual_rand_srate_visual_random = \
            individual_ntfd_sucessrate(RAND_SUBJECTS, MAIN_DIR, PLOTS_FOLDER,
                                       SESSTYPES, N_TRIALS, random=True,
                                       flatten=False)

    # Compute stats
    _, pval_norand_srate_audio = stats.ttest_rel(
        individual_norand_srate_audio_beat,
        individual_norand_srate_audio_interval,
        alternative='two-sided')

    _, pval_norand_srate_visual = stats.ttest_rel(
        individual_norand_srate_visual_beat,
        individual_norand_srate_visual_interval,
        alternative='two-sided')

    # ###

    _, pval_rand_srate_audio_bi = stats.ttest_rel(
        individual_rand_srate_audio_beat,
        individual_rand_srate_audio_interval,
        alternative='two-sided')

    _, pval_rand_srate_visual_bi = stats.ttest_rel(
        individual_rand_srate_visual_beat,
        individual_rand_srate_visual_interval,
        alternative='two-sided')

    _, pval_rand_srate_audio_br = stats.ttest_rel(
        individual_rand_srate_audio_beat,
        individual_rand_srate_audio_random,
        alternative='two-sided')

    _, pval_rand_srate_visual_br = stats.ttest_rel(
        individual_rand_srate_visual_beat,
        individual_rand_srate_visual_random,
        alternative='two-sided')

    _, pval_rand_srate_audio_ir = stats.ttest_rel(
        individual_rand_srate_audio_interval,
        individual_rand_srate_audio_random,
        alternative='two-sided')

    _, pval_rand_srate_visual_ir = stats.ttest_rel(
        individual_rand_srate_visual_interval,
        individual_rand_srate_visual_random,
        alternative='two-sided')

    # Group analysis
    group_successrate_norand(
        individual_norand_srate_audio_beat,
        individual_norand_srate_audio_interval,
        individual_norand_srate_visual_beat,
        individual_norand_srate_visual_interval,
        pval_norand_srate_audio,
        pval_norand_srate_visual,
        MAIN_DIR, PLOTS_FOLDER)

    group_successrate_rand(
        individual_rand_srate_audio_beat,
        individual_rand_srate_audio_interval,
        individual_rand_srate_audio_random,
        individual_rand_srate_visual_beat,
        individual_rand_srate_visual_interval,
        individual_rand_srate_visual_random,
        pval_rand_srate_audio_bi,
        pval_rand_srate_audio_br,
        pval_rand_srate_audio_ir,
        pval_rand_srate_visual_bi,
        pval_rand_srate_visual_br,
        pval_rand_srate_visual_ir,
        MAIN_DIR, PLOTS_FOLDER)
