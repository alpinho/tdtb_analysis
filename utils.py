import os
import glob
import csv
import re

import numpy as np

from scipy import stats
from nilearn.image import load_img, new_img_like, math_img


def extract_timestamp(filename):
    """Function to extract a timestamp from XPD and CSV filenames."""
    # Old XPD names use YYYYMMDDHHMM before the extension.
    match = re.search(r"(\d{12})(?=\.(xpd|csv)$)", filename)
    if match:
        return int(match.group(1))

    # New CSV names use YYYYMMDD_HHMM before the extension.
    match = re.search(r"(\d{8})_(\d{4})(?=\.(xpd|csv)$)", filename)
    if match:
        return int(match.group(1) + match.group(2))

    return 0


def _get_metadata_value(inputs_list, field):
    """Return the value of a '#x field: value' metadata line."""
    prefix = field + ':'
    for row in inputs_list:
        if row and row[0].startswith(prefix):
            return row[0].split(':', 1)[1].strip()

    return None


def _normalise_session(value):
    """Remove leading zeroes from a numeric session label."""
    if value.isdigit():
        return str(int(value))

    return value


def _normalise_csv_logfile(inputs_list, subject_no):
    """Convert newer CSV rows to the legacy XPD row layout.

    Newer CSV logs start with ``session_id`` in the first data
    column. Legacy XPD-derived rows start with the subject number.
    The downstream parsers use fixed column indices, so every CSV
    trial row is rebuilt with ``subject_no`` explicitly forced into
    column 0. This applies to production and perception logs.
    """
    header_idx = None
    for i, row in enumerate(inputs_list):
        if row and row[0] == 'session_id':
            header_idx = i
            break

    if header_idx is None:
        return inputs_list

    subject = str(subject_no)

    header = [[
        'subject_id', 'session_number', 'run_number', 'trial_number',
        'trial_id', 'condition', 'onset', 'duration',
        'theoretical_isi/feedback', 'real_isi/feedback', 'rt', 'key'
    ]]

    trials = []
    for row in inputs_list[header_idx + 1:]:
        if not row:
            continue
        if len(row) < 11:
            continue

        event = row[4]
        if event.startswith('isi_'):
            event = event.replace('isi_', 'interval_', 1)

        trials.append([
            subject,
            _normalise_session(row[0]),
            row[1],
            row[2],
            row[3],
            event,
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
        ])

    return inputs_list[:header_idx] + header + trials


def _is_selected_task(inputs_list, task, sesstype):
    """Return True if a logfile corresponds to the selected task."""
    sesstype_tag = sesstype.replace('_', ' ')
    xpd_task = task + ' - ' + sesstype_tag

    modality, _, task_name = task.partition(' ')
    csv_modality = modality.lower()
    csv_modality = {'auditory': 'audio'}.get(csv_modality, csv_modality)
    csv_task = 'st ' + csv_modality + ' ' + task_name.upper()

    for row in inputs_list[:12]:
        if not row:
            continue

        line = row[0]
        if line.startswith('#e Task:'):
            task_label = line.split(':', 1)[1].strip()
            return task_label in [xpd_task, csv_task]

    return False


def parse_logfile(parent_dir, subject_no, sesstypes, task, n_trials,
                  ttl=False, concatenate=True, sessions=None,
                  renumber_sessions=False, reject_pilot=True):

    allsessions = []
    for sesstype in sesstypes:
        sesstype_path = os.path.join(parent_dir, 'sub-%02d' % subject_no,
                                     sesstype + 's')
        # If the participant only did behaviour
        if sesstype == 'imaging_session' and not os.path.exists(sesstype_path):
            break

        # Do not consider behavioural data from imaging session of
        # sub-04 (pilot).
        # Unless, we want to extract paradigm descriptors.
        elif (sesstype == 'imaging_session'
              and subject_no == 4 and reject_pilot):
            break
        elif sessions is None:
            selected_sessions = os.listdir(sesstype_path)
            selected_sessions.sort()
        else:
            selected_sessions = sessions

        for session in selected_sessions:
            logpath = os.path.join(sesstype_path, session)
            logfiles = [
                f for ext in ('*.xpd', '*.csv')
                for f in glob.glob(os.path.join(logpath, ext))
            ]
            logfiles.sort(key=extract_timestamp)

            inputs_lists = []
            for logfile in logfiles:
                with open(logfile, newline='') as open_file:
                    inputs_list = [
                        line for line in csv.reader(open_file, delimiter=',')
                    ]
                if logfile.endswith('.csv'):
                    inputs_list = _normalise_csv_logfile(
                        inputs_list, subject_no)
                inputs_lists.append(inputs_list)

            # Pick log files of selected task
            allruns = []
            for i, inputs_list in enumerate(inputs_lists, 1):
                ttag = task + ' - ' + sesstype.replace('_', ' ')
                if _is_selected_task(inputs_list, task, sesstype):
                    liste = inputs_list
                    # Extract trial information from log file
                    for r, row in enumerate(liste):
                        if row[0] == str(subject_no):
                            break
                        else:
                            continue
                    # Remove TTL line or not?
                    if not ttl:
                        trials_info = liste[r+1:]
                    else:
                        trials_info = liste[r:]

                    # For behavioral sessions
                    if sesstype == 'behavioral_session':
                        trials_info = trials_info[:-1]
                    # For imaging sessions
                    else:
                        assert sesstype == 'imaging_session'
                        if ttag in inputs_list[8][0][9:]:
                            for li, line in enumerate(trials_info):
                                if li and \
                                   trials_info[li-1][3] == str(n_trials) and \
                                   line[4] == 'fixcross':
                                    break
                                else:
                                    continue
                            trials_info = trials_info[:li+2]
                        elif ttag in inputs_list[9][0][9:]:
                            for li, line in enumerate(trials_info):
                                if line[4] == 'baseline' and \
                                   trials_info[li-2][3] == str(n_trials):
                                    break
                                else:
                                    continue
                            trials_info = trials_info[li+1:]

                        # Renumbering the imaging sessions
                        if renumber_sessions:
                            for trial in trials_info:
                                if trial[1] == '1':
                                    trial[1] = '4'
                                else:
                                    assert trial[1] == '2'
                                    trial[1] = '5'

                    if concatenate:
                        allruns.extend(trials_info)
                    else:
                        allruns.append(trials_info)
                    if i == len(inputs_lists):
                        break

                if i == len(inputs_lists) and not allruns:
                    raise NameError(
                        'Log file for selected task does not exist!')

            if concatenate:
                allsessions.extend(allruns)
            else:
                allsessions.append(allruns)

    return allsessions


def adjacent_values(vals, q1, q3):
    vals.sort()
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)

    return lower_adjacent_value, upper_adjacent_value


def customize_vplot(datum, ax, pos):
    q1, median, q3 = np.percentile(datum, [25, 50, 75])
    whiskers = np.array([adjacent_values(datum, q1, q3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.scatter(pos, median, marker='o', color='white', s=6, zorder=3)
    ax.vlines(pos, q1, q3, color='k', linestyle='-', lw=5)
    ax.vlines(pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def ffx(audio_beat, audio_interval, visual_beat, visual_interval,
        metric='mean'):
    # Inputs shape (n_subjects, n_isi, n_trials)
    # Computes mean of elements in the third dimension
    # Swaps dimensions and returns array w/ shape (n_isi, n_subjects)

    if metric == 'mean':
        ffx_audio_beat = [[np.mean(ab2)
                           for ab2 in ab1] for ab1 in audio_beat]
        ffx_audio_interval = [[np.mean(ai2)
                               for ai2 in ai1] for ai1 in audio_interval]
        ffx_visual_beat = [[np.mean(vb2)
                             for vb2 in vb1] for vb1 in visual_beat]
        ffx_visual_interval = [[np.mean(vi2)
                                for vi2 in vi1] for vi1 in visual_interval]
    else:
        assert metric == 'std'
        ffx_audio_beat = [[np.std(ab2)
                           for ab2 in ab1] for ab1 in audio_beat]
        ffx_audio_interval = [[np.std(ai2)
                               for ai2 in ai1] for ai1 in audio_interval]
        ffx_visual_beat = [[np.std(vb2)
                             for vb2 in vb1] for vb1 in visual_beat]
        ffx_visual_interval = [[np.std(vi2)
                                for vi2 in vi1] for vi1 in visual_interval]

    ffx_audio_beat = np.swapaxes(ffx_audio_beat, 0, 1)
    ffx_audio_interval = np.swapaxes(ffx_audio_interval, 0, 1)
    ffx_visual_beat = np.swapaxes(ffx_visual_beat, 0, 1)
    ffx_visual_interval = np.swapaxes(ffx_visual_interval, 0, 1)

    return (ffx_audio_beat, ffx_audio_interval, ffx_visual_beat,
            ffx_visual_interval)


def zval_conversion(tval, dof):
    """
    Convert t-values to z-values, given the degrees of freedom.

    This function converts t-values (from a t-distribution with the
        specified degrees of freedom) to z-values (from a standard
        normal distribution). It does so by computing the upper-tail
        probability using the survival function (sf) and the lower-tail
        probability using the cumulative distribution function (cdf) of
        the t-distribution, then converting these probabilities to
        z-values using the inverse survival function (isf) and the
        percent point function (ppf) of the normal distribution.
        For cases where the z-value obtained via isf is negative, the
        corresponding ppf value is used instead.

    Parameters
    ----------
    tval : array_like
        t-values (or an array of t-values) from a t-distribution.
    dof : int or array_like
        Degrees of freedom for the t-distribution.

    Returns
    -------
    zval : ndarray
        z-values corresponding to the input t-values.
    """
    pval = stats.t.sf(tval, dof)
    one_minus_pval = stats.t.cdf(tval, dof)
    zval_sf = stats.norm.isf(pval)
    zval_cdf = stats.norm.ppf(one_minus_pval)
    zval = np.empty(pval.shape)
    use_cdf = zval_sf < 0
    use_sf = np.logical_not(use_cdf)
    zval[np.atleast_1d(use_cdf)] = zval_cdf[use_cdf]
    zval[np.atleast_1d(use_sf)] = zval_sf[use_sf]

    return zval


def tval_conversion(zval, dof):
    """
    Convert z-values back into t-values, given the degrees of freedom.
    
    For positive z-values, we assume:
        p = norm.sf(z)
        t = t.isf(p, dof)
    
    For negative z-values, we assume:
        one_minus_p = norm.cdf(z)
        t = t.ppf(one_minus_p, dof)
    
    Parameters
    ----------
    zval : array_like
        Input z-values.
    dof : int or array_like
        Degrees of freedom for the t distribution.
    
    Returns
    -------
    tval : array_like
        The corresponding t-values.
    """
    zval = np.asarray(zval)
    tval = np.empty(zval.shape)
    
    pos_mask = zval >= 0
    neg_mask = ~pos_mask
    
    if np.any(pos_mask):
        p = stats.norm.sf(zval[pos_mask])
        # t.isf(q, dof) is equivalent to t.ppf(1-q, dof)
        tval[pos_mask] = stats.t.isf(p, dof)
    
    if np.any(neg_mask):
        one_minus_p = stats.norm.cdf(zval[neg_mask])
        tval[neg_mask] = stats.t.ppf(one_minus_p, dof)
    
    return tval


def combine_maps(mpath1, mpath2, combined_mpaths):

    # Load
    map1 = load_img(mpath1)
    map2 = load_img(mpath2)

    # Get data
    map1_val = map1.get_fdata()
    map2_val = map2.get_fdata()

    # Merge masks in one single file
    combined_map_vals = map1_val + map2_val
    combined_map = new_img_like(map1, combined_map_vals)

    # Save file
    combined_map.to_filename(combined_mpaths)


def mask_map(map_path, mask_path, masked_map_path):
    # Apply the mask: retain values where mask > 0, set others to zero
    masked_map_img = math_img('map * (mask > 0)', map=map_path, mask=mask_path)

    # Save the masked image
    masked_map_img.to_filename(masked_map_path + '.nii.gz')


def combine_masks(maskpath1, maskpath2, combined_maskpath):

    # Load
    mask1 = load_img(maskpath1)
    mask2 = load_img(maskpath2)

    # Get data
    mask1_val = mask1.get_fdata().astype(np.uint8)
    mask2_val = mask2.get_fdata().astype(np.uint8)

    # Merge masks in one single file
    combined_mask_val = mask1_val + mask2_val
    combined_mask_val[combined_mask_val == 2] = 1
    combined_mask = new_img_like(mask1, combined_mask_val)

    # Save file
    combined_mask.to_filename(combined_maskpath)
