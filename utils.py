import os
import glob
import csv

import numpy as np


def parse_logfile(parent_dir, subject_no, sesstype, n_sess, tasks,
                  ttl=False, concatenate=True):

    sessions = ['sess-%02d' %s for s in np.arange(1, n_sess + 1)]
    allsessions = []
    for session in sessions:
        logpath = os.path.join(parent_dir, 'sub-%02d' % subject_no, session)
        logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
        logfiles.sort()
        inputs_lists = [[line for line in csv.reader(open(logfile),
                                                     delimiter=',')]
                        for logfile in logfiles]
        # Pick log files of selected task
        allruns = []
        for i, inputs_list in enumerate(inputs_lists, 1):
            for task_name in tasks:
                ttag = task_name + ' - ' + sesstype
                if ttag in inputs_list[8][0][9:]:
                    liste = inputs_list
                    # Extract trial information from log file
                    for r, row in enumerate(liste):
                        if row[0] == str(subject_no):
                            break
                        else:
                            continue
                    if not ttl:
                        trials_info = liste[r+1:]
                    else:
                        trials_info = liste[r:]
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
