"""
Script to extract behavioral data from logfiles for the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 2022
Last update: May 2022

Compatibility: Python 3.7.11

"""
import os
import glob
import csv
import numpy as np


# %%
# =========================== FUNCTIONS ================================


def parse_logfile(parent_dir, subject_no, task_name):
    logpath = os.path.join(parent_dir, 'sub-%02d' % subject_no, 'logfiles')
    logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
    logfiles.sort()
    inputs_lists = [[line for line in csv.reader(open(logfile), delimiter=',')]
                    for logfile in logfiles]
    # Pick log file of selected task
    for i, inputs_list in enumerate(inputs_lists, 1):
        if inputs_list[6][0][9:] == task_name:
            liste = inputs_list
            break
        if i == len(inputs_lists):
            raise NameError('Log file for selected task does not exist!')
    # Extract trial information from log file
    for r, row in enumerate(liste):
        if row[0] == str(subject_no):
            break
        else:
            continue
    trials_info = liste[r:]

    return trials_info


# %%
# =========================== INPUTS ===================================

# subjects = [1, 2, 3, 4]
subjects = [1]

# tasks = ['Auditory Production',
#          'Auditory Perception',
#          'Auditory No-Temporal Feature Discrimination',
#          'Visual Production',
#          'Visual Perception',
#          'Visual No-Temporal Feature Discrimination']

tasks = ['Auditory Production']

# %%
# ========================= PARAMETERS =================================

this_dir = os.path.dirname(os.path.abspath(__file__))

# Subject, task, condition, reaction time, response

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    for subject in subjects:
        for task in tasks:
            data = parse_logfile(this_dir, subject, task)
            # ****************** Production Tasks **********************
            # #### Signed Synchrony ####





