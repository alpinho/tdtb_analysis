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


# subjects_no = [1, 2, 3, 4]
subjects_no = [1]

subjects = ['sub-%02d' % s for s in subjects_no]

# Load log file
this_dir = os.path.dirname(os.path.abspath(__file__))
subject_logpath = os.path.join(this_dir, 'sub-01', 'logfiles')
subject_logfiles = glob.glob(os.path.join(subject_logpath, '*.xpd'))
subject_logfile = subject_logfiles[0]
inputs_list = [line for line in csv.reader(open(subject_logfile), delimiter=',')]

# Start parsing from the correct row
for r, row in enumerate(inputs_list):
    if row[0] == str(subjects_no[0]):
        break
    else:
        continue
data = inputs_list[r:]
