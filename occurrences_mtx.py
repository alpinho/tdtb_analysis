"""
Script to visualize occurrences of cognitive components from a set of
 experimental conditions

Author: Ana Luisa Pinho

Date: November 2021

Compatibility: Python 3.7
"""

import os

import numpy as np
import pandas as pd


def condcomp(selected_tasks, df):
    """
    Extract conditions and components from selected tasks in the data frame
    """
    cond_list = []
    comp_list = []
    for tk in selected_tasks:
        task_conditions = [tk + '_' + cd
                           for cd in df[df.task == tk].condition.tolist()]
        cond_list.extend(task_conditions)
        cogcomps = df[df.task == tk].components.apply(
            lambda x: x[1:-1].split(',')).tolist()
        cogcomp_array = []
        for cogcomp in cogcomps:
            cogcomp = [cc.replace("'", "") for cc in cogcomp]
            cogcomp_array.append(cogcomp)
        comp_list.extend(cogcomp_array)
    return cond_list, comp_list


def utags(tags_lists):
    """
    Return unique components list.
    """
    tags_flatten = [item for sublist in tags_lists for item in sublist]
    unique_tags = np.unique(tags_flatten).tolist()
    return unique_tags


def occurences_matrix(tgs_list, unique_tgs, conds):
    """
    Create occurrences matrix of cognitive components
    Return data-frame
    """
    occur_mtx = []
    for tlist in tgs_list:
        occur = []
        for coco in unique_tgs:
            if coco in tlist:
                occur.append(1)
            else:
                occur.append(0)
        occur_mtx.append(occur)
    data_frame = pd.DataFrame(occur_mtx, columns=unique_tgs,
                              index=conds)
    return data_frame


# ############################# INPUTS ########################################

fpath = os.path.abspath('conditions_components.tsv')

selected_tasks = ['zatorre1994', 'ramnani2001', 'pope2005', 'grahn2007',
                  'chen2008', 'thaut2008', 'karabanov2009', 'bengtsson2006']


# #################################### RUN ####################################

if __name__ == "__main__":
    # Read file
    df_condcomp = pd.read_csv(fpath, sep='\t')
    # Create conditions and components lists
    conditions_list, components_list = condcomp(selected_tasks, df_condcomp)
    # Build the design matrix of components occurrences
    unique_components = utags(components_list)
    dmtx = occurences_matrix(components_list, unique_components,
                             conditions_list)
