#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether MDS component 3 from two analyses differs only by sign.

Loads the two ROI similarity matrices using the same path logic as the
main scripts, runs classical MDS on both, and computes the correlation
between component 3 scores.

Interpretation:
- correlation close to +1 : same orientation
- correlation close to -1 : mirrored axis (sign flip)
- otherwise              : genuinely different structure

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 24th of April 2026
Last Update: April 2026

Compatibility: Python 3.10.16
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import PcmPy as pcm


# =========================== USER INPUTS =========================== #

MODEL = 'rwls'
MASKING = 'wb'
HRF = 'hrf128'

INDIV = 'i'
HEMI = 'bh'
MODALITY = 'Both'
N_ROIS = 8

FILETAG = 'withrestrand'

# rm-corr matrix
ENC_DIRNAME = 'encoding_restrand'

# subject-wise matrix
SUBJ_DIRNAME = 'subjectcorr_paired_restrand'


# ============================= PATHS ============================== #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_ALL = os.path.join(
    WORKING_DIR,
    f'roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed',
    'bothmod_allmain_tasks',
    'profile_similarity',
)

path_rm = os.path.join(
    BASE_ALL,
    ENC_DIRNAME,
    INDIV,
    'matrices',
    f'matrix_r_{INDIV}_{MODALITY}_{HEMI}_{N_ROIS}-rois_'
    f'{FILETAG}.tsv',
)

path_subj = os.path.join(
    BASE_ALL,
    SUBJ_DIRNAME,
    INDIV,
    'matrices',
    f'matrix_mean_r_{INDIV}_{MODALITY}_{HEMI}_{N_ROIS}-rois_'
    f'{FILETAG}.tsv',
)


# =========================== FUNCTIONS ============================ #

def run_mds(path: str) -> tuple[list[str], np.ndarray]:
    """
    Load matrix and run classical MDS.
    """
    df = pd.read_csv(path, sep='\t', index_col=0)
    labels = df.index.tolist()

    mtx = df.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(mtx)
    mtx = mtx / rank

    scores, eigval = pcm.util.classical_mds(mtx)
    return labels, scores


# ============================== RUN =============================== #

if __name__ == '__main__':

    print('RM-CORR MATRIX:')
    print(path_rm)
    print()

    print('SUBJECT-WISE MATRIX:')
    print(path_subj)
    print()

    labels1, scores1 = run_mds(path_rm)
    labels2, scores2 = run_mds(path_subj)

    if labels1 != labels2:
        raise ValueError('ROI order differs between matrices.')

    comp = 2  # Python index for component 3

    old_c3 = scores1[:, comp]
    new_c3 = scores2[:, comp]

    r_val = np.corrcoef(old_c3, new_c3)[0, 1]

    print('ROI labels:')
    print(labels1)
    print()

    print('Old MDS3 scores:')
    print(np.round(old_c3, 6))
    print()

    print('New MDS3 scores:')
    print(np.round(new_c3, 6))
    print()

    print(f'Correlation old vs new MDS3 = {r_val:.6f}')
    print()

    if r_val < -0.95:
        print('Conclusion: component 3 is mirrored (sign flip).')
    elif r_val > 0.95:
        print('Conclusion: component 3 has same orientation.')
    else:
        print('Conclusion: component 3 changed beyond a simple sign flip.')