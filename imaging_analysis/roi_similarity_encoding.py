"""
Compute roi-similarity matrices for the encoding period

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 7th of October 2025
Last Update: October 2025

Compatibility: Python 3.10.16
"""

import os

# ########################## FUNCTIONS ############################## #

# ########################### INPUTS ################################ #
ALPHA = 0.05

N_ROIS = 8
INDIVID_LEVELS = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]

ROI_LABELS: Dict[str, str] = {
    'dstr': 'Dorsal Striatum',
    'sma': 'SMA',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'PreSMA',
    'heschl': 'Heschl Gyrus',
    'occipital': 'Occipital Lobe',
}

# Enforced listing/sorting order
HEMIS = ['bh', 'lh', 'rh']
TASKS = ['Production', 'Perception', 'NTFD']
MODALITIES = ['Both', 'Auditory', 'Visual']

HEMI_ORDER = {h: i for i, h in enumerate(HEMIS)}
MOD_ORDER = {m: i for i, m in enumerate(MODALITIES)}

# ########################### PATHS ################################# #
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'rwls'
MASKING = 'wb'
HRF = 'hrf128'

BASE_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'main_tasks',
)

# ############################ RUN ################################## #

if __name__ == "__main__":

    # Open dataframes
    for indiv in INDIVID_LEVELS:
        df_dir = os.path.join(BASE_DIR, 'df_rois_volume')
        df_path = os.path.join(df_dir, f"dfrois_{indiv}_{N_ROIS}-rois.tsv")
        # Check if dataframe exists
        if not os.path.exists(df_path):
            print(f"[WARN] Missing file for {indiv}: {df_path}")
            continue
