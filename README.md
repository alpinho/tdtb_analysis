# TDTB analysis

Analysis of the neuroimaging and behavioral data collected for the *Timing Domain Task
Battery* (TDTB). The protocols that produced these data — stimulus generation, task scripts
and timing validation — are in
[alpinho/tdtb_protocols](https://github.com/alpinho/tdtb_protocols).

The battery crosses three tasks (production, perception, and non-temporal feature
discrimination) with two temporal conditions (beat and interval) and two sensory modalities
(auditory and visual), in behavioral and imaging sessions. The code here takes the raw
logfiles and functional images through to the statistics and figures reported in the
associated publications.

## Organization

| Path | Contents |
| --- | --- |
| `behavioral_analysis/` | Parsing of logfiles and analysis of behavioral performance, one subdirectory per task, plus the cross-cohort comparison |
| `imaging_analysis/` | Preprocessing and first-level modelling in MATLAB, and second-level, ROI and multivariate analyses in Python; includes the striatal and fs_LR32k surface meshes used for projection |
| `utils.py` | Helpers shared across the repository (logfile naming, timestamps, image handling) |
| `xpd_isi_analysis.py` | Comparison of nominal and delivered intervals as logged in the `.xpd` files of an acquisition |

### Behavioral analysis

The path is the same for each task: `*_df.py` parses the logfiles into one dataframe per
session grouping, `*_bad-trials.py` screens trials for exclusion, and a task-specific script
computes the dependent measure — a linear mixed model over the mean signed asynchrony for
production (`production_lmm.py`), psychometric fits and difference limens for perception
(`perception_analysis.py`), and reaction-time scores for NTFD (`ntfd_rtscore.py`).
`cross_cohort_behaviour.py` and `cohort_estimation_plot.py` then test and plot whether the
condition effect replicates across cohorts. Each task directory documents its own missing
data in a `README_missing_data_*` file.

### Imaging analysis

`msdtb_imana.m` is the MATLAB entry point for everything up to the first-level contrasts. It
is organized as cases called by name, in stages: `ANAT:` and `SURF:` for anatomical
processing and surface reconstruction, `SUIT:` for the cerebellum, `FUNC:` for realignment,
distortion correction and coregistration, `GLM:` and `CON:` for design, estimation and
contrasts, and `GROUP:` for the second level. The design matrices are built from the paradigm
descriptors written by `paradigm_descriptors.py`.

The Python scripts take over from the first-level maps: projection to volume, surface and
SUIT space, ROI definition and extraction, ANOVAs on the extracted ROIs, and the reliability,
RSA and MDS analyses.

## Running the scripts

Most scripts are standalone and are configured by editing the block of constants at the top,
then run without arguments. A few take command-line arguments — among them
`roi_extraction_msdtb.py`, `volume_to_surface.py`, `volume_to_suit.py`,
`summarize_significance.py`, `cohort_estimation_plot.py` and `xpd_isi_analysis.py`; each
documents its usage in its own header. The MATLAB pipeline is called by stage, for example
`msdtb_imana('FUNC:realign_unwarp', 'sn', 3)`.

Requirements: Python 3.10 with numpy, pandas, scipy, statsmodels, nilearn, PcmPy and
matplotlib; MATLAB with SPM12, the SUIT toolbox, FreeSurfer and the Diedrichsen lab
`dataframe`, `imaging/tools` and `rwls` toolboxes. Paths to the data directories are set at
the top of each script.

The data themselves are not distributed in this repository.

## Author

- Ana Luísa Pinho, 2021 - present
