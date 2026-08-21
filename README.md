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

## Author

- Ana Luísa Pinho, 2021 - present
