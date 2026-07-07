"""
Cross-cohort comparison of behavioural performance for the TDTB project.

For each task (Production, Perception, NTFD) the auditory (or visual) condition
effect (Interval - Beat) is estimated within each of two cohorts, and the
Cohort x Condition interaction is tested. The design is a 2x2 with one
between-subject factor (Cohort) and one within-subject factor (Condition, two
levels); the interaction therefore reduces to a two-sample comparison of each
participant's Interval - Beat difference, tested with the Welch (unequal
variance) t and a default-prior JZS Bayes factor BF01 (evidence for no
interaction).

Metrics (one value per participant and condition, auditory data):
  Production -- Mean Signed Asynchrony (no latency correction: the *_0_0_0 files)
  Perception -- Difference Limen (DL), averaged over Standards (postfit files)
  NTFD       -- Reaction Time (Beat/Interval only)

The script is driven by COHORTS below: set, per cohort, the batch ('fb'/'sb'),
the session grouping (any SESSION_CONFIG tag, e.g. 'behav12', 'behavses',
'ses-01'), and the participant set ('imaging', 'all', or an explicit list). It
reads the same per-grouping dataframes produced by production_df.py,
perception_analysis.py and ntfd_df.py, so it stays in sync with new data.

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: 6th of July 2026
Last update: July 2026

Compatibility: Python 3.10.14
"""

import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import re
import numpy as np
import pandas as pd
from scipy import stats, integrate

# %%
# =========================== CONFIGURATION ===========================
# Root of the behavioral_analysis tree.
BEHAV_ROOT = "."

# The two cohorts to compare. Edit 'grouping' to use any session grouping, and
# 'subjects' to restrict the sample ('imaging' = participants with imaging data,
# 'all', or an explicit list of integer IDs).
COHORTS = {
    "First cohort":  {"batch": "fb", "grouping": "behav12",  "subjects": "imaging"},
    "Second cohort": {"batch": "sb", "grouping": "behavses", "subjects": "all"},
}

MODALITIES = ["auditory", "visual"]   # modalities to include; both = full table
PROD_LATENCY = "0_0_0"       # production latency tag; "0_0_0" = no correction
BF_PRIOR_SCALE = 0.707       # JZS Cauchy prior scale (default)

# NTFD reaction time is latency-corrected (unlike Production): ms subtracted =
# presentation latency + button press, per modality and batch (from
# ntfd_rtscore.py). This is a constant per-cohort shift, so it cancels in the
# Interval - Beat effect; it only sets the absolute RT level.
NTFD_LATENCY = {
    "auditory": {"fb": 133 + 20, "sb": 63 + 20},
    "visual":   {"fb": 35 + 20,  "sb": 35 + 20},
}
# If True, exclude NTFD bad trials before averaging (RT outside [100, 700] ms or
# incorrect discrimination, score == 0), matching the manuscript's bad-trial
# definition. Your ntfd_rtscore.py currently does NOT filter, so leave this
# False to reproduce those results; set True for the bad-trial-excluded metric.
NTFD_FILTER_BAD_TRIALS = False

# Standard weighting. True: average within each Standard first, then across the
# five Standards (marginal mean over the designed Standard factor; matches the
# Production LMM's estimated marginal means, and treats the accidental
# trial-count imbalance across Standards as a nuisance). False: pool all trials
# (trial-count weighted). Perception is unaffected (one DL per Standard already).
EQUAL_WEIGHT_STANDARDS = True

OUTPUT_DIR = "cross_cohort_results"
# =====================================================================


# ========================== I/O AND METRICS ==========================
def _subj_int(x):
    """Normalise a subject label ('sub-03', 3, '3') to an int."""
    return int(re.search(r"(\d+)", str(x)).group(1))


def _batch_word(batch):
    return "first" if batch == "fb" else "second"


def _prod_path(batch, grouping):
    return os.path.join(
        BEHAV_ROOT, "production", "production_results", "dataframes",
        f"df_production_{batch}_{PROD_LATENCY}_{grouping}.tsv")


def _perc_path(batch, grouping):
    return os.path.join(
        BEHAV_ROOT, "perception",
        f"perception_results_{_batch_word(batch)}_batch", "anovas",
        f"df_perception_postfit_{grouping}.tsv")


def _ntfd_path(batch, grouping):
    return os.path.join(
        BEHAV_ROOT, "ntfd", f"ntfd_results_{_batch_word(batch)}_batch",
        "dataframes", f"df_ntfd_{grouping}.tsv")


def _aggregate(d):
    """Collapse [subject, condition, standard, value] to one value per
    subject and condition, equal-weighting Standards or pooling trials."""
    if EQUAL_WEIGHT_STANDARDS:
        cell = (d.groupby(["subject", "condition", "standard"])["value"]
                .mean().reset_index())
        g = (cell.groupby(["subject", "condition"])["value"]
             .mean().reset_index())
    else:
        g = d.groupby(["subject", "condition"])["value"].mean().reset_index()
    return g


def load_metric(task, batch, grouping, modality):
    """Return a tidy frame [subject, condition, value] for one task/cohort.

    'value' is the per-participant, per-condition metric (auditory or visual):
    mean signed asynchrony (Production), mean DL (Perception), or mean reaction
    time (NTFD, Beat/Interval only). NTFD reaction times are latency-corrected;
    Standards are collapsed per EQUAL_WEIGHT_STANDARDS.
    """
    if task == "Production":
        d = pd.read_csv(_prod_path(batch, grouping), sep="\t")
        d = d[d["modality"] == modality]
        d = d.rename(columns={"signed_asynchrony": "value"})
    elif task == "Perception":
        d = pd.read_csv(_perc_path(batch, grouping), sep="\t")
        pmod = "audio" if modality == "auditory" else "visual"
        d = d[d["Modality"] == pmod]
        d = d.rename(columns={"Subject": "subject", "Condition": "condition",
                              "Standard": "standard", "DL": "value"})
    elif task == "NTFD":
        d = pd.read_csv(_ntfd_path(batch, grouping), sep="\t")
        d = d[(d["modality"] == modality)
              & (d["condition"].isin(["beat", "interval"]))].copy()
        d = d.dropna(subset=["reaction_time"])
        d["reaction_time"] = (d["reaction_time"].astype(float)
                              - NTFD_LATENCY[modality][batch])
        if NTFD_FILTER_BAD_TRIALS:
            d = d[(d["reaction_time"] >= 100) & (d["reaction_time"] <= 700)
                  & (d["score"] == 1)]
        d = d.rename(columns={"reaction_time": "value"})
    else:
        raise ValueError(f"Unknown task: {task}")

    d = d[["subject", "condition", "standard", "value"]].copy()
    d["subject"] = d["subject"].apply(_subj_int)
    return _aggregate(d)


def imaging_subjects():
    """Participants with imaging data (those present in the fb imgses file)."""
    d = pd.read_csv(_prod_path("fb", "imgses"), sep="\t")
    return set(d["subject"].apply(_subj_int).unique())


def resolve_subjects(spec):
    if spec == "all":
        return None
    if spec == "imaging":
        return imaging_subjects()
    return set(int(s) for s in spec)


def condition_diffs(g, subjects):
    """Per-participant Interval - Beat difference (needs both conditions)."""
    if subjects is not None:
        g = g[g["subject"].isin(subjects)]
    wide = g.pivot(index="subject", columns="condition", values="value")
    wide = wide.dropna(subset=["beat", "interval"])
    means = {"beat": wide["beat"].mean(), "interval": wide["interval"].mean()}
    return (wide["interval"] - wide["beat"]), means


# ============================ STATISTICS =============================
def effect_ci(diffs):
    """Mean Interval - Beat, its 95% CI, and one-sample t p-value vs zero."""
    x = diffs.to_numpy(dtype=float)
    n = len(x)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    half = sem * stats.t.ppf(0.975, n - 1)
    _, p = stats.ttest_1samp(x, 0.0)
    return dict(n=n, mean=m, ci_low=m - half, ci_high=m + half, p=p)


def jzs_bf10(t, n1, n2, r=BF_PRIOR_SCALE):
    """Default-prior JZS Bayes factor BF10 for a two-sample t statistic."""
    nu = n1 + n2 - 2.0
    neff = (n1 * n2) / (n1 + n2)

    def integrand(g):
        return ((1 + neff * g * r ** 2) ** -0.5
                * (1 + t ** 2 / ((1 + neff * g * r ** 2) * nu))
                ** (-(nu + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))

    den, _ = integrate.quad(integrand, 0, np.inf)
    num = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    return den / num


def interaction(diffs_a, diffs_b):
    """Welch two-sample test of the Cohort x Condition interaction + BF01."""
    a = diffs_a.to_numpy(dtype=float)
    b = diffs_b.to_numpy(dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)          # Welch
    v1, v2, n1, n2 = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    df = ((v1 / n1 + v2 / n2) ** 2
          / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
    bf01 = 1.0 / jzs_bf10(t, n1, n2)
    return dict(t=t, df=df, p=p, bf01=bf01)


# =========================== LATEX OUTPUT ============================
TASK_LABEL = {"Production": "Production (Mean SA)",
              "Perception": "Perception (DL)",
              "NTFD": "NTFD (RT, ms)"}
TASK_DEC = {"Production": 3, "Perception": 3, "NTFD": 1}


def _fmt(x, dec):
    s = f"{x:.{dec}f}"
    if float(s) == 0:                       # avoid a spurious "-0.000"
        s = s.lstrip("-")
    return s


def _cell(eff, lo, hi, dec):
    return (f"${_fmt(eff, dec)}$ [${_fmt(lo, dec)}$, ${_fmt(hi, dec)}$]")


def gen_latex(summary, names, ns, modalities):
    """Assemble the copy-paste LaTeX table from the summary frame."""
    def hcell(text):
        return r"\multicolumn{1}{c}{\textbf{" + text + r"}}"

    L = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{\textbf{$\!\!\!\mid\!$ Comparison of Behavioural "
        r"Performance between Cohorts.}}",
        r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}",
        r"\setlength{\extrarowheight}{2.5pt}\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\rowcolors{2}{white}{nhbGreenRow}",
        r"\footnotesize",
        r"% --- column widths: edit the \hsize factors; they MUST sum to 6 ---",
        r"\begin{tabularx}{\textwidth}{",
        r"  >{\hsize=1.4\hsize\raggedright\arraybackslash}X   % 1 Task",
        r"  >{\hsize=1.5\hsize\raggedleft\arraybackslash}X    % 2 First cohort",
        r"  >{\hsize=1.5\hsize\raggedleft\arraybackslash}X    % 3 Second cohort",
        r"  >{\hsize=0.8\hsize\raggedleft\arraybackslash}X    % 4 Welch t (df)",
        r"  >{\hsize=0.4\hsize\raggedleft\arraybackslash}X    % 5 p",
        r"  >{\hsize=0.4\hsize\raggedleft\arraybackslash}X}   % 6 BF01",
        r"\toprule",
        r"\rowcolor{nhbGreenHeader}",
        r"\textbf{Task (Effect)} & "
        + hcell(f"{names[0]} ($N={ns[names[0]]}$)") + " & "
        + hcell(f"{names[1]} ($N={ns[names[1]]}$)") + " & "
        + hcell(r"Welch $t$ (df)") + " & " + hcell(r"$p$") + " & "
        + hcell(r"$\mathrm{BF}_{01}$") + r" \\",
        r"\rowcolor{nhbGreenHeader}",
        " & " + hcell(r"Interval$-$Beat [95\% CI]") + " & "
        + hcell(r"Interval$-$Beat [95\% CI]") + r" &  &  & \\",
    ]
    for mi, mod in enumerate(modalities):
        L.append(r"\midrule")
        L.append(r"\multicolumn{6}{l}{\textit{" + mod.capitalize()
                 + r"}} \\")
        L.append(r"\midrule")
        for task in ["Production", "Perception", "NTFD"]:
            r = summary[(summary["modality"] == mod)
                        & (summary["task"] == task)].iloc[0]
            dec = TASK_DEC[task]
            c0 = _cell(r["eff0"], r["ci0_low"], r["ci0_high"], dec)
            c1 = _cell(r["eff1"], r["ci1_low"], r["ci1_high"], dec)
            welch = f"${r['welch_t']:.2f}$ ({r['welch_df']:.1f})"
            L.append(f"{TASK_LABEL[task]} & {c0} & {c1} & {welch} & "
                     f"${r['p']:.2f}$ & ${r['bf01']:.1f}$ " + r"\\")
    L += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"",
        r"\vspace{6pt}",
        r"{\footnotesize",
        r"\leftskip=0pt \rightskip=0pt \parfillskip=0pt plus 1fil "
        r"\parindent=0pt",
        r"The condition effect (Interval $-$ Beat) is given per cohort with "
        r"its $95\%$ CI, followed by the Welch two-sample test of the Cohort "
        r"$\times$ Condition interaction -- equivalently, whether the "
        r"cohorts' condition effects differ -- and a default-prior JZS Bayes "
        r"factor $\mathrm{BF}_{01}$ (evidence for no interaction; $1$--$3$ "
        r"weak, $3$--$10$ moderate). The auditory tasks test the "
        r"re-implementation, in which the second cohort used PsychoPy; the "
        r"visual tasks, run in Expyriment for both cohorts, are a "
        r"same-software control isolating the sample. Metrics: \textit{Mean "
        r"Signed Asynchrony} (Production, proportion of $S$), \textit{"
        r"Difference Limen} (Perception, proportion), Reaction Time (NTFD, "
        r"ms). The first cohort is the imaging sample.\par}",
        r"\label{etab:cohort_consistency}",
        r"\end{table*}",
    ]
    return "\n".join(L)


# =============================== MAIN ================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = list(COHORTS.keys())
    assert len(names) == 2, "Exactly two cohorts must be configured."

    per_cohort_rows, summary_rows = [], []
    cohort_n = {n: 0 for n in names}
    for modality in MODALITIES:
        for task in ["Production", "Perception", "NTFD"]:
            diffs, effs = {}, {}
            for cname in names:
                cfg = COHORTS[cname]
                g = load_metric(task, cfg["batch"], cfg["grouping"], modality)
                d, means = condition_diffs(g, resolve_subjects(cfg["subjects"]))
                diffs[cname] = d
                e = effect_ci(d)
                effs[cname] = e
                cohort_n[cname] = max(cohort_n[cname], e["n"])
                per_cohort_rows.append(dict(
                    modality=modality, task=task, cohort=cname,
                    grouping=cfg["grouping"], n=e["n"],
                    mean_beat=means["beat"], mean_interval=means["interval"],
                    effect=e["mean"], ci_low=e["ci_low"], ci_high=e["ci_high"],
                    p_effect=e["p"]))
            it = interaction(diffs[names[0]], diffs[names[1]])
            summary_rows.append(dict(
                modality=modality, task=task,
                eff0=effs[names[0]]["mean"], ci0_low=effs[names[0]]["ci_low"],
                ci0_high=effs[names[0]]["ci_high"],
                eff1=effs[names[1]]["mean"], ci1_low=effs[names[1]]["ci_low"],
                ci1_high=effs[names[1]]["ci_high"],
                welch_t=it["t"], welch_df=it["df"], p=it["p"], bf01=it["bf01"]))

    per_cohort = pd.DataFrame(per_cohort_rows)
    summary = pd.DataFrame(summary_rows)
    per_cohort.to_csv(os.path.join(OUTPUT_DIR, "per_cohort_effects.tsv"),
                      sep="\t", index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "interaction_summary.tsv"),
                   sep="\t", index=False)
    tex = gen_latex(summary, names, cohort_n, MODALITIES)
    with open(os.path.join(OUTPUT_DIR, "cohort_table.tex"), "w") as fh:
        fh.write(tex + "\n")

    pd.set_option("display.width", 170,
                  "display.float_format", lambda v: f"{v:.4f}")
    print("Cohorts:")
    for cname in names:
        c = COHORTS[cname]
        print(f"  {cname} (N={cohort_n[cname]}): batch={c['batch']}, "
              f"grouping={c['grouping']}, subjects={c['subjects']}")
    print("\n--- Per-cohort condition means and effect (Interval - Beat) ---")
    print(per_cohort.to_string(index=False))
    print("\n--- Cohort x Condition interaction (Welch) + BF01 ---")
    print(summary.to_string(index=False))
    print("\n--- LaTeX table (also written to "
          f"{OUTPUT_DIR}/cohort_table.tex) ---\n")
    print(tex)
    return per_cohort, summary


if __name__ == "__main__":
    main()
