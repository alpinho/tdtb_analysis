#!/usr/bin/env python3
"""
Deface T1w anatomical images of the TDTB dataset for public release.

Wraps `pydeface` (FSL FLIRT-based registration to a face template, then a
validated deface mask applied in that registered space) over every
subject's T1w found under a BIDS root, and optionally checks the result
against a cerebellum mask to catch any defacing that reached into brain
tissue before the output is trusted.

An earlier attempt at this used a custom brain mask (built from SUIT/SPM
tissue-segmentation output) plus `quickshear`; it twice produced masks/cuts
that damaged frontal cortex in ways that passed a naive automated check
but failed on visual review. `pydeface` was adopted instead specifically
because it is the community-validated tool used for OpenNeuro submissions,
and its behaviour was re-verified end to end (independent cerebellum-mask
overlap check + full-cohort visual review) before being trusted for the
public release. Do not revert to a from-scratch mask-building approach
without at least that same level of verification.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: September 2026
Last Update: September 2026

Compatibility: Python 3.10.16, pydeface 2.1.0, FSL 6.0.7.23 (FLIRT 6.0)

Prerequisites
-------------
- FSL must be installed and its environment sourced before running this
  script, so that `flirt` resolves on PATH, e.g.:
      export FSLDIR=/home/analu/fsl
      source $FSLDIR/etc/fslconf/fsl.sh
      export PATH=$FSLDIR/share/fsl/bin:$PATH
      export FSLOUTPUTTYPE=NIFTI_GZ
  The script checks for `flirt` up front and exits before doing any work
  if it is not found, rather than failing partway through a batch.
- `pydeface` must be installed, e.g. `pip install pydeface` -- kept in its
  own conda env in this project (`deface-pydeface210-py3124`), separate
  from FSL and from whatever env `--qc-cerebellum` is run in.
- `--qc-cerebellum` additionally needs `nibabel`, `numpy`, `scipy` and
  `nilearn` (the `deface-nilearn0141-py3124` env in this project). It is
  only imported if that flag is passed, so plain defacing does not need
  those packages.

How to run
----------
Deface every subject's T1w under a BIDS root, writing to a staging
directory (never touches the input files):

    python deface_t1w.py /cifs/diedrichsen/data/Cerebellum/tdtb/openneuro \\
        --outdir /cifs/diedrichsen/data/Cerebellum/tdtb/pydeface_staging

Specific subjects only, 6-way parallel (each pydeface call is a separate
`flirt` registration and is the bottleneck, ~6-10 min/subject serially):

    python deface_t1w.py <bids_root> --outdir <outdir> \\
        --subjects sub-03 sub-07 --jobs 6

Also run the independent cerebellum-overlap QC check after defacing
(requires corr_cerebellum_masks/<c_sub-NN_T1w_pcereb_corr.nii> alongside
the dataset root; default location assumes the layout used in this
project, override with --cerebellum-dir):

    python deface_t1w.py <bids_root> --outdir <outdir> --qc-cerebellum

Promote already-staged, already-reviewed output into the BIDS root,
overwriting the original T1w files in place. This is the only destructive
step in this script and is never run implicitly -- inspect --outdir
first (ideally with --qc-cerebellum and a visual check) before using it:

    python deface_t1w.py <bids_root> --outdir <outdir> --promote
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# %%
# ======================== HELPERS ======================================


def find_flirt():
    """Return the path to `flirt`, or None if it is not on PATH."""
    return shutil.which("flirt")


def find_t1w_files(bids_root, subjects=None):
    """Return sorted T1w paths under a BIDS root, optionally filtered."""
    pattern = os.path.join(bids_root, "sub-*", "ses-*", "anat", "*_T1w.nii.gz")
    files = sorted(glob.glob(pattern))
    if subjects:
        wanted = set(subjects)
        files = [f for f in files
                 if os.path.basename(f).split("_")[0] in wanted]
    return files


def deface_one(t1_path, outdir):
    """
    Run pydeface on a single T1w file.

    Writes <outdir>/<subject>/<stem>_defaced.nii.gz and a matching
    pydeface.log capturing pydeface's own console output. Returns a dict
    with the outcome, for the caller to tally and report.
    """
    fn = os.path.basename(t1_path)
    subject = fn.split("_")[0]
    stem = fn[:-len(".nii.gz")]

    subj_dir = os.path.join(outdir, subject)
    os.makedirs(subj_dir, exist_ok=True)
    out_path = os.path.join(subj_dir, f"{stem}_defaced.nii.gz")
    log_path = os.path.join(subj_dir, "pydeface.log")

    t0 = time.time()
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["pydeface", t1_path, "--outfile", out_path, "--force"],
            stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    return {
        "subject": subject,
        "t1_path": t1_path,
        "out_path": out_path,
        "log_path": log_path,
        "ok": result.returncode == 0,
        "elapsed": elapsed,
    }


def run_batch(t1_files, outdir, jobs):
    """Run deface_one over every file, up to `jobs` at a time."""
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(deface_one, f, outdir): f for f in t1_files}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            status = "ok" if r["ok"] else "FAILED"
            print(f"[{i}/{len(t1_files)}] {r['subject']}: {status} "
                  f"({r['elapsed']:.1f}s)", flush=True)
            if not r["ok"]:
                print(f"    see {r['log_path']}", flush=True)
            results.append(r)
    return results


def cerebellum_overlap_qc(t1_path, defaced_path, cerebellum_dir):
    """
    Check whether any voxel removed by defacing overlaps the subject's
    cerebellum mask. Lazily imports nibabel/nilearn so that a plain
    defacing run does not need them installed.

    Returns (n_cerebellum_voxels, n_overlap_voxels). n_overlap_voxels > 0
    means defacing reached into the cerebellum and the result must not be
    trusted without investigation -- see the module docstring.
    """
    import numpy as np
    import nibabel as nib
    from nilearn.image import resample_to_img

    subject = os.path.basename(t1_path).split("_")[0]
    cereb_path = os.path.join(cerebellum_dir, f"c_{subject}_T1w_pcereb_corr.nii")
    if not os.path.exists(cereb_path):
        raise FileNotFoundError(
            f"No cerebellum mask found for {subject} at {cereb_path}; "
            "pass --cerebellum-dir if masks live elsewhere.")

    t1_img = nib.load(t1_path)
    t1_data = np.asanyarray(t1_img.dataobj)
    defaced_data = np.asanyarray(nib.load(defaced_path).dataobj)
    removed = (defaced_data == 0) & (t1_data > 0)

    cereb_img = nib.load(cereb_path)
    cereb_on_t1 = resample_to_img(
        cereb_img, t1_img, interpolation="nearest", force_resample=True,
        copy_header=True)
    cereb_mask = cereb_on_t1.get_fdata() > 0.5

    return int(cereb_mask.sum()), int((removed & cereb_mask).sum())


def promote(results, bids_root, outdir):
    """
    Copy each successfully-defaced file from outdir over the matching
    original in bids_root, preserving the original filename. Handles the
    BIDS root's files commonly being read-only (checked-out DataLad/CIFS
    content) by chmod'ing before and after the copy.
    """
    n = 0
    for r in results:
        if not r["ok"]:
            continue
        rel = os.path.relpath(r["t1_path"], bids_root)
        target = os.path.join(bids_root, rel)
        if not os.path.exists(target):
            print(f"  SKIP (target missing): {target}")
            continue
        os.chmod(target, 0o644)
        shutil.copy2(r["out_path"], target)
        os.chmod(target, 0o644)
        n += 1
        print(f"  promoted: {rel}")
    print(f"promoted {n}/{len(results)} files")


# %%
# =========================== MAIN =======================================


def main():
    parser = argparse.ArgumentParser(
        description="Deface T1w anatomical images with pydeface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("bids_root", help="Path to the BIDS dataset root "
                         "(e.g. .../openneuro).")
    parser.add_argument("--outdir", required=True,
                         help="Staging directory for defaced output. "
                         "Never overwrites bids_root unless --promote is "
                         "also given.")
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to these subject IDs (e.g. "
                         "sub-03 sub-07). Default: every subject found.")
    parser.add_argument("--jobs", type=int, default=1,
                         help="Number of pydeface calls to run in "
                         "parallel (default: 1, sequential). Each call is "
                         "a separate flirt registration, ~6-10 min/subject "
                         "serially; 6 was used for the 32-subject cohort "
                         "on a 12-core machine.")
    parser.add_argument("--qc-cerebellum", action="store_true",
                         help="After defacing, check each result against "
                         "the subject's cerebellum mask (see "
                         "--cerebellum-dir). Requires nibabel/nilearn.")
    parser.add_argument("--cerebellum-dir", default=None,
                         help="Directory containing c_<subject>_T1w_"
                         "pcereb_corr.nii masks. Default: "
                         "<bids_root>/../corr_cerebellum_masks.")
    parser.add_argument("--promote", action="store_true",
                         help="Copy already-staged, already-reviewed "
                         "output from --outdir over the originals in "
                         "bids_root. The only destructive step here -- "
                         "run this only after inspecting --outdir.")
    args = parser.parse_args()

    if args.promote:
        # Re-derive the same file list and expected output paths so
        # promotion can run standalone (e.g. in a later invocation, after
        # a prior defacing run has already completed and been reviewed).
        t1_files = find_t1w_files(args.bids_root, args.subjects)
        results = []
        for t1_path in t1_files:
            fn = os.path.basename(t1_path)
            subject = fn.split("_")[0]
            stem = fn[:-len(".nii.gz")]
            out_path = os.path.join(args.outdir, subject,
                                     f"{stem}_defaced.nii.gz")
            results.append({
                "t1_path": t1_path,
                "out_path": out_path,
                "ok": os.path.exists(out_path),
            })
        missing = [r for r in results if not r["ok"]]
        if missing:
            print(f"WARNING: {len(missing)} expected defaced file(s) not "
                  f"found in {args.outdir}; skipping those.")
        promote(results, args.bids_root, args.outdir)
        return 0

    if find_flirt() is None:
        print("ERROR: `flirt` not found on PATH. Source FSL's environment "
              "before running this script -- see the module docstring.",
              file=sys.stderr)
        return 1

    t1_files = find_t1w_files(args.bids_root, args.subjects)
    if not t1_files:
        print(f"No T1w files found under {args.bids_root} "
              f"(subjects filter: {args.subjects}).", file=sys.stderr)
        return 1
    print(f"Found {len(t1_files)} T1w file(s).")

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    results = run_batch(t1_files, args.outdir, args.jobs)
    elapsed = time.time() - t_start

    n_ok = sum(1 for r in results if r["ok"])
    n_failed = len(results) - n_ok
    print(f"\n=== DEFACING SUMMARY ===")
    print(f"Total: {len(results)}  OK: {n_ok}  FAILED: {n_failed}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if args.qc_cerebellum:
        cerebellum_dir = args.cerebellum_dir or os.path.join(
            args.bids_root, os.pardir, "corr_cerebellum_masks")
        print(f"\n=== CEREBELLUM-OVERLAP QC ===")
        n_flagged = 0
        for r in results:
            if not r["ok"]:
                continue
            try:
                n_cereb, n_overlap = cerebellum_overlap_qc(
                    r["t1_path"], r["out_path"], cerebellum_dir)
            except FileNotFoundError as e:
                print(f"  {r['subject']}: SKIPPED ({e})")
                continue
            flag = " <=== FLAG" if n_overlap > 0 else ""
            if flag:
                n_flagged += 1
            print(f"  {r['subject']}: cereb_overlap={n_overlap}/{n_cereb}"
                  f"{flag}")
        if n_flagged:
            print(f"\n{n_flagged} subject(s) flagged for review -- do "
                  f"NOT --promote until investigated (see module "
                  f"docstring on why this check exists).")
        else:
            print("\nNo subject flagged. Still recommended to visually "
                  "review a sample before --promote.")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
