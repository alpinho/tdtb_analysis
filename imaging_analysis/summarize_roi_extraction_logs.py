#!/usr/bin/env python3
"""
Summarize ROI extraction logs produced by extract_roi_means_surface(...).

- Accepts one or more CSV paths or glob patterns
- Prints:
    1) Overall summary
    2) By ROI (region/atlas/roi)
    3) By ROI x hemisphere
    4) Rings-used distribution (how often 0,1,2,…,-1)
    5) Fallback types distribution
- Optionally writes the above as CSVs.

Usage:
    python summarize_roi_extraction_logs.py \
        /path/to/**/rois_surf_extraction/*_extraction_log.csv \
        --out-dir /path/to/summary

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 23rd of September 2025
Last update: September 2025

Compatibility: Python 3.10.16
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
import csv

try:
    import pandas as pd
except Exception:
    pd = None


CATEGORICAL_KEYS = [
    "region", "atlas", "roi", "tag", "subject", "hemisphere",
    "task", "contrast_id", "contrast_name"
]

NUMERIC_KEYS = [
    "n_vertices", "seed_nonzero", "seed_allowed",
    "finite_total", "valid_total", "initial_roi_finite",
    "expanded", "rings_used", "final_mask_size",
    "returned_mean", "used_nan", "cortex_frac"
]


def read_logs(paths_or_globs: list[str]) -> list[dict]:
    paths = []
    for p in paths_or_globs:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        print("[WARN] No log files matched.", file=sys.stderr)
        return []

    rows: list[dict] = []
    for p in paths:
        with open(p, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                # Normalize dtypes a bit
                clean = dict(row)
                for k in NUMERIC_KEYS:
                    if k in clean:
                        try:
                            if k in ("cortex_frac", "returned_mean"):
                                clean[k] = float(clean[k]) if clean[k] != "" else float("nan")
                            else:
                                clean[k] = int(float(clean[k])) if clean[k] != "" else 0
                        except Exception:
                            clean[k] = 0
                rows.append(clean)
    return rows


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):5.1f}%" if d else "  0.0%"


def summarize_overall(rows: list[dict]) -> dict:
    N = len(rows)
    exp = sum(r.get("expanded", 0) for r in rows)
    nan = sum(r.get("used_nan", 0) for r in rows)
    rings = [r.get("rings_used", -1) for r in rows if r.get("rings_used", -1) >= 0]
    dist = Counter(rings)
    fallback = Counter(r.get("fallback", "none") for r in rows)
    return {
        "n_rows": N,
        "expanded_n": exp,
        "expanded_pct": _pct(exp, N),
        "used_nan_n": nan,
        "used_nan_pct": _pct(nan, N),
        "rings_distribution": dict(sorted(dist.items())),
        "fallback_distribution": dict(sorted(fallback.items())),
    }


def _group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k, "") for k in keys)].append(r)
    return buckets


def _summarize_bucket(bucket: list[dict]) -> dict:
    N = len(bucket)
    exp = sum(r.get("expanded", 0) for r in bucket)
    nan = sum(r.get("used_nan", 0) for r in bucket)
    rings = [r.get("rings_used", -1) for r in bucket if r.get("rings_used", -1) >= 0]
    dist = Counter(rings)
    return {
        "n_rows": N,
        "expanded_n": exp,
        "expanded_pct": _pct(exp, N),
        "used_nan_n": nan,
        "used_nan_pct": _pct(nan, N),
        "rings_mean": (sum(rings) / len(rings)) if rings else float("nan"),
        "rings_dist": dict(sorted(dist.items())),
    }


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def print_overall(summary: dict):
    print_section("OVERALL")
    print(f"rows           : {summary['n_rows']}")
    print(f"expanded       : {summary['expanded_n']} ({summary['expanded_pct']})")
    print(f"used_nan       : {summary['used_nan_n']} ({summary['used_nan_pct']})")
    print(f"rings dist     : {summary['rings_distribution']}")
    print(f"fallback dist  : {summary['fallback_distribution']}")


def print_table(rows: list[dict], headers: list[str]):
    # minimal pretty printing
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))


def summarize_by_roi(rows: list[dict]):
    keys = ["region", "atlas", "roi"]
    buckets = _group(rows, keys)
    out_rows = []
    for key, bucket in sorted(buckets.items()):
        s = _summarize_bucket(bucket)
        out_rows.append({
            "region": key[0], "atlas": key[1], "roi": key[2],
            "rows": s["n_rows"],
            "expanded_n": s["expanded_n"], "expanded_pct": s["expanded_pct"],
            "used_nan_n": s["used_nan_n"], "used_nan_pct": s["used_nan_pct"],
            "rings_mean": f"{s['rings_mean']:.2f}" if s["rings_mean"] == s["rings_mean"] else "nan",
            "rings_dist": s["rings_dist"],
        })
    print_section("BY ROI (region/atlas/roi)")
    print_table(out_rows, ["region", "atlas", "roi", "rows", "expanded_n",
                           "expanded_pct", "used_nan_n", "used_nan_pct",
                           "rings_mean", "rings_dist"])
    return out_rows


def summarize_by_roi_hemi(rows: list[dict]):
    keys = ["region", "atlas", "roi", "hemisphere"]
    buckets = _group(rows, keys)
    out_rows = []
    for key, bucket in sorted(buckets.items()):
        s = _summarize_bucket(bucket)
        out_rows.append({
            "region": key[0], "atlas": key[1], "roi": key[2],
            "hemi": key[3],
            "rows": s["n_rows"],
            "expanded_n": s["expanded_n"], "expanded_pct": s["expanded_pct"],
            "used_nan_n": s["used_nan_n"], "used_nan_pct": s["used_nan_pct"],
            "rings_mean": f"{s['rings_mean']:.2f}" if s["rings_mean"] == s["rings_mean"] else "nan",
            "rings_dist": s["rings_dist"],
        })
    print_section("BY ROI × HEMISPHERE")
    print_table(out_rows, ["region", "atlas", "roi", "hemi", "rows", "expanded_n",
                           "expanded_pct", "used_nan_n", "used_nan_pct",
                           "rings_mean", "rings_dist"])
    return out_rows


def summarize_rings_hist(rows: list[dict]):
    rings = [r.get("rings_used", -1) for r in rows]
    hist = Counter(rings)
    print_section("RINGS USED HISTOGRAM (including -1 for fallback)")
    out = [{"rings_used": k, "count": v} for k, v in sorted(hist.items())]
    print_table(out, ["rings_used", "count"])
    return out


def summarize_fallback(rows: list[dict]):
    fb = Counter(r.get("fallback", "none") for r in rows)
    print_section("FALLBACK TYPE HISTOGRAM")
    out = [{"fallback": k, "count": v} for k, v in sorted(fb.items())]
    print_table(out, ["fallback", "count"])
    return out


def maybe_write_csv(out_dir: str | None, name: str, rows: list[dict]):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    else:
        # minimal writer
        cols = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if cols:
                w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[WROTE] {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize ROI extraction logs.")
    ap.add_argument("logs", nargs="+",
                    help="CSV files or glob patterns for *_extraction_log.csv")
    ap.add_argument("--out-dir", default=None,
                    help="If set, write summary CSVs here.")
    args = ap.parse_args(argv)

    rows = read_logs(args.logs)
    if not rows:
        return 1

    # Overall
    overall = summarize_overall(rows)
    print_overall(overall)

    # Detailed tables
    by_roi = summarize_by_roi(rows)
    by_roi_hemi = summarize_by_roi_hemi(rows)
    rings_hist = summarize_rings_hist(rows)
    fb_hist = summarize_fallback(rows)

    # Optional outputs
    maybe_write_csv(args.out_dir, "by_roi", by_roi)
    maybe_write_csv(args.out_dir, "by_roi_hemi", by_roi_hemi)
    maybe_write_csv(args.out_dir, "rings_hist", rings_hist)
    maybe_write_csv(args.out_dir, "fallback_hist", fb_hist)

    # Also write a tiny overall.json-like CSV
    overall_rows = [{
        "rows": overall["n_rows"],
        "expanded_n": overall["expanded_n"],
        "expanded_pct": overall["expanded_pct"],
        "used_nan_n": overall["used_nan_n"],
        "used_nan_pct": overall["used_nan_pct"],
        "rings_distribution": overall["rings_distribution"],
        "fallback_distribution": overall["fallback_distribution"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }]
    maybe_write_csv(args.out_dir, "overall", overall_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())