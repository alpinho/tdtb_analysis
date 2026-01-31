#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write per-folder TSV with posthoc-significant rows and one master index
under bothmod_allmain_tasks.

Rules
-----
- Map "*_anova.tsv" -> "*_posthoc.tsv" in the same folder.
- Include ANOVAs only if they have at least one effect with corrected
  p < alpha and at least one matching posthoc row with p-unc < alpha.
- Ignore main effect "ROI" in ANOVA.
- Per-folder TSV columns:
  file, hemisphere, individualization, (all cols before mean/std),
  T (if found), p-unc, p-corr, corrected_sig ('yes' if p-corr < alpha).
- Master index TSV at bothmod_allmain_tasks: columns folder,
  n_rows (# significant posthoc rows), n_files (# unique posthoc files).
- If a folder has no posthoc-significant rows, no log is written.
- If a log exists, delete it before writing new.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 25th of September 2025
Last Update: January 2026

Compatibility: Python 3.10.16
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

ALPHA_DEFAULT = 0.05

ANOVA_PCORR_CANDS = ['p-GG-corr', 'p-HF-corr', 'p-corr', 'p_corrected']
POSTHOC_PUNC_CANDS = ['p-unc', 'p_unc', 'punc']
POSTHOC_PCORR_CANDS = ['p-corr', 'p_corr', 'pcorr']
TSTAT_CANDS = ['T', 't', 'T-val', 'tval', 't-stat', 't_stat']

HEMIS = ['bh', 'lh', 'rh']
INDIVS = ['i', 'i9a', 'i8a', 'i7a', 'i6a',
          'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']


def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        try:
            return pd.read_csv(path, sep='\t', header=None)
        except Exception:
            return None


def _first_present(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _find_anova_pcorr(df: pd.DataFrame) -> Optional[str]:
    return _first_present(ANOVA_PCORR_CANDS, df)


def _detect_eff_col(df: pd.DataFrame) -> str:
    for c in ['Source', 'Effect', 'contrast', 'Contrast',
              'Within', 'within']:
        if c in df.columns:
            return c
    return df.columns[0]


def _posthoc_path(anova_path: Path) -> Path:
    name = anova_path.name
    if name.endswith('_anova.tsv'):
        return anova_path.with_name(name[:-10] + '_posthoc.tsv')
    return anova_path.with_name(name.replace('anova.tsv', 'posthoc.tsv'))


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _find_tcol(df: pd.DataFrame) -> Optional[str]:
    c = _first_present(TSTAT_CANDS, df)
    if c:
        return c
    for col in df.columns:
        if re.fullmatch(r'(?i)t(\W*|$).*', str(col)):
            return col
    return None


def _first_meanstd_idx(df: pd.DataFrame) -> int:
    for i, col in enumerate(df.columns):
        s = str(col).lower()
        if 'mean' in s or 'std' in s:
            return i
    return len(df.columns)


def _parse_tags_from_name(name: str) -> Tuple[str, str]:
    """Parse hemisphere and individualization from filename tokens."""
    stem = name.split('.')[0]
    toks = stem.split('_')
    indiv = toks[0] if len(toks) > 0 else ''
    hemi = toks[1] if len(toks) > 1 else ''
    hemi_out = hemi if hemi in HEMIS else 'other'
    indiv_out = indiv if indiv in INDIVS else 'other'
    return hemi_out, indiv_out


def _collect_rows_for_anova(anova_path: Path, alpha: float
                            ) -> Optional[pd.DataFrame]:
    """Return DataFrame of posthoc rows (p-unc < alpha) for sig effects."""
    df_a = _read_tsv(anova_path)
    if df_a is None or df_a.empty:
        return None

    eff_col = _detect_eff_col(df_a)
    pc_col = _find_anova_pcorr(df_a)
    if pc_col is None:
        return None

    df_a[pc_col] = df_a[pc_col].map(_to_float)

    sig_eff = []
    for _, row in df_a.iterrows():
        eff = str(row.get(eff_col, '')).strip()
        if not eff or eff.upper() == 'ROI':
            continue
        p_corr = row.get(pc_col, np.nan)
        if np.isfinite(p_corr) and p_corr < alpha:
            sig_eff.append(eff)

    if not sig_eff:
        return None

    posthoc_path = _posthoc_path(anova_path)
    df_p = _read_tsv(posthoc_path)
    if df_p is None or df_p.empty:
        return None

    punc_col = _first_present(POSTHOC_PUNC_CANDS, df_p)
    pcorr_col = _first_present(POSTHOC_PCORR_CANDS, df_p)
    t_col = _find_tcol(df_p)

    if punc_col:
        df_p[punc_col] = pd.to_numeric(df_p[punc_col], errors='coerce')
    if pcorr_col:
        df_p[pcorr_col] = pd.to_numeric(df_p[pcorr_col], errors='coerce')

    ph_eff_col = _detect_eff_col(df_p)

    hemi, indiv = _parse_tags_from_name(anova_path.name)
    rows = []

    for eff in sig_eff:
        try:
            mask = df_p[ph_eff_col].astype(str).str.contains(
                eff, case=False, regex=False
            )
        except Exception:
            mask = pd.Series([True] * len(df_p), index=df_p.index)

        df_eff = df_p.loc[mask].copy()
        if not punc_col or punc_col not in df_eff.columns:
            continue

        df_sig = df_eff.loc[df_eff[punc_col] < alpha].copy()
        if df_sig.empty:
            continue

        first_ms = _first_meanstd_idx(df_sig)
        base_cols = list(df_sig.columns[:first_ms])
        sel_cols = base_cols.copy()
        if t_col and t_col in df_sig.columns and t_col not in sel_cols:
            sel_cols.append(t_col)
        if punc_col and punc_col not in sel_cols:
            sel_cols.append(punc_col)
        if pcorr_col and pcorr_col not in sel_cols:
            sel_cols.append(pcorr_col)

        seen = set()
        uniq_cols = []
        for c in sel_cols:
            if c not in seen:
                uniq_cols.append(c)
                seen.add(c)

        for _, rr in df_sig.iterrows():
            rec = {
                'file': posthoc_path.name,
                'hemisphere': hemi,
                'individualization': indiv
            }
            for c in uniq_cols:
                rec[c] = rr.get(c, '')
            corr_hit = False
            if pcorr_col and pd.notna(rr.get(pcorr_col)):
                try:
                    corr_hit = float(rr.get(pcorr_col)) < alpha
                except Exception:
                    corr_hit = False
            rec['corrected_sig'] = 'yes' if corr_hit else ''
            rows.append(rec)

    if not rows:
        return None

    df_out = pd.DataFrame(rows)
    posthoc_cols = [c for c in df_out.columns
                    if c not in ['file', 'hemisphere',
                                 'individualization', 'corrected_sig']]
    ordered = (['file', 'hemisphere', 'individualization'] +
               posthoc_cols + ['corrected_sig'])
    return df_out[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=('Per-folder TSVs of posthoc-significant rows and a '
                     'master index at bothmod_allmain_tasks.')
    )
    parser.add_argument('root_dir', type=str, help='Root directory to scan')
    parser.add_argument('--alpha', type=float, default=ALPHA_DEFAULT,
                        help='Significance threshold')
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        print(f'[ERROR] Root dir does not exist: {root}', file=sys.stderr)
        sys.exit(1)

    # find the common bothmod_allmain_tasks dir
    bothmod_root = None
    for parent in root.parents:
        if parent.name == 'bothmod_allmain_tasks':
            bothmod_root = parent
            break
    if bothmod_root is None:
        bothmod_root = root

    anovas = sorted(root.rglob('*_anova.tsv'))
    if not anovas:
        print('[WARN] No "*_anova.tsv" files found.', file=sys.stderr)
        sys.exit(0)

    per_folder: Dict[Path, List[pd.DataFrame]] = {}
    for ap in anovas:
        df_rows = _collect_rows_for_anova(ap, alpha=args.alpha)
        if df_rows is None:
            continue
        per_folder.setdefault(ap.parent, []).append(df_rows)
        print(f'[OK] posthoc-significant rows from: {ap}')

    if not per_folder:
        print('[INFO] No posthoc-significant rows under root.')
        sys.exit(0)

    index_rows = []
    for folder, df_list in per_folder.items():
        if not df_list:
            continue
        log_tsv = folder / 'stats_log.tsv'
        for old in [folder / 'stats_log.txt', log_tsv]:
            if old.exists():
                try:
                    old.unlink()
                except Exception:
                    pass

        df_all = pd.concat(df_list, ignore_index=True)
        df_all['hemisphere'] = pd.Categorical(
            df_all['hemisphere'], categories=HEMIS + ['other'], ordered=True
        )
        df_all['individualization'] = pd.Categorical(
            df_all['individualization'],
            categories=INDIVS + ['other'], ordered=True
        )
        df_all.sort_values(['hemisphere', 'individualization'],
                           inplace=True)
        df_all.to_csv(log_tsv, sep='\t', index=False)
        print(f'[WROTE] {log_tsv}')

        n_rows = int(df_all.shape[0])
        n_files = int(df_all['file'].nunique())
        index_rows.append({
            'folder': str(folder),
            'n_rows': n_rows,
            'n_files': n_files
        })

    index_path = bothmod_root / 'significant_folders.tsv'
    if index_path.exists():
        try:
            index_path.unlink()
        except Exception:
            pass
    idx_df = pd.DataFrame(index_rows).sort_values('folder')
    idx_df.to_csv(index_path, sep='\t', index=False)
    print(f'[WROTE] {index_path}')

    print('[DONE] Wrote per-folder logs and master index.')


if __name__ == '__main__':
    main()