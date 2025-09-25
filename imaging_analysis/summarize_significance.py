#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate significant posthocs per folder and write a single log.

Rules
-----
- Map "*_anova.tsv" -> "*_posthoc.tsv" in the same folder.
- ANOVA significance uses corrected p only: first of
  ['p-GG-corr', 'p-HF-corr', 'p-corr', 'p_corrected'].
- Ignore main effect "ROI".
- Posthoc rows are listed when p-unc < alpha; mark "(corrected SIG)"
  when p-corr < alpha.
- If an ANOVA has no significant posthoc rows for any effect, it is
  omitted from the log.
- In the log, include all columns before any mean/std column, plus T,
  p-unc, and p-corr if present.
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

ALPHA_DEFAULT = 0.05

ANOVA_PCORR_CANDS = ['p-GG-corr', 'p-HF-corr', 'p-corr', 'p_corrected']
POSTHOC_PUNC_CANDS = ['p-unc', 'p_unc', 'punc']
POSTHOC_PCORR_CANDS = ['p-corr', 'p_corr', 'pcorr']
TSTAT_CANDS = ['T', 't', 'T-val', 'tval', 't-stat', 't_stat']


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


def _summarize_one_anova(anova_path: Path, alpha: float) -> Optional[str]:
    """
    Return a text block only if there are posthoc rows (p-unc < alpha)
    for at least one significant ANOVA effect (p-corr < alpha).
    """
    df_a = _read_tsv(anova_path)
    if df_a is None or df_a.empty:
        return None

    eff_col = _detect_eff_col(df_a)
    pc_col = _find_anova_pcorr(df_a)
    if pc_col is None:
        return None

    df_a[pc_col] = df_a[pc_col].map(_to_float)

    sig = []
    for _, row in df_a.iterrows():
        eff = str(row.get(eff_col, '')).strip()
        if not eff or eff.upper() == 'ROI':
            continue
        p_corr = row.get(pc_col, np.nan)
        if np.isfinite(p_corr) and p_corr < alpha:
            sig.append((eff, p_corr))

    if not sig:
        return None

    posthoc_path = _posthoc_path(anova_path)
    df_p = _read_tsv(posthoc_path)
    if df_p is None or df_p.empty:
        return None

    punc_col = _first_present(POSTHOC_PUNC_CANDS, df_p)
    pcorr_col = _first_present(POSTHOC_PCORR_CANDS, df_p)
    t_col = _find_tcol(df_p)

    blocks: List[str] = []
    for eff, p_corr in sig:
        ph_eff_col = _detect_eff_col(df_p)
        try:
            mask = df_p[ph_eff_col].astype(str).str.contains(
                eff, case=False, regex=False
            )
        except Exception:
            mask = pd.Series([True] * len(df_p), index=df_p.index)

        df_eff = df_p.loc[mask].copy()
        if punc_col:
            df_eff[punc_col] = pd.to_numeric(df_eff[punc_col],
                                             errors='coerce')
        if pcorr_col:
            df_eff[pcorr_col] = pd.to_numeric(df_eff[pcorr_col],
                                              errors='coerce')

        if not punc_col or punc_col not in df_eff.columns:
            continue

        df_sig = df_eff.loc[df_eff[punc_col] < alpha].copy()
        if df_sig.empty:
            continue

        lines = []
        lines.append(f'ANOVA: {anova_path.name}')
        lines.append(f'- Effect: {eff}  |  p-corr={p_corr:.4g}')
        lines.append(f'  Posthoc: {posthoc_path.name}')

        first_ms = _first_meanstd_idx(df_sig)
        base_cols = list(df_sig.columns[:first_ms])
        sel_cols = base_cols.copy()
        if t_col and t_col in df_sig.columns and t_col not in sel_cols:
            sel_cols.append(t_col)
        if punc_col and punc_col not in sel_cols:
            sel_cols.append(punc_col)
        if pcorr_col and pcorr_col not in sel_cols:
            sel_cols.append(pcorr_col)

        lines.append('    ' + '\t'.join(sel_cols))
        for _, rr in df_sig.iterrows():
            row_txt = '\t'.join(str(rr.get(c, '')) for c in sel_cols)
            mark = ''
            if pcorr_col and pd.notna(rr.get(pcorr_col)):
                try:
                    if float(rr.get(pcorr_col)) < alpha:
                        mark = '  (corrected SIG)'
                except Exception:
                    pass
            lines.append(f'    {row_txt}{mark}')
        lines.append('')

        blocks.append('\n'.join(lines).rstrip())

    if not blocks:
        return None

    # Deduplicate identical heads if multiple effects wrote the same ANOVA
    # name line; keep as-is for clarity per effect.
    return '\n'.join(blocks) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate significant posthocs per folder.'
    )
    parser.add_argument('root_dir', type=str, help='Root directory')
    parser.add_argument('--alpha', type=float, default=ALPHA_DEFAULT,
                        help='Significance threshold')
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        print(f'[ERROR] Root dir does not exist: {root}', file=sys.stderr)
        sys.exit(1)

    anovas = sorted(root.rglob('*_anova.tsv'))
    if not anovas:
        print('[WARN] No "*_anova.tsv" files found.', file=sys.stderr)
        sys.exit(0)

    folder_blocks: Dict[Path, List[str]] = {}
    for ap in anovas:
        block = _summarize_one_anova(ap, alpha=args.alpha)
        if block is None:
            continue
        folder = ap.parent
        folder_blocks.setdefault(folder, []).append(block)
        print(f'[OK] posthoc-significant: {ap}')

    if not folder_blocks:
        print('[INFO] No posthoc-significant effects under root.')
        sys.exit(0)

    for folder, blocks in folder_blocks.items():
        log_path = folder / 'stats_log.txt'
        header = (f'=== SIGNIFICANCE SUMMARY (alpha={args.alpha}) ===\n'
                  f'Folder: {folder.name}\n')
        body = '\n'.join(blocks)
        log_path.write_text(header + '\n' + body + '\n')
        print(f'[WROTE] {log_path}')

    print('[DONE] Wrote logs for folders with posthoc significance.')


if __name__ == '__main__':
    main()