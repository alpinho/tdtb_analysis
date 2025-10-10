
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze real-vs-theoretical ISI from .xpd logs inside a ZIP archive.

The script scans all .xpd files (optionally only those under
".../imaging_sessions/"), extracts the theoretical ISI and the
measured real ISI from each row, and computes summary tables per
user-provided groups of target theoretical ISIs.

Outputs are TSV files per group with columns:
  theoretical, count, mean_real, mean_diff, std_diff, min_diff,
  max_diff, mean_abs, perc_large_5

It also writes a TSV of outliers (>5 ms absolute deviation) per
group with columns:
  subject, session, block, file, theoretical, real, diff

Usage example:
  python xpd_isi_analysis.py \\
    --zip /path/to/logfiles_filtered.zip \\
    --out out_dir \\
    --imaging-only \\
    --group "fast:367,404,408,449,450,468,490" \\
    --group "classic:459,510,561,612,663"

Notes
-----
* Lines are expected to be comma-separated like:
    47,2,3,2,interval03,interval_1,370425,482,561,561,-,-
  where columns -4 and -3 are theoretical and real ISI.
* Rows that cannot be parsed are skipped.
* All statistics are in milliseconds.
"""

import argparse
import csv
import io
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


@dataclass
class Event:
# Keep lines <=79 chars for PEP8 friendliness
    file: str
    subject: str
    session: str
    block: str
    theoretical: int
    real: int

    @property
    def diff(self) -> int:
        return self.real - self.theoretical

    @property
    def abs_diff(self) -> int:
        return abs(self.diff)


def parse_groups(raw_groups: Sequence[str]) -> Dict[str, List[int]]:
    """
    Parse --group arguments of the form "name:comma,separated,ints".
    """
    groups = {}
    for g in raw_groups:
        if ':' not in g:
            raise ValueError(
                f'Invalid --group "{g}". Use "name:val1,val2,..."'
            )
        name, vals = g.split(':', 1)
        nums = [int(v.strip()) for v in vals.split(',') if v.strip()]
        if not nums:
            raise ValueError(f'Group "{name}" has no numbers.')
        groups[name] = nums
    return groups


def iter_events(zip_path: Path,
                imaging_only: bool = True) -> Iterable[Event]:
    """
    Yield Events from all .xpd files inside the given ZIP archive.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.xpd'):
                continue
            if imaging_only and '/imaging_sessions/' not in name:
                continue

            with zf.open(name) as fh:
                for raw in fh:
                    try:
                        line = raw.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        continue
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    # Need at least the last 4 columns
                    if len(parts) < 11:
                        continue
                    try:
                        theor = int(parts[-4])
                        real = int(parts[-3])
                    except ValueError:
                        continue

                    subject = parts[0]
                    session = parts[1] if len(parts) > 1 else ''
                    block = parts[2] if len(parts) > 2 else ''

                    yield Event(
                        file=name,
                        subject=subject,
                        session=session,
                        block=block,
                        theoretical=theor,
                        real=real,
                    )


def summarize(events: Iterable[Event],
              targets: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build summary and outlier tables for the provided target ISIs.
    """
    rows = []
    outlier_rows = []
    targets_set = set(targets)
    for ev in events:
        if ev.theoretical not in targets_set:
            continue
        rows.append({
            'file': ev.file,
            'subject': ev.subject,
            'session': ev.session,
            'block': ev.block,
            'theoretical': ev.theoretical,
            'real': ev.real,
            'diff': ev.diff,
        })
        if ev.abs_diff > 5:
            outlier_rows.append({
                'subject': ev.subject,
                'session': ev.session,
                'block': ev.block,
                'file': ev.file,
                'theoretical': ev.theoretical,
                'real': ev.real,
                'diff': ev.diff,
            })

    if not rows:
        return (pd.DataFrame(columns=[
                    'theoretical', 'count', 'mean_real', 'mean_diff',
                    'std_diff', 'min_diff', 'max_diff', 'mean_abs',
                    'perc_large_5'
                ]),
                pd.DataFrame(columns=[
                    'subject', 'session', 'block', 'file', 'theoretical',
                    'real', 'diff'
                ]))

    df = pd.DataFrame(rows)
    df['abs_diff'] = df['diff'].abs()

    grp = (
        df.groupby('theoretical')
          .agg(
              count=('diff', 'size'),
              mean_real=('real', 'mean'),
              mean_diff=('diff', 'mean'),
              std_diff=('diff', 'std'),
              min_diff=('diff', 'min'),
              max_diff=('diff', 'max'),
              mean_abs=('abs_diff', 'mean'),
          )
          .reset_index()
          .sort_values('theoretical')
    )
    # percentage of trials with abs deviation > 5 ms
    perc = (
        df.assign(flag=lambda x: (x['abs_diff'] > 5).astype(int))
          .groupby('theoretical')
          .agg(perc_large_5=('flag', lambda s: 100 * s.mean()))
          .reset_index()
    )
    summary = grp.merge(perc, on='theoretical', how='left')

    outliers = pd.DataFrame(outlier_rows)
    return summary, outliers


def write_tsv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False, float_format='%.6f')


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(
        description='Summarize ISI deviations from .xpd logs in a ZIP.'
    )
    p.add_argument('--zip', required=True, type=Path,
                   help='Path to ZIP containing .xpd files.')
    p.add_argument('--out', required=True, type=Path,
                   help='Directory to write TSV outputs.')
    p.add_argument('--imaging-only', action='store_true',
                   help='Restrict to ".../imaging_sessions/".')
    p.add_argument('--group', action='append', default=[],
                   help='Group spec "name:v1,v2,...". Can be repeated.')
    args = p.parse_args(argv)

    if not args.group:
        p.error('Please provide at least one --group "name:vals".')

    groups = parse_groups(args.group)

    events = list(iter_events(args.zip, imaging_only=args.imaging_only))

    for name, targets in groups.items():
        summary, outliers = summarize(iter(events), targets)
        write_tsv(summary, args.out / f'summary_{name}.tsv')
        write_tsv(outliers, args.out / f'outliers_{name}.tsv')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
