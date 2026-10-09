"""
Measure the accuracy of the transcription on the scans that already have
curated sheets.

Usage:
    python -m transcription.validate [--subjects 3 4 ...] [--no-model]

For every subject folder in tdtb_private/forms_scanned, the scans are
transcribed into tdtb_private/forms_validation/sub-NN/ (drafts and
review PDF, kept in the private folder) and each draft is compared with
the curated sheet of tdtb_private/forms. Subjects already compared are
skipped, so an interrupted run can be resumed.

Privacy: the output holds counts only (per form, per kind of answer and
per question), never an answer, a name or a file name; errors are
reported by their type only. The counts are also written to
tdtb_private/forms_validation/accuracy.txt.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import argparse
import collections
import csv
import difflib
import glob
import json
import os
import re
import time

from . import layout, transcribe
from .sheets import MISSING

SCAN_DIR = os.path.join(transcribe.PRIVATE_DIR, 'forms_scanned')
SHEET_DIR = os.path.join(transcribe.PRIVATE_DIR, 'forms')
OUT_DIR = os.path.join(transcribe.PRIVATE_DIR, 'forms_validation')
# Rows not read from the page: not counted
NOT_READ = {'Subject #', 'Date of Birth (DD/MM/YYYY)'}


def _norm(s):
    s = s.replace('’', "'").replace('‘', "'").lower()
    return ' '.join(s.split()).rstrip('.')


def _read(path, header):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    return rows[1:] if header else rows


def _kinds():
    """{form: {row: kind}} where kind is mark, text or yes_text."""
    out = {}
    for form, fields in layout.load_fields().items():
        out[form] = {fl.row: ('mark' if fl.kind in ('circle', 'tick')
                              else fl.kind) for fl in fields}
    return out


def compare_subject(sub_dir, subject, kinds):
    """Counts of agreement between the drafts and the curated sheets."""
    c = collections.Counter()
    per_q = collections.defaultdict(collections.Counter)
    sheet_dir = os.path.join(SHEET_DIR, 'sub-%02d' % subject)
    drafts = {os.path.basename(p).replace('.draft.tsv', '.tsv'): p
              for p in glob.glob(os.path.join(sub_dir, '*.draft.tsv'))}
    sheets = {os.path.basename(p): p
              for p in glob.glob(os.path.join(sheet_dir, '*.tsv'))}
    for name in sheets:
        if name not in drafts:
            c['sheets without a draft'] += 1
    for name, dpath in drafts.items():
        if name not in sheets:
            c['drafts without a sheet'] += 1
            continue
        form = transcribe_form(name)
        truth = {r[0]: r[1] for r in _read(sheets[name], False) if len(r) > 1}
        for row in _read(dpath, True):
            q, got, status = row[0], row[1], row[2]
            if q in NOT_READ or q not in truth:
                continue
            if form == 'gold_msi' and q == 'Date':
                kind = 'copied date'
            else:
                kind = kinds[form].get(q, 'other')
            t = truth[q]
            if got == t:
                res = 'same'
            elif _norm(got) == _norm(t):
                res = 'same but for case/spaces'
            elif kind in ('text', 'yes_text') and difflib.SequenceMatcher(
                    None, _norm(got), _norm(t)).ratio() >= 0.8:
                res = 'close (>=80% similar)'
            else:
                res = 'different'
            c[(form, kind, status, res)] += 1
            per_q[(form, q)][(status, res)] += 1
            if kind == 'yes_text':
                base = 'same' if got.split(':')[0] == t.split(':')[0] \
                    else 'different'
                c[(form, 'yes_text Yes/No part', status, base)] += 1
            if t == MISSING or got == MISSING:
                c[(form, kind, 'n/a in sheet' if t == MISSING
                   else 'n/a in draft only', res)] += 1
    return c, per_q


def transcribe_form(sheet_name):
    if sheet_name.startswith('demographic'):
        return 'demographic'
    if sheet_name.startswith('gold_msi'):
        return 'gold_msi'
    return 'postexp'


def report(total, per_q, n_subjects, errors, seconds):
    lines = ['Transcription accuracy (counts only)',
             'subjects compared: %d; errors: %s; time: %.1f h'
             % (n_subjects, dict(errors) or 0, seconds / 3600), '']
    lines.append('%-12s %-24s %-7s %-26s %6s' % ('form', 'answer kind',
                                                 'status', 'vs sheet', 'n'))
    keys = sorted((k for k in total if isinstance(k, tuple)), key=str)
    for k in keys:
        lines.append('%-12s %-24s %-7s %-26s %6d' % (k + (total[k],)))
    for k in sorted(k for k in total if not isinstance(k, tuple)):
        lines.append('%s: %d' % (k, total[k]))
    lines += ['', 'Per question: different from the sheet (marked ok / '
              'marked check) and number compared',
              '']
    for (form, q), cnt in sorted(per_q.items()):
        n = sum(cnt.values())
        bad_ok = sum(v for (s, r), v in cnt.items()
                     if s == 'ok' and r == 'different')
        bad_check = sum(v for (s, r), v in cnt.items()
                        if s == 'check' and r == 'different')
        if bad_ok or bad_check:
            lines.append('%-12s %4d / %4d of %4d  %s'
                         % (form, bad_ok, bad_check, n, q[:70]))
    return '\n'.join(lines)


def reversed_report():
    """Reversed-scale ratings (reversed_scales_screen.rate) from the
    drafts and from the curated sheets, against the ratings by hand."""
    import reversed_scales_screen as rs     # in the questionnaires folder
    hand = rs.recorded_ratings()
    drafts = rs.load(OUT_DIR, '*postexp*.draft.tsv', True)
    sheets = rs.load(SHEET_DIR, '*postexp*.tsv', False)
    lines = ['Reversed scales: rating by hand (rows) against the automatic '
             'rating from the drafts / from the curated sheets', '']
    a = rs.agreement(rs.rate_all(drafts), hand)
    b = rs.agreement(rs.rate_all(sheets), hand)
    for key in sorted(set(a) | set(b)):
        lines.append('%-15s %-15s %4d / %4d'
                     % (key + (a.get(key, 0), b.get(key, 0))))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description='Accuracy of the '
                                'transcription on already curated scans.')
    p.add_argument('--subjects', type=int, nargs='*')
    p.add_argument('--no-model', action='store_true')
    a = p.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    kinds = _kinds()
    folders = sorted(glob.glob(os.path.join(SCAN_DIR, 'sub-*')))
    t0 = time.time()
    errors = collections.Counter()
    for folder in folders:
        m = re.fullmatch(r'sub-(\d+)', os.path.basename(folder))
        if not m:
            continue
        subject = int(m.group(1))
        if a.subjects and subject not in a.subjects:
            continue
        out = os.path.join(OUT_DIR, 'sub-%02d' % subject)
        done = os.path.join(out, 'counts.json')
        if os.path.exists(done):
            continue
        pdfs = sorted(glob.glob(os.path.join(folder, '*.pdf')))
        # Tried twice: Ollama restarts by itself after a crash
        for attempt in (1, 2):
            try:
                transcribe.run(subject, pdfs, out,
                               use_model=not a.no_model, log=lambda *x: None)
                c, per_q = compare_subject(out, subject, kinds)
                break
            except Exception as e:          # report the type only
                errors[type(e).__name__] += 1
                print('a subject failed (%s), attempt %d'
                      % (type(e).__name__, attempt), flush=True)
                time.sleep(60)
        else:
            continue
        with open(done, 'w') as f:
            json.dump({'total': [[list(k) if isinstance(k, tuple) else k, v]
                                 for k, v in c.items()],
                       'per_q': [[list(k), [[list(kk), v]
                                            for kk, v in cnt.items()]]
                                 for k, cnt in per_q.items()]}, f)
        n_done = len(glob.glob(os.path.join(OUT_DIR, 'sub-*',
                                            'counts.json')))
        print('%d subjects done (%.1f h)' % (n_done,
                                             (time.time() - t0) / 3600),
              flush=True)
    # Totals over all compared subjects (also earlier runs)
    total = collections.Counter()
    per_q = collections.defaultdict(collections.Counter)
    files = glob.glob(os.path.join(OUT_DIR, 'sub-*', 'counts.json'))
    for path in files:
        d = json.load(open(path))
        for k, v in d['total']:
            total[tuple(k) if isinstance(k, list) else k] += v
        for k, cnt in d['per_q']:
            for kk, v in cnt:
                per_q[tuple(k)][tuple(kk)] += v
    text = report(total, per_q, len(files), errors, time.time() - t0)
    text += '\n\n' + reversed_report()
    with open(os.path.join(OUT_DIR, 'accuracy.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
