"""
Check the corrected draft sheets of one subject and write the final
sheets read by the questionnaire scripts.

Usage:
    python -m transcription.finalize SUBJECT [DRAFT ...] [--reviewed]
        [--overwrite] [--drafts DIR] [--forms DIR]

Run it after checking the review PDF and correcting the answers in the
draft sheets (*.draft.tsv; only the answer column matters). Rows still
marked "check" stop the script, unless --reviewed tells it that they were
all checked against the paper forms.

Every answer is checked: the rows and their wording, the options of the
circled and ticked questions, numbers, dates (DD/MM/YYYY) and
percentages; missing answers must be n/a. The date of the Gold-MSI sheet
is the date of the demographic form. Final sheets are 2-column TSVs
(UTF-8, no BOM, LF line ends), written to DIR/sub-NN (default: the forms
folder of tdtb_private).

Which drafts: without DRAFT, every draft of the subject that has no
final sheet yet (drafts whose sheet exists are skipped, so forms added
later are finalized alone); with --overwrite, every draft. With DRAFT
(one or more draft files, by name or path), only those; replacing an
existing sheet then needs --overwrite.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import argparse
import csv
import glob
import os
import re
import sys

from . import layout
from .sheets import MISSING, SHEET_ROWS
from .transcribe import DRAFT_DIR, FORMS_DIR

PATTERNS = {'number': r'\d+(\.\d+)?(-\d+(\.\d+)?)?',
            'date': r'\d{2}/\d{2}/\d{4}',
            'percent': r'\d+(\.\d+)?%'}


def sheet_kind(fname):
    for kind in ('demographic', 'gold_msi', 'behav_postexp', 'mri_postexp'):
        if fname.startswith(kind.replace('demographic', 'demographic_info')):
            return kind
    return None


def read_draft(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    if not rows or rows[0][:2] != ['question', 'answer']:
        raise ValueError('not a draft sheet (no header)')
    return [r + [''] * (4 - len(r)) for r in rows[1:]]


def check(kind, rows, subject, fields):
    """Return the problems of a draft, as a list of strings."""
    problems = []
    expected = SHEET_ROWS[kind]
    got = [r[0] for r in rows]
    if got != expected:
        missing = [q for q in expected if q not in got]
        extra = [q for q in got if q not in expected]
        problems.append('rows differ from the sheet format (missing: %s; '
                        'unexpected: %s)' % (missing, extra))
        return problems
    form = 'postexp' if kind.endswith('postexp') else kind
    by_row = {fl.row: fl for fl in fields[form]}
    for q, a, status, _ in rows:
        short = q if len(q) < 50 else q[:47] + '...'
        if a != a.strip() or '\t' in a or '\n' in a or not a:
            problems.append('"%s": empty answer or extra spaces (use n/a '
                            'for missing answers)' % short)
            continue
        if q == 'Subject #':
            if a != str(subject):
                problems.append('"Subject #" is %s, not %d' % (a, subject))
            continue
        if a == MISSING:
            continue
        fl = by_row[q]
        values = [v for v, _ in fl.options]
        if fl.kind == 'tick' and len(values) == 1:
            ok = a in ('Yes', 'No')
        elif fl.kind == 'yes_text':
            ok = a in ('Yes', 'No') or a.startswith('Yes: ')
        elif fl.kind == 'tick' and 'Other' in values:
            ok = a in values or a.startswith('Other: ')
        elif fl.kind in ('circle', 'tick'):
            ok = a in values
        elif fl.text_kind in PATTERNS:
            ok = re.fullmatch(PATTERNS[fl.text_kind], a) is not None
        else:
            ok = True
        if not ok:
            allowed = (values if values and fl.kind != 'text'
                       else 'a %s' % fl.text_kind)
            problems.append('"%s": "%s" is not allowed (expected %s or n/a)'
                            % (short, a, allowed))
    return problems


def write_sheet(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        for q, a, *_ in rows:
            w.writerow([q, a])


def select_drafts(subject, drafts_dir, forms_dir, files, overwrite, log):
    """Paths of the drafts to finalize, or None if a named one is wrong."""
    sub = 'sub-%02d' % subject
    out_dir = os.path.join(forms_dir, sub)
    if files:
        paths, ok = [], True
        for f in files:
            path = f if os.path.exists(f) else os.path.join(
                drafts_dir, sub, os.path.basename(f))
            if not path.endswith('.draft.tsv') or not os.path.isfile(path):
                log('%s: no such draft (in %s)'
                    % (f, os.path.join(drafts_dir, sub)))
                ok = False
            paths.append(path)
        return paths if ok else None
    drafts = sorted(glob.glob(os.path.join(drafts_dir, sub, '*.draft.tsv')))
    if overwrite:
        return drafts
    new = []
    for path in drafts:
        fname = os.path.basename(path).replace('.draft.tsv', '.tsv')
        if os.path.exists(os.path.join(out_dir, fname)):
            log('%s: skipped, already final (name it and use --overwrite '
                'to replace)' % fname)
        else:
            new.append(path)
    return new


def run(subject, drafts_dir, forms_dir, reviewed=False, overwrite=False,
        files=None, log=print):
    sub = 'sub-%02d' % subject
    drafts = select_drafts(subject, drafts_dir, forms_dir, files, overwrite,
                           log)
    if drafts is None:
        log('Nothing written.')
        return False
    if not drafts:
        log('No drafts to finalize in %s' % os.path.join(drafts_dir, sub))
        return not files
    fields = layout.load_fields()
    out_dir = os.path.join(forms_dir, sub)
    sheets, failed = {}, False
    for path in drafts:
        fname = os.path.basename(path).replace('.draft.tsv', '.tsv')
        kind = sheet_kind(fname)
        if kind is None or '_copy' in fname or 'unassigned' in fname:
            log('%s: rename it to its sheet name (e.g. '
                'behav_postexp_sub-NN_ses-03.draft.tsv) or delete it'
                % os.path.basename(path))
            failed = True
            continue
        try:
            rows = read_draft(path)
        except ValueError as e:
            log('%s: %s' % (fname, e))
            failed = True
            continue
        problems = check(kind, rows, subject, fields)
        unchecked = [r[0] for r in rows if r[2] == 'check']
        if unchecked and not reviewed:
            problems.append('%d rows still marked "check" (check them, then '
                            'run again with --reviewed)' % len(unchecked))
        if os.path.exists(os.path.join(out_dir, fname)) and not overwrite:
            problems.append('the sheet already exists in %s (use '
                            '--overwrite to replace it)' % out_dir)
        for p in problems:
            log('%s: %s' % (fname, p))
        failed |= bool(problems)
        sheets[fname] = (kind, rows)
    if failed:
        log('Nothing written: fix the problems above and run again.')
        return False
    # The Gold-MSI form has no date: it is the demographic form's date
    demo = next((rows for k, rows in sheets.values()
                 if k == 'demographic'), None)
    if demo is None:
        final_demo = os.path.join(out_dir, 'demographic_info_%s.tsv' % sub)
        if os.path.exists(final_demo):
            with open(final_demo, newline='', encoding='utf-8') as f:
                demo = list(csv.reader(f, delimiter='\t'))
    for fname, (kind, rows) in sheets.items():
        if kind == 'gold_msi' and demo:
            date = next(r[1] for r in demo if r[0] == 'Date')
            for r in rows:
                if r[0] == 'Date':
                    r[1] = date
    os.makedirs(out_dir, exist_ok=True)
    for fname, (kind, rows) in sheets.items():
        write_sheet(os.path.join(out_dir, fname), rows)
        log('written %s' % os.path.join(out_dir, fname))
    log('Next: run reversed_scales_screen.py (questionnaires folder) to '
        'list the post-session questionnaires to assess for a reversed '
        'scale.')
    return True


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Check the corrected drafts of one subject and write '
                    'the final sheets (see the module docstring).')
    p.add_argument('subject', type=int)
    p.add_argument('files', nargs='*', metavar='DRAFT',
                   help='draft(s) to finalize, by name or path (default: '
                        'those without a final sheet)')
    p.add_argument('--reviewed', action='store_true',
                   help='all rows marked "check" were checked')
    p.add_argument('--overwrite', action='store_true',
                   help='replace existing final sheets')
    p.add_argument('--drafts', default=DRAFT_DIR)
    p.add_argument('--forms', default=FORMS_DIR)
    a = p.parse_args(argv)
    ok = run(a.subject, a.drafts, a.forms, a.reviewed, a.overwrite,
             a.files)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
