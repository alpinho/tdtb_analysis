"""
Transcribe the scanned forms of one subject into draft sheets.

Usage:
    python -m transcription.transcribe SUBJECT SCAN.pdf [SCAN.pdf ...]
        [--out DIR] [--no-model]

The scans can hold any forms in any order (e.g. one PDF with everything
but the post-session form of session 2, plus one PDF with that form):
each page is identified by comparing it with the blank forms.

Output, in DIR/sub-NN/ (default: the forms_drafts folder of tdtb_private):
  - one draft sheet per form (*.draft.tsv): question, answer, status
    ("ok" or "check") and how the answer was read;
  - review_sub-NN.pdf: each answer next to the part of the scan it comes
    from.
Check the rows marked "check" against the paper forms, correct the
answers in the draft sheets, then run finalize.py.

Post-session forms are assigned to sessions by page order (across the
PDFs in the order given), except that a form from a PDF whose file name
holds a session label ("ses-02") takes that session; behavioral sessions
come before imaging sessions. Forms whose dates do not follow this order
are marked "check". (On the curated scans, page order was right for 93%
of the forms; the dates, misread for a quarter of the forms, and the
labels written on the forms, misread by the model, were worse guides.)

Nothing leaves the computer: the handwriting model runs locally.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import argparse
import collections
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import pymupdf

from . import align, layout, read, review
from .sheets import MISSING, SHEET_FNAMES

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subject_lists  # noqa: E402

PRIVATE_DIR = os.path.expanduser('~/OneDrive/tdtb_admin/tdtb_private')
DRAFT_DIR = os.path.join(PRIVATE_DIR, 'forms_drafts')

N_PAGES = {'demographic': 3, 'gold_msi': 8, 'postexp': 2}
# Least coverage ratio (see align.COVERAGE_RATIO) for a look-alike page
# to be identified by elimination
ELIMINATION_RATIO = 1.1
SESSION_BAND = (300, 20, 595, 98)   # top of the post-session form (points)


@dataclass
class ScanPage:
    pdf: str
    index: int
    page: object
    match: object
    by_elimination: bool = False


@dataclass
class Form:
    """One filled-in form: its pages, in order."""
    form: str
    pages: dict = field(default_factory=dict)     # form page -> ScanPage
    warped: dict = field(default_factory=dict)
    answers: list = field(default_factory=list)
    session_label: str = None
    session_from: str = ''
    sheet: str = None
    note: str = ''

    def source(self):
        return ', '.join('%s p.%d' % (os.path.basename(p.pdf), p.index + 1)
                         for _, p in sorted(self.pages.items()))


def group_forms(pages):
    """Group identified pages into forms (a new form starts at its page
    1, or when a page of the current form repeats)."""
    forms, current = [], {}
    for sp in pages:
        f = sp.match.form
        cur = current.get(f)
        if cur is None or sp.match.page in cur.pages or sp.match.page == 0:
            cur = Form(f)
            forms.append(cur)
            current[f] = cur
        cur.pages[sp.match.page] = sp
    return forms


def _date(s):
    try:
        return datetime.strptime(s, '%d/%m/%Y')
    except (TypeError, ValueError):
        return None


def _label_from_name(pdf):
    name = os.path.basename(pdf).lower()
    m = (re.search(r'ses[-_ ]?0?(\d)', name)
         or re.search(r'postbehav\d+-(\d)\.pdf$', name))
    return m.group(1) if m else None


def assign_sessions(posts, subject):
    """Name the sheet of each post-session form (behav before MRI)."""
    first_batch = subject in subject_lists.BEHAV_SUBJECTS
    known = first_batch or subject in (subject_lists.GOOD_SB_SUBJECTS
                                       + subject_lists.GOOD_TB_SUBJECTS)
    n_behav = 3 if first_batch else 2
    n_mri = 2 if subject in subject_lists.IMG_SUBJECTS else 0
    notes = []
    if not known:
        notes.append('sub-%02d is not in subject_lists.py yet: assumed 2 '
                      'behavioral sessions and no imaging session'
                      % subject)
    slots = (['behav_postexp', s] for s in range(1, n_behav + 1))
    slots = list(slots) + [['mri_postexp', s] for s in range(1, n_mri + 1)]
    # Page order, but a form from a PDF labelled with a session in its file
    # name takes that session
    fixed = {}
    for i, p in enumerate(posts):
        lab = p.session_label
        if lab and 1 <= int(lab) <= n_behav and int(lab) - 1 not in fixed:
            fixed[int(lab) - 1] = i
    rest = iter(i for i in range(len(posts)) if i not in fixed.values())
    order = [fixed[r] if r in fixed else next(rest)
             for r in range(len(posts))]
    # Dates (only those within a year of the median: others are misread)
    # must follow the order
    dates = [_date(p.date) for p in posts]
    read_dates = sorted(d for d in dates if d)
    if read_dates:
        med = read_dates[len(read_dates) // 2]
        dates = [d if d and abs((d - med).days) < 365 else None
                 for d in dates]
    seq = [dates[i] for i in order if dates[i]]
    dates_disagree = any(x >= y for x, y in zip(seq, seq[1:]))
    for rank, i in enumerate(order):
        p = posts[i]
        if rank < len(slots):
            kind, ses = slots[rank]
            p.sheet = SHEET_FNAMES[kind] % (subject, '%02d' % ses)
            p.note = ('session from the file name' if i in fixed.values()
                      else 'session from the page order')
            if dates_disagree:
                p.note += ('; the dates of the forms DISAGREE with this '
                           'order: check the sessions')
        else:
            p.sheet = 'postexp_sub-%02d_unassigned-%d.tsv' % (subject, rank)
            p.note = 'more post-session forms than sessions'
    return notes


def fix_swapped_dates(forms):
    """Swap day and month of a date that matches another form's date only
    when swapped. The model sometimes swaps them (it reads 12/03 as 03/12),
    and the demographic form is usually filled in on the day of the first
    post-session form."""
    dates = {}
    for fm in forms:
        a = next((a for a in fm.answers if a.row == 'Date'), None)
        if a is not None and _date(a.answer):
            dates[id(fm)] = a
    for fm in forms:
        a = dates.get(id(fm))
        if a is None:
            continue
        d, m, y = a.answer.split('/')
        swapped = '%s/%s/%s' % (m, d, y)
        others = {b.answer for k, b in dates.items() if k != id(fm)}
        if d != m and _date(swapped) and swapped in others \
                and a.answer not in others:
            a.answer = fm.date = swapped
            a.detail += ('; day and month swapped to match the date of '
                         'another form')
            a.status = 'check'


def write_draft(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['question', 'answer', 'status', 'how it was read'])
        for r in rows:
            w.writerow(r)


def run(subject, pdfs, out_dir, use_model=True, log=print):
    t0 = time.time()
    templates = align.Templates()
    fields = layout.load_fields()
    reader = None
    if use_model:
        from .handwriting import Reader
        reader = Reader()

    # 1. Identify every page
    pages, unknown = [], []
    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        for i, page in enumerate(doc):
            m = templates.identify(page)
            sp = ScanPage(pdf, i, page, m)
            (pages if m.ok else unknown).append(sp)
    # A look-alike page (Gold-MSI pages 2-4) whose identity is uncertain is
    # taken as its best guess when that page is otherwise missing (on the
    # curated scans, every uncertain page was the missing Gold-MSI page 4)
    found = collections.Counter((sp.match.form, sp.match.page)
                                for sp in pages)
    for sp in list(unknown):
        m = sp.match
        key = (m.form, m.page)
        if m.coverage_ratio is not None \
                and m.inliers >= align.MIN_INLIERS \
                and m.coverage_ratio >= ELIMINATION_RATIO \
                and found[key] < found[(m.form, 0)]:
            sp.by_elimination = True
            found[key] += 1
            unknown.remove(sp)
            pages.append(sp)
    order = {p: k for k, p in enumerate(pdfs)}
    pages.sort(key=lambda sp: (order[sp.pdf], sp.index))
    log('%d pages identified, %d not identified (%.0f s)'
        % (len(pages), len(unknown), time.time() - t0))
    forms = group_forms(pages)

    # 2. Read every form
    for fm in forms:
        for pno, sp in sorted(fm.pages.items()):
            warped = templates.align(sp.page, sp.match)
            fm.warped[pno] = warped
            blank = templates.pages[(fm.form, pno)]['img']
            on_page = [fl for fl in fields[fm.form] if fl.page == pno]
            for a in read.read_page(on_page, warped, blank, fm.form,
                                    reader):
                a.page = pno
                if sp.by_elimination:
                    a.status = 'check'
                    a.detail = ((a.detail + '; ') if a.detail else '') + \
                        'page identified by elimination'
                fm.answers.append(a)
        if fm.form == 'postexp' and 0 in fm.pages:
            band = read.text_crop(fm.warped[0], SESSION_BAND, (0, 0), False,
                                  templates.pages[('postexp', 0)]['img'])
            fm.band = band
            fm.session_label = _label_from_name(fm.pages[0].pdf)
            fm.session_from = 'file name' if fm.session_label else ''
        by_row = {a.row: a for a in fm.answers}
        fm.date = by_row['Date'].answer if 'Date' in by_row else None
    log('pages read (%.0f s)' % (time.time() - t0))

    fix_swapped_dates(forms)

    # 3. Sheets
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    demo = [f for f in forms if f.form == 'demographic']
    gold = [f for f in forms if f.form == 'gold_msi']
    posts = [f for f in forms if f.form == 'postexp']
    for f in demo:
        f.sheet = SHEET_FNAMES['demographic'] % subject
    for f in gold:
        f.sheet = SHEET_FNAMES['gold_msi'] % subject
    notes += assign_sessions(posts, subject)
    for kind, found in (('demographic', demo), ('gold_msi', gold)):
        if len(found) != 1:
            notes.append('%d %s forms found (expected 1)'
                         % (len(found), kind))
    demo_date = demo[0].date if demo else MISSING

    rev = review.Review('sub-%02d - transcription drafts - check the '
                        'orange rows against the paper forms' % subject)
    if notes or unknown:
        rev.heading('Notes', '')
        for n in notes:
            rev.row('Note', '', 'check', n)
        for sp in unknown:
            rev.image('Page not identified: %s p.%d (best guess %s p.%s, '
                      '%d matches)' % (os.path.basename(sp.pdf),
                                       sp.index + 1, sp.match.form,
                                       (sp.match.page or 0) + 1,
                                       sp.match.inliers),
                      review.thumbnail(align.render(sp.page, 60)))
    written, counts = [], {'ok': 0, 'check': 0}
    seen = {}
    for fm in demo + gold + posts:
        sheet = fm.sheet
        if sheet in seen:
            seen[sheet] += 1
            sheet = sheet.replace('.tsv', '_copy%d.tsv' % seen[sheet])
        seen.setdefault(sheet, 1)
        form_key = fm.form
        by_row = {a.row: a for a in fm.answers}
        rows = []
        missing_pages = [p + 1 for p in range(N_PAGES[form_key])
                         if p not in fm.pages]
        rev.heading(sheet.replace('.tsv', ''),
                    'Source: %s.%s%s' % (
                        fm.source(),
                        ' Missing pages: %s.' % missing_pages
                        if missing_pages else '',
                        ' ' + fm.note if fm.note else ''))
        if form_key == 'postexp' and hasattr(fm, 'band'):
            rev.row('Session label area', fm.session_label or MISSING,
                    'check' if 'DISAGREE' in fm.note else 'ok',
                    'label from %s' % fm.session_from
                    if fm.session_label else 'no label found',
                    fm.band)
        for fl in fields[form_key]:
            if fl.row == 'Subject #':
                rows.append([fl.row, str(subject), 'ok', 'subject number'])
                continue
            if fl.kind == 'derived':
                if form_key == 'gold_msi' and fl.row == 'Date':
                    rows.append([fl.row, demo_date, 'ok', 'copy of the date '
                                 'of the demographic form (updated by '
                                 'finalize)'])
                else:
                    rows.append([fl.row, MISSING, 'check', fl.note])
                rev.row(fl.row, rows[-1][1], rows[-1][2], rows[-1][3])
                continue
            a = by_row.get(fl.row)
            if a is None:
                rows.append([fl.row, MISSING, 'check',
                             'page %d missing' % (fl.page + 1)])
                rev.row(fl.row, MISSING, 'check', rows[-1][3])
                continue
            rows.append([fl.row, a.answer, a.status, a.detail])
            rev.row(fl.row, a.answer, a.status, a.detail,
                    review.crop_for(fm.warped[a.page], a, form_key))
        if 'DISAGREE' in fm.note or 'unassigned' in sheet:
            rows[1][2] = 'check'
            rows[1][3] = (rows[1][3] + '; ' + fm.note).strip('; ')
        for r in rows:
            counts[r[2]] = counts.get(r[2], 0) + 1
        path = os.path.join(out_dir, sheet.replace('.tsv', '.draft.tsv'))
        write_draft(path, rows)
        written.append(path)
    review_path = os.path.join(out_dir, 'review_sub-%02d.pdf' % subject)
    rev.save(review_path)
    log('%d draft sheets and %s written to %s (%.0f s)'
        % (len(written), os.path.basename(review_path), out_dir,
           time.time() - t0))
    log('answers: %d ok, %d to check' % (counts['ok'], counts['check']))
    if reader:
        log('handwriting model: %d calls, %.0f s in total'
            % (reader.calls, reader.seconds))
    for n in notes:
        log('NOTE: ' + n)
    return written, review_path


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Transcribe the scanned forms of one subject into '
                    'draft sheets (see the module docstring).')
    p.add_argument('subject', type=int)
    p.add_argument('scans', nargs='+', help='scanned PDF(s)')
    p.add_argument('--out', default=DRAFT_DIR,
                   help='folder for the drafts (default: %(default)s)')
    p.add_argument('--no-model', action='store_true',
                   help='do not read handwriting (marks only)')
    a = p.parse_args(argv)
    out = os.path.join(a.out, 'sub-%02d' % a.subject)
    run(a.subject, a.scans, out, use_model=not a.no_model)


if __name__ == '__main__':
    main()
