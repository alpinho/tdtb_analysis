"""
Make fake filled-in forms to test the transcription without real data.

The blank forms are filled with random answers (circles, ticks and
script-font "handwriting"), then degraded like a scan: enlarged ~6%,
slightly tilted and shifted, turned by 0/90/180/270 degrees and blurred.
The true answers are written next to the PDF, as sheets.

Usage:
    python -m transcription.synthetic OUT_DIR [--seed N] [--subject N]

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import pymupdf

from . import layout, read
from .sheets import MISSING, SHEET_FNAMES

SCAN_DPI = 300
PT = SCAN_DPI / 72
FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

FAKE_TEXT = {
    'Date': ['12/03/2026', '05/11/2026', '28/07/2026'],
    'What is your first language?': ['English', 'Spanish', 'Mandarin'],
    'What other languages do you know?': ['French, Spanish', 'English'],
    'Please also rank the degree of fluency.': ['French: 2, Spanish: 4',
                                                'English: 1'],
    'What do you consider your dominant/main language:': ['English'],
    'If yes, which instrument(s)?': ['Piano', 'Guitar, Violin'],
    'What type of training did you receive?': ['Private lessons',
                                               'Self-taught'],
    'The instrument I play best (including voice) is _.': ['Piano',
                                                           'Voice'],
    'Nationality': ['Canadian', 'Brazilian'],
    'Country in which you spent the formative years of your childhood '
    'and youth:': ['Canada', 'Portugal'],
    'Country of current residency:': ['Canada'],
    'If yes, please describe briefly:': ['I counted the beats',
                                         'Tapped my foot'],
    'Do you have any additional comments regarding the experiment?':
        ['Very interesting', 'No'],
}
FAKE_KIND = {'number': ['2', '5', '10', '21', '3-4'],
             'date': ['12/03/2026', '05/11/2026'],
             'percent': ['80%', '50%', '100%']}


def fake_answers(fields, rng, blank_rate=0.08):
    """Random answer for every field of a form (value, text or n/a)."""
    out = {}
    for fl in fields:
        if fl.kind == 'derived':
            continue
        if rng.random() < blank_rate and fl.kind != 'tick':
            out[fl.row] = MISSING
        elif fl.kind in ('circle', 'tick'):
            value = rng.choice(fl.options)[0]
            if len(fl.options) == 1:
                value = rng.choice(['Yes', 'No'])
            if value == 'Other':
                value = 'Other: Trade school'
            out[fl.row] = value
        elif fl.kind == 'yes_text':
            out[fl.row] = rng.choice(['No', 'No', 'No', 'Yes: Asthma'])
        else:
            pool = FAKE_TEXT.get(fl.row) or FAKE_KIND.get(fl.text_kind) \
                or ['Something']
            out[fl.row] = rng.choice(pool)
    # The two language rows share one area on the paper: only the
    # fluency ("French: 2, Spanish: 4") is written, the list follows
    fluency = 'Please also rank the degree of fluency.'
    if fluency in out:
        langs = [x.split(':')[0] for x in out[fluency].split(', ')]
        out['What other languages do you know?'] = (
            MISSING if out[fluency] == MISSING else ', '.join(langs))
    return out


def _circle(img, rect, rng):
    x0, y0, x1, y1 = (v * PT for v in rect)
    c = (int((x0 + x1) / 2 + rng.uniform(-3, 3)),
         int((y0 + y1) / 2 + rng.uniform(-3, 3)))
    axes = (int((x1 - x0) / 2 + rng.uniform(8, 16)),
            int((y1 - y0) / 2 + rng.uniform(6, 12)))
    cv2.ellipse(img, c, axes, rng.uniform(-10, 10), rng.uniform(0, 30),
                rng.uniform(330, 380), 40, int(rng.uniform(3, 5)))


def _tick(img, rect, rng):
    x0, y0, x1, y1 = (v * PT for v in rect)
    a = (int(x0 + rng.uniform(-0.1, 0.2) * (x1 - x0)), int((y0 + y1) / 2))
    b = (int(x0 + 0.45 * (x1 - x0)), int(y1 + rng.uniform(2, 8)))
    c = (int(x1 + rng.uniform(15, 30)), int(y0 - rng.uniform(15, 30)))
    cv2.line(img, a, b, 40, 4)
    cv2.line(img, b, c, 40, 4)


def _write(img, rect, text, sideways, rng):
    """Script-font text in rect (turned for sideways pages)."""
    x0, y0, x1, y1 = (int(v * PT) for v in rect)
    w, h = (y1 - y0, x1 - x0) if sideways else (x1 - x0, y1 - y0)
    scale = 1.6
    while scale > 0.6:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 3)
        if tw < w * 0.95 and th < h * 0.8:
            break
        scale -= 0.1
    canvas = np.full((h, w), 255, np.uint8)
    cv2.putText(canvas, text, (int(rng.uniform(5, 20)), int(h * 0.7)),
                FONT, scale, 30, 3, cv2.LINE_AA)
    if sideways:
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    region = img[y0:y0 + canvas.shape[0], x0:x0 + canvas.shape[1]]
    np.minimum(region, canvas[:region.shape[0], :region.shape[1]],
               out=region)


def fill_form(doc, fields, answers, form, rng):
    """Return the filled pages of a form as 300-dpi images."""
    pages = []
    sideways = form in read.SIDEWAYS
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.h, pix.w).copy()
        for fl in fields:
            if fl.page != i or fl.kind == 'derived':
                continue
            value = answers[fl.row]
            if value == MISSING \
                    or fl.row == 'What other languages do you know?':
                continue
            if fl.kind in ('circle', 'tick', 'yes_text'):
                choice = value.split(':')[0]
                if len(fl.options) == 1:
                    if value == 'No':
                        continue
                    choice = fl.options[0][0]
                for v, r in fl.options:
                    if v == choice:
                        (_tick if fl.kind != 'circle' else _circle)(
                            img, r, rng)
                if ':' in value and fl.rect:
                    _write(img, fl.rect, value.split(': ', 1)[1],
                           sideways, rng)
            else:
                _write(img, fl.rect, value, sideways, rng)
        pages.append(img)
    return pages


def degrade(img, rng):
    """Make a clean page look scanned."""
    h, w = img.shape
    s = rng.uniform(1.03, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.2, 1.2), s)
    M[:, 2] += (rng.uniform(-40, 40), rng.uniform(-40, 40))
    out = cv2.warpAffine(img, M, (w, h), borderValue=255)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, 6, out.shape)
    out = np.clip(out.astype(float) + noise, 0, 255).astype(np.uint8)
    code = rng.choice(list(read.align.ROTATIONS.values()))
    return out if code is None else cv2.rotate(out, code)


def make(out_dir, subject=99, seed=0, n_post=2):
    """Write a fake subject's scans and true sheets into out_dir."""
    rng = random.Random(seed)
    fields = layout.load_fields()
    os.makedirs(out_dir, exist_ok=True)
    docs = {k: pymupdf.open(os.path.join(layout.TEMPLATE_DIR, v))
            for k, v in layout.TEMPLATE_FILES.items()}
    truth, pages = {}, []
    for form in ('demographic', 'gold_msi'):
        ans = fake_answers(fields[form], rng)
        truth[form] = ans
        pages += fill_form(docs[form], fields[form], ans, form, rng)
    post_pages = []
    for k in range(1, n_post + 1):
        ans = fake_answers(fields['postexp'], rng)
        ans['Date'] = '%02d/0%d/2026' % (10 + k, k)
        truth['behav_postexp', k] = ans
        post_pages.append(fill_form(docs['postexp'], fields['postexp'],
                                    ans, 'postexp', rng))
    # As often in practice: everything but session 2 in one PDF, session 2
    # in its own PDF
    files = {'all%02d.pdf' % subject: pages + post_pages[0]
             + sum(post_pages[2:], []),
             'postbehav%02d-2.pdf' % subject: post_pages[1]
             if n_post > 1 else []}
    for name, imgs in files.items():
        if not imgs:
            continue
        pdf = pymupdf.open()
        for img in imgs:
            img = degrade(img, rng)
            ok, jpg = cv2.imencode('.jpg', img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            page = pdf.new_page(width=img.shape[1] * 72 / SCAN_DPI,
                                height=img.shape[0] * 72 / SCAN_DPI)
            page.insert_image(page.rect, stream=jpg.tobytes())
        pdf.save(os.path.join(out_dir, name))
    truth_dir = os.path.join(out_dir, 'truth')
    os.makedirs(truth_dir, exist_ok=True)
    for key, ans in truth.items():
        if isinstance(key, tuple):
            fname = SHEET_FNAMES[key[0]] % (subject, '%02d' % key[1])
            rows = layout.load_fields()['postexp']
        else:
            fname = SHEET_FNAMES[key] % subject
            rows = fields[key]
        with open(os.path.join(truth_dir, fname), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            for fl in rows:
                if fl.row == 'Subject #':
                    w.writerow([fl.row, str(subject)])
                elif fl.kind == 'derived':
                    # Gold-MSI date: copy of the demographic date
                    w.writerow([fl.row, truth['demographic']['Date']
                                if fl.row == 'Date' else MISSING])
                else:
                    w.writerow([fl.row, ans[fl.row]])
    return list(files)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    p.add_argument('out_dir')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--subject', type=int, default=99)
    p.add_argument('--n-post', type=int, default=2)
    a = p.parse_args()
    print(make(a.out_dir, a.subject, a.seed, a.n_post))
