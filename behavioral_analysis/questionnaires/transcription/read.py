"""
Read the answers of an aligned page.

Circled options and tick boxes: the page is compared with the blank form
to find the ink added by hand. Each pen stroke (connected added ink) is
given whole to the option it touches whose centre is nearest, among all
the options of the page: a circle is centred on its option even where it
crosses into the box of the next option or question, and a tick on its
box. The option with most ink is the answer, unless no option or several
options were marked.

Handwriting: the area of the answer is sent to the local handwriting
model (handwriting.py) if it holds added ink.

Every answer gets a status: "ok" (clear mark, or nothing written), or
"check" (unclear mark, or text read by the model, which can be wrong).

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import align
from .sheets import MISSING

PT = align.DPI / 72          # pixels per point

# Ink given to an option (pixels at align.DPI): below BLANK, not marked;
# above MARKED, marked; in between, unclear. A second option with at
# least SECOND x the ink of the first makes the answer unclear. (On the
# curated scans, true blanks had under 15 pixels, and an option with
# 15-50 pixels and twice the ink of any other was the answer 174 times
# out of 178.)
BLANK = 15
MARKED = 100
SECOND = 0.5
# Strokes smaller than this (pixels) are specks, not marks
MIN_STROKE = 15
# Added ink pixels in a handwriting area below which it is empty
MIN_TEXT_PIXELS = 80

# Gold-MSI pages are printed sideways; text is turned upright before it
# is read
SIDEWAYS = {'gold_msi'}

# School level: if several boxes are ticked, the highest level is kept
SCHOOL_RANK = ['Elementary School', 'Less than Grade 12',
               'High school diploma',
               'Some university undergraduate schooling',
               'College Degree (2 years)', "Bachelor's degree",
               'Postgraduate degree']


@dataclass
class Answer:
    row: str
    answer: str = MISSING
    status: str = 'ok'
    detail: str = ''
    crop: tuple = None              # area shown in the review (points)
    marks: list = field(default_factory=list)   # (rect, chosen) shown
    page: int = None


def ink_masks(warped, blank):
    """Return (scan ink, ink added to the blank form)."""
    otsu, _ = cv2.threshold(warped, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scan_ink = warped < np.clip(otsu, 140, 200)
    # Light grey printing (table lines) counts as printed too: it comes
    # out darker on the scans. Printing is widened by ~1 mm, as scans are
    # never aligned perfectly.
    printed = cv2.dilate((blank < 240).astype(np.uint8),
                         np.ones((9, 9), np.uint8)) > 0
    return scan_ink, scan_ink & ~printed


def _px(rect, shift=(0, 0)):
    return (rect[0] * PT + shift[0], rect[1] * PT + shift[1],
            rect[2] * PT + shift[0], rect[3] * PT + shift[1])


def _union(rects):
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def _padding(r, page_rects):
    """Padding of an option box: up to 8 points, but never reaching
    halfway to the next option of the page, whatever its question."""
    px = py = 8.0
    for o in page_rects:
        if tuple(o) == tuple(r):
            continue
        if min(r[3], o[3]) - max(r[1], o[1]) > 0:      # side by side
            gap = max(o[0] - r[2], r[0] - o[2])
            px = min(px, max(1.0, gap / 2 - 0.5))
        if min(r[2], o[2]) - max(r[0], o[0]) > 0:      # one above the other
            gap = max(o[1] - r[3], r[1] - o[3])
            py = min(py, max(1.0, gap / 2 - 0.5))
    return r[0] - px, r[1] - py, r[2] + px, r[3] + py


@dataclass
class Option:
    field: int        # index of the field on the page
    value: str
    rect: tuple       # padded box (points)
    box: tuple        # padded box (pixels, shifted)
    centre: tuple     # (pixels, shifted)
    ink: int = 0


def _inside(stroke, rects):
    """Fraction of a stroke's box that lies inside the largest overlap
    with one of rects (all in pixels)."""
    x, y, w, h = stroke
    best = 0.0
    for r in rects:
        ox = min(x + w, r[2]) - max(x, r[0])
        oy = min(y + h, r[3]) - max(y, r[1])
        if ox > 0 and oy > 0:
            best = max(best, ox * oy / max(1, w * h))
    return best


def mark_ink(fields, shifts, added):
    """Give each stroke of added ink to an option; return the options.

    Strokes lying mostly in a handwriting area are writing, not marks.
    """
    writing = [_px(fl.rect, shifts[i]) for i, fl in enumerate(fields)
               if fl.rect]
    page_rects = [r for fl in fields for _, r in fl.options]
    options = []
    for i, fl in enumerate(fields):
        for k, (value, r) in enumerate(fl.options):
            if fl.cells:
                # The whole table cell (inside its lines): a mark can be
                # anywhere in it
                cell = fl.cells[k]
                rect = (cell[0] + 1, cell[1] + 1, cell[2] - 1, cell[3] - 1)
                c = box = _px(rect, shifts[i])
            else:
                rect = _padding(r, page_rects)
                box = _px(rect, shifts[i])
                c = _px(r, shifts[i])
            options.append(Option(i, value, rect, box,
                                  ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)))
    if not options:
        return options
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8)),
        connectivity=8)
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        if _inside((x, y, w, h), writing) > 0.6:
            continue
        touching = [o for o in options
                    if x < o.box[2] and x + w > o.box[0]
                    and y < o.box[3] and y + h > o.box[1]]
        if not touching:
            continue
        pixels = int(added[y:y + h, x:x + w][labels[y:y + h, x:x + w]
                                             == k].sum())
        if pixels < MIN_STROKE:
            continue
        cx, cy = centroids[k]
        nearest = min(touching, key=lambda o: (o.centre[0] - cx) ** 2
                      + (o.centre[1] - cy) ** 2)
        # A circle or a tick is centred on its option; handwriting that
        # merely touches an option box is not
        reach = 0.5 * np.hypot(nearest.box[2] - nearest.box[0],
                               nearest.box[3] - nearest.box[1]) + 8 * PT
        if np.hypot(nearest.centre[0] - cx, nearest.centre[1] - cy) > reach:
            continue
        nearest.ink += pixels
    return options


def choose(scores, multiple=False):
    """Return (values, status, detail) from [(value, ink)]."""
    ranked = sorted(scores, key=lambda s: -s[1])
    best = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0
    detail = 'ink ' + ', '.join('%s=%d' % (v, s) for v, s in scores)
    if best < BLANK:
        # Never silently blank: on the curated scans, more questions read
        # as blank had a mark outside the options than were really blank
        return [], 'check', detail + ' (no mark found)'
    if best < MARKED:
        return [ranked[0][0]], 'check', detail + ' (faint mark)'
    if second >= SECOND * best:
        marked = [v for v, s in ranked if s >= SECOND * best]
        return (marked if multiple else [ranked[0][0]], 'check',
                detail + ' (several marked)')
    if second >= MARKED:
        # A crossed-out answer has more ink than a circle: on the curated
        # scans, the answer was the second option in most such cases
        return ([ranked[0][0]] if not multiple else
                [v for v, s in ranked if s >= MARKED], 'check',
                detail + ' (another option also marked: one may be '
                'crossed out)')
    return [ranked[0][0]], 'ok', detail


def text_crop(warped, rect, shift, sideways, blank=None):
    """Return the image of a handwriting area, upright.

    With the blank page, the printed form (text, lines) is erased, so the
    handwriting model only sees what was written by hand. The eraser is
    thin, so handwriting that crosses a printed line is mostly kept.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in _px(
        (rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3), shift))
    x0, y0 = max(0, x0), max(0, y0)
    crop = warped[y0:y1, x0:x1].copy()
    if blank is not None:
        # The blank page is not shifted: take its matching window
        bx0, by0 = int(round(x0 - shift[0])), int(round(y0 - shift[1]))
        b = blank[max(0, by0):max(0, by0) + crop.shape[0],
                  max(0, bx0):max(0, bx0) + crop.shape[1]]
        printed = cv2.dilate((b < 200).astype(np.uint8),
                             np.ones((3, 3), np.uint8)) > 0
        crop[:printed.shape[0], :printed.shape[1]][printed] = 255
        _erase_strays(crop, warped, blank, shift, (x0, y0),
                      _px(rect, shift))
    if sideways:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def _erase_strays(crop, warped, blank, shift, origin, inner, margin=60):
    """Erase from crop the strokes centred outside the answer area inner
    (pixels), e.g. a circle drawn around the option of the next question
    that reaches into the area. Strokes are seen whole in a wider window."""
    h, w = warped.shape
    wx0, wy0 = max(0, int(inner[0]) - margin), max(0, int(inner[1]) - margin)
    wx1, wy1 = min(w, int(inner[2]) + margin), min(h, int(inner[3]) + margin)
    win = warped[wy0:wy1, wx0:wx1]
    otsu, _ = cv2.threshold(warped, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = win < np.clip(otsu, 140, 200)
    # Without the printed form, so that writing on a printed line is not
    # joined to the line (the blank page is not shifted)
    bx0, by0 = int(round(wx0 - shift[0])), int(round(wy0 - shift[1]))
    b = blank[max(0, by0):max(0, by0) + win.shape[0],
              max(0, bx0):max(0, bx0) + win.shape[1]]
    printed = cv2.dilate((b < 200).astype(np.uint8),
                         np.ones((3, 3), np.uint8)) > 0
    ink[:printed.shape[0], :printed.shape[1]] &= ~printed
    ink = ink.astype(np.uint8)
    n, labels, _, cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    stray = np.zeros(n, bool)
    for k in range(1, n):
        cx, cy = cent[k][0] + wx0, cent[k][1] + wy0
        stray[k] = not (inner[0] <= cx <= inner[2]
                        and inner[1] <= cy <= inner[3])
    mask = stray[labels]
    # The part of the window that the crop covers
    ox, oy = origin[0] - wx0, origin[1] - wy0
    sub = mask[max(0, oy):oy + crop.shape[0], max(0, ox):ox + crop.shape[1]]
    crop[:sub.shape[0], :sub.shape[1]][sub] = 255


def has_text(warped, blank, rect, shift):
    """True if the area holds handwriting: ink that is not printed, with
    a thin margin around the printing (writing often sits on the lines)."""
    crop = text_crop(warped, rect, shift, False, blank)
    otsu, _ = cv2.threshold(warped, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int((crop < np.clip(otsu, 140, 200)).sum()) >= MIN_TEXT_PIXELS


def clean_text(text, text_kind):
    """Normalise a model answer; return (answer, problem or '')."""
    t = ' '.join(text.strip().strip('"').split())
    if not t or t.upper().strip('.') == 'EMPTY':
        return MISSING, ''
    t = t.rstrip('.')
    if text_kind == 'number':
        t = t.replace(' ', '').replace(',', '.')
        if not re.fullmatch(r'\d+(\.\d+)?(-\d+(\.\d+)?)?', t):
            return t, 'not a number'
    elif text_kind == 'date':
        m = re.fullmatch(r'(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*'
                         r'(\d{2}|\d{4})', t)
        if not m:
            return t, 'not a DD/MM/YYYY date'
        d, mo, y = m.groups()
        if len(y) == 2:
            return '%02d/%02d/20%s' % (int(d), int(mo), y), 'two-digit year'
        t = '%02d/%02d/%s' % (int(d), int(mo), y)
        if not (1 <= int(d) <= 31 and 1 <= int(mo) <= 12):
            return t, 'impossible date'
        if int(d) <= 12 and int(mo) <= 12 and d != mo:
            return t, 'day and month could be swapped'
    elif text_kind == 'languages':
        # Only the languages: the fluency numbers go in the next row
        t = re.sub(r'\s*[:\-\u2013(]?\s*\d+\)?', '', t).strip(' ,')
        t = ', '.join(x.strip() for x in t.split(',') if x.strip())
    elif text_kind == 'percent':
        t = t.replace(' ', '')
        if re.fullmatch(r'\d+(\.\d+)?', t):
            t += '%'
        if not re.fullmatch(r'\d+(\.\d+)?%', t):
            return t, 'not a percentage'
    return t, ''


def read_page(fields, warped, blank, form, reader):
    """Read the fields of one aligned page.

    reader is the handwriting model (handwriting.Reader), or None to skip
    handwriting. All the handwriting of the page is read in one call.
    """
    _, added = ink_masks(warped, blank)
    sideways = form in SIDEWAYS
    fields = [fl for fl in fields if fl.kind != 'derived']
    shifts, areas = [], []
    for fl in fields:
        rects = [r for _, r in fl.options] + ([fl.rect] if fl.rect else [])
        area = _union(rects)
        areas.append(area)
        shifts.append(align.local_shift(warped, blank, _px(area)))
    options = mark_ink(fields, shifts, added)

    # 1. Marks, and the handwriting areas that need reading
    answers, wanted = [], []        # wanted: (answer index, field)
    for i, fl in enumerate(fields):
        area = areas[i]
        a = Answer(fl.row, crop=(area[0] - 10, area[1] - 10,
                                 area[2] + 10, area[3] + 10))
        if fl.kind in ('circle', 'tick', 'yes_text'):
            mine = [o for o in options if o.field == i]
            multiple = fl.row.startswith('What level did you attain')
            values, a.status, a.detail = choose(
                [(o.value, o.ink) for o in mine], multiple)
            a.marks = [(o.rect, o.value in values) for o in mine]
            if fl.kind == 'tick' and len(fl.options) == 1:
                # Single box ("do not contact"): ticked = Yes, else No
                a.answer = 'Yes' if values else 'No'
                if not values:
                    a.status = 'ok'     # an empty box is an answer
            elif multiple and len(values) > 1:
                ranked = [v for v in SCHOOL_RANK if v in values]
                a.answer = ranked[-1] if ranked else values[0]
                a.detail += '; highest level kept'
            elif values:
                a.answer = values[0]
            # "Other (please specify)" and health "Yes, describe"
            if fl.rect and a.answer in ('Other', 'Yes') \
                    and has_text(warped, blank, fl.rect, shifts[i]):
                wanted.append((i, fl))
        elif fl.kind == 'text':
            if has_text(warped, blank, fl.rect, shifts[i]):
                wanted.append((i, fl))
        answers.append(a)

    # 2. Handwriting, one model call for the page
    crops = [text_crop(warped, fl.rect, shifts[i], sideways, blank)
             for i, fl in wanted]
    texts = (reader.read_many(list(zip(crops, [fl for _, fl in wanted])))
             if reader and wanted else [None] * len(wanted))
    for (i, fl), text in zip(wanted, texts):
        a = answers[i]
        if text is None:
            a.status = 'check'
            a.detail = (a.detail + '; ' if a.detail else '') + \
                'handwriting not read (model off)'
            continue
        value, problem = clean_text(text, fl.text_kind)
        if value == MISSING:
            # Ink was found but the model read nothing: never silently n/a
            a.status = 'check'
            a.detail = (a.detail + '; ' if a.detail else '') + \
                'handwriting found but not read'
            continue
        if fl.kind == 'text':
            a.answer = value
            a.detail = 'handwriting' + (' (%s)' % problem if problem else '')
        else:
            a.answer = '%s: %s' % (a.answer, value)
            a.detail += '; handwriting'
        a.status = 'check'
    return answers
