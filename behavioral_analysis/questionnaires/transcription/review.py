"""
Review PDF: each answer next to the part of the scan it was read from.

Rows in orange need a check against the paper form (unclear marks and
all handwriting, which the model may misread); rows in green were read
from clear marks or are empty. On each cut-out, the chosen option is
framed in green and the other options in grey.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import textwrap

import cv2
import pymupdf

from . import read

PAGE_W, PAGE_H = 595, 842
MARGIN = 30
COL_LABEL, COL_ANSWER, COL_IMAGE = 30, 240, 375
IMG_W, IMG_H = 190, 80
COLORS = {'ok': (0.1, 0.55, 0.1), 'check': (0.9, 0.45, 0.0)}


def _crop(warped, a, sideways):
    """Cut-out of a page around an answer, with its options framed."""
    pt = read.PT
    img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    for rect, chosen in a.marks:
        x0, y0, x1, y1 = (int(round(v * pt)) for v in rect)
        cv2.rectangle(img, (x0, y0), (x1, y1),
                      (40, 160, 40) if chosen else (170, 170, 170),
                      3 if chosen else 1)
    x0, y0, x1, y1 = (int(round(v * pt)) for v in a.crop)
    h, w = warped.shape
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if sideways:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


class Review:
    def __init__(self, title):
        self.doc = pymupdf.open()
        self.title = title
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page.insert_text((MARGIN, 22), self.title, fontsize=8,
                              color=(0.4, 0.4, 0.4))
        self.y = 40

    def heading(self, text, sub=''):
        if self.y > PAGE_H - 120:
            self._new_page()
        self.y += 8
        self.page.insert_text((MARGIN, self.y + 12), text, fontsize=13,
                              fontname='hebo')
        self.y += 18
        for line in textwrap.wrap(sub, 110):
            self.page.insert_text((MARGIN, self.y + 9), line, fontsize=8)
            self.y += 11
        self.y += 4

    def row(self, label, answer, status, detail='', image=None):
        label_lines = textwrap.wrap(label, 48) or ['']
        answer_lines = textwrap.wrap(answer, 26) or ['']
        detail_lines = textwrap.wrap(detail, 40)[:3]
        text_h = max(len(label_lines) * 10,
                     len(answer_lines) * 12 + len(detail_lines) * 8 + 10)
        img_h = 0
        if image is not None and image.size:
            h, w = image.shape[:2]
            scale = min(IMG_W / w, IMG_H / h)
            img_h = h * scale
        height = max(text_h, img_h) + 8
        if self.y + height > PAGE_H - MARGIN:
            self._new_page()
        color = COLORS.get(status, (0, 0, 0))
        self.page.draw_rect(pymupdf.Rect(MARGIN - 6, self.y,
                                         MARGIN - 3, self.y + height - 4),
                            color=color, fill=color)
        y = self.y + 9
        for line in label_lines:
            self.page.insert_text((COL_LABEL, y), line, fontsize=8)
            y += 10
        y = self.y + 10
        for line in answer_lines:
            self.page.insert_text((COL_ANSWER, y), line, fontsize=10,
                                  fontname='hebo')
            y += 12
        self.page.insert_text((COL_ANSWER, y), status.upper(), fontsize=7,
                              color=color)
        y += 9
        for line in detail_lines:
            self.page.insert_text((COL_ANSWER, y), line, fontsize=6,
                                  color=(0.4, 0.4, 0.4))
            y += 8
        if img_h:
            ok, png = cv2.imencode('.png', image)
            h, w = image.shape[:2]
            scale = min(IMG_W / w, IMG_H / h)
            r = pymupdf.Rect(COL_IMAGE, self.y + 2, COL_IMAGE + w * scale,
                             self.y + 2 + h * scale)
            self.page.insert_image(r, stream=png.tobytes())
        self.page.draw_line((MARGIN, self.y + height - 2),
                            (PAGE_W - MARGIN, self.y + height - 2),
                            color=(0.85, 0.85, 0.85), width=0.5)
        self.y += height

    def image(self, caption, image, max_h=250):
        h, w = image.shape[:2]
        scale = min((PAGE_W - 2 * MARGIN) / w, max_h / h)
        if self.y + h * scale + 20 > PAGE_H - MARGIN:
            self._new_page()
        self.page.insert_text((MARGIN, self.y + 9), caption, fontsize=8)
        ok, png = cv2.imencode('.png', image)
        r = pymupdf.Rect(MARGIN, self.y + 12, MARGIN + w * scale,
                         self.y + 12 + h * scale)
        self.page.insert_image(r, stream=png.tobytes())
        self.y += h * scale + 20

    def save(self, path):
        self.doc.save(path, garbage=3, deflate=True)


def thumbnail(img, width=500):
    scale = width / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def crop_for(warped, answer, form):
    if answer.crop is None:
        return None
    return _crop(warped, answer, form in read.SIDEWAYS)
