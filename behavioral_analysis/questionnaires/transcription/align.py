"""
Identify the form page of each scanned page and align it to the blank
form.

Pages are matched to every page of the blank forms with ORB features,
in the four possible orientations; the best match gives the page and a
homography, refined with ECC, that maps the scan onto the blank page.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf

from . import layout

ID_DPI = 100      # resolution used to identify pages
DPI = 200         # resolution used to read the answers
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
N_FEATURES = 2000
MIN_INLIERS = 40  # fewer matches than this: the page is not identified
MIN_RATIO = 2.0   # best match / best other page; below: identity uncertain
COVERAGE_RATIO = 1.3   # look-alike pages: printed text found on the
                       # best page / on the next page, at least


def render(page, dpi):
    """Return a PDF page as a grayscale image."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(
        pix.h, pix.w).copy()


def rotate(img, rotation):
    code = ROTATIONS[rotation]
    return img if code is None else cv2.rotate(img, code)


@dataclass
class Match:
    form: str = None       # demographic, gold_msi or postexp
    page: int = None       # 0-based page of the form
    rotation: int = 0
    inliers: int = 0
    ratio: float = 0.0     # inliers of the best page / of the next page
    H: np.ndarray = None   # scan (ID_DPI, rotated) -> blank (ID_DPI)
    coverage_ratio: float = None   # set for look-alike pages

    @property
    def ok(self):
        if self.inliers < MIN_INLIERS:
            return False
        if self.coverage_ratio is not None:
            return self.coverage_ratio >= COVERAGE_RATIO
        return self.ratio >= MIN_RATIO


class Templates:
    """The pages of the blank forms, ready for matching and reading."""

    def __init__(self):
        self.orb = cv2.ORB_create(N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.pages = {}    # (form, page) -> dict
        for form, fname in layout.TEMPLATE_FILES.items():
            doc = pymupdf.open(os.path.join(layout.TEMPLATE_DIR, fname))
            for i, page in enumerate(doc):
                small = render(page, ID_DPI)
                kp, des = self.orb.detectAndCompute(small, None)
                full = render(page, DPI)
                self.pages[(form, i)] = {'kp': kp, 'des': des,
                                         'img': full}

    def _homography(self, kp, des, key):
        t = self.pages[key]
        pairs = self.bf.knnMatch(des, t['des'], k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return 0, None
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([t['kp'][m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            return 0, None
        return int(mask.sum()), H

    def identify(self, page):
        """Return the Match of a scanned page (a pymupdf page)."""
        small = render(page, ID_DPI)
        results = []
        for rotation in ROTATIONS:
            kp, des = self.orb.detectAndCompute(rotate(small, rotation),
                                                None)
            if des is None:
                continue
            for key in self.pages:
                n, H = self._homography(kp, des, key)
                results.append((n, key, rotation, H))
        if not results:
            return Match()
        results.sort(key=lambda r: -r[0])
        n, key, rotation, H = results[0]
        other = next((r[0] for r in results[1:] if r[1] != key), 0)
        match = Match(key[0], key[1], rotation, n, n / max(1, other), H)
        if match.inliers >= MIN_INLIERS and match.ratio < MIN_RATIO:
            match = self._by_coverage(small, results)
        return match

    def _by_coverage(self, small, results):
        """Tell apart pages with the same layout (Gold-MSI pages 2-4) by
        how much of their printed text is found on the scan."""
        best = {}
        for n, key, rotation, H in results:
            if H is not None and key not in best:
                best[key] = (n, rotation, H)
        top = results[0][0]
        scored = []
        for key, (n, rotation, H) in best.items():
            if n < top / MIN_RATIO:
                continue
            blank = cv2.resize(self.pages[key]['img'], None,
                               fx=ID_DPI / DPI, fy=ID_DPI / DPI,
                               interpolation=cv2.INTER_AREA)
            warped = cv2.warpPerspective(rotate(small, rotation), H,
                                         blank.shape[::-1], borderValue=255)
            printed = blank < 128
            found = cv2.dilate((warped < 128).astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
            cov = float((printed & found).sum()) / max(1, printed.sum())
            scored.append((cov, n, key, rotation, H))
        scored.sort(key=lambda s: -s[0])
        cov, n, key, rotation, H = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        ratio = n / max(1, results[0][0] if key != results[0][1]
                        else next((r[0] for r in results[1:]
                                   if r[1] != key), 0))
        return Match(key[0], key[1], rotation, n, ratio, H,
                     coverage_ratio=cov / max(1e-6, second))

    def align(self, page, match):
        """Return the scanned page warped onto the blank page, at DPI."""
        img = rotate(render(page, DPI), match.rotation)
        s = DPI / ID_DPI
        S = np.diag([s, s, 1.0])
        H = S @ match.H @ np.linalg.inv(S)
        blank = self.pages[(match.form, match.page)]['img']
        h, w = blank.shape
        H = _refine(img, blank, H)
        return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderValue=255)


def _refine(img, blank, H, scale=0.25):
    """Refine the homography with ECC on blurred, reduced images."""
    small_blank = cv2.GaussianBlur(
        cv2.resize(blank, None, fx=scale, fy=scale), (5, 5), 0)
    S = np.diag([scale, scale, 1.0])
    H0 = (S @ H @ np.linalg.inv(S)).astype(np.float32)
    h, w = small_blank.shape
    small_img = cv2.GaussianBlur(
        cv2.resize(img, None, fx=scale, fy=scale), (5, 5), 0)
    warped = cv2.warpPerspective(small_img, H0, (w, h), borderValue=255)
    try:
        _, D = cv2.findTransformECC(
            small_blank.astype(np.float32), warped.astype(np.float32),
            np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
            None, 5)
    except cv2.error:
        return H
    # D maps blank coordinates to warped-image coordinates
    H1 = np.linalg.inv(D) @ H0
    H1 = np.linalg.inv(S) @ H1 @ S
    # Keep the refinement only if it is a small correction
    corners = np.float32([[0, 0], [blank.shape[1], 0],
                          [0, blank.shape[0]],
                          [blank.shape[1], blank.shape[0]]]).reshape(-1, 1, 2)
    shift = np.abs(cv2.perspectiveTransform(corners, H1)
                   - cv2.perspectiveTransform(corners, H)).max()
    return H1 if shift < 0.02 * DPI * 8 else H


def local_shift(warped, blank, rect_px, pad=40, max_shift=20):
    """Translation (dx, dy) that best lines up the printed text around a
    field; (0, 0) if it cannot be estimated reliably."""
    x0, y0, x1, y1 = (int(v) for v in rect_px)
    h, w = blank.shape
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(w, x1 + pad), min(h, y1 + pad)
    a = cv2.GaussianBlur(blank[Y0:Y1, X0:X1], (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(warped[Y0:Y1, X0:X1], (5, 5), 0).astype(np.float32)
    if a.std() < 5 or b.std() < 5:
        return 0, 0
    try:
        _, M = cv2.findTransformECC(
            a, b, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
            None, 5)
    except cv2.error:
        return 0, 0
    dx, dy = float(M[0, 2]), float(M[1, 2])
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return 0, 0
    return dx, dy
