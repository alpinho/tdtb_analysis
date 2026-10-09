"""
Read handwriting with a vision model running locally in Ollama.

The model runs on this computer (Ollama listens on 127.0.0.1 only); no
image or text leaves the machine. Only small images of single answers
are sent to it, which keeps each call short.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import re
import time

import cv2
import numpy as np

MODEL = 'qwen2.5vl:3b'
HOST = 'http://127.0.0.1:11434'   # local only; never a remote address
# Short answers need little context; a small context lets the model fit
# on a 4 GB graphics card, which is several times faster
NUM_CTX = 4096
# Each call costs ~20 s whatever the image size, so the answers of a page
# are read together: stacked in one image, numbered, at most MAX_ROWS
# rows and MAX_HEIGHT pixels (before reduction by SHEET_SCALE) per call
MAX_ROWS = 12
MAX_HEIGHT = 1400
SHEET_SCALE = 0.6

# A short prompt: long prompts (e.g. with the question text) are slower
# and lead the model to answer "None" for short words such as "No". The
# model is only asked when the area holds handwriting (see read.has_text),
# so the prompt does not offer it a "blank" answer: given that choice, it
# sometimes calls clear writing blank.
PROMPT = ('This image shows a handwritten answer on a form. Transcribe it '
          'exactly. %s Reply with the transcription only.')
HINTS = {
    'text': '',
    'list': 'If there are several items, separate them with a comma and a '
            'space.',
    'number': 'It is a number or a range such as 2-3; write it in digits.',
    'date': 'It is a date (day/month/year); write it as DD/MM/YYYY.',
    'percent': 'It is a percentage.',
    'languages': 'Separate the items with a comma and a space.',
    'fluency': 'Write each language with its number, as "Language: '
               'number", separated by a comma and a space.',
}


BATCH_PROMPT = ('This image shows numbered handwritten answers from a form, '
                'one per row. Transcribe each exactly. %sReply with one line '
                'per row, as "number: transcription".')
BATCH_HINTS = {'date': 'a date: write it as DD/MM/YYYY',
               'number': 'a number or a range such as 2-3, in digits',
               'percent': 'a percentage',
               'languages': 'a list of languages',
               'fluency': 'languages with numbers: write each as '
                          '"Language: number", separated by commas'}


def _sheet(images):
    """Stack images in one numbered image."""
    width = max(img.shape[1] for img in images) + 70
    rows = []
    for k, img in enumerate(images, 1):
        row = np.full((img.shape[0] + 16, width), 255, np.uint8)
        row[8:8 + img.shape[0], 70:70 + img.shape[1]] = img
        cv2.putText(row, '%d)' % k, (5, row.shape[0] // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        cv2.line(row, (0, row.shape[0] - 1), (width, row.shape[0] - 1),
                 120, 1)
        rows.append(row)
    sheet = np.vstack(rows)
    return cv2.resize(sheet, None, fx=SHEET_SCALE, fy=SHEET_SCALE,
                      interpolation=cv2.INTER_AREA)


class Reader:
    """Callable that reads the handwriting of one answer."""

    def __init__(self, model=MODEL):
        import ollama   # only needed when the model is used
        self.client = ollama.Client(host=HOST)
        self.model = model
        self.calls, self.seconds = 0, 0.0

    def __call__(self, image, fl):
        ok, png = cv2.imencode('.png', image)
        hint = fl.prompt or HINTS[fl.text_kind]
        prompt = ' '.join((PROMPT % hint).split())
        t0 = time.time()
        r = self.client.chat(
            model=self.model, keep_alive='10m',
            options={'temperature': 0, 'num_predict': 60,
                     'num_ctx': NUM_CTX},
            messages=[{'role': 'user', 'content': prompt,
                       'images': [png.tobytes()]}])
        self.calls += 1
        self.seconds += time.time() - t0
        return r['message']['content']

    def read_many(self, items):
        """Read [(image, field)]; return the texts, in the same order."""
        out, chunk, height = [None] * len(items), [], 0
        chunks = []
        for k, (img, fl) in enumerate(items):
            if chunk and (len(chunk) == MAX_ROWS
                          or height + img.shape[0] > MAX_HEIGHT):
                chunks.append(chunk)
                chunk, height = [], 0
            chunk.append(k)
            height += img.shape[0] + 16
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            if len(chunk) == 1:
                out[chunk[0]] = self(*items[chunk[0]])
                continue
            hints = '; '.join(
                'row %d is %s' % (n, BATCH_HINTS[items[k][1].text_kind])
                for n, k in enumerate(chunk, 1)
                if items[k][1].text_kind in BATCH_HINTS)
            prompt = BATCH_PROMPT % (hints[0].upper() + hints[1:] + '. '
                                     if hints else '')
            ok, png = cv2.imencode('.png', _sheet([items[k][0]
                                                   for k in chunk]))
            t0 = time.time()
            r = self.client.chat(
                model=self.model, keep_alive='10m',
                options={'temperature': 0, 'num_predict': 40 * len(chunk),
                         'num_ctx': NUM_CTX},
                messages=[{'role': 'user', 'content': prompt,
                           'images': [png.tobytes()]}])
            self.calls += 1
            self.seconds += time.time() - t0
            got = {}
            for line in r['message']['content'].splitlines():
                m = re.match(r'\s*(\d+)\s*[:.)]\s*(.*)$', line)
                if m:
                    got.setdefault(int(m.group(1)), m.group(2).strip())
            for n, k in enumerate(chunk, 1):
                # A row the model skipped is read on its own
                out[k] = got[n] if n in got else self(*items[k])
        return out

    def ask(self, image, prompt):
        """Free question about an image (e.g. a session label)."""
        ok, png = cv2.imencode('.png', image)
        r = self.client.chat(
            model=self.model, keep_alive='10m',
            options={'temperature': 0, 'num_predict': 20,
                     'num_ctx': NUM_CTX},
            messages=[{'role': 'user', 'content': prompt,
                       'images': [png.tobytes()]}])
        return r['message']['content']
