"""
Where each answer is on the blank forms.

The positions come from the text of the blank forms themselves (the PDFs
in music-sdtb_protocols, give_to_the_participant_2026): every option
("1" ... "6", "Yes", "Male", a tick box) is located by its printed text
near a position given here, so the boxes follow the PDFs exactly. All
coordinates are PDF points of the blank page (1/72 inch).

Fields that would identify the participant (Participant ID, Gold-MSI
name, email and anonymous ID) are not defined, so they are never read.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

import os
from dataclasses import dataclass, field

import pymupdf

from . import sheets

TEMPLATE_DIR = os.path.expanduser(
    '~/mygit/music_sdtb/music-sdtb_protocols/experiments/behavioral/'
    'ethics/give_to_the_participant_2026')
TEMPLATE_FILES = {'demographic': 'demographic_questionnaire_Grahn_2019.pdf',
                  'gold_msi': 'GOLD-MSI.pdf',
                  'postexp': 'post_experiment_questionnaire.pdf'}


@dataclass
class Field:
    """One answer of a form.

    kind is one of:
      circle   -- one option circled (options: [(value, rect)])
      tick     -- one tick box ticked (options: [(value, rect)])
      yes_text -- Yes/No tick boxes plus a text line read when Yes is
                  ticked; the answer is "No", "Yes" or "Yes: <text>"
      text     -- handwriting in rect; text_kind tells its format
      derived  -- not read from the page (subject number, copies)
    """
    form: str
    page: int                 # 0-based page of the form
    row: str                  # row of the sheet
    kind: str
    options: list = field(default_factory=list)
    rect: tuple = None        # x0, y0, x1, y1 (points)
    text_kind: str = 'text'   # text, number, date, percent, list
    prompt: str = ''          # what the handwriting model is asked
    note: str = ''
    cells: list = None        # table cell of each option, if any (points)


def _words(page):
    return [(x0, y0, x1, y1, w) for x0, y0, x1, y1, w, *_ in
            page.get_text('words')]


def _find(words, text, y=None, x=None, tol=5):
    """Return the box of the word `text` near (x, y)."""
    hits = [w for w in words if w[4] == text
            and (y is None or abs(w[1] - y) < tol)
            and (x is None or abs(w[0] - x) < tol)]
    if len(hits) != 1:
        raise ValueError('Found %d words "%s" near x=%s, y=%s'
                         % (len(hits), text, x, y))
    return tuple(hits[0][:4])


def _find_at(words, x, y, tol=3):
    """Return the box of the word starting at (x, y), whatever its text."""
    hits = [w for w in words if abs(w[0] - x) < tol and abs(w[1] - y) < tol]
    if len(hits) != 1:
        raise ValueError('Found %d words at x=%s, y=%s' % (len(hits), x, y))
    return tuple(hits[0][:4])


def _search(page, text, clip, nth=0):
    """Return the box of the nth occurrence of `text` inside clip."""
    hits = page.search_for(text, clip=pymupdf.Rect(clip))
    return tuple(hits[nth])


def _union(*rects):
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def _demographic(doc):
    rows = sheets.DEMOGRAPHIC_ROWS
    f = []
    p1, p2, p3 = (_words(doc[i]) for i in range(3))

    def circle(page, row, labels, y, words, values=None, x=None):
        boxes = [_find(words, lab, y, x[i] if x else None)
                 for i, lab in enumerate(labels)]
        f.append(Field('demographic', page, row, 'circle',
                       list(zip(values or labels, boxes))))

    def scale(page, row, y, words, n0, n1):
        circle(page, row, [str(i) for i in range(n0, n1 + 1)], y, words)

    def text(page, row, rect, text_kind='text', prompt=''):
        f.append(Field('demographic', page, row, 'text', rect=rect,
                       text_kind=text_kind, prompt=prompt))

    f.append(Field('demographic', 0, rows[0], 'derived',
                   note='subject number'))
    text(0, rows[1], (330, 100, 412, 128), 'date')
    circle(0, rows[2], ['Male', 'Female'], 240, p1)
    f.append(Field('demographic', 0, rows[3], 'derived',
                   note='not on the form; fill in by hand if known'))
    text(0, rows[4], (82, 254, 205, 282), 'number')
    circle(0, rows[5], ['Right', 'Left'], 293, p1)
    # School level: tick boxes, the value is the printed label
    school = [('Elementary School', 57, 361), ('Less than Grade 12', 57, 382),
              ('High school diploma', 57, 403),
              ('Some university undergraduate schooling', 57, 423),
              ('College Degree (2 years)', 305, 361),
              ("Bachelor's degree", 305, 382),
              ('Postgraduate degree', 305, 403), ('Other', 57, 444)]
    f.append(Field('demographic', 0, rows[6], 'tick',
                   [(v, _find(p1, '□', y, x)) for v, x, y in school],
                   rect=(176, 434, 412, 458), note='Other: text in rect'))
    text(0, rows[7], (205, 494, 400, 520))
    langs = (55, 578, 430, 624)
    text(0, rows[8], langs, 'languages',
         'It lists languages, each maybe followed by a number. Write only '
         'the languages, separated by a comma and a space.')
    text(0, rows[9], langs, 'fluency',
         'It lists languages, each followed by a number from 1 to 6. '
         'Write them as "Language: number", separated by a comma and a '
         'space.')
    yes, no = _find(p1, 'Yes', 634), _find(p1, 'No', 634)
    f.append(Field('demographic', 0, rows[10], 'circle',
                   [('Yes', _union(_find_at(p1, 281, 634.6), yes)),
                    ('No', _union(_find_at(p1, 335, 634.6), no))]))
    text(0, rows[11], (278, 650, 480, 692))
    text(0, rows[12], (278, 732, 480, 775), 'percent')
    # Page 2
    scale(1, rows[13], 118, p2, 1, 6)
    circle(1, rows[14], ['Yes', 'No'], 159, p2)
    text(1, rows[15], (199, 168, 540, 194), 'list')
    text(1, rows[16], (276, 189, 372, 215), 'number')
    text(1, rows[17], (55, 230, 540, 258))
    circle(1, rows[18], ['Yes', 'No'], 283.5, p2)
    text(1, rows[19], (313, 292, 540, 318), 'number')
    scale(1, rows[20], 366, p2, 1, 6)
    circle(1, rows[21], ['No', 'Right', 'Left', 'Both'], 427.6, p2)
    circle(1, rows[22], ['Sometimes', 'Always', 'Never'], 469, p2)
    circle(1, rows[23], ['Both', 'Left', 'Right'], 489.8, p2)
    scale(1, rows[24], 545, p2, 1, 6)
    scale(1, rows[25], 628, p2, 0, 10)
    scale(1, rows[26], 725, p2, 0, 10)
    # Page 3
    scale(2, rows[27], 138.5, p3, 0, 10)
    scale(2, rows[28], 235.3, p3, 0, 10)
    scale(2, rows[29], 332.2, p3, 0, 10)
    # Health: Yes/No tick boxes and a line for the description
    health = [(rows[30], 384.6, 410.5, (364, 380, 540, 401)),
              (rows[31], 443.6, 469.5, (397, 439, 540, 460)),
              (rows[32], 502.5, 528.4, (397, 498, 540, 519)),
              (rows[33], 564.4, 590.3, (397, 560, 540, 581)),
              (rows[34], 626.3, 652.1, (372, 622, 540, 643)),
              (rows[35], 685.3, 711.1, (397, 681, 545, 702))]
    for row, y_yes, y_no, rect in health:
        # The option covers the box and its label: some participants
        # circle the word instead of ticking the box
        f.append(Field('demographic', 2, row, 'yes_text',
                       [('Yes', _union(_find_at(p3, 281, y_yes),
                                       _find(p3, 'Yes,', y_yes))),
                        ('No', _union(_find_at(p3, 281, y_no),
                                      _find(p3, 'No', y_no)))], rect=rect))
    return f


def _postexp(doc):
    rows = sheets.POSTEXP_ROWS
    p1, p2 = _words(doc[0]), _words(doc[1])
    f = [Field('postexp', 0, rows[0], 'derived', note='subject number'),
         Field('postexp', 0, rows[1], 'text', rect=(339, 100, 420, 128),
               text_kind='date')]
    for i, (page, words, y) in enumerate(
            [(0, p1, 314), (0, p1, 412), (0, p1, 511), (0, p1, 609),
             (0, p1, 708), (1, p2, 131), (1, p2, 229)]):
        f.append(Field('postexp', page, rows[2 + i], 'circle',
                       [(str(v), _find(words, str(v), y))
                        for v in range(1, 7)]))
    f.append(Field('postexp', 1, rows[9], 'tick',
                   [('Yes', _find(p2, '□', 335, 242)),
                    ('No', _find(p2, '□', 335, 313))]))
    f.append(Field('postexp', 1, rows[10], 'text', rect=(62, 380, 536, 456)))
    f.append(Field('postexp', 1, rows[11], 'text', rect=(62, 504, 536, 580)))
    return f


def _gold_msi(doc):
    """Gold-MSI fields. The pages are printed sideways: each line of text
    is a column of the page (x), read from top to bottom (y)."""
    rows = sheets.GOLD_MSI_ROWS
    f = [Field('gold_msi', 0, rows[0], 'derived', note='subject number'),
         Field('gold_msi', 0, rows[1], 'derived',
               note='no date on the form; copied from the demographic form')]
    # Items 1-31: digits 1-7 printed in the column of each item
    first_item = {1: 1, 2: 11, 3: 21}
    for page in (1, 2, 3):
        words = _words(doc[page])
        items = sorted(int(w[4][:-1]) for w in words
                       if w[4][:-1].isdigit() and w[4].endswith('.'))
        for item in items:
            col = next(w for w in words if w[4] == '%d.' % item)
            opts = []
            for v in range(1, 8):
                hits = [w for w in words if w[4] == str(v)
                        and abs(w[0] - col[0]) < 3]
                if len(hits) != 1:
                    raise ValueError('Gold-MSI item %d: option %d not found'
                                     % (item, v))
                opts.append((str(v), tuple(hits[0][:4])))
            f.append(Field('gold_msi', page, 'question_%02d' % item,
                           'circle', opts,
                           cells=[_cell(doc[page], r) for _, r in opts]))
        if items[0] != first_item[page]:
            raise ValueError('Unexpected items on Gold-MSI page %d'
                             % (page + 1))
    # Items 32-38: options printed inside the sentence
    p5 = doc[4]
    options = {32: ['0', '1', '2', '3', '4-5', '6-9', '10 or more'],
               33: ['0', '0.5', '1', '1.5', '2', '3-4', '5 or more'],
               34: ['0', '1', '2', '3', '4-6', '7-10', '11 or more'],
               35: ['0', '0.5', '1', '2', '3', '4-6', '7 or more'],
               36: ['0', '0.5', '1', '2', '3-5', '6-9', '10 or more'],
               37: ['0', '1', '2', '3', '4', '5', '6 or more'],
               38: ['0-15 min', '15-30 min', '30-60 min', '60-90 min',
                    '2 hrs', '2-3 hrs', '4 hrs or more']}
    x_item = {32: 474, 33: 451, 34: 428, 35: 406, 36: 383, 37: 347, 38: 325}
    for item, opts in options.items():
        f.append(Field('gold_msi', 4, rows[item + 1], 'circle',
                       _inline_options(p5, x_item[item], opts)))
    f.append(Field('gold_msi', 4, rows[40], 'text', rect=(296, 338, 322, 600)))
    # Page 6: occupational status and genre (tick one box)
    w6 = _words(doc[5])
    occupation = ['Still at School', 'At University',
                  'In Full-time employment', 'In Part-time employment',
                  'Self-employed', 'Homemaker/full time parent',
                  'Unemployed', 'Retired']
    f.append(Field('gold_msi', 5, rows[41], 'tick',
                   _tick_lines(w6, occupation)))
    f.append(Field('gold_msi', 5, rows[42], 'tick',
                   _tick_lines(w6, ['Rock/Pop', 'Jazz', 'Classical Music'])))
    # Page 7: education (tick one box); the value drops the "(e.g. ...)"
    w7 = _words(doc[6])
    attained = [('Did not complete any school qualification', None),
                ('Completed first school qualification at about 16 years',
                 '(e.g.'),
                ('Completed Second qualification', '(e.g'),
                ('Undergraduate degree or professional qualification', None),
                ('Postgraduate degree', None),
                ('I am still in education', None)]
    expected = [('First school qualification', '(e.g.'),
                ('Post-16 vocational course', None),
                ('Second school qualification', '(e.g.'),
                ('Undergraduate degree or professional qualification', None),
                ('Postgraduate degree', None), ('Not applicable', None)]
    f.append(Field('gold_msi', 6, rows[43], 'tick',
                   _tick_lines(w7, attained, x_range=(370, 500))))
    f.append(Field('gold_msi', 6, rows[44], 'tick',
                   _tick_lines(w7, expected, x_range=(210, 350))))
    # Page 8: age, gender, nationality, countries, contact
    p8 = doc[7]
    f.append(Field('gold_msi', 7, rows[45], 'text', rect=(466, 118, 485, 160),
                   text_kind='number'))
    gender = (455, 72, 468, 200)
    female = _search(p8, 'Female', gender)
    male = [r for r in p8.search_for('Male', clip=pymupdf.Rect(gender))
            if r.y0 > female[3] - 1][0]
    f.append(Field('gold_msi', 7, rows[46], 'circle',
                   [('Female', female), ('Male', tuple(male))]))
    f.append(Field('gold_msi', 7, rows[47], 'text', rect=(439, 128, 458, 420)))
    f.append(Field('gold_msi', 7, rows[48], 'text', rect=(425, 442, 445, 780)))
    # Stops short of the email line (x < 414), which must never be read
    f.append(Field('gold_msi', 7, rows[49], 'text', rect=(415, 211, 431, 520)))
    w8 = _words(doc[7])
    box = next(w for w in w8 if abs(w[0] - 361) < 3 and w[4] == '2')
    f.append(Field('gold_msi', 7, rows[50], 'tick', [('Yes', tuple(box[:4]))],
                   note='ticked = Yes, empty = No'))
    return f


def _cell(page, r):
    """The table cell around the option rect r, from the lines drawn on
    the page. Some participants tick the middle of the cell rather than
    circle the digit printed in its corner."""
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    hor, ver = [], []
    for d in page.get_drawings():
        for it in d['items']:
            if it[0] != 'l':
                continue
            p, q = it[1], it[2]
            if abs(p.y - q.y) < 1 and min(p.x, q.x) <= cx <= max(p.x, q.x):
                hor.append(p.y)
            elif abs(p.x - q.x) < 1 and min(p.y, q.y) <= cy <= max(p.y, q.y):
                ver.append(p.x)
    cell = (max(x for x in ver if x <= r[0]), max(y for y in hor if y <= r[1]),
            min(x for x in ver if x >= r[2]), min(y for y in hor if y >= r[3]))
    return cell


def _inline_options(page, x, opts):
    """Boxes of the options of a Gold-MSI sentence (items 32-38)."""
    line = (x - 2, 60, x + 15, 790)
    text = page.get_textbox(pymupdf.Rect(line))
    joined = ' / '.join(opts)
    if joined not in ' '.join(text.split()):
        raise ValueError('Options not found: %s' % joined)
    # Search the whole run of options, then each option inside the run
    run = page.search_for(joined, clip=pymupdf.Rect(line))[0]
    out, start = [], run.y0
    for opt in opts:
        hits = sorted((r for r in page.search_for(opt, clip=run)
                       if r.y0 >= start - 0.5), key=lambda r: r.y0)
        r = hits[0]
        out.append((opt, (r.x0, r.y0, r.x1, r.y1)))
        start = r.y1
    return out


def _tick_lines(words, labels, x_range=(0, 600)):
    """Tick boxes of Gold-MSI lists: the box is the "2" glyph that starts
    the line of each label (the font draws a square for it)."""
    out = []
    for label in labels:
        value, stop = label if isinstance(label, tuple) else (label, None)
        first = value.split()[0]
        hits = []
        for w in words:
            if w[4] == '2' and x_range[0] <= w[0] <= x_range[1]:
                line = sorted((v for v in words if abs(v[0] - w[0]) < 2),
                              key=lambda v: v[1])
                text = ' '.join(v[4] for v in line[1:])
                text = text.replace('ﬁ', 'fi')
                if stop and stop in text:
                    text = text[:text.index(stop)].strip()
                if text == value or (text.startswith(first)
                                     and text == value):
                    hits.append(w)
        if len(hits) != 1:
            raise ValueError('Tick box "%s": %d found' % (value, len(hits)))
        out.append((value, tuple(hits[0][:4])))
    return out


def load_fields():
    """Return {form: [Field, ...]} for the three forms."""
    docs = {k: pymupdf.open(os.path.join(TEMPLATE_DIR, v))
            for k, v in TEMPLATE_FILES.items()}
    fields = {'demographic': _demographic(docs['demographic']),
              'gold_msi': _gold_msi(docs['gold_msi']),
              'postexp': _postexp(docs['postexp'])}
    for form, rows in (('demographic', sheets.DEMOGRAPHIC_ROWS),
                       ('gold_msi', sheets.GOLD_MSI_ROWS),
                       ('postexp', sheets.POSTEXP_ROWS)):
        got = [fl.row for fl in fields[form]]
        if got != rows:
            missing = [r for r in rows if r not in got]
            raise ValueError('Fields of %s do not match the sheet rows; '
                             'missing: %s' % (form, missing))
    return fields
