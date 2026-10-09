"""
List the post-session questionnaires that still need a reversed-scale
assessment.

Some participants answer questions 1-6 of the post-session questionnaire
as if the scale were reversed (see postses_results.py). The assessment
is done by hand, on paper, and recorded in
postsess_suspected_reversed_scales.tsv (private folder); its outcome
goes in REVERSED_SCALES of postses_results.py. This script rates every
post-session questionnaire and lists those that call for an assessment
but are not recorded yet.

Rating. A session is rated only if understanding (Q1) or concentration
(Q3), whose best answer is 1, was answered 5 or 6, or if the
transcription found a possibly crossed-out answer to questions 1-6.
The "bad" answer is then weighed against the participant's other
answers (score, see rate()):

  +2  both Q1 and Q3 at 5-6
  +1  rarely guessed (Q5 at 1-2)
  +1  motivated (Q6 at 3-6); -1 if not (Q6 at 1-2, or missing)
  -1  found the task very difficult (Q2 at 5-6): consistent with a
      genuine "did not understand"
  +1  the same question answered 1-2 in another session

  score >= 4: likely; 2-3: possible; <= 1: unlikely.

A possibly crossed-out answer to questions 1-6 (a participant who
noticed the scale direction) raises the rating by one level, or rates
the session "self-corrected" if no answer is at the bad end.

Against the earlier assessment by hand (sheets of the 74 first
participants): 21 of the 24 sessions rated likely or very likely by
hand scored >= 2, and none of the 3 sessions rated unlikely did.

Run it whenever sheets are added or corrected. The list goes to
tdtb_private/reversed_scales_to_assess.tsv; only counts are printed.
The rating is a screen: the decision is taken on paper and recorded by
hand, as before.

Usage:
    python reversed_scales_screen.py [--drafts DIR]

--drafts reads the transcription drafts from DIR instead (e.g.
tdtb_private/forms_validation, after transcribing already curated scans).

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.10.14
"""

import argparse
import ast
import csv
import glob
import os
import re

PRIVATE_DIR = os.path.expanduser('~/OneDrive/tdtb_admin/tdtb_private')
FORMS_DIR = os.path.join(PRIVATE_DIR, 'forms')
DRAFT_DIR = os.path.join(PRIVATE_DIR, 'forms_drafts')
ASSESSED = os.path.join(PRIVATE_DIR, 'postsess_suspected_reversed_scales.tsv')
OUTPUT = os.path.join(PRIVATE_DIR, 'reversed_scales_to_assess.tsv')
RESULTS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'postses_results.py')

QUESTIONS = {
    'Q1': 'How well did you understand the task?',
    'Q2': 'How difficult did you find the task?',
    'Q3': 'How strongly did you concentrate on the task?',
    'Q4': 'Did your concentration on the task change throughout the '
          'experiment?',
    'Q5': 'Were there occasions where you guessed when responding?',
    'Q6': 'How motivated were you for the experiment?'}
# Questions whose best answer is 1: an answer of BAD_END or more is
# suspicious
BEST_IS_ONE = ('Q1', 'Q3')
BAD_END = 5
CROSSED_OUT = 'one may be crossed out'


def _session(fname):
    """('behav_ses-02', subject) from a sheet or draft file name."""
    m = re.search(r'(behav|mri)_postexp_sub-(\d+)_ses-(\d+)', fname)
    return int(m.group(2)), '%s_ses-%s' % (m.group(1), m.group(3))


def _read(path, header):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    return rows[1:] if header else rows


def recorded():
    """Sessions already assessed: {(subject, session): source}."""
    out = {}
    src = open(RESULTS_SCRIPT, encoding='utf-8').read()
    i = src.index('REVERSED_SCALES = [')
    j = src.index('\n]', i)
    for s, ses, _ in ast.literal_eval(src[i + len('REVERSED_SCALES = '):
                                          j + 2]):
        out[(s, ses)] = 'REVERSED_SCALES'
    for s, ses in re.findall(r"#\s*\((\d+), '(\w+-\d+)'", src[i:j]):
        out.setdefault((int(s), ses), 'REVERSED_SCALES (commented out)')
    if os.path.exists(ASSESSED):
        for r in csv.DictReader(open(ASSESSED, newline='', encoding='utf-8'),
                                delimiter='\t'):
            s = int(re.sub(r'\D', '', r['subject']))
            codes = r['sessions_affected']
            if codes.strip() == 'all':
                codes = 'beh1 beh2 beh3 mri1 mri2'
            # Sessions named anywhere in the entry, self-corrected included
            for kind, n in re.findall(r'(beh|mri)\s*(\d)', codes):
                ses = '%s_ses-0%s' % ('behav' if kind == 'beh' else 'mri', n)
                out.setdefault((s, ses), 'assessment file')
    return out


def answers(rows):
    """{'Q1': 3, ...} (None if missing) and the questions with a possibly
    crossed-out answer, from the rows of a sheet or a draft."""
    t = {r[0]: r for r in rows if len(r) > 1}
    v, crossed = {}, set()
    for q, text in QUESTIONS.items():
        r = t.get(text)
        v[q] = int(r[1]) if r and r[1].isdigit() else None
        if r and len(r) > 3 and CROSSED_OUT in r[3]:
            crossed.add(q)
    return v, crossed


def rate(v, others, crossed=()):
    """(rating, score, reasons) of one session, or None if not rated.

    v: answers of the session; others: answers of the participant's other
    sessions; crossed: questions with a possibly crossed-out answer.
    """
    bad = [q for q in BEST_IS_ONE if v[q] is not None and v[q] >= BAD_END]
    reasons = ['%s answered %d (1 = best)' % (q, v[q]) for q in bad]
    if crossed:
        reasons.append('possibly crossed-out answer (%s): scale-direction '
                       'confusion on paper' % ', '.join(sorted(crossed)))
    if not bad:
        return ('self-corrected', None, reasons) if crossed else None
    score = 0
    if len(bad) == 2:
        score += 2
    if v['Q5'] is not None and v['Q5'] <= 2:
        score += 1
        reasons.append('rarely guessed (Q5=%d)' % v['Q5'])
    if v['Q6'] is not None and v['Q6'] >= 3:
        score += 1
        reasons.append('motivated (Q6=%d)' % v['Q6'])
    else:
        score -= 1
        reasons.append('not motivated (Q6=%s)' % v['Q6'])
    if v['Q2'] is not None and v['Q2'] >= 5:
        score -= 1
        reasons.append('found the task very difficult (Q2=%d)' % v['Q2'])
    if any(o[q] is not None and o[q] <= 2 for o in others for q in bad):
        score += 1
        reasons.append('same question answered 1-2 in another session')
    rating = 'likely' if score >= 4 else 'possible' if score >= 2 \
        else 'unlikely'
    if crossed:
        rating = {'unlikely': 'possible', 'possible': 'likely',
                  'likely': 'very likely'}[rating]
    return rating, score, reasons


def rate_all(sessions):
    """{(subject, session): (rating, score, reasons)} from
    {subject: {session: (answers, crossed)}}."""
    out = {}
    for s, by_ses in sessions.items():
        for ses, (v, crossed) in by_ses.items():
            others = [o for k, (o, _) in by_ses.items() if k != ses]
            r = rate(v, others, crossed)
            if r:
                out[(s, ses)] = r
    return out


def agreement(rated, recorded):
    """Counts of (rating by hand, rating by rate()) over the sessions
    rated by either; sessions in neither are not counted."""
    out = {}
    for k in set(rated) | set(recorded):
        key = (recorded.get(k, 'not assessed'),
               rated[k][0] if k in rated else 'not rated')
        out[key] = out.get(key, 0) + 1
    return out


def load(folder, pattern, header):
    """{subject: {session: (answers, crossed)}} of the sheets or drafts."""
    out = {}
    for path in glob.glob(os.path.join(folder, 'sub-*', pattern)):
        if not re.search(r'_ses-\d+', path):
            continue        # e.g. a form not assigned to a session
        s, ses = _session(path)
        out.setdefault(s, {})[ses] = answers(_read(path, header))
    return out


def recorded_ratings():
    """The ratings given by hand: {(subject, session): rating}, with
    rating very likely, likely, possible, unlikely or self-corrected."""
    out = {}
    src = open(RESULTS_SCRIPT, encoding='utf-8').read()
    i = src.index('REVERSED_SCALES = [')
    j = src.index('\n]', i)
    for line in src[i:j].splitlines():
        m = re.search(r"\((\d+), '(\w+-\d+)', .*\),\s*#\s*(.*)$", line)
        if m:
            note = m.group(3)
            out[(int(m.group(1)), m.group(2))] = (
                'very likely' if note.startswith('very likely')
                else 'likely' if note.startswith('likely') else 'possible')
    if os.path.exists(ASSESSED):
        for r in csv.DictReader(open(ASSESSED, newline='', encoding='utf-8'),
                                delimiter='\t'):
            s = int(re.sub(r'\D', '', r['subject']))
            for part in r['sessions_affected'].split(','):
                for kind, n in re.findall(r'(beh|mri)\s*(\d)', part):
                    ses = '%s_ses-0%s' % ('behav' if kind == 'beh'
                                          else 'mri', n)
                    if 'self-corrected' in part:
                        out.setdefault((s, ses), 'self-corrected')
                    elif r['likelihood'].startswith('unlikely'):
                        out.setdefault((s, ses), 'unlikely')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description='List the post-session '
                                'questionnaires still to assess for a '
                                'reversed scale.')
    p.add_argument('--drafts', default=DRAFT_DIR,
                   help='folder of transcription drafts (sub-NN/)')
    a = p.parse_args(argv)
    done = recorded()
    sessions = load(FORMS_DIR, '*postexp*.tsv', False)
    # Crossed-out answers are only seen by the transcription
    for s, by_ses in load(a.drafts, '*postexp*.draft.tsv', True).items():
        for ses, (_, crossed) in by_ses.items():
            if ses in sessions.get(s, {}):
                sessions[s][ses] = (sessions[s][ses][0], crossed)
    rated = rate_all(sessions)
    todo = sorted(k for k, r in rated.items()
                  if k not in done and r[0] != 'unlikely')
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['subject', 'session', 'rating', 'score', 'reasons'])
        for k in todo:
            rating, score, reasons = rated[k]
            w.writerow(['sub-%02d' % k[0], k[1], rating,
                        '' if score is None else score, '; '.join(reasons)])
    print('sessions rated: %d' % len(rated))
    print('  already assessed: %d' % sum(k in done for k in rated))
    print('  still to assess (rated possible or more, or self-corrected): '
          '%d (%d participants), listed in %s'
          % (len(todo), len({s for s, _ in todo}), OUTPUT))
    print('Ratings by hand (rows) against this rating:')
    for (hand, auto), n in sorted(agreement(rated,
                                            recorded_ratings()).items()):
        print('  %-15s %-15s %3d' % (hand, auto, n))

if __name__ == '__main__':
    main()
