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
  +1  the extreme answer, 6
  +2  "did not understand" (Q1 at 5-6) a task found easy (Q2 at 1-2)
  +1  the same question answered 1-2 in another session

  score >= 4: likely; 2-3: possible; <= 1: unlikely.

A possibly crossed-out answer to questions 1-6, at least FAR from the
other marked answer (a participant who noticed the scale direction),
raises the rating by one level, or rates
the session "self-corrected" if no answer is at the bad end.

Against the assessments by hand (October 2026, 78 participants): 26 of
the 27 sessions rated likely or very likely are rated possible or more
(listed for assessment), against 2 of the 6 rated unlikely.

Run it whenever sheets are added or corrected. The list goes to
tdtb_private/reversed_scales_to_assess.tsv; only counts are printed.
The rating is a screen: the decision is taken on paper and recorded by
hand in the assessment file, as before: each session to assess gets a
row "to assess" in the assessment file, with the screen's rating and
reasons, to complete after the assessment on paper (status "verified on
paper", likelihood, evidence).

REVERSED_SCALES then follows the assessment file: each run adds the
assessed sessions not listed yet, as an active entry if rated likely or
very likely (the answers are corrected), commented out if rated possible
(not corrected); sessions rated unlikely, none or self-corrected are not
added. Entries already listed are never changed: if one disagrees with
the assessment file (e.g. edited by hand), a warning is printed. Run
postses_results.py after entries are added.

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
# Status of the rows added by the screen to the assessment file, until
# the assessment on paper is recorded
TO_ASSESS = 'to assess'
ASSESSED_COLUMNS = ['subject', 'status', 'sessions_affected',
                    'items_affected', 'Q1_understand (1=best)',
                    'Q3_concentrate (1=best)', 'Q6_motivated (6=best)',
                    'likelihood', 'evidence']
# A crossed-out answer counts only if it is this far from the other marked
# answer: a correction to a neighbouring number (5 -> 4) is not a scale
# reversal (on the paper forms: of 13 sessions flagged, the 4 false ones
# were such corrections or large circles)
FAR = 3


def _session(fname):
    """('behav_ses-02', subject) from a sheet or draft file name."""
    m = re.search(r'(behav|mri)_postexp_sub-(\d+)_ses-(\d+)', fname)
    return int(m.group(2)), '%s_ses-%s' % (m.group(1), m.group(3))


def _read(path, header):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    return rows[1:] if header else rows


def _rows():
    if not os.path.exists(ASSESSED):
        return []
    return list(csv.DictReader(open(ASSESSED, newline='', encoding='utf-8'),
                               delimiter='\t'))


def _assessed_rows():
    """Rows of the assessment file, without those still to assess."""
    return [r for r in _rows() if not r['status'].startswith(TO_ASSESS)]


def _code(ses):
    """'beh1' from 'behav_ses-01'."""
    kind, n = ses.split('_ses-')
    return ('beh' if kind == 'behav' else 'mri') + str(int(n))


def add_draft_rows(todo, rated, sessions):
    """Add to the assessment file a row "to assess" per participant for
    the sessions of todo not in it yet; return the number of sessions.
    The row is completed by hand after the assessment on paper."""
    pending = set()
    for r in _rows():
        if r['status'].startswith(TO_ASSESS):
            s = int(re.sub(r'\D', '', r['subject']))
            pending |= {(s, _ses(k, n)) for k, n in
                        re.findall(SESSION_CODE, r['sessions_affected'])}
    new = {}
    for k in todo:
        if k not in pending:
            new.setdefault(k[0], []).append(k[1])
    rows = []
    for s, sess in sorted(new.items()):
        items = set()
        for ses in sess:
            v, crossed = sessions[s][ses]
            items |= {q for q in BEST_IS_ONE
                      if v[q] is not None and v[q] >= BAD_END} | set(crossed)
        by_ses = sorted(sessions[s].items())
        cols = [' '.join('%s=%s' % (_code(ses), 'NA' if v[q] is None
                                    else v[q]) for ses, (v, _) in by_ses)
                for q in ('Q1', 'Q3', 'Q6')]
        rows.append(['sub-%02d' % s, TO_ASSESS, ', '.join(map(_code, sess)),
                     ', '.join(sorted(items))] + cols + [
            '%s (screen: %s)' % (TO_ASSESS, '; '.join(
                '%s (%s)' % (rated[(s, ses)][0], _code(ses))
                for ses in sess)),
            'Screen: ' + ' '.join('%s: %s.' % (_code(ses), '; '.join(
                rated[(s, ses)][2])) for ses in sess) +
            ' After the assessment on paper, set status to "verified on '
            'paper", likelihood to the rating and this text to the '
            'evidence.'])
    if rows:
        raw = open(ASSESSED, encoding='utf-8').read() \
            if os.path.exists(ASSESSED) else ''
        with open(ASSESSED, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            if not raw:
                w.writerow(ASSESSED_COLUMNS)
            elif not raw.endswith('\n'):
                f.write('\n')
            w.writerows(rows)
    return sum(len(v) for v in new.values())


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
        for r in _assessed_rows():
            s = int(re.sub(r'\D', '', r['subject']))
            codes = r['sessions_affected']
            if codes.strip() == 'all':
                codes = 'beh1 beh2 beh3 mri1 mri2'
            # Sessions named anywhere in the entry, self-corrected included
            for kind, n in re.findall(r'(beh|mri)\s*(\d)', codes):
                ses = '%s_ses-0%s' % ('behav' if kind == 'beh' else 'mri', n)
                out.setdefault((s, ses), 'assessment file')
    return out


def _far(detail):
    """True if the two most-inked options of a draft answer ("ink 1=230,
    2=0, ...") are at least FAR apart on the scale."""
    ink = sorted(((int(n), int(v)) for v, n in
                  re.findall(r'(\d+)=(\d+)', detail)), reverse=True)
    return len(ink) > 1 and abs(ink[0][1] - ink[1][1]) >= FAR


def answers(rows):
    """{'Q1': 3, ...} (None if missing) and the questions with a possibly
    crossed-out answer, from the rows of a sheet or a draft."""
    t = {r[0]: r for r in rows if len(r) > 1}
    v, crossed = {}, set()
    for q, text in QUESTIONS.items():
        r = t.get(text)
        v[q] = int(r[1]) if r and r[1].isdigit() else None
        if r and len(r) > 3 and CROSSED_OUT in r[3] and _far(r[3]):
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
    if any(v[q] == 6 for q in bad):
        score += 1
        reasons.append('the extreme answer 6')
    if 'Q1' in bad and v['Q2'] is not None and v['Q2'] <= 2:
        score += 2
        reasons.append('did not understand an easy task (Q2=%d)' % v['Q2'])
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
        for r in _assessed_rows():
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


# REVERSED_SCALES follows the assessment file: sessions rated likely or
# very likely are corrected (active entry), possible ones are listed
# commented out, others (unlikely, none, self-corrected) are not listed
ACTIVE = ('very likely', 'likely')
COMMENT_AT = 45         # column of the rating comment in REVERSED_SCALES
SESSION_CODE = r'(beh|mri)\s*(\d)'


def _rating(text):
    """very likely, likely or possible (else None) from a free text."""
    t = text.strip().lower()
    for r in ('very likely', 'likely', 'possible'):
        if t.startswith(r):
            return r
    return None


def _ses(kind, n):
    return '%s_ses-0%s' % ('behav' if kind == 'beh' else 'mri', n)


def _subject_sessions(s):
    """Sessions with a post-session sheet (for "all")."""
    out = []
    for path in glob.glob(os.path.join(FORMS_DIR, 'sub-%02d' % s,
                                       '*postexp*.tsv')):
        out.append(_session(path)[1])
    return sorted(out) or [_ses(k, n) for k, n in
                           (('beh', 1), ('beh', 2), ('beh', 3), ('mri', 1),
                            ('mri', 2))]


def assessed_entries():
    """What REVERSED_SCALES should hold according to the assessment file:
    {(subject, session): {item: rating}}, rating likely, very likely or
    possible."""
    out = {}
    if not os.path.exists(ASSESSED):
        return out
    for r in _assessed_rows():
        s = int(re.sub(r'\D', '', r['subject']))
        # Sessions: a part "(self-corrected)" or "none" is left out; a note
        # in brackets ("beh1 (beh2 self-corrected)") is not a session
        sessions = []
        for part in r['sessions_affected'].split(';'):
            if re.search(r'\(\s*self-corrected\s*\)', part):
                continue
            part = re.sub(r'\([^)]*\)', '', part)
            if re.search(r'\bnone\b', part):
                continue
            if part.strip() == 'all':
                sessions += _subject_sessions(s)
            sessions += [_ses(k, n) for k, n in
                         re.findall(SESSION_CODE, part)]
        # Items, without notes in brackets and self-corrected or
        # not-reversed parts
        items = []
        for part in r['items_affected'].split(';'):
            if re.search(r'\(\s*self-corrected\s*\)', part):
                continue
            part = re.sub(r'\([^)]*\)', '', part)
            if re.search(r'self-corrected|not reversed|\bnone\b', part):
                continue
            items += re.findall(r'Q\d', part)
        # Ratings: "likely", or per session or item: "likely (beh2);
        # possible (beh1)", "very likely (Q1); possible (Q3)"
        default, by_ses, by_item = None, {}, {}
        for part in r['likelihood'].split(';'):
            rating = _rating(part)
            scope = re.search(r'\(([^)]*)\)', part)
            scope = scope.group(1) if scope else ''
            codes = [_ses(k, n) for k, n in re.findall(SESSION_CODE, scope)]
            qs = re.findall(r'Q\d', scope)
            for c in codes:
                by_ses[c] = rating
            for q in qs:
                by_item[q] = rating
            if not codes and not qs and default is None:
                default = rating
        for ses in sessions:
            for q in items:
                rating = by_item.get(q, by_ses.get(ses, default))
                if rating:
                    out.setdefault((s, ses), {})[q] = rating
    return out


def listed_entries(src):
    """REVERSED_SCALES of postses_results.py: {(subject, session): {item:
    (rating, active)}}."""
    out = {}
    i = src.index('REVERSED_SCALES = [')
    j = src.index('\n]', i)
    for line in src[i:j].splitlines()[1:]:
        m = re.match(r"\s*(#\s*)?\((\d+), '(\w+-\d+)', (.*)\),\s*#\s*(.*)$",
                     line)
        if not m:
            continue
        items = ast.literal_eval(m.group(4))
        items = (items,) if isinstance(items, str) else items
        for q in items:
            out.setdefault((int(m.group(2)), m.group(3)), {})[q] = (
                _rating(m.group(5)), not m.group(1))
    return out


def sync_reversed_scales(write=True):
    """Add to REVERSED_SCALES the assessed sessions it does not list yet;
    return (lines added, warnings). Listed entries are never changed: a
    disagreement with the assessment file is only reported."""
    src = open(RESULTS_SCRIPT, encoding='utf-8').read()
    want, have = assessed_entries(), listed_entries(src)
    added, warnings = [], []
    for key in sorted(want):
        if key in have:
            for q in sorted(set(want[key]) | set(have[key])):
                w, h = want[key].get(q), have[key].get(q)
                expect = (w, w in ACTIVE) if w else None
                if expect != h:
                    warnings.append(
                        'sub-%02d %s %s: assessed %s, REVERSED_SCALES has %s'
                        % (key + (q, w or 'not to correct',
                                  ('%s, %s' % (h[0], 'active' if h[1]
                                               else 'commented out'))
                                  if h else 'nothing')))
            continue
        by_rating = {}
        for q, rating in sorted(want[key].items()):
            by_rating.setdefault(rating, []).append(q)
        for rating, qs in by_rating.items():
            items = repr(qs[0]) if len(qs) == 1 else repr(tuple(qs))
            entry = '%s(%d, %r, %s),' % ('' if rating in ACTIVE else '# ',
                                         key[0], key[1], items)
            added.append(('    ' + entry).ljust(COMMENT_AT) + '# ' + rating)
    for key in sorted(set(have) - set(want)):
        warnings.append('sub-%02d %s: in REVERSED_SCALES but not assessed as '
                        'likely or possible in the assessment file' % key)
    if added and write:
        j = src.index('\n]', src.index('REVERSED_SCALES = ['))
        src = src[:j] + '\n' + '\n'.join(added) + src[j:]
        with open(RESULTS_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(src)
    return added, warnings


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
    n_new = add_draft_rows(todo, rated, sessions)
    print('sessions rated: %d' % len(rated))
    print('  already assessed: %d' % sum(k in done for k in rated))
    print('  still to assess (rated possible or more, or self-corrected): '
          '%d (%d participants), listed in %s'
          % (len(todo), len({s for s, _ in todo}), OUTPUT))
    print('  rows "%s" added to %s: %d sessions'
          % (TO_ASSESS, os.path.basename(ASSESSED), n_new))
    added, warnings = sync_reversed_scales()
    print('REVERSED_SCALES: %d entries added' % len(added))
    for line in added:
        print('  ' + line.strip())
    if added:
        print('  run postses_results.py to apply them')
    for w in warnings:
        print('WARNING: ' + w)
    print('Ratings by hand (rows) against this rating:')
    for (hand, auto), n in sorted(agreement(rated,
                                            recorded_ratings()).items()):
        print('  %-15s %-15s %3d' % (hand, auto, n))

if __name__ == '__main__':
    main()
