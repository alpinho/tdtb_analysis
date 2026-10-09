# Transcription of the scanned forms

Turns the scanned paper forms of a new participant (demographic
questionnaire, Gold-MSI, post-session questionnaires) into the sheets read
by the questionnaire scripts (`demographic_data_extraction.py`,
`parse_score_goldmsi.py`, `postses_results.py`).

The computer drafts every answer; you check the unclear ones against the
paper forms before the sheets are written. **Everything runs on this
computer**: circles and ticks are read by image processing, handwriting by
a vision model running locally in [Ollama](https://ollama.com). No scan,
image or answer is sent anywhere.

## Setup (once)

```
conda create -n forms-ocr-py312 python=3.12
conda activate forms-ocr-py312
pip install opencv-python-headless pymupdf numpy ollama

curl -fsSL https://ollama.com/install.sh | sh    # Ollama (needs sudo)
ollama pull qwen2.5vl:3b                         # the handwriting model
```

The blank forms used as templates are in `music-sdtb_protocols`,
`experiments/behavioral/ethics/give_to_the_participant_2026/`:
`GOLD-MSI.pdf`, and `demographic_questionnaire_Grahn_2019.pdf` and
`post_experiment_questionnaire.pdf` **exported from Microsoft Word**
(LibreOffice lays these pages out differently from the printed copies).
If a form changes, re-export it and update `layout.py`.

## Use (per participant)

Run from `behavioral_analysis/questionnaires/`, in the
`forms-ocr-py312` environment.

1. **Draft.** Give the subject number and all of their scanned PDFs, in
   any order and grouping (e.g. one PDF with everything but the
   post-session form of session 2, plus one PDF with that form):

   ```
   python -m transcription.transcribe 78 ~/scans/sub-78/*.pdf
   ```

   It writes to `~/OneDrive/tdtb_admin/tdtb_private/forms_drafts/sub-78/`:
   one draft per sheet (`*.draft.tsv`) and `review_sub-78.pdf`.
   It takes about 4-5 minutes per participant (mostly the handwriting
   model; close other large applications, as the model needs memory).

2. **Review.** Open `review_sub-78.pdf` next to the paper forms. Each
   answer is shown next to the part of the scan it was read from, with the
   chosen option framed in green. **Orange rows** must be checked: unclear
   marks, and all handwriting (the model can misread). Compare the answer
   of every orange row with its crop, not only that the row is orange
   (e.g. a date with day and month swapped is only seen this way). Correct
   the answers
   in the draft sheets (column `answer`; any spreadsheet editor, saved as
   TSV). Missing answers are `n/a`.

3. **Finalize.**

   ```
   python -m transcription.finalize 78 --reviewed
   ```

   It checks every answer (rows, options, numbers, dates `DD/MM/YYYY`,
   percentages) and writes the final sheets to
   `~/OneDrive/tdtb_admin/tdtb_private/forms/sub-78/`. Without
   `--reviewed` it refuses while rows are still marked `check`. It never
   replaces an existing sheet unless given `--overwrite`.

4. **Screen for reversed scales.**

   ```
   cd ..; python reversed_scales_screen.py
   ```

   Rates each post-session questionnaire for a reversed scale (likely,
   possible, unlikely, self-corrected), weighing an understanding or
   concentration answer of 5-6 against the participant's other answers and
   any crossed-out answer, and lists in
   `tdtb_private/reversed_scales_to_assess.tsv` those rated possible or
   more that are not yet recorded in
   `postsess_suspected_reversed_scales.tsv`; it also prints how its
   ratings compare with the ratings given by hand.
   Record the assessment there, and its outcome in `REVERSED_SCALES` of
   `postses_results.py`. The sheets always keep the answers as written on
   the paper (a crossed-out answer is not an answer).

5. Add the subject to `subject_lists.py` (and the lists of the other
   behavioral scripts) as usual.

## What to know

- **Post-session forms** are assigned to sessions by page order (across
  the PDFs, in the order given on the command line), except that a form
  from a PDF whose file name holds a session label (`ses-02`) takes that
  session. Behavioral sessions come before imaging sessions; the number of
  sessions comes from `subject_lists.py` (a new subject not yet listed is
  assumed to have 2 behavioral sessions and no imaging session). If the
  dates read on the forms do not follow that order, the forms are marked
  `check`. On the curated scans this rule was right for 94% of the forms;
  sorting by the dates (a quarter of them misread) or by session labels
  written on the forms (mostly misread by the model) did worse.
- **Not on the forms:** *Date of Birth* (the demographic form has no such
  field; `n/a` unless filled in by hand) and the Gold-MSI *Date* (copied
  from the demographic form by `finalize`).
- **Never read:** the Participant ID on every form and the name, email and
  anonymous ID of Gold-MSI page 8. They are not defined in `layout.py`,
  so they are never read or written.
- **Marks** (circles, ticks) are found by the ink they add to the blank
  form. An answer is marked `check` when no option has a mark (`no mark
  found`: the mark may be outside the options), when the mark is faint,
  or when a second option also has a clear mark (`one may be crossed
  out`: a crossed-out answer has more ink than a circle; it is not an
  answer). On Gold-MSI pages 2-4 the whole table cell of an option
  counts, as some participants tick the middle of the cell.
- **Handwriting:** strokes centred outside an answer area (e.g. a circle
  around the answer of the question above) are ignored. An area with no
  handwriting gives `n/a`, marked `ok`.
- **Dates** are `DD/MM/YYYY`. The model sometimes swaps day and month: a
  date that matches another form's date only when swapped is swapped
  back (and marked `check`), and a date whose day and month could be
  swapped says so.
- **Gold-MSI pages 2-4 look alike.** A page whose identity is uncertain
  is taken as the one page missing from the form (`page identified by
  elimination`; its answers are marked `check`).
- **School level** (demographic form): if several boxes are ticked, the
  highest level is kept and the row is marked `check`.
- Unidentified pages and missing pages are listed at the top of the
  review PDF.

## Files

| File | Role |
| ---- | ---- |
| `transcribe.py` | Command 1: scans to drafts and review PDF |
| `finalize.py` | Command 3: checked drafts to final sheets |
| `layout.py` | Where every answer is on the blank forms |
| `align.py` | Identifies each scanned page and aligns it to the blank form |
| `read.py` | Reads circles, ticks and handwriting areas; thresholds at the top |
| `handwriting.py` | Calls the local handwriting model |
| `review.py` | Builds the review PDF |
| `sheets.py` | Rows of the sheets (wording must not change) |
| `synthetic.py` | Makes fake filled-in forms for testing |
| `validate.py` | Accuracy check on the scans that already have sheets |
| `../reversed_scales_screen.py` | Step 4: reversed-scale rating and list |

## Testing without real data

```
python -m transcription.synthetic /tmp/fake --seed 0
python -m transcription.transcribe 99 /tmp/fake/*.pdf --out /tmp/fake/drafts
```

`/tmp/fake/truth/` holds the true answers of the fake forms; the drafts
are in `/tmp/fake/drafts/sub-99/`.

## Accuracy check on the curated scans

```
python -m transcription.validate
```

Transcribes the scans of every participant that already has curated
sheets (`tdtb_private/forms_scanned/`) into
`tdtb_private/forms_validation/sub-NN/` and compares each draft with its
sheet. It takes 5-6 hours for the 74 participants; a participant already
compared is skipped, so an interrupted run resumes where it stopped
(rename or remove the folder to start again). Only counts are written, to
`forms_validation/accuracy.txt` and `run.log`: agreement per form, kind of
answer and status (`ok`/`check`), the questions that disagree most, and
the reversed-scale ratings (from the drafts and from the sheets) against
the ratings given by hand. The answers that matter most are those marked
`ok` that disagree with the sheet: the review does not catch them.

To run it unattended:

```
nohup setsid ~/anaconda3/envs/forms-ocr-py312/bin/python \
    -m transcription.validate > /dev/null 2>&1 &
```
