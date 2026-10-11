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
   any order and grouping:

   ```
   python -m transcription.transcribe 78 ~/scans/sub-78/*.pdf
   ```

   It writes to `~/OneDrive/tdtb_admin/tdtb_private/forms_drafts/sub-78/`
   one draft per sheet (`*.draft.tsv`) and `review_sub-78.pdf`, in about
   5 minutes. Close other large applications: the model needs memory.

2. **Review.** Open `review_sub-78.pdf` next to the paper forms. Each
   answer is shown next to the part of the scan it was read from, with the
   chosen option framed in green.
   - **Orange rows** (`check`) must be checked: unclear marks and all
     handwriting. Compare the answer with its crop, as the reason given
     may not be the only problem (e.g. day and month swapped in a date).
   - **Green rows** (`ok`) were read from clear marks.
   - **Grey rows** are empty (`n/a`).
   - Notes at the top list unidentified, missing or uncertain pages.

   Correct the answers in the draft sheets (column `answer`; any
   spreadsheet editor, saved as TSV). Missing answers are `n/a`.

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

   Rates each post-session questionnaire for a reversed scale and adds a
   `to assess` row for each new session to
   `tdtb_private/postsess_suspected_reversed_scales.tsv`. After checking
   the paper, complete that row (status `verified on paper`, likelihood,
   evidence) and run the screen again: it adds the session to
   `REVERSED_SCALES` in `postses_results.py`, active if likely or very
   likely, commented out if possible, and warns if an entry already
   listed disagrees with the assessment. Then run `postses_results.py`.
   The sheets always keep the answers as written on the paper.

5. Add the subject to `subject_lists.py` (and the lists of the other
   behavioral scripts) as usual.

## What to know when reviewing

- **Post-session forms** are assigned to sessions in page order (across
  the PDFs, in the order given), unless the PDF file name holds a session
  label (`ses-02`). Behavioral sessions come before imaging sessions; the
  number of sessions comes from `subject_lists.py` (a subject not yet
  listed is assumed to have 2 behavioral sessions). If the dates on the
  forms do not follow that order, the forms are marked `check`.
- **Never read:** the Participant ID on every form, and the name, email
  and anonymous ID on Gold-MSI page 8.
- **Not on the forms:** *Date of Birth* (`n/a` unless filled in by hand)
  and the Gold-MSI *Date* (copied from the reviewed demographic date by
  `finalize`).
- **Crossed-out answers:** when two options are marked, the post-session
  form keeps the one with less ink (a crossed-out answer has more ink),
  the other forms the one with more ink; both are marked `check`.
- **Several levels ticked** (school level, Gold-MSI education): the
  highest level obtained is kept ("still in education" is not a level).
- **Strategies** (post-session form): "No" with a strategy described
  below becomes "Yes" (marked `check`).
- **Empty areas:** an area with ink around it but not inside is marked
  `check` (`written outside?`). Handwriting the model could not read
  (e.g. "N/A", or a line drawn to leave the area blank) is `n/a`, marked
  `check`.
- **Dates** are `DD/MM/YYYY`. A date whose day and month could be swapped
  says so; one that matches another form only when swapped is swapped
  back.
- **Gold-MSI pages 2-4 look alike:** a page identified by elimination is
  noted at the top of the review PDF; check that its questions match the
  paper.

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
| `../reversed_scales_screen.py` | Step 4: reversed-scale screen |

## Testing without real data

```
python -m transcription.synthetic /tmp/fake --seed 0
python -m transcription.transcribe 99 /tmp/fake/*.pdf --out /tmp/fake/drafts
```

`/tmp/fake/truth/` holds the true answers of the fake forms; the drafts
are in `/tmp/fake/drafts/sub-99/`.

## Accuracy check on the curated scans

```
nohup setsid ~/anaconda3/envs/forms-ocr-py312/bin/python \
    -m transcription.validate > /dev/null 2>&1 &
```

Transcribes the scans of every participant that already has curated
sheets into `tdtb_private/forms_validation/sub-NN/` and compares each
draft with its sheet, writing only counts to
`forms_validation/accuracy.txt`. It takes 5-6 hours for 74 participants
and resumes where it stopped if interrupted. Run it after changing the
code that reads the forms.

Result (October 2026, 74 participants, 8,658 answers): 0.4% of the
answers were marked `ok` but differ from the sheet (these are not caught
by the review); about 29% are marked `check`.
