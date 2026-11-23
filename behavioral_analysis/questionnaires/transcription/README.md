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

The same steps work on macOS, Windows and Linux. Commands are typed in a
terminal (macOS, Linux) or in the Anaconda Prompt (Windows).

### Requirements

- **Anaconda or Miniconda.**
- **Ollama** with the handwriting model (step 2). It needs a fair amount
    of memory: close other large applications while transcribing.
- **The `tdtb_admin` folder** on the local computer, anywhere. The
    participant data are never in the repository; the scripts read and
    write them inside `tdtb_admin`, whose structure is fixed:

        tdtb_admin/
        └── tdtb_private/
            ├── forms_scanned/sub-NN/    scanned PDFs (input)
            ├── forms_drafts/sub-NN/     drafts and review PDF (transcribe.py)
            └── forms/sub-NN/            final sheets (finalize.py)

### Steps

1. **Python environment.**

        conda create -n forms-ocr-py312 python=3.12
        conda activate forms-ocr-py312
        pip install opencv-python-headless pymupdf numpy ollama

2. **Handwriting model.** Install Ollama from
    <https://ollama.com/download>, then:

        ollama pull qwen2.5vl:3b

3. **Path of `tdtb_admin`.** Open `behavioral_analysis/questionnaires/admin_paths.py`
    and set `ADMIN_DIR` to where `tdtb_admin` is on your computer, e.g.

        ADMIN_DIR = '~/OneDrive/tdtb_admin'                       # macOS, Linux
        ADMIN_DIR = r'C:\Users\you\Documents\tdtb_admin'          # Windows

    Do not commit this change: it is your own path. To keep git from
    picking it up, run once, from the repository:

        git update-index --skip-worktree behavioral_analysis/questionnaires/admin_paths.py

4. **Check.** From `behavioral_analysis/questionnaires/`:

        python admin_paths.py

    It prints the `tdtb_admin` folder the scripts will use, or stops if
    `tdtb_private` is not inside it.

### Blank forms

The blank forms used as templates are in `templates/`, next to the
scripts. They are copies of the forms in `music-sdtb_protocols`,
`experiments/behavioral/ethics/give_to_the_participant_2026/`:
`GOLD-MSI.pdf`, and `demographic_questionnaire_Grahn_2019.pdf` and
`post_experiment_questionnaire.pdf` **exported from Microsoft Word**
(LibreOffice lays these pages out differently from the printed copies).
If a form changes, re-export it, copy it into `templates/` and update
`layout.py`.

## Use (per participant)

Run every command from `behavioral_analysis/questionnaires/`, in the
`forms-ocr-py312` environment.

1. **Subject lists.** Add the subject to `subject_lists.py` (and to the
    lists of the other behavioral scripts) as usual. The transcription
    reads there how many behavioral and imaging sessions the subject has.

2. **Draft.** Give the subject number and the scanned PDFs, in any order
    and grouping:

        python -m transcription.transcribe 78 ~/scans/sub-78/*.pdf

    It writes to `tdtb_admin/tdtb_private/forms_drafts/sub-78/` one draft
    per sheet (`*.draft.tsv`) and a review PDF for the forms of this run
    (e.g. `review_sub-78_demo_goldmsi_behav01.pdf`), in about 5 minutes.
    Close other large applications: the model needs memory.

3. **Review.** Open the review PDF next to the paper forms. Each answer
    is shown next to the part of the scan it was read from, with the
    chosen option framed in green.
    - **Orange rows** (`check`) must be checked: unclear marks and all
      handwriting. Compare the answer with its crop, as the reason given
      may not be the only problem (e.g. day and month swapped in a date).
    - **Green rows** (`ok`) were read from clear marks.
    - **Grey rows** are empty (`n/a`).
    - Notes at the top list unidentified, missing or uncertain pages.

    Correct the answers in the draft sheets (column `answer`; any
    spreadsheet editor, saved as TSV). Missing answers are `n/a`.

4. **Finalize.**

        python -m transcription.finalize 78 --reviewed

    It checks every answer (rows, options, numbers, dates `DD/MM/YYYY`,
    percentages) and writes the final sheets to
    `tdtb_admin/tdtb_private/forms/sub-78/`. Without `--reviewed` it
    refuses while rows are still marked `check`. Without draft names it
    finalizes only the drafts that have no final sheet yet; give draft
    names to finalize only those (see below).

5. **Screen for reversed scales.**

        python reversed_scales_screen.py

    Rates each post-session questionnaire for a reversed scale and adds a
    `to assess` row for each new session to
    `tdtb_private/postsess_suspected_reversed_scales.tsv`. After checking
    the paper, complete that row (status `verified on paper`, likelihood,
    evidence) and run the screen again: it adds the session to
    `REVERSED_SCALES` in `postses_results.py`, active if likely or very
    likely, commented out if possible, and warns if an entry already
    listed disagrees with the assessment. Then run `postses_results.py`.
    The sheets always keep the answers as written on the paper.

### Adding, correcting or redoing forms later

Forms do not have to be transcribed all at once: e.g. the forms of the
first session now, the post-session form of session 2 later. By default
`transcribe` only **adds** forms: it never replaces a draft or a final
sheet, and refuses a scanned PDF it has already transcribed (listed in
`scans_done.tsv`, in the drafts folder). `finalize` only writes sheets
that do not exist yet. Replacing anything needs `--overwrite`.

What to give each command (always followed by the review, step 3,
whenever `transcribe` runs):

<table border="1" cellpadding="6">
  <tr><th>You want to…</th><th><code>transcribe</code></th><th><code>finalize</code></th></tr>
  <tr><td>add a new form</td><td>the new scanned PDF</td><td>nothing extra</td></tr>
  <tr><td>fix an answer in a finished session</td><td>not needed</td><td>the draft, <code>--overwrite</code></td></tr>
  <tr><td>redo a session from a new scanned PDF</td><td>its scanned PDF, <code>--overwrite</code></td><td>the draft, <code>--overwrite</code></td></tr>
  <tr><td>redo everything</td><td>all scanned PDFs, <code>--overwrite</code></td><td><code>--overwrite</code></td></tr>
</table>

The commands, for sub-78 (session 1 as example):

**Add a new form** (e.g. the post-session form of session 2):

```
python -m transcription.transcribe 78 new_scanned_form.pdf
python -m transcription.finalize 78 --reviewed
```

**Fix an answer in a finished session:** correct it in the draft, then

```
python -m transcription.finalize 78 behav_postexp_sub-78_ses-01.draft.tsv --overwrite
```

**Redo a session from a new scanned PDF** (its file name must give the
session):

```
python -m transcription.transcribe 78 sub-78_ses-01.pdf --overwrite
python -m transcription.finalize 78 behav_postexp_sub-78_ses-01.draft.tsv --reviewed --overwrite
```

**Redo everything:**

```
python -m transcription.transcribe 78 ~/scans/sub-78/*.pdf --overwrite
python -m transcription.finalize 78 --reviewed --overwrite
```

Rules:

- A new post-session form takes the **next session without a sheet**
  (e.g. session 2 when session 1 is done), unless the file name of its
  scanned PDF gives the session (`ses-02`).
- To **redo only some sessions** with `--overwrite`, the file name of
  each scanned PDF must give its session (`sub-78_ses-01.pdf`); otherwise the
  run stops, as it could not tell which session the form belongs to.
- `transcribe --overwrite` replaces only the drafts of the forms in the
  scanned PDFs given, never the final sheets: those change only with
  `finalize --overwrite`.
- If a run would replace something without `--overwrite`, it stops
  before reading the forms and says why; nothing is written.

## What to know when reviewing

- **Post-session forms** are assigned to sessions in page order (across
  the PDFs, in the order given), starting from the first session without
  a sheet, unless the PDF file name holds a session label (`ses-02`).
  Behavioral sessions come before imaging sessions; the
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
- **Accuracy:** on the scans of 74 participants that already had curated
  sheets (October 2026, 8,658 answers), about 29% of the answers were
  marked `check`, and 0.4% were marked `ok` but differed from the sheet.
  Green rows are very reliable but not infallible: if one looks wrong in
  the review PDF, check it too.

## Files

| File | Role |
| ---- | ---- |
| `transcribe.py` | Step 2: scanned PDFs to drafts and review PDF |
| `finalize.py` | Step 4: checked drafts to final sheets |
| `layout.py` | Where every answer is on the blank forms |
| `align.py` | Identifies each scanned page and aligns it to the blank form |
| `read.py` | Reads circles, ticks and handwriting areas; thresholds at the top |
| `handwriting.py` | Calls the local handwriting model |
| `review.py` | Builds the review PDF |
| `sheets.py` | Rows of the sheets (wording must not change) |
| `synthetic.py` | For development: fake filled-in forms to test the code (usage in the file) |
| `validate.py` | For development: accuracy check after changing the code that reads the forms (usage in the file) |
| `../reversed_scales_screen.py` | Step 5: reversed-scale screen |
