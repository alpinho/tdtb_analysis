"""
Rows of the curated questionnaire sheets, as read by the questionnaire
scripts (demographic_data_extraction.py, parse_score_goldmsi.py and
postses_results.py).

Each sheet is a 2-column TSV (question, answer) with exactly these rows,
in this order. The question wording, typos included, must not change:
the scripts look the rows up by their text.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.12
"""

# Missing values follow the BIDS convention
MISSING = 'n/a'

DEMOGRAPHIC_ROWS = [
    'Subject #',
    'Date',
    'Are you (circle one):',
    'Date of Birth (DD/MM/YYYY)',
    'Age:',
    'Which hand do you write with (circle one):',
    'What level did you attain in school (please check one):',
    'What is your first language?',
    'What other languages do you know?',
    'Please also rank the degree of fluency.',
    'Do you consider yourself bilingual?',
    'What do you consider your dominant/main language:',
    'What percentage of the time do you speak your main langauge(s)?',
    'How would you describe you musical skills/experiences (please '
    'circle one number)?',
    'Have you ever played a musical instrument?',
    'If yes, which instrument(s)?',
    'For how many years did/have you played?',
    'What type of training did you receive?',
    'Are you currently practicing music?',
    'If yes, how many hours per week do you practice?',
    'How important is music to your identity?',
    'Do you wear a hearing aid?',
    'Do you have ringing in your ears?',
    'If yes, which ear(s)?',
    'How would you describe your general hearing abilities (please '
    'circle one number)?',
    'When you talk with someone at a place that strongly '
    'reverberates/echoes (e.g., in a church or train station), can you '
    'understand what the person says?',
    "When you are with a group (~5 people) in a lively restaurant, can you "
    "follow the group's conversation?",
    'Based on the sound of a bus or truck, can you tell whether it is '
    'moving towards or away from you?',
    'When you are in an unknown environment, can you tell from which '
    'direction a brief sound originates?',
    'Are you able to ignore distracting sound when you concentrate on a '
    'specific aspect of your acoustic surrounding?',
    'Does difficulty with your hearing ever upset you?',
    'Have you ever been diagnosed with a neurological disorder (e.g., '
    'epilepsy)?',
    'Have you ever been diagnosed with ADHD?',
    'Have you ever been diagnosed with a psychological illness (e.g., '
    'phobia, anxiety)?',
    'Do you take any medications regularly?',
    'Have you ever had a serious head injury (concussion, unconscious, '
    'etc.)?',
]

GOLD_MSI_ROWS = (
    ['Subject #', 'Date']
    + ['question_%02d' % i for i in range(1, 32)]
    + [
        'I engaged in regular, daily practice of a musical instrument '
        '(including voice) for _ years.',
        'At the peak of my interest, I practiced _ hours per day on my '
        'primary instrument.',
        'I have attended _ live music events as an audience member in the '
        'past twelve months.',
        'I have had formal training in music theory for _ years.',
        'I have had _ years of formal training on a musical instrument '
        '(including voice) during my lifetime.',
        'I can play _ musical instruments.',
        'I listen attentively to music for _ per day.',
        'The instrument I play best (including voice) is _.',
        'Occupational status',
        'What is the musical genre you mainly listen to?',
        'What is the highest educational qualification you have attained?',
        'If you are still in education, what is the highest qualification '
        'you expect to obtain?',
        'Your age',
        'Gender',
        'Nationality',
        'Country in which you spent the formative years of your childhood '
        'and youth:',
        'Country of current residency:',
        'Do not contact me again',
    ])

POSTEXP_ROWS = [
    'Subject #',
    'Date',
    'How well did you understand the task?',
    'How difficult did you find the task?',
    'How strongly did you concentrate on the task?',
    'Did your concentration on the task change throughout the experiment?',
    'If your concentration changed, in what direction did it change?',
    'Were there occasions where you guessed when responding?',
    'How motivated were you for the experiment?',
    'Did you use/develop any specific strategies during the experiment to '
    'solve the task?',
    'If yes, please describe briefly:',
    'Do you have any additional comments regarding the experiment?',
]

# Sheet file names, per form; %02d is the subject, %s the session
SHEET_FNAMES = {
    'demographic': 'demographic_info_sub-%02d.tsv',
    'gold_msi': 'gold_msi_sub-%02d.tsv',
    'behav_postexp': 'behav_postexp_sub-%02d_ses-%s.tsv',
    'mri_postexp': 'mri_postexp_sub-%02d_ses-%s.tsv',
}

SHEET_ROWS = {'demographic': DEMOGRAPHIC_ROWS, 'gold_msi': GOLD_MSI_ROWS,
              'behav_postexp': POSTEXP_ROWS, 'mri_postexp': POSTEXP_ROWS}
