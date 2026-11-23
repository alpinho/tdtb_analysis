"""
Where the private folders of the project are on this computer.

Set ADMIN_DIR below to the path of the tdtb_admin folder on your
computer (~ is your home folder). Everything inside tdtb_admin is fixed
and given relative to it (tdtb_private/forms, tdtb_private/forms_drafts,
...; see transcription/README.md, Setup).

Run this file to check the path:
    python admin_paths.py

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: November 2026
"""

import os

# ======================= SET YOUR PATH HERE ===========================
ADMIN_DIR = '~/OneDrive/tdtb_admin'
# ======================================================================

ADMIN_DIR = os.path.expanduser(ADMIN_DIR)
PRIVATE_DIR = os.path.join(ADMIN_DIR, 'tdtb_private')

if not os.path.isdir(PRIVATE_DIR):
    raise FileNotFoundError(
        'no tdtb_private folder in %s: set ADMIN_DIR in %s to the path of '
        'tdtb_admin on this computer' % (ADMIN_DIR, os.path.abspath(__file__)))

if __name__ == '__main__':
    print('tdtb_admin: %s' % ADMIN_DIR)
