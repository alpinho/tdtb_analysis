#!/usr/bin/env python3
#
# Author: Ana Luisa Pinho
# Email: agrilopi@uwo.ca

# Creation: 8th of June 2026
# Last Update: June 2026

# Compatibility: Python 3.10.14
#
# =============================================================================
# Computation of the whole-brain peak table for the Random vs. Non-Random
# contrast, directly from the group Z-maps and the provided atlases.
#
# For every region/hemisphere the script computes:
#   1. peak coordinate, peak Z and two-sided BH-FDR q-value (q = 0.05),
#      the threshold being fdr_threshold(|z|, 0.05);
#   2. activation vs. deactivation, from the sign of Random vs. Rest and
#      Non-Random vs. Rest at the peak;
#   3. overlap with the predictive-timing network in EACH hemisphere: the
#      percentage of the region's Random > Non-Random voxels that also survive
#      the same-sign FDR-thresholded Encoding vs. Rest map (reported for left
#      and right separately, with voxel counts);
#   4. per-modality direction (Activated/Deactivated): the sign of the mean
#      task response 1/2[(Random - Rest) + (Non-Random - Rest)] over the
#      region's supra-threshold cluster, evaluated separately for the auditory
#      and visual runs (the cerebellar peak is read at the voxel rather than
#      the cluster mean, because the cerebellar atlas and the functional maps
#      are defined in different MNI variants).
# =============================================================================
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from scipy.stats import norm
from nilearn.image import resample_to_img
from nilearn.glm.thresholding import fdr_threshold

ALPHA = 0.05
INP = '.'                                       # directory holding the inputs

MAPS = {
    'diff'  : '17_random_vs_non-random_zmap.nii.gz',   # Random - Non-Random (pooled)
    'rnd'   : '9_random_zmap.nii.gz',                  # Random     vs Rest
    'nonrnd': '8_non-random_zmap.nii.gz',              # Non-Random vs Rest
    'enc'   : '1_encoding_zmap.nii.gz',                # Encoding   vs Rest
    'aud_r' : '21_auditory_random_zmap.nii.gz',        # Auditory Random     vs Rest
    'aud_n' : '20_auditory_non-random_zmap.nii.gz',    # Auditory Non-Random vs Rest
    'vis_r' : '33_visual_random_zmap.nii.gz',          # Visual   Random     vs Rest
    'vis_n' : '32_visual_non-random_zmap.nii.gz',      # Visual   Non-Random vs Rest
}
img = {k: nib.load(f'{INP}/{v}') for k, v in MAPS.items()}
dat = {k: img[k].get_fdata() for k in img}
ref = img['diff']; aff = ref.affine; inv = np.linalg.inv(aff)
brain = dat['diff'] != 0                         # shared non-zero support of the maps

# ---- two-sided BH-FDR thresholds on |z| (q = 0.05) --------------------------
THR = {k: float(fdr_threshold(np.abs(dat[k][brain]), ALPHA)) for k in ('diff', 'enc')}

# ---- per-voxel two-sided BH q-map for the pooled contrast -------------------
def bh_qmap(z, mask):
    p = 2 * norm.sf(np.abs(z[mask]))
    order = np.argsort(p); n = p.size
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    qs = np.empty(n); mn = 1.0
    for idx in order[::-1]:
        mn = min(mn, q[idx]); qs[idx] = mn
    out = np.full(z.shape, np.nan); out[mask] = qs
    return out
QDIFF = bh_qmap(dat['diff'], brain)

# ---- atlases resampled to the Z-map grid (nearest neighbour) ----------------
def load_lbl(p):
    return resample_to_img(nib.load(p), ref, interpolation='nearest',
                           force_resample=True, copy_header=True).get_fdata().astype(int)
CORT = load_lbl(f'{INP}/harvardoxford-cortical_label_all.nii.gz')   # FSL HO cortical 1..48
HMAT = load_lbl(f'{INP}/hmat/HMAT_website/HMAT.nii')                # 1..12 (HMAT key)
CERE = load_lbl(f'{INP}/atl-NettekovenSym128_space-MNI152NLin2009cSymC_dseg.nii')
HO = dict(Insula=2, MFG=4, STGant=9, STGpost=10, MTGto=13, SMGpost=20, FOC=33, PP=44, PT=46)
HK = dict(PMd_L=10, PMv_R=11)
CK = dict(D1Li=30)

# ---- hemisphere split by world-x; supra-threshold positive clusters ---------
ii, jj, kk = np.indices(ref.shape)
WX = aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 2] * kk + aff[0, 3]
LEFT, RIGHT = WX < 0, WX > 0
POS = (dat['diff'] >= THR['diff']) & brain                         # Random > Non-Random voxels
CC, _ = ndi.label(POS, structure=np.ones((3, 3, 3)))               # 26-connectivity

def vox(c):
    v = inv @ np.array([c[0], c[1], c[2], 1.0]); return tuple(int(round(x)) for x in v[:3])
def world(p):
    v = aff @ np.array([p[0], p[1], p[2], 1.0]); return tuple(int(round(x)) for x in v[:3])
def comp_at(coord):
    p = vox(coord)
    if POS[p]: return CC[p]
    best, bd = None, 1e9                                            # nearest supra voxel (<=4 vox)
    for di in range(-4, 5):
        for dj in range(-4, 5):
            for dk in range(-4, 5):
                a, b, c = p[0] + di, p[1] + dj, p[2] + dk
                if POS[a, b, c] and di * di + dj * dj + dk * dk < bd:
                    bd = di * di + dj * dj + dk * dk; best = CC[a, b, c]
    return best

def overlap_in(region, hemimask, side):
    """% of the region's Random>Non-Random voxels in one hemisphere that also
    survive the same-sign FDR-thresholded Encoding vs. Rest map."""
    enc = (dat['enc'] <= -THR['enc']) if side == 'deactivation' else (dat['enc'] >= THR['enc'])
    V = region & hemimask & POS
    n = int(V.sum())
    if n == 0:
        return float('nan'), 0
    return 100.0 * float((V & enc).sum()) / n, n

def evaluate(region, hemi, anchor, peak='anchor', section='activation', mod_at_peak=False):
    hemimask = LEFT if hemi == 'L' else RIGHT
    comp = comp_at(anchor)
    Vpk = region & hemimask & POS
    if comp is not None:
        Vpk = Vpk & (CC == comp)                                   # peak cluster (for peak/modality)
    if int(Vpk.sum()) == 0:
        return None
    pk = (np.unravel_index(np.argmax(np.where(Vpk, dat['diff'], -np.inf)), Vpk.shape)
          if peak == 'region' else vox(anchor))
    zr, zn = dat['rnd'][pk], dat['nonrnd'][pk]
    kind = ('activation' if (zr > 0 and zn > 0) else
            'deactivation' if (zr < 0 and zn < 0) else 'mixed')
    ovL, nL = overlap_in(region, LEFT,  section)
    ovR, nR = overlap_in(region, RIGHT, section)
    # per-modality direction: mean task response over the cluster (peak voxel
    # for the cerebellum, where atlas and functional maps differ in MNI variant)
    if mod_at_peak:
        aud = 0.5 * (dat['aud_r'][pk] + dat['aud_n'][pk])
        vis = 0.5 * (dat['vis_r'][pk] + dat['vis_n'][pk])
    else:
        aud = 0.5 * (dat['aud_r'][Vpk] + dat['aud_n'][Vpk]).mean()
        vis = 0.5 * (dat['vis_r'][Vpk] + dat['vis_n'][Vpk]).mean()
    return dict(coord=world(pk), Z=float(dat['diff'][pk]), q=float(QDIFF[pk]),
                kind=kind, ovL=ovL, ovR=ovR, nL=nL, nR=nR,
                ov_peak=(ovL if hemi == 'L' else ovR),
                aud='Activated' if aud > 0 else 'Deactivated',
                vis='Activated' if vis > 0 else 'Deactivated')

# (label, region mask, peak hemisphere, anchor coordinate, section, peak-mode, cerebellar flag)
ROWS = [
    # --- Activations (Random > Non-Random >= Rest) ---
    ('SMG, posterior',     CORT == HO['SMGpost'], 'R', (68, -34, 18),  'activation',   'anchor', False),
    ('LPMC / PMV',         HMAT == HK['PMv_R'],   'R', (38, 8, 30),    'activation',   'anchor', False),
    ('Cerebellum / D1Li',  CERE == CK['D1Li'],    'L', (-27, -67, -50),'activation',   'anchor', True),
    ('STG, posterior',     CORT == HO['STGpost'], 'R', (53, -20, -2),  'activation',   'anchor', False),
    ('Insula, anterior',   CORT == HO['Insula'],  'R', (30, 26, 0),    'activation',   'anchor', False),
    ('PT',                 CORT == HO['PT'],      'L', (-50, -42, 18), 'activation',   'anchor', False),
    ('SMG, posterior',     CORT == HO['SMGpost'], 'L', (-60, -42, 15), 'activation',   'anchor', False),
    ('LPMC / PMD',         HMAT == HK['PMd_L'],   'L', (-37, -4, 45),  'activation',   'anchor', False),
    # --- Deactivations (Rest >= Random > Non-Random) ---
    ('MFG',                CORT == HO['MFG'],     'R', (38, 23, 22),   'deactivation', 'anchor', False),
    ('STG, anterior',      CORT == HO['STGant'],  'R', (53, 3, -12),   'deactivation', 'anchor', False),
    ('FOC',                CORT == HO['FOC'],     'L', (-34, 26, 0),   'deactivation', 'anchor', False),
    ('MTG, temporooccip.', CORT == HO['MTGto'],   'R', (66, -52, 10),  'deactivation', 'anchor', False),
    ('PP',                 CORT == HO['PP'],      'L', (-47, -10, -8), 'deactivation', 'region', False),
]

if __name__ == '__main__':
    print('Two-sided BH-FDR thresholds (|z|, q=0.05): '
          f"diff = {THR['diff']:.3f},  enc = {THR['enc']:.3f}\n")
    head = (f"{'region':20s}{'hemi':5s}{'coord':>15s}{'Z':>6s}{'qFDR':>10s}"
            f"{'class':>13s}{'ov%(pk)':>8s}{'ovL%':>6s}{'ovR%':>6s}"
            f"{'nL':>6s}{'nR':>6s}{'Aud':>12s}{'Vis':>12s}")
    print(head); print('-' * len(head))
    for lab, m, h, a, sec, pk, cb in ROWS:
        r = evaluate(m, h, a, peak=pk, section=sec, mod_at_peak=cb)
        if r is None:
            print(f'{lab:20s}{h:5s}  (no supra-threshold voxel)'); continue
        fmt = lambda x: '   .' if (isinstance(x, float) and np.isnan(x)) else f'{x:6.0f}'
        print(f"{lab:20s}{h:5s}{str(r['coord']):>15s}{r['Z']:6.2f}{r['q']:10.2e}"
              f"{r['kind']:>13s}{fmt(r['ov_peak'])}{fmt(r['ovL'])}{fmt(r['ovR'])}"
              f"{r['nL']:6d}{r['nR']:6d}{r['aud']:>12s}{r['vis']:>12s}")