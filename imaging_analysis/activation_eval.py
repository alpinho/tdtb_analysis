# =============================================================================
# Supplementary Table E.3  --  Random vs. Non-Random whole-brain peaks
#
# Recomputes the table directly from the group z-maps and atlases, with the
# peak list REGENERATED from the maps (no hard-coded coordinates). For each
# peak the script reports four things, each documented at its function below:
#
#   1. PEAK DETECTION  -> detect_peaks(): local maxima of the pooled
#      Random - Non-Random map within the FDR-thresholded positive set,
#      thinned to one peak per 10 mm, labelled by atlas, one peak kept per
#      (region, hemisphere).
#   2. Z and qFDR at the peak (two-sided BH-FDR on the pooled contrast).
#   3. OVERLAP  -> overlap(): % of the region's Random>Non-Random voxels (per
#      hemisphere) that also survive the same-sign FDR-thresholded Encoding map.
#   4. MODALITY  -> modality(): sign of the cluster MEDIAN auditory / visual
#      task response, thresholded at each modality's own FDR level (else Rest).
#
# Atlas spaces: the functional maps and the Harvard-Oxford / HMAT atlases are
# all in FSL MNI152 (NLin6); the Nettekoven cerebellar atlas is in 2009cSymC,
# a few-mm mismatch, so the cerebellar modality is read at the peak voxel.
# =============================================================================
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from scipy.stats import norm
from nilearn.image import resample_to_img
from nilearn.glm.thresholding import fdr_threshold

ALPHA   = 0.05
MIN_SEP = 10.0        # mm; minimum separation between reported peaks
NEAR_R  = 2           # voxels; radius for nearest-label fallback on unlabelled peaks
INP     = '.'         # directory holding the inputs

# restrict the printed table to the manuscript regions (True) or show every
# detected cortical / premotor / cerebellar peak (False)
MANUSCRIPT_ONLY = False

# report only peaks whose PEAK VOXEL lies OUTSIDE the two-sided Encoding-vs-Rest
# network (|z_enc| < THR_enc) -- i.e. clusters outside the predictive timing
# network, where unpredictability recruits regions the timing task itself does
# not engage. Set False to print all peaks (the encoding z is shown either way).
OUTSIDE_NETWORK_ONLY = True

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
dat = {k: nib.load(f'{INP}/{v}').get_fdata() for k, v in MAPS.items()}
ref = nib.load(f"{INP}/{MAPS['diff']}"); aff = ref.affine; inv = np.linalg.inv(aff)
brain = dat['diff'] != 0                              # shared non-zero support

# ---- two-sided BH-FDR thresholds on |z| (q = 0.05) --------------------------
# diff/enc: thresholds for the contrast and the encoding overlap map.
# aud/vis : thresholds of the per-modality task-response map (the Rest cutoff).
THR = {k: float(fdr_threshold(np.abs(dat[k][brain]), ALPHA)) for k in ('diff', 'enc')}
AUD = 0.5 * (dat['aud_r'] + dat['aud_n'])
VIS = 0.5 * (dat['vis_r'] + dat['vis_n'])
THR['aud'] = float(fdr_threshold(np.abs(AUD[brain]), ALPHA))
THR['vis'] = float(fdr_threshold(np.abs(VIS[brain]), ALPHA))

# ---- per-voxel two-sided BH q-map for the pooled contrast (qFDR column) ------
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

# ---- atlases resampled to the z-map grid (nearest neighbour) ----------------
def load_lbl(p):
    return resample_to_img(nib.load(p), ref, interpolation='nearest',
                           force_resample=True, copy_header=True).get_fdata().astype(int)
CORT = load_lbl(f'{INP}/harvardoxford-cortical_label_all.nii.gz')   # FSL HO cortical 1..48
HMAT = load_lbl(f'{INP}/hmat/HMAT_website/HMAT.nii')                # premotor (HMAT key)
CERE = load_lbl(f'{INP}/atl-NettekovenSym128_space-MNI152NLin2009cSymC_dseg.nii')

HOname = {1:'Frontal Pole',2:'Insula, anterior',3:'SFG',4:'MFG',5:'IFG, pars tri.',
    6:'IFG, pars oper.',7:'Precentral',8:'Temporal Pole',9:'STG, anterior',
    10:'STG, posterior',11:'MTG, anterior',12:'MTG, posterior',13:'MTG, temporooccip.',
    14:'ITG, anterior',15:'ITG, posterior',16:'ITG, temporooccip.',17:'Postcentral',
    18:'SPL',19:'SMG, anterior',20:'SMG, posterior',21:'Angular',22:'LOC, superior',
    23:'LOC, inferior',24:'Intracalcarine',25:'Frontal Medial',26:'JLC / SMA',
    27:'Subcallosal',28:'Paracingulate',29:'Cingulate, anterior',30:'Cingulate, posterior',
    31:'Precuneus',32:'Cuneal',33:'FOC',34:'PHG, anterior',35:'PHG, posterior',36:'Lingual',
    37:'TFusC, anterior',38:'TFusC, posterior',39:'TOFusC',40:'Occipital Fusiform',
    41:'Frontal Operculum',42:'Central Operculum',43:'Parietal Operculum',44:'PP',
    45:"Heschl's Gyrus",46:'PT',47:'Supracalcarine',48:'Occipital Pole'}
HMname = {9:'LPMC / PMD', 10:'LPMC / PMD', 11:'LPMC / PMV', 12:'LPMC / PMV'}

# ---- hemisphere split; supra-threshold positive (Random>Non-Random) set -----
ii, jj, kk = np.indices(ref.shape)
WX = aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 2] * kk + aff[0, 3]
LEFT, RIGHT = WX < 0, WX > 0
POS = (dat['diff'] >= THR['diff']) & brain
CC, _ = ndi.label(POS, structure=np.ones((3, 3, 3)))   # 26-connected clusters

def world(p):
    v = aff @ np.array([p[0], p[1], p[2], 1.0]); return tuple(int(round(x)) for x in v[:3])

def nearest_label(p, r=NEAR_R):
    """Nearest non-zero Harvard-Oxford label within r voxels. The hard HO dseg
    leaves some lateral perisylvian voxels unlabelled (0); a small-radius
    nearest-label fill recovers them the way a probabilistic-HO assignment would,
    while deep voxels (no cortical label within r) stay unlabelled and are dropped."""
    best, bd = 0, 1e9
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                a, b, c = p[0] + di, p[1] + dj, p[2] + dk
                if (0 <= a < CORT.shape[0] and 0 <= b < CORT.shape[1] and 0 <= c < CORT.shape[2]
                        and CORT[a, b, c] > 0):
                    d = di * di + dj * dj + dk * dk
                    if d < bd: bd, best = d, CORT[a, b, c]
    return best

# =============================================================================
# 1. PEAK DETECTION
# -----------------------------------------------------------------------------
# - POS is the set of voxels where the pooled Random - Non-Random map exceeds
#   the two-sided FDR threshold (i.e. significant Random > Non-Random).
# - Local maxima are voxels equal to the maximum of the map in their 3x3x3
#   neighbourhood (maximum_filter size=3) and inside POS.
# - Candidates are sorted by Z and thinned greedily: a peak is kept only if it
#   is at least MIN_SEP mm from every higher peak already kept.
# - Each kept peak is labelled, in priority order, by HMAT premotor parcel,
#   then the Nettekoven cerebellar mask, then the Harvard-Oxford parcel (with a
#   nearest-label fallback for unlabelled perisylvian voxels). Peaks with no
#   cortical / premotor / cerebellar label (deep subcortical) are dropped.
# - Of several peaks sharing a (region, hemisphere) only the highest is kept,
#   matching the one-peak-per-region-per-hemisphere convention of the table.
# - The activation / deactivation section follows the table's definition from
#   the sign relative to rest: activation when Non-Random >= Rest at the peak
#   (Random > Non-Random >= Rest), deactivation otherwise (Rest >= Non-Random).
# =============================================================================
def region_of(p):
    if HMAT[p] in HMname:                       # premotor first (HMAT overrides HO)
        return HMname[HMAT[p]], (HMAT == HMAT[p])
    if CERE[p] > 0:                             # whole cerebellar gray matter
        return 'Cerebellum / D1Li', (CERE > 0)
    lab = CORT[p] if CORT[p] > 0 else nearest_label(p)
    if lab in HOname:
        return HOname[lab], (CORT == lab)
    return None, None                           # deep subcortical -> dropped

def detect_peaks():
    mx = ndi.maximum_filter(dat['diff'], size=3)
    cand = sorted([tuple(p) for p in np.argwhere((dat['diff'] == mx) & POS)],
                  key=lambda p: -dat['diff'][p])
    kept = []
    for p in cand:
        w = np.array(world(p))
        if all(np.linalg.norm(w - np.array(world(q))) >= MIN_SEP for q in kept):
            kept.append(p)
    best = {}                                   # (region, hemi) -> (peak, mask)
    for p in kept:
        name, mask = region_of(p)
        if name is None: continue
        h = 'L' if LEFT[p] else 'R'
        if (name, h) not in best or dat['diff'][p] > dat['diff'][best[(name, h)][0]]:
            best[(name, h)] = (p, mask)
    peaks = []
    for (name, h), (p, mask) in best.items():
        section = 'activation' if dat['nonrnd'][p] >= 0 else 'deactivation'
        peaks.append(dict(name=name, hemi=h, vox=p, mask=mask,
                          coord=world(p), Z=float(dat['diff'][p]), q=float(QDIFF[p]),
                          section=section, cb=name.startswith('Cerebellum'),
                          enc=float(dat['enc'][p]),
                          in_network=bool(abs(dat['enc'][p]) >= THR['enc'])))
    return peaks

# =============================================================================
# 2. OVERLAP
# -----------------------------------------------------------------------------
# Per hemisphere, the percentage of the region's Random > Non-Random voxels
# (region mask AND hemisphere AND POS) that also fall in the same-sign
# FDR-thresholded Encoding vs. Rest map: Encoding > Rest for activation rows,
# Rest > Encoding for deactivation rows. Computed on the region mask (not the
# connected cluster) so adjacent structures in the same cluster are excluded;
# reported for left and right separately.
# =============================================================================
def overlap(mask, hemimask, section):
    encmask = (dat['enc'] <= -THR['enc']) if section == 'deactivation' else (dat['enc'] >= THR['enc'])
    V = mask & hemimask & POS
    n = int(V.sum())
    if n == 0:
        return float('nan'), 0
    return 100.0 * float((V & encmask).sum()) / n, n

# =============================================================================
# 3. MODALITY  (auditory / visual z-summaries)
# -----------------------------------------------------------------------------
# For each modality the per-voxel task response is 1/2[(Random-Rest)+(Non-Random
# -Rest)] = the mean of that modality's two condition-vs-rest z-maps. The cell
# value is the MEDIAN of that response over the peak cluster, taken in the PEAK
# hemisphere only (region mask AND peak hemisphere AND POS AND the peak's
# connected component). The cerebellar value is read at the single peak voxel,
# because the cerebellar atlas and the functional maps differ in MNI variant.
# The label is Activated / Deactivated when |median| exceeds that modality's own
# two-sided FDR threshold (THR['aud'] / THR['vis']), otherwise Rest -- i.e. the
# cluster carries the Random>Non-Random difference without a net modality response.
# =============================================================================
def modality(peak):
    p, mask, h = peak['vox'], peak['mask'], peak['hemi']
    if peak['cb']:
        aud = float(0.5 * (dat['aud_r'][p] + dat['aud_n'][p]))
        vis = float(0.5 * (dat['vis_r'][p] + dat['vis_n'][p]))
    else:
        V = mask & (LEFT if h == 'L' else RIGHT) & POS & (CC == CC[p])
        aud = float(np.median(0.5 * (dat['aud_r'][V] + dat['aud_n'][V])))
        vis = float(np.median(0.5 * (dat['vis_r'][V] + dat['vis_n'][V])))
    lab = lambda x, t: 'Activated' if x >= t else 'Deactivated' if x <= -t else 'Rest'
    return lab(aud, THR['aud']), aud, lab(vis, THR['vis']), vis

# ---- manuscript peaks (for marking matches only; not used in computation) ---
MANUSCRIPT = {(68,-34,18),(38,8,30),(-27,-67,-50),(53,-20,-2),(30,26,0),(-50,-42,18),
              (-60,-42,15),(-37,-4,45),(38,23,22),(53,3,-12),(-34,26,0),(66,-52,10),(-44,-10,-10)}

if __name__ == '__main__':
    print('Two-sided BH-FDR thresholds (|z|, q=0.05): '
          f"diff={THR['diff']:.3f}  enc={THR['enc']:.3f}  "
          f"aud={THR['aud']:.3f}  vis={THR['vis']:.3f}     (* = within 6 mm of a manuscript peak)\n")
    peaks = detect_peaks()
    peaks.sort(key=lambda r: (r['section'] != 'activation', -r['Z']))
    head = (f"{'':2s}{'Z':>5s}{'qFDR':>10s}{'z_enc':>7s}  {'region':20s}{'hemi':5s}"
            f"{'coord':>16s}{'sect':>7s}{'ovL%':>6s}{'ovR%':>6s}{'Auditory':>16s}{'Visual':>16s}")
    print(head); print('-' * len(head))
    for pk in peaks:
        if MANUSCRIPT_ONLY and not any(np.linalg.norm(np.array(pk['coord']) - np.array(m)) <= 6
                                       for m in MANUSCRIPT):
            continue
        if OUTSIDE_NETWORK_ONLY and pk['in_network']:
            continue
        oL, _ = overlap(pk['mask'], LEFT,  pk['section'])
        oR, _ = overlap(pk['mask'], RIGHT, pk['section'])
        al, az, vl, vz = modality(pk)
        star = '*' if any(np.linalg.norm(np.array(pk['coord']) - np.array(m)) <= 6
                          for m in MANUSCRIPT) else ' '
        f = lambda x: '   . ' if np.isnan(x) else f'{x:5.0f}'
        a = f'{al} ({az:+.1f})'; v = f'{vl} ({vz:+.1f})'
        print(f"{star:2s}{pk['Z']:5.2f}{pk['q']:10.2e}{pk['enc']:7.2f}  {pk['name']:20s}{pk['hemi']:5s}"
              f"{str(pk['coord']):>16s}{pk['section'][:5]:>7s}{f(oL)}{f(oR)}{a:>16s}{v:>16s}")