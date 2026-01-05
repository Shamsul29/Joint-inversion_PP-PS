#!/usr/bin/env python3
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matplotlib.colors import ListedColormap, BoundaryNorm

# ================== CONFIG ==================

ROOT = r"/ddnB/work/shamsul/for_prestack_3D_inversion_25Dec_merged/Bakken_lamda-rho_mu-rho_YM_PR_files/"

INLINE_MIN = 127
INLINE_MAX = 790
N_INLINE   = INLINE_MAX - INLINE_MIN + 1

# Zone ?-bounds (same meaning as your crossplot ?-lines)
PR_MIN   = 0.224 #0.22
PR_SPLIT = 0.298 #0.30
PR_MAX   = 0.338 #0.34

# E targets used to FIT the E-lines
E_MIN    = 13.8 #15.0   # GPa
E_SPLIT  = 25.8 #25.0   # GPa
E_MAX    = 32.8 #33.0   # GPa

E_TOL_LIST = (4.0, 8.0, 10.0, 15.0)  #3.0 for 30Dec # auto-widen if not enough points for E-line fitting

# Rotation angle
ROT_ANGLE = -1.621  # degrees

FIGSIZE = (3.54, 3.0)
DPI     = 300
OUTFIG  = "Bakken_four_zone_map_25Dec_LINECONSISTENT_with_calibration_3Jan.jpeg"

# Zone colors (1..4)
ZONE_COLORS = [
    "red",         # 1 brittle rich
    "orange",      # 2 brittle poor
    "cyan", # 3 ductile poor
    "green",        # 4 ductile rich
]

ZONE_LABELS = {
    1: "Brittle rich",
    2: "Brittle poor",
    3: "Ductile poor",
    4: "Ductile rich",
}

# Filenames
LAM_FILE = "lambda_rho_bakken_window.csv"
MU_FILE  = "mu_rho_bakken_window.csv"
YM_FILE  = "young_modulus_bakken_window.csv"
PR_FILE  = "poissons_ratio_bakken_window.csv"

# ================== HELPERS ==================

def extract_xline_from_dirname(dname: str):
    m = re.search(r"(\d+)", dname)
    return int(m.group(1)) if m else None

def fit_E_line(lam_flat, mu_flat, E_flat, Et, tol_list=(3.0, 8.0, 10.0, 15.0), min_pts=50):
    """
    Fit µ = a  + b using points where E is near Et.
    Auto-widen tolerance until we get enough points.
    """
    for tol in tol_list:
        m = (E_flat >= Et - tol) & (E_flat <= Et + tol)
        n = int(np.sum(m))
        if n >= min_pts:
            a, b = np.polyfit(lam_flat[m], mu_flat[m], 1)
            return a, b, tol, n
    raise RuntimeError(f"Not enough points to fit E={Et} even with tol={tol_list[-1]}. "
                       f"Try increasing E_TOL_LIST or check YM units.")

# ================== LOAD DATA INTO 2D MAPS ==================

xl_dirs = sorted(glob.glob(os.path.join(ROOT, "XL*")))
if not xl_dirs:
    raise RuntimeError(f"No XL* folders found under {ROOT}.")

crosslines = []
for d in xl_dirs:
    base = os.path.basename(d)
    xl = extract_xline_from_dirname(base)
    if xl is not None:
        crosslines.append(xl)

if not crosslines:
    raise RuntimeError("No valid crossline numbers parsed from XL folders.")

crosslines, xl_dirs = zip(*sorted(zip(crosslines, xl_dirs)))
crosslines = list(crosslines)
xl_dirs    = list(xl_dirs)

n_xl = len(crosslines)
inlines = np.arange(INLINE_MIN, INLINE_MAX + 1)

print("Crosslines:", crosslines[0], "…", crosslines[-1], f"(count = {n_xl})")

# Maps: inline x crossline
LAM_map = np.full((N_INLINE, n_xl), np.nan, dtype=float)
MU_map  = np.full((N_INLINE, n_xl), np.nan, dtype=float)
YM_map  = np.full((N_INLINE, n_xl), np.nan, dtype=float)
PR_map  = np.full((N_INLINE, n_xl), np.nan, dtype=float)  # not used for zoning (kept for diagnostics)

for j, (xl, d) in enumerate(zip(crosslines, xl_dirs)):
    lam_path = os.path.join(d, LAM_FILE)
    mu_path  = os.path.join(d, MU_FILE)
    ym_path  = os.path.join(d, YM_FILE)
    pr_path  = os.path.join(d, PR_FILE)

    if not (os.path.exists(lam_path) and os.path.exists(mu_path) and os.path.exists(ym_path) and os.path.exists(pr_path)):
        print(f"[WARN] Missing one or more files in {d}, skipping XL{xl}")
        continue

    lam_arr = np.atleast_2d(np.loadtxt(lam_path, dtype=float, delimiter=","))
    mu_arr  = np.atleast_2d(np.loadtxt(mu_path,  dtype=float, delimiter=","))
    ym_arr  = np.atleast_2d(np.loadtxt(ym_path,  dtype=float, delimiter=","))
    pr_arr  = np.atleast_2d(np.loadtxt(pr_path,  dtype=float, delimiter=","))

    # Mean across available rows (your current behavior)
    #lam_vals = lam_arr[1:-1,:].mean(axis=0)
    #mu_vals  = mu_arr[1:-1,:].mean(axis=0)
    #ym_vals  = ym_arr[1:-1,:].mean(axis=0)
    #pr_vals  = pr_arr[1:-1,:].mean(axis=0)

    # indices of the two smallest mu values in each column
    idx2 = np.argpartition(mu_arr, kth=1, axis=0)[:2, :]   # (2, n_cols)
    
    # take same two row-indices from all arrays (per column)
    mu_win  = np.take_along_axis(mu_arr,  idx2, axis=0)    # (2, n_cols)
    lam_win = np.take_along_axis(lam_arr, idx2, axis=0)
    ym_win  = np.take_along_axis(ym_arr,  idx2, axis=0)
    pr_win  = np.take_along_axis(pr_arr,  idx2, axis=0)

    # reduce to one value per column (inline)
    mu_vals  = mu_win.mean(axis=0)
    lam_vals = lam_win.mean(axis=0)
    ym_vals  = ym_win.mean(axis=0)
    pr_vals  = pr_win.mean(axis=0)
    
    if lam_vals.shape[0] != N_INLINE:
        print(f"[WARN] Length mismatch in {d}: got {lam_vals.shape[0]}, expected {N_INLINE}")
        continue

    # Convert YM if needed:
    # Keep ONLY if YM is stored in Pa. If YM already in GPa, comment this out.
    ym_vals = ym_vals / 1e9

    LAM_map[:, j] = lam_vals
    MU_map[:, j]  = mu_vals
    YM_map[:, j]  = ym_vals
    PR_map[:, j]  = pr_vals

print("YM range:", np.nanmin(YM_map), "to", np.nanmax(YM_map))
print("PR range:", np.nanmin(PR_map), "to", np.nanmax(PR_map))
print("LAM range:", np.nanmin(LAM_map), "to", np.nanmax(LAM_map))
print("MU  range:", np.nanmin(MU_map),  "to", np.nanmax(MU_map))

# ================== LINE-CONSISTENT 4-ZONE CLASSIFICATION ==================
# This matches the previous crossplot code:
# - ? is computed from (??, µ?) so ?-zones align with ?-lines.
# - E zones are defined by which side of the FITTED E-lines (µ=a?+b) the point falls in.

zone_map = np.full(YM_map.shape, np.nan, dtype=float)

valid = np.isfinite(LAM_map) & np.isfinite(MU_map) & np.isfinite(YM_map)

# ? from ?? and µ?  (? cancels)
den = 2.0 * (LAM_map + MU_map)
NU_map = np.full_like(LAM_map, np.nan, dtype=float)
np.divide(LAM_map, den, out=NU_map, where=(den != 0.0))

valid_nu = valid & np.isfinite(NU_map)

# --- Fit E-lines globally in (??, µ?) using points near each target E ---
lam_flat = LAM_map[valid_nu].ravel()
mu_flat  = MU_map[valid_nu].ravel()
E_flat   = YM_map[valid_nu].ravel()

a13, b13, tol13, n13 = fit_E_line(lam_flat, mu_flat, E_flat, E_MIN,   tol_list=E_TOL_LIST)
a19, b19, tol19, n19 = fit_E_line(lam_flat, mu_flat, E_flat, E_SPLIT, tol_list=E_TOL_LIST)
a32, b32, tol32, n32 = fit_E_line(lam_flat, mu_flat, E_flat, E_MAX,   tol_list=E_TOL_LIST)

print(f"Fitted E={E_MIN}  line with tol={tol13} using n={n13} points: µ = {a13:.4f}? + {b13:.4f}")
print(f"Fitted E={E_SPLIT} line with tol={tol19} using n={n19} points: µ = {a19:.4f}? + {b19:.4f}")
print(f"Fitted E={E_MAX}  line with tol={tol32} using n={n32} points: µ = {a32:.4f}? + {b32:.4f}")

# Evaluate µ on the fitted E-lines for every gridpoint (same shape as maps)
MU_Emin   = a13 * LAM_map + b13
MU_Esplit = a19 * LAM_map + b19
MU_Emax   = a32 * LAM_map + b32

# “Between two lines” robustly (in case lines cross somewhere)
lo_13_19 = np.minimum(MU_Emin, MU_Esplit)
hi_13_19 = np.maximum(MU_Emin, MU_Esplit)

lo_19_32 = np.minimum(MU_Esplit, MU_Emax)
hi_19_32 = np.maximum(MU_Esplit, MU_Emax)

# ? bands (match ?-lines exactly)
brittle = valid_nu & (NU_map >= PR_MIN)   & (NU_map <  PR_SPLIT)
ductile = valid_nu & (NU_map >= PR_SPLIT) & (NU_map <= PR_MAX)

# E bands (match fitted E-lines)
E_low  = valid_nu & (MU_map >= lo_13_19) & (MU_map <  hi_13_19)   # between E_MIN and E_SPLIT
E_high = valid_nu & (MU_map >= lo_19_32) & (MU_map <= hi_19_32)   # between E_SPLIT and E_MAX

# Assign zones (same semantics as your mask naming)
zone_map[brittle & E_low]  = 1  # brittle rich
zone_map[brittle & E_high] = 2  # brittle poor
zone_map[ductile & E_high] = 3  # ductile poor
zone_map[ductile & E_low]  = 4  # ductile rich

from scipy.ndimage import generic_filter

def majority_filter_1d_xl(zone_map: np.ndarray, win_xl: int = 5) -> np.ndarray:
    """
    Majority filter across crossline direction ONLY (axis=1).
    Keeps zone codes {1,2,3,4}; preserves NaNs.
    win_xl must be odd (e.g., 3,5,7,9).
    """
    if win_xl % 2 == 0:
        raise ValueError("win_xl must be odd (3,5,7,...)")

    def majority(vals):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan
        # round to nearest int to be safe
        vals = vals.astype(int)
        counts = np.bincount(vals, minlength=5)  # indices 0..4 (0 unused)
        counts[0] = 0
        if counts.sum() == 0:
            return np.nan
        return np.argmax(counts)

    # footprint: 1 row × win_xl columns (only crossline smoothing)
    footprint = np.zeros((1, win_xl), dtype=bool)
    footprint[0, :] = True

    return generic_filter(
        zone_map,
        function=majority,
        footprint=footprint,
        mode="constant",
        cval=np.nan
    )

# --- apply smoothing across crossline ---
zone_map = majority_filter_1d_xl(zone_map, win_xl=5)  # try 3, 5, or 7


print("Zone counts (LINE-CONSISTENT):")
for code in (1, 2, 3, 4):
    cnt = int(np.sum(zone_map == code))
    print(f"  {code} ({ZONE_LABELS[code]}): {cnt}")
print("  outside:", int(np.sum(np.isnan(zone_map))))

# ================== PLOT 4-ZONE MAP (ROTATED) ==================

df_zone = pd.DataFrame(zone_map, index=inlines, columns=crosslines)
arr2 = df_zone.values

# Flip so inline 127 at bottom
arr_flipped = np.flipud(arr2)

# Rotate with nearest neighbor, NaN outside
rotated = rotate(
    arr_flipped,
    angle=ROT_ANGLE,
    reshape=True,
    order=0,
    mode='constant',
    cval=np.nan
)

# Colormap; NaN -> white
cmap = ListedColormap(ZONE_COLORS)
cmap.set_bad("black")

bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

extent = [crosslines[0], crosslines[-1], inlines[0], inlines[-1]]

im = ax.imshow(
    rotated,
    aspect='auto',
    cmap=cmap,
    norm=norm,
    extent=extent
)

cbar = plt.colorbar(im, ticks=[1, 2, 3, 4])
cbar.ax.set_yticklabels([
    "Brittle rich",
    "Brittle poor",
    "Ductile poor",
    "Ductile rich",
])
cbar.ax.tick_params(labelsize=7)

ax.set_xlabel("Crossline Number", fontsize=8)
ax.set_ylabel("Inline Number", fontsize=8)

ax.set_xticks(np.linspace(crosslines[0], crosslines[-1], 10, dtype=int))
ax.set_yticks(np.linspace(inlines[0], inlines[-1], 10, dtype=int))
ax.tick_params(axis='both', which='both', labelsize=7)

plt.tight_layout()
plt.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("Saved", OUTFIG)


