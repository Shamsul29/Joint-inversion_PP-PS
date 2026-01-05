#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================

ROOT = Path(r"/ddnB/work/shamsul/for_prestack_3D_inversion_25Dec_merged/Bakken_lamda-rho_mu-rho_YM_PR_files")
CROSSLINES    = range(369, 851)
CL_FOLDER_FMT = "XL{cl}"

LAM_FILE = "lambda_rho_bakken_window.csv"
MU_FILE  = "mu_rho_bakken_window.csv"
YM_FILE  = "young_modulus_bakken_window.csv"

NU_VALUES = [0.224, 0.298, 0.338] #[0.22, 0.30, 0.34] for 30Dec#[0.16, 0.27, 0.32] #[0.23, 0.31, 0.34]   # PR_MIN, PR_SPLIT, PR_MAX
E_TARGETS = [13.8, 25.8, 32.8] #[15.0, 25.0, 33.0] for 30Dec#[13.0, 19.5, 32.0]    # E_MIN, E_SPLIT, E_MAX
E_TOL     = 4.0                   # for fitting the E-lines

FIGSIZE = (3.0, 3.0)
DPI     = 300
OUTFIG  = "Bakken_lambda_mu_crossplot_4zones_by_lines_3Jan_with_calibration.png"

DRAW_LINES = True   # draw ν lines + E lines (no labels)

# ================== LOAD λρ, μρ, YM ==================

lam_list, mu_list, E_list = [], [], []

for cl in CROSSLINES:
    cl_dir = ROOT / CL_FOLDER_FMT.format(cl=cl)
    lam_path = cl_dir / LAM_FILE
    mu_path  = cl_dir / MU_FILE
    ym_path  = cl_dir / YM_FILE

    if not (lam_path.is_file() and mu_path.is_file() and ym_path.is_file()):
        continue

    lam_vals = np.atleast_2d(np.loadtxt(lam_path, dtype=float, delimiter=","))
    mu_vals  = np.atleast_2d(np.loadtxt(mu_path,  dtype=float, delimiter=","))
    E_vals   = np.atleast_2d(np.loadtxt(ym_path,  dtype=float, delimiter=","))

    # Keep ONLY if YM is in Pa. If YM already in GPa, comment this out.
    E_vals = E_vals / 1e9

    #lam_mean = np.nanmean(lam_vals[1:-1,:], axis=0)
    #mu_mean  = np.nanmean(mu_vals[1:-1,:],  axis=0)
    #E_mean   = np.nanmean(E_vals[1:-1,:],   axis=0)

    # indices of the two smallest mu values in each column
    idx2 = np.argpartition(mu_vals, kth=1, axis=0)[:2, :]   # (2, n_cols)
    
    # take same two row-indices from all arrays (per column)
    mu_win  = np.take_along_axis(mu_vals,  idx2, axis=0)    # (2, n_cols)
    lam_win = np.take_along_axis(lam_vals, idx2, axis=0)
    E_win  = np.take_along_axis(E_vals,  idx2, axis=0)
    #pr_win  = np.take_along_axis(pr_arr,  idx2, axis=0)

    # reduce to one value per column (inline)
    mu_mean  = mu_win.mean(axis=0)
    lam_mean = lam_win.mean(axis=0)
    E_mean  = E_win.mean(axis=0)
    #pr_vals  = pr_win.mean(axis=0)
    
    lam_list.append(lam_mean.ravel())
    mu_list.append(mu_mean.ravel())
    E_list.append(E_mean.ravel())

if not lam_list:
    raise RuntimeError("No valid data found. Check paths/filenames.")

lam = np.concatenate(lam_list)
mu  = np.concatenate(mu_list)
E   = np.concatenate(E_list)

# finite only
m0 = np.isfinite(lam) & np.isfinite(mu) & np.isfinite(E)
lam, mu, E = lam[m0], mu[m0], E[m0]

# ================== ν FROM (λρ, μρ) so it matches ν-lines ==================
# ν = λ / [2(λ+μ)]
den = 2.0 * (lam + mu)
nu = np.full_like(lam, np.nan, dtype=float)
np.divide(lam, den, out=nu, where=(den != 0.0))

m1 = np.isfinite(nu)
lam, mu, E, nu = lam[m1], mu[m1], E[m1], nu[m1]

# ================== FIT E-LINES (μ = a λ + b) for each target ==================

Efits = {}  # Et -> (a,b)
for Et in E_TARGETS:
    mE = (E >= Et - E_TOL) & (E <= Et + E_TOL)
    if np.sum(mE) < 10:
        raise RuntimeError(f"Not enough points to fit E={Et}±{E_TOL}. Found {np.sum(mE)}.")
    a, b = np.polyfit(lam[mE], mu[mE], 1)
    Efits[Et] = (a, b)

def mu_on_line(Et, lam_arr):
    a, b = Efits[Et]
    return a * lam_arr + b

mu13 = mu_on_line(13.8, lam)
mu19 = mu_on_line(25.8, lam)
mu31 = mu_on_line(32.8, lam)

# “Between two lines” robustly, even if they cross:
band_13_19_lo = np.minimum(mu13, mu19)
band_13_19_hi = np.maximum(mu13, mu19)

band_19_31_lo = np.minimum(mu19, mu31)
band_19_31_hi = np.maximum(mu19, mu31)

# ================== BUILD 4 ZONES FROM LINES ==================
PR_MIN, PR_SPLIT, PR_MAX = NU_VALUES

# ν-band membership (between ν-lines)
brittle = (nu >= PR_MIN) & (nu <  PR_SPLIT)   # between 0.24 and 0.295
ductile = (nu >= PR_SPLIT) & (nu <= PR_MAX)   # between 0.295 and 0.33

# E-band membership (between fitted E-lines)
E_low  = (mu >= band_13_19_lo) & (mu <  band_13_19_hi)  # between E=13 and E=19
E_high = (mu >= band_19_31_lo) & (mu <= band_19_31_hi)  # between E=19 and E=31

zone = np.zeros(lam.shape, dtype=np.uint8)  # 0 = outside (black)

# 1) brittle rich: PR_MIN ≤ PR < PR_SPLIT AND E_MIN ≤ YM < E_SPLIT  -> between ν(0.24,0.295) and between E(13,19)
zone[brittle & E_low]  = 1

# 2) brittle poor: PR_MIN ≤ PR < PR_SPLIT AND E_SPLIT ≤ YM ≤ E_MAX  -> between ν(0.24,0.295) and between E(19,31)
zone[brittle & E_high] = 2

# 3) ductile poor: PR_SPLIT ≤ PR ≤ PR_MAX AND E_SPLIT ≤ YM ≤ E_MAX
zone[ductile & E_high] = 3

# 4) ductile rich: PR_SPLIT ≤ PR ≤ PR_MAX AND E_MIN ≤ YM < E_SPLIT
zone[ductile & E_low]  = 4

print("Counts:", {z: int(np.sum(zone == z)) for z in [0,1,2,3,4]})

# ================== PLOT ==================

x_min, x_max = lam.min(), lam.max()
y_min, y_max = mu.min(),  mu.max()
x_pad = 0.05 * (x_max - x_min)
y_pad = 0.05 * (y_max - y_min)
x_min_plot, x_max_plot = x_min - x_pad, x_max + x_pad
y_min_plot, y_max_plot = y_min - y_pad, y_max + y_pad

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

# outside (black)
m_out = (zone == 0)
ax.scatter(lam[m_out], mu[m_out], s=4, alpha=0.45, edgecolor="none", color="k")

# zones 1–4
colors = {1:"tab:red", 2:"tab:orange", 3:"tab:cyan", 4:"tab:green"}
for z in [1,2,3,4]:
    mz = (zone == z)
    ax.scatter(lam[mz], mu[mz], s=4, alpha=0.55, edgecolor="none", color=colors[z])

# ---- draw all lines (no labels) ----
if DRAW_LINES:
    # ν lines: λ = slope * μ
    mu_line = np.linspace(y_min_plot, y_max_plot, 400)
    for nu0 in NU_VALUES:
        slope = (2.0 * nu0) / (1.0 - 2.0 * nu0)
        lam_line = slope * mu_line
        m = (lam_line >= x_min_plot) & (lam_line <= x_max_plot)
        if np.any(m):
            ax.plot(lam_line[m], mu_line[m], color="0.2", linewidth=0.8)

    # E lines from fitted parameters
    lam_line = np.linspace(x_min_plot, x_max_plot, 300)
    for Et in E_TARGETS:
        a, b = Efits[Et]
        ax.plot(lam_line, a*lam_line + b, color="0.2", linestyle="--", linewidth=0.8)

ax.set_xlim(x_min_plot, x_max_plot)
ax.set_ylim(y_min_plot, y_max_plot)
ax.set_xlabel(r"$\lambda\rho$  (GPa·g/cc)", fontsize=8)
ax.set_ylabel(r"$\mu\rho$  (GPa·g/cc)", fontsize=8)
ax.tick_params(axis="both", which="both", labelsize=7)
ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

plt.tight_layout()
plt.savefig(OUTFIG, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved", OUTFIG)
