# joint_pp_ps_inversion_3D_tiled_C_v3.py
# -*- coding: utf-8 -*-
"""
3D joint PP–PS prestack inversion (log domain, background-anchored), C-order, tiled.

Key structural fixes for stubborn ~0.9 misfit:
- Forward has NO lateral smoothing.
- vsvp is depth-varying per trace (from Vs0/Vp0).
- **Angle-wise** forward: one operator per angle per (ix,iy).
- **Per-angle amplitude/polarity calibration** using L m0 vs data (robust dot gain).
- Gaussian prior is on dm: (I - GxGy) dm ~ 0 (zero RHS).

Requires: numpy, pandas, scipy, pylops >= 2.5.0
500-898 ms (200 samples)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, lsqr

import pylops
from pylops.avo.prestack import PrestackLinearModelling
from pylops.basicoperators import VStack, Identity, Restriction

# ============================== USER CONFIG =========================================

ROOT = Path(r"/ddnB/work/shamsul/for_prestack_3D_inversion_20Nov")

WAVELET_DIR   = ROOT / "wavelets_pylops_ready"  # PP_family.csv, PS_family.csv
BACKGROUND_DIR= ROOT / "background_models_by_XL"
#LF_DIR        = ROOT / "outputs_lowpass"

DATA_PP_DIRS: Dict[int, Path] = {
    7:  ROOT / "pp_0_15deg",
    22: ROOT / "pp_15_30deg",
    37: ROOT / "pp_30_45deg",
}
DATA_PS_DIRS: Dict[int, Path] = {
    20: ROOT / "ps_15_25deg", #25
    27: ROOT / "ps_20_30deg", #32
    34: ROOT / "ps_30_40deg", #45
}

OUT_DIR = ROOT /"new_inversion_24Dec_488_659"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PER_XL  = True
SAVE_VOLUMES = True

LINEARIZATION_PP = "fatti"
LINEARIZATION_PS = "ps"      # needs PyLops build with 'ps'

# Data weights and regularization
W_PP, W_PS   = 1.0, 0.8 #1.0, 1.0
ALPHA_T      = 0.00
GAMMA_T2     = 0.00#0.05
ALPHA_X      = 0.0#0.03
ALPHA_Y      = 0.0#0.08
BETA_DAMP    = 0.001#0.0001#0.50
BETA_VP      = 0.0#2.0
BETA_VS      = 0.0#2.0
BETA_RHO     = 20.0#0.0#4.0
ITER_LIM     = 15#350#400

# Wavelet equalization (per angle)
EQUALIZE_WAVELETS = False#True   # <- enable per-angle zero-phase equalizer
EQ_TAPS            = 41     # odd length, e.g., 21–41
EQ_SMOOTH_BINS     = 7      # freq smoothing bins (3–9)


# Gaussian PRIOR on dm (NOT in forward!)
USE_GPRIOR   = False#True
SIGMA_X_TR   = 1.0
SIGMA_Y_TR   = 0.0
KLEN_MAX     = 15
LAMBDA_GPR   = 0.0

# Diagnostics
PRINT_SUMMARY = True
XL_WHITELIST: List[int] = [488,659]
#XL_WHITELIST=list(range(680,711))

# Trust-region + physical bounds (applied in log-space then exp)
VP_MIN, VP_MAX   = 2500.0, 7000.0 #3500.0, 7000.0
VS_MIN, VS_MAX   = 1500.0, 5200.0 #1800.0,  5000.0
RHO_MIN, RHO_MAX = 2200.0, 3000.0 #2480.0,  3000.0
TR_DOWN, TR_UP   = 1.0,1.0 #0.6, 1.5

# Tiling
IL_BATCH = 664#nx#128
CL_BATCH = 1#ny#8
OVL_X    = 0#3
OVL_Y    = 0#1

# Amplitude / polarity calibration
USE_AMP_CAL_ANGLE = False#True   # <-- angle-wise gains

# Optional: remove DC per trace (helps if stacks carry offsets)
REMOVE_TRACE_MEAN = True

# Optional time gate (indices) applied to data rows; set to None to use full
TIME_GATE: Optional[Tuple[int,int]] = (80,250) #(90,190) #None  # e.g., (160, 475) for ~320–950 ms at 2 ms dt

# =============================== HELPERS (I/O) ======================================

_XL_RE = re.compile(r"XL(\d+)\.csv$", flags=re.IGNORECASE)

def _read_wavelet_csv(path: Path) -> np.ndarray:
    w = pd.read_csv(path, header=None).values.squeeze().astype(np.float64)
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

def _demean(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, float).ravel() - float(np.mean(w))

def _unit_energy(w: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(w)
    return w if n == 0 else w / n

def _read_xl_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None, comment="#", engine="c")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.nan_to_num(df.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def _list_xl_ids(folder: Path) -> List[int]:
    ids = []
    for p in folder.glob("XL*.csv"):
        m = _XL_RE.search(p.name)
        if m: ids.append(int(m.group(1)))
    return sorted(set(ids))

def _common_xl_ids(pp_dirs: Dict[int, Path], ps_dirs: Dict[int, Path]) -> List[int]:
    if not pp_dirs or not ps_dirs:
        return []
    pp_sets = [set(_list_xl_ids(d)) for d in pp_dirs.values()]
    ps_sets = [set(_list_xl_ids(d)) for d in ps_dirs.values()]
    return sorted(set.intersection(*pp_sets, *ps_sets)) if pp_sets and ps_sets else []

def _find_bg_for_xl(bg_dir: Path, xl: int, prop: str) -> Optional[Path]:
    token = f"xl{xl}".lower(); prop = prop.lower()
    for p in list(bg_dir.glob("*.csv")) + list(bg_dir.glob("**/*.csv")):
        s = p.name.lower()
        if prop in s and token in s: return p
    return None

def _load_bg_xl(bg_dir: Path, xl: int, nt: int, nx: int):
    vp_p, vs_p, rh_p = (_find_bg_for_xl(bg_dir, xl, k) for k in ("vp", "vs", "rho"))
    if not (vp_p and vs_p and rh_p): return None
    Vp0, Vs0, Rho0 = _read_xl_csv(vp_p), _read_xl_csv(vs_p), _read_xl_csv(rh_p)
    if Vp0.shape!=(nt,nx) or Vs0.shape!=(nt,nx) or Rho0.shape!=(nt,nx):
        print(f"[warn] XL{xl} background shapes mismatch; expected {(nt,nx)}.")
        return None
    Vp0  = np.clip(np.nan_to_num(Vp0,  3000.0),  500.0, 10000.0)
    Vs0  = np.clip(np.nan_to_num(Vs0,  1600.0),  200.0,  6000.0)
    Rho0 = np.clip(np.nan_to_num(Rho0, 2300.0), 1000.0,  3500.0)
    return Vp0, Vs0, Rho0

def _read_two_col_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None, usecols=[0,1])
    vals = df.iloc[:,1].to_numpy(np.float64)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

def _broadcast_series(vals: np.ndarray, nt: int, nx: int) -> np.ndarray:
    vals = np.asarray(vals, float).reshape(-1)
    out  = np.zeros((nt, nx), float)
    L    = min(nt, vals.size)
    if L>0:
        out[:L,:] = vals[:L,None]
        if nt>L: out[L:,:] = vals[L-1]
    return out

"""def _load_bg_fallback(nt: int, nx: int):
    cands = list(LF_DIR.glob("*.csv"))
    def _find(k): 
        for p in cands:
            if k in p.name.lower(): return p
        return None
    vp_p, vs_p, rh_p = (_find(k) for k in ("vp","vs","rho"))
    vp = _read_two_col_series(vp_p) if vp_p else np.full(nt, 3000.0)
    vs = _read_two_col_series(vs_p) if vs_p else np.maximum(vp/1.9, 200.0)
    rh = _read_two_col_series(rh_p) if rh_p else np.full(nt, 2300.0)
    Vp0 = _broadcast_series(vp, nt, nx)
    Vs0 = _broadcast_series(vs, nt, nx)
    Rho0= _broadcast_series(rh, nt, nx)
    Vp0 = np.clip(Vp0,  500.0, 10000.0)
    Vs0 = np.clip(Vs0,  200.0,  6000.0)
    Rho0= np.clip(Rho0, 1000.0,  3500.0)
    return Vp0, Vs0, Rho0
"""
def _scale_with_dims(op, w):
    if w is None or float(w)==1.0: return op
    Sop = w * op
    try:
        if not hasattr(Sop, "dims") and hasattr(op, "dims"):
            setattr(Sop, "dims", op.dims)
    except Exception:
        pass
    return Sop

import pylops as _pylops
import numpy as _np

class _DimsFtoC(_pylops.LinearOperator):
    def __init__(self, opF: _pylops.LinearOperator, dims):
        self.opF  = opF
        self.dims = tuple(int(d) for d in dims)
        N = int(_np.prod(self.dims))
        super().__init__(dtype=getattr(opF, "dtype", _np.float64), shape=(opF.shape[0], N))
    def _matvec(self, xC):
        vecF = _np.asarray(xC).reshape(self.dims, order='C').ravel(order='F')
        return self.opF.matvec(vecF)
    def _rmatvec(self, y):
        xF = self.opF.rmatvec(y)
        return _np.asarray(xF).reshape(self.dims, order='F').ravel(order='C')

def _FirstDerivative_compat(*, dims, axis: int):
    try:
        opF = _pylops.FirstDerivative(dims=dims, axis=axis)
    except TypeError:
        opF = _pylops.FirstDerivative(dims=dims, dir=axis)
    return _DimsFtoC(opF, dims)

def _SecondDerivative_compat(*, dims, axis: int):
    try:
        opF = _pylops.SecondDerivative(dims=dims, axis=axis)
    except TypeError:
        opF = _pylops.SecondDerivative(dims=dims, dir=axis)
    return _DimsFtoC(opF, dims)

def _signed_gain(Y_list, D_list):
    num = den = 0.0
    for y, d in zip(Y_list, D_list):
        y = y.ravel(); d = d.ravel()
        num += float(np.dot(y, d))
        den += float(np.dot(y, y) + 1e-12)
    return (num / den) if den > 0 else 1.0

# ===== Gaussian kernels (PRIOR only) =====

def _gaussian_kernel_odd(sigma_tr: float, ntr: int, kcap: int) -> np.ndarray:
    if sigma_tr<=0 or ntr<=2: return np.array([1.0], float)
    klen = int(6.0*max(sigma_tr,1e-6)+1) | 1
    klen = max(1, min(klen, kcap, 2*ntr - 3))
    half = (klen - 1)//2
    x = np.arange(-half, half+1, dtype=np.float64)
    h = np.exp(-0.5*(x/max(sigma_tr,1e-6))**2)
    return h / (h.sum() + 1e-12)

def _convolve1d_same_zero(arr: np.ndarray, h: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, np.float64); h = np.asarray(h, np.float64).ravel(); k = int(h.size)
    if k%2: pL=pR=(k-1)//2
    else:   pL=k//2 - 1; pR=k//2
    pad = [(0,0)]*arr.ndim; pad[axis]=(pL,pR)
    x = np.pad(arr, pad, mode='constant', constant_values=0.0)
    x = np.moveaxis(x, axis, 0)
    Lorig = x.shape[0] - (pL+pR)
    out = np.empty((Lorig,*x.shape[1:]), np.float64)
    hr = h[::-1]
    for i in range(Lorig):
        seg = x[i:i+k]
        out[i] = np.tensordot(seg, hr, axes=(0,0))
    return np.moveaxis(out, 0, axis)

class Conv1D_C(LinearOperator):
    def __init__(self, dims: Tuple[int,int,int,int], h: np.ndarray, axis: int):
        self.dims = tuple(int(x) for x in dims)
        self.N = int(np.prod(self.dims))
        self.axis = int(axis)
        self.h = np.asarray(h, float).ravel()
        super().__init__(dtype=np.float64, shape=(self.N, self.N))
    def _matvec(self, x):
        m = np.asarray(x, np.float64).reshape(self.dims, order='C')
        y = _convolve1d_same_zero(m, self.h, axis=self.axis)
        return y.reshape(self.N, order='C')
    def _rmatvec(self, y):
        dy = np.asarray(y, np.float64).reshape(self.dims, order='C')
        z  = _convolve1d_same_zero(dy, self.h[::-1], axis=self.axis)
        return z.reshape(self.N, order='C')

def _indices_C(nt:int,nx:int,ny:int)->np.ndarray:
    return np.arange(nt*nx*ny*3).reshape((nt,nx,ny,3), order='C')

def _restriction_C(nt:int,nx:int,ny:int,ix:int,iy:int)->Restriction:
    idx = _indices_C(nt,nx,ny)[:,ix,iy,:].ravel(order='C')
    return Restriction(nt*nx*ny*3, idx)

"""
def _backtransform_log_to_linear_clipped(m_log, Vp0, Vs0, Rho0):
    vp_lo = np.maximum(VP_MIN,  TR_DOWN*Vp0); vp_hi = np.minimum(VP_MAX,  TR_UP*Vp0)
    vs_lo = np.maximum(VS_MIN,  TR_DOWN*Vs0); vs_hi = np.minimum(VS_MAX,  TR_UP*Vs0)
    rh_lo = np.maximum(RHO_MIN, TR_DOWN*Rho0); rh_hi = np.minimum(RHO_MAX, TR_UP*Rho0)
    l_vp_lo,l_vp_hi = np.log(vp_lo.astype(np.float64)), np.log(vp_hi.astype(np.float64))
    l_vs_lo,l_vs_hi = np.log(vs_lo.astype(np.float64)), np.log(vs_hi.astype(np.float64))
    l_rh_lo,l_rh_hi = np.log(rh_lo.astype(np.float64)), np.log(rh_hi.astype(np.float64))
    vp_log = np.clip(m_log[...,0], l_vp_lo, l_vp_hi)
    vs_log = np.clip(m_log[...,1], l_vs_lo, l_vs_hi)
    rh_log = np.clip(m_log[...,2], l_rh_lo, l_rh_hi)
    Vp  = np.exp(vp_log, dtype=np.float64).astype(np.float32)
    Vs  = np.exp(vs_log, dtype=np.float64).astype(np.float32)
    Rho = np.exp(rh_log, dtype=np.float64).astype(np.float32)
    print(f"[post] clamp fractions: Vp={(np.mean((m_log[...,0]<=l_vp_lo+1e-12)|(m_log[...,0]>=l_vp_hi-1e-12))):.3f}, "
          f"Vs={(np.mean((m_log[...,1]<=l_vs_lo+1e-12)|(m_log[...,1]>=l_vs_hi-1e-12))):.3f}, "
          f"Rho={(np.mean((m_log[...,2]<=l_rh_lo+1e-12)|(m_log[...,2]>=l_rh_hi-1e-12))):.3f}")
    return Vp,Vs,Rho
"""

def _backtransform_log_to_linear_physical_only(m_est_log: np.ndarray
                                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Just exp(log-model) and clamp to broad physical limits.
    No trust region around background.
    m_est_log shape: (nt,nx,ny,3)
    """
    vp_log = m_est_log[..., 0]
    vs_log = m_est_log[..., 1]
    rh_log = m_est_log[..., 2]

    Vp  = np.exp(vp_log, dtype=np.float64)
    Vs  = np.exp(vs_log, dtype=np.float64)
    Rho = np.exp(rh_log, dtype=np.float64)

    Vp  = np.clip(Vp,  VP_MIN,  VP_MAX ).astype(np.float32)
    Vs  = np.clip(Vs,  VS_MIN,  VS_MAX ).astype(np.float32)
    Rho = np.clip(Rho, RHO_MIN, RHO_MAX).astype(np.float32)

    # diagnostics (optional)
    frac_vp = np.mean((Vp <= VP_MIN+1e-3) | (Vp >= VP_MAX-1e-3))
    frac_vs = np.mean((Vs <= VS_MIN+1e-3) | (Vs >= VS_MAX-1e-3))
    frac_rh = np.mean((Rho<=RHO_MIN+1e-3)| (Rho>=RHO_MAX-1e-3))
    print(f"[post] physical clamp: Vp={frac_vp:.3f}, Vs={frac_vs:.3f}, Rho={frac_rh:.3f}")

    return Vp, Vs, Rho


def _zero_phase_equalizer(y_list: List[np.ndarray],
                          d_list: List[np.ndarray],
                          taps: int = 31,
                          smooth_bins: int = 5) -> np.ndarray:
    """
    Build a short zero-phase FIR that maps y (L m0) to d (data).
    Returns a real, symmetric, zero-phase kernel of odd length 'taps' and ~unit energy.
    """
    assert taps % 2 == 1, "EQ_TAPS must be odd (zero-phase)"
    if len(y_list) == 0:
        h = np.zeros(taps, float); h[taps//2] = 1.0
        return h

    # Ensure all pairs have the same length (they should; each op is one angle at (ix,iy))
    nt = int(y_list[0].size)
    for y, d in zip(y_list, d_list):
        if y.size != nt or d.size != nt:
            raise ValueError(f"Equalizer expects fixed nt; got {y.size} and {d.size} (expected {nt}).")

    # Choose an FFT size and corresponding rFFT length
    nfft = 1
    while nfft < 2 * nt:
        nfft <<= 1
    nfreq = nfft // 2 + 1

    num = np.zeros(nfreq, np.complex128)  # <D, Y*>
    den = np.zeros(nfreq, np.float64)     # |Y|^2

    eps = 1e-12
    for y, d in zip(y_list, d_list):
        y0 = y.astype(np.float64) - np.mean(y)
        d0 = d.astype(np.float64) - np.mean(d)
        Y  = np.fft.rfft(y0, nfft)   # (nfreq,)
        D  = np.fft.rfft(d0, nfft)   # (nfreq,)
        num += D * np.conj(Y)
        den += (Y.real * Y.real + Y.imag * Y.imag)

    H = num / (den + eps)  # complex transfer estimate on [0, nyquist]

    # Optional simple frequency smoothing (boxcar)
    if smooth_bins and smooth_bins > 1:
        k = int(smooth_bins)
        pad = k // 2
        # real and imag smoothed separately to avoid phase distortion here
        Hr = np.pad(H.real, (pad, pad), mode='edge')
        Hi = np.pad(H.imag, (pad, pad), mode='edge')
        ker = np.ones(k, dtype=np.float64) / k
        Hrs = np.convolve(Hr, ker, mode='valid')
        His = np.convolve(Hi, ker, mode='valid')
        H   = Hrs + 1j * His

    # Zero-phase equalizer: take magnitude only
    A = np.abs(H)  # (nfreq,)

    # Bring back to time domain as a real, even sequence
    a_full = np.fft.irfft(A, nfft)  # length nfft, centered at t=0 (even)
    # Extract symmetric window of length 'taps' around t=0
    half = taps // 2
    # indices [-half ... +half] modulo nfft
    idxs = [(i % nfft) for i in range(-half, half + 1)]
    h = a_full[idxs].copy()

    # Taper to reduce ringing and normalize to ~unit energy
    w = np.hanning(taps)
    w /= (w.sum() + 1e-12)
    h *= w
    h /= (np.linalg.norm(h) + 1e-12)

    return h



# ---------- Build angle-wise forward ops & data ----------
def _build_anglewise_ops_and_data(
    PP, PS,
    wav_pp, wav_ps,             # can be 1D (base) OR list of per-angle 1D arrays
    theta_pp, theta_ps,
    Vp0, Vs0, nt0_pp: int, nt0_ps: int,
    remove_trace_mean: bool = True,
):
    """
    Build one operator per angle per (ix,iy), return aligned op lists and data parts.
    wav_pp/wav_ps can be:
      - 1D base wavelet (same for all angles), or
      - List[1D] of length npp / nps for per-angle wavelets.
    """
    nt, nx, ny, npp = PP.shape
    _,  _,  _, nps  = PS.shape

    if remove_trace_mean:
        PP = PP - PP.mean(axis=0, keepdims=True)
        PS = PS - PS.mean(axis=0, keepdims=True)

    def _pick_pp_wav(j):
        return wav_pp[j] if isinstance(wav_pp, (list, tuple)) else wav_pp
    def _pick_ps_wav(j):
        return wav_ps[j] if isinstance(wav_ps, (list, tuple)) else wav_ps

    I_full = Identity(nt * nx * ny * 3, dtype=float)

    pp_ops, ps_ops = [], []
    pp_data_parts, ps_data_parts = [], []
    pp_angle2opix = [[] for _ in range(npp)]
    ps_angle2opix = [[] for _ in range(nps)]

    for iy in range(ny):
        for ix in range(nx):
            Rixy = _restriction_C(nt, nx, ny, ix, iy)
            vsvp_1d = np.clip(
                Vs0[:, ix, iy] / np.maximum(Vp0[:, ix, iy], 1e-6),
                1e-3, 0.999
            ).astype(np.float64)

            for j in range(npp):
                PPop = PrestackLinearModelling(
                    _pick_pp_wav(j),
                    np.array([float(theta_pp[j])], np.float64),
                    vsvp=vsvp_1d, nt0=nt0_pp, linearization=LINEARIZATION_PP,
                )
                op = PPop * Rixy * I_full
                pp_angle2opix[j].append(len(pp_ops))
                pp_ops.append(op)
                pp_data_parts.append(PP[:, ix, iy, j].ravel(order="C"))

            for j in range(nps):
                PSop = PrestackLinearModelling(
                    _pick_ps_wav(j),
                    np.array([float(theta_ps[j])], np.float64),
                    vsvp=vsvp_1d, nt0=nt0_ps, linearization=LINEARIZATION_PS,
                )
                op = PSop * Rixy * I_full
                ps_angle2opix[j].append(len(ps_ops))
                ps_ops.append(op)
                ps_data_parts.append(PS[:, ix, iy, j].ravel(order="C"))

    return (
        pp_ops, ps_ops,
        pp_data_parts, ps_data_parts,
        pp_angle2opix, ps_angle2opix,
    )



# ---------- Angle-wise amplitude/polarity calibration ----------
def _apply_anglewise_calibration(
    pp_ops, ps_ops,
    pp_data_parts, ps_data_parts,
    pp_angle2opix, ps_angle2opix,
    m0_vec,
):
    """
    Compute per-angle gains g = sum_k <y_k, d_k> / sum_k <y_k, y_k>,
    where k runs over all operators that belong to that angle.
    Everything is flattened to 1-D to avoid shape (N,1) vs (N,) issues.
    """
    # Forward on m0, per-op (aligned with *_ops and *_data_parts)
    ypp_parts = [(op @ m0_vec).ravel() for op in pp_ops]
    yps_parts = [(op @ m0_vec).ravel() for op in ps_ops]

    # Gather all-PP / all-PS pairs (after EQ, per-op predictions)
    gPP0 = _signed_gain(ypp_parts, pp_data_parts) if len(ypp_parts) else 1.0
    gPS0 = _signed_gain(yps_parts, ps_data_parts) if len(yps_parts) else 1.0

    # Apply global scales to every op in the component
    for k in range(len(pp_ops)):
      pp_ops[k] = _scale_with_dims(pp_ops[k], gPP0)
    for k in range(len(ps_ops)):
      ps_ops[k] = _scale_with_dims(ps_ops[k], gPS0)

    print(f"[amp-cal] global scales: gPP0={gPP0:.3f}  gPS0={gPS0:.3f}")

    # Refresh predictions after global scaling
    ypp_parts = [(op @ m0_vec).ravel() for op in pp_ops]
    yps_parts = [(op @ m0_vec).ravel() for op in ps_ops]
    
    # Gains for PP
    gpp = []
    for j, op_idxs in enumerate(pp_angle2opix):
        if not op_idxs:
            gpp.append(1.0); continue
        num = 0.0; den = 0.0
        for k in op_idxs:
            y = ypp_parts[k]; d = pp_data_parts[k]
            num += float(np.dot(y, d))
            den += float(np.dot(y, y) + 1e-12)
        gpp.append(num / den if den > 0 else 1.0)

    # Gains for PS
    gps = []
    for j, op_idxs in enumerate(ps_angle2opix):
        if not op_idxs:
            gps.append(1.0); continue
        num = 0.0; den = 0.0
        for k in op_idxs:
            y = yps_parts[k]; d = ps_data_parts[k]
            num += float(np.dot(y, d))
            den += float(np.dot(y, y) + 1e-12)
        gps.append(num / den if den > 0 else 1.0)

    # Apply the angle gains to each operator belonging to that angle
    for j, op_idxs in enumerate(pp_angle2opix):
        for k in op_idxs:
            pp_ops[k] = _scale_with_dims(pp_ops[k], gpp[j])
    for j, op_idxs in enumerate(ps_angle2opix):
        for k in op_idxs:
            ps_ops[k] = _scale_with_dims(ps_ops[k], gps[j])

    return pp_ops, ps_ops, np.array(gpp), np.array(gps)


# ---------- Gaussian PRIOR on dm ----------
def _gaussian_prior_op(nt:int,nx:int,ny:int,sigma_x:float,sigma_y:float,klen_max:int,weight:float):
    if weight<=0.0 or (sigma_x<=0 and sigma_y<=0): return None
    dims=(nt,nx,ny,3)
    hx = _gaussian_kernel_odd(sigma_x, nx, klen_max) if sigma_x>0 else np.array([1.0])
    hy = _gaussian_kernel_odd(sigma_y, ny, klen_max) if sigma_y>0 else np.array([1.0])
    Gx = Conv1D_C(dims, hx, axis=1)
    Gy = Conv1D_C(dims, hy, axis=2)
    G  = Gy * Gx
    I  = Identity(int(np.prod(dims)), dtype=float)
    HighPass = I - G
    return _scale_with_dims(HighPass, weight)

# ================================ INVERSION CORE ====================================

def invert_3d_tile(
    PP, PS, Vp0, Vs0, Rho0, wav_pp, wav_ps, theta_pp, theta_ps,
    alpha_t=ALPHA_T, gamma_t2=GAMMA_T2, alpha_x=ALPHA_X, alpha_y=ALPHA_Y,
    beta_damp=BETA_DAMP, beta_vp=BETA_VP, beta_vs=BETA_VS, beta_rho=BETA_RHO,
    w_pp=W_PP, w_ps=W_PS, iter_lim=ITER_LIM,
    use_gprior=USE_GPRIOR, sigma_x=SIGMA_X_TR, sigma_y=SIGMA_Y_TR, klen_max=KLEN_MAX, lambda_g=LAMBDA_GPR,
    nt0_pp=None, nt0_ps=None, print_summary=PRINT_SUMMARY,
):
    nt,nx,ny,_ = PP.shape
    if nt0_pp is None: nt0_pp = nt
    if nt0_ps is None: nt0_ps = nt

    # background in log-params
    m0 = np.zeros((nt,nx,ny,3), float)
    m0[...,0] = np.log(np.clip(Vp0,  100.0, 3e4))
    m0[...,1] = np.log(np.clip(Vs0,   50.0, 2e4))
    m0[...,2] = np.log(np.clip(Rho0,  10.0, 1e5))
    m0_vec = m0.ravel(order='C')

    # Build angle-wise ops & data (optionally time-gated, de-meaned)
    # Build per-operator ops & data (angle-wise), then angle-wise amp/pol calibration
    # --- Build ops with base wavelets (to get y=m0 predictions) ---
    pp_ops, ps_ops, pp_data_parts, ps_data_parts, pp_map, ps_map = _build_anglewise_ops_and_data(
      PP, PS, wav_pp, wav_ps, theta_pp, theta_ps, Vp0, Vs0,
      nt0_pp=nt0_pp, nt0_ps=nt0_ps, remove_trace_mean=REMOVE_TRACE_MEAN,
    )

    # Predicted seismograms from m0 (per-op)
    ypp_parts = [(op @ m0_vec).ravel() for op in pp_ops]
    yps_parts = [(op @ m0_vec).ravel() for op in ps_ops]

        # ---------------- GLOBAL PP/PS SCALING (time-gated) ----------------
    # Use the TIME_GATE to estimate one scalar for PP and one for PS
    if TIME_GATE is not None:
        i1, i2 = TIME_GATE
    else:
        i1, i2 = 0, nt  # full time if no gate

    # Global PP scale
    gPP0 = 1.0
    if len(pp_ops):
        num = den = 0.0
        for y, d in zip(ypp_parts, pp_data_parts):
            yg = y[i1:i2].astype(np.float64)
            dg = d[i1:i2].astype(np.float64)
            num += float(np.dot(yg, dg))
            den += float(np.dot(yg, yg) + 1e-12)
        gPP0 = num / den if den > 0 else 1.0
        for k in range(len(pp_ops)):
            pp_ops[k] = _scale_with_dims(pp_ops[k], gPP0)

    # Global PS scale
    gPS0 = 1.0
    if len(ps_ops):
        num = den = 0.0
        for y, d in zip(yps_parts, ps_data_parts):
            yg = y[i1:i2].astype(np.float64)
            dg = d[i1:i2].astype(np.float64)
            num += float(np.dot(yg, dg))
            den += float(np.dot(yg, yg) + 1e-12)
        gPS0 = num / den if den > 0 else 1.0
        for k in range(len(ps_ops)):
            ps_ops[k] = _scale_with_dims(ps_ops[k], gPS0)

    print(f"[amp-cal-gate] gPP0={gPP0:.3f}  gPS0={gPS0:.3f}")

    # refresh y=m0 predictions after scaling
    ypp_parts = [(op @ m0_vec).ravel() for op in pp_ops]
    yps_parts = [(op @ m0_vec).ravel() for op in ps_ops]

    
    # --- Per-angle zero-phase wavelet equalization (optional) ---
    if EQUALIZE_WAVELETS:
      # Build per-angle equalizers for PP
      wav_pp_angles = []
      for j, op_idxs in enumerate(pp_map):
        y_list = [ypp_parts[k] for k in op_idxs]
        d_list = [pp_data_parts[k] for k in op_idxs]
        h = _zero_phase_equalizer(y_list, d_list, taps=EQ_TAPS, smooth_bins=EQ_SMOOTH_BINS)
        wav_pp_angles.append(np.convolve(wav_pp if wav_pp.ndim==1 else wav_pp[0], h, mode="full"))
        # keep unit-energy to stabilize scaling; scalars will adjust amplitude
        w = wav_pp_angles[-1]; wav_pp_angles[-1] = (w - np.mean(w)); 
        e = np.linalg.norm(wav_pp_angles[-1]); wav_pp_angles[-1] = wav_pp_angles[-1] / (e + 1e-12)

      # Per-angle equalizers for PS
      wav_ps_angles = []
      for j, op_idxs in enumerate(ps_map):
        y_list = [yps_parts[k] for k in op_idxs]
        d_list = [ps_data_parts[k] for k in op_idxs]
        h = _zero_phase_equalizer(y_list, d_list, taps=EQ_TAPS, smooth_bins=EQ_SMOOTH_BINS)
        wav_ps_angles.append(np.convolve(wav_ps if wav_ps.ndim==1 else wav_ps[0], h, mode="full"))
        w = wav_ps_angles[-1]; wav_ps_angles[-1] = (w - np.mean(w)); 
        e = np.linalg.norm(wav_ps_angles[-1]); wav_ps_angles[-1] = wav_ps_angles[-1] / (e + 1e-12)

      # Rebuild ops with per-angle wavelets
      pp_ops, ps_ops, pp_data_parts, ps_data_parts, pp_map, ps_map = _build_anglewise_ops_and_data(
        PP, PS, wav_pp_angles, wav_ps_angles, theta_pp, theta_ps, Vp0, Vs0,
        nt0_pp=nt0_pp, nt0_ps=nt0_ps, remove_trace_mean=REMOVE_TRACE_MEAN,
      )
      # Refresh y=m0 predictions after equalization
      ypp_parts = [(op @ m0_vec).ravel() for op in pp_ops]
      yps_parts = [(op @ m0_vec).ravel() for op in ps_ops]

    # --- Angle-wise scalar gains (should now be near ±1) ---
    gpp, gps = [], []
    if USE_AMP_CAL_ANGLE:
      # PP scalar per angle
      for j, op_idxs in enumerate(pp_map):
        num = den = 0.0
        for k in op_idxs:
            y = ypp_parts[k]; d = pp_data_parts[k]
            num += float(np.dot(y, d))
            den += float(np.dot(y, y) + 1e-12)
        gpp.append(num/den if den>0 else 1.0)
      # PS scalar per angle
      for j, op_idxs in enumerate(ps_map):
        num = den = 0.0
        for k in op_idxs:
            y = yps_parts[k]; d = ps_data_parts[k]
            num += float(np.dot(y, d))
            den += float(np.dot(y, y) + 1e-12)
        gps.append(num/den if den>0 else 1.0)
      # Apply scalars
      for j, idxs in enumerate(pp_map):
        for k in idxs:
            pp_ops[k] = _scale_with_dims(pp_ops[k], gpp[j])
      for j, idxs in enumerate(ps_map):
        for k in idxs:
            ps_ops[k] = _scale_with_dims(ps_ops[k], gps[j])
      print(f"[amp-cal] (post-EQ) gPP={np.round(np.array(gpp),3).tolist()}  gPS={np.round(np.array(gps),3).tolist()}")

    # Concatenate data (aligned with ops)
    dPP_big = np.concatenate(pp_data_parts) if pp_data_parts else np.array([], float)
    dPS_big = np.concatenate(ps_data_parts) if ps_data_parts else np.array([], float)

    # Assemble L and RHS (anchored to m0), with data weights
    Lpp = VStack(pp_ops) if pp_ops else None
    Lps = VStack(ps_ops) if ps_ops else None
    blocks, rhs_parts = [], []
    if Lpp is not None:
      Lw_pp = _scale_with_dims(Lpp, np.sqrt(w_pp))
      blocks.append(Lw_pp)
      rhs_parts.append(np.sqrt(w_pp) * dPP_big - (Lw_pp @ m0_vec))
    if Lps is not None:
      Lw_ps = _scale_with_dims(Lps, np.sqrt(w_ps))
      blocks.append(Lw_ps)
      rhs_parts.append(np.sqrt(w_ps) * dPS_big - (Lw_ps @ m0_vec))
    L    = VStack(blocks)
    rhs0 = np.concatenate(rhs_parts) if rhs_parts else np.array([], float)



    ops, rhs = [L], [rhs0]

    # Derivative priors anchored to m0
    if alpha_t>0.0:
        Dt = _FirstDerivative_compat(dims=(nt,nx,ny,3), axis=0)
        ops.append(_scale_with_dims(Dt, alpha_t)); rhs.append(-alpha_t * (Dt @ m0_vec))
    if gamma_t2>0.0:
        D2t = _SecondDerivative_compat(dims=(nt,nx,ny,3), axis=0)
        ops.append(_scale_with_dims(D2t, gamma_t2)); rhs.append(-gamma_t2 * (D2t @ m0_vec))
    if alpha_x>0.0:
        Dx = _FirstDerivative_compat(dims=(nt,nx,ny,3), axis=1)
        ops.append(_scale_with_dims(Dx, alpha_x)); rhs.append(-alpha_x * (Dx @ m0_vec))
    if alpha_y>0.0:
        Dy = _FirstDerivative_compat(dims=(nt,nx,ny,3), axis=2)
        ops.append(_scale_with_dims(Dy, alpha_y)); rhs.append(-alpha_y * (Dy @ m0_vec))

    # Gaussian PRIOR on dm
    if use_gprior and lambda_g>0 and (sigma_x>0 or sigma_y>0):
        HP = _gaussian_prior_op(nt,nx,ny,sigma_x,sigma_y,klen_max,lambda_g)
        if HP is not None:
            ops.append(HP); rhs.append(np.zeros(m0_vec.size, float))

    # per-parameter L2 and global damp on dm
    if any(b>0.0 for b in (beta_vp,beta_vs,beta_rho)):
        nmod = m0_vec.size
        idx_vp = np.arange(0, nmod, 3); idx_vs = idx_vp+1; idx_rh = idx_vp+2
        if beta_vp>0:  ops.append(_scale_with_dims(Restriction(nmod, idx_vp), beta_vp));  rhs.append(np.zeros(idx_vp.size))
        if beta_vs>0:  ops.append(_scale_with_dims(Restriction(nmod, idx_vs), beta_vs));  rhs.append(np.zeros(idx_vs.size))
        if beta_rho>0: ops.append(_scale_with_dims(Restriction(nmod, idx_rh), beta_rho)); rhs.append(np.zeros(idx_rh.size))
    if beta_damp>0.0:
        I = Identity(m0_vec.size, dtype=float); ops.append(_scale_with_dims(I, beta_damp)); rhs.append(np.zeros(m0_vec.size))

    A = VStack(ops); b = np.concatenate(rhs)

    out = lsqr(A, b, iter_lim=iter_lim, atol=1e-10, btol=1e-10, conlim=1e12, show=False)
    dm = out[0]
    info = dict(istop=out[1], itn=out[2], r1norm=out[3], r2norm=out[4])

    """
    if print_summary:
        res  = (L @ dm) - rhs0
        dcat = np.concatenate(rhs_parts) if rhs_parts else np.array([], float)
        rel_tot = np.linalg.norm(res) / (np.linalg.norm(np.concatenate([np.sqrt(W_PP)*dPP_big, np.sqrt(W_PS)*dPS_big])) + 1e-12)
        print(info); print(f"[3D summary] rel_misfit_total={rel_tot:.3f}")
    """
    if print_summary:
      # Full log-model
      m_est_vec = m0_vec + dm

      # If TIME_GATE is set, compute misfit only inside that window
      if TIME_GATE is not None:
        i1, i2 = TIME_GATE
        num = 0.0  # ||res||^2 over gate
        den = 0.0  # ||data||^2 over gate

        # PP component
        for k, op in enumerate(pp_ops):
            syn = (op @ m_est_vec).ravel().astype(np.float64)      # synthetic PP
            dat = pp_data_parts[k].astype(np.float64)              # real PP
            syn_w = np.sqrt(w_pp) * syn
            dat_w = np.sqrt(w_pp) * dat
            res_t = syn_w[i1:i2] - dat_w[i1:i2]
            num += float(np.dot(res_t, res_t))
            den += float(np.dot(dat_w[i1:i2], dat_w[i1:i2]) + 1e-12)

        # PS component
        for k, op in enumerate(ps_ops):
            syn = (op @ m_est_vec).ravel().astype(np.float64)      # synthetic PS
            dat = ps_data_parts[k].astype(np.float64)              # real PS
            syn_w = np.sqrt(w_ps) * syn
            dat_w = np.sqrt(w_ps) * dat
            res_t = syn_w[i1:i2] - dat_w[i1:i2]
            num += float(np.dot(res_t, res_t))
            den += float(np.dot(dat_w[i1:i2], dat_w[i1:i2]) + 1e-12)

        rel_gate = (num ** 0.5) / (den ** 0.5) if den > 0 else 0.0
        print(info)
        print(f"[3D summary] rel_misfit_gate={rel_gate:.3f}  (samples {i1}:{i2})")

      else:
        # Fallback: original full-time misfit
        res_data = (L @ dm) - rhs0
        d_full = np.concatenate(
            [np.sqrt(w_pp) * dPP_big, np.sqrt(w_ps) * dPS_big]
        ) if (dPP_big.size + dPS_big.size) > 0 else np.array([], float)
        rel_tot = np.linalg.norm(res_data) / (np.linalg.norm(d_full) + 1e-12)
        print(info)
        print(f"[3D summary] rel_misfit_total={rel_tot:.3f}")
    
    m_est = (m0_vec + dm).reshape(nt,nx,ny,3, order='C')
    #Vp,Vs,Rho = _backtransform_log_to_linear_clipped(m_est, Vp0, Vs0, Rho0)
    Vp, Vs, Rho = _backtransform_log_to_linear_physical_only(m_est)
    return Vp,Vs,Rho,info

"""
# ============================== For synthetic generation =================================================
def create_synthetics_from_model(
    Vp: np.ndarray,
    Vs: np.ndarray,
    Rho: np.ndarray,
    Vp0: np.ndarray,
    Vs0: np.ndarray,
    wav_pp,
    wav_ps,
    theta_pp: np.ndarray,
    theta_ps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    
    #Build synthetic PP / PS 4D volumes from final model:
        #Vp, Vs, Rho  (nt,nx,ny)
    #using per-angle wavelets and PyLops PrestackLinearModelling.

    #Returns:
        #PPsyn: (nt, nx, ny, npp)
        #PSsyn: (nt, nx, ny, nps)
    
    Vp  = np.asarray(Vp,  np.float64)
    Vs  = np.asarray(Vs,  np.float64)
    Rho = np.asarray(Rho, np.float64)
    Vp0 = np.asarray(Vp0, np.float64)
    Vs0 = np.asarray(Vs0, np.float64)

    nt, nx, ny = Vp.shape
    theta_pp = np.asarray(theta_pp, np.float64)
    theta_ps = np.asarray(theta_ps, np.float64)
    npp = theta_pp.size
    nps = theta_ps.size

    PPsyn = np.zeros((nt, nx, ny, npp), np.float32)
    PSsyn = np.zeros((nt, nx, ny, nps), np.float32)

    for iy in range(ny):
        for ix in range(nx):
            # vsvp from background model (same as inversion)
            vsvp_1d = np.clip(
                Vs0[:, ix, iy] / np.maximum(Vp0[:, ix, iy], 1e-6),
                1e-3, 0.999,
            ).astype(np.float64)

            # log-parameters for this trace (same clipping as in m0)
            logvp = np.log(np.clip(Vp[:, ix, iy],  100.0, 3e4))
            logvs = np.log(np.clip(Vs[:, ix, iy],   50.0, 2e4))
            logrh = np.log(np.clip(Rho[:, ix, iy],  10.0, 1e5))

            # shape (nt,3): [logVp, logVs, logRho]
            m_mat = np.column_stack([logvp, logvs, logrh]).astype(np.float64)
            # Flatten in Fortran order to match PyLops dims=(nt,3)
            m_vec = m_mat.ravel(order="F")

            # PP angles
            for j, th in enumerate(theta_pp):
                wj = wav_pp[j] if isinstance(wav_pp, (list, tuple)) else wav_pp
                op_pp = PrestackLinearModelling(
                    wj,
                    np.array([float(th)], np.float64),
                    vsvp=vsvp_1d,
                    nt0=nt,
                    linearization=LINEARIZATION_PP,
                )
                tr_pp = (op_pp @ m_vec).astype(np.float32).ravel()
                if tr_pp.size != nt:
                    tr_pp = tr_pp[:nt]
                PPsyn[:, ix, iy, j] = tr_pp

            # PS angles
            for j, th in enumerate(theta_ps):
                wj = wav_ps[j] if isinstance(wav_ps, (list, tuple)) else wav_ps
                op_ps = PrestackLinearModelling(
                    wj,
                    np.array([float(th)], np.float64),
                    vsvp=vsvp_1d,
                    nt0=nt,
                    linearization=LINEARIZATION_PS,
                )
                tr_ps = (op_ps @ m_vec).astype(np.float32).ravel()
                if tr_ps.size != nt:
                    tr_ps = tr_ps[:nt]
                PSsyn[:, ix, iy, j] = tr_ps

    return PPsyn, PSsyn

def save_synthetics_as_csv(
    PPsyn: np.ndarray,
    PSsyn: np.ndarray,
    theta_pp: np.ndarray,
    theta_ps: np.ndarray,
    xl_ids: List[int],
    out_dir: Path,
):
    
    #Save synthetic PP/PS volumes as CSVs, mimicking your input layout:
        #synthetics/
          #PP/PP_7/XL488.csv  (nt x nx)
          #PP/PP_22/XL488.csv
          #...
          #PS/PS_20/XL488.csv
          #...
    
    out_dir = Path(out_dir)
    syn_root = out_dir / "synthetics"
    syn_pp_root = syn_root / "PP"
    syn_ps_root = syn_root / "PS"
    syn_pp_root.mkdir(parents=True, exist_ok=True)
    syn_ps_root.mkdir(parents=True, exist_ok=True)

    nt, nx, ny, npp = PPsyn.shape
    _,  _,  ny2, nps = PSsyn.shape
    assert ny == len(xl_ids) == ny2

    # PP angles
    for j, th in enumerate(theta_pp):
        folder = syn_pp_root / f"PP_{int(round(th))}"
        folder.mkdir(parents=True, exist_ok=True)
        for iy, xl in enumerate(xl_ids):
            arr = PPsyn[:, :, iy, j]
            pd.DataFrame(arr).to_csv(
                folder / f"XL{xl}.csv",
                header=False,
                index=False,
                float_format="%.6f",
            )

    # PS angles
    for j, th in enumerate(theta_ps):
        folder = syn_ps_root / f"PS_{int(round(th))}"
        folder.mkdir(parents=True, exist_ok=True)
        for iy, xl in enumerate(xl_ids):
            arr = PSsyn[:, :, iy, j]
            pd.DataFrame(arr).to_csv(
                folder / f"XL{xl}.csv",
                header=False,
                index=False,
                float_format="%.6f",
            )

    print(f"[save] Synthetic PP/PS CSVs written under {syn_root}")
# ============================== For synthetic generation =================================================
"""

# ============================== TILING =================================================

def _hann1d(n, ovl):
    if ovl<=0 or 2*ovl>=n: return np.ones(n, np.float32)
    w = np.ones(n, np.float32); t = np.linspace(0, np.pi/2, ovl, np.float32)
    ramp = np.sin(t)**2; w[:ovl]*=ramp; w[-ovl:]*=ramp[::-1]; return w

def _blend2d(nx, ny, ovl_x, ovl_y):
    return (_hann1d(nx,ovl_x)[:,None] * _hann1d(ny,ovl_y)[None,:]).astype(np.float32)

def run_3d_tiled(
    PP, PS, Vp0, Vs0, Rho0, wav_pp, wav_ps, theta_pp, theta_ps, xl_ids: List[int], out_dir: Path,
    il_batch=IL_BATCH, cl_batch=CL_BATCH, ovl_x=OVL_X, ovl_y=OVL_Y,
    **inv_kwargs,
):
    nt,nx,ny,_ = PP.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    vp_acc = np.memmap(out_dir / "Vp_acc.float32", dtype="float32", mode="w+", shape=(nt,nx,ny))
    vs_acc = np.memmap(out_dir / "Vs_acc.float32", dtype="float32", mode="w+", shape=(nt,nx,ny))
    rh_acc = np.memmap(out_dir / "Rho_acc.float32",dtype="float32", mode="w+", shape=(nt,nx,ny))
    w_acc  = np.memmap(out_dir / "W_acc.float32", dtype="float32", mode="w+", shape=(1,nx,ny))
    vp_acc[:]=0; vs_acc[:]=0; rh_acc[:]=0; w_acc[:]=0

    il_starts = list(range(0, nx, max(1, il_batch - ovl_x))) if nx>il_batch else [0]
    cl_starts = list(range(0, ny, max(1, cl_batch - ovl_y))) if ny>cl_batch else [0]

    for x0 in il_starts:
        x1 = min(nx, x0+il_batch); il_subset = list(range(x0,x1)); nx_tile = x1-x0
        for y0 in cl_starts:
            time_count1 = time.perf_counter()
            y1 = min(ny, y0+cl_batch); cl_subset = list(range(y0,y1)); ny_tile = y1-y0
            w2d = _blend2d(nx_tile, ny_tile, ovl_x, ovl_y); w3d = w2d[None,:,:]

            Vp_t,Vs_t,Rho_t,_ = invert_3d_tile(
                PP[:, il_subset, :][:, :, cl_subset, :],
                PS[:, il_subset, :][:, :, cl_subset, :],
                Vp0[:, il_subset, :][:, :, cl_subset],
                Vs0[:, il_subset, :][:, :, cl_subset],
                Rho0[:, il_subset, :][:, :, cl_subset],
                wav_pp, wav_ps, theta_pp, theta_ps,
                **inv_kwargs
            )
            vp_acc[:,x0:x1,y0:y1]+= (Vp_t * w3d).astype(np.float32)
            vs_acc[:,x0:x1,y0:y1]+= (Vs_t * w3d).astype(np.float32)
            rh_acc[:,x0:x1,y0:y1]+= (Rho_t* w3d).astype(np.float32)
            w_acc[0,x0:x1,y0:y1] += w2d.astype(np.float32)
            time_count2 = time.perf_counter() - time_count1
            print(f"subset {y0+1} of {len(cl_starts)} in {time_count2:.2f} s")

        w = w_acc[0]
        w[w == 0] = 1.0
        Vp  = (vp_acc / w[None, :, :]).astype(np.float32)
        Vs  = (vs_acc / w[None, :, :]).astype(np.float32)
        Rho = (rh_acc / w[None, :, :]).astype(np.float32)

        # ----- NEW: Impedances -----
        Ip = (Vp * Rho).astype(np.float32)   # Acoustic impedance
        Is = (Vs * Rho).astype(np.float32)   # Shear impedance

        # Save 3D volumes
        np.save(out_dir / 'Vp3D.npy',  Vp)
        np.save(out_dir / 'Vs3D.npy',  Vs)
        np.save(out_dir / 'Rho3D.npy', Rho)
        np.save(out_dir / 'Ip3D.npy',  Ip)
        np.save(out_dir / 'Is3D.npy',  Is)

        # Per-XL CSVs
        for iy, xl in enumerate(xl_ids):
          xl_dir = out_dir / f"XL{xl}"
          xl_dir.mkdir(parents=True, exist_ok=True)

          # velocity & density
          pd.DataFrame(Vp[:, :, iy]).to_csv(xl_dir / "Vp.csv",
                                          header=False, index=False, float_format="%.6f")
          pd.DataFrame(Vs[:, :, iy]).to_csv(xl_dir / "Vs.csv",
                                          header=False, index=False, float_format="%.6f")
          pd.DataFrame(Rho[:, :, iy]).to_csv(xl_dir / "Rho.csv",
                                           header=False, index=False, float_format="%.6f")

          # ----- NEW: Ip, Is per XL -----
          pd.DataFrame(Ip[:, :, iy]).to_csv(xl_dir / "Ip.csv",
                                          header=False, index=False, float_format="%.6f")
          pd.DataFrame(Is[:, :, iy]).to_csv(xl_dir / "Is.csv",
                                          header=False, index=False, float_format="%.6f")

        print("[tile-run] done. volumes + impedances written.")
        return Vp, Vs, Rho


# ================================ I/O & MAIN ========================================

def _load_volume_from_xls(angle_dirs: Dict[int, Path], xl_ids: List[int]):
    thetas = np.array(sorted(angle_dirs.keys()), dtype=np.float64)
    probe  = _read_xl_csv(angle_dirs[int(thetas[0])] / f"XL{xl_ids[0]}.csv")
    nt,nx = probe.shape; ny=len(xl_ids); ntheta=len(thetas)
    vol = np.zeros((nt,nx,ny,ntheta), np.float64)
    for j,th in enumerate(thetas):
        folder = angle_dirs[int(th)]
        for iy,xl in enumerate(xl_ids):
            arr = _read_xl_csv(folder / f"XL{xl}.csv")
            if arr.shape!=(nt,nx):
                raise ValueError(f"Angle θ={th}, XL{xl}: got {arr.shape}, expected {(nt,nx)}")
            vol[:,:,iy,j]=arr
    return vol, nt, nx, ny, thetas

def _load_background_volume(xl_ids: List[int], nt: int, nx: int):
    Vp0 = np.zeros((nt,nx,len(xl_ids)), float)
    Vs0 = np.zeros_like(Vp0); Rho0 = np.zeros_like(Vp0)
    have_all=True
    for iy,xl in enumerate(xl_ids):
        bg = _load_bg_xl(BACKGROUND_DIR, xl, nt, nx)
        if bg is None: have_all=False; break
        vpi,vsi,rhoi = bg
        Vp0[:,:,iy]=vpi; Vs0[:,:,iy]=vsi; Rho0[:,:,iy]=rhoi
    if not have_all:
        print("[warn] Missing per-XL backgrounds; using fallback LF broadcast")
        vpf,vsf,rhf = _load_bg_fallback(nt, nx)
        for iy in range(len(xl_ids)):
            Vp0[:,:,iy]=vpf; Vs0[:,:,iy]=vsf; Rho0[:,:,iy]=rhf
    return Vp0,Vs0,Rho0

def main():
    """
    pp_thetas = np.array(sorted(DATA_PP_DIRS.keys()), dtype=np.float64)
    ps_thetas = np.array(sorted(DATA_PS_DIRS.keys()), dtype=np.float64)
    print(f"[info] PP angles: {pp_thetas.tolist()}   PS angles: {ps_thetas.tolist()}")

    wav_pp = _unit_energy(_demean(_read_wavelet_csv(WAVELET_DIR/"PP_family.csv")))
    wav_ps = _unit_energy(_demean(_read_wavelet_csv(WAVELET_DIR/"PS_family.csv")))
    print(f"[info] Loaded wavelets: PP({wav_pp.size}), PS({wav_ps.size}) [demeaned, unit-energy]")
    """

    pp_thetas = np.array(sorted(DATA_PP_DIRS.keys()), dtype=np.float64)
    ps_thetas = np.array(sorted(DATA_PS_DIRS.keys()), dtype=np.float64)
    print(f"[info] PP angles: {pp_thetas.tolist()}   PS angles: {ps_thetas.tolist()}")

    # ---- per-angle wavelets ----
    wav_pp_list = []
    for th in pp_thetas:
        fname = WAVELET_DIR / f"PP_{int(round(th))}.csv"   # e.g., PP_7.csv, PP_22.csv, ...
        w = _read_wavelet_csv(fname)
        w = _unit_energy(_demean(w))
        wav_pp_list.append(-w)

    wav_ps_list = []
    for th in ps_thetas:
        fname = WAVELET_DIR / f"PS_{int(round(th))}.csv"   # e.g., PS_20.csv, PS_27.csv, ...
        w = _read_wavelet_csv(fname)
        w = _unit_energy(_demean(w))
        wav_ps_list.append(-w)

    wav_pp = wav_pp_list
    wav_ps = wav_ps_list

    print("[info] Loaded per-angle PP wavelets:", [len(w) for w in wav_pp])
    print("[info] Loaded per-angle PS wavelets:", [len(w) for w in wav_ps])

    
    xl_ids = _common_xl_ids(DATA_PP_DIRS, DATA_PS_DIRS)
    if XL_WHITELIST: xl_ids = [xl for xl in xl_ids if xl in XL_WHITELIST]
    if not xl_ids: raise RuntimeError("No common XL###.csv found across PP/PS folders.")
    print(f"[info] Using {len(xl_ids)} XL slices, e.g., {xl_ids[:5]}")

    PP, nt, nx, ny, theta_pp = _load_volume_from_xls(DATA_PP_DIRS, xl_ids)
    PS, nt2, nx2, ny2, theta_ps = _load_volume_from_xls(DATA_PS_DIRS, xl_ids)
    assert (nt,nx,ny)==(nt2,nx2,ny2)
    print(f"[info] Volume shape: nt={nt}, nx={nx}, ny={ny}")

    Vp0,Vs0,Rho0 = _load_background_volume(xl_ids, nt, nx)

    """
    # ---------------- SMALL TEST WINDOW (INLINE CROP) ----------------
    # pick a small inline window around the middle
    win_nx = 250          # number of inlines for quick run (change if you like)
    i_center = nx // 2
    i0 = max(0, i_center - win_nx // 2)
    i1 = min(nx, i0 + win_nx)

    print(f"[small-run] Cropping inlines: i0={i0}, i1={i1} (nx_win={i1-i0})")

    PP   = PP[:, i0:i1, :, :]
    PS   = PS[:, i0:i1, :, :]
    Vp0  = Vp0[:, i0:i1, :]
    Vs0  = Vs0[:, i0:i1, :]
    Rho0 = Rho0[:, i0:i1, :]
    """
    #nt, nx_win, ny, _ = PP.shape
    #print(f"[small-run] New volume shape: nt={nt}, nx={nx_win}, ny={ny}")
    # -------- Upto this and commenting PP,PS next ------
    
    """
    #---------------- (TIME CROP) ----------------
    # ---- time crop: keep 500–900 ms → indices [90:290] ----
    i0, i1 = 100, 320  # Python slice is [i0 : i1] -> 90..289

    PP   = PP[i0:i1, ...]      # (nt_crop, nx, ny, npp)
    PS   = PS[i0:i1, ...]      # (nt_crop, nx, ny, nps)
    Vp0  = Vp0[i0:i1, ...]     # (nt_crop, nx, ny)
    Vs0  = Vs0[i0:i1, ...]
    Rho0 = Rho0[i0:i1, ...]
    nt   = PP.shape[0]

    print(f"[crop] New volume shape: nt={nt}, nx={nx}, ny={ny}")
    
    #---------------- (TIME CROP) ----------------
    """
    
    PP = np.nan_to_num(PP, nan=0.0, posinf=0.0, neginf=0.0)
    PS = np.nan_to_num(PS, nan=0.0, posinf=0.0, neginf=0.0)

    Vp,Vs,Rho = run_3d_tiled(
        PP, PS, Vp0, Vs0, Rho0, wav_pp, wav_ps, theta_pp, theta_ps, xl_ids, OUT_DIR,
        il_batch=IL_BATCH, cl_batch=CL_BATCH, ovl_x=OVL_X, ovl_y=OVL_Y,
        alpha_t=ALPHA_T, gamma_t2=GAMMA_T2, alpha_x=ALPHA_X, alpha_y=ALPHA_Y,
        beta_damp=BETA_DAMP, beta_vp=BETA_VP, beta_vs=BETA_VS, beta_rho=BETA_RHO,
        w_pp=W_PP, w_ps=W_PS, iter_lim=ITER_LIM, print_summary=PRINT_SUMMARY,
        use_gprior=USE_GPRIOR, sigma_x=SIGMA_X_TR, sigma_y=SIGMA_Y_TR, klen_max=KLEN_MAX, lambda_g=LAMBDA_GPR,
        nt0_pp=nt, nt0_ps=nt,
    )

    """
        # ----- NEW: build and save final synthetics -----
    PPsyn, PSsyn = create_synthetics_from_model(
        Vp, Vs, Rho, Vp0, Vs0, wav_pp, wav_ps, theta_pp, theta_ps
    )
    save_synthetics_as_csv(PPsyn, PSsyn, theta_pp, theta_ps, xl_ids, OUT_DIR)
    # ----- NEW: build and save final synthetics -----
    """
    
    if SAVE_VOLUMES: print(f"[save] Volumes saved under {OUT_DIR}")
    if SAVE_PER_XL:  print(f"[save] Per-XL slices written under {OUT_DIR}")
    print("\n[done] 3D joint PP–PS inversion (tiled, C-order, v3) complete.")

if __name__ == "__main__":
    try:
        _ = PrestackLinearModelling(np.array([1.0]), np.array([25.0]), vsvp=0.5, nt0=10, linearization=LINEARIZATION_PS)
    except Exception as e:
        raise RuntimeError("PyLops 'ps' linearization is not available (or failed to init).\n"
                           f"Original error: {e}\nTip: upgrade PyLops (>=2.5) or provide a custom callable.")
    main()
