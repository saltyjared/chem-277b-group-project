#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthotropic-Elastic Post-Processor (robust SVD and links)
=========================================================
Purpose
-------
Process an Excel sheet of orthotropic stiffness entries C11,C22,C33,C12,C13,C23,C44,C55,C66
(one row per material) and:

1) Build the 6×6 stiffness matrix C in Voigt engineering order (11,22,33,23,13,12) for an
   orthotropic solid (no normal–shear couplings). Orthotropy/Voigt reference:
   https://nilspv.folk.ntnu.no/TMM4175/hookes-law.html

2) Symmetrise C, compute the 2‑norm condition number cond_C, and invert to compliance S:
   • If well‑conditioned → exact inverse
   • Else → Moore–Penrose pseudoinverse with robust SVD settings, flagged by pinv_debug=1
   Condition‑number background: https://cs357.cs.illinois.edu/textbook/notes/condition.html

3) Extract engineering constants from S (Voigt convention): E1,E2,E3; G23,G13,G12; ν12,ν13,ν23
   and reciprocals ν21,ν31,ν32. See orthotropic overview:
   https://nilspv.folk.ntnu.no/TMM4175/hookes-law.html

4) Compute Voigt/Reuss bounds and universal anisotropy index:
   AU = 5*GV/GR + BV/BR − 6, with formulas from:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9413398/

5) Write two CSVs (17‑digit precision), overwriting on each run:
   • compliance_and_engineering_constraints.csv   (full dataset)
   • engineering_constraints.csv                  (pass‑through + constants)

Debug tail (last four columns): cond_C_debug, sym_resid_debug, invert_warn_debug, pinv_debug
"""

import re, numpy as np, pandas as pd
from pathlib import Path

# ---------------- user settings ----------------
IN_FILE  = Path("input_data_full.xlsx")
SHEET    = 0
OUT_DIR  = Path(".")
COND_MAX = 1e12           # switch to pinv if cond(C) > COND_MAX
SYM_TOL  = 1e-10          # symmetry warning tolerance
FLOAT_FMT = "%.17g"       # CSV float precision
PINV_RCOND = 1e-12        # base rcond for SVD pseudoinverse
# ------------------------------------------------

# Exact legacy prefix from your HTML (do NOT edit it)
OLD_STR = """ρ,1x1,1x2,2x2,1x3,2x3,3x3,1x4,2x4,3x4,4x4,1x5,2x5,3x5,4x5,5x5,1x6,2x6,3x6,4x6,5x6,6x6,1x7,2x7,3x7,4x7,5x7,6x7,7x7,1x8,2x8,3x8,4x8,5x8,6x8,7x8,8x8,1x9,2x9,3x9,4x9,5x9,6x9,7x9,8x9,9x9,1x10,2x10,3x10,4x10,5x10,6x10,7x10,8x10,9x10,10x10,1x11,2x11,3x11,4x11,5x11,6x11,7x11,8x11,9x11,10x11,11x11,S11,S12,S13,S14,S15,S16,S21,S22,S23,S24,S25,S26,S31,S32,S33,S34,S35,S36,S41,S42,S43,S44,S45,S46,S51,S52,S53,S54,S55,S56,S61,S62,S63,S64,S65,S66,E1,E2,E3,G23,G13,G12,nu12,nu13,nu23,nu21,nu31,nu32,cond_C,sym_resid,invert_warn"""
OLD_PREFIX = OLD_STR.split(",")

# ---------- helpers ----------
def is_nxn(name: str) -> bool:
    return bool(re.fullmatch(r"\d+x\d+", str(name)))

C_LABELS = [f"C{i}{j}" for i in range(1,7) for j in range(1,7)]
REQUIRED_ORTHO = ("C11","C22","C33","C12","C13","C23","C44","C55","C66")

def passthrough_cols(df):
    return [c for c in df.columns
            if (c not in C_LABELS) or str(c).strip() == "ρ" or is_nxn(c)]

def to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def build_C_orthotropic(row):
    # NaN for any missing/invalid required entries (never default to zero)
    C = np.full((6,6), np.nan, float)
    v = {k: to_float_or_nan(row.get(k, np.nan)) for k in REQUIRED_ORTHO}
    if np.isfinite(v["C11"]): C[0,0] = v["C11"]
    if np.isfinite(v["C22"]): C[1,1] = v["C22"]
    if np.isfinite(v["C33"]): C[2,2] = v["C33"]
    if np.isfinite(v["C12"]): C[0,1] = C[1,0] = v["C12"]
    if np.isfinite(v["C13"]): C[0,2] = C[2,0] = v["C13"]
    if np.isfinite(v["C23"]): C[1,2] = C[2,1] = v["C23"]
    if np.isfinite(v["C44"]): C[3,3] = v["C44"]
    if np.isfinite(v["C55"]): C[4,4] = v["C55"]
    if np.isfinite(v["C66"]): C[5,5] = v["C66"]
    return C

def robust_pinv(A, base_rcond=PINV_RCOND):#Dont worry as much about this, this is just if we have a very messed up geometry and will give a pseudoinverse and ping it as bad in the output. We have yet to find a structure with this issue
    # Replace NaNs/Infs that can crash SVD; keep symmetry by fixing diagonal first
    A = A.copy()
    bad = ~np.isfinite(A)
    if bad.any():
        # push bad diagonals to a large positive number to retain SPD-ish behavior
        for i in range(6):
            if bad[i, i]:
                A[i, i] = 1e30
        # remaining bad off-diagonals to 0 to avoid contaminating SVD
        for i in range(6):
            for j in range(6):
                if bad[i, j] and i != j:
                    A[i, j] = 0.0
    # Symmetrise before SVD
    A = 0.5*(A + A.T)
    # Compute scale-aware rcond (helps when units scale)
    try:
        smax = np.linalg.svd(A, compute_uv=False, hermitian=True)[0]
        rcond = base_rcond if smax == 0 or not np.isfinite(smax) else base_rcond * smax
    except np.linalg.LinAlgError:
        rcond = base_rcond
    # Try hermitian pinv first, then generic pinv if needed
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False, hermitian=True)
    except Exception:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.array([1/si if si > rcond else 0.0 for si in s], dtype=float)
    return (Vt.T * s_inv) @ U.T

def invert_with_flags(C):
    Csym = 0.5*(C + C.T)
    sym_resid = np.linalg.norm(Csym - C) / max(1e-16, np.linalg.norm(Csym))
    try:
        condC = np.linalg.cond(Csym)
    except Exception:
        condC = np.inf
    # Use exact inverse when safe
    if np.isfinite(condC) and condC <= COND_MAX:
        try:
            S = np.linalg.inv(Csym)
            return S, condC, sym_resid, (sym_resid > SYM_TOL), 0
        except np.linalg.LinAlgError:
            pass
    # Otherwise robust pseudoinverse
    try:
        S = robust_pinv(Csym, base_rcond=PINV_RCOND)
        return S, condC, sym_resid, True, 1
    except Exception:
        # last-resort generic pinv
        S = np.linalg.pinv(Csym, rcond=PINV_RCOND)
        return S, condC, sym_resid, True, 1

def constants_from_S(S):
    E1,E2,E3 = 1/S[0,0], 1/S[1,1], 1/S[2,2]
    G23,G13,G12 = 1/S[3,3], 1/S[4,4], 1/S[5,5]
    nu12,nu13,nu23 = -S[0,1]/S[0,0], -S[0,2]/S[0,0], -S[1,2]/S[1,1]
    nu21,nu31,nu32 = -S[1,0]/S[1,1], -S[2,0]/S[2,2], -S[2,1]/S[2,2]
    return dict(E1=E1,E2=E2,E3=E3,G23=G23,G13=G13,G12=G12,
                nu12=nu12,nu13=nu13,nu23=nu23,nu21=nu21,nu31=nu31,nu32=nu32)

def voigt_reuss_AU(C, S):
    BV = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2]))/9.0
    GV = ((C[0,0]+C[1,1]+C[2,2] - C[0,1]-C[0,2]-C[1,2])/15.0
          + (C[3,3]+C[4,4]+C[5,5])/5.0)
    BR = 1.0/(S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))
    GR = 15.0/(4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2])
               + 3*(S[3,3]+S[4,4]+S[5,5]))
    AU = 5.0*(GV/GR) + (BV/BR) - 6.0
    return BV,GV,BR,GR,AU

# ---------- main ----------
df = pd.read_excel(IN_FILE, sheet_name=SHEET, engine="openpyxl")
pass_cols = passthrough_cols(df)

rows_full, rows_consts = [], []
for _, r in df.iterrows():
    C = build_C_orthotropic(r)
    S, condC, sym_resid, invert_warn, pinv_flag = invert_with_flags(C)
    consts = constants_from_S(S)
    BV,GV,BR,GR,AU = voigt_reuss_AU(C, S)

    base = {c: r[c] for c in pass_cols}
    for i in range(6):
        for j in range(6):
            base[f"S{i+1}{j+1}"] = S[i,j]
    base.update(consts)
    base.update(dict(BV=BV, GV=GV, BR=BR, GR=GR, AU=AU,
                     cond_C_debug=condC, sym_resid_debug=sym_resid,
                     invert_warn_debug=invert_warn, pinv_debug=pinv_flag))
    rows_full.append(base)

    cview = {c: r[c] for c in pass_cols}
    cview.update(consts)
    cview.update(dict(BV=BV, GV=GV, BR=BR, GR=GR, AU=AU,
                      cond_C_debug=condC, sym_resid_debug=sym_resid,
                      invert_warn_debug=invert_warn, pinv_debug=pinv_flag))
    rows_consts.append(cview)

# Column order: legacy prefix, then BV..AU, then debug tail
S_cols   = [f"S{i}{j}" for i in range(1,7) for j in range(1,7)]
CONST_L  = ["E1","E2","E3","G23","G13","G12","nu12","nu13","nu23","nu21","nu31","nu32"]
ANISO    = ["BV","GV","BR","GR","AU"]
DEBUG_TAIL = ["cond_C_debug","sym_resid_debug","invert_warn_debug","pinv_debug"]

ordered_prefix = [c for c in OLD_PREFIX if c in (pass_cols + S_cols + CONST_L)]
full_cols  = ordered_prefix + [c for c in (S_cols + CONST_L + ANISO) if c not in ordered_prefix] + DEBUG_TAIL
const_cols = [c for c in pass_cols] + CONST_L + ANISO + DEBUG_TAIL

# Filenames and write
OUT_DIR.mkdir(parents=True, exist_ok=True)
full_path  = OUT_DIR / "compliance_and_engineering_constraints.csv"
const_path = OUT_DIR / "engineering_constraints.csv"

pd.DataFrame(rows_full ).reindex(columns=full_cols ).to_csv(full_path,  index=False, float_format=FLOAT_FMT)
pd.DataFrame(rows_consts).reindex(columns=const_cols).to_csv(const_path, index=False, float_format=FLOAT_FMT)

print(f"Wrote {full_path.resolve()} and {const_path.resolve()} (precision {FLOAT_FMT})")
