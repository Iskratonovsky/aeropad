"""Validate the aeropad port against the dissertation reference numbers.

Semi-empirical: 4 methods x 10 cases x {CL, CD} vs standalone_metrics.csv
Kriging:        10 cases x {CL, CD}          vs gpr_clarky_final.csv / gpr_naca_final.csv

Any |ΔR²| or |ΔMAE| above 5e-3 is flagged.
"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

from aeropad import (CaseConfig, AirfoilSpec, FlowSpec, DataSpec,
                     PipelineSpec, reconstruct_polar)

TOL = 5e-3
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'sample_data')
REF = os.path.join(HERE, '..', 'benchmark')

CLARK_RE = {0.108: 1898100, 0.144: 2530632, 0.160: 2817138}
CLARK_FILES = {0.108: 'clark_y_Ma0108.csv', 0.144: 'clark_y_Ma014.csv',
               0.160: 'clark_y_Ma016.csv'}


def clark_config(Ma, method):
    return CaseConfig(
        airfoil=AirfoilSpec(
            name=f"Clark_Y_M{Ma}", symmetry="asymmetric",
            TC=0.117, HC=0.036, RLE_C=0.02, AR=1.548,
            alpha_s_2D_pos=15.0, CL_s_2D_pos=1.52,
            alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.78,
            delta_nose_deg=15.08, delta_tail_deg=7.13),
        flow=FlowSpec(Re=CLARK_RE[Ma], M=Ma),
        data=DataSpec(LF_CL_column="CL_PM", LF_CD_column="CD_PM",
                      HF_CL_column="CL_CFD", HF_CD_column="CD_CFD",
                      flip_aoa_sign=True),
        pipeline=PipelineSpec(extrapolator=method, PM_cutoff=25.0,
                              blend_end=45.0, fine_step=1.0))


def naca_config(Ma, method):
    return CaseConfig(
        airfoil=AirfoilSpec(
            name=f"NACA0012_M{Ma}", symmetry="symmetric",
            TC=0.12, HC=0.0, RLE_C=0.0158,
            alpha_s_2D_pos=15.0, CL_s_2D_pos=1.52,
            alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.78),
        flow=FlowSpec(Re=2_000_000, M=Ma),
        data=DataSpec(LF_CL_column="CL_XFOIL", LF_CD_column="CD_XFOIL",
                      HF_CL_column="CL_CFD", HF_CD_column="CD_CFD",
                      flip_aoa_sign=False),
        pipeline=PipelineSpec(extrapolator=method, PM_cutoff=19.25,
                              blend_end=45.0, fine_step=1.0))


def load(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    return df


CASES = [('Clark Y', Ma, CLARK_FILES[Ma], clark_config)
         for Ma in (0.108, 0.144, 0.160)]
CASES += [('NACA 0012', round(0.1 * i, 1), f'naca_0012_m0{i}.csv', naca_config)
          for i in range(1, 8)]

# ── 1 · Semi-empirical validation ───────────────────────────────────
ref_se = pd.read_csv(os.path.join(REF, 'standalone_metrics.csv'))
fails, checked = [], 0

print("=" * 78)
print("SEMI-EMPIRICAL VALIDATION (4 methods × 10 cases × CL/CD = 80 metrics pairs)")
print("=" * 78)
for airfoil, Ma, fname, cfg_fn in tqdm(CASES, desc='semi-empirical cases'):
    df = load(fname)
    for method in ('battisti', 'aerodas', 'montgomerie', 'lindenburg'):
        config = cfg_fn(Ma, method)
        res = reconstruct_polar(df, config, route="semi-empirical")
        for coeff in ('CL', 'CD'):
            got = res.metrics.get(coeff, {})
            ref_row = ref_se[(ref_se.airfoil == airfoil)
                             & (np.isclose(ref_se.Ma, Ma))
                             & (ref_se.method == method)
                             & (ref_se.coefficient == coeff)]
            if ref_row.empty:
                continue
            r = ref_row.iloc[0]
            dr2 = abs(got.get('R2', np.nan) - r.R2)
            dmae = abs(got.get('MAE', np.nan) - r.MAE)
            checked += 1
            if not (dr2 < TOL and dmae < TOL):
                fails.append((airfoil, Ma, method, coeff,
                              got.get('R2'), r.R2, dr2, dmae))

print(f"checked {checked} metric pairs "
      f"({len(fails)} outside tolerance {TOL})")
for f in fails:
    print("  MISMATCH:", f)

# ── 2 · Kriging validation ──────────────────────────────────────────
print()
print("=" * 78)
print("KRIGING VALIDATION (10 cases × CL/CD = 20 metric pairs)")
print("=" * 78)
ref_k = pd.concat([
    pd.read_csv(os.path.join(REF, 'gpr_clarky_final.csv')),
    pd.read_csv(os.path.join(REF, 'gpr_naca_final.csv')),
])
kfails, kchecked = [], 0

for airfoil, Ma, fname, cfg_fn in tqdm(CASES, desc='kriging cases'):
    df = load(fname)
    # GPR route is flip-neutral (stationary kernels); run unflipped to
    # match the original scripts' fold assignments exactly.
    config = cfg_fn(Ma, 'auto')
    config.data.flip_aoa_sign = False
    res = reconstruct_polar(df, config, route="kriging",
                            bracket_stall=False)
    label = (f"Clark Y, Ma={Ma:.3f}" if airfoil == 'Clark Y'
             else f"NACA 0012, Ma={Ma}")
    for coeff, col in (('CL', 'CL_CFD'), ('CD', 'CD_CFD')):
        got = res.metrics.get(coeff, {})
        ref_row = ref_k[(ref_k.case == label) & (ref_k.coefficient == col)]
        if ref_row.empty:
            # try alternative Ma formatting
            alt = f"Clark Y, Ma={Ma}" if airfoil == 'Clark Y' else label
            ref_row = ref_k[(ref_k.case == alt) & (ref_k.coefficient == col)]
        if ref_row.empty:
            print(f"  no ref row for {label} {col}")
            continue
        r = ref_row.iloc[0]
        dr2 = abs(got.get('R2', np.nan) - r.R2)
        dmae = abs(got.get('MAE', np.nan) - r.MAE)
        kchecked += 1
        flag = "" if (dr2 < TOL and dmae < TOL) else "  <-- MISMATCH"
        print(f"  {label:<22s} {coeff}: aeropad R²={got.get('R2', float('nan')):.4f} "
              f"ref={r.R2:.4f} (Δ={dr2:.1e})  "
              f"[{got.get('protocol','?')}]{flag}")
        if flag:
            kfails.append((label, coeff, got.get('R2'), r.R2))

print()
print(f"kriging: checked {kchecked} pairs, {len(kfails)} mismatches")
print()
print("VALIDATION " + ("PASSED" if not fails and not kfails else "HAS MISMATCHES"))
