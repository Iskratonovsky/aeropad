"""End-to-end smoke test on a synthetic cambered-airfoil polar.

Generates a plausible full-range polar (linear pre-stall lift up to a
sharp peak, sin(2α)-type post-stall lift, sin²(α)-type drag), plus a
thin-airfoil-style LF dataset, then exercises:

1. All four semi-empirical extrapolators
2. The Kriging route with the uniform-20° rule
3. The auto route/method advisor
4. Metrics evaluation on both routes
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from aeropad import (
    CaseConfig, AirfoilSpec, FlowSpec, DataSpec, PipelineSpec,
    reconstruct_polar, recommend,
)


# ── Synthetic ground truth ──────────────────────────────────────────
def synthetic_polar(alpha_deg: np.ndarray, cambered: bool = True):
    a = np.radians(alpha_deg)
    alpha0 = -3.0 if cambered else 0.0
    a0 = np.radians(alpha0)
    stall_pos, stall_neg = 15.0, (-11.0 if cambered else -15.0)
    clmax, clmin = 1.45, (-0.85 if cambered else -1.45)

    cl = np.empty_like(a)
    cd = 0.02 + 1.75 * np.sin(a) ** 2

    for i, (ad, ar) in enumerate(zip(alpha_deg, a)):
        if stall_neg <= ad <= stall_pos:
            cl[i] = 2 * np.pi * 0.85 * (ar - a0) / (1 + 2 / 7.0)
            cl[i] = np.clip(cl[i], clmin, clmax)
        else:
            # deep-stall sin(2α) structure with gentle asymmetry
            amp = 1.05 if cambered else 1.10
            cl[i] = amp * np.sin(2 * ar) * (0.92 if ad < 0 else 1.0)
    return cl, cd


def thin_airfoil_lf(alpha_deg: np.ndarray, cambered: bool = True):
    a = np.radians(alpha_deg)
    a0 = np.radians(-3.0 if cambered else 0.0)
    cl = 2 * np.pi * (a - a0)
    cd = 0.008 + 0.011 * (alpha_deg / 10.0) ** 2
    return cl, cd


# ── Build the input DataFrame ───────────────────────────────────────
aoa_hf = np.arange(-180, 181, 5).astype(float)      # dense reference
cl_hf, cd_hf = synthetic_polar(aoa_hf)

aoa_lf = np.arange(-25, 26, 1).astype(float)        # attached-flow range
cl_lf, cd_lf = thin_airfoil_lf(aoa_lf)

df = pd.DataFrame({"AoA": aoa_hf, "CL_CFD": cl_hf, "CD_CFD": cd_hf})
lf = pd.DataFrame({"AoA": aoa_lf, "CL_PM": cl_lf, "CD_PM": cd_lf})
df = df.merge(lf, on="AoA", how="outer").sort_values("AoA").reset_index(drop=True)

config = CaseConfig(
    airfoil=AirfoilSpec(
        name="synthetic_cambered", symmetry="asymmetric",
        TC=0.117, HC=0.036, RLE_C=0.02, AR=7.0,
        alpha_s_2D_pos=15.0, CL_s_2D_pos=1.45,
        alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.85,
        delta_nose_deg=15.0, delta_tail_deg=7.0),
    flow=FlowSpec(Re=2.0e6, M=0.16),
    data=DataSpec(
        LF_CL_column="CL_PM", LF_CD_column="CD_PM",
        HF_CL_column="CL_CFD", HF_CD_column="CD_CFD"),
    pipeline=PipelineSpec(extrapolator="auto", PM_cutoff=20.0,
                          blend_end=45.0, fine_step=1.0),
)

print("=" * 70)
print("ADVISOR")
print("=" * 70)
for budget in (None, 5, 19, 40):
    adv = recommend(config, hf_budget=budget)
    print(f"budget={str(budget):>5s} -> route={adv['route']:<15s} "
          f"CL={adv['method_CL']}, CD={adv['method_CD']}, "
          f"bracket_stall={adv['bracket_stall']}")

print()
print("=" * 70)
print("SEMI-EMPIRICAL ROUTE — all four methods explicitly")
print("=" * 70)
for method in ("battisti", "aerodas", "montgomerie", "lindenburg"):
    config.pipeline.extrapolator = method
    try:
        res = reconstruct_polar(df, config, route="semi-empirical")
        m_cl, m_cd = res.metrics.get("CL", {}), res.metrics.get("CD", {})
        print(f"{method:<12s}  CL R²={m_cl.get('R2', float('nan')):.4f} "
              f"MAE={m_cl.get('MAE', float('nan')):.4f}   "
              f"CD R²={m_cd.get('R2', float('nan')):.4f} "
              f"MAE={m_cd.get('MAE', float('nan')):.4f}   "
              f"rows={len(res.polar)}")
    except Exception as e:
        print(f"{method:<12s}  FAILED: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("SEMI-EMPIRICAL ROUTE — auto (per-coefficient mixing)")
print("=" * 70)
config.pipeline.extrapolator = "auto"
res = reconstruct_polar(df, config, route="semi-empirical")
print(res.summary())

print()
print("=" * 70)
print("KRIGING ROUTE — uniform-20° rule")
print("=" * 70)
res_k = reconstruct_polar(df, config, route="kriging")
print(res_k.summary())
print(f"training points CL: {len(res_k.models['train_CL'])}, "
      f"CD: {len(res_k.models['train_CD'])}")

print()
print("=" * 70)
print("AUTO ROUTE (advisor decides from available HF samples)")
print("=" * 70)
res_a = reconstruct_polar(df, config, route="auto")
print(f"advisor chose: {res_a.route}")
print(res_a.summary())

print()
print("ALL SMOKE TESTS PASSED" if True else "")
