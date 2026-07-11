"""Minimal aeropad demo using the bundled synthetic dataset.

Run from the repository root::

    python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from aeropad import (
    AirfoilSpec, CaseConfig, DataSpec, FlowSpec,
    recommend, reconstruct_polar,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "sample_data",
                   "synthetic_cambered_polar.csv")

df = pd.read_csv(CSV)

config = CaseConfig(
    airfoil=AirfoilSpec(
        name="synthetic_cambered", symmetry="asymmetric",
        TC=0.117, alpha_s_2D_pos=15.0, CL_s_2D_pos=1.45,
        alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.85,
        delta_nose_deg=15.0, delta_tail_deg=7.0),
    flow=FlowSpec(Re=2.0e6, M=0.16),
    data=DataSpec(LF_CL_column="CL_PM", LF_CD_column="CD_PM",
                  HF_CL_column="CL_CFD", HF_CD_column="CD_CFD"),
)

# 1 · Ask the advisor what it would do at different CFD budgets
for budget in (0, 19):
    adv = recommend(config, hf_budget=budget)
    print(f"budget={budget:>2d} -> {adv['route']}: {adv['notes']}")

# 2 · Reconstruct with the auto route (advisor decides from the data)
result = reconstruct_polar(df, config, route="auto")
print()
print(result.summary())

# 3 · Export the reconstructed full-range polar
out = os.path.join(HERE, "reconstructed_polar.csv")
result.polar.to_csv(out, index=False)
print(f"\nWrote {out}")
