"""aeropad.sizing demo — detailed rotorcraft dataset, multi-feature.

Workflow: correlation heatmap -> multi-feature family comparison
(rotor radius = f(MTOW, cruise speed, tail blade count)) -> classic
two-feature power-law radius relation with 3D surface.

The 'symbolic' family (PySR) joins the comparison automatically when
a working PySR/Julia backend is detected.
"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from aeropad.sizing import (
    SizingModel, compare_families, correlation_heatmap,
    prediction_surface, actual_vs_predicted,
)
from aeropad.sizing.models import FAMILIES, symbolic_available

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sizing_output")
os.makedirs(OUT, exist_ok=True)

raw = pd.read_csv(os.path.join(HERE, "..", "sample_data",
                               "rotorcraft_dataset_jane_detailed.csv"))
df = raw.rename(columns={
    "MTOW (kg)": "MTOW",
    "Main rotor radius (m)": "Rotor radius",
    "Disc loading (kg/m2)": "Disc loading",
    "Cruising speed (km/h)": "Cruising speed",
    "Tail No. Blades": "Tail blades",
})
num_cols = ["MTOW", "Rotor radius", "Disc loading",
            "Cruising speed", "Tail blades"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"Dataset: {len(df)} rotorcraft (detailed)\n")

# 1 - Correlation heatmap over all numeric sizing parameters
fig, corr = correlation_heatmap(
    df, columns=num_cols,
    save=os.path.join(OUT, "correlation_heatmap_detailed.png"))
print("Correlation matrix:")
print(corr.round(3).to_string(), "\n")

# 2 - Multi-feature comparison: radius = f(MTOW, cruise speed, tail blades)
features = ["MTOW", "Cruising speed", "Tail blades"]
target = "Rotor radius"
fams = [f for f in FAMILIES
        if f not in ("power_law",)               # tail blades has zeros
        and (f != "symbolic" or symbolic_available())]
table, models = compare_families(df, features, target, families=fams)
cols = ["R2_train", "R2_test", "RMSE_test", "MAE_test", "best_params"]
print("=" * 100)
print(f"FAMILY COMPARISON: {target} = f({', '.join(features)})")
print("=" * 100)
print(table[cols].round(4).to_string(), "\n")
table.to_csv(os.path.join(OUT, "family_comparison_detailed.csv"))

# 3 - Classic two-feature power-law radius relation (Rand-style)
m = SizingModel("power_law").fit(df, ["MTOW", "Cruising speed"], target)
print("Two-feature power-law radius relation:")
print(" ", m.equation())
print(f"  R2_test = {m.metrics['R2_test']:.4f}\n")
prediction_surface(m, df,
                   save=os.path.join(OUT, "radius_powerlaw_surface.png"))
actual_vs_predicted(m, save=os.path.join(OUT, "radius_powerlaw_avp.png"))

# 4 - Point prediction
q = {"MTOW": 5000.0, "Cruising speed": 250.0}
print(f"Predicted rotor radius for {q}: {m.predict(q)[0]:.2f} m")
print(f"\nOutputs -> {OUT}/")
