# aeropad

**AEROnautical Preliminary & conceptual Aircraft Design toolkit**

A modular Python toolkit for conceptual and preliminary aircraft design
assistance. The first module, `aeropad.polar`, reconstructs full-range
(±180°) aerodynamic polars from the limited data typically available in
early design phases — when dense high-fidelity CFD sweeps are
unaffordable but full-envelope coefficient data is still required for
load computation, stability analysis, and simulation.

## The problem

Aerodynamic coefficients across the full ±180° angle-of-attack range are
required for reliable load computation in rotorcraft and fixed-wing
applications, yet CFD coverage of this range at 5° resolution costs ~72
RANS solutions *per operating condition*. Early design phases cannot
afford this, and attached-flow tools (panel methods, XFOIL) only cover
roughly ±20°.

## Two reconstruction routes

`aeropad.polar` offers two independent routes, each validated against
dense RANS CFD references on two airfoils of contrasting character
(a cambered finite wing and a symmetric 2D section) across ten
(airfoil, Mach) conditions:

| Route | Input required | Typical accuracy |
|---|---|---|
| **Semi-empirical extrapolation** | LF attached-flow data + geometry (zero CFD) | R² ≈ 0.88–0.97 |
| **Kriging surrogate** | 19 CFD points (uniform-20° rule); 10 for symmetric airfoils | R² ≥ 0.99 on smooth polars |

Four semi-empirical methods are implemented — **AERODAS** (Spera 2008),
**Montgomerie** (2004), **Lindenburg** (2003), and **Battisti et al.**
(2020) — with per-case best-method selection built in: cambered
profiles favour Montgomerie; symmetric profiles favour AERODAS for drag
and Battisti (low Mach) or AERODAS (high Mach) for lift.

The Kriging route implements the **uniform-20° sampling rule**: 19
training stations reconstruct the full polar at ~26% of the CFD cost of
a 5° sweep, halved again to 10 unique evaluations for symmetric
airfoils via geometric mirroring. Known limitation: sharp low-Mach
lift-stall peaks fall between stations; the built-in advisor prescribes
two supplementary stations at ±α_stall for those conditions.

**Why not combine the routes?** Stacking them — e.g. using a
semi-empirical composite as a prior for the surrogate — compounds the
error sources of both stages: biases in the empirical correlations
propagate into the surrogate where they can no longer be diagnosed
against the reconstruction, degrading trustworthiness precisely in the
data-poor settings this toolkit targets. `aeropad` treats **route
selection**, not route stacking, as the design decision, and automates
it with a budget- and regime-aware advisor.

## Quick start (Python API)

```python
import pandas as pd
from aeropad import CaseConfig, AirfoilSpec, FlowSpec, DataSpec, reconstruct_polar

df = pd.read_csv("my_polar.csv")   # AoA + LF columns (+ optional CFD samples)

config = CaseConfig(
    airfoil=AirfoilSpec(symmetry="asymmetric", TC=0.117,
                        alpha_s_2D_pos=15.0, CL_s_2D_pos=1.52,
                        alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.78,
                        delta_nose_deg=15.1, delta_tail_deg=7.1),
    flow=FlowSpec(Re=2.8e6, M=0.16),
    data=DataSpec(LF_CL_column="CL_PM", LF_CD_column="CD_PM",
                  HF_CL_column="CL_CFD", HF_CD_column="CD_CFD"),
)

result = reconstruct_polar(df, config, route="auto")
print(result.summary())
result.polar.to_csv("full_polar.csv", index=False)
```

### Airfoil geometry from coordinate files

The extrapolators need geometric parameters (thickness, camber, LE
radius, nose/tail camber angles) that are tedious to measure by hand.
`aeropad.polar.geometry.analyze_dat("clark_y.dat")` computes all of
them from a standard Selig/Lednicer coordinate file (validated against
analytic NACA 4-digit values), and the GUI's "Load geometry from .dat"
button fills the airfoil fields directly.

## Quick start (GUI)

```bash
pip install -e .
aeropad-gui          # or: python -m aeropad.gui.app
```

Load a polar CSV, map columns (auto-detected for common naming), set
airfoil and flow parameters, pick a route or use **Advise**, and hit
**Run**. Export the reconstructed polar as CSV or the figure as PNG.

A synthetic demo dataset ships in `sample_data/`.

## Module 2 — `aeropad.sizing`: aircraft sizing by statistics

Statistical sizing relations over historical aircraft databases, in
the spirit of sizing-by-statistics methods for rapid conceptual
design. Seven regression families behind one tune–fit–evaluate
interface — polynomial, ridge, lasso, RBF kernel ridge, power law
(the classical closed-form sizing relation), Gaussian process, and
PySR symbolic regression — each tuned leakage-safe and evaluated on a
common held-out split. GPR and symbolic regression are the flagship
pair: GPR delivers the highest accuracy with implicit models, while
symbolic regression discovers explicit empirical equations suitable
for standardisation and further engineering development. The symbolic
family requires PySR (`pip install pysr`; Julia backend auto-installs
on first use) and is skipped gracefully where unavailable.

```python
from aeropad.sizing import compare_families, correlation_heatmap

fig, corr = correlation_heatmap(df)          # explore the database
table, models = compare_families(df,          # benchmark all families
    features=["MTOW", "Cruising speed"], target="Disc Loading")
print(models["power_law"].equation())
# Disc Loading = 0.05438 * MTOW^0.3012 * Cruising speed^0.7083
models["gpr"].predict({"MTOW": 5000, "Cruising speed": 250})
```

Diagnostics ship alongside: correlation heatmaps (dataset
exploration), hyperparameter tuning curves/heatmaps, 3D prediction
surfaces, actual-vs-predicted plots, and learning curves. Bundled demonstration datasets: a 225-entry rotorcraft database (cleaned), a 277-entry detailed rotorcraft database, and a 49-aircraft fighter starter dataset spanning piston, turboprop and jet eras (public-domain figures — see its README for verification caveats). The module is fully dataset-agnostic.

### Dataset guarding and results export (`aeropad.sizing.dataio`)

Real aircraft databases fail in recurring ways; the audit utilities
encode the fixes: unit-contamination detection and repair (mm values
mixed into metre columns), collinearity screening with VIFs (a second
predictor at |r| ≈ 0.9 can *worsen* hold-out accuracy), propulsion-aware
missing-by-design vs missing-by-error classification, and a hold-out
usability screen that flags targets which resist statistical
prediction as documented negative results. `export_results` writes
fitted relations — explicit equations included, symbolic regression's
especially — and metrics to CSV. `audit_dataset` bundles the guards
into one report.

The GUI (`aeropad-gui`) exposes both modules as tabs: polar
reconstruction, and sizing by statistics (dataset load, correlation
heatmap, family fitting/comparison, and point prediction).

## Roadmap

- Fixed-wing sizing database.
- Pitching-moment (CM) reconstruction.
- Joint surrogates over (Mach, Reynolds, α) for parametric studies.

## References

- Battisti, L. et al. (2020). *Wind Turbines in Cold Climates.*
- Spera, D. (2008). NASA/CR-2008-215434 (AERODAS).
- Montgomerie, B. (2004). FOI-R--1305--SE.
- Lindenburg, C. (2003). ECN-C--03-025.
