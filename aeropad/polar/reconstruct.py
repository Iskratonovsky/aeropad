"""
aeropad.polar.reconstruct
=========================
Unified full-range polar reconstruction: one entry point, two routes.

**Route 1 — semi-empirical extrapolation.** Requires only low-fidelity
attached-flow data (panel method, XFOIL) plus airfoil geometry. Zero
high-fidelity CFD needed. Four methods available; the per-case best
choice is encoded in :func:`recommend` from a systematic benchmark
across two airfoil families and ten Mach conditions.

**Route 2 — Kriging surrogate.** Requires a modest high-fidelity budget
(the uniform-20° rule: 19 CFD evaluations per condition, halved to 10
for symmetric airfoils). Delivers R² ≥ 0.99 wherever the polar carries
no sharp features narrower than the station spacing.

The two routes are deliberately kept **independent**. Stacking them —
e.g. using a semi-empirical composite as a prior or trend for the
surrogate — compounds the error sources of both stages: biases in the
empirical correlations propagate into the surrogate and can no longer
be diagnosed against the reconstruction, degrading trustworthiness in
exactly the preliminary-design settings where no dense reference exists
to catch the drift. aeropad therefore treats route selection, not route
stacking, as the design decision — see :func:`recommend`.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CaseConfig
from .extrapolation import get_extrapolator
from .kriging import (
    KrigingReconstructor, recommended_stations, sample_training,
    mirror_symmetric, loo_cv,
)
from .metrics import evaluate_curve


# ── Advisor ─────────────────────────────────────────────────────────

#: Mach number above which compressibility broadens the lift-stall peak
#: enough for the uniform-20° grid to resolve it (Regime 1 → Regime 2
#: transition observed between Ma 0.4 and 0.5 on NACA 0012 at Re = 2e6).
REGIME_CROSSOVER_MACH = 0.45


def _default_methods(config: CaseConfig) -> tuple:
    """Per-case best semi-empirical methods (CL, CD) by airfoil family.

    Encodes the per-case benchmark winners: cambered profiles →
    Montgomerie for both coefficients; symmetric profiles → AERODAS for
    CD at all Mach, Battisti for CL at low Mach with AERODAS above the
    regime crossover.
    """
    sym = config.airfoil.symmetry == "symmetric"
    low_mach = config.flow.M <= REGIME_CROSSOVER_MACH
    if sym:
        return ("battisti" if low_mach else "aerodas", "aerodas")
    return ("montgomerie", "montgomerie")


def recommend(config: CaseConfig,
              hf_budget: Optional[int] = None) -> dict:
    """Recommend a reconstruction route and method for a case.

    Encodes the operating-regime findings of the underlying benchmark:

    - **No HF budget** → semi-empirical route. Method by airfoil family:
      cambered profiles → Montgomerie (most consistent across cambered
      test conditions); symmetric profiles → AERODAS for CD at all Mach,
      Battisti for CL at low Mach (≤ ~0.45) and AERODAS above.
    - **HF budget ≥ 10** → Kriging route with the uniform-20° rule
      (10 unique evaluations suffice for symmetric airfoils via
      mirroring; 19 otherwise). At low Mach, two supplementary stations
      at ±α_stall are advised to bracket the sharp lift peak that
      otherwise falls between the 0° and ±20° stations.

    Parameters
    ----------
    config : CaseConfig
    hf_budget : int, optional
        Number of high-fidelity CFD evaluations available for this
        condition. ``None`` or 0 means no HF budget.

    Returns
    -------
    dict with keys ``route``, ``method_CL``, ``method_CD``,
    ``bracket_stall``, ``notes``.
    """
    sym = config.airfoil.symmetry == "symmetric"
    mach = config.flow.M
    low_mach = mach <= REGIME_CROSSOVER_MACH
    min_budget = 10 if sym else 19

    if hf_budget is None or hf_budget < min_budget:
        method_cl, method_cd = _default_methods(config)
        return {
            "route": "semi-empirical",
            "method_CL": method_cl,
            "method_CD": method_cd,
            "bracket_stall": False,
            "notes": (
                f"HF budget ({hf_budget or 0}) below the uniform-20° "
                f"requirement ({min_budget} for this airfoil); "
                f"semi-empirical route selected."),
        }

    return {
        "route": "kriging",
        "method_CL": None,
        "method_CD": None,
        "bracket_stall": low_mach,
        "notes": (
            "HF budget sufficient for the uniform-20° Kriging rule."
            + (" Low-Mach condition: supplementary ±stall stations "
               "advised to bracket the sharp lift peak (Regime 1)."
               if low_mach else "")),
    }


# ── Result container ────────────────────────────────────────────────

@dataclass
class PolarResult:
    """Reconstructed full-range polar plus provenance and diagnostics."""
    polar: pd.DataFrame                    # AoA, CL_full, CD_full[, source]
    route: str                             # "semi-empirical" | "kriging"
    method_CL: Optional[str] = None        # semi-empirical method used
    method_CD: Optional[str] = None
    metrics: dict = field(default_factory=dict)   # per-coefficient dicts
    models: dict = field(default_factory=dict)    # fitted objects
    advisory: dict = field(default_factory=dict)  # recommend() output

    def summary(self) -> str:
        lines = [f"Route: {self.route}"]
        if self.route == "semi-empirical":
            lines.append(f"Method (CL): {self.method_CL}")
            lines.append(f"Method (CD): {self.method_CD}")
        for coeff, m in self.metrics.items():
            if m:
                lines.append(
                    f"{coeff}: R²={m.get('R2', float('nan')):.4f}  "
                    f"MAE={m.get('MAE', float('nan')):.4f}  "
                    f"({m.get('protocol', 'full-reference')})")
        return "\n".join(lines)


# ── Unified entry point ─────────────────────────────────────────────

def reconstruct_polar(df: pd.DataFrame,
                      config: CaseConfig,
                      route: str = "auto",
                      hf_budget: Optional[int] = None,
                      evaluate: bool = True,
                      bracket_stall: Optional[bool] = None) -> PolarResult:
    """Reconstruct the full ±180° polar from the data in ``df``.

    Parameters
    ----------
    df : DataFrame
        Must contain ``AoA`` plus the LF columns named in
        ``config.data`` (semi-empirical route) and/or the HF columns
        (Kriging route and evaluation).
    config : CaseConfig
    route : {"auto", "semi-empirical", "kriging"}
        ``"auto"`` calls :func:`recommend` using ``hf_budget`` (or, if
        not given, the number of available HF samples in ``df``).
    hf_budget : int, optional
        Available HF evaluations; used by the advisor. If ``None`` and
        HF columns are present, the available sample count is used.
    evaluate : bool
        If True and HF reference columns are present, evaluate the
        reconstruction against them (dense reference where available;
        LOO-CV for the Kriging route when the reference is itself the
        training set).
    bracket_stall : bool, optional
        Kriging route only: force stall-bracketing stations on
        (``True``) or off (``False``). ``None`` defers to the advisor
        (on at low Mach). Set ``False`` to reproduce the plain
        uniform-spacing baseline.
    """
    d = config.data
    if d.flip_aoa_sign:
        df = df.copy()
        df["AoA"] = -df["AoA"]
        df = df.sort_values("AoA").reset_index(drop=True)

    # Infer available HF budget if not stated
    hf_cl = d.HF_CL_column if d.HF_CL_column in (df.columns if d.HF_CL_column else []) else None
    hf_cd = d.HF_CD_column if d.HF_CD_column in (df.columns if d.HF_CD_column else []) else None
    if hf_budget is None and hf_cl:
        hf_budget = int(df[hf_cl].notna().sum())

    advice = recommend(config, hf_budget)
    chosen = advice["route"] if route == "auto" else route

    if chosen == "semi-empirical":
        result = _run_semi_empirical(df, config, advice)
    elif chosen == "kriging":
        result = _run_kriging(df, config, advice,
                              bracket_stall=bracket_stall)
    else:
        raise ValueError(
            f"route must be 'auto', 'semi-empirical' or 'kriging', "
            f"got {route!r}")

    result.advisory = advice

    if evaluate:
        _evaluate(result, df, config)
    return result


# ── Route implementations ───────────────────────────────────────────

def _run_semi_empirical(df: pd.DataFrame,
                        config: CaseConfig,
                        advice: dict) -> PolarResult:
    method = config.pipeline.extrapolator
    if method == "auto":
        # Derive per-coefficient winners directly (the advisor may have
        # recommended the kriging route, in which case its method slots
        # are None — the user has still explicitly asked for this route).
        method_cl, method_cd = _default_methods(config)
    else:
        method_cl = method_cd = method

    def build(m: str) -> pd.DataFrame:
        cfg = config
        old = cfg.pipeline.extrapolator
        cfg.pipeline.extrapolator = m
        try:
            ext = get_extrapolator(cfg)
            out = ext.build(df)
        finally:
            cfg.pipeline.extrapolator = old
        return out

    if method_cl == method_cd:
        polar = build(method_cl)
    else:
        # Mixed per-coefficient winners: run both, take each coefficient
        # from its per-case best method.
        out_cl = build(method_cl)
        out_cd = build(method_cd)
        polar = out_cl[["AoA", "CL_full", "source"]].merge(
            out_cd[["AoA", "CD_full"]], on="AoA", how="inner")
        polar = polar[["AoA", "CL_full", "CD_full", "source"]]

    return PolarResult(polar=polar, route="semi-empirical",
                       method_CL=method_cl, method_CD=method_cd)


def _run_kriging(df: pd.DataFrame,
                 config: CaseConfig,
                 advice: dict,
                 bracket_stall: Optional[bool] = None) -> PolarResult:
    d = config.data
    if not d.HF_CL_column or not d.HF_CD_column:
        raise ValueError(
            "Kriging route requires HF_CL_column and HF_CD_column "
            "to be set in config.data and present in the DataFrame.")

    sym = config.airfoil.symmetry == "symmetric"
    stall = abs(config.airfoil.alpha_s_2D_pos or 15.0)
    do_bracket = advice.get("bracket_stall", False) \
        if bracket_stall is None else bracket_stall
    stations = recommended_stations(
        spacing=config.pipeline.kriging_spacing,
        bracket_stall=do_bracket,
        alpha_stall=stall)

    fine = np.arange(-180.0, 180.0 + config.pipeline.fine_step,
                     config.pipeline.fine_step)

    models, curves, train_sets = {}, {}, {}
    for coeff, col in (("CL", d.HF_CL_column), ("CD", d.HF_CD_column)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            train = sample_training(df, col, stations)
        if sym:
            # Mirror the positive branch only if the negative branch is
            # not already covered by the supplied data — evaluating only
            # non-negative stations is the cost-saving option, but real
            # samples always take precedence over mirrored copies.
            n_neg_requested = int((stations < 0).sum())
            n_neg_found = int((train["AoA"] < 0).sum())
            if n_neg_found < n_neg_requested:
                pos = train[train["AoA"] >= 0.0]
                train = mirror_symmetric(pos, coeff, col)
        rec = KrigingReconstructor()
        rec.fit(train["AoA"].values, train[col].values)
        models[coeff] = rec
        curves[coeff] = rec.predict(fine)
        train_sets[coeff] = train

    polar = pd.DataFrame({
        "AoA": fine,
        "CL_full": curves["CL"],
        "CD_full": curves["CD"],
        "source": "kriging",
    })
    res = PolarResult(polar=polar, route="kriging", models=models)
    res.models["train_CL"] = train_sets["CL"]
    res.models["train_CD"] = train_sets["CD"]
    return res


# ── Evaluation ──────────────────────────────────────────────────────

def _evaluate(result: PolarResult,
              df: pd.DataFrame,
              config: CaseConfig) -> None:
    d = config.data
    for coeff, ref_col in (("CL", d.HF_CL_column), ("CD", d.HF_CD_column)):
        if not ref_col or ref_col not in df.columns:
            continue
        n_ref = int(df[ref_col].notna().sum())
        curve_col = f"{coeff}_full"

        if result.route == "kriging":
            train = result.models.get(f"train_{coeff}")
            n_train = 0 if train is None else len(train)
            if n_ref <= n_train + 2:
                # Reference is (essentially) the training set: dense
                # evaluation is meaningless; use LOO-CV instead.
                ref = df.dropna(subset=[ref_col])
                m = loo_cv(ref["AoA"].values, ref[ref_col].values)
                m.pop("y_pred", None)
                m["protocol"] = "LOO-CV (sparse reference)"
                m["n_ref"] = n_ref
                result.metrics[coeff] = m
                continue

        m = evaluate_curve(df, result.polar, ref_col, curve_col)
        m["protocol"] = "full-reference"
        result.metrics[coeff] = m
