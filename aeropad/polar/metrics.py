"""
aeropad.polar.metrics
=====================
Accuracy metrics for full-range polar reconstruction.

Both R² and MAE are reported throughout aeropad because the two metrics
characterise different error structures and can disagree on the better
reconstruction. Semi-empirical extrapolation tends to produce *globally
consistent* errors (uniform deviation → tight residual variance → high
R², moderate MAE); Kriging tends to be *locally optimal* (near-perfect
at most stations, large isolated deviations where sharp features fall
between training points → low MAE, but R² penalised by the few
mismatched stations). Which metric matters depends on the downstream
use: integrated-load computations weight mean deviation (MAE), while
worst-case-bounded analyses weight maximum deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean(
        (np.asarray(y_true, dtype=float)
         - np.asarray(y_pred, dtype=float)) ** 2)))


def evaluate_curve(reference: pd.DataFrame,
                   curve: pd.DataFrame,
                   ref_col: str,
                   curve_col: str,
                   aoa_col: str = "AoA") -> dict:
    """Evaluate a reconstructed curve against a reference polar.

    The reconstruction is linearly interpolated onto the reference
    angle-of-attack stations before computing metrics, so the two inputs
    need not share a grid.

    Returns
    -------
    dict with keys ``R2``, ``MAE``, ``RMSE``, ``n_ref``.
    """
    ref = reference[[aoa_col, ref_col]].dropna().sort_values(aoa_col)
    cur = curve[[aoa_col, curve_col]].dropna().sort_values(aoa_col)
    if len(ref) == 0:
        return {"R2": float("nan"), "MAE": float("nan"),
                "RMSE": float("nan"), "n_ref": 0}
    y_hat = np.interp(ref[aoa_col].values,
                      cur[aoa_col].values,
                      cur[curve_col].values)
    y_ref = ref[ref_col].values
    return {"R2": r2(y_ref, y_hat), "MAE": mae(y_ref, y_hat),
            "RMSE": rmse(y_ref, y_hat), "n_ref": int(len(ref))}
