"""
aeropad.sizing.plots
====================
Diagnostics for statistical sizing studies.

:func:`correlation_heatmap` is the dataset-exploration entry point:
which parameters correlate strongly enough to anchor a sizing relation.
The remaining functions diagnose a fitted :class:`SizingModel` —
hyperparameter tuning curve, 3D prediction surface (two-feature fits),
actual-vs-predicted, and a leakage-safe learning curve.

All functions return the matplotlib figure and accept ``save=`` for
direct file export.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:      # graceful degradation to plain matplotlib
    _HAS_SEABORN = False


# ── Dataset exploration ─────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame,
                        columns: Optional[Sequence[str]] = None,
                        method: str = "pearson",
                        annot: bool = True,
                        mask_upper: bool = True,
                        cmap: str = "coolwarm",
                        figsize: tuple = (8, 6),
                        save: Optional[str] = None):
    """Correlation heatmap over the numeric columns of a sizing dataset.

    The natural first step of any sizing-by-statistics study: identify
    which design parameters carry exploitable statistical relationships
    before committing to feature/target choices.

    Parameters
    ----------
    df : DataFrame
        The dataset. Non-numeric columns are ignored automatically.
    columns : sequence of str, optional
        Restrict the heatmap to these columns.
    method : {"pearson", "spearman", "kendall"}
        Correlation measure. Spearman is often preferable for sizing
        data spanning orders of magnitude (rank-based, monotonic).
    mask_upper : bool
        Show only the lower triangle (the matrix is symmetric).

    Returns
    -------
    (fig, corr) : (matplotlib Figure, DataFrame)
        The figure and the correlation matrix itself.
    """
    num = df[list(columns)] if columns else df.select_dtypes("number")
    corr = num.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) \
        if mask_upper else None

    fig, ax = plt.subplots(figsize=figsize)
    if _HAS_SEABORN:
        sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f",
                    cmap=cmap, vmin=-1.0, vmax=1.0, square=True,
                    cbar_kws={"label": f"{method.title()} correlation"},
                    ax=ax)
    else:
        data = corr.where(~mask) if mask is not None else corr
        im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45,
                      ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        if annot:
            for i in range(len(corr)):
                for j in range(i + 1 if mask_upper else len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label=f"{method.title()} correlation")

    ax.set_title(f"{method.title()} correlation heatmap")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig, corr


# ── Model diagnostics ───────────────────────────────────────────────

def tuning_curve(model,
                 figsize: tuple = (8, 5),
                 save: Optional[str] = None):
    """CV RMSE across the hyperparameter grid of a fitted SizingModel.

    One-parameter grids render as a curve (log-x when the grid spans
    decades); two-parameter grids render as an RMSE heatmap. Families
    without hyperparameters (power_law) raise a ValueError.
    """
    res = model.tuning_results_
    if res is None:
        raise ValueError(
            f"family '{model.family}' has no hyperparameter grid.")
    params = [c for c in res.columns if c != "cv_rmse"]

    fig, ax = plt.subplots(figsize=figsize)
    if len(params) == 1:
        p = params[0]
        x = res[p]
        if np.issubdtype(np.asarray(x).dtype, np.number):
            ax.plot(x, res.cv_rmse, marker="o")
            if x.max() / max(x.min(), 1e-12) > 100:
                ax.set_xscale("log")
        else:  # non-numeric grid (e.g. kernel objects) — index axis
            ax.plot(range(len(x)), res.cv_rmse, marker="o")
            ax.set_xticks(range(len(x)),
                          [str(v)[:18] for v in x],
                          rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(p)
        ax.set_ylabel("CV RMSE")
    elif len(params) == 2:
        pivot = res.pivot_table(index=params[0], columns=params[1],
                                values="cv_rmse")
        im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(pivot.shape[1]),
                      [f"{v:.3g}" for v in pivot.columns],
                      rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]),
                      [f"{v:.3g}" for v in pivot.index])
        ax.set_xlabel(params[1])
        ax.set_ylabel(params[0])
        fig.colorbar(im, ax=ax, label="CV RMSE")
    else:
        raise ValueError("tuning_curve supports 1- or 2-parameter grids.")

    ax.set_title(f"{model.family}: tuning grid (CV RMSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def prediction_surface(model,
                       df: pd.DataFrame,
                       resolution: int = 60,
                       figsize: tuple = (10, 7),
                       save: Optional[str] = None):
    """3D fitted-surface plot for two-feature sizing models.

    Observed data is overlaid as scatter, following the reporting style
    of the underlying study.
    """
    if len(model.features) != 2:
        raise ValueError(
            "prediction_surface requires a model with exactly 2 features.")
    f1, f2 = model.features
    data = df[[f1, f2, model.target]].dropna()

    g1 = np.linspace(data[f1].min(), data[f1].max(), resolution)
    g2 = np.linspace(data[f2].min(), data[f2].max(), resolution)
    G1, G2 = np.meshgrid(g1, g2)
    Z = model.predict(np.column_stack([G1.ravel(), G2.ravel()])) \
        .reshape(G1.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(G1, G2, Z, cmap="viridis", alpha=0.78,
                           linewidth=0, antialiased=True)
    ax.scatter(data[f1], data[f2], data[model.target],
               color="black", s=18, alpha=0.7, label="Observed data")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(model.target)
    ax.set_title(f"{model.family}: {model.target} = f({f1}, {f2})")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08,
                 label=f"Predicted {model.target}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def actual_vs_predicted(model,
                        bounds: Optional[float] = None,
                        figsize: tuple = (6, 5),
                        save: Optional[str] = None):
    """Held-out-test actual-vs-predicted scatter with the y = x line.

    ``bounds`` adds ±bounds guide lines around perfect prediction
    (in target units), as used in the underlying study's reporting.
    """
    X_tr, X_te, y_tr, y_te = model._split
    y_hat = model.predict(X_te)

    lo = min(y_te.min(), y_hat.min())
    hi = max(y_te.max(), y_hat.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([lo, hi], [lo, hi], color="red", linestyle=":",
            linewidth=1.5, label="Perfect prediction (y = x)")
    if bounds:
        ax.plot([lo, hi], [lo + bounds, hi + bounds], color="blue",
                linestyle=":", linewidth=1.2, label=f"± {bounds:g} band")
        ax.plot([lo, hi], [lo - bounds, hi - bounds], color="blue",
                linestyle=":", linewidth=1.2)
    ax.scatter(y_te, y_hat, s=28, alpha=0.8)
    ax.set_xlabel(f"Actual {model.target}")
    ax.set_ylabel(f"Predicted {model.target}")
    ax.set_title(f"{model.family}: actual vs. predicted "
                 f"(R² = {model.metrics['R2_test']:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def learning_curve_plot(model,
                        figsize: tuple = (8, 5),
                        save: Optional[str] = None):
    """Leakage-safe learning curve of the tuned estimator (train split)."""
    from sklearn.model_selection import KFold, learning_curve

    X_tr, _, y_tr, _ = model._split
    n_splits = min(model.cv, len(X_tr))
    cv = KFold(n_splits=n_splits, shuffle=True,
               random_state=model.random_state)
    sizes, tr_scores, te_scores = learning_curve(
        model.model, X_tr, y_tr, cv=cv,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 6))

    tr_rmse = np.sqrt(-tr_scores.mean(axis=1))
    te_rmse = np.sqrt(-te_scores.mean(axis=1))
    te_std = np.sqrt(-te_scores).std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sizes, tr_rmse, marker="o", label="Training RMSE")
    ax.plot(sizes, te_rmse, marker="s", label="CV RMSE")
    ax.fill_between(sizes, te_rmse - te_std, te_rmse + te_std,
                    alpha=0.15)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model.family}: learning curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
