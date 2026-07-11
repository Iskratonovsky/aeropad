"""
aeropad.sizing.regression
=========================
Unified tune–fit–evaluate workflow for statistical sizing regressions.

:class:`SizingModel` wraps one regression family behind a consistent
interface: leakage-safe hyperparameter tuning (KFold + grid search on
the training split only), held-out-test evaluation, prediction, and —
for closed-form families — a human-readable sizing equation.

:func:`compare_families` runs every family on the same split and
returns a single comparison table, replacing the per-notebook
``aggregate.csv`` workflow with one call.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from .models import FAMILIES, build_family
from ..polar.metrics import mae, r2, rmse


class SizingModel:
    """One regression family, tuned and evaluated on a sizing dataset.

    Parameters
    ----------
    family : str
        One of :data:`aeropad.sizing.models.FAMILIES`.
    cv : int
        Folds for the tuning cross-validation (on the training split).
    test_size : float
        Held-out fraction for final evaluation.
    random_state : int
        Seed for the split, the CV shuffling, and stochastic families.

    Examples
    --------
    >>> m = SizingModel("power_law")
    >>> m.fit(df, features=["MTOW", "Cruising speed"],
    ...       target="Disc Loading")
    >>> m.metrics["R2_test"]
    >>> m.equation()
    'Disc Loading = 0.9 * MTOW^0.31 * Cruising speed^0.18'
    >>> m.predict({"MTOW": 5000.0, "Cruising speed": 250.0})
    """

    def __init__(self,
                 family: str = "polynomial",
                 cv: int = 5,
                 test_size: float = 0.3,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 family_options: Optional[dict] = None) -> None:
        self.family = family.lower()
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.family_options = dict(family_options or {})
        self._name_map: dict = {}

        self.features: list[str] = []
        self.target: str = ""
        self.model = None            # fitted best estimator
        self.best_params_: dict = {}
        self.metrics: dict = {}
        self.tuning_results_: Optional[pd.DataFrame] = None
        self._split = None           # (X_train, X_test, y_train, y_test)

    # ── fitting ──────────────────────────────────────────────────────

    def fit(self,
            df: pd.DataFrame,
            features: Sequence[str],
            target: str) -> "SizingModel":
        self.features = list(features)
        self.target = target

        data = df[self.features + [target]].dropna()
        X = data[self.features].values.astype(float)
        y = data[target].values.astype(float)

        opts = dict(self.family_options)
        loss_grid = opts.pop("loss_grid", None)   # symbolic only
        spec = build_family(self.family, self.random_state, **opts)
        if spec["tags"]["positive_only"]:
            bad = [c for c, col in zip(self.features + [target],
                                       np.column_stack([X, y]).T)
                   if (col <= 0).any()]
            if bad:
                raise ValueError(
                    f"family '{self.family}' requires strictly positive "
                    f"values (log–log fit); offending column(s): "
                    f"{', '.join(bad)}. Filter those rows or drop the "
                    f"column, or choose another family.")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state)
        self._split = (X_tr, X_te, y_tr, y_te)

        if self.family == "symbolic":
            self._fit_symbolic(spec, X_tr, y_tr, loss_grid, opts)
        elif spec["param_grid"]:
            n_splits = min(self.cv, len(X_tr))
            cv = KFold(n_splits=n_splits, shuffle=True,
                       random_state=self.random_state)
            search = GridSearchCV(
                spec["estimator"], spec["param_grid"], cv=cv,
                scoring="neg_mean_squared_error", n_jobs=self.n_jobs)
            search.fit(X_tr, y_tr)
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            self.tuning_results_ = self._tidy_cv(search)
        else:
            self.model = spec["estimator"].fit(X_tr, y_tr)
            self.best_params_ = {}
            self.tuning_results_ = None

        self._evaluate(X_tr, X_te, y_tr, y_te)
        return self

    def _fit_symbolic(self, spec, X_tr, y_tr, loss_grid, opts) -> None:
        """Bespoke path for PySR symbolic regression.

        Feature names are sanitised to valid identifiers and passed as
        a DataFrame so the discovered equations carry real variable
        names. If ``loss_grid`` is provided (a list of Huber deltas),
        candidate models with ``HuberLoss(d)`` are fitted on an 80/20
        sub-split of the training data and the delta with the lowest
        validation MSE is selected before the final training fit.
        """
        import re
        from .models import build_family as _bf

        safe = [re.sub(r"\W+", "_", f) for f in self.features]
        self._name_map = dict(zip(safe, self.features))
        X_tr_df = pd.DataFrame(X_tr, columns=safe)

        if loss_grid:
            X_sub, X_val, y_sub, y_val = train_test_split(
                X_tr_df, y_tr, test_size=0.2,
                random_state=self.random_state)
            rows, best_d, best_mse = [], None, np.inf
            for d in loss_grid:
                cand_opts = dict(opts,
                                 elementwise_loss=f"HuberLoss({d})")
                cand = _bf("symbolic", self.random_state,
                           **cand_opts)["estimator"]
                cand.fit(X_sub, y_sub)
                mse = float(np.mean(
                    (np.asarray(cand.predict(X_val)).ravel()
                     - y_val) ** 2))
                rows.append({"huber_delta": d, "cv_rmse": np.sqrt(mse)})
                if mse < best_mse:
                    best_mse, best_d = mse, d
            self.tuning_results_ = pd.DataFrame(rows)
            self.best_params_ = {"elementwise_loss":
                                 f"HuberLoss({best_d})"}
            spec = _bf("symbolic", self.random_state,
                       **dict(opts,
                              elementwise_loss=f"HuberLoss({best_d})"))

        self.model = spec["estimator"].fit(X_tr_df, y_tr)
        if not loss_grid:
            self.best_params_ = {"model_selection":
                                 getattr(self.model, "model_selection",
                                         "best")}
            self.tuning_results_ = None

    @staticmethod
    def _tidy_cv(search: GridSearchCV) -> pd.DataFrame:
        """Tidy GridSearchCV results: one row per grid point."""
        res = pd.DataFrame(search.cv_results_["params"])
        res["cv_rmse"] = np.sqrt(-search.cv_results_["mean_test_score"])
        return res

    def _evaluate(self, X_tr, X_te, y_tr, y_te) -> None:
        y_hat_te = self.model.predict(X_te)
        y_hat_tr = self.model.predict(X_tr)
        abs_err = np.abs(y_te - y_hat_te)
        self.metrics = {
            "family": self.family,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "R2_train": r2(y_tr, y_hat_tr),
            "R2_test": r2(y_te, y_hat_te),
            "RMSE_test": rmse(y_te, y_hat_te),
            "MAE_test": mae(y_te, y_hat_te),
            "MaxErr_test": float(abs_err.max()),
            "pearson_r_test": float(np.corrcoef(y_te, y_hat_te)[0, 1]),
            "best_params": self.best_params_,
        }

    # ── prediction ───────────────────────────────────────────────────

    def predict(self, X, return_std: bool = False):
        """Predict the target for new designs.

        ``X`` may be an array, a DataFrame with the feature columns, or
        a single dict of feature values. ``return_std`` is available
        for the ``gpr`` family only.
        """
        if self.model is None:
            raise RuntimeError("fit() must be called first.")
        if isinstance(X, dict):
            X = np.array([[X[f] for f in self.features]], dtype=float)
        elif isinstance(X, pd.DataFrame):
            X = X[self.features].values.astype(float)
        else:
            X = np.atleast_2d(np.asarray(X, dtype=float))

        if self.family == "symbolic" and self._name_map:
            X = pd.DataFrame(X, columns=list(self._name_map.keys()))

        if return_std:
            if self.family != "gpr":
                raise ValueError(
                    "Predictive std is only available for family='gpr'.")
            scaler = self.model.named_steps["scaler"]
            gpr = self.model.named_steps["gpr"]
            return gpr.predict(scaler.transform(X), return_std=True)
        return self.model.predict(X)

    # ── introspection ────────────────────────────────────────────────

    def equation(self, precision: int = 4) -> Optional[str]:
        """Human-readable sizing equation for closed-form families.

        Power-law equations are exact in original units. Linear-family
        equations (polynomial / ridge / lasso) are expressed over
        standardized inputs ``z(·)`` — the form the model was actually
        fitted in — matching the reporting convention of the underlying
        study. Returns ``None`` for kernel families (kernel_ridge, gpr).
        """
        if self.model is None:
            raise RuntimeError("fit() must be called first.")
        p = precision

        if self.family == "symbolic":
            expr = str(self.model.sympy())
            # restore original feature names in the discovered equation
            for safe_name, orig in sorted(self._name_map.items(),
                                          key=lambda kv: -len(kv[0])):
                expr = expr.replace(safe_name, orig)
            return f"{self.target} = {expr}"

        if self.family == "power_law":
            a = self.model.amplitude_
            parts = [f"{self.target} = {a:.{p}g}"]
            for f, b in zip(self.features, self.model.coef_):
                parts.append(f"{f}^{b:.{p}g}")
            return " * ".join(parts)

        if self.family == "polynomial":
            poly = self.model.named_steps["poly"]
            ols = self.model.named_steps["ols"]
            names = poly.get_feature_names_out(self.features)
            terms = [f"{ols.intercept_:.{p}g}"]
            for name, c in zip(names, np.ravel(ols.coef_)):
                terms.append(f"({c:.{p}g})*z({name})")
            return f"{self.target} = " + " + ".join(terms)

        if self.family in ("ridge", "lasso"):
            step = self.model.named_steps[self.family]
            terms = [f"{step.intercept_:.{p}g}"]
            for f, c in zip(self.features, np.ravel(step.coef_)):
                terms.append(f"({c:.{p}g})*z({f})")
            return f"{self.target} = " + " + ".join(terms)

        return None

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Family: {self.family}",
            f"Fit: {self.target} = f({', '.join(self.features)})",
            f"Split: {m['n_train']} train / {m['n_test']} test",
            f"Best params: {m['best_params'] or '—'}",
            f"R² train/test: {m['R2_train']:.4f} / {m['R2_test']:.4f}",
            f"RMSE: {m['RMSE_test']:.4f}   MAE: {m['MAE_test']:.4f}   "
            f"MaxErr: {m['MaxErr_test']:.4f}",
        ]
        eq = self.equation()
        if eq:
            lines.append(f"Equation: {eq}")
        return "\n".join(lines)


def compare_families(df: pd.DataFrame,
                     features: Sequence[str],
                     target: str,
                     families: Optional[Sequence[str]] = None,
                     **model_kwargs) -> tuple:
    """Fit every requested family on one common split; return a table.

    All families share the same ``random_state``-controlled split, so
    the comparison is apples-to-apples. Families whose requirements the
    data violates (e.g. power_law on non-positive data) are skipped
    with a note in the table.

    Returns
    -------
    (table, models) : (pd.DataFrame, dict[str, SizingModel])
    """
    rows, models = [], {}
    for fam in (families or FAMILIES):
        try:
            m = SizingModel(fam, **model_kwargs).fit(df, features, target)
            models[fam] = m
            rows.append({k: v for k, v in m.metrics.items()
                         if k != "best_params"}
                        | {"best_params": str(m.metrics["best_params"])})
        except Exception as exc:
            rows.append({"family": fam,
                         "best_params": f"SKIPPED: {exc}"})
    table = pd.DataFrame(rows).set_index("family")
    return table, models
