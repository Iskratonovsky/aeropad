"""
aeropad.sizing.models
=====================
Regression-family registry for aircraft sizing by statistics.

Each family is returned as an sklearn estimator (usually a Pipeline)
plus a hyperparameter grid, so a single tuning/evaluation workflow in
:mod:`aeropad.sizing.regression` serves them all. The families and
their grids follow the benchmarked notebook study on the rotorcraft
statistical-sizing problem (in the spirit of Rand & Khromov's
helicopter sizing by statistics):

==============  ====================================================
``polynomial``  PolynomialFeatures (deg 1–10) → scaler → OLS
``ridge``       scaler → Ridge, α ∈ logspace(−4, 3, 12)
``lasso``       scaler → Lasso, α ∈ logspace(−4, 1, 12)
``kernel_ridge``scaler → RBF KernelRidge, α × γ grid, scaled target
``power_law``   log–log OLS: y = a·x₁^b¹·x₂^b²·…  (closed form)
``gpr``         scaler → GP (C·RBF + White), init length-scale grid
==============  ====================================================
"""

from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """Multivariate power-law regression via log–log ordinary least squares.

    Fits ``y = a · x₁^b₁ · x₂^b₂ · …`` by linear regression of ``ln y``
    on ``ln x``. All features and the target must be strictly positive.
    The fitted exponents are directly interpretable as sensitivity
    elasticities — the classical form of statistical sizing relations.
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if (X <= 0).any() or (y <= 0).any():
            raise ValueError(
                "PowerLawRegressor requires strictly positive features "
                "and target (log–log fit).")
        self._ols = LinearRegression()
        self._ols.fit(np.log(X), np.log(y))
        self.coef_ = self._ols.coef_          # exponents b_i
        self.intercept_ = self._ols.intercept_  # ln(a)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "_ols")
        X = check_array(X)
        return np.exp(self._ols.predict(np.log(X)))

    @property
    def amplitude_(self) -> float:
        """The multiplicative constant ``a`` in original units."""
        check_is_fitted(self, "_ols")
        return float(np.exp(self.intercept_))


def build_family(name: str, random_state: int = 42,
                 **options) -> dict:
    """Return ``{"estimator": ..., "param_grid": ..., "tags": ...}``.

    ``tags`` flags: ``closed_form`` (an equation string is available),
    ``supports_std`` (predictive uncertainty available),
    ``positive_only`` (features/target must be > 0), ``bespoke``
    (family manages its own model selection; no grid search).

    ``**options`` are forwarded to the family constructor where
    supported (currently ``symbolic``: any PySRRegressor keyword,
    e.g. ``niterations=100`` or ``maxsize=15``).
    """
    name = name.lower()

    if name == "polynomial":
        est = Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("ols", LinearRegression()),
        ])
        grid = {"poly__degree": np.arange(1, 11)}
        tags = dict(closed_form=True, supports_std=False,
                    positive_only=False)

    elif name == "ridge":
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        grid = {"ridge__alpha": np.logspace(-4, 3, 12)}
        tags = dict(closed_form=True, supports_std=False,
                    positive_only=False)

    elif name == "lasso":
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=50_000)),
        ])
        grid = {"lasso__alpha": np.logspace(-4, 1, 12)}
        tags = dict(closed_form=True, supports_std=False,
                    positive_only=False)

    elif name == "kernel_ridge":
        inner = Pipeline([
            ("scaler", StandardScaler()),
            ("krr", KernelRidge(kernel="rbf")),
        ])
        est = TransformedTargetRegressor(
            regressor=inner, transformer=StandardScaler())
        grid = {
            "regressor__krr__alpha": np.logspace(-4, 1, 6),
            "regressor__krr__gamma": np.logspace(-3, 2, 6),
        }
        tags = dict(closed_form=False, supports_std=False,
                    positive_only=False)

    elif name == "power_law":
        est = PowerLawRegressor()
        grid = {}  # closed-form OLS in log space: nothing to tune
        tags = dict(closed_form=True, supports_std=False,
                    positive_only=True)

    elif name == "gpr":
        kernels = [
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1.0,
                          noise_level_bounds=(1e-8, 1e1))
            for ls in np.logspace(-1, 2, 10)
        ]
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                n_restarts_optimizer=8,
                random_state=random_state,
                normalize_y=True)),
        ])
        grid = {"gpr__kernel": kernels}
        tags = dict(closed_form=False, supports_std=True,
                    positive_only=False)

    elif name == "symbolic":
        try:
            from pysr import PySRRegressor
        except Exception as exc:  # ImportError, or Julia bootstrap failure
            raise ImportError(
                "family 'symbolic' requires PySR with a working Julia "
                "backend (pip install pysr; Julia is downloaded "
                "automatically on first use — requires internet "
                "access).") from exc

        opts = dict(
            niterations=30,
            populations=40,
            model_selection="best",
            binary_operators=["-", "+", "*", "/", "^"],
            unary_operators=[
                "exp", "abs",
                "sq(x) = x * x",
                "cub(x) = x * x * x",
                "inv(x) = 1 / x",
            ],
            extra_sympy_mappings={
                "sq": lambda x: x ** 2,
                "cub": lambda x: x ** 3,
                "inv": lambda x: 1 / x,
            },
            elementwise_loss="HuberLoss()",
            maxsize=25,
            parsimony=0.005,
            adaptive_parsimony_scaling=1050,
            nested_constraints={"exp": {"exp": 0}, "^": {"^": 0}},
            random_state=random_state,
            deterministic=False,
            progress=False,
            verbosity=0,
        )
        opts.update(options)
        est = PySRRegressor(**opts)
        grid = {}  # evolutionary search performs its own model selection
        tags = dict(closed_form=True, supports_std=False,
                    positive_only=False, bespoke=True)

    else:
        raise ValueError(
            f"Unknown family {name!r}. Valid: polynomial, ridge, lasso, "
            f"kernel_ridge, power_law, gpr, symbolic.")

    return {"estimator": est, "param_grid": grid, "tags": tags}


_SYMBOLIC_OK: bool | None = None


def symbolic_available() -> bool:
    """True if the PySR/Julia backend is usable in this environment.

    The result is cached: the first check may take several seconds if
    the Julia bootstrap has to time out against a blocked network.
    """
    global _SYMBOLIC_OK
    if _SYMBOLIC_OK is None:
        try:
            import pysr  # noqa: F401
            _SYMBOLIC_OK = True
        except Exception:
            _SYMBOLIC_OK = False
    return _SYMBOLIC_OK


FAMILIES = ("polynomial", "ridge", "lasso",
            "kernel_ridge", "power_law", "gpr", "symbolic")
