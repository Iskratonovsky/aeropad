"""
aeropad.sizing.dataio
=====================
Dataset standardisation, integrity guarding, and results export for
statistical sizing studies.

Real aircraft databases are messy in recurring, diagnosable ways. The
utilities here encode failure modes met (and fixed) in practice:

- **Unit contamination** — a subset of a column entered in different
  units (classically millimetres mixed into a metres column, values
  ~1000× the rest). :func:`detect_unit_contamination` finds such
  entries by cluster-ratio testing; :func:`fix_unit_contamination`
  normalises them.
- **Collinearity** — adding a second predictor that is strongly
  correlated with the first (|r| ≳ 0.9) can *worsen* hold-out
  performance rather than improve it. :func:`collinearity_screen`
  flags risky feature pairs and reports variance inflation factors.
- **Missing-by-design vs missing-by-error** — in mixed databases some
  columns are structurally absent for whole categories (piston
  aircraft have engine power, jets have thrust). Treating those as
  errors corrupts imputation and screening.
  :func:`missingness_report` separates the two patterns.
- **Targets that resist prediction** — some design parameters do not
  reduce to gross-parameter laws (they are design freedoms, not
  consequences). :func:`usability_screen` runs quick hold-out fits and
  flags targets with non-positive test R² as *not usable*, so an
  unreliable relation never enters a sizing chain silently.

:func:`audit_dataset` bundles the guards into one report;
:func:`export_results` writes fitted equations and metrics from any
model family (symbolic regression included) to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ── Standardise ─────────────────────────────────────────────────────

def standardize_dataset(df: pd.DataFrame,
                        rename: Optional[dict] = None,
                        numeric: str | Sequence[str] = "auto",
                        min_numeric_frac: float = 0.5) -> tuple:
    """Clean a raw dataset into analysis-ready form.

    Steps: strip whitespace/BOM from column names, apply ``rename``,
    coerce numeric columns (``numeric="auto"`` treats any column at
    least ``min_numeric_frac`` parseable as numeric; or pass an
    explicit list), and drop fully-empty rows/columns.

    Returns
    -------
    (df_clean, report) : (DataFrame, dict)
        ``report`` records renames applied, columns coerced, and any
        values lost to coercion (count per column).
    """
    report = {"renamed": {}, "numeric_columns": [],
              "coercion_losses": {}, "dropped_empty_cols": []}

    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "")
                   for c in out.columns]
    if rename:
        applied = {k: v for k, v in rename.items() if k in out.columns}
        out = out.rename(columns=applied)
        report["renamed"] = applied

    if numeric == "auto":
        candidates = []
        for c in out.columns:
            coerced = pd.to_numeric(out[c], errors="coerce")
            frac = coerced.notna().sum() / max(out[c].notna().sum(), 1)
            if frac >= min_numeric_frac and out[c].notna().any():
                candidates.append(c)
    else:
        candidates = [c for c in numeric if c in out.columns]

    for c in candidates:
        before = int(out[c].notna().sum())
        out[c] = pd.to_numeric(out[c], errors="coerce")
        lost = before - int(out[c].notna().sum())
        report["numeric_columns"].append(c)
        if lost:
            report["coercion_losses"][c] = lost

    empty_cols = [c for c in out.columns if out[c].isna().all()]
    out = out.drop(columns=empty_cols).dropna(how="all")
    report["dropped_empty_cols"] = empty_cols
    return out, report


# ── Guard: unit contamination ───────────────────────────────────────

def detect_unit_contamination(series: pd.Series,
                              factor: float = 1000.0,
                              rel_tol: float = 0.35) -> pd.Series:
    """Flag entries that sit ~``factor``× the column's main cluster.

    The classic case is millimetre values mixed into a metres column.
    The main cluster is taken as the median of the lower group after a
    log-scale split; entries whose ratio to that cluster median lies
    within ``rel_tol`` (log-relative) of ``factor`` are flagged.

    Returns a boolean mask (True = contaminated) aligned to the input.
    """
    x = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    v = x.dropna()
    v = v[v > 0]
    if len(v) < 5:
        return mask
    logv = np.log10(v)
    split = logv.median() + np.log10(factor) / 2.0
    low, high = v[logv < split], v[logv >= split]
    if len(high) == 0 or len(low) < 3:
        return mask
    ratio = high / low.median()
    hit = high.index[np.abs(np.log10(ratio / factor))
                     <= np.log10(1 + rel_tol)]
    mask.loc[hit] = True
    return mask


def fix_unit_contamination(df: pd.DataFrame,
                           column: str,
                           factor: float = 1000.0,
                           **detect_kwargs) -> tuple:
    """Divide contaminated entries of ``column`` by ``factor``.

    Returns ``(df_fixed, n_fixed)``. Non-destructive: operates on a
    copy.
    """
    mask = detect_unit_contamination(df[column], factor=factor,
                                     **detect_kwargs)
    out = df.copy()
    out.loc[mask, column] = pd.to_numeric(
        out.loc[mask, column], errors="coerce") / factor
    return out, int(mask.sum())


# ── Guard: collinearity ─────────────────────────────────────────────

def collinearity_screen(df: pd.DataFrame,
                        features: Sequence[str],
                        threshold: float = 0.9) -> dict:
    """Flag feature pairs whose |Pearson r| meets ``threshold``, with VIFs.

    Empirical motivation: on sizing databases, adding a second
    predictor correlated ~0.9 with the first has been observed to
    *worsen* hold-out R² — the extra feature adds variance, not
    information. Treat flagged pairs as candidates for dropping one
    member rather than as richer inputs.

    Returns ``{"pairs": DataFrame, "vif": Series}``.
    """
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").dropna()
    corr = X.corr()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j],
                              "pearson_r": float(r)})

    vifs = {}
    Xv = (X - X.mean()) / X.std(ddof=0)
    for c in cols:
        others = [o for o in cols if o != c]
        if not others:
            vifs[c] = 1.0
            continue
        A = np.column_stack([Xv[others].values,
                             np.ones(len(Xv))])
        coef, *_ = np.linalg.lstsq(A, Xv[c].values, rcond=None)
        resid = Xv[c].values - A @ coef
        r2 = 1 - resid.var() / Xv[c].values.var()
        vifs[c] = float(1.0 / max(1 - r2, 1e-9))

    return {"pairs": pd.DataFrame(pairs),
            "vif": pd.Series(vifs, name="VIF")}


# ── Guard: missingness ──────────────────────────────────────────────

def missingness_report(df: pd.DataFrame,
                       by: Optional[str] = None,
                       design_hi: float = 0.9,
                       design_lo: float = 0.1) -> pd.DataFrame:
    """Per-column missingness, optionally split by a category column.

    When ``by`` is given, a column is classified *missing-by-design*
    if it is nearly absent (≥ ``design_hi`` missing) in at least one
    category while nearly complete (≤ ``design_lo`` missing) in
    another — the structural pattern of, e.g., Power vs Thrust across
    propulsion types. Everything else with missing values is
    *missing-by-error* (scattered gaps needing attention).
    """
    rows = []
    for c in df.columns:
        if c == by:
            continue
        total_miss = float(df[c].isna().mean())
        entry = {"column": c, "missing_frac": round(total_miss, 4),
                 "classification": "complete" if total_miss == 0
                 else "missing-by-error"}
        if by is not None and total_miss > 0:
            per = df.groupby(by)[c].apply(lambda s: s.isna().mean())
            entry["per_group"] = {k: round(float(v), 3)
                                  for k, v in per.items()}
            if (per >= design_hi).any() and (per <= design_lo).any():
                entry["classification"] = "missing-by-design"
        rows.append(entry)
    return pd.DataFrame(rows)


# ── Guard: target usability ─────────────────────────────────────────

def usability_screen(df: pd.DataFrame,
                     targets: Sequence[str],
                     features: Sequence[str],
                     family: str = "polynomial",
                     min_r2: float = 0.0,
                     **model_kwargs) -> pd.DataFrame:
    """Quick hold-out screen: which targets support a sizing relation?

    For each target, one model of ``family`` is fitted on the given
    features and evaluated on the held-out split. Targets with test R²
    below ``min_r2`` (default 0: worse than predicting the mean) are
    flagged **not usable** — a documented negative result, preventing
    an unreliable estimate from silently entering a sizing chain.
    """
    from .regression import SizingModel

    rows = []
    for tgt in targets:
        feats = [f for f in features if f != tgt]
        try:
            m = SizingModel(family, **model_kwargs).fit(df, feats, tgt)
            r2t = m.metrics["R2_test"]
            rows.append({
                "target": tgt, "family": family,
                "R2_test": round(r2t, 4),
                "RMSE_test": round(m.metrics["RMSE_test"], 4),
                "verdict": "usable" if r2t >= min_r2
                else "NOT USABLE (R2 < %.2g)" % min_r2,
            })
        except Exception as exc:
            rows.append({"target": tgt, "family": family,
                         "R2_test": float("nan"),
                         "RMSE_test": float("nan"),
                         "verdict": f"SKIPPED: {exc}"})
    return pd.DataFrame(rows)


# ── Bundled audit ───────────────────────────────────────────────────

@dataclass
class AuditReport:
    shape: tuple
    n_duplicates: int
    missingness: pd.DataFrame
    unit_warnings: dict = field(default_factory=dict)
    collinearity: Optional[dict] = None

    def summary(self) -> str:
        lines = [f"Rows × cols: {self.shape[0]} × {self.shape[1]}",
                 f"Duplicate rows: {self.n_duplicates}"]
        by_design = self.missingness.query(
            "classification == 'missing-by-design'")["column"].tolist()
        by_error = self.missingness.query(
            "classification == 'missing-by-error'")["column"].tolist()
        if by_design:
            lines.append(f"Missing-by-design: {', '.join(by_design)}")
        if by_error:
            lines.append(f"Missing-by-error:  {', '.join(by_error)}")
        if self.unit_warnings:
            for c, n in self.unit_warnings.items():
                lines.append(f"UNIT CONTAMINATION suspected: {c} "
                             f"({n} entries ~1000× the main cluster)")
        else:
            lines.append("Unit contamination: none detected")
        if self.collinearity is not None \
                and len(self.collinearity["pairs"]):
            for _, r in self.collinearity["pairs"].iterrows():
                lines.append(f"Collinearity: {r.feature_a} ~ "
                             f"{r.feature_b} (r = {r.pearson_r:+.3f})")
        return "\n".join(lines)


def audit_dataset(df: pd.DataFrame,
                  by: Optional[str] = None,
                  features: Optional[Sequence[str]] = None,
                  unit_factor: float = 1000.0) -> AuditReport:
    """Run the standard guards and bundle the findings.

    Parameters
    ----------
    by : str, optional
        Category column for missing-by-design classification.
    features : sequence of str, optional
        Numeric feature set for the collinearity screen (skipped if
        not given).
    """
    numeric_cols = df.select_dtypes("number").columns
    unit_warnings = {}
    for c in numeric_cols:
        n = int(detect_unit_contamination(df[c],
                                          factor=unit_factor).sum())
        if n:
            unit_warnings[c] = n

    return AuditReport(
        shape=df.shape,
        n_duplicates=int(df.duplicated().sum()),
        missingness=missingness_report(df, by=by),
        unit_warnings=unit_warnings,
        collinearity=(collinearity_screen(df, features)
                      if features else None))


# ── Export ──────────────────────────────────────────────────────────

def export_results(models: dict,
                   path: str,
                   table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Write fitted relations and metrics to a CSV.

    One row per fitted :class:`SizingModel` in ``models`` (as returned
    by :func:`compare_families`): family, target, features, tuned
    hyperparameters, train/test metrics, and — for closed-form
    families including symbolic regression — the explicit equation
    string, ready for standardisation or downstream use.

    Returns the exported DataFrame.
    """
    rows = []
    for fam, m in models.items():
        eq = None
        try:
            eq = m.equation()
        except Exception:
            pass
        rows.append({
            "family": fam,
            "target": m.target,
            "features": " | ".join(m.features),
            "best_params": str(m.best_params_),
            **{k: v for k, v in m.metrics.items()
               if k not in ("family", "best_params")},
            "equation": eq or "(implicit model)",
        })
    out = pd.DataFrame(rows)
    if table is not None:
        skipped = table.index.difference(out["family"])
        for fam in skipped:
            out.loc[len(out)] = {"family": fam,
                                 "equation": str(
                                     table.loc[fam, "best_params"])}
    out.to_csv(path, index=False)
    return out
