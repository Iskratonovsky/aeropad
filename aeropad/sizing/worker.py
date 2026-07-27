"""Process-pool worker for GUI-driven sizing fits.

Lives in its own module (no GUI imports) so worker processes spawned
on Windows re-import only the lightweight sizing stack.
"""

from __future__ import annotations


def fit_family(family: str, df, features, target):
    """Fit one family; never raises — returns (family, model|None, err|None)."""
    try:
        from .regression import SizingModel
        m = SizingModel(family).fit(df, features, target)
        return family, m, None
    except Exception as exc:  # noqa: BLE001 — reported to the GUI
        return family, None, f"{type(exc).__name__}: {exc}"
