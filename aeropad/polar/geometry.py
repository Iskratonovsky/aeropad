"""
aeropad.polar.geometry
======================
Airfoil geometry parameter extraction from coordinate files.

The semi-empirical extrapolators require geometric parameters —
thickness ratio (TC), camber ratio (HC), leading-edge radius (RLE_C),
and nose/tail wedge angles — that are tedious to determine by hand.
This module computes all of them directly from a standard airfoil
coordinate file (Selig or Lednicer ``.dat`` format, as distributed by
the UIUC database and airfoiltools.com).

Typical use::

    from aeropad.polar.geometry import analyze_dat
    geo = analyze_dat("clark_y.dat")
    print(geo.summary())
    spec = geo.to_airfoil_spec(alpha_s_2D_pos=15.0, CL_s_2D_pos=1.52,
                               alpha_s_2D_neg=-11.0, CL_s_2D_neg=-0.78)

Conventions
-----------
- Coordinates are normalised to unit chord with the leading edge at
  x = 0 before analysis.
- The leading-edge radius is obtained by least-squares circle fit to
  the surface points nearest the LE (Kåsa fit).
- Nose/tail wedge angles are the opening angles between straight-line
  (secant) fits to the upper and lower surfaces over configurable
  chordwise windows at each end. Different post-stall methods define
  these windows differently; the defaults (nose 5%, tail 10% chord)
  are a reasonable general convention, and both are adjustable to
  match a specific method's definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── Parsing ─────────────────────────────────────────────────────────

def load_dat(path: str) -> tuple:
    """Load an airfoil ``.dat`` file (Selig or Lednicer format).

    Returns ``(name, xy)`` where ``xy`` is an (N, 2) array running
    continuously from the trailing edge over one surface to the LE and
    back along the other surface (Selig ordering; Lednicer files are
    converted).
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    name = lines[0].strip() if lines else Path(path).stem

    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(rows) < 6:
        raise ValueError(f"{path}: fewer than 6 coordinate pairs found.")
    arr = np.asarray(rows, dtype=float)

    # Lednicer format: first data line is the two point counts (>1.5),
    # followed by upper surface LE→TE then lower surface LE→TE.
    if arr[0, 0] > 1.5 or arr[0, 1] > 1.5:
        n_up, n_lo = int(round(arr[0, 0])), int(round(arr[0, 1]))
        pts = arr[1:]
        upper = pts[:n_up]              # LE → TE
        lower = pts[n_up:n_up + n_lo]   # LE → TE
        xy = np.vstack([upper[::-1], lower[1:]])  # TE→LE→TE (Selig)
    else:
        xy = arr
    return name, xy


# ── Geometry container ──────────────────────────────────────────────

@dataclass
class AirfoilGeometry:
    """Computed geometric parameters of an airfoil section."""
    name: str
    TC: float                 # max thickness / chord
    x_TC: float               # chordwise location of max thickness
    HC: float                 # max camber / chord (signed)
    x_HC: float               # chordwise location of max camber
    RLE_C: float              # leading-edge radius / chord
    delta_nose_deg: float     # camber-line slope angle at the LE (deg)
    delta_tail_deg: float     # camber-line slope angle at the TE (deg)
    wedge_nose_deg: float     # surface opening angle over the nose window
    wedge_tail_deg: float     # surface opening angle over the tail window
    symmetry: str             # "symmetric" | "asymmetric"
    n_points: int

    def summary(self) -> str:
        return "\n".join([
            f"Airfoil: {self.name}  ({self.n_points} points)",
            f"  symmetry:   {self.symmetry}",
            f"  TC   = {self.TC:.4f}  (at x/c = {self.x_TC:.3f})",
            f"  HC   = {self.HC:+.4f}  (at x/c = {self.x_HC:.3f})",
            f"  RLE/C = {self.RLE_C:.5f}",
            f"  delta_nose (camber slope @ LE) = {self.delta_nose_deg:.2f} deg",
            f"  delta_tail (camber slope @ TE) = {self.delta_tail_deg:.2f} deg",
            f"  surface wedge nose/tail = {self.wedge_nose_deg:.1f} / "
            f"{self.wedge_tail_deg:.2f} deg",
        ])

    def to_airfoil_spec(self, **stall_and_flow_params):
        """Build an :class:`aeropad.config.AirfoilSpec`.

        Geometric fields are filled from this analysis; stall
        characteristics (``alpha_s_2D_pos`` etc.) are aerodynamic and
        must be supplied by the caller.
        """
        from ..config import AirfoilSpec
        return AirfoilSpec(
            name=self.name,
            symmetry=self.symmetry,
            TC=self.TC,
            HC=abs(self.HC),
            RLE_C=self.RLE_C,
            delta_nose_deg=self.delta_nose_deg,
            delta_tail_deg=self.delta_tail_deg,
            **stall_and_flow_params)


# ── Analysis ────────────────────────────────────────────────────────

def _split_surfaces(xy: np.ndarray) -> tuple:
    """Split Selig-ordered points into upper/lower surfaces (LE→TE)."""
    i_le = int(np.argmin(xy[:, 0]))
    a = xy[:i_le + 1][::-1]   # LE → TE
    b = xy[i_le:]             # LE → TE
    if np.trapezoid(a[:, 1], a[:, 0]) >= np.trapezoid(b[:, 1], b[:, 0]):
        upper, lower = a, b
    else:
        upper, lower = b, a
    return upper, lower


def _le_radius_parabola(pts: np.ndarray) -> float:
    """Leading-edge radius via parabola curvature at the nose vertex.

    Near a rounded LE the surface is well described by x(y) = a + b·y
    + c·y²; at the vertex the tangent is vertical and the osculating
    radius is 1/(2|c|). This is markedly more accurate than a circle
    fit over the same window, which is biased by points that have
    already departed from the osculating circle.
    """
    x, y = pts[:, 0], pts[:, 1]
    if len(x) < 4:
        return float("nan")
    a, b, cc = np.polyfit(y, x, 2)[::-1]
    if abs(cc) < 1e-12:
        return float("nan")
    return float(1.0 / (2.0 * abs(cc)))


def _secant_angle_deg(x: np.ndarray, y: np.ndarray) -> float:
    """Slope angle (deg) of a least-squares line through (x, y)."""
    p = np.polyfit(x, y, 1)
    return float(np.degrees(np.arctan(p[0])))


def analyze_coordinates(xy: np.ndarray,
                        name: str = "airfoil",
                        nose_window: float = 0.05,
                        tail_window: float = 0.10,
                        le_fit_window: float = 0.01,
                        symmetry_tol: float = 0.004) -> AirfoilGeometry:
    """Compute geometric parameters from raw Selig-ordered coordinates.

    Parameters
    ----------
    nose_window, tail_window : float
        Chord fractions over which the upper/lower surface secants are
        fitted for the wedge angles. Match these to the convention of
        the post-stall method the parameters will feed.
    le_fit_window : float
        Chord fraction of points included in the LE circle fit.
    symmetry_tol : float
        |max camber| below which the section is declared symmetric.
    """
    xy = np.asarray(xy, dtype=float)

    # Normalise: LE at x=0, unit chord
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError("Degenerate coordinates: zero chord.")
    xy = (xy - [x_min, 0.0]) / chord

    upper, lower = _split_surfaces(xy)

    # Common cosine-spaced grid for thickness/camber distributions
    xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 201)))
    xg = np.clip(xg, upper[:, 0].min(), upper[:, 0].max())
    yu = np.interp(xg, upper[:, 0], upper[:, 1])
    yl = np.interp(xg, lower[:, 0], lower[:, 1])

    t = yu - yl
    c = 0.5 * (yu + yl)
    i_t, i_c = int(np.argmax(t)), int(np.argmax(np.abs(c)))

    # LE radius: parabola-vertex curvature over the nose points.
    # The window adapts outward if the coordinate file is sparse near
    # the LE (at least 5 points are required for a stable fit).
    w = le_fit_window
    le_pts = xy[xy[:, 0] <= w]
    while len(le_pts) < 5 and w < 0.08:
        w *= 1.6
        le_pts = xy[xy[:, 0] <= w]
    rle = _le_radius_parabola(le_pts)

    # Wedge angles from surface secants over the end windows
    def window(surface, lo, hi):
        m = (surface[:, 0] >= lo) & (surface[:, 0] <= hi)
        return surface[m]

    nu, nl = window(upper, 0.0, nose_window), window(lower, 0.0, nose_window)
    tu, tl = (window(upper, 1 - tail_window, 1.0),
              window(lower, 1 - tail_window, 1.0))
    w_nose = abs(_secant_angle_deg(nu[:, 0], nu[:, 1])
                 - _secant_angle_deg(nl[:, 0], nl[:, 1])) \
        if len(nu) >= 2 and len(nl) >= 2 else float("nan")
    w_tail = abs(_secant_angle_deg(tu[:, 0], tu[:, 1])
                 - _secant_angle_deg(tl[:, 0], tl[:, 1])) \
        if len(tu) >= 2 and len(tl) >= 2 else float("nan")

    # Camber-line slope angles at LE and TE — the convention used
    # for the Lindenburg delta_nose/delta_tail inputs. A quadratic is
    # fitted to the camber distribution over each end window and its
    # tangent evaluated at the endpoint itself, which converges to the
    # true LE/TE camber slope rather than the window-averaged secant.
    def _endpoint_slope_deg(xs, ys, at):
        if len(xs) < 3:
            return float("nan")
        q = np.polyfit(xs, ys, 2)
        return float(np.degrees(np.arctan(2 * q[0] * at + q[1])))

    mn = xg <= nose_window
    mt = xg >= 1 - tail_window
    d_nose = abs(_endpoint_slope_deg(xg[mn], c[mn], 0.0))
    d_tail = abs(_endpoint_slope_deg(xg[mt], c[mt], 1.0))

    hc = float(c[i_c])
    return AirfoilGeometry(
        name=name,
        TC=float(t[i_t]), x_TC=float(xg[i_t]),
        HC=hc, x_HC=float(xg[i_c]),
        RLE_C=rle,
        delta_nose_deg=d_nose,
        delta_tail_deg=d_tail,
        wedge_nose_deg=w_nose,
        wedge_tail_deg=w_tail,
        symmetry="symmetric" if abs(hc) < symmetry_tol else "asymmetric",
        n_points=len(xy))


def analyze_dat(path: str, **kwargs) -> AirfoilGeometry:
    """Load a ``.dat`` file and compute its geometric parameters."""
    name, xy = load_dat(path)
    return analyze_coordinates(xy, name=name, **kwargs)


# ── NACA 4-digit generator (validation + convenience) ───────────────

def naca4_coordinates(code: str, n: int = 120,
                      closed_te: bool = True) -> np.ndarray:
    """Generate Selig-ordered coordinates for a NACA 4-digit section.

    Provided both as a convenience and as the module's validation
    anchor: 4-digit sections have analytic TC, HC and LE radius
    (r_LE/c = 1.1019 t²), against which the extractor can be checked.
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError("code must be a 4-digit string, e.g. '2412'")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 + a4 * x ** 4)

    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p,
                      m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p,
                       2 * m / p ** 2 * (p - x),
                       2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)

    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.vstack([np.column_stack([xu, yu])[::-1],
                      np.column_stack([xl, yl])[1:]])
