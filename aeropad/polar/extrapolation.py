"""
aeropad.polar.extrapolation
===========================
Semi-empirical full-range polar reconstruction from low-fidelity data.

Four interchangeable extrapolation methods handle CL and CD across
the full ±180° angle-of-attack range. The active method is selected
via ``pipeline.extrapolator`` in the case configuration.

:class:`BattistiExtrapolator`
    Battisti et al. (2020).

:class:`AERODASExtrapolator`
    Spera (2008) NASA/CR-2008-215434.

:class:`MontgomerieExtrapolator`
    Montgomerie (2004) FOI-R--1305--SE. CL and CD only.

:class:`LindenburgExtrapolator`
    Lindenburg (2003) ECN-C--03-025.

One additional extrapolator handles CM only and always runs alongside
whichever CL/CD method is active when ``run_CM=True``:

:class:`MontgomerieCMExtrapolator`
    Montgomerie (2004) CM arm method.

Use :func:`get_extrapolator` to obtain the active CL/CD extrapolator.

Notes
-----
No extrapolator performs file I/O. All methods receive DataFrames and
return DataFrames with columns ``AoA``, ``CL_full``, ``CD_full``, and
``source`` (which segment of the composite each point came from:
LF, blend, or extrapolation).

References
----------
- Battisti, L. et al. (2020). Wind Turbines in Cold Climates. Eqs. 10-22, 27-29.
- Spera, D. (2008). NASA/CR-2008-215434 (AERODAS).
- Montgomerie, B. (2004). FOI-R--1305--SE.
- Lindenburg, C. (2003). ECN-C--03-025.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..config import CaseConfig


# ── Shared helpers ────────────────────────────────────────────────────────────

def _derive_alpha0(df: pd.DataFrame,
                   lf_cl_col: str,
                   symmetry: str) -> float:
    """Interpolate zero-lift angle from LF CL data.

    For symmetric airfoils returns 0.0 exactly.
    For asymmetric airfoils interpolates CL_LF = 0 in [-10°, 10°].
    """
    if symmetry == "symmetric":
        return 0.0
    lf = df.dropna(subset=[lf_cl_col])
    lf = lf[(lf["AoA"] >= -10) & (lf["AoA"] <= 10)]
    f  = interp1d(lf[lf_cl_col].values, lf["AoA"].values, kind="linear")
    return float(f(0.0))


def _build_lf_cl_interp(df: pd.DataFrame,
                         lf_cl_col: str) -> interp1d:
    """Return a linear interpolator for LF CL."""
    lf = df.dropna(subset=[lf_cl_col])
    return interp1d(lf["AoA"], lf[lf_cl_col],
                    kind="linear", fill_value="extrapolate",
                    bounds_error=False)


def _build_lf_cd_interp(df: pd.DataFrame,
                         lf_cd_col: str) -> interp1d | None:
    """Return a linear interpolator for LF CD, or None if column missing."""
    if lf_cd_col not in df.columns or df[lf_cd_col].isna().all():
        return None
    lf = df.dropna(subset=[lf_cd_col])
    return interp1d(lf["AoA"], lf[lf_cd_col],
                    kind="linear", fill_value="extrapolate",
                    bounds_error=False)


# ── Abstract base ─────────────────────────────────────────────────────────────

class ExtrapolatorBase(ABC):
    """Abstract base for CL/CD composite-dataset extrapolators.

    All concrete extrapolators must implement :meth:`build`.
    The :attr:`ALPHA0` property exposes the zero-lift angle for use
    by :class:`MontgomerieCMExtrapolator`.

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.
    """

    def __init__(self, config: CaseConfig) -> None:
        self.config   = config
        self._alpha0  : float | None = None
        self._derived : bool         = False

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full ±180° composite dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame. AoA must be in standard aerodynamic convention
            (sign flip already applied by the pipeline).

        Returns
        -------
        pd.DataFrame
            Columns: ``AoA``, ``CL_full``, ``CD_full``, ``source``.
        """
        ...

    @property
    def ALPHA0(self) -> float | None:
        """Zero-lift angle of attack (degrees). ``None`` if not yet built."""
        return self._alpha0

    @property
    def parameters(self) -> dict:
        """Derived parameters as a dictionary. Empty if not yet built."""
        return {}


# ── Factory ───────────────────────────────────────────────────────────────────

def get_extrapolator(config: CaseConfig) -> ExtrapolatorBase:
    """Return the CL/CD extrapolator specified in ``config.pipeline.extrapolator``.

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Returns
    -------
    ExtrapolatorBase

    Raises
    ------
    ValueError
        If the extrapolator name is not recognised.
    """
    _dispatch = {
        "battisti"   : BattistiExtrapolator,
        "aerodas"    : AERODASExtrapolator,
        "montgomerie": MontgomerieExtrapolator,
        "lindenburg" : LindenburgExtrapolator,
    }
    method = config.pipeline.extrapolator.lower()
    if method not in _dispatch:
        raise ValueError(
            f"Unknown extrapolator '{method}'. "
            f"Valid options: {sorted(_dispatch)}."
        )
    return _dispatch[method](config)


# ── Battisti (2020) ───────────────────────────────────────────────────────────

class BattistiExtrapolator(ExtrapolatorBase):
    """Build the full-range CL and CD composite dataset using Battisti (2020).

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Attributes
    ----------
    CD90, CD270 : float
        Drag at 90° (suction) and 270° (pressure). Set after :meth:`build`.
    CD_F : float
        Skin friction coefficient (Prandtl-Schlichting).

    Examples
    --------
    >>> ext = BattistiExtrapolator(cfg)
    >>> polar = ext.build(df)
    >>> print(ext.parameters)

    References
    ----------
    Battisti, L. et al. (2020). Eqs. 10-22, 27-29.
    """

    _CL_SINGULARITY_GUARD: float = 1e-4

    def __init__(self, config: CaseConfig) -> None:
        super().__init__(config)
        self.CD90            = None
        self.CD270           = None
        self.CD_F            = None
        self._alpha_ds_pos   = None
        self._alpha_ds_neg   = None

    # ── Public ────────────────────────────────────────────────────────────────

    def derive_parameters(self, df: pd.DataFrame) -> None:
        """Compute Battisti geometric and flow parameters from LF data."""
        g  = self.config.airfoil
        fl = self.config.flow
        d  = self.config.data
        p  = self.config.pipeline

        # ── CD90 / CD270 (Eqs. 10-12) ────────────────────────────────────────
        if g.symmetry == "symmetric":
            self.CD90 = self.CD270 = 1.98 - 0.64*g.RLE_C - 0.44*g.TC
        else:
            self.CD90  = 1.98 - 0.64*g.RLE_C - 0.44*g.TC + 1.39*g.HC
            self.CD270 = 1.98 - 0.64*g.RLE_C - 0.44*g.TC - 1.39*g.HC

        # ── CD_F (Eq. 18) ─────────────────────────────────────────────────────
        self.CD_F = 0.455 / (np.log10(fl.Re) ** 2.58) - 1700.0 / fl.Re

        # ── ALPHA0 ────────────────────────────────────────────────────────────
        self._alpha0 = _derive_alpha0(df, d.LF_CL_column, g.symmetry)

        # ── αDS ───────────────────────────────────────────────────────────────
        self._alpha_ds_pos =  p.PM_cutoff
        self._alpha_ds_neg = -p.PM_cutoff

        self._derived = True

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full ±180° composite dataset.

        Calls :meth:`derive_parameters` internally if not yet called.
        """
        if not self._derived:
            self.derive_parameters(df)

        g = self.config.airfoil
        d = self.config.data
        p = self.config.pipeline

        f_cl_lf = _build_lf_cl_interp(df, d.LF_CL_column)
        f_cd_lf = _build_lf_cd_interp(df, d.LF_CD_column)
        if f_cd_lf is None:
            warnings.warn(
                f'"{d.LF_CD_column}" not found or all NaN. '
                "Battisti blend anchor uses analytical CD(α) approximation.",
                UserWarning, stacklevel=2,
            )

        out = self._build_curve(f_cl_lf, f_cd_lf,
                                p.PM_cutoff, p.blend_end, p.fine_step)

        if p.apply_3D_correction:
            out["CD_full"] = [
                self._CD_3D_correction(a, cd, g.AR)
                for a, cd in zip(out["AoA"], out["CD_full"])
            ]
        return out

    @property
    def parameters(self) -> dict:
        if not self._derived:
            raise RuntimeError(
                "Parameters not yet derived. Call build() or derive_parameters() first."
            )
        return {
            "CD90"  : self.CD90,
            "CD270" : self.CD270,
            "CD_F"  : self.CD_F,
            "ALPHA0": self._alpha0,
        }

    # ── Private: curve builder ────────────────────────────────────────────────

    def _build_curve(self, f_cl_lf, f_cd_lf,
                     pm_cutoff: float,
                     blend_end: float,
                     fine_step: float) -> pd.DataFrame:
        aoa_fine = np.arange(-180.0, 180.0 + fine_step, fine_step)
        cl_out, cd_out, source = [], [], []

        for a in aoa_fine:
            a_abs    = abs(a)
            alpha_ds = self._alpha_ds_pos if a >= 0 else self._alpha_ds_neg

            if a_abs <= pm_cutoff:
                cl_val = float(f_cl_lf(a))
                cd_val = (float(f_cd_lf(a)) if f_cd_lf is not None
                          else abs(self._CD_alpha_2D(
                              a, self.CD90, self.CD270, self._alpha0)))
                cl_out.append(cl_val)
                cd_out.append(cd_val)
                source.append("LF")

            elif a_abs <= blend_end:
                f = self._blend_f(a, alpha_ds)

                cl_ds = float(f_cl_lf(alpha_ds))
                cd_ds = (float(f_cd_lf(alpha_ds)) if f_cd_lf is not None
                         else abs(self._CD_alpha_2D(
                             alpha_ds, self.CD90, self.CD270, self._alpha0)))
                CN_ds, CT_ds = self._cl_cd_to_cn_ct(alpha_ds, cl_ds, cd_ds)

                CDa    = self._CD_alpha_2D(a, self.CD90, self.CD270, self._alpha0)
                CN_sep = self._CN(a, CDa, self.config.airfoil.TC)
                CT_sep = self._CT(a, CDa, self.config.airfoil.TC, self.CD_F)

                CN_b  = CN_ds * f + CN_sep * (1.0 - f)
                CT_b  = CT_ds * f + CT_sep * (1.0 - f)
                a_rad = np.radians(a)

                cl_out.append(CN_b * np.cos(a_rad) + CT_b * np.sin(a_rad))
                cd_out.append(abs(CN_b * np.sin(a_rad) - CT_b * np.cos(a_rad)))
                source.append("blend")

            else:
                CL_s, CD_s = self._CL_CD_sep(
                    a, self.CD90, self.CD270, self._alpha0,
                    self.config.airfoil.TC, self.CD_F,
                )
                cl_out.append(CL_s)
                cd_out.append(abs(CD_s))
                source.append("battisti")

        return pd.DataFrame({"AoA": aoa_fine, "CL_full": cl_out,
                              "CD_full": cd_out, "source": source})

    # ── Private: Battisti core equations (all static, pure functions) ─────────

    @staticmethod
    def _beta_star(alpha_deg: float, alpha0_deg: float) -> float:
        return alpha_deg - alpha0_deg * np.cos(np.radians(alpha_deg))

    @staticmethod
    def _CD_alpha_2D(alpha_deg: float, CD90: float,
                     CD270: float, alpha0_deg: float) -> float:
        b = BattistiExtrapolator._beta_star(alpha_deg, alpha0_deg)
        return ((CD90 + CD270) / 2.0
                + (CD90 - CD270) / 2.0 * np.sin(np.radians(b)))

    @staticmethod
    def _CN(alpha_deg: float, CD_alpha: float, TC: float) -> float:
        a_rad  = np.radians(alpha_deg)
        sin_a  = np.sin(a_rad)
        sin_2a = np.sin(2.0 * a_rad)
        cos_a  = np.cos(a_rad)
        return CD_alpha * (sin_a + 0.0023 * sin_2a) / (
            0.38 + 0.62 * abs(sin_a) + 3.7 * TC * cos_a ** 8
        )

    @staticmethod
    def _CT(alpha_deg: float, CD_alpha: float,
            TC: float, CD_F: float) -> float:
        a_rad  = np.radians(alpha_deg)
        sin_a  = np.sin(a_rad)
        sin_2a = np.sin(2.0 * a_rad)
        cos_a  = np.cos(a_rad)
        return (CD_alpha * 0.3 * TC
                * abs(sin_a + 0.1 * sin_2a) * (1.0 - 2.0 * cos_a)
                - CD_F * cos_a)

    @staticmethod
    def _CL_CD_sep(alpha_deg: float, CD90: float, CD270: float,
                   alpha0_deg: float, TC: float,
                   CD_F: float) -> tuple[float, float]:
        a_rad = np.radians(alpha_deg)
        CDa   = BattistiExtrapolator._CD_alpha_2D(alpha_deg, CD90, CD270, alpha0_deg)
        CN    = BattistiExtrapolator._CN(alpha_deg, CDa, TC)
        CT    = BattistiExtrapolator._CT(alpha_deg, CDa, TC, CD_F)
        return (CN * np.cos(a_rad) + CT * np.sin(a_rad),
                CN * np.sin(a_rad) - CT * np.cos(a_rad))

    @staticmethod
    def _cl_cd_to_cn_ct(alpha_deg: float,
                         CL: float, CD: float) -> tuple[float, float]:
        a_rad = np.radians(alpha_deg)
        return (CL * np.cos(a_rad) + CD * np.sin(a_rad),
                CL * np.sin(a_rad) - CD * np.cos(a_rad))

    @staticmethod
    def _blend_f(alpha_deg: float, alpha_DS: float) -> float:
        return float(np.clip(
            ((abs(alpha_deg) - 45.0) / (abs(alpha_DS) - 45.0)) ** 2,
            0.0, 1.0,
        ))

    @staticmethod
    def _CD_3D_correction(alpha_deg: float,
                           CD_val: float,
                           AR: float | None) -> float:
        if AR is None:
            return CD_val
        sin_a = abs(np.sin(np.radians(alpha_deg)))
        if sin_a < 1e-6:
            return CD_val
        AR_eff = AR / sin_a
        return CD_val * (1.0 - 0.40 * (1.0 - np.exp(-11.4 / AR_eff)))


# ── AERODAS / Viterna-Corrigan (Spera 2008) ───────────────────────────────────

class AERODASExtrapolator(ExtrapolatorBase):
    """Build the full-range CL and CD composite dataset using the AERODAS model (Spera 2008).

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Notes
    -----
    Stall angles (ACL1) are estimated via:

    1. **Prandtl finite-wing correction** applied to user-supplied 2D stall
       data (``airfoil.alpha_s_2D_pos``, ``airfoil.CL_s_2D_pos``, etc.).
    2. **Fallback** — if 2D stall data is not supplied, ``ACL1`` is set
       to ``PM_cutoff`` directly. A :class:`UserWarning` is raised.

    References
    ----------
    Spera, D. (2008). NASA/CR-2008-215434. Eqs. 3, 6-12.
    """

    def __init__(self, config: CaseConfig) -> None:
        super().__init__(config)
        self._params: dict = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def derive_parameters(self, df: pd.DataFrame) -> None:
        """Derive AERODAS parameters from LF data and config geometry."""
        g  = self.config.airfoil
        fl = self.config.flow
        d  = self.config.data
        p  = self.config.pipeline

        f_cl_lf = _build_lf_cl_interp(df, d.LF_CL_column)
        f_cd_lf = _build_lf_cd_interp(df, d.LF_CD_column)

        # ── A0, S1, CD0 from PM ───────────────────────────────────────────────
        self._alpha0 = _derive_alpha0(df, d.LF_CL_column, g.symmetry)
        A0 = self._alpha0

        slope_pts = df[(df["AoA"] >= 2) & (df["AoA"] <= 15)].dropna(
            subset=[d.LF_CL_column])
        S1 = float(np.polyfit(slope_pts["AoA"].values,
                               slope_pts[d.LF_CL_column].values, 1)[0])
        CD0 = float(f_cd_lf(A0)) if f_cd_lf is not None else 0.01

        # ── CL2max, CD2max from geometry (Eqs. 9-10) ─────────────────────────
        TC, AR = g.TC, g.AR if g.AR is not None else 1e6
        CL2max = 1.190 * (1.0 - TC**2) * (0.65 + 0.35 * np.exp(-(9.0 / AR)**2.3))
        CD2max = 2.300 * np.exp(-(0.65 * TC)**0.90) \
                 * (0.52 + 0.48 * np.exp(-(6.5 / AR)**1.1))

        # ── ACL1 — positive branch ────────────────────────────────────────────
        if g.alpha_s_2D_pos is not None and g.CL_s_2D_pos is not None:
            delta_pos = np.degrees(g.CL_s_2D_pos / (np.pi * AR))
            ACL1 = g.alpha_s_2D_pos + delta_pos
        else:
            warnings.warn(
                "airfoil.alpha_s_2D_pos / CL_s_2D_pos not provided. "
                f"AERODAS ACL1 (positive) set to PM_cutoff = {p.PM_cutoff}°. "
                "For better accuracy supply 2D stall data from literature.",
                UserWarning, stacklevel=3,
            )
            ACL1 = p.PM_cutoff

        # ── ACL1 — negative branch ────────────────────────────────────────────
        if g.alpha_s_2D_neg is not None and g.CL_s_2D_neg is not None:
            delta_neg = np.degrees(g.CL_s_2D_neg / (np.pi * AR))
            ACL1_neg = g.alpha_s_2D_neg + delta_neg
        else:
            warnings.warn(
                "airfoil.alpha_s_2D_neg / CL_s_2D_neg not provided. "
                f"AERODAS ACL1 (negative) set to -PM_cutoff = -{p.PM_cutoff}°. "
                "For better accuracy supply 2D stall data from literature.",
                UserWarning, stacklevel=3,
            )
            ACL1_neg = -p.PM_cutoff

        CL1max     = float(f_cl_lf(ACL1))
        CL1max_neg = abs(float(f_cl_lf(ACL1_neg)))
        CD1max     = float(f_cd_lf(ACL1))     if f_cd_lf is not None else CD0
        CD1max_neg = float(f_cd_lf(ACL1_neg)) if f_cd_lf is not None else CD0

        self._params = dict(
            A0=A0, S1=S1, CD0=CD0,
            CL2max=CL2max, CD2max=CD2max,
            ACL1=ACL1, ACL1_neg=ACL1_neg,
            CL1max=CL1max, CL1max_neg=CL1max_neg,
            CD1max=CD1max, CD1max_neg=CD1max_neg,
        )
        self._f_cl_lf = f_cl_lf
        self._f_cd_lf = f_cd_lf
        self._derived = True

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full ±180° AERODAS composite dataset."""
        if not self._derived:
            self.derive_parameters(df)

        p   = self.config.param = self.config.pipeline
        pm  = p.PM_cutoff
        step = p.fine_step
        pr  = self._params

        aoa_fine = np.arange(-180.0, 180.0 + step, step)
        cl_out, cd_out, source = [], [], []

        for a in aoa_fine:
            if abs(a) <= pm:
                cl_out.append(float(self._f_cl_lf(a)))
                cd_out.append(float(self._f_cd_lf(a))
                               if self._f_cd_lf is not None else 0.0)
                source.append("PM")
            elif a >= 0:
                cl, cd = self._aerodas_cl_cd(
                    a, pr["A0"], pr["S1"], pr["ACL1"], pr["CL1max"],
                    pr["CD0"], pr["CD1max"], pr["CL2max"], pr["CD2max"])
                cl_out.append(cl); cd_out.append(cd)
                source.append("aerodas_pos")
            else:
                cl, cd = self._aerodas_cl_cd(
                    -a, pr["A0"], pr["S1"], abs(pr["ACL1_neg"]),
                    pr["CL1max_neg"], pr["CD0"], pr["CD1max_neg"],
                    pr["CL2max"], pr["CD2max"])
                cl_out.append(-cl); cd_out.append(cd)
                source.append("aerodas_neg")

        return pd.DataFrame({"AoA": aoa_fine, "CL_full": cl_out,
                              "CD_full": cd_out, "source": source})

    @property
    def parameters(self) -> dict:
        if not self._derived:
            raise RuntimeError(
                "Parameters not yet derived. Call build() first."
            )
        return dict(self._params)

    # ── Private: AERODAS core equations ──────────────────────────────────────

    @staticmethod
    def _prestall_cl(alpha_deg: float, A0: float, S1: float,
                     ACL1: float, CL1max: float) -> float:
        """AERODAS pre-stall CL — Spera Eqs. 6a/6b."""
        RCL1 = S1 * (ACL1 - A0) - CL1max
        N1   = 1.0 + CL1max / RCL1
        if alpha_deg >= A0:
            return S1*(alpha_deg - A0) - RCL1*((alpha_deg - A0)/(ACL1 - A0))**N1
        else:
            return S1*(alpha_deg - A0) + RCL1*((A0 - alpha_deg)/(ACL1 - A0))**N1

    @staticmethod
    def _prestall_cd(alpha_deg: float, A0: float, ACL1: float,
                     CD0: float, CD1max: float, M: float = 2.0) -> float:
        """AERODAS pre-stall CD — Spera Eq. 7a."""
        lo = 2.0 * A0 - ACL1
        if lo <= alpha_deg <= ACL1:
            return CD0 + (CD1max - CD0) * ((alpha_deg - A0) / (ACL1 - A0)) ** M
        return 0.0

    @staticmethod
    def _poststall_cl(alpha_deg: float, A0: float,
                      ACL1: float, CL2max: float) -> float:
        """AERODAS post-stall CL — Spera Eqs. 11b/11c."""
        RCL2 = 1.632 - CL2max
        if RCL2 <= 0:
            raise ValueError(f"RCL2={RCL2:.4f} — CL2max={CL2max:.4f} exceeds 1.632.")
        N2 = 1.0 + CL2max / RCL2

        if alpha_deg < 0:
            mapped = -alpha_deg + 2.0 * A0
            if mapped < ACL1:
                return 0.0
            return -AERODASExtrapolator._poststall_cl(mapped, A0, ACL1, CL2max)
        if alpha_deg < ACL1:
            return 0.0
        elif alpha_deg <= 92.0:
            return -0.032*(alpha_deg - 92.0) - RCL2*((92.0 - alpha_deg)/51.0)**N2
        else:
            return -0.032*(alpha_deg - 92.0) + RCL2*((alpha_deg - 92.0)/51.0)**N2

    @staticmethod
    def _poststall_cd(alpha_deg: float, A0: float, ACL1: float,
                      CD1max: float, CD2max: float) -> float:
        """AERODAS post-stall CD — Spera Eq. 12b."""
        if alpha_deg <= (2.0 * A0 - ACL1):
            mapped = -alpha_deg + 2.0 * A0
            if mapped < ACL1:
                return 0.0
            return AERODASExtrapolator._poststall_cd(mapped, A0, ACL1,
                                                      CD1max, CD2max)
        if alpha_deg < ACL1:
            return 0.0
        arg = (alpha_deg - ACL1) / (90.0 - ACL1) * 90.0
        return CD1max + (CD2max - CD1max) * np.sin(np.radians(arg))

    @staticmethod
    def _aerodas_cl_cd(alpha_deg: float, A0: float, S1: float,
                        ACL1: float, CL1max: float,
                        CD0: float, CD1max: float,
                        CL2max: float, CD2max: float) -> tuple[float, float]:
        """Full AERODAS CL and CD — additive blending (Spera Eq. 3)."""
        lo = 2.0 * A0 - ACL1
        if lo <= alpha_deg <= ACL1:
            CL1 = AERODASExtrapolator._prestall_cl(alpha_deg, A0, S1, ACL1, CL1max)
            CD1 = AERODASExtrapolator._prestall_cd(alpha_deg, A0, ACL1, CD0, CD1max)
        else:
            CL1 = CD1 = 0.0
        CL2 = AERODASExtrapolator._poststall_cl(alpha_deg, A0, ACL1, CL2max)
        CD2 = AERODASExtrapolator._poststall_cd(alpha_deg, A0, ACL1, CD1max, CD2max)
        return CL1 + CL2, abs(CD1 + CD2)


# ── Montgomerie (2004) — CL and CD ────────────────────────────────────────────

class MontgomerieExtrapolator(ExtrapolatorBase):
    """Build the full-range CL and CD composite dataset using Montgomerie (2004).

    This extrapolator handles CL and CD only. CM is handled separately by
    :class:`MontgomerieCMExtrapolator`.

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Notes
    -----
    Blending parameters ``alphaM`` and ``k`` are fitted independently for
    the positive and negative AoA branches from two PM anchor points per
    branch (FOI-R--1305--SE §2.2.2, Eqs. 4-9).

    Constants used from Montgomerie defaults:
    - ``CL90 = 0.036`` (§2.2.4 default)
    - ``CD_FRICTION = 0.006`` (§2.3.2 default)

    References
    ----------
    Montgomerie, B. (2004). FOI-R--1305--SE.
    """

    #: CL at 90° — Montgomerie §2.2.4 default.
    _CL90: float = 0.036
    #: Friction drag — Montgomerie §2.3.2 default.
    _CD_FRICTION: float = 0.006

    def __init__(self, config: CaseConfig) -> None:
        super().__init__(config)
        self._params: dict = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def derive_parameters(self, df: pd.DataFrame) -> None:
        """Fit Montgomerie blending parameters from LF data."""
        g = self.config.airfoil
        d = self.config.data
        p = self.config.pipeline

        f_cl_lf = _build_lf_cl_interp(df, d.LF_CL_column)
        f_cd_lf = _build_lf_cd_interp(df, d.LF_CD_column)

        # ── A0, CLalpha, CL0, CD0 ─────────────────────────────────────────────
        self._alpha0 = _derive_alpha0(df, d.LF_CL_column, g.symmetry)
        A0      = self._alpha0
        slp_pts = df[(df["AoA"] >= 2) & (df["AoA"] <= 15)].dropna(
            subset=[d.LF_CL_column])
        CLalpha = float(np.polyfit(slp_pts["AoA"].values,
                                    slp_pts[d.LF_CL_column].values, 1)[0])
        CL0 = float(f_cl_lf(0.0))

        # ── CD90 (FOI §2.3.1) ─────────────────────────────────────────────────
        CD_THINPLATE = 1.45
        CD90_pos = (CD_THINPLATE - 0.83*g.RLE_C - 1.46*(g.TC/2) + 1.46*g.HC)
        CD90_neg = (CD_THINPLATE - 0.83*g.RLE_C - 1.46*(g.TC/2) - 1.46*g.HC)

        # ── CLmax / CLmin for fitting point selection ──────────────────────────
        CLmax = float(f_cl_lf( p.PM_cutoff))
        CLmin = float(f_cl_lf(-p.PM_cutoff))

        # ── DALPHA_MIN ────────────────────────────────────────────────────────
        DALPHA_MIN = (CLmin - CL0) / (2.0 * np.pi) - A0

        # ── Fit k and alphaM per branch ───────────────────────────────────────
        is_symmetric = self.config.airfoil.symmetry == "symmetric"
        (a1p, f1p), (a2p, f2p), \
        (a1n, f1n), (a2n, f2n) = self._select_fitting_points(
            CL0, CLalpha, CLmax, CLmin, A0,
            self._CL90, CD90_pos, CD90_neg, DALPHA_MIN,
            symmetric=is_symmetric,
        )
        alphaM_pos, k_pos = self._fit_k_alphaM(a1p, f1p, a2p, f2p)
        alphaM_neg, k_neg = self._fit_k_alphaM(a1n, f1n, a2n, f2n)

        assert k_pos > 0, f"k_pos={k_pos:.6f} must be positive"
        assert k_neg > 0, f"k_neg={k_neg:.6f} must be positive"

        self._params = dict(
            A0=A0, CLalpha=CLalpha, CL0=CL0,
            CD90_pos=CD90_pos, CD90_neg=CD90_neg,
            alphaM_pos=alphaM_pos, k_pos=k_pos,
            alphaM_neg=alphaM_neg, k_neg=k_neg,
        )
        self._f_cl_lf = f_cl_lf
        self._f_cd_lf = f_cd_lf
        self._derived = True

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full ±180° Montgomerie composite dataset."""
        if not self._derived:
            self.derive_parameters(df)

        p    = self.config.pipeline
        pm   = p.PM_cutoff
        step = p.fine_step
        pr   = self._params

        aoa_fine = np.arange(-180.0, 180.0 + step, step)
        cl_out, cd_out, source = [], [], []

        for a in aoa_fine:
            if abs(a) <= pm:
                cl_out.append(float(self._f_cl_lf(a)))
                cd_out.append(float(self._f_cd_lf(a))
                               if self._f_cd_lf is not None else 0.0)
                source.append("PM")
            elif a >= 0:
                cl = self._montgomerie_cl(
                    a, pr["A0"], pr["CL0"], pr["CLalpha"],
                    self._CL90, pr["CD90_pos"],
                    pr["alphaM_pos"], pr["k_pos"])
                cd = self._montgomerie_cd(
                    a, pr["A0"], pr["CL0"], pr["CLalpha"],
                    self._CL90, pr["CD90_pos"],
                    pr["alphaM_pos"], pr["k_pos"], self._f_cd_lf)
                cl_out.append(cl); cd_out.append(abs(cd))
                source.append("montgomerie_pos")
            else:
                cl = self._montgomerie_cl(
                    a, pr["A0"], pr["CL0"], pr["CLalpha"],
                    self._CL90, pr["CD90_neg"],
                    pr["alphaM_neg"], pr["k_neg"])
                cd = self._montgomerie_cd(
                    a, pr["A0"], pr["CL0"], pr["CLalpha"],
                    self._CL90, pr["CD90_neg"],
                    pr["alphaM_neg"], pr["k_neg"], self._f_cd_lf)
                cl_out.append(cl); cd_out.append(abs(cd))
                source.append("montgomerie_neg")

        return pd.DataFrame({"AoA": aoa_fine, "CL_full": cl_out,
                              "CD_full": cd_out, "source": source})

    @property
    def parameters(self) -> dict:
        if not self._derived:
            raise RuntimeError(
                "Parameters not yet derived. Call build() first."
            )
        return dict(self._params)

    # ── Private: Montgomerie core equations ───────────────────────────────────

    @staticmethod
    def _s_func(alpha_deg: float, A0: float, CL0: float,
                 CL90: float, CD90: float) -> float:
        """Thin-plate (separated) CL — FOI §2.2.4."""
        a_rad  = np.radians(alpha_deg)
        delta1 = 57.6 * CL90 * np.sin(a_rad)
        delta2 = A0 * np.cos(a_rad)
        beta   = np.radians(alpha_deg - delta1 - delta2)
        A_amp  = 1.0 + (CL0 / np.sin(np.radians(45.0))) * np.sin(a_rad)
        return A_amp * CD90 * np.sin(beta) * np.cos(beta)

    @staticmethod
    def _t_func(alpha_deg: float, A0: float,
                 CL0: float, CLalpha: float) -> float:
        """Potential-flow (attached) CL — FOI §2.2.3."""
        return CL0 + CLalpha * (alpha_deg - A0)

    @staticmethod
    def _f_func(alpha_deg: float, alphaM: float, k: float) -> float:
        """Blending function — FOI §2.2.2, Eq. 6."""
        return 1.0 / (1.0 + k * (alphaM - alpha_deg) ** 4)

    @staticmethod
    def _compute_f_from_curve(alpha_deg: float, CL_val: float,
                               A0: float, CL0: float, CLalpha: float,
                               CL90: float, CD90: float) -> float:
        """Invert f = (CL - s) / (t - s) at a known CL point."""
        t = MontgomerieExtrapolator._t_func(alpha_deg, A0, CL0, CLalpha)
        s = MontgomerieExtrapolator._s_func(alpha_deg, A0, CL0, CL90, CD90)
        denom = t - s
        if abs(denom) < 1e-10:
            raise ValueError(f"t ≈ s at α={alpha_deg}° — cannot compute f")
        return (CL_val - s) / denom

    @staticmethod
    def _fit_k_alphaM(alpha1: float, f1: float,
                       alpha2: float, f2: float) -> tuple[float, float]:
        """Fit k and alphaM from two (alpha, f) points — FOI Eqs. 7-9."""
        import warnings

        # Guard: f must be in (0, 1) for G = (1/f - 1)^0.25 to be real.
        # f >= 1 occurs on symmetric airfoils where CL at PM_cutoff slightly
        # exceeds the linear theory at the anchor point (structural issue —
        # reducing PM_cutoff does not help). Clamp to 0.95 as a fallback.
        for name, f in [("f1", f1), ("f2", f2)]:
            if f >= 1.0 or f <= 0.0:
                warnings.warn(
                    f"Montgomerie fitting: {name}={f:.4f} outside (0, 1). "
                    f"This typically occurs on symmetric airfoils (NACA 00xx) "
                    f"where CL at PM_cutoff exceeds the linear theory at the "
                    f"anchor point. Clamping to 0.95 as fallback — results "
                    f"may be approximate.",
                    UserWarning,
                )
        f1 = max(0.01, min(0.99, f1))
        f2 = max(0.01, min(0.99, f2))

        G1 = ((1.0 / f1 - 1.0)) ** 0.25
        G2 = ((1.0 / f2 - 1.0)) ** 0.25
        if abs(G2 - G1) < 1e-10:
            raise ValueError("G1 ≈ G2 — fitting points too similar")
        G = G1 / G2
        alphaM = (alpha1 - alpha2 * G) / (1.0 - G)
        denom  = (alpha1 - alphaM) ** 4
        if abs(denom) < 1e-10:
            raise ValueError("alpha1 ≈ alphaM — degenerate fit")
        k = (1.0 / f1 - 1.0) / denom
        return alphaM, k

    @classmethod
    def _select_fitting_points(cls, CL0: float, CLalpha: float,
                                CLmax: float, CLmin: float,
                                A0: float, CL90: float,
                                CD90_pos: float, CD90_neg: float,
                                DALPHA_MIN: float,
                                symmetric: bool = False):
        """Select two anchor points per branch for k/alphaM fitting.

        For symmetric airfoils the positive branch anchor uses the same
        5% offset strategy as the negative branch. Using CL_max as the
        positive anchor (the standard approach) causes f₁ > 1 on symmetric
        profiles because CL_max at PM_cutoff > t(α₁) — a structural issue
        the FOI report never addresses since the method was calibrated for
        cambered profiles. Per FOI §2.2.2: 'the positive side analysis is
        usually not necessary to carry out. It is already in basic data.'
        """
        # ── Positive branch ───────────────────────────────────────────────
        a1p = (CLmax - CL0) / CLalpha + DALPHA_MIN
        a2p = a1p + 15.0
        CL2p = cls._s_func(a2p, A0, CL0, CL90, CD90_pos) + 0.03

        if symmetric:
            # Use 5% offset (same as negative branch) to guarantee f1p ∈ (0,1)
            t_a1p = cls._t_func(a1p, A0, CL0, CLalpha)
            s_a1p = cls._s_func(a1p, A0, CL0, CL90, CD90_pos)
            CL1p  = t_a1p + 0.05 * (s_a1p - t_a1p)
        else:
            CL1p = CLmax

        f1p = cls._compute_f_from_curve(a1p, CL1p, A0, CL0, CLalpha, CL90, CD90_pos)
        f2p = cls._compute_f_from_curve(a2p, CL2p, A0, CL0, CLalpha, CL90, CD90_pos)

        # ── Negative branch ───────────────────────────────────────────────
        a1n  = (CLmin - CL0) / CLalpha - DALPHA_MIN
        a2n  = a1n - 15.0
        CL2n = cls._s_func(a2n, A0, CL0, CL90, CD90_neg) - 0.03
        t_a1 = cls._t_func(a1n, A0, CL0, CLalpha)
        s_a1 = cls._s_func(a1n, A0, CL0, CL90, CD90_neg)
        CL1n = t_a1 + 0.05 * (s_a1 - t_a1)
        f1n  = cls._compute_f_from_curve(a1n, CL1n, A0, CL0, CLalpha, CL90, CD90_neg)
        f2n  = cls._compute_f_from_curve(a2n, CL2n, A0, CL0, CLalpha, CL90, CD90_neg)

        return (a1p, f1p), (a2p, f2p), (a1n, f1n), (a2n, f2n)

    @classmethod
    def _montgomerie_cl(cls, alpha_deg: float, A0: float, CL0: float,
                         CLalpha: float, CL90: float, CD90: float,
                         alphaM: float, k: float) -> float:
        """CL = f·t + (1-f)·s."""
        f = cls._f_func(alpha_deg, alphaM, k)
        t = cls._t_func(alpha_deg, A0, CL0, CLalpha)
        s = cls._s_func(alpha_deg, A0, CL0, CL90, CD90)
        return f * t + (1.0 - f) * s

    @classmethod
    def _montgomerie_cd(cls, alpha_deg: float, A0: float, CL0: float,
                         CLalpha: float, CL90: float, CD90: float,
                         alphaM: float, k: float,
                         f_cd_lf) -> float:
        """CD = f·CD_e + (1-f)·CD_thinPlate — FOI §2.3.2."""
        f        = cls._f_func(alpha_deg, alphaM, k)
        CD_e     = float(f_cd_lf(alpha_deg)) if f_cd_lf is not None else 0.0
        CD_thin  = CD90 * np.sin(np.radians(alpha_deg))**2 + cls._CD_FRICTION
        return f * CD_e + (1.0 - f) * CD_thin


# ── Lindenburg (2003) ─────────────────────────────────────────────────────────

class LindenburgExtrapolator(ExtrapolatorBase):
    """Build the full-range CL and CD composite dataset using Lindenburg (2003) StC model.

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Notes
    -----
    Requires ``airfoil.delta_nose_deg`` and ``airfoil.delta_tail_deg`` to be
    set in the config. These are the nose and tail wedge angles derived from
    the airfoil coordinate geometry.

    No 3D AR correction is applied — the CFD data is already from a 3D
    finite-AR simulation.

    References
    ----------
    Lindenburg, C. (2003). ECN-C--03-025. §2.2–2.4.
    """

    def __init__(self, config: CaseConfig) -> None:
        super().__init__(config)
        self._params: dict = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def derive_parameters(self, df: pd.DataFrame) -> None:
        """Compute Lindenburg geometry and flow parameters."""
        g  = self.config.airfoil
        fl = self.config.flow
        d  = self.config.data

        if g.delta_nose_deg is None or g.delta_tail_deg is None:
            raise ValueError(
                "LindenburgExtrapolator requires airfoil.delta_nose_deg "
                "and airfoil.delta_tail_deg to be set in the config."
            )

        self._alpha0 = _derive_alpha0(df, d.LF_CL_column, g.symmetry)

        delta_nose = np.radians(g.delta_nose_deg)
        delta_tail = np.radians(g.delta_tail_deg)

        # ── CD_LAM — Prandtl-Schlichting (§2.3.2) ────────────────────────────
        CD_LAM = 0.455 / (np.log10(fl.Re) ** 2.58) - 1700.0 / fl.Re

        # ── CD90 — Lindenburg §2.2 ────────────────────────────────────────────
        CD90 = (1.7 + (0.3 - delta_nose) * (0.2 + 0.08 * delta_nose)
                * (1.0 - 1.8 * np.sqrt(g.RLE_C))
                - delta_tail * (0.2 + 0.08 * delta_tail))

        self._params = dict(
            CD90=CD90, CD_LAM=CD_LAM,
            AR=g.AR if g.AR is not None else 1e6,
            RLE_C=g.RLE_C,
        )
        self._f_cl_lf = _build_lf_cl_interp(df, d.LF_CL_column)
        self._f_cd_lf = _build_lf_cd_interp(df, d.LF_CD_column)
        self._derived = True

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full ±180° Lindenburg composite dataset."""
        if not self._derived:
            self.derive_parameters(df)

        p    = self.config.pipeline
        pm   = p.PM_cutoff
        step = p.fine_step
        pr   = self._params

        aoa_fine = np.arange(-180.0, 180.0 + step, step)
        cl_out, cd_out, source = [], [], []

        for a in aoa_fine:
            a_abs = abs(a)

            if a_abs <= pm:
                cl_out.append(float(self._f_cl_lf(a)))
                cd_out.append(float(self._f_cd_lf(a))
                               if self._f_cd_lf is not None else 0.0)
                source.append("PM")

            elif a_abs < 170.0:
                # CN/CT model covers attached, stall, and deep stall up to 170°
                cl, cd = self._lindenburg_CL_CD(
                    a, pr["CD90"], pr["AR"], pr["RLE_C"], pr["CD_LAM"])
                cl_out.append(cl); cd_out.append(abs(cd))
                source.append("lindenburg")

            else:
                # Reversed-flow formula — valid only for |α| ∈ [170°, 190°]
                cl, cd = self._lindenburg_reversed(a, pr["RLE_C"])
                cl_out.append(cl); cd_out.append(cd)
                source.append("reversed")

        return pd.DataFrame({"AoA": aoa_fine, "CL_full": cl_out,
                              "CD_full": cd_out, "source": source})

    @property
    def parameters(self) -> dict:
        if not self._derived:
            raise RuntimeError(
                "Parameters not yet derived. Call build() first."
            )
        return dict(self._params)

    # ── Private: Lindenburg core equations ───────────────────────────────────

    @staticmethod
    def _CN(alpha_deg: float, CD90: float,
            AR: float, RLE_C: float) -> float:
        """Normal force CN — Lindenburg §2.3.1.

        Flat-plate term: 1 / (0.56 + 0.44·|sinα|)
        """
        a_rad  = np.radians(alpha_deg)
        sin_a  = np.sin(a_rad)
        sin_2a = np.sin(2.0 * a_rad)
        AR_eff = AR / max(abs(sin_a), 1e-6)

        flat_plate = 1.0 / (0.56 + 0.44 * abs(sin_a))
        AR_term    = 0.41 * (1.0 - np.exp(-17.0 / AR_eff))
        skewness   = sin_a + 0.1 * np.sqrt(RLE_C) * sin_2a

        return CD90 * (flat_plate - AR_term) * skewness

    @staticmethod
    def _CT(alpha_deg: float, CN: float,
            RLE_C: float, CD_LAM: float) -> float:
        """Tangential force CT — Lindenburg §2.3.2."""
        a_rad   = np.radians(alpha_deg)
        cos_a   = np.cos(a_rad)
        sign_ct = 1.0 if alpha_deg >= 0 else -1.0

        viscous = -0.5 * CD_LAM * cos_a
        suction = abs(CN) * np.sqrt(RLE_C) * (sign_ct * 0.3 - 0.55 * cos_a)
        return viscous + suction

    @classmethod
    def _lindenburg_CL_CD(cls, alpha_deg: float, CD90: float,
                           AR: float, RLE_C: float,
                           CD_LAM: float) -> tuple[float, float]:
        """CL and CD from CN/CT via standard rotation."""
        a_rad = np.radians(alpha_deg)
        CN    = cls._CN(alpha_deg, CD90, AR, RLE_C)
        CT    = cls._CT(alpha_deg, CN, RLE_C, CD_LAM)
        CL    = CN * np.cos(a_rad) + CT * np.sin(a_rad)
        CD    = CN * np.sin(a_rad) - CT * np.cos(a_rad)
        return CL, CD

    @staticmethod
    def _lindenburg_reversed(alpha_deg: float,
                              RLE_C: float) -> tuple[float, float]:
        """Reversed-flow model — Lindenburg §2.4. Valid for |α| ∈ [170°, 190°].

        CL
        --
        Slope from Hoerner ellipsis characteristics (α in degrees):

            C_Lα = 0.108 − 1.5 · (r_nose/c)   [per degree]

        If C_Lα × 10° < 0.8 (slope too small to reach CL = 0.8 at 190°),
        enforce CL_max = 0.8 → C_Lα = 0.08 per degree.

        CL = sign · C_Lα · (|α| − 180°)   [α in degrees]

        CD
        --
        At 180° (Hoerner ellipsis in turbulent reversed flow):

            CD(180°) = 0.005 · (2 + √(2·r_nose/c) · (4 + 240·r_nose/c))

        Parabolic growth away from 180°:

            CD(α) = CD(180°) + 0.0003 · (α[deg] − 180°)²

        Notes
        -----
        - Valid range is strictly [170°, 190°] per §2.4. The paper states
          this is "a very rough attempt" with no continuity guarantee at 170°.
        - The previous implementation applied this formula for ALL |α| > 90°,
          which caused unphysical CL ~ ±7. The structural fix (range → 170°)
          is the primary correction. Formulas are as stated in ECN-C--03-025.
        """
        sign  = 1.0 if alpha_deg >= 0 else -1.0
        a_abs = abs(alpha_deg)

        # ── CL ────────────────────────────────────────────────────────────────
        cl_alpha = 0.108 - 1.5 * RLE_C                    # per degree, §2.4
        if cl_alpha * 10.0 < 0.8:                          # cap: CL_max ≥ 0.8
            cl_alpha = 0.08
        dev_deg = a_abs - 180.0                            # ∈ [−10, 0]
        CL = sign * cl_alpha * dev_deg

        # ── CD ────────────────────────────────────────────────────────────────
        CD_180 = 0.005 * (2.0 + np.sqrt(2.0 * RLE_C)
                          * (4.0 + 240.0 * RLE_C))         # §2.4 Hoerner
        CD = CD_180 + 0.0003 * dev_deg ** 2

        return CL, abs(CD)


# ── Montgomerie CM extrapolator ───────────────────────────────────────────────

class MontgomerieCMExtrapolator:
    """Build the full-range CM composite dataset using Montgomerie (2004).

    Parameters
    ----------
    config : CaseConfig
        Full pipeline configuration.

    Notes
    -----
    Implements Montgomerie (2004) Eqs. 37-57 with two modifications:

    1. Empirical arm flip for cambered airfoils (Eq. 45/57):
       ``arm_deep = CM_arm_flip_offset - arm_deep``
       Montgomerie's flat-plate arm assumption gives the wrong CM sign
       for cambered airfoils. Mirroring arm_deep around 0.25 corrects
       this. Offset calibrated at 0.1 for Clark Y.

    2. LF arm anchor at ±PM_CUTOFF replaces Eq. 48 armCalc to guarantee
       continuity at the LF/blend boundary.

    3D data note: Montgomerie is strictly 2D. For a straight unswept
    rectangular wing, CMy ≈ 2D CM. A warning is raised if AR is not None.

    References
    ----------
    Montgomerie, B. (2004). ECN-C-04-054. Eqs. 37-57.
    """

    _CL_SINGULARITY_GUARD: float = 1e-3

    def __init__(self, config: CaseConfig) -> None:
        self.config  = config
        self._params = None

    # ── Public ────────────────────────────────────────────────────────────────

    def derive_parameters(self, df: pd.DataFrame,
                           ALPHA0: float) -> None:
        """Derive Montgomerie arm parameters from LF data."""
        d      = self.config.data
        p      = self.config.pipeline
        cm_col = d.LF_CM_column
        cl_col = d.LF_CL_column

        lf_cl = df.dropna(subset=[cl_col])
        lf_cm = df.dropna(subset=[cm_col])

        f_cl_lf = interp1d(lf_cl["AoA"], lf_cl[cl_col],
                            kind="linear", fill_value="extrapolate",
                            bounds_error=False)
        f_cm_lf = interp1d(lf_cm["AoA"], lf_cm[cm_col],
                            kind="linear", fill_value="extrapolate",
                            bounds_error=False)

        CL0 = float(f_cl_lf(0.0))
        CM0 = float(f_cm_lf(0.0))

        arm0 = (0.25 - CM0 / CL0) if abs(CL0) > self._CL_SINGULARITY_GUARD else 0.25

        eps      = 1.0
        CL_slope = (float(f_cl_lf(eps)) - float(f_cl_lf(-eps))) / (2.0 * eps)

        offset = 0.5111 - 1.337e-3 * ALPHA0
        slope  = 1.653e-3 + 1.6e-4  * ALPHA0

        def _arm_at(alpha_b: float) -> float:
            cl  = float(f_cl_lf(alpha_b))
            cm  = float(f_cm_lf(alpha_b))
            CN  = cl * np.cos(np.radians(alpha_b))
            return (0.25 - cm / CN) if abs(CN) > self._CL_SINGULARITY_GUARD else arm0

        arm_anchor_pos = _arm_at( p.PM_cutoff)
        arm_anchor_neg = _arm_at(-p.PM_cutoff)

        alpha0_abs = abs(ALPHA0)
        x_A = alpha0_abs
        y_A = offset + slope * (x_A - 90.0)
        x_B = -180.0 - alpha0_abs
        y_B = offset + slope * 90.0
        denom = x_B - x_A
        k_neg = (y_B - y_A) / denom if abs(denom) > 1e-8 else 0.0

        self._params = {
            "arm0"          : arm0,
            "CL0"           : CL0,
            "CM0"           : CM0,
            "CL_slope"      : CL_slope,
            "offset"        : offset,
            "slope"         : slope,
            "arm_anchor_pos": arm_anchor_pos,
            "arm_anchor_neg": arm_anchor_neg,
            "x_A"           : x_A,
            "y_A"           : y_A,
            "k_neg"         : k_neg,
        }
        self._f_cm_lf = f_cm_lf
        self._ALPHA0  = ALPHA0

    def build(self, df: pd.DataFrame,
              CL_full: np.ndarray,
              CD_full: np.ndarray,
              aoa_fine: np.ndarray,
              ALPHA0: float) -> np.ndarray:
        """Build the full ±180° composite CM curve."""
        if self._params is None:
            raise RuntimeError(
                "MontgomerieCMExtrapolator.build() called before "
                "derive_parameters(). Call derive_parameters() first."
            )

        p           = self.config.pipeline
        pm_cutoff   = p.PM_cutoff
        flip_offset = p.CM_arm_flip_offset

        arm0           = self._params["arm0"]
        CL0            = self._params["CL0"]
        CL_slope       = self._params["CL_slope"]
        offset         = self._params["offset"]
        slope_arm      = self._params["slope"]
        arm_anchor_pos = self._params["arm_anchor_pos"]
        arm_anchor_neg = self._params["arm_anchor_neg"]
        x_A            = self._params["x_A"]
        y_A            = self._params["y_A"]
        k_neg          = self._params["k_neg"]
        f_cm_lf        = self._f_cm_lf

        cm_out = []

        for i, a in enumerate(aoa_fine):
            a_rad  = np.radians(a)
            CL_val = float(CL_full[i])
            CD_val = float(CD_full[i])
            CN_val = CL_val * np.cos(a_rad) + CD_val * np.sin(a_rad)

            if abs(a) <= pm_cutoff:
                cm_out.append(float(f_cm_lf(a)))

            else:
                arm_anchor = arm_anchor_pos if a >= 0 else arm_anchor_neg
                armCalc    = float(np.clip(arm_anchor, 0.0, 1.0))

                if a >= 0:
                    arm_deep = offset + slope_arm * (a - 90.0)
                else:
                    arm_deep = y_A + k_neg * (a - x_A)
                arm_deep = float(np.clip(arm_deep, 0.0, 1.0))
                arm_deep = flip_offset - arm_deep

                CL_pot  = CL_slope * (a - ALPHA0)
                CL_thin = np.sin(2.0 * a_rad)
                a_f     = CL_pot  - CL_val
                b_f     = CL_val  - CL_thin
                denom_f = a_f + b_f

                f = float(np.clip((a_f / denom_f) ** 2, 0.0, 1.0)) \
                    if abs(denom_f) > 1e-8 else 1.0

                arm_final = (1.0 - f) * armCalc + f * arm_deep
                cm_out.append(-(arm_final - 0.25) * CN_val)

        return np.array(cm_out)
