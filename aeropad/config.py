"""
aeropad.config
==============
Configuration dataclasses for the aeropad polar-reconstruction pipeline.

A :class:`CaseConfig` bundles four sub-specifications:

- :class:`AirfoilSpec`   — geometry and stall characteristics
- :class:`FlowSpec`      — freestream operating condition
- :class:`DataSpec`      — column mapping for the input DataFrame
- :class:`PipelineSpec`  — reconstruction pipeline parameters

Configs can be built directly in Python or loaded from a YAML file
with :meth:`CaseConfig.from_yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AirfoilSpec:
    """Airfoil geometry and 2D stall characteristics.

    Attributes
    ----------
    name : str
        Identifier used in outputs and plots.
    symmetry : str
        ``"symmetric"`` or ``"asymmetric"``. Controls zero-lift-angle
        derivation and symmetric-mirroring optimisations.
    TC : float
        Thickness-to-chord ratio.
    HC : float
        Camber-to-chord ratio.
    RLE_C : float
        Leading-edge radius to chord ratio.
    AR : float, optional
        Aspect ratio for finite wings. ``None`` for 2D sections.
    alpha_s_2D_pos, CL_s_2D_pos : float, optional
        Positive-branch 2D stall angle (deg) and lift coefficient.
    alpha_s_2D_neg, CL_s_2D_neg : float, optional
        Negative-branch 2D stall angle (deg) and lift coefficient.
    delta_nose_deg, delta_tail_deg : float, optional
        Nose and tail wedge angles (deg); required by the Lindenburg
        method for cambered profiles (defaults to 0 for symmetric).
    """
    name: str = "unnamed_airfoil"
    symmetry: str = "asymmetric"
    TC: float = 0.12
    HC: float = 0.0
    RLE_C: float = 0.0
    AR: Optional[float] = None
    alpha_s_2D_pos: Optional[float] = None
    CL_s_2D_pos: Optional[float] = None
    alpha_s_2D_neg: Optional[float] = None
    CL_s_2D_neg: Optional[float] = None
    delta_nose_deg: Optional[float] = None
    delta_tail_deg: Optional[float] = None

    def __post_init__(self) -> None:
        if self.symmetry not in ("symmetric", "asymmetric"):
            raise ValueError(
                f"symmetry must be 'symmetric' or 'asymmetric', "
                f"got {self.symmetry!r}"
            )
        # Lindenburg requires wedge angles; symmetric airfoils default to 0.
        if self.symmetry == "symmetric":
            if self.delta_nose_deg is None:
                self.delta_nose_deg = 0.0
            if self.delta_tail_deg is None:
                self.delta_tail_deg = 0.0


@dataclass
class FlowSpec:
    """Freestream operating condition."""
    Re: float = 1.0e6
    M: float = 0.1


@dataclass
class DataSpec:
    """Column mapping for the input polar DataFrame.

    The input DataFrame must contain an ``AoA`` column (degrees).
    Low-fidelity (LF) columns hold attached-flow data from a panel
    method, XFOIL, or similar; high-fidelity (HF) columns, when
    present, hold sparse CFD samples used by the Kriging route and
    for evaluation.
    """
    LF_CL_column: str = "CL_LF"
    LF_CD_column: str = "CD_LF"
    LF_CM_column: str = "CM_LF"
    HF_CL_column: Optional[str] = None
    HF_CD_column: Optional[str] = None
    flip_aoa_sign: bool = False


@dataclass
class PipelineSpec:
    """Reconstruction pipeline parameters.

    Attributes
    ----------
    extrapolator : str
        Semi-empirical method for the extrapolation route:
        ``"battisti"``, ``"aerodas"``, ``"montgomerie"``,
        ``"lindenburg"``, or ``"auto"`` (per-case recommendation).
    PM_cutoff : float
        Angle of attack (deg) beyond which LF attached-flow data is no
        longer trusted; extrapolation takes over past the blend region.
    blend_end : float
        End of the LF-to-extrapolation blend region (deg).
    fine_step : float
        Output grid resolution (deg) of the reconstructed polar.
    kriging_spacing : float
        Training-station spacing (deg) for the Kriging route.
        The uniform-20° rule is the validated default.
    run_CM : bool
        Whether to also reconstruct the pitching-moment coefficient
        (semi-empirical route only; requires LF CM data).
    CM_arm_flip_offset : float
        Moment-arm sign-flip offset used by the CM extrapolator.
    """
    extrapolator: str = "auto"
    PM_cutoff: float = 20.0
    blend_end: float = 45.0
    fine_step: float = 1.0
    kriging_spacing: float = 20.0
    apply_3D_correction: bool = False
    run_CM: bool = False
    CM_arm_flip_offset: float = 0.1


@dataclass
class CaseConfig:
    """Full configuration for one (airfoil, flow condition) case."""
    airfoil: AirfoilSpec = field(default_factory=AirfoilSpec)
    flow: FlowSpec = field(default_factory=FlowSpec)
    data: DataSpec = field(default_factory=DataSpec)
    pipeline: PipelineSpec = field(default_factory=PipelineSpec)
    # Legacy internal alias slot (set by some extrapolators at runtime).
    param: object = None

    @classmethod
    def from_yaml(cls, path: str) -> "CaseConfig":
        """Load a CaseConfig from a YAML file.

        The YAML structure mirrors the dataclass structure::

            airfoil:
              name: Clark_Y
              symmetry: asymmetric
              ...
            flow:
              Re: 2817138
              M: 0.16
            data:
              LF_CL_column: CL_PM
              ...
            pipeline:
              extrapolator: montgomerie
              ...
        """
        import yaml  # local import; PyYAML optional at runtime
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            airfoil=AirfoilSpec(**raw.get("airfoil", {})),
            flow=FlowSpec(**raw.get("flow", {})),
            data=DataSpec(**raw.get("data", {})),
            pipeline=PipelineSpec(**raw.get("pipeline", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
