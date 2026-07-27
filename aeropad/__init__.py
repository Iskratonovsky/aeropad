"""
aeropad — AEROnautical Preliminary & conceptual Aircraft Design toolkit
=======================================================================

A modular toolkit for conceptual and preliminary aircraft design
assistance. Current modules:

- :mod:`aeropad.polar` — full-range (±180°) aerodynamic polar
  reconstruction from limited data, via semi-empirical post-stall
  extrapolation or sparse-sample Kriging surrogates.

- :mod:`aeropad.sizing` — aircraft sizing by statistics: regression
  families (polynomial, ridge, lasso, kernel ridge, power law, GPR)
  over historical aircraft databases, with correlation-heatmap dataset
  exploration and cross-family comparison.
"""

__version__ = "0.6.2"

from .config import (
    CaseConfig, AirfoilSpec, FlowSpec, DataSpec, PipelineSpec,
)
from .polar.reconstruct import reconstruct_polar, recommend, PolarResult
from .sizing import SizingModel, compare_families, correlation_heatmap

__all__ = [
    "CaseConfig", "AirfoilSpec", "FlowSpec", "DataSpec", "PipelineSpec",
    "reconstruct_polar", "recommend", "PolarResult",
    "SizingModel", "compare_families", "correlation_heatmap",
]
