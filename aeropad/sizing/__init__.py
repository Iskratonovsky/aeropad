"""aeropad.sizing — aircraft sizing by statistics.

Regression-family workflows for statistical sizing relations over
historical aircraft databases, plus dataset-exploration diagnostics
(correlation heatmaps) for identifying exploitable relationships.
"""

from .regression import SizingModel, compare_families
from .models import FAMILIES, PowerLawRegressor, build_family
from .dataio import (
    standardize_dataset, detect_unit_contamination,
    fix_unit_contamination, collinearity_screen, missingness_report,
    usability_screen, audit_dataset, export_results, AuditReport,
)
from .plots import (
    correlation_heatmap, tuning_curve, prediction_surface,
    actual_vs_predicted, learning_curve_plot,
)

__all__ = [
    "SizingModel", "compare_families",
    "FAMILIES", "PowerLawRegressor", "build_family",
    "standardize_dataset", "detect_unit_contamination",
    "fix_unit_contamination", "collinearity_screen",
    "missingness_report", "usability_screen", "audit_dataset",
    "export_results", "AuditReport",
    "correlation_heatmap", "tuning_curve", "prediction_surface",
    "actual_vs_predicted", "learning_curve_plot",
]
