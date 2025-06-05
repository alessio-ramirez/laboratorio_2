# --- START OF FILE __init__.py ---
"""
Guezzi - A Scientific Utility Library for Physics Students

Guezzi aims to simplify common tasks encountered in experimental physics labs,
such as handling measurements with uncertainties, performing curve fitting,
basic statistical analysis, and generating publication-ready visualizations
and LaTeX tables.

Core Features:
  - Measurement Class: Represents values with errors, propagates uncertainties automatically.
  - Curve Fitting: Performs fits using standard methods (Least Squares, ODR)
    and provides detailed results.
  - Statistics: Includes compatibility testing and weighted averaging.
  - Output Generation: Creates LaTeX tables and plots suitable for reports.
"""

__version__ = "2.1.0-docs-fix" # Updated version

# Core components
from .measurement import Measurement
from .fitting import perform_fit, FitResult
from .stats import test_comp, weighted_mean

# Output generation
from .tables import latex_table_data, latex_table_fit
from .plot import plot_fit, plot_measurements, PlotStyle

# Utilities (can be accessed via gz.utils if needed, e.g., gz.utils.SI_PREFIXES)
from . import utils

# --- END OF FILE __init__.py ---