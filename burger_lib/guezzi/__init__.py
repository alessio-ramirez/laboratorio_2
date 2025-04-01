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

Example Usage:
  import guezzi as gz
  import numpy as np

  # Create measurements
  voltage = gz.Measurement([1.1, 2.0, 3.2], [0.1, 0.1, 0.2], unit='V', name='Voltage')
  current = gz.Measurement([0.23, 0.41, 0.65], [0.02, 0.03, 0.04], unit='A', name='Current')

  # Define a fit function (Ohm's Law: V = R*I)
  def linear_func(i, r):
      return r * i

  # Perform fit (ODR is used automatically due to x-errors)
  fit_result = gz.perform_fit(current, voltage, linear_func, parameter_names=['Resistance'])
  print(fit_result)

  # Plot the fit
  gz.plot_fit(fit_result, show_params=True, show_stats=True, plot_residuals=True, show_plot=True)

  # Generate a LaTeX table of the fit results
  print(gz.latex_table_fit(fit_result, param_labels={'Resistance': '$R$'}))
"""

__version__ = "2.1.0-docs-fix" # Updated version

# Core components
from .measurement import Measurement
from .fitting import perform_fit, FitResult
from .stats import test_comp, weighted_mean

# Output generation
from .tables import latex_table_data, latex_table_fit
from .plotting import plot_fit, plot_measurements

# Utilities (can be accessed via gz.utils if needed, e.g., gz.utils.SI_PREFIXES)
from . import utils

# --- END OF FILE __init__.py ---