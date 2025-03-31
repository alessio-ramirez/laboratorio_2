"""
Burger: A Python library for handling measurements with uncertainties,
error propagation, curve fitting, and reporting.
"""

__version__ = "0.1.0"  # Example version

# Import core components to make them available directly from 'burger'
from .measurement import Measurement#, create_measurement
#from .propagation import error_prop
#from .fitting import perform_fit, create_best_fit_line
#from .stats import test_comp
#from .tables import latex_table, fit_results_table

# Define what gets imported with 'from burger import *' (optional but good practice)
# Also defines the public API for documentation tools like Sphinx.
__all__ = [
    # From measurement.py
    'Measurement']
""",
    'create_measurement',

    # From propagation.py
    'error_prop',

    # From fitting.py
    'perform_fit',
    'create_best_fit_line',

    # From stats.py
    'test_comp',

    # From tables.py
    'latex_table',
    'fit_results_table',
]

# Optional: Define module-level constants if any are truly global
# DEFAULT_COLORS could potentially go here, but often better kept closer to usage (fitting.py)"""