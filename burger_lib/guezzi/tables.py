# --- START OF FILE tables.py ---
"""
LaTeX Table Generation Utilities

Provides functions to create formatted LaTeX tables from Measurement objects
and FitResult objects, suitable for inclusion in reports.
"""

import numpy as np
from typing import List, Optional, Dict, Union, Sequence, Any # Added Sequence
import warnings
try:
    # pyperclip allows copying the generated LaTeX code to the clipboard
    import pyperclip
    _HAS_PYPERCLIP = True
except ImportError:
    _HAS_PYPERCLIP = False

from .measurement import Measurement
from .fitting import FitResult # Import FitResult
from .utils import _format_value_error_eng # Import the corrected formatter

# Helper function for formatting non-Measurement stats
def _format_stat(value: Any) -> str:
    """Formats statistics (like chi^2, DoF) for the table."""
    if value is None:
        return "---"
    if isinstance(value, int):
        return f"{value}"
    if isinstance(value, float):
        # Format floats reasonably, e.g., 3-4 significant figures or scientific notation
        if np.isclose(value, 0.0): return "0"
        if 0.001 <= abs(value) < 10000:
            # Use general format, trying to get ~3-4 digits
            return f"{value:.3g}"
        else:
            # Use scientific notation for very large/small numbers
            return f"{value:.3e}"
    return str(value) # Fallback for other types

def latex_table_data(*measurements: Measurement,
                     labels: Optional[List[str]] = None,
                     orientation: str = 'h',
                     sig_figs_error: int = 1,
                     caption: Optional[str] = None,
                     table_spec: Optional[str] = None,
                     copy_to_clipboard: bool = True) -> str:
    """
    Generate a LaTeX table string from Measurement objects using engineering notation.

    This function takes one or more Measurement objects and arranges them into
    a LaTeX `tabular` environment, enclosed within a `table` float environment.
    Values and errors are formatted using engineering notation (SI prefixes)
    via `Measurement.to_eng_string`, ensuring appropriate significant figures.

    Args:
        *measurements: One or more Measurement objects to include. If measurements
                       are arrays, they must have broadcast-compatible shapes.
        labels: List of strings for row/column headers corresponding to measurements.
                If None, uses `Measurement.name` or generates default names like "Data 1".
                Must match the number of Measurement objects provided.
        orientation: 'h' for horizontal layout (labels as row headers, multiple
                     entries per measurement go across columns).
                     'v' for vertical layout (labels as column headers, multiple
                     entries per measurement go down rows). Default 'h'.
        sig_figs_error: Number of significant figures to use for displaying the
                        error part of each measurement (passed to formatting function).
                        Typically 1 or 2. (Default 1).
        caption: Optional LaTeX table caption text. If provided, a `\\caption{}`
                 command is added within the table environment.
        table_spec: Optional custom LaTeX table column specification string
                    (e.g., '|c|c|r|'). If None, a default specification is
                    generated based on orientation and number of entries.
        copy_to_clipboard: If True and `pyperclip` is installed, attempts to copy
                           the generated LaTeX code to the system clipboard.
                           (Default True).

    Returns:
        str: A string containing the full LaTeX code for the table.

    Raises:
        ValueError: If the number of labels does not match the number of measurements,
                    if measurement shapes are incompatible, or if orientation is invalid.
    """
    if not measurements:
        return "" # Return empty string if no measurements provided

    # Validate inputs
    n_meas = len(measurements)
    if not all(isinstance(m, Measurement) for m in measurements):
        raise TypeError("All arguments must be Measurement objects.")

    if labels is None:
        # Generate default labels using Measurement names or placeholders
        labels = [m.name if m.name else f"Data {i+1}" for i, m in enumerate(measurements)]
    elif len(labels) != n_meas:
        raise ValueError(f"Number of labels provided ({len(labels)}) must match the "
                         f"number of measurements ({n_meas}).")

    if orientation not in ['h', 'v']:
        raise ValueError("Orientation must be 'h' (horizontal) or 'v' (vertical).")

    # Determine common shape and number of entries per measurement
    shapes = [m.shape for m in measurements]
    try:
        # Use broadcasting to find common shape and check compatibility
        common_shape = np.broadcast(*[np.empty(s) for s in shapes]).shape
        n_entries = int(np.prod(common_shape)) if common_shape else 1 # np.prod(()) is 1.0 -> int
        is_scalar_like = (common_shape == ())
    except ValueError:
        raise ValueError(f"Measurement shapes {shapes} are not compatible for table generation. "
                         "They must be broadcastable to a common shape.")

    # Format all measurement data points using the engineering string method
    formatted_data = []
    for m in measurements:
        # Use the Measurement's built-in formatter which calls the util function
        formatter = np.vectorize(lambda v, e, u: m.to_eng_string(sig_figs_error=sig_figs_error, value=v, error=e, unit=u),
                                 excluded=['sig_figs_error', 'unit'], # vectorize over value and error
                                 otypes=[str])
        # Ensure value/error are broadcastable to common shape before formatting
        val_bcast = np.broadcast_to(m.value, common_shape)
        err_bcast = np.broadcast_to(m.error, common_shape)
        formatted_array = formatter(v=val_bcast, e=err_bcast, u=m.unit)
        formatted_data.append(formatted_array.flatten()) # Flatten for simple row/column access

    # --- Generate LaTeX Code ---
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]") # Standard float placement options
    latex_lines.append("\\centering")

    # Build the tabular environment
    if orientation == 'h':
        # Horizontal: Labels are row headers. Data entries form columns.
        n_cols = n_entries # Number of columns is the number of data points per measurement
        # Default table spec: left-aligned label column, centered data columns
        if table_spec is None:
            col_spec = '|l|' + ('c' * n_cols) + '|' if n_cols > 0 else '|l|'
        else:
            col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Optional header row for multi-entry data
        if n_entries > 1 and not is_scalar_like:
             header_row = ["Index"] + [str(i+1) for i in range(n_entries)]
             latex_lines.append(" & ".join(header_row) + " \\\\\\hline") # Double hline after header

        # Data rows: One row per measurement
        for i, label in enumerate(labels):
            # Escape LaTeX special characters in labels (simple version)
            safe_label = label.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
            row_content = [safe_label] + list(formatted_data[i])
            latex_lines.append(" & ".join(row_content) + " \\\\\\hline")

        latex_lines.append("\\end{tabular}")

    elif orientation == 'v':
        # Vertical: Labels are column headers. Data entries form rows.
        n_cols = n_meas # Number of columns is the number of measurements
        # Default table spec: centered columns for each measurement
        if table_spec is None:
             col_spec = '|' + ('c|' * n_cols) if n_cols > 0 else '|'
        else:
             col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Header row: Measurement labels
        safe_labels = [lbl.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&') for lbl in labels]
        latex_lines.append(" & ".join(safe_labels) + " \\\\\\hline\\hline") # Double hline after header

        # Data rows: One row per data point index
        for i in range(n_entries):
            row_content = [formatted_data[j][i] for j in range(n_meas)]
            latex_lines.append(" & ".join(row_content) + " \\\\") # No hline between data rows typical here
        latex_lines.append("\\hline") # Add hline at the end of data
        latex_lines.append("\\end{tabular}")

    # Add caption if provided
    if caption:
        # Basic escaping for caption text
        safe_caption = caption.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        latex_lines.append(f"\\caption{{{safe_caption}}}")
        # Consider adding automatic \label{tab:...} here? Maybe too complex.
        # Example: label_base = "".join(filter(str.isalnum, caption)).lower()[:10]
        # latex_lines.append(f"\\label{{tab:{label_base}}}")

    latex_lines.append("\\end{table}")

    # --- Final Output and Clipboard ---
    latex_string = "\n".join(latex_lines)

    if copy_to_clipboard and _HAS_PYPERCLIP:
        try:
            pyperclip.copy(latex_string)
        except Exception as e:
            # Provide a more specific warning if copy fails
            warnings.warn(f"Failed to copy table to clipboard. pyperclip might not be configured correctly "
                          f"or lack permissions. Error: {e}", RuntimeWarning)
    elif copy_to_clipboard and not _HAS_PYPERCLIP:
        warnings.warn("pyperclip library not found. Cannot copy table to clipboard. "
                      "Install it (`pip install pyperclip`) to enable this feature.", RuntimeWarning)

    return latex_string


def latex_table_fit(*fit_results: FitResult,
                    orientation: str = 'h',
                    sig_figs_error: int = 2,
                    include_stats: List[str] = ['chi_square', 'dof', 'reduced_chi_square'],
                    fit_labels: Optional[List[str]] = None,
                    param_labels: Optional[Dict[str, str]] = None,
                    stat_labels: Optional[Dict[str, str]] = None,
                    caption: Optional[str] = None,
                    table_spec: Optional[str] = None,
                    copy_to_clipboard: bool = True) -> str:
    """
    Generate a LaTeX table summarizing one or more FitResult objects.

    Displays fitted parameters (with uncertainties, formatted using engineering notation)
    and selected fit statistics (like Chi², DoF, Reduced Chi²).

    Args:
        *fit_results: One or more FitResult objects to include in the table.
        orientation: 'h' for horizontal layout (parameters/stats as rows, fits as columns).
                     'v' for vertical layout (fits as rows, parameters/stats as columns).
                     (Default 'h').
        sig_figs_error: Number of significant figures for displaying the uncertainty
                        of the fitted parameters. (Default 2).
        include_stats: List of statistic keys (attribute names from FitResult) to
                       include in the table. Common choices: 'chi_square', 'dof',
                       'reduced_chi_square'. The function will try to retrieve these
                       attributes from each FitResult. (Default shown above).
        fit_labels: Optional list of labels for each FitResult (used as column/row headers).
                    If None, generated automatically like "Fit 1", "Fit 2". Must match
                    the number of FitResult objects.
        param_labels: Optional dictionary mapping internal parameter names (from
                      `FitResult.parameter_names`) to desired display names in the
                      table (e.g., {'a': '$A_0$', 'b': '$\\tau$'}). Allows LaTeX math mode.
        stat_labels: Optional dictionary mapping internal statistic keys (from
                     `include_stats`) to desired display names (e.g.,
                     {'reduced_chi_square': '$\\chi^2/\\nu$'}). Allows LaTeX math mode.
        caption: Optional LaTeX table caption text.
        table_spec: Optional custom LaTeX table column specification string. If None,
                    a default specification is generated.
        copy_to_clipboard: If True and `pyperclip` is installed, attempts to copy
                           the generated LaTeX code to the system clipboard.
                           (Default True).

    Returns:
        str: A string containing the full LaTeX code for the summary table.

    Raises:
        ValueError: If the number of fit_labels does not match the number of fits,
                    or if orientation is invalid.
    """
    if not fit_results:
        return ""

    # Validate inputs
    n_fits = len(fit_results)
    if not all(isinstance(res, FitResult) for res in fit_results):
        raise TypeError("All arguments must be FitResult objects.")

    if fit_labels is None:
        fit_labels = [f"Fit {i+1}" for i in range(n_fits)]
    elif len(fit_labels) != n_fits:
        raise ValueError(f"Number of fit_labels provided ({len(fit_labels)}) must match the "
                         f"number of FitResult objects ({n_fits}).")

    if orientation not in ['h', 'v']:
        raise ValueError("Orientation must be 'h' (horizontal) or 'v' (vertical).")

    # --- Data Collection and Formatting ---

    # Gather all unique parameter names and requested stat keys across all fits
    all_param_names = []
    all_stat_keys = []
    for res in fit_results:
        for name in res.parameter_names:
            if name not in all_param_names:
                all_param_names.append(name)
        if include_stats:
            for key in include_stats:
                 # Check if stat exists, is requested, and is not None in this *specific* result
                 stat_val = getattr(res, key, None)
                 if key not in all_stat_keys and stat_val is not None:
                     all_stat_keys.append(key) # Add if present in at least one fit

    # Create display labels, using provided dictionaries or defaults
    # Use internal names as fallback if no label provided
    param_display_map = {name: param_labels.get(name, name) for name in all_param_names} if param_labels else {name: name for name in all_param_names}

    # Define default pretty labels for common stats
    default_stat_display = {'chi_square': '$\\chi^2$',
                            'dof': 'DoF', # Degrees of Freedom
                            'reduced_chi_square': '$\\chi^2/\\nu$'} # Reduced Chi-squared (nu = DoF)
    stat_display_map = {key: stat_labels.get(key, default_stat_display.get(key, key)) for key in all_stat_keys} if stat_labels else {key: default_stat_display.get(key, key) for key in all_stat_keys}


    # Prepare the data for the table: Dict[row_label, List[formatted_entry_for_each_fit]]
    table_data: Dict[str, List[str]] = {}

    # Format parameters
    for param_name in all_param_names:
        row_label = param_display_map[param_name]
        row_entries = []
        for res in fit_results:
            param_meas = res.parameters.get(param_name) # Get Measurement object
            if param_meas is not None:
                # Use Measurement's formatting method with specified sig figs
                row_entries.append(param_meas.to_eng_string(sig_figs_error=sig_figs_error))
            else:
                row_entries.append("---") # Placeholder if parameter not in this fit
        table_data[row_label] = row_entries

    # Format statistics
    for stat_key in all_stat_keys:
        row_label = stat_display_map[stat_key]
        row_entries = []
        for res in fit_results:
            stat_val = getattr(res, stat_key, None)
            # Use the helper function for consistent formatting of stats
            row_entries.append(_format_stat(stat_val))
        table_data[row_label] = row_entries

    # --- Generate LaTeX Code ---
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")

    # Define the order of rows (parameters first, then statistics)
    # Use the display names (values of the maps) as the keys for ordering
    ordered_row_labels = list(param_display_map.values()) + list(stat_display_map.values())
    # Filter out any row labels that ended up having no data (e.g., a stat requested but None in all fits)
    ordered_row_labels = [label for label in ordered_row_labels if label in table_data]


    if orientation == 'h': # Parameters/Stats as rows, Fits as columns
        n_cols = n_fits # Number of fits determines data columns
        # Default spec: Left-aligned label column, centered fit columns
        if table_spec is None:
            col_spec = '|l|' + ('c' * n_cols) + '|' if n_cols > 0 else '|l|'
        else:
            col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Header row: Fit labels (plus a header for the first column)
        # Escape special characters in fit labels
        safe_fit_labels = [lbl.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&') for lbl in fit_labels]
        header = ["Parameter/Stat"] + safe_fit_labels
        latex_lines.append(" & ".join(header) + " \\\\\\hline\\hline") # Double hline after header

        # Data rows
        for row_label in ordered_row_labels:
             row_content = [row_label] + table_data[row_label]
             latex_lines.append(" & ".join(row_content) + " \\\\\\hline") # Hline after each row

        latex_lines.append("\\end{tabular}")

    elif orientation == 'v': # Fits as rows, Parameters/Stats as columns
        n_cols = len(ordered_row_labels) # Number of params/stats determines data columns
        # Default spec: Left-aligned fit label column, centered data columns
        if table_spec is None:
             col_spec = '|l|' + ('c' * n_cols) + '|' if n_cols > 0 else '|l|'
        else:
             col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Header row: Parameter/Stat labels (already display-ready)
        header = ["Fit Label"] + ordered_row_labels
        latex_lines.append(" & ".join(header) + " \\\\\\hline\\hline") # Double hline after header

        # Data rows: One row per fit
        for i, fit_label in enumerate(fit_labels):
             safe_fit_label = fit_label.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
             # Get the i-th entry from each column's data list
             row_entries = [table_data[row_label][i] for row_label in ordered_row_labels]
             row_content = [safe_fit_label] + row_entries
             latex_lines.append(" & ".join(row_content) + " \\\\\\hline") # Hline after each fit row

        latex_lines.append("\\end{tabular}")

    # Add caption if provided
    if caption:
        safe_caption = caption.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        latex_lines.append(f"\\caption{{{safe_caption}}}")
        # Consider adding \label here as well

    latex_lines.append("\\end{table}")

    # --- Final Output and Clipboard ---
    latex_string = "\n".join(latex_lines)

    if copy_to_clipboard and _HAS_PYPERCLIP:
        try:
            pyperclip.copy(latex_string)
        except Exception as e:
            warnings.warn(f"Failed to copy table to clipboard. Error: {e}", RuntimeWarning)
    elif copy_to_clipboard and not _HAS_PYPERCLIP:
        warnings.warn("pyperclip not found. Cannot copy table to clipboard.", RuntimeWarning)

    return latex_string

# --- END OF FILE tables.py ---