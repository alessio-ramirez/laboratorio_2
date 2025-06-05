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
# Ensure this import path is correct based on your project structure
from .utils import _format_value_error_eng 

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
        # General format for typical range, trying for about 3-4 sig figs visible
        if 0.001 <= abs(value) < 10000: 
            # Using .3g might not always show trailing zeros if they are significant.
            # A more complex formatter might be needed for perfect sig fig control here.
            # For now, .3g is a reasonable default.
            return f"{value:.3g}" 
        else:
            # Use scientific notation for very large/small numbers
            return f"{value:.2e}" # Usually 2-3 sig figs in sci notation is good
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
    via the library's internal formatting utilities, ensuring appropriate significant figures.

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
    # This uses the .value attribute of each measurement for shape determination.
    shapes = [m.shape for m in measurements] # m.shape is from m.value.shape
    try:
        # Use broadcasting to find common shape and check compatibility
        # Create dummy arrays with these shapes for broadcasting check
        dummy_arrays = [np.empty(s) for s in shapes]
        common_shape = np.broadcast(*dummy_arrays).shape
        n_entries = int(np.prod(common_shape)) if common_shape else 1 
        is_scalar_like_table = (n_entries == 1) # If the whole table effectively has one entry per measurement
    except ValueError:
        raise ValueError(f"Measurement shapes {shapes} are not compatible for table generation. "
                         "They must be broadcastable to a common shape.")

    # Format all measurement data points
    formatted_data = [] # This will store 1D arrays of strings for each measurement
    
    # Define the vectorized formatter for _format_value_error_eng ONCE
    # It applies _format_value_error_eng to each element of its input arrays.
    vectorized_core_formatter = np.vectorize(
        _format_value_error_eng,
        excluded=['unit_symbol', 'sig_figs_error'], # These are fixed per call below
        otypes=[str] # Output type is string
    )
            
    for m in measurements:
        # `m` is one of the input Measurement objects.
        # Broadcast its value and error to the `common_shape` determined for the table.
        val_bcast = np.broadcast_to(m.value, common_shape)
        err_bcast = np.broadcast_to(m.error, common_shape)

        # Apply the pre-defined vectorized formatter
        formatted_array_for_m = vectorized_core_formatter(
            value=val_bcast,
            error=err_bcast,
            unit_symbol=m.unit, # Use unit from the original measurement m
            sig_figs_error=sig_figs_error # Use sig_figs_error from latex_table_data args
        )
        
        # formatted_array_for_m will have `common_shape`. Flatten it for table construction.
        formatted_data.append(formatted_array_for_m.flatten())

    # --- Generate LaTeX Code ---
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]") # Standard float placement options
    latex_lines.append("\\centering")

    # Build the tabular environment
    if orientation == 'h':
        # Horizontal: Labels are row headers. Data entries form columns.
        # n_cols is the number of data points per measurement (after broadcasting and flattening)
        actual_n_cols = n_entries 
        if table_spec is None:
            col_spec = '|l|' + ('c' * actual_n_cols) + '|' if actual_n_cols > 0 else '|l|'
        else:
            col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Optional header row for multi-entry data (if not a scalar-like table for each measurement)
        if n_entries > 1: # If each measurement contributes more than one cell
             header_row_entries = [str(i+1) for i in range(n_entries)]
             # Check if all measurements were originally scalar and got broadcast
             all_orig_scalar = all(meas.size == 1 for meas in measurements)
             if not (is_scalar_like_table and all_orig_scalar) : # Avoid "Index 1" if only one broadcasted column
                header_row = ["Label"] + header_row_entries # Changed "Index" to "Label" for clarity
                latex_lines.append(" & ".join(header_row) + " \\\\\\hline\\hline") # Double hline after header


        # Data rows: One row per measurement
        for i, label in enumerate(labels):
            safe_label = label.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
            # formatted_data[i] is already a flat list/array of strings
            row_content = [safe_label] + list(formatted_data[i])
            latex_lines.append(" & ".join(row_content) + " \\\\\\hline")

        latex_lines.append("\\end{tabular}")

    elif orientation == 'v':
        # Vertical: Labels are column headers. Data entries form rows.
        actual_n_cols = n_meas # Number of columns is the number of measurements
        if table_spec is None:
             col_spec = '|' + ('c|' * actual_n_cols) if actual_n_cols > 0 else '|'
        else:
             col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Header row: Measurement labels
        safe_labels = [lbl.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&') for lbl in labels]
        latex_lines.append(" & ".join(safe_labels) + " \\\\\\hline\\hline") 

        # Data rows: One row per data point index (up to n_entries)
        for i in range(n_entries):
            # Get the i-th formatted string from each measurement's flattened list
            row_content = [formatted_data[j][i] for j in range(n_meas)]
            latex_lines.append(" & ".join(row_content) + " \\\\") 
        latex_lines.append("\\hline") 
        latex_lines.append("\\end{tabular}")

    # Add caption if provided
    if caption:
        safe_caption = caption.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        latex_lines.append(f"\\caption{{{safe_caption}}}")
        # Automatic label generation could be:
        # label_base = "".join(filter(str.isalnum, caption)).lower()[:15].replace("__", "_")
        # if label_base: latex_lines.append(f"\\label{{tab:{label_base}}}")

    latex_lines.append("\\end{table}")

    # --- Final Output and Clipboard ---
    latex_string = "\n".join(latex_lines)

    if copy_to_clipboard and _HAS_PYPERCLIP:
        try:
            pyperclip.copy(latex_string)
        except Exception as e:
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
        fit_labels = [f"Fit {i+1} ({res.method})" if res.method else f"Fit {i+1}" for i, res in enumerate(fit_results)]
    elif len(fit_labels) != n_fits:
        raise ValueError(f"Number of fit_labels provided ({len(fit_labels)}) must match the "
                         f"number of FitResult objects ({n_fits}).")

    if orientation not in ['h', 'v']:
        raise ValueError("Orientation must be 'h' (horizontal) or 'v' (vertical).")

    # --- Data Collection and Formatting ---

    all_param_names_ordered = [] # Maintain order of first appearance
    seen_param_names = set()
    all_stat_keys_ordered = [] # Maintain order from include_stats
    seen_stat_keys = set()

    for res in fit_results:
        for name in res.parameter_names: # Use the defined order in FitResult
            if name not in seen_param_names:
                all_param_names_ordered.append(name)
                seen_param_names.add(name)
    
    if include_stats:
        for key in include_stats: # Keep user's preferred order for stats
            # Check if this stat key is present (not None) in at least one fit result
            # to decide if it should be a column/row in the table.
            is_key_present_in_any_fit = any(getattr(res, key, None) is not None for res in fit_results)
            if key not in seen_stat_keys and is_key_present_in_any_fit:
                all_stat_keys_ordered.append(key)
                seen_stat_keys.add(key)


    param_display_map = {name: param_labels.get(name, name.replace('_', '\\_')) for name in all_param_names_ordered} if param_labels else \
                        {name: name.replace('_', '\\_') for name in all_param_names_ordered}

    default_stat_display = {'chi_square': '$\\chi^2$',
                            'dof': 'DoF', 
                            'reduced_chi_square': '$\\chi^2/\\nu$'}
    stat_display_map = {key: stat_labels.get(key, default_stat_display.get(key, key.replace('_', '\\_'))) for key in all_stat_keys_ordered} if stat_labels else \
                       {key: default_stat_display.get(key, key.replace('_', '\\_')) for key in all_stat_keys_ordered}


    table_data_rows: Dict[str, List[str]] = {} # Key: display_label, Value: list of entries for each fit

    # Format parameters
    for param_name in all_param_names_ordered:
        display_label = param_display_map[param_name]
        row_entries = []
        for res in fit_results:
            param_meas = res.parameters.get(param_name) 
            if param_meas is not None:
                # param_meas.to_eng_string() returns an np.ndarray of strings.
                # Since parameters are scalar Measurements (size 1), it will be a 1-element array.
                # We need the string item itself.
                formatted_str_array = param_meas.to_eng_string(sig_figs_error=sig_figs_error)
                row_entries.append(formatted_str_array.item()) # Get the single string
            else:
                row_entries.append("---") 
        table_data_rows[display_label] = row_entries

    # Format statistics
    for stat_key in all_stat_keys_ordered:
        display_label = stat_display_map[stat_key]
        row_entries = []
        for res in fit_results:
            stat_val = getattr(res, stat_key, None)
            row_entries.append(_format_stat(stat_val))
        table_data_rows[display_label] = row_entries
    
    # Define the final order of rows for the table
    ordered_display_row_labels = [param_display_map[p] for p in all_param_names_ordered] + \
                                 [stat_display_map[s] for s in all_stat_keys_ordered]


    # --- Generate LaTeX Code ---
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")

    if orientation == 'h': 
        n_data_cols = n_fits 
        if table_spec is None:
            col_spec = '|l|' + ('c' * n_data_cols) + '|' if n_data_cols > 0 else '|l|'
        else:
            col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        safe_fit_labels_display = [lbl.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&') for lbl in fit_labels]
        header = ["Property"] + safe_fit_labels_display # Changed from "Parameter/Stat"
        latex_lines.append(" & ".join(header) + " \\\\\\hline\\hline") 

        for row_display_label in ordered_display_row_labels:
             # Ensure row_display_label is properly escaped if it contains LaTeX special chars not handled by maps
             safe_row_display_label = row_display_label # Assumes maps already handled this
             row_content_for_fits = table_data_rows[row_display_label]
             latex_lines.append(safe_row_display_label + " & " + " & ".join(row_content_for_fits) + " \\\\\\hline")

        latex_lines.append("\\end{tabular}")

    elif orientation == 'v': 
        n_data_cols = len(ordered_display_row_labels)
        if table_spec is None:
             col_spec = '|l|' + ('c' * n_data_cols) + '|' if n_data_cols > 0 else '|l|'
        else:
             col_spec = table_spec
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Header row: Parameter/Stat display labels (already prepared)
        header = ["Fit"] + ordered_display_row_labels # Changed from "Fit Label"
        latex_lines.append(" & ".join(header) + " \\\\\\hline\\hline") 

        for i, fit_disp_label in enumerate(fit_labels):
             safe_fit_disp_label = fit_disp_label.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
             row_entries_for_this_fit = [table_data_rows[prop_disp_label][i] for prop_disp_label in ordered_display_row_labels]
             latex_lines.append(safe_fit_disp_label + " & " + " & ".join(row_entries_for_this_fit) + " \\\\\\hline")

        latex_lines.append("\\end{tabular}")

    if caption:
        safe_caption = caption.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        latex_lines.append(f"\\caption{{{safe_caption}}}")
        # label_base = "".join(filter(str.isalnum, caption)).lower()[:15].replace("__", "_")
        # if label_base: latex_lines.append(f"\\label{{tab:fit_{label_base}}}")


    latex_lines.append("\\end{table}")

    latex_string = "\n".join(latex_lines)

    if copy_to_clipboard and _HAS_PYPERCLIP:
        try:
            pyperclip.copy(latex_string)
        except Exception as e:
            warnings.warn(f"Failed to copy table to clipboard. Error: {e}", RuntimeWarning)
    elif copy_to_clipboard and not _HAS_PYPERCLIP:
        warnings.warn("pyperclip not found. Cannot copy table to clipboard.", RuntimeWarning)

    return latex_string
