"""
Plotting Utilities for Measurements and Fits

Provides functions to visualize Measurement data with error bars and
to plot FitResult objects, including data, fit curve, and residuals.
Uses Matplotlib for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Any, Union
import warnings
import os # For path handling if needed

# Import local package components
from .measurement import Measurement
from .fitting import FitResult

# Use Matplotlib's default property cycle for colors if none provided
try:
    DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
except KeyError: # Fallback if 'color' isn't the key (unlikely but possible)
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def plot_measurements(*measurements: Measurement,
                      labels: Optional[List[str]] = None,
                      ax: Optional[plt.Axes] = None,
                      xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None,
                      title: Optional[str] = None,
                      colors: Optional[List[str]] = None,
                      fmts: Union[str, List[str]] = 'o',
                      use_names_units: bool = True,
                      show_plot: bool = False,
                      save_path: Optional[str] = None,
                      **errorbar_kwargs) -> plt.Axes:
    """
    Plots one or more pairs of Measurement objects with error bars using Matplotlib.

    This function is designed to quickly visualize experimental data points where
    both x and y values have associated uncertainties.

    Args:
        *measurements: Measurement objects to plot, provided in pairs (x1, y1, x2, y2, ...).
                       Each pair represents a dataset to be plotted.
        labels: Optional list of labels for each (x, y) dataset pair, used in the legend.
                If None, default labels ("Data 1", "Data 2", ...) are generated.
        ax: Optional Matplotlib Axes object to plot on. If None, a new figure and
            axes are created automatically.
        xlabel, ylabel, title: Optional strings for overriding the default axis labels
                               and plot title. If None, defaults are constructed using
                               the `name` and `unit` attributes from the *first* x/y
                               Measurement pair if `use_names_units` is True.
        colors: Optional list of color strings (e.g., 'red', '#FF0000') for each dataset.
                If None, colors are cycled from Matplotlib's default property cycle.
        fmts: Format string(s) passed to `plt.errorbar` controlling the marker style.
              Can be a single string (applied to all datasets) or a list of strings
              (one per dataset). Common examples: 'o' (dots), '.' (pixels), 's' (squares),
              'none' (no marker, only error bars). (Default 'o').
        use_names_units: If True, automatically generate axis labels and a default title
                         using the `name` and `unit` attributes from the first (x, y)
                         measurement pair provided. (Default True).
        show_plot (bool): If True, call `plt.show()` after creating the plot to display
                          it interactively. Blocks script execution until plot window is closed.
                          (Default False).
        save_path (Optional[str]): If provided, the plot will be saved to this file path.
                                   The format is inferred from the extension (e.g., '.png', '.pdf').
                                   Saving uses `bbox_inches='tight'` to minimize whitespace.
                                   (Default None, do not save).
        **errorbar_kwargs: Additional keyword arguments passed directly to the
                           `ax.errorbar` function for customizing the plot appearance.
                           Examples: `markersize=5`, `capsize=3`, `elinewidth=1`,
                           `linestyle='none'` (often used with markers).
                           **Avoid passing 'color' here; use the 'colors' argument instead.**

    Returns:
        matplotlib.axes.Axes: The Axes object containing the plot.

    Raises:
        ValueError: If an odd number of Measurement objects is provided (must be pairs),
                    or if the lengths of `labels`, `colors`, or `fmts` (if lists)
                    do not match the number of datasets.
        TypeError: If non-Measurement objects are passed.
    """
    if not all(isinstance(m, Measurement) for m in measurements):
        raise TypeError("All arguments must be Measurement objects.")
    if len(measurements) % 2 != 0:
        raise ValueError("Measurements must be provided in (x, y) pairs (even number of arguments).")

    # --- Input Validation and Setup ---
    # Check for misuse of 'color' kwarg
    if 'color' in errorbar_kwargs:
        warnings.warn("Keyword argument 'color' (singular) was passed via **errorbar_kwargs. "
                      "Please use the 'colors' (plural) argument to specify plot colors for distinct datasets. "
                      "Ignoring the 'color' keyword argument.", UserWarning)
        errorbar_kwargs.pop('color') # Remove it to avoid conflict

    num_datasets = len(measurements) // 2

    # Setup axes
    if ax is None:
        # Create new figure and axes
        fig, ax = plt.subplots(figsize=errorbar_kwargs.pop('figsize', (8, 5))) # Pop figsize if user provided it
    else:
        # Use provided axes
        fig = ax.get_figure()
        if 'figsize' in errorbar_kwargs:
             warnings.warn("Ignoring 'figsize' keyword argument because 'ax' was provided.", UserWarning)
             errorbar_kwargs.pop('figsize')

    # Set default errorbar style if not specified by user
    errorbar_kwargs.setdefault('linestyle', 'none') # Common default for scatter-like data
    errorbar_kwargs.setdefault('markersize', 5)
    errorbar_kwargs.setdefault('capsize', 3) # Small caps on error bars

    # Normalize labels, colors, fmts to lists of the correct length
    if labels is None: labels = [f"Data {i+1}" for i in range(num_datasets)]
    _colors = colors # Use temp variable
    if _colors is None: _colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(num_datasets)]
    if isinstance(fmts, str): fmts = [fmts] * num_datasets

    # Check lengths after defaulting/normalizing
    if len(labels) != num_datasets: raise ValueError(f"Mismatch: Expected {num_datasets} labels, got {len(labels)}")
    if len(_colors) != num_datasets: raise ValueError(f"Mismatch: Expected {num_datasets} colors, got {len(_colors)}")
    if len(fmts) != num_datasets: raise ValueError(f"Mismatch: Expected {num_datasets} fmts, got {len(fmts)}")

    # --- Plotting Loop ---
    x_combined_name, x_combined_unit = "", ""
    y_combined_name, y_combined_unit = "", ""
    has_plotted_anything = False

    for i in range(num_datasets):
        x_data = measurements[2*i]
        y_data = measurements[2*i+1]

        # Ensure data is 1D for errorbar plotting
        try:
             x_val_flat = np.atleast_1d(x_data.value).flatten()
             x_err_flat = np.atleast_1d(x_data.error).flatten()
             y_val_flat = np.atleast_1d(y_data.value).flatten()
             y_err_flat = np.atleast_1d(y_data.error).flatten()
             # Broadcast errors if they were scalar
             x_err_flat = np.broadcast_to(x_err_flat, x_val_flat.shape)
             y_err_flat = np.broadcast_to(y_err_flat, y_val_flat.shape)
        except Exception as e:
             warnings.warn(f"Could not process data for dataset {i+1} (Label: '{labels[i]}'). Skipping. Error: {e}", RuntimeWarning)
             continue

        if len(x_val_flat) != len(y_val_flat):
             warnings.warn(f"Dataset {i+1} (Label: '{labels[i]}') has mismatched x ({len(x_val_flat)}) "
                           f"and y ({len(y_val_flat)}) lengths after flattening. Skipping.", RuntimeWarning)
             continue
        if len(x_val_flat) == 0:
            warnings.warn(f"Dataset {i+1} (Label: '{labels[i]}') contains no data points. Skipping.", RuntimeWarning)
            continue

        # Plot the current dataset using its specific color, fmt, label
        ax.errorbar(x_val_flat, y_val_flat,
                    xerr=x_err_flat, yerr=y_err_flat,
                    fmt=fmts[i], color=_colors[i], label=labels[i],
                    **errorbar_kwargs) # Pass remaining kwargs
        has_plotted_anything = True

        # Collect names/units from the *first* dataset for potential axis labels
        if i == 0 and use_names_units:
             x_combined_name = getattr(x_data, 'name', "")
             x_combined_unit = getattr(x_data, 'unit', "")
             y_combined_name = getattr(y_data, 'name', "")
             y_combined_unit = getattr(y_data, 'unit', "")

    # --- Final Touches ---
    # Set labels and title
    if use_names_units:
        # Construct default labels if specific ones not provided by user
        final_xlabel = xlabel if xlabel else (f"{x_combined_name} [{x_combined_unit}]" if x_combined_unit else x_combined_name)
        final_ylabel = ylabel if ylabel else (f"{y_combined_name} [{y_combined_unit}]" if y_combined_unit else y_combined_name)
        default_title = f"{y_combined_name} vs {x_combined_name}" if y_combined_name and x_combined_name else "Measurement Plot"
        final_title = title if title else default_title
    else:
        # Use user-provided labels/title or leave blank
        final_xlabel = xlabel if xlabel else ""
        final_ylabel = ylabel if ylabel else ""
        final_title = title if title else ""

    ax.set_xlabel(final_xlabel)
    ax.set_ylabel(final_ylabel)
    ax.set_title(final_title)

    # Add legend only if labels were provided or generated and something was plotted
    handles, legend_labels = ax.get_legend_handles_labels()
    if legend_labels and has_plotted_anything:
        ax.legend(handles=handles, labels=legend_labels)
    ax.grid(True, linestyle=':') # Add a subtle grid

    # --- Save and Show Logic ---
    if save_path:
        try:
            # Use bbox_inches='tight' to minimize whitespace around the plot
            # Use a reasonable default DPI for saved figures
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot successfully saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to '{save_path}'. Error: {e}", RuntimeWarning)

    if show_plot:
        plt.show() # Display the plot

    return ax


def plot_fit(fit_result: FitResult,
             ax: Optional[plt.Axes] = None,
             plot_data: bool = True,
             plot_masked_data: bool = True,
             plot_fit_line: bool = True,
             n_fit_points: int = 200,
             plot_residuals: bool = False,
             color: Optional[str] = None,
             fmt: str = 'o',
             masked_fmt: str = 'x',
             masked_color_offset: bool = True, # Option to use different color for masked
             masked_alpha: float = 0.5,
             fit_linestyle: str = '-',
             data_label: Optional[str] = None,
             fit_label: Optional[str] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             use_names_units: bool = True,
             show_params: bool = False,
             show_stats: bool = False,
             annotation_fontsize: int = 9, # Slightly smaller default for annotations
             annotation_location: str = 'best', # Control annotation box location
             show_plot: bool = False,
             save_path: Optional[str] = None,
             **errorbar_kwargs) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Plots the data, best-fit curve, and optionally residuals from a FitResult object.

    This function provides a comprehensive visualization of a fitting procedure,
    showing the original data points (with error bars), the fitted model curve,
    and optionally a plot of the residuals (data - fit) to help assess the
    quality and identify systematic trends in the fit.

    Args:
        fit_result: The FitResult object obtained from `perform_fit`.
        ax: Optional Matplotlib Axes for the main plot. If `plot_residuals` is True,
            this should be the *top* axes of a pre-existing 2-panel figure setup
            (e.g., created with `plt.subplots(2, 1, sharex=True)`). If None,
            a suitable figure/axes setup is created automatically.
        plot_data: If True, plot the original data points (respecting the mask). (Default True).
        plot_masked_data: If True and a mask was used in the fit, show the points
                          that were *excluded* by the mask. They are typically shown
                          with a different marker/color/alpha. (Default True).
        plot_fit_line: If True and the fit was successful, draw the fitted curve. (Default True).
        n_fit_points: Number of points used to draw a smooth curve for the fit line
                      over the range of the fitted data. (Default 200).
        plot_residuals: If True, create a second subplot below the main plot showing
                        the residuals (y_data - y_fit) with their y-error bars.
                        Helps visualize the goodness of fit. Requires `ax` to be None
                        or the top axes of a shared-x 2-panel setup. (Default False).
        color: Base color string (e.g., 'blue', '#00FF00') for the primary data points
               and the fit line. If None, automatically cycles through Matplotlib's
               default color sequence.
        fmt: Format string for *unmasked* data points (passed to `errorbar`). (Default 'o').
        masked_fmt: Format string for *masked* (excluded) data points. (Default 'x').
        masked_color_offset: If True, uses the *next* color in the cycle for masked points,
                             otherwise uses the same base `color` but with `masked_alpha`. (Default True).
        masked_alpha: Alpha transparency (0 to 1) for masked data points. (Default 0.5).
        fit_linestyle: Linestyle string for the fit curve (e.g., '-', '--', ':'). (Default '-').
        data_label: Custom label for the *unmasked* data points in the legend.
                    If None, defaults to the y_data name or "Data".
        fit_label: Custom label for the fit line in the legend. If None, defaults
                   to something like "Fit (curve_fit)" or "Fit (odr)".
        xlabel, ylabel, title: Optional strings to override default axis labels/title.
                               Defaults are derived from data names/units if `use_names_units=True`.
        use_names_units: If True, use `name` and `unit` from the `fit_result`'s
                         x_data and y_data for default axis labels/title. (Default True).
        show_params: If True and fit was successful, annotate the plot with the
                     fitted parameter values and uncertainties using `to_eng_string`. (Default False).
        show_stats: If True and fit was successful, annotate the plot with key
                    goodness-of-fit statistics (like χ²/DoF). (Default False).
        annotation_fontsize: Font size for the parameter/stats annotations box. (Default 9).
        annotation_location: Location string for the annotation box (passed to `ax.text` via
                             heuristic placement, e.g., 'best', 'upper left', 'lower right').
                             'best' tries to find a good spot automatically. (Default 'best').
        show_plot (bool): If True, call `plt.show()` to display the plot interactively. (Default False).
        save_path (Optional[str]): If provided, save the plot to this file path. (Default None).
        **errorbar_kwargs: Additional keyword arguments passed to `ax.errorbar` for plotting
                           the *data points*. Examples: `markersize`, `capsize`, `elinewidth`.
                           **Avoid passing 'color' or 'label' here.**

    Returns:
        matplotlib.axes.Axes or Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
            If `plot_residuals` is False, returns the main Axes object.
            If `plot_residuals` is True, returns a tuple `(main_ax, residual_ax)`.

    Raises:
        ValueError: If `fit_result` is missing necessary data (x_data, y_data) or if
                    mask shape is incompatible.
        TypeError: If `fit_result` is not a valid FitResult object.
    """
    if not isinstance(fit_result, FitResult):
         raise TypeError("Input 'fit_result' must be a FitResult object.")
    if fit_result.x_data is None or fit_result.y_data is None:
        raise ValueError("FitResult object is missing required x_data or y_data attributes.")

    x_data = fit_result.x_data
    y_data = fit_result.y_data
    mask = fit_result.mask # Mask used *during* the fit (True=used)

    # --- Setup Axes ---
    main_ax: plt.Axes
    res_ax: Optional[plt.Axes] = None
    fig: plt.Figure

    if plot_residuals:
        # Need a 2-panel figure (main plot + residuals below)
        if ax is not None:
             # User provided an axes, assume it's the top one for the main plot.
             # We need to find or create the bottom one for residuals.
             fig = ax.get_figure()
             # Check if 'ax' already has a shared-x axes below it
             axes_list = fig.get_axes()
             shared_axes_group = ax.get_shared_x_axes()
             found_res_ax = None
             for other_ax in axes_list:
                 if other_ax is ax: continue
                 pos_ax = ax.get_position()
                 pos_other = other_ax.get_position()
                 # Heuristic: Check if other_ax is below, aligned, and shares x-axis
                 is_below = pos_other.y1 < pos_ax.y0
                 is_aligned = abs(pos_other.x0 - pos_ax.x0) < 0.01 and abs(pos_other.x1 - pos_ax.x1) < 0.01
                 shares_x = shared_axes_group.joined(ax, other_ax)
                 if is_below and is_aligned and shares_x:
                     found_res_ax = other_ax
                     break
             if found_res_ax:
                 main_ax = ax
                 res_ax = found_res_ax
                 print("Info: Found existing shared-x axes below the provided 'ax'. Using it for residuals.")
             else:
                 # Could not find suitable existing axes, create new figure.
                 warnings.warn("Provided 'ax' with 'plot_residuals=True', but couldn't find a suitable existing "
                               "shared-x axes below it. Creating a new 2-panel figure.", RuntimeWarning)
                 fig, (main_ax, res_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), # Slightly taller
                                                      gridspec_kw={'height_ratios': [3, 1]}) # Main plot gets more space
        else:
             # Create a new 2-panel figure from scratch
             fig, (main_ax, res_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                                                   gridspec_kw={'height_ratios': [3, 1]})
    else:
        # Only need a single panel for the main plot
        if ax is None:
             fig, main_ax = plt.subplots(figsize=errorbar_kwargs.pop('figsize',(8, 5))) # Pop figsize if provided
        else:
             main_ax = ax
             fig = main_ax.get_figure()
             if 'figsize' in errorbar_kwargs:
                 warnings.warn("Ignoring 'figsize' keyword argument because 'ax' was provided.", UserWarning)
                 errorbar_kwargs.pop('figsize')


    # --- Color and Labels ---
    # Determine base color
    if color is None:
        # Get the next color from the axes' property cycle if none provided
        # _get_lines is semi-private, but common practice. Use public API if possible.
        # Alternatively: `color = next(ax._get_lines.prop_cycler)['color']` but need care if cycle empty
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors_in_cycle = prop_cycle.by_key().get('color', DEFAULT_COLORS)
        # Find current cycle state - complex, easier to just get next color directly
        try:
             color = main_ax._get_lines.get_next_color()
        except AttributeError: # Fallback if axes state is unusual
             color = DEFAULT_COLORS[0]

    # Determine color for masked points
    masked_color = color # Default to same color
    if plot_masked_data and masked_color_offset:
         # Try to get the *next* color in the cycle for masked points
         try:
              # This is tricky, might advance cycle. Clone state? Simpler: find index and advance manually
              current_colors = main_ax._get_lines.prop_cycler.by_key().get('color', DEFAULT_COLORS)
              current_index = -1
              try:
                  # Find index of base color in the cycle list
                  current_index = list(current_colors).index(color)
              except ValueError:
                  current_index = 0 # Default if base color not in list
              masked_color = current_colors[(current_index + 1) % len(current_colors)]
         except Exception: # Fallback if getting next color fails
             masked_color = DEFAULT_COLORS[1 % len(DEFAULT_COLORS)] # Use second default color

    # Define default labels using data names if not provided
    y_data_name = getattr(y_data, 'name', 'Data')
    x_data_name = getattr(x_data, 'name', 'X')
    if data_label is None: data_label = y_data_name if y_data_name != "y" else "Data" # Avoid generic 'Data' vs 'x'
    if fit_label is None: fit_label = f"Fit ({fit_result.method})"


    # --- Prepare Data for Plotting ---
    errorbar_kwargs.setdefault('linestyle', 'none')
    errorbar_kwargs.setdefault('markersize', 5)
    errorbar_kwargs.setdefault('capsize', 3)

    # Ensure data are 1D numpy arrays for consistent indexing
    x_val = np.atleast_1d(x_data.value).flatten()
    x_err = np.atleast_1d(x_data.error).flatten()
    y_val = np.atleast_1d(y_data.value).flatten()
    y_err = np.atleast_1d(y_data.error).flatten()
    # Broadcast errors to match values shape
    try:
        x_err = np.broadcast_to(x_err, x_val.shape)
        y_err = np.broadcast_to(y_err, y_val.shape)
    except ValueError:
        raise ValueError("Cannot broadcast errors to match data values shape.")

    # Define the active mask (points used in fit)
    if mask is None:
        # If no mask was used in fit, all points are considered active
        active_mask = np.ones_like(x_val, dtype=bool)
    else:
        # Use the mask stored in FitResult
        try:
            active_mask = np.broadcast_to(np.asarray(mask, dtype=bool), x_val.shape)
        except ValueError:
             raise ValueError(f"Mask shape {np.asarray(mask).shape} incompatible with data shape {x_val.shape}.")

    inactive_mask = ~active_mask
    any_active = np.any(active_mask)
    any_inactive = np.any(inactive_mask)

    # --- Plot Data Points ---
    # Plot active (unmasked) points
    plotted_active = False
    if plot_data and any_active:
        main_ax.errorbar(x_val[active_mask], y_val[active_mask],
                         xerr=x_err[active_mask], yerr=y_err[active_mask],
                         fmt=fmt, color=color, label=data_label, **errorbar_kwargs)
        plotted_active = True

    # Plot inactive (masked) points
    plotted_masked = False
    if plot_masked_data and any_inactive:
         masked_kwargs = errorbar_kwargs.copy()
         masked_kwargs['alpha'] = masked_kwargs.get('alpha', masked_alpha) # Apply transparency
         # Only add label for masked points if active points weren't plotted, or if specifically desired
         masked_points_label = f"{data_label} (masked)" if not plotted_active else None
         # Pass the chosen masked color
         main_ax.errorbar(x_val[inactive_mask], y_val[inactive_mask],
                          xerr=x_err[inactive_mask], yerr=y_err[inactive_mask],
                          fmt=masked_fmt, color=masked_color, label=masked_points_label, **masked_kwargs)
         plotted_masked = True


    # --- Plot Fit Line ---
    plotted_fit_line = False
    if plot_fit_line and fit_result.success and fit_result.function and any_active:
        # Extract optimal parameter values
        try:
            # Parameters are stored as Measurement objects in FitResult
             params_opt = [p.value for p in fit_result.parameters.values()]
        except (AttributeError, TypeError, StopIteration):
             warnings.warn("Could not extract optimal parameter values from fit_result.parameters. "
                           "Skipping fit line plot.", RuntimeWarning)
             params_opt = None

        if params_opt is not None:
            try:
                # Determine plotting range for the fit line based on *active* data
                x_active_val = x_val[active_mask]
                x_min, x_max = np.min(x_active_val), np.max(x_active_val)

                # Handle case of single active point or all points at same x
                if np.isclose(x_min, x_max):
                     # Create a small range around the single point
                     x_range_ext = np.abs(x_min) * 0.1 if not np.isclose(x_min, 0) else 1.0 # 10% or 1 unit
                     # Consider extending based on x-error if available
                     if np.any(x_err[active_mask] > 1e-15):
                         x_range_ext = max(x_range_ext, 5 * np.mean(x_err[active_mask])) # e.g., ±5 sigma range
                     x_fit_min = x_min - x_range_ext
                     x_fit_max = x_max + x_range_ext
                     # Ensure range is not zero width
                     if np.isclose(x_fit_min, x_fit_max): x_fit_max += 1.0 # Add 1 unit width
                else:
                     # Extend the range slightly beyond min/max data points (e.g., 5%)
                     x_range_ext = (x_max - x_min) * 0.05
                     x_fit_min = x_min - x_range_ext
                     x_fit_max = x_max + x_range_ext

                # Generate smooth x values for plotting the curve
                x_fit_curve = np.linspace(x_fit_min, x_fit_max, max(2, n_fit_points))

                # Calculate corresponding y values using the fitted function
                y_fit_curve = fit_result.function(x_fit_curve, *params_opt)

                # Plot the fit line
                main_ax.plot(x_fit_curve, y_fit_curve, color=color, linestyle=fit_linestyle, label=fit_label)
                plotted_fit_line = True

            except Exception as e:
                warnings.warn(f"Could not evaluate fit function for plotting: {e}. Skipping fit line.", RuntimeWarning)

    # --- Plot Residuals ---
    plotted_residuals = False
    if plot_residuals and res_ax is not None and fit_result.success and fit_result.function and params_opt is not None:
        try:
            # Plot residuals for active points
            if any_active:
                x_active = x_val[active_mask]
                y_active = y_val[active_mask]
                y_err_active = y_err[active_mask]
                y_pred_active = fit_result.function(x_active, *params_opt)
                residuals_active = y_active - y_pred_active
                # Use same marker/color as main plot, no label needed for residuals
                res_kwargs = errorbar_kwargs.copy()
                res_kwargs.pop('label', None) # Remove label if present
                res_ax.errorbar(x_active, residuals_active, yerr=y_err_active, # Only y-error on residuals usually shown
                                fmt=fmt, color=color, **res_kwargs)

            # Plot residuals for masked points
            if plot_masked_data and any_inactive:
                x_inactive = x_val[inactive_mask]
                y_inactive = y_val[inactive_mask]
                y_err_inactive = y_err[inactive_mask]
                y_pred_inactive = fit_result.function(x_inactive, *params_opt)
                residuals_inactive = y_inactive - y_pred_inactive
                masked_res_kwargs = errorbar_kwargs.copy()
                masked_res_kwargs['alpha'] = masked_res_kwargs.get('alpha', masked_alpha)
                masked_res_kwargs.pop('label', None)
                res_ax.errorbar(x_inactive, residuals_inactive, yerr=y_err_inactive,
                                fmt=masked_fmt, color=masked_color, **masked_res_kwargs)

            # Add horizontal line at zero for reference
            res_ax.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=0) # Place behind points

            # Label the residual axis
            res_y_label = "Residuals"
            y_data_unit = getattr(y_data, 'unit', None)
            if y_data_unit and use_names_units:
                res_y_label += f" [{y_data_unit}]"
            res_ax.set_ylabel(res_y_label)
            res_ax.grid(True, linestyle=':')
            plotted_residuals = True

        except Exception as e:
            warnings.warn(f"Could not calculate or plot residuals: {e}.", RuntimeWarning)
            if res_ax: res_ax.set_ylabel("Residuals (Error)") # Indicate failure

    # --- Annotations (Parameters and Stats) ---
    annotation_lines = []
    if show_params and fit_result.success and fit_result.parameters:
         annotation_lines.append("Parameters:")
         param_items = list(fit_result.parameters.items())
         for name, param in param_items:
              # Use Measurement's formatting (respects sig figs from utils fix)
              # Use 2 sig figs for error in annotations by default
              try:
                   param_str = param.to_eng_string(sig_figs_error=2)[0]
              except Exception as fmt_err:
                   warnings.warn(f"Could not format parameter {name} for annotation: {fmt_err}", RuntimeWarning)
                   param_str = f"{param.value:.3g} \u00B1 {param.error:.2g} (format err)" # Fallback
              annotation_lines.append(f"  {name} = {param_str}")

    if show_stats and fit_result.success:
         stats_added = False
         # Prefer Reduced Chi Square if available and meaningful
         if fit_result.reduced_chi_square is not None and fit_result.dof is not None and fit_result.dof > 0:
             try:
                 # Format numbers nicely for annotation
                 chi2_nu_str = f"{fit_result.reduced_chi_square:.3g}"
                 chi2_str = f"{fit_result.chi_square:.3g}" if fit_result.chi_square is not None else "N/A"
                 dof_str = f"{fit_result.dof}"
                 if annotation_lines: annotation_lines.append("") # Add space before stats
                 annotation_lines.append("Goodness of Fit:")
                 annotation_lines.append(f"  $\\chi^2/\\nu = {chi2_nu_str}$") # Use LaTeX math mode
                 annotation_lines.append(f"  ($\\chi^2={chi2_str}, \\nu={dof_str}$)")
                 stats_added = True
             except (AttributeError, TypeError, ValueError) as fmt_err:
                 warnings.warn(f"Could not format fit statistics for annotation: {fmt_err}", RuntimeWarning)
         # Fallback to Chi Square if reduced is not available/applicable
         elif fit_result.chi_square is not None and fit_result.dof is not None and not stats_added:
              try:
                  chi2_str = f"{fit_result.chi_square:.3g}"
                  dof_str = f"{fit_result.dof}"
                  if annotation_lines: annotation_lines.append("")
                  annotation_lines.append("Goodness of Fit:")
                  annotation_lines.append(f"  $\\chi^2 = {chi2_str}$ ($\\nu={dof_str}$)")
                  stats_added = True
              except (AttributeError, TypeError, ValueError) as fmt_err:
                  warnings.warn(f"Could not format fit statistics (Chi^2) for annotation: {fmt_err}", RuntimeWarning)

    # Add annotation box to the main plot if there are lines to show
    if annotation_lines:
         text_str = "\n".join(annotation_lines)
         # Use a standard box style
         props = dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.75)
         # Place the text box based on annotation_location ('best' is heuristic)
         # We need numerical coordinates for ax.text. Map 'best' etc. to locations.
         loc_map = {
             'best': (0.03, 0.97), 'upper right': (0.97, 0.97), 'upper left': (0.03, 0.97),
             'lower left': (0.03, 0.03), 'lower right': (0.97, 0.03), 'center left': (0.03, 0.5),
             'center right': (0.97, 0.5), 'lower center': (0.5, 0.03), 'upper center': (0.5, 0.97),
             'center': (0.5, 0.5)
         }
         # Determine horizontal/vertical alignment based on location
         ha_map = {'left': 'left', 'right': 'right', 'center': 'center'}
         va_map = {'upper': 'top', 'lower': 'bottom', 'center': 'center'}
         loc_key = annotation_location.lower().replace(' ', '')

         # Default placement: upper left
         x_coord, y_coord = loc_map.get(loc_key, loc_map['upper left'])
         h_align = 'left' if 'left' in loc_key else 'right' if 'right' in loc_key else 'center'
         v_align = 'top' if 'upper' in loc_key else 'bottom' if 'lower' in loc_key else 'center'
         if loc_key == 'best': # Default 'best' to upper left placement
              x_coord, y_coord = loc_map['upper left']
              h_align, v_align = 'left', 'top'


         main_ax.text(x_coord, y_coord, text_str, transform=main_ax.transAxes,
                      fontsize=annotation_fontsize, verticalalignment=v_align,
                      horizontalalignment=h_align, bbox=props, zorder=10) # High zorder to be on top


    # --- Final Axis Labels, Title, Legend, Grid ---
    if use_names_units:
        x_unit = getattr(x_data, 'unit', None)
        y_unit = getattr(y_data, 'unit', None)
        final_xlabel = xlabel if xlabel is not None else (f"{x_data_name} [{x_unit}]" if x_unit else x_data_name)
        final_ylabel = ylabel if ylabel is not None else (f"{y_data_name} [{y_unit}]" if y_unit else y_data_name)
        default_base_title = f"{y_data_name} vs {x_data_name}" if (y_data_name != "y" or x_data_name != "x") else "Fit Plot"
        final_title = title if title is not None else default_base_title
    else:
        final_xlabel = xlabel if xlabel is not None else ""
        final_ylabel = ylabel if ylabel is not None else ""
        final_title = title if title is not None else ""

    # Apply labels/title
    # X label only on the bottom-most plot (main_ax or res_ax)
    bottom_ax = res_ax if plotted_residuals else main_ax
    bottom_ax.set_xlabel(final_xlabel)

    main_ax.set_ylabel(final_ylabel)

    if plotted_residuals:
        # Title goes on the top plot (main_ax)
        main_ax.set_title(final_title)
        # Remove x-tick labels from the top plot for cleaner look
        main_ax.tick_params(axis='x', labelbottom=False)
    else:
        # Title on the single plot
        main_ax.set_title(final_title)

    # Add legend to main plot if any labels were generated
    handles, legend_labels = main_ax.get_legend_handles_labels()
    if legend_labels:
        # Try placing legend somewhat intelligently, avoiding annotation box if possible
        legend_loc = 'best'
        # Heuristic: if annotation likely in upper left/right, try opposite corner
        if (show_params or show_stats):
             anno_loc_key = annotation_location.lower().replace(' ', '')
             if 'left' in anno_loc_key: legend_loc = 'upper right'
             elif 'right' in anno_loc_key: legend_loc = 'upper left'
             # If annotation center, 'best' might be okay.
        main_ax.legend(handles=handles, labels=legend_labels, loc=legend_loc)

    # Add grid to main plot
    main_ax.grid(True, linestyle=':')

    # Adjust layout to prevent labels overlapping (only if we created the figure)
    # If user provided 'ax', they are responsible for final layout adjustments.
    if ax is None:
        try:
            # tight_layout adjusts subplot params for a tight layout.
            # rect can reserve space for suptitle if needed.
            fig.tight_layout(rect=[0, 0, 1, 0.97] if final_title else None) # Leave space for title
        except Exception as e:
            # tight_layout can sometimes fail with complex annotations or layouts
            warnings.warn(f"fig.tight_layout() failed: {e}. Plot spacing may be suboptimal.", RuntimeWarning)


    # --- Save and Show Logic ---
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot successfully saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to '{save_path}'. Error: {e}", RuntimeWarning)

    if show_plot:
        plt.show()

    # --- Return Value ---
    if plotted_residuals:
         return main_ax, res_ax
    else:
         return main_ax
