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
from dataclasses import dataclass

# Import local package components
from .measurement import Measurement
from .fitting import FitResult

# Use Matplotlib's default property cycle for colors if none provided
try:
    DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
except KeyError:
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


@dataclass
class PlotStyle:
    """Configuration for plot styling"""
    colors: Optional[List[str]] = None
    marker: str = 'o'
    masked_marker: str = 'x'
    masked_alpha: float = 0.5
    linestyle: str = '-'
    markersize: int = 5
    capsize: int = 3
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = DEFAULT_COLORS


class PlotHelper:
    """Helper class for common plotting operations"""
    
    @staticmethod
    def validate_measurements(*measurements: Measurement) -> None:
        """Validate that all inputs are Measurement objects and come in pairs"""
        if not all(isinstance(m, Measurement) for m in measurements):
            raise TypeError("All arguments must be Measurement objects.")
        if len(measurements) % 2 != 0:
            raise ValueError("Measurements must be provided in (x, y) pairs.")
    
    @staticmethod
    def setup_axes(ax: Optional[plt.Axes], figsize: Tuple[float, float] = (8, 5)) -> Tuple[plt.Figure, plt.Axes]:
        """Setup matplotlib axes, creating new ones if needed"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        return fig, ax
    
    
    @staticmethod
    def generate_labels(measurements: List[Measurement], custom_labels: Optional[List[str]] = None) -> List[str]:
        """Generate appropriate labels for datasets"""
        num_datasets = len(measurements) // 2
        if custom_labels is None:
            return [f"Data {i+1}" for i in range(num_datasets)]
        if len(custom_labels) != num_datasets:
            raise ValueError(f"Expected {num_datasets} labels, got {len(custom_labels)}")
        return custom_labels
    
    @staticmethod
    def get_axis_labels(x_data: Measurement, y_data: Measurement, 
                       custom_xlabel: Optional[str] = None, 
                       custom_ylabel: Optional[str] = None) -> Tuple[str, str]:
        """Generate axis labels from measurement names and units"""
        if custom_xlabel is not None:
            xlabel = custom_xlabel
        else:
            x_name = getattr(x_data, 'name', 'X')
            x_unit = getattr(x_data, 'unit', '')
            xlabel = f"{x_name} [{x_unit}]" if x_unit else x_name
        
        if custom_ylabel is not None:
            ylabel = custom_ylabel
        else:
            y_name = getattr(y_data, 'name', 'Y')
            y_unit = getattr(y_data, 'unit', '')
            ylabel = f"{y_name} [{y_unit}]" if y_unit else y_name
        
        return xlabel, ylabel
    
    @staticmethod
    def save_and_show(fig: plt.Figure, save_path: Optional[str] = None, show_plot: bool = False) -> None:
        """Handle saving and displaying plots"""
        if save_path:
            try:
                fig.savefig(save_path, bbox_inches='tight', dpi=600)
                print(f"Plot successfully saved to: {save_path}")
            except Exception as e:
                warnings.warn(f"Failed to save plot to '{save_path}'. Error: {e}", RuntimeWarning)
        
        if show_plot:
            plt.show()


class DataPlotter:
    """Handles plotting of measurement data points"""
    
    def __init__(self, style: PlotStyle):
        self.style = style
    
    def plot_dataset(self, ax: plt.Axes, x_data: Measurement, y_data: Measurement, 
                    label: str, color: str, **kwargs) -> None:
        """Plot a single dataset with error bars"""
        try:
            # We are assuming .value and .error contains 1d numpy array
            x_val, x_err = (x_data.value, x_data.error)
            y_val, y_err = (y_data.value, y_data.error)
            
            if len(x_val) == 0:
                warnings.warn(f"Dataset '{label}' contains no data points. Skipping.", RuntimeWarning)
                return
            
            ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err,
                       fmt=self.style.marker, color=color, label=label,
                       linestyle='none', markersize=self.style.markersize,
                       capsize=self.style.capsize, **kwargs)
        
        except Exception as e:
            warnings.warn(f"Could not plot dataset '{label}': {e}", RuntimeWarning)
    
    def plot_masked_data(self, ax: plt.Axes, x_data: Measurement, y_data: Measurement,
                        mask: np.ndarray, color: str, **kwargs) -> None:
        """Plot masked (excluded) data points"""
        try:
            x_val, x_err = (x_data.value, x_data.error)
            y_val, y_err = (y_data.value, y_data.error)
            
            inactive_mask = ~mask
            if not np.any(inactive_mask):
                return # Return if mask is useless
            
            ax.errorbar(x_val[inactive_mask], y_val[inactive_mask],
                       xerr=x_err[inactive_mask], yerr=y_err[inactive_mask],
                       fmt=self.style.masked_marker, color=color, alpha=self.style.masked_alpha,
                       linestyle='none', markersize=self.style.markersize,
                       capsize=self.style.capsize, **kwargs)
        
        except Exception as e:
            warnings.warn(f"Could not plot masked data: {e}", RuntimeWarning)


class FitPlotter:
    """Handles plotting of fit results"""
    
    def __init__(self, style: PlotStyle):
        self.style = style
    
    def plot_fit_curve(self, ax: plt.Axes, fit_result: FitResult, color: str, 
                      n_points: int = 1000, label: str = "Fit") -> None:
        """Plot the fitted curve"""
        if not fit_result.success or not fit_result.function:
            return
        
        try:
            params = [p.value for p in fit_result.parameters.values()]
            x_data = fit_result.x_data
            
            # Get active data range
            x_val = x_data.value
            if fit_result.mask is not None:
                active_mask = np.broadcast_to(np.asarray(fit_result.mask, dtype=bool), x_val.shape)
                x_active = x_val[active_mask]
            else:
                x_active = x_val
            
            if len(x_active) == 0:
                return
            
            # Generate smooth curve
            x_min, x_max = np.min(x_active), np.max(x_active)
            x_range = (x_max - x_min) * 0.1
            x_min -= x_range
            x_max += x_range
            
            x_curve = np.linspace(x_min, x_max, n_points)
            y_curve = fit_result.function(x_curve, *params)
            
            ax.plot(x_curve, y_curve, color=color, linestyle=self.style.linestyle, label=label)
        
        except Exception as e:
            warnings.warn(f"Could not plot fit curve: {e}", RuntimeWarning)
    
    def plot_residuals(self, ax: plt.Axes, fit_result: FitResult, color: str) -> None:
        """Plot residuals in a separate axes"""
        if not fit_result.success or not fit_result.function:
            return
        
        try:
            params = [p.value for p in fit_result.parameters.values()]
            x_data = fit_result.x_data
            y_data = fit_result.y_data
            
            x_val = x_data.value
            y_val, y_err = (y_data.value, y_data.error)
            
            if fit_result.mask is not None:
                active_mask = np.broadcast_to(np.asarray(fit_result.mask, dtype=bool), x_val.shape) # Useless
                x_active = x_val[active_mask]
                y_active = y_val[active_mask]
                y_err_active = y_err[active_mask]
            else:
                x_active = x_val
                y_active = y_val
                y_err_active = y_err
            
            if len(x_active) == 0:
                return
            
            y_pred = fit_result.function(x_active, *params)
            residuals = y_active - y_pred
            
            ax.errorbar(x_active, residuals, yerr=y_err_active,
                       fmt=self.style.marker, color=color, linestyle='none',
                       markersize=self.style.markersize, capsize=self.style.capsize)
            
            ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add residuals for masked data if present
            if fit_result.mask is not None:
                inactive_mask = ~active_mask
                if np.any(inactive_mask):
                    x_inactive = x_val[inactive_mask]
                    y_inactive = y_val[inactive_mask]
                    y_err_inactive = y_err[inactive_mask]
                    y_pred_inactive = fit_result.function(x_inactive, *params)
                    residuals_inactive = y_inactive - y_pred_inactive
                    
                    ax.errorbar(x_inactive, residuals_inactive, yerr=y_err_inactive,
                               fmt=self.style.masked_marker, color=color, alpha=self.style.masked_alpha,
                               linestyle='none', markersize=self.style.markersize, capsize=self.style.capsize)
        
        except Exception as e:
            warnings.warn(f"Could not plot residuals: {e}", RuntimeWarning)


class AnnotationHandler:
    """Handles plot annotations for parameters and statistics"""
    
    @staticmethod
    def create_parameter_text(fit_result: FitResult) -> List[str]:
        """Create text lines for parameter annotations"""
        if not fit_result.success or not fit_result.parameters:
            return []
        
        lines = ["Parameters:"]
        for name, param in fit_result.parameters.items():
            try:
                param_str = param.to_eng_string(sig_figs_error=2)[0]
            except Exception:
                param_str = f"{param.value:.3g} ± {param.error:.2g}"
            lines.append(f"  {name} = {param_str}")
        
        return lines
    
    @staticmethod
    def create_stats_text(fit_result: FitResult) -> List[str]:
        """Create text lines for fit statistics"""
        if not fit_result.success:
            return []
        
        lines = []
        if fit_result.reduced_chi_square is not None and fit_result.dof is not None and fit_result.dof > 0:
            lines.extend([
                "Goodness of Fit:",
                f"  χ²/ν = {fit_result.reduced_chi_square:.3g}",
                f"  (χ²={fit_result.chi_square:.3g}, ν={fit_result.dof})"
            ])
        elif fit_result.chi_square is not None and fit_result.dof is not None:
            lines.extend([
                "Goodness of Fit:",
                f"  χ² = {fit_result.chi_square:.3g} (ν={fit_result.dof})"
            ])
        
        return lines
    
    @staticmethod
    def add_annotation(ax: plt.Axes, lines: List[str], location: str = 'upper left') -> None:
        """Add annotation box to plot"""
        if not lines:
            return
        
        text = "\n".join(lines)
        
        # Location mapping
        loc_coords = {
            'upper left': (0.03, 0.97), 'upper right': (0.97, 0.97),
            'lower left': (0.03, 0.03), 'lower right': (0.97, 0.03),
            'center': (0.5, 0.5)
        }
        
        x_coord, y_coord = loc_coords.get(location.lower(), loc_coords['upper left'])
        h_align = 'left' if 'left' in location.lower() else 'right' if 'right' in location.lower() else 'center'
        v_align = 'top' if 'upper' in location.lower() else 'bottom' if 'lower' in location.lower() else 'center'
        
        props = dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.75)
        ax.text(x_coord, y_coord, text, transform=ax.transAxes, fontsize=9,
                verticalalignment=v_align, horizontalalignment=h_align, bbox=props)


def _group_fits_by_data(fit_results: List[FitResult]) -> List[List[FitResult]]:
    """Group fit results that share the same data"""
    groups = []
    processed = set()
    
    for i, fit1 in enumerate(fit_results):
        if i in processed:
            continue
        
        group = [fit1]
        processed.add(i)
        
        for j, fit2 in enumerate(fit_results[i+1:], i+1):
            if j in processed:
                continue
            
            # Check if fits share the same data
            if (np.array_equal(fit1.x_data.value, fit2.x_data.value) and 
                np.array_equal(fit1.y_data.value, fit2.y_data.value)):
                group.append(fit2)
                processed.add(j)
        
        groups.append(group)
    
    return groups


def plot_measurements(*measurements: Measurement,
                      labels: Optional[List[str]] = None,
                      ax: Optional[plt.Axes] = None,
                      xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None,
                      title: Optional[str] = None,
                      style: Optional[PlotStyle] = None,
                      show_plot: bool = False,
                      save_path: Optional[str] = None) -> plt.Axes:
    """
    Plots one or more pairs of Measurement objects with error bars.
    
    Args:
        *measurements: Measurement objects in pairs (x1, y1, x2, y2, ...)
        labels: Optional labels for each dataset
        ax: Optional Matplotlib Axes object
        xlabel, ylabel, title: Optional axis labels and title
        style: Optional PlotStyle object for customization
        show_plot: If True, display the plot
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    PlotHelper.validate_measurements(*measurements)
    
    if style is None:
        style = PlotStyle()
    
    fig, ax = PlotHelper.setup_axes(ax)
    data_plotter = DataPlotter(style)
    
    num_datasets = len(measurements) // 2
    labels = PlotHelper.generate_labels(measurements, labels)
    
    # Plot each dataset
    for i in range(num_datasets):
        x_data = measurements[2*i]
        y_data = measurements[2*i+1]
        color = style.colors[i % len(style.colors)]
        data_plotter.plot_dataset(ax, x_data, y_data, labels[i], color)
    
    # Set labels using first dataset if not provided
    if num_datasets > 0:
        x_data = measurements[0]
        y_data = measurements[1]
        final_xlabel, final_ylabel = PlotHelper.get_axis_labels(x_data, y_data, xlabel, ylabel)
        ax.set_xlabel(final_xlabel)
        ax.set_ylabel(final_ylabel)
        
        if title is None:
            x_name = getattr(x_data, 'name', 'X')
            y_name = getattr(y_data, 'name', 'Y')
            title = f"{y_name} vs {x_name}"
        ax.set_title(title)
    
    # Finalize plot
    ax.legend()
    ax.grid(True, linestyle=':')
    
    PlotHelper.save_and_show(fig, save_path, show_plot)
    return ax


def plot_fit(*fit_results: FitResult,
             ax: Optional[plt.Axes] = None,
             plot_data: bool = True,
             plot_masked_data: bool = True,
             plot_residuals: bool = False,
             show_params: bool = False,
             show_stats: bool = False,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             style: Optional[PlotStyle] = None,
             show_plot: bool = False,
             save_path: Optional[str] = None) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Plots fit results with data, fit curves, and optionally residuals.
    
    Can handle multiple fits of the same data by automatically grouping them.
    
    Args:
        *fit_results: One or more FitResult objects
        ax: Optional Matplotlib Axes object
        plot_data: If True, plot the data points
        plot_masked_data: If True, plot masked data points
        plot_residuals: If True, create residuals subplot
        show_params: If True, show parameter annotations
        show_stats: If True, show fit statistics
        xlabel, ylabel, title: Optional axis labels and title
        style: Optional PlotStyle object
        show_plot: If True, display the plot
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib.axes.Axes or tuple of axes if residuals are plotted
    """
    if not fit_results:
        raise ValueError("At least one FitResult must be provided")
    
    if not all(isinstance(fr, FitResult) for fr in fit_results):
        raise TypeError("All arguments must be FitResult objects")
    
    if style is None:
        style = PlotStyle()
    
    # Group fits by shared data
    fit_groups = _group_fits_by_data(list(fit_results))
    
    # Setup axes
    main_ax = ax
    res_ax = None
    
    if plot_residuals:
        if ax is None:
            fig, (main_ax, res_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                                                  gridspec_kw={'height_ratios': [3, 1]})
        else:
            # Try to find or create residual axes
            fig = ax.get_figure()
            main_ax = ax
            # For simplicity, create new figure if residuals requested with existing ax
            warnings.warn("Creating new figure for residuals when ax is provided", UserWarning)
            fig, (main_ax, res_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                                                  gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, main_ax = PlotHelper.setup_axes(main_ax)
    
    data_plotter = DataPlotter(style)
    fit_plotter = FitPlotter(style)
    
    # Plot each group
    for group_idx, fit_group in enumerate(fit_groups):
        base_color = style.colors[group_idx % len(style.colors)]
        
        # Plot data once per group
        if plot_data and fit_group:
            representative_fit = fit_group[0]
            data_label = getattr(representative_fit.y_data, 'name', f'Data {group_idx + 1}')
            data_plotter.plot_dataset(main_ax, representative_fit.x_data, representative_fit.y_data,
                                    data_label, base_color)
            
            # Plot masked data
            if plot_masked_data and representative_fit.mask is not None:
                mask = np.broadcast_to(np.asarray(representative_fit.mask, dtype=bool), 
                                    np.atleast_1d(representative_fit.x_data.value).shape)
                data_plotter.plot_masked_data(main_ax, representative_fit.x_data, representative_fit.y_data,
                                            mask, base_color)
        
        # Plot fit curves for each fit in the group
        for fit_idx, fit_result in enumerate(fit_group):
            fit_color = base_color if len(fit_group) == 1 else style.colors[(group_idx * 10 + fit_idx) % len(style.colors)]
            fit_label = f"Fit ({fit_result.method})"
            if len(fit_group) > 1:
                fit_label += f" {fit_idx + 1}"
            
            fit_plotter.plot_fit_curve(main_ax, fit_result, fit_color, label=fit_label)
            
            # Plot residuals
            if plot_residuals and res_ax is not None:
                fit_plotter.plot_residuals(res_ax, fit_result, fit_color)
    
    # Set labels and title
    if fit_groups and fit_groups[0]:
        representative_fit = fit_groups[0][0]
        final_xlabel, final_ylabel = PlotHelper.get_axis_labels(
            representative_fit.x_data, representative_fit.y_data, xlabel, ylabel)
        
        bottom_ax = res_ax if res_ax is not None else main_ax
        bottom_ax.set_xlabel(final_xlabel)
        main_ax.set_ylabel(final_ylabel)
        
        if title is None:
            x_name = getattr(representative_fit.x_data, 'name', 'X')
            y_name = getattr(representative_fit.y_data, 'name', 'Y')
            title = f"{y_name} vs {x_name} - Fit"
        main_ax.set_title(title)
        
        # Add annotations for first successful fit
        annotation_lines = []
        if show_params:
            annotation_lines.extend(AnnotationHandler.create_parameter_text(representative_fit))
        if show_stats:
            if annotation_lines:
                annotation_lines.append("")
            annotation_lines.extend(AnnotationHandler.create_stats_text(representative_fit))
        
        AnnotationHandler.add_annotation(main_ax, annotation_lines)
    
    # Finalize plots
    main_ax.legend()
    main_ax.grid(True, linestyle=':')
    
    if res_ax is not None:
        res_ax.set_ylabel("Residuals")
        res_ax.grid(True, linestyle=':')
        main_ax.tick_params(axis='x', labelbottom=False)
        try:
            fig.tight_layout()
        except:
            pass
    
    PlotHelper.save_and_show(fig, save_path, show_plot)
    
    return (main_ax, res_ax) if res_ax is not None else main_ax