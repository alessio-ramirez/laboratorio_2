# --- START OF FILE interactive_plot_generator.py ---
"""
Interactive Plot Generator for Guezzi Fit Results

Provides functionality to generate interactive HTML reports for visualizing
and exploring FitResult objects. Users can manipulate fit parameters with
sliders and compare multiple fits.
"""

import panel as pn
import panel.io.state as pn_state # Explicit import for pn.state
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Optional, Dict, Callable, Any
import os
import warnings
import traceback # For detailed error printing

# Assuming these are correctly importable from your library structure
from .fitting import FitResult # If in a package
from .measurement import Measurement # If in a package
# If running as standalone scripts, you might need:
# from fitting import FitResult
# from measurement import Measurement


# --- Panel Extension Setup ---
THEME_TO_USE = None
try:
    from panel.theme import FastDarkTheme
    THEME_TO_USE = FastDarkTheme
    print("INFO: Using Panel's FastDarkTheme.")
except ImportError:
    warnings.warn(
        "Panel's 'Fast*' themes not directly importable from panel.theme. "
        "Attempting to use string name for theme or Panel default.", UserWarning
    )
    if hasattr(pn, '__version__') and pn.__version__ >= "1.0.0":
        try:
            pn.extension(theme="fast-dark", skip_existing=True)
            THEME_TO_USE = "fast-dark"
            print("INFO: Using Panel's 'fast-dark' theme string.")
        except Exception as e_theme:
            warnings.warn(f"String theme 'fast-dark' also failed: {e_theme}. Using Panel default theme.", UserWarning)
            THEME_TO_USE = None
    else:
        THEME_TO_USE = None

pn.extension(
    sizing_mode="stretch_width",
    notifications=True,
    mathjax=True,
    theme=THEME_TO_USE,
    # loading_spinner='dots' # Example spinner
)
print(f"INFO: Panel version: {pn.__version__}")
print(f"INFO: Matplotlib backend: {plt.get_backend()}")


def _get_slider_range_and_step(param_name: str, value: float, error: Optional[float], k: float = 5.0, n_steps: int = 300):
    # (Function remains largely the same as your well-developed version)
    min_val, max_val, step_val = 0.0, 1.0, 0.01
    if np.isnan(value):
        warnings.warn(f"Parameter '{param_name}' has NaN value. Using default slider range [-1, 1].", UserWarning)
        return -1.0, 1.0, 0.01
    has_valid_error = error is not None and np.isfinite(error) and error > 1e-18
    relative_range_preferred = False
    if not np.isclose(value, 0.0):
        if not has_valid_error: relative_range_preferred = True
        elif abs(k * error / value) < 0.05: relative_range_preferred = True
    if relative_range_preferred:
        f1, f2 = 0.2, 5.0
        if abs(k * error / value) < 0.01 and has_valid_error: f1, f2 = 0.5, 2.0
        if value > 0: min_val, max_val = value * f1, value * f2
        else: min_val, max_val = value * f2, value * f1
        if min_val >= max_val:
             min_val, max_val = min(value * f1, value * f2), max(value * f1, value * f2)
             if np.isclose(min_val, max_val):
                min_val = value - abs(value) * 0.5 if not np.isclose(value,0) else -1.0
                max_val = value + abs(value) * 0.5 if not np.isclose(value,0) else 1.0
    elif has_valid_error:
        min_val, max_val = value - k * error, value + k * error
        if min_val >= max_val:
            if not np.isclose(value, 0.0): min_val, max_val = value - abs(value) * k * 0.2, value + abs(value) * k * 0.2
            else: min_val, max_val = -k * error, k * error
            if min_val >= max_val: min_val, max_val = value - abs(k*error if error else 1.0), value + abs(k*error if error else 1.0)
    else:
        default_abs_range = max(1.0, abs(value if value else 1.0) * k * 0.5)
        min_val, max_val = value - default_abs_range, value + default_abs_range
    if np.isclose(min_val, max_val): max_val = min_val + max(abs(min_val * 0.1) if not np.isclose(min_val,0) else 1.0, 1e-9)
    if min_val > max_val: min_val, max_val = max_val, min_val
    dynamic_range = max_val - min_val
    if dynamic_range > 1e-15: step_val = dynamic_range / float(n_steps)
    else: step_val = abs(value) * 1e-4 if not np.isclose(value, 0) else 1e-4
    if np.isclose(step_val, 0.0) or step_val < 1e-18: step_val = (max_val - min_val) / float(n_steps) if (max_val - min_val) > 1e-15 else 1e-6
    step_val = abs(step_val)
    if np.isclose(step_val, 0.0) or step_val < 1e-18: step_val = 1e-6
    return float(min_val), float(max_val), float(step_val)


class InteractiveFitPlotter:
    def __init__(self, fit_results: Union[FitResult, List[FitResult]]):
        print("DEBUG: InteractiveFitPlotter.__init__ started")
        if isinstance(fit_results, FitResult):
            self.fit_results_list = [fit_results]
        else:
            self.fit_results_list = list(fit_results)

        if not self.fit_results_list:
            raise ValueError("No FitResult objects provided.")

        self.shared_x_data = self.fit_results_list[0].x_data
        self.shared_y_data = self.fit_results_list[0].y_data

        if self.shared_x_data is None or self.shared_y_data is None or \
           self.shared_x_data.value is None or self.shared_y_data.value is None:
            raise ValueError("The first FitResult object must contain valid x_data and y_data (with .value).")

        self.n_fit_points = 300
        self.fig, self.ax_main = plt.subplots(figsize=(8, 6))
        print(f"DEBUG: Matplotlib figure created: id={id(self.fig)}")

        self.param_sliders_map: Dict[str, pn.widgets.FloatSlider] = {}
        self.current_params_values: Dict[str, float] = {}

        self.stats_display = pn.pane.Markdown(
            "### Fit Statistics\nSelect an active fit and adjust sliders to see its stats.",
            width=360, margin=(10,5,10,5), css_classes=['guezzistatsdisplay'],
            stylesheets=['.guezzistatsdisplay {border: 1px solid lightgray; padding: 10px; background-color: #f9f9f9;}']
        )

        self.active_fit_index = 0

        fit_options_for_selector = {
            f"Fit {i+1}: {fr.method} ({fr.function.__name__ if fr.function else 'N/A'})": i
            for i, fr in enumerate(self.fit_results_list)
        }
        self.fit_selector = pn.widgets.Select(
            name='Active Fit for Sliders', options=fit_options_for_selector, value=0,
            margin=(5,5,10,5)
        )
        self.fit_selector.param.watch(self._on_active_fit_change, 'value')

        self.fit_visibility_toggles = pn.widgets.CheckBoxGroup(
            name='Visible Fits', options=list(fit_options_for_selector.keys()),
            value=[list(fit_options_for_selector.keys())[0]] if fit_options_for_selector else [],
            margin=(5,5,10,5)
        )
        self.fit_visibility_toggles.param.watch(self._trigger_plot_update_no_event, 'value')

        self.data_plot_artist = None
        self.masked_data_plot_artist = None
        self.fit_line_artists: Dict[int, plt.Line2D] = {}

        self.reset_button = pn.widgets.Button(
            name='Reset Active Fit Sliders', button_type='primary',
            width=200, margin=(10,5,15,5)
        )
        self.reset_button.on_click(self._reset_active_sliders)

        self.manual_update_button = pn.widgets.Button(
            name='DEBUG: Manual Plot Update', button_type='warning', width=200, margin=(5,5,5,5)
        )
        self.manual_update_button.on_click(self._trigger_plot_update_no_event) # Connect to the same update logic

        self.sliders_pane_column = pn.Column(sizing_mode='stretch_width', margin=(0,5,0,5))

        self.matplotlib_pane = pn.pane.Matplotlib(
            self.fig, tight=True, dpi=96,
            sizing_mode='stretch_both', margin=(0,10,10,5)
        )
        print(f"DEBUG: Matplotlib pane created, holding figure id={id(self.matplotlib_pane.object)}")


        self._setup_plot_base_elements()
        self._create_param_sliders_for_active_fit()
        self._trigger_plot_update_no_event() # Initial plot
        print("DEBUG: InteractiveFitPlotter.__init__ finished")

    def _robust_scalar_convert(self, raw_val, param_name_context="parameter"):
        # (Function remains the same)
        if isinstance(raw_val, (int, float, np.integer, np.floating)): return float(raw_val)
        if isinstance(raw_val, np.ndarray):
            if raw_val.size == 1: return float(raw_val.item())
            elif raw_val.size == 0:
                warnings.warn(f"Empty array for {param_name_context}. Using NaN.", UserWarning); return float('nan')
            else:
                warnings.warn(f"{param_name_context} array ({raw_val.size} el). Using first: {raw_val.flat[0]}.", UserWarning)
                return float(raw_val.flat[0])
        elif isinstance(raw_val, np.generic): return float(raw_val.item())
        else:
            try: return float(raw_val)
            except (TypeError, ValueError):
                warnings.warn(f"Could not convert {param_name_context} ({raw_val}) to float. NaN.", UserWarning)
                return float('nan')

    def _reset_active_sliders(self, event):
        print(f"DEBUG: _reset_active_sliders called for event: {event}")
        if not self.fit_results_list: return
        active_fr = self.fit_results_list[self.active_fit_index]
        for p_name in active_fr.parameter_names:
            slider = self.param_sliders_map.get(p_name)
            if slider and p_name in active_fr.parameters:
                param_meas = active_fr.parameters[p_name]
                current_best_fit_scalar = self._robust_scalar_convert(param_meas.value, f"best-fit for '{p_name}'")
                s_start, s_end = slider.start, slider.end
                if np.isnan(current_best_fit_scalar): current_best_fit_scalar = slider.value
                clamped_reset_val = current_best_fit_scalar
                if np.isfinite(current_best_fit_scalar) and np.isfinite(s_start) and np.isfinite(s_end) and s_start < s_end:
                    clamped_reset_val = float(np.clip(current_best_fit_scalar, s_start, s_end))
                elif np.isfinite(current_best_fit_scalar):
                     warnings.warn(f"Slider '{p_name}' invalid range [{s_start}, {s_end}] for reset.")
                if np.isnan(clamped_reset_val) and np.isfinite(current_best_fit_scalar):
                     warnings.warn(f"Clamping reset for '{p_name}' to [{s_start:.3e},{s_end:.3e}] gave NaN from {current_best_fit_scalar:.3e}.")
                     clamped_reset_val = current_best_fit_scalar
                slider.value = clamped_reset_val
                self.current_params_values[p_name] = clamped_reset_val
        self._trigger_plot_update_no_event()

    def _setup_plot_base_elements(self):
        print("DEBUG: _setup_plot_base_elements called")
        # (Function remains largely the same)
        fr_ref = self.fit_results_list[0]
        x_val = np.asarray(self.shared_x_data.value).flatten()
        y_val = np.asarray(self.shared_y_data.value).flatten()
        x_err_raw = self.shared_x_data.error if hasattr(self.shared_x_data, 'error') else None
        y_err_raw = self.shared_y_data.error if hasattr(self.shared_y_data, 'error') else None
        x_err = np.asarray(x_err_raw if x_err_raw is not None else 0).flatten()
        y_err = np.asarray(y_err_raw if y_err_raw is not None else 0).flatten()
        if x_err.size == 1 and x_val.size > 1: x_err = np.full_like(x_val, x_err[0])
        if y_err.size == 1 and y_val.size > 1: y_err = np.full_like(y_val, y_err[0])
        if x_err.shape != x_val.shape: x_err = np.zeros_like(x_val)
        if y_err.shape != y_val.shape: y_err = np.zeros_like(y_val)
        mask = fr_ref.mask if fr_ref.mask is not None else np.ones_like(x_val, dtype=bool)
        active_mask = np.asarray(mask, dtype=bool).flatten()
        if active_mask.shape != x_val.shape:
            warnings.warn("Mask shape incompatible, ignoring.", UserWarning); active_mask = np.ones_like(x_val, dtype=bool)
        inactive_mask = ~active_mask
        data_label_text = f"{getattr(self.shared_y_data, 'name', 'Data') or 'Data'}"
        if np.any(x_val[active_mask]):
            self.data_plot_artist = self.ax_main.errorbar(
                x_val[active_mask], y_val[active_mask],
                xerr=x_err[active_mask] if np.any(x_err) else None,
                yerr=y_err[active_mask] if np.any(y_err) else None,
                fmt='o', label=data_label_text, markersize=5, capsize=3, zorder=1, color='royalblue'
            )
        if np.any(x_val[inactive_mask]):
            self.masked_data_plot_artist = self.ax_main.errorbar(
                x_val[inactive_mask], y_val[inactive_mask],
                xerr=x_err[inactive_mask] if np.any(x_err) else None,
                yerr=y_err[inactive_mask] if np.any(y_err) else None,
                fmt='x', label='Masked Data', markersize=5, capsize=3, alpha=0.5, color='darkgray', zorder=1
            )
        prop_cycle = plt.rcParams['axes.prop_cycle']
        mpl_colors = prop_cycle.by_key()['color']
        for i, fr in enumerate(self.fit_results_list):
            fit_display_name = list(self.fit_selector.options.keys())[list(self.fit_selector.options.values()).index(i)]
            color_val = mpl_colors[(i+1) % len(mpl_colors)]
            line, = self.ax_main.plot([], [], lw=2, label=fit_display_name, zorder=2, color=color_val, alpha=0.8)
            self.fit_line_artists[i] = line
        x_name = getattr(self.shared_x_data, 'name', 'X') or 'X'
        x_unit = getattr(self.shared_x_data, 'unit', '') or ''
        y_name = getattr(self.shared_y_data, 'name', 'Y') or 'Y'
        y_unit = getattr(self.shared_y_data, 'unit', '') or ''
        x_axis_label = f"{x_name}" + (f" [{x_unit}]" if x_unit else "")
        y_axis_label = f"{y_name}" + (f" [{y_unit}]" if y_unit else "")
        self.ax_main.set_xlabel(x_axis_label, fontsize=12)
        self.ax_main.set_ylabel(y_axis_label, fontsize=12)
        self.ax_main.set_title("Interactive Fit Explorer", fontsize=14, fontweight='bold') # Initial title
        self.ax_main.legend(fontsize='small', loc='best')
        self.ax_main.grid(True, linestyle=':', alpha=0.6)
        self.fig.tight_layout()

    def _create_param_sliders_for_active_fit(self):
        print(f"DEBUG: _create_param_sliders_for_active_fit for fit_index: {self.active_fit_index}")
        # (Function remains largely the same)
        if not self.fit_results_list: self.sliders_pane_column.objects = [pn.pane.Markdown("No fit results.")]; return
        active_fr = self.fit_results_list[self.active_fit_index]
        self.param_sliders_map.clear()
        new_sliders_list = []
        if not active_fr.parameter_names or not active_fr.parameters:
             self.sliders_pane_column.objects = [pn.pane.Markdown("No parameters for this fit.")]; return
        for p_name in active_fr.parameter_names:
            if p_name not in active_fr.parameters:
                warnings.warn(f"Param '{p_name}' in names but not in FitResult.parameters.", UserWarning)
                new_sliders_list.append(pn.pane.Markdown(f"<i>Error: Param '{p_name}' data missing.</i>")); continue
            param_meas = active_fr.parameters[p_name]
            val_orig = self._robust_scalar_convert(param_meas.value, f"val for '{p_name}'")
            err_raw = param_meas.error
            err = self._robust_scalar_convert(err_raw, f"err for '{p_name}'") if err_raw is not None else None
            initial_slider_val = self.current_params_values.get(p_name, val_orig)
            if np.isnan(initial_slider_val) and np.isfinite(val_orig): initial_slider_val = val_orig
            min_v, max_v, step_v = _get_slider_range_and_step(p_name, initial_slider_val if np.isfinite(initial_slider_val) else val_orig, err)
            clamped_val_for_slider = initial_slider_val
            if np.isnan(initial_slider_val): clamped_val_for_slider = (min_v + max_v) / 2.0 if np.isfinite(min_v) and np.isfinite(max_v) else 0.0
            elif np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v :
                 clamped_val_for_slider = float(np.clip(initial_slider_val, min_v, max_v))
                 if not np.isclose(initial_slider_val, clamped_val_for_slider):
                    warnings.warn(f"Initial '{p_name}' ({initial_slider_val:.3e}) outside range [{min_v:.3e},{max_v:.3e}]. Clamped to {clamped_val_for_slider:.3e}.", UserWarning)
            slider = pn.widgets.FloatSlider(name=p_name, start=min_v, end=max_v, value=clamped_val_for_slider, step=step_v, margin=(0,5,8,5), format='0.3e')
            slider.param.watch(self._on_slider_value_change, 'value')
            self.param_sliders_map[p_name] = slider
            self.current_params_values[p_name] = clamped_val_for_slider
            new_sliders_list.append(slider)
        self.sliders_pane_column.objects = new_sliders_list if new_sliders_list else [pn.pane.Markdown("No params.")]

    def _on_active_fit_change(self, event):
        print(f"DEBUG: _on_active_fit_change: new active_fit_index={event.new}")
        self.active_fit_index = event.new
        self._create_param_sliders_for_active_fit()
        self._trigger_plot_update_no_event()

    def _on_slider_value_change(self, event):
        print(f"DEBUG: _on_slider_value_change: param='{event.obj.name}', new_value={event.new:.4e}")
        changed_param_name = event.obj.name
        self.current_params_values[changed_param_name] = event.new
        self._trigger_plot_update_no_event()

    def _update_plot_contents(self):
        """Core logic to update plot artists based on current state."""
        print(f"DEBUG: _update_plot_contents called for active_fit_index: {self.active_fit_index}")
        if not self.fit_results_list: return

        # --- Update plot title for visual feedback ---
        self.ax_main.set_title(f"Fit Explorer (Update #{np.random.randint(1000)})", fontsize=14, fontweight='bold')

        visible_fit_display_names = self.fit_visibility_toggles.value
        name_to_index_map = {name: idx for name, idx in self.fit_selector.options.items()}
        visible_fit_indices = [name_to_index_map[name] for name in visible_fit_display_names if name in name_to_index_map]

        x_data_vals_flat = np.asarray(self.shared_x_data.value).flatten()
        if len(x_data_vals_flat) == 0:
            for fit_idx in self.fit_line_artists:
                self.fit_line_artists[fit_idx].set_data([],[])
                self.fit_line_artists[fit_idx].set_visible(False)
            return # No data to plot against

        x_min_data, x_max_data = np.min(x_data_vals_flat), np.max(x_data_vals_flat)
        plot_range_extension = (x_max_data - x_min_data) * 0.05 if (x_max_data - x_min_data) > 1e-9 else 0.05
        x_plot_min = x_min_data - plot_range_extension
        x_plot_max = x_max_data + plot_range_extension
        if np.isclose(x_plot_min, x_plot_max): x_plot_min -= 1.0; x_plot_max += 1.0
        x_curve_points = np.linspace(x_plot_min, x_plot_max, self.n_fit_points)

        for fit_idx, fr_object in enumerate(self.fit_results_list):
            line_artist = self.fit_line_artists[fit_idx]
            if fit_idx in visible_fit_indices and fr_object.function and fr_object.parameter_names:
                params_to_plot = []
                valid_params = True
                current_param_source = {}
                if fit_idx == self.active_fit_index: current_param_source = self.current_params_values
                else:
                    for p_name_orig in fr_object.parameter_names:
                         if p_name_orig in fr_object.parameters:
                            current_param_source[p_name_orig] = self._robust_scalar_convert(fr_object.parameters[p_name_orig].value, p_name_orig)
                         else: current_param_source[p_name_orig] = float('nan')
                for p_name in fr_object.parameter_names:
                    val = current_param_source.get(p_name, float('nan')) # Use .get for safety
                    if np.isfinite(val): params_to_plot.append(val)
                    else:
                        original_val = float('nan')
                        if p_name in fr_object.parameters: original_val = self._robust_scalar_convert(fr_object.parameters[p_name].value, p_name)
                        if np.isfinite(original_val):
                            params_to_plot.append(original_val)
                            if fit_idx == self.active_fit_index:
                                warnings.warn(f"Param '{p_name}' for active fit {fit_idx+1} was NaN/missing. Using best-fit: {original_val:.3e}", UserWarning)
                        else:
                            valid_params = False; warnings.warn(f"Param '{p_name}' for fit {fit_idx+1} is NaN. Cannot plot.", UserWarning); break
                if valid_params:
                    try:
                        # print(f"DEBUG: Plotting Fit {fit_idx+1} with params: {params_to_plot}")
                        y_curve_values = fr_object.function(x_curve_points, *params_to_plot)
                        if np.all(np.isfinite(y_curve_values)):
                            line_artist.set_data(x_curve_points, y_curve_values)
                            line_artist.set_visible(True)
                        else:
                            warnings.warn(f"Fit {fit_idx+1} produced non-finite values. Line not updated.", RuntimeWarning)
                            line_artist.set_data([],[]); line_artist.set_visible(False)
                    except Exception as e:
                        warnings.warn(f"Error evaluating fit func for Fit {fit_idx+1} ({fr_object.function.__name__ if fr_object.function else 'N/A'}): {e}", RuntimeWarning)
                        # traceback.print_exc() # More detailed error
                        line_artist.set_data([],[]); line_artist.set_visible(False)
                else: line_artist.set_data([],[]); line_artist.set_visible(False)
            else: line_artist.set_data([],[]); line_artist.set_visible(False)

        self._recalculate_stats_for_active_fit()
        self.ax_main.legend(fontsize='small', loc='best') # Redraw legend

    def _trigger_plot_update_no_event(self, *args):
        """Wrapper to call update logic, ensuring it's executed by Panel."""
        print(f"DEBUG: _trigger_plot_update_no_event called with args: {args}")
        try:
            self._update_plot_contents() # The actual logic
            
            # CRITICAL: Notify Panel that the figure object has changed
            print(f"DEBUG: Assigning self.fig (id={id(self.fig)}) to self.matplotlib_pane.object (current id={id(self.matplotlib_pane.object) if self.matplotlib_pane.object else None})")
            self.matplotlib_pane.object = self.fig
            
            if self.fig.canvas:
                # print("DEBUG: Calling self.fig.canvas.draw_idle()")
                self.fig.canvas.draw_idle() # Ensure Matplotlib internal state is updated
            print("DEBUG: _trigger_plot_update_no_event finished successfully")

        except Exception as e:
            print(f"ERROR in _trigger_plot_update_no_event: {e}")
            traceback.print_exc()
            if pn_state.curdoc and pn_state.notifications:
                pn_state.notifications.error(f"Plot update failed: {e}", duration=5000)


    def _recalculate_stats_for_active_fit(self):
        # (Function remains largely the same - assumed to be less critical for "static plot" issue)
        # ... (ensure robust_scalar_convert and fallbacks are used)
        if not self.fit_results_list: return
        active_fr = self.fit_results_list[self.active_fit_index]
        active_fit_display_name = self._get_active_fit_display_name()
        if not active_fr.parameter_names or not active_fr.parameters or not active_fr.function:
            self.stats_display.object = f"**Stats ({active_fit_display_name}):**\nNo params/func."; return
        current_slider_params, valid_current_params = [], True
        for p_name in active_fr.parameter_names:
            slider_val = self.current_params_values.get(p_name)
            if slider_val is not None and np.isfinite(slider_val): current_slider_params.append(float(slider_val))
            else:
                original_val = self._robust_scalar_convert(active_fr.parameters[p_name].value, p_name)
                if np.isfinite(original_val): current_slider_params.append(original_val); warnings.warn(f"Using best-fit for '{p_name}' in stats (bad slider val).", UserWarning)
                else: valid_current_params = False; self.stats_display.object = f"**Stats ({active_fit_display_name}):**\nParam '{p_name}' invalid for stats."; return
        x_data_obj, y_data_obj = active_fr.x_data, active_fr.y_data
        x_val, y_val = np.asarray(x_data_obj.value).flatten(), np.asarray(y_data_obj.value).flatten()
        y_err_raw = y_data_obj.error if hasattr(y_data_obj, 'error') else None
        y_err_val = np.asarray(y_err_raw if y_err_raw is not None else 1.0).flatten()
        if y_err_val.size == 1 and y_val.size > 1: y_err_val = np.full_like(y_val, y_err_val[0])
        if y_err_val.shape != y_val.shape: y_err_val = np.ones_like(y_val)
        mask_arr = np.asarray(active_fr.mask if active_fr.mask is not None else True, dtype=bool).flatten()
        if mask_arr.size == 1 and x_val.size > 1: mask_arr = np.full_like(x_val, mask_arr[0], dtype=bool)
        if mask_arr.shape != x_val.shape: mask_arr = np.ones_like(x_val, dtype=bool)
        x_val_masked, y_val_masked, y_err_val_masked = x_val[mask_arr], y_val[mask_arr], y_err_val[mask_arr]
        num_points_for_stats = len(x_val_masked)
        if num_points_for_stats == 0: self.stats_display.object = f"**Stats ({active_fit_display_name}):**\nNo data points for stats."; return
        sigma_safe = np.where(np.isclose(y_err_val_masked, 0.0) | ~np.isfinite(y_err_val_masked), 1.0, y_err_val_masked)
        chi2_val = np.nan
        if valid_current_params:
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    y_pred = active_fr.function(x_val_masked, *current_slider_params)
                    residuals = (y_val_masked - y_pred) / sigma_safe
                    finite_mask_res = np.isfinite(residuals) & np.isfinite(y_pred)
                    if np.any(finite_mask_res): chi2_val = np.sum(residuals[finite_mask_res]**2)
            except Exception as e: warnings.warn(f"Error calculating Chi² for stats ({active_fit_display_name}): {e}", RuntimeWarning)
        dof_val = active_fr.dof
        if dof_val is None: dof_val = num_points_for_stats - len(current_slider_params)
        if dof_val is None or dof_val < 0 : dof_val = 0
        reduced_chi2_val = chi2_val / dof_val if dof_val > 0 and np.isfinite(chi2_val) else np.nan
        stats_md = f"**Stats for Active Fit ({active_fit_display_name}):**\n"
        param_summary_parts = [f"  - *{name}*: {self.current_params_values.get(name, float('nan')):.4g}" for i, name in enumerate(active_fr.parameter_names)]
        stats_md += "Current Parameters (from sliders):\n" + "\n".join(param_summary_parts) + "\n"
        stats_md += f"\n$\\chi^2$: {chi2_val:.4g if np.isfinite(chi2_val) else 'N/A'}\n"
        stats_md += f"DoF ($\\nu$): {int(dof_val) if dof_val is not None else 'N/A'}\n"
        stats_md += f"Reduced $\\chi^2$ ($\\chi^2/\\nu$): {reduced_chi2_val:.4g if np.isfinite(reduced_chi2_val) else 'N/A'}\n"
        self.stats_display.object = stats_md


    def _get_active_fit_display_name(self) -> str:
        try: return next(key for key, value in self.fit_selector.options.items() if value == self.active_fit_index)
        except StopIteration: return f"Fit Index {self.active_fit_index + 1}"

    def get_panel_layout(self):
        print("DEBUG: get_panel_layout called")
        controls_title = pn.pane.Markdown("### Parameter Controls", margin=(0,5,5,5))
        visibility_title = pn.pane.Markdown("### Fit Visibility", margin=(15,5,5,5))

        sidebar_widgets = [controls_title]
        if len(self.fit_results_list) > 1:
            sidebar_widgets.append(self.fit_selector)

        sidebar_widgets.append(self.sliders_pane_column)
        sidebar_widgets.append(self.reset_button)
        sidebar_widgets.append(self.manual_update_button) # Add debug button

        if len(self.fit_results_list) > 1:
             sidebar_widgets.append(visibility_title)
             sidebar_widgets.append(self.fit_visibility_toggles)

        sidebar_widgets.append(self.stats_display)

        sidebar = pn.Column(
            *sidebar_widgets, width=380, height=700, scroll=True,
            sizing_mode='fixed', margin=(0, 5, 0, 10)
        )

        main_layout = pn.Row(
            self.matplotlib_pane, sidebar,
            sizing_mode='stretch_width'
        )
        return main_layout

    def save_report(self, filename="interactive_fit_report.html"):
        layout = self.get_panel_layout()
        try:
            self._trigger_plot_update_no_event() # Ensure plot is current
            # embed_json=True and save_path='./' are correct for creating the JSON dir
            layout.save(filename, embed=True, embed_json=True, save_path='./')
            abs_path = os.path.abspath(os.path.join('./', filename)) # Get absolute path of HTML
            print(f"Interactive report successfully saved to: {abs_path}")
            print(f"Associated JSON data is in a subdirectory next to this HTML file.")
            print(f"To view with full interactivity, serve this directory using a local HTTP server (e.g., 'python -m http.server' in this directory) and open http://localhost:8000/{filename}")

            if pn_state.curdoc and pn_state.notifications:
                pn_state.notifications.success(f"Report saved to {os.path.basename(abs_path)}", duration=5000)
        except Exception as e:
            error_msg = f"Failed to save report: {e}"
            if pn_state.curdoc and pn_state.notifications: pn_state.notifications.error(error_msg, duration=0)
            warnings.warn(f"Error saving report to '{filename}': {e}", RuntimeWarning); print(error_msg)
            traceback.print_exc()


    def show_report(self, port=0, threaded=False, title="Interactive Fit Report"):
        print(f"DEBUG: show_report called (port={port}, threaded={threaded})")
        layout = self.get_panel_layout()
        self._trigger_plot_update_no_event() # Ensure plot is current

        if threaded:
            print("DEBUG: Starting server in a thread.")
            layout.show(port=port, threaded=True, title=title)
        else:
            print("DEBUG: Starting blocking server.")
            pn.serve(layout, port=port, show=True, title=title)


def generate_interactive_report(fit_results: Union[FitResult, List[FitResult]],
                                output_filename: Optional[str] = "interactive_fit_report.html",
                                auto_open_in_browser: bool = False,
                                show_in_server: bool = False,
                                server_port: int = 0,
                                server_threaded: bool = False):
    print("DEBUG: generate_interactive_report called")
    try:
        plotter = InteractiveFitPlotter(fit_results)
        if output_filename:
            plotter.save_report(output_filename)
            if auto_open_in_browser:
                try:
                    import webbrowser
                    abs_path = os.path.abspath(os.path.join('./', output_filename))
                    webbrowser.open(f"file://{abs_path}")
                except Exception as e_web:
                    warnings.warn(f"Could not auto-open: {e_web}. Open '{os.path.realpath(output_filename)}' manually.", UserWarning)
        if show_in_server:
            plotter.show_report(port=server_port, threaded=server_threaded)
        return plotter
    except Exception as e_gen:
        error_msg = f"Failed to generate interactive report: {e_gen}"
        warnings.warn(error_msg, RuntimeWarning)
        if pn_state.curdoc and pn_state.notifications: pn_state.notifications.error(error_msg, duration=0)
        traceback.print_exc()
        raise

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Starting Example Usage ---")
    # --- Dummy Measurement and FitResult classes for standalone testing ---
    from dataclasses import dataclass, field
    # Measurement and FitResult classes are defined in their respective files
    # For this example, we'll use simplified dummy versions if not found.
    try:
        # Attempt to import the actual classes if they are in the same directory or accessible
        from measurement import Measurement as ActualMeasurement
        from fitting import FitResult as ActualFitResult
        print("INFO: Using ActualMeasurement and ActualFitResult from local files.")
        Measurement = ActualMeasurement
        FitResult = ActualFitResult
    except ImportError:
        print("WARNING: Actual Measurement/FitResult not found. Using dummy classes for example.")
        @dataclass
        class DummyMeasurement:
            value: Any; error: Optional[Any] = None; name: Optional[str] = None; unit: Optional[str] = None
            @property
            def shape(self): return np.asarray(self.value).shape
        @dataclass
        class DummyFitResult:
            parameters: Dict[str, DummyMeasurement] = field(default_factory=dict)
            covariance_matrix: Optional[np.ndarray] = None; chi_square: Optional[float] = None
            dof: Optional[int] = None; reduced_chi_square: Optional[float] = None
            function: Optional[Callable] = None; parameter_names: List[str] = field(default_factory=list)
            x_data: Optional[DummyMeasurement] = None; y_data: Optional[DummyMeasurement] = None
            method: str = ""; mask: Optional[np.ndarray] = None; success: bool = False
            fit_object: Optional[Any] = None
        Measurement = DummyMeasurement # type: ignore
        FitResult = DummyFitResult # type: ignore
    # --- End Dummy classes logic ---

    def linear_func(x_arg, m_param, c_param): # Renamed to avoid clashes
        return m_param * x_arg + c_param

    def quadratic_func(x_arg, a_param, b_param, c_param): # Renamed
        return a_param * x_arg**2 + b_param * x_arg + c_param

    np.random.seed(0)
    x_vals = np.linspace(0, 10, 20) # Fewer points for faster example
    y_true_linear = linear_func(x_vals, m_param=2, c_param=1)
    y_noise_linear = np.random.normal(0, 0.5, size=x_vals.shape)
    y_linear_data = y_true_linear + y_noise_linear

    y_true_quad = quadratic_func(x_vals, a_param=0.5, b_param=-2, c_param=5)
    y_noise_quad = np.random.normal(0, 1.0, size=x_vals.shape)
    y_quad_data = y_true_quad + y_noise_quad

    dummy_x = Measurement(x_vals, error=0.1, name="Time", unit="s")

    fit_res1 = FitResult(
        parameters={"m": Measurement(2.1, 0.05), "c": Measurement(0.9, 0.1)},
        function=linear_func, parameter_names=["m_param", "c_param"], # Match renamed func params
        x_data=dummy_x, y_data=Measurement(y_linear_data, error=0.5, name="Voltage", unit="V"),
        method="Linear Fit Ex1", success=True, chi_square=25.0, dof=18
    )

    fit_res2 = FitResult(
        parameters={"a": Measurement(0.55,0.08),"b": Measurement(-2.2,0.1),"c": Measurement(5.1,0.2)},
        function=quadratic_func, parameter_names=["a_param", "b_param", "c_param"], # Match renamed
        x_data=dummy_x, y_data=Measurement(y_quad_data, error=1.0, name="Position", unit="m"),
        method="Quadratic Fit Ex2", success=True, chi_square=35.0, dof=17
    )
    fit_res2.y_data = fit_res1.y_data # Forcing shared Y to test data point plotting

    print("DEBUG: Dummy FitResult objects created.")
    print("--- Generating report with dummy data (will run server) ---")
    
    # This will start a server and block if server_threaded=False
    generate_interactive_report(
        [fit_res1, fit_res2],
        output_filename="test_interactive_report.html", # It will also save
        auto_open_in_browser=True, # Attempt to open the saved HTML
        show_in_server=True,       # Crucially, run the server
        server_port=0,             # Use a random available port
        server_threaded=False      # Make it blocking for easy testing from script
    )
    print("--- Example Usage Finished (Server might still be running if threaded) ---")

# --- END OF FILE interactive_plot_generator.py ---