# Guezzi: A Scientific Utility Library for Physics Students

**Version:** 2.1.0

## Introduction

Guezzi is a Python library designed to assist physics students with common tasks encountered in laboratory courses and data analysis. It provides tools for handling measurements with uncertainties, performing curve fitting, basic statistical analysis, and generating outputs like plots and LaTeX tables suitable for reports.

**Core Goals:**

*   **Simplify Uncertainty Propagation:** Automate the often tedious process of error propagation.
*   **Streamline Fitting:** Provide easy access to standard fitting algorithms (`curve_fit`, `ODR`) with clear result interpretation.
*   **Facilitate Reporting:** Generate plots and LaTeX tables quickly.
*   **Physics Context:** Use terminology and provide explanations relevant to experimental physics.

**Installation:**

Currently, Guezzi might be used by placing the source files (`guezzi` directory containing `__init__.py`, `measurement.py`, etc.) in your project or Python path. For future development, installation via `pip` might be considered.

```bash
# Example (if packaged later):
# pip install guezzi
```

## Core Concept: `Measurement` Object (`measurement.py`)

The foundation of Guezzi is the `Measurement` class. It represents a physical quantity that has both a nominal `value` and an associated `error` (uncertainty), along with optional `unit` and `name` metadata.

```python
import guezzi as gz
import numpy as np

# Create scalar measurements
length = gz.Measurement(10.5, 0.2, unit='cm', name='Length')
time = gz.Measurement(value=5.12, error=0.05, unit='s', name='Time')
temp = gz.Measurement(295, 1, unit='K') # Name is optional

# Create array measurements
voltages = gz.Measurement([1.1, 2.0, 3.2], [0.1, 0.1, 0.2], unit='V', name='Voltage')
# Scalar error applied to all values
currents = gz.Measurement([0.2, 0.4, 0.6], 0.02, unit='A', name='Current')

print(length)         # Output: 10.5 ± 0.2 cm
print(voltages)       # Output: [1.1 ± 0.1, 2.0 ± 0.1, 3.2 ± 0.2] V
print(length.value)   # Output: 10.5
print(length.error)   # Output: 0.2
print(length.unit)    # Output: cm
print(voltages.shape) # Output: (3,)
```

### Error Propagation Theory

Guezzi automatically propagates uncertainties through calculations using **first-order Taylor expansion (linear error propagation)**. For a function `f(x, y, ...)` where `x`, `y`, ... are independent measurements `x ± σ_x`, `y ± σ_y`, ..., the uncertainty `σ_f` in `f` is approximated by:

`σ_f² ≈ (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)² + ...`

Where `∂f/∂x` is the partial derivative of `f` with respect to `x`, evaluated at the nominal values.

**Key Assumptions & Limitations:**

1.  **Small Uncertainties / Linearity:** The formula assumes uncertainties are small enough that the function `f` is approximately linear over the range defined by the errors (e.g., `x ± σ_x`). If the function has significant curvature within this range, the linear approximation may be inaccurate.
2.  **Independence:** Calculations between *different* `Measurement` objects (e.g., `m1 + m2`, `m1 * m2`) assume the uncertainties `σ1` and `σ2` are statistically independent (uncorrelated). Operations involving the *same* object (e.g., `m - m`, `m * m`, `np.sin(m)`) correctly handle the correlations. The library does *not* track correlations across multiple separate operations.
3.  **Standard Deviation:** The `error` attribute is assumed to represent one standard deviation (1σ) of the underlying error distribution.
4.  **Units:** Basic unit tracking occurs for addition/subtraction (requires identical units) and simple unary operations. For multiplication, division, powers, and most functions, the resulting unit is cleared (`""`). Users must manage complex unit conversions manually.

### Calculations with Measurements

Standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) and many NumPy ufuncs (`np.sin`, `np.log`, `np.exp`, etc.) work directly with `Measurement` objects.

```python
speed = length / time
print(speed)
# Output ≈ 2.05 ± 0.09 cm/s (Unit logic basic, user verifies cm/s)
# Note: Guezzi clears the unit for division by default.

resistance = voltages / currents # Element-wise division for arrays
print(resistance)
# Output ≈ [5.5 ± 0.7, 5.0 ± 0.4, 5.3 ± 0.5] V/A (unit cleared)

angle = gz.Measurement(0.5, 0.01, unit='rad', name='Angle')
sin_angle = np.sin(angle) # Use NumPy ufuncs
print(sin_angle)
# Output ≈ 0.479 ± 0.009 (unit cleared - dimensionless)

# Comparisons use nominal values only:
m1 = gz.Measurement(5, 1)
m2 = gz.Measurement(5.1, 1)
print(m1 == m2)  # Output: False
print(m1 < m2)   # Output: True
# For compatibility check including errors, use gz.test_comp(m1, m2)
```

### Formatting Output

Use `to_eng_string()` for engineering notation with SI prefixes or `round_to_error()` for standard rounding based on error.

```python
val = gz.Measurement(12345, 67, unit='Hz')
print(val.to_eng_string(sig_figs_error=1)) # Output: 12.35 ± 0.07 kHz
print(val.to_eng_string(sig_figs_error=2)) # Output: 12.345 ± 0.067 kHz
print(val.round_to_error(sig_figs_error=1)) # Output: 12350 ± 70 Hz
print(val.round_to_error(sig_figs_error=2)) # Output: 12345 ± 67 Hz
```

## Curve Fitting (`fitting.py`)

The `perform_fit` function provides a unified interface for fitting data using standard techniques.

```python
import guezzi as gz
import numpy as np

# Example Data (Voltage vs Current for Ohm's Law V = R*I)
current = gz.Measurement([0.1, 0.2, 0.3, 0.4, 0.5], 0.01, unit='A', name='Current')
voltage = gz.Measurement([1.9, 4.1, 5.9, 8.2, 10.1], 0.2, unit='V', name='Voltage')

# Define the model function: f(x, param1, param2, ...)
def linear_model(i, resistance):
  """Ohm's Law V = R*I"""
  return resistance * i

# Perform the fit
# 'auto' method selects curve_fit (y-errors only) or ODR (x and y errors)
# Provide parameter names for clarity in results
fit_result = gz.perform_fit(
    x_data=current,
    y_data=voltage,
    func=linear_model,
    parameter_names=['Resistance']
    # p0=[10] # Optional: provide initial guess for Resistance
)

# Print the detailed FitResult object summary
print(fit_result)

# Access results:
params = fit_result.parameters # Dictionary: {'Resistance': Measurement(value=..., error=...)}
resistance_fit = params['Resistance']
print(f"Fitted Resistance: {resistance_fit.to_eng_string(2)}")

print(f"Reduced Chi-squared: {fit_result.reduced_chi_square:.3f}")
print(f"Covariance Matrix:\n{fit_result.covariance_matrix}")
```

### Choosing the Fit Method (`curve_fit` vs. `ODR`)

*   **`scipy.optimize.curve_fit` (Least Squares):**
    *   **Assumes:** Independent variable (`x_data`) is known exactly (no error). Minimizes the sum of squared residuals weighted by the dependent variable's uncertainties (`y_data.error`).
    *   **Use when:** X-errors are negligible compared to y-errors or the range of x-values. Faster and often sufficient.
    *   **Guezzi Usage:** `method='curve_fit'` or `method='auto'` if `x_data` has no errors.
*   **`scipy.odr` (Orthogonal Distance Regression):**
    *   **Assumes:** Both `x_data` and `y_data` have uncertainties. Minimizes the sum of squared *orthogonal* distances from data points to the fitted curve, considering both `x_data.error` and `y_data.error`.
    *   **Use when:** X-errors are significant. More complex but statistically more appropriate in this case.
    *   **Guezzi Usage:** `method='odr'` or `method='auto'` if `x_data` has errors.

### Interpreting `FitResult`

The `FitResult` object contains:

*   `parameters`: A dictionary where keys are parameter names and values are `Measurement` objects containing the best-fit value and its standard error.
*   `covariance_matrix`: Shows the variances (on diagonal) and covariances (off-diagonal) of the fitted parameters. Helps understand correlations between parameters.
*   `chi_square` (χ²): The weighted sum of squared residuals. Measures overall agreement between model and data.
*   `dof` (Degrees of Freedom): Number of data points used minus the number of fitted parameters.
*   `reduced_chi_square` (χ²/DoF): The χ² value normalized by DoF.
    *   **χ²/DoF ≈ 1:** A good fit, deviations are consistent with errors.
    *   **χ²/DoF >> 1:** Poor fit (bad model?), underestimated errors, or non-statistical noise.
    *   **χ²/DoF << 1:** Errors might be overestimated, or the model might be overfitting (less common).
*   `success`: Boolean indicating if the underlying optimization algorithm converged. *Does not guarantee a good or physically meaningful fit!* Always check χ²/DoF and visualize the fit and residuals.

## Statistical Tests (`stats.py`)

### Compatibility Test (`test_comp`)

Checks if two measurements (`m1`, `m2`) are statistically compatible (i.e., could they represent the same underlying true value?).

```python
m1 = gz.Measurement(10.0, 0.5, name='Exp A')
m2 = gz.Measurement(10.8, 0.4, name='Exp B')
m3 = gz.Measurement(10.1, 0.1, name='Exp C')

comp_AB = gz.test_comp(m1, m2, alpha=0.05) # Default alpha=5%
comp_AC = gz.test_comp(m1, m3)

print(f"Compatibility A vs B: {comp_AB['interpretation']}")
# Output ≈ Compatibility A vs B: Measurements are compatible at alpha=0.05 (assuming independence). (Z=1.25, p=0.212)

print(f"Compatibility A vs C: {comp_AC['interpretation']}")
# Output ≈ Compatibility A vs C: Measurements are NOT compatible at alpha=0.05 (assuming independence). (Z=1.96, p=0.0498)
```

*   **Theory:** Performs a Z-test on the difference `d = m1 - m2`. Calculates `Z = |d| / σ_d`. The p-value is the probability of observing such a difference (or larger) if `m1` and `m2` were truly measuring the same value. If `p-value < alpha`, the difference is considered statistically significant, and the measurements are deemed incompatible.
*   `alpha`: Significance level (risk of falsely rejecting compatibility).
*   `assume_correlated=True`: Use if `m1`, `m2` might be negatively correlated. Provides a more conservative test (less likely to find incompatibility). Assumes maximum negative correlation (error on difference = `σ1 + σ2`).

### Weighted Mean (`weighted_mean`)

Calculates the best estimate of a quantity from multiple independent measurements, weighting each by its inverse variance (`1 / error²`).

```python
results = [
    gz.Measurement(9.8, 0.2),
    gz.Measurement(9.9, 0.3),
    gz.Measurement(9.7, 0.1) # This one has smallest error, gets highest weight
]

avg = gz.weighted_mean(results)
print(f"Weighted Mean: {avg.to_eng_string(2)}")
# Output ≈ Weighted Mean: 9.73 ± 0.08
```

*   **Theory:** Minimizes the variance of the combined result. Assumes measurements are independent. Measurements with smaller errors contribute more to the weighted average.

## Generating Tables (`tables.py`)

Create LaTeX tables for data or fit results.

### Data Table (`latex_table_data`)

```python
# Using voltage, current from earlier example
latex_code_data = gz.latex_table_data(
    current, voltage,
    labels=['Current', 'Voltage'],
    orientation='v', # Vertical layout
    sig_figs_error=1,
    caption='Voltage vs Current Measurements'
)
print(latex_code_data)
# Output: LaTeX code for a table with Current and Voltage columns
```

### Fit Results Table (`latex_table_fit`)

```python
# Using fit_result from earlier example
latex_code_fit = gz.latex_table_fit(
    fit_result, # Can pass multiple FitResult objects
    fit_labels=['Ohmic Fit'], # Label for this fit column/row
    param_labels={'Resistance': '$R_{fit}$'}, # Use LaTeX math for parameter name
    stat_labels={'reduced_chi_square': '$\\chi^2/\\nu$'}, # Pretty label for stat
    sig_figs_error=2,
    caption='Linear Fit Results for Resistance'
)
print(latex_code_fit)
# Output: LaTeX code summarizing the fit parameters and stats
```

*   These functions format `Measurement` objects using `to_eng_string` and generate the necessary LaTeX boilerplate (`\begin{table}`, `\begin{tabular}`, etc.).
*   `pyperclip` library (optional) allows automatically copying the LaTeX code to the clipboard.

## Plotting (`plotting.py`)

Visualize data and fits using Matplotlib.

### Plotting Measurements (`plot_measurements`)

Plot data points with x and y error bars.

```python
# Using voltage, current from earlier example
ax = gz.plot_measurements(
    current, voltage, # Pass x, y pairs
    labels=['Measured Data'],
    xlabel='Current (A)', # Override default label
    ylabel='Voltage (V)', # Override default label
    title='Voltage vs Current',
    fmts='o', # Use circle markers
    markersize=6,
    capsize=4
    # save_path='voltage_vs_current.png' # Optional: save the plot
    # show_plot=True # Optional: display plot interactively
)
# Further customize the plot using the returned axes 'ax' if needed
# e.g., ax.set_xlim(0, 0.6)
plt.show() # Show plot if show_plot=False was used
```

### Plotting Fits (`plot_fit`)

Visualize the results of `perform_fit`.

```python
# Using fit_result from earlier example
ax_main, ax_res = gz.plot_fit(
    fit_result,
    plot_residuals=True, # Show residuals plot below main plot
    show_params=True,    # Annotate plot with fitted parameters
    show_stats=True,     # Annotate plot with chi^2/DoF
    data_label='Measurements',
    fit_label='Linear Fit (V=RI)',
    color='blue',
    # save_path='fit_plot.pdf' # Optional: save
    # show_plot=True           # Optional: display
)
plt.show() # Show plot if show_plot=False was used
```

*   `plot_fit` shows data points (respecting masks), the fitted curve, and optionally residuals (`y_data - y_fit`).
*   Residual plots are crucial for diagnosing fit problems (e.g., systematic trends indicate the model might be inadequate).
*   Annotations (`show_params`, `show_stats`) provide key results directly on the plot.

## Utilities (`utils.py`)

Contains internal helpers:

*   `_format_value_error_eng`: The core function for formatting `value ± error` strings with SI prefixes and correct significant figures (used by `Measurement.to_eng_string`).
*   `round_to_significant_figures`: Rounds a number to a specified number of significant figures.
*   `get_si_prefix`: Determines the appropriate SI prefix for a number.
*   `SI_PREFIXES`: Dictionary mapping exponents to SI prefix symbols.

## Examples

*(This section would ideally contain more complete, runnable examples demonstrating common workflows, like:*
*   *Reading data from a file.*
*   *Performing a fit with multiple parameters.*
*   *Comparing two different fits to the same data.*
*   *Combining results using weighted mean.*
*   *Generating a full report section with plots and tables.*)
