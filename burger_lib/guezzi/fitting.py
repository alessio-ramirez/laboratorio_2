# --- START OF FILE fitting.py ---
"""
Curve Fitting Functionality

Provides the `perform_fit` function to fit data using common models and
methods (scipy.optimize.curve_fit, scipy.odr), returning results in a
structured `FitResult` object.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
import inspect
from typing import Callable, Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import warnings

from .measurement import Measurement # Import Measurement class

@dataclass
class FitResult:
    """
    Holds the results of a curve fitting operation performed by `perform_fit`.

    This object bundles together the fitted parameters, their uncertainties,
    goodness-of-fit statistics, and references to the input data and function.

    Attributes:
        parameters (Dict[str, Measurement]): Dictionary mapping parameter names
            (str) to Measurement objects. Each Measurement holds the optimal
            parameter value and its standard error (uncertainty).
        covariance_matrix (Optional[np.ndarray]): The covariance matrix (P) of the
            fitted parameters. The diagonal elements `P[i, i]` are the variances
            (σ_i^2) of the parameters, and the off-diagonal elements `P[i, j]`
            are the covariances between parameters i and j. Useful for analyzing
            parameter correlations. Can be None if the fit failed or wasn't computed.
        chi_square (Optional[float]): The Chi-squared (χ²) statistic value for the fit.
            Calculated as Σ [ (y_i - f(x_i, params)) / σ_y_i ]^2 for curve_fit
            (or a related ODR sum of squares if x-errors are present). Measures
            the overall agreement between the data and the model, weighted by
            data uncertainties. Only meaningful if uncertainties (at least y_err)
            are provided and non-zero. Can be None if fit failed or stats
            not calculated.
        dof (Optional[int]): Degrees of Freedom for the fit. Typically calculated as
            (Number of data points used in fit) - (Number of fitted parameters).
            Represents the number of independent pieces of information available
            to estimate the goodness-of-fit after determining the parameters.
            Can be None if fit failed or not enough data points.
        reduced_chi_square (Optional[float]): The Reduced Chi-squared (χ²/DoF or χ²_ν).
            Calculated as `chi_square / dof`. This value normalizes the χ² by the
            degrees of freedom.
            Interpretation:
                - χ²/DoF ≈ 1: Indicates a good fit, where deviations between data
                  and model are consistent with the estimated data uncertainties.
                - χ²/DoF > 1: May suggest a poor model, underestimated errors, or
                  non-statistical fluctuations.
                - χ²/DoF < 1: May suggest overestimated errors or an over-fitted model
                  (though less common).
            Can be None if chi_square or dof is None, or if dof <= 0.
        function (Optional[Callable]): The model function used for fitting (e.g., `f(x, p1, p2)`).
        parameter_names (List[str]): Ordered list of parameter names corresponding
            to the function signature and the covariance matrix rows/columns.
        x_data (Optional[Measurement]): Reference to the input independent variable data
            (Measurement object) used for the fit. Includes original values and errors.
        y_data (Optional[Measurement]): Reference to the input dependent variable data
            (Measurement object) used for the fit. Includes original values and errors.
        method (str): The fitting method used ('curve_fit' or 'odr').
        mask (Optional[np.ndarray]): The boolean mask applied to the data before fitting
            (True = point used, False = point excluded). None if no mask was applied.
        success (bool): Flag indicating if the underlying fitting routine
            (curve_fit or ODR) reported success. Note that success doesn't guarantee
            a scientifically meaningful fit (check χ²/DoF and residuals).
    """
    parameters: Dict[str, Measurement] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None
    chi_square: Optional[float] = None
    dof: Optional[int] = None
    reduced_chi_square: Optional[float] = None
    function: Optional[Callable] = None
    parameter_names: List[str] = field(default_factory=list)
    x_data: Optional[Measurement] = None
    y_data: Optional[Measurement] = None
    method: str = ""
    mask: Optional[np.ndarray] = None
    success: bool = False

    def __str__(self) -> str:
        """Provides a concise, human-readable summary string of the fit result."""
        lines = [f"--- Fit Result ({self.method}) ---"]
        lines.append(f"  Success: {self.success}")
        if self.function:
             func_name = getattr(self.function, '__name__', 'anonymous')
             lines.append(f"  Function: {func_name}")

        lines.append("  Parameters:")
        if self.parameters:
            # Use a consistent sig fig setting for the summary print
            param_sig_figs = 2
            for name, param in self.parameters.items():
                 # Use Measurement's formatter for consistent output
                 lines.append(f"    {name}: {param.to_eng_string(sig_figs_error=param_sig_figs)}")
        else:
            lines.append("    (No parameters fitted or available)")

        if self.chi_square is not None and self.dof is not None and self.dof > 0:
             lines.append("  Goodness of Fit:")
             lines.append(f"    Chi²: {self.chi_square:.4g}")
             lines.append(f"    Degrees of Freedom (DoF): {self.dof}")
             if self.reduced_chi_square is not None:
                  lines.append(f"    Reduced Chi² (χ²/DoF): {self.reduced_chi_square:.4g}")
        elif self.chi_square is not None: # Case where DoF might be <= 0
             lines.append("  Goodness of Fit:")
             lines.append(f"    Chi²: {self.chi_square:.4g}")
             lines.append(f"    Degrees of Freedom (DoF): {self.dof if self.dof is not None else 'N/A'} (Reduced Chi² not computed)")


        if self.mask is not None:
             points_used = np.sum(self.mask)
             total_points = len(self.mask)
             lines.append(f"  Mask Applied: {points_used} / {total_points} data points used.")
        lines.append("--------------------------")
        return "\n".join(lines)

    def __repr__(self) -> str:
        # More technical representation, useful for debugging
        # Show subset of info to keep it manageable
        params_repr = {name: f"({p.value:.3g} ± {p.error:.2g})" for name, p in self.parameters.items()}
        return (f"FitResult(method='{self.method}', success={self.success}, "
                f"parameters={params_repr}, chi_square={self.chi_square:.4g}, "
                f"dof={self.dof}, function={getattr(self.function,'__name__','?')}, ...)")


def perform_fit(x_data: Union[Measurement, np.ndarray, List, Tuple],
                y_data: Union[Measurement, np.ndarray, List, Tuple],
                func: Callable,
                p0: Optional[Union[List[float], Tuple[float, ...]]] = None,
                parameter_names: Optional[List[str]] = None,
                method: str = 'auto',
                mask: Optional[np.ndarray] = None,
                calculate_stats: bool = True) -> FitResult:
    """
    Performs curve fitting using either `scipy.optimize.curve_fit` (Least Squares)
    or `scipy.odr` (Orthogonal Distance Regression).

    Choice of Method:
    - `curve_fit`: Standard least-squares fitting. It minimizes the sum of squared
      residuals weighted by the *y-uncertainties* (`sigma` argument). It assumes
      the x-data is known exactly (no x-uncertainty). Faster and simpler for many cases.
    - `odr`: Orthogonal Distance Regression. Minimizes the sum of squared orthogonal
      distances from the data points to the fitted curve, considering *both x and y
      uncertainties*. More appropriate when x-uncertainties are significant compared
      to the range of x-values or the scale of features in the model. Can be slower.
    - `auto`: Automatically selects 'odr' if the input `x_data` has non-zero
      uncertainties (specifically, if `np.any(x_data.error > 1e-15)`), otherwise
      defaults to 'curve_fit'. This is often a sensible default.

    Args:
        x_data: Independent variable data (horizontal axis). Can be:
            - A `Measurement` object (containing values and errors).
            - NumPy array, list, or tuple of values (errors assumed to be zero).
        y_data: Dependent variable data (vertical axis). Can be:
            - A `Measurement` object.
            - NumPy array, list, or tuple of values (errors assumed to be zero).
            Must have the same shape as x_data after broadcasting.
        func: The model function to fit. It must have the signature `f(x, p1, p2, ...)`
              where `x` is the independent variable (or array) and `p1, p2, ...` are
              the parameters to be fitted.
        p0: Initial guess for the parameters `[p1_guess, p2_guess, ...]`.
            If None, defaults to 1.0 for all parameters. Providing good initial
            guesses is often crucial for the fit to converge correctly.
        parameter_names: Optional list of strings providing names for the parameters
                         `['name1', 'name2', ...]`, in the same order as they appear
                         in the function signature. If None, names are inferred from the
                         function signature (e.g., 'p1', 'p2' if func is `f(x, p1, p2)`).
        method: Fitting method to use: 'auto', 'curve_fit', or 'odr'. (Default 'auto').
        mask: Optional boolean NumPy array of the same shape as x_data/y_data.
              If provided, only data points where the mask is `True` will be used
              in the fitting process. Points where the mask is `False` are ignored.
        calculate_stats: If True, calculate Chi-squared (χ²), Degrees of Freedom (DoF),
                         and Reduced Chi-squared (χ²/DoF) goodness-of-fit statistics.
                         Requires valid uncertainties in the data (at least y_err for
                         curve_fit) and sufficient data points (more points than parameters).
                         (Default True).

    Returns:
        FitResult: An object containing the results, including fitted parameters
                   (as Measurement objects with uncertainties), covariance matrix,
                   goodness-of-fit statistics, and references to input data/function.

    Raises:
        ValueError: If x_data and y_data have incompatible shapes, if the number
                    of provided parameter names or initial guesses doesn't match the
                    function signature, if the mask shape is incompatible, or if an
                    invalid method is specified.
        TypeError: If x_data or y_data cannot be converted to Measurement objects.
    """
    # --- Input Validation and Conversion to Measurement Objects ---
    if not isinstance(x_data, Measurement):
        try:
            # Create a basic Measurement, infer name if possible
            x_name = getattr(x_data, 'name', 'x') # Use existing name if array has one
            x_data = Measurement(values=x_data, name=x_name)
        except Exception as e:
            raise TypeError(f"Could not convert x_data to Measurement: {e}")
    if not isinstance(y_data, Measurement):
        try:
            y_name = getattr(y_data, 'name', 'y')
            y_data = Measurement(values=y_data, name=y_name)
        except Exception as e:
            raise TypeError(f"Could not convert y_data to Measurement: {e}")

    # Check for shape compatibility after conversion
    try:
        # Allow broadcasting (e.g., scalar against array), find common shape
        common_shape = np.broadcast(x_data.value, y_data.value).shape
        # Note: Measurement internally handles broadcasting of value/error
    except ValueError:
        raise ValueError(f"Shape mismatch: x_data ({x_data.shape}) and y_data ({y_data.shape}) "
                         "cannot be broadcast together.")

    # Extract NumPy arrays for fitting
    x_val = np.asarray(x_data.value)
    x_err = np.asarray(x_data.error)
    y_val = np.asarray(y_data.value)
    y_err = np.asarray(y_data.error)

    # --- Masking ---
    # Start with a mask that includes all points
    active_mask = np.ones_like(x_val, dtype=bool)
    original_mask_provided = mask is not None # Keep track if user provided a mask

    if original_mask_provided:
        mask_arr = np.asarray(mask, dtype=bool)
        # Check mask compatibility with broadcasted data shape
        try:
             mask_bcast = np.broadcast_to(mask_arr, common_shape)
             active_mask = mask_bcast
        except ValueError:
             raise ValueError(f"Provided mask shape {mask_arr.shape} is incompatible "
                              f"with broadcasted data shape {common_shape}")

    # Apply the active mask to get the data subset for fitting
    # Using boolean indexing creates copies, which is safe for fitting routines
    x_fit = x_val[active_mask]
    y_fit = y_val[active_mask]
    x_err_fit = x_err[active_mask]
    y_err_fit = y_err[active_mask]

    n_points = len(x_fit) # Number of points actually used in the fit

    if n_points == 0:
        warnings.warn("No data points selected after applying the mask. Fit cannot be performed.", RuntimeWarning)
        # Return a FitResult indicating failure
        return FitResult(
            x_data=x_data, y_data=y_data, function=func, method="N/A (no data)",
            mask=mask if original_mask_provided else None, success=False
        )

    # --- Parameter Setup ---
    try:
        sig = inspect.signature(func)
        func_param_names = list(sig.parameters.keys())
        # Expecting func(x, p1, p2, ...)
        if len(func_param_names) < 2:
            raise ValueError("Fit function must accept at least two arguments: "
                             "the independent variable (x) and one parameter.")
        # Assume the first argument is the independent variable 'x'
        actual_param_names_from_func = func_param_names[1:]
        n_params = len(actual_param_names_from_func)
    except Exception as e:
         raise ValueError(f"Could not inspect function signature for '{getattr(func,'__name__','anonymous')}': {e}")

    # Determine parameter names to use
    if parameter_names is None:
        parameter_names = actual_param_names_from_func # Use names from function signature
    elif len(parameter_names) != n_params:
        raise ValueError(f"Number of provided parameter_names ({len(parameter_names)}) "
                         f"does not match the number of parameters expected by the function "
                         f"'{getattr(func,'__name__','anonymous')}' ({n_params}).")

    # Set up initial parameter guesses
    if p0 is None:
        p0 = [1.0] * n_params # Default guess
        warnings.warn(f"No initial guess (p0) provided. Using default guess {[1.0]*n_params} "
                      f"for parameters {parameter_names}. Fit convergence may be poor. "
                      "Providing good initial guesses is strongly recommended.", UserWarning)
    elif len(p0) != n_params:
        raise ValueError(f"Number of initial guesses in p0 ({len(p0)}) does not match "
                         f"the number of parameters expected by the function ({n_params}).")

    # --- Method Selection ---
    # Use ODR if x_data has significant errors, otherwise use curve_fit
    # Use a small tolerance to check for non-zero errors
    has_significant_x_errors = np.any(x_err_fit > 1e-15 * np.abs(x_fit)) or np.any(x_err_fit > 1e-9) # Relative or absolute check

    if method == 'auto':
        use_odr = has_significant_x_errors
        fit_method_name = 'odr' if use_odr else 'curve_fit'
        if use_odr:
            print("Info: Significant X errors detected. Using ODR method for fitting.") # Inform user
        else:
             print("Info: No significant X errors detected. Using curve_fit method.")
    elif method == 'curve_fit':
        use_odr = False
        fit_method_name = 'curve_fit'
        if has_significant_x_errors:
            warnings.warn("Method forced to 'curve_fit', but significant X errors were detected in the data. "
                          "These X errors will be ignored by curve_fit. Consider using 'odr' or 'auto'.", UserWarning)
    elif method == 'odr':
        use_odr = True
        fit_method_name = 'odr'
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'auto', 'curve_fit', or 'odr'")

    # --- Perform the Fit ---
    params_opt: np.ndarray = np.full(n_params, np.nan) # Initialize with NaN
    params_std_err: np.ndarray = np.full(n_params, np.nan) # Initialize with NaN
    cov_matrix: Optional[np.ndarray] = None
    fit_success = False
    odr_output = None # To store ODR output if used

    try:
        if use_odr:
            # Define the function wrapper for ODR: f(beta, x) where beta are parameters
            def odr_func_wrapper(beta, x):
                try:
                    return func(x, *beta)
                except Exception as e:
                     # ODR can be sensitive to errors inside the function
                     warnings.warn(f"Error during ODR function evaluation: {e}", RuntimeWarning)
                     # Return NaN or raise to indicate failure within ODR step?
                     # Returning NaN might allow ODR to sometimes recover or fail gracefully.
                     return np.full_like(x, np.nan) if isinstance(x, np.ndarray) else np.nan


            odr_model = Model(odr_func_wrapper)

            # Create RealData object, providing errors only if they are non-zero/significant
            # ODR handles sx=None or sy=None gracefully. Using small numbers can sometimes cause issues.
            sx = x_err_fit if np.any(x_err_fit > 1e-15) else None
            sy = y_err_fit if np.any(y_err_fit > 1e-15) else None

            # Check if any errors are zero or very small - ODR might treat them as exact
            if sx is None: warnings.warn("ODR: X errors appear negligible or zero, treating x as exact.", UserWarning)
            if sy is None: warnings.warn("ODR: Y errors appear negligible or zero, treating y as exact.", UserWarning)

            data = RealData(x_fit, y_fit, sx=sx, sy=sy)

            # Configure and run ODR
            # Set maxit to a reasonable number, e.g., 1000
            odr = ODR(data, odr_model, beta0=p0, maxit=1000)
            # ODR convergence can be sensitive, consider different job settings if needed
            # e.g., job=10 for explicit derivatives if provided, default is 0.
            output = odr.run()
            odr_output = output # Store for potential chi2 calculation later

            # Check ODR success code. See ODRPACK guide p.38.
            # Codes 0, 1, 2, 3 usually indicate success or acceptable termination.
            if output.info <= 3:
                 params_opt = output.beta        # Optimal parameters
                 params_std_err = output.sd_beta # Standard errors of parameters
                 # ODR covariance matrix (cov_beta) needs scaling by residual variance (res_var)
                 # if the fit was not implicitly scaled (which is the default if errors provided)
                 # According to scipy docs: "The covariance matrix is res_var * cov_beta."
                 # This seems correct regardless of whether errors were provided.
                 cov_matrix = output.cov_beta * output.res_var
                 fit_success = True
            else:
                 # Provide more context from ODR output if possible
                 reason = output.stopreason
                 warnings.warn(f"ODR fitting routine indicated potential issues. "
                               f"Stop reason: '{reason}'. ODR info code: {output.info}. "
                               "Results may be unreliable.", RuntimeWarning)
                 # Still store results, but mark success as False
                 params_opt = output.beta
                 params_std_err = output.sd_beta
                 cov_matrix = output.cov_beta * output.res_var if output.cov_beta is not None else None
                 fit_success = False # Mark as failed despite potential parameter output

        else: # Use curve_fit
            # curve_fit requires sigma (y errors).
            # If y_err_fit contains zeros, curve_fit interprets them as infinite weights,
            # forcing the curve through those points exactly. This is usually desired.
            # If *all* errors are zero, absolute_sigma=False (default) scales the
            # cov matrix using residual variance. If *any* error is non-zero,
            # absolute_sigma=True should be used to treat sigma as actual std deviations.
            sigma_y = y_err_fit
            # Use absolute_sigma=True if we have actual error estimates
            use_absolute_sigma = np.any(sigma_y > 1e-15)

            if not use_absolute_sigma:
                 warnings.warn("curve_fit: No significant Y errors provided (or all are zero). "
                               "Uncertainties on fitted parameters will be scaled by residual variance "
                               "(absolute_sigma=False). These uncertainties might not reflect the true "
                               "parameter uncertainty if the model is imperfect.", UserWarning)
                 # Use None for sigma if all are effectively zero, let curve_fit estimate variance
                 sigma_y = None if np.all(np.isclose(sigma_y, 0.0)) else sigma_y


            # Perform the fit
            popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0,
                                   sigma=sigma_y, absolute_sigma=use_absolute_sigma,
                                   maxfev=5000) # Increase max function evaluations

            params_opt = popt
            # Parameter uncertainties are the sqrt of the diagonal elements of the covariance matrix
            params_std_err = np.sqrt(np.diag(pcov))
            cov_matrix = pcov
            fit_success = True # curve_fit raises an exception on failure

    except RuntimeError as e:
        # Catch specific curve_fit/ODR runtime errors (e.g., max iterations)
        warnings.warn(f"Fitting failed: Runtime error during optimization. "
                      f"Could not find optimal parameters. Check initial guess (p0) "
                      f"and function behavior. Error: {e}", RuntimeWarning)
        # params_opt, params_std_err remain NaN, cov_matrix is None
        fit_success = False
    except Exception as e:
        # Catch other potential errors during fitting
        warnings.warn(f"An unexpected error occurred during the fitting process: {e}", RuntimeWarning)
        # params_opt, params_std_err remain NaN, cov_matrix is None
        fit_success = False


    # --- Package Results into FitResult Object ---
    fit_params_dict = {}
    for i, name in enumerate(parameter_names):
        # Store each parameter as a Measurement object, using name for clarity
        fit_params_dict[name] = Measurement(params_opt[i], params_std_err[i], name=name)

    result = FitResult(
        parameters=fit_params_dict,
        covariance_matrix=cov_matrix,
        function=func,
        parameter_names=parameter_names,
        x_data=x_data, # Store reference to original Measurement object
        y_data=y_data, # Store reference to original Measurement object
        method=fit_method_name,
        mask=mask if original_mask_provided else None, # Store original mask if one was passed
        success=fit_success
    )

    # --- Calculate Goodness-of-Fit Statistics (Optional) ---
    # Requires a successful fit and enough data points for DoF > 0
    if calculate_stats and fit_success and n_points > n_params:
        # Calculate predicted y values using the fitted parameters
        y_pred = func(x_fit, *params_opt)

        # Calculate Degrees of Freedom (DoF)
        result.dof = n_points - n_params

        # Calculate Chi-squared (χ²)
        chi2_val = None
        if use_odr:
            # For ODR, the relevant sum of squares is directly available in output.sum_square
            # This value accounts for both x and y errors if provided.
            if odr_output is not None:
                 chi2_val = odr_output.sum_square # This is the weighted sum of squared residuals
            else:
                 warnings.warn("Cannot calculate Chi²: ODR output not available.", RuntimeWarning)

        else: # curve_fit case
            # Chi² calculation depends on whether y-errors were provided
            if sigma_y is not None and np.any(sigma_y > 1e-15):
                # Use provided y-errors for weighting (standard Chi² formula)
                # Avoid division by zero if any sigma_y were exactly zero (though unlikely if use_absolute_sigma was True)
                sigma_y_safe = np.where(np.isclose(sigma_y, 0.0), 1.0, sigma_y) # Avoid dividing by zero, use 1 if error was 0
                residuals = (y_fit - y_pred) / sigma_y_safe
                chi2_val = np.sum(residuals**2)
            else:
                # If no y-errors were provided (sigma_y=None for curve_fit), a meaningful Chi²
                # based on data uncertainties cannot be calculated. curve_fit effectively minimized
                # the unweighted sum of squares. We avoid calculating a potentially misleading Chi².
                 warnings.warn("Cannot calculate meaningful Chi²: No significant Y errors were provided to curve_fit.", UserWarning)
                 chi2_val = None # Chi-squared is not statistically meaningful here

        result.chi_square = chi2_val

        # Calculate Reduced Chi-squared (χ²/DoF) if possible and meaningful
        if chi2_val is not None and result.dof > 0:
            result.reduced_chi_square = chi2_val / result.dof
        elif chi2_val is not None and result.dof <= 0:
            warnings.warn("Degrees of Freedom (DoF) is zero or negative. Reduced Chi² cannot be calculated.", RuntimeWarning)

    elif calculate_stats and (not fit_success or n_points <= n_params):
         reason = "Fit failed" if not fit_success else f"Insufficient data points (N={n_points}, Params={n_params})"
         warnings.warn(f"Cannot calculate fit statistics (Chi², DoF): {reason}.", RuntimeWarning)

    return result

# --- END OF FILE fitting.py ---