# --- START OF FILE fitting.py ---
"""
Curve Fitting Functionality

Provides the `perform_fit` function to fit data using common models and
methods (scipy.optimize.curve_fit, scipy.odr, iminuit), returning results in a
structured `FitResult` object.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
import inspect
from typing import Callable, Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import warnings
import sys # For float max

# Assuming measurement.py contains the Measurement class as before
# If measurement.py is in the same directory, use:
# from measurement import Measurement
# If it's a sub-module of a package, use:
try:
    from .measurement import Measurement
except ImportError:
    # Fallback if running as a script or structure differs
    print("Warning: Could not import Measurement using relative path '.measurement'. Trying direct import.")
    try:
        from measurement import Measurement
    except ImportError:
        raise ImportError("Could not find the Measurement class. Ensure measurement.py is accessible.")


# Optional dependency: iminuit
try:
    from iminuit import Minuit
    from iminuit.cost import LeastSquares # Import LeastSquares
    _has_iminuit = True
except ImportError:
    _has_iminuit = False
    # Define dummy classes/functions if iminuit is not available to avoid NameErrors later
    # But the code will raise an ImportError if 'minuit' method is actually chosen.
    class Minuit: pass
    class LeastSquares: pass


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
            Calculated as Σ [ (y_i - f(x_i, params)) / σ_y_i ]^2 for curve_fit/minuit
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
        method (str): The fitting method used ('curve_fit', 'odr', or 'minuit').
        mask (Optional[np.ndarray]): The boolean mask applied to the data before fitting
            (True = point used, False = point excluded). None if no mask was applied.
        success (bool): Flag indicating if the underlying fitting routine
            (curve_fit, ODR, or Minuit) reported success/validity. Note that success
            doesn't guarantee a scientifically meaningful fit (check χ²/DoF and residuals).
        fit_object (Optional[Any]): Reference to the underlying fit object if available
            (e.g., the `iminuit.Minuit` instance or `scipy.odr.Output` instance)
            for advanced inspection.
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
    fit_object: Optional[Any] = None

    def __str__(self) -> str:
        """Provides a concise, human-readable summary string of the fit result."""
        lines = [f"--- Fit Result ({self.method}) ---"]
        lines.append(f"  Success: {self.success}")
        if self.function:
             func_name = getattr(self.function, '__name__', 'anonymous')
             lines.append(f"  Function: {func_name}")

        lines.append("  Parameters:")
        if self.parameters:
            param_sig_figs = 2 # Use consistent sig figs for summary print
            for name in self.parameter_names: # Print in consistent order
                if name in self.parameters:
                     param = self.parameters[name]
                     # Handle potential NaN values gracefully in output
                     if np.isnan(param.value) or np.isnan(param.error):
                         lines.append(f"    {name}: NaN ± NaN")
                     else:
                         lines.append(f"    {name}: {param.to_eng_string(sig_figs_error=param_sig_figs)}")
                else:
                     lines.append(f"    {name}: (Parameter not found in results?)") # Should not happen
        else:
            lines.append("    (No parameters fitted or available)")

        if self.covariance_matrix is not None and np.all(np.isfinite(self.covariance_matrix)):
             lines.append("  Covariance Matrix: Provided")
        elif self.covariance_matrix is not None:
             lines.append("  Covariance Matrix: Provided (may contain NaN/inf)")
        else:
             lines.append("  Covariance Matrix: None")

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
        # Check if fit_object is Minuit and show fval if chi2 wasn't calculated formally but fval exists
        elif self.method == 'minuit' and self.fit_object is not None and hasattr(self.fit_object, 'fval'):
            lines.append("  Goodness of Fit:")
            lines.append(f"    Minuit FCN Value (χ²): {self.fit_object.fval:.4g}") # fval IS chi2 when using LeastSquares
            lines.append(f"    Degrees of Freedom (DoF): {self.dof if self.dof is not None else 'N/A'}")


        if self.mask is not None:
             points_used = np.sum(self.mask) # True=1, False=0
             total_points = len(self.mask)
             lines.append(f"  Mask Applied: {points_used} / {total_points} data points used.")
        lines.append("--------------------------")
        return "\n".join(lines)

    def __repr__(self) -> str:
        # More technical representation, useful for debugging
        params_repr = {name: f"({p.value:.3g} ± {p.error:.2g})" for name, p in self.parameters.items()}
        chi2_str = f"{self.chi_square:.4g}" if self.chi_square is not None else 'None'
        return (f"FitResult(method='{self.method}', success={self.success}, "
                f"parameters={params_repr}, chi_square={chi2_str}, "
                f"dof={self.dof}, function={getattr(self.function,'__name__','?!')}, ...)")


def perform_fit(x_data: Union[Measurement, np.ndarray, List, Tuple],
                y_data: Union[Measurement, np.ndarray, List, Tuple],
                func: Callable,
                p0: Optional[Union[List[float], Tuple[float, ...]]] = None, # Changed: Always list/tuple
                parameter_names: Optional[List[str]] = None,
                method: str = 'auto',
                mask: Optional[np.ndarray] = None,
                calculate_stats: bool = True,
                minuit_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
                **kwargs) -> FitResult:
    """
    Performs curve fitting using `scipy.optimize.curve_fit` (Least Squares),
    `scipy.odr` (Orthogonal Distance Regression), or `iminuit.Minuit` (robust minimization).

    Choice of Method:
    - `curve_fit`: Standard least-squares fitting. Minimizes sum of squared residuals
      weighted by *y-uncertainties*. Assumes x-data is exact. Good for simple cases
      with negligible x-error.
    - `odr`: Orthogonal Distance Regression. Minimizes orthogonal distances considering
      *both x and y uncertainties*. Best when x-uncertainties are significant.
    - `minuit`: Uses the MINUIT2 algorithm (via `iminuit`) to minimize a cost function.
      Uses `iminuit.cost.LeastSquares` by default, equivalent to a Chi-squared cost
      function `sum(((y - model)/y_err)**2)`, making it a robust alternative to
      `curve_fit`, often better with complex functions or poor initial guesses.
      Requires y-errors for weighting; ignores x-errors in this mode.
    - `auto`: Selects 'odr' if `x_data` has significant errors (relative error > 0.5%),
      otherwise defaults to 'curve_fit' (or 'minuit' if iminuit is installed).

    Args:
        x_data: Independent variable data. Measurement, array, list, or tuple.
        y_data: Dependent variable data. Measurement, array, list, or tuple.
        func: Model function `f(x, p1, p2, ...)` or `f(x, **params)`. The order
              of parameters must match `p0` and `parameter_names`.
        p0: Initial guess for parameters as a list or tuple `[p1_guess, p2_guess, ...]`.
            The order must match the function signature (after `x`) and the optional
            `parameter_names`. If None, defaults to 1.0 for all parameters.
            Providing good guesses is highly recommended, especially for non-linear models.
        parameter_names: List `['name1', 'name2', ...]`. Order must match function
                         signature (excluding x) and `p0`. If None, inferred from the
                         function signature using `inspect`. Required if the function
                         signature cannot be introspected easily (e.g., lambda, partial).
        method: Fitting method: 'auto', 'curve_fit', 'odr', 'minuit'. (Default 'auto').
        mask: Optional boolean NumPy array to select data points (True = use).
        calculate_stats: If True, calculate Chi², DoF, Reduced Chi². Requires valid
                         uncertainties (at least y_err for curve_fit/minuit) and N > n_params.
                         (Default True).
        minuit_limits (Dict): Optional parameter limits for Minuit, e.g.,
                              `{'param_name': (lower_bound, upper_bound)}`. Use None for
                              no limit on one side (e.g., `(0, None)`). Assumes parameter
                              names in the dict are correct. (Default None).
        **kwargs: Additional keyword arguments passed directly to the underlying
                  fitting routine (`curve_fit`, `ODR.run`, `Minuit.migrad`). Common
                  options include `maxfev` (max function evaluations for curve_fit),
                  `maxit` (max iterations for ODR), `ncall` (max function calls for Minuit).

    Returns:
        FitResult: Object containing fitted parameters, stats, etc.

    Raises:
        ValueError: Incompatible inputs, shapes, names, or invalid method.
        TypeError: Inputs not convertible to Measurement objects or invalid p0 type.
        ImportError: If `method='minuit'` is chosen but `iminuit` is not installed.

    Notes on Difficult Fits:
        - If the fit fails or `success=False`, check the initial guess `p0`. Visualizing
          the model with `p0` against the data can help.
        - For `minuit`, use `minuit_limits` to constrain parameters to physically
          reasonable ranges.
        - Ensure the model function `func` is correctly defined and behaves as expected.
        - Consider if the chosen `method` is appropriate for the data's error structure.
    """
    # --- Input Validation and Conversion ---
    if not isinstance(x_data, Measurement):
        try: x_data = Measurement(values=x_data, name=getattr(x_data, 'name', 'x'))
        except Exception as e: raise TypeError(f"Could not convert x_data to Measurement: {e}")
    if not isinstance(y_data, Measurement):
        try: y_data = Measurement(values=y_data, name=getattr(y_data, 'name', 'y'))
        except Exception as e: raise TypeError(f"Could not convert y_data to Measurement: {e}")

    try:
        common_shape = np.broadcast(x_data.value, y_data.value).shape
    except ValueError:
        raise ValueError(f"Shape mismatch: x_data ({x_data.shape}) and y_data ({y_data.shape}) "
                         "cannot be broadcast together.")

    x_val = np.asarray(x_data.value).flatten() # Ensure 1D for simplicity here
    x_err = np.asarray(x_data.error).flatten()
    y_val = np.asarray(y_data.value).flatten()
    y_err = np.asarray(y_data.error).flatten()

    # Ensure data is broadcast to common shape *before* flattening if needed
    # This part might need adjustment if multi-dimensional fits are intended
    if x_val.shape != common_shape: x_val = np.broadcast_to(x_val, common_shape).flatten()
    if x_err.shape != common_shape: x_err = np.broadcast_to(x_err, common_shape).flatten()
    if y_val.shape != common_shape: y_val = np.broadcast_to(y_val, common_shape).flatten()
    if y_err.shape != common_shape: y_err = np.broadcast_to(y_err, common_shape).flatten()


    # --- Masking ---
    active_mask = np.ones_like(x_val, dtype=bool)
    original_mask_provided = mask is not None
    if original_mask_provided:
        mask_arr = np.asarray(mask, dtype=bool).flatten()
        # Ensure mask can be broadcast or matches the flattened data shape
        if mask_arr.size == 1:
             active_mask[:] = mask_arr # Broadcast single value
        elif mask_arr.shape == active_mask.shape:
             active_mask = mask_arr
        else:
             raise ValueError(f"Mask shape {mask_arr.shape} incompatible with flattened data shape {active_mask.shape}")

    # Apply mask *after* ensuring consistent shapes
    x_fit = x_val[active_mask]
    y_fit = y_val[active_mask]
    x_err_fit = x_err[active_mask]
    y_err_fit = y_err[active_mask]

    n_points = len(x_fit)
    if n_points == 0:
        warnings.warn("No data points after masking. Fit cannot proceed.", RuntimeWarning)
        return FitResult(x_data=x_data, y_data=y_data, function=func, method="N/A (no data)",
                         mask=mask if original_mask_provided else None, success=False)

    # --- Parameter Setup ---
    try:
        sig = inspect.signature(func)
        func_param_names_all = list(sig.parameters.keys())
        if len(func_param_names_all) < 2:
            raise ValueError("Fit function needs >= 2 args: independent var (x) and >= 1 parameter.")
        actual_param_names_from_func = func_param_names_all[1:]
        n_params = len(actual_param_names_from_func)
    except Exception as e:
         raise ValueError(f"Could not inspect function signature for '{getattr(func,'__name__','anonymous')}': {e}")

    # Determine final parameter names
    if parameter_names is None:
        parameter_names = actual_param_names_from_func # Use names from function
    elif len(parameter_names) != n_params:
        raise ValueError(f"Provided parameter_names count ({len(parameter_names)}) != function parameters ({n_params}). "
                         f"Function expects: {actual_param_names_from_func}")
    # Now parameter_names holds the definitive list of names in order

    # --- Method Selection ---
    if method == 'minuit' and not _has_iminuit:
        raise ImportError("Method 'minuit' selected, but iminuit is not installed. "
                          "Please install it (`pip install iminuit`).")


    has_significant_x_errors = np.any(x_err_fit/x_fit > 0.005) # Relative error > 0.5%
    has_any_y_errors = np.any(y_err_fit > 0) # Check if *any* y-errors are provided
    y_errors_all_zero = not has_any_y_errors # True if all y_err are zero or effectively zero

    resolved_method = method
    if method == 'auto':
        if _has_iminuit: # Prefer minuit if available and no significant X errors
            resolved_method = 'odr' if has_significant_x_errors else 'minuit'
        else: # Fallback if iminuit is not installed
             resolved_method = 'odr' if has_significant_x_errors else 'curve_fit'
        print(f"Info: Method 'auto' resolved to '{resolved_method}'.") # Simplified auto-resolve logic slightly
    elif method == 'curve_fit':
        if has_significant_x_errors:
            warnings.warn("Method 'curve_fit' selected, but significant X errors detected. "
                          "X errors will be ignored. Consider 'odr' or 'auto'.", UserWarning)
        if y_errors_all_zero:
             warnings.warn("curve_fit: No Y errors provided (or all zero). "
                           "Fit will be unweighted. Parameter errors will be scaled by residuals "
                           "and may not be statistically meaningful. Chi² cannot be calculated.", UserWarning)
    elif method == 'odr':
        if not has_significant_x_errors and not y_errors_all_zero:
            warnings.warn("Method 'odr' selected, but X errors appear negligible. "
                          "'curve_fit' or 'minuit' might be more appropriate if only Y errors are present.", UserWarning)
        if y_errors_all_zero and not has_significant_x_errors:
             warnings.warn("ODR: Neither significant X nor Y errors provided. Fit will be unweighted.", UserWarning)

    elif method == 'minuit':
        if has_significant_x_errors:
             warnings.warn("Method 'minuit' (with LeastSquares cost) selected, but significant X errors detected. "
                           "This cost function ignores X errors. Results might be biased. Consider 'odr'.", UserWarning)
        if y_errors_all_zero:
             # This is fatal for LeastSquares, raise error before attempting fit
             raise ValueError("Method 'minuit' (with LeastSquares cost) requires non-zero y-errors for weighting. All provided y_err are zero or errors are missing.")
        elif np.any(np.isclose(y_err_fit, 0.0)):
             # Non-fatal, but good to know
             warnings.warn("Minuit/LeastSquares: Found zero values in y_err_fit. "
                           "These points will have infinite weight.", UserWarning)

    elif method not in ['curve_fit', 'odr']:
        raise ValueError(f"Invalid method '{method}'. Must be 'auto', 'curve_fit', 'odr', or 'minuit'")

    fit_method_name = resolved_method

    # --- Initial Guess Setup (Unified) ---
    p0_list: List[float] = []
    if p0 is None:
        p0_list = [1.0] * n_params
        warnings.warn(f"No initial guess (p0) provided. Using default guess {p0_list} "
                      f"for parameters {parameter_names}. Fit convergence may be poor.", UserWarning)
    elif isinstance(p0, (list, tuple)):
        if len(p0) != n_params:
            raise ValueError(f"p0 list/tuple length ({len(p0)}) != number of parameters ({n_params}). "
                             f"Expected {n_params} guesses for parameters: {parameter_names}.")
        p0_list = list(p0)
    else:
        raise TypeError("p0 must be a list or tuple of initial parameter guesses "
                        f"(or None). Received type: {type(p0)}.")

    # --- Perform the Fit ---
    params_opt: np.ndarray = np.full(n_params, np.nan)
    params_std_err: np.ndarray = np.full(n_params, np.nan)
    cov_matrix: Optional[np.ndarray] = None
    fit_success = False
    fit_extra_info = None # Store Minuit object or ODR output
    m: Optional[Minuit] = None # Explicitly define m for Minuit case

    try:
        if fit_method_name == 'odr':
            # Check if ODR is viable
            if y_errors_all_zero and not has_significant_x_errors:
                 warnings.warn("ODR selected but no errors provided for weighting. Performing unweighted fit.", UserWarning)
                 sx = None
                 sy = None
            else:
                 sx = None if np.all(x_err_fit==0) else x_err_fit # ODR handles None
                 sy = y_err_fit if has_any_y_errors else None

            def odr_func_wrapper(beta, x): # beta = params
                try: return func(x, *beta)
                except Exception as e:
                     return np.full_like(x, np.nan) if isinstance(x, np.ndarray) else np.nan

            odr_model = Model(odr_func_wrapper)
            data = RealData(x_fit, y_fit, sx=sx, sy=sy)
            # Sensible default for maxit, allow override via kwargs
            odr_instance = ODR(data, odr_model, beta0=p0_list, maxit=kwargs.pop('maxit', 1000), **kwargs)
            output = odr_instance.run()
            fit_extra_info = output

            # ODR success codes: 0, 1, 2, 3 indicate some level of success/convergence
            if output.info in [0, 1, 2, 3]:
                 params_opt = output.beta
                 params_std_err = output.sd_beta
                 if output.cov_beta is not None:
                      # ODR scales cov_beta by res_var ONLY if fit is implicitly scaled (no errors given)
                      # If errors ARE given (sx or sy not None), cov_beta is the correct covariance matrix.
                      # If errors are NOT given, then cov_beta must be multiplied by res_var.
                      # Scipy ODR docs are a bit confusing here, but common practice (and curve_fit comparison)
                      # suggests this interpretation.
                      if sx is not None or sy is not None: # Weighted fit
                           cov_matrix = output.cov_beta
                           # Check if res_var is reasonable, ODR sets it to 1 if weighted?
                           # if not np.isclose(output.res_var, 1.0/(n_points - n_params)) and not np.isclose(output.res_var, 1.0):
                           #      warnings.warn(f"ODR weighted fit has res_var={output.res_var:.3g}. Check documentation for covariance interpretation.", RuntimeWarning)
                      else: # Unweighted fit, variance is estimated from residuals
                           cov_matrix = output.cov_beta * output.res_var
                 else:
                      warnings.warn("ODR fit successful but covariance matrix (cov_beta) is None.", RuntimeWarning)
                 fit_success = True
                 if output.info > 0:
                      warnings.warn(f"ODR completed but with potential issues. Stop reason: '{output.stopreason}'. Info code: {output.info}.", RuntimeWarning)
            else:
                 warnings.warn(f"ODR fitting failed. Stop reason: '{output.stopreason}'. Info code: {output.info}.", RuntimeWarning)
                 params_opt = output.beta if hasattr(output, 'beta') else params_opt
                 params_std_err = output.sd_beta if hasattr(output, 'sd_beta') else params_std_err
                 fit_success = False

        # --- Modified Minuit Section ---
        elif fit_method_name == 'minuit':
            # Ensure y_errors_all_zero was checked earlier and raised ValueError if true

            # Instantiate Cost Function
            try:
                # LeastSquares requires y_err > 0. Already checked for all zeros.
                cost_func = LeastSquares(x_fit, y_fit, y_err_fit, func)
            except Exception as e:
                 raise ValueError(f"Failed to initialize iminuit.cost.LeastSquares: {e}. Check function signature and data.")

            # Create initial parameters dictionary for Minuit
            p0_dict = dict(zip(parameter_names, p0_list))

            # Instantiate Minuit
            m = Minuit(cost_func, **p0_dict)
            fit_extra_info = m # Store Minuit object

            # Apply limits if provided (simplified)
            if minuit_limits:
                try:
                    for name, limits in minuit_limits.items():
                        if name in m.parameters: # Basic check
                            m.limits[name] = limits
                        else:
                            # Warn once if a name is wrong, but don't stop
                            warnings.warn(f"Parameter name '{name}' in minuit_limits not found in function parameters.", UserWarning)
                except Exception as e:
                     warnings.warn(f"Error applying Minuit limits: {e}", RuntimeWarning)


            # Run Migrad minimizer
            m.migrad(**kwargs) # Pass extra kwargs like ncall

            # Run Hesse for accurate errors if Migrad finished
            if m.valid:
                try:
                    m.hesse() # Calculate Hessian errors
                except RuntimeError as e:
                    warnings.warn(f"Minuit HESSE calculation failed: {e}. Errors from MIGRAD might be less accurate.", RuntimeWarning)

            # Check final validity
            fit_success = m.valid and m.fmin.is_valid # Check if function minimum is valid

            if fit_success:
                # Get values/errors in the order defined by parameter_names
                params_opt = np.array([m.values[name] for name in parameter_names])
                params_std_err = np.array([m.errors[name] for name in parameter_names])

                # --- CORRECTED COVARIANCE EXTRACTION ---
                if m.covariance is not None:
                    # Create numpy array and populate using indices
                    cov_matrix = np.zeros((n_params, n_params))
                    param_indices = {name: i for i, name in enumerate(parameter_names)}
                    for p1_name in parameter_names:
                        for p2_name in parameter_names:
                            idx1 = param_indices[p1_name]
                            idx2 = param_indices[p2_name]
                            # Access covariance matrix elements using indices or names
                            # Note: m.covariance indexing might depend slightly on iminuit version
                            # Trying direct tuple indexing first, common in newer versions
                            try:
                                cov_matrix[idx1, idx2] = m.covariance[idx1, idx2]
                            except (TypeError, IndexError):
                                # Fallback if tuple indexing fails (maybe older version expects names?)
                                try:
                                    cov_matrix[idx1, idx2] = m.covariance[p1_name, p2_name]
                                except Exception as e:
                                    warnings.warn(f"Could not access covariance element for ({p1_name}, {p2_name}): {e}. Covariance matrix might be incomplete.", RuntimeWarning)
                                    cov_matrix[idx1, idx2] = np.nan # Mark as unavailable

                else:
                    warnings.warn("Minuit fit valid, but covariance matrix could not be obtained (is None).", RuntimeWarning)
                    # Leave cov_matrix as None
                # --- END CORRECTION ---

            else:
                 warnings.warn(f"Minuit optimization failed or did not converge properly. "
                               f"(valid={m.valid}, fmin.is_valid={m.fmin.is_valid}). "
                               f"Check results carefully.", RuntimeWarning)
                 # Attempt to grab parameters even on failure
                 try:
                     params_opt = np.array([m.values[name] for name in parameter_names])
                     params_std_err = np.array([m.errors[name] for name in parameter_names])
                 except Exception:
                     pass # Keep NaNs if extraction fails
                 # Leave cov_matrix as None if fit failed
        # --- End of Modified Minuit Section ---

        else: # curve_fit
            sigma_y = y_err_fit if has_any_y_errors else None
            use_absolute_sigma = has_any_y_errors # Use absolute if errors were provided

            if not use_absolute_sigma:
                 warnings.warn("curve_fit: No Y errors provided. Using absolute_sigma=False. "
                               "Parameter errors will be scaled by residuals. "
                               "Chi² cannot be calculated.", UserWarning)

            if use_absolute_sigma and np.any(np.isclose(sigma_y, 0.0)):
                warnings.warn("curve_fit: Found zero values in y_err (sigma). "
                              "These points will cause issues (division by zero). "
                              "Result or errors might be inf/nan.", UserWarning)
                # curve_fit handles infinite weights poorly, often results in inf/nan cov matrix
                # No automatic fix applied here, user must handle zero errors if using curve_fit

            try:
                popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0_list,
                                       sigma=sigma_y,
                                       absolute_sigma=use_absolute_sigma,
                                       check_finite=kwargs.pop('check_finite', True), # Default True
                                       maxfev=kwargs.pop('maxfev', 5000), # Sensible default
                                       **kwargs)

                params_opt = popt
                with np.errstate(invalid='ignore'): # Ignore sqrt(negative/inf) warning
                    params_std_err = np.sqrt(np.diag(pcov))
                cov_matrix = pcov

                # Check if pcov contains inf/nan which indicates failure even if no exception was raised
                if np.any(~np.isfinite(pcov)):
                     warnings.warn("curve_fit completed, but covariance matrix contains Inf/NaN values. "
                                   "This often indicates a poorly constrained fit or issues with zero errors.", RuntimeWarning)
                     fit_success = False # Treat as failure if cov is unusable
                     params_std_err.fill(np.nan) # Mark errors as invalid
                else:
                     fit_success = True # curve_fit raises exception on primary failure

            except RuntimeError as e:
                # This usually means optimal parameters not found
                warnings.warn(f"curve_fit failed: {e}", RuntimeWarning)
                fit_success = False
                # p0 might be returned in popt, or it might be nonsense. Leave as NaN.
            except ValueError as e:
                 # Can happen with shape mismatches internally, or NaN/inf in input if check_finite=False
                 warnings.warn(f"curve_fit failed due to ValueError: {e}", RuntimeWarning)
                 fit_success = False


    # --- General Error Handling (Catches errors within fit blocks if not caught internally) ---
    except ImportError as e: # Catch the iminuit import error here as well
         raise e # Re-raise import error
    except ValueError as e: # Catch ValueErrors from data checks, param mismatches etc.
         warnings.warn(f"Fitting failed due to input error ({fit_method_name}): {e}", RuntimeWarning)
         fit_success = False
    except TypeError as e: # Catch TypeErrors from p0 etc.
        warnings.warn(f"Fitting failed due to type error ({fit_method_name}): {e}", RuntimeWarning)
        fit_success = False
    except Exception as e:
        # Catch any other unexpected errors during the fitting process itself
        import traceback
        warnings.warn(f"Unexpected error during fitting process ({fit_method_name}): {e}\n{traceback.format_exc()}", RuntimeWarning)
        # Try to capture the state before failing if possible
        if m is not None: fit_extra_info = m # Already assigned in minuit block
        elif 'output' in locals() and isinstance(output, ODR.Output): fit_extra_info = output
        fit_success = False


    # --- Package Results ---
    fit_params_dict = {}
    for i, name in enumerate(parameter_names):
        # Ensure index is within bounds before accessing potentially modified arrays
        val = params_opt[i] if i < len(params_opt) and np.isfinite(params_opt[i]) else np.nan
        err = params_std_err[i] if i < len(params_std_err) and np.isfinite(params_std_err[i]) else np.nan
        # Ensure NaN propagates if value is NaN, even if error calculation somehow yielded a number
        if np.isnan(val): err = np.nan
        fit_params_dict[name] = Measurement(val, err, name=name)

    result = FitResult(
        parameters=fit_params_dict,
        covariance_matrix=cov_matrix, # Might be None or contain NaN/inf
        function=func,
        parameter_names=parameter_names, # Store the definitive names list
        x_data=x_data, y_data=y_data,
        method=fit_method_name,
        mask=mask if original_mask_provided else None,
        success=fit_success,
        fit_object=fit_extra_info # Store Minuit/ODR object
    )

    # --- Calculate Goodness-of-Fit Statistics ---
    result.dof = n_points - n_params # Calculate DoF regardless of success, might be <= 0

    # Conditions for calculating Chi2: Fit succeeded, stats requested, DoF calculable, and errors appropriate for method
    can_calculate_stats = calculate_stats and fit_success and result.dof is not None
    meaningful_errors_for_chi2 = False
    chi2_source = "N/A"

    if fit_method_name == 'odr':
        # ODR's sum_square is chi2 if fit was weighted
        if fit_extra_info and hasattr(fit_extra_info, 'sum_square'):
            if sx is not None or sy is not None: # Weighted fit?
                 meaningful_errors_for_chi2 = True
                 chi2_source = "ODR sum_square (weighted)"
            # else: warnings.warn("Cannot calculate meaningful Chi² for unweighted ODR fit.", UserWarning) # Already warned

    elif fit_method_name == 'minuit':
        # Minuit's fval with LeastSquares cost IS the Chi² (since non-zero y_err was required)
        if fit_extra_info and hasattr(fit_extra_info, 'fval'):
            meaningful_errors_for_chi2 = True
            chi2_source = "Minuit fval (LeastSquares)"
        # else: y_errors_all_zero case already raised an error

    else: # curve_fit case
        # Chi² meaningful only if absolute_sigma=True (i.e., weighted fit)
        if use_absolute_sigma:
             meaningful_errors_for_chi2 = True
             chi2_source = "Manual calculation (curve_fit, absolute_sigma=True)"
        # else: warnings.warn("Cannot calculate meaningful Chi² for curve_fit: Fit used absolute_sigma=False (unweighted).", UserWarning) # Already warned

    # Proceed if conditions met
    if can_calculate_stats and meaningful_errors_for_chi2:
        chi2_val = None
        try:
            if fit_method_name == 'odr':
                chi2_val = fit_extra_info.sum_square
            elif fit_method_name == 'minuit':
                chi2_val = fit_extra_info.fval
            else: # curve_fit with absolute_sigma=True
                # Check params_opt are finite before using them
                if np.all(np.isfinite(params_opt)):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_pred = func(x_fit, *params_opt)
                        # Use y_err_fit which was passed as sigma. Handle potential zeros.
                        sigma_safe = np.where(np.isclose(y_err_fit, 0.0), 1.0, y_err_fit) # Avoid division by zero visually
                        residuals = (y_fit - y_pred) / sigma_safe
                        # Check for NaNs/Infs resulting from calculation (e.g., func output issues) or zero errors
                        finite_mask = np.isfinite(residuals)
                        if np.all(finite_mask):
                             chi2_val = np.sum(residuals**2)
                        else:
                             # If some residuals are non-finite, calculate sum only on finite ones and warn.
                             chi2_val = np.sum(residuals[finite_mask]**2)
                             warnings.warn("Chi² calculated for curve_fit excluded non-finite residuals (check function or zero errors).", RuntimeWarning)
                else:
                    warnings.warn("Cannot calculate Chi² for curve_fit: Optimal parameters contain NaN/inf.", RuntimeWarning)


            # Assign if calculation was successful (might be None if issues above)
            if chi2_val is not None and np.isfinite(chi2_val):
                result.chi_square = chi2_val
                if result.dof > 0:
                     result.reduced_chi_square = chi2_val / result.dof
                else: # DoF <= 0 case
                     warnings.warn(f"DoF = {result.dof} <= 0. Reduced Chi² cannot be calculated.", RuntimeWarning)
            else:
                 warnings.warn(f"Could not compute a valid Chi² value ({chi2_source}).", RuntimeWarning)

        except Exception as e:
             warnings.warn(f"Error during Chi² calculation ({chi2_source}): {e}", RuntimeWarning)

    # Add final warnings if stats were requested but couldn't be calculated
    elif calculate_stats and not fit_success:
         warnings.warn(f"Cannot calculate fit statistics (Chi², etc.): Fit failed (success=False).", RuntimeWarning)
    elif calculate_stats and not meaningful_errors_for_chi2:
         warnings.warn(f"Fit successful, but cannot calculate meaningful Chi² statistic: Method '{fit_method_name}' requires appropriate data errors for weighting (check {chi2_source}).", RuntimeWarning)
    elif calculate_stats and result.dof is None: # Should not happen if n_points > 0
         warnings.warn(f"Cannot calculate fit statistics: Degrees of freedom calculation failed.", RuntimeWarning)

    # Final check on DoF for reduced chi2 even if calculated above
    if result.chi_square is not None and (result.dof is None or result.dof <= 0):
         result.reduced_chi_square = None # Ensure it's None if DoF is invalid


    return result
# --- END OF FILE fitting.py ---