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
from .measurement import Measurement
from iminuit import Minuit
from iminuit.cost import LeastSquares


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
            param_sig_figs = 2
            for name in self.parameter_names:
                if name in self.parameters:
                     param = self.parameters[name]
                     if np.isnan(param.value) or np.isnan(param.error):
                         lines.append(f"    {name}: NaN ± NaN")
                     else:
                         lines.append(f"    {name}: {param.to_eng_string(sig_figs_error=param_sig_figs)}")
                else:
                     lines.append(f"    {name}: (Parameter not found in results?)")
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
        elif self.chi_square is not None:
             lines.append("  Goodness of Fit:")
             lines.append(f"    Chi²: {self.chi_square:.4g}")
             lines.append(f"    Degrees of Freedom (DoF): {self.dof if self.dof is not None else 'N/A'} (Reduced Chi² not computed)")
        elif self.method == 'minuit' and self.fit_object is not None and hasattr(self.fit_object, 'fval'):
            lines.append("  Goodness of Fit:")
            lines.append(f"    Minuit FCN Value (χ²): {self.fit_object.fval:.4g}")
            lines.append(f"    Degrees of Freedom (DoF): {self.dof if self.dof is not None else 'N/A'}")

        if self.mask is not None:
             points_used = np.sum(self.mask)
             total_points = len(self.mask)
             lines.append(f"  Mask Applied: {points_used} / {total_points} data points used.")
        lines.append("--------------------------")
        return "\n".join(lines)

    def __repr__(self) -> str:
        params_repr = {name: f"({p.value:.3g} ± {p.error:.2g})" for name, p in self.parameters.items()}
        chi2_str = f"{self.chi_square:.4g}" if self.chi_square is not None else 'None'
        return (f"FitResult(method='{self.method}', success={self.success}, "
                f"parameters={params_repr}, chi_square={chi2_str}, "
                f"dof={self.dof}, function={getattr(self.function,'__name__','?!')}, ...)")


class _FitWarnings:
    """Centralized warning management for fit operations."""
    
    def __init__(self):
        self.warnings = []
    
    def add(self, message: str, category: str = 'general'):
        """Add a warning message to be emitted later."""
        self.warnings.append((category, message))
    
    def emit_all(self):
        """Emit all collected warnings."""
        for category, message in self.warnings:
            warnings.warn(message, UserWarning if category == 'user' else RuntimeWarning)


def _validate_and_prepare_data(x_data, y_data, mask=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """Validate inputs and prepare data arrays for fitting."""
    # Convert to Measurement objects if needed
    if not isinstance(x_data, Measurement):
        x_data = Measurement(values=x_data, name=getattr(x_data, 'name', 'x'))
    if not isinstance(y_data, Measurement):
        y_data = Measurement(values=y_data, name=getattr(y_data, 'name', 'y'))
    
    # Check shape compatibility
    try:
        common_shape = np.broadcast(x_data.value, y_data.value).shape
    except ValueError:
        raise ValueError(f"Shape mismatch: x_data ({x_data.shape}) and y_data ({y_data.shape}) cannot be broadcast together.")
    
    # Extract and flatten arrays
    x_vals = np.broadcast_to(x_data.value, common_shape).flatten()
    y_vals = np.broadcast_to(y_data.value, common_shape).flatten()
    x_errs = np.broadcast_to(x_data.error, common_shape).flatten()
    y_errs = np.broadcast_to(y_data.error, common_shape).flatten()
    
    # Apply mask
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).flatten()
        if mask_arr.size == 1:
            active_mask = np.full_like(x_vals, mask_arr[0], dtype=bool)
        elif mask_arr.shape == x_vals.shape:
            active_mask = mask_arr
        else:
            raise ValueError(f"Mask shape {mask_arr.shape} incompatible with data shape {x_vals.shape}")
        
        x_vals, y_vals = x_vals[active_mask], y_vals[active_mask]
        x_errs, y_errs = x_errs[active_mask], y_errs[active_mask]
    else:
        active_mask = None
    
    n_points = len(x_vals)
    if n_points == 0:
        raise ValueError("No data points available after masking.")
    
    return x_vals, y_vals, x_errs, y_errs, n_points, active_mask


def _extract_parameter_info(func, parameter_names=None) -> Tuple[List[str], int]:
    """Extract parameter names and count from function signature."""
    try:
        sig = inspect.signature(func)
        func_params = list(sig.parameters.keys())[1:]  # Skip first parameter (x)
        n_params = len(func_params)
        
        if n_params == 0:
            raise ValueError("Function must have at least one parameter besides x.")
        
        if parameter_names is None:
            return func_params, n_params
        elif len(parameter_names) != n_params:
            raise ValueError(f"Provided parameter_names count ({len(parameter_names)}) != function parameters ({n_params})")
        else:
            return parameter_names, n_params
            
    except Exception as e:
        raise ValueError(f"Could not inspect function signature: {e}")


def _determine_method(x_vals, y_vals, x_errs, y_errs, n_points, method='auto') -> Tuple[str, _FitWarnings]:
    """Intelligent method selection based on data characteristics."""
    warn_mgr = _FitWarnings()
    
    if method != 'auto':
        return method, warn_mgr
    
    has_x_errors = np.any(x_errs > 0.005 * x_vals) # Error greater than 0.5%
    has_y_errors = np.any(y_errs > 0)
    
    # Selection logic
    if has_x_errors:
        selected = 'odr'
        warn_mgr.add("Method 'auto' selected 'odr' due to significant x-errors.", 'general')
    elif not has_y_errors:
        selected = 'curve_fit'
        warn_mgr.add("Method 'auto' selected 'curve_fit' (no errors provided - unweighted fit).", 'general')
    elif n_points < 20:  # For small datasets, use robust minuit
        selected = 'minuit'
        warn_mgr.add("Method 'auto' selected 'minuit' for robust fitting with small dataset.", 'general')
    else:
        selected = 'minuit'  # Default for well-behaved cases
    
    return selected, warn_mgr


def _validate_method_compatibility(method, x_vals, y_vals, x_errs, y_errs, warn_mgr):
    """Check method-data compatibility and add appropriate warnings."""
    has_x_errors = np.any(x_errs > 0.005 * x_vals)
    has_y_errors = np.any(y_errs > 0)
    
    if method == 'curve_fit':
        if has_x_errors:
            warn_mgr.add("curve_fit ignores x-errors. Consider 'odr' for x-error handling.", 'user')
        if not has_y_errors:
            warn_mgr.add("curve_fit with no y-errors: unweighted fit, parameter uncertainties scaled by residuals.", 'user')
    
    elif method == 'odr':
        if not has_x_errors and has_y_errors:
            warn_mgr.add("ODR selected but x-errors negligible. 'minuit' might be more appropriate.", 'user')
        if not has_x_errors and not has_y_errors:
            warn_mgr.add("ODR with no significant errors: unweighted fit.", 'user')
    
    elif method == 'minuit':
        if has_x_errors:
            warn_mgr.add("minuit (LeastSquares) ignores x-errors. Consider 'odr' for x-error handling.", 'user')
        if not has_y_errors:
            raise ValueError("minuit with LeastSquares cost requires non-zero y-errors for weighting.")


def _setup_initial_guess(p0, n_params, parameter_names, warn_mgr) -> List[float]:
    """Setup and validate initial parameter guess."""
    if p0 is None:
        p0_default = 1.0
        p0_list = [p0_default] * n_params
        warn_mgr.add(f"No initial guess provided. Using default value of {p0_default} for {parameter_names}.", 'user')
        return p0_list
    
    if not isinstance(p0, (list, tuple)):
        raise TypeError("p0 must be a list or tuple of initial parameter guesses.")
    
    if len(p0) != n_params:
        raise ValueError(f"p0 length ({len(p0)}) != number of parameters ({n_params})")
    
    return list(p0)


def _fit_with_curve_fit(func, x_vals, y_vals, y_errs, p0_list, warn_mgr, **kwargs) -> Dict[str, Any]:
    """Optimized scipy.curve_fit wrapper."""
    has_y_errors = np.any(y_errs > 0)
    sigma_y = y_errs if has_y_errors else None
    absolute_sigma = has_y_errors # If True, the errors are NOT scaled to give reduced χ²=1
    
    # Check for problematic zero errors
    if has_y_errors and np.any(y_errs==0.0):
        warn_mgr.add("Found zero y-errors in curve_fit - may cause numerical issues.", 'general')
    
    # Extract curve_fit specific parameters with sensible defaults
    cf_kwargs = {
        'sigma': sigma_y,
        'absolute_sigma': absolute_sigma,
        'method': kwargs.pop('method', 'lm'),
        'check_finite': kwargs.pop('check_finite', True),
        **kwargs
    }
    
    try:
        popt, pcov = curve_fit(func, x_vals, y_vals, p0=p0_list, **cf_kwargs)
        
        # Check covariance matrix validity
        if np.any(~np.isfinite(pcov)):
            warn_mgr.add("Covariance matrix contains Inf/NaN - poorly constrained fit.", 'general')
            return {'success': False, 'params': popt, 'errors': np.full_like(popt, np.nan), 'cov_matrix': pcov}
        
        param_errors = np.sqrt(np.diag(pcov))
        return {'success': True, 'params': popt, 'errors': param_errors, 'cov_matrix': pcov}
        
    except (RuntimeError, ValueError) as e:
        warn_mgr.add(f"curve_fit failed: {e}", 'general')
        return {'success': False, 'params': np.full(len(p0_list), np.nan), 
                'errors': np.full(len(p0_list), np.nan), 'cov_matrix': None}


def _fit_with_odr(func, x_vals, y_vals, x_errs, y_errs, p0_list, warn_mgr, **kwargs) -> Dict[str, Any]:
    """Optimized scipy.odr wrapper."""
    has_x_errors = np.any(x_errs > 0)
    has_y_errors = np.any(y_errs > 0)
    
    # Setup weights (ODR prefers weights over errors when possible)
    sx = x_errs if has_x_errors else None
    sy = y_errs if has_y_errors else None
    
    def odr_func_wrapper(beta, x):
        try:
            return func(x, *beta)
        except Exception:
            return np.full_like(x, np.nan) if hasattr(x, '__len__') else np.nan
    
    try:
        model = Model(odr_func_wrapper)
        data = RealData(x_vals, y_vals, sx=sx, sy=sy)
        
        # ODR-specific parameters with defaults
        odr_kwargs = {
            'maxit': kwargs.pop('maxit', 1000),
            **kwargs
        }
        
        odr_obj = ODR(data, model, beta0=p0_list, **odr_kwargs)
        output = odr_obj.run()
        
        # Check success (info codes 0-3 indicate convergence)
        success = output.info in [0, 1, 2, 3]
        if not success:
            warn_mgr.add(f"ODR failed: {output.stopreason} (info={output.info})", 'general')
        elif output.info > 0:
            warn_mgr.add(f"ODR converged with warnings: {output.stopreason}", 'general')
        
        # Handle covariance matrix scaling
        cov_matrix = None
        if output.cov_beta is not None:
            if sx is not None or sy is not None:  # Weighted fit
                cov_matrix = output.cov_beta
            else:  # Unweighted - scale by residual variance
                cov_matrix = output.cov_beta * output.res_var
        
        param_errors = np.sqrt(np.diag(output.cov_beta)) if success else np.full(len(p0_list), np.nan)
        
        return {'success': success, 'params': output.beta, 'errors': param_errors, 
                'cov_matrix': cov_matrix, 'fit_object': output}
                
    except Exception as e:
        warn_mgr.add(f"ODR fitting failed: {e}", 'general')
        return {'success': False, 'params': np.full(len(p0_list), np.nan),
                'errors': np.full(len(p0_list), np.nan), 'cov_matrix': None}


def _fit_with_minuit(func, x_vals, y_vals, y_errs, p0_list, parameter_names, warn_mgr, 
                     minuit_limits=None, **kwargs) -> Dict[str, Any]:
    """Optimized iminuit wrapper."""
    try:
        # Create cost function
        cost_func = LeastSquares(x_vals, y_vals, y_errs, func)
        
        # Setup initial parameters
        p0_dict = dict(zip(parameter_names, p0_list))
        
        # Create Minuit instance
        minuit_obj = Minuit(cost_func, **p0_dict)
        
        # Apply parameter limits
        if minuit_limits:
            for name, limits in minuit_limits.items():
                if name in minuit_obj.parameters:
                    minuit_obj.limits[name] = limits
                else:
                    warn_mgr.add(f"Parameter '{name}' in minuit_limits not found.", 'user')
        
        # Run optimization
        minuit_obj.migrad(**kwargs)
        
        # Calculate Hessian for accurate errors
        if minuit_obj.valid:
            try:
                minuit_obj.hesse()
            except RuntimeError as e:
                warn_mgr.add(f"HESSE calculation failed: {e}", 'general')
        
        success = minuit_obj.valid and minuit_obj.fmin.is_valid
        
        if success:
            # Extract results in parameter order
            params = np.array([minuit_obj.values[name] for name in parameter_names])
            errors = np.array([minuit_obj.errors[name] for name in parameter_names])
            
            # Extract covariance matrix
            cov_matrix = None
            if minuit_obj.covariance is not None:
                cov_matrix = np.array([[minuit_obj.covariance[i, j] 
                                      for j in range(len(parameter_names))] 
                                     for i in range(len(parameter_names))])
            
            return {'success': True, 'params': params, 'errors': errors, 
                    'cov_matrix': cov_matrix, 'fit_object': minuit_obj}
        else:
            warn_mgr.add(f"Minuit optimization failed (valid={minuit_obj.valid})", 'general')
            # Try to extract parameters even on failure
            try:
                params = np.array([minuit_obj.values[name] for name in parameter_names])
                errors = np.array([minuit_obj.errors[name] for name in parameter_names])
            except Exception:
                params = np.full(len(parameter_names), np.nan)
                errors = np.full(len(parameter_names), np.nan)
            
            return {'success': False, 'params': params, 'errors': errors, 
                    'cov_matrix': None, 'fit_object': minuit_obj}
            
    except Exception as e:
        warn_mgr.add(f"Minuit setup/execution failed: {e}", 'general')
        return {'success': False, 'params': np.full(len(p0_list), np.nan),
                'errors': np.full(len(p0_list), np.nan), 'cov_matrix': None}


def _calculate_chi_square(x_vals, y_vals, y_errs, func, params, method, fit_object=None) -> Optional[float]:
    """Unified chi-square calculation for all methods."""
    try:
        if method == 'odr' and fit_object and hasattr(fit_object, 'sum_square'):
            # ODR provides sum_square directly
            return fit_object.sum_square # fit_object.res_var is the reduced chi square instead
            
        elif method == 'minuit' and fit_object and hasattr(fit_object, 'fval'):
            # Minuit LeastSquares cost function value IS chi-square
            return fit_object.fval
            
        else:  # curve_fit or manual calculation
            if not np.all(np.isfinite(params)):
                return None
                
            with np.errstate(divide='ignore', invalid='ignore'):
                y_pred = func(x_vals, *params)
                
                # Handle zero errors safely
                sigma_safe = np.where(y_errs > 0, y_errs, 1.0)
                residuals = (y_vals - y_pred) / sigma_safe
                
                # Only include finite residuals
                finite_mask = np.isfinite(residuals)
                if np.any(finite_mask):
                    return np.sum(residuals[finite_mask]**2)
                    
        return None
        
    except Exception:
        return None


def _calculate_fit_statistics(chi_square, n_points, n_params, method, has_meaningful_errors) -> Dict[str, Optional[float]]:
    """Calculate degrees of freedom and reduced chi-square."""
    dof = n_points - n_params if n_points > n_params else None
    reduced_chi_square = None
    
    if chi_square is not None and dof is not None and dof > 0 and has_meaningful_errors:
        reduced_chi_square = chi_square / dof
    
    return {'dof': dof, 'reduced_chi_square': reduced_chi_square}


def perform_fit(x_data: Union[Measurement, np.ndarray, List, Tuple],
                y_data: Union[Measurement, np.ndarray, List, Tuple],
                func: Callable,
                p0: Optional[Union[List[float], Tuple[float, ...]]] = None,
                parameter_names: Optional[List[str]] = None,
                method: str = 'auto',
                mask: Optional[np.ndarray] = None,
                calculate_stats: bool = True,
                minuit_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
                **kwargs) -> FitResult:
    """
    Performs curve fitting using scipy.optimize.curve_fit, scipy.odr, or iminuit.

    Args:
        x_data: Independent variable data (Measurement, array, list, or tuple)
        y_data: Dependent variable data (Measurement, array, list, or tuple)  
        func: Model function f(x, p1, p2, ...) with x as first parameter
        p0: Initial parameter guess as list/tuple. Defaults to [1.0, ...] if None
        parameter_names: Parameter names list. Inferred from function if None
        method: Fitting method ('auto', 'curve_fit', 'odr', 'minuit')
        mask: Boolean array to select data points (True = use point)
        calculate_stats: Whether to calculate chi-square statistics
        minuit_limits: Parameter bounds for minuit as {'param': (lower, upper)}
        **kwargs: Additional arguments passed to underlying fitting routine

    Returns:
        FitResult: Complete fitting results with parameters, statistics, etc.

    Raises:
        ValueError: For incompatible inputs or invalid method selection
        TypeError: For incorrect input types
        ImportError: If minuit requested but not available
    """
    warn_mgr = _FitWarnings()
    
    try:
        # Input validation and data preparation
        x_vals, y_vals, x_errs, y_errs, n_points, active_mask = _validate_and_prepare_data(x_data, y_data, mask)
        
        # Parameter setup
        parameter_names, n_params = _extract_parameter_info(func, parameter_names)
        
        # Method selection and validation
        resolved_method, method_warnings = _determine_method(x_vals, y_vals, x_errs, y_errs, n_points, method)
        warn_mgr.warnings.extend(method_warnings.warnings)
        _validate_method_compatibility(resolved_method, x_vals, y_vals, x_errs, y_errs, warn_mgr)
        
        # Initial guess setup
        p0_list = _setup_initial_guess(p0, n_params, parameter_names, warn_mgr)
        
        # Perform fitting based on selected method
        if resolved_method == 'curve_fit':
            fit_result = _fit_with_curve_fit(func, x_vals, y_vals, y_errs, p0_list, warn_mgr, **kwargs)
        elif resolved_method == 'odr':
            fit_result = _fit_with_odr(func, x_vals, y_vals, x_errs, y_errs, p0_list, warn_mgr, **kwargs)
        elif resolved_method == 'minuit':
            fit_result = _fit_with_minuit(func, x_vals, y_vals, y_errs, p0_list, parameter_names, 
                                        warn_mgr, minuit_limits, **kwargs)
        else:
            raise ValueError(f"Unknown method: {resolved_method}")
        
        # Package parameters into Measurement objects
        fit_params = {}
        params_array = fit_result['params']
        errors_array = fit_result['errors']
        
        for i, name in enumerate(parameter_names):
            val = params_array[i] if i < len(params_array) else np.nan
            err = errors_array[i] if i < len(errors_array) else np.nan
            if np.isnan(val):
                err = np.nan  # Ensure NaN propagation
            fit_params[name] = Measurement(val, err, name=name)
        
        # Calculate statistics if requested
        chi_square = None
        stats = {'dof': None, 'reduced_chi_square': None}
        
        if calculate_stats and fit_result['success']:
            # Determine if errors are meaningful for chi-square calculation
            has_meaningful_errors = (
                (resolved_method == 'odr' and (np.any(x_errs > 0) or np.any(y_errs > 0))) or
                (resolved_method in ['curve_fit', 'minuit'] and np.any(y_errs > 0))
            )
            
            chi_square = _calculate_chi_square(
                x_vals, y_vals, y_errs, func, params_array, 
                resolved_method, fit_result.get('fit_object')
            )
            
            stats = _calculate_fit_statistics(
                chi_square, n_points, n_params, resolved_method, has_meaningful_errors
            )
        
        # Create and populate FitResult object
        result = FitResult(
            parameters=fit_params,
            covariance_matrix=fit_result.get('cov_matrix'),
            chi_square=chi_square,
            dof=stats['dof'],
            reduced_chi_square=stats['reduced_chi_square'],
            function=func,
            parameter_names=parameter_names,
            x_data=x_data if isinstance(x_data, Measurement) else Measurement(values=x_data, name='x'),
            y_data=y_data if isinstance(y_data, Measurement) else Measurement(values=y_data, name='y'),
            method=resolved_method,
            mask=active_mask,
            success=fit_result['success'],
            fit_object=fit_result.get('fit_object')
        )
        
        # Emit all collected warnings
        warn_mgr.emit_all()
        
        return result
        
    except Exception as e:
        # Create failure result
        warn_mgr.add(f"Fit operation failed: {e}", 'general')
        warn_mgr.emit_all()
        
        # Try to extract basic info for failure result
        try:
            param_names, n_params = _extract_parameter_info(func, parameter_names)
        except:
            param_names = parameter_names or []
            n_params = len(param_names)
        
        failure_params = {name: Measurement(np.nan, np.nan, name=name) for name in param_names}
        
        return FitResult(
            parameters=failure_params,
            parameter_names=param_names,
            function=func,
            x_data=x_data if isinstance(x_data, Measurement) else Measurement(values=x_data, name='x'),
            y_data=y_data if isinstance(y_data, Measurement) else Measurement(values=y_data, name='y'),
            method=method,
            success=False
        )