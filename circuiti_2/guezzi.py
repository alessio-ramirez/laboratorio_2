"""
guezzi.py - A Scientific Utility Library for Physics Students

This library provides tools for physics students to handle measurement data,
perform error propagation, statistical analysis, and visualize results.
The main focus is to simplify the work for the laboratory course at Bicocca's
University, visit https://github.com/alessio-ramirez/laboratorio_2 to see
the devolepment and use of this library. 

Mr Guezzi Legacy (MGL)

Key Features:
-------------
1. Data Management:
   - create_dataset(): Create standardized datasets with values and errors

2. Error Analysis:
   - error_prop(): Perform error propagation through arbitrary functions
   - test_comp(): Compare two measurements with their uncertainties

3. Curve Fitting:
   - perform_fit(): Fit data to models with automatic error handling
   - create_best_fit_line(): Plot data with best-fit curves

4. Visualization:
   - create_best_fit_line(): Plot data with fitted curves
   - latex_table(): Generate LaTeX tables for reporting results

Usage Notes:
-----------
- Always convert raw data to the standard format using create_dataset()
  since most functions require inputs in the form {'value': array, 'error': array}
- Functions handle errors appropriately when provided, but to mantain
  things simple the user is asked to read carefully the docs before using,
  for example, order of lists or dictionaries is a crucial part that the user
  should ensure
- For visualizations, the library uses matplotlib which must be installed.
- For LaTeX tables and other statistics, pyperclip is required to copy to clipboard.

Examples:
---------
>>> import guezzi as gz
>>> # Create a dataset
>>> data = gz.create_dataset([1.2, 2.3, 3.4], [0.1, 0.2, 0.3])
>>> # Fit to a model
>>> model = lambda x, a, b: a*x + b
>>> params, errors = gz.perform_fit(x_data, y_data, model)
"""

import numpy as np
import matplotlib.pyplot as plt
import pyperclip
import sympy as sp
import math
import re
import warnings
from itertools import combinations
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
import inspect
from typing import Union, List, Dict, Callable, Optional

# Module-level constants
DEFAULT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# ---------------------------- Dataset Creation ------------------------------

def create_dataset(values: Union[float, List[float], Dict], errors: Union[float, List[float]] = None, magnitude: int = 0) -> Dict[str, np.ndarray]:
    """
    Create a standardized dataset dictionary for measurements with errors.
    
    Parameters:
    -----------
    values : float, list, or dict
        - Single measurement (float)
        - List of measured values
        - Dictionary with {value: error} pairs (if provided, errors should be None)
    errors : float or list, optional
        Single error value (same error for each measured value) or list of errors.
        If None and values is not a dict, zero errors are assumed.
    magnitude : int, optional (default=0)
        Order of magnitude adjustment. Values and errors are multiplied by 10^magnitude.
    
    Returns:
    --------
    dict: {'value': np.array, 'error': np.array}
        Standard format dataset to use with other functions in this library.
    
    Examples:
    ---------
    >>> create_dataset(5.0, 1.0)  # Single measurement
    {'value': array([5.]), 'error': array([1.])}
    
    >>> create_dataset([2, 4], 0.5)  # Multiple measurements with uniform error
    {'value': array([2., 4.]), 'error': array([0.5, 0.5])}
    
    >>> create_dataset({10: 0.1, 20: 0.2})  # Dictionary input
    {'value': array([10., 20.]), 'error': array([0.1, 0.2])}
    
    >>> create_dataset([1.2, 3.4], [0.1, 0.2], magnitude=-3)  # With magnitude adjustment
    {'value': array([0.0012, 0.0034]), 'error': array([0.0001, 0.0002])}
    """
    # Case 1: Dictionary input - extract values and errors
    if isinstance(values, dict):
        if errors is not None:
            raise ValueError("Errors must be None when using dictionary input")
        return {
            'value': np.array(list(values.keys()), dtype=float) * 10**magnitude,
            'error': np.array(list(values.values()), dtype=float) * 10**magnitude
        }
    
    # Case 2: Scalar value - convert to list
    if np.isscalar(values):
        values = [float(values)]
    
    # Convert to numpy array
    values = np.array(values, dtype=float)
    
    # Process errors
    if errors is None:
        # Zero errors if not provided
        errors = np.zeros_like(values)
    elif np.isscalar(errors):
        # Same error for all values
        errors = np.full_like(values, errors, dtype=float)
    else:
        errors = np.array(errors, dtype=float)
        if errors.shape != values.shape:
            raise ValueError("Errors length must match values length")
    
    # Apply magnitude adjustment
    return {
        'value': values * 10**magnitude,
        'error': errors * 10**magnitude
    }

# ---------------------------- Error Propagation -----------------------------

def error_prop(f: Callable, *variables, covariance_matrix=None, copy_latex: bool = False, round_latex: int = 3) -> Dict[str, np.ndarray]:
    """
    Calculate error propagation through a function using partial derivatives.
    Among the hypothesis for the formula with partial derivatives and covariances to be
    valid there is the requirement of errors to be relatively small, since this way of
    propagate errors relies on the fact that the first order Taylor's expansion is a good
    approximation. Mind that covariance is needed when errors are related, so if the errors
    are the standard deviation estimators, for example, then covariance matrix is needed, 
    but if the errors correspond to the sensitivity of measuring instruments, it's not.
    
    Parameters:
    -----------
    f : callable
        Function to propagate errors through. Must take individual arguments (not arrays).
    *variables : dict
        Variable datasets in the form of {'value': array, 'error': array}.
    covariance_matrix : numpy.ndarray, optional
        Precomputed covariance matrix of shape (n_variables, n_variables).
        If None, variables are assumed to be independent.
    copy_latex : bool, optional (default=False)
        If True, copy the LaTeX formula to clipboard.
    round_latex : int, optional (default=3)
        Number of significant digits for LaTeX output.
    
    Returns:
    --------
    dict: {'value': np.array, 'error': np.array}
        Results and errors in standard format
    
    Example:
    --------
    >>> f = lambda x, y: x*y
    >>> x_data = create_dataset([1,2], 0.1)
    >>> y_data = create_dataset([3,4], 0.2)
    >>> error_prop(f, x_data, y_data)
    >>> # With covariance matrix
    >>> cov_matrix = np.array([[0.01, 0.005], [0.005, 0.04]])
    >>> error_prop(f, x_data, y_data, covariance_matrix=cov_matrix)
    """
    # Validate inputs
    for var in variables:
        if not isinstance(var, dict) or 'value' not in var or 'error' not in var:
            raise TypeError("Each variable must be a dict with 'value' and 'error' keys.")
    
    # Extract values and errors
    values_list = [np.array(var['value']) for var in variables]
    errors_list = [np.array(var['error']) for var in variables]
    
    # Ensure all datasets have the same length
    n_points = len(values_list[0])
    for values in values_list:
        if len(values) != n_points:
            raise ValueError("All variables must have the same number of data points.")
    
    # Create symbolic variables for differentiation
    n_vars = len(variables)
    symbols = sp.symbols(f'x:{n_vars}')
    
    # Create symbolic function
    f_sym = f(*symbols)
    
    # Calculate derivatives
    derivatives = [sp.diff(f_sym, sym) for sym in symbols]
    
    # Generate symbolic error formula
    error_formula = sp.sqrt(
        sum(
            (sp.diff(f_sym, symbols[i]) * sp.symbols(f'sigma_{i}'))**2
            for i in range(n_vars)
        )
    )
    
    # Validate covariance terms if provided
    if covariance_matrix is not None:
        if covariance_matrix.shape != (n_vars, n_vars):
            raise ValueError(f"Covariance matrix must have shape ({n_vars}, {n_vars})")
        
        # Add covariance terms to error formula
        cov_terms = 0
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if not np.isclose(covariance_matrix[i, j], 0):
                    cov_term = 2 * sp.diff(f_sym, symbols[i]) * sp.diff(f_sym, symbols[j]) * sp.symbols(f'cov_{i}_{j}')
                    cov_terms += cov_term
        
        error_formula = sp.sqrt(error_formula**2 + cov_terms)
    
    # Handle LaTeX output
    if copy_latex:
        if 'pyperclip' not in globals():
            raise ImportError("pyperclip is required for LaTeX clipboard copy")
        
        # Evaluate the expressions to a numerical value with specified significant digits
        f_sym_rounded = sp.N(f_sym, round_latex)
        error_formula_rounded = sp.N(error_formula, round_latex)
        
        # Convert the rounded expressions to LaTeX
        f_latex = sp.latex(f_sym_rounded, mode="equation")
        sigma_latex = sp.latex(error_formula_rounded, mode="equation")
        
        pyperclip.copy(f"Function:\n{f_latex}\n\nPropagated Error:\n{sigma_latex}")
    
    # Calculate results for each data point
    values = []
    errors = []
    for i in range(n_points):
        # Get values for this data point
        point_values = [values[i] for values in values_list]
        point_errors = [errors[i] for errors in errors_list]
        
        # Calculate function value at this point
        value = f(*point_values)
        
        # Calculate derivatives at this point
        deriv_values = []
        for deriv in derivatives:
            # Substitute values into derivative expression
            subs_dict = {sym: val for sym, val in zip(symbols, point_values)}
            deriv_val = float(deriv.subs(subs_dict).evalf())
            deriv_values.append(deriv_val)
        
        # Calculate error components
        error_squared = sum((d**2) * (e**2) for d, e in zip(deriv_values, point_errors))
        
        # Add covariance components if provided
        if covariance_matrix is not None:
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    if not np.isclose(covariance_matrix[i, j], 0):
                        error_squared += 2 * deriv_values[i] * deriv_values[j] * covariance_matrix[i, j]
        
        error = np.sqrt(error_squared)
        values.append(value)
        errors.append(error)
    
    return create_dataset(values, errors)

# ---------------------------- Curve Fitting --------------------------------

def perform_fit(x: Union[Dict, np.ndarray], y: Union[Dict, np.ndarray], 
                func: Callable, p0: Optional[Union[List[float], float]] = None, 
                chi_square: bool = False, method: str = 'auto',
                parameter_names: Optional[List[str]] = None,
                mask: Optional[Union[np.ndarray, List[bool]]] = None) -> Dict:
    """
    Perform curve fitting with automatic error-aware method selection and data masking.
    
    Parameters:
    -----------
    x : dict or array
        Independent variable data. If dict, must have 'value' and optionally 'error' keys.
    y : dict or array
        Dependent variable data. If dict, must have 'value' and optionally 'error' keys.
    func : callable
        Model function in the form f(x, *params).
    p0 : list or float, optional
        Initial parameter guesses. If None, defaults to [1.0, 1.0, ...].
    chi_square : bool, optional (default=False)
        If True, calculate and return chi-square statistics.
    method : str, optional (default='auto')
        Fitting method: 'auto', 'curve_fit', or 'odr'.
        - 'auto': Use ODR if x has errors, otherwise use curve_fit
        - 'curve_fit': Force use of scipy.optimize.curve_fit
        - 'odr': Force use of scipy.odr.ODR
    parameter_names : list of str, optional
        Names for the parameters. If None, defaults to ["param1", "param2", ...].
    mask : array-like of bool, optional
        Boolean mask to select which data points to use in the fit.
        True values indicate data points to include.
        Will be applied to both x and y data arrays.
    
    Returns:
    --------
    dict: A dictionary with fit results formatted for easy table creation:
        {
            'parameters': {
                'value': np.array of parameter values,
                'error': np.array of parameter errors
            },
            'parameter_names': list of parameter names,
            'stats': {
                'chi_square': value (if requested),
                'dof': degrees of freedom (if chi_square=True),
                'reduced_chi_square': value (if chi_square=True)
            },
            'method': the method used ('curve_fit' or 'odr'),
            'mask': the mask used for the fit (if provided)
        }
    
    Notes:
    ------
    - ODR is used when x errors are present or method='odr'
    - curve_fit is used when no x errors are present or method='curve_fit'
    - Function signature is automatically detected and adapted for the chosen method
    - The mask will be applied to both x and y datasets, with the same indices masked
    
    Example:
    --------
    >>> model = lambda x, a, b: a*x + b
    >>> x_data = create_dataset([1,2,3,4,5], 0.1)
    >>> y_data = create_dataset([2,4,6,7,10], 0.2)
    >>> # Use only the first 3 data points
    >>> mask = [True, True, True, False, False]
    >>> fit_results = perform_fit(x_data, y_data, model, mask=mask, parameter_names=["slope", "intercept"])
    >>> latex_table_from_fit(fit_results, title="Linear Fit Results")
    """
    # Process x data
    if isinstance(x, dict):
        x_val = np.asarray(x['value'])
        x_err = np.asarray(x.get('error', np.zeros_like(x_val)))
    else:
        x_val = np.asarray(x)
        x_err = np.zeros_like(x_val)
    
    # Process y data
    if isinstance(y, dict):
        y_val = np.asarray(y['value'])
        y_err = np.asarray(y.get('error', np.zeros_like(y_val)))
    else:
        y_val = np.asarray(y)
        y_err = np.zeros_like(y_val)
    
    # Process mask if provided
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(x_val):
            raise ValueError(f"Mask length ({len(mask)}) must match data length ({len(x_val)})")
        
        # Apply mask to all arrays
        x_val = x_val[mask]
        x_err = x_err[mask]
        y_val = y_val[mask]
        y_err = y_err[mask]
    
    # Handle p0 cases
     # Get function parameter count
    sig = inspect.signature(func)
    n_params = len(sig.parameters) - 1  # Subtract the x parameter
    if p0 is None: # Auto-generate initial parameters if not provided
        sig = inspect.signature(func)
        p0 = [1.0] * n_params
    elif isinstance(p0, (int, float)):
        p0 = [p0] * n_params
    elif len(p0) != n_params:
        raise TypeError(f"The length of p0 ({len(p0)}) must match the number of parameters ({n_params})")

    # Auto-generate parameter names if not provided
    if parameter_names is None:
        parameter_names = [f"param{i+1}" for i in range(n_params)]
    elif len(parameter_names) != len(p0):
        raise ValueError(f"Number of parameter names ({len(parameter_names)}) must match number of parameters ({len(p0)})")
    
    # Determine fitting method
    has_x_errors = np.any(x_err > 0)
    
    if method == 'auto':
        use_odr = has_x_errors
    elif method == 'curve_fit':
        use_odr = False
    elif method == 'odr':
        use_odr = True
    else:
        raise ValueError("Method must be 'auto', 'curve_fit', or 'odr'")
    
    # Prepare result dictionary
    result = {
        'parameter_names': parameter_names,
        'stats': {},
        'method': 'odr' if use_odr else 'curve_fit'
    }
    
    # Store mask in result if provided
    if mask is not None:
        result['mask'] = mask
    
    # Perform fitting
    if use_odr:
        def odr_wrapper(beta, x): # Convert from curve_fit style to ODR style
            return func(x, *beta)
        fit_func = odr_wrapper
    
        # Prepare data for ODR
        from scipy.odr import Model, ODR, RealData
        odr_model = Model(fit_func)
        data = RealData(x_val, y_val, sx=x_err, sy=y_err)
        odr = ODR(data, odr_model, beta0=p0)
        output = odr.run()
        
        params = output.beta
        params_err = output.sd_beta
        
        # Store parameters in result dictionary
        result['parameters'] = {
            'value': np.array(params),
            'error': np.array(params_err)
        }
        
        if chi_square:
            chi_squared = output.sum_square
            dof = len(y_val) - len(params)
            reduced_chi_sq = chi_squared / dof if dof > 0 else np.nan
            
            # Store statistics in result dictionary
            result['stats'] = {
                'chi_square': chi_squared,
                'dof': dof,
                'reduced_chi_square': reduced_chi_sq
            }
            
            # Copy chi-square statistics to clipboard if available
            try:
                import pyperclip
                pyperclip.copy(f"$\\chi^2 = {chi_squared:.3f}$, dof$={dof}$, $\\tilde{{\\chi^2}} = {reduced_chi_sq:.3f}$")
            except (ImportError, AttributeError):
                pass
        
    else:  # Use curve_fit
        # Handle y errors
        from scipy.optimize import curve_fit
        if np.any(y_err > 0):
            # Replace zero errors with small values to avoid division by zero
            y_err_adj = np.where(y_err == 0, 1e-10, y_err)
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0, sigma=y_err_adj, absolute_sigma=True) #Doesn't change y errors
        else:
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0)
        
        params = popt
        params_err = np.sqrt(np.diag(pcov))
        
        # Store parameters in result dictionary
        result['parameters'] = {
            'value': np.array(params),
            'error': np.array(params_err)
        }
        
        if chi_square:
            # Calculate chi-square
            y_pred = func(x_val, *params)
            
            if np.any(y_err > 0):
                y_err_adj = np.where(y_err == 0, 1e-10, y_err) # Check for zero division
                residuals = (y_val - y_pred) / y_err_adj
                chi_squared = np.sum(residuals**2)
            else:
                residuals = y_val - y_pred # It has no statistical meaning
                chi_squared = np.sum(residuals**2)
                warnings.warn("Chi squared test with no y errors has no meaning")
            
            dof = len(y_val) - len(params)
            reduced_chi_sq = chi_squared / dof if dof > 0 else np.nan
            
            # Store statistics in result dictionary
            result['stats'] = {
                'chi_square': chi_squared,
                'dof': dof,
                'reduced_chi_square': reduced_chi_sq
            }
            
            # Copy chi-square statistics to clipboard if available
            try:
                import pyperclip
                pyperclip.copy(f"$\\chi^2 = {chi_squared:.3f}$, dof$={dof}$, $\\tilde{{\\chi^2}} = {reduced_chi_sq:.3f}$")
            except (ImportError, AttributeError):
                pass
    
    return result

# ---------------------------- Latex Formatting -----------------------------

def latex_table(*args, orientation="h", magnitude=None):
    """
    Generate a LaTeX table from provided datasets with consistent formatting.
    
    The function rounds values based on the error's first significant digit and
    applies a consistent order of magnitude for better readability.
    
    Parameters:
    -----------
    *args : str, dict pairs
        Pairs of names and datasets. Each dataset is a dict with 'value' and 'error' arrays.
    orientation : str, optional (default='h')
        'h' for horizontal (names as row headers), 'v' for vertical (names as column headers).
    magnitude : int, optional (default=None)
        If provided, all values and errors will be scaled by 10^magnitude.
        If None, magnitude is determined automatically for each dataset.
    
    Returns:
    --------
    str: LaTeX table code (also copied to clipboard).
    
    Example:
    --------
    >>> data1 = create_dataset([10.1, 10.2], [0.2, 0.3])
    >>> data2 = create_dataset([20.5, 21.0], [0.5, 0.6])
    >>> latex_table("Dataset 1", data1, "Dataset 2", data2, magnitude=-1)
    """
    # Validate input pairs
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of name and dataset")
    
    pairs = []
    for i in range(0, len(args), 2):
        name = args[i]
        dataset = args[i+1]
        if not isinstance(dataset, dict) or 'value' not in dataset or 'error' not in dataset:
            raise ValueError("Dataset must be a dict with 'value' and 'error' keys")
        if len(dataset['value']) != len(dataset['error']):
            raise ValueError("Dataset value and error arrays must have the same length")
        pairs.append((name, dataset))
    
    # Check all datasets have the same length
    lengths = [len(d['value']) for (_, d) in pairs]
    if len(set(lengths)) != 1:
        raise ValueError("All datasets must have the same length")
    n_entries = lengths[0]
    
    formatted_data = []
    for name, dataset in pairs:
        values = np.array(dataset['value'], dtype=float)
        errors = np.array(dataset['error'], dtype=float)
        
        # Apply user-provided magnitude scaling
        if magnitude is not None:
            scaling_factor = 10 ** (-magnitude)
            exponent = magnitude
        else:
            # Determine scaling factor based on the maximum error
            if np.all(np.isclose(errors, 0.0, atol=1e-12)):
                # Use max value if all errors are zero
                max_val = np.max(np.abs(values)) if len(values) > 0 else 0.0
                if np.isclose(max_val, 0.0, atol=1e-12):
                    scaling_factor = 1.0
                    exponent = 0
                else:
                    exponent = int(np.floor(np.log10(max_val)))
                    scaling_factor = 10 ** (-exponent)
            else:
                # Use average error order of magnitude
                non_zero_errors = errors[~np.isclose(errors, 0.0, atol=1e-12)]
                if len(non_zero_errors) > 0:
                    exponent = int(np.floor(np.mean(np.log10(non_zero_errors))))
                    scaling_factor = 10 ** (-exponent)
                else:
                    scaling_factor = 1.0
                    exponent = 0
        
        # Create the new name with magnitude indicator
        if exponent != 0:
            magnitude_str = f" $\\times 10^{{{int(exponent)}}}$"
        else:
            magnitude_str = ""
        new_name = f"{name}{magnitude_str}"
        
        # Process each entry in the dataset
        entries = []
        for v, e in zip(values, errors):
            scaled_v = v * scaling_factor
            scaled_e = e * scaling_factor
            
            if np.isclose(scaled_e, 0.0, atol=1e-12):
                # No error, format value with up to 3 significant digits
                entries.append(f"{scaled_v:.3g}")
            else:
                # Find the position of the first significant digit in the error
                if scaled_e < 1e-12:
                    decimal_places = 0
                else:
                    decimal_places = -int(np.floor(np.log10(scaled_e)))
                    decimal_places = max(0, decimal_places)
                
                # Round value and error to the appropriate decimal places
                rounded_v = round(scaled_v, decimal_places)
                # Ensure error has exactly one significant digit
                rounded_e = round(scaled_e, decimal_places)
                
                # Format value and error strings
                if decimal_places > 0:
                    str_v = f"{rounded_v:.{decimal_places}f}"
                    str_e = f"{rounded_e:.{decimal_places}f}"
                else:
                    str_v = f"{int(rounded_v)}"
                    str_e = f"{int(rounded_e)}"
                
                entries.append(f"${str_v} \\pm {str_e}$")
        
        formatted_data.append((new_name, entries))
    
    # Generate LaTeX code based on orientation
    if orientation == 'h':
        # Horizontal: each name is a row with its entries
        columns = '|l|' + 'c' * n_entries + '|'
        latex = f"\\begin{{tabular}}{{{columns}}}\n\\hline\n"
        for name, entries in formatted_data:
            row = [name] + entries
            latex += " & ".join(row) + " \\\\\\hline\n"
        latex += "\\end{tabular}"
    elif orientation == 'v':
        # Vertical: names are column headers, entries are in columns
        num_columns = len(pairs)
        latex = f"\\begin{{tabular}}{{{'|'+'c|' * num_columns}}}\n\\hline\n"
        headers = [name for (name, _) in formatted_data]
        latex += " & ".join(headers) + " \\\\\n\\hline\n"
        for i in range(n_entries):
            row = [entries[i] for (_, entries) in formatted_data]
            latex += " & ".join(row) + " \\\\\n"
        latex += "\\hline\n\\end{tabular}"
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    
    # Copy to clipboard and return
    pyperclip.copy(latex)
    return latex

def fit_results_table(*args, orientation="h", parameter_labels: Optional[List[str]] = None,
                      include_stats: List[str] = None, precision: int = 3) -> str:
    """
    Generate a LaTeX table of fit results.
    
    Parameters:
    -----------
    *args : Dict
        Fit result dictionaries from perform_fit, with optional labels
    orientation : str
        Table orientation: 'h' (horizontal) or 'v' (vertical)
    parameter_labels : Optional[List[str]]
        Custom labels for parameters (overrides parameter_names in fit results)
    include_stats : List[str]
        Statistics to include in the table (e.g., 'chi_square', 'r_squared')
    precision : int
        Decimal precision for values
        
    Returns:
    --------
    str
        LaTeX code for the table
    """
    # Process arguments
    results = []
    labels = []
    
    for i, arg in enumerate(args):
        if i % 2 == 0 and i + 1 < len(args) and isinstance(args[i+1], dict) and 'parameters' in args[i+1]:
            # This is a label for the next fit result
            labels.append(str(arg))
        elif isinstance(arg, dict) and 'parameters' in arg:
            # This is a fit result
            results.append(arg)
            # If no label was provided
            if len(labels) < len(results):
                labels.append(f"Fit {len(results)}")
    
    if not results:
        raise ValueError("No fit results provided")
    
    # Default statistics to include
    if include_stats is None:
        include_stats = ['chi_square', 'reduced_chi_square', 'r_squared', 'dof']
    
    # Get all parameter names
    all_param_names = []
    for result in results:
        param_names = result.get('parameter_names', [])
        for name in param_names:
            if name not in all_param_names:
                all_param_names.append(name)
    
    # Use custom parameter labels if provided
    if parameter_labels is not None:
        if len(parameter_labels) != len(all_param_names):
            raise ValueError(f"Number of parameter labels ({len(parameter_labels)}) must match number of parameters ({len(all_param_names)})")
        param_display_names = parameter_labels
    else:
        param_display_names = all_param_names
    
    # Format values with errors
    def format_value_error(value, error):
        if np.isclose(error, 0.0):
            return f"{value:.{precision}g}"
        else:
            # Find appropriate decimal places based on error
            if error < 1e-10:
                decimal_places = precision
            else:
                decimal_places = -int(np.floor(np.log10(error))) + (precision - 1)
                decimal_places = max(0, decimal_places)
            
            formatted_value = f"{value:.{decimal_places}f}"
            formatted_error = f"{error:.{decimal_places}f}"
            return f"${formatted_value} \\pm {formatted_error}$"
    
    # Format statistic value
    def format_stat_value(value):
        if isinstance(value, (int, np.integer)):
            return f"{value}"
        else:
            return f"{value:.{precision}g}"
    
    # Generate horizontal table (parameters as rows, fits as columns)
    if orientation == 'h':
        # Create header row
        header = ["Parameter"] + labels
        rows = []
        
        # Add parameter rows
        for i, param_name in enumerate(all_param_names):
            row = [param_display_names[i]]
            for result in results:
                param_index = result.get('parameter_names', []).index(param_name) if param_name in result.get('parameter_names', []) else -1
                if param_index >= 0:
                    value = result['parameters']['value'][param_index]
                    error = result['parameters']['error'][param_index]
                    row.append(format_value_error(value, error))
                else:
                    row.append("--")
            rows.append(row)
        
        # Add statistics rows
        for stat in include_stats:
            if any(stat in result.get('stats', {}) for result in results):
                row = [stat.replace("_", " ").title()]
                for result in results:
                    if stat in result.get('stats', {}):
                        row.append(format_stat_value(result['stats'][stat]))
                    else:
                        row.append("--")
                rows.append(row)
        
        # Generate LaTeX code
        cols = "|l|" + "c|" * len(results)
        latex = f"\\begin{{tabular}}{{{cols}}}\n\\hline\n"
        latex += " & ".join(header) + " \\\\\n\\hline\\hline\n"
        for row in rows:
            latex += " & ".join(row) + " \\\\\n\\hline\n"
        latex += "\\end{tabular}"
        
    # Generate vertical table (fits as rows, parameters as columns)
    else:  # orientation == 'v'
        # Create header row with parameter names
        header = ["Fit"] + param_display_names + [stat.replace("_", " ").title() for stat in include_stats if any(stat in result.get('stats', {}) for result in results)]
        rows = []
        
        # Add a row for each fit
        for i, (label, result) in enumerate(zip(labels, results)):
            row = [label]
            
            # Add parameter values
            for param_name in all_param_names:
                param_index = result.get('parameter_names', []).index(param_name) if param_name in result.get('parameter_names', []) else -1
                if param_index >= 0:
                    value = result['parameters']['value'][param_index]
                    error = result['parameters']['error'][param_index]
                    row.append(format_value_error(value, error))
                else:
                    row.append("--")
            
            # Add statistics
            for stat in include_stats:
                if any(stat in res.get('stats', {}) for res in results):
                    if stat in result.get('stats', {}):
                        row.append(format_stat_value(result['stats'][stat]))
                    else:
                        row.append("--")
            
            rows.append(row)
        
        # Generate LaTeX code
        cols = "|l|" + "c|" * (len(all_param_names) + sum(1 for stat in include_stats if any(stat in result.get('stats', {}) for result in results)))
        latex = f"\\begin{{tabular}}{{{cols}}}\n\\hline\n"
        latex += " & ".join(header) + " \\\\\n\\hline\\hline\n"
        for row in rows:
            latex += " & ".join(row) + " \\\\\n\\hline\n"
        latex += "\\end{tabular}"
    
    # Copy to clipboard and return
    pyperclip.copy(latex)
    return latex

# ---------------------------- Visualization --------------------------------

def create_best_fit_line(*args: Union[Dict, np.ndarray], func: Callable, 
                         p0: Optional[Union[List[float], List[List[float]]]] = None,
                         xlabel: str = None, ylabel: str = None, title: str = None,
                         colors: List[str] = None, labels: List[str] = None, 
                         fit_line: bool = True, label_fit: List[str] = None,
                         together: bool = True, figsize: tuple = (10, 6),
                         save_path: str = None, dpi: int = 300,
                         grid: bool = False, legend_loc: str = 'best',
                         fit_points: int = 1000, fit_range: tuple = None,
                         confidence_interval: float = None, 
                         residuals: bool = False, fmt: str = '+',
                         markersize: int = 6, linewidth: float = 1.5,
                         show_fit_params: bool = False,
                         show_chi_squared: bool = False,
                         xlim: tuple = None, ylim: tuple = None,
                         capsize: float = 3, axis_fontsize: int = 12,
                         title_fontsize: int = 14,
                         masks: List[np.ndarray] = None,
                         parameter_names: List[List[str]] = None,
                         method: str = 'auto',
                         show_masked_points: bool = True,
                         masked_fmt: str = 'x',
                         masked_color: str = None,
                         masked_alpha: float = 0.5) -> plt.Figure:
    """
    Create a plot with data points and their best-fit curves, using perform_fit for the fitting.
    
    Parameters:
    -----------
    *args : Dict or np.ndarray
        Pairs of x and y datasets. Each dataset should be a dictionary with 'value' and 'error' keys.
    func : callable
        The function to fit the data to. Should take x as first argument followed by parameters.
    p0 : list or list of lists, optional
        Initial parameters for the fit. Can be:
        - None: Auto-generate with 1.0 for each parameter
        - Single list for all datasets: [a, b, ...]
        - List of lists for multiple datasets: [[a1, b1, ...], [a2, b2, ...]]
    xlabel, ylabel : str, optional
        Labels for the x and y axes.
    title : str, optional
        Title for the plot.
    colors : list of str, optional
        Colors for each dataset. If None, default colors will be used.
    labels : list of str, optional
        Labels for the data points in the legend.
    fit_line : bool, optional (default=True)
        Whether to plot the fit line.
    label_fit : list of str, optional
        Labels for the fitted curves in the legend.
    together : bool, optional (default=True)
        If True, plot all datasets on the same figure. If False, create separate figures.
    figsize : tuple, optional (default=(10, 6))
        Size of the figure in inches.
    save_path : str, optional
        If provided, save the figure to this path.
    dpi : int, optional (default=300)
        Resolution for the saved figure.
    grid : bool, optional (default=False)
        Whether to show grid lines.
    legend_loc : str, optional (default='best')
        Location of the legend.
    fit_points : int, optional (default=1000)
        Number of points to use for the fit line.
    fit_range : tuple, optional
        Range to plot the fit line (min_x, max_x). If None, use data range.
    confidence_interval : float, optional
        If provided, draw confidence interval bands (0-1).
    residuals : bool, optional (default=False)
        If True, add a subplot with residuals.
    fmt : str, optional (default='+')
        Format string for the data points.
    markersize : int, optional (default=6)
        Size of the markers.
    linewidth : float, optional (default=1.5)
        Width of the fit line.
    show_fit_params : bool, optional (default=False)
        If True, display fit parameters on the plot.
    show_chi_squared : bool, optional (default=False)
        If True, calculate and display chi-squared statistics.
    xlim, ylim : tuple, optional
        Limits for x and y axes.
    capsize : float, optional (default=3)
        Size of the error bar caps.
    axis_fontsize : int, optional (default=12)
        Font size for axis labels.
    title_fontsize : int, optional (default=14)
        Font size for the title.
    masks : list of arrays, optional
        List of boolean masks for each dataset. True values indicate data points to include in the fit.
        Each mask will be applied to the corresponding dataset pair.
    parameter_names : list of lists, optional
        Names for parameters in each fit. Can be a list of lists for multiple datasets.
    method : str, optional (default='auto')
        Fitting method: 'auto', 'curve_fit', or 'odr' to pass to perform_fit.
    show_masked_points : bool, optional (default=True)
        If True, also display the masked (excluded) points with a different style.
    masked_fmt : str, optional (default='x')
        Format string for the masked data points.
    masked_color : str, optional
        Color for masked points. If None, uses the same color as unmasked but with transparency.
    masked_alpha : float, optional (default=0.5)
        Alpha transparency for masked points.
    
    Returns:
    --------
    fig : matplotlib.pyplot.Figure
        The figure object.
    
    Examples:
    ---------
    >>> x = create_dataset([1, 2, 3, 4, 5], 0.1)
    >>> y = create_dataset([2.1, 3.9, 6.2, 8.1, 9.8], 0.2)
    >>> model = lambda x, a, b: a*x + b
    >>> fig = create_best_fit_line(x, y, func=model, show_fit_params=True)
    
    >>> # Multiple datasets with masks
    >>> x1 = create_dataset([1, 2, 3, 4, 5], 0.1)
    >>> y1 = create_dataset([2.1, 3.9, 6.2, 8.1, 9.8], 0.2)
    >>> x2 = create_dataset([1, 2, 3, 4, 5], 0.1)
    >>> y2 = create_dataset([1.1, 1.9, 3.2, 4.1, 4.8], 0.2)
    >>> # Mask to exclude the last point of the first dataset
    >>> mask1 = [True, True, True, True, False]
    >>> fig = create_best_fit_line(x1, y1, x2, y2, func=model, 
    ...                            masks=[mask1, None], 
    ...                            labels=['Dataset 1', 'Dataset 2'])
    """
    
    # Parameter normalization
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be pairs of x and y datasets.")
    
    num_datasets = len(args) // 2
    
    # Get function parameter count
    sig = inspect.signature(func)
    n_params = len(sig.parameters) - 1  # Subtract x parameter
    
    # Handle p0
    if p0 is None:
        p0 = [[1.0] * n_params for _ in range(num_datasets)]
    elif not isinstance(p0[0], (list, tuple, np.ndarray)):
        p0 = [p0] * num_datasets
    elif len(p0) != num_datasets:
        raise ValueError(f"p0 length ({len(p0)}) must match number of datasets ({num_datasets}).")
    
    # Handle parameter_names
    if parameter_names is None:
        parameter_names = [[f"p{j}" for j in range(n_params)] for _ in range(num_datasets)]
    elif not isinstance(parameter_names[0], (list, tuple)):
        parameter_names = [parameter_names] * num_datasets
    elif len(parameter_names) != num_datasets:
        raise ValueError(f"parameter_names length ({len(parameter_names)}) must match number of datasets ({num_datasets}).")
    
    # Handle masks
    if masks is None:
        masks = [None] * num_datasets
    elif len(masks) != num_datasets:
        raise ValueError(f"masks length ({len(masks)}) must match number of datasets ({num_datasets}).")
    
    # Ensure other parameters are lists of the right length
    def ensure_list(param, name):
        if param is None:
            return [None] * num_datasets
        if not isinstance(param, (list, tuple)):
            return [param] * num_datasets
        if len(param) != num_datasets:
            raise ValueError(f"{name} length must match number of datasets.")
        return param
    
    colors = ensure_list(colors, "colors")
    labels = ensure_list(labels, "labels")
    label_fit = ensure_list(label_fit, "label_fit")
    
    # Use default colors if None is provided
    for i in range(num_datasets):
        if colors[i] is None:
            colors[i] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    
    # Create figure
    if residuals:
        fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=figsize, 
                                             gridspec_kw={'height_ratios': [3, 1]},
                                             sharex=True)
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
    
    # Setup plot
    if title:
        ax_main.set_title(title, fontsize=title_fontsize)
    if xlabel:
        if residuals:
            ax_res.set_xlabel(xlabel, fontsize=axis_fontsize)
        else:
            ax_main.set_xlabel(xlabel, fontsize=axis_fontsize)
    if ylabel:
        ax_main.set_ylabel(ylabel, fontsize=axis_fontsize)
        if residuals:
            ax_res.set_ylabel('Residuals', fontsize=axis_fontsize)
    
    # Set axis limits if provided
    if xlim:
        ax_main.set_xlim(xlim)
    if ylim:
        ax_main.set_ylim(ylim)
    
    # Enable grid if requested
    if grid:
        ax_main.grid(True, linestyle='--', alpha=0.7)
        if residuals:
            ax_res.grid(True, linestyle='--', alpha=0.7)
    
    # Store fit parameters and chi-squared values for display
    fit_params_text = []
    chi_squared_values = []
    
    # Process each dataset
    for i in range(num_datasets):
        x_data = args[2*i]
        y_data = args[2*i+1]
        mask = masks[i]
        
        # Extract data
        if isinstance(x_data, dict):
            x_val = np.array(x_data['value'])
            x_err = x_data.get('error', np.zeros_like(x_val))
            if np.isscalar(x_err):
                x_err = np.full_like(x_val, x_err)
        else:
            x_val = np.array(x_data)
            x_err = np.zeros_like(x_val)
        
        if isinstance(y_data, dict):
            y_val = np.array(y_data['value'])
            y_err = y_data.get('error', np.zeros_like(y_val))
            if np.isscalar(y_err):
                y_err = np.full_like(y_val, y_err)
        else:
            y_val = np.array(y_data)
            y_err = np.zeros_like(y_val)
        
        # Create data label
        data_label = labels[i]
        if data_label is None and labels == [None] * num_datasets:
            data_label = f"Dataset {i+1}"
        
        # Create inverse mask for displaying masked points
        if mask is not None:
            inverse_mask = ~np.array(mask)
        else:
            inverse_mask = np.zeros_like(x_val, dtype=bool)
        
        # Perform fit using the mask
        if show_chi_squared:
            fit_result = perform_fit(x_data, y_data, func, p0=p0[i], 
                                  chi_square=True, mask=mask, 
                                  parameter_names=parameter_names[i],
                                  method=method)
            params = fit_result['parameters']['value']
            params_err = fit_result['parameters']['error']
            chi_squared = fit_result['stats']['chi_square']
            dof = fit_result['stats']['dof']
            reduced_chi_sq = fit_result['stats']['reduced_chi_square']
            chi_squared_values.append((chi_squared, dof, reduced_chi_sq))
        else:
            fit_result = perform_fit(x_data, y_data, func, p0=p0[i], 
                                  mask=mask,
                                  parameter_names=parameter_names[i],
                                  method=method)
            params = fit_result['parameters']['value']
            params_err = fit_result['parameters']['error']
        
        # Format fit parameters for display
        if show_fit_params:
            param_str = []
            param_names = parameter_names[i]
            for j, (name, p, e) in enumerate(zip(param_names, params, params_err)):
                param_str.append(f"{name} = {p:.4g} ± {e:.4g}")
            fit_params_text.append(", ".join(param_str))
        
        # Plot unmasked data points
        if mask is not None:
            # Plot points included in the fit
            ax_main.errorbar(x_val[mask], y_val[mask], xerr=x_err[mask], yerr=y_err[mask], 
                             fmt=fmt, color=colors[i], label=data_label, 
                             markersize=markersize, capsize=capsize)
            
            # Plot masked points if requested
            if show_masked_points and np.any(inverse_mask):
                masked_c = masked_color if masked_color else colors[i]
                ax_main.errorbar(x_val[inverse_mask], y_val[inverse_mask], 
                                 xerr=x_err[inverse_mask], yerr=y_err[inverse_mask], 
                                 fmt=masked_fmt, color=masked_c, alpha=masked_alpha,
                                 markersize=markersize, capsize=capsize,
                                 label=f"{data_label} (excluded)")
        else:
            # Plot all data points if no mask
            ax_main.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt=fmt, 
                             color=colors[i], label=data_label, markersize=markersize,
                             capsize=capsize)
        
        # Plot fit line
        if fit_line:
            # Determine fit range
            if fit_range is None:
                min_x = np.min(x_val)
                max_x = np.max(x_val)
            else:
                min_x, max_x = fit_range
            
            x_fit = np.linspace(min_x, max_x, fit_points)
            y_fit = func(x_fit, *params)
            
            # Create fit label
            fit_label = label_fit[i]
            if fit_label is None and label_fit == [None] * num_datasets:
                fit_label = f"Fit {i+1}"
            
            ax_main.plot(x_fit, y_fit, color=colors[i], linestyle='solid', 
                         label=fit_label, linewidth=linewidth)
            
            # Add confidence interval if requested
            if confidence_interval is not None:
                # Calculate confidence interval
                from scipy.stats import t
                
                alpha = 1.0 - confidence_interval
                n = len(x_val) if mask is None else np.sum(mask)
                p = len(params)
                dof = max(0, n - p)
                
                if dof > 0:
                    t_val = t.ppf(1 - alpha/2, dof)
                    
                    # Standard error of the regression
                    if mask is not None:
                        x_val_fit = x_val[mask]
                        y_val_fit = y_val[mask]
                    else:
                        x_val_fit = x_val
                        y_val_fit = y_val
                    
                    y_pred = func(x_val_fit, *params)
                    ss_residuals = np.sum((y_val_fit - y_pred)**2)
                    mse = ss_residuals / dof if dof > 0 else 0
                    se = np.sqrt(mse)
                    
                    # Design matrix
                    X = np.zeros((n, p))
                    for j in range(p):
                        # Numerical approximation of partial derivatives
                        h = 1e-8
                        new_params = params.copy()
                        new_params[j] += h
                        X[:, j] = (func(x_val_fit, *new_params) - y_pred) / h
                    
                    # Covariance matrix
                    try:
                        covariance = np.linalg.inv(X.T @ X) if dof > 0 else np.eye(p)
                    except np.linalg.LinAlgError:
                        # Handle singular matrix
                        covariance = np.eye(p)
                        print(f"Warning: Singular matrix in confidence interval calculation for dataset {i+1}")
                    
                    # Confidence interval
                    y_err_fit = np.zeros_like(x_fit)
                    for j, x in enumerate(x_fit):
                        X_pred = np.zeros(p)
                        for k in range(p):
                            h = 1e-8
                            new_params = params.copy()
                            new_params[k] += h
                            X_pred[k] = (func(x, *new_params) - func(x, *params)) / h
                        
                        y_err_fit[j] = t_val * se * np.sqrt(X_pred @ covariance @ X_pred.T)
                    
                    ax_main.fill_between(x_fit, y_fit - y_err_fit, y_fit + y_err_fit,
                                        color=colors[i], alpha=0.2)
            
            # Plot residuals if requested
            if residuals:
                if mask is not None:
                    x_val_fit = x_val[mask]
                    y_val_fit = y_val[mask]
                    y_err_fit = y_err[mask]
                else:
                    x_val_fit = x_val
                    y_val_fit = y_val
                    y_err_fit = y_err
                
                y_pred = func(x_val_fit, *params)
                residual = y_val_fit - y_pred
                ax_res.errorbar(x_val_fit, residual, yerr=y_err_fit, fmt=fmt, 
                                color=colors[i], markersize=markersize,
                                capsize=capsize)
                
                # Also show excluded points in residuals plot if requested
                if show_masked_points and mask is not None and np.any(inverse_mask):
                    x_val_excl = x_val[inverse_mask]
                    y_val_excl = y_val[inverse_mask]
                    y_pred_excl = func(x_val_excl, *params)
                    residual_excl = y_val_excl - y_pred_excl
                    y_err_excl = y_err[inverse_mask]
                    masked_c = masked_color if masked_color else colors[i]
                    ax_res.errorbar(x_val_excl, residual_excl, yerr=y_err_excl, 
                                    fmt=masked_fmt, color=masked_c, alpha=masked_alpha,
                                    markersize=markersize, capsize=capsize)
                
                # Add a horizontal line at y=0 for reference
                ax_res.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Display fit parameters and chi-squared if requested
    if show_fit_params:
        for i, param_text in enumerate(fit_params_text):
            ax_main.annotate(param_text, xy=(0.02, 0.98 - i*0.05), xycoords='axes fraction',
                             fontsize=14, va='top', color=colors[i])
    
    if show_chi_squared:
        for i, (chi_sq, dof, red_chi_sq) in enumerate(chi_squared_values):
            chi_sq_text = f"χ²={chi_sq:.4g}, dof={dof}, χ²/dof={red_chi_sq:.4g}"
            y_pos = 0.98 - i*0.05
            if show_fit_params:
                y_pos -= len(fit_params_text) * 0.05
            ax_main.annotate(chi_sq_text, xy=(0.02, y_pos), xycoords='axes fraction',
                            fontsize=14, va='top', color=colors[i])
    
    # Add legend if appropriate
    if any(l is not None for l in labels) or (fit_line and any(l is not None for l in label_fit)):
        ax_main.legend(loc=legend_loc)
    
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show the figure if in a Jupyter notebook
    plt.show()
    
    return fig

# Helper function to detect if we're in a notebook environment
def in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False

# ---------------------------- Statistical Test -----------------------------
def test_comp(a: float, sigma_a: float, b: float, sigma_b: float, 
             method: str = 'normal', alpha: float = 0.05, 
             n_a: int = None, n_b: int = None,
             corr_coef: float = 0.0, output: str = 'full',
             visualize: bool = False) -> Dict:
    """
    Compare two measurements with uncertainties and determine if they are compatible.
    
    Parameters:
    -----------
    a : float
        First measurement value.
    sigma_a : float
        Uncertainty of the first measurement.
    b : float
        Second measurement value.
    sigma_b : float
        Uncertainty of the second measurement.
    method : str, optional (default='normal')
        Method to use for comparison:
        - 'normal': Use normal distribution (default)
        - 'student': Use Student's t-distribution
    alpha : float, optional (default=0.05)
        Significance level for hypothesis testing (default is 5%).
    n_a : int, optional
        Sample size for the first measurement (required for 'student' method).
    n_b : int, optional
        Sample size for the second measurement (required for 'student' method).
    corr_coef : float, optional (default=0.0)
        Correlation coefficient between the two measurements (-1 to 1).
    output : str, optional (default='full')
        Output format:
        - 'full': Return all statistics
        - 'pvalue': Return only p-value
        - 'compatible': Return True/False based on compatibility
    visualize : bool, optional (default=False)
        If True, create a visualization of the comparison.
    
    Returns:
    --------
    dict: Statistics about the comparison including:
        - difference: Absolute difference between measurements
        - sigma_difference: Uncertainty of the difference
        - z_score: Number of standard deviations of difference
        - p_value: Probability that the difference could occur by chance
        - compatible: Boolean indicating if measurements are compatible
        - critical_value: Critical value for the chosen significance level
        - method: Method used for the comparison
    
    Examples:
    ---------
    >>> # Basic comparison
    >>> result = test_comp(5.2, 0.3, 5.5, 0.4)
    >>> print(f"Compatible: {result['compatible']}, p-value: {result['p_value']:.4f}")
    
    >>> # Using Student's t-distribution
    >>> result = test_comp(5.2, 0.3, 5.5, 0.4, method='student', n_a=10, n_b=15)
    
    >>> # With correlation between measurements
    >>> result = test_comp(5.2, 0.3, 5.5, 0.4, corr_coef=0.5)
    
    >>> # With visualization
    >>> test_comp(5.2, 0.3, 5.5, 0.4, visualize=True)
    
    Notes:
    ------
    - For the 'student' method, sample sizes (n_a, n_b) must be provided.
    - The correlation coefficient should be between -1 and 1.
    - The p-value is two-tailed, testing the null hypothesis that a = b.
    """
    # Validate inputs
    if sigma_a <= 0 or sigma_b <= 0:
        raise ValueError("Uncertainties must be positive.")
    if corr_coef < -1 or corr_coef > 1:
        raise ValueError("Correlation coefficient must be between -1 and 1.")
    if method not in ['normal', 'student']:
        raise ValueError("Method must be either 'normal' or 'student'.")
    if method == 'student' and (n_a is None or n_b is None):
        raise ValueError("Sample sizes must be provided for Student's t-test.")
    if output not in ['full', 'pvalue', 'compatible']:
        raise ValueError("Output must be 'full', 'pvalue', or 'compatible'.")
    
    # Calculate difference and its uncertainty
    diff = a - b
    
    # Handle correlated measurements
    if corr_coef != 0:
        sigma_diff = np.sqrt(sigma_a**2 + sigma_b**2 - 2 * corr_coef * sigma_a * sigma_b)
    else:
        sigma_diff = np.sqrt(sigma_a**2 + sigma_b**2)
    
    # Calculate z-score (number of standard deviations)
    z_score = diff / sigma_diff if sigma_diff > 0 else 0
    
    # Calculate p-value based on the chosen method
    if method == 'normal':
        # Using normal distribution
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        critical_value = norm.ppf(1 - alpha/2)
    else:
        # Using Student's t-distribution
        from scipy.stats import t
        
        # Calculate degrees of freedom using Welch-Satterthwaite equation
        if n_a == 1 or n_b == 1:
            dof = 1
        else:
            dof = (sigma_a**2 + sigma_b**2)**2 / ((sigma_a**4 / (n_a - 1)) + (sigma_b**4 / (n_b - 1)))
            dof = max(1, int(dof))
        
        p_value = 2 * (1 - t.cdf(abs(z_score), dof))
        critical_value = t.ppf(1 - alpha/2, dof)
    
    # Determine compatibility
    compatible = p_value >= alpha
    
    # Prepare result based on output format
    if output == 'pvalue':
        return p_value
    elif output == 'compatible':
        return compatible
    else:
        result = {
            'difference': diff,
            'sigma_difference': sigma_diff,
            'z_score': z_score,
            'p_value': p_value,
            'compatible': compatible,
            'critical_value': critical_value,
            'method': method
        }
        
        if method == 'student':
            result['degrees_of_freedom'] = dof
        
        # Add interpretations
        if compatible:
            result['interpretation'] = f"The measurements are compatible at {alpha*100:.1f}% significance level."
        else:
            result['interpretation'] = f"The measurements differ significantly at {alpha*100:.1f}% significance level."
        
        # Add effect size (Cohen's d)
        result['effect_size'] = abs(diff) / np.sqrt((sigma_a**2 + sigma_b**2) / 2)
        if result['effect_size'] < 0.2:
            result['effect_size_interpretation'] = "Negligible effect"
        elif result['effect_size'] < 0.5:
            result['effect_size_interpretation'] = "Small effect"
        elif result['effect_size'] < 0.8:
            result['effect_size_interpretation'] = "Medium effect"
        else:
            result['effect_size_interpretation'] = "Large effect"
    
    # Visualize the comparison if requested
    if visualize:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the measurements with error bars
        ax.errorbar([1], [a], yerr=[sigma_a], fmt='o', color='blue', capsize=5, markersize=8, label='Measurement A')
        ax.errorbar([2], [b], yerr=[sigma_b], fmt='o', color='red', capsize=5, markersize=8, label='Measurement B')
        
        # Add a line connecting the measurements
        ax.plot([1, 2], [a, b], '--', color='gray', alpha=0.5)
        
        # Set labels and title
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['A', 'B'])
        ax.set_ylabel('Value')
        ax.set_title('Comparison of Measurements')
        
        # Add compatibility information
        compatibility_text = "Compatible" if compatible else "Not Compatible"
        color = "green" if compatible else "red"
        ax.text(1.5, max(a, b) + 2*max(sigma_a, sigma_b), 
                f"p-value: {p_value:.4f}\n{compatibility_text}", 
                ha='center', va='center', fontsize=12, color=color,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add legends
        ax.legend()
        
        # Create a second subplot for the distributions
        ax2 = ax.twinx()
        
        # Create x-axis for distributions
        x_min = min(a - 4*sigma_a, b - 4*sigma_b)
        x_max = max(a + 4*sigma_a, b + 4*sigma_b)
        x = np.linspace(x_min, x_max, 1000)
        
        # Plot normal distributions
        from scipy.stats import norm
        ax2.plot(norm.pdf(x, a, sigma_a), x, color='blue', alpha=0.5)
        ax2.plot(norm.pdf(x, b, sigma_b), x, color='red', alpha=0.5)
        
        # Set y-axis invisible
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    return result

def test_comp_advanced(a, sigma_a, b, sigma_b, method='normal', alpha=0.05, 
                      n_a=None, n_b=None, corr_coef=0.0, test_type='equivalence',
                      delta=None, equiv_margin=None, visualize=False):
    """
    Advanced comparison of two measurements with comprehensive statistical tests.
    
    Parameters:
    -----------
    a, sigma_a : float
        First measurement value and its uncertainty.
    b, sigma_b : float
        Second measurement value and its uncertainty.
    method : str, optional (default='normal')
        Statistical method: 'normal', 'student', or 'bootstrap'.
    alpha : float, optional (default=0.05)
        Significance level (Type I error rate).
    n_a, n_b : int, optional
        Sample sizes for Student's t-test.
    corr_coef : float, optional (default=0.0)
        Correlation coefficient between measurements (-1 to 1).
    test_type : str, optional (default='equivalence')
        Type of test:
        - 'difference': Test if values are different (traditional null hypothesis)
        - 'equivalence': Test if values are equivalent within a margin
        - 'non-inferiority': Test if a is not worse than b
        - 'superiority': Test if a is better than b
    delta : float, optional
        Clinically significant difference for non-inferiority/superiority tests.
    equiv_margin : float, optional
        Equivalence margin for equivalence tests.
    visualize : bool, optional (default=False)
        Generate visualization of the comparison.
    
    Returns:
    --------
    dict: Comprehensive test results.
    
    Examples:
    ---------
    >>> # Equivalence test
    >>> test_comp_advanced(5.2, 0.3, 5.5, 0.4, test_type='equivalence', equiv_margin=0.5)
    
    >>> # Non-inferiority test
    >>> test_comp_advanced(5.2, 0.3, 5.5, 0.4, test_type='non-inferiority', delta=0.3)
    """
    # Validate inputs
    if test_type not in ['difference', 'equivalence', 'non-inferiority', 'superiority']:
        raise ValueError("test_type must be 'difference', 'equivalence', 'non-inferiority', or 'superiority'")
    
    # Check for required parameters based on test type
    if test_type == 'equivalence' and equiv_margin is None:
        raise ValueError("equiv_margin must be provided for equivalence tests")
    if test_type in ['non-inferiority', 'superiority'] and delta is None:
        raise ValueError("delta must be provided for non-inferiority and superiority tests")
    
    # Initialize results dictionary
    results = {
        'measurement_a': {'value': a, 'uncertainty': sigma_a},
        'measurement_b': {'value': b, 'uncertainty': sigma_b},
        'difference': a - b,
        'method': method,
        'alpha': alpha,
        'test_type': test_type
    }
    
    # Calculate standard error of the difference
    if corr_coef == 0:
        sigma_diff = np.sqrt(sigma_a**2 + sigma_b**2)
    else:
        # When measurements are correlated, adjust standard error calculation
        sigma_diff = np.sqrt(sigma_a**2 + sigma_b**2 - 2 * corr_coef * sigma_a * sigma_b)
    
    results['standard_error'] = sigma_diff
    
    # Calculate test statistic
    diff = a - b
    z_score = diff / sigma_diff if sigma_diff > 0 else 0
    results['z_score'] = z_score
    
    # Perform appropriate test based on specified method
    if method == 'normal':
        from scipy.stats import norm
        
        # Standard two-sided test
        results['p_value_two_sided'] = 2 * (1 - norm.cdf(abs(z_score)))
        
        # One-sided tests
        results['p_value_less'] = norm.cdf(z_score)       # P(a < b)
        results['p_value_greater'] = 1 - norm.cdf(z_score) # P(a > b)
        
        # Critical values
        results['critical_value_two_sided'] = norm.ppf(1 - alpha/2)
        results['critical_value_one_sided'] = norm.ppf(1 - alpha)
        
        # Calculate confidence interval
        margin = results['critical_value_two_sided'] * sigma_diff
        results['confidence_interval'] = (diff - margin, diff + margin)
        
    elif method == 'student':
        from scipy.stats import t
        
        # Ensure sample sizes are provided
        if n_a is None or n_b is None:
            raise ValueError("Sample sizes must be provided for Student's t-test")
        
        # Calculate degrees of freedom (Welch-Satterthwaite equation)
        if n_a <= 1 or n_b <= 1:
            dof = 1
        else:
            dof = (sigma_a**2 + sigma_b**2)**2 / ((sigma_a**4 / (n_a - 1)) + (sigma_b**4 / (n_b - 1)))
            dof = max(1, int(dof))
        
        results['degrees_of_freedom'] = dof
        
        # Calculate p-values
        results['p_value_two_sided'] = 2 * (1 - t.cdf(abs(z_score), dof))
        results['p_value_less'] = t.cdf(z_score, dof)
        results['p_value_greater'] = 1 - t.cdf(z_score, dof)
        
        # Critical values
        results['critical_value_two_sided'] = t.ppf(1 - alpha/2, dof)
        results['critical_value_one_sided'] = t.ppf(1 - alpha, dof)
        
        # Calculate confidence interval
        margin = results['critical_value_two_sided'] * sigma_diff
        results['confidence_interval'] = (diff - margin, diff + margin)
        
    elif method == 'bootstrap':
        # Implement bootstrap method for comparison
        # This is a non-parametric approach useful when the distribution is unknown
        if n_a is None or n_b is None:
            raise ValueError("Sample sizes must be provided for bootstrap method")
        
        # Generate bootstrap samples
        np.random.seed(42)  # for reproducibility
        n_bootstrap = 10000
        
        # Create synthetic data based on the provided means and standard deviations
        data_a = np.random.normal(a, sigma_a, size=(n_bootstrap, n_a))
        data_b = np.random.normal(b, sigma_b, size=(n_bootstrap, n_b))
        
        # Calculate bootstrap means
        means_a = np.mean(data_a, axis=1)
        means_b = np.mean(data_b, axis=1)
        
        # Calculate bootstrap differences
        bootstrap_diffs = means_a - means_b
        
        # Calculate p-values
        results['p_value_two_sided'] = np.mean(abs(bootstrap_diffs) >= abs(diff))
        results['p_value_less'] = np.mean(bootstrap_diffs <= 0)
        results['p_value_greater'] = np.mean(bootstrap_diffs >= 0)
        
        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_diffs, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        results['confidence_interval'] = (lower_bound, upper_bound)
    
    else:
        raise ValueError("Method must be 'normal', 'student', or 'bootstrap'")
    
    # Determine test conclusion based on test_type
    if test_type == 'difference':
        # Traditional null hypothesis testing
        results['conclusion'] = {
            'reject_null': results['p_value_two_sided'] < alpha,
            'interpretation': f"The measurements are {'significantly different' if results['p_value_two_sided'] < alpha else 'not significantly different'} at alpha={alpha}.",
            'p_value': results['p_value_two_sided']
        }
    
    elif test_type == 'equivalence':
        # Two one-sided tests (TOST) for equivalence
        lower_p = 1 - norm.cdf((diff + equiv_margin) / sigma_diff) if method == 'normal' else 1 - t.cdf((diff + equiv_margin) / sigma_diff, dof)
        upper_p = norm.cdf((diff - equiv_margin) / sigma_diff) if method == 'normal' else t.cdf((diff - equiv_margin) / sigma_diff, dof)
        
        results['equivalence_tests'] = {
            'lower_bound_p': lower_p,
            'upper_bound_p': upper_p,
            'equiv_margin': equiv_margin
        }
        
        # Conclusion for equivalence test
        reject_null = max(lower_p, upper_p) < alpha
        results['conclusion'] = {
            'reject_null': reject_null,
            'interpretation': f"The measurements are {'equivalent' if reject_null else 'not equivalent'} within the margin of {equiv_margin} at alpha={alpha}.",
            'p_value': max(lower_p, upper_p)
        }
    
    elif test_type == 'non-inferiority':
        # Non-inferiority test (one-sided)
        # Null hypothesis: a is inferior to b by at least delta
        # Alternative hypothesis: a is non-inferior to b (a - b > -delta)
        non_inf_p = 1 - results['p_value_less'] if diff > -delta else 1.0
        
        results['non_inferiority_test'] = {
            'non_inferiority_margin': delta,
            'non_inferiority_p': non_inf_p
        }
        
        # Conclusion for non-inferiority test
        reject_null = non_inf_p < alpha
        results['conclusion'] = {
            'reject_null': reject_null,
            'interpretation': f"Measurement A is {'non-inferior' if reject_null else 'inferior'} to measurement B with a margin of {delta} at alpha={alpha}.",
            'p_value': non_inf_p
        }
    
    elif test_type == 'superiority':
        # Superiority test (one-sided)
        # Null hypothesis: a is not superior to b by at least delta
        # Alternative hypothesis: a is superior to b (a - b > delta)
        sup_p = results['p_value_greater'] if diff > delta else 1.0
        
        results['superiority_test'] = {
            'superiority_margin': delta,
            'superiority_p': sup_p
        }
        
        # Conclusion for superiority test
        reject_null = sup_p < alpha
        results['conclusion'] = {
            'reject_null': reject_null,
            'interpretation': f"Measurement A is {'superior' if reject_null else 'not superior'} to measurement B with a margin of {delta} at alpha={alpha}.",
            'p_value': sup_p
        }
    
    # Generate visualization if requested
    if visualize:
        results['visualization'] = _generate_visualization(a, sigma_a, b, sigma_b, 
                                                         test_type, results, 
                                                         delta, equiv_margin)
    
    return results

def _generate_visualization(a, sigma_a, b, sigma_b, test_type, results, delta, equiv_margin):
    """
    Generate visualization for the comparison test.
    
    Parameters:
    -----------
    a, sigma_a : float
        First measurement value and its uncertainty.
    b, sigma_b : float
        Second measurement value and its uncertainty.
    test_type : str
        Type of test performed.
    results : dict
        Results dictionary from the main function.
    delta : float or None
        Clinically significant difference for non-inferiority/superiority tests.
    equiv_margin : float or None
        Equivalence margin for equivalence tests.
    
    Returns:
    --------
    str: Base64 encoded image or path to saved image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot probability distributions
    x = np.linspace(min(a, b) - 3*max(sigma_a, sigma_b), 
                    max(a, b) + 3*max(sigma_a, sigma_b), 1000)
    
    # Normal distributions for each measurement
    y_a = stats.norm.pdf(x, a, sigma_a)
    y_b = stats.norm.pdf(x, b, sigma_b)
    
    ax.plot(x, y_a, 'b-', label=f'Measurement A: {a:.2f} ± {sigma_a:.2f}')
    ax.plot(x, y_b, 'r-', label=f'Measurement B: {b:.2f} ± {sigma_b:.2f}')
    
    # Plot vertical lines for means
    ax.axvline(a, color='b', linestyle='--', alpha=0.5)
    ax.axvline(b, color='r', linestyle='--', alpha=0.5)
    
    # Add shaded regions and markers based on test type
    if test_type == 'equivalence':
        # Shade equivalence region
        ax.axvspan(b - equiv_margin, b + equiv_margin, color='g', alpha=0.2, 
                   label=f'Equivalence region (±{equiv_margin:.2f})')
        
    elif test_type == 'non-inferiority':
        # Shade non-inferiority region
        ax.axvspan(b - delta, max(x), color='g', alpha=0.2, 
                   label=f'Non-inferiority region (>{b-delta:.2f})')
        
    elif test_type == 'superiority':
        # Shade superiority region
        ax.axvspan(b + delta, max(x), color='g', alpha=0.2, 
                   label=f'Superiority region (>{b+delta:.2f})')
    
    # Add confidence interval
    if 'confidence_interval' in results:
        ci_low, ci_high = results['confidence_interval']
        ax.plot([ci_low, ci_high], [0.05, 0.05], 'k-', linewidth=2)
        ax.plot([ci_low], [0.05], 'k|', markersize=10)
        ax.plot([ci_high], [0.05], 'k|', markersize=10)
        ax.text((ci_low + ci_high)/2, 0.07, f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]', 
                ha='center', va='bottom')
    
    # Add test conclusion
    if 'conclusion' in results:
        conclusion = results['conclusion']['interpretation']
        ax.text(0.5, 0.95, conclusion, 
                ha='center', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Measurement Value')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Statistical Comparison: {test_type.capitalize()} Test')
    ax.legend(loc='upper right')
    
    # Remove the y-axis for cleaner appearance
    ax.yaxis.set_ticklabels([])
    
    # Save the figure or return as base64
    plt.tight_layout()
    
    # Save figure to a temporary file and return the path
    # In a real application, you might want to use IO to convert to base64
    # or save to a specific location
    fig_path = 'comparison_visualization.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return fig_path