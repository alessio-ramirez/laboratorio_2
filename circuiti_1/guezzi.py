import numpy as np
import matplotlib.pyplot as plt
import pyperclip
import sympy as sp
import math
import re 
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
import inspect

def create_dataset(values, errors=None):
    """
    Create a standardized dataset dictionary for measurements.
    
    Parameters:
    values (array-like or dict): Array of measured values or a dictionary where keys are values and values are errors.
    errors (array-like/float, optional): Array of errors or a single scalar error. Default is None.
    
    Returns:
    dict: {'value': np.array, 'error': np.array} with float arrays.
    
    """
    # Handle case where values is a dictionary
    if isinstance(values, dict):
        if errors is not None:
            raise ValueError("If values is a dictionary, errors must be None")
        # Convert keys and values to float arrays
        value_arr = np.array(list(values.keys()), dtype=float)
        error_arr = np.array(list(values.values()), dtype=float)
        if len(value_arr) != len(error_arr):
            raise ValueError("Dictionary keys and values must have the same length")
        return {'value': value_arr, 'error': error_arr}
    
    # Convert values to a float array
    values = np.asarray(values, dtype=float)
    
    # Determine the errors array
    if errors is not None:
        errors = np.asarray(errors, dtype=float)
        if errors.ndim == 0:  # Scalar error
            errors = np.full(values.shape, errors.item())
        else:
            if errors.shape != values.shape:
                raise ValueError("Errors must be a scalar or have the same shape as values")
    else:
        errors = np.zeros_like(values)
    
    return {'value': values, 'error': errors}


def error_prop(f, variables, covariances=None, use_covariance=False, copy_latex=False, round_latex=3):
    """
    Calculate error propagation through a function using partial derivatives
    
    Parameters:
    f (callable): Function to propagate errors through
    variables (list of dicts): List of {'value': [], 'error': []} datasets
    covariances (dict): Precomputed covariances between variables
    use_covariance (bool): Calculate covariances from data
    copy_latex (bool): Copy LaTeX formula to clipboard
    round_latex (int): Rounding digits for LaTeX output
    
    Returns:
    list: [(result, error)] for each data point
    
    Example:
    >>> f = lambda x, y: x*y
    >>> x_data = create_dataset([1,2], 0.1)
    >>> y_data = create_dataset([3,4], 0.2)
    >>> error_prop(f, [x_data, y_data])
    """
    converted_vars = []
    for var in variables:
        if not isinstance(var, dict) or 'value' not in var or 'error' not in var:
            raise TypeError("Each variable must be a dict with 'value' and 'error' keys.")
        values = var['value']
        errors = var['error']
        if isinstance(errors, (int, float)):
            errors = [errors] * len(values)
        if len(errors) != len(values):
            raise ValueError(f"Errors length ({len(errors)}) must match values ({len(values)}).")
        converted_vars.append(list(zip(values, errors)))
    
    # Rest of the function remains the same...
    # Validate data points
    num_data_points = len(converted_vars[0])
    for var in converted_vars:
        if len(var) != num_data_points:
            raise ValueError("All variables must have the same number of data points.")
    
    # Compute covariances if needed
    if use_covariance:
        n_vars = len(converted_vars)
        computed_cov = {}
        var_values = []
        for var in converted_vars:
            values = [point[0] for point in var]
            var_values.append(values)
        for i, j in combinations(range(n_vars), 2):
            values_i = var_values[i]
            values_j = var_values[j]
            mean_i = sum(values_i) / num_data_points
            mean_j = sum(values_j) / num_data_points
            cov = sum((xi - mean_i) * (xj - mean_j) for xi, xj in zip(values_i, values_j))
            computed_cov[(i, j)] = cov / (num_data_points - 1)
        covariances = computed_cov
    else:
        covariances = {(min(i, j), max(i, j)): val for (i, j), val in (covariances or {}).items()}
    
    # Create symbolic variables and derivatives
    n = len(converted_vars)
    symbols = sp.symbols(f'x_0:{n}')
    f_sym = f(*symbols)
    derivatives = [f_sym.diff(sym) for sym in symbols]
    
    # Generate symbolic error formula
    sigma_syms = sp.symbols(f'sigma_0:{n}')
    sum_term = sum((derivatives[i] * sigma_syms[i])**2 for i in range(n))
    cov_term = 0
    if use_covariance:
        cov_terms = [2 * derivatives[i] * derivatives[j] * sp.symbols(f'cov_{i}{j}')
                     for i in range(n) for j in range(i+1, n)]
        cov_term = sum(cov_terms)
    sigma_f_sym = sp.sqrt(sum_term + cov_term)
    
    # Handle LaTeX output
    if copy_latex:
        if not pyperclip:
            raise ImportError("pyperclip is required for LaTeX clipboard copy")
        
        # Evaluate the expressions to a numerical value with 3 significant digits.
        f_sym_rounded = sp.N(f_sym, round_latex)
        sigma_f_sym_rounded = sp.N(sigma_f_sym, round_latex)
        
        # Convert the rounded expressions to LaTeX.
        f_latex = sp.latex(f_sym_rounded, mode="equation")
        sigma_latex = sp.latex(sigma_f_sym_rounded, mode="equation")
        
        pyperclip.copy(f"Function:\n{f_latex}\n\nPropagated Error:\n{sigma_latex}")

    
    # Calculate results for each data point
    results = []
    for dp_idx in range(num_data_points):
        values = [var[dp_idx][0] for var in converted_vars]
        errors = [var[dp_idx][1] for var in converted_vars]
        
        # Calculate function value
        subs = dict(zip(symbols, values))
        f_value = f_sym.subs(subs).evalf()
        f_value = float(f_value) if isinstance(f_value, sp.Expr) else f_value
        
        # Calculate error components
        deriv_values = [float(deriv.subs(subs).evalf()) for deriv in derivatives]
        
        var_term = sum((dv**2) * (err**2) for dv, err in zip(deriv_values, errors))
        cov_term = 0.0
        if use_covariance:
            for (i, j), cov in covariances.items():
                cov_term += 2 * deriv_values[i] * deriv_values[j] * cov
        
        sigma_f = sp.sqrt(var_term + cov_term).evalf()
        sigma_f = float(sigma_f) if sigma_f.is_real else complex(sigma_f)
        
        results.append((f_value, sigma_f))
    
    return results


def perform_fit(x, y, func, p0):
    """
    Perform curve fitting with automatic error-aware method selection
    
    Parameters:
    x: Independent variable data (array or dataset dict)
    y: Dependent variable data (array or dataset dict)
    func (callable): Model function (curve_fit-style: f(x, *params))
    p0 (list): Initial parameter guesses
    
    Returns:
    tuple: (parameters array, parameter_errors array)
    
    Notes:
    - Uses ODR if x errors are present, otherwise uses curve_fit
    - Handles both curve_fit and ODR function signatures automatically
    
    Example:
    >>> model = lambda x, a, b: a*x + b
    >>> x_data = create_dataset([1,2,3], 0.1)
    >>> y_data = create_dataset([2,4,6], 0.2)
    >>> params, errors = perform_fit(x_data, y_data, model, [1,1])
    """
    # Process x data
    if isinstance(x, dict):
        x_val = np.asarray(x['value'])
        x_err = np.asarray(x['error']) if 'error' in x else np.zeros_like(x_val)
    else:
        x_val = np.asarray(x)
        x_err = np.zeros_like(x_val)
    
    # Process y data
    if isinstance(y, dict):
        y_val = np.asarray(y['value'])
        y_err = np.asarray(y['error']) if 'error' in y else np.zeros_like(y_val)
    else:
        y_val = np.asarray(y)
        y_err = np.zeros_like(y_val)
    
    # Determine fitting method
    use_odr = np.any(x_err != 0)

    if use_odr:
        # Wrap curve_fit-style functions for ODR
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        # Check if function is in curve_fit format (x first)
        if len(params) > 0 and params[0].name == 'x':
            # Create a wrapper for ODR
            def odr_wrapper(beta, x):
                return func(x, *beta)
            fit_func = odr_wrapper
        else:
            fit_func = func  # Assume already ODR-compatible

        # Use ODR with the wrapped function
        odr_model = Model(fit_func)
        data = RealData(x_val, y_val, sx=x_err, sy=y_err)
        odr = ODR(data, odr_model, beta0=p0)
        output = odr.run()
        params = output.beta
        params_err = output.sd_beta

    else:
        # Use curve_fit directly
        if np.any(y_err != 0):
            y_err = np.where(y_err == 0, 1e-10, y_err)
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0, sigma=y_err, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0)
        params = popt
        params_err = np.sqrt(np.diag(pcov))

    return params, params_err

def create_best_fit_line(*args, func, p0, xlabel=None, ylabel=None, title=None, colors=None, labels=None, fit_line=True, label_fit=None):
    """
    Create a plot with data points and best-fit lines
    
    Parameters:
    *args: Alternating x-y data pairs (e.g., x1, y1, x2, y2)
    func: Model function for fitting
    p0: Initial parameter guesses (list for each dataset)
    xlabel/ylabel: Axis labels
    title: Plot title
    colors: Colors for each dataset
    labels: Legend labels for datasets
    fit_line: Whether to plot fit lines
    
    Example:
    >>> x = create_dataset([1,2,3], 0.1)
    >>> y = create_dataset([2,4,6], 0.2)
    >>> create_best_fit_line(x, y, func=linear_func, p0=[1,1])
    """
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be pairs of x and y datasets.")
    num_datasets = len(args) // 2
    
    # Handle parameters
    if not isinstance(p0, (list, tuple)):
        p0 = [p0] * num_datasets
    elif len(p0) != num_datasets:
        raise ValueError("p0 length must match number of datasets.")
    
    # Defaults
    colors = colors or [None] * num_datasets
    labels = labels or [None] * num_datasets
    
    plt.figure()
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    
    for i in range(num_datasets):
        x_data = args[2*i]
        y_data = args[2*i+1]
        
        # Extract data
        x_val = np.array(x_data['value'])
        x_err = x_data.get('error', 0.0)
        x_err = x_err if np.isscalar(x_err) else np.array(x_err)
        
        y_val = np.array(y_data['value'])
        y_err = y_data.get('error', 0.0)
        y_err = y_err if np.isscalar(y_err) else np.array(y_err)
        
        # Perform fit
        params, params_err = perform_fit(x_data, y_data, func, p0[i])
        
        # Plot data
        plt.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt='o', 
                    color=colors[i], label=labels[i])
        
        # Plot fit
        if fit_line:
            x_fit = np.linspace(x_val.min(), x_val.max(), 1000)
            plt.plot(x_fit, func(x_fit, *params), 
                    color=colors[i], linestyle='--', label=label_fit[i])
    
    plt.legend()
    plt.show()