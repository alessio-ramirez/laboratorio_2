"""
guezzi.py - A scientific utility library to help Bicocca's physics students, like Mr Guezzi did.
the documentation is not updated, and the code is generally inefficient, but it works :D.
convert every set of measurements with create_dataset before using other functions,
the only one that returns (value, error) format (so not usable by the other functions) is error_prop.
the fits are made with ODR or least squares, in both cases the chi square is intended to be a useful
test only for fits linear with the parameters, for non linear fits the code might work, but the chi square
test will be useless.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyperclip
import sympy as sp
import math
import re
from itertools import combinations
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
import inspect
from typing import Union, List, Dict, Callable, Optional

# ---------------------------- Dataset Creation ------------------------------
def create_dataset(values: Union[float, List[float], Dict], errors: Union[float, List[float]] = None, magnitude=0) -> Dict[str, np.ndarray]:
    """
    Create a standardized dataset dictionary for measurements with errors.

    Parameters:
    values : float, list, or dict
        - Single measurement (float)
        - List of measured values
        - Dictionary with {value: error} pairs (if provided errors should be None)
    errors : float or list, optional
        Single error value (same error for each measured value) or list of errors

    Returns:
    dict: {'value': np.array, 'error': np.array}
        Standard format of the measurements for this library

    Examples:
    >>> create_dataset(5.0, 1.0)  # Single measurement
    {'value': array([5.]), 'error': array([1.])}
    
    >>> create_dataset([2, 4], 0.5)  # Multiple measurements with uniform error
    {'value': array([2., 4.]), 'error': array([0.5, 0.5])}
    
    >>> create_dataset({10: 0.1, 20: 0.2})  # Dictionary input
    {'value': array([10., 20.]), 'error': array([0.1, 0.2])}
    """
    # Scalar input case
    if isinstance(values, (int, float)):
        values = [float(values)] # Add a float just because
    # Easy dictionary case
    elif isinstance(values, dict):
        if errors is not None:
            raise ValueError("Errors must be None when using dictionary input")
        return {
            'value' : np.array(list(values.keys()), dtype=float),
            'error' : np.array(list(values.values()), dtype=float)
        }
    
    # Convert values to a float array
    values = np.array(values, dtype=float)
    
    # Determine the errors array
    if errors is not None:
        errors = np.array(errors, dtype=float)
        if errors.ndim == 0:  # Scalar error
            errors = np.full(values.shape, errors.item())
        else:
            if errors.shape != values.shape:
                raise ValueError("Errors length have to be the same of Values")
    else:
        errors = np.zeros_like(values) # Zero errors if not provided
    
    return {'value': values*10**(magnitude), 'error': errors*10**(magnitude)}

# ---------------------------- Error Propagation -----------------------------
def error_prop(f: Callable, variables: List[Dict], covariances: Optional[Dict] = None, 
               use_covariance: bool = False, copy_latex: bool = False, round_latex: int = 3) -> List[tuple]:
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
        converted_vars.append(list(zip(values, errors)))
    
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

# ---------------------------- Curve Fitting --------------------------------
def perform_fit(x: Union[Dict, np.ndarray], y: Union[Dict, np.ndarray], 
                func: Callable, p0: Optional[List[float]] = None, chi_square=False) -> tuple:
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
    >>> params, errors = perform_fit(x_data, y_data, model)  # Auto p0 = [1,1]
    """
    # Auto-generate p0 if not provided
    if p0 is None:
        sig = inspect.signature(func)
        n_params = len(sig.parameters) - 1  # Subtract x parameter
        p0 = [1.0] * n_params

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
        if chi_square:
            chi_squared = output.sum_square
            dof = len(y_val) - len(params)
            return params, params_err, chi_squared, dof

    else:
        # Use curve_fit directly
        if np.any(y_err != 0):
            y_err = np.where(y_err == 0, 1e-10, y_err)
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0, sigma=y_err, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0)
        params = popt
        params_err = np.sqrt(np.diag(pcov))
        if chi_square:
            y_pred = func(x_val, *params)
            if np.any(y_err != 0):
                residuals = (y_val - y_pred) / y_err
            else:
                residuals = y_val - y_pred  # Unweighted residuals
            chi_squared = np.sum(residuals**2)
            dof = len(y_val) - len(params)
            return params, params_err, chi_squared, dof

    return params, params_err

# ---------------------------- Visualization --------------------------------
def create_best_fit_line(*args: Union[Dict, np.ndarray], func: Callable, 
                        p0: Optional[Union[List[float], List[List[float]]]] = None,
                        xlabel: str = None, ylabel: str = None, title: str = None,
                        colors: List[str] = None, labels: List[str] = None, 
                        fit_line: bool = True, label_fit: List[str] = None,
                        together = True) -> None:
    """
    Enhanced to handle flexible p0 formats and auto-generate initial parameters.

    Parameters:
    p0 : Can be:
        - Single list for one dataset (e.g., [1,1])
        - List of lists for multiple datasets (e.g., [[1,1], [2,2]])
        - None for auto-generated 1's

    Example:
    >>> x = create_dataset([1,2,3], 0.1)
    >>> y = create_dataset([2,4,6], 0.2)
    >>> create_best_fit_line(x, y, func=lambda x,a,b: a*x+b)  # Auto p0
    """
    # Parameter normalization
    num_datasets = len(args) // 2
    if p0 is None:
        sig = inspect.signature(func)
        n_params = len(sig.parameters) - 1  # Infer from function
        p0 = [[1.0]*n_params for _ in range(num_datasets)]
    elif not isinstance(p0[0], (list, np.ndarray)):  # Single dataset format
        p0 = [p0]

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
    label_fit = label_fit or [None] * num_datasets
    
    if together:
        plt.figure()
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
    
    for i in range(num_datasets):
        if not together:
            plt.figure()
            if title: plt.title(title)
            if xlabel: plt.xlabel(xlabel)
            if ylabel: plt.ylabel(ylabel)
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
        if not together:
            plt.legend()
            plt.show()
            
    if together:
        plt.legend()
        plt.show()

# ---------------------------- Statistical Test -----------------------------
def test_comp(a: float, sigma_a: float, b: float, sigma_b: float) -> float:
    """
    Compare two measurements with error propagation.
    
    Returns:
    float: Probability that the difference exceeds |a-b|
    """
     #si suppongono le variabili indipendenti
    sigma_eq = math.sqrt(sigma_a**2 + sigma_b**2)
    t_sigma =  abs(a-b)/sigma_eq
    t = sp.symbols('t')
    prob_entro_t_sigma = sp.erf(t / sp.sqrt(2)).subs(t, t_sigma).evalf()
    if prob_entro_t_sigma > 0.95:
        print("Ci che hai fatto")
    print("probabilità entro t sigma =", prob_entro_t_sigma)
    return 1 - prob_entro_t_sigma

# ---------------------------- Latex Formatting -----------------------------
def latex_table(*args, orientation="h"):
    """
    Generate a LaTeX table from provided datasets with homogeneous order of magnitude for each dataset,
    and copy it to the clipboard. Values are rounded based on the error's first significant digit.

    Parameters:
    *args : str, dict pairs
        Pairs of names and datasets. Each dataset is a dict with 'value' and 'error' arrays.
    orientation : str, optional
        'h' for horizontal (names as row headers), 'v' for vertical (names as column headers).

    Returns:
    str: LaTeX table code.
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
        values = dataset['value']
        errors = dataset['error']
        
        # Determine scaling factor based on the maximum error or value
        if np.all(errors == 0):
            # Use max value if all errors are zero
            max_val = np.max(np.abs(values)) if len(values) > 0 else 0.0
            if np.isclose(max_val, 0.0, atol=1e-12):
                scaling_factor = 1.0
                exponent = 0
            else:
                rounded_max = float(f"{max_val:.1g}")
                if rounded_max == 0:
                    scaling_factor = 1.0
                    exponent = 0
                else:
                    exponent = np.floor(np.log10(rounded_max))
                    scaling_factor = 10 ** exponent
        else:
            # Use max error
            max_error = np.max(np.abs(errors))
            rounded_max = float(f"{max_error:.1g}")
            if rounded_max == 0:
                scaling_factor = 1.0
                exponent = 0
            else:
                exponent = np.floor(np.log10(rounded_max))
                scaling_factor = 10 ** exponent
        
        # Handle potential division by zero or invalid scaling factor
        if scaling_factor == 0:
            scaling_factor = 1.0
            exponent = 0
        
        # Create the new name with magnitude
        if exponent != 0:
            magnitude_str = f" $\\times 10^{{{int(exponent)}}}$"
        else:
            magnitude_str = ""
        new_name = f"{name}{magnitude_str}"
        
        # Process each entry in the dataset
        entries = []
        for v, e in zip(values, errors):
            scaled_v = v / scaling_factor
            scaled_e = e / scaling_factor
            
            if np.isclose(scaled_e, 0.0, atol=1e-12):
                # No error, format value with up to 3 significant digits
                entries.append(f"{scaled_v:.3g}")
            else:
                # Round error to one significant digit
                rounded_e = float(f"{scaled_e:.1g}")
                # Determine decimal places based on rounded error
                if rounded_e == 0:
                    decimal_places = 0
                else:
                    exp_e = np.floor(np.log10(abs(rounded_e))) if rounded_e != 0 else 0
                    decimal_places = int(-exp_e) if exp_e < 0 else 0
                
                # Round value to the determined decimal places
                rounded_v = round(scaled_v, decimal_places)
                
                # Format value and error strings
                str_v = f"{rounded_v:.{decimal_places}f}".rstrip('0').rstrip('.') if decimal_places !=0 else f"{int(rounded_v)}"
                str_e = f"{rounded_e:.{decimal_places}f}".rstrip('0').rstrip('.') if decimal_places !=0 else f"{int(rounded_e)}"
                
                entries.append(f"${str_v} \\pm {str_e}$")
        
        formatted_data.append((new_name, entries))
    
    # Generate LaTeX code based on orientation
    if orientation == 'h':
        # Horizontal: each name is a row with its entries
        columns = 'l' + ' c' * n_entries
        latex = f"\\begin{{tabular}}{{{columns}}}\n\\hline\n"
        for name, entries in formatted_data:
            row = [name] + entries
            latex += " & ".join(row) + " \\\\\n"
        latex += "\\hline\n\\end{tabular}"
    elif orientation == 'v':
        # Vertical: names are column headers, entries are in columns
        num_columns = len(pairs)
        latex = f"\\begin{{tabular}}{{{'|c' * num_columns}}}\n\\hline\n"
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