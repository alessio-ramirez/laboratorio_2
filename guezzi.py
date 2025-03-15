import numpy as np
import matplotlib.pyplot as plt
import pyperclip
import sympy as sp
import math
import re
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR

def error_prop(f, variables, covariances=None, use_covariance=False, copy_latex=False, round_latex=3):
    # Convert variables to lists of (value, error) tuples
    converted_vars = []
    for var in variables:
        if isinstance(var, dict):
            var_list = [(key, var[key]) for key in var.keys()]
            converted_vars.append(var_list)
        elif isinstance(var, list):
            var_list = []
            for item in var:
                if isinstance(item, (int, float)):
                    var_list.append((item, 0.0))
                else:
                    var_list.append((item[0], item[1]))
            converted_vars.append(var_list)
        else:
            raise TypeError("Each variable must be a dict or list.")

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
    Perform a fit on data x and y with a given function, considering errors in x and/or y.

    Parameters:
    x (array or dict): Independent variable data. If a dict, must have 'value' and optionally 'error'.
    y (array or dict): Dependent variable data. If a dict, must have 'value' and optionally 'error'.
    func (callable): The model function, must have signature f(x, *params).
    p0 (list): Initial guess for the parameters.

    Returns:
    tuple: (parameters, errors) as numpy arrays.
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
        # Use Orthogonal Distance Regression (ODR)
        odr_model = Model(func)
        data = RealData(x_val, y_val, sx=x_err, sy=y_err)
        odr = ODR(data, odr_model, beta0=p0)
        output = odr.run()
        params = output.beta
        params_err = output.sd_beta
    else:
        # Use curve_fit (ordinary or weighted least squares)
        if np.any(y_err != 0):
            # Handle zero errors to avoid division by zero
            y_err = np.where(y_err == 0, 1e-10, y_err)
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0, sigma=y_err, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(func, x_val, y_val, p0=p0)
        params = popt
        params_err = np.sqrt(np.diag(pcov))
    
    return params, params_err