import numpy as np
import sympy as sp
import pyperclip
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
from collections import OrderedDict

class ExperimentData:
    """Class for managing experimental data with error propagation capabilities"""
    def __init__(self):
        self.variables = OrderedDict()
        self.covariances = {}
        self._symbols = {}
        self._symbolic_expr = None
        
    def add_variable(self, name, values, errors=None):
        """Add a measured variable with associated errors"""
        values = np.asarray(values)
        
        if errors is None:
            errors = np.zeros_like(values)
        elif np.isscalar(errors):
            errors = np.full_like(values, errors)
        else:
            errors = np.asarray(errors)
            
        if values.shape != errors.shape:
            raise ValueError("Values and errors must have same shape")
            
        self.variables[name] = {'values': values, 'errors': errors}
        self._symbols[name] = sp.Symbol(f'x_{name}')
        
    def set_covariance(self, var1, var2, covariance):
        """Set covariance between two variables"""
        self.covariances[(var1, var2)] = covariance
        
    def error_propagation(self, func, use_covariance=False, copy_latex=False):
        """Perform symbolic error propagation analysis"""
        # Create symbolic expression
        sym_vars = [self._symbols[name] for name in self.variables]
        expr = func(*sym_vars)
        self._symbolic_expr = expr
        
        # Calculate partial derivatives
        derivatives = {name: expr.diff(var) for name, var in zip(self.variables, sym_vars)}
        
        # Build error formula
        error_terms = []
        for name in self.variables:
            error_terms.append((derivatives[name] * sp.Symbol(f'σ_{name}'))**2)
            
        if use_covariance:
            for (var1, var2), cov in self.covariances.items():
                term = 2 * derivatives[var1] * derivatives[var2] * sp.Symbol(f'cov_{var1}_{var2}')
                error_terms.append(term)
                
        total_error = sp.sqrt(sum(error_terms))
        
        # Prepare numeric evaluation
        results = []
        n_points = len(next(iter(self.variables.values()))['values'])
        
        for i in range(n_points):
            # Substitute numeric values
            subs = {var: self.variables[name]['values'][i] 
                   for name, var in zip(self.variables, sym_vars)}
            
            # Calculate function value
            f_value = float(expr.subs(subs).evalf())
            
            # Calculate error components
            error_subs = {}
            for name in self.variables:
                error_subs[sp.Symbol(f'σ_{name}')] = self.variables[name]['errors'][i]
                
            for (var1, var2), cov in self.covariances.items():
                error_subs[sp.Symbol(f'cov_{var1}_{var2}')] = cov[i] if hasattr(cov, '__iter__') else cov
                
            sigma_value = total_error.subs(error_subs).subs(subs).evalf()
            sigma_value = float(sigma_value) if sigma_value.is_real else np.nan
            
            results.append((f_value, sigma_value))
        
        # LaTeX generation
        if copy_latex:
            latex_expr = sp.latex(expr, mode='equation')
            latex_error = sp.latex(total_error, mode='equation')
            pyperclip.copy(f"Function:\n{latex_expr}\n\nPropagated Error:\n{latex_error}")
            
        return np.array(results), (expr, total_error)

    def perform_fit(self, func, p0, x_vars, y_var, use_odr=False):
        """Perform curve fitting with error propagation"""
        # Prepare data
        x_data = [self.variables[var]['values'] for var in x_vars]
        x_errs = [self.variables[var]['errors'] for var in x_vars]
        y_data = self.variables[y_var]['values']
        y_err = self.variables[y_var]['errors']
        
        if use_odr:
            model = Model(func)
            data = RealData(x_data, y_data, sx=x_errs, sy=y_err)
            odr = ODR(data, model, beta0=p0)
            output = odr.run()
            return output.beta, output.sd_beta
        else:
            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, sigma=y_err)
            return popt, np.sqrt(np.diag(pcov))

# Example usage:
if __name__ == "__main__":
    exp = ExperimentData()
    
    # Add variables with different error types
    exp.add_variable('V', [1.0, 2.0, 3.0], errors=0.1)  # Constant error
    exp.add_variable('I', [0.1, 0.2, 0.3], errors=[0.01, 0.02, 0.03])  # Variable error
    
    # Define covariance between variables (optional)
    exp.set_covariance('V', 'I', 0.005)  # Could be array or scalar
    
    # Define physical relationship symbolically
    def resistance(V, I):
        return V / I
    
    # Perform error propagation
    results, (expr, error_expr) = exp.error_propagation(resistance, use_covariance=True, copy_latex=True)
    
    # Print results
    print("Computed resistance with errors:")
    for val, err in results:
        print(f"{val:.2f} ± {err:.2f} Ω")
    
    # Perform curve fitting example
    exp.add_variable('t', [1, 2, 3], errors=0.1)
    exp.add_variable('T', [2.1, 4.3, 6.2], errors=0.2)
    
    def linear_model(t, a, b):
        return a * t + b
    
    params, errors = exp.perform_fit(linear_model, [1, 0], x_vars=['t'], y_var='T')
    print(f"\nFit parameters: {params} ± {errors}")