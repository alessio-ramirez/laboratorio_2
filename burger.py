 # %% Import necessary libraries
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
from typing import Union, List, Dict, Callable, Optional, Tuple, Any
from numbers import Number

# Module-level constants
DEFAULT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Helper function to detect if we're in a notebook environment (for plt.show())
def _in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is None: return False
        if 'IPKernelApp' not in get_ipython().config: return False
        return True
    except ImportError:
        return False

# ---------------------------- Measurement Class -----------------------------

class Measurement:
    """
    Represents a physical quantity with a value and uncertainty.

    Supports arithmetic operations (+, -, *, /, **) with automatic error
    propagation (assuming independence) and integration with NumPy ufuncs.

    Attributes:
    -----------
    value : np.ndarray
        The nominal value(s) of the measurement.
    error : np.ndarray
        The uncertainty (standard deviation) associated with the value(s).
        Must have the same shape as value or be broadcastable.
    """
    __array_priority__ = 1000 # Ensure Measurement methods are called before NumPy's

    def __init__(self, values: Union[float, List[float], Dict], errors: Union[float, List[float]] = None, magnitude: int = 0):
        """
        Initializes a Measurement object.

        Parameters:
        -----------
        value : number, list, np.ndarray
            The nominal value(s) of the measurement.
        error : number, list, np.ndarray, optional
            The uncertainty (standard deviation) associated with the value(s).
            If None or 0, uncertainty is assumed to be zero.
            If a scalar, it's applied to all values.
            If an array, its shape must match 'value' or be broadcastable.
        """
        # Case 1: Dictionary input - extract values and errors
        if isinstance(values, dict):
            if errors is not None:
                raise ValueError("Errors must be None when using dictionary input")
            self.value = np.asarray(list(values.keys()), dtype=float) * (10 ** magnitude)
            self.error = np.asarray(list(values.values()), dtype=float) * (10 ** magnitude)
        
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
        
        self.value = values * (10 ** magnitude)
        self.error = errors * (10 ** magnitude)

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def size(self) -> int:
        return self.value.size

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: Any) -> 'Measurement':
        return Measurement(self.value[key], self.error[key])

    def __repr__(self) -> str:
        """Technical representation."""
        return f"Measurement(value={self.value!r}, error={self.error!r})" # Repr() of numpy arrays

    def __str__(self) -> str:
        """User-friendly string representation."""
        if self.shape == (1,): # Scalar
            # Basic formatting, could be improved for significant digits
             val_str = f"{self.value:.3g}"
             err_str = f"{self.error:.2g}"
             return f"{val_str} ± {err_str}"
        elif self.size > 5: # Large array
            return f"Measurement(value=\n{self.value},\n error=\n{self.error}\n shape={self.shape})"
        else: # Small array
            parts = [f"{v:.3g} ± {e:.2g}" for v, e in zip(self.value.flat, self.error.flat)]
            if self.ndim == 1:
                return "[" + ", ".join(parts) + "]"
            else:
                 # Needs better formatting for multi-dim arrays
                 return f"Measurement(\n value={self.value},\n error={self.error}\n)"


    # --- Arithmetic Operations ---

    def __add__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            new_value = self.value + other.value
            # Assuming independence: sigma_sum = sqrt(sigma_a^2 + sigma_b^2)
            new_error = np.sqrt(self.error**2 + other.error**2)
            return Measurement(new_value, new_error)
        elif isinstance(other, (Number, np.ndarray)):
            # Add scalar or array (treat as having zero error)
            new_value = self.value + other
            return Measurement(new_value, self.error) # Error remains unchanged
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> 'Measurement':
        return self.__add__(other) # Addition is commutative

    def __sub__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            new_value = self.value - other.value
            # Assuming independence: sigma_diff = sqrt(sigma_a^2 + sigma_b^2)
            new_error = np.sqrt(self.error**2 + other.error**2)
            return Measurement(new_value, new_error)
        elif isinstance(other, (Number, np.ndarray)):
            new_value = self.value - other
            return Measurement(new_value, self.error) # Error remains unchanged
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> 'Measurement':
        if isinstance(other, (Number, np.ndarray)):
            new_value = other - self.value
            # Result has the same error as self
            return Measurement(new_value, self.error)
        else:
            # Let __sub__ handle Measurement - Measurement
            return NotImplemented # Or -(self - other)? Check consistency, for now is good.


    def __mul__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            a, b = self.value, other.value
            sa, sb = self.error, other.error
            new_value = a * b
            # Assuming independence: sigma_prod = |a*b| * sqrt((sa/a)^2 + (sb/b)^2)
            # Safer formula: sqrt( (b*sa)^2 + (a*sb)^2 ) to avoid division by zero
            term1_sq = (b * sa)**2
            term2_sq = (a * sb)**2
            # Handle cases where value might be zero causing NaN in propagation
            # If a value is zero, its contribution to error via the *other* error is zero
            term1_sq = np.where(np.isclose(a, 0), 0.0, term1_sq)
            term2_sq = np.where(np.isclose(b, 0), 0.0, term2_sq)

            new_error = np.sqrt(term1_sq + term2_sq)
            return Measurement(new_value, new_error)
        elif isinstance(other, (Number, np.ndarray)):
            # Multiply by scalar or array (treat as having zero error)
            k = np.asarray(other)
            new_value = self.value * k
            new_error = np.abs(k) * self.error # sigma_k*x = |k|*sigma_x
            return Measurement(new_value, new_error)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> 'Measurement':
        return self.__mul__(other) # Multiplication is commutative

    def __truediv__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            a, b = self.value, other.value
            sa, sb = self.error, other.error

            if np.any(np.isclose(b, 0)):
                 warnings.warn("Division by Measurement with zero value encountered.", RuntimeWarning)
                 # Result value will be inf, error propagation needs care

            new_value = a / b
            # Assuming independence: sigma_div = |a/b| * sqrt((sa/a)^2 + (sb/b)^2)
            # Safer formula: sqrt( (sa/b)^2 + (a*sb / b^2)^2 )
            # Or even simpler: (1/|b|) * sqrt( sa^2 + (a*sb/b)^2 )
            term_a_sq = (sa / b)**2
            term_b_sq = (a * sb / b**2)**2

            # Handle cases where a or b might be zero
            # If a is 0, term_b_sq is 0. If b is 0, result is inf, error is inf/complex?
            term_a_sq = np.where(np.isclose(b, 0), np.inf, term_a_sq) # Error becomes infinite if dividing by zero exactly
            term_b_sq = np.where(np.isclose(b, 0), np.inf, term_b_sq)
            #term_a_sq = np.where(np.isclose(a, 0) & ~np.isclose(b, 0), (sa/b)**2, term_a_sq) # If a=0, only sa/b matters
            term_b_sq = np.where(np.isclose(a, 0) & ~np.isclose(b, 0), 0.0, term_b_sq)       # If a=0, a*sb/b^2 = 0

            new_error = np.sqrt(term_a_sq + term_b_sq)

            # Handle division by zero in value array -> leads to inf
            new_value = np.where(np.isclose(b, 0) & ~np.isclose(a, 0), np.sign(a)*np.sign(b)*np.inf, new_value)
            # Handle 0/0 -> leads to NaN
            new_value = np.where(np.isclose(b, 0) & np.isclose(a, 0), np.nan, new_value)
            # Error is NaN if value is NaN or Inf
            new_error = np.where(np.isnan(new_value) | np.isinf(new_value), np.nan, new_error)


            return Measurement(new_value, new_error)

        elif isinstance(other, (Number, np.ndarray)):
            # Divide by scalar or array (treat as having zero error)
            k = np.asarray(other)
            if np.any(np.isclose(k, 0)):
                warnings.warn("Division by zero encountered.", RuntimeWarning)
                # Value will be inf/nan, error should reflect this
            new_value = self.value / k
            new_error = np.abs(self.error / k) # sigma_(x/k) = sigma_x / |k|
            # Error is NaN if k=0
            new_error = np.where(np.isclose(k, 0), np.nan, new_error)
            return Measurement(new_value, new_error)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> 'Measurement':
        if isinstance(other, (Number, np.ndarray)):
            # Scalar or array divided by Measurement
            k = np.asarray(other)
            a, sa = self.value, self.error

            if np.any(np.isclose(a, 0)):
                 warnings.warn("Division by Measurement with zero value encountered.", RuntimeWarning)

            new_value = k / a
            # Formula: sigma_(k/a) = |k * sa / a^2|
            new_error = np.abs(k * sa / a**2)

             # Handle division by zero in value array -> leads to inf
            new_value = np.where(np.isclose(a, 0) & ~np.isclose(k, 0), np.sign(k)*np.inf, new_value)
            # Handle 0/0 -> leads to NaN
            new_value = np.where(np.isclose(a, 0) & np.isclose(k, 0), np.nan, new_value)
             # Error is NaN if value is NaN or Inf
            new_error = np.where(np.isnan(new_value) | np.isinf(new_value), np.nan, new_error)

            return Measurement(new_value, new_error)
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> 'Measurement':
        if isinstance(other, (Number, np.ndarray)):
            # Measurement ** scalar (or array)
            n = np.asarray(other) # Exponent (no error)
            a, sa = self.value, self.error

            # Avoid issues with negative base to non-integer power, or 0^negative
            if np.any((a < 0) & (n % 1 != 0)) or np.any((np.isclose(a, 0)) & (n < 0)):
                warnings.warn(f"Potentially invalid power operation: {a} ** {n}", RuntimeWarning)
                # Let numpy handle value calculation, may result in NaN

            new_value = a ** n
             # Formula: sigma_(a^n) = |n * a^(n-1) * sa|
            deriv = n * (a ** (n - 1))
            # Handle edge case: 0^n where n > 1. Derivative is 0.
            # Handle edge case: a^0 = 1. Derivative is 0.
            deriv = np.where(np.isclose(a, 0) & (n > 1), 0.0, deriv)
            deriv = np.where(np.isclose(n, 0), 0.0, deriv) # If n=0, a^n=1, error is 0

            # Handle potentially complex results or infinities from derivative
            # e.g., a=0, n<1 -> a^(n-1) is inf.
            # e.g., a<0, n is non-integer -> a^(n-1) is complex
            deriv = np.where(np.isclose(a, 0) & (n == 1), 1.0, deriv) # a^1 -> deriv is 1
            deriv = np.where(np.isclose(a, 0) & (n < 1) & (n > 0), np.inf, deriv)
            deriv = np.where(np.isclose(a, 0) & (n <= 0), np.nan, deriv) # Or Inf? Let's use NaN for undefined slope

            new_error = np.abs(deriv * sa)

            # Error is NaN if value is NaN
            new_error = np.where(np.isnan(new_value), np.nan, new_error)
            # If value is Inf, error is Inf
            new_error = np.where(np.isinf(new_value), np.inf, new_error)


            return Measurement(new_value, new_error)

        elif isinstance(other, Measurement):
             # Measurement ** Measurement (e.g., a^b)
             # f(a,b) = a^b
             # df/da = b * a^(b-1)
             # df/db = a^b * ln(a)
             a, sa = self.value, self.error
             b, sb = other.value, other.error

             if np.any(a < 0):
                 warnings.warn(f"Base Measurement has negative values {a}; power operation may yield complex numbers or NaN.", RuntimeWarning)
             if np.any(np.isclose(a, 0) & (b <= 0)):
                 warnings.warn(f"Power operation 0 ** <=0 encountered.", RuntimeWarning)


             new_value = a ** b

             df_da = b * (a ** (b - 1))
             df_db = (a ** b) * np.log(a) # Will be NaN for a < 0, -inf for a = 0

             # Handle edge cases for derivatives
             # If a=0: df_da is 0 if b>1, inf if 0<b<1, NaN if b<=0
             df_da = np.where(np.isclose(a, 0) & (b > 1), 0.0, df_da)
             df_da = np.where(np.isclose(a, 0) & (b == 1), 1.0, df_da) # 0^1 -> deriv wrt a is 1
             df_da = np.where(np.isclose(a, 0) & (b < 1) & (b > 0), np.inf, df_da)
             df_da = np.where(np.isclose(a, 0) & (b <= 0), np.nan, df_da)
             # If a=0: df_db = 0 * log(0) -> usually NaN or 0? Let's treat as 0 since 0^b is 0 for b>0.
             df_db = np.where(np.isclose(a, 0) & (b > 0), 0.0, df_db)
             df_db = np.where(np.isclose(a, 0) & (b <= 0), np.nan, df_db) # 0^0, 0^-1 -> undefined derivative wrt b

             term_a_sq = (df_da * sa)**2
             term_b_sq = (df_db * sb)**2

             # Avoid NaN propagation where component is zero
             term_a_sq = np.where(np.isclose(sa, 0), 0.0, term_a_sq)
             term_b_sq = np.where(np.isclose(sb, 0), 0.0, term_b_sq)
             term_a_sq = np.where(np.isnan(df_da), np.nan, term_a_sq)
             term_b_sq = np.where(np.isnan(df_db), np.nan, term_b_sq)


             new_error = np.sqrt(term_a_sq + term_b_sq)
             # Error is NaN if value is NaN or Inf
             new_error = np.where(np.isnan(new_value) | np.isinf(new_value), np.nan, new_error)


             return Measurement(new_value, new_error)

        else:
            return NotImplemented

    def __rpow__(self, other: Any) -> 'Measurement':
         if isinstance(other, (Number, np.ndarray)):
             # Scalar (or array) ** Measurement (e.g., k^a)
             # f(a) = k^a => df/da = k^a * ln(k)
             k = np.asarray(other)
             a, sa = self.value, self.error

             if np.any(k < 0):
                  warnings.warn(f"Base {k} is negative; power operation may yield complex numbers or NaN.", RuntimeWarning)
             if np.any(np.isclose(k, 0) & (a <= 0)):
                 warnings.warn(f"Power operation 0 ** <=0 encountered.", RuntimeWarning)

             new_value = k ** a
             deriv = new_value * np.log(k) # NaN if k<0, -Inf if k=0

             # Handle edge cases
             deriv = np.where(np.isclose(k, 0) & (a > 0), 0.0, deriv) # 0^a for a>0 -> deriv is 0
             deriv = np.where(np.isclose(k, 0) & (a <= 0), np.nan, deriv) # 0^0, 0^-1 -> undefined deriv

             new_error = np.abs(deriv * sa)
             # Error is NaN if value is NaN or Inf
             new_error = np.where(np.isnan(new_value) | np.isinf(new_value), np.nan, new_error)

             return Measurement(new_value, new_error)
         else:
             return NotImplemented

    def __neg__(self) -> 'Measurement':
        return Measurement(-self.value, self.error) # Negation doesn't change error magnitude

    def __pos__(self) -> 'Measurement':
        return self # Positive does nothing

    def __abs__(self) -> 'Measurement':
        # abs(x) -> Error is tricky near x=0.
        # Derivative is sign(x), which is discontinuous.
        # However, error propagation assumes linearity, so error remains the same.
        # This matches behaviour of some other libraries (e.g. `uncertainties`)
        return Measurement(np.abs(self.value), self.error)

    # --- NumPy Integration ---

    def __array__(self, dtype=None) -> np.ndarray:
        """Allows direct use in NumPy functions that call np.asarray(). Returns nominal values."""
        warnings.warn("Casting Measurement to np.ndarray loses error information. Returning nominal values.", UserWarning)
        return np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Handles NumPy ufuncs (like np.sin, np.add, etc.)."""

        # --- Input Processing ---
        input_values = []
        input_errors = []
        input_symbols = []
        measurement_inputs = [] # Keep track of Measurement args
        has_measurement = False

        # Create unique symbols for SymPy
        sympy_vars = sp.symbols(f'x:{len(inputs)}')

        for i, x in enumerate(inputs):
            if isinstance(x, Measurement):
                input_values.append(x.value)
                input_errors.append(x.error)
                input_symbols.append(sympy_vars[i])
                measurement_inputs.append(x)
                has_measurement = True
            elif isinstance(x, (Number, np.ndarray)):
                input_values.append(np.asarray(x))
                 # Treat non-Measurement inputs as having zero error
                input_errors.append(np.zeros_like(np.asarray(x)))
                input_symbols.append(sympy_vars[i])
                measurement_inputs.append(None) # Placeholder
            else:
                # Cannot handle this type of input with the ufunc
                return NotImplemented

        if not has_measurement:
            # If no Measurement objects are involved, just call the ufunc directly
            # This shouldn't typically happen if __array_priority__ is set,
            # but good to have as a fallback.
            return ufunc(*input_values, **kwargs)

        # Check for 'out' keyword argument - we don't support it easily with error prop.
        if 'out' in kwargs and kwargs['out'] is not None:
            # It's complex to handle putting results into existing arrays,
            # especially when creating new Measurement objects.
            warnings.warn("The 'out' keyword argument is not supported for ufuncs involving Measurement objects. A new Measurement object will be returned.", UserWarning)
            # Remove 'out' to prevent errors, let NumPy create a new result array
            kwargs.pop('out')
            # return NotImplemented # Alternative: refuse to proceed


        # --- Value Calculation ---
        # Apply the ufunc to the nominal values
        try:
             # Use inputs list directly with ufunc
            result_value = ufunc(*input_values, **kwargs)
        except Exception as e:
             # Catch potential errors during value calculation (e.g., domain errors)
             warnings.warn(f"Error during ufunc ({ufunc.__name__}) value calculation: {e}", RuntimeWarning)
             # Try to proceed to error calculation if possible, result might be nan/inf
             # Create a dummy result shape for error calculation if needed
             try:
                 # Determine output shape based on broadcasting rules
                 broadcast_shape = np.broadcast(*input_values).shape
                 result_value = np.full(broadcast_shape, np.nan) # Fill with NaN on error
             except ValueError: # Cannot broadcast inputs
                 return NotImplemented # Give up if inputs incompatible


        # --- Error Propagation ---
        if ufunc.nin == 1: # Unary functions (e.g., np.sin(x))
            if isinstance(inputs[0], Measurement): # Ensure the input was a Measurement
                x = measurement_inputs[0]
                x_sym = input_symbols[0]

                # Special handling for common functions where derivative is known
                if ufunc is np.negative: deriv_val = -1.0
                elif ufunc is np.abs: deriv_val = np.sign(x.value) # Note: discontinuous at 0
                elif ufunc is np.exp: deriv_val = np.exp(x.value)
                elif ufunc is np.log: deriv_val = 1.0 / x.value
                elif ufunc is np.log10: deriv_val = 1.0 / (x.value * np.log(10))
                elif ufunc is np.sqrt: deriv_val = 0.5 / np.sqrt(x.value)
                elif ufunc is np.sin: deriv_val = np.cos(x.value)
                elif ufunc is np.cos: deriv_val = -np.sin(x.value)
                elif ufunc is np.tan: deriv_val = 1.0 / (np.cos(x.value)**2)
                # Add more common cases if needed...
                else:
                    # General case: Use SymPy for derivative
                    try:
                        f_sym = ufunc(x_sym) # Apply numpy ufunc to sympy symbol
                        deriv_sym = sp.diff(f_sym, x_sym)
                        # Lambdify for numerical evaluation
                        deriv_func = sp.lambdify(x_sym, deriv_sym, modules=['numpy', {'conjugate': np.conjugate}]) # Add conjugate for complex ufuncs
                        deriv_val = deriv_func(x.value)
                    except (TypeError, AttributeError, NotImplementedError) as e:
                         warnings.warn(f"SymPy could not differentiate ufunc '{ufunc.__name__}'. Cannot propagate error. Error will be NaN. Sympy error: {e}", RuntimeWarning)
                         deriv_val = np.nan

                # Error formula: sigma_f = |df/dx * sigma_x|
                with np.errstate(invalid='ignore', divide='ignore'): # Ignore warnings from deriv_val potentially being NaN/inf
                    result_error = np.abs(deriv_val * x.error)
                 # Ensure error is NaN if value is NaN or derivative failed
                result_error = np.where(np.isnan(result_value) | np.isnan(deriv_val), np.nan, result_error)

            else:
                # Input was not a Measurement, result error is 0
                result_error = np.zeros_like(result_value)

        elif ufunc.nin == 2: # Binary functions (e.g., np.add(x, y))
            x, y = measurement_inputs[0], measurement_inputs[1]
            x_sym, y_sym = input_symbols[0], input_symbols[1]
            vx, sx = (x.value, x.error) if x is not None else (input_values[0], input_errors[0])
            vy, sy = (y.value, y.error) if y is not None else (input_values[1], input_errors[1])

            # Special handling for basic arithmetic (faster than SymPy)
            if ufunc is np.add or ufunc is np.subtract:
                var_sq = sx**2 + sy**2
            elif ufunc is np.multiply:
                term1_sq = (vy * sx)**2
                term2_sq = (vx * sy)**2
                term1_sq = np.where(np.isclose(vx, 0), 0.0, term1_sq)
                term2_sq = np.where(np.isclose(vy, 0), 0.0, term2_sq)
                var_sq = term1_sq + term2_sq
            elif ufunc is np.true_divide:
                term_a_sq = (sx / vy)**2
                term_b_sq = (vx * sy / vy**2)**2
                term_a_sq = np.where(np.isclose(vy, 0), np.inf, term_a_sq)
                term_b_sq = np.where(np.isclose(vy, 0), np.inf, term_b_sq)
                term_a_sq = np.where(np.isclose(vx, 0) & ~np.isclose(vy, 0), (sx/vy)**2, term_a_sq)
                term_b_sq = np.where(np.isclose(vx, 0) & ~np.isclose(vy, 0), 0.0, term_b_sq)
                var_sq = term_a_sq + term_b_sq
            elif ufunc is np.power:
                 # This depends on which arg is Measurement (handled by __pow__ / __rpow__)
                 # If both are Measurement, SymPy is needed anyway for the general case.
                 # Fall through to SymPy case for np.power
                 var_sq = None # Signal to use SymPy
            else:
                 var_sq = None # Signal to use SymPy for other binary funcs

            if var_sq is not None:
                 with np.errstate(invalid='ignore'): # Ignore sqrt(negative) if var_sq is somehow negative
                    result_error = np.sqrt(var_sq)
            else:
                 # General case: Use SymPy
                 try:
                     f_sym = ufunc(x_sym, y_sym)
                     df_dx_sym = sp.diff(f_sym, x_sym)
                     df_dy_sym = sp.diff(f_sym, y_sym)

                     # Lambdify for numerical evaluation
                     # Need to handle cases where x or y wasn't Measurement
                     symbols_to_eval = []
                     values_to_eval = []
                     if x is not None: symbols_to_eval.append(x_sym); values_to_eval.append(vx)
                     if y is not None: symbols_to_eval.append(y_sym); values_to_eval.append(vy)
                     # If only one Measurement was involved, we need values for both symbols
                     if len(symbols_to_eval) == 1:
                         if x is None: symbols_to_eval.insert(0, x_sym); values_to_eval.insert(0, vx)
                         if y is None: symbols_to_eval.append(y_sym); values_to_eval.append(vy)


                     # Handle potential complex results from numpy functions in sympy
                     modules = ['numpy', {'conjugate': np.conjugate}]

                     df_dx_func = sp.lambdify(symbols_to_eval, df_dx_sym, modules=modules)
                     df_dy_func = sp.lambdify(symbols_to_eval, df_dy_sym, modules=modules)

                     df_dx_val = df_dx_func(*values_to_eval)
                     df_dy_val = df_dy_func(*values_to_eval)

                     # Error formula: sigma_f^2 = (df/dx * sx)^2 + (df/dy * sy)^2 (assuming independence)
                     term1_sq = (df_dx_val * sx)**2
                     term2_sq = (df_dy_val * sy)**2

                     # Handle NaNs from derivatives if components have zero error
                     term1_sq = np.where(np.isclose(sx, 0), 0.0, term1_sq)
                     term2_sq = np.where(np.isclose(sy, 0), 0.0, term2_sq)
                     # If derivative itself is NaN (e.g. log(neg)), propagate NaN
                     term1_sq = np.where(np.isnan(df_dx_val), np.nan, term1_sq)
                     term2_sq = np.where(np.isnan(df_dy_val), np.nan, term2_sq)


                     with np.errstate(invalid='ignore'): # Ignore adding NaNs
                         var_sq = term1_sq + term2_sq
                     result_error = np.sqrt(var_sq)

                 except (TypeError, AttributeError, NotImplementedError, ValueError) as e:
                     warnings.warn(f"SymPy could not differentiate ufunc '{ufunc.__name__}'. Cannot propagate error. Error will be NaN. Error details: {e}", RuntimeWarning)
                     result_error = np.full_like(result_value, np.nan)

            # Ensure error is NaN if value is NaN
            result_error = np.where(np.isnan(result_value), np.nan, result_error)
             # Ensure error is Inf if value is Inf
            result_error = np.where(np.isinf(result_value), np.inf, result_error)


        else: # Ufuncs with more than 2 inputs - NotImplemented
            warnings.warn(f"Error propagation for ufunc '{ufunc.__name__}' with {ufunc.nin} inputs is not implemented.", UserWarning)
            return NotImplemented
            # # Potential future implementation using SymPy
            # try:
            #     f_sym = ufunc(*input_symbols)
            #     var_sq = 0
            #     symbols_to_eval = []
            #     values_to_eval = []
            #     for i, meas in enumerate(measurement_inputs):
            #         if meas is not None:
            #             symbols_to_eval.append(input_symbols[i])
            #             values_to_eval.append(meas.value)

            #     # Add non-measurement values needed for evaluation
            #     full_eval_values = []
            #     current_meas_idx = 0
            #     for i, meas in enumerate(measurement_inputs):
            #          if meas is not None:
            #              full_eval_values.append(values_to_eval[current_meas_idx])
            #              current_meas_idx += 1
            #          else:
            #              full_eval_values.append(input_values[i])


            #     for i, meas in enumerate(measurement_inputs):
            #         if meas is not None: # Only differentiate wrt Measurement inputs
            #             deriv_sym = sp.diff(f_sym, input_symbols[i])
            #             # Lambdify needs all symbols involved in the expression
            #             all_involved_symbols = list(deriv_sym.free_symbols) # Get symbols actually in derivative
            #             eval_indices = [input_symbols.index(s) for s in all_involved_symbols] # Find indices in original inputs
            #             eval_values_subset = [full_eval_values[idx] for idx in eval_indices]

            #             # Check if lambdify arguments match required symbols
            #             if set(all_involved_symbols) != set(input_symbols[j] for j, m in enumerate(measurement_inputs) if m is not None):
            #                 # Need to handle cases where derivative simplifies and removes a variable
            #                 # For now, try lambdifying with all original measurement symbols
            #                  try:
            #                     deriv_func = sp.lambdify(symbols_to_eval, deriv_sym, modules=['numpy', {'conjugate': np.conjugate}])
            #                     deriv_val = deriv_func(*values_to_eval)
            #                  except Exception as le:
            #                      warnings.warn(f"Lambdify error for derivative {i} of {ufunc.__name__}: {le}. Trying with reduced symbols.", RuntimeWarning)
            #                      try:
            #                          deriv_func = sp.lambdify(all_involved_symbols, deriv_sym, modules=['numpy', {'conjugate': np.conjugate}])
            #                          deriv_val = deriv_func(*eval_values_subset)
            #                          # Need to broadcast deriv_val back to original shape if necessary
            #                          # This part is tricky and requires careful shape management
            #                      except Exception as le2:
            #                          warnings.warn(f"Secondary Lambdify error for derivative {i} of {ufunc.__name__}: {le2}. Giving up.", RuntimeWarning)
            #                          deriv_val = np.nan
            #             else:
            #                 deriv_func = sp.lambdify(symbols_to_eval, deriv_sym, modules=['numpy', {'conjugate': np.conjugate}])
            #                 deriv_val = deriv_func(*values_to_eval)


            #             term_sq = (deriv_val * meas.error)**2
            #             term_sq = np.where(np.isnan(deriv_val), np.nan, term_sq) # Propagate NaN from deriv
            #             var_sq += term_sq

            #     result_error = np.sqrt(var_sq)
            #     result_error = np.where(np.isnan(result_value), np.nan, result_error) # Error is NaN if value is NaN

            # except Exception as e:
            #      warnings.warn(f"SymPy error during propagation for ufunc '{ufunc.__name__}' with {ufunc.nin} inputs: {e}. Error will be NaN.", RuntimeWarning)
            #      result_error = np.full_like(result_value, np.nan)


        # --- Result ---
        return Measurement(result_value, result_error)


    # --- Comparison Methods ---
    # Comparisons should generally compare nominal values, ignoring errors,
    # unless specifically comparing compatibility (see test_comp).
    def __eq__(self, other):
        if isinstance(other, Measurement):
            return np.array_equal(self.value, other.value) # Compare only values
        elif isinstance(other, (Number, np.ndarray)):
             return np.array_equal(self.value, np.asarray(other))
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        val = other.value if isinstance(other, Measurement) else other
        try:
            return self.value < val
        except TypeError:
             return NotImplemented

    def __le__(self, other):
        val = other.value if isinstance(other, Measurement) else other
        try:
            return self.value <= val
        except TypeError:
             return NotImplemented

    def __gt__(self, other):
        val = other.value if isinstance(other, Measurement) else other
        try:
            return self.value > val
        except TypeError:
             return NotImplemented

    def __ge__(self, other):
        val = other.value if isinstance(other, Measurement) else other
        try:
            return self.value >= val
        except TypeError:
             return NotImplemented


    # --- Helper Methods ---
    @property
    def nominal_value(self) -> np.ndarray:
        """Returns the nominal value(s)."""
        return self.value

    @property
    def std_dev(self) -> np.ndarray:
        """Returns the uncertainty (standard deviation)."""
        return self.error

    @property
    def variance(self) -> np.ndarray:
         """Returns the variance (square of uncertainty)."""
         return self.error**2

    def round_to_error(self, n_sig_figs: int = 1) -> str:
        """
        Formats the measurement to a string, rounding the value
        based on the uncertainty's significant figures.

        Args:
            n_sig_figs (int): Number of significant figures for the error (usually 1 or 2).

        Returns:
            str: Formatted string like "value ± error". Handles arrays element-wise.
        """
        if self.shape == (): # Scalar
            return self._round_single_to_error(self.value, self.error, n_sig_figs)
        else: # Array
            strs = [self._round_single_to_error(v, e, n_sig_figs)
                    for v, e in np.nditer([self.value, self.error])]
            # Reshape the list of strings back into the original array shape (if > 1D)
            if self.ndim > 1:
                 # This is tricky to represent nicely as a multi-line string automatically
                 # For now, just return a flat list representation or similar
                 return np.array(strs).reshape(self.shape).__str__() # Use numpy's array str
            else:
                return "[" + ", ".join(strs) + "]"


    @staticmethod
    def _round_single_to_error(value: float, error: float, n_sig_figs: int = 1) -> str:
        """Helper to format a single value-error pair."""
        if np.isnan(value) or np.isnan(error):
            return "NaN"
        if np.isinf(value) or np.isinf(error):
             vs = "inf" if np.isinf(value) else f"{value:.3g}"
             es = "inf" if np.isinf(error) else f"{error:.2g}"
             return f"{vs} ± {es}"
        if np.isclose(error, 0.0):
            # No error or negligible error, format value reasonably
            # Try to show similar precision as if error was present
             order_val = np.floor(np.log10(abs(value))) if not np.isclose(value, 0) else 0
             decimals = max(0, int(2 - order_val)) # Show a few decimals
             return f"{value:.{decimals}f} ± 0" # Indicate zero error

        # Find order of magnitude of the first significant digit of the error
        order_err = np.floor(np.log10(error))
        # Determine the decimal place to round to
        decimal_place = int(order_err - (n_sig_figs - 1))

        # Round error to n significant figures
        factor = 10**(-decimal_place)
        rounded_error_scaled = round(error * factor)
        rounded_error = rounded_error_scaled / factor

        # Round value to the same decimal place as the rounded error
        rounded_value = round(value, -decimal_place) if decimal_place < 0 else round(value * factor) / factor

        # Format the strings
        # Ensure correct number of decimal places shown
        fmt_dp = max(0, -decimal_place)
        val_str = f"{rounded_value:.{fmt_dp}f}"
        err_str = f"{rounded_error:.{fmt_dp}f}"

        # Handle cases where error rounds to 1.0*10^n -> needs one more digit sometimes?
        # Check if rounded error has fewer apparent sig figs than intended
        # This logic can get complex, basic rounding is often sufficient

        return f"{val_str} ± {err_str}"


# ---------------------------- Dataset Creation ------------------------------

def create_measurement(values: Union[float, List[float], np.ndarray, Dict],
                       errors: Union[float, List[float], np.ndarray] = None,
                       magnitude: int = 0) -> Measurement:
    """
    Create a Measurement object from values and errors.

    Parameters:
    -----------
    values : float, list, np.ndarray, or dict
        - Single measurement value (float)
        - List/array of measured values
        - Dictionary with {value: error} pairs (if provided, errors arg must be None)
    errors : float, list, or np.ndarray, optional
        Single error value (same error for all measured values) or list/array of errors.
        If None and values is not a dict, zero errors are assumed.
    magnitude : int, optional (default=0)
        Order of magnitude adjustment. Values and errors are multiplied by 10^magnitude.

    Returns:
    --------
    Measurement: An object encapsulating the values and errors.

    Examples:
    ---------
    >>> m1 = create_measurement(5.0, 0.1)
    >>> print(m1)
    5.0 ± 0.1

    >>> m2 = create_measurement([2, 4], 0.5)
    >>> print(m2)
    [2.0 ± 0.5, 4.0 ± 0.5]

    >>> m3 = create_measurement({10: 0.1, 20: 0.2})
    >>> print(m3)
    [10.0 ± 0.1, 20.0 ± 0.2]

    >>> m4 = create_measurement([1.2, 3.4], [0.1, 0.2], magnitude=-3)
    >>> print(m4.value)
    [0.0012 0.0034]
    >>> print(m4.error)
    [0.0001 0.0002]
    """
    factor = 10.0**magnitude

    if isinstance(values, dict):
        if errors is not None:
            raise ValueError("Errors argument must be None when using dictionary input for values")
        val_array = np.array(list(values.keys()), dtype=float) * factor
        err_array = np.array(list(values.values()), dtype=float) * factor
        return Measurement(val_array, err_array)
    else:
        val_array = np.asarray(values, dtype=float) * factor
        if errors is None:
            err_array = np.zeros_like(val_array)
        else:
            err_array = np.asarray(errors, dtype=float)
            if err_array.ndim == 0: # Scalar error
                err_array = np.full_like(val_array, err_array.item(), dtype=float) * factor
            else:
                 err_array = err_array * factor # Apply magnitude to array error
                 if err_array.shape != val_array.shape:
                      # Try broadcasting
                      try:
                          np.broadcast(val_array, err_array)
                          err_array = np.broadcast_to(err_array, val_array.shape)
                      except ValueError:
                          raise ValueError(f"Errors shape {err_array.shape} must match values shape {val_array.shape} or be broadcastable")
        return Measurement(val_array, err_array)

# --------------------- Error Propagation (General Function) ------------------

def error_prop(f: Callable, *variables: Measurement,
              covariance_matrix: Optional[np.ndarray] = None,
              copy_latex: bool = False, round_latex: int = 3) -> Measurement:
    """
    Calculate error propagation through an arbitrary function using partial derivatives.

    This function is useful when the operation is not standard arithmetic or a NumPy ufunc
    supported directly by the Measurement class.

    Parameters:
    -----------
    f : callable
        Function to propagate errors through. Should accept arguments corresponding
        to the nominal values of the input Measurement objects.
        Example: `f(x, y)` if called as `error_prop(f, meas_x, meas_y)`.
    *variables : Measurement
        Input Measurement objects.
    covariance_matrix : numpy.ndarray, optional
        Precomputed covariance matrix of shape (n_variables, n_variables).
        If None, variables are assumed to be independent.
        Note: Covariance is handled only if explicitly provided. Arithmetic operations
        within the Measurement class currently assume independence.
    copy_latex : bool, optional (default=False)
        If True, copy the LaTeX formula (symbolic derivatives) to clipboard using pyperclip.
    round_latex : int, optional (default=3)
        Number of decimal places for coefficients in the LaTeX formula.

    Returns:
    --------
    Measurement: Result of the function application with propagated error.

    Example:
    --------
    >>> def my_func(x, y): return x * np.sin(y)
    >>> x = create_measurement(2.0, 0.1)
    >>> y = create_measurement(np.pi/4, 0.05)
    >>> result = error_prop(my_func, x, y)
    >>> print(result)
    1.41 ± 0.0857
    """
    n_vars = len(variables)
    values_list = [v.value for v in variables]
    errors_list = [v.error for v in variables]

    # Ensure all inputs are compatible (e.g., same shape or broadcastable)
    try:
        broadcast_shape = np.broadcast(*values_list).shape
    except ValueError:
        raise ValueError("Input Measurement variables have incompatible shapes for broadcasting.")

    # Adapt values and errors to the broadcast shape
    values_broadcast = [np.broadcast_to(v, broadcast_shape) for v in values_list]
    errors_broadcast = [np.broadcast_to(e, broadcast_shape) for e in errors_list]

    # Calculate nominal result value
    result_value = f(*values_broadcast)

    # Use SymPy for derivatives
    symbols = sp.symbols(f'x:{n_vars}')
    # Try applying the function symbolically
    try:
        f_sym = f(*symbols)
    except Exception as e:
         warnings.warn(f"Could not represent function '{f.__name__}' symbolically with SymPy: {e}. Cannot calculate derivatives for error propagation.", RuntimeWarning)
         # Return result with NaN error
         return Measurement(result_value, np.full_like(result_value, np.nan))


    derivatives_sym = [sp.diff(f_sym, sym) for sym in symbols]

    # Lambdify derivatives for numerical evaluation
    deriv_funcs = []
    try:
        for deriv_sym in derivatives_sym:
            # Include numpy module for common functions
            deriv_funcs.append(sp.lambdify(symbols, deriv_sym, modules=['numpy']))
    except Exception as e:
        warnings.warn(f"Could not lambdify symbolic derivatives for '{f.__name__}': {e}. Cannot calculate error. Error will be NaN.", RuntimeWarning)
        return Measurement(result_value, np.full_like(result_value, np.nan))

    # Evaluate derivatives at nominal values
    deriv_values = [func(*values_broadcast) for func in deriv_funcs]

    # Calculate squared error terms
    variance_sq = 0
    for i in range(n_vars):
        variance_sq += (deriv_values[i] * errors_broadcast[i])**2

    # Add covariance terms if provided
    if covariance_matrix is not None:
        if covariance_matrix.shape != (n_vars, n_vars):
            raise ValueError(f"Covariance matrix must have shape ({n_vars}, {n_vars})")
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                 # Ensure covariance term is broadcastable if necessary
                 # This assumes cov_matrix contains scalar covariances
                 cov_ij = np.broadcast_to(covariance_matrix[i, j], broadcast_shape)
                 variance_sq += 2 * deriv_values[i] * deriv_values[j] * cov_ij
                 # Note: This assumes cov(xi, xj) is constant across all elements
                 # if inputs are arrays. A full covariance matrix *between elements*
                 # is much more complex to handle here.

    with np.errstate(invalid='ignore'): # Ignore sqrt(negative) if variance is somehow negative
        result_error = np.sqrt(variance_sq)

    # Handle LaTeX output
    if copy_latex:
        try:
            import pyperclip
            latex_error_terms = []
            for i in range(n_vars):
                 term = (sp.N(derivatives_sym[i], round_latex) * sp.symbols(f'\\sigma_{{{symbols[i]}}}'))**2
                 latex_error_terms.append(term)

            # Add covariance terms to LaTeX if applicable
            if covariance_matrix is not None:
                for i in range(n_vars):
                    for j in range(i+1, n_vars):
                        # Check if covariance is significant enough to include
                        # Note: using np.isclose might be tricky with symbolic/rounded
                        # We add it if provided, regardless of value for the formula.
                        cov_sym = sp.symbols(f'\\sigma_{{{symbols[i]},{symbols[j]}}}') # Or cov?
                        term = 2 * sp.N(derivatives_sym[i], round_latex) * sp.N(derivatives_sym[j], round_latex) * cov_sym
                        latex_error_terms.append(term)

            sigma_f_sym = sp.sqrt(sum(latex_error_terms))
            latex_formula = f"\\sigma_f = {sp.latex(sigma_f_sym)}"
            pyperclip.copy(latex_formula)
        except ImportError:
            warnings.warn("pyperclip not installed. Cannot copy LaTeX formula to clipboard.")
        except Exception as e:
             warnings.warn(f"Failed to generate or copy LaTeX formula: {e}")


    return Measurement(result_value, result_error)

# ---------------------------- Curve Fitting (Core) ---------------------------

def perform_fit(x: Union[Measurement, np.ndarray, list],
                y: Union[Measurement, np.ndarray, list],
                func: Callable,
                p0: Optional[List[float]] = None,
                parameter_names: Optional[List[str]] = None,
                method: str = 'auto',
                mask: Optional[np.ndarray] = None,
                absolute_sigma: bool = True,
                maxfev: Optional[int] = None,
                full_output: bool = True, # For ODR
                **kwargs) -> Dict[str, Any]:
    """
    Performs curve fitting using scipy.optimize.curve_fit or scipy.odr.

    Selects ODR if x data has errors, otherwise uses curve_fit. Returns a
    dictionary containing comprehensive fit results.

    Parameters:
    -----------
    x : Measurement, np.ndarray, or list
        Independent variable data. If Measurement, errors are used.
    y : Measurement, np.ndarray, or list
        Dependent variable data. If Measurement, errors are used.
    func : callable
        Model function to fit. Signature depends on the method:
        - curve_fit: `f(x, p1, p2, ...)`
        - ODR: Requires a specific `f(beta, x)` signature where beta is a list
          of parameters. `perform_fit` handles the wrapping automatically.
    p0 : list or float, optional
        Initial parameter guesses. Auto-generated if None ([1.0, 1.0, ...]).
    parameter_names : list of str, optional
        Names for the parameters. Auto-generated if None ("p0", "p1", ...).
    method : str, optional (default='auto')
        Fitting method: 'auto', 'curve_fit', or 'odr'.
        - 'auto': Use ODR if x has errors > 0, otherwise use curve_fit.
        - 'curve_fit': Force use of scipy.optimize.curve_fit (ignores x errors).
        - 'odr': Force use of scipy.odr.ODR.
    mask : array-like of bool, optional
        Boolean mask to select data points for fitting (True = include).
        Applied to x and y.
    absolute_sigma : bool, optional (default=True)
        For `curve_fit`: If True, `sigma` (y_err) is used in an absolute sense
        and the estimated parameter covariance reflects this. If False, only
        relative weights matter. For ODR, errors are always treated absolutely.
    maxfev : int, optional
        Maximum number of function evaluations for `curve_fit`. Passed to `scipy.optimize.curve_fit`.
        For ODR, use `maxit` within `kwargs`.
    full_output : bool, optional (default=True)
         For ODR: If True, returns the full ODR output object in the results dict.
    **kwargs : dict, optional
        Additional keyword arguments passed directly to the underlying fit function
        (`curve_fit` or `ODR.run()`). E.g., `maxit` for ODR.

    Returns:
    --------
    dict: A dictionary containing comprehensive fit results:
        - 'parameters': Measurement object with best-fit values and errors.
        - 'parameter_names': List of parameter names.
        - 'covariance_matrix': Parameter covariance matrix (np.ndarray).
        - 'correlation_matrix': Parameter correlation matrix (np.ndarray).
        - 'statistics': Dict with 'chi_squared', 'dof', 'reduced_chi_squared',
                        'r_squared', 'p_value' (chi-squared probability).
        - 'fit_method': The method used ('curve_fit' or 'odr').
        - 'model_function': The original model function `func`.
        - 'initial_guess': The initial parameter guess `p0` used.
        - 'mask': The boolean mask used, if any.
        - 'n_points_total': Total number of data points before masking.
        - 'n_points_fit': Number of data points used in the fit (after masking).
        - 'success': Boolean indicating if the fit likely converged.
        - 'status_message': Status message from the fitter (if available).
        - 'raw_output': The raw output object from `curve_fit` (tuple) or
                       `ODR.run()` (output object) if available.

    Raises:
    -------
    ValueError: If inputs are inconsistent (e.g., shape mismatch, invalid method).
    TypeError: If inputs are not of the expected types.
    RuntimeError: If the fitting process fails catastrophically.
    """
    # --- 1. Input Processing and Validation ---
    if isinstance(x, Measurement):
        x_val = x.value
        x_err = x.error
    elif isinstance(x, (np.ndarray, list)):
        x_val = np.asarray(x)
        x_err = np.zeros_like(x_val)
    else:
        raise TypeError("x must be a Measurement object, numpy array, or list.")

    if isinstance(y, Measurement):
        y_val = y.value
        y_err = y.error
    elif isinstance(y, (np.ndarray, list)):
        y_val = np.asarray(y)
        # Use small non-zero error if none provided, for curve_fit's sigma
        # But only if absolute_sigma=True, otherwise weights are relative.
        # Let's default to zeros and handle in curve_fit call.
        y_err = np.zeros_like(y_val)
    else:
        raise TypeError("y must be a Measurement object, numpy array, or list.")

    if x_val.shape != y_val.shape:
        raise ValueError(f"Shape mismatch: x has shape {x_val.shape}, y has shape {y_val.shape}")
    if x_err.shape != x_val.shape:
         # Try broadcasting error shape if possible (e.g., scalar error)
        try:
            x_err = np.broadcast_to(x_err, x_val.shape)
        except ValueError:
            raise ValueError(f"x_err shape {x_err.shape} incompatible with x_val shape {x_val.shape}")
    if y_err.shape != y_val.shape:
        try:
            y_err = np.broadcast_to(y_err, y_val.shape)
        except ValueError:
            raise ValueError(f"y_err shape {y_err.shape} incompatible with y_val shape {y_val.shape}")

    n_points_total = x_val.size # Use size for multi-dimensional compatibility? Flatten?
    if x_val.ndim > 1:
         warnings.warn("Input data has more than 1 dimension. Flattening for fitting.", UserWarning)
         x_val = x_val.flatten()
         x_err = x_err.flatten()
         y_val = y_val.flatten()
         y_err = y_err.flatten()
         n_points_total = x_val.size

    # --- 2. Apply Mask ---
    original_indices = np.arange(n_points_total)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).flatten()
        if mask.shape != x_val.shape:
            raise ValueError(f"Mask shape {mask.shape} must match data shape {x_val.shape}")

        x_val_fit = x_val[mask]
        x_err_fit = x_err[mask]
        y_val_fit = y_val[mask]
        y_err_fit = y_err[mask]
        fit_indices = original_indices[mask]
    else:
        x_val_fit = x_val
        x_err_fit = x_err
        y_val_fit = y_val
        y_err_fit = y_err
        fit_indices = original_indices

    n_points_fit = x_val_fit.size
    if n_points_fit == 0:
         raise ValueError("No data points remaining after applying the mask.")

    # --- 3. Parameter Setup ---
    sig = inspect.signature(func)
    n_params = len(sig.parameters) - 1 # Subtract the independent variable 'x'
    if n_params <= 0:
        raise ValueError(f"Model function signature invalid. Needs at least one parameter besides x: {sig}")

    if p0 is None:
        p0 = [1.0] * n_params
    elif isinstance(p0, (int, float)):
        p0 = [p0] * n_params
    elif len(p0) != n_params:
        raise ValueError(f"p0 length ({len(p0)}) must match number of function parameters ({n_params})")
    p0 = list(p0) # Ensure it's a list for ODR

    if parameter_names is None:
        # Default names like p0, p1, ... or use names from signature if possible
        param_sig_names = list(sig.parameters.keys())[1:] # Skip x
        if len(param_sig_names) == n_params:
             parameter_names = param_sig_names
        else: # Fallback if signature inspection fails
            parameter_names = [f"p{i}" for i in range(n_params)]
    elif len(parameter_names) != n_params:
        raise ValueError(f"parameter_names length ({len(parameter_names)}) must match number of parameters ({n_params})")


    # --- 4. Choose and Perform Fit ---
    has_x_errors = np.any(x_err_fit > 1e-15) # Use a small tolerance for float comparison

    if method == 'auto':
        use_odr = has_x_errors
    elif method == 'curve_fit':
        use_odr = False
        if has_x_errors:
            warnings.warn("Method forced to 'curve_fit', but x data has errors. X errors will be ignored.", UserWarning)
    elif method == 'odr':
        use_odr = True
        if not has_x_errors and np.all(np.isclose(y_err_fit, 0)):
             warnings.warn("Method forced to 'odr', but neither x nor y has significant errors. ODR might be overkill or less stable.", UserWarning)
    else:
        raise ValueError(f"Invalid fitting method '{method}'. Choose 'auto', 'curve_fit', or 'odr'.")

    fit_method_name = 'odr' if use_odr else 'curve_fit'
    popt = None
    pcov = None
    fit_success = False
    fit_message = "Fit not attempted."
    raw_output = None

    try:
        if use_odr:
            # --- ODR Fit ---
            # ODR requires func(beta, x) where beta is the list of parameters
            odr_model = Model(func, fjacb=None, fjacd=None) # Let ODR estimate Jacobians

            # Use RealData. Errors (sx, sy) must be std deviations, not variances
            # Handle zero errors: ODR doesn't like zero errors, replace with small number?
            # ODR documentation suggests it handles zero internally by assigning large weight.
            # Let's pass them as is unless issues arise.
            odr_data = RealData(x_val_fit, y_val_fit, sx=x_err_fit, sy=y_err_fit)

            odr_obj = ODR(odr_data, odr_model, beta0=p0, **kwargs) # Pass extra args like maxit
            output = odr_obj.run()
            raw_output = output

            if output.info <= 0: # Check ODR status code (<=0 means error or failure)
                 raise RuntimeError(f"ODR fitting failed. Info code: {output.info}. Message: {output.stopreason}")

            popt = output.beta
            pcov = output.cov_beta # Covariance matrix
            fit_success = True # Assume success if info > 0
            fit_message = output.stopreason

        else:
            # --- curve_fit Fit ---
            # curve_fit requires f(x, p1, p2, ...)
            # Handle y errors (sigma)
            effective_sigma = None
            if np.any(y_err_fit > 1e-15):
                 # Replace zero errors with a very small number if absolute_sigma=True
                 # to avoid division by zero and ensure points are weighted.
                 # If absolute_sigma=False, zeros are fine (infinite weight).
                 if absolute_sigma:
                    y_err_adjusted = np.where(y_err_fit <= 1e-15, 1e-15, y_err_fit)
                    effective_sigma = y_err_adjusted
                 else:
                    effective_sigma = y_err_fit # Use original errors (including zero)

            # Prepare curve_fit arguments
            curve_fit_kwargs = {'p0': p0, 'sigma': effective_sigma, 'absolute_sigma': absolute_sigma}
            if maxfev is not None:
                 curve_fit_kwargs['maxfev'] = maxfev
            curve_fit_kwargs.update(kwargs) # Add any other user kwargs

            popt, pcov = curve_fit(func, x_val_fit, y_val_fit, **curve_fit_kwargs)
            raw_output = (popt, pcov) # Store the direct output
            fit_success = True # curve_fit raises error on failure
            fit_message = "Converged (curve_fit)." # curve_fit doesn't provide detailed messages like ODR

    except RuntimeError as e:
        fit_message = f"Fitting failed: {e}"
        # Return NaNs or raise error? Let's return NaNs for parameters/stats
        popt = np.full(n_params, np.nan)
        pcov = np.full((n_params, n_params), np.nan)
        fit_success = False
        warnings.warn(fit_message, RuntimeWarning)
    except Exception as e:
         fit_message = f"An unexpected error occurred during fitting: {e}"
         popt = np.full(n_params, np.nan)
         pcov = np.full((n_params, n_params), np.nan)
         fit_success = False
         warnings.warn(fit_message, RuntimeWarning)
         # Re-raise if it's not a standard fitting convergence issue? Maybe not.


    # --- 5. Post-processing and Statistics ---
    perr = np.sqrt(np.diag(pcov)) if fit_success else np.full(n_params, np.nan)
    parameters = Measurement(popt, perr)

    # Chi-squared calculation
    dof = n_points_fit - n_params
    chi_squared = np.nan
    reduced_chi_squared = np.nan
    p_value_chi2 = np.nan
    y_pred_fit = np.full_like(y_val_fit, np.nan) # Predicted values for fitted points

    if fit_success and dof > 0:
        y_pred_fit = func(x_val_fit, *popt)
        residuals = y_val_fit - y_pred_fit

        # Chi-squared definition depends on error source and method
        if use_odr:
            # ODR provides sum_square, which is the generalized chi-squared
             if raw_output and hasattr(raw_output, 'sum_square'):
                chi_squared = raw_output.sum_square
                # ODR also has res_var (reduced chi-sq), use it?
                # reduced_chi_squared = raw_output.res_var
                # Let's recalculate for consistency
                reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan
             else: # Should not happen if fit succeeded
                 warnings.warn("Could not retrieve sum_square from ODR output.", RuntimeWarning)

        else: # curve_fit
             # Standard chi-squared: sum(((y_i - f(x_i))/sigma_yi)^2)
             if effective_sigma is not None and np.any(effective_sigma > 1e-15):
                 # Use the sigma provided to curve_fit
                 weights = 1.0 / effective_sigma**2
                 chi_squared = np.sum(residuals**2 * weights)
                 reduced_chi_squared = chi_squared / dof
             else:
                 # No meaningful errors provided for y, chi-squared is just sum of squares
                 # Chi-squared test is not statistically valid here.
                 chi_squared = np.sum(residuals**2)
                 reduced_chi_squared = chi_squared / dof
                 warnings.warn("Chi-squared calculated without y-errors; statistical interpretation is invalid.", UserWarning)

        # P-value for chi-squared test (probability of exceeding observed chi-squared)
        if not np.isnan(chi_squared) and dof > 0:
            p_value_chi2 = 1.0 - stats.chi2.cdf(chi_squared, dof)

    # R-squared (Coefficient of Determination)
    r_squared = np.nan
    if fit_success and n_points_fit > 1:
        ss_residuals = np.sum(residuals**2) if fit_success else np.nan
        mean_y = np.mean(y_val_fit)
        ss_total = np.sum((y_val_fit - mean_y)**2)
        if ss_total > 1e-15: # Avoid division by zero if all y values are the same
             r_squared = 1.0 - (ss_residuals / ss_total)
        elif np.isclose(ss_residuals, 0): # Perfect fit to constant data
             r_squared = 1.0
        else: # Bad fit to constant data (shouldn't happen if ss_total is ~0)
             r_squared = 0.0 # Or NaN? Let's use 0


    # Correlation Matrix
    correlation_matrix = np.full_like(pcov, np.nan)
    if fit_success:
        std_devs = np.sqrt(np.diag(pcov)).reshape(-1, 1)
        # Avoid division by zero if parameter error is zero
        if np.all(std_devs > 1e-15):
            correlation_matrix = pcov / (std_devs @ std_devs.T)
        else:
             # Handle zero standard deviations carefully
             correlation_matrix = np.eye(n_params) # Initialize as identity
             non_zero_mask = (std_devs > 1e-15).flatten()
             if np.any(non_zero_mask):
                  outer_prod = std_devs[non_zero_mask] @ std_devs[non_zero_mask].T
                  sub_pcov = pcov[np.ix_(non_zero_mask, non_zero_mask)]
                  sub_corr = sub_pcov / outer_prod
                  correlation_matrix[np.ix_(non_zero_mask, non_zero_mask)] = sub_corr
             # Entries involving zero-error params remain 0 off-diagonal, 1 on-diagonal.

    # --- 6. Assemble Results Dictionary ---
    results = {
        'parameters': parameters,
        'parameter_names': parameter_names,
        'covariance_matrix': pcov,
        'correlation_matrix': correlation_matrix,
        'statistics': {
            'chi_squared': chi_squared,
            'dof': dof,
            'reduced_chi_squared': reduced_chi_squared,
            'p_value': p_value_chi2, # Chi-squared probability
            'r_squared': r_squared,
        },
        'fit_method': fit_method_name,
        'model_function': func,
        'initial_guess': p0,
        'mask': mask,
        'n_points_total': n_points_total,
        'n_points_fit': n_points_fit,
        'success': fit_success,
        'status_message': fit_message,
        'raw_output': raw_output,
        # Include fitted values? Maybe not essential here, can be recalc'd.
        # 'y_predicted': y_pred_fit, # Predicted y values for the *fitted* points
        # 'residuals': residuals,   # Residuals for the *fitted* points
        # Add arguments used for clarity?
        'fit_arguments': {
            'absolute_sigma': absolute_sigma if not use_odr else None,
            'maxfev': maxfev if not use_odr else None,
            'odr_kwargs': kwargs if use_odr else {},
            'curve_fit_kwargs': kwargs if not use_odr else {},
        }
    }

    return results


# ------------------------- Visualization (Plotting) --------------------------

def create_best_fit_line(
    fit_results: Dict[str, Any],
    x_data: Union[Measurement, np.ndarray, list],
    y_data: Union[Measurement, np.ndarray, list],
    *, # Force subsequent arguments to be keyword-only
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    data_label: Optional[str] = "Data",
    fit_label: Optional[str] = "Fit",
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    # Data Point Options
    fmt: str = 'o',
    markersize: int = 5,
    capsize: float = 3,
    data_alpha: float = 0.8,
    # Masked Point Options
    show_masked_points: bool = True,
    masked_fmt: str = 'x',
    masked_color: str = 'gray',
    masked_alpha: float = 0.5,
    masked_label: Optional[str] = "_nolegend_", # Default: don't label masked points
    # Fit Line Options
    fit_linewidth: float = 1.5,
    fit_linestyle: str = '-',
    fit_points: int = 500,
    fit_range_padding: float = 0.05, # Extend fit line range by 5% beyond data
    # Confidence/Prediction Band Options (To be implemented)
    # show_confidence_band: bool = False,
    # confidence_level: float = 0.95,
    # band_alpha: float = 0.2,
    # Residuals Plot Options
    show_residuals: bool = False,
    residuals_ylabel: str = "Residuals",
    residuals_fmt: str = 'o',
    # Annotation Options
    show_fit_params: bool = False,
    show_stats: Optional[List[str]] = None, # e.g., ['reduced_chi_squared', 'r_squared']
    annotation_pos: Tuple[float, float] = (0.03, 0.97),
    annotation_fontsize: int = 10,
    annotation_vspacing: float = 0.045,
    # Plot Styling Options
    grid: bool = True,
    legend_loc: str = 'best',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    residuals_ylim: Optional[Tuple[float, float]] = None,
    axis_fontsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a plot visualizing the data points and the best-fit curve from perform_fit results.

    Parameters:
    -----------
    fit_results : dict
        The dictionary returned by `perform_fit`.
    x_data : Measurement, np.ndarray, or list
        The *original* independent variable data (including any points masked during the fit).
    y_data : Measurement, np.ndarray, or list
        The *original* dependent variable data (including any points masked during the fit).
    xlabel, ylabel : str, optional
        Labels for the x and y axes.
    title : str, optional
        Title for the main plot.
    data_label : str, optional
        Label for the (unmasked) data points in the legend.
    fit_label : str, optional
        Label for the fitted curve in the legend.
    color : str, optional
        Color used for data points and fit line. Defaults to Matplotlib's default cycle.
    ax : plt.Axes, optional
        Existing Matplotlib Axes object to plot on. If None, a new Figure and Axes are created.
    figsize : tuple, optional
        Size of the figure if `ax` is None and `show_residuals` is False.
    show_plot : bool, optional
        If True (default), calls `plt.show()` at the end (if not in a notebook).
    save_path : str, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving the figure.
    fmt : str, optional
        Format string for the unmasked data points (matplotlib style).
    markersize : int, optional
        Marker size for data points.
    capsize : float, optional
        Size of the error bar caps.
    data_alpha : float, optional
        Transparency for unmasked data points.
    show_masked_points : bool, optional
        If True, display points excluded by the mask (if a mask was used).
    masked_fmt : str, optional
        Format string for masked data points.
    masked_color : str, optional
        Color for masked points.
    masked_alpha : float, optional
        Transparency for masked points.
    masked_label : str, optional
        Legend label for masked points. Default is "_nolegend_" (hidden).
    fit_linewidth : float, optional
        Line width for the fit curve.
    fit_linestyle : str, optional
        Line style for the fit curve.
    fit_points : int, optional
        Number of points used to draw the smooth fit curve.
    fit_range_padding : float, optional
        Fractional padding to extend the fit line beyond the min/max x-data range.
    show_residuals : bool, optional
        If True, add a subplot below the main plot showing residuals (y_data - y_fit).
    residuals_ylabel : str, optional
        Label for the y-axis of the residuals plot.
    residuals_fmt : str, optional
        Format string for the residual points.
    show_fit_params : bool, optional
        If True, display the fitted parameter values and errors as text on the plot.
    show_stats : list of str, optional
        List of statistic keys (from `fit_results['statistics']`) to display as text.
        E.g., ['reduced_chi_squared', 'r_squared', 'p_value'].
    annotation_pos : tuple, optional
        (x, y) position in axes coordinates (0-1) for the top-left corner of annotations.
    annotation_fontsize : int, optional
        Font size for annotations.
    annotation_vspacing : float, optional
        Vertical spacing between annotation lines in axes coordinates.
    grid : bool, optional
        If True, display grid lines on the plot(s).
    legend_loc : str, optional
        Location of the legend (e.g., 'best', 'upper left').
    xlim, ylim : tuple, optional
        Set fixed limits for the x and y axes of the main plot.
    residuals_ylim : tuple, optional
        Set fixed limits for the y-axis of the residuals plot.
    axis_fontsize : int, optional
        Font size for axis labels.
    title_fontsize : int, optional
        Font size for the title.
    legend_fontsize : int, optional
        Font size for the legend.

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib Figure object and the main Axes object (or array of Axes if residuals=True).

    Raises:
    ------
    ValueError: If `fit_results` is missing essential keys or data is inconsistent.
    TypeError: If data types are incorrect.
    """
    # --- 1. Extract Data and Fit Info ---
    if not isinstance(fit_results, dict) or 'parameters' not in fit_results:
        raise ValueError("fit_results must be a dictionary returned by perform_fit.")

    # Extract from fit_results
    params_meas = fit_results['parameters']
    popt = params_meas.value
    # perr = params_meas.error # Not directly used here, but available
    func = fit_results['model_function']
    mask = fit_results.get('mask') # Might be None
    fit_success = fit_results.get('success', False)
    param_names = fit_results.get('parameter_names', [f'p{i}' for i in range(len(popt))])
    stats = fit_results.get('statistics', {})

    # Process original x_data
    if isinstance(x_data, Measurement):
        x_val = x_data.value
        x_err = x_data.error
    else:
        x_val = np.asarray(x_data)
        x_err = np.zeros_like(x_val)

    # Process original y_data
    if isinstance(y_data, Measurement):
        y_val = y_data.value
        y_err = y_data.error
    else:
        y_val = np.asarray(y_data)
        y_err = np.zeros_like(y_val)

    # Basic shape validation
    if x_val.shape != y_val.shape:
        raise ValueError("Original x_data and y_data must have the same shape.")
    if x_val.ndim > 1: # Flatten original data if necessary
         x_val = x_val.flatten()
         x_err = x_err.flatten() if x_err.size > 1 else np.full_like(x_val, x_err.item() if x_err.size==1 else 0)
         y_val = y_val.flatten()
         y_err = y_err.flatten() if y_err.size > 1 else np.full_like(y_val, y_err.item() if y_err.size==1 else 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).flatten()
        if mask.shape != x_val.shape:
             raise ValueError("Mask shape must match original data shape.")
        unmasked_indices = np.where(mask)[0]
        masked_indices = np.where(~mask)[0]
    else:
        unmasked_indices = np.arange(x_val.size)
        masked_indices = np.array([], dtype=int)

    # --- 2. Setup Plot ---
    if ax is None:
        if show_residuals:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
            ax_main = axes[0]
            ax_res = axes[1]
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
            axes = ax_main
            ax_res = None
    else:
        # User provided axes. Assume it's compatible (single or array if residuals)
        if show_residuals:
            if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) != 2:
                raise ValueError("If show_residuals is True, 'ax' must be a list/tuple/array of two Axes objects.")
            axes = ax
            ax_main = ax[0]
            ax_res = ax[1]
            fig = ax_main.get_figure()
        else:
            if isinstance(ax, (list, tuple, np.ndarray)):
                 warnings.warn("Received multiple Axes but show_residuals is False. Plotting only on the first Axes.", UserWarning)
                 ax_main = ax[0]
            else:
                ax_main = ax
            axes = ax_main
            ax_res = None
            fig = ax_main.get_figure()

    # Use default color cycle if color not specified
    if color is None:
        color = next(ax_main._get_lines.prop_cycler)['color']

    # --- 3. Plot Data Points ---
    # Unmasked points (used in fit)
    if len(unmasked_indices) > 0:
        ax_main.errorbar(x_val[unmasked_indices], y_val[unmasked_indices],
                         xerr=x_err[unmasked_indices], yerr=y_err[unmasked_indices],
                         fmt=fmt, color=color, markersize=markersize, capsize=capsize,
                         alpha=data_alpha, label=data_label)

    # Masked points (excluded from fit)
    if show_masked_points and len(masked_indices) > 0:
        ax_main.errorbar(x_val[masked_indices], y_val[masked_indices],
                         xerr=x_err[masked_indices], yerr=y_err[masked_indices],
                         fmt=masked_fmt, color=masked_color, markersize=markersize, capsize=capsize,
                         alpha=masked_alpha, label=masked_label, ecolor=masked_color) # Match error bar color

    # --- 4. Plot Fit Line ---
    if fit_success:
        # Determine range for plotting the fit line
        if len(unmasked_indices) > 0:
             x_fit_min = np.min(x_val[unmasked_indices])
             x_fit_max = np.max(x_val[unmasked_indices])
             x_range = x_fit_max - x_fit_min
             plot_min = x_fit_min - x_range * fit_range_padding
             plot_max = x_fit_max + x_range * fit_range_padding
        else: # Should not happen if perform_fit succeeded, but as fallback
             plot_min = np.min(x_val)
             plot_max = np.max(x_val)

        x_fit_line = np.linspace(plot_min, plot_max, fit_points)
        try:
            y_fit_line = func(x_fit_line, *popt)
            ax_main.plot(x_fit_line, y_fit_line, color=color,
                         linewidth=fit_linewidth, linestyle=fit_linestyle,
                         label=fit_label, alpha=0.9)
        except Exception as e:
            warnings.warn(f"Could not evaluate model function for plotting fit line: {e}", RuntimeWarning)


        # --- 5. Plot Confidence/Prediction Bands (Placeholder) ---
        # This requires more complex calculations using the covariance matrix
        # To be implemented later if needed.
        # if show_confidence_band:
        #    y_fit_err = calculate_band_error(...) # Function needed
        #    ax_main.fill_between(x_fit_line, y_fit_line - y_fit_err, y_fit_line + y_fit_err,
        #                         color=color, alpha=band_alpha, label=f"{confidence_level*100:.0f}% CI")


        # --- 6. Plot Residuals ---
        if show_residuals and ax_res is not None and len(unmasked_indices) > 0:
            x_unmasked = x_val[unmasked_indices]
            y_unmasked = y_val[unmasked_indices]
            y_err_unmasked = y_err[unmasked_indices]

            try:
                y_pred_unmasked = func(x_unmasked, *popt)
                residuals = y_unmasked - y_pred_unmasked

                # Plot residuals with y-errors
                ax_res.errorbar(x_unmasked, residuals, yerr=y_err_unmasked,
                                fmt=residuals_fmt, color=color, markersize=markersize,
                                alpha=data_alpha, capsize=capsize)
                ax_res.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
                ax_res.set_ylabel(residuals_ylabel, fontsize=axis_fontsize)
                if grid:
                     ax_res.grid(True, linestyle=':', alpha=0.6)
                if residuals_ylim:
                    ax_res.set_ylim(residuals_ylim)
                # Consider plotting masked point residuals? Maybe not standard.

            except Exception as e:
                 warnings.warn(f"Could not calculate or plot residuals: {e}", RuntimeWarning)


    # --- 7. Annotations ---
    annotation_lines = []
    if show_fit_params and fit_success:
        for i, name in enumerate(param_names):
             # Use round_to_error formatting from Measurement
             param_str = params_meas[i].round_to_error(n_sig_figs=2) # Show 2 sig figs for error?
             annotation_lines.append(f"{name} = {param_str}")
        if annotation_lines: annotation_lines.append("") # Add spacer

    if show_stats and fit_success:
        stat_map = { # Nicer names for display
            'chi_squared': "χ²",
            'reduced_chi_squared': "χ²/dof",
            'dof': "DoF",
            'r_squared': "R²",
            'p_value': "P(χ²)"
        }
        for key in show_stats:
            if key in stats and not np.isnan(stats[key]):
                display_key = stat_map.get(key, key)
                value = stats[key]
                # Format based on type
                if isinstance(value, int) or key == 'dof':
                    annotation_lines.append(f"{display_key} = {value}")
                elif key == 'p_value':
                     annotation_lines.append(f"{display_key} = {value:.3g}") # Scientific notation if small
                else:
                    annotation_lines.append(f"{display_key} = {value:.3f}") # 3 decimal places? Or sig figs?
            # else:
            #     annotation_lines.append(f"{stat_map.get(key, key)} = N/A")


    if annotation_lines:
        anc = ax_main.annotate("\n".join(annotation_lines),
                               xy=annotation_pos, xycoords='axes fraction',
                               fontsize=annotation_fontsize, va='top', ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --- 8. Final Plot Styling ---
    if title:
        ax_main.set_title(title, fontsize=title_fontsize)
    if xlabel:
        # Set xlabel on the lowest subplot if shared
        xlabel_ax = ax_res if show_residuals else ax_main
        xlabel_ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    if ylabel:
        ax_main.set_ylabel(ylabel, fontsize=axis_fontsize)

    if grid:
        ax_main.grid(True, linestyle=':', alpha=0.6)

    # Set limits
    if xlim:
         ax_main.set_xlim(xlim) # Works even if axes are shared
    if ylim:
         ax_main.set_ylim(ylim)

    # Add legend
    handles, labels = ax_main.get_legend_handles_labels()
    if handles: # Only show legend if there are labeled items
        ax_main.legend(loc=legend_loc, fontsize=legend_fontsize)


    # --- 9. Save and Show ---
    if save_path:
        try:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            warnings.warn(f"Failed to save figure to '{save_path}': {e}")

    if show_plot and not _in_notebook():
        plt.show()

    return fig, axes # Return tuple (fig, ax or [ax_main, ax_res])


# ---------------------------- Latex Formatting -----------------------------
# TODO: Update latex_table and fit_results_table to use Measurement objects
# and the new fit_results dictionary structure.

def latex_table(*args, orientation="h", magnitude=None, precision=2):
    """
    Generate a LaTeX table from named Measurement objects.

    Rounds values based on error using Measurement.round_to_error logic.

    Parameters:
    -----------
    *args : str, Measurement pairs
        Pairs of names (str) and Measurement objects.
    orientation : str, optional (default='h')
        'h' for horizontal (names as row headers), 'v' for vertical.
    magnitude : int, optional
        If provided, scale values by 10**magnitude *before* formatting.
        Note: This overrides the inherent scaling of Measurement formatting.
    precision : int, optional (default=2)
        Number of significant figures to target for the error when formatting.

    Returns:
    --------
    str: LaTeX table code (also copied to clipboard if pyperclip available).

    Example:
    --------
    >>> m1 = create_measurement([10.123, 10.245], [0.021, 0.038])
    >>> m2 = create_measurement([20.55, 21.01], [0.52, 0.69])
    >>> print(latex_table("Set A", m1, "Set B", m2, orientation='v'))
    """
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of name (str) and Measurement object.")

    pairs = []
    data_lengths = set()
    for i in range(0, len(args), 2):
        name = args[i]
        data = args[i+1]
        if not isinstance(name, str):
            raise TypeError(f"Expected string for name at index {i}, got {type(name)}")
        if not isinstance(data, Measurement):
             raise TypeError(f"Expected Measurement object for data at index {i+1}, got {type(data)}")
        pairs.append((name, data))
        data_lengths.add(data.size)

    if len(data_lengths) > 1:
         warnings.warn(f"Input Measurement objects have different sizes: {data_lengths}. Table might look irregular.", UserWarning)
         # Proceed anyway, might work for horizontal if lengths differ? Vertical needs same length.
         if orientation == 'v':
              raise ValueError("All Measurement objects must have the same size for vertical table orientation.")

    n_entries = max(data_lengths) if data_lengths else 0 # Max length for horizontal

    formatted_data = []
    for name, meas in pairs:
        # Apply scaling if requested
        if magnitude is not None:
             meas_scaled = meas * (10.0**magnitude) # Apply scaling
             name_str = f"{name} ($\\times 10^{{{-magnitude}}}$)" if magnitude != 0 else name
        else:
             meas_scaled = meas
             name_str = name # No scaling applied

        # Format each entry
        if meas_scaled.size == 0:
             entries = []
        elif meas_scaled.ndim == 0: # Scalar
            entries = [meas_scaled.round_to_error(precision)]
        else: # Array
            entries = [Measurement(v, e).round_to_error(precision)
                       for v, e in np.nditer([meas_scaled.value, meas_scaled.error])]

        formatted_data.append({'name': name_str, 'entries': entries, 'size': meas_scaled.size})

    # Generate LaTeX code
    latex = ""
    try:
        if orientation == 'h':
            max_len = max(d['size'] for d in formatted_data) if formatted_data else 0
            columns = '|l|' + 'c' * max_len + '|'
            latex += f"\\begin{{tabular}}{{{columns}}}\n\\hline\n"
            for data_dict in formatted_data:
                # Pad entries if shorter than max_len
                padded_entries = data_dict['entries'] + ['--'] * (max_len - data_dict['size'])
                row = [data_dict['name']] + padded_entries
                latex += " & ".join(row) + " \\\\\\hline\n"
            latex += "\\end{tabular}"
        elif orientation == 'v':
            if not data_lengths or len(data_lengths) > 1:
                 raise ValueError("Inconsistent data lengths for vertical table.")
            num_cols = len(pairs)
            num_rows = n_entries
            latex += f"\\begin{{tabular}}{{{'|'+'c|' * num_cols}}}\n\\hline\n"
            headers = [d['name'] for d in formatted_data]
            latex += " & ".join(headers) + " \\\\\n\\hline\\hline\n" # Double hline after header
            for i in range(num_rows):
                row = [d['entries'][i] for d in formatted_data]
                latex += " & ".join(row) + " \\\\\n"
            latex += "\\hline\n\\end{tabular}"
        else:
            raise ValueError("Orientation must be 'h' or 'v'")

        # Copy to clipboard
        pyperclip.copy(latex)

    except ImportError:
        warnings.warn("pyperclip not installed. Cannot copy LaTeX table to clipboard.")
    except Exception as e:
        warnings.warn(f"Failed to generate or copy LaTeX table: {e}")

    return latex


def fit_results_table(
    *fit_results_args: Union[Dict[str, Any], str],
    orientation="v",
    params_to_show: Optional[List[str]] = None,
    stats_to_show: Optional[List[str]] = ['reduced_chi_squared', 'r_squared', 'p_value'],
    param_precision: int = 2,
    stat_precision: int = 3,
    title: Optional[str] = None, # Add title support?
    label_dict: Optional[Dict[str, str]] = None # Map param/stat keys to nice names
    ) -> str:
    """
    Generates a LaTeX table summarizing results from one or more fits.

    Parameters:
    -----------
    *fit_results_args : Dict or str
        A sequence of fit result dictionaries (from `perform_fit`) and
        optional string labels preceding each dictionary.
        Example: `fit_results_table("Linear", res1, "Quadratic", res2)`
    orientation : str, optional (default='v')
        Table orientation: 'v' (vertical, fits as rows) or 'h' (horizontal, fits as columns).
    params_to_show : list of str, optional
        List of parameter names (from `parameter_names` in fit results) to include.
        If None, includes all parameters found in the results.
    stats_to_show : list of str, optional
        List of statistic keys (from `fit_results['statistics']`) to include.
        Defaults to ['reduced_chi_squared', 'r_squared', 'p_value'].
        Set to [] to show no statistics.
    param_precision : int, optional (default=2)
        Number of significant figures for the error part of parameters when formatting.
    stat_precision : int, optional (default=3)
        Number of significant figures or decimal places for statistic values.
    title : str, optional
        Optional caption for the LaTeX table (uses `\caption{}`).
    label_dict : dict, optional
        Dictionary mapping internal parameter/statistic names to desired LaTeX display names.
        Example: {'p0': '$a_0$', 'reduced_chi_squared': '$\\tilde{\\chi}^2$'}

    Returns:
    --------
    str: LaTeX code for the table, including optional `tabular` and `caption`.
         Also copied to clipboard if pyperclip is available.
    """

    fits = []
    current_label = None
    for i, arg in enumerate(fit_results_args):
        if isinstance(arg, str):
            current_label = arg
        elif isinstance(arg, dict) and 'parameters' in arg:
            label = current_label if current_label is not None else f"Fit {len(fits)+1}"
            fits.append({'label': label, 'results': arg})
            current_label = None # Reset label
        else:
            raise TypeError(f"Invalid argument type at index {i}: {type(arg)}. Expected fit result dict or string label.")

    if not fits:
        warnings.warn("No valid fit results provided to fit_results_table.")
        return ""

    # --- Identify all parameters and stats across fits ---
    all_param_names = []
    all_stat_keys = set(stats_to_show) if stats_to_show else set()
    temp_stat_keys = set()

    for fit in fits:
        res = fit['results']
        p_names = res.get('parameter_names', [])
        for name in p_names:
            if name not in all_param_names:
                all_param_names.append(name)
        # Check which requested stats are actually present
        if 'statistics' in res:
            for key in (stats_to_show or []):
                 if key in res['statistics']:
                      temp_stat_keys.add(key)

    # Filter parameters if requested
    if params_to_show is not None:
         # Keep original order but only include specified params
         all_param_names = [p for p in all_param_names if p in params_to_show]

    # Use only stats that are actually present and requested
    final_stat_keys = sorted(list(temp_stat_keys))

    # --- Setup Display Labels ---
    if label_dict is None: label_dict = {}
    param_display_names = [label_dict.get(p, p.replace('_', '\\_')) for p in all_param_names]
    stat_display_names = [label_dict.get(s, s.replace('_', ' ').title()) for s in final_stat_keys]
    stat_map = dict(zip(final_stat_keys, stat_display_names)) # For lookup

    # --- Format Data ---
    table_data = []
    fit_labels = [fit['label'] for fit in fits]

    for fit in fits:
        res = fit['results']
        params: Measurement = res['parameters']
        stats: Dict = res.get('statistics', {})
        param_map = dict(zip(res.get('parameter_names', []), params)) if params.size > 0 else {} # Map name to Measurement slice

        row_data = {}
        # Parameters
        for p_name in all_param_names:
            if p_name in param_map:
                 # Find the index corresponding to the name
                 try:
                     idx = res.get('parameter_names', []).index(p_name)
                     p_meas = Measurement(params.value[idx], params.error[idx]) # Single Measurement
                     row_data[p_name] = p_meas.round_to_error(param_precision)
                 except (ValueError, IndexError):
                     row_data[p_name] = '--' # Should not happen if param_map is built correctly
            else:
                row_data[p_name] = '--' # Parameter not present in this fit

        # Statistics
        for s_key in final_stat_keys:
            if s_key in stats and not np.isnan(stats[s_key]):
                value = stats[s_key]
                if isinstance(value, int) or s_key == 'dof':
                    row_data[s_key] = f"{value}"
                else:
                    # Use sig figs formatting
                    row_data[s_key] = f"{value:.{stat_precision}g}"
            else:
                row_data[s_key] = '--' # Statistic not calculated or NaN

        table_data.append(row_data)

    # --- Generate LaTeX ---
    latex = ""
    if title:
         latex += f"\\caption{{{title}}}\n"

    num_fits = len(fits)
    num_params = len(all_param_names)
    num_stats = len(final_stat_keys)

    try:
        if orientation == 'v': # Fits as rows
            num_cols = 1 + num_params + num_stats
            col_spec = '|l|' + ('c' * num_params) + ('c' * num_stats) + '|' if num_params + num_stats > 0 else '|l|'
            latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
            header = ["Fit"] + param_display_names + stat_display_names
            latex += " & ".join(header) + " \\\\\n\\hline\\hline\n"
            for i, fit_label in enumerate(fit_labels):
                row = [fit_label]
                row.extend(table_data[i].get(p_name, '--') for p_name in all_param_names)
                row.extend(table_data[i].get(s_key, '--') for s_key in final_stat_keys)
                latex += " & ".join(row) + " \\\\\n"
            latex += "\\hline\n\\end{tabular}"

        elif orientation == 'h': # Fits as columns
            num_rows = num_params + num_stats
            num_cols = 1 + num_fits
            col_spec = '|l|' + 'c' * num_fits + '|'
            latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
            header = ["Quantity"] + fit_labels
            latex += " & ".join(header) + " \\\\\n\\hline\\hline\n"
            # Parameter rows
            for i, p_name in enumerate(all_param_names):
                row = [param_display_names[i]]
                row.extend(table_data[j].get(p_name, '--') for j in range(num_fits))
                latex += " & ".join(row) + " \\\\\n"
            # Separator line if stats follow params
            if num_params > 0 and num_stats > 0:
                 latex += "\\hline\n"
             # Statistic rows
            for i, s_key in enumerate(final_stat_keys):
                row = [stat_display_names[i]]
                row.extend(table_data[j].get(s_key, '--') for j in range(num_fits))
                latex += " & ".join(row) + " \\\\\n"
            latex += "\\hline\n\\end{tabular}"
        else:
            raise ValueError("Orientation must be 'h' or 'v'")

        # Copy to clipboard
        pyperclip.copy(latex)

    except ImportError:
        warnings.warn("pyperclip not installed. Cannot copy LaTeX table to clipboard.")
    except Exception as e:
        warnings.warn(f"Failed to generate or copy fit results LaTeX table: {e}\nLaTeX so far:\n{latex}")

    return latex


# ---------------------------- Statistical Test -----------------------------
# TODO: Update test_comp and test_comp_advanced to accept Measurement objects

def test_comp(a: Union[Measurement, float], b: Union[Measurement, float],
              sigma_a: Optional[float] = None, sigma_b: Optional[float] = None,
              corr_coef: float = 0.0,
              method: str = 'normal', alpha: float = 0.05,
              n_a: Optional[int] = None, n_b: Optional[int] = None, # For t-test
              output: str = 'full', visualize: bool = False) -> Dict:
    """
    Compares two measurements (value ± uncertainty) for compatibility.

    Accepts either Measurement objects or value/sigma pairs.

    Parameters:
    -----------
    a : Measurement or float
        First measurement. If float, `sigma_a` must be provided.
    b : Measurement or float
        Second measurement. If float, `sigma_b` must be provided.
    sigma_a : float, optional
        Uncertainty of the first measurement (required if `a` is float).
    sigma_b : float, optional
        Uncertainty of the second measurement (required if `b` is float).
    corr_coef : float, optional (default=0.0)
        Correlation coefficient between a and b (-1 to 1).
    method : str, optional (default='normal')
        Method: 'normal' (Z-test) or 'student' (t-test).
    alpha : float, optional (default=0.05)
        Significance level for hypothesis testing.
    n_a, n_b : int, optional
        Sample sizes (required for 'student' method).
    output : str, optional (default='full')
        Output format: 'full', 'pvalue', 'compatible'.
    visualize : bool, optional (default=False)
        If True, create a plot visualizing the comparison.

    Returns:
    --------
    dict or float or bool: Depending on `output` format.
        'full': Dict with 'difference', 'sigma_difference', 'z_score', 'p_value',
                'compatible', 'critical_value', 'method', etc.
        'pvalue': The calculated p-value (float).
        'compatible': Boolean indicating compatibility at the given alpha.

    Examples:
    ---------
    >>> m1 = create_measurement(5.2, 0.3)
    >>> m2 = create_measurement(5.5, 0.4)
    >>> result = test_comp(m1, m2)
    >>> print(f"Compatible: {result['compatible']}, p-value: {result['p_value']:.3f}")

    >>> result_t = test_comp(5.2, 5.5, sigma_a=0.3, sigma_b=0.4, method='student', n_a=10, n_b=15)
    """

    # Extract values and errors
    if isinstance(a, Measurement):
        val_a = a.value.item() if a.ndim == 0 else a.value # Allow scalar result? Assume scalar input
        err_a = a.error.item() if a.ndim == 0 else a.error
        if a.size > 1: raise TypeError("test_comp expects scalar Measurements or floats.")
        val_a, err_a = val_a.item(), err_a.item() # Ensure float
    elif isinstance(a, Number):
        if sigma_a is None: raise ValueError("sigma_a must be provided if a is float.")
        val_a = float(a)
        err_a = float(sigma_a)
    else:
        raise TypeError("a must be a Measurement object or a number.")

    if isinstance(b, Measurement):
        val_b = b.value.item() if b.ndim == 0 else b.value
        err_b = b.error.item() if b.ndim == 0 else b.error
        if b.size > 1: raise TypeError("test_comp expects scalar Measurements or floats.")
        val_b, err_b = val_b.item(), err_b.item()
    elif isinstance(b, Number):
        if sigma_b is None: raise ValueError("sigma_b must be provided if b is float.")
        val_b = float(b)
        err_b = float(sigma_b)
    else:
        raise TypeError("b must be a Measurement object or a number.")

    # Validate inputs
    if err_a < 0 or err_b < 0:
        raise ValueError("Uncertainties must be non-negative.")
    if corr_coef < -1 or corr_coef > 1:
        raise ValueError("Correlation coefficient must be between -1 and 1.")
    if method not in ['normal', 'student']:
        raise ValueError("Method must be either 'normal' or 'student'.")
    if method == 'student' and (n_a is None or n_b is None):
        raise ValueError("Sample sizes n_a and n_b must be provided for Student's t-test.")
    if method == 'student' and (n_a <= 1 or n_b <= 1):
         warnings.warn("Student's t-test with n<=1 has limited meaning.", UserWarning)
         # Adjust DoF calculation below? Welch-Satterthwaite needs n > 1.
         # Let scipy handle potential errors for now.

    if output not in ['full', 'pvalue', 'compatible']:
        raise ValueError("Output must be 'full', 'pvalue', or 'compatible'.")

    # Calculate difference and its uncertainty
    diff = val_a - val_b
    var_diff = err_a**2 + err_b**2 - 2 * corr_coef * err_a * err_b
    if var_diff < 0:
         # This can happen if correlation is high and errors are similar
         warnings.warn(f"Calculated variance of difference is negative ({var_diff:.3g}). Check correlation and errors. Setting sigma_diff to 0.", RuntimeWarning)
         sigma_diff = 0.0
    else:
        sigma_diff = np.sqrt(var_diff)


    # Calculate test statistic (z_score or t_score)
    test_statistic = diff / sigma_diff if sigma_diff > 1e-15 else np.inf * np.sign(diff)
    if np.isclose(sigma_diff, 0) and np.isclose(diff, 0): test_statistic = 0.0 # Handle 0/0

    # Calculate p-value and critical value
    p_value = np.nan
    critical_value = np.nan
    dof = None

    if method == 'normal':
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        critical_value = stats.norm.ppf(1 - alpha/2)
    else: # student
        # Welch-Satterthwaite degrees of freedom
        if n_a > 1 and n_b > 1 and err_a > 1e-15 and err_b > 1e-15:
            num = (err_a**2 / n_a + err_b**2 / n_b)**2
            den = (err_a**4 / (n_a**2 * (n_a - 1))) + (err_b**4 / (n_b**2 * (n_b - 1)))
            if den > 1e-15:
                dof = num / den
            else: # Denominator is zero, implies infinite DoF? or just use Normal?
                warnings.warn("Could not calculate Welch-Satterthwaite DoF reliably (denominator near zero).", RuntimeWarning)
                # Fallback? Use a large DoF or Normal? Let scipy handle NaN DoF maybe?
                dof = np.inf # Treat as normal case
        elif n_a > 1 and n_b <=1: # Only A has valid DoF source
             dof = n_a - 1
        elif n_b > 1 and n_a <=1: # Only B has valid DoF source
             dof = n_b - 1
        else: # Neither have valid DoF source
             warnings.warn("Cannot calculate DoF for t-test with n_a <= 1 and n_b <= 1.", UserWarning)
             dof = np.nan # Indicate failure

        if dof is not None and not np.isnan(dof):
             if np.isinf(dof): # Handle fallback to normal
                  p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
                  critical_value = stats.norm.ppf(1 - alpha/2)
             else:
                 dof = max(1, dof) # Ensure DoF is at least 1 if calculated small
                 p_value = 2 * (1 - stats.t.cdf(abs(test_statistic), df=dof))
                 critical_value = stats.t.ppf(1 - alpha/2, df=dof)

    # Determine compatibility
    compatible = p_value >= alpha if not np.isnan(p_value) else False

    # Prepare result
    if output == 'pvalue':
        return p_value
    elif output == 'compatible':
        return compatible
    else: # 'full'
        result = {
            'measurement_a': Measurement(val_a, err_a),
            'measurement_b': Measurement(val_b, err_b),
            'difference': diff,
            'sigma_difference': sigma_diff,
            'test_statistic': test_statistic, # Was z_score
            'p_value': p_value,
            'alpha': alpha,
            'compatible': compatible,
            'critical_value': critical_value,
            'method': method,
            'correlation': corr_coef,
        }
        if method == 'student':
            result['dof'] = dof
            result['n_a'] = n_a
            result['n_b'] = n_b

        # Add interpretation
        if np.isnan(p_value):
             result['interpretation'] = "Compatibility test could not be completed."
        elif compatible:
            result['interpretation'] = f"The measurements are compatible at the {alpha*100:.1f}% significance level (p={p_value:.3g})."
        else:
            result['interpretation'] = f"The measurements differ significantly at the {alpha*100:.1f}% significance level (p={p_value:.3g})."

        # Optional: Add effect size (Cohen's d using pooled std dev if appropriate, or just diff/sigma_diff?)
        # pooled_std = np.sqrt(((n_a-1)*err_a**2 + (n_b-1)*err_b**2) / (n_a+n_b-2)) if method=='student' and n_a>1 and n_b>1 else np.sqrt((err_a**2 + err_b**2)/2)
        # result['cohens_d'] = abs(diff) / pooled_std if pooled_std > 1e-15 else np.inf

        if visualize:
            try:
                 # Pass actual values used
                 _test_comp_visualize(val_a, err_a, val_b, err_b, result)
            except Exception as e:
                 warnings.warn(f"Failed to generate visualization: {e}")

        return result


def _test_comp_visualize(val_a, err_a, val_b, err_b, results):
     """Helper to generate plot for test_comp."""
     fig, ax = plt.subplots(figsize=(8, 5))

     # Plot the measurements with error bars
     ax.errorbar([0], [val_a], yerr=[err_a], fmt='o', color='blue', capsize=5, markersize=8, label=f'A: {results["measurement_a"].round_to_error()}')
     ax.errorbar([1], [val_b], yerr=[err_b], fmt='o', color='red', capsize=5, markersize=8, label=f'B: {results["measurement_b"].round_to_error()}')

     # Add connecting line (optional, visual aid)
     # ax.plot([0, 1], [val_a, val_b], '--', color='gray', alpha=0.5)

     # Set plot limits and labels
     ax.set_xlim(-0.5, 1.5)
     ax.set_xticks([0, 1])
     ax.set_xticklabels(['Measurement A', 'Measurement B'])
     ax.set_ylabel('Value')
     ax.set_title('Measurement Comparison')
     ax.grid(True, linestyle=':', alpha=0.6)

     # Add compatibility information text
     p_value = results['p_value']
     compatible = results['compatible']
     alpha = results['alpha']
     if np.isnan(p_value):
          comp_text = "Test Inconclusive"
          color = "orange"
     else:
          comp_text = f"Compatible (p={p_value:.3g} >= {alpha})" if compatible else f"Incompatible (p={p_value:.3g} < {alpha})"
          color = "green" if compatible else "red"

     # Position text - adjust based on data range
     y_range = ax.get_ylim()
     y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0]) # Near top
     ax.text(0.5, y_pos, comp_text,
             ha='center', va='top', fontsize=11, color='white',
             bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.4'))

     ax.legend(loc='best')
     plt.tight_layout()
     if not _in_notebook():
         plt.show()
     return fig


# TODO: Implement test_comp_advanced (more complex statistical tests) if needed.
# It would follow a similar pattern, taking Measurement objects or values+errors.

# --- Example Usage Placeholder ---
if __name__ == "__main__":
    print("--- Measurement Class Examples ---")
    m1 = Measurement(10.0, 0.5)
    m2 = create_measurement([1, 2, 3], [0.1, 0.1, 0.2])
    print(f"m1: {m1}")
    print(f"m2: {m2}")
    print(f"m1 + 5: {m1 + 5}")
    print(f"m1 * m2[0]: {m1 * m2[0]}") # Scalar * Scalar
    try:
        print(f"m1 + m2: {m1 + m2}") # Broadcast scalar + array
    except ValueError as e:
        print(f"Error broadcasting m1 + m2: {e}") # Expect error if shapes mismatch

    m3 = Measurement([10, 20], [0.5, 1.0])
    m4 = Measurement([2, 4], [0.1, 0.2])
    print(f"m3 + m4: {m3 + m4}")
    print(f"m3 * m4: {m3 * m4}")
    print(f"m3 / m4: {m3 / m4}")
    print(f"m3 ** 2: {m3 ** 2}")
    print(f"2 ** m4: {2 ** m4}")
    print(f"m3 ** m4: {m3 ** m4}")

    print("\n--- NumPy ufunc Examples ---")
    print(f"np.sin(m2): {np.sin(m2)}")
    print(f"np.exp(m2[0]): {np.exp(m2[0])}")
    print(f"np.add(m3, m4): {np.add(m3, m4)}")
    print(f"np.log(m3): {np.log(m3)}")

    print("\n--- Formatting Example ---")
    m_fmt = Measurement(1234.56, 23.8)
    m_fmt_small = Measurement(0.00123, 0.00045)
    print(f"{m_fmt.round_to_error(n_sig_figs=1)}") # Should be 1230 ± 20
    print(f"{m_fmt.round_to_error(n_sig_figs=2)}") # Should be 1235 ± 24
    print(f"{m_fmt_small.round_to_error(n_sig_figs=1)}") # Should be 0.0012 ± 0.0005
    print(f"{m_fmt_small.round_to_error(n_sig_figs=2)}") # Should be 0.00123 ± 0.00045

    print("\n--- Fitting Example ---")
    # Generate some data with noise
    np.random.seed(42)
    true_a, true_b, true_c = 1.5, -2.0, 0.5
    x_true = np.linspace(0, 5, 20)
    y_true = true_a * x_true**2 + true_b * x_true + true_c
    x_meas = Measurement(x_true, 0.1) # Add x errors
    y_noise = np.random.normal(0, 0.8, size=x_true.shape)
    y_meas = Measurement(y_true + y_noise, 0.8) # Constant y error

    # Define model
    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c

    # Perform fit (using ODR because x has errors)
    fit_res = perform_fit(x_meas, y_meas, quadratic_model, parameter_names=['a', 'b', 'c'])

    print("Fit Successful:", fit_res['success'])
    print("Fit Method:", fit_res['fit_method'])
    print("Parameters:")
    for name, param in zip(fit_res['parameter_names'], fit_res['parameters']):
        print(f"  {name}: {param.round_to_error(2)}")
    print("Statistics:")
    for key, val in fit_res['statistics'].items():
        print(f"  {key}: {val:.4g}")

    # Generate fit results table
    print("\n--- Fit Results Table (LaTeX) ---")
    print(fit_results_table("Quad Fit", fit_res, orientation='v', stat_precision=3))

    # Plotting
    print("\n--- Plotting Fit ---")
    fig, ax = create_best_fit_line(
        fit_res, x_meas, y_meas,
        xlabel="X Value", ylabel="Y Value", title="Quadratic Fit Example",
        data_label="Measured Data", fit_label="Quadratic Fit",
        show_fit_params=True, show_stats=['reduced_chi_squared', 'r_squared'],
        show_residuals=True,
        show_plot=False # Don't block in script
    )
    # If running interactively or in a notebook, uncomment:
    if _in_notebook(): plt.show()


    print("\n--- Compatibility Test ---")
    m_a = Measurement(5.2, 0.3)
    m_b = Measurement(5.5, 0.4)
    comp_res = test_comp(m_a, m_b, visualize=False) # Set visualize=True to see plot
    print(comp_res['interpretation'])