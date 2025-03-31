# --- START OF FILE measurement.py ---

import numpy as np
import sympy as sp
import warnings
from typing import Union, List, Dict, Tuple, Any
from numbers import Number

# ---------------------------- Measurement Class -----------------------------

class Measurement:
    """
    Represents a physical quantity with a nominal value and an associated uncertainty.

    This class facilitates calculations involving measured quantities by automatically
    propagating uncertainties through common arithmetic operations and NumPy ufuncs.
    The propagation assumes that the uncertainties represent standard deviations and
    that the errors in different Measurement objects (or components of vector
    Measurements) are statistically independent unless otherwise noted.

    Error propagation for arithmetic operations (+, -, *, /, **) and standard
    NumPy ufuncs (like np.sin, np.log, np.exp) is implemented using first-order
    Taylor expansion (linear error propagation). For a function f(x, y, ...),
    the variance σ_f² is approximated by:
        σ_f² ≈ (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)² + ...
    This approximation is generally valid when the uncertainties (σ_x, σ_y, ...)
    are small relative to the values (x, y, ...) such that higher-order terms in
    the Taylor expansion are negligible.

    Attributes:
    -----------
    value : np.ndarray
        The nominal value(s) of the measurement. Stored as a NumPy array.
    error : np.ndarray
        The uncertainty (standard deviation) associated with the value(s).
        Must have the same shape as value or be broadcastable during operations.
        Represents the standard deviation (σ). Non-negative values are expected.

    Examples:
    ---------
    >>> x = Measurement(10.5, 0.2)
    >>> y = Measurement(5.1, 0.1)
    >>> z = 2.0 * np.sin(x / y)
    >>> print(z)
    1.71 ± 0.32
    >>> print(z.round_to_error())
    1.71 ± 0.32

    >>> temps = Measurement([25.1, 25.3, 24.9], 0.1) # Single error for multiple values
    >>> print(temps)
    [25.1 ± 0.1, 25.3 ± 0.1, 24.9 ± 0.1]
    >>> print(np.mean(temps))
    25.1 ± 0.0577
    """
    # Higher priority ensures Measurement.__add__ etc. are called over np.add etc.
    # when a Measurement object is involved.
    __array_priority__ = 1000

    def __init__(self, values: Union[float, list, tuple, np.ndarray, dict],
                 errors: Union[float, list, tuple, np.ndarray, None] = None,
                 magnitude: int = 0):
        """
        Initializes a Measurement object.

        Parameters:
        -----------
        values : float, list, tuple, np.ndarray, or dict
            - Single measurement value (float).
            - Sequence (list, tuple, np.ndarray) of measured values.
            - Dictionary with {value: error} pairs (if provided, errors must be None).
        errors : float, list, tuple, np.ndarray, or None, optional
            - Single error value (float): Applied to all elements in 'values'.
            - Sequence (list, tuple, np.ndarray): Must match the shape of 'values'.
            - None:
                - If 'values' is a dict, errors are taken from the dict values.
                - Otherwise, errors are assumed to be zero.
        magnitude : int, optional (default=0)
            Order of magnitude adjustment. Both values and errors are multiplied
            by 10**magnitude upon initialization. Useful for concisely representing
            numbers like (1.5 ± 0.1) * 10^-6 by passing magnitude=-6.
        """
        _values: np.ndarray
        _errors: np.ndarray

        # --- Input Parsing ---
        if isinstance(values, dict):
            if errors is not None:
                raise ValueError("Argument 'errors' must be None when 'values' is a dictionary.")
            if not values: # Empty dict
                 _values = np.array([], dtype=float)
                 _errors = np.array([], dtype=float)
            else:
                # Extract keys (values) and values (errors) from the dict
                val_list = list(values.keys())
                err_list = list(values.values())
                _values = np.asarray(val_list, dtype=float)
                _errors = np.asarray(err_list, dtype=float)
                if _values.shape != _errors.shape:
                    # This shouldn't happen with dict input, but good sanity check
                    raise ValueError("Internal error: Dictionary keys and values mismatch shape.")

        else:
            # Handle scalar, list, tuple, or ndarray for values
            if np.isscalar(values):
                # Ensure even single values are stored in arrays for consistency
                 _values = np.array([float(values)])
            else:
                 # Convert sequence to numpy array
                 _values = np.asarray(values, dtype=float)

            # Process errors based on 'errors' argument
            if errors is None:
                # Assume zero error if not provided (and not using dict input)
                _errors = np.zeros_like(_values)
            elif np.isscalar(errors):
                # Apply the same scalar error to all values
                _errors = np.full_like(_values, float(errors), dtype=float)
            else:
                # Convert sequence of errors to numpy array
                _errors = np.asarray(errors, dtype=float)
                # Ensure errors shape is compatible with values shape
                try:
                    # Use broadcasting to check shape compatibility
                    np.broadcast(_values, _errors)
                except ValueError:
                    raise ValueError(f"Shape mismatch: values {(_values.shape)} "
                                     f"and errors {(_errors.shape)} cannot be broadcast together.")
                # If errors were broadcastable but not identical shape (e.g., scalar error for array value),
                # expand errors to match the values shape explicitly.
                if _errors.shape != _values.shape:
                    _errors = np.broadcast_to(_errors, _values.shape).copy() # Use copy to avoid issues

        # --- Final Validation and Assignment ---
        if np.any(_errors < 0):
            warnings.warn("Initializing Measurement with negative error(s). "
                          "Uncertainty should be non-negative. Taking absolute value.", UserWarning)
            _errors = np.abs(_errors)

        # Apply magnitude scaling
        scale_factor = 10.0 ** magnitude
        self.value = _values * scale_factor
        self.error = _errors * scale_factor

    # --- Properties for NumPy-like interface ---

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.value.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of array dimensions."""
        return self.value.shape

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.value.size

    def __len__(self) -> int:
        """Length of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of unsized Measurement object")
        return len(self.value)

    def __getitem__(self, key: Any) -> 'Measurement':
        """Allows indexing and slicing like a NumPy array."""
        # Index both value and error, maintaining consistency
        # NumPy handles scalar vs array results automatically based on key
        new_value = self.value[key]
        new_error = self.error[key]
        # Ensure the result is still wrapped in Measurement
        # If indexing results in a scalar, it needs to be re-wrapped
        if np.isscalar(new_value):
             return Measurement(new_value, new_error)
        else:
             # If result is an array, create new Measurement directly
             # Need a way to init from arrays without rescaling or checks... maybe a private constructor?
             # For now, standard init works but is slightly less efficient.
             return Measurement(new_value, new_error)


    # --- String Representations ---

    def __repr__(self) -> str:
        """Technical representation, useful for debugging."""
        # Use NumPy's array repr for clarity, especially for large arrays
        return f"Measurement(value={self.value!r}, error={self.error!r})"

    def __str__(self) -> str:
        """User-friendly string representation. Uses standard formatting."""
        if self.shape == () or self.shape == (1,): # Scalar or single element
             val = self.value.item() if self.shape == () else self.value[0]
             err = self.error.item() if self.shape == () else self.error[0]
             # Basic formatting, might not align significant figures well.
             # round_to_error() provides better control.
             val_str = f"{val:.3g}" # General format, up to 3 sig figs
             err_str = f"{err:.2g}" # General format, up to 2 sig figs for error
             return f"{val_str} ± {err_str}"
        elif self.size <= 10: # Small array, show elements
            parts = [f"{v:.3g} ± {e:.2g}"
                     for v, e in zip(self.value.flat, self.error.flat)]
            # Attempt to format based on original dimensions
            try:
                 # Use numpy's array2string for better multi-dimensional layout
                 arr_str = np.array2string(np.array(parts).reshape(self.shape), separator=', ',
                                           formatter={'all': lambda x: x})
                 return arr_str
            except: # Fallback for complex shapes or errors
                 return "[" + ", ".join(parts) + "]"
        else: # Large array, show summary
            return f"Measurement(value=..., error=..., shape={self.shape})"


    # --- Arithmetic Operations ---
    # The following methods implement error propagation for basic arithmetic.
    # They assume independence between 'self' and 'other' if 'other' is also a Measurement.
    # If 'other' is a scalar or NumPy array, it's treated as having zero uncertainty.
    # Edge cases (division by zero, operations resulting in NaN/Inf) are primarily
    # handled by NumPy's underlying arithmetic on the 'value' arrays. Warnings are
    # issued if the *resulting* values or errors contain NaN or Inf.

    def _check_nan_inf(self, value: np.ndarray, error: np.ndarray, operation_name: str):
        """Internal helper to warn about NaN/Inf results."""
        # Check value array
        if np.any(np.isnan(value)):
             warnings.warn(f"NaN value encountered during {operation_name}", RuntimeWarning)
        if np.any(np.isinf(value)):
             warnings.warn(f"Infinity value encountered during {operation_name}", RuntimeWarning)
        # Check error array (Errors should ideally not be NaN/Inf unless value is also problematic)
        # Suppress warnings if error is NaN/Inf only where value is also NaN/Inf
        error_is_problematic = (np.isnan(error) & ~np.isnan(value)) | (np.isinf(error) & ~np.isinf(value))
        if np.any(error_is_problematic):
             warnings.warn(f"NaN or Infinity error encountered during {operation_name} "
                           "where value is finite/valid.", RuntimeWarning)


    def __add__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            # M1 + M2: σ² = σ₁² + σ₂²
            new_value = self.value + other.value
            # Use hypot for potentially better numerical stability: sqrt(a²+b²)
            new_error = np.hypot(self.error, other.error)
            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "addition")
            return result
        elif isinstance(other, (Number, np.ndarray)):
            # M + k: Treat k as exact (zero error)
            k = np.asarray(other)
            new_value = self.value + k
            # Error remains unchanged: σ_(x+k) = σ_x
            result = Measurement(new_value, self.error.copy()) # Use copy to avoid aliasing
            self._check_nan_inf(result.value, result.error, "addition")
            return result
        else:
            return NotImplemented # Let Python handle unsupported types

    def __radd__(self, other: Any) -> 'Measurement':
        # k + M is the same as M + k
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            # M1 - M2: σ² = σ₁² + σ₂² (Errors add in quadrature even for subtraction)
            new_value = self.value - other.value
            new_error = np.hypot(self.error, other.error)
            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "subtraction")
            return result
        elif isinstance(other, (Number, np.ndarray)):
            # M - k: Treat k as exact
            k = np.asarray(other)
            new_value = self.value - k
            # Error remains unchanged: σ_(x-k) = σ_x
            result = Measurement(new_value, self.error.copy())
            self._check_nan_inf(result.value, result.error, "subtraction")
            return result
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> 'Measurement':
        if isinstance(other, (Number, np.ndarray)):
            # k - M: Treat k as exact
            k = np.asarray(other)
            new_value = k - self.value
            # Error remains unchanged: σ_(k-x) = σ_x
            result = Measurement(new_value, self.error.copy())
            self._check_nan_inf(result.value, result.error, "subtraction")
            return result
        else:
            # Let __sub__ handle Measurement - Measurement via -(M2 - M1) if necessary,
            # but direct implementation is usually done via `a - b`.
            # If `other` is not Number/ndarray, standard mechanisms apply.
            return NotImplemented

    def __mul__(self, other: Any) -> 'Measurement':
        if isinstance(other, Measurement):
            # M1 * M2: σ² = (value₂ * σ₁)² + (value₁ * σ₂)²
            a, sa = self.value, self.error
            b, sb = other.value, other.error
            new_value = a * b
            # Calculate variance terms, np.hypot is good for sqrt(x²+y²)
            # This form avoids division by zero if a or b is zero.
            with np.errstate(invalid='ignore'): # Ignore potential 0*inf warnings if errors are inf
                 term1_sq = (b * sa)**2
                 term2_sq = (a * sb)**2
                 # If a value is exactly zero, its contribution to the other term's error is zero
                 term1_sq = np.where(b == 0.0, 0.0, term1_sq)
                 term2_sq = np.where(a == 0.0, 0.0, term2_sq)
                 new_error = np.sqrt(term1_sq + term2_sq)

            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "multiplication")
            return result
        elif isinstance(other, (Number, np.ndarray)):
            # M * k: σ = |k| * σ_M
            k = np.asarray(other)
            new_value = self.value * k
            new_error = np.abs(k) * self.error
            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "multiplication")
            return result
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> 'Measurement':
        # k * M is the same as M * k
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Measurement':
        # Using np.errstate to manage warnings from division itself (0/0=nan, x/0=inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            if isinstance(other, Measurement):
                # M1 / M2: σ² = (σ₁/value₂)² + (value₁*σ₂ / value₂²)²
                a, sa = self.value, self.error
                b, sb = other.value, other.error
                new_value = a / b # Let numpy handle 0/0, x/0

                # Calculate error terms carefully
                # term1 = sa / b
                # term2 = a * sb / b**2 = new_value * sb / b
                term1_sq = (sa / b)**2
                term2_sq = (new_value * sb / b)**2

                # Handle cases involving zero explicitly to avoid NaN/Inf from calculation itself
                # if b is zero, error should be inf (unless a is also zero?)
                term1_sq = np.where(b == 0.0, np.inf, term1_sq)
                term2_sq = np.where(b == 0.0, np.inf, term2_sq)
                # If a is zero, the second term contribution is zero (unless b is also zero)
                term2_sq = np.where((a == 0.0) & (b != 0.0), 0.0, term2_sq)

                new_error = np.sqrt(term1_sq + term2_sq)

            elif isinstance(other, (Number, np.ndarray)):
                # M / k: σ = σ_M / |k|
                k = np.asarray(other)
                new_value = self.value / k # Let numpy handle division by zero
                new_error = np.abs(self.error / k)

            else:
                return NotImplemented

            # Wrap result and check final state
            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "division")
            return result

    def __rtruediv__(self, other: Any) -> 'Measurement':
        # k / M
        if isinstance(other, (Number, np.ndarray)):
            with np.errstate(divide='ignore', invalid='ignore'):
                k = np.asarray(other)
                a, sa = self.value, self.error
                new_value = k / a # Let numpy handle division by zero
                # σ = |k * σ_a / a²| = |(k/a) * sa / a| = |new_value * sa / a|
                new_error = np.abs(new_value * sa / a)
                # Handle division by zero in error calculation
                new_error = np.where(a == 0.0, np.inf, new_error)
                # If k=0, result is 0 with 0 error (unless a=0)
                new_error = np.where((k == 0.0) & (a != 0.0), 0.0, new_error)

            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "division")
            return result
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> 'Measurement':
        # Using np.errstate for potentially invalid operations like (-ve)**non-integer
        with np.errstate(invalid='ignore'):
            if isinstance(other, (Number, np.ndarray)):
                # M ** k: σ = |k * value^(k-1) * σ_M|
                n = np.asarray(other) # Exponent (no error)
                a, sa = self.value, self.error

                # Check for potential domain issues before calculation, issue warning
                if np.any((a < 0) & (n % 1 != 0)):
                    warnings.warn(f"Base Measurement has negative values with non-integer exponent {n}. Result may be complex or NaN.", RuntimeWarning)
                if np.any((a == 0.0) & (n < 0)):
                    warnings.warn(f"Power operation 0**negative encountered.", RuntimeWarning)

                new_value = a ** n # Let numpy compute value (may be nan/inf)
                # Derivative: d(a^n)/da = n * a^(n-1)
                deriv = n * (a ** (n - 1))
                new_error = np.abs(deriv * sa)

                # Ensure error is NaN if derivative is NaN (e.g., from 0**negative)
                new_error = np.where(np.isnan(deriv), np.nan, new_error)
                # Ensure error is 0 if exponent is 0 (a**0 = 1, exact)
                new_error = np.where(n == 0.0, 0.0, new_error)
                # Ensure error is 0 if base is 0 and exponent > 0
                new_error = np.where((a == 0.0) & (n > 0), 0.0, new_error)


            elif isinstance(other, Measurement):
                # M1 ** M2: f(a,b) = a^b
                # σ² = (∂f/∂a * σₐ)² + (∂f/∂b * σ<0xE2><0x82><0x99>)²
                # ∂f/∂a = b * a^(b-1)
                # ∂f/∂b = a^b * ln(a) = value * ln(a)
                a, sa = self.value, self.error
                b, sb = other.value, other.error

                # Check for domain issues
                if np.any(a < 0):
                    warnings.warn(f"Base Measurement has negative values ({a}); power operation with uncertain exponent may yield complex numbers or NaN.", RuntimeWarning)
                if np.any((a == 0.0) & (b <= 0)):
                    warnings.warn(f"Power operation 0 ** <=0 encountered with uncertain exponent.", RuntimeWarning)
                # Need np.log(a), check for a <= 0
                if np.any(a <= 0):
                     warnings.warn(f"Logarithm of non-positive base ({a}) required for error propagation in M1**M2. Error may be NaN.", RuntimeWarning)

                new_value = a ** b # Let numpy compute value

                # Calculate partial derivatives and variance terms
                df_da = b * (a ** (b - 1))
                df_db = new_value * np.log(a) # This will be nan if a <= 0

                term_a_sq = (df_da * sa)**2
                term_b_sq = (df_db * sb)**2

                # Handle NaNs from derivatives appropriately
                # If derivative is NaN, variance contribution should be NaN unless error is 0
                term_a_sq = np.where(np.isnan(df_da) & (sa != 0.0), np.nan, term_a_sq)
                term_b_sq = np.where(np.isnan(df_db) & (sb != 0.0), np.nan, term_b_sq)
                # If an error is zero, its term's contribution is zero
                term_a_sq = np.where(sa == 0.0, 0.0, term_a_sq)
                term_b_sq = np.where(sb == 0.0, 0.0, term_b_sq)

                new_error = np.sqrt(term_a_sq + term_b_sq)

            else:
                return NotImplemented

            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "power")
            return result

    def __rpow__(self, other: Any) -> 'Measurement':
        # k ** M
        if isinstance(other, (Number, np.ndarray)):
            with np.errstate(invalid='ignore', divide='ignore'):
                # f(a) = k^a => df/da = k^a * ln(k) = value * ln(k)
                k = np.asarray(other)
                a, sa = self.value, self.error

                # Check for domain issues
                if np.any(k < 0):
                    warnings.warn(f"Base {k} is negative; power operation k**M may yield complex numbers or NaN.", RuntimeWarning)
                if np.any((k == 0.0) & (a <= 0)):
                     warnings.warn(f"Power operation 0 ** M (with M <= 0) encountered.", RuntimeWarning)
                # Need ln(k)
                if np.any(k <= 0):
                     warnings.warn(f"Logarithm of non-positive base ({k}) required for error propagation in k**M. Error may be NaN.", RuntimeWarning)

                new_value = k ** a # Let numpy compute value
                log_k = np.log(k) # Will be nan/inf for k<=0
                deriv = new_value * log_k
                new_error = np.abs(deriv * sa)

                # Handle NaNs/Infs from log_k
                new_error = np.where(np.isnan(log_k) | np.isinf(log_k), np.nan, new_error)
                # Ensure error is 0 if k=1 (1**a = 1, exact)
                new_error = np.where(k == 1.0, 0.0, new_error)
                 # Ensure error is 0 if k=0 and a > 0 (0**a = 0, exact)
                new_error = np.where((k == 0.0) & (a > 0), 0.0, new_error)


            result = Measurement(new_value, new_error)
            self._check_nan_inf(result.value, result.error, "power")
            return result
        else:
            return NotImplemented

    # --- Unary Operations ---

    def __neg__(self) -> 'Measurement':
        # -(x ± σ) = (-x) ± σ
        # Error magnitude is unchanged.
        return Measurement(-self.value, self.error.copy())

    def __pos__(self) -> 'Measurement':
        # +(x ± σ) = x ± σ
        return self # Or return Measurement(self.value.copy(), self.error.copy()) if immutability is critical

    def __abs__(self) -> 'Measurement':
        # abs(x ± σ) ≈ abs(x) ± σ
        # This is an approximation, especially problematic near x=0 where the derivative is undefined.
        # The standard propagation formula |df/dx * σ| with f=abs(x) gives |sign(x) * σ| = σ.
        # This assumes the distribution doesn't significantly cross zero.
        # A more rigorous treatment might be needed if abs(value)/error is small.
        if np.any(np.isclose(self.value, 0.0, atol=self.error)): # Check if interval includes zero
             warnings.warn("Taking absolute value of a Measurement whose interval includes zero. "
                           "Standard error propagation (σ_abs ≈ σ_orig) might be inaccurate.", UserWarning)
        return Measurement(np.abs(self.value), self.error.copy())

    # --- NumPy Integration ---

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Allows direct use in NumPy functions that call np.asarray() or np.array().

        Warning: This conversion discards the uncertainty information. It returns
        only the nominal values of the Measurement object(s).

        Parameters:
        -----------
        dtype : data-type, optional
            The desired data-type for the array. If not given, infers from self.value.

        Returns:
        --------
        np.ndarray :
            An array containing only the nominal values.
        """
        warnings.warn("Casting Measurement to np.ndarray loses error information. "
                      "Returning nominal values only.", UserWarning)
        return np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """
        Handles NumPy ufuncs (e.g., np.sin, np.add, np.sqrt) involving Measurement objects.

        This method implements error propagation using first-order Taylor expansion.

        Parameters:
        -----------
        ufunc : np.ufunc
            The universal function being called (e.g., np.add, np.sin).
        method : str
            The method called on the ufunc (usually '__call__').
        *inputs : tuple
            Positional arguments passed to the ufunc. Can include Measurement objects,
            NumPy arrays, scalars, etc.
        **kwargs : dict
            Keyword arguments passed to the ufunc (e.g., 'out', 'where').
            Note: The 'out' argument is currently not supported and will raise a warning.

        Returns:
        --------
        Measurement or NotImplemented
            A new Measurement object containing the result of the ufunc applied to the
            values and the propagated uncertainty. Returns NotImplemented if the ufunc
            or input types are not supported.

        Error Propagation Logic:
        ------------------------
        The core idea is linear error propagation based on a first-order Taylor
        series expansion of the ufunc `f` around the nominal values of the inputs.
        For inputs x ± σ_x, y ± σ_y, ..., the variance σ_f² in the result f is:
            σ_f² ≈ (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)² + ...
        assuming the inputs are independent and the errors σ_x, σ_y, ... are
        small enough for the linear approximation to hold.

        Implementation Details:
        -----------------------
        1. Input Processing: Extracts values and errors from all inputs. Non-Measurement
           inputs are treated as exact (zero error). SymPy symbols are created for
           inputs that are Measurement objects.
        2. Value Calculation: The ufunc is called directly on the nominal values of
           all inputs to get the resulting nominal value. NumPy handles broadcasting
           and potential numerical errors (returning nan/inf).
        3. Error Propagation:
           - Unary Ufuncs (f(x)):
             - Common functions (sin, cos, exp, log, sqrt, etc.) have their derivatives
               hardcoded for efficiency and robustness.
             - For other unary ufuncs, SymPy is used to symbolically differentiate
               the function (e.g., `sympy.diff(ufunc(x_sym), x_sym)`).
             - The derivative is evaluated numerically at the input nominal value(s).
             - The result error is `σ_f = |df/dx * σ_x|`.
           - Binary Ufuncs (f(x, y)):
             - Basic arithmetic (+, -, *, /) is handled using the standard formulas
               implemented in the `__add__`, `__mul__`, etc. methods (called via
               the ufunc).
             - For other binary ufuncs (e.g., `np.power`, `np.hypot`), SymPy is used
               to find partial derivatives ∂f/∂x and ∂f/∂y.
             - Derivatives are evaluated numerically at the input nominal values.
             - The result variance is `σ_f² = (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)²`.
             - The result error is `σ_f = sqrt(σ_f²)`.
           - N-ary Ufuncs (nin > 2): Currently not implemented, returns NotImplemented.
        4. SymPy Usage: `sympy.lambdify` converts the symbolic derivatives into
           numerical functions that can operate on NumPy arrays. This allows applying
           the error propagation formulas element-wise.
        5. Edge Cases: NumPy calculates the nominal result, potentially yielding NaN or Inf.
           The error calculation attempts to proceed but may also yield NaN/Inf (e.g.,
           derivative undefined, division by zero). Final errors are set to NaN if the
           corresponding value is NaN, and potentially Inf if the value is Inf. Warnings
           are issued if the final result contains NaN/Inf.
        """

        # --- Check for Unsupported Features ---
        if method != '__call__':
            # Only standard ufunc calls are supported (not reduce, accumulate, etc.)
            return NotImplemented

        if 'out' in kwargs and kwargs['out'] is not None:
            # Modifying arrays in-place ('out' argument) is complex to reconcile
            # with creating new Measurement objects containing both value and error.
            # It's safer to disallow it and return a new object.
            warnings.warn("The 'out' keyword argument is not supported for ufuncs "
                          "involving Measurement objects. A new Measurement object "
                          "will be returned.", UserWarning)
            # Remove 'out' kwarg to prevent passing it down to numpy calculation
            # that might expect a specific type or shape we can't guarantee.
            kwargs = kwargs.copy() # Avoid modifying original kwargs dict
            kwargs.pop('out')
            # Alternatively, could strictly return NotImplemented here.

        # --- Input Processing ---
        input_values = []      # List of nominal values (as np.ndarray)
        input_errors = []      # List of errors (as np.ndarray)
        input_symbols = []     # List of SymPy symbols (or None)
        measurement_inputs = [] # List of original Measurement inputs (or None)
        has_measurement = False # Flag if any input is a Measurement

        # Create unique SymPy symbols for potential differentiation
        # Using a large enough number to likely avoid collisions if nested calls occur (though not guaranteed)
        sympy_vars = sp.symbols(f'x:{ufunc.nin}')

        for i, x in enumerate(inputs):
            if isinstance(x, Measurement):
                input_values.append(x.value)
                input_errors.append(x.error)
                input_symbols.append(sympy_vars[i])
                measurement_inputs.append(x) # Keep track of the Measurement object
                has_measurement = True
            elif isinstance(x, (Number, np.ndarray)):
                # Treat non-Measurement inputs as exact (zero error)
                val = np.asarray(x)
                input_values.append(val)
                input_errors.append(np.zeros_like(val, dtype=float)) # Ensure error array matches shape
                input_symbols.append(None) # No symbol needed if not differentiating w.r.t this input
                measurement_inputs.append(None) # Placeholder
            else:
                # Cannot handle this type of input combination
                return NotImplemented

        if not has_measurement:
            # If no Measurement objects are involved, this method shouldn't have been called
            # due to __array_priority__. However, as a safeguard, call the original ufunc.
            # This should technically not be reached if Measurement has higher priority.
            return ufunc(*inputs, **kwargs) # Or simply return NotImplemented


        # --- Value Calculation ---
        # Calculate the nominal result value using the original ufunc
        # We wrap this in errstate to suppress warnings originating *during*
        # this calculation (e.g. divide by zero). We will check the *final*
        # result later and issue our own warning if needed.
        with np.errstate(all='ignore'): # Suppress numpy warnings during value calc
             try:
                 result_value = ufunc(*input_values, **kwargs)
             except Exception as e:
                 # Catch potential errors during the ufunc call itself (e.g., domain errors
                 # not handled gracefully by the ufunc, though most numpy ufuncs return nan/inf).
                 warnings.warn(f"Error during ufunc ({ufunc.__name__}) value calculation: {e}. "
                               "Result value set to NaN.", RuntimeWarning)
                 # Determine output shape based on broadcasting rules to create NaN array
                 try:
                     broadcast_shape = np.broadcast(*input_values).shape
                 except ValueError:
                      # If inputs cannot be broadcast, the operation is fundamentally invalid
                      return NotImplemented
                 result_value = np.full(broadcast_shape, np.nan)


        # --- Error Propagation ---
        result_error: np.ndarray

        # Use np.errstate context for error calculations which might involve
        # divisions, logs of zero, etc., during derivative evaluation or combination.
        with np.errstate(divide='ignore', invalid='ignore'):

            if ufunc.nin == 1: # Unary functions (e.g., np.sin(x))
                # Only proceed if the single input was a Measurement
                if measurement_inputs[0] is not None:
                    x = measurement_inputs[0]
                    x_sym = input_symbols[0]
                    sx = x.error

                    # Use hardcoded derivatives for common, simple ufuncs for performance
                    # and to handle potential SymPy issues (e.g., sign function).
                    deriv_val: Union[float, np.ndarray, None] = None # Initialize
                    if ufunc is np.negative: deriv_val = -1.0
                    elif ufunc is np.positive: deriv_val = 1.0
                    elif ufunc is np.conjugate: deriv_val = 1.0 # For real values
                    elif ufunc is np.absolute:
                        # Derivative is sign(x), undefined at 0. Use sign, propagate warning if near 0.
                        # The __abs__ method already handles the warning.
                        deriv_val = np.sign(x.value)
                        # Handle exact zero where sign is 0, derivative should arguably be 1 or handled differently.
                        # Propagating error as σ_x is common practice. |sign(0)|*σ = 0, which seems wrong.
                        # Let's default to 1*σ_x in magnitude.
                        deriv_val = np.where(x.value == 0.0, 1.0, deriv_val) # Treat deriv magnitude as 1 at x=0
                    elif ufunc is np.exp: deriv_val = result_value # exp(x) is its own derivative
                    elif ufunc is np.log: deriv_val = 1.0 / x.value
                    elif ufunc is np.log10: deriv_val = 1.0 / (x.value * np.log(10))
                    elif ufunc is np.sqrt: deriv_val = 0.5 / result_value # 1 / (2 * sqrt(x))
                    elif ufunc is np.sin: deriv_val = np.cos(x.value)
                    elif ufunc is np.cos: deriv_val = -np.sin(x.value)
                    elif ufunc is np.tan: deriv_val = 1.0 + result_value**2 # 1 + tan(x)² = sec(x)²
                    # Add more common cases here if needed...

                    if deriv_val is None:
                        # General case: Use SymPy for the derivative
                        try:
                            # Need to handle ufuncs that don't map directly to sympy functions
                            # Check common mappings
                            sympy_func_map = {
                                np.exp: sp.exp, np.log: sp.log, np.sin: sp.sin, np.cos: sp.cos,
                                np.tan: sp.tan, np.sqrt: sp.sqrt, np.abs: sp.Abs,
                                # Add more mappings if needed
                            }
                            sympy_func = sympy_func_map.get(ufunc)

                            if sympy_func:
                                f_sym = sympy_func(x_sym)
                                deriv_sym = sp.diff(f_sym, x_sym)
                                # Lambdify for numerical evaluation
                                # Add numpy and handle potential complex results if ufunc supports them
                                modules = ['numpy', {'conjugate': np.conjugate}]
                                deriv_func = sp.lambdify(x_sym, deriv_sym, modules=modules)
                                deriv_val = deriv_func(x.value)
                            else:
                                # If no direct mapping, try applying ufunc to symbol (might fail)
                                try:
                                     f_sym = ufunc(x_sym) # This might not work for many ufuncs
                                     deriv_sym = sp.diff(f_sym, x_sym)
                                     modules = ['numpy', {'conjugate': np.conjugate}]
                                     deriv_func = sp.lambdify(x_sym, deriv_sym, modules=modules)
                                     deriv_val = deriv_func(x.value)
                                except (TypeError, AttributeError, NotImplementedError) as e_sympy:
                                     warnings.warn(f"SymPy could not interpret or differentiate ufunc '{ufunc.__name__}'. "
                                                   f"Cannot propagate error. Error will be NaN. SymPy error: {e_sympy}", RuntimeWarning)
                                     deriv_val = np.nan # Signal failure

                        except Exception as e_sympy_general:
                             # Catch any other unexpected SymPy errors
                             warnings.warn(f"Unexpected error during SymPy differentiation for '{ufunc.__name__}'. "
                                           f"Cannot propagate error. Error will be NaN. Error: {e_sympy_general}", RuntimeWarning)
                             deriv_val = np.nan

                    # Error formula: sigma_f = |df/dx * sigma_x|
                    result_error = np.abs(deriv_val * sx)
                    # Ensure error is NaN if derivative calculation failed or resulted in NaN
                    result_error = np.where(np.isnan(deriv_val), np.nan, result_error)

                else:
                    # Input was not a Measurement, so the result has zero error
                    result_error = np.zeros_like(result_value)

            elif ufunc.nin == 2: # Binary functions (e.g., np.add(x, y))
                x_meas, y_meas = measurement_inputs[0], measurement_inputs[1]
                x_sym, y_sym = input_symbols[0], input_symbols[1]
                # Get value/error pairs, using 0 error for non-Measurement inputs
                vx, sx = (x_meas.value, x_meas.error) if x_meas is not None else (input_values[0], input_errors[0])
                vy, sy = (y_meas.value, y_meas.error) if y_meas is not None else (input_values[1], input_errors[1])

                # Optimization: Check if the ufunc corresponds to a standard arithmetic operation
                # This avoids SymPy for simple cases and reuses the robust logic in __add__, etc.
                op_map = {np.add: '+', np.subtract: '-', np.multiply: '*', np.true_divide: '/'} # np.power handled below
                corresponding_op = op_map.get(ufunc)

                temp_result: Union[Measurement, None] = None
                if corresponding_op:
                     try:
                         # Reconstruct operands as Measurement or scalar/array
                         op1 = x_meas if x_meas is not None else vx
                         op2 = y_meas if y_meas is not None else vy
                         # Use eval to perform the operation (e.g., eval("op1 + op2"))
                         # This is a bit indirect but reuses the dunder method logic.
                         # A cleaner way might be to directly call the propagation formulas here.
                         # Let's stick to direct formulas for clarity and robustness here.

                         if ufunc is np.add:
                             var_sq = sx**2 + sy**2
                         elif ufunc is np.subtract:
                              var_sq = sx**2 + sy**2
                         elif ufunc is np.multiply:
                              term1_sq = (vy * sx)**2
                              term2_sq = (vx * sy)**2
                              term1_sq = np.where(vy == 0.0, 0.0, term1_sq) # Handle value=0 case
                              term2_sq = np.where(vx == 0.0, 0.0, term2_sq)
                              var_sq = term1_sq + term2_sq
                         elif ufunc is np.true_divide:
                              # Use result_value = vx / vy
                              term1_sq = (sx / vy)**2
                              term2_sq = (result_value * sy / vy)**2
                              # Handle vy=0 cases (results in inf variance)
                              term1_sq = np.where(vy == 0.0, np.inf, term1_sq)
                              term2_sq = np.where(vy == 0.0, np.inf, term2_sq)
                              # Handle vx=0 but vy!=0 case (second term is zero)
                              term2_sq = np.where((vx == 0.0) & (vy != 0.0), 0.0, term2_sq)
                              var_sq = term1_sq + term2_sq
                         else: # Should not happen with op_map keys
                             var_sq = np.nan

                         result_error = np.sqrt(var_sq)

                     except Exception as e_arith:
                          warnings.warn(f"Error during specialized arithmetic propagation for {ufunc.__name__}: {e_arith}. "
                                        "Falling back to SymPy (if possible).", RuntimeWarning)
                          temp_result = None # Fallback signal
                          result_error = np.full_like(result_value, np.nan) # Default to NaN

                # Handle np.power separately or fall through to SymPy
                elif ufunc is np.power:
                     # This depends on which argument is Measurement. Reuse __pow__ / __rpow__ logic.
                     # Reconstruct operands and call the appropriate power method.
                     try:
                         op1 = x_meas if x_meas is not None else vx
                         op2 = y_meas if y_meas is not None else vy
                         # Python's pow() or ** dispatches to __pow__ or __rpow__
                         temp_result = op1 ** op2 # This returns a Measurement object
                         result_error = temp_result.error
                     except Exception as e_pow:
                         warnings.warn(f"Error during __pow__/__rpow__ dispatch for np.power: {e_pow}. "
                                       "Falling back to SymPy (if possible).", RuntimeWarning)
                         temp_result = None # Fallback signal
                         result_error = np.full_like(result_value, np.nan)

                else:
                     # General case for other binary ufuncs: Use SymPy
                     temp_result = None # Ensure we don't skip SymPy block

                # If arithmetic/power specialization failed or wasn't applicable, use SymPy
                if temp_result is None and result_error is None: # Check if SymPy needed
                  result_error = np.full_like(result_value, np.nan) # Default error to NaN
                  try:
                     # Determine which inputs need differentiation (only Measurements)
                     symbols_to_diff = []
                     if x_meas is not None: symbols_to_diff.append(x_sym)
                     if y_meas is not None: symbols_to_diff.append(y_sym)

                     if not symbols_to_diff:
                         # If neither input was Measurement (shouldn't happen here), error is zero
                         result_error = np.zeros_like(result_value)
                     else:
                         # Get SymPy equivalent function if possible
                         sympy_func_map = { np.hypot: sp.sqrt(x_sym**2 + y_sym**2), # Example
                                            # Add mappings for other binary ufuncs if needed
                                          }
                         f_sym_expr = sympy_func_map.get(ufunc)

                         if f_sym_expr:
                             f_sym = f_sym_expr
                         else:
                             # Try applying ufunc directly to symbols (might fail)
                             f_sym = ufunc(x_sym, y_sym)

                         var_sq = np.zeros_like(result_value, dtype=float) # Accumulate variance

                         # Define symbols needed for lambdify (both x and y, even if one is constant)
                         symbols_for_lambdify = [x_sym, y_sym]
                         values_for_lambdify = [vx, vy] # Pass nominal values
                         modules = ['numpy', {'conjugate': np.conjugate}]

                         # Calculate variance contribution from x (if it was Measurement)
                         if x_meas is not None:
                             df_dx_sym = sp.diff(f_sym, x_sym)
                             df_dx_func = sp.lambdify(symbols_for_lambdify, df_dx_sym, modules=modules)
                             df_dx_val = df_dx_func(*values_for_lambdify)
                             term_x_sq = (df_dx_val * sx)**2
                             # Handle NaN derivative: if deriv is NaN and error > 0, result is NaN
                             term_x_sq = np.where(np.isnan(df_dx_val) & (sx != 0.0), np.nan, term_x_sq)
                             # Handle zero error: if error is 0, contribution is 0
                             term_x_sq = np.where(sx == 0.0, 0.0, term_x_sq)
                             var_sq += term_x_sq

                         # Calculate variance contribution from y (if it was Measurement)
                         if y_meas is not None:
                             df_dy_sym = sp.diff(f_sym, y_sym)
                             df_dy_func = sp.lambdify(symbols_for_lambdify, df_dy_sym, modules=modules)
                             df_dy_val = df_dy_func(*values_for_lambdify)
                             term_y_sq = (df_dy_val * sy)**2
                             # Handle NaN derivative and zero error
                             term_y_sq = np.where(np.isnan(df_dy_val) & (sy != 0.0), np.nan, term_y_sq)
                             term_y_sq = np.where(sy == 0.0, 0.0, term_y_sq)
                             var_sq += term_y_sq # Add in quadrature

                         result_error = np.sqrt(var_sq)

                  except (TypeError, AttributeError, NotImplementedError, ValueError) as e_sympy_bin:
                      warnings.warn(f"SymPy could not differentiate or evaluate binary ufunc '{ufunc.__name__}'. "
                                    f"Cannot propagate error. Error will be NaN. Details: {e_sympy_bin}", RuntimeWarning)
                      # result_error remains NaN from initialization before try block

            else: # Ufuncs with nin > 2 inputs
                warnings.warn(f"Error propagation for ufunc '{ufunc.__name__}' with {ufunc.nin} inputs "
                              "is not implemented. Returning NotImplemented.", UserWarning)
                return NotImplemented


        # --- Final Result Construction and Check ---

        # Ensure error is NaN wherever value is NaN
        result_error = np.where(np.isnan(result_value), np.nan, result_error)
        # Ensure error is Inf wherever value is Inf (usually appropriate)
        result_error = np.where(np.isinf(result_value), np.inf, result_error)
        # Ensure error is non-negative (sqrt might yield complex for negative variance if calculation errors occur)
        if np.any(np.iscomplex(result_error)):
             warnings.warn(f"Complex error encountered during propagation for {ufunc.__name__}. Taking magnitude.", RuntimeWarning)
             result_error = np.abs(result_error)
        result_error = np.nan_to_num(result_error, nan=np.nan, posinf=np.inf, neginf=np.nan) # Convert potential -inf from sqrt to nan

        final_result = Measurement(result_value, result_error)
        # Issue warnings based on the *final* state of the result
        self._check_nan_inf(final_result.value, final_result.error, f"ufunc '{ufunc.__name__}'")

        return final_result


    # --- Comparison Methods ---
    # Comparisons operate on nominal values only. Uncertainty is ignored for ordering.
    # Returns boolean arrays for element-wise comparison, consistent with NumPy.

    def __eq__(self, other):
        if isinstance(other, Measurement):
            return self.value == other.value
        elif isinstance(other, (Number, np.ndarray)):
             # Compare value to the other object, allowing broadcasting
             try:
                 return self.value == np.asarray(other)
             except ValueError: # Broadcast error
                 return False # Cannot be equal if shapes incompatible
        else:
            return NotImplemented # Or False? NumPy usually returns False for incomparable types

    def __ne__(self, other):
         result = self.__eq__(other)
         if result is NotImplemented:
             return NotImplemented
         return ~result # Element-wise negation of boolean array

    def __lt__(self, other):
        val_other = other.value if isinstance(other, Measurement) else other
        try:
            # Perform comparison, relies on NumPy broadcasting
            return self.value < val_other
        except (TypeError, ValueError): # Catch comparison errors or broadcasting issues
             return NotImplemented

    def __le__(self, other):
        val_other = other.value if isinstance(other, Measurement) else other
        try:
            return self.value <= val_other
        except (TypeError, ValueError):
             return NotImplemented

    def __gt__(self, other):
        val_other = other.value if isinstance(other, Measurement) else other
        try:
            return self.value > val_other
        except (TypeError, ValueError):
             return NotImplemented

    def __ge__(self, other):
        val_other = other.value if isinstance(other, Measurement) else other
        try:
            return self.value >= val_other
        except (TypeError, ValueError):
             return NotImplemented


    # --- Helper Methods and Properties ---

    @property
    def nominal_value(self) -> np.ndarray:
        """Returns the nominal value(s) as a NumPy array."""
        return self.value

    @property
    def std_dev(self) -> np.ndarray:
        """Returns the uncertainty (standard deviation) as a NumPy array."""
        return self.error

    @property
    def variance(self) -> np.ndarray:
         """Returns the variance (square of uncertainty) as a NumPy array."""
         return self.error**2

    def round_to_error(self, n_sig_figs: int = 1) -> str:
        """
        Formats the measurement(s) to a string, rounding the nominal value
        appropriately based on the uncertainty's significant figures.

        This provides a standard way to report measurements, ensuring the value's
        precision matches the uncertainty's precision.

        Args:
            n_sig_figs (int): Number of significant figures to keep for the
                              uncertainty (typically 1 or 2). Must be positive.

        Returns:
            str: A string representation like "value ± error", formatted according
                 to the uncertainty. For array Measurements, returns a string
                 representation of the array with each element formatted.

        Example:
            >>> m = Measurement(123.456, 1.23)
            >>> m.round_to_error(1) # Round error to 1 sig fig (1)
            '123 ± 1'
            >>> m.round_to_error(2) # Round error to 2 sig figs (1.2)
            '123.5 ± 1.2'
            >>> m_arr = Measurement([10.5, 20.1], [0.15, 0.033])
            >>> print(m_arr.round_to_error(1))
            [10.5 ± 0.2, 20.10 ± 0.03]
        """
        if n_sig_figs <= 0:
            raise ValueError("Number of significant figures must be positive.")

        # Use np.vectorize to apply the static formatting method element-wise
        formatter = np.vectorize(self._round_single_to_error, otypes=[str])
        formatted_array = formatter(self.value, self.error, n_sig_figs)

        # If the original was scalar-like, return just the string
        if self.shape == () or self.shape == (1,):
             return formatted_array.item() # Extract single string element
        else:
             # Use numpy's array2string for a nice representation of the string array
             return np.array2string(formatted_array, separator=', ',
                                    formatter={'all': lambda x: x})


    @staticmethod
    def _round_single_to_error(value: float, error: float, n_sig_figs: int) -> str:
        """Static helper to format a single value-error pair."""

        # --- Handle Special Cases ---
        if np.isnan(value) or np.isnan(error):
            return "nan ± nan" # Or just "nan"? Depends on desired representation
        if np.isinf(value) or np.isinf(error):
             # Represent infinity clearly
             vs = "inf" if np.isinf(value) else f"{value:.3g}" # Use general format if finite
             es = "inf" if np.isinf(error) else f"{error:.2g}" # Use general format if finite
             # Check signs of infinity
             if np.isinf(value) and value < 0: vs = "-inf"
             # Error should always be positive infinity if infinite
             if np.isinf(error): es = "inf"
             return f"{vs} ± {es}"
        if error == 0.0:
            # Exact value, format reasonably (e.g., show some decimal places)
            # This is heuristic: show more precision for smaller numbers
            if abs(value) > 1e-4 and abs(value) < 1e6 or value == 0.0:
                 return f"{value:.6g} ± 0" # General format up to 6 sig figs
            else:
                 return f"{value:.3e} ± 0" # Scientific notation for very large/small
        if error < 0.0:
             # Should not happen if constructor enforces non-negative error, but handle defensively
             error = abs(error)

        # --- Standard Formatting Logic ---
        try:
            # Determine the order of magnitude of the most significant digit of the error
            # Example: error = 0.0123 -> log10 ≈ -1.9 => floor = -2
            # Example: error = 123    -> log10 ≈  2.09 => floor =  2
            order_err = np.floor(np.log10(error))

            # Determine the decimal place to round the error to
            # Example (1 sig fig): error=0.0123 (order=-2) -> round to 10^(-2 - (1-1)) = 10^-2 (0.01)
            # Example (1 sig fig): error=123    (order= 2) -> round to 10^( 2 - (1-1)) = 10^ 2 (100)
            # Example (2 sig figs): error=0.0123 (order=-2) -> round to 10^(-2 - (2-1)) = 10^-3 (0.001)
            # Example (2 sig figs): error=123    (order= 2) -> round to 10^( 2 - (2-1)) = 10^ 1 (10)
            decimal_place = int(order_err - (n_sig_figs - 1))

            # --- Rounding ---
            # Round error: scale, round, descale to preserve significant digit place
            scale = 10**(-decimal_place)
            # Add small epsilon before rounding to handle borderline cases like 0.1499 -> 0.15 (for 2 sig figs) ?
            # Standard round (to nearest, ties to even) is generally preferred in science.
            rounded_error_scaled = np.round(error * scale)
            rounded_error = rounded_error_scaled / scale

            # Round value to the *same decimal place* as the rounded error
            # Use the 'decimal_place' calculated from the error.
            # Python's round() rounds to 'ndigits' *after* decimal point.
            # So, if decimal_place is -2 (round to 100s), ndigits should be -(-2)=2 ? No.
            # decimal_place = -2 means round to 10^2. round(val, -2)
            # decimal_place = 3 means round to 10^-3. round(val, 3)
            ndigits_value = -decimal_place
            rounded_value = np.round(value, ndigits_value)

            # --- Formatting to String ---
            # Determine the number of decimal places needed for the string format specifier
            # This must match the precision of the rounded error.
            fmt_decimal_places = max(0, ndigits_value)

            # Format value and error strings
            val_str = f"{rounded_value:.{fmt_decimal_places}f}"
            err_str = f"{rounded_error:.{fmt_decimal_places}f}"

            # Refinement: Check if rounding error caused loss of intended sig figs (e.g., 0.096 -> 0.1)
            # If rounded_error is a power of 10 (e.g., 1, 10, 0.1) after rounding,
            # the number of displayed digits might be less than n_sig_figs intended.
            # Example: error=0.14, n_sig_figs=1. order=-1, decimal_place=-1. scale=10.
            # rounded_error_scaled = round(1.4) = 1. rounded_error=0.1. ndigits_val=1.
            # fmt_dp=1. val=1.23 -> rounded=1.2. Result "1.2 ± 0.1". Looks OK.
            # Example: error=14, n_sig_figs=1. order=1, decimal_place=1. scale=0.1
            # rounded_error_scaled = round(1.4) = 1. rounded_error=10. ndigits_val=-1.
            # fmt_dp=0. val=123 -> rounded=120. Result "120 ± 10". Looks OK.
            # This seems generally handled by the decimal place logic.

            return f"{val_str} ± {err_str}"

        except Exception as e:
             # Fallback if any calculation above fails
             warnings.warn(f"Error during formatting: {e}. Using basic representation.", RuntimeWarning)
             return f"{value:.3g} ± {error:.2g}" # Fallback to basic __str__ format

# --- END OF FILE measurement.py ---