# --- START OF FILE measurement.py ---

"""
Measurement Class for Handling Quantities with Uncertainties.

Defines the `Measurement` class, the core of the library for representing
physical quantities with associated uncertainties (errors) and units.
It automatically handles error propagation for standard arithmetic operations
and many NumPy functions.
"""

import numpy as np
import sympy as sp
import warnings
from typing import Union, List, Dict, Tuple, Any, Optional, Callable
from numbers import Number
import math

# Import formatting utilities (crucial for correct output)
try:
    from .utils import _format_value_error_eng, round_to_significant_figures, SI_PREFIXES, get_si_prefix
except ImportError:
    # Fallback stubs if running standalone (less useful without utils)
    warnings.warn("Could not import from .utils. Using basic formatting stubs.", ImportWarning)
    # Provide *very* basic placeholders if utils not found
    def round_to_significant_figures(value, sig_figs): return round(value, sig_figs if sig_figs > 0 else 1)
    def _format_value_error_eng(value, error, unit_symbol, sig_figs_error):
        err_fmt = f".{max(0, sig_figs_error-1)}g" if error != 0 else ".1f"
        val_fmt = ".3g" # Basic value format
        try:
            if error != 0:
                order_err = math.floor(math.log10(abs(round_to_significant_figures(error, sig_figs_error))))
                dp = max(0, -int(order_err - (sig_figs_error - 1)))
                val_fmt = f".{dp}f"
                err_fmt = f".{dp}f"
            return f"{value:{val_fmt}} ± {error:{err_fmt}} {unit_symbol}".strip()
        except: # Catch formatting errors
             return f"{value:.3g} ± {error:.1g} {unit_symbol}".strip()
    SI_PREFIXES = {0: ''}
    def get_si_prefix(exponent): return '', 0


# ---------------------------- Measurement Class -----------------------------

class Measurement:
    """
    Represents a physical quantity with a nominal value and standard uncertainty.

    This class facilitates calculations involving measured quantities by automatically
    propagating uncertainties using first-order Taylor expansion (linear error
    propagation). It supports standard arithmetic operations (+, -, *, /, **) and
    many NumPy universal functions (ufuncs) like np.sin, np.log, np.exp.

    Error Propagation Theory:
        For a function f(x, y, ...) where x, y, ... are independent measurements
        with uncertainties σ_x, σ_y, ..., the uncertainty σ_f in f is approximated by:
            σ_f² ≈ (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)² + ...

        This approximation is generally valid when the uncertainties are "small"
        relative to the values, meaning higher-order terms in the Taylor expansion
        are negligible. The class assumes uncertainties represent one standard deviation (1σ).

        Key Assumptions:
        1.  **Linearity:** The function f is approximately linear over the range
            defined by the uncertainties (e.g., x ± σ_x).
        2.  **Independence:** Uncertainties in different `Measurement` operands are
            assumed to be statistically independent unless specifically handled
            otherwise (e.g., `m - m` correctly yields zero error, but `m1 - m2`
            assumes independence between `m1` and `m2`). Correlations are not
            tracked automatically between separate operations.
        3.  **Gaussian Errors (Implied):** While not strictly required for the propagation
            formula itself, interpreting the uncertainty as a standard deviation is
            most meaningful if the underlying error distribution is approximately Normal.

    Features:
        - Automatic error propagation for +, -, *, /, **, and NumPy ufuncs.
        - Support for scalar and array-based measurements.
        - Basic unit tracking (units propagated correctly for +/- if identical,
          cleared otherwise; unit consistency is user's responsibility for * / **).
        - String formatting methods (`__str__`, `to_eng_string`, `round_to_error`)
          for clear presentation of results, including SI prefixes.
        - Compatibility with NumPy functions (though casting to array loses uncertainty).

    Attributes:
        value (Union[float, np.ndarray]): The nominal value(s) of the quantity.
        error (Union[float, np.ndarray]): The uncertainty (standard deviation, σ)
                                         associated with the value. Must be non-negative
                                         and broadcastable with `value`.
        unit (str): A string representing the physical unit (e.g., "V", "m/s", "kg").
                    Used primarily for display purposes in formatting methods.
        name (str): An optional descriptive name for the quantity (e.g., "Voltage", "Length").
    """
    __array_priority__ = 1000 # Ensure Measurement ops override NumPy ops when mixed

    def __init__(self,
                 values: Union[float, complex, list, tuple, np.ndarray, dict, 'Measurement'],
                 errors: Union[float, list, tuple, np.ndarray, None] = None,
                 magnitude: int = 0,
                 unit: str = "",
                 name: str = ""):
        """
        Initializes a Measurement object.

        Args:
            values: Nominal value(s). Can be:
                - Scalar (float, int, complex).
                - Sequence (list, tuple, NumPy array).
                - Dictionary `{value1: error1, value2: error2, ...}` (errors must be None).
                - Another Measurement object (creates a copy, ignores other args).
            errors: Uncertainty(ies) corresponding to values. Can be:
                - Scalar (float, int). Applied to all values if values is array-like.
                - Sequence (list, tuple, NumPy array). Must be broadcastable with values.
                - None: If `values` is a dict, `errors` MUST be None. If `values` is not
                  a dict, `errors`=None implies zero uncertainty.
            magnitude: A power-of-10 scaling factor applied to both values and errors
                       upon initialization. E.g., `magnitude=-3` with `values=5`
                       creates a measurement of 5e-3. Useful for inputting data
                       with common prefixes (e.g., measured in mV). (Default 0).
            unit: Physical unit symbol string (e.g., "V", "kg", "m/s"). (Default "").
            name: Descriptive name string for the quantity. (Default "").

        Raises:
            ValueError: If `values` is a dict and `errors` is not None.
            ValueError: If `values` and `errors` (when provided) cannot be broadcast
                        to a compatible shape.
            TypeError: If inputs cannot be converted to appropriate numeric types.
        """
        _values_in: Any = values # Keep track of original type for scalar handling
        _errors_in: Any = errors

        # --- Input Parsing ---
        if isinstance(values, Measurement): # Handle initialization from another Measurement
             # Create a copy, ignore other arguments
             self.value = np.copy(values.value)
             self.error = np.copy(values.error)
             # Allow overriding unit/name, otherwise inherit
             self.unit = unit if unit else values.unit
             self.name = name if name else values.name
             if errors is not None or magnitude != 0:
                  warnings.warn("Ignoring 'errors' and 'magnitude' when initializing "
                                "from an existing Measurement object.", UserWarning)
        elif isinstance(values, dict):
            # Initialize from {value: error} dictionary
            if errors is not None:
                raise ValueError("Argument 'errors' must be None when 'values' is a dictionary.")
            if not values: # Handle empty dictionary
                 _vals_arr = np.array([], dtype=float)
                 _errs_arr = np.array([], dtype=float)
            else:
                # Extract keys (values) and values (errors)
                val_list = list(values.keys())
                err_list = list(values.values())
                _vals_arr = np.asarray(val_list, dtype=float) # Force float for consistency
                _errs_arr = np.asarray(err_list, dtype=float)
                if _vals_arr.shape != _errs_arr.shape:
                    # Should not happen if dict structure is correct, but check
                    raise ValueError("Internal error: Dictionary keys and values "
                                     "yielded incompatible shapes after conversion.")
            # Apply magnitude scaling
            scale_factor = 10.0 ** magnitude
            self.value = _vals_arr * scale_factor
            self.error = _errs_arr * scale_factor
            self.unit = unit
            self.name = name
        else:
            # Handle scalar, list, tuple, or ndarray for values
            # Use float as default internal type for consistency in calculations
            _vals_arr = np.atleast_1d(np.asarray(values, dtype=float))

            # Process errors based on 'errors' argument
            if errors is None:
                _errs_arr = np.zeros_like(_vals_arr, dtype=float) # Default to zero error
            else:
                 _errs_arr = np.atleast_1d(np.asarray(errors, dtype=float))

            # Apply magnitude scaling *before* broadcasting check
            scale_factor = 10.0 ** magnitude
            _vals_arr_scaled = _vals_arr * scale_factor
            _errs_arr_scaled = _errs_arr * scale_factor

            # --- Broadcasting and Final Assignment ---
            try:
                 # Check shape compatibility and broadcast arrays if necessary
                 # np.broadcast handles the check and determines the final shape
                 bcast_shape = np.broadcast(_vals_arr_scaled, _errs_arr_scaled).shape

                 # Store internally, ensuring they have the common broadcast shape
                 self.value = np.broadcast_to(_vals_arr_scaled, bcast_shape)
                 self.error = np.broadcast_to(_errs_arr_scaled, bcast_shape)

                 # Optimization: If original input was scalar and resulted in a 0-dim array,
                 # store as Python float internally for potential slight speedup in scalar ops.
                 # Check original types, not just the arrays after atleast_1d.
                 was_scalar_input = isinstance(_values_in, Number) and \
                                    (errors is None or isinstance(_errors_in, Number))

                 if was_scalar_input and self.value.ndim == 0:
                      self.value = self.value.item() # Convert 0-dim array to Python float
                      self.error = self.error.item()

            except ValueError:
                  # Broadcasting failed - shapes are incompatible
                  raise ValueError(f"Shape mismatch: Input values (shape={_vals_arr.shape}) "
                                   f"and errors (shape={_errs_arr.shape}) "
                                   "cannot be broadcast together.")

            # Assign metadata
            self.unit = unit
            self.name = name

        # --- Final Validation ---
        # Ensure errors are non-negative (standard deviation cannot be negative)
        if np.any(np.asarray(self.error) < 0): # Use asarray to handle scalar/array cases
            warnings.warn("Initializing Measurement with negative error(s). "
                          "Uncertainty (standard deviation) must be non-negative. "
                          "Taking the absolute value.", UserWarning)
            self.error = np.abs(self.error)

    # --- NumPy-like Properties ---
    # These allow the Measurement object to mimic some behaviors of NumPy arrays
    @property
    def ndim(self) -> int:
        """Number of array dimensions of the value/error."""
        # Return ndim of the value; error is guaranteed to be broadcastable
        return np.ndim(self.value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of array dimensions of the value/error."""
        return np.shape(self.value)

    @property
    def size(self) -> int:
        """Total number of elements in the value/error array."""
        return np.size(self.value)

    def __len__(self) -> int:
        """Length of the first dimension (if array is not 0-dimensional)."""
        s = self.shape
        if not s: # Empty tuple for 0-dim array
             # Mimics NumPy behavior for 0-d arrays
             raise TypeError("len() of unsized object (0-dimensional Measurement)")
        return s[0] # Return length of first dimension

    def __getitem__(self, key: Any) -> 'Measurement':
        """
        Allows indexing and slicing like a NumPy array.

        Returns a *new* Measurement object representing the selected subset.
        The name of the new object is appended with the index/slice key.
        Units are preserved.
        """
        if self.ndim == 0:
             raise IndexError("Cannot index a 0-dimensional Measurement object.")

        try:
            new_value = np.asarray(self.value)[key]
            # Error needs careful handling: if error was scalar, it applies to all elements.
            # If error was array, slice it the same way as value.
            if np.ndim(self.error) == 0:
                # If error is scalar, the sliced element still has the same scalar error
                new_error = self.error
            else:
                # If error is an array, slice it correspondingly
                new_error = np.asarray(self.error)[key]

            # Determine if the result of indexing is a scalar or an array
            is_result_scalar = (np.ndim(new_value) == 0)

            # Create a descriptive name for the sliced Measurement
            try:
                 # Attempt a clean string representation of the key
                 key_str = str(key)
                 # Basic sanitization for common problematic chars in names
                 key_str = key_str.replace(':', '-').replace(' ', '').replace(',', '_')
            except:
                 key_str = "slice" # Fallback
            new_name = f"{self.name}[{key_str}]" if self.name else f"Measurement[{key_str}]"

            # Return a new Measurement object
            # Important: pass scalar values if result is scalar, else pass arrays
            if is_result_scalar:
                 return Measurement(new_value.item(), new_error.item() if np.ndim(new_error) == 0 else new_error,
                                    unit=self.unit, name=new_name)
            else:
                 return Measurement(new_value, new_error, unit=self.unit, name=new_name)

        except IndexError as e:
            raise IndexError(f"Error indexing Measurement: {e}")
        except Exception as e: # Catch other potential slicing errors
             raise TypeError(f"Invalid index or slice key for Measurement: {key}. Error: {e}")


    # --- Convenience Properties ---
    @property
    def nominal_value(self) -> Union[float, np.ndarray]:
        """Alias for `self.value` (the nominal value(s))."""
        return self.value

    @property
    def std_dev(self) -> Union[float, np.ndarray]:
        """Alias for `self.error` (the uncertainty/standard deviation)."""
        return self.error

    @property
    def variance(self) -> Union[float, np.ndarray]:
         """Returns the variance (square of the uncertainty `self.error`)."""
         return np.square(self.error)

    # --- String Representations ---

    def __repr__(self) -> str:
        """
        Provides a detailed, unambiguous string representation of the object,
        useful for debugging. Shows value, error, name, and unit.
        """
        name_str = f", name='{self.name}'" if self.name else ""
        unit_str = f", unit='{self.unit}'" if self.unit else ""
        # Use np.repr for array representation if needed
        val_repr = np.array_repr(self.value) if isinstance(self.value, np.ndarray) else repr(self.value)
        err_repr = np.array_repr(self.error) if isinstance(self.error, np.ndarray) else repr(self.error)
        return f"Measurement(value={val_repr}, error={err_repr}{name_str}{unit_str})"

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation, suitable for printing.

        Defaults to using engineering notation (`to_eng_string`) with 1 significant
        figure for the error if a unit is present. If no unit is present, it uses
        standard rounding based on error (`round_to_error`) with 1 significant figure.
        Handles both scalar and array Measurements.
        """
        # Check if the Measurement holds scalar or array data
        is_scalar = (self.ndim == 0)

        # Choose formatting based on whether a unit is specified
        if self.unit:
            # Use engineering notation if unit exists
            formatter_func = lambda v, e, u, sf: _format_value_error_eng(v, e, u, sf)
            sig_figs = 1 # Default sig figs for __str__
        else:
            # Use standard rounding if no unit
            formatter_func = lambda v, e, u, sf: self._round_single_to_error_std(v, e, sf)
            sig_figs = 1 # Default sig figs for __str__

        # Vectorize the chosen formatter to handle arrays
        # We exclude unit and sig_figs as they are constant for the call
        vectorized_formatter = np.vectorize(formatter_func, excluded=['u', 'sf'], otypes=[str])

        # Apply formatting
        formatted_array = vectorized_formatter(v=self.value, e=self.error, u=self.unit, sf=sig_figs)

        # Return scalar string or formatted array string
        if is_scalar:
            return formatted_array.item() # Extract scalar string from 0-dim array
        else:
             # Use numpy's array2string for clean array output, prevent truncation
             with np.printoptions(threshold=np.inf):
                 # Use 'all' formatter to apply the identity function (string is already formatted)
                 return np.array2string(formatted_array, separator=', ',
                                        formatter={'all': lambda x: x})

    def to_eng_string(self, sig_figs_error: int = 1, **kwargs) -> str:
        """
        Formats the measurement(s) using engineering notation with SI prefixes.

        This method provides a standardized way to display measurements,
        especially common in physics and engineering, where values span many
        orders of magnitude. The error is rounded to the specified number of
        significant figures, and the value is rounded to the corresponding
        decimal place. An appropriate SI prefix (like k, m, µ, n) is chosen
        based on the value's magnitude.

        Args:
            sig_figs_error: The number of significant figures to use for displaying
                            the error part. Typically 1 or 2. (Default 1).
            **kwargs: Internal use for vectorization. Do not provide.

        Returns:
            str: A formatted string (for scalar) or a string representation of the
                 formatted array (for array Measurement). Example: "1.23 ± 0.05 mV".

        Raises:
            ValueError: If `sig_figs_error` is not positive.
        """
        if sig_figs_error <= 0:
            raise ValueError("Number of significant figures for error must be positive.")

        # Extract value, error, unit - handle internal call from vectorization
        value = kwargs.get('value', self.value)
        error = kwargs.get('error', self.error)
        unit = kwargs.get('unit', self.unit)
        is_scalar = (np.ndim(value) == 0)

        # Core formatting logic is in the utility function
        formatter = np.vectorize(_format_value_error_eng, excluded=['unit_symbol', 'sig_figs_error'], otypes=[str])
        formatted_array = formatter(value=value, error=error, unit_symbol=unit, sig_figs_error=sig_figs_error)

        if is_scalar:
            return formatted_array.item()
        else:
             # Format array output cleanly
             with np.printoptions(threshold=np.inf):
                 return np.array2string(formatted_array, separator=', ',
                                        formatter={'all': lambda x: x})

    def round_to_error(self, n_sig_figs: int = 1) -> str:
        """
        Formats the measurement(s) by rounding the value based on error significance.

        This method uses standard decimal rounding without SI prefixes.
        The error is rounded to `n_sig_figs` significant figures, and the value
        is then rounded to the same decimal place as the least significant digit
        of the rounded error. This is a common convention for displaying results
        where engineering notation is not required or desired.

        Args:
            n_sig_figs: Number of significant figures for the uncertainty (error).
                        Typically 1 or 2. (Default 1).

        Returns:
            str: Formatted string (for scalar) or array representation (for array).
                 Example: "123.4 ± 1.2" or "0.0056 ± 0.0003".

        Raises:
            ValueError: If `n_sig_figs` is not positive.
        """
        if n_sig_figs <= 0:
            raise ValueError("Number of significant figures must be positive.")

        is_scalar = (self.ndim == 0)

        # Vectorize the static helper method for standard rounding
        formatter = np.vectorize(self._round_single_to_error_std, excluded=['n_sig_figs'], otypes=[str])
        formatted_array = formatter(value=self.value, error=self.error, n_sig_figs=n_sig_figs)

        if is_scalar:
             return formatted_array.item()
        else:
             with np.printoptions(threshold=np.inf):
                 return np.array2string(formatted_array, separator=', ',
                                        formatter={'all': lambda x: x})

    @staticmethod
    def _round_single_to_error_std(value: float, error: float, n_sig_figs: int) -> str:
        """
        Static helper method for formatting a single value-error pair using
        standard rounding based on the error's significant figures.
        (Internal use, called by `round_to_error`).
        """
        # Handle non-finite numbers gracefully
        if np.isnan(value) or np.isnan(error): return "nan ± nan"
        if np.isinf(value) or np.isinf(error): return f"{value} ± {error}" # Basic representation
        if error < 0: error = abs(error) # Ensure error is non-negative

        # --- Case 1: Error is effectively zero ---
        if np.isclose(error, 0.0):
            # Format value reasonably when error is zero. Avoid excessive digits.
            # Use scientific notation for very large/small numbers.
            if abs(value) > 1e-4 and abs(value) < 1e6 or np.isclose(value, 0.0):
                 # Heuristic: ~6 significant digits for typical range
                 return f"{value:.6g} ± 0"
            else:
                 # Scientific notation for large/small magnitudes
                 return f"{value:.3e} ± 0"

        # --- Case 2: Error is non-zero ---
        try:
            # 1. Round the error to the specified significant figures
            rounded_error = round_to_significant_figures(error, n_sig_figs)
            if np.isclose(rounded_error, 0.0):
                # If error rounds to zero (e.g., 0.0001 rounded to 1 sf), treat as zero case.
                decimal_place = 0 # Default, but should fall back to zero error formatting ideally
                # Re-call formatting for zero error case might be cleaner, but simple approach:
                return Measurement._round_single_to_error_std(value, 0.0, n_sig_figs)

            # 2. Determine the decimal place of the least significant digit of the *rounded error*
            # This determines how many decimal places the value should be rounded to.
            # Example: rounded_error = 123 (3 sf) -> last digit is units place (dp=0)
            # Example: rounded_error = 120 (2 sf) -> last digit is tens place (dp=-1) -> round value to nearest 10
            # Example: rounded_error = 0.12 (2 sf) -> last digit is hundredths place (dp=2) -> round value to 0.01
            # Example: rounded_error = 0.012 (2 sf) -> last digit is thousandths place (dp=3) -> round value to 0.001
            order_rounded_err = math.floor(math.log10(abs(rounded_error)))
            # The decimal place relative to point '.' is -(order - (sig_figs - 1))
            decimal_place = max(0, -int(order_rounded_err - (n_sig_figs - 1)))

            # 3. Round the value to this decimal place
            scale = 10.0**decimal_place
            rounded_value = round(value * scale) / scale

            # 4. Format both rounded value and rounded error to this decimal place
            # The f-string format specifier ensures trailing zeros are kept if needed
            val_str = f"{rounded_value:.{decimal_place}f}"
            err_str = f"{rounded_error:.{decimal_place}f}"
            return f"{val_str} ± {err_str}"

        except OverflowError:
             warnings.warn(f"Overflow encountered during standard formatting for value={value}, error={error}. "
                           "Result may be inaccurate.", RuntimeWarning)
             return f"{value:.3g} ± {error:.2g}" # Fallback to general format
        except Exception as e:
             # Catch other potential errors (e.g., log10(0)) although zero error handled above
             warnings.warn(f"Error during standard formatting for value={value}, error={error}: {e}. "
                           "Using basic representation.", RuntimeWarning)
             return f"{value:.3g} ± {error:.2g}" # Fallback


    # --- Metadata Propagation Helper ---
    def _propagate_metadata(self, other: Any, operation: str) -> Tuple[str, str]:
        """
        Internal helper to determine the name and unit for the result of an operation.
        Provides basic logic for simple cases (e.g., addition, negation).
        More complex unit tracking (e.g., V * A -> W) is not implemented.
        """
        res_name = "" # Name is generally lost unless operation is simple unary
        res_unit = "" # Default to dimensionless/unitless unless specific rules apply

        # Extract metadata from the 'other' operand if it's a Measurement
        is_other_meas = isinstance(other, Measurement)
        other_unit = other.unit if is_other_meas else None
        other_name = other.name if is_other_meas else None

        # --- Unit Logic ---
        if operation in ['add', 'sub']:
            # Addition/Subtraction: Require identical units for the result to have that unit.
            if self.unit and other_unit and self.unit == other_unit:
                res_unit = self.unit # Units match, preserve it
            elif self.unit and other_unit is None and isinstance(other, (Number, np.ndarray)):
                 res_unit = self.unit # Adding/subtracting a dimensionless constant preserves unit
            elif not self.unit and other_unit:
                 res_unit = other_unit # Adding/subtracting to a dimensionless constant preserves unit
            elif self.unit and other_unit and self.unit != other_unit:
                 # Units are present but different - result unit is ambiguous/cleared
                 warnings.warn(f"Operation '{operation}' on Measurements with incompatible units: "
                               f"'{self.unit}' and '{other_unit}'. Resulting unit is cleared.", UserWarning)
                 res_unit = ""
            # else: both units empty or one is empty and other is const -> unit remains empty

        elif operation in ['neg', 'pos', 'abs']:
             # Unary operations that don't change dimensionality preserve the unit
             res_unit = self.unit

        # Operations like mul, div, pow generally result in different units.
        # Automatic unit parsing (e.g., 'm' / 's' -> 'm/s') is complex and not implemented.
        # User should manage units manually for these operations if needed.
        # By default, res_unit remains "".

        # --- Name Logic (Basic) ---
        # Try to generate a simple name indicating the operation.
        op_symbols = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
        op_sym = op_symbols.get(operation, operation) # Use symbol or operation name

        # Represent operands in the name
        self_name_part = f"({self.name})" if self.name else 'm1'
        if is_other_meas:
            other_name_part = f"({other_name})" if other_name else 'm2'
        elif isinstance(other,(Number, np.ndarray)):
            other_name_part = 'const' # Indicate a constant value
        else:
            other_name_part = 'other'

        if operation in ['neg', 'pos', 'abs']:
             # Unary operations
             res_name = f"{op_sym}{self_name_part}"
        elif is_other_meas or isinstance(other, (Number, np.ndarray)):
             # Binary operations
             # Special case for r-operations (e.g., const + m1)
             if operation.startswith('r') and len(operation) > 1:
                 # Reverse the order for the name e.g., radd -> const + m1
                 res_name = f"{other_name_part} {op_sym} {self_name_part}"
             else:
                 res_name = f"{self_name_part} {op_sym} {other_name_part}"
        # else: Operation doesn't fit simple patterns, name remains empty

        return res_name, res_unit

    # --- Nan/Inf Check Helper ---
    def _check_nan_inf(self, value: np.ndarray, error: np.ndarray, operation_name: str):
        """
        Internal helper to issue warnings if NaN or Infinity values appear
        in the result value or error after an operation.
        """
        # Check value array for NaN/Inf
        value_arr = np.asarray(value) # Ensure array for np.any
        if np.any(np.isnan(value_arr)):
             warnings.warn(f"NaN value resulted from operation '{operation_name}'.", RuntimeWarning)
        if np.any(np.isinf(value_arr)):
             warnings.warn(f"Infinity value resulted from operation '{operation_name}'.", RuntimeWarning)

        # Check error array - more critical if error is NaN/Inf when value is NOT
        error_arr = np.asarray(error)
        # Identify problematic errors: NaN/Inf error where value is finite
        error_is_problematic = (np.isnan(error_arr) | np.isinf(error_arr)) & \
                               (~np.isnan(value_arr) & ~np.isinf(value_arr))
        if np.any(error_is_problematic):
             warnings.warn(f"NaN or Infinity error resulted from operation '{operation_name}' "
                           "where the corresponding value is finite. This often indicates issues like "
                           "division by zero or invalid function domains during error propagation.",
                           RuntimeWarning)


    # --- Arithmetic Operations ---
    # Each operation calculates the new value and error based on propagation rules,
    # propagates metadata (unit, name), checks for NaN/Inf, and returns a new Measurement.

    def __add__(self, other: Any) -> 'Measurement':
        op = "add"
        res_name, res_unit = self._propagate_metadata(other, op)
        if isinstance(other, Measurement):
            # (x ± σx) + (y ± σy) = (x+y) ± sqrt(σx² + σy²)
            new_value = self.value + other.value
            new_error = np.hypot(self.error, other.error) # sqrt(sx^2 + sy^2)
        elif isinstance(other, (Number, np.ndarray)):
            # (x ± σx) + k = (x+k) ± σx (error unchanged)
            other_arr = np.asarray(other)
            new_value = self.value + other_arr
            new_error = self.error # Additive constant doesn't change error
        else:
            return NotImplemented # Indicate operation is not supported for this type
        # Create result and check for issues
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __radd__(self, other: Any) -> 'Measurement':
        # k + (x ± σx) = (k+x) ± σx. Same logic as __add__.
        # We can just call __add__ because addition is commutative.
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Measurement':
        op = "sub"
        res_name, res_unit = self._propagate_metadata(other, op)
        if isinstance(other, Measurement):
            # (x ± σx) - (y ± σy) = (x-y) ± sqrt(σx² + σy²) (errors add in quadrature)
            new_value = self.value - other.value
            new_error = np.hypot(self.error, other.error)
        elif isinstance(other, (Number, np.ndarray)):
            # (x ± σx) - k = (x-k) ± σx (error unchanged)
            other_arr = np.asarray(other)
            new_value = self.value - other_arr
            new_error = self.error # Subtractive constant doesn't change error
        else:
            return NotImplemented
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rsub__(self, other: Any) -> 'Measurement':
        # k - (x ± σx) = (k-x) ± σx
        op = "rsub" # Use specific name for metadata if needed, though propagate handles it
        # Correct metadata propagation for k - M
        res_name, res_unit = self._propagate_metadata(other, op) # Will generate 'const - (m1)' etc.

        if isinstance(other, (Number, np.ndarray)):
            other_arr = np.asarray(other)
            new_value = other_arr - self.value
            new_error = self.error # Error magnitude is the same as for -(x ± σx)
        else:
             return NotImplemented
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __mul__(self, other: Any) -> 'Measurement':
        op = "mul"
        res_name, res_unit = self._propagate_metadata(other, op) # Unit will be cleared by default
        if isinstance(other, Measurement):
            # (x ± σx) * (y ± σy) ≈ (x*y) ± sqrt( (y*σx)² + (x*σy)² )
            a, sa = self.value, self.error
            b, sb = other.value, other.error
            new_value = a * b
            # Use np.errstate to suppress potential warnings during intermediate calculation
            # (e.g., inf*0 results in nan, which is handled below)
            with np.errstate(invalid='ignore'):
                 term1 = b * sa
                 term2 = a * sb
                 # Handle cases where a or b is exactly zero: the corresponding error term is zero.
                 term1 = np.where(np.isclose(b, 0.0), 0.0, term1)
                 term2 = np.where(np.isclose(a, 0.0), 0.0, term2)
                 # Combine terms in quadrature
                 new_error = np.hypot(term1, term2)
        elif isinstance(other, (Number, np.ndarray)):
            # k * (x ± σx) = (k*x) ± |k|*σx
            k = np.asarray(other)
            new_value = self.value * k
            # Error scales by the absolute value of the constant factor
            new_error = np.abs(k) * self.error
        else:
            return NotImplemented
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rmul__(self, other: Any) -> 'Measurement':
        # k * M is the same as M * k. Multiplication is commutative.
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Measurement':
        op = "div"
        res_name, res_unit = self._propagate_metadata(other, op) # Unit cleared by default
        # Use errstate to manage division by zero warnings during calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            if isinstance(other, Measurement):
                # (x ± σx) / (y ± σy) ≈ (x/y) ± sqrt( (σx/y)² + (x*σy / y²)² )
                #                        = (x/y) ± sqrt( (σx/y)² + ((x/y)*σy / y)² )
                a, sa = self.value, self.error
                b, sb = other.value, other.error
                new_value = a / b # Let numpy handle 0/0 -> nan, x/0 -> inf

                # Calculate error terms, carefully handling potential zeros
                # Term 1: (∂f/∂a * sa) = (1/b * sa)
                term1 = sa / b # Might be inf if b=0
                # Term 2: (∂f/∂b * sb) = (-a/b² * sb) = -(a/b)*(sb/b) = -new_value*(sb/b)
                term2 = new_value * (sb / b) # Might involve inf/inf=nan, 0/0=nan etc.

                # Refine error terms where inputs are zero
                term1 = np.where(np.isclose(b, 0.0), np.inf if not np.isclose(sa, 0.0) else 0.0, term1)
                term2 = np.where(np.isclose(b, 0.0), np.inf if not (np.isclose(a, 0.0) or np.isclose(sb, 0.0)) else 0.0, term2)
                # If a is zero, only the sa/b term contributes (unless b is also zero)
                term2 = np.where(np.isclose(a, 0.0) & ~np.isclose(b, 0.0), 0.0, term2)

                # Combine in quadrature
                new_error = np.hypot(term1, term2)

            elif isinstance(other, (Number, np.ndarray)):
                # (x ± σx) / k = (x/k) ± (|1/k|) * σx = (x/k) ± σx / |k|
                k = np.asarray(other)
                new_value = self.value / k
                # Error scales by absolute value of 1/k
                new_error = np.abs(self.error / k) # Will be inf if k=0

            else:
                return NotImplemented

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rtruediv__(self, other: Any) -> 'Measurement':
        # k / (x ± σx) ≈ (k/x) ± | -k*σx / x² | = (k/x) ± | (k/x) * σx / x |
        op = "rtruediv"
        res_name, res_unit = self._propagate_metadata(other, op) # Unit cleared by default
        if isinstance(other, (Number, np.ndarray)):
            with np.errstate(divide='ignore', invalid='ignore'):
                k = np.asarray(other)
                a, sa = self.value, self.error
                new_value = k / a # Numpy handles k/0 -> inf

                # Calculate error: | new_value * sa / a |
                new_error = np.abs(new_value * sa / a)
                # Handle cases involving zero
                new_error = np.where(np.isclose(a, 0.0), np.inf if not np.isclose(k*sa, 0.0) else 0.0, new_error)
                # If k=0, result is 0 with 0 error (unless a=0, which gives nan handled above)
                new_error = np.where(np.isclose(k, 0.0) & ~np.isclose(a, 0.0), 0.0, new_error)
        else:
            return NotImplemented
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __pow__(self, other: Any) -> 'Measurement':
        op = "pow"
        res_name, res_unit = self._propagate_metadata(other, op) # Unit cleared by default
        with np.errstate(invalid='ignore'): # Manage 0**neg, neg**frac warnings
            if isinstance(other, (Number, np.ndarray)): # Case 1: M ** k
                n = np.asarray(other)
                a, sa = self.value, self.error
                # Check for domain issues that yield complex numbers or NaN/Inf
                # Negative base with fractional exponent -> complex
                if np.any((a < 0) & (n % 1 != 0)):
                     warnings.warn(f"Power M**k: Base Measurement has negative values with non-integer exponent {n}. "
                                   "Result may be complex or NaN. Error propagation assumes real result.", RuntimeWarning)
                # 0 ** negative -> inf
                if np.any((np.isclose(a, 0.0)) & (n < 0)):
                     warnings.warn(f"Power M**k: Operation 0**negative encountered.", RuntimeWarning)

                # Calculate value: a^n
                new_value = a ** n

                # Calculate error: σ_f ≈ | ∂f/∂a * σ_a | = | n * a^(n-1) * σ_a |
                deriv = n * (a ** (n - 1))
                new_error = np.abs(deriv * sa)

                # Refine error for special cases where derivative might be NaN/Inf or rule simple
                # If deriv is NaN (e.g., from 0**-1), error should be NaN or Inf
                new_error = np.where(np.isnan(deriv) | np.isinf(deriv), deriv, new_error)
                # If exponent n=0, result is 1 exactly, error is 0
                new_error = np.where(np.isclose(n, 0.0), 0.0, new_error)
                # If base a=0 and n>0, result is 0 exactly, error is 0
                new_error = np.where(np.isclose(a, 0.0) & (n > 0), 0.0, new_error)
                # If base a=0 and n=1, result 0, error is |1*0^0*sa| - handle 0^0 as 1? No, use limit -> error is sa
                new_error = np.where(np.isclose(a, 0.0) & np.isclose(n, 1.0), sa, new_error)


            elif isinstance(other, Measurement): # Case 2: M1 ** M2
                # f(a, b) = a^b
                # ∂f/∂a = b * a^(b-1)
                # ∂f/∂b = a^b * ln(a) = f * ln(a)
                # σ_f² ≈ ( (b * a^(b-1)) * σ_a )² + ( (a^b * ln(a)) * σ_b )²
                a, sa = self.value, self.error
                b, sb = other.value, other.error

                # Domain warnings for M1**M2 are more complex
                if np.any(a < 0):
                     warnings.warn(f"Power M1**M2: Base M1 has negative values ({a}). "
                                   "Result or error propagation may involve complex numbers or NaN.", RuntimeWarning)
                if np.any((np.isclose(a, 0.0)) & (b <= 0)):
                     warnings.warn(f"Power M1**M2: Operation 0 ** (M2 <= 0) encountered.", RuntimeWarning)
                # ln(a) is needed for error propagation, requires a > 0
                if np.any(a <= 0):
                     warnings.warn(f"Power M1**M2: Logarithm of non-positive base M1 ({a}) required for error propagation. "
                                   "Error contribution from M2 uncertainty (σ_b) may be NaN.", RuntimeWarning)

                # Calculate value
                new_value = a ** b

                # Calculate partial derivatives and error terms
                df_da = b * (a ** (b - 1))
                # Use np.log for potential array input, handle log(<=0) -> nan/inf
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warning
                     log_a = np.log(a)
                df_db = new_value * log_a # This will be nan if a <= 0

                term_a = df_da * sa
                term_b = df_db * sb

                # Handle NaNs arising from derivatives or inputs
                # If derivative is NaN, term is NaN (unless error is 0)
                term_a = np.where(np.isnan(df_da) & ~np.isclose(sa, 0.0), np.nan, term_a)
                term_b = np.where(np.isnan(df_db) & ~np.isclose(sb, 0.0), np.nan, term_b)
                # If error is zero, term is zero
                term_a = np.where(np.isclose(sa, 0.0), 0.0, term_a)
                term_b = np.where(np.isclose(sb, 0.0), 0.0, term_b)

                # Combine in quadrature
                new_error = np.hypot(term_a, term_b)
            else:
                return NotImplemented

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rpow__(self, other: Any) -> 'Measurement':
        # k ** M
        op = "rpow"
        res_name, res_unit = self._propagate_metadata(other, op) # Unit cleared
        if isinstance(other, (Number, np.ndarray)):
            with np.errstate(invalid='ignore', divide='ignore'):
                k = np.asarray(other) # Base
                a, sa = self.value, self.error # Exponent M = a ± sa
                # f(a) = k^a => ∂f/∂a = k^a * ln(k)
                # σ_f ≈ | (k^a * ln(k)) * σ_a |

                 # Domain warnings
                if np.any(k < 0):
                     warnings.warn(f"Power k**M: Base k={k} is negative. Result may be complex or NaN.", RuntimeWarning)
                if np.any((np.isclose(k, 0.0)) & (a <= 0)):
                     warnings.warn(f"Power k**M: Operation 0 ** (M <= 0) encountered.", RuntimeWarning)
                # ln(k) requires k > 0 for real result
                if np.any(k <= 0):
                     warnings.warn(f"Power k**M: Logarithm of non-positive base k ({k}) required for error propagation. "
                                   "Error may be NaN.", RuntimeWarning)

                # Calculate value
                new_value = k ** a

                # Calculate derivative and error
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_k = np.log(k) # Will be nan/inf for k<=0
                deriv = new_value * log_k
                new_error = np.abs(deriv * sa)

                # Refine error for special cases
                # If ln(k) was NaN/Inf, error is NaN/Inf (unless sa=0)
                new_error = np.where((np.isnan(log_k) | np.isinf(log_k)) & ~np.isclose(sa, 0.0), np.nan, new_error)
                new_error = np.where(np.isclose(sa, 0.0), 0.0, new_error) # If exponent error is zero, result error is zero
                # If base k=1, result is 1 exactly, error is 0
                new_error = np.where(np.isclose(k, 1.0), 0.0, new_error)
                # If base k=0 and exponent a > 0, result is 0 exactly, error is 0
                new_error = np.where(np.isclose(k, 0.0) & (a > 0), 0.0, new_error)
        else:
            return NotImplemented
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result


    # --- Unary Operations ---
    def __neg__(self) -> 'Measurement':
        # -(x ± σx) = (-x) ± σx
        op = "neg"
        res_name, res_unit = self._propagate_metadata(None, op) # Preserves unit
        # Negating the value does not change the magnitude of the uncertainty
        return Measurement(-self.value, self.error, unit=res_unit, name=res_name)

    def __pos__(self) -> 'Measurement':
        # +(x ± σx) = x ± σx
        op = "pos"
        res_name, res_unit = self._propagate_metadata(None, op) # Preserves unit
        # The positive operator should ideally return an immutable copy
        # Creating a new Measurement ensures this.
        return Measurement(self.value, self.error, unit=res_unit, name=res_name)

    def __abs__(self) -> 'Measurement':
        # abs(x ± σx) ≈ abs(x) ± σx (approximation, potentially inaccurate near zero)
        op = "abs"
        res_name, res_unit = self._propagate_metadata(None, op) # Preserves unit

        # Check if the interval [x - n*σx, x + n*σx] (e.g., n=3 for ~3 sigma) includes zero.
        # If it does, the simple error propagation σ_abs ≈ σx might be poor.
        # The distribution of abs(X) is non-Gaussian near zero.
        # A more sophisticated treatment might be needed for high-accuracy cases near zero.
        if np.any(np.abs(self.value) < 3 * np.asarray(self.error)): # Check if |value| < 3*error
             warnings.warn("Taking absolute value of a Measurement whose uncertainty interval "
                           "likely includes zero (|value| < 3*error). Standard error propagation "
                           "(σ_abs ≈ σ_orig) is used but might be inaccurate in this region.", UserWarning)

        # Apply abs to value, keep error magnitude the same (first-order approx)
        new_value = np.abs(self.value)
        new_error = self.error # Error magnitude approx unchanged

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op) # Check for NaNs (can happen with complex input)
        return result


    # --- NumPy Integration ---

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Allows direct use of Measurement objects in NumPy functions that expect arrays
        (e.g., `np.asarray(m)`, `np.array([m1, m2])`).

        Warning: This conversion *discards* the uncertainty, unit, and name, returning
        only the nominal value(s) as a NumPy array. Use with caution. For calculations
        preserving uncertainty, use the overloaded arithmetic operators or rely on
        `__array_ufunc__` where implemented.

        Args:
            dtype: Optional desired dtype for the resulting NumPy array.

        Returns:
            np.ndarray: The nominal value(s) as a NumPy array.
        """
        warnings.warn("Casting Measurement to np.ndarray using np.asarray() or np.array() "
                      "discards uncertainty, unit, and name information. Returning nominal values only. "
                      "For uncertainty propagation, use Measurement arithmetic or compatible ufuncs.",
                      UserWarning)
        return np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """
        Handles NumPy universal functions (ufuncs) applied to Measurement objects.

        This method intercepts calls like `np.sin(m)`, `np.add(m1, m2)`, etc.
        It calculates the result's nominal value by applying the ufunc to the
        input values and propagates the uncertainties using first-order Taylor
        expansion (via `sympy` for differentiation).

        Current Implementation Notes:
        - Supports ufuncs with 1 or 2 inputs where at least one is a Measurement.
        - Uses `sympy.diff` to calculate partial derivatives for error propagation.
          This provides generality but might be slower than hardcoded rules.
        - Assumes independence of errors between different Measurement inputs.
        - Basic unit propagation for simple cases (neg, abs, add/sub with same units).
          Most ufuncs result in a dimensionless Measurement (unit="").
        - Does *not* support ufuncs with more than 2 inputs or methods other
          than `__call__` (e.g., `ufunc.at`, `ufunc.reduce`).
        - The 'out' keyword argument is not supported.

        Args:
            ufunc (np.ufunc): The ufunc being called (e.g., `np.add`, `np.sin`).
            method (str): The ufunc method called (usually '__call__').
            *inputs (Any): The inputs to the ufunc. Can be Measurement objects,
                           NumPy arrays, scalars, etc.
            **kwargs (Any): Keyword arguments passed to the ufunc (e.g., `where`).
                            'out' is explicitly disallowed.

        Returns:
            Measurement or NotImplemented: A new Measurement object with the calculated
            value and propagated uncertainty, or NotImplemented if the ufunc/method
            is not supported.
        """
        # --- Basic Checks and Input Preparation ---
        if method != '__call__':
            # Only handle standard ufunc calls, not methods like .reduce, .accumulate, etc.
            return NotImplemented

        if 'out' in kwargs:
            # In-place operations with 'out' are complex to handle correctly with
            # error propagation. Disallow them.
            warnings.warn("The 'out' keyword argument is not supported for ufuncs involving "
                          "Measurement objects. A new Measurement object will always be returned.",
                          UserWarning)
            # We could try to remove 'out' and proceed, but safer to return NotImplemented
            # if the user explicitly tried to use it. Alternatively, modify kwargs:
            # kwargs = {k: v for k, v in kwargs.items() if k != 'out'} # Proceed without 'out'
            return NotImplemented # Be strict: if user specified 'out', don't handle it.


        # Convert inputs for processing: extract value, error, metadata
        input_values = []
        input_errors = []
        input_units = []
        input_names = []
        input_is_measurement = [] # Track which inputs are Measurements

        for x in inputs:
            if isinstance(x, Measurement):
                input_values.append(x.value)
                input_errors.append(x.error)
                input_units.append(x.unit)
                input_names.append(x.name)
                input_is_measurement.append(True)
            elif isinstance(x, (Number, np.ndarray)):
                # Treat NumPy arrays or scalars as having zero error
                val = np.asarray(x)
                input_values.append(val)
                input_errors.append(np.zeros_like(val, dtype=float)) # Zero error
                input_units.append("") # No unit
                input_names.append("") # No name
                input_is_measurement.append(False)
            else:
                # If any input is not a Measurement, number, or array, cannot handle it
                return NotImplemented

        # If no Measurement objects were involved, this method shouldn't have been called
        # due to __array_priority__, but as a safeguard:
        if not any(input_is_measurement):
             return NotImplemented

        # --- Value Calculation ---
        # Calculate the nominal value of the result using the ufunc
        try:
            # Use errstate to avoid warnings from numpy during intermediate calculations
            # (e.g., log(0), sqrt(-1)) - these should result in nan/inf appropriately.
            with np.errstate(all='ignore'):
                 result_value = ufunc(*input_values, **kwargs)
        except Exception as e:
            # If the ufunc itself fails on the values (e.g., type mismatch not caught earlier)
            warnings.warn(f"Error during ufunc '{ufunc.__name__}' value calculation: {e}. "
                          "Result value calculation failed.", RuntimeWarning)
            # Attempt to determine expected output shape for returning NaNs
            try:
                 broadcast_shape = np.broadcast(*input_values).shape
                 result_value = np.full(broadcast_shape, np.nan)
            except ValueError:
                 # If even broadcasting fails, cannot proceed
                 return NotImplemented


        # --- Metadata Propagation (Result Unit and Name) ---
        # Default to dimensionless, generate generic name
        res_unit = ""
        operand_names = [name if name else f"op{i+1}" for i, name in enumerate(input_names)]
        res_name = f"{ufunc.__name__}({', '.join(operand_names)})"

        # Simple unit rules based on ufunc type
        if ufunc.nin == 1 and input_is_measurement[0]:
             # Unary functions
             if ufunc in [np.negative, np.positive, np.conjugate, np.absolute]:
                 res_unit = input_units[0] # Preserve unit
             # Most others (trig, exp, log, sqrt, square) likely change or clear unit -> default "" is okay
             # np.sqrt('m^2') -> 'm', np.square('m') -> 'm^2' - requires parsing, not done.

        elif ufunc.nin == 2:
             # Binary functions
             unit1 = input_units[0]
             unit2 = input_units[1]
             if ufunc in [np.add, np.subtract, np.hypot]:
                 # Addition-like: require compatible units
                 if unit1 and unit1 == unit2:
                      res_unit = unit1
                 elif unit1 and not unit2 and not input_is_measurement[1]: # M + const
                      res_unit = unit1
                 elif unit2 and not unit1 and not input_is_measurement[0]: # const + M
                      res_unit = unit2
                 elif unit1 and unit2 and unit1 != unit2:
                      warnings.warn(f"Ufunc '{ufunc.__name__}' on Measurements with incompatible units: "
                                    f"'{unit1}' and '{unit2}'. Result unit cleared.", UserWarning)
                      res_unit = "" # Clear unit if incompatible and both were Measurements
             # Most others (multiply, divide, power, etc.) -> unit logic complex, default ""

        # --- Error Propagation using SymPy ---
        result_error: Union[float, np.ndarray] = np.nan # Default to NaN

        try:
            # Create SymPy symbols for the inputs that are Measurements
            sympy_vars = sp.symbols(f'x:{ufunc.nin}') # Max arity symbols needed
            sympy_inputs_map = {} # Map index to symbol
            measurement_indices = [i for i, is_meas in enumerate(input_is_measurement) if is_meas]

            # Build the SymPy expression corresponding to the ufunc
            # Map NumPy ufuncs to their SymPy equivalents where possible
            # This mapping needs to be extended for more ufunc support
            sympy_func_map = {
                np.add: lambda x, y: x + y,
                np.subtract: lambda x, y: x - y,
                np.multiply: lambda x, y: x * y,
                np.true_divide: lambda x, y: x / y,
                np.power: lambda x, y: x**y,
                np.negative: lambda x: -x,
                np.positive: lambda x: +x,
                np.exp: sp.exp,
                np.log: sp.log,
                np.log10: lambda x: sp.log(x, 10),
                np.sqrt: sp.sqrt,
                np.sin: sp.sin,
                np.cos: sp.cos,
                np.tan: sp.tan,
                np.arcsin: sp.asin,
                np.arccos: sp.acos,
                np.arctan: sp.atan,
                np.absolute: sp.Abs, # Sympy Abs
                np.conjugate: sp.conjugate,
                np.hypot: lambda x, y: sp.sqrt(x**2 + y**2),
                # Add more mappings as needed...
                # np.square: lambda x: x**2, # Covered by power?
            }

            sympy_f = sympy_func_map.get(ufunc)
            if not sympy_f:
                 raise NotImplementedError(f"SymPy mapping for ufunc '{ufunc.__name__}' is not implemented.")

            # Create the symbolic expression using the first 'nin' symbols
            symbolic_expr = sympy_f(*sympy_vars[:ufunc.nin])

            # Calculate sum of squared error contributions: Σ (∂f/∂xi * σi)²
            variance_sq_sum = np.zeros_like(np.asarray(result_value), dtype=float)
            numeric_values_for_eval = input_values[:ufunc.nin] # Values needed for derivative evaluation

            for i in measurement_indices:
                 if i >= ufunc.nin: continue # Skip if index out of ufunc arity

                 # Get the corresponding symbol and error
                 var_sym = sympy_vars[i]
                 sigma_i = np.asarray(input_errors[i]) # Ensure error is array

                 # Skip calculation if error is zero for this input
                 if np.all(np.isclose(sigma_i, 0.0)):
                     continue

                 # Calculate the partial derivative symbolically
                 deriv_sym = sp.diff(symbolic_expr, var_sym)

                 # Lambdify the derivative for numerical evaluation using NumPy functions
                 # Pass all necessary symbols for the original function's arity
                 # Use 'numpy' module for numerical functions (sin, cos, etc.)
                 # Add specific mappings if needed (e.g., Abs -> np.abs)
                 modules = ['numpy', {'Abs': np.abs, 'conjugate': np.conjugate}]
                 deriv_func = sp.lambdify(sympy_vars[:ufunc.nin], deriv_sym, modules=modules)

                 # Evaluate the derivative numerically at the input values
                 with np.errstate(all='ignore'): # Ignore potential numerical issues during derivative eval
                      deriv_val = deriv_func(*numeric_values_for_eval)

                 # Calculate the variance contribution: (∂f/∂xi * σi)²
                 var_term = np.square(deriv_val * sigma_i)

                 # Add to the sum, handling NaNs carefully
                 # If term is NaN, the sum becomes NaN. If sum is already NaN, it stays NaN.
                 current_sum_is_nan = np.isnan(variance_sq_sum)
                 term_is_nan = np.isnan(var_term)
                 variance_sq_sum = np.add(variance_sq_sum, np.nan_to_num(var_term)) # Add non-NaN part
                 # Re-apply NaN status
                 variance_sq_sum = np.where(current_sum_is_nan | term_is_nan, np.nan, variance_sq_sum)


            # Final error is the square root of the summed variances
            with np.errstate(invalid='ignore'): # Ignore sqrt(negative) warnings -> NaN
                 result_error = np.sqrt(variance_sq_sum)

        except NotImplementedError as e:
            warnings.warn(f"Cannot propagate error for ufunc '{ufunc.__name__}': {e}. "
                          "Resulting error will be NaN.", RuntimeWarning)
            # Ensure error shape matches value shape, filled with NaN
            result_error = np.full_like(np.asarray(result_value), np.nan)
        except Exception as e_sympy:
            warnings.warn(f"Unexpected error during SymPy error propagation for '{ufunc.__name__}': {e_sympy}. "
                          "Resulting error will be NaN.", RuntimeWarning)
            result_error = np.full_like(np.asarray(result_value), np.nan)


        # --- Final Result Construction ---
        # Post-process error: Ensure NaN/Inf consistency with value, ensure non-negative
        value_arr = np.asarray(result_value)
        error_arr = np.asarray(result_error)

        # Where value is NaN or Inf, error should also be NaN or Inf respectively
        error_arr = np.where(np.isnan(value_arr), np.nan, error_arr)
        error_arr = np.where(np.isinf(value_arr), np.inf, error_arr)

        # If error calculation resulted in complex numbers (e.g., sqrt(-ve variance)), warn and take magnitude
        if np.iscomplexobj(error_arr):
             warnings.warn(f"Complex error encountered during propagation for '{ufunc.__name__}'. "
                           "Taking the magnitude (absolute value) of the error.", RuntimeWarning)
             error_arr = np.abs(error_arr)

        # Ensure error is non-negative (abs) and convert remaining NaNs from calculation if value is finite
        # (nan_to_num converts nan to 0, but we want to keep them if value is also nan)
        error_arr = np.abs(error_arr)
        error_arr = np.where(np.isnan(value_arr) | np.isinf(value_arr), error_arr, # Keep NaN/Inf if value has them
                             np.nan_to_num(error_arr, nan=np.nan, posinf=np.inf, neginf=np.nan)) # Convert others if needed


        # Create the final Measurement object
        final_result = Measurement(result_value, error_arr, unit=res_unit, name=res_name)

        # Perform final check for NaN/Inf issues in the result
        self._check_nan_inf(final_result.value, final_result.error, f"ufunc '{ufunc.__name__}'")

        # --- Match Return Type (Scalar/Array) ---
        # Try to return a scalar Measurement if the operation naturally resulted in a scalar
        # Heuristic: Check if all *Measurement* inputs were scalar-like (0-dim) AND output value is 0-dim
        all_meas_inputs_scalar = all(np.ndim(inputs[i].value) == 0
                                      for i in measurement_indices)
        output_is_scalar = (final_result.ndim == 0)

        if all_meas_inputs_scalar and output_is_scalar:
             # Return a scalar Measurement (access .value, .error directly which handle scalar/0-dim array)
              return Measurement(final_result.value, final_result.error, unit=final_result.unit, name=final_result.name)
        else:
             # Return the potentially array-based Measurement
             return final_result


    # --- Comparison Methods ---
    # Comparisons operate on nominal values ONLY. Uncertainty, name, unit are ignored.
    # This follows the behavior of libraries like `uncertainties`.
    # Returns boolean or boolean arrays, consistent with NumPy comparison behavior.

    def _compare(self, other: Any, comparison_operator: Callable) -> Union[bool, np.ndarray]:
        """Internal helper for comparison operations."""
        # Extract nominal value of other operand if it's a Measurement
        other_val = other.value if isinstance(other, Measurement) else other
        try:
            # Perform comparison on nominal values
            return comparison_operator(self.value, other_val)
        except (TypeError, ValueError):
            # If comparison fails (e.g., incompatible types), return NotImplemented
            return NotImplemented

    def __eq__(self, other: Any) -> Union[bool, np.ndarray]:
        """Equality comparison (==) based on nominal values."""
        return self._compare(other, np.equal)

    def __ne__(self, other: Any) -> Union[bool, np.ndarray]:
        """Inequality comparison (!=) based on nominal values."""
        return self._compare(other, np.not_equal)

    def __lt__(self, other: Any) -> Union[bool, np.ndarray]:
        """Less than comparison (<) based on nominal values."""
        return self._compare(other, np.less)

    def __le__(self, other: Any) -> Union[bool, np.ndarray]:
        """Less than or equal comparison (<=) based on nominal values."""
        return self._compare(other, np.less_equal)

    def __gt__(self, other: Any) -> Union[bool, np.ndarray]:
        """Greater than comparison (>) based on nominal values."""
        return self._compare(other, np.greater)

    def __ge__(self, other: Any) -> Union[bool, np.ndarray]:
        """Greater than or equal comparison (>=) based on nominal values."""
        return self._compare(other, np.greater_equal)

# --- END OF FILE measurement.py ---