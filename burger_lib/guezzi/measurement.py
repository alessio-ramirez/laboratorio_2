# --- START OF FILE measurement.py ---

"""
Measurement Class for Handling Quantities with Uncertainties.

Defines the `Measurement` class, the core of the library for representing
physical quantities with associated uncertainties (errors) and units.
It automatically handles error propagation for standard arithmetic operations
and many NumPy universal functions (ufuncs).
"""

import numpy as np
import sympy as sp # Will be used later in __array_ufunc__
import warnings
from typing import Union, List, Dict, Tuple, Any, Optional, Callable
from numbers import Number # Used for type checking scalar numbers

# Import formatting utilities (crucial for correct output)
# _format_value_error_eng is the primary formatter now.
# round_to_significant_figures is used internally by _format_value_error_eng.
from .utils import _format_value_error_eng, round_to_significant_figures

# ---------------------------- Measurement Class -----------------------------

class Measurement:
    """
    Represents a real-valued physical quantity with a nominal value and standard uncertainty.

    This class facilitates calculations involving measured quantities by automatically
    propagating uncertainties using first-order Taylor expansion (linear error
    propagation). It supports standard arithmetic operations (+, -, *, /, **) and
    many NumPy universal functions (ufuncs) like np.sin, np.log, np.exp.

    Error Propagation Theory:
        For a function f(x, y, ...) where x, y, ... are independent measurements
        with uncertainties σ_x, σ_y, ..., the uncertainty σ_f in f is approximated by:
            σ_f² ≈ (∂f/∂x * σ_x)² + (∂f/∂y * σ_y)² + ...

        This approximation is generally valid when the uncertainties are "small"
        relative to the values. The class assumes uncertainties represent 1σ.

        Key Assumptions:
        1.  **Real Values:** Input values/errors assumed real. Complex inputs trigger
            a warning, and only the real part is used.
        2.  **Linearity:** f is approximately linear over the uncertainty range.
        3.  **Independence:** Uncertainties in different Measurement operands are
            assumed independent unless handled otherwise (e.g., m - m).
        4.  **Gaussian Errors (Implied):** Interpretation of uncertainty as 1σ
            is most meaningful for approximately Normal error distributions.

    Attributes:
        value (np.ndarray): The nominal value(s) of the quantity (at least 1D).
        error (np.ndarray): The uncertainty (standard deviation, σ) associated
                            with the value (at least 1D, non-negative, and
                            broadcastable with `value`).
        unit (str): Physical unit string (e.g., "V", "m/s").
        name (str): Optional descriptive name (e.g., "Voltage").
    """
    __array_priority__ = 1000

    @staticmethod
    def _process_complex_input(data: Any, data_name: str) -> np.ndarray:
        """
        Processes input, handles complex numbers (warns, takes real part),
        and converts to a NumPy float array.
        """
        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            warnings.warn(
                f"Input '{data_name}' contains complex numbers. "
                "Using only the real part. Ensure this is intended.",
                UserWarning
            )
            return np.real(data).astype(float)
        elif isinstance(data, (list, tuple)):
            try:
                arr_data = np.asarray(data) # Convert first for robust complex check
                if np.iscomplexobj(arr_data):
                    warnings.warn(
                        f"Input '{data_name}' (list/tuple) contains complex numbers. "
                        "Using only the real part. Ensure this is intended.",
                        UserWarning
                    )
                    return np.real(arr_data).astype(float)
                return arr_data.astype(float)
            except (TypeError, ValueError) as e:
                 raise TypeError(f"Could not convert '{data_name}' to a numeric array: {e}")
        elif isinstance(data, Number): # Handles float, int, complex scalars
            if isinstance(data, complex):
                warnings.warn(
                    f"Input '{data_name}' is a complex scalar. "
                    "Using only the real part. Ensure this is intended.",
                    UserWarning
                )
                return np.array([np.real(data)], dtype=float) # Output as 1-element array
            return np.array([data], dtype=float) # Output as 1-element array
        
        # Fallback for other types, attempt direct conversion
        try:
            return np.asarray(data, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Input '{data_name}' could not be converted to a numeric array: {e}")


    def __init__(self,
                 values: Union[float, list, tuple, np.ndarray, dict, 'Measurement'],
                 errors: Union[float, list, tuple, np.ndarray, None] = None,
                 magnitude: int = 0,
                 unit: str = "",
                 name: str = ""):
        """
        Initializes a Measurement object. Must represent at least one value.

        Args:
            values: Nominal value(s). Scalar, sequence, dict, or Measurement.
                    Complex inputs: real part used with warning.
            errors: Uncertainty(ies). Scalar, sequence, or None (implies zero error).
                    Complex inputs: real part used with warning.
            magnitude: Power-of-10 scaling factor (e.g., `magnitude=-3` for milli).
            unit: Physical unit symbol string.
            name: Descriptive name string.

        Raises:
            ValueError: If `values` is an empty dict, or if processed `values`/`errors`
                        result in an empty array (size 0), or if shapes are incompatible,
                        or `values` is dict and `errors` is not None.
            TypeError: If inputs cannot be converted to numeric arrays.
        """

        if isinstance(values, Measurement):
            self.value = np.copy(values.value)
            self.error = np.copy(values.error)
            self.unit = unit if unit else values.unit
            self.name = name if name else values.name
            if errors is not None or magnitude != 0:
                 warnings.warn("Ignoring 'errors' and 'magnitude' when initializing "
                               "from an existing Measurement object.", UserWarning)
            # A Measurement copied from another is inherently valid (non-empty).
            return

        scale_factor = 10.0 ** magnitude

        if isinstance(values, dict):
            if not values:
                raise ValueError("Cannot initialize Measurement from an empty dictionary.")
            if errors is not None:
                raise ValueError("Argument 'errors' must be None when 'values' is a dictionary.")
            
            _vals_arr_raw = list(values.keys())
            _errs_arr_raw = list(values.values())

            _vals_arr_processed = self._process_complex_input(_vals_arr_raw, "values (from dict keys)")
            _errs_arr_processed = self._process_complex_input(_errs_arr_raw, "errors (from dict values)")

            self.value = np.atleast_1d(_vals_arr_processed * scale_factor)
            self.error = np.atleast_1d(_errs_arr_processed * scale_factor)
        else:
            _vals_arr_processed = self._process_complex_input(values, "values")

            if errors is None:
                _errs_arr_processed = np.zeros_like(_vals_arr_processed, dtype=float)
            else:
                 _errs_arr_processed = self._process_complex_input(errors, "errors")

            self.value = np.atleast_1d(_vals_arr_processed * scale_factor)
            self.error = np.atleast_1d(_errs_arr_processed * scale_factor)

        # Ensure non-empty after processing
        if self.value.size == 0:
            raise ValueError("Cannot initialize Measurement with empty values after processing. "
                             "Input 'values' might have been an empty list/tuple or resulted in an empty array.")

        # Broadcast values and errors to a common shape
        try:
             bcast_shape = np.broadcast(self.value, self.error).shape
             self.value = np.broadcast_to(self.value, bcast_shape)
             self.error = np.broadcast_to(self.error, bcast_shape)
        except ValueError:
              # This error should ideally be caught if self.value.size check is robust
              raise ValueError(f"Shape mismatch: Processed values (shape: {self.value.shape}) "
                               f"and errors (shape: {self.error.shape}) "
                               "cannot be broadcast together.")

        self.unit = unit
        self.name = name

        if np.any(self.error < 0):
            warnings.warn("Initializing Measurement with negative error(s). "
                          "Uncertainty must be non-negative. Taking absolute value.", UserWarning)
            self.error = np.abs(self.error)

    # --- NumPy-like Properties ---
    @property
    def ndim(self) -> int:
        """Number of array dimensions of the value/error."""
        return self.value.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of array dimensions of the value/error."""
        return self.value.shape

    @property
    def size(self) -> int:
        """Total number of elements in the value/error array."""
        return self.value.size # Always > 0 due to __init__ checks

    def __len__(self) -> int:
        """Length of the first dimension."""
        return self.shape[0]

    def __getitem__(self, key: Any) -> 'Measurement':
        """
        Allows indexing and slicing. Returns a *new* Measurement object.
        If slicing results in an empty Measurement, a ValueError is raised by the constructor.
        """
        try:
            new_value_array = self.value[key]
            new_error_array = self.error[key]

            try:
                 key_str = str(key).replace(':', '-').replace(' ', '').replace(',', '_')
            except:
                 key_str = "slice"

            new_name = f"{self.name}[{key_str}]" if self.name else f"Data[{key_str}]"
            # Measurement constructor will raise ValueError if new_value_array.size is 0
            return Measurement(new_value_array, new_error_array, unit=self.unit, name=new_name)

        except IndexError as e:
            raise IndexError(f"Error indexing Measurement: {e}")
        except ValueError as e: # Propagate ValueError from Measurement constructor (e.g. empty slice)
            raise ValueError(f"Slicing resulted in an invalid Measurement: {e}")
        except Exception as e:
             raise TypeError(f"Invalid index or slice key for Measurement: {key}. Error: {e}")

    # --- Convenience Properties ---
    @property
    def nominal_value(self) -> np.ndarray:
        """Alias for `self.value` (the nominal value(s) as np.ndarray)."""
        return self.value

    @property
    def std_dev(self) -> np.ndarray:
        """Alias for `self.error` (the uncertainty/standard deviation as np.ndarray)."""
        return self.error

    @property
    def variance(self) -> np.ndarray:
         """Returns the variance (square of the uncertainty `self.error`)."""
         return np.square(self.error)

    # --- String Representations ---

    def __repr__(self) -> str:
        """
        Detailed, unambiguous string representation for debugging.
        """
        name_str = f", name='{self.name}'" if self.name else ""
        unit_str = f", unit='{self.unit}'" if self.unit else ""
        val_repr = np.array_repr(self.value)
        err_repr = np.array_repr(self.error)
        return f"Measurement(value={val_repr}, error={err_repr}{name_str}{unit_str})"

    def __str__(self) -> str:
        """
        User-friendly string representation, using engineering notation.
        Defaults to 1 significant figure for the error.
        Returns a single string, even for array-like Measurements (uses np.array2string).
        """
        sig_figs_error_default = 1

        # Core scalar formatter
        scalar_formatter_func = lambda v, e: _format_value_error_eng(
            v, e, self.unit, sig_figs_error_default
        )

        if self.size == 1: # Handles Measurements with a single value/error pair
            try:
                return scalar_formatter_func(self.value.item(), self.error.item())
            except Exception as e:
                warnings.warn(f"Error during scalar formatting in __str__: {e}", RuntimeWarning)
                val_item = self.value.item()
                err_item = self.error.item()
                return f"{val_item} \u00B1 {err_item}" + (f" {self.unit}" if self.unit else "")
        else: # For Measurements with multiple elements
            vectorized_formatter = np.vectorize(scalar_formatter_func, otypes=[str])
            try:
                formatted_array = vectorized_formatter(self.value, self.error)
                # np.array2string is appropriate here as __str__ should return a single string
                return np.array2string(formatted_array, separator=', ',
                                       formatter={'all': lambda x: x})
            except Exception as e:
                 warnings.warn(f"Error during array formatting in __str__: {e}", RuntimeWarning)
                 return (f"Measurement Array (value_shape={self.shape}, error_shape={self.shape})"
                         " - Formatting Error")


    def to_eng_string(self, sig_figs_error: int = 1) -> np.ndarray:
        """
        Formats the measurement(s) using engineering notation with SI prefixes,
        returning a NumPy array of formatted strings.
        The value is rounded based on the error's significant figures.

        Args:
            sig_figs_error (int): Number of significant figures for the error. (Default 1).

        Returns:
            np.ndarray: A NumPy array of formatted strings. The shape of this array
                        matches the shape of `self.value`.
        """
        if not isinstance(sig_figs_error, int) or sig_figs_error <= 0:
            raise ValueError("Number of significant figures for error must be positive.")

        # Vectorize the core formatting utility _format_value_error_eng.
        # It will be applied element-wise to self.value and self.error.
        # self.unit and sig_figs_error are passed as fixed arguments to each call.
        vectorized_core_formatter = np.vectorize(
            _format_value_error_eng,
            excluded=['unit_symbol', 'sig_figs_error'], # These are fixed for all elements
            otypes=[str] # Specify the output type of the vectorized function
        )

        try:
            # Apply the vectorized formatter
            formatted_strings_array = vectorized_core_formatter(
                value=self.value,         # Operate on self.value
                error=self.error,         # Operate on self.error
                unit_symbol=self.unit,    # Pass self.unit to each internal call
                sig_figs_error=sig_figs_error # Pass sig_figs_error to each internal call
            )
            return formatted_strings_array
        except Exception as e:
            warnings.warn(f"Error during array formatting in to_eng_string: {e}", RuntimeWarning)
            # Fallback: return an array of strings indicating the error
            error_indicator_str = "FormatError"
            return np.full(self.shape, error_indicator_str, dtype=object)

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
        other_is_const = isinstance(other, (Number, np.ndarray)) and not is_other_meas

        # --- Unit Logic ---
        if operation in ['add', 'sub']:
            # Addition/Subtraction: Require identical units for the result to have that unit.
            if self.unit and other_unit and self.unit == other_unit:
                res_unit = self.unit # Units match, preserve it
            elif self.unit and (other_unit is None or other_is_const):
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
        op_symbols = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**',
                      'radd': '+', 'rsub': '-', 'rmul': '*', 'rtruediv': '/', 'rpow': '**'}
        op_sym = op_symbols.get(operation, operation) # Use symbol or operation name

        # Represent operands in the name
        self_name_part = f"({self.name})" if self.name else 'm1'
        if is_other_meas:
            other_name_part = f"({other_name})" if other_name else 'm2'
        elif other_is_const:
            other_name_part = 'const' # Indicate a constant value
        else:
            other_name_part = 'other' # Fallback for unknown types

        if operation in ['neg', 'pos', 'abs']:
             # Unary operations
             res_name = f"{op_sym}{self_name_part}"
        elif is_other_meas or other_is_const:
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


    # --- Broadcasting Helper ---
    def _get_broadcasted_operands(self, other: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Internal helper to get value and error arrays for self and other,
        broadcasted to a common shape. Handles Measurement, scalar, or ndarray inputs.

        Returns:
            Tuple(self_val, self_err, other_val, other_err) broadcasted, or None if incompatible.
        """
        self_val = np.asarray(self.value)
        self_err = np.asarray(self.error)

        if isinstance(other, Measurement):
            other_val = np.asarray(other.value)
            other_err = np.asarray(other.error)
        elif isinstance(other, (Number, np.ndarray)):
            other_val = np.asarray(other)
            # Assume zero error for constants/arrays unless they are Measurement objects
            other_err = np.zeros_like(other_val, dtype=float)
        else:
            # Indicate incompatible type
            return None

        # Perform broadcasting
        try:
            bcast = np.broadcast(self_val, self_err, other_val, other_err)
            shape = bcast.shape
            return (np.broadcast_to(self_val, shape),
                    np.broadcast_to(self_err, shape),
                    np.broadcast_to(other_val, shape),
                    np.broadcast_to(other_err, shape))
        except ValueError:
             # Broadcasting failed
             raise ValueError(f"Operands with shapes {self_val.shape} and {other_val.shape} "
                              "cannot be broadcast together.")


    # --- Arithmetic Operations (Revised with Broadcasting) ---

    def __add__(self, other: Any) -> 'Measurement':
        op = "add"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented # Incompatible type

        a, sa, b, sb = operands
        new_value = a + b
        # Errors add in quadrature, assuming independence
        new_error = np.hypot(sa, sb)

        # Create result and check for issues
        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __radd__(self, other: Any) -> 'Measurement':
        # Use __add__ since addition is commutative.
        # Metadata handled correctly by __add__ when called as other + self.
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Measurement':
        op = "sub"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        a, sa, b, sb = operands
        new_value = a - b
        # Errors add in quadrature for subtraction too (variances add)
        new_error = np.hypot(sa, sb)

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rsub__(self, other: Any) -> 'Measurement':
        # k - (a ± sa) = (k-a) ± sa
        op = "rsub"
        res_name, res_unit = self._propagate_metadata(other, op) # Correct metadata order
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        # Note: _get_broadcasted_operands returns (k, 0, a, sa)
        k, _, a, sa = operands # Error on k (other) is zero if it wasn't a Measurement
        new_value = a - k
        new_error = _ # Error is just the error from self

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __mul__(self, other: Any) -> 'Measurement':
        op = "mul"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        a, sa, b, sb = operands
        new_value = a * b

        # Error: sqrt( (b*sa)² + (a*sb)² )
        with np.errstate(invalid='ignore'): # Handle potential 0*inf etc.
             term1 = b * sa
             term2 = a * sb
             # Handle cases where a or b is exactly zero: the corresponding error term is zero.
             # (hypot handles NaN correctly, but explicit zeroing is safer for clarity)
             term1 = np.where(np.isclose(b, 0.0), 0.0, term1)
             term2 = np.where(np.isclose(a, 0.0), 0.0, term2)
             new_error = np.hypot(term1, term2)

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rmul__(self, other: Any) -> 'Measurement':
        # k * M is the same as M * k.
        return self.__mul__(other)

    # --- REPLACE this method in measurement.py ---
    def __truediv__(self, other: Any) -> 'Measurement':
        op = "div"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        a, sa, b, sb = operands

        with np.errstate(divide='ignore', invalid='ignore'):
            new_value = a / b # Let numpy handle 0/0 -> nan, x/0 -> inf

            # Error: sqrt( (sa/b)² + (a*sb / b²)² )
            #      = sqrt( (sa/b)² + (new_value * sb / b)² )
            term1 = sa / b
            term2 = new_value * (sb / b)

            # --- Refine error terms ---
            # Where denominator b is zero:
            # The term1 contribution (from sa) is inf if sa is non-zero, else 0.
            value_if_b_is_zero_for_term1 = np.where(np.isclose(sa, 0.0), 0.0, np.inf)
            term1 = np.where(np.isclose(b, 0.0), value_if_b_is_zero_for_term1, term1)

            # The term2 contribution (from sb) is inf if b=0, unless the numerator (a*sb or new_value*sb) is also zero.
            # Check if b is zero AND the numerator components (new_value and sb) are NOT both effectively zero.
            # Note: Using new_value = a/b. If b=0, new_value is inf (if a!=0) or nan (if a=0).
            # It's safer to check based on original a and sb.
            # term2 should be inf if b=0 and a!=0 and sb!=0
            is_inf_term2 = np.isclose(b, 0.0) & ~np.isclose(a, 0.0) & ~np.isclose(sb, 0.0)
            term2 = np.where(is_inf_term2, np.inf, term2)

            # Where numerator a is zero (and b is not zero):
            # term1 = sa/b (can be non-zero)
            # term2 = 0 * (sb/b) = 0
            term2 = np.where(np.isclose(a, 0.0) & ~np.isclose(b, 0.0), 0.0, term2)

            # Where error sb is zero (and b is not zero):
            # term2 = new_value * (0/b) = 0
            term2 = np.where(np.isclose(sb, 0.0) & ~np.isclose(b, 0.0), 0.0, term2)

            # Combine in quadrature
            new_error = np.hypot(term1, term2)

            # Final check: if value is 0, error might need adjustment (e.g. 0/1)
            # If new_value is exactly 0 (e.g., 0 / non_zero_b), then error is |sa/b|.
            new_error = np.where(np.isclose(new_value, 0.0) & ~np.isclose(b, 0.0), np.abs(term1), new_error)

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __rtruediv__(self, other: Any) -> 'Measurement':
        # k / (a ± sa)
        op = "rtruediv"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        k, _, a, sa = operands # other_err (sk) is 0

        with np.errstate(divide='ignore', invalid='ignore'):
            new_value = k / a

            # Error: | (-k*sa / a²) | = | (k/a) * sa / a | = | new_value * sa / a |
            new_error = np.abs(new_value * (sa / a))

            # Handle cases involving zero
            # If a=0, error is inf (unless k=0 or sa=0)
            new_error = np.where(np.isclose(a, 0.0),
                                 np.inf if not (np.isclose(k, 0.0) or np.isclose(sa, 0.0)) else 0.0,
                                 new_error)
            # If k=0, result is 0 with 0 error (unless a=0, gives nan/inf handled above)
            new_error = np.where(np.isclose(k, 0.0) & ~np.isclose(a, 0.0), 0.0, new_error)
            # If sa=0, error is 0 (unless a=0)
            new_error = np.where(np.isclose(sa, 0.0) & ~np.isclose(a, 0.0), 0.0, new_error)

        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result

    def __pow__(self, other: Any) -> 'Measurement':
        op = "pow"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        a, sa, b, sb = operands

        with np.errstate(invalid='ignore'): # Manage 0**neg, neg**frac warnings etc.
             # Check for domain issues that yield complex numbers or NaN/Inf
             if np.any((a < 0) & (b % 1 != 0)):
                  warnings.warn(f"Power M**k or M1**M2: Base has negative values with non-integer exponent. "
                                "Result may be complex or NaN. Error propagation assumes real result.", RuntimeWarning)
             if np.any((np.isclose(a, 0.0)) & (b < 0)):
                  warnings.warn(f"Power: Operation 0**negative encountered.", RuntimeWarning)
             # Check if M2 involved (sb != 0) and base a<=0 for log(a) issue
             if np.any(sb != 0) and np.any(a <= 0):
                  warnings.warn(f"Power M1**M2: Logarithm of non-positive base M1 ({a}) required for error propagation "
                                "due to M2 uncertainty. Error contribution from M2 may be NaN.", RuntimeWarning)

             # Calculate value
             new_value = a ** b

             # Calculate error contributions
             # Term 1: (∂f/∂a * sa) = (b * a**(b-1) * sa)
             df_da = b * (a ** (b - 1))
             term_a = df_da * sa

             # Term 2: (∂f/∂b * sb) = (a**b * ln(a) * sb) = (new_value * ln(a) * sb)
             # Only calculate if sb is non-zero
             term_b = np.zeros_like(new_value, dtype=float)
             has_sb_error = np.any(sb != 0.0)
             if has_sb_error:
                 with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warning
                     log_a = np.log(a)
                 df_db = new_value * log_a # Will be nan if a <= 0
                 term_b = df_db * sb

             # Handle NaNs/Infs arising from derivatives or inputs
             # If derivative is NaN/Inf, term is NaN/Inf (unless error is 0)
             term_a = np.where((np.isnan(df_da) | np.isinf(df_da)) & ~np.isclose(sa, 0.0), df_da, term_a)
             term_b = np.where((np.isnan(df_db) | np.isinf(df_db)) & ~np.isclose(sb, 0.0), df_db, term_b)
             # If error is zero, term is zero
             term_a = np.where(np.isclose(sa, 0.0), 0.0, term_a)
             term_b = np.where(np.isclose(sb, 0.0), 0.0, term_b)
             # If base a=0:
             # If b>1, a**(b-1)=0 -> df_da=0 -> term_a=0. If b=1, a**0=1 -> df_da=1 -> term_a=sa. If 0<b<1, a**(b-1)=inf. If b=0, a**-1=inf. If b<0, a**(b-1)=inf.
             # If base a=0: ln(a)=-inf -> df_db=-inf (if new_value=0). term_b = -inf * sb.

             # More robust handling for a=0:
             # Case: a=0, b>0 => new_value=0. Error requires care.
             # If b > 1: df_da=0, term_a=0. log(a)=-inf, df_db=-inf, term_b=-inf*sb. Error is inf if sb!=0.
             # If b = 1: new_value=0. df_da=1*0^0=1 (convention?), term_a=sa. log(a)=-inf, df_db=0*(-inf)=nan?, term_b=nan. Error is sa.
             # If 0 < b < 1: new_value=0. df_da=inf, term_a=inf*sa. df_db=-inf, term_b=-inf*sb. Error is inf if sa!=0 or sb!=0.
             # If b=0: new_value=1. df_da=0*0^-1=inf, term_a=inf*sa. df_db=1*log(0)=-inf, term_b=-inf*sb. Error is inf if sa!=0 or sb!=0.
             # If b < 0: new_value=inf.

             # Combine in quadrature - hypot handles inf correctly (sqrt(inf^2 + x^2) = inf)
             new_error = np.hypot(term_a, term_b)

             # Specific overrides based on simpler logic where possible
             # If exponent b=0, result is 1, error is 0 (overrides inf calculations above if a=0)
             new_error = np.where(np.isclose(b, 0.0), 0.0, new_error)
             new_value = np.where(np.isclose(b, 0.0), 1.0, new_value)
             # If exponent b=1, result is a, error is sa
             new_error = np.where(np.isclose(b, 1.0), sa, new_error)
             new_value = np.where(np.isclose(b, 1.0), a, new_value)
             # If base a=0 and b>0 and sb=0, result is 0, error is 0 (unless b=1, handled above)
             is_a0_bpos_sb0 = np.isclose(a, 0.0) & (b > 0) & np.isclose(sb, 0.0) & ~np.isclose(b, 1.0)
             new_error = np.where(is_a0_bpos_sb0, 0.0, new_error)


        result = Measurement(new_value, new_error, unit=res_unit, name=res_name)
        self._check_nan_inf(result.value, result.error, op)
        return result


    def __rpow__(self, other: Any) -> 'Measurement':
        # k ** (a ± sa)
        op = "rpow"
        res_name, res_unit = self._propagate_metadata(other, op)
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented

        k, _, a, sa = operands # other_err (sk) is 0

        with np.errstate(invalid='ignore', divide='ignore'):
            # Domain warnings
            if np.any(k < 0):
                 warnings.warn(f"Power k**M: Base k={k} is negative. Result may be complex or NaN.", RuntimeWarning)
            if np.any((np.isclose(k, 0.0)) & (a <= 0)):
                 warnings.warn(f"Power k**M: Operation 0 ** (M <= 0) encountered.", RuntimeWarning)
            if np.any(k <= 0):
                 warnings.warn(f"Power k**M: Logarithm of non-positive base k ({k}) required for error propagation. "
                               "Error may be NaN.", RuntimeWarning)

            # Calculate value
            new_value = k ** a

            # Calculate error: σ_f ≈ | ∂f/∂a * σ_a | = | (k^a * ln(k)) * σ_a | = | new_value * ln(k) * sa |
            with np.errstate(divide='ignore', invalid='ignore'):
                log_k = np.log(k) # Will be nan/inf for k<=0
            deriv = new_value * log_k
            new_error = np.abs(deriv * sa)

            # Refine error for special cases
            # If ln(k) was NaN/Inf, error is NaN/Inf (unless sa=0)
            new_error = np.where((np.isnan(log_k) | np.isinf(log_k)) & ~np.isclose(sa, 0.0), np.nan, new_error)
            # If exponent error sa=0, result error is 0 (unless base/exponent invalid -> NaN/Inf)
            new_error = np.where(np.isclose(sa, 0.0) & np.isfinite(new_value), 0.0, new_error)
            # If base k=1, result is 1 exactly, error is 0
            new_error = np.where(np.isclose(k, 1.0), 0.0, new_error)
            new_value = np.where(np.isclose(k, 1.0), 1.0, new_value)
            # If base k=0 and exponent a > 0, result is 0 exactly, error is 0
            is_k0_apos = np.isclose(k, 0.0) & (a > 0)
            new_error = np.where(is_k0_apos, 0.0, new_error)
            new_value = np.where(is_k0_apos, 0.0, new_value)

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
        if np.any(np.abs(np.asarray(self.value)) < 3 * np.asarray(self.error)): # Check if |value| < 3*error
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
            return NotImplemented # Be strict: if user specified 'out', don't handle it.


        # Convert inputs for processing: extract value, error, metadata, handle broadcasting implicitly later
        processed_inputs = [] # Store tuples of (value, error, unit, name, is_measurement)
        has_measurement = False
        input_shapes = []

        for x in inputs:
            if isinstance(x, Measurement):
                val = np.asarray(x.value)
                err = np.asarray(x.error)
                processed_inputs.append((val, err, x.unit, x.name, True))
                has_measurement = True
                input_shapes.append(val.shape)
            elif isinstance(x, (Number, np.ndarray)):
                val = np.asarray(x)
                err = np.zeros_like(val, dtype=float) # Zero error
                processed_inputs.append((val, err, "", "", False))
                input_shapes.append(val.shape)
            else:
                return NotImplemented # Unsupported input type

        if not has_measurement:
             return NotImplemented # Should not happen if __array_priority__ is set

        # Determine common broadcast shape
        try:
            bcast_shape = np.broadcast(*(inp[0] for inp in processed_inputs)).shape
        except ValueError:
             # Shape mismatch even before ufunc call
             shapes = [inp[0].shape for inp in processed_inputs]
             raise ValueError(f"Input shapes {shapes} for ufunc '{ufunc.__name__}' cannot be broadcast together.")

        # Broadcast all input values and errors
        broadcast_inputs = []
        for val, err, unit, name, is_meas in processed_inputs:
            b_val = np.broadcast_to(val, bcast_shape)
            b_err = np.broadcast_to(err, bcast_shape)
            broadcast_inputs.append((b_val, b_err, unit, name, is_meas))

        input_values = [inp[0] for inp in broadcast_inputs]
        input_errors = [inp[1] for inp in broadcast_inputs]
        input_units = [inp[2] for inp in broadcast_inputs]
        input_names = [inp[3] for inp in broadcast_inputs]
        input_is_measurement = [inp[4] for inp in broadcast_inputs]


        # --- Value Calculation ---
        try:
            with np.errstate(all='ignore'):
                 result_value = ufunc(*input_values, **kwargs)
        except Exception as e:
            warnings.warn(f"Error during ufunc '{ufunc.__name__}' value calculation: {e}. "
                          "Result value calculation failed.", RuntimeWarning)
            result_value = np.full(bcast_shape, np.nan)


        # --- Metadata Propagation (Result Unit and Name) ---
        res_unit = ""
        operand_names = [name if name else f"op{i+1}" for i, name in enumerate(input_names)]
        res_name = f"{ufunc.__name__}({', '.join(operand_names)})"

        # Simple unit rules (as before)
        if ufunc.nin == 1 and input_is_measurement[0]:
             if ufunc in [np.negative, np.positive, np.conjugate, np.absolute]:
                 res_unit = input_units[0]
        elif ufunc.nin == 2:
             unit1 = input_units[0]
             unit2 = input_units[1]
             if ufunc in [np.add, np.subtract, np.hypot]:
                 if unit1 and unit1 == unit2: res_unit = unit1
                 elif unit1 and not unit2 and not input_is_measurement[1]: res_unit = unit1
                 elif unit2 and not unit1 and not input_is_measurement[0]: res_unit = unit2
                 elif unit1 and unit2 and unit1 != unit2:
                      warnings.warn(f"Ufunc '{ufunc.__name__}' on Measurements with incompatible units: "
                                    f"'{unit1}' and '{unit2}'. Result unit cleared.", UserWarning)
                      res_unit = ""

        # --- Error Propagation using SymPy ---
        result_error: Union[float, np.ndarray] = np.nan

        try:
            sympy_vars = sp.symbols(f'x:{ufunc.nin}')
            sympy_func_map = {
                np.add: lambda x, y: x + y, np.subtract: lambda x, y: x - y,
                np.multiply: lambda x, y: x * y, np.true_divide: lambda x, y: x / y,
                np.power: lambda x, y: x**y, np.negative: lambda x: -x,
                np.positive: lambda x: +x, np.exp: sp.exp, np.log: sp.log,
                np.log10: lambda x: sp.log(x, 10), np.sqrt: sp.sqrt,
                np.sin: sp.sin, np.cos: sp.cos, np.tan: sp.tan,
                np.arcsin: sp.asin, np.arccos: sp.acos, np.arctan: sp.atan,
                np.absolute: sp.Abs, np.conjugate: sp.conjugate,
                np.hypot: lambda x, y: sp.sqrt(x**2 + y**2),
                np.square: lambda x: x**2,
                # Add more...
            }

            sympy_f = sympy_func_map.get(ufunc)
            if not sympy_f:
                 raise NotImplementedError(f"SymPy mapping for ufunc '{ufunc.__name__}'")

            symbolic_expr = sympy_f(*sympy_vars[:ufunc.nin])
            variance_sq_sum = np.zeros_like(np.asarray(result_value), dtype=float)
            numeric_values_for_eval = input_values[:ufunc.nin] # Use broadcasted values

            for i in range(ufunc.nin):
                 sigma_i = input_errors[i] # Use broadcasted errors
                 if not input_is_measurement[i] or np.all(np.isclose(sigma_i, 0.0)):
                     continue # Skip if not a measurement or has zero error

                 var_sym = sympy_vars[i]
                 deriv_sym = sp.diff(symbolic_expr, var_sym)
                 modules = ['numpy', {'Abs': np.abs, 'conjugate': np.conjugate}]
                 deriv_func = sp.lambdify(sympy_vars[:ufunc.nin], deriv_sym, modules=modules)

                 with np.errstate(all='ignore'):
                      deriv_val = deriv_func(*numeric_values_for_eval)

                 var_term = np.square(deriv_val * sigma_i)
                 current_sum_is_nan = np.isnan(variance_sq_sum)
                 term_is_nan = np.isnan(var_term)
                 variance_sq_sum = np.add(variance_sq_sum, np.nan_to_num(var_term))
                 variance_sq_sum = np.where(current_sum_is_nan | term_is_nan, np.nan, variance_sq_sum)

            with np.errstate(invalid='ignore'):
                 result_error = np.sqrt(variance_sq_sum)

        except NotImplementedError as e:
            warnings.warn(f"Cannot propagate error for ufunc '{ufunc.__name__}': {e}. Error -> NaN.", RuntimeWarning)
            result_error = np.full_like(np.asarray(result_value), np.nan)
        except Exception as e_sympy:
            warnings.warn(f"SymPy error propagation failed for '{ufunc.__name__}': {e_sympy}. Error -> NaN.", RuntimeWarning)
            result_error = np.full_like(np.asarray(result_value), np.nan)


        # --- Final Result Construction ---
        value_arr = np.asarray(result_value)
        error_arr = np.asarray(result_error)
        error_arr = np.where(np.isnan(value_arr), np.nan, error_arr)
        error_arr = np.where(np.isinf(value_arr), np.inf, error_arr)
        if np.iscomplexobj(error_arr):
             warnings.warn(f"Complex error for '{ufunc.__name__}'. Taking magnitude.", RuntimeWarning)
             error_arr = np.abs(error_arr)
        error_arr = np.abs(error_arr)
        error_arr = np.where(np.isnan(value_arr) | np.isinf(value_arr), error_arr,
                             np.nan_to_num(error_arr, nan=np.nan, posinf=np.inf, neginf=np.nan))


        final_result = Measurement(result_value, error_arr, unit=res_unit, name=res_name)
        self._check_nan_inf(final_result.value, final_result.error, f"ufunc '{ufunc.__name__}'")

        # --- Match Return Type (Scalar/Array) ---
        # Return scalar if all measurement inputs were scalar AND output is scalar
        all_meas_inputs_scalar = all(inp[0].ndim == 0 for inp in broadcast_inputs if inp[4])
        output_is_scalar = (final_result.ndim == 0)

        if all_meas_inputs_scalar and output_is_scalar:
             # Ensure value/error are python floats before creating new Measurement
             return Measurement(final_result.value.item(), final_result.error.item(),
                                unit=final_result.unit, name=final_result.name)
        else:
             return final_result


    # --- Comparison Methods ---
    # Comparisons operate on nominal values ONLY.

    def _compare(self, other: Any, comparison_operator: Callable) -> Union[bool, np.ndarray]:
        """Internal helper for comparison operations."""
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, _, b, _ = operands # Don't need errors for comparison
        try:
            return comparison_operator(a, b)
        except (TypeError, ValueError):
            return NotImplemented

    def __eq__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.equal)

    def __ne__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.not_equal)

    def __lt__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.less)

    def __le__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.less_equal)

    def __gt__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.greater)

    def __ge__(self, other: Any) -> Union[bool, np.ndarray]:
        return self._compare(other, np.greater_equal)

# --- END OF FILE measurement.py ---