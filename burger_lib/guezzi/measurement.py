# --- START OF FILE measurement.py ---

"""
Measurement Class for Handling Quantities with Uncertainties.

Defines the `Measurement` class, the core of the library for representing
physical quantities with associated uncertainties (errors) and units.
It automatically handles error propagation for standard arithmetic operations
and many NumPy universal functions (ufuncs).
"""

import numpy as np
import warnings
from typing import Union, List, Dict, Tuple, Any, Optional, Callable
from numbers import Number

from .utils import _format_value_error_eng, round_to_significant_figures

# ---------------------------- Ufunc Handlers ---------------------------------
# For __array_ufunc__, maps ufunc to (value_func, [derivative_funcs])
# derivative_funcs take the same number of arguments as the ufunc.
# Values are passed to derivative funcs, not Measurement objects.

UFUNC_HANDLERS: Dict[np.ufunc, Tuple[Callable, Callable]] = {
    np.negative: (np.negative, lambda x: -1.0),
    np.positive: (np.positive, lambda x: 1.0),
    np.exp: (np.exp, lambda x: np.exp(x)),
    np.log: (np.log, lambda x: 1.0/x),
    np.log10: (np.log10, lambda x: 1.0/(x * np.log(10))),
    np.sqrt: (np.sqrt, lambda x: 0.5/np.sqrt(x)),
    np.sin: (np.sin, lambda x: np.cos(x)),
    np.cos: (np.cos, lambda x: -np.sin(x)),
    np.tan: (np.tan, lambda x: 1.0/(np.cos(x)**2)),
    np.arcsin: (np.arcsin, lambda x: 1.0/np.sqrt(1 - x**2)),
    np.arccos: (np.arccos, lambda x: -1.0/np.sqrt(1 - x**2)),
    np.arctan: (np.arctan, lambda x: 1.0/(1 + x**2)),
    np.square: (np.square, lambda x: 2.0*x),
}


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
        Processes input, raises TypeError for complex numbers,
        and converts to a NumPy float array.
        """
        is_complex = False
        if isinstance(data, complex):
            is_complex = True
        elif isinstance(data, np.ndarray) and np.iscomplexobj(data):
            is_complex = True
        elif isinstance(data, (list, tuple)):
            # Check if any element in the list/tuple is complex
            if any(isinstance(x, complex) for x in data):
                is_complex = True
            else: # Try converting to array and check again
                try:
                    temp_arr = np.asarray(data)
                    if np.iscomplexobj(temp_arr):
                        is_complex = True
                except (TypeError, ValueError): # If conversion fails here, it will fail later anyway
                    pass

        if is_complex:
            raise TypeError(
                f"Input '{data_name}' contains complex numbers. "
                "Measurement class handles real-valued quantities only."
            )

        # Attempt conversion to float array
        try:
            # For scalars, ensure they become 1-element arrays
            if isinstance(data, Number): # float, int (complex already rejected)
                return np.array([data], dtype=float)
            
            arr_data = np.asarray(data, dtype=float)
            if arr_data.ndim == 0 : # Handle case where asarray results in 0-dim array from scalar-like
                return arr_data.reshape(1)
            return arr_data

        except (TypeError, ValueError) as e:
            raise TypeError(f"Input '{data_name}' could not be converted to a numeric float array: {e}")


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

        if self.value.size == 0: # Should be caught by _process_complex_input if input was empty list
            raise ValueError("Cannot initialize Measurement with empty values. "
                             "Input 'values' might have been empty or an unconvertible type.")

        # Broadcast values and errors to a common shape
        try:
             bcast_shape = np.broadcast(self.value, self.error).shape
             self.value = np.broadcast_to(self.value, bcast_shape)
             self.error = np.broadcast_to(self.error, bcast_shape)
        except ValueError:
              raise ValueError(f"Shape mismatch: Processed values (shape {self.value.shape}) "
                               f"and errors (shape {self.error.shape}) "
                               "cannot be broadcast together.")

        self.unit = unit
        self.name = name

        if np.any(self.error < 0):
            warnings.warn("Initializing Measurement with negative error(s). "
                          "Uncertainty must be non-negative. Taking absolute value.", UserWarning)
            self.error = np.abs(self.error)

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
        return self.shape[0]

    def __getitem__(self, key: Any) -> 'Measurement':
        try:
            new_value_array = self.value[key]
            new_error_array = self.error[key]
            
            # Ensure result is at least 1D for consistency if a scalar is selected
            if not isinstance(new_value_array, np.ndarray) or new_value_array.ndim == 0:
                new_value_array = np.array([new_value_array])
                new_error_array = np.array([new_error_array])
            elif new_value_array.size == 0: # Slicing resulted in an empty array
                 raise ValueError("Slicing resulted in an empty Measurement.")


            return Measurement(new_value_array, new_error_array, unit=self.unit, name=self.name) # Name/unit are copied
        except IndexError as e:
            raise IndexError(f"Error indexing Measurement: {e}")
        except ValueError as e: # Propagate ValueError (e.g. empty slice)
            raise ValueError(f"Slicing resulted in an invalid Measurement: {e}")
        except Exception as e: # Broader catch for other unexpected indexing issues
             raise TypeError(f"Invalid index or slice key for Measurement: {key}. Error: {e}")


    @property
    def nominal_value(self) -> np.ndarray:
        return self.value

    @property
    def std_dev(self) -> np.ndarray:
        return self.error

    @property
    def variance(self) -> np.ndarray:
         return np.square(self.error)

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
                return f"{val_item} ± {err_item}" + (f" {self.unit}" if self.unit else "")
        else: # For Measurements with multiple elements
            vectorized_formatter = np.vectorize(scalar_formatter_func, otypes=[str])
            try:
                formatted_array = vectorized_formatter(self.value, self.error)
                return np.array2string(formatted_array, separator=', ',
                                       formatter={'all': lambda x: x})
            except Exception as e:
                 warnings.warn(f"Error during array formatting in __str__: {e}", RuntimeWarning)
                 return (f"Measurement Array (value_shape={self.shape}, error_shape={self.shape})"
                         " - Formatting Error")

    def to_eng_string(self, sig_figs_error: int = 1) -> Union[str, np.ndarray]:
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

        vectorized_core_formatter = np.vectorize(
            _format_value_error_eng,
            excluded=['unit_symbol', 'sig_figs_error'], # These are fixed for all elements
            otypes=[str] # Specify the output type of the vectorized function
        )
        try:
            formatted_strings_array = vectorized_core_formatter(
                value=self.value, error=self.error,
                unit_symbol=self.unit, sig_figs_error=sig_figs_error
            )
            return formatted_strings_array
        except Exception as e:
            warnings.warn(f"Error during array formatting in to_eng_string: {e}", RuntimeWarning)
            # Fallback: return an array of strings indicating the error
            error_indicator_str = "FormatError"
            return np.full(self.shape, error_indicator_str, dtype=object)


    def _check_nan_inf(self, value: np.ndarray, error: np.ndarray, operation_name: str):
        """
        Internal helper to issue warnings if NaN or Infinity values appear
        in the result value or error after an operation.
        """

        value_arr = np.asarray(value)
        if np.any(np.isnan(value_arr)):
             warnings.warn(f"NaN value resulted from operation '{operation_name}'.", RuntimeWarning)
        if np.any(np.isinf(value_arr)):
             warnings.warn(f"Infinity value resulted from operation '{operation_name}'.", RuntimeWarning)

        error_arr = np.asarray(error)
        error_is_problematic = (np.isnan(error_arr) | np.isinf(error_arr)) & \
                               (~np.isnan(value_arr) & ~np.isinf(value_arr))
        if np.any(error_is_problematic):
             warnings.warn(f"NaN or Infinity error resulted from operation '{operation_name}' "
                           "where the corresponding value is valid.", RuntimeWarning)


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


    def __add__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands
        new_value = a + b
        new_error = np.hypot(sa, sb) # Sqrt(sa^2 + sb^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "add")
        return result

    def __radd__(self, other: Any) -> 'Measurement':
        return self.__add__(other) # Addition is commutative

    def __sub__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands
        new_value = a - b
        new_error = np.hypot(sa, sb) # Sqrt(sa^2 + sb^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "sub")
        return result

    def __rsub__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands
        new_value = b - a # other - self
        new_error = np.hypot(sa, sb) # Sqrt(sa^2 + sb^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "rsub")
        return result

    def __mul__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands
        new_value = a * b
        new_error = np.hypot(a*sb, b*sa) # Sqrt((a*sb)^2 + (b*sa)^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "mul")
        return result

    def __rmul__(self, other: Any) -> 'Measurement':
        return self.__mul__(other) # Multiplication is commutative

    def __truediv__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands # self / other = a / b
        new_value = a / b
        new_error = np.hypot(sa/b, a*sb/(b**2)) # Sqrt((sa/b)^2 + (a*sb/b^2)^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "div")
        return result

    def __rtruediv__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands # other / self = b / a
        new_value = b / a
        new_error = np.hypot(sb/a, b*sa/(a**2)) # Sqrt((sb/a)^2 + (b*sa/a^2)^2)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "rtruediv")
        return result

    def __pow__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        a, sa, b, sb = operands # self ** other = a ** b
        with np.errstate(invalid='ignore', divide='ignore'):
            if np.any((a < 0.0) & (b % 1.0 != 0.0)):
                 warnings.warn("Power M1**M2: Base M1 has negative values with non-integer exponent. ", RuntimeWarning)
            if np.any((a == 0.0) & (b < 0.0)):
                 warnings.warn("Power: Operation 0**negative encountered.", RuntimeWarning)
            if np.any(sb != 0.0) and np.any(a <= 0.0): # log(a) for error term
                 warnings.warn("Power M1**M2: Logarithm of non-positive base M1 required for error propagation "
                               "due to M2 uncertainty. Error contribution from M2 may be NaN.", RuntimeWarning)

            new_value = a ** b
            df_da= b * (a**(b-1))
            df_db= new_value * np.log(a) # log(a) can be nan if a <= 0
            new_error = np.hypot(df_da*sa, df_db*sb)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "pow")
        return result

    def __rpow__(self, other: Any) -> 'Measurement':
        operands = self._get_broadcasted_operands(other)
        if operands is None: return NotImplemented
        b, sb, a, sa = operands # other ** self = a ** b
        with np.errstate(invalid='ignore', divide='ignore'):
            if np.any((a < 0.0) & (b % 1.0 != 0.0)):
                 warnings.warn("Power M1**M2: Base M1 has negative values with non-integer exponent. ", RuntimeWarning)
            if np.any((a == 0.0) & (b < 0.0)):
                 warnings.warn("Power: Operation 0**negative encountered.", RuntimeWarning)
            if np.any(sb != 0.0) and np.any(a <= 0.0): # log(a) for error term
                 warnings.warn("Power M1**M2: Logarithm of non-positive base M1 required for error propagation "
                               "due to M2 uncertainty. Error contribution from M2 may be NaN.", RuntimeWarning)

            new_value = a ** b
            df_da= b * (a**(b-1))
            df_db= new_value * np.log(a) # log(a) can be nan if a <= 0
            new_error = np.hypot(df_da*sa, df_db*sb)

        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "pow")
        return result

    def __neg__(self) -> 'Measurement':
        return Measurement(-self.value, self.error, unit="", name="") # Name/unit not propagated

    def __pos__(self) -> 'Measurement':
        return Measurement(self.value, self.error, unit="", name="") # Name/unit not propagated

    def __abs__(self) -> 'Measurement':
        if np.any(np.abs(self.value) < 3 * self.error):
             warnings.warn("Taking absolute value of a Measurement whose uncertainty interval "
                           "likely includes zero (|value| < 3*error). Standard error propagation "
                           "(σ_abs ≈ σ_orig) is used but might be inaccurate.", UserWarning)
        
        new_value = np.abs(self.value)
        new_error = self.error # Approximation: error magnitude unchanged.
        
        result = Measurement(new_value, new_error, unit="", name="")
        self._check_nan_inf(result.value, result.error, "abs")
        return result

    def __array__(self, dtype=None) -> np.ndarray:
        warnings.warn("Casting Measurement to np.ndarray discards uncertainty, unit, and name. "
                      "Returning nominal values only.", UserWarning)
        if dtype is not None:
            return np.asarray(self.value, dtype=dtype)
        return np.asarray(self.value) # Keep original dtype if possible, usually float

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        # We only handle '__call__' method for unary ufuncs.
        if method != '__call__':
            return NotImplemented

        # Check if 'out' kwarg is present; we don't support it.
        if 'out' in kwargs:
            warnings.warn("The 'out' keyword argument is not supported for ufuncs with Measurement objects "
                          "and will be ignored.", UserWarning)
            return NotImplemented

        if ufunc.nin != 1: # Check if the ufunc is indeed unary
           raise TypeError(f"Try to use '{ufunc.__name__}' with 1 argument instead of {ufunc.nin}") 

        value = self.value
        error = self.error

        handler_tuple = UFUNC_HANDLERS.get(ufunc)

        if not handler_tuple:
            warnings.warn(
                f"Error propagation for unary ufunc '{ufunc.__name__}' is not implemented "
                "in UFUNC_HANDLERS. Resulting error will be NaN.", RuntimeWarning
            )
            result_value = ufunc(value, **kwargs)
            result_error = np.full_like(result_value, np.nan, dtype=float)

        else:
            _, derivative_func = handler_tuple
            result_value = ufunc(value, **kwargs)

            # --- Error Propagation ---
            # Error formula for f(x) is: sigma_f = |df/dx| * sigma_x
            with np.errstate(all='ignore'): # Suppress warnings during derivative calculation
                df_dx = derivative_func(value) # Derivative is a function of the value
                result_error = np.abs(df_dx * error)

        # Create the resulting Measurement object
        result_measurement = Measurement(result_value, result_error, unit="", name="")
        self._check_nan_inf(result_measurement.value, result_measurement.error, f"ufunc '{ufunc.__name__}'")
        if self.size == 1 and result_measurement.size == 1:
            return Measurement(result_measurement.value.item(), result_measurement.error.item(),
                               unit=result_measurement.unit, name=result_measurement.name)
        return result_measurement


    def _compare(self, other: Any, comparison_operator: Callable) -> Union[bool, np.ndarray]:
        # Comparisons operate on nominal values ONLY.
        # Do not use _get_broadcasted_operands as it forces 'other' to Measurement-like error structure
        
        self_val = self.value
        
        if isinstance(other, Measurement):
            other_val = other.value
        elif isinstance(other, (Number, np.ndarray)):
            if np.iscomplexobj(np.asarray(other)):
                raise TypeError("Cannot compare Measurement with complex number/array.")
            other_val = np.asarray(other, dtype=float) # Ensure 'other' is float for consistent comparison
        else:
            return NotImplemented

        try:
            # Perform comparison with broadcasting
            return comparison_operator(self_val, other_val)
        except (TypeError, ValueError): # Should be caught by broadcasting or ufunc if types are truly bad
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