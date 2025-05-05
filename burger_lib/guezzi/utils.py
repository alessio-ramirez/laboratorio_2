# --- START OF FILE utils.py ---
"""
Internal Utility Functions for Guezzi Library

Contains helper functions for tasks like SI prefix handling and value/error formatting,
primarily used by other modules within the library (Measurement, tables, plotting).
"""

import numpy as np
import math
from typing import Tuple, Optional
import warnings
import decimal # Use decimal for more precise rounding control if needed, though float arithmetic is usually sufficient here

# Dictionary mapping exponents (powers of 10) to SI prefixes
# Covers standard prefixes from Yotta (10^24) to yocto (10^-24)
SI_PREFIXES = {
    24: 'Y', 21: 'Z', 18: 'E', 15: 'P', 12: 'T', 9: 'G', 6: 'M', 3: 'k',
     0: '',  # Base unit (no prefix)
    -3: 'm', -6: 'µ', -9: 'n', -12: 'p', -15: 'f', -18: 'a', -21: 'z', -24: 'y'
}
# Sorted list of exponents for easier lookup
_SI_EXPONENTS = sorted(SI_PREFIXES.keys(), reverse=True) # Reverse is set to true since sorted() will put smallest as first

def get_si_prefix(value: float) -> Tuple[str, int, float]:
    """
    Determines the most appropriate SI prefix, exponent, and scaling factor for a given value.

    Chooses the prefix such that the scaled value falls within a reasonable range
    (typically 1 to 1000, but adjusted for standard prefixes).

    Args:
        value (float): The numerical value for which to find the prefix. Can be zero.

    Returns:
        Tuple[str, int, float]: A tuple containing:
            - prefix (str): The SI prefix symbol (e.g., 'k', 'm', 'µ', or '').
            - exponent (int): The corresponding power-of-10 exponent (e.g., 3, -3, -6, 0).
            - scale_factor (float): The factor needed to scale the original value
                                    (value / scale_factor = scaled_value). (10**exponent).
    """
    if np.isnan(value) or np.isinf(value) or value==0:
        return '', 0, 1.0 # No prefix for nan's and inf's
    
    # Use Decimal for potentially higher precision in edge cases
    d_value = decimal.Decimal(value)
    exponent = d_value.adjusted()

    # Find the largest SI exponent less than or equal to the value's exponent
    best_exponent = 0 # Default to base unit
    for si_exponent in _SI_EXPONENTS:
        if exponent >= si_exponent:
            best_exponent = si_exponent # Could be an issue for VERY LARGE numbers
            break
        # If value is smaller than the smallest prefix, use the smallest prefix
        best_exponent = _SI_EXPONENTS[-1]

    prefix = SI_PREFIXES[best_exponent]
    scale_factor = 10.0**best_exponent

    return prefix, best_exponent, scale_factor
        
def round_to_significant_figures(value: float, sig_figs: int) -> float:
    """
    Rounds a floating-point number to a specified number of significant figures.
    Handles positive, negative, and zero values correctly.
    Args:
        value (float): The number to round.
        sig_figs (int): The desired number of significant figures (must be > 0).
    Returns:
        float: The value rounded to the specified significant figures.
    Raises:
        ValueError: If sig_figs is not a positive integer.
        TypeError: If value cannot be converted to float or during log calculation.
        RuntimeError: For other unexpected errors.
    """

    if not isinstance(sig_figs, int) or sig_figs <= 0: # Check for meaningless sig figs
        raise ValueError("Number of significant figures must be a positive integer.")

    try:
        # Using Decimal(value) directly takes the binary float's actual value,
        # which might already have slight inaccuracies (e.g., float 0.1 is not exactly 0.1).
        # For ultimate precision from user input strings, Decimal(str(value)) is better.
        d_value = decimal.Decimal(str(value))
    except decimal.InvalidOperation:
        if np.isnan(value) or np.isinf(value) or value == 0.0: # Standard float NaN and Inf often can be converted to Decimal NaN/Inf.
            return float(value) # Return edge cases as they are
        raise TypeError(f"Cannot convert value '{value}' to Decimal for rounding.") from None

    if d_value.is_zero() or d_value.is_nan() or d_value.is_infinite(): # Decimal has its own representations for Zero, NaN, and Infinity.
        return float(d_value)

    try:
        # d_value.adjusted() returns the exponent of the leftmost digit.
        # It's like floor(log10(abs(value))) but works directly on the Decimal representation.
        # Examples:
        #   Decimal('123.45').adjusted() -> 2 (since 123.45 = 1.2345 * 10^2)
        #   Decimal('0.0123').adjusted() -> -2 (since 0.0123 = 1.23 * 10^-2)
        #   Decimal('-567').adjusted() -> 2 (since -567 = -5.67 * 10^2)
        order = d_value.adjusted()

        # Example (value=12345, sig_figs=3):
        #   order = 4 (from 10^4 place)
        #   We want to keep 3 digits (1, 2, 3). The last one (3) is at the 10^2 place.
        #   decimal_place_exponent = 4 - (3 - 1) = 4 - 2 = 2.
        # Example (value=0.01234, sig_figs=2):
        #   order = -2 (from 10^-2 place)
        #   We want to keep 2 digits (1, 2). The last one (2) is at the 10^-3 place.
        #   decimal_place_exponent = -2 - (2 - 1) = -2 - 1 = -3.
        decimal_place_exponent = order - (sig_figs - 1)

        # Example (decimal_place_exponent = 2):
        #   rounding_exponent = Decimal('1e2') which is Decimal('100')
        # Example (decimal_place_exponent = -3):
        #   rounding_exponent = Decimal('1e-3') which is Decimal('0.001')
        rounding_exponent = decimal.Decimal('1e' + str(decimal_place_exponent))

        # Example (d_value=Decimal('123.45'), rounding_exponent=Decimal('1e0') i.e. 1):
        #   quantize(Decimal('1')) rounds to the nearest integer -> Decimal('123')
        # Example (d_value=Decimal('123.45'), rounding_exponent=Decimal('1e-1') i.e. 0.1):
        #   quantize(Decimal('0.1')) rounds to the nearest tenth -> Decimal('123.5')
        # rounding=decimal.ROUND_HALF_UP: Specifies the rounding rule.
        #   ROUND_HALF_UP means round 5s away from zero (e.g., 2.5 -> 3, -2.5 -> -3).
        #   Other options exist like ROUND_HALF_EVEN (Python 3 default, rounds 5s to nearest *even* digit).
        rounded_d_value = d_value.quantize(rounding_exponent, rounding=decimal.ROUND_HALF_UP)

        # The function is defined to return a float, so convert the precisely
        # rounded Decimal value back into a standard Python float.
        # Note: This final conversion can potentially reintroduce tiny binary
        # representation inaccuracies if the rounded number isn't exactly
        # representable as a binary float, but the rounding itself was performed
        # with Decimal precision based on the significant figure rule.
        return float(rounded_d_value)

    except ValueError as e:
        raise TypeError(f"Cannot round value '{value}'. Ensure it's a valid number. Error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during rounding of {value} to {sig_figs} sf: {e}") from e

def _format_value_error_eng(value: float, error: float, unit_symbol: str = "", sig_figs_error: int = 1) -> str:
    """
    Formats a value-error pair using engineering notation with SI prefixes,
    rounding the value based on the error's significant figures.

    The error is rounded to `sig_figs_error` significant figures. The decimal
    place of the least significant digit of this rounded error then determines
    the decimal place to which the value is rounded. Both are presented with
    the SI prefix determined by the original value's magnitude.

    Args:
        value (float): The central value.
        error (float): The uncertainty (error) associated with the value.
        unit_symbol (str, optional): The unit symbol to append (e.g., "V", "Ω"). Defaults to "".
        sig_figs_error (int, optional): The number of significant figures to use
                                        for rounding the error. Defaults to 1. Must be > 0.

    Returns:
        str: A formatted string like "12.3 ± 0.2 kΩ" or "50.0 ± 0.0 mV".

    Raises:
        ValueError: If sig_figs_error is not a positive integer.

    Examples:
        >>> format_value_error_eng(12345, 567, "Hz", 1)
        '12.3 ± 0.6 kHz'
        >>> format_value_error_eng(12345, 567, "Hz", 2)
        '12.35 ± 0.57 kHz'
        >>> format_value_error_eng(0.0123, 0.00052, "V", 1)
        '12.3 ± 0.5 mV'
        >>> format_value_error_eng(0.0123, 0.00052, "V", 2)
        '12.30 ± 0.52 mV'
        >>> format_value_error_eng(50.0, 0.0, "mV")
        '50.0 ± 0.0 mV'
        >>> format_value_error_eng(100, 0.5, "m", 1)
        '100.0 ± 0.5 m'
        >>> format_value_error_eng(100, 7, "m", 1)
        '100 ± 7 m'
        >>> format_value_error_eng(100, 70, "m", 1) # Error rounded to 1sf is 70. Last digit place is 10^1.
        '100 ± 70 m' # Value rounded to 10^1 place is 100.
        >>> format_value_error_eng(99, 7, "m", 1) # Error rounded to 1sf is 7. Last digit place is 10^0.
        '99 ± 7 m' # Value rounded to 10^0 place is 99.
        >>> format_value_error_eng(1.2e-7, 3.4e-9, "A", 1) # Error 3e-9 -> 1sf = 3e-9. Place=10^-9.
        '120 ± 3 nA' # Value 1.2e-7 = 120e-9. Rounded to 10^-9 place = 120.
         >>> format_value_error_eng(1.2e-7, 3.4e-9, "A", 2) # Error 3.4e-9 -> 2sf = 3.4e-9. Place=10^-10.
        '120.0 ± 3.4 nA' # Value 1.2e-7 = 120.0e-9. Rounded to 10^-10 place = 120.0.
    """
    # --- Checks ---
    if not isinstance(sig_figs_error, int) or sig_figs_error <= 0:
        raise ValueError("Number of significant figures for error must be positive.")

    if np.isnan(value) or np.isnan(error) or np.isinf(value) or np.isinf(error): # Edge cases returned as they are
        return f"{value} ± {error} {unit_symbol}".strip() # Strip useful when unit_symbol is ""

    error = abs(error) # Ensure error is non-negative

    ref_value_for_prefix = value if value != 0.0 else (error if error != 0.0 else 1.0) # If value is zero we want to use the error, otherwise 1.0
    prefix, _, scale_factor = get_si_prefix(ref_value_for_prefix)

    # --- Scale Value and Error ---
    try:
        scaled_value = value / scale_factor
        scaled_error = error / scale_factor
    except OverflowError:
        warnings.warn("Overflow during value/error scaling. Using unscaled values.", RuntimeWarning)
        scaled_value = value
        scaled_error = error
        prefix = ""
        scale_factor = 1.0

    # --- Handle Zero Error Case ---
    if error == 0.0:
        temp_val_str = f"{scaled_value:.15g}"
        if '.' in temp_val_str:
            decimal_places = len(temp_val_str.split('.')[-1].split('e')[0]) # Handle potential sci notation from g format
        else:
            decimal_places = 0

        val_str = f"{scaled_value:.{decimal_places}f}" # Format value and error to the determined decimal places
        err_str = f"{0.0:.{decimal_places}f}" # Ensure error has same number of zeros

        return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()

    # --- Handle Non-Zero Error Case ---
    scaled_rounded_error = round_to_significant_figures(scaled_error, sig_figs_error)

    # Check if error rounded to zero (can happen if error is very small)
    if np.isclose(scaled_rounded_error, 0.0, atol=1e-18): # atol=1e-18 => RELATIVE error of 1/10^18
            # If error rounds to zero, treat similarly to the exact zero case,
            # but base the precision on the original unrounded scaled error's magnitude
            order_original_err = math.floor(math.log10(abs(scaled_error)))
            decimal_place_exponent = order_original_err - (sig_figs_error - 1)
            final_decimal_places = max(0, -decimal_place_exponent)

            # Round value to these decimal places
            scale_val_round = 10.0**final_decimal_places
            scaled_rounded_value = round(scaled_value * scale_val_round) / scale_val_round

            val_str = f"{scaled_rounded_value:.{final_decimal_places}f}"
            err_str = f"{0.0:.{final_decimal_places}f}" # Error is effectively zero at this precision

            return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()

    d_scaled_rounded_error = decimal.Decimal(str(scaled_rounded_error))
    order_rounded_err = d_scaled_rounded_error.adjusted() # Determine the decimal place of the least significant digit of the rounded error
    decimal_place_exponent = order_rounded_err - (sig_figs_error - 1) # The exponent corresponding to the place value of the last significant digit
    final_decimal_places = max(0, -decimal_place_exponent) # The number of decimal places needed is the negative of this exponent (if exponent is negative)
    scale_val_round = 10.0**final_decimal_places
    d_scaled_value = decimal.Decimal(str(scaled_value))
    rounding_exp_val = decimal.Decimal('1e' + str(-final_decimal_places))
    d_scaled_rounded_value = d_scaled_value.quantize(rounding_exp_val, rounding=decimal.ROUND_HALF_UP)
    scaled_rounded_value = float(d_scaled_rounded_value)

    val_str = f"{scaled_rounded_value:.{final_decimal_places}f}" # Format both using the calculated number of decimal places
    err_str = f"{scaled_rounded_error:.{final_decimal_places}f}"

    return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()

# --- END OF FILE utils.py ---