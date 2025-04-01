# --- START OF FILE utils.py ---
"""
Internal Utility Functions for Guezzi Library

Contains helper functions for tasks like SI prefix handling and value/error formatting,
primarily used by other modules within the library (Measurement, tables, plotting).
"""

import numpy as np
import math
from typing import Tuple

# Dictionary mapping exponents (powers of 10) to SI prefixes
# Covers standard prefixes from Yotta (10^24) to yocto (10^-24)
SI_PREFIXES = {
    24: 'Y', 21: 'Z', 18: 'E', 15: 'P', 12: 'T', 9: 'G', 6: 'M', 3: 'k',
     0: '',  # Base unit (no prefix)
    -3: 'm', -6: 'µ', -9: 'n', -12: 'p', -15: 'f', -18: 'a', -21: 'z', -24: 'y'
}
# Sorted list of exponents for easier lookup
_SI_EXPONENTS = sorted(SI_PREFIXES.keys(), reverse=True)

def get_si_prefix(value: float) -> Tuple[str, int, float]:
    """
    Determines the most appropriate SI prefix, exponent, and scaling factor for a given value.

    Chooses the prefix such that the scaled value falls within a reasonable range
    (typically 1 to 1000, but adjusted for standard prefixes).

    Args:
        value (float): The numerical value for which to find the prefix.

    Returns:
        Tuple[str, int, float]: A tuple containing:
            - prefix (str): The SI prefix symbol (e.g., 'k', 'm', 'µ', or '').
            - exponent (int): The corresponding power-of-10 exponent (e.g., 3, -3, -6, 0).
            - scale_factor (float): The factor needed to scale the original value
                                    (value / scale_factor = scaled_value). (10**exponent).
    """
    if np.isclose(value, 0.0) or np.isnan(value) or np.isinf(value):
        return '', 0, 1.0 # No prefix for zero, nan, inf

    exponent = math.floor(math.log10(abs(value)))

    # Find the largest SI exponent less than or equal to the value's exponent
    best_exponent = 0 # Default to base unit
    for si_exponent in _SI_EXPONENTS:
        if exponent >= si_exponent:
            best_exponent = si_exponent
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
        TypeError: If value cannot be converted to float.
    """
    if not isinstance(sig_figs, int) or sig_figs <= 0:
        raise ValueError("Number of significant figures must be a positive integer.")
    if value == 0 or np.isnan(value) or np.isinf(value):
        return float(value) # Return 0, nan, inf as is

    try:
        # Determine the order of magnitude of the most significant digit
        # Use log10, handle potential errors for non-positive values (already checked for 0)
        order = math.floor(math.log10(abs(value)))

        # Calculate the decimal place to round to.
        # Example: value=12345, sig_figs=3. order=4. place = -(4 - (3-1)) = -2. Round to 10^2 = 100s place.
        # Example: value=0.01234, sig_figs=2. order=-2. place = -(-2 - (2-1)) = 3. Round to 10^-3 = 0.001 place.
        decimal_place_exponent = order - (sig_figs - 1)
        scale = 10.0**(-decimal_place_exponent)

        # Round the scaled value to the nearest integer, then scale back
        rounded_value = round(value * scale) / scale
        return rounded_value
    except ValueError as e: # Catch potential math domain errors if not caught earlier
         raise TypeError(f"Cannot round value '{value}'. Ensure it's a valid number. Error: {e}")
    except Exception as e: # Catch other potential issues
         raise RuntimeError(f"Unexpected error during rounding of {value} to {sig_figs} sf: {e}")


def _format_value_error_eng(value: float, error: float, unit_symbol: str = "", sig_figs_error: int = 1) -> str:
    """
    Formats a value-error pair using engineering notation with SI prefixes (Revised Logic).

    This function aims to produce a standard representation like "1.23 ± 0.05 mV".
    1. The error is rounded to `sig_figs_error` significant figures.
    2. An appropriate SI prefix is chosen based on the *value*.
    3. Both the value and the rounded error are scaled using this SI prefix.
    4. The scaled value is rounded to the same decimal place as the least significant
       digit of the *scaled rounded error*.
    5. The final string is constructed.

    Args:
        value (float): Nominal value.
        error (float): Uncertainty (standard deviation).
        unit_symbol (str): The base unit symbol (e.g., "V", "m"). (Default "").
        sig_figs_error (int): Number of significant figures for the error (typically 1 or 2).
                              (Default 1).

    Returns:
        str: Formatted string like "1.23 ± 0.05 mV".

    Raises:
        ValueError: If sig_figs_error is not positive.
    """
    if not isinstance(sig_figs_error, int) or sig_figs_error <= 0:
        raise ValueError("Number of significant figures for error must be positive.")

    # --- Handle Special Cases ---
    if np.isnan(value) or np.isnan(error): return "NaN ± NaN" + (f" {unit_symbol}" if unit_symbol else "")
    if np.isinf(value): return f"{value} ± {error}" + (f" {unit_symbol}" if unit_symbol else "") # Let inf value dominate
    if np.isinf(error): return f"{value} ± {error}" + (f" {unit_symbol}" if unit_symbol else "") # Error is infinite
    if error < 0: error = abs(error) # Ensure error is non-negative

    # --- Case 1: Error is Zero (or effectively zero) ---
    if np.isclose(error, 0.0):
        prefix, _, scale_factor = get_si_prefix(value)
        scaled_value = value / scale_factor
        # Format zero error value with reasonable precision (~6 sig figs default)
        if np.isclose(scaled_value, 0.0):
             val_str = "0" # Simply "0" if value is also zero
             err_str = "0"
             # Determine decimal places needed based on a small number if value isn't exactly 0? No, keep simple.
             final_decimal_places = 0
             if not np.isclose(value, 0.0): # If value was non-zero but scaled to ~0
                  # Try formatting with general precision
                   temp_fmt = f"{scaled_value:.6g}"
                   if '.' in temp_fmt: final_decimal_places = len(temp_fmt.split('.')[-1])
                   else: final_decimal_places = 0 # Treat as integer if no decimal part
             val_str = f"{scaled_value:.{final_decimal_places}f}"
             err_str = f"{0.0:.{final_decimal_places}f}" # Match value's decimals
        else:
             # Heuristic: Show ~6 sig figs for non-zero value with zero error
             order_scaled = math.floor(math.log10(abs(scaled_value)))
             final_decimal_places = max(0, 5 - int(order_scaled)) # Aim for ~6 digits total
             val_str = f"{scaled_value:.{final_decimal_places}f}"
             err_str = f"{0.0:.{final_decimal_places}f}"

        return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()


    # --- Case 2: Error is Non-Zero ---
    # 1. Choose SI prefix based on the *value*'s magnitude
    prefix, eng_exponent, scale_factor = get_si_prefix(value)

    # 2. Scale value and error by this factor
    scaled_value = value / scale_factor
    scaled_error = error / scale_factor

    # 3. Round the *scaled error* to `sig_figs_error` significant figures
    scaled_rounded_error = round_to_significant_figures(scaled_error, sig_figs_error)

    # Check if scaled error rounded to zero (can happen if error was very small)
    if np.isclose(scaled_rounded_error, 0.0):
        # Treat as zero-error case, but use the chosen prefix
        # Format scaled value reasonably, error is 0
        if np.isclose(scaled_value, 0.0):
            val_str = "0"
            err_str = "0"
            dp = 0
        else:
            order_scaled = math.floor(math.log10(abs(scaled_value)))
            dp = max(0, 5 - int(order_scaled)) # Aim for ~6 digits total
            val_str = f"{scaled_value:.{dp}f}"
            err_str = f"{0.0:.{dp}f}"
        return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()


    # 4. Determine the decimal place of the least significant digit of the *scaled rounded error*
    order_scaled_rounded_err = math.floor(math.log10(abs(scaled_rounded_error)))
    # This is the number of digits to keep *after* the decimal point
    final_decimal_places = max(0, -int(order_scaled_rounded_err - (sig_figs_error - 1)))

    # 5. Round the *scaled value* to this same decimal place
    scale_value_round = 10.0**final_decimal_places
    scaled_rounded_value = round(scaled_value * scale_value_round) / scale_value_round

    # 6. Format the scaled rounded value and scaled rounded error to these decimal places
    val_str = f"{scaled_rounded_value:.{final_decimal_places}f}"
    err_str = f"{scaled_rounded_error:.{final_decimal_places}f}"

    # 7. Construct the final string
    return f"{val_str} ± {err_str} {prefix}{unit_symbol}".strip()


# --- END OF FILE utils.py ---