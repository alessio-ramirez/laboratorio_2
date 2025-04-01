# --- START OF FILE stats.py ---
"""
Statistical Analysis Tools for Measurements

This module provides functions for common statistical tasks involving
Measurement objects, such as testing compatibility and calculating weighted means.
"""

import numpy as np
from scipy import stats as scistats
from typing import Tuple, Union, List, Dict, Optional, Any
import warnings

from .measurement import Measurement

def test_comp(m1: Union[Measurement, Tuple[float, float]],
              m2: Union[Measurement, Tuple[float, float]],
              alpha: float = 0.05,
              assume_correlated: bool = False) -> Dict[str, Any]:
    """
    Compare two measurements for compatibility using a Z-test.

    Theory:
        This function tests the null hypothesis that two measurements, m1 and m2,
        represent the same underlying true value. It calculates the difference
        d = m1 - m2 and its uncertainty σ_d. Assuming the uncertainties σ1 and σ2
        are standard deviations and the measurements are normally distributed,
        the difference d is also normally distributed.

        The test statistic is the Z-score: Z = |d| / σ_d.
        This score measures how many standard deviations the observed difference
        is away from zero (the expected difference under the null hypothesis).

        If m1 and m2 are independent, the uncertainty of the difference is
        σ_d = sqrt(σ1^2 + σ2^2). This is handled automatically by Measurement
        subtraction.

        If the measurements might be negatively correlated (e.g., derived from
        the same underlying data in opposing ways), `assume_correlated=True`
        provides a more conservative test. It uses the maximum possible error
        for the difference, which occurs for perfect negative correlation (corr=-1):
        σ_d(corr=-1) = sqrt(σ1^2 + σ2^2 - 2*(-1)*σ1*σ2) = sqrt((σ1+σ2)^2) = σ1 + σ2.
        This increases σ_d, decreases Z, and makes it harder to reject the null
        hypothesis (i.e., more likely to find them compatible). Note: A proper
        treatment requires the actual covariance.

        The p-value is the probability of observing a difference as large as |d|
        (or larger) if the null hypothesis were true. A small p-value (< alpha)
        suggests the observed difference is statistically significant, and the
        measurements are considered incompatible at the chosen significance level.

    Args:
        m1: First measurement (Measurement object or (value, error) tuple).
        m2: Second measurement (Measurement object or (value, error) tuple).
        alpha: Significance level (Type I error rate). Common values are 0.05 (5%)
               or 0.01 (1%). The probability of rejecting the null hypothesis
               when it is actually true. (Default 0.05).
        assume_correlated: If True, assumes maximum negative correlation,
                           giving a lower bound on compatibility (more conservative).
                           (Default False, assumes independence).

    Returns:
        Dict containing:
        - 'difference': Value of m1 - m2 (Measurement object).
        - 'z_score': Compatibility score Z = |m1 - m2| / σ_diff.
        - 'p_value': Two-tailed p-value for the Z-test. Probability of observing
                     a Z-score this large or larger if the measurements were compatible.
        - 'compatible': Boolean, True if p_value >= alpha.
        - 'alpha': Significance level used.
        - 'interpretation': String summarizing the compatibility result.

    Raises:
        TypeError: If m1 or m2 are not Measurement objects or (value, error) tuples.
        ValueError: If uncertainties are negative.
    """
    # Input processing and validation
    if isinstance(m1, Measurement):
        v1, s1 = m1.value, m1.error
        meas1 = m1 # Keep original Measurement object
    elif isinstance(m1, (tuple, list)) and len(m1) == 2:
        v1, s1 = float(m1[0]), float(m1[1])
        meas1 = Measurement(v1, s1) # Convert for difference calculation
    else:
        raise TypeError("m1 must be a Measurement or a (value, error) tuple/list")

    if isinstance(m2, Measurement):
        v2, s2 = m2.value, m2.error
        meas2 = m2 # Keep original Measurement object
    elif isinstance(m2, (tuple, list)) and len(m2) == 2:
        v2, s2 = float(m2[0]), float(m2[1])
        meas2 = Measurement(v2, s2) # Convert for difference calculation
    else:
        raise TypeError("m2 must be a Measurement or a (value, error) tuple/list")

    if s1 < 0 or s2 < 0:
        raise ValueError("Uncertainties must be non-negative.")

    # Calculate difference using Measurement subtraction for error propagation
    # By default, Measurement subtraction assumes independence: sigma_diff^2 = s1^2 + s2^2
    diff_meas = meas1 - meas2
    diff_val = diff_meas.value
    diff_err_independent = diff_meas.error

    # Use the appropriate error for the Z-test based on correlation assumption
    if assume_correlated:
        # Recalculate error assuming max negative correlation (corr = -1)
        # sigma_diff^2 = s1^2 + s2^2 - 2*corr*s1*s2
        # sigma_diff^2 = s1^2 + s2^2 + 2*s1*s2 = (s1 + s2)^2
        diff_err = s1 + s2
        warnings.warn("Assuming maximum negative correlation (corr=-1) for compatibility test. "
                      "This provides a conservative estimate (lower Z-score). "
                      "The calculated 'difference' Measurement object still uses independent errors.", UserWarning)
    else:
        # Assume independence
        diff_err = diff_err_independent
        # If independence was explicitly assumed and errors add up, the diff_meas.error is correct.

    # Calculate Z-score and p-value
    # Handle potential division by zero if errors are zero
    if np.isclose(diff_err, 0.0):
        # If error is zero, measurements are only compatible if values are identical
        is_identical = np.isclose(diff_val, 0.0)
        z_score = 0.0 if is_identical else np.inf
        p_value = 1.0 if is_identical else 0.0
    else:
        # Standard Z-test calculation
        z_score = np.abs(diff_val / diff_err)
        # p-value from standard normal CDF (two-tailed test)
        p_value = 2 * (1.0 - scistats.norm.cdf(z_score))

    # Determine compatibility based on p-value and alpha
    compatible = p_value >= alpha

    # Generate interpretation string
    corr_assumption_str = " (assuming max negative correlation)" if assume_correlated else " (assuming independence)"
    interp = f"Measurements are {'compatible' if compatible else 'NOT compatible'} " \
             f"at alpha={alpha:.3f}{corr_assumption_str}. " \
             f"(Z={z_score:.2f}, p={p_value:.3g})"

    return {
        'difference': diff_meas, # Note: This difference Measurement assumes independence
        'z_score': z_score,
        'p_value': p_value,
        'compatible': compatible,
        'alpha': alpha,
        'interpretation': interp
    }


def weighted_mean(measurements: List[Measurement]) -> Measurement:
    """
    Calculates the weighted mean of a list of Measurement objects.

    Theory:
        When combining multiple measurements (x_i ± σ_i) of the same quantity,
        the optimal way to combine them (yielding the minimum uncertainty on the
        result) is using a weighted mean. The weight for each measurement (w_i)
        should be inversely proportional to its variance (σ_i^2).
            w_i = 1 / σ_i^2

        The weighted mean value (x̄_w) is calculated as:
            x̄_w = Σ(w_i * x_i) / Σ(w_i)

        The uncertainty (standard deviation) of the weighted mean (σ_x̄_w) is:
            σ_x̄_w^2 = 1 / Σ(w_i)
            σ_x̄_w = sqrt(1 / Σ(w_i))

        This method assumes the measurements are independent. If they are
        correlated, a more complex calculation involving the covariance matrix
        is required.

        Special Handling:
        - Measurements with zero uncertainty (σ_i = 0) have infinite weight.
          If one or more such measurements exist and are consistent (have the
          same value), the weighted mean is simply that value with zero uncertainty.
          If they are inconsistent, it indicates a problem with the input data.

    Args:
        measurements: A list of Measurement objects. They should ideally represent
                      the same physical quantity with compatible units.

    Returns:
        Measurement: The weighted mean and its uncertainty.
                     The unit is preserved only if all input measurements have the
                     exact same unit string. Otherwise, the unit is cleared.

    Raises:
        ValueError: If the input list is empty, if any uncertainty is negative,
                    or if inconsistent measurements with zero error are found.
    """
    if not measurements:
        raise ValueError("Input list of measurements cannot be empty.")
    if not all(isinstance(m, Measurement) for m in measurements):
         raise TypeError("All items in the input list must be Measurement objects.")

    # Extract values and errors, ensuring they are flat arrays
    values = np.array([m.value for m in measurements]).flatten()
    errors = np.array([m.error for m in measurements]).flatten()

    if len(values) != len(errors):
         # This should typically not happen if inputs are Measurements, but safeguard
         raise ValueError("Internal shape mismatch between values and errors after flattening.")

    if np.any(errors < 0):
        raise ValueError("Uncertainties must be non-negative.")

    # Handle measurements with zero error (infinite weight)
    zero_error_indices = np.where(np.isclose(errors, 0.0))[0]
    if len(zero_error_indices) > 0:
        first_zero_idx = zero_error_indices[0]
        # Check if all zero-error values are consistent
        if not np.allclose(values[zero_error_indices], values[first_zero_idx]):
            raise ValueError("Inconsistent measurements with zero error found. "
                             "Cannot compute a meaningful weighted mean.")
        # If consistent, the result is the value of the first zero-error measurement
        # Preserve unit/name from that specific measurement
        origin_meas = measurements[first_zero_idx]
        res_unit = origin_meas.unit
        res_name = f"Weighted Mean ({origin_meas.name})" if origin_meas.name else "Weighted Mean"
        warnings.warn("Found measurements with zero error. Result is taken directly from these measurements.", UserWarning)
        return Measurement(values[first_zero_idx], 0.0, unit=res_unit, name=res_name)

    # Standard calculation for non-zero errors
    # Calculate weights (handle potential division by zero if errors very small but not zero)
    variances = errors**2
    # Avoid division by zero in weights calculation for safety, though zero errors handled above
    variances = np.where(np.isclose(variances, 0.0), np.finfo(float).eps, variances)
    weights = 1.0 / variances

    # Calculate weighted mean value
    weighted_sum = np.sum(weights * values)
    sum_of_weights = np.sum(weights)
    mean_value = weighted_sum / sum_of_weights

    # Calculate uncertainty of the weighted mean
    mean_variance = 1.0 / sum_of_weights
    mean_error = np.sqrt(mean_variance)

    # Handle units - preserve only if all inputs have the same, non-empty unit
    first_unit = measurements[0].unit
    all_units_same = all(m.unit == first_unit for m in measurements)
    res_unit = first_unit if (first_unit and all_units_same) else ""
    if first_unit and not all_units_same:
         warnings.warn("Input measurements have different units. Resulting unit is cleared.", UserWarning)

    # Construct a generic name
    res_name = "Weighted Mean"

    return Measurement(mean_value, mean_error, unit=res_unit, name=res_name)

# --- END OF FILE stats.py ---