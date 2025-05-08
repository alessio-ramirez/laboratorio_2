# run_interactive_fit.py
import numpy as np
from burger_lib.guezzi import *
# 1. Define a model function (Python version)
def gaussian_model(x, amp, mu, sigma, offset):
    """A simple Gaussian model."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

# 2. Define the corresponding JavaScript model function string
#    It MUST be named "modelFunction" and have parameters in the same order.
js_gaussian_model_str = """
function modelFunction(x, amp, mu, sigma, offset) {
    // Ensure parameters are numbers; they might be strings from sliders initially
    amp = parseFloat(amp);
    mu = parseFloat(mu);
    sigma = parseFloat(sigma);
    offset = parseFloat(offset);

    if (Math.abs(sigma) < 1e-9) { // Avoid division by zero or extremely small sigma
        return offset; // Or handle as appropriate, e.g., return amp + offset for a delta function
    }
    return amp * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) + offset;
}
"""

# Example for a linear model:
# def linear_model_py(x, m, c):
# return m * x + c
# js_linear_model_str = """
# function modelFunction(x, m, c) {
#     m = parseFloat(m);
#     c = parseFloat(c);
#     return m * x + c;
# }"""


def main():
    # 3. Generate some sample data
    np.random.seed(42)
    x_true = np.linspace(-5, 5, 50)
    
    # True parameters for data generation
    params_true_dict = {'amp': 5.0, 'mu': 0.5, 'sigma': 1.0, 'offset': 1.0}
    
    y_true = gaussian_model(x_true, 
                            params_true_dict['amp'], 
                            params_true_dict['mu'], 
                            params_true_dict['sigma'], 
                            params_true_dict['offset'])
    
    y_noise_level = 0.5
    y_noise = y_noise_level * np.random.normal(size=x_true.size)
    y_measured_vals = y_true + y_noise
    y_errors_vals = np.full_like(y_measured_vals, y_noise_level) # Constant error

    x_m = Measurement(x_true, name="X Values (units)")
    y_m = Measurement(y_measured_vals, y_errors_vals, name="Y Values (units)")

    # 4. Perform the fit
    # Initial guess for parameters (amp, mu, sigma, offset)
    initial_guess_list = [4.0, 0.0, 0.8, 0.5] 
    # Parameter names MUST match the order in gaussian_model (after x) and initial_guess_list
    param_names_list = ['amp', 'mu', 'sigma', 'offset']

    print("Starting fit...")
    fit_result: FitResult = perform_fit(
        x_data=x_m,
        y_data=y_m,
        func=gaussian_model,
        p0=initial_guess_list,
        parameter_names=param_names_list, 
        method='curve_fit', # or 'minuit' if iminuit is installed and preferred
        # Example of passing kwargs to curve_fit:
        # maxfev=10000 
    )

    print("\nFit Result from Python:")
    print(fit_result) # Uses FitResult.__str__

    if fit_result is not None and fit_result.success:
        # 5. Generate the interactive HTML report
        generate_interactive_report(
            fit_result=fit_result,
            js_model_function_str=js_gaussian_model_str,
            output_html_path="gaussian_fit_report.html",
            plot_title="Interactive Gaussian Fit Explorer"
        )
    elif fit_result is not None:
        print("\nFit was not successful. Cannot generate full interactive report.")
        # Optionally, you could still generate a basic plot of data if fit_result.x_data etc. are populated
    else:
        print("\nFit failed catastrophically (result is None). No report generated.")


if __name__ == "__main__":
    main()