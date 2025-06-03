import numpy as np
import scipy
import qnmfits

from bgp_qnm_fits.utils import get_time_shift, sim_interpolator_data
from bgp_qnm_fits.gp_kernels import compute_kernel_matrix
from scipy.optimize import minimize


def get_residuals(sim_main, sim_lower, t0, T, dt=None):
    """

    Computes the residuals between two simulations by performing a time shift,
    masking the data, interpolating onto the same grid, then subtracting two waveform
    levels.

    Args:
        sim_main (qnmfits.Custom): The main level simulation object (typically L5, R2).
        sim_lower (qnmfits.Custom): The lower level simulation object (typically L4, R2).
        t0 (float): The start time of the analysis, where the residuals are computed.
        T (float): The duration of the analysis, from t0 to t0 + T.
        dt (float, optional): The time step for interpolation. If None, it will be calculated.
    Returns:
        dict: A dictionary containing the residuals for each mode, with keys as (ell, m) tuples.

    """

    # Perform a time shift

    time_shift = get_time_shift(sim_main, sim_lower, delta=0.0001, alpha=0.1, t0=-100, T=100)
    sim_lower.zero_time = -time_shift
    sim_lower.time_shift()

    # Mask data THEN interpolate onto the same grid

    analysis_mask_main = (sim_main.times >= t0 - 1e-9) & (sim_main.times < t0 + T - 1e-9)
    analysis_mask_lower = (sim_lower.times >= t0 - 1e-9) & (sim_lower.times < t0 + T - 1e-9)

    sim_main_masked = {key: sim_main.h[key][analysis_mask_main] for key in sim_main.h.keys()}
    sim_lower_masked = {key: sim_lower.h[key][analysis_mask_lower] for key in sim_lower.h.keys()}

    sim_main_times_masked = sim_main.times[analysis_mask_main]
    sim_lower_times_masked = sim_lower.times[analysis_mask_lower]

    if dt is None:
        dt = (
            sim_main_times_masked[np.argmin(np.abs(sim_main_times_masked - t0)) + 1]
            - sim_main_times_masked[np.argmin(np.abs(sim_main_times_masked - t0))]
        )

    new_times = np.arange(t0, t0 + T, dt)
    sim_main_interp = sim_interpolator_data(sim_main_masked, sim_main_times_masked, new_times)
    sim_lower_interp = sim_interpolator_data(sim_lower_masked, sim_lower_times_masked, new_times)

    return {key: sim_main_interp[key] - sim_lower_interp[key] for key in sim_main_interp.keys()}


def get_params(
    residual_dict,
    Mf,
    chif_mag,
    ringdown_start,
    smoothness,
    epsilon,
    spherical_modes=None,
):
    """
    This function computes the initial parameters for the Gaussian Process kernel based on the residuals.
    Args:
        residual_dict (dict): A dictionary containing the residuals for each spherical mode.
        Mf (float): The remnant mass of the black hole.
        chif_mag (float): The magnitude of the remnant dimensionless spin.
        ringdown_start (float): The start time of the ringdown phase (determined from mismatch curves in BH cartography).
        smoothness (float): The smoothness parameter for the kernel.
        epsilon (float): An artifact from an earlier version of the code, used to avoid numerical issues.
        spherical_modes (list, optional): A list of spherical modes to consider. If None, all modes in residual_dict are used.
    Returns:
        dict: A dictionary containing the parameters for each spherical mode, with keys as (ell, m) tuples. Some items in the
        dictionary are not used in the kernel, but are included as artifacts for testing.
    """

    if spherical_modes is None:
        spherical_modes = residual_dict.keys()

    param_dict_lm = {
        (ell, m): {
            "sigma_max": np.max(np.abs(R := residual_dict[(ell, m)])),
            "sigma_min": np.max(np.abs(R)) * epsilon,
            "t_s": ringdown_start,
            "smoothness": smoothness,
            "length_scale": -1 / (omega := qnmfits.qnm.omega(ell, m, 0, -1 if m < 0 else 1, chif_mag, Mf)).imag,
            "period": (2 * np.pi) / omega.real,
            # For the complicated kernel only
            "length_scale_2": -1 / omega.imag,
            "period_2": (2 * np.pi) / omega.real,
            "a": 0.5,
            # For jitter
            "jitter_scale": np.max(np.abs(R)),
        }
        for ell, m in spherical_modes
    }

    return param_dict_lm


def train_hyper_params(
    training_start_time,
    training_end_time,
    time_step,
    initial_params,
    bounds,
    param_dict,
    R_dict,
    hyperparam_rule_dict,
    kernel,
    training_modes,
    mode_rules,
):
    """
    Trains the hyperparameters for the Gaussian Process kernel by minimizing the negative log likelihood across chosen simulations
    and modes.

    Args:
        training_start_time (float): The start time for the training phase (must match the residual timespan).
        training_end_time (float): The end time for the training phase (must match the residual timespan).
        time_step (float): The time step for the analysis (must match the residual timestep).
        initial_params (list): Initial guess for the hyperparameters.
        bounds (list): Bounds for the hyperparameters.
        param_dict (dict): Dictionary containing initial parameters for each simulation and mode.
        R_dict (dict): Dictionary containing residuals for each simulation and mode.
        hyperparam_rule_dict (dict): Dictionary defining how to update hyperparameters.
        kernel (function): The kernel function to be used in the Gaussian Process.
        training_modes (list): List of modes to be considered for training.
        mode_rules (dict): Dictionary defining rules for modes inclusion for a given simulation.
    Returns:
        tuple: A tuple containing the optimized hyperparameters, log evidence, and tuned parameters for each simulation.

    """

    analysis_times = np.arange(
        training_start_time,
        training_start_time + training_end_time,
        time_step,
    )

    hyperparam_list, le = get_minimised_hyperparams(
        initial_params,
        bounds,
        param_dict,
        R_dict,
        hyperparam_rule_dict,
        analysis_times,
        kernel,
        training_modes,
        mode_rules,
    )

    print(
        "Optimal parameters:",
        dict(zip(hyperparam_rule_dict.keys(), hyperparam_list)),
        "Log evidence:",
        le,
    )

    print("Tuning parameters...")

    tuned_params_sim_lm = {}

    for sim_id in mode_rules.keys():
        tuned_params_sim_lm[sim_id] = get_tuned_params(
            param_dict[sim_id],
            hyperparam_list,
            hyperparam_rule_dict,
        )

    return hyperparam_list, le, tuned_params_sim_lm


def GP_log_likelihood(K, f):
    """
    Computes the log likelihood of a Gaussian Process with given hyperparameters.
    Args:
        K (ndarray): Covariance matrix of the Gaussian Process.
        f (ndarray): Residual values (observations) for which the log likelihood is computed.
    Returns:
        float: The log likelihood of the Gaussian Process.
    """
    _, logdet = np.linalg.slogdet(K)
    # Check for numerical instability
    try:
        # Check condition number
        eigvals = scipy.linalg.eigvalsh(K)
        condition_number = np.max(eigvals) / np.min(eigvals)
        
        # Check for negative eigenvalues or very small eigenvalues
        if np.any(eigvals <= 0):
            print(f"Warning: K has very small or negative eigenvalues: {min(eigvals)}")
            
        if condition_number > 1e8:
            print(f"Warning: K is ill-conditioned with condition number {condition_number}")
    except:
        print("Error: Numerical instability detected in covariance matrix K")
    return -0.5 * (np.dot(f, scipy.linalg.solve(K, f, assume_a="pos")) + logdet + len(f) * np.log(2 * np.pi))


def get_new_params(param_dict, hyperparam_list, rule_dict):
    """
    Updates the parameters based on the hyperparameter list and the rules defined in rule_dict.
    Args:
        param_dict (dict): Dictionary containing the initial parameters for the Gaussian Process.
        hyperparam_list (list): List of hyperparameters to be applied to the initial parameters.
        rule_dict (dict): Dictionary defining how each parameter should be updated (e.g., "multiply", "sum", "replace").
    Returns:
        dict: A new dictionary containing the updated parameters based on the hyperparameter list and rules.
    """
    new_params = {}
    for i, (param, rule) in enumerate(rule_dict.items()):
        if rule == "multiply":
            new_params[param] = param_dict[param] * hyperparam_list[i]
        elif rule == "sum":
            new_params[param] = param_dict[param] + hyperparam_list[i]
        elif rule == "replace":
            new_params[param] = hyperparam_list[i]

    # new_params["sigma_min"] = param_dict["sigma_min"] * hyperparam_list[list(rule_dict.keys()).index("sigma_max")]
    new_params.update({param: param_dict[param] for param in param_dict if param not in new_params})

    return new_params


def prior_logpdf(s, mu=np.log(0.01), sigma=1.0):
  return -0.5*((np.log(s)-mu)/sigma)**2 - np.log(s*sigma*np.sqrt(2*np.pi))


def get_total_log_likelihood(
    hyperparam_list,
    param_dict_sim_lm,
    f_dict_sim_lm,
    rule_dict,
    analysis_times,
    kernel,
    spherical_modes,
    mode_rules,
    alpha=2,
    beta=2,
):
    """
    Computes the total log likelihood for a Gaussian Process across multiple simulations and spherical modes.
    Args:
        hyperparam_list (list): List of hyperparameters to be applied to the Gaussian Process initial parameters.
        param_dict_sim_lm (dict): Dictionary containing initial parameters for each simulation and mode.
        f_dict_sim_lm (dict): Dictionary containing residuals for each simulation and mode.
        rule_dict (dict): Dictionary defining how to update hyperparameters.
        analysis_times (ndarray): Array of times at which the analysis is performed.
        kernel (function): The kernel function used in the Gaussian Process.
        spherical_modes (list): List of spherical modes to be considered.
        mode_rules (dict): Dictionary defining rules for modes inclusion for a given simulation.
        alpha (float, optional): Hyperparameter for the Beta distribution prior on a. Default is 2.
        beta (float, optional): Hyperparameter for the Beta distribution prior on a. Default is 2.
    Returns:
        float: The negative total log likelihood of the Gaussian Process across all simulations and modes.

    """

    mode_filters = {
        "PE": lambda mode: mode[1] >= 0 and mode[1] % 2 == 0,
        "P": lambda mode: mode[1] >= 0,
        "E": lambda mode: mode[1] % 2 == 0,
        "ALL": lambda mode: True,
    }

    total_log_likelihood = 0

    for sim_id, mode_rule in mode_rules.items():

        spherical_mode_choice = [mode for mode in spherical_modes if mode_filters[mode_rule](mode)]

        for mode in spherical_mode_choice:
            param_dict = param_dict_sim_lm[sim_id][mode]
            new_param_dict = get_new_params(param_dict, hyperparam_list, rule_dict)
            K = compute_kernel_matrix(analysis_times, new_param_dict, kernel)
            f = f_dict_sim_lm[sim_id][mode]
            log_likelihood_real = GP_log_likelihood(K, f.real)
            log_likelihood_imag = GP_log_likelihood(K, f.imag)
            total_log_likelihood += log_likelihood_real + log_likelihood_imag
            if mode_rule in {"P", "PE"}:
                total_log_likelihood += log_likelihood_real + log_likelihood_imag

    if "a" in rule_dict.keys():
        a_hyperparam_index = list(rule_dict.keys()).index("a")
        current_a_value = hyperparam_list[a_hyperparam_index]
        total_log_likelihood += (alpha - 1) * np.log(current_a_value) + (beta - 1) * np.log(1 - current_a_value)

    if "smoothness" in rule_dict.keys():
        smoothness_hyperparam_index = list(rule_dict.keys()).index("smoothness")
        current_smoothness_value = hyperparam_list[smoothness_hyperparam_index]
        total_log_likelihood += prior_logpdf(current_smoothness_value)

    return -total_log_likelihood


def get_minimised_hyperparams(
    initial_params,
    bounds,
    param_dict_sim_lm,
    R_dict_sim_lm,
    hyperparam_rule_dict,
    analysis_times,
    kernel,
    spherical_modes,
    mode_rules,
):
    """
    Minimizes the negative log likelihood of a Gaussian Process across multiple simulations and spherical modes to find the optimal hyperparameters.
    Args:
        initial_params (list): Initial guess for the hyperparameters.
        bounds (list): Bounds for the hyperparameters.
        param_dict_sim_lm (dict): Dictionary containing initial parameters for each simulation and mode.
        R_dict_sim_lm (dict): Dictionary containing residuals for each simulation and mode.
        hyperparam_rule_dict (dict): Dictionary defining how to update hyperparameters.
        analysis_times (ndarray): Array of times at which the analysis is performed.
        kernel (function): The kernel function used in the Gaussian Process.
        spherical_modes (list): List of spherical modes to be considered.
        mode_rules (dict): Dictionary defining rules for modes inclusion for a given simulation.
    Returns:
        tuple: A tuple containing the optimized hyperparameters and the log evidence of the Gaussian Process.
    """
    args = (
        param_dict_sim_lm,
        R_dict_sim_lm,
        hyperparam_rule_dict,
        analysis_times,
        kernel,
        spherical_modes,
        mode_rules,
    )

    result = minimize(
        get_total_log_likelihood,
        initial_params,
        args=args,
        method="Nelder-Mead",
        bounds=bounds,
    )
    return result.x, result.fun


def get_tuned_params(param_dict_lm, hyperparams, hyperparam_rule_dict, spherical_modes=None):
    """
    Returns a dictionary of tuned parameters for each spherical mode based on the hyperparameters, the initial parameters,
    and rules defined in hyperparam_rule_dict.
    Args:
        param_dict_lm (dict): Dictionary containing initial parameters for each spherical mode.
        hyperparams (list): List of hyperparameters to be applied to the initial parameters.
        hyperparam_rule_dict (dict): Dictionary defining how to update hyperparameters.
        spherical_modes (list, optional): List of spherical modes to be considered. If None, all modes in param_dict_lm are used.
    Returns:
        dict: A dictionary containing the tuned parameters for each spherical mode, with keys as (ell, m) tuples.
    """
    if spherical_modes is None:
        spherical_modes = param_dict_lm.keys()
    return {mode: get_new_params(param_dict_lm[mode], hyperparams, hyperparam_rule_dict) for mode in spherical_modes}


def kl_divergence(p, q):
    dim = p.shape[0]
    trace_term = np.trace(np.linalg.solve(q, p))
    log_det_p = np.linalg.slogdet(p)[1]
    log_det_q = np.linalg.slogdet(q)[1]
    kl_div = 0.5 * (trace_term - dim + log_det_q - log_det_p)
    return kl_div


def js_divergence(p, q):
    """
    The symmetric Kullback-Leibler divergence between two probability distributions p and q.
    """
    return kl_divergence(p, q) / 2 + kl_divergence(q, p) / 2