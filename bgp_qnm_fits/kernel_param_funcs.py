import numpy as np
import scipy
import time
import qnmfits

from bgp_qnm_fits.utils import get_time_shift, sim_interpolator
from bgp_qnm_fits.GP_funcs import compute_kernel_matrix
from scipy.optimize import minimize


def get_residuals(sim_main, sim_lower, t0, T, dt=None):

    # Perform a time shift

    time_shift = get_time_shift(sim_main, sim_lower, delta=0.0001, alpha=0.1, t0=-100, T=100)
    sim_lower.zero_time = -time_shift
    sim_lower.time_shift()

    # Interpolate onto the same grids

    if dt is None:
        dt = (
            sim_main.times[np.argmin(np.abs(sim_main.times - t0)) + 1]
            - sim_main.times[np.argmin(np.abs(sim_main.times - t0))]
        )

    new_times = np.arange(sim_main.times[0], sim_main.times[-1], dt)
    sim_main_interp = sim_interpolator(sim_main, new_times)
    sim_lower_interp = sim_interpolator(sim_lower, new_times)

    # Mask the data around the region of interest for param estimation and training

    analysis_mask = (sim_main_interp.times >= t0 - 1e-9) & (sim_main_interp.times < t0 + T - 1e-9)

    #TODO this is currently running from -10 to 90 M not 100! 

    return {
        key: sim_main_interp.h[key][analysis_mask] - sim_lower_interp.h[key][analysis_mask]
        for key in sim_main_interp.h.keys()
    }


def get_params(
    residual_dict,
    Mf,
    chif_mag,
    ringdown_start,
    smoothness,
    epsilon,
    spherical_modes=None,
):
    if spherical_modes is None:
        spherical_modes = residual_dict.keys()

    param_dict_lm = {
        (ell, m): {
            "sigma_max": np.max(np.abs(R := residual_dict[(ell, m)])),
            "sigma_min": np.max(np.abs(R)) * epsilon,
            "t_s": ringdown_start,
            "sharpness": smoothness,
            "length_scale": -1 / (omega := qnmfits.qnm.omega(ell, m, 0, -1 if m < 0 else 1, chif_mag, Mf)).imag,
            "period": (2 * np.pi) / omega.real,
            # For the complicated kernel only
            "length_scale_2": -1 / omega.imag,
            "period_2": (2 * np.pi) / omega.real,
            "a": 0.5,
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
    This trains on the chosen data and modes to get the
    hyperparameters and the tuned parameters.

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


def log_evidence(K, f):
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * (np.dot(f, scipy.linalg.solve(K, f, assume_a="pos")) + logdet + len(f) * np.log(2 * np.pi)) 


def get_new_params(param_dict, hyperparam_list, rule_dict):
    new_params = {}
    for i, (param, rule) in enumerate(rule_dict.items()):
        if rule == "multiply":
            new_params[param] = param_dict[param] * hyperparam_list[i]
        elif rule == "sum":
            new_params[param] = param_dict[param] + hyperparam_list[i]
        elif rule == "replace":
            new_params[param] = hyperparam_list[i]

    #new_params["sigma_min"] = param_dict["sigma_min"] * hyperparam_list[list(rule_dict.keys()).index("sigma_max")]
    new_params.update({param: param_dict[param] for param in param_dict if param not in new_params})

    return new_params


def get_total_log_evidence(
    hyperparam_list,
    param_dict_sim_lm,
    f_dict_sim_lm,
    rule_dict,
    analysis_times,
    kernel,
    spherical_modes,
    mode_rules,
    alpha=2,
    beta=2
):

    mode_filters = {
        "PE": lambda mode: mode[1] >= 0 and mode[1] % 2 == 0,
        "P": lambda mode: mode[1] >= 0,
        "E": lambda mode: mode[1] % 2 == 0,
        "ALL": lambda mode: True,
    }

    total_log_evidence = 0

    for sim_id, mode_rule in mode_rules.items():

        spherical_mode_choice = [mode for mode in spherical_modes if mode_filters[mode_rule](mode)]

        for mode in spherical_mode_choice:
            param_dict = param_dict_sim_lm[sim_id][mode]
            new_param_dict = get_new_params(param_dict, hyperparam_list, rule_dict)
            K = compute_kernel_matrix(analysis_times, new_param_dict, kernel)
            f = f_dict_sim_lm[sim_id][mode]
            log_evidence_real = log_evidence(K, f.real)
            log_evidence_imag = log_evidence(K, f.imag)
            total_log_evidence += log_evidence_real + log_evidence_imag
            if mode_rule in {"P", "PE"}:
                total_log_evidence += log_evidence_real + log_evidence_imag

    if "a" in rule_dict.keys():
        a_hyperparam_index = list(rule_dict.keys()).index("a")
        current_a_value = hyperparam_list[a_hyperparam_index]
        total_log_evidence += (alpha - 1) * np.log(current_a_value) + (beta - 1) * np.log(1 - current_a_value) 

    return -total_log_evidence


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
        get_total_log_evidence,
        initial_params,
        args=args,
        method="Nelder-Mead",
        bounds=bounds,
    )
    return result.x, result.fun


def get_tuned_params(param_dict_lm, hyperparams, hyperparam_rule_dict, spherical_modes=None):
    if spherical_modes is None:
        spherical_modes = param_dict_lm.keys()
    return {mode: get_new_params(param_dict_lm[mode], hyperparams, hyperparam_rule_dict) for mode in spherical_modes}
