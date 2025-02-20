import numpy as np
import scipy 
from funcs.likelihood_funcs import *
from funcs.GP_funcs import *
from funcs.utils import *
from scipy.optimize import minimize


def get_residuals(sim_main, sim_lower, t0, T, dt=None):

    # Perform a time shift

    # TODO shift sensitive to range, maybe because of cutting the waveform and doing fft (but it's not a huge effect; on the order of 0.01)
    time_shift = get_time_shift(sim_main, sim_lower, delta=0.0001, range=35)
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

    analysis_mask = (sim_main_interp.times >= t0 - 1e-9) & (
        sim_main_interp.times < t0 + T
    )

    return {
        key: sim_main_interp.h[key][analysis_mask]
        - sim_lower_interp.h[key][analysis_mask]
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
            "length_scale": -1
            / (
                omega := qnmfits.qnm.omega(ell, m, 0, -1 if m < 0 else 1, chif_mag, Mf)
            ).imag,
            "period": (2 * np.pi) / omega.real,
            # For the complicated kernel only
            "length_scale_2": -1 / omega.imag,
            "period_2": (2 * np.pi) / omega.real,
            "a": 0.5,
        }
        for ell, m in spherical_modes
    }

    return param_dict_lm


def log_evidence(K, f):
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * (
        np.dot(f, scipy.linalg.solve(K, f, assume_a="pos"))
        + logdet
        + len(f) * np.log(2 * np.pi)
    )


def get_new_params(param_dict, hyperparam_list, rule_dict):
    new_params = {
        param: (
            param_dict[param] * hyperparam_list[i]
            if rule == "multiply"
            else param_dict[param] + hyperparam_list[i]
        )
        for i, (param, rule) in enumerate(rule_dict.items())
    }

    new_params["sigma_min"] = (
        param_dict["sigma_min"]
        * hyperparam_list[list(rule_dict.keys()).index("sigma_max")]
    )
    new_params.update(
        {param: param_dict[param] for param in param_dict if param not in new_params}
    )

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
):

    mode_filters = {
        "PE": lambda mode: mode[1] >= 0 and mode[1] % 2 == 0,
        "P": lambda mode: mode[1] >= 0,
        "E": lambda mode: mode[1] % 2 == 0,
        "ALL": lambda mode: True,
    }

    total_log_evidence = 0

    for sim_id, mode_rule in mode_rules.items():

        spherical_mode_choice = [
            mode for mode in spherical_modes if mode_filters[mode_rule](mode)
        ]

        for mode in spherical_mode_choice:
            param_dict = param_dict_sim_lm[sim_id][mode]
            new_param_dict = get_new_params(param_dict, hyperparam_list, rule_dict)
            K = compute_kernel_matrix(analysis_times, new_param_dict, kernel)
            f = f_dict_sim_lm[sim_id][mode]
            total_log_evidence += log_evidence(K, f.real) + log_evidence(K, f.imag)
            # To do include negative m modes (with the same log evidence as the positive m modes but excluded from the initial sum)
            if mode_rule in {"P", "PE"}:
                total_log_evidence += log_evidence(K, f.real) + log_evidence(K, f.imag)

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


def get_tuned_params(
    param_dict_lm, hyperparams, hyperparam_rule_dict, spherical_modes=None
):
    if spherical_modes == None:
        spherical_modes = param_dict_lm.keys()
    return {
        mode: get_new_params(param_dict_lm[mode], hyperparams, hyperparam_rule_dict)
        for mode in spherical_modes
    }
