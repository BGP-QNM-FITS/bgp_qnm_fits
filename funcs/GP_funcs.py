import numpy as np
from scipy.stats import wasserstein_distance

def squared_exp_element(t1, t2, period):
    dist = np.abs(t1[:, None] - t2[None, :])
    return np.exp(-0.5 * dist**2 / period**2)


def logoneplusexp(t):
    return np.log(1 + np.exp(-np.abs(t))) + np.maximum(t, 0)


def smoothclip(t, sigma_min, sigma_max, sharpness):
    return (
        t
        - (1.0 / sharpness) * logoneplusexp(sharpness * (t - sigma_max))
        + (1.0 / sharpness) * logoneplusexp(-sharpness * (t - sigma_min))
    )


def softclip(t, sigma_min, sigma_max, sharpness):
    return np.exp(
        smoothclip(np.log(t), np.log(sigma_min), np.log(sigma_max), sharpness)
    )


def exponential_func(t, length_scale, t_s, sigma_max):
    return sigma_max * np.exp(-(t - t_s) / length_scale)


def new_func(t, length_scale, t_s, sigma_min, sigma_max, sharpness):
    t = np.asarray(t)
    return softclip(
        exponential_func(t, length_scale, t_s, sigma_max),
        sigma_min,
        sigma_max,
        sharpness,
    )


def periodic_kernel(t1, t2, length_scale, period):
    dist = np.abs(t1[:, None] - t2[None, :])
    return np.exp(-2 * np.sin(np.pi * dist / period) ** 2 / length_scale**2)


def kernel_s(analysis_times, **kwargs):
    return kwargs["sigma_max"] ** 2 * np.eye(len(analysis_times))


def kernel_main(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        squared_exp_element(t1, t2, kwargs["period"])
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            kwargs["sigma_max"],
            kwargs["sharpness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            kwargs["sigma_max"],
            kwargs["sharpness"],
        )[None, :]
    )

def kernel_test_altsigma(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        kwargs["sigma_max"]**2 * squared_exp_element(t1, t2, kwargs["period"])
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            1,
            kwargs["sharpness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            1,
            kwargs["sharpness"],
        )[None, :]
    )

def kernel_test_stationary(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        kwargs["sigma_max"] ** 2 * squared_exp_element(t1, t2, kwargs["period"])
    )

def kernel_c(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        (
            squared_exp_element(t1, t2, kwargs["period"]) ** kwargs["a"]
            * periodic_kernel(t1, t2, kwargs["length_scale_2"], kwargs["period_2"])
            ** (1 - kwargs["a"])
        )
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            kwargs["sigma_max"],
            kwargs["sharpness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_min"],
            kwargs["sigma_max"],
            kwargs["sharpness"],
        )[None, :]
    )


def compute_kernel_matrix(analysis_times, hyperparams, kernel):
    return (
        kernel(np.asarray(analysis_times), **hyperparams) 
        + np.eye(len(analysis_times)) * 1e-13
    )


def kl_divergence(p, q):
    dim = p.shape[0]
    inv_q = np.linalg.inv(q)
    trace_term = np.trace(inv_q @ p)
    log_det_p = np.linalg.slogdet(p)[1]
    log_det_q = np.linalg.slogdet(q)[1]
    kl_div = 0.5 * (trace_term - dim + log_det_q - log_det_p)
    return kl_div


def hellinger_distance(p, q):
    # TODO get this to work
    det_p = np.linalg.det(p)
    det_q = np.linalg.det(q)
    denom = np.linalg.det(0.5 * (p + q)) ** 0.5
    if denom == 0:
        return 0
    else:
        return np.sqrt(1 - (det_p**0.25 * det_q**0.25) / denom)


def js_divergence(p, q):
    return kl_divergence(p, q) / 2 + kl_divergence(q, p) / 2


def ws_distance(p, q):
    return wasserstein_distance(p.flatten(), q.flatten())


def get_GP_covariance_matrix(
    analysis_times, kernel, tuned_param_dict, spherical_modes=None
):
    if spherical_modes == None:
        spherical_modes = tuned_param_dict.keys()
    kernel_dict = {
        mode: compute_kernel_matrix(analysis_times, tuned_param_dict[mode], kernel)
        for mode in spherical_modes
    }

    return np.array(
        [np.linalg.inv(kernel_dict[mode]) for mode in spherical_modes],
        dtype=np.complex128,
    )
