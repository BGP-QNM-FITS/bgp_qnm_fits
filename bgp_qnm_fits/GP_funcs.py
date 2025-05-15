import numpy as np
import jax
import jax.numpy as jnp
from bgp_qnm_fits.utils import get_inverse

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def squared_exp_element(t1, t2, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-0.5 * dist**2 / period**2)


def logoneplusexp(t):
    return jnp.log(1 + jnp.exp(-jnp.abs(t))) + jnp.maximum(t, 0)


def smoothclip(t, sigma_min, sigma_max, sharpness):
    return (
        t
        - (1.0 / sharpness) * logoneplusexp(sharpness * (t - sigma_max))
        + (1.0 / sharpness) * logoneplusexp(-sharpness * (t - sigma_min))
    )


def softclip(t, sigma_min, sigma_max, sharpness):
    return jnp.exp(smoothclip(jnp.log(t), jnp.log(sigma_min), jnp.log(sigma_max), sharpness))


def exponential_func(t, length_scale, t_s, sigma_max):
    return sigma_max * jnp.exp(-(t - t_s) / length_scale)


def new_func(t, length_scale, t_s, sigma_min, sigma_max, sharpness):
    t = jnp.asarray(t)
    return softclip(
        exponential_func(t, length_scale, t_s, sigma_max),
        sigma_min,
        sigma_max,
        sharpness,
    )


def periodic_kernel(t1, t2, length_scale, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-2 * jnp.sin(np.pi * dist / period) ** 2 / length_scale**2)


def kernel_s(analysis_times, **kwargs):
    return kwargs["sigma_max"] ** 2 * jnp.eye(len(analysis_times))


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


def kernel_c(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        (
            squared_exp_element(t1, t2, kwargs["period"]) ** kwargs["a"]
            * periodic_kernel(t1, t2, kwargs["length_scale_2"], kwargs["period_2"]) ** (1 - kwargs["a"])
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
    return kernel(jnp.asarray(analysis_times), **hyperparams) + jnp.eye(len(analysis_times)) * 1e-9


def get_inv_GP_covariance_matrix(analysis_times, kernel, tuned_param_dict, spherical_modes=None):
    if spherical_modes is None:
        spherical_modes = tuned_param_dict.keys()
    kernel_dict = {
        mode: compute_kernel_matrix(analysis_times, tuned_param_dict[mode], kernel) for mode in spherical_modes
    }

    return np.array(
        [get_inverse(kernel_dict[mode]) for mode in spherical_modes],
        dtype=np.complex128,
    )


def kl_divergence(p, q):
    dim = p.shape[0]
    trace_term = np.trace(np.linalg.solve(q, p))
    log_det_p = np.linalg.slogdet(p)[1]
    log_det_q = np.linalg.slogdet(q)[1]
    kl_div = 0.5 * (trace_term - dim + log_det_q - log_det_p)
    return kl_div


def js_divergence(p, q):
    return kl_divergence(p, q) / 2 + kl_divergence(q, p) / 2
