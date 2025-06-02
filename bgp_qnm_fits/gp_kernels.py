import os 
import jax
import jax.numpy as jnp

from bgp_qnm_fits.utils import get_inverse

def squared_exp_element(t1, t2, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-0.5 * dist**2 / period**2)


def exponential_func(t, length_scale, t_s, sigma_max):
    return sigma_max * jnp.exp(-(t - t_s) / length_scale)


def smoothmax(x, x_max, smoothness):
    return (x + x_max - jnp.sqrt((x - x_max) ** 2 + smoothness*x_max**2)) * 0.5


def new_func(t, length_scale, t_s, sigma_max, smoothness):
    t = jnp.asarray(t)
    return jnp.exp(smoothmax(
        jnp.log(exponential_func(t, length_scale, t_s, sigma_max)),
        jnp.log(sigma_max),
        smoothness,
    )) 


def periodic_kernel(t1, t2, length_scale, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-2 * jnp.sin(np.pi * dist / period) ** 2 / length_scale**2)


def kernel_WN(analysis_times, **kwargs):
    return kwargs["sigma_max"] ** 2 * jnp.eye(len(analysis_times))


def kernel_GP(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        squared_exp_element(t1, t2, kwargs["period"])
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["smoothness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["smoothness"],
        )[None, :]
    )


def kernel_GPC(analysis_times, **kwargs):
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
            kwargs["sigma_max"],
            kwargs["smoothness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["smoothness"],
        )[None, :]
    )


def compute_kernel_matrix(analysis_times, hyperparams, kernel, epsilon=1e-6):
    return (
        kernel(jnp.asarray(analysis_times), **hyperparams)
        + jnp.eye(len(analysis_times)) * hyperparams["jitter_scale"] ** 2 * epsilon
    )


def get_inv_GP_covariance_matrix(analysis_times, kernel, tuned_param_dict, spherical_modes=None):
    if spherical_modes is None:
        spherical_modes = tuned_param_dict.keys()
    kernel_dict = {
        mode: compute_kernel_matrix(analysis_times, tuned_param_dict[mode], kernel) for mode in spherical_modes
    }

    return jnp.array(
        [get_inverse(kernel_dict[mode]) for mode in spherical_modes],
        dtype=jnp.complex128,
    )