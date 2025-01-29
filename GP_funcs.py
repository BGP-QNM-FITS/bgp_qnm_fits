import numpy as np
import matplotlib.pyplot as plt
import qnmfits
import CCE 
from likelihood_funcs import * 
import corner
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as spline


def data_mismatch(sim1, sim2, t0=0, modes=None, T=100, dt=0.01, shift=0):

    # TODO: Use Fourier transform to do the time shift 

    new_times = np.arange(t0, t0 + T, dt)

    if modes is None:
        modes = list(sim1.h.keys())

    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0

    for mode in modes:

        h1 = sim1.h[mode]
        h2 = sim2.h[mode]

        interp_h1 = np.interp(new_times, sim1.times, h1)
        interp_h2 = np.interp(new_times - shift, sim2.times, h2)

        numerator += np.abs(np.trapz(interp_h1 * np.conjugate(interp_h2), x=new_times))
        denominator1 += np.abs(
            np.trapz(interp_h1 * np.conjugate(interp_h1), x=new_times)
        )
        denominator2 += np.abs(
            np.trapz(interp_h2 * np.conjugate(interp_h2), x=new_times)
        )

    denominator = np.sqrt(denominator1 * denominator2)

    return 1 - (numerator / denominator)


def get_residual(sim_main, sims_other, spherical_modes=None):

    if spherical_modes == None:
        spherical_modes = list(sim_main.h.keys())

    # TODO: Search all levels/radii or just choose one? And is sum the best measure for the 'worst'? 

    R = {}
    for i, sim in enumerate(sims_other):
        for ell, m in spherical_modes:
            diff = sim_main.h[ell,m] - sim[ell,m]
            if (ell, m) not in R.keys() or np.abs(np.sum(diff)) > np.abs(np.sum(R[ell,m])):
                R[ell,m] = diff 

    return R


def squared_exp_element(t1, t2, period):
    time_diff = np.abs(t1[:, None] - t2[None, :])
    return np.exp(-0.5 * time_diff**2 / period**2)

def logoneplusexp(t):
    return np.log(1 + np.exp(-np.abs(t))) + np.maximum(t, 0)

def smoothclip(t, sigma_min, sigma_max, sharpness):
    return (t - 
            (1./sharpness)*logoneplusexp(sharpness*(t - sigma_max)) + 
            (1./sharpness)*logoneplusexp(-sharpness*(t - sigma_min))
    )

def softclip(t, sigma_min, sigma_max, sharpness):
    return np.exp(smoothclip(np.log(t), np.log(sigma_min), np.log(sigma_max), sharpness))

def exponential_func(t, length_scale, t_s, sigma_max):
    return sigma_max * np.exp(-(t-t_s)/length_scale)

def new_func(t, length_scale, t_s, t_min, sigma_max, sharpness):
    t = np.asarray(t)
    return softclip(exponential_func(t, length_scale, t_s, sigma_max), t_min, sigma_max, sharpness)

def kernel(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        squared_exp_element(t1, t2, kwargs['period'])
        * new_func(t1, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[:, None]
        * new_func(t2, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[None, :]
    )

def compute_kernel(analysis_times, spherical_modes, hyperparam_dict):
    kernel_dict = {} 
    for (ell, m) in spherical_modes:
        kernel_dict[(ell, m)] = kernel(np.asarray(analysis_times), **hyperparam_dict[ell,m]) + np.eye(len(analysis_times)) * 1e-14

def compute_kernel_matrix(analysis_times, hyperparams):
    return kernel(np.asarray(analysis_times), **hyperparams) + np.eye(len(analysis_times)) * 1e-13

def log_evidence(K, f):
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * np.dot(f, np.linalg.solve(K, f)) - 0.5 * logdet - 0.5 * len(f) * np.log(2 * np.pi)

def log_evidence_range(R, param, analysis_times, analysis_mask, hyperparam_dict, values):
    f = R[analysis_mask]
    hyperparam_dict_new = hyperparam_dict.copy()
    log_evidence_values = np.zeros(len(values))
    for i, value in enumerate(values):
        hyperparam_dict_new[param] = value
        K = compute_kernel_matrix(analysis_times, hyperparam_dict_new)
        log_evidence_values[i] = log_evidence(K, f)
    return log_evidence_values