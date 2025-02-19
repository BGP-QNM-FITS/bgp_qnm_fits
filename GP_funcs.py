import numpy as np
import qnmfits
import matplotlib.pyplot as plt
import scipy
import sys
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from likelihood_funcs import * 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.signal import correlate

sys.setrecursionlimit(10000)

def sim_interpolator_data(data_dict, data_times, new_times):

    h = {}
    for mode in data_dict.keys():
        h_real = spline(data_times, np.real(data_dict[mode]))(new_times)
        h_imag = spline(data_times, np.imag(data_dict[mode]))(new_times)
        h[mode] = h_real + 1j * h_imag

    return h


def sim_interpolator(sim, new_times):

    data_dict = sim.h 
    data_times = sim.times

    h_interp = sim_interpolator_data(data_dict, data_times, new_times)

    sim_new = qnmfits.Custom(
        new_times,
        h_interp,
        metadata={
            "remnant_mass": sim.Mf,
            "remnant_dimensionless_spin": sim.chif,
        },
        # Data does not need shifting again 
        zero_time=0,
    )
    
    return sim_new


def get_time_shift(sim1, sim2, dt=0.0001, range = 10):

    # Mask data to speed up interpolating (and shifting) 

    mask1 = (sim1.times >= -range) & (sim1.times < range)
    mask2 = (sim2.times >= -range) & (sim2.times < range)

    sim1_times_masked = sim1.times[mask1]
    sim1_data_masked = {mode: sim1.h[mode][mask1] for mode in sim1.h.keys()}

    sim2_times_masked = sim2.times[mask2]
    sim2_data_masked = {mode: sim2.h[mode][mask2] for mode in sim2.h.keys()}

    new_times = np.arange(sim1_times_masked[0], sim1_times_masked[-1], dt)

    h1_int = sim_interpolator_data(sim1_data_masked, sim1_times_masked, new_times)
    h2_int = sim_interpolator_data(sim2_data_masked, sim2_times_masked, new_times)

    overlap = np.sum(
        [np.fft.ifft(np.fft.fft(h1_int[mode]) * np.conjugate(np.fft.fft(h2_int[mode]))) for mode in sim1.h.keys()],
        axis=0
    )

    max_shift_index = np.argmax(np.abs(overlap))

    if max_shift_index > len(new_times) // 2:
        max_shift_index -= len(new_times)

    time_shift = max_shift_index * dt

    return time_shift


def log_evidence(K, f):
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * (np.dot(f, scipy.linalg.solve(K, f, assume_a = 'pos')) + logdet + len(f) * np.log(2 * np.pi)) 


def get_new_params(param_dict, hyperparam_list, rule_dict):
    new_params = {}

    epsilon = 1./10

    for i, param in enumerate(rule_dict.keys()):
        rule = rule_dict[param]
        if rule == "multiply":
            new_params[param] = param_dict[param] * hyperparam_list[i]
        elif rule == "sum":
            new_params[param] = param_dict[param] + hyperparam_list[i]
        else:
            print("Rule not recognised for parameter", param)
            
    for param in param_dict.keys():
        if param not in new_params.keys():
            if param == "sigma_min":
                new_params[param] = new_params["sigma_max"] * epsilon
            else:
                new_params[param] = param_dict[param]

    return new_params

def get_total_log_evidence(hyperparam_list, param_dict_sim_lm, f_dict_sim_lm, rule_dict, analysis_times, kernel, sims = None, modes = None):

    NO_NEGATIVE_MS = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0010", "0011", "0012"]

    if sims == None:
        sims = param_dict_sim_lm.keys() 

    total_log_evidence = 0

    for sim in sims:
        if modes == None:
            modes = param_dict_sim_lm[sim].keys()
        for mode in modes:
            param_dict = param_dict_sim_lm[sim][mode]
            new_param_dict = get_new_params(param_dict, hyperparam_list, rule_dict)
            K = compute_kernel_matrix(analysis_times, new_param_dict, kernel)
            f = f_dict_sim_lm[sim][mode]
            total_log_evidence += log_evidence(K, f.real) 
            total_log_evidence += log_evidence(K, f.imag)
            # To do include negative m modes (with the same log evidence as the positive m modes but excluded from the initial sum)
            if sim in NO_NEGATIVE_MS:
                total_log_evidence += log_evidence(K, f.real) 
                total_log_evidence += log_evidence(K, f.imag)

    # Minus sign to minimse 

    return -total_log_evidence 

def squared_exp_element(t1, t2, period):
    dist = np.abs(t1[:, None] - t2[None, :])
    return np.exp(-0.5 * dist**2 / period**2)

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

def new_func(t, length_scale, t_s, sigma_min, sigma_max, sharpness):
    t = np.asarray(t)
    return softclip(exponential_func(t, length_scale, t_s, sigma_max), sigma_min, sigma_max, sharpness)

def periodic_kernel(t1, t2, length_scale, period):
    dist = np.abs(t1[:, None] - t2[None, :])
    return  np.exp(-2*np.sin(np.pi*dist/period)**2 / length_scale**2)

def kernel_s(analysis_times, **kwargs):
    return kwargs['sigma_max']**2 * np.eye(len(analysis_times))

def kernel(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        squared_exp_element(t1, t2, kwargs['period'])
        * new_func(t1, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[:, None]
        * new_func(t2, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[None, :]
    )

def kernel_c(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        (
            squared_exp_element(t1, t2, kwargs['period']) ** kwargs['a']
            * periodic_kernel(t1, t2, kwargs['length_scale_2'], kwargs['period_2']) ** (1 - kwargs['a'])
         ) 
        * new_func(t1, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[:, None]
        * new_func(t2, kwargs['length_scale'], kwargs['t_s'], kwargs['sigma_min'], kwargs['sigma_max'], kwargs['sharpness'])[None, :]
    )

def compute_kernel_matrix(analysis_times, hyperparams, kernel):
    return kernel(np.asarray(analysis_times), **hyperparams) + np.eye(len(analysis_times)) * 1e-13

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
    denom = np.linalg.det(0.5 * (p + q))**0.5
    if denom == 0:
        return 0
    else:
        return np.sqrt(1 - (det_p**0.25 * det_q**0.25) / denom)

def js_divergence(p, q):
    return kl_divergence(p,q)/2 + kl_divergence(q, p)/2

def ws_distance(p, q):
    return wasserstein_distance(p.flatten(), q.flatten())  

def get_GP_covariance_matrix(analysis_times, param_dict_lm, hyperparams, hyperparam_rule_dict, kernel, spherical_modes=None):
    """

    Determines the covariance matrix for one simulation (all given modes) 

    """
    if spherical_modes == None:
        spherical_modes = param_dict_lm.keys()
    tuned_param_dict = {mode: get_new_params(param_dict_lm[mode], hyperparams, hyperparam_rule_dict) for mode in spherical_modes}
    kernel_dict = {mode: compute_kernel_matrix(analysis_times, tuned_param_dict[mode], kernel) for mode in spherical_modes}
    return np.array([np.linalg.inv(kernel_dict[mode]) for mode in spherical_modes], dtype=np.complex128)
