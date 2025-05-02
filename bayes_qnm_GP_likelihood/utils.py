"""

This file contains miscellaneous utility functions for the bayes_qnm_GP_likelihood package.

"""

import numpy as np
import qnmfits
import scipy 
from scipy.interpolate import make_interp_spline as spline


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


def get_time_shift(sim1, sim2, modes=None, delta=0.0001, alpha=0.1, t0=-100, T=100):

    # Mask data to speed up interpolating (and shifting) 
    # But need to apply a smooth buffer 

    if modes is None:
        modes = sim1.h.keys()

    mask1 = (sim1.times >= t0) & (sim1.times < T)
    mask2 = (sim2.times >= t0) & (sim2.times < T)

    sim1_times_masked = sim1.times[mask1]
    sim2_times_masked = sim2.times[mask2]

    def taper_window(times_masked, alpha):
        window = np.ones_like(times_masked)
        taper_length = int(alpha * len(times_masked) / 2)
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_length) / taper_length))
        window[:taper_length] = taper
        window[-taper_length:] = taper[::-1]
        return window

    window_1 = taper_window(sim1_times_masked, alpha)
    window_2 = taper_window(sim2_times_masked, alpha)

    sim1_data_masked = {mode: sim1.h[mode][mask1]*window_1 for mode in sim1.h.keys()}
    sim2_data_masked = {mode: sim2.h[mode][mask2]*window_2 for mode in sim2.h.keys()}

    new_times = np.arange(sim1_times_masked[0], sim1_times_masked[-1], delta)
    h1_int = sim_interpolator_data(sim1_data_masked, sim1_times_masked, new_times)
    h2_int = sim_interpolator_data(sim2_data_masked, sim2_times_masked, new_times)

    overlap = np.sum(
        [
            np.fft.ifft(
                np.fft.fft(h1_int[mode]) * np.conjugate(np.fft.fft(h2_int[mode]))
            )
            for mode in modes
        ],
        axis=0,
    )

    max_shift_index = np.argmax(np.abs(overlap))

    if max_shift_index > len(new_times) // 2:
        max_shift_index -= len(new_times)

    time_shift = max_shift_index * delta

    return time_shift


def get_inverse(matrix, epsilon=1e-10):
    """Get the inverse of the fisher matrix - not general""" 
    vals, vecs = scipy.linalg.eigh(matrix)
    vals = np.maximum(vals, epsilon)
    return np.einsum('ik, k, jk -> ij', vecs, 1/vals, vecs)


def weighted_quantile(values, quantiles, weights=None):
    values = np.array(values)
    quantiles = np.array(quantiles)
    if weights is None:
        weights = np.ones(values.shape[0])
    weights = np.array(weights)

    # Sort values and weights along the first axis
    sorter = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, sorter, axis=0)
    sorted_weights = np.take_along_axis(weights[:, None], sorter, axis=0)

    # Compute cumulative weights
    cumulative_weights = np.cumsum(sorted_weights, axis=0) - 0.5 * sorted_weights
    cumulative_weights /= cumulative_weights[-1, :]
    
    # Interpolate quantiles
    quantile_values = np.empty((len(quantiles), values.shape[1]))
    for i in range(values.shape[1]):
        quantile_values[:, i] = np.interp(quantiles, cumulative_weights[:, i], sorted_values[:, i])

    return quantile_values


def mismatch(wf_array_1, wf_array_2, inv_noise_covariance_matrix=None):

    """
    Compute the phase maximised, mismatch between two waveforms. If inv_noise_covariance_matrix is provided
    then the mismatch is computed using the inverse noise covariance matrix.

    """

    if inv_noise_covariance_matrix is None:
        numerator = np.abs(np.sum(wf_array_1 * np.conj(wf_array_2), axis=(0, 1)))
        wf_1_norm = np.abs(np.sum(wf_array_1 * np.conj(wf_array_1), axis=(0, 1)))
        wf_2_norm = np.abs(np.sum(wf_array_2 * np.conj(wf_array_2), axis=(0, 1)))
    else: 
        numerator = np.abs(np.einsum("bi,bj,bij->", wf_array_1, np.conj(wf_array_2), inv_noise_covariance_matrix)) 
        wf_1_norm = np.abs(np.einsum("bi,bj,bij->", wf_array_1, np.conj(wf_array_1), inv_noise_covariance_matrix))
        wf_2_norm = np.abs(np.einsum("bi,bj,bij->", wf_array_2, np.conj(wf_array_2), inv_noise_covariance_matrix))
    
    denominator = np.sqrt(wf_1_norm * wf_2_norm)
    
    return 1 - (numerator / denominator)


def log_likelihood(data_array, model_array, inv_covariance_matrix):
        """
        Compute the log-likelihood of the data given the model.

        Args:
            data_array (array): The data array.
            model_array (array): The model array.
            inv_covariance_matrix (array): The inverse covariance matrix.

        Returns:
            float: The log-likelihood value.
        """

        # TODO Check this is valid (why is there a small imaginary component?)

        residual = data_array - model_array
        return np.real(-0.5 * np.einsum(
            "bi,bj,bij->", np.conj(residual), residual, inv_covariance_matrix
        )) 

