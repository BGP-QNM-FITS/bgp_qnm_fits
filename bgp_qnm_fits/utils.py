import numpy as np
import bgp_qnm_fits.qnmfits_funcs as qnmfits
import jax.numpy as jnp
from jax.scipy.linalg import eigh, cholesky, solve_triangular
from scipy.interpolate import make_interp_spline as spline


def sim_interpolator_data(data_dict, data_times, new_times):
    """
    Interpolates the data_dict onto new_times using cubic splines.
    Args:
        data_dict (dict): Dictionary containing the data to be interpolated.
        data_times (array): The times corresponding to the data in data_dict.
        new_times (array): The times at which to interpolate the data.
    Returns:
        dict: A dictionary containing the interpolated data at new_times.
    """
    h = {}
    for mode in data_dict.keys():
        h_real = spline(data_times, np.real(data_dict[mode]))(new_times)
        h_imag = spline(data_times, np.imag(data_dict[mode]))(new_times)
        h[mode] = h_real + 1j * h_imag

    return h


def sim_interpolator(sim, new_times):
    """
    Interpolates the simulation data onto new_times using cubic splines.
    Args:
        sim (qnmfits.Custom): The simulation object containing the data to be interpolated.
        new_times (array): The times at which to interpolate the data.
    Returns:
        qnmfits.Custom: A new simulation object with the interpolated data.
    """
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
    """
    Computes the time and phase shift between two simulations by maximizing the overlap of their waveforms.
    Args:
        sim1 (qnmfits.Custom): The first simulation object.
        sim2 (qnmfits.Custom): The second simulation object.
        modes (list, optional): List of modes to consider for the overlap. If None, all modes are used.
        delta (float, optional): The time step for the new time grid. Default is 0.0001 (demonstrated to
        be small enough and converge in BH cartography).
        alpha (float, optional): The tapering factor for the window function. Default is 0.1.
        t0 (float, optional): The start time for the analysis. Default is -100.
        T (float, optional): The end time for the analysis. Default is 100.
    Returns:
        float: The time shift between the two simulations.
    """

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

    sim1_data_masked = {mode: sim1.h[mode][mask1] * window_1 for mode in sim1.h.keys()}
    sim2_data_masked = {mode: sim2.h[mode][mask2] * window_2 for mode in sim2.h.keys()}

    new_times = np.arange(sim1_times_masked[0], sim1_times_masked[-1], delta)
    h1_int = sim_interpolator_data(sim1_data_masked, sim1_times_masked, new_times)
    h2_int = sim_interpolator_data(sim2_data_masked, sim2_times_masked, new_times)

    overlap = np.sum(
        [np.fft.ifft(np.fft.fft(h1_int[mode]) * np.conjugate(np.fft.fft(h2_int[mode]))) for mode in modes],
        axis=0,
    )

    max_shift_index = np.argmax(np.abs(overlap))

    if max_shift_index > len(new_times) // 2:
        max_shift_index -= len(new_times)

    time_shift = max_shift_index * delta

    return time_shift


def get_inverse(matrix, epsilon=1e-10):
    """
    Compute the inverse of a matrix using its eigenvalues and eigenvectors.

    Args:
        matrix (array): The matrix to be inverted.
        epsilon (float): A small value to ensure numerical stability.
    Returns:
        array: The inverse of the input matrix.

    """
    vals, vecs = eigh(matrix)
    vals = jnp.maximum(vals, epsilon)
    return jnp.einsum("ik, k, jk -> ij", vecs, 1 / vals, vecs)


def mismatch(wf_array_1, wf_array_2, noise_covariance_matrix=None):
    """
    Compute the phase maximised, mismatch between two waveforms. If inv_noise_covariance_matrix is provided
    then the mismatch is computed using the inverse noise covariance matrix.

    Args:
        wf_array_1 (array): The first waveform array.
        wf_array_2 (array): The second waveform array.
        inv_noise_covariance_matrix (array, optional): The inverse noise covariance matrix. If None, the
            mismatch is computed without it.
    Returns:
        float: The mismatch value between the two waveforms.

    """

    if noise_covariance_matrix is None:

        numerator = np.abs(np.sum(wf_array_1 * np.conj(wf_array_2), axis=(0, 1)))
        wf_1_norm = np.abs(np.sum(wf_array_1 * np.conj(wf_array_1), axis=(0, 1)))
        wf_2_norm = np.abs(np.sum(wf_array_2 * np.conj(wf_array_2), axis=(0, 1)))

    else:

        L = cholesky(noise_covariance_matrix, lower=True)

        L_wf1 = solve_triangular(L, wf_array_1, lower=True)
        L_wf2 = solve_triangular(L, wf_array_2, lower=True)

        Kinv_wf1 = solve_triangular(jnp.transpose(L, [0, 2, 1]), L_wf1, lower=False)
        Kinv_wf2 = solve_triangular(jnp.transpose(L, [0, 2, 1]), L_wf2, lower=False)

        numerator = jnp.abs(jnp.einsum("st, st -> ", jnp.conj(wf_array_1), Kinv_wf2).item())
        wf_1_norm = jnp.abs(jnp.einsum("st, st -> ", jnp.conj(wf_array_1), Kinv_wf1).item())
        wf_2_norm = jnp.abs(jnp.einsum("st, st -> ", jnp.conj(wf_array_2), Kinv_wf2).item())

    denominator = np.sqrt(wf_1_norm * wf_2_norm)

    return 1 - (numerator / denominator)
