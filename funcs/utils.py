"""This module contains miscellaneous (non-essential) functions to streamline workbooks."""

import numpy as np
import qnmfits
from scipy.interpolate import InterpolatedUnivariateSpline as spline


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


def get_time_shift(sim1, sim2, delta=0.0001, range=10):

    # Mask data to speed up interpolating (and shifting)

    mask1 = (sim1.times >= -range) & (sim1.times < range)
    mask2 = (sim2.times >= -range) & (sim2.times < range)

    sim1_times_masked = sim1.times[mask1]
    sim1_data_masked = {mode: sim1.h[mode][mask1] for mode in sim1.h.keys()}

    sim2_times_masked = sim2.times[mask2]
    sim2_data_masked = {mode: sim2.h[mode][mask2] for mode in sim2.h.keys()}

    new_times = np.arange(sim1_times_masked[0], sim1_times_masked[-1], delta)

    h1_int = sim_interpolator_data(sim1_data_masked, sim1_times_masked, new_times)
    h2_int = sim_interpolator_data(sim2_data_masked, sim2_times_masked, new_times)

    overlap = np.sum(
        [
            np.fft.ifft(
                np.fft.fft(h1_int[mode]) * np.conjugate(np.fft.fft(h2_int[mode]))
            )
            for mode in sim1.h.keys()
        ],
        axis=0,
    )

    max_shift_index = np.argmax(np.abs(overlap))

    if max_shift_index > len(new_times) // 2:
        max_shift_index -= len(new_times)

    time_shift = max_shift_index * delta

    return time_shift


##################################################################################################################################################


def get_amplitude_params(
    sim, t0, T=100, l_max=8, n_max=7, include_chif=False, include_Mf=False
):
    qnm_list = [
        (ell, m, n, p)
        for ell in np.arange(2, l_max + 1)
        for m in np.arange(-ell, ell + 1)
        for n in np.arange(0, n_max + 1)
        for p in [-1, 1]
    ]
    spherical_modes = [
        (ell, m) for ell in np.arange(2, l_max + 1) for m in np.arange(-ell, ell + 1)
    ]

    data_times = sim.times
    data = sim.h
    Mf_0 = sim.Mf
    chif_mag_0 = sim.chif_mag

    ls_fit = qnmfits.multimode_ringdown_fit(
        data_times,
        data,
        modes=qnm_list,
        Mf=Mf_0,
        chif=chif_mag_0,
        t0=t0,
        T=T,
        spherical_modes=spherical_modes,
    )

    C_0 = ls_fit["C"]

    true_params = []
    for re_c, im_c in zip(np.real(ls_fit["C"]), np.imag(ls_fit["C"])):
        true_params.append(re_c)
        true_params.append(im_c)

    if include_chif:
        true_params = true_params + [chif_mag_0]
    if include_Mf:
        true_params = true_params + [Mf_0]

    return (
        qnm_list,
        spherical_modes,
        data_times,
        data,
        Mf_0,
        chif_mag_0,
        C_0,
        true_params,
    )


def get_tidy_list(
    qnm_list, spherical_modes, t0, data_times, data, Mf_0, chif_mag_0, inv_cov
):
    """Get a tidied list of QNMs and their significance."""

    fisher_matrix = get_fisher_matrix(
        qnm_list,
        spherical_modes,
        t0,
        data_times,
        Mf_0,
        chif_mag_0,
        inv_cov,
        T=100,
        C_0=None,
        include_chif=False,
        include_Mf=False,
    )

    b_vec = get_b_vector(
        qnm_list,
        spherical_modes,
        t0,
        data_times,
        data,
        Mf_0,
        chif_mag_0,
        inv_cov,
        T=100,
        C_0=None,
        include_chif=False,
        include_Mf=False,
    )

    mean_vector = np.linalg.solve(fisher_matrix, b_vec)
    sig_list = get_significance_list(qnm_list, mean_vector, fisher_matrix)
    qnm_list, sig_list = zip(
        *sorted(zip(qnm_list, sig_list), key=lambda x: x[1], reverse=True)
    )
    for sig, qnm in zip(sig_list, qnm_list):
        print(qnm, sig)


def get_fisher_and_b(sim, t0, T=100, l_max=8, n_max=7):
    # TODO: Build in inv_cov functionality

    qnm_list, spherical_modes, data_times, data, Mf_0, chif_mag_0, C_0, true_params = (
        get_params(sim, t0, T=T, l_max=l_max, n_max=n_max)
    )

    fisher_matrix = get_fisher_matrix(
        qnm_list, spherical_modes, t0, data_times, Mf_0, chif_mag_0, inv_cov=1, T=T
    )

    b_vec = get_b_vector(
        qnm_list,
        spherical_modes,
        t0,
        data_times,
        data,
        Mf_0,
        chif_mag_0,
        inv_cov=1,
        T=T,
    )

    mean_vector = np.linalg.solve(fisher_matrix, b_vec)
    covariance_matrix = np.linalg.inv(fisher_matrix)

    return fisher_matrix, b_vec, mean_vector, covariance_matrix, true_params
