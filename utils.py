"""This module contains miscellaneous (non-essential) functions to streamline workbooks."""

import numpy as np
import qnmfits
from likelihood_funcs import *
from qnm_selecting_funcs import *


def get_params(sim, t0, T=100, l_max=8, n_max=7, include_chif=False, include_Mf=False):
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


def get_tidy_list(qnm_list, spherical_modes, t0, data_times, data, Mf_0, chif_mag_0, inv_cov):
    """Get a tidied list of QNMs and their significance."""

    fisher_matrix = get_fisher_matrix(
        qnm_list,
        spherical_modes,
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
    qnm_list, sig_list = zip(*sorted(zip(qnm_list, sig_list), key=lambda x: x[1], reverse=True))
    for sig, qnm in zip(sig_list, qnm_list):
        print(qnm, sig)


def get_fisher_and_b(sim, t0, T=100, l_max=8, n_max=7):
    # TODO: Build in inv_cov functionality

    qnm_list, spherical_modes, data_times, data, Mf_0, chif_mag_0, C_0, true_params = get_params(
        sim, t0, T=T, l_max=l_max, n_max=n_max
    )

    fisher_matrix = get_fisher_matrix(
        qnm_list, spherical_modes, data_times, Mf_0, chif_mag_0, inv_cov=1, T=T
    )

    b_vec = get_b_vector(
        qnm_list, spherical_modes, t0, data_times, data, Mf_0, chif_mag_0, inv_cov=1, T=T
    )

    mean_vector = np.linalg.solve(fisher_matrix, b_vec)
    covariance_matrix = np.linalg.inv(fisher_matrix)

    return fisher_matrix, b_vec, mean_vector, covariance_matrix, true_params
