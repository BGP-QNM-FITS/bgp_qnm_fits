"""This module contains functions for computing the Fisher matrix and b vector for a given set of
QNMs and spherical modes.

It also includes functions for marginalising over parameters and computing the significance of
parameters.
"""

import numpy as np
import qnmfits
from funcs.GP_funcs import *


def domega_dchif(ell, m, n, p, Mf_0, chif_0, delta=0.01):
    """Compute domega/dchif for a given QNM."""
    chifs = np.linspace(chif_0 - delta, chif_0 + delta, 11)
    omegas = qnmfits.qnm.omega(ell, m, n, p, chifs, Mf=Mf_0)
    df = np.gradient(omegas, chifs)
    return df[5]


def dmu_dchif(lp, mp, ell, m, n, p, chif_0, delta=0.01):
    """Compute dmu/dchif for a given QNM."""
    chifs = np.linspace(chif_0 - delta, chif_0 + delta, 11)
    mus = qnmfits.qnm.mu(lp, mp, ell, m, n, p, chifs)
    if isinstance(mus, int):
        mus = np.zeros(11)
    df = np.gradient(mus, chifs)
    return df[5]


def const_model_term_generator(qnms, spherical_modes, t_list, C_0, Mf_0, chif_mag_0):
    """Computes the H_* term for a given set of QNMs. Returns a len(spherical_modes) x len(t_list)
    array."""
    lm_matrix = np.zeros((len(spherical_modes), len(t_list)), dtype=complex)
    omegas = qnmfits.qnm.omega_list(
        qnms, chif_mag_0, Mf_0
    )  # Note this returns omega / Mf
    for i, mode in enumerate(spherical_modes):
        mus = qnmfits.qnm.mu_list([mode + indices for indices in qnms], chif_mag_0)
        lm_matrix[i] = sum(
            C_0[j] * mus[j] * np.exp(-1j * omegas[j] * t_list)
            for j, indices in enumerate(qnms)
        )
    return lm_matrix


def re_amplitude_model_term_generator(
    indices, spherical_modes, t_list, chif_mag_0, omega
):
    """Computes the real part of the amplitude model term for a given set of QNMs. Returns a
    len(spherical_modes) x len(t_list) array."""
    return [
        np.exp(-1j * omega * t_list)
        * qnmfits.qnm.mu_list([mode + indices], chif_mag_0)[0]
        for mode in spherical_modes
    ]


def im_amplitude_model_term_generator(
    indices, spherical_modes, t_list, chif_mag_0, omega
):
    """Computes the imaginary part of the amplitude model term for a given set of QNMs. Returns a
    len(spherical_modes) x len(t_list) array."""
    return [
        1j
        * np.exp(-1j * omega * t_list)
        * qnmfits.qnm.mu_list([mode + indices], chif_mag_0)[0]
        for mode in spherical_modes
    ]


def mass_model_term_generator(
    qnms, spherical_modes, t_list, C_0, Mf_0, chif_mag_0, omegas
):
    """Computes the Mf term for a given set of QNMs. Returns a len(spherical_modes) x len(t_list)
    array."""
    lm_matrix = np.zeros((len(spherical_modes), len(t_list)), dtype=complex)
    for i, mode in enumerate(spherical_modes):
        mus = qnmfits.qnm.mu_list([mode + indices for indices in qnms], chif_mag_0)
        lm_matrix[i] = sum(
            C_0[i]
            * np.exp(-1j * omegas[i] * t_list)
            * (1j * mus[i] * omegas[i] * t_list)
            / Mf_0
            for i in range(len(qnms))
        )
    return lm_matrix


def chif_model_term_generator(
    qnms, spherical_modes, t_list, C_0, Mf_0, chif_mag_0, omegas
):
    """Computes the chif_mag term for a given set of QNMs. Returns a len(spherical_modes) x
    len(t_list) array."""
    lm_matrix = np.zeros((len(spherical_modes), len(t_list)), dtype=complex)
    for i, mode in enumerate(spherical_modes):
        lp, mp = mode
        term = np.zeros(len(t_list), dtype=complex)
        for j, indices in enumerate(qnms):
            l, m, n, p = indices
            term += (
                C_0[j]
                * np.exp(-1j * omegas[j] * t_list)
                * (
                    dmu_dchif(lp, mp, l, m, n, p, chif_mag_0)
                    - 1j
                    * qnmfits.qnm.mu_list([mode + indices], chif_mag_0)[0]
                    * t_list
                    * domega_dchif(l, m, n, p, Mf_0, chif_mag_0)
                )
            )
        lm_matrix[i] = term
    return lm_matrix


def precompute_dict(
    param_list, qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0
):
    """This precomputes the terms in the linearised model for each parameter in param_list.

    Returns a len(param_list) x len(spherical_modes) x len(analysis_times) array.
    """

    sph_matrix = np.zeros(
        (len(param_list), len(spherical_modes), len(analysis_times)), dtype=complex
    )

    omegas = qnmfits.qnm.omega_list(qnm_list, chif_mag_0, Mf_0)  # This is omega / Mf

    for i, param in enumerate(param_list):
        if param == "chif":
            list = chif_model_term_generator(
                qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0, omegas
            )
        elif param == "Mf":
            list = mass_model_term_generator(
                qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0, omegas
            )
        elif i % 2 == 0:
            indices = param
            list = re_amplitude_model_term_generator(
                indices, spherical_modes, analysis_times, chif_mag_0, omegas[i // 2]
            )
        else:
            indices = param
            list = im_amplitude_model_term_generator(
                indices, spherical_modes, analysis_times, chif_mag_0, omegas[i // 2]
            )

        sph_matrix[i] = list

    return sph_matrix


def get_element(array1, array2, analysis_times, covariance_matrix):
    """
    This computes a single element of the Fisher matrix corresponding to dict1 and dict2.

    Returns a real scalar.
    """

    #return np.real(
    #    np.einsum("bi,bj,bij->", array1, array2, covariance_matrix)
    #    / (analysis_times[-1] - analysis_times[0]) 
    #)

    if np.allclose(covariance_matrix[0], np.diag(np.diagonal(covariance_matrix[0]))): # should really make this true for every mode
        return np.real(
            np.einsum("bi,bj,bij->", array1, array2, covariance_matrix)
            * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
        )
    else:
        return np.real(
            np.einsum("bi,bj,bij->", array1, array2, covariance_matrix)
        )
 
    #return np.real(
    #    np.einsum("bi,bj,bij->", array1, array2, covariance_matrix)
    #    * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
    #)


def get_fisher_matrix(
    qnm_list,
    spherical_modes,
    analysis_times,
    Mf_0,
    chif_mag_0,
    covariance_matrix,
    C_0=None,
    include_chif=False,
    include_Mf=False,
):
    """This computes the Fisher matrix for the parameters in param_list.

    Returns a len(param_list) x len(param_list) array.
    """

    param_list = [qnm for qnm in qnm_list for _ in range(2)]
    if include_chif:
        param_list = param_list + ["chif"]
    if include_Mf:
        param_list = param_list + ["Mf"]

    # C_0 only needed if chif or Mf are included. Otherwise use a dummy value.
    if not (include_chif or include_Mf):
        C_0 = [0] * len(qnm_list)

    fisher_matrix = np.zeros((len(param_list), len(param_list)))
    sph_matrix = precompute_dict(
        param_list, qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0
    )

    matrix1 = np.conj(sph_matrix.copy())
    matrix2 = sph_matrix.copy()

    for i in range(len(param_list)):
        for j in range(i + 1):
            element = get_element(
                matrix1[i, :, :], matrix2[j, :, :], analysis_times, covariance_matrix
            )
            fisher_matrix[i, j] = element
            if i != j:
                fisher_matrix[j, i] = element

    return fisher_matrix


def get_b_vector(
    qnm_list,
    spherical_modes,
    analysis_times,
    data,  # this must be masked in advance
    Mf_0,
    chif_mag_0,
    covariance_matrix,
    C_0=None,
    include_chif=False,
    include_Mf=False,
):
    """This computes the b vector for the parameters in param_list.

    Returns a len(param_list) array.
    """

    param_list = [qnm for qnm in qnm_list for _ in range(2)]
    if include_chif:
        param_list = param_list + ["chif"]
    if include_Mf:
        param_list = param_list + ["Mf"]

    # C_0 only needed if chif or Mf are included. Otherwise use a dummy value.
    if not (include_chif or include_Mf):
        C_0 = [0] * len(qnm_list)
        h_0 = np.zeros((len(spherical_modes), len(analysis_times)))
    else:
        print(
            "Note that the mean vector is now cov_matrix @ b PLUS theta_0 (the true parameter values)"
        )
        h_0 = const_model_term_generator(
            qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0
        )

    b_vector = np.zeros((len(param_list)))
    sph_matrix = precompute_dict(
        param_list, qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0
    )

    data_array = np.array([np.array(data[mode]) for mode in spherical_modes])
    data_array_new = data_array - h_0

    for i in range(len(param_list)):
        element = get_element(
            np.conj(sph_matrix[i, :, :]),
            data_array_new,
            analysis_times,
            covariance_matrix,
        )
        b_vector[i] = element

    return b_vector


def qnm_BGP_fit(
    times,
    data_dict,
    modes,
    Mf,
    chif,
    t0,
    kernel_param_dict,
    kernel,
    t0_method="geq",
    T=100,
    spherical_modes=None,
):

    # Use the requested spherical modes
    if spherical_modes is None:
        spherical_modes = list(data_dict.keys())

    epsilon = 1e-9

    # Mask the data with the requested method
    if t0_method == "geq":

        data_mask = (times >= t0 - epsilon) & (times < t0 + T)

        times = times[data_mask]
        data = np.concatenate([data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {lm: data_dict[lm][data_mask] for lm in spherical_modes}

    elif t0_method == "closest":

        start_index = np.argmin((times - t0) ** 2)
        end_index = np.argmin((times - t0 - T) ** 2)

        times = times[start_index:end_index]
        data = np.concatenate(
            [data_dict[lm][start_index:end_index] for lm in spherical_modes]
        )
        data_dict_mask = {
            lm: data_dict[lm][start_index:end_index] for lm in spherical_modes
        }

    else:
        print(
            """Requested t0_method is not valid. Please choose between
              'geq' and 'closest'."""
        )

    model_times = times - t0

    # Frequencies
    # -----------

    frequencies = np.array(qnmfits.qnm.omega_list(modes, chif, Mf))

    # Noise covariance matrix
    # -----------

    noise_covariance_matrix = get_GP_covariance_matrix(
        times, kernel, kernel_param_dict, spherical_modes
    )

    # Mean & covariance calculations 
    # -----------

    fisher_matrix = get_fisher_matrix(
        modes, spherical_modes, model_times, Mf, chif, noise_covariance_matrix
    )

    b_vector = get_b_vector(
        modes,
        spherical_modes,
        model_times,
        data_dict_mask,
        Mf,
        chif,
        noise_covariance_matrix,
    )

    mean_vector = np.linalg.solve(fisher_matrix, b_vector)
    covariance_matrix = np.linalg.inv(fisher_matrix)

    labels = [str(mode) for mode in modes]

    # Store all useful information to a output dictionary
    best_fit = {
        "mean": mean_vector,
        "covariance": covariance_matrix,
        "fisher_matrix": fisher_matrix,
        "b_vector": b_vector,
        "noise_covariance": noise_covariance_matrix,
        "data": data_dict,
        "model_times": times,
        "t0": t0,
        "modes": modes,
        "mode_labels": labels,
        "frequencies": frequencies,
    }

    # Return the output dictionary
    return best_fit
