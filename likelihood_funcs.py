"""This module contains functions for computing the Fisher matrix and b vector for a given set of
QNMs and spherical modes.

It also includes functions for marginalising over parameters and computing the significance of
parameters.
"""

import numpy as np
import qnmfits
import time
from scipy.linalg import cholesky


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
    omegas = qnmfits.qnm.omega_list(qnms, chif_mag_0, Mf_0)  # Note this returns omega / Mf
    for i, mode in enumerate(spherical_modes):
        mus = qnmfits.qnm.mu_list([mode + indices for indices in qnms], chif_mag_0)
        lm_matrix[i] = sum(
            C_0[j] * mus[j] * np.exp(-1j * omegas[j] * t_list) for j, indices in enumerate(qnms)
        )
    return lm_matrix


def re_amplitude_model_term_generator(indices, spherical_modes, t_list, chif_mag_0, omega):
    """Computes the real part of the amplitude model term for a given set of QNMs. Returns a
    len(spherical_modes) x len(t_list) array."""
    return [
        np.exp(-1j * omega * t_list) * qnmfits.qnm.mu_list([mode + indices], chif_mag_0)[0]
        for mode in spherical_modes
    ]


def im_amplitude_model_term_generator(indices, spherical_modes, t_list, chif_mag_0, omega):
    """Computes the imaginary part of the amplitude model term for a given set of QNMs. Returns a
    len(spherical_modes) x len(t_list) array."""
    return [
        1j * np.exp(-1j * omega * t_list) * qnmfits.qnm.mu_list([mode + indices], chif_mag_0)[0]
        for mode in spherical_modes
    ]


def mass_model_term_generator(qnms, spherical_modes, t_list, C_0, Mf_0, chif_mag_0, omegas):
    """Computes the Mf term for a given set of QNMs. Returns a len(spherical_modes) x len(t_list)
    array."""
    lm_matrix = np.zeros((len(spherical_modes), len(t_list)), dtype=complex)
    for i, mode in enumerate(spherical_modes):
        mus = qnmfits.qnm.mu_list([mode + indices for indices in qnms], chif_mag_0)
        lm_matrix[i] = sum(
            C_0[i] * np.exp(-1j * omegas[i] * t_list) * (1j * mus[i] * omegas[i] * t_list) / Mf_0
            for i in range(len(qnms))
        )
    return lm_matrix


def chif_model_term_generator(qnms, spherical_modes, t_list, C_0, Mf_0, chif_mag_0, omegas):
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


def precompute_dict(param_list, qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0):
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


def get_element(array1, array2, analysis_times, inv_cov):
    """This computes a single element of the Fisher matrix corresponding to dict1 and dict2.

    Returns a real scalar.
    """

    # TODO: Once testing is finished, default inv_cov to id matrix if None to remove integration step

    if isinstance(inv_cov, (int, float)):
        product = array1 * array2
        int_re = np.trapz(np.real(product), axis=1, x=analysis_times) / inv_cov**2
        element = np.sum(int_re)
    else:
        element = np.real(
            np.einsum("bi,bj,bij->", array1, array2, inv_cov)
            * (analysis_times[1] - analysis_times[0])
        )

    return element


def get_fisher_matrix(
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
):
    """This computes the Fisher matrix for the parameters in param_list.

    Returns a len(param_list) x len(param_list) array.
    """

    epsilon = 1e-9

    analysis_times = data_times[(data_times >= t0 - epsilon) & (data_times < t0 + T)] - t0

    param_list = [qnm for qnm in qnm_list for _ in range(2)]
    if include_chif:
        param_list = param_list + ["chif"]
    if include_Mf:
        param_list = param_list + ["Mf"]

    # C_0 only needed if chif or Mf are included. Otherwise use a dummy value.
    if not (include_chif or include_Mf):
        C_0 = [0] * len(qnm_list)

    fisher_matrix = np.zeros((len(param_list), len(param_list)))
    start_time = time.time()
    sph_matrix = precompute_dict(
        param_list, qnm_list, spherical_modes, analysis_times, C_0, Mf_0, chif_mag_0
    )
    print("Precomputation time: ", time.time() - start_time)

    matrix1 = np.conj(sph_matrix.copy())
    matrix2 = sph_matrix.copy()

    start_time = time.time()
    for i in range(len(param_list)):
        for j in range(i + 1):
            element = get_element(matrix1[i, :, :], matrix2[j, :, :], analysis_times, inv_cov)
            fisher_matrix[i, j] = element
            if i != j:
                fisher_matrix[j, i] = element
    print("Fisher matrix computation time: ", time.time() - start_time)

    return fisher_matrix


def get_b_vector(
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
):
    """This computes the b vector for the parameters in param_list.

    Returns a len(param_list) array.
    """

    epsilon = 1e-9

    analysis_times = data_times[(data_times >= t0 - epsilon) & (data_times < t0 + T)] - t0

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

    data_mask = (data_times >= t0 - epsilon) & (data_times < t0 + T)
    data_array_new = data_array[:, data_mask] - h_0

    for i in range(len(param_list)):
        element = get_element(
            np.conj(sph_matrix[i, :, :]), data_array_new, analysis_times, inv_cov
        )
        b_vector[i] = element

    return b_vector


def marginalise(parameter_choice, parameters, mean_vector, fisher_matrix):
    """This marginalises over the parameters in parameter_choice.

    Inputs:

    parameter_choice : list
        The parameters to marginalise.
    parameters : list
        The full list of parameters (i.e. each QNM repeated for the real and imaginary cpt + mass and spin)
    mean_vector : np.array
        The mean vector of all parameters.
    fisher_matrix : np.array
        The Fisher matrix of all parameters.

    Returns:
    marginal_mean : np.array
        The mean vector of the marginalised parameters.
    marginal_fisher : np.array
        The Fisher matrix of the marginalised parameters.
    """

    keep_indices = [i for i, p in enumerate(parameters) if p in parameter_choice]
    marginalize_indices = [i for i in range(len(parameters)) if i not in keep_indices]
    marginal_mean = mean_vector[keep_indices]
    Q_11 = fisher_matrix[np.ix_(keep_indices, keep_indices)]
    Q_12 = fisher_matrix[np.ix_(keep_indices, marginalize_indices)]
    Q_22 = fisher_matrix[np.ix_(marginalize_indices, marginalize_indices)]
    marginal_fisher = Q_11 - Q_12 @ np.linalg.solve(Q_22, Q_12.T)

    return marginal_mean, marginal_fisher


def get_significance(marginal_mean, marginal_covariance):
    """This computes the significance of the marginalised parameters.

    Inputs:

    marginal_mean : np.array
        The mean vector of the marginalised parameters.
    marginal_covariance : np.array
        The covariance matrix of the marginalised parameters.

    Returns:
    significance : float
        The significance of the marginalised parameters.
    """

    # TODO: Fine tune b_a threshold for significance

    L = cholesky(marginal_covariance)
    b_a = -np.dot(np.linalg.inv(L), marginal_mean)

    if np.dot(b_a, b_a) > 3:
        return -np.exp(-0.5 * np.dot(b_a, b_a))
    else:
        return np.log(1 - np.exp(-0.5 * np.dot(b_a, b_a)))
