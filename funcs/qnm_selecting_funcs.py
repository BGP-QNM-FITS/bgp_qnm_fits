"""This module contains functions for selecting significant QNMs in a model."""

import numpy as np
from likelihood_funcs import *
from GP_funcs import *
from utils import *
from kernel_param_funcs import *


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


def get_significance_list(qnm_list, mean_vector, fisher_matrix):
    """This computes the significance of the marginalised parameters in qnm_list_new.

    Inputs:

    qnm_list : list
        The list of QNMs to compute the significance of (this includes QNMs only
        once as both the real and imaginary part are used to compute the significance,
        in contrast to the param_list which includes each QNM twice).
    mean_vector : np.array
        The mean vector of all parameters.
    fisher_matrix : np.array
        The Fisher matrix of all parameters.

    Returns:
    sig_list : list
        The list of significances of the marginalised parameters in qnm_list_new.
    """

    param_list = [qnm for qnm in qnm_list for _ in range(2)]

    sig_list = []
    for qnm in qnm_list:
        marginal_mean, marginal_fisher = marginalise(
            [qnm], param_list, mean_vector, fisher_matrix
        )
        marginal_covariance = np.linalg.inv(marginal_fisher)
        try:
            significance = get_significance(marginal_mean, marginal_covariance)
        except np.linalg.LinAlgError as e:
            # This is just to catch rare instances where Cholesky decomposition fails.
            print(f"Cholesky decomposition failed for {qnm}: {e}")
            significance = np.nan
        sig_list.append(significance)
    return sig_list


def find_worst_qnm(qnm_list, mean_vector, fisher_matrix):
    """Takes a list of QNMs and returns the QNM with the lowest significance and corresponding
    significance value."""
    sig_list = get_significance_list(qnm_list, mean_vector, fisher_matrix)
    filtered_qnm_sig_pairs = [
        (qnm, sig) for qnm, sig in zip(qnm_list, sig_list) if not np.isnan(sig)
    ]
    filtered_qnm_sig_pairs.sort(key=lambda pair: pair[1])
    worst_qnm, worst_sig = filtered_qnm_sig_pairs[0]
    return worst_sig, worst_qnm


def recursive_qnm_finder(
    qnm_list_initial, b_vec, fisher_matrix, threshold_sig=np.log(0.9)
):
    """This function recursively removes the QNM with the lowest significance until all QNMs have a
    significance above a threshold."""

    fisher_matrix_init = fisher_matrix.copy()
    b_vec_init = b_vec.copy()
    param_list_init = [qnm for qnm in qnm_list_initial for _ in range(2)]
    qnm_list_reduced = qnm_list_initial.copy()

    sig_val = -1

    while sig_val < threshold_sig:
        mask = np.array([qnm in qnm_list_reduced for qnm in param_list_init])
        fisher_matrix_reduced = fisher_matrix_init[mask][:, mask]
        b_vec_reduced = b_vec_init[mask]
        mean_vector_reduced = np.linalg.solve(fisher_matrix_reduced, b_vec_reduced)
        sig_val, worst_qnm = find_worst_qnm(
            qnm_list_reduced, mean_vector_reduced, fisher_matrix_reduced
        )
        if sig_val < threshold_sig:
            qnm_list_reduced.remove(worst_qnm)
            print(f"{worst_qnm} removed with significance {sig_val}")

    return qnm_list_reduced


def get_qnm_timeseries(
    intital_qnm_list,
    spherical_modes,
    t0_list,
    data_times,
    data,
    Mf_0,
    chif_mag_0,
    inv_cov,
    T=100,
    threshold_sig=np.log(0.9),
):
    """Gets timeseries of QNMs 'in the model'.

    After each timestep, modes that fall below the significance threshold are removed from the
    model. Returns a list of lists of QNMs at each timestep.
    """

    qnm_list_timeseries = []
    qnm_list_new = intital_qnm_list.copy()

    for t0 in t0_list:
        print(f"t0 = {t0}")
        fisher_matrix = get_fisher_matrix(
            qnm_list_new,
            spherical_modes,
            t0,
            data_times,
            Mf_0,
            chif_mag_0,
            inv_cov,
            T=T,
        )
        b_vec = get_b_vector(
            qnm_list_new,
            spherical_modes,
            t0,
            data_times,
            data,
            Mf_0,
            chif_mag_0,
            inv_cov,
            T=T,
        )
        qnm_list_new = recursive_qnm_finder(
            qnm_list_new, b_vec, fisher_matrix, threshold_sig=threshold_sig
        )
        qnm_list_timeseries.append(qnm_list_new)

    return qnm_list_timeseries
