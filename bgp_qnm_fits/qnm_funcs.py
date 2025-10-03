import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve, solve_triangular
from bgp_qnm_fits.utils import get_inverse


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

    keep_indices = jnp.array([i for i, p in enumerate(parameters) if p in parameter_choice])
    marginalize_indices = jnp.array([i for i in range(len(parameters)) if i not in keep_indices])
    marginal_mean = mean_vector[keep_indices]
    Q_11 = fisher_matrix[jnp.ix_(keep_indices, keep_indices)]
    Q_12 = fisher_matrix[jnp.ix_(keep_indices, marginalize_indices)]
    Q_22 = fisher_matrix[jnp.ix_(marginalize_indices, marginalize_indices)]
    marginal_fisher = Q_11 - Q_12 @ solve(Q_22, Q_12.T)

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
    b_a = solve_triangular(L, marginal_mean)
    return 1 - jnp.exp(-0.5 * jnp.dot(b_a, b_a))

    # if np.dot(b_a, b_a) > 3:
    #    return -np.exp(-0.5 * np.dot(b_a, b_a))
    # else:
    #    return np.log(1 - np.exp(-0.5 * np.dot(b_a, b_a)))


def get_log_significance(marginal_mean, marginal_covariance): 
    L = cholesky(marginal_covariance)
    b_a = solve_triangular(L, marginal_mean)
    dot_product = np.dot(b_a, b_a)
    if dot_product > 10:
        return dot_product, -np.exp(-0.5 * dot_product)
    else:
        return dot_product, np.log(1 - np.exp(-0.5 * dot_product))


def get_log_significance_mode(mode, qnm_list, mean_vector, fisher_matrix, include_chif=False, include_Mf=False):
        """
        Computes the LOG significance of a single mode from a fit to many modes.
        
        """
        #TODO this is the log significance ! 
        param_list = [qnm for qnm in qnm_list for _ in range(2)]
        if include_chif:
            param_list += ["chif"]
        if include_Mf:
            param_list += ["Mf"]
        marginal_mean, marginal_fisher = marginalise([mode], param_list, mean_vector, fisher_matrix)
        marginal_covariance = get_inverse(marginal_fisher)
        return get_log_significance(marginal_mean, marginal_covariance)


def get_significance_list(qnm_list, mean_vector, fisher_matrix, include_chif=False, include_Mf=False):
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
    if include_chif:
        param_list += ["chif"]
    if include_Mf:
        param_list += ["Mf"]

    sig_list = []
    for qnm in qnm_list:
        marginal_mean, marginal_fisher = marginalise([qnm], param_list, mean_vector, fisher_matrix)
        marginal_covariance = get_inverse(marginal_fisher)
        significance = get_significance(marginal_mean, marginal_covariance)
        sig_list.append(significance)
    return sig_list


def find_worst_qnm(qnm_list, mean_vector, fisher_matrix):
    """Takes a list of QNMs and returns the QNM with the lowest significance and corresponding
    significance value."""
    sig_list = get_significance_list(qnm_list, mean_vector, fisher_matrix)
    filtered_qnm_sig_pairs = [(qnm, sig) for qnm, sig in zip(qnm_list, sig_list) if not np.isnan(sig)]
    filtered_qnm_sig_pairs.sort(key=lambda pair: pair[1])
    worst_qnm, worst_sig = filtered_qnm_sig_pairs[0]
    return worst_sig, worst_qnm


def recursive_qnm_finder(qnm_list_initial, b_vec, fisher_matrix, threshold_sig=np.log(0.9)):
    """This function recursively removes the QNM with the lowest significance until all QNMs
    have a significance above a threshold."""

    fisher_matrix_init = fisher_matrix.copy()
    b_vec_init = b_vec.copy()
    param_list_init = [qnm for qnm in qnm_list_initial for _ in range(2)]
    qnm_list_reduced = qnm_list_initial.copy()

    sig_val = -100

    while sig_val < threshold_sig:
        mask = np.array([qnm in qnm_list_reduced for qnm in param_list_init])
        fisher_matrix_reduced = fisher_matrix_init[mask][:, mask]
        b_vec_reduced = b_vec_init[mask]
        mean_vector_reduced = np.linalg.solve(fisher_matrix_reduced, b_vec_reduced)
        sig_val, worst_qnm = find_worst_qnm(qnm_list_reduced, mean_vector_reduced, fisher_matrix_reduced)
        if sig_val < threshold_sig:
            qnm_list_reduced.remove(worst_qnm)
            print(f"{worst_qnm} removed with significance {sig_val}")

    return qnm_list_reduced


def recursive_qnm_finder_ordered(qnm_list_initial, b_vec, fisher_matrix, qnm_order, threshold_sig=np.log(0.9)):
    fisher_matrix_init = fisher_matrix.copy()
    b_vec_init = b_vec.copy()
    param_list_init = [qnm for qnm in qnm_list_initial for _ in range(2)]

    qnm_list_reduced = qnm_list_initial.copy()
    qnm_order_reduced = qnm_order.copy()

    sig_val = -100

    while sig_val < threshold_sig:
        mask = np.array([qnm in qnm_list_reduced for qnm in param_list_init])
        fisher_matrix_reduced = fisher_matrix_init[mask][:, mask]
        b_vec_reduced = b_vec_init[mask]
        mean_vector_reduced = np.linalg.solve(fisher_matrix_reduced, b_vec_reduced)

        qnm_choice = qnm_order_reduced[0]
        param_list_reduced = [qnm for qnm in qnm_list_reduced for _ in range(2)]
        marginal_mean, marginal_fisher = marginalise(
            [qnm_choice], param_list_reduced, mean_vector_reduced, fisher_matrix_reduced
        )
        marginal_covariance = get_inverse(marginal_fisher)
        sig_val = get_significance(marginal_mean, marginal_covariance)

        if np.isnan(sig_val):
            continue

        if sig_val < threshold_sig:
            qnm_list_reduced.remove(qnm_choice)
            qnm_order_reduced.remove(qnm_choice)
            print(f"{qnm_choice} removed with significance {sig_val}")

    return qnm_list_reduced, qnm_order_reduced


def get_qnm_timeseries(
    intital_qnm_list,
    spherical_modes,
    t0_list,
    data_times,
    data,
    Mf_0,
    chif_mag_0,
    tuned_params_lm,
    kernel,
    T=100,
    threshold_sig=np.log(0.9),
    qnm_ordering=None,
):
    """
    Gets timeseries of QNMs 'in the model'.

    After each timestep, modes that fall below the significance threshold are removed from the
    model. Returns a list of lists of QNMs at each timestep.

    If an ordering is given (e.g. overtones highest to lowest), qnms are considered in that order.
    If the mode that is highest in the list does not fall below significance, then the time is stepped
    forward regardless of the significance of the other modes.

    """

    qnm_list_timeseries = []
    qnm_list_new = intital_qnm_list.copy()
    qnm_order_reduced = qnm_ordering.copy() if qnm_ordering is not None else None

    for t0 in t0_list:
        print(f"t0 = {t0}")
        fit = qnm_BGP_fit(
            data_times,
            data,
            qnm_list_new,
            Mf_0,
            chif_mag_0,
            t0,
            tuned_params_lm,
            kernel,
            T=T,
            spherical_modes=spherical_modes,
        )
        if qnm_ordering is None:
            qnm_list_new = recursive_qnm_finder(qnm_list_new, fit["b_vector"], fit["fisher_matrix"], threshold_sig)
        else:
            qnm_list_new, qnm_order_reduced = recursive_qnm_finder_ordered(
                qnm_list_new,
                fit["b_vector"],
                fit["fisher_matrix"],
                qnm_order_reduced,
                threshold_sig,
            )

        qnm_list_timeseries.append(qnm_list_new)

        if qnm_order_reduced == [] or qnm_list_new == []:  # TODO: This will break when there's only one element
            break

    return qnm_list_timeseries
