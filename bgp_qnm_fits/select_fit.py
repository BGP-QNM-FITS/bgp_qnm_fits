import numpy as np
import bgp_qnm_fits.qnmfits_funcs as qnmfits
import jax
import jax.numpy as jnp
import time

from bgp_qnm_fits.base_fit import Base_BGP_fit
from bgp_qnm_fits.utils import get_inverse
from bgp_qnm_fits.qnm_funcs import get_log_significance_mode
from jax.scipy.linalg import cholesky

jax.config.update("jax_enable_x64", True)


class BGP_select(Base_BGP_fit):
    """
    A class for selecting QNMs by performing an optimised version of the 
    Bayes GP fit code implemented fully in main_fit. 
    """

    def __init__(
        self,
        *args,
        t0,
        candidate_modes,
        log_threshold, 
        candidate_type="sequential",
        n_max=6,
        num_draws=1e3,
        **kwargs,
    ):
        """
        Initializes the BGP_select class.
        Args:
            *args: Positional arguments for the Base_BGP_fit class.
            t0 (float, int, list, or tuple): The time at which to fit the QNM parameters.
            use_nonlinear_params (bool): Whether to use nonlinear parameters in the fit
            (note this is slower and provides only a small improvement on the linearised approximation).
            decay_corrected (bool): Whether to apply decay correction to the amplitudes
            (note this takes a long time as it is computed for each sample).
            num_samples (int): Number of samples to draw from the posterior distribution.
            quantiles (list): List of quantiles to compute from the amplitude distribution.
            **kwargs: Keyword arguments for the Base_BGP_fit class.
        """

        super().__init__(*args, **kwargs)
        self.key = jax.random.PRNGKey(int(time.time()))
        self.log_threshold = log_threshold
        self.candidate_modes = candidate_modes
        self.n_max = n_max
        self.num_draws = int(num_draws)
        self.candidate_type = candidate_type 
        self.get_fit_at_t0(t0)

    def _get_ls_amplitudes_candidates(self, t0, Mf, chif, modes, t0_method="closest"):
        """
        Perform the least-squares fit to obtain reference amplitudes for the QNM modes.

        Args:
            t0 (float): The start time for the fitting.
            Mf (float): Reference remnant mass.
            chif (float): Reference remnant dimensionless spin.
            t0_method (str): Method for determining the start time of the fitting ("geq" or "closest").
        Returns:
            tuple: A tuple containing the complex coefficients (C_0) and the reference parameters
            (real and imaginary parts, plus mass and spin if requested).
        """

        # Perform the multimode ringdown fit using qnmfits
        ls_fit = qnmfits.multimode_ringdown_fit(
            self.times,
            self.data_dict,
            modes=modes,
            Mf=Mf,
            chif=chif,
            t0=t0,
            T=self.T,
            spherical_modes=self.spherical_modes,
            t0_method=t0_method,
        )

        # Extract the complex coefficients
        C_0 = jnp.array(ls_fit["C"], dtype=jnp.complex128)

        # Construct the reference parameters
        ref_params = jnp.concatenate([jnp.stack([jnp.real(C_0), jnp.imag(C_0)], axis=-1).reshape(-1)])
        if self.include_chif:
            ref_params = jnp.append(ref_params, chif)
        if self.include_Mf:
            ref_params = jnp.append(ref_params, Mf)

        return C_0, ref_params

    def determine_next_modes(self, current_modes, candidate_modes, n_limit_prograde, n_limit_retrograde, n_max, type="sequential"):
        possible_new_modes = []

        QQNM1 = (2,2,0,1,2,2,0,1)
        QQNM2 = (3,3,0,1,3,3,0,1)
        CQNM = (2,2,0,1,2,2,0,1,2,2,0,1) 

        if type=="sequential":

            for s in self.spherical_modes:
                if n_limit_prograde[s] + 1 <= n_max:
                    if (s[0], s[1], n_limit_prograde[s] + 1, 1) in candidate_modes:
                        possible_new_modes.append((s[0], s[1], n_limit_prograde[s] + 1, 1))
                if n_limit_retrograde[s] + 1 <= n_max:
                    if (s[0], s[1], n_limit_retrograde[s] + 1, -1) in candidate_modes:
                        possible_new_modes.append((s[0], s[1], n_limit_retrograde[s] + 1, -1))

            if (2, 2, 0, 1) in current_modes:
                if QQNM1 not in current_modes and (QQNM1 in candidate_modes):
                    possible_new_modes.append(QQNM1)
                if CQNM not in current_modes and (CQNM in candidate_modes):
                    possible_new_modes.append(CQNM)
            elif (3, 3, 0, 1) in current_modes and QQNM2 not in current_modes and (QQNM2 in candidate_modes):
                possible_new_modes.append(QQNM2)

        elif type=="all":
            for candidate_mode in candidate_modes:
                if candidate_mode not in current_modes:
                        possible_new_modes.append(candidate_mode)

        elif type=="prograde_sequential":
            for s in self.spherical_modes:
                if n_limit_prograde[s] + 1 <= n_max:
                    if (s[0], s[1], n_limit_prograde[s] + 1, 1) in candidate_modes:
                        possible_new_modes.append((s[0], s[1], n_limit_prograde[s] + 1, 1))

                for n in range(n_max + 1):
                    qnm = (s[0], s[1], n + 1, -1)
                    if qnm in candidate_modes and qnm not in current_modes:
                        possible_new_modes.append(qnm)

            if (2, 2, 0, 1) in current_modes:
                if QQNM1 not in current_modes and (QQNM1 in candidate_modes):
                    possible_new_modes.append(QQNM1)
                if CQNM not in current_modes and (CQNM in candidate_modes):
                    possible_new_modes.append(CQNM)
            elif (3, 3, 0, 1) in current_modes and QQNM2 not in current_modes and (QQNM2 in candidate_modes):
                possible_new_modes.append(QQNM2)

        # CONSTANTS 

        for candidate_mode in candidate_modes:
            if len(candidate_mode) == 2 and candidate_mode not in current_modes:
                possible_new_modes.append(candidate_mode)

        return possible_new_modes
    

    def _get_samples(self, mean_vector, covariance_matrix, num_draws):
        key, subkey = jax.random.split(self.key)
        samples = jax.random.multivariate_normal(
            subkey,
            mean_vector,
            covariance_matrix,
            shape=(num_draws),
        )
                
        return samples, key
    

    def get_model_linear(self, constant_term, mean_vector, ref_params, model_terms):
        return constant_term + np.einsum("p,stp->st", mean_vector - ref_params, model_terms)
    

    def get_expected_chi_squared(self, noise_covariance, num_draws=1e3):

        # TODO - check this is working for multimode fits + make more compact!! 

        dist_samples = np.zeros((num_draws))
        for i in range(len(self.spherical_modes)):
            normal_samples = np.random.normal(0, 1, size=(num_draws, len(noise_covariance[0])))
            eigvals = np.linalg.eigvals(noise_covariance)[i].real
            dist_samples += 2 * np.sum(eigvals * normal_samples**2, axis=1)

        return dist_samples
 
    def get_model_chi_squared(self, masked_data_array, constant_term, ref_params, model_terms, mean_vector, covariance_matrix, num_draws=1e3):
        samples, key = self._get_samples(mean_vector, covariance_matrix, num_draws)
        r_squareds = np.zeros(num_draws)
        for j in range(num_draws):
            theta_j = samples[j, :]
            sample_model = self.get_model_linear(constant_term, theta_j, ref_params, model_terms)
            residual = masked_data_array - sample_model
            r_squared = np.einsum("st, st -> ", np.conj(residual), residual).real
            r_squareds[j] = r_squared
        return r_squareds
    

    def get_fit_arrays(self, t0, model_times, masked_data_array, Mf_ref, chif_ref, modes, noise_covariance_lower_triangular):
        full_ls_amplitudes, full_ref_params = self._get_ls_amplitudes_candidates(t0, Mf_ref, chif_ref, modes)
        full_frequencies = jnp.array([self._get_frequency(mode, chif_ref, Mf_ref) for mode in modes])
        full_frequency_derivatives = jnp.array([self._get_domega_dchif(mode, chif_ref, Mf_ref) for mode in modes])
        full_mixing_coefficients = jnp.array([self._get_mixing(mode, sph_mode, chif_ref) for sph_mode in self.spherical_modes for mode in modes]).reshape(len(self.spherical_modes), len(modes))
        full_mixing_derivatives = jnp.array([self._get_dmu_dchif(mode, sph_mode, chif_ref) for sph_mode in self.spherical_modes for mode in modes]).reshape(len(self.spherical_modes), len(modes))
        full_exponential_terms = self._get_exponential_terms(model_times, full_frequencies)

        model_terms = self.get_model_terms(
            model_times,
            full_ls_amplitudes,
            full_exponential_terms,
            full_mixing_coefficients,
            full_mixing_derivatives,
            full_frequencies,
            full_frequency_derivatives,
            Mf_ref,
        )
        constant_term = self.get_const_term(full_ls_amplitudes, full_exponential_terms, full_mixing_coefficients)

        fisher_matrix = self.get_fisher_matrix(model_times, model_terms, noise_covariance_lower_triangular)
        b_vector = self.get_b_vector(
            masked_data_array,
            constant_term,
            model_times,
            model_terms,
            noise_covariance_lower_triangular,
        )

        covariance_matrix = get_inverse(fisher_matrix)
        mean_vector = jnp.linalg.solve(fisher_matrix, b_vector) + full_ref_params

        return full_ref_params, model_terms, constant_term, fisher_matrix, b_vector, covariance_matrix, mean_vector


    def get_fit_at_t0(self, t0):
        """
        Perform the fit at a specific time t0.
        Args:
            t0 (float): The time at which to perform the fit.
        Returns:
            fit (dict): A dictionary containing relevant results of the fit.
        """

        analysis_times, masked_data_array = self._mask_data(t0)
        model_times = analysis_times - t0

        chif_ref = self.chif_ref
        Mf_ref = self.Mf_ref

        noise_covariance_matrix = self.get_noise_covariance_matrix(analysis_times)
        noise_covariance_lower_triangular = cholesky(noise_covariance_matrix, lower=True)

        if len(self.modes) != 0:
            # TODO this is for a very specific trial case 
            n_limit_prograde = {s: 0 for s in self.spherical_modes}
            n_limit_retrograde = {s: 0 for s in self.spherical_modes}
        else:
            n_limit_prograde = {s: -1 for s in self.spherical_modes}
            n_limit_retrograde = {s: -1 for s in self.spherical_modes}

        modes = self.modes.copy() 

        mode_dot_products = [] 

        # TODO remove print statements at some stage 

        while True:

            log_significance = [] 
            dot_products = []

            candidate_modes_considered = self.determine_next_modes(modes, self.candidate_modes, n_limit_prograde, n_limit_retrograde, self.n_max, type=self.candidate_type)

            if len(candidate_modes_considered) == 0:
                print("Stopping: no more modes to add.")
                print("Final mode content", modes)
                break

            self.modes_length += 1
            self.params_length += 2

            for candidate_mode in candidate_modes_considered:

                try_modes = modes + [candidate_mode]

                full_ref_params, model_terms, constant_term, fisher_matrix, b_vector, \
                covariance_matrix, mean_vector = self.get_fit_arrays(
                    t0, model_times, masked_data_array, Mf_ref, chif_ref, 
                    try_modes, noise_covariance_lower_triangular
                )
                dot_product, log_s = get_log_significance_mode(candidate_mode, 
                                                                    try_modes, 
                                                                    mean_vector, 
                                                                    fisher_matrix, 
                                                                    include_chif=self.include_chif, 
                                                                    include_Mf=self.include_Mf)
                log_significance.append(log_s)
                dot_products.append(dot_product)

            max_log_sig = max(log_significance)
            mode_dot_products.append(max(dot_products))

            if max_log_sig < self.log_threshold:
                print("Stopping: no more significant modes")
                print(f"Next mode is {candidate_modes_considered[np.argmax(log_significance)]} with log significance {max_log_sig}")
                print("Final mode content", modes)
                self.modes_length -= 1
                self.params_length -= 2
                break

            best_idx = np.argmax(dot_products)
            mode_to_add = candidate_modes_considered[best_idx]
            modes.append(mode_to_add)
            print(f"Adding mode {mode_to_add} with significance {np.exp(max_log_sig)}.")

            if len(mode_to_add)==4 and mode_to_add[3]==1:
                n_limit_prograde[(mode_to_add[0], mode_to_add[1])] += 1
            elif len(mode_to_add)==4 and mode_to_add[3]==-1:
                n_limit_retrograde[(mode_to_add[0], mode_to_add[1])] += 1

        full_ref_params, model_terms, constant_term, fisher_matrix, b_vector, \
        covariance_matrix, mean_vector = self.get_fit_arrays(
            t0, model_times, masked_data_array, Mf_ref, chif_ref, 
            modes, noise_covariance_lower_triangular
        )
        expected_chi_squared = self.get_expected_chi_squared(noise_covariance_matrix, num_draws=self.num_draws)

        model_chi_squared = self.get_model_chi_squared(masked_data_array, constant_term, full_ref_params, model_terms, mean_vector, covariance_matrix, num_draws=self.num_draws)
        p_values = np.array([np.sum(expected_chi_squared < chi_sq) / self.num_draws for chi_sq in model_chi_squared])

        #model_chi_squared_mean, model_chi_squared_lower, model_chi_squared_upper = np.mean(model_chi_squared), np.percentile(model_chi_squared, 25), np.percentile(model_chi_squared, 75)
        #p_value_mean = np.sum(expected_chi_squared < model_chi_squared_mean) / self.num_draws

        sample_model = self.get_model_linear(constant_term, mean_vector, full_ref_params, model_terms)
        residual = masked_data_array - sample_model
        r_squared_mean = np.einsum("st, st -> ", np.conj(residual), residual).real
        p_value_mean = np.sum(expected_chi_squared < r_squared_mean) / self.num_draws

        self.r_squareds = model_chi_squared
        self.r_squared_mean = r_squared_mean

        self.p_values = p_values
        self.p_value_mean = p_value_mean

        self.full_modes = modes

        self.dot_products = mode_dot_products