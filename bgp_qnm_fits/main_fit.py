import numpy as np
import bgp_qnm_fits.qnmfits_funcs as qnmfits
import jax
import jax.numpy as jnp
import time

from bgp_qnm_fits.base_fit import Base_BGP_fit
from bgp_qnm_fits.utils import get_inverse
from tqdm import tqdm
from jax.scipy.linalg import cholesky

jax.config.update("jax_enable_x64", True)


class BGP_fit(Base_BGP_fit):
    """
    A class for fitting QNM parameters to data using Bayesian inference.
    This class extends the Base_BGP_fit class and provides methods for fitting QNM parameters
    at a specific time or list of times.
    """

    def __init__(
        self,
        *args,
        t0,
        use_nonlinear_params=False,
        decay_corrected=False,
        num_samples=10000,
        quantiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        **kwargs,
    ):
        """
        Initializes the BGP_fit class.
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
        self.num_samples = num_samples
        self.quantiles = quantiles
        self.use_nonlinear_params = use_nonlinear_params
        self.decay_corrected = decay_corrected

        # Decide whether to perform the fit once or loop over multiple t0 values.
        # In the latter case, the fits will be stored in a list.
        if isinstance(t0, (float, int)):
            self.fit = self.get_fit_at_t0(t0)
        elif isinstance(t0, (list, tuple, np.ndarray)):
            self.fits = []
            for t0_val in tqdm(t0, desc="Fitting at t0 values"):
                self.fits.append(self.get_fit_at_t0(t0_val))
        else:
            raise ValueError("t0 must be a float, int, list or tuple")

    def _get_samples(self, mean_vector, covariance_matrix):
        """
        Draw samples from a multivariate normal distribution with the given mean vector and covariance matrix.
        Args:
            mean_vector (array): The mean vector of the multivariate normal distribution.
            covariance_matrix (array): The covariance matrix of the multivariate normal distribution.
        Returns:
            samples (array): Samples drawn from the multivariate normal distribution.
            key (jax.random.PRNGKey): The updated random key after sampling.
        """
        key, subkey = jax.random.split(self.key)
        samples = jax.random.multivariate_normal(
            subkey,
            mean_vector,
            covariance_matrix,
            shape=(self.num_samples),
        )
                
        return samples, key

    def linearised_frequency(self, mode, chif_sample, Mf_sample):
        i = self.modes.index(mode)
        return (
            self.frequencies[i]
            - (Mf_sample - self.Mf_ref) * self.chif_ref / self.Mf_ref
            + (chif_sample - self.chif_ref) * self.frequency_derivatives[i]
        )

    def get_amplitude_phase(self, mean, samples, t0):
        """
        Compute the amplitude and phase of the QNM parameters from the mean and samples.
        Args:
            mean (array): The mean vector of the parameters.
            samples (array): Samples drawn from the posterior distribution.
            t0 (float): The time at which the fit is performed.
        Returns:
            mean_amplitude (array): The mean amplitude of the QNM parameters.
            mean_phase (array): The mean phase of the QNM parameters.
            sample_amplitudes (array): The amplitudes of the samples.
            sample_phases (array): The phases of the samples.
            samples_weights (array): Weights for the samples for using `Prior 2'
            neff (float): Effective number of samples after reweighting.
        """

        num_amplitude_params = self.modes_length * 2

        mean_re, mean_im = mean[:num_amplitude_params:2], mean[1:num_amplitude_params:2]
        samples_re, samples_im = (
            samples[:, :num_amplitude_params:2],
            samples[:, 1:num_amplitude_params:2],
        )

        mean_amplitude = jnp.sqrt(mean_re**2 + mean_im**2)
        mean_phase = jnp.arctan2(mean_im, mean_re)

        sample_amplitudes = jnp.sqrt(samples_re**2 + samples_im**2)
        sample_phases = jnp.arctan2(samples_im, samples_re)

        if self.decay_corrected:
            decay_corrected_sample_amplitudes = jnp.zeros_like(sample_amplitudes)

            if self.include_chif:
                chif_samples = samples[:, -2]
            else:
                chif_samples = jnp.full((self.num_samples,), self.chif_ref)
            if self.include_Mf:
                Mf_samples = samples[:, -1]
            else:
                Mf_samples = jnp.full((self.num_samples,), self.Mf_ref)

            for i, mode in enumerate(self.modes):
                decay_times = self.linearised_frequency(mode, chif_samples, Mf_samples).imag
                correction_factors = jnp.exp(-decay_times * t0)
                decay_corrected_sample_amplitudes = decay_corrected_sample_amplitudes.at[:, i].set(
                    sample_amplitudes[:, i] * correction_factors
                )
            sample_amplitudes = decay_corrected_sample_amplitudes

        log_samples = jnp.log(sample_amplitudes)
        samples_weights = jnp.exp(-jnp.sum(log_samples, axis=1))

        neff = jnp.sum(samples_weights) ** 2 / jnp.sum(samples_weights**2)

        # print(self.num_samples)

        return (mean_amplitude, mean_phase, sample_amplitudes, sample_phases, samples_weights, neff)

    def weighted_quantile(self, values, quantiles, weights=None):
        """
        Compute the weighted quantiles of a set of values.
        Args:
            values (array): The values for which to compute the quantiles.
            quantiles (array): The quantiles to compute.
            weights (array, optional): Weights for the values. If None, equal weights are assumed and
            this function just reverts to a classic quantile caculator.
        Returns:
            quantile_values (array): The computed quantiles for each value.
        """

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

    def get_amplitude_quantiles(self, sample_amplitudes, quantiles, samples_weights=None):
        """
        Compute the quantiles of the absolute amplitudes of the samples.
        Args:
            sample_amplitudes (array): The amplitudes of the samples.
            quantiles (list): The quantiles to compute.
            samples_weights (array, optional): Weights for the samples. If None, equal weights are assumed.
        Returns:
            weighted_abs_amplitude_quantiles_dict (dict): A dictionary where keys are quantiles and values are the
            corresponding quantile values for the absolute amplitudes.
        """
        weighted_abs_amplitude_quantiles = self.weighted_quantile(sample_amplitudes, quantiles, weights=samples_weights)
        weighted_abs_amplitude_quantiles_dict = {
            quantiles: weighted_abs_amplitude_quantiles[i] for i, quantiles in enumerate(quantiles)
        }
        return weighted_abs_amplitude_quantiles_dict

    def get_model_nonlinear(self, mean_vector, analysis_times, Mf_val, chif_val):
        """
        Compute the nonlinear model array based on the mean vector and analysis times.
        Args:
            mean_vector (array): The mean vector of the parameters.
            analysis_times (array): The times at which the model is evaluated.
            Mf_val (float): The mass parameter for the model.
            chif_val (float): The dimensionless spin parameter for the model.
        Returns:
            model_array (array): The model array.
        """

        if self.include_chif and self.include_Mf:
            chif = mean_vector[-2]
            Mf = mean_vector[-1]
        elif self.include_chif:
            chif = mean_vector[-1]
            Mf = Mf_val
        elif self.include_Mf:
            chif = chif_val
            Mf = mean_vector[-1]
        else:
            chif = chif_val
            Mf = Mf_val

        frequencies = qnmfits.qnm.omega_list(self.modes, chif, Mf=Mf)
        indices_lists = [[lm_mode + mode for mode in self.modes] for lm_mode in self.spherical_modes]
        mu_lists = [qnmfits.qnm.mu_list(indices, chif) for indices in indices_lists]

        a = np.concatenate(
            [
                np.array(
                    [mu_lists[i][j] * np.exp(-1j * frequencies[j] * analysis_times) for j in range(len(frequencies))]
                ).T
                for i in range(len(self.spherical_modes))
            ]
        )

        # Create an array of complex values
        C = mean_vector[: 2 * len(self.modes) : 2] + 1j * mean_vector[1 : 2 * len(self.modes) : 2]

        # Evaluate the model
        model = jnp.einsum("ij,j->i", a, C)

        model_array = jnp.array(
            [model[i * len(analysis_times) : (i + 1) * len(analysis_times)] for i in range(self.spherical_modes_length)]
        )

        return model_array

    def get_model_linear(self, constant_term, mean_vector, ref_params, model_terms):
        """
        Compute the linear model array based on the mean vector and model terms.
        Args:
            constant_term (float): The constant term in the model.
            mean_vector (array): The mean vector of the parameters.
            ref_params (array): Reference parameters for the model.
            model_terms (array): The model terms.
        Returns:
            model_array (array): The linear model array.
        """
        return constant_term + jnp.einsum("p,stp->st", mean_vector - ref_params, model_terms)
    

    def get_expected_chi_squared(self, noise_covariance):
        eigvals = np.linalg.eigvals(noise_covariance)[0].real
        normal_samples = np.random.normal(0, 1, size=(self.num_samples, len(eigvals)))
        dist_samples = 2 * np.sum(eigvals * normal_samples**2, axis=1)
        return dist_samples 
    

    def get_model_chi_squared(self, masked_data_array, constant_term, ref_params, model_terms, mean_vector, covariance_matrix):
        samples, key = self._get_samples(mean_vector, covariance_matrix)
        r_squareds = np.zeros(self.num_samples)
        for j in range(self.num_samples):
            theta_j = samples[j, :]
            sample_model = self.get_model_linear(constant_term, theta_j, ref_params, model_terms)
            residual = masked_data_array - sample_model
            r_squared = np.einsum("st, st -> ", np.conj(residual), residual).real
            r_squareds[j] = r_squared
        return r_squareds

    def strain_correct(self, samples):

        corrected_samples = jnp.zeros_like(samples)

        if self.include_chif:
            chif_samples = samples[:, -2]
            corrected_samples = corrected_samples.at[:, -2].set(chif_samples)
        else:
            chif_samples = jnp.full((self.num_samples,), self.chif_ref)
        if self.include_Mf:
            Mf_samples = samples[:, -1]
            corrected_samples = corrected_samples.at[:, -1].set(Mf_samples)
        else:
            Mf_samples = jnp.full((self.num_samples,), self.Mf_ref)

        for i, mode in enumerate(self.modes):
            omegas = self.linearised_frequency(mode, chif_samples, Mf_samples)
            re_A = samples[:, 2*i]
            im_A = samples[:, 2*i+1]
            if self.data_type == 'strain':
                corrected_samples = corrected_samples.at[:, 2*i].set(
                    re_A
                ) 
                corrected_samples = corrected_samples.at[:, 2*i+1].set(
                    im_A
                )
            if self.data_type == 'news':
                A_news = re_A + 1j * im_A
                A_strain = A_news / (-1j * omegas)
                corrected_samples = corrected_samples.at[:, 2*i].set(
                    A_strain.real
                )
                corrected_samples = corrected_samples.at[:, 2*i+1].set(
                    A_strain.imag
                )
            if self.data_type == 'psi4':
                A_psi4 = re_A + 1j * im_A
                A_strain = (A_psi4 / (-omegas**2)) * -2 # Factor of -2 to match scri convention 
                corrected_samples = corrected_samples.at[:, 2*i].set(
                    A_strain.real
                )
                corrected_samples = corrected_samples.at[:, 2*i+1].set(
                    A_strain.imag
                )

        return corrected_samples 
    
    def solve_via_cholesky(self, fisher, b):
        L = cholesky(fisher, lower=True) 
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        x = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
        return x
    
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

        chif_nonlinear, Mf_nonlinear = self.get_nonlinear_mf_chif(t0, self.T, self.spherical_modes, self.chif_ref, self.Mf_ref)
        _, ref_params_nonlinear = self._get_ls_amplitudes(
            t0, Mf_nonlinear, chif_nonlinear
        )  

        if self.use_nonlinear_params:
            chif_ref, Mf_ref = chif_nonlinear, Mf_nonlinear
            frequencies, frequency_derivatives, mixing_coefficients, mixing_derivatives = (
                self._get_mixing_frequency_terms(chif_ref, Mf_ref)
            )
        else:
            chif_ref = self.chif_ref
            Mf_ref = self.Mf_ref
            frequencies, frequency_derivatives, mixing_coefficients, mixing_derivatives = (
                self.frequencies,
                self.frequency_derivatives,
                self.mixing_coefficients,
                self.mixing_derivatives,
            )

        exponential_terms = self._get_exponential_terms(model_times, frequencies)
        ls_amplitudes, ref_params = self._get_ls_amplitudes(
            t0, Mf_ref, chif_ref
        )  
        model_terms = self.get_model_terms(
            model_times,
            ls_amplitudes,
            exponential_terms,
            mixing_coefficients,
            mixing_derivatives,
            frequencies,
            frequency_derivatives,
            Mf_ref,
        )
        constant_term = self.get_const_term(ls_amplitudes, exponential_terms, mixing_coefficients)

        noise_covariance_matrix = self.get_noise_covariance_matrix(analysis_times)
        noise_covariance_lower_triangular = cholesky(noise_covariance_matrix, lower=True)

        fisher_matrix = self.get_fisher_matrix(model_times, model_terms, noise_covariance_lower_triangular)
        b_vector = self.get_b_vector(
            masked_data_array,
            constant_term,
            model_times,
            model_terms,
            noise_covariance_lower_triangular,
        )

        # TODO use cholesky solve? 
        mean_vector = jnp.linalg.solve(fisher_matrix, b_vector) + ref_params
        covariance_matrix = get_inverse(fisher_matrix)
        samples, key = self._get_samples(mean_vector, covariance_matrix)

        if self.strain_parameters:
            samples = self.strain_correct(samples)
            ref_params = self.strain_correct(ref_params[None, :])[0, :]
            ref_params_nonlinear = self.strain_correct(ref_params_nonlinear[None, :])[0, :]
            #mean_vector = self.strain_correct(mean_vector[None, :])[0, :]
            #ls_amplitudes_corrected = jnp.zeros_like(ls_amplitudes)
            #for i, mode in enumerate(self.modes):
            #    re_A = ref_params[2*i]
            #    im_A = ref_params[2*i+1]
            #    ls_amplitudes_corrected = ls_amplitudes_corrected.at[i].set(
            #        re_A + 1j * im_A
            #    )
            #ls_amplitudes = ls_amplitudes_corrected
            #constant_term = self.get_const_term(ls_amplitudes, exponential_terms, mixing_coefficients)
            #model_terms = self.get_model_terms(
            #                    model_times,
            #                    ls_amplitudes,
            #                    exponential_terms,
            #                    mixing_coefficients,
            #                    mixing_derivatives,
            #                    frequencies,
            #                    frequency_derivatives,
            #                    Mf_ref,
            #                    )
            
        (
            mean_amplitude,
            mean_phase,
            sample_amplitudes,
            sample_phases,
            samples_weights,
            neff,
        ) = self.get_amplitude_phase(mean_vector, samples, t0)

        # weighted_quantiles_dict = self.get_amplitude_quantiles(sample_amplitudes, self.quantiles, samples_weights)
        unweighted_quantiles_dict = self.get_amplitude_quantiles(sample_amplitudes, self.quantiles)
        model_array_linear = self.get_model_linear(constant_term, mean_vector, ref_params, model_terms)
        model_array_nonlinear = self.get_model_nonlinear(mean_vector, analysis_times, Mf_ref, chif_ref)

        expected_chi_squared = self.get_expected_chi_squared(noise_covariance_matrix)

        model_chi_squared = self.get_model_chi_squared(masked_data_array, constant_term, ref_params, model_terms, mean_vector, covariance_matrix)
        p_values = np.array([np.sum(expected_chi_squared < chi_sq) / len(expected_chi_squared) for chi_sq in model_chi_squared])

        #model_chi_squared_mean, model_chi_squared_lower, model_chi_squared_upper = np.mean(model_chi_squared), np.percentile(model_chi_squared, 25), np.percentile(model_chi_squared, 75)
        #p_value_mean = np.sum(expected_chi_squared < model_chi_squared_mean) / len(expected_chi_squared)

        sample_model = self.get_model_linear(constant_term, mean_vector, ref_params, model_terms)
        residual = masked_data_array - sample_model
        mean_r_squared = np.einsum("st, st -> ", np.conj(residual), residual).real
        p_value_mean = np.sum(expected_chi_squared < mean_r_squared) / len(expected_chi_squared)

        fit = {
            "ls_amplitudes": ls_amplitudes,
            "chif_ref": chif_ref,
            "Mf_ref": Mf_ref,
            "frequencies": frequencies,
            "frequency_derivatives": frequency_derivatives,
            "mixing_coefficients": mixing_coefficients,
            "mixing_derivatives": mixing_derivatives,
            "noise_covariance": noise_covariance_matrix,
            "noise_covariance_lower_triangular": noise_covariance_lower_triangular,
            "fisher_matrix": fisher_matrix,
            "covariance": covariance_matrix,
            "b_vector": b_vector,
            "mean": mean_vector,
            "mean_amplitude": mean_amplitude,
            "mean_phase": mean_phase,
            "samples": samples,
            "sample_amplitudes": sample_amplitudes,
            "sample_phases": sample_phases,
            "samples_weights": samples_weights,
            "N_effective_samples": neff,
            "unweighted_quantiles": unweighted_quantiles_dict,
            # "weighted_quantiles": weighted_quantiles_dict,
            "model_array_linear": model_array_linear,
            "model_array_nonlinear": model_array_nonlinear,
            "analysis_times": analysis_times,
            "data_array_masked": masked_data_array,
            "constant_term": constant_term,
            "model_terms": model_terms,
            "ref_params": ref_params,
            "ref_params_nonlinear": ref_params_nonlinear,
            "param_names": self.params,
            "p_values": p_values,
            "p_value_mean": p_value_mean,
            "r_squareds": model_chi_squared,
            "r_squared_mean": mean_r_squared
        }

        return fit