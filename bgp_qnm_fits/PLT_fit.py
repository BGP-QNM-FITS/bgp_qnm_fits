import numpy as np
import jax
import jax.numpy as jnp

from bgp_qnm_fits.base_fit import Base_BGP_fit
from bgp_qnm_fits.utils import get_inverse
from tqdm import tqdm
from jax.scipy.linalg import cholesky, solve_triangular
import emcee

jax.config.update("jax_enable_x64", True)


class PLT_BGP_fit(Base_BGP_fit):
    """
    A class for fitting PLTs via Bayesian inference using MCMC sampling.
    """

    def __init__(
        self,
        *args,
        t0,
        PLT_modes,
        A_PLT_val=None,  # these should be lists of length equal to number of PLT modes
        t_PLT_val=None,
        lam_PLT_val=None,
        A_PLT_prior=(
            -0.01,
            0.01,
        ),  # these should be tuples of (min, max) for uniform prior (we assume the same prior on all modes)
        t_PLT_prior=(-50, 40),
        lam_PLT_prior=(1, 14),
        nsteps=1000,
        nwalkers=20,
        use_nonlinear_params=False,
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
        self.use_nonlinear_params = use_nonlinear_params
        self.PLT_modes = PLT_modes
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.A_PLT_val = A_PLT_val
        self.t_PLT_val = t_PLT_val
        self.lam_PLT_val = lam_PLT_val
        self.A_PLT_prior = A_PLT_prior
        self.t_PLT_prior = t_PLT_prior
        self.lam_PLT_prior = lam_PLT_prior

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

    def linearised_frequency(self, mode, chif_sample, Mf_sample):
        i = self.modes.index(mode)
        return (
            self.frequencies[i]
            - (Mf_sample - self.Mf_ref) * self.chif_ref / self.Mf_ref
            + (chif_sample - self.chif_ref) * self.frequency_derivatives[i]
        )

    def A_prior(self, A):
        # Flat prior: 0 inside bounds, -inf outside
        if np.all((self.A_PLT_prior[0] < A) & (A <= self.A_PLT_prior[1])):
            return 0.0
        else:
            return -np.inf

    def T_prior(self, analysis_times, t):
        # Flat prior: 0 inside bounds, -inf outside
        if np.all((self.t_PLT_prior[0] < t) & (t < self.t_PLT_prior[1])):
            return 0.0
        else:
            return -np.inf

    def lam_prior(self, lam):
        # Flat prior: 0 inside bounds, -inf outside
        if np.all((self.lam_PLT_prior[0] < lam) & (lam <= self.lam_PLT_prior[1])):
            return 0.0
        else:
            return -np.inf

    def _get_PLT_term(self, t0, PLT_mode_indices, analysis_times, A_PLT_real, A_PLT_imag, t_PLT, lam_PLT):
        """

        Computes the PLT term for a given set of QNMs.

        """

        PLT_term = jnp.zeros((len(self.spherical_modes), len(analysis_times)), dtype=jnp.complex128)
        for i, mode_index in enumerate(PLT_mode_indices):
            PLT_term = PLT_term.at[mode_index].set(
                (A_PLT_real[i] + 1j * A_PLT_imag[i])
                * ((analysis_times - t_PLT[i]) / (analysis_times[0] - t_PLT[i])) ** (-lam_PLT[i])
            )

        return PLT_term

    def _get_b_vector_PLT(
        self,
        data_array_new,
        analysis_times,
        Kinv_model_array,
        inv_fisher_matrix,
    ):
        """
        Computes the b vector for the parameters in `params`. This method uses the Cholesky decomposition
        of the noise covariance matrix to compute the b vector efficiently and without direct inversion of the covariance matrix.

        Args:
            data_array (jnp.ndarray): The data array of shape: spherical modes x analysis times.
            constant_term (jnp.ndarray): The constant term computed with reference parameters.
            analysis_times (jnp.ndarray): The times at which the b vector is evaluated.
            model_terms (jnp.ndarray): A 3D array of model terms of shape: spherical modes x analysis times x model parameters.
            noise_covariance_matrix (jnp.ndarray): The noise covariance matrix for the GP kernel.
        Returns:
            jnp.ndarray: A 1D JAX array representing the b vector, where each element corresponds to a parameter in `params`.
        """

        b_vector = jnp.real(jnp.einsum("st, stp -> p", jnp.conj(data_array_new), Kinv_model_array))
        b_vector_hc = jnp.real(jnp.einsum("stp, st -> p", jnp.conj(Kinv_model_array), data_array_new))

        b_val = jnp.einsum("pq, p, q ->", inv_fisher_matrix, b_vector, b_vector_hc)

        if self.is_GP_diagonal:
            raise ValueError("Diagonal GP covariance is not supported in this implementation.")
        else:
            return jnp.real(b_val)

    def _get_residual_PLT(
        self,
        data_array_new,
        analysis_times,
        noise_covariance_lower_triangular,
    ):

        L_data_array = solve_triangular(noise_covariance_lower_triangular, data_array_new, lower=True)
        Kinv_data_array = solve_triangular(
            jnp.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_data_array, lower=False
        )
        residual_val = jnp.einsum("st, st ->", jnp.conj(data_array_new), Kinv_data_array)

        if self.is_GP_diagonal:
            return jnp.real(residual_val) * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
        else:
            return jnp.real(residual_val)

    def _get_PLT_log_likelihood(
        self,
        t0,
        PLT_mode_indices,
        analysis_times,
        masked_data_array,
        Kinv_model_array,
        constant_term,
        inv_fisher_matrix,
        noise_covariance_lower_triangular,
        A_PLT_real,
        A_PLT_imag,
        t_PLT,
        lam_PLT,
    ):

        PLT_term = self._get_PLT_term(t0, PLT_mode_indices, analysis_times, A_PLT_real, A_PLT_imag, t_PLT, lam_PLT)
        data_array_new = masked_data_array - constant_term - PLT_term

        ll = (
            self.A_prior(A_PLT_real)
            + self.A_prior(A_PLT_imag)
            + self.T_prior(analysis_times, t_PLT)
            + self.lam_prior(lam_PLT)
            - 0.5
            * self._get_residual_PLT(
                data_array_new,
                analysis_times,
                noise_covariance_lower_triangular,
            )
            + 0.5 * self._get_b_vector_PLT(data_array_new, analysis_times, Kinv_model_array, inv_fisher_matrix)
        )

        return ll

    def PLT_search(
        self,
        t0,
        PLT_mode_indices,
        analysis_times,
        masked_data_array,
        Kinv_model_array,
        constant_term,
        inv_fisher_matrix,
        noise_covariance_lower_triangular,
    ):

        N = len(PLT_mode_indices)
        param_names = []
        bounds = []

        # Build parameter list and bounds
        if self.A_PLT_val is None:
            param_names += ["A_PLT_real", "A_PLT_imag"]
            bounds += [self.A_PLT_prior] * N + [self.A_PLT_prior] * N
        if self.t_PLT_val is None:
            param_names += ["t_PLT"]
            bounds += [self.t_PLT_prior] * N
        if self.lam_PLT_val is None:
            param_names += ["lam_PLT"]
            bounds += [self.lam_PLT_prior] * N

        ndim = len(bounds)
        bounds_array = np.array(bounds)

        # Prepare fixed values
        fixed_A_real = np.array(self.A_PLT_val.real) if self.A_PLT_val is not None else None
        fixed_A_imag = np.array(self.A_PLT_val.imag) if self.A_PLT_val is not None else None
        fixed_t_PLT = np.array(self.t_PLT_val) if self.t_PLT_val is not None else None
        fixed_lam_PLT = np.array(self.lam_PLT_val) if self.lam_PLT_val is not None else None

        def log_probability(params):

            if np.any(params < bounds_array[:, 0]) or np.any(params > bounds_array[:, 1]):
                return -np.inf

            current_index = 0
            if fixed_A_real is not None:
                A_PLT_real = fixed_A_real
            else:
                A_PLT_real = params[current_index : current_index + N]
                current_index += N
            if fixed_A_imag is not None:
                A_PLT_imag = fixed_A_imag
            else:
                A_PLT_imag = params[current_index : current_index + N]
                current_index += N
            if fixed_t_PLT is not None:
                t_PLT = fixed_t_PLT
            else:
                t_PLT = params[current_index : current_index + N]
                current_index += N
            if fixed_lam_PLT is not None:
                lam_PLT = fixed_lam_PLT
            else:
                lam_PLT = params[current_index : current_index + N]

            result = self._get_PLT_log_likelihood(
                t0,
                PLT_mode_indices,
                analysis_times,
                masked_data_array,
                Kinv_model_array,
                constant_term,
                inv_fisher_matrix,
                noise_covariance_lower_triangular,
                A_PLT_real,
                A_PLT_imag,
                t_PLT,
                lam_PLT,
            )
            return result

        initial_pos = np.array(
            [np.random.uniform(low=low, high=high, size=self.nwalkers) for (low, high) in bounds]
        ).T  # shape (nwalkers, ndim)

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, log_probability, threads=4)
        sampler.run_mcmc(initial_pos, self.nsteps, progress=True)

        return sampler

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

        chif_nonlinear, Mf_nonlinear = self.get_nonlinear_mf_chif(
            t0, self.T, self.spherical_modes, self.chif_ref, self.Mf_ref
        )
        _, ref_params_nonlinear = self._get_ls_amplitudes(t0, Mf_nonlinear, chif_nonlinear)

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
        ls_amplitudes, ref_params = self._get_ls_amplitudes(t0, Mf_ref, chif_ref)
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
        inv_fisher_matrix = get_inverse(fisher_matrix)

        # Get model array with inverse noise covariance applied

        L_model_terms = solve_triangular(noise_covariance_lower_triangular, model_terms, lower=True)
        Kinv_model_array = solve_triangular(
            jnp.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_model_terms, lower=False
        )

        # Determine indices of PLT modes in spherical modes

        PLT_mode_indices = [self.spherical_modes.index(mode) for mode in self.spherical_modes if mode in self.PLT_modes]

        # Perform PLT search
        sampler = self.PLT_search(
            t0,
            PLT_mode_indices,
            analysis_times,
            masked_data_array,
            Kinv_model_array,
            constant_term,
            inv_fisher_matrix,
            noise_covariance_lower_triangular,
        )

        mcmc_samples = sampler.get_chain(flat=True)

        self.sampler = sampler
        self.mcmc_samples = mcmc_samples
