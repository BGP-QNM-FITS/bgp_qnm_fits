import qnmfits
import os 
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from functools import partial

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


class Base_BGP_fit:
    """

    The base class for performing Bayesian Quasinormal Mode (QNM) fitting using Gaussian Processes (GP).
    This contains the core methods and attributes needed for the fitting process.
    It is designed to be subclassed for specific fitting implementations
    (for either a single t0 or a list of t0s), such as BGP_fit.

    """

    def __init__(
        self,
        times,
        data_dict,
        modes,
        Mf,
        chif,
        kernel_param_dict,
        kernel,
        t0_method="closest",
        T=100,
        spherical_modes=None,
        include_chif=False,
        include_Mf=False,
    ):
        """
        Initialize the BGP_fit object with the required parameters.

        Args:
            times (array): The full simulation time array.
            data_dict (dict): The full dictionary of data for spherical modes directly from the sim object.
            modes (list): List of QNMs.
            Mf (float): Remnant mass (from metadata).
            chif (float): Remnant spin (from metadata).
            t0 (float or list): Start time(s).
            kernel_param_dict (dict): Kernel parameters for GP.
            kernel (callable): Kernel function for GP.
            T (float): Duration of the analysis window (i.e. anlysis runs to T + t0).
            spherical_modes (list): List of spherical modes.
        """

        # Get initial attributes

        self.times = jnp.array(times)
        self.data_dict = data_dict
        self.t0_method = t0_method
        self.T = T
        self.modes = modes
        self.modes_length = len(self.modes)
        self.include_chif = include_chif
        self.include_Mf = include_Mf
        self.params = self._get_params()
        self.params_length = len(self.params)
        self.kernel_param_dict = kernel_param_dict
        self.kernel = kernel
        self.spherical_modes = spherical_modes or list(data_dict.keys())
        self.spherical_modes_length = len(self.spherical_modes)
        self.data_array = jnp.array([data_dict[mode] for mode in self.spherical_modes])

        # By default, the base class sets the ABD mass and spin (NR values) as attributes; subclasses may override 
        # this behavior to implement nonlinear mass and spin from least-squares minimization.

        self.Mf_ref = Mf
        self.chif_ref = chif

        self.frequencies, self.frequency_derivatives, self.mixing_coefficients, self.mixing_derivatives = (
            self._get_mixing_frequency_terms(self.chif_ref, self.Mf_ref)
        )

        # Check if the kernel is diagonal
        test_times = jnp.linspace(0, 10, 10)
        test_kernel_matrix = jnp.array(
            self.kernel(jnp.array(test_times), **self.kernel_param_dict[self.spherical_modes[0]])
        )
        self.is_GP_diagonal = jnp.allclose(test_kernel_matrix, jnp.diag(jnp.diagonal(test_kernel_matrix)))

        # print("##################### Checks #####################")
        # print("Is kernel diagonal?", self.is_GP_diagonal)
        # print("Params in model:", self.params)
        # print("##################################################")

    def _mask_data(self, t0):
        """
        Mask the data based on the specified t0_method.

        Args:
            t0_method (str): Method for masking data ("geq" or "closest").

        Returns:
            tuple: Masked times and data dictionary.
        """
        # T = self.T - t0 # TODO: REMEMBER THIS IS THE CURRENT CONVENTION!!
        start_index = jnp.argmin((self.times - t0) ** 2)
        end_index = jnp.argmin((self.times - t0 - self.T) ** 2)
        times_mask = self.times[start_index:end_index]
        data_array_mask = self.data_array[:, start_index:end_index]
        return jnp.array(times_mask), data_array_mask

    def _get_params(self):
        """
        Get the parameters for the QNM fitting.

        Returns:
            jnp.ndarray: JAX array of parameters for the QNM fitting.
        """
        params = []

        for mode in self.modes:
            params.append(f"Re_C_{mode}")
            params.append(f"Im_C_{mode}")
        if self.include_chif:
            params.append("chif")
        if self.include_Mf:
            params.append("Mf")
        return params

    def _mf_chif_mismatch(self, chif_mf, t0, T, spherical_modes):
        """
        Compute the mismatch for given mass and spin parameters.
        """
        chif_mag, Mf = chif_mf
        best_fit = qnmfits.multimode_ringdown_fit(
            self.times,
            self.data_dict,
            self.modes,
            Mf,
            chif_mag,
            t0,
            t0_method="closest",
            T=T,
            spherical_modes=spherical_modes,
        )
        return best_fit["mismatch"]

    def _get_nonlinear_mf_chif(self, t0, T, spherical_modes, chif_ref, Mf_ref):
        """
        Compute mass and spin parameters for each t0 using least-squares and mismatch minimization.
        """
        # TODO: Optimise this
        initial_params = (chif_ref, Mf_ref)
        Mf_RANGE = (Mf_ref * 0.5, Mf_ref * 1.5)
        chif_mag_RANGE = (0.1, 0.99)
        bounds = (chif_mag_RANGE, Mf_RANGE)

        result = minimize(
            self._mf_chif_mismatch,
            initial_params,
            args=(t0, T, spherical_modes),
            method="Nelder-Mead",
            bounds=bounds,
        )

        return result.x[0], result.x[1]

    def _get_frequency(self, mode, chif, Mf):
        """
        Compute the frequency for a given QNM mode and spin.

        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            chif (float): Remnant spin.

        Returns:
            complex: Frequency of the QNM.
        """
        return sum(
            [
                qnmfits.qnm.omega(ell, m, n, sign, chif, Mf, s=-2)
                for ell, m, n, sign in [mode[i : i + 4] for i in range(0, len(mode), 4)]
            ]
        )

    def _get_mixing(self, mode, sph_mode, chif):
        """
        Compute the mixing coefficient for a given QNM mode.

        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            chif (float): Remnant spin.

        Returns:
            complex: Mixing coefficient of the QNM. Or, if quadratic or cubic, 1.
        """
        ell, m = sph_mode
        if len(mode) == 4:
            lp, mp, nprime, sign = mode
            return qnmfits.qnm.mu(ell, m, lp, mp, nprime, sign, chif)
        elif len(mode) == 8:
            return 1 + 0j
        elif len(mode) == 12:
            return 1 + 0j

    def _get_ls_amplitudes(self, t0, Mf, chif, t0_method="closest"):
        """
        Compute least-squares amplitudes and reference parameters.
        """
        # Perform the multimode ringdown fit using qnmfits
        ls_fit = qnmfits.multimode_ringdown_fit(
            self.times,
            self.data_dict,
            modes=self.modes,
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

    def _get_mixing_frequency_terms(self, chif, Mf):
        frequencies = jnp.array([self._get_frequency(mode, chif, Mf) for mode in self.modes])
        frequency_derivatives = jnp.array([self._get_domega_dchif(mode, chif, Mf) for mode in self.modes])
        mixing_coefficients = jnp.array(
            [self._get_mixing(mode, sph_mode, chif) for sph_mode in self.spherical_modes for mode in self.modes]
        ).reshape(len(self.spherical_modes), len(self.modes))
        mixing_derivatives = jnp.array(
            [self._get_dmu_dchif(mode, sph_mode, chif) for sph_mode in self.spherical_modes for mode in self.modes]
        ).reshape(len(self.spherical_modes), len(self.modes))
        return frequencies, frequency_derivatives, mixing_coefficients, mixing_derivatives

    def get_inverse(self, matrix, epsilon=1e-10):
        vals, vecs = jnp.linalg.eigh(matrix)
        vals = jnp.maximum(vals, epsilon)
        return jnp.einsum("ik, k, jk -> ij", vecs, 1 / vals, vecs)

    def compute_kernel_matrix(self, hyperparams, analysis_times):
        return jnp.array(self.kernel(jnp.array(analysis_times), **hyperparams) + jnp.eye(len(analysis_times)) * 1e-13)

    @partial(jax.jit, static_argnames=["self"])
    def get_inverse_noise_covariance_matrix(self, analysis_times):
        return jnp.array(
            [
                self.get_inverse(self.compute_kernel_matrix(self.kernel_param_dict[mode], analysis_times)).real
                for mode in self.spherical_modes
            ],
            dtype=jnp.float64,
        )

    def _get_exponential_terms(self, analysis_times, frequencies):
        """Compute the exponential terms for the QNM fitting."""
        return jnp.array([jnp.exp(-1j * frequencies[i] * analysis_times) for i in range(self.modes_length)])

    def _get_domega_dchif(self, mode, chif, Mf, delta=1.0e-6):
        """Compute domega/dchif for a given QNM."""
        omega_plus = self._get_frequency(mode, chif + delta, Mf)
        omega_minus = self._get_frequency(mode, chif - delta, Mf)
        return (omega_plus - omega_minus) / (2 * delta)

    def _get_dmu_dchif(self, mode, sph_mode, chif, delta=1.0e-6):
        """Compute dmu/dchif for a given QNM."""
        mu_plus = self._get_mixing(mode, sph_mode, chif=chif + delta)
        mu_minus = self._get_mixing(mode, sph_mode, chif=chif - delta)
        return (mu_plus - mu_minus) / (2 * delta)

    def _get_re_amplitude_term(self, exponential_terms, mixing_coefficients):
        """Computes the real part of the amplitude model term for a given set of QNMs. Returns a
        len(spherical_modes) x len(t_list) array."""
        return jnp.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

    def _get_im_amplitude_term(self, exponential_terms, mixing_coefficients):
        """Computes the imaginary part of the amplitude model term for a given set of QNMs. Returns a
        len(spherical_modes) x len(t_list) array."""
        return jnp.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

    def _get_mass_term(self, analysis_times, ls_amplitudes, frequencies, exponential_terms, mixing_coefficients, Mf):
        """Computes the Mf term for a given set of QNMs. Returns a len(spherical_modes) x len(t_list)
        array."""
        return jnp.einsum(
            "p,sp,pt->st",
            1j * frequencies / Mf * ls_amplitudes,
            mixing_coefficients,
            exponential_terms * analysis_times,
        )

    def _get_chif_term(
        self,
        analysis_times,
        ls_amplitudes,
        exponential_terms,
        mixing_coefficients,
        mixing_derivatives,
        frequency_derivatives,
    ):
        """Computes the chif_mag term for a given set of QNMs. Returns a len(spherical_modes) x
        len(t_list) array."""
        return jnp.einsum(
            "p,pt,spt->st",
            ls_amplitudes,
            exponential_terms,
            mixing_derivatives[:, :, None]
            - 1j
            * mixing_coefficients[:, :, None]
            * analysis_times[None, None, :]
            * frequency_derivatives[None, :, None],
        )

    def get_const_term(self, ls_amplitudes, exponential_terms, mixing_coefficients):
        """Computes the H_* term for a given set of QNMs."""
        return jnp.einsum("p,sp,pt->st", ls_amplitudes, mixing_coefficients, exponential_terms)

    # @partial(jax.jit, static_argnames=["self"]) - can't do this now in case Mf is updated!
    def get_model_terms(
        self,
        model_times,
        ls_amplitudes,
        exponential_terms,
        mixing_coefficients,
        mixing_derivatives,
        frequencies,
        frequency_derivatives,
        Mf,
    ):
        sph_matrix = jnp.zeros(
            (self.params_length, self.spherical_modes_length, len(model_times)),
            dtype=jnp.complex128,
        )

        re_model_terms = self._get_re_amplitude_term(exponential_terms, mixing_coefficients)
        sph_matrix = sph_matrix.at[: 2 * self.modes_length : 2].set(re_model_terms)
        sph_matrix = sph_matrix.at[1 : 2 * self.modes_length : 2].set(1j * re_model_terms)

        if self.include_chif:
            sph_matrix = sph_matrix.at[-1 if not self.include_Mf else -2].set(
                self._get_chif_term(
                    model_times,
                    ls_amplitudes,
                    exponential_terms,
                    mixing_coefficients,
                    mixing_derivatives,
                    frequency_derivatives,
                )
            )

        if self.include_Mf:
            sph_matrix = sph_matrix.at[-1].set(
                self._get_mass_term(model_times, ls_amplitudes, frequencies, exponential_terms, mixing_coefficients, Mf)
            )

        return sph_matrix

    def get_fisher_matrix(self, analysis_times, model_terms, inverse_noise_covariance_matrix):
        """

        Compute the Fisher matrix.

        Returns:
            jnp.ndarray: A 2D array representing the Fisher matrix,
            where the element at (i, j) corresponds to the Fisher information
            between the i-th and j-th parameters.
        """

        if self.is_GP_diagonal:
            # Use the diagonal version of get_element
            fisher_matrix = (
                jnp.einsum(
                    "pst,qsu,stu->pq",
                    jnp.conj(model_terms),
                    model_terms,
                    inverse_noise_covariance_matrix,
                )
                * (analysis_times[-1] - analysis_times[0])
                / len(analysis_times)
            )
        else:
            # Use the general version of get_element
            fisher_matrix = jnp.einsum(
                "pst,qsu,stu->pq",
                jnp.conj(model_terms),
                model_terms,
                inverse_noise_covariance_matrix,
            )

        return jnp.real(fisher_matrix)

    def get_b_vector(
        self,
        data_array,
        constant_term,
        analysis_times,
        model_terms,
        inverse_noise_covariance_matrix,
    ):
        """
        Computes the b vector for the parameters in `params`.

        The b vector is calculated based on the difference between the data
        and a constant model term (`h_0`). Each element of the b vector corresponds to
        a parameter in `params`.

        Returns:
            jnp.ndarray: A 1D JAX array representing the b vector, where
            each element corresponds to a parameter in `params`.
        """
        data_array_new = data_array - constant_term
        if self.is_GP_diagonal:
            # Use the diagonal version of get_element
            b_vector = (
                jnp.einsum(
                    "pst,su,stu->p",
                    jnp.conj(model_terms),
                    data_array_new,
                    inverse_noise_covariance_matrix,
                )
                * (analysis_times[-1] - analysis_times[0])
                / len(analysis_times)
            )
        else:
            # Use the general version of get_element
            b_vector = jnp.einsum(
                "pst,su,stu->p",
                jnp.conj(model_terms),
                data_array_new,
                inverse_noise_covariance_matrix,
            )

        return jnp.real(b_vector)
