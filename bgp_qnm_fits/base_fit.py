import qnmfits
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from bgp_qnm_fits.gp_kernels import compute_kernel_matrix
from scipy.optimize import minimize

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
        strain_parameters=False,
        data_type=None
    ):
        """
        Initialize the Base_BGP_fit class.
        This method sets up the initial attributes and computes the necessary parameters for the fitting process.

        Args:
            times (array): List of time values corresponding to the data_dict.
            data_dict (dict): Dictionary containing the data for each spherical mode.
            modes (list): List of QNM modes to fit.
            Mf (float): Reference remnant mass.
            chif (float): Reference remnant dimensionless spin.
            kernel_param_dict (dict): Dictionary containing the hyperparameters for the kernel.
            kernel (callable): Kernel function to be used in the GP fitting.
            t0_method (str): Method for determining the start time of the fitting ("geq" or "closest").
            T (float): Duration of the fitting window.
            spherical_modes (list, optional): List of spherical modes to consider. If None, defaults to keys of data_dict.
            include_chif (bool): Whether to include the remnant spin as a parameter in the fitting.
            include_Mf (bool): Whether to include the remnant mass as a parameter in the fitting.
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
        self.strain_parameters = strain_parameters
        self.data_type = data_type

        if self.strain_parameters:
            if self.data_type not in ["news", "psi4", "strain"]:
                raise ValueError(
                    "If strain_parameters is True, data_type must be one of ['news', 'psi4', 'strain']." \
                    "Parameters will be corrected to match the strain domain values"
                )

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
        self.is_GP_diagonal = jnp.count_nonzero(test_kernel_matrix - jnp.diag(jnp.diagonal(test_kernel_matrix))) == 0

        #print("##################### Checks #####################")
        #print("Is kernel diagonal?", self.is_GP_diagonal)
        #print("Params in model:", self.params)
        #print("##################################################")

    def _mask_data(self, t0):
        """
        Mask the data based on the specified t0_method.

        Args:
            t0 (float): The start time for the fitting.

        Returns:
            tuple: A tuple containing the masked times and the corresponding data array.
        """
        # T = self.T - t0
        start_index = jnp.argmin((self.times - t0) ** 2)
        end_index = jnp.argmin((self.times - t0 - self.T) ** 2)
        times_mask = self.times[start_index:end_index]
        data_array_mask = self.data_array[:, start_index:end_index]
        return jnp.array(times_mask), data_array_mask

    def _get_params(self):
        """
        Get the parameter names for the QNM fitting.

        Returns:
            list: A list of parameter names for the fitting.
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
        Compute the mismatch for a given mass and spin parameters.

        Args:
            chif_mf (tuple): Tuple containing the remnant spin magnitude and mass.
            t0 (float): The start time for the fitting.
            T (float): Duration of the fitting window.
            spherical_modes (list): List of spherical modes to consider.
        Returns:
            float: The mismatch value for the given mass and spin parameters.
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

    def get_nonlinear_mf_chif(self, t0, T, spherical_modes, chif_ref, Mf_ref):
        """
        Perform a nonlinear fit to find the remnant mass and spin parameters

        Args:
            t0 (float): The start time for the fitting.
            T (float): Duration of the fitting window.
            spherical_modes (list): List of spherical modes to consider.
            chif_ref (float): Reference remnant spin.
            Mf_ref (float): Reference remnant mass.
        Returns:
            tuple: A tuple containing the best-fit remnant spin and mass.
        """

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
        Compute the frequency for a given QNM mode.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            chif (float): Remnant spin.
            Mf (float): Remnant mass.
        Returns:
            complex: Frequency of the QNM.
        """
        return qnmfits.qnm.omega_list([mode], chif, Mf)[0] 
    

    #def _get_frequency(self, mode, chif, Mf):
        """
        Compute the frequency for a given QNM mode.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            chif (float): Remnant spin.
            Mf (float): Remnant mass.
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
        Compute the mixing coefficient for a given QNM mode and spherical mode.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            sph_mode (tuple): Spherical mode (ell, m).
            chif (float): Remnant spin.
        Returns:
            complex: Mixing coefficient for the QNM mode.
        """
        return qnmfits.qnm.mu_list([sph_mode + mode], chif)[0] 
    

    #def _get_mixing(self, mode, sph_mode, chif):
        """
        Compute the mixing coefficient for a given QNM mode and spherical mode.
        For higher order QNMs, this is a placeholder that returns 1 + 0j.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            sph_mode (tuple): Spherical mode (ell, m).
            chif (float): Remnant spin.
        Returns:
            complex: Mixing coefficient for the QNM mode.
        """
        ell, m = sph_mode
        if len(mode) == 4:
            lp, mp, nprime, sign = mode
            return qnmfits.qnm.mu(ell, m, lp, mp, nprime, sign, chif)
        elif len(mode) == 8:
            lp, mp, nprime, sign, ell2, m2, nprime2, sign2 = mode
            if mp + m2 == m:
                return 1 + 0j
            else:
                return 0 + 0j
        elif len(mode) == 12:
            ell1, m1, n1, p1, ell2, m2, n2, p2, ell3, m3, n3, p3 = mode
            if m1 + m2 + m3 == m:
                return 1 + 0j
            else:
                return 0 + 0j


    def _get_ls_amplitudes(self, t0, Mf, chif, t0_method="closest"):
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
        """
        Compute the mixing coefficients and frequencies for the QNM modes.
        Args:
            chif (float): Remnant spin.
            Mf (float): Remnant mass.
        Returns:
            tuple: A tuple containing:
                - frequencies: Array of frequencies for each QNM mode.
                - frequency_derivatives: Array of frequency derivatives with respect to chif.
                - mixing_coefficients: 2D array of mixing coefficients for each spherical mode and QNM mode.
                - mixing_derivatives: 2D array of mixing coefficient derivatives with respect to chif.
        """
        frequencies = jnp.array([self._get_frequency(mode, chif, Mf) for mode in self.modes])
        frequency_derivatives = jnp.array([self._get_domega_dchif(mode, chif, Mf) for mode in self.modes])
        mixing_coefficients = jnp.array(
            [self._get_mixing(mode, sph_mode, chif) for sph_mode in self.spherical_modes for mode in self.modes]
        ).reshape(len(self.spherical_modes), len(self.modes))
        mixing_derivatives = jnp.array(
            [self._get_dmu_dchif(mode, sph_mode, chif) for sph_mode in self.spherical_modes for mode in self.modes]
        ).reshape(len(self.spherical_modes), len(self.modes))
        return frequencies, frequency_derivatives, mixing_coefficients, mixing_derivatives

    # @partial(jax.jit, static_argnames=["self"])
    def get_noise_covariance_matrix(self, analysis_times):
        """
        Compute the noise covariance matrix for the GP kernel.

        Args:
            analysis_times (array): The times at which the kernel is evaluated.
        Returns:
            jnp.ndarray: A 2D array representing the noise covariance matrix for each spherical mode.
        """

        # TODO It would be better for the noise to be calculated before it's passed to this class,
        # then only a single noise specification would need to be passed (calculated at all times) and sliced as needed. 
        # This would bring it in alignment with qnmfits. 
        # Have the shape of the specification determine the covariance matrix form i.e.
        # L ~ 1 x beta (scalar) diagonal constant
        # L ~ 2 x beta (vector) diagonal variable
        # L ~ 3 x beta (matrix) full GP covariance matrix

        return jnp.array(
            [
                compute_kernel_matrix(analysis_times, self.kernel_param_dict[mode], self.kernel).real
                for mode in self.spherical_modes
            ],
            dtype=jnp.float64,
        )

    def _get_exponential_terms(self, analysis_times, frequencies):
        """
        Compute the exponential terms for the QNM modes at the given analysis times.
        Args:
            analysis_times (array): The times at which the exponential terms are evaluated.
            frequencies (array): The frequencies of the QNM modes.
        Returns:
            jnp.ndarray: A 2D array of exponential terms of shape: QNMs x analysis times.

        """
        return jnp.array([jnp.exp(-1j * frequencies[i] * analysis_times) for i in range(self.modes_length)])

    def _get_domega_dchif(self, mode, chif, Mf, delta=1.0e-6):
        """
        Compute the derivative of the frequency with respect to chif for a given QNM mode.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            chif (float): Reference remnant spin.
            Mf (float): Reference remnant mass.
        Returns:
            float: The derivative of the frequency with respect to chif.

        """
        omega_plus = self._get_frequency(mode, chif + delta, Mf)
        omega_minus = self._get_frequency(mode, chif - delta, Mf)
        return (omega_plus - omega_minus) / (2 * delta)

    def _get_dmu_dchif(self, mode, sph_mode, chif, delta=1.0e-6):
        """
        Compute the derivative of the mixing coefficient with respect to chif for a given QNM mode and spherical mode.
        Args:
            mode (tuple): QNM mode (ell, m, n, sign).
            sph_mode (tuple): Spherical mode (ell, m).
            chif (float): Reference remnant spin.

        Returns:
            float: The derivative of the mixing coefficient with respect to chif.

        """
        mu_plus = self._get_mixing(mode, sph_mode, chif=chif + delta)
        mu_minus = self._get_mixing(mode, sph_mode, chif=chif - delta)
        return (mu_plus - mu_minus) / (2 * delta)

    def _get_re_amplitude_term(self, exponential_terms, mixing_coefficients):
        """
        Computes the model term for the real part of the amplitudes for a given set of QNMs.

        Args:
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
        Returns:
            jnp.ndarray: A 3D array of shape: QNMs x spherical modes x analysis times.

        """
        return jnp.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

    def _get_im_amplitude_term(self, exponential_terms, mixing_coefficients):
        """
        Computes the model term for the imaginary part of the amplitudes for a given set of QNMs.

        Args:
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
        Returns:
            jnp.ndarray: A 3D array of shape: QNMs x spherical modes x analysis times.

        """
        return jnp.einsum("pt,sp->pst", exponential_terms, mixing_coefficients)

    def _get_mass_term(self, analysis_times, ls_amplitudes, frequencies, exponential_terms, mixing_coefficients, Mf):
        """
        Computes the mass term for a given set of QNMs.

        Args:
            analysis_times (jnp.ndarray): The times at which the mass term is evaluated.
            ls_amplitudes (jnp.ndarray): The least-squares reference amplitudes for the QNMs.
            frequencies (jnp.ndarray): The frequencies of the QNMs.
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x analysis times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
        Returns:
            jnp.ndarray: A 2D array of shape: spherical modes x analysis times representing the mass term.
        """
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
        """

        Computes the chif term for a given set of QNMs.

        Args:
            analysis_times (jnp.ndarray): The times at which the chif term is evaluated.
            ls_amplitudes (jnp.ndarray): The least-squares reference amplitudes for the QNMs.
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x analysis times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
            mixing_derivatives (jnp.ndarray): A 2D array of mixing coefficient derivatives with respect to chif.
            frequency_derivatives (jnp.ndarray): A 2D array of frequency derivatives with respect to chif.
        Returns:
            jnp.ndarray: A 2D array of shape: spherical modes x analysis times representing the chif term.

        """
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
        """
        Computes the constant (reference) term for the model.

        Args:
            ls_amplitudes (jnp.ndarray): The least-squares reference amplitudes for the QNMs.
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x analysis times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
        Returns:
            jnp.ndarray: A 2D array of shape: spherical modes x analysis times representing the constant term.

        """
        return jnp.einsum("p,sp,pt->st", ls_amplitudes, mixing_coefficients, exponential_terms)

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
        """
        Computes the model terms for the QNM fitting.

        Args:
            model_times (jnp.ndarray): The times at which the model terms are evaluated.
            ls_amplitudes (jnp.ndarray): The least-squares reference amplitudes for the QNMs.
            exponential_terms (jnp.ndarray): A 2D array of exponential terms of shape: QNMs x analysis times.
            mixing_coefficients (jnp.ndarray): A 2D array of mixing coefficients of shape: spherical modes x QNMs.
            mixing_derivatives (jnp.ndarray): A 2D array of mixing coefficient derivatives with respect to chif.
            frequencies (jnp.ndarray): The frequencies of the QNMs.
            frequency_derivatives (jnp.ndarray): A 2D array of frequency derivatives with respect to chif.
            Mf (float): Reference remnant mass.
        Returns:
            jnp.ndarray: A 3D array of shape:  spherical modes x model times x model parameters representing the model terms.
            The ordering speeds up computations using the Cholesky decomposition of the noise covariance matrix later.

        """

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

        return jnp.moveaxis(sph_matrix, 0, -1)

    def get_fisher_matrix(self, analysis_times, model_terms, noise_covariance_lower_triangular):
        """

        Computes the Fisher matrix for the parameters in `params`. This method uses the Cholesky decomposition
        of the noise covariance matrix to compute the Fisher matrix efficiently and without direct inversion of the covariance matrix.

        Args:
            analysis_times (jnp.ndarray): The times at which the Fisher matrix is evaluated.
            model_terms (jnp.ndarray): A 3D array of model terms of shape: spherical modes x analysis times x model parameters.
            noise_covariance_matrix (jnp.ndarray): The noise covariance matrix for the GP kernel: spherical modes x analysis times x analysis times.
        Returns:
            jnp.ndarray: A 2D JAX array representing the Fisher matrix, where each element corresponds to a pair of parameters in `params`.

        """

        L_model_terms = solve_triangular(
            noise_covariance_lower_triangular, model_terms, lower=True
        )  # This has shape: spherical modes x analysis times x model parameters
        Kinv_model_terms = solve_triangular(
            jnp.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_model_terms, lower=False
        )  # This has shape: spherical modes x analysis times x model parameters
        fisher_matrix = jnp.einsum(
            "btm, btn -> mn", jnp.conj(model_terms), Kinv_model_terms
        )  # This has shape: model parameters x model parameters

        if self.is_GP_diagonal:
            return jnp.real(fisher_matrix) * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
        else:
            return jnp.real(fisher_matrix)

    def get_b_vector(
        self,
        data_array,
        constant_term,
        analysis_times,
        model_terms,
        noise_covariance_lower_triangular,
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
        data_array_new = data_array - constant_term

        L_model_terms = solve_triangular(noise_covariance_lower_triangular, model_terms, lower=True)
        Kinv_model_terms = solve_triangular(
            jnp.transpose(noise_covariance_lower_triangular, [0, 2, 1]), L_model_terms, lower=False
        )
        b_vector = jnp.einsum("st, stp -> p", jnp.conj(data_array_new), Kinv_model_terms)

        if self.is_GP_diagonal:
            return jnp.real(b_vector) * (analysis_times[-1] - analysis_times[0]) / len(analysis_times)
        else:
            return jnp.real(b_vector) 