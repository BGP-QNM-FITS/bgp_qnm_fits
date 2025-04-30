import numpy as np
import qnmfits
from bayes_qnm_GP_likelihood.utils import *
from bayes_qnm_GP_likelihood.GP_funcs import *
import scipy.stats

import time 

# Jax
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax import debug
from functools import partial

class BGP_fit:
    """
    A class for performing Bayesian Quasinormal Mode (QNM) fitting using Gaussian Processes (GP).
    """

    def __init__(self, 
                 times, 
                 data_dict, 
                 modes, 
                 Mf, 
                 chif, 
                 t0, 
                 kernel_param_dict, 
                 kernel, 
                 t0_method='closest', 
                 T=100, 
                 spherical_modes=None,
                 include_chif=False,
                 include_Mf=False):
        """
        Initialize the BGP_fit object with the required parameters.

        Args:
            times (array): The full simulation time array.
            data_dict (dict): The full dictionary of data for spherical modes directly from the sim object.
            modes (list): List of QNMs.
            Mf (float): Remnant mass (from metadata).
            chif (float): Remnant spin (from metadata).
            t0 (float): Start time.
            kernel_param_dict (dict): Kernel parameters for GP.
            kernel (callable): Kernel function for GP.
            T (float): Duration of the analysis window (i.e. anlysis runs to T + t0).
            spherical_modes (list): List of spherical modes.
        """

        # Get initial attributes 

        self.times = times
        self.data_dict = data_dict
        self.t0 = t0
        self.T = T
        self.analysis_times, self.masked_data_dict = self._mask_data(t0_method)
        self.modes = modes
        self.modes_length = len(self.modes)
        self.include_chif = include_chif
        self.include_Mf = include_Mf
        self.params = self._get_params()
        self.params_length = len(self.params)
        self.Mf = Mf
        self.chif = chif
        self.kernel_param_dict = kernel_param_dict
        self.kernel = kernel
        self.spherical_modes = spherical_modes or list(data_dict.keys())
        self.spherical_modes_length = len(self.spherical_modes)

        self.data_array = jnp.array([self.masked_data_dict[mode] for mode in self.spherical_modes])

        # Determine base QNM params 

        self.frequencies = jnp.array([
            self._get_frequency(mode) for mode in self.modes
        ])

        self.frequency_derivatives = jnp.array([
            self._get_domega_dchif(mode) for mode in self.modes
        ])

        self.mixing_coefficients = jnp.array([
            self._get_mixing(mode, sph_mode) for sph_mode in self.spherical_modes for mode in self.modes
        ]).reshape(len(self.spherical_modes), len(self.modes))

        self.mixing_derivatives = jnp.array([
            self._get_dmu_dchif(mode, sph_mode) for sph_mode in self.spherical_modes for mode in self.modes
        ]).reshape(len(self.spherical_modes), len(self.modes))

        self.exponential_terms = self._get_exponential_terms() 
        self.ls_amplitudes, self.ref_params = self._get_ls_amplitudes()

        #### Begin BGP fitting ####

        # Get inverse noise covariance matrix 
        self.inverse_noise_covariance_matrix = self.get_inverse_noise_covariance_matrix() 

        # TODO make this more robust 
        self.is_GP_diagonal = jnp.allclose(
            self.inverse_noise_covariance_matrix[0],
            jnp.diag(jnp.diagonal(self.inverse_noise_covariance_matrix[0]))
        )

        self.model_terms = self.get_model_terms()
        self.fisher_matrix = self.get_fisher_matrix() 
        self.b_vector = self.get_b_vector()
        self.mean_vector = jnp.linalg.solve(self.fisher_matrix, self.b_vector) + self.ref_params 
        self.covariance_matrix = self.get_inverse(self.fisher_matrix)

        self.samples = self._get_samples() 

    def _mask_data(self, t0_method="geq"):
        """
        Mask the data based on the specified t0_method.

        Args:
            t0_method (str): Method for masking data ("geq" or "closest").

        Returns:
            tuple: Masked times and data dictionary.
        """
        if t0_method == "geq":
            data_mask = (self.times >= self.t0 - self.epsilon) & (self.times < self.t0 + self.T - self.epsilon)
            times_mask = self.times[data_mask]
            data_dict_mask = {lm: self.data_dict[lm][data_mask] for lm in self.data_dict.keys()}
        elif t0_method == "closest":
            start_index = jnp.argmin((self.times - self.t0) ** 2)
            end_index = jnp.argmin((self.times - self.t0 - self.T) ** 2)
            times_mask = self.times[start_index:end_index]
            data_dict_mask = {lm: jnp.array(self.data_dict[lm][start_index:end_index]) for lm in self.data_dict.keys()}
        else:
            raise ValueError("Invalid t0_method. Choose between 'geq' and 'closest'.")
        return jnp.array(times_mask), data_dict_mask
    
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

    def _get_frequency(self, mode, chif=None, Mf=None):
        """
        Compute the frequency for a given QNM mode and spin.

        Args:
            mode (tuple): QNM mode (lp, mp).
            chif (float): Remnant spin.

        Returns:
            complex: Frequency of the QNM.
        """
        # This could be replaced with custom function 
        chif = chif if chif is not None else self.chif
        Mf = Mf if Mf is not None else self.Mf
        return sum([
                qnmfits.qnm.omega(ell, m, n, sign, chif, Mf, s=-2)
                for ell, m, n, sign in [
                    mode[i:i+4] for i in range(0, len(mode), 4)
                ]
            ])

    def _get_mixing(self, mode, sph_mode, chif=None, Mf=None):

        """
        Compute the mixing coefficient for a given QNM mode.

        Args:
            mode (tuple): QNM mode (lp, mp).
            chif (float): Remnant spin.

        Returns:
            complex: Mixing coefficient of the QNM. Or, if quadratic or cubic, 1. 
        """
        chif = chif if chif is not None else self.chif
        Mf = Mf if Mf is not None else self.Mf
        ell, m = sph_mode
        if len(mode) == 4:
            lp, mp, nprime, sign = mode
            return qnmfits.qnm.mu(ell, m, lp, mp, nprime, sign, chif)
        elif len(mode) == 8:
            return 1 + 0j
        elif len(mode) == 12:
            return 1 + 0j

    def _get_ls_amplitudes(self, t0_method="closest"):
        """
        Compute least-squares amplitudes and reference parameters using JAX-compatible operations.
        """
        # Perform the multimode ringdown fit using qnmfits
        ls_fit = qnmfits.multimode_ringdown_fit(
            self.times,
            self.data_dict,
            modes=self.modes,
            Mf=self.Mf,
            chif=self.chif,
            t0=self.t0,
            T=self.T,
            spherical_modes=self.spherical_modes,
            t0_method=t0_method
        )

        # Extract the complex coefficients
        C_0 = jnp.array(ls_fit["C"], dtype=jnp.complex128)

        # Construct the reference parameters
        ref_params = jnp.concatenate([
            jnp.stack([jnp.real(C_0), jnp.imag(C_0)], axis=-1).reshape(-1)
        ])
        if self.include_chif:
            ref_params = jnp.append(ref_params, self.chif)
        if self.include_Mf:
            ref_params = jnp.append(ref_params, self.Mf)

        return C_0, ref_params

    def get_inverse(self, matrix, epsilon=1e-10):
        vals, vecs = jnp.linalg.eigh(matrix)
        vals = jnp.maximum(vals, epsilon)
        return jnp.einsum('ik, k, jk -> ij', vecs, 1/vals, vecs)
    
    def compute_kernel_matrix(self, hyperparams):
        return jnp.array(
            self.kernel(np.array(self.analysis_times), **hyperparams) 
            + np.eye(len(self.analysis_times)) * 1e-13
        )

    def get_inverse_noise_covariance_matrix(self):
        return jnp.array(
            [self.get_inverse(self.compute_kernel_matrix(self.kernel_param_dict[mode])) for mode in self.spherical_modes],
            dtype=jnp.complex128,
        )
    
    def _get_exponential_terms(self):
        """Compute the exponential terms for the QNM fitting."""
        return jnp.array([
            jnp.exp(-1j * self.frequencies[i] * self.analysis_times)
            for i in range(self.modes_length)
        ])

    def _get_domega_dchif(self, mode, delta=1.0e-3):
        """Compute domega/dchif for a given QNM."""
        omega_plus = self._get_frequency(mode, chif=self.chif + delta)
        omega_minus = self._get_frequency(mode, chif=self.chif - delta)
        return (omega_plus-omega_minus)/(2*delta)

    def _get_dmu_dchif(self, mode, sph_mode, delta=0.01):
        """Compute dmu/dchif for a given QNM."""
        mu_plus = self._get_mixing(mode, sph_mode, chif=self.chif + delta)
        mu_minus = self._get_mixing(mode, sph_mode, chif=self.chif - delta)
        return (mu_plus-mu_minus)/(2*delta)
    
    def re_amplitude_model_term_generator(self, i):
        """Computes the real part of the amplitude model term for a given set of QNMs. Returns a
        len(spherical_modes) x len(t_list) array."""
        return jnp.array([
                self.exponential_terms[i] * self.mixing_coefficients[j, i]
                for j in range(self.spherical_modes_length)
            ])

    def im_amplitude_model_term_generator(self, i):
        """Computes the imaginary part of the amplitude model term for a given set of QNMs. Returns a
        len(spherical_modes) x len(t_list) array."""
        return jnp.array([
            1j * self.exponential_terms[i] * self.mixing_coefficients[j, i]
            for j in range(self.spherical_modes_length)
        ])

    def mass_model_term_generator(self):
        """Computes the Mf term for a given set of QNMs. Returns a len(spherical_modes) x len(t_list)
        array."""
        lm_matrix = jnp.einsum(
            "p,sp,pt->st",
            1j * self.frequencies / self.Mf * self.ls_amplitudes,
            self.mixing_coefficients,
            self.exponential_terms * self.analysis_times,
        )
        return lm_matrix


    def chif_model_term_generator(self):
        """Computes the chif_mag term for a given set of QNMs. Returns a len(spherical_modes) x
        len(t_list) array."""
        # TODO: may be a way to properly vectorise this? 
        lm_matrix = jnp.zeros((self.spherical_modes_length, len(self.analysis_times)), dtype=complex)
        for j in range(self.spherical_modes_length):
            term = jnp.zeros(len(self.analysis_times), dtype=complex)
            for i in range(self.modes_length):
                term += (
                    self.ls_amplitudes[i]
                    * self.exponential_terms[i]
                    * (
                        self.mixing_derivatives[j, i]
                        - 1j
                        * self.mixing_coefficients[j, i]
                        * self.analysis_times
                        * self.frequency_derivatives[i]
                    )
                )
            lm_matrix = lm_matrix.at[j].set(term)
        return lm_matrix
    
    

    def const_model_term_generator(self):
        """Computes the H_* term for a given set of QNMs."""
        lm_matrix = jnp.zeros((self.spherical_modes_length, len(self.analysis_times)), dtype=complex)
        for j in range(self.spherical_modes_length):
            lm_matrix = lm_matrix.at[j].set(
                sum(
                    self.ls_amplitudes[i]
                    * self.mixing_coefficients[j, i]
                    * self.exponential_terms[i] 
                    for i in range(self.modes_length)
                )
            )
        return lm_matrix


    def get_model_terms(self):
        sph_matrix = jnp.zeros((self.params_length, self.spherical_modes_length, len(self.analysis_times)), dtype=jnp.complex128)

        for i in range(self.modes_length):
            model_term = self.re_amplitude_model_term_generator(i)
            sph_matrix = sph_matrix.at[2*i].set(model_term)
            sph_matrix = sph_matrix.at[2*i + 1].set(1j * model_term)

        if self.include_chif:
            sph_matrix = sph_matrix.at[-1 if not self.include_Mf else -2].set(self.chif_model_term_generator())
        if self.include_Mf:
            sph_matrix = sph_matrix.at[-1].set(self.mass_model_term_generator())

        return sph_matrix


    def get_fisher_matrix(self):
        """
        Computes the Fisher information matrix for the parameters in `self.params`.

        The Fisher matrix is a square matrix that provides a measure of the amount 
        of information that an observable random variable carries about unknown 
        parameters upon which the likelihood depends. This implementation assumes 
        that the Fisher matrix is symmetric.

        Returns:
            jnp.ndarray: A 2D array representing the Fisher information matrix, 
            where the element at (i, j) corresponds to the Fisher information 
            between the i-th and j-th parameters.
        """

        #fisher_matrix = jnp.zeros((params_length, params_length), dtype=jnp.float64)
        matrix1 = jnp.conj(self.model_terms)
        matrix2 = self.model_terms
        if self.is_GP_diagonal:
        # Use the diagonal version of get_element
            fisher_matrix = jnp.einsum(
                "pst,qsu,stu->pq",
                matrix1,
                matrix2,
                self.inverse_noise_covariance_matrix,
            ) * (self.analysis_times[-1] - self.analysis_times[0]) / len(self.analysis_times)
        else:
            # Use the general version of get_element
            fisher_matrix = jnp.einsum(
                "pst,qsu,stu->pq",
                matrix1,
                matrix2,
                self.inverse_noise_covariance_matrix,
            )

        return jnp.real(fisher_matrix)


    def get_b_vector(self):
        """
        Computes the b vector for the parameters in `params`.

        The b vector is calculated based on the difference between the data 
        and a constant model term (`h_0`), projected onto the spherical 
        harmonics model terms. Each element of the b vector corresponds to 
        a parameter in `params`.

        Returns:
            jnp.ndarray: A 1D JAX array representing the b vector, where 
            each element corresponds to a parameter in `params`.
        """
        data_array_new = self.data_array - self.const_model_term_generator()
        if self.is_GP_diagonal:
        # Use the diagonal version of get_element
            b_vector = jnp.einsum(
                "pst,su,stu->p",
                jnp.conj(self.model_terms),
                data_array_new,
                self.inverse_noise_covariance_matrix,
        ) * (self.analysis_times[-1] - self.analysis_times[0]) / len(self.analysis_times)
        else:
            # Use the general version of get_element
            b_vector = jnp.einsum(
                "pst,su,stu->p",
                jnp.conj(self.model_terms),
                data_array_new,
                self.inverse_noise_covariance_matrix,
            )

        return jnp.real(b_vector)


    def _get_samples(self):
        """
        Generate samples from the posterior distribution of the parameters.

        Returns:
            jnp.ndarray: A 2D array of samples from the posterior distribution.
        """
        identity = np.eye(len(self.mean_vector))
        return scipy.stats.multivariate_normal(
            self.mean_vector, self.covariance_matrix, allow_singular=True
        ).rvs(size=1000)
