import numpy as np
import pandas as pd
import time
import pickle

# Plotting
import matplotlib.pyplot as plt
import corner

# Jax
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from functools import partial

# MCMC sampling
import numpyro
from numpyro.distributions import Beta
from numpyro.infer import MCMC, NUTS, ESS

# Eliot's qnmfits and Richard's CCE and Bayesian codes
import qnmfits
#from funcs.CCE import SXS_CCE
#from funcs.GP_funcs import get_inv_GP_covariance_matrix, kernel_main
from bayes_qnm_GP_likelihood import *


def inject_tail(times, amp, t0, tau, lambda_):
    r"""
    Used for injecting a power-law tail (PLT). 

    For t<t0, the function returns a constant A, for t>t0 the function is
    .. math::
        h(t) = A \\left( \\frac{\\tau-t}/{t_0-\\tau} \\right)^{-\\lambda}.

    INPUTS
    ------
    times: ndarray or jax array, shape=(num_times,)
        The times at which to evaluate the function.
    amp: complex
        The amplitude of the tail at time t0.
    t0: float
        The start time of the ringdown tail.
    tau: float
        Time before t0 when the function diverges. Must be negative, tau<0.
    lambda_: float
        The exponent of the PLT. Must be positive, lambda_>0.

    RETURNS
    -------
    h: complex ndarray, shaped like times
        The PLT evaluated at the sample times.
    """
    assert tau<0, "tau must be negative."
    assert lambda_>0, "lambda must be positive."
    h = amp * np.ones_like(times, dtype=complex)
    mask = times>t0
    h[mask] = amp * ((times[mask]-tau)/(t0-tau))**(-lambda_)
    return h


def get_analysis_times(sim, t0, T, t0_method='geq', epsilon=1.0e-9):
    """
    Given a numerical relativity (NR) simulation object sim, find the ringdown 
    analysis times.

    INPUTS
    ------
    sim: simulation object
        The SXS simulation, containing the simulation data.
    t0: float
        The start time of the ringdown model.
    T: float
        The duration of the analysis data. The end time is t0 + T.
    t0_method: str, optional
        A requested ringdown start time will in general lie between times on
        the time array (the same is true for the end time of the analysis). 
        There are different approaches to deal with this, which can be specified 
        here. [Defaults to 'geq'.]
            - 'geq'. Take data at times greater than or equal to t0. Note that
                we still treat the ringdown start time as occuring at t0, so the 
                best fit coefficients are defined with respect to t0.
            - 'closest'. Identify the data point occuring at a time closest to 
                t0, and take times from there.
    epsilon: float, optional
        A small tolerance value used for the 'geq' method. 
        [Defaults to 1e-9.]

    RETURNS
    -------
    times: jax array, shape=(num_times,)
        The analysis times.
    data_mask: jax array, boolean shape=(num_sim_times,)
        A mask for the sim data array.
    """
    if t0_method == "geq":
        data_mask = ( (sim.times >= t0 - epsilon) & \
                      (sim.times < t0 + T - epsilon) )
        
        times = sim.times[data_mask]

    elif t0_method == "closest":
        start_index = np.argmin((sim.times - t0) ** 2)
        end_index = np.argmin((sim.times - t0 - T) ** 2)

        data_mask = ( (start_index<=np.arange(len(sim.times))) & \
                      (np.arange(len(sim.times))<end_index) )
        
        times = sim.times[data_mask]

    else:
        ValueError("Invalid t0_method, choose between geq and closest.")

    times = jnp.array(times)
    data_mask = jnp.array(data_mask)

    return times, data_mask


def get_cov_inverse(cov, tol=1.0e-10):
    """
    Get the inverse of a covariance matrix using its eigenvalue decomposition.

    INPUTS
    ------
    cov: jax array, shape=(..., N,N)
        The covariance matrix. Must be real symmetric positive-definite.
    tol: float
        A small tolerance value. This is the minimum allowed eigenvalue.
        [Defaults to 1.0e-10.]
    
    RETURNS
    -------
    inv_cov: jax array, shaped like cov
        The inverse of the covariance matrix.
    """ 
    vals, vecs = jax.numpy.linalg.eigh(cov)
    vals = jnp.maximum(vals, tol)
    return jnp.einsum('...ik,...k,...jk->...ij', vecs, 1/vals, vecs)


class QNM_PLT_BAYES_FIT:
    r"""
    TEXT
    """

    def __init__(self, sim, modes, spherical_modes, t0, T, inv_noise_covs,
                 t0_method="geq", 
                 include_Mf=False, include_chif=False, include_lambda=False, 
                 tau_range=(-10.0, -1.0), lambda_range=(1.0, 10.0),
                 epsilon=1.0e-9, tol=1.0e-10, prior_eps=0.1,
                 num_chains=None, num_warmup=10, num_samples=1000, thin=1,
                 mcmc_method='numpyro_ess', diagnostic_plots=False,
                 reweight_samples=False):
        """
        INPUTS
        ------
        sim: SXS object
            The SXS simulation, containing the numerical relativity simulation 
            data as well as the mass and spin of the remnant BH.
        modes: list of tuples
            This determines what terms to include in the model.
            A 4-tuple (l,m,n,p) indicates a linear QNM, an 8-tuple 
            (l,m,n,p,l',m',n',p') indicates a quadratic QNM and a 12-tuple 
            (l,m,n,p,l',m',n',p',l'',m'',n'',p'') indicates a cubic QNM.
            A 2-tuple (l, m) indicates a constant offset (i.e. a zero-frequency 
            QNM). And a 3-tuple of the form (l, m, 'T') indicates a power-law 
            tail in the corresponding spherical harmonic.
            E.g. [(2,2,0,1), (2,2,0,2,2,2,0,1), (2,2), (2,2,'T')].
        spherical_modes: list of tuples, length=num_spherical_modes
            The spherical harmonic modes \beta=(l,m) to be used in the fit.
            E.g. [(2,2), (4,4)].
        t0: float
            The start time of the ringdown model.
        T: float
            The duration of the analysis data. The end time is t0 + T. 
        inv_noise_covariances: ndarray or jax array 
            The inverse noise covariance matrices for each spherical mode.
            Must have shape = (num_spherical_modes, num_times, num_times).

        OPTIONAL INPUTS
        ---------------
        include_Mf: bool, optional
            Whether to include the BH mass as a free parameter. If False, then 
            the mass will be fixed to the NR value. [Defaults to False.]
        include_chif: bool, optional
            Whether to include the BH spin as a free parameter. If False, then 
            the spin will be fixed to the NR value. [Defaults to False.]
        include_lambda: bool, optional
            Whether to include the power-law tail index as a free parameter. If 
            False, then lambda will be fixed to the Price value l+3 in each 
            mode. [Defaults to False.]
        tau_range: tuple, optional
            The prior range (tau_min, tau_max) of the power-law tail time 
            constants. Values must be negative. [Defaults to (-10.0, -1.0).]
        lambda_range: tuple, optional
            The prior range (lambda_min, lambda_max) of the power-law tail 
            indices lambda. Only used if include_lambda is True. Values must be 
            positive. [Defaults to (-10.0, -1.0).]
        epsilon: float, optional
            A small tolerance value for the t0_method='geq' method of getting 
            the analysis times. [Defaults to 1.0e-9.]
        tol: float, optional
            A small tolerance value used for inverting the covariance matrices.
            [Defaults to 1.0e-10.]
        prior_eps: float, optional
            The priors on tau and lambda are shifted and scaled symmetric Beta 
            distributions with shape parameter 1+prior_eps. As prior_eps->0, the 
            prior becomes uniform. [Defaults to 0.1.]
        num_chains: int, optional
            The number of chains to use in numpyro. If using ESS then this is 
            the number of walkers in the ensemble. If using NUTS, then this is 
            the number of independent chains that are run. If None, then the
            following default values are used. [Defaults to None.]
                - numpyro_ess: 10
                - numpyro_nuts: 1
        num_warmup: int, optional
            The number of warmup (burn in) NUTS steps. [Defaults to 10.]
        num_samples: int, optional
            The number of posterior samples to draw. [Defaults to 1000.]
        thin: int, optional
            The thinning factor for the MCMC chain. 
            [Defaults to 1 which means no thinning; be careful.]
        mcmc_method: str, optional
            Which MCMC algorithm do you want to use? The options are 
            'numpyro_nuts' and 'numpyro_ess'. [Defaults to 'numpyro_ess'.]
        diagnostic_plots: bool, optional
            Whether to plot the diagnostic plots after the MCMC sampling.
            [Defaults to True.]
        reweight_samples: bool, optional
            Whether to calculate the importance sampling weights for the 
            priors that are flat in the amplitudes. [Defaults to False.]
        """
        self.sim = sim
        self.modes = modes
        self.spherical_modes = spherical_modes
        self.t0 = t0
        self.T = T
        self.inv_sigma_array = jnp.array(inv_noise_covs)
        self.t0_method = t0_method
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.include_lambda = include_lambda
        self.tau_range = tau_range
        self.lambda_range = lambda_range
        self.epsilon = epsilon
        self.tol = tol
        self.prior_eps = prior_eps
        if num_chains is None: # default values for num_chains
            self.num_chains = {'numpyro_nuts': 1, 
                               'numpyro_ess': 10}[mcmc_method]
        else:
            self.num_chains = int(num_chains)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.thin = int(thin)
        self.mcmc_method = mcmc_method
        self.diagnostic_plots = diagnostic_plots
        
        self.times, self.data_mask = get_analysis_times(sim, t0, T, 
                                                    t0_method=self.t0_method, 
                                                    epsilon=self.epsilon)

        num_spherical_modes = len(spherical_modes)
        num_times = len(self.times)

        assert len(spherical_modes)==len(set(spherical_modes)), \
                            "Error: duplicate spherical harmonics."
        assert len(modes)==len(set(modes)), \
                            "Error: duplicate modes."
        assert inv_noise_covs.shape==(num_spherical_modes,num_times,num_times),\
                            "Error: inv_noise_covs has wrong shape."
        assert tau_range[0]<0 and tau_range[1]<0, \
                            "Error: tau values must be negative."
        assert lambda_range[0]>0 and lambda_range[1]>0, \
                            "Error: lambda values must be positive."

        # This is used in the prior for the PLT psi parameters.
        self.nearly_uniform = Beta(1.0+self.prior_eps, 1.0+self.prior_eps)

        # this is used to help index the self.plt_psi_params array later
        self.skip = (2 if include_lambda else 1) 

        # jax random number generator key, seeded using the current time
        self.key = jax.random.PRNGKey(int(time.time()))
        
        self.data = self.get_data_array()
        self.get_model_content()
        self.qnm_params, self.plt_phi_params, self.plt_psi_params = \
                                        self.get_model_parameters()
        self.psi_bounds_array = self.get_psi_bounds_array()
        self.ref_model, self.ref_residuals, self.qnm_ref_params = \
                                        self.get_reference_model_and_params()
        self.qnm_frequencies = self.get_qnm_frequncies()
        self.qnm_frequency_derivs = self.get_qnm_frequncy_derivatives()
        self.qnm_mus = self.get_qnm_mode_mixing_coefficients()
        self.qnm_mu_derivs = self.get_qnm_mode_mixing_coefficient_derivatives()
        self.plt_mus = self.get_plt_mode_mixing_coefficients()
        self.qnm_model_matrix = self.get_qnm_model_matrix()
        self.qnm_covariance_matrix, self.qnm_inv_covariance_matrix = \
                                        self.get_qnm_covariance_matrix()
        self.sample_posterior()
        if reweight_samples:
            self.reweight_samples()

    def get_data_array(self):
        """
        Arrange all the NR spherical harmonic mode analysis data into a single 
        jax array.

        RETURNS
        -------
        data : complex jax array shape=(num_spherical_modes, num_times)
            The analysis data.
        """
        data = np.zeros((len(self.spherical_modes), 
                          len(self.times)), dtype=complex)
        
        for i, mode in enumerate(self.spherical_modes):
            data[i] = self.sim.h[mode][self.data_mask]

        return jnp.array(data)
    
    def get_model_content(self):
        """
        Go through the list of modes and determine how many modes there are 
        of each type. The modes are split into linear, quadratic, cubic,
        constant offsets and power-law tails. 

        self.linear_qnms : list of tuples of form (l,m,n,p)
        self.quadratic_qnms : list of tuples of form (l,m,n,p,l',m',n',p')
        self.cubic_qnms : list of tuples  of form (l,m,n,p,l',m',n',p',L,M,N,P)
        self.constant_offsets : list of tuples of form (l,m)
        self.power_law_tails : list of tuples of form (l,m)
        """
        self.linear_qnms = []
        self.quadratic_qnms = []
        self.cubic_qnms = []
        self.constant_offsets = []
        self.power_law_tails = []

        for mode in self.modes:
            if len(mode) == 4: 
                self.linear_qnms.append(mode)
            elif len(mode) == 8: 
                self.quadratic_qnms.append(mode)
            elif len(mode) == 12: 
                self.cubic_qnms.append(mode)
            elif len(mode) == 2:
                self.constant_offsets.append(mode)
            elif len(mode) == 3:
                assert mode[2] == "T", f"Invalid mode, don't recognise {mode}."
                self.power_law_tails.append((mode[0], mode[1]))
            else:
                raise ValueError(f"Not implemented. You requested {mode=}")
            
        self.num_qnms = len(self.linear_qnms) + len(self.quadratic_qnms) + \
                        len(self.cubic_qnms) + len(self.constant_offsets)
        
        self.num_plts = len(self.power_law_tails)

    def get_model_parameters(self):
        """
        Create a list of the names of all the parameters in both parts of the 
        model and create jax arrays to store these parameters.

        self.qnm_param_names : list of strings, length = num_qnm_params
        self.plt_phi_param_names : list of strings, length = num_plt_phi_params
        self.plt_psi_param_names : list of strings, length = num_plt_psi_params

        RETURNS
        -------
        qnm_params : jax array, shape = (num_qnm_params,)
            An array for the QNM theta parameters. 
        plt_phi_params : jax array, shape = (num_plt_phi_params,)
            An array for the PLT phi parameters. 
        plt_psi_params : jax array, shape = (num_plt_psi_params,)
            An array for the PLT psi parameters.
        """
        self.qnm_param_names = []
        for mode in self.linear_qnms:
            self.qnm_param_names.append(f"Re C_{mode}")
            self.qnm_param_names.append(f"Im C_{mode}")
        for mode in self.quadratic_qnms:
            self.qnm_param_names.append(f"Re C_{mode}")
            self.qnm_param_names.append(f"Im C_{mode}")
        for mode in self.cubic_qnms:
            self.qnm_param_names.append(f"Re C_{mode}")
            self.qnm_param_names.append(f"Im C_{mode}")
        for mode in self.constant_offsets:
            self.qnm_param_names.append(f"Re B_{mode}")
            self.qnm_param_names.append(f"Im B_{mode}")
        if self.include_Mf:
            self.qnm_param_names.append("Mf")
        if self.include_chif:
            self.qnm_param_names.append("chif")

        self.plt_phi_param_names = []
        for mode in self.power_law_tails:
            self.plt_phi_param_names.append(f"Re A_{mode}")
            self.plt_phi_param_names.append(f"Im A_{mode}")

        self.plt_psi_param_names = []
        for mode in self.power_law_tails:
            self.plt_psi_param_names.append(f"tau_{mode}")
            if self.include_lambda:
                self.plt_psi_param_names.append(f"lambda_{mode}")

        qnm_params = jnp.zeros(len(self.qnm_param_names))

        plt_phi_params = jnp.zeros(len(self.plt_phi_param_names))

        plt_psi_params = np.zeros(len(self.plt_psi_param_names))
        for i, name in enumerate(self.plt_psi_param_names):
            if name.startswith("tau_"):
                plt_psi_params[i] = np.mean(self.tau_range)
            elif name.startswith("lambda_"):
                plt_psi_params[i] = np.mean(self.lambda_range)
            else:
                raise ValueError(f"Invalid name {name}.")
        plt_psi_params = jnp.array(plt_psi_params)

        return qnm_params, plt_phi_params, plt_psi_params
    
    def get_psi_bounds_array(self):
        """
        Create a jax array of the prior bounds on the PLT psi parameters.

        RETURNS
        -------
        psi_bounds_array: jax array, shape=(num_plt_psi_params, 2)
            The prior bounds. 
        """
        psi_bounds_array = np.zeros((len(self.plt_psi_param_names), 2))

        for i, name in enumerate(self.plt_psi_param_names):
            if name.startswith("tau_"):
                psi_bounds_array[i] = np.array(self.tau_range)
            elif name.startswith("lambda_"):
                psi_bounds_array[i] = np.array(self.lambda_range)
            else:
                raise ValueError(f"Invalid name {name}.")

        return jnp.array(psi_bounds_array)

    def get_reference_model_and_params(self):
        """
        A least-squares fit using just the linear QNMs (with the mass and spin 
        fixed to the NR values) is used to define the reference model.

        RETURNS
        -------
        ref_model: complex jax array, shape=(num_spherical_modes, num_times)
            The reference waveform.
        ref_residuals: complex jax array, shape=(num_spherical_modes, num_times)
            The data minus the reference model.
        qnm_ref_params: jax array, shape=(num_qnm_params,)
            The reference QNM theta parameters.
        """
        self.ls_linear_qnm_fit = qnmfits.multimode_ringdown_fit(
                        self.sim.times, self.sim.h, modes=self.linear_qnms,
                        Mf=self.sim.Mf, chif=self.sim.chif_mag, t0=self.t0, 
                        T=self.T, t0_method='geq', # THIS SHOULD BE FIXED!
                        spherical_modes=self.spherical_modes)

        ref_model = self.zero_model()

        for i, mode in enumerate(self.spherical_modes):
            ref_model[i] = self.ls_linear_qnm_fit['model'][mode]

        ref_residuals = np.array(self.data) - ref_model

        qnm_ref_params = np.zeros(len(self.qnm_param_names))

        for i, qnm in enumerate(self.linear_qnms):
            C = self.ls_linear_qnm_fit['C'][i]
            qnm_ref_params[2*i] = C.real
            qnm_ref_params[2*i+1] = C.imag

        if self.include_Mf:
            idx = self.qnm_param_names.index("Mf")
            qnm_ref_params[idx] = self.sim.Mf

        if self.include_chif:
            idx = self.qnm_param_names.index("chif")
            qnm_ref_params[idx] = self.sim.chif_mag

        ref_model = jnp.array(ref_model)
        ref_residuals = jnp.array(ref_residuals)
        qnm_ref_params = jnp.array(qnm_ref_params)

        self.qnm_params = qnm_ref_params.copy()

        return ref_model, ref_residuals, qnm_ref_params
    
    def zero_model(self, dtype=complex):
        """
        Return a numpy array of zeros for the model waveform.

        RETURNS
        -------
        model: ndarray, shape=(num_spherical_modes, num_times)
            2D array of zeros.
        """
        return np.zeros((len(self.spherical_modes), len(self.times)), 
                        dtype=dtype)
    
    def get_qnm_frequncies(self, chif=None):
        """
        Calculate the frequencies of all the QNMs in the model. This includes
        all the linear, quadratic, cubic QNMs and the constant offsets.

        INPUTS
        ------
        chif: float, optional
            The BH spin. If None, the value in self.sim is used.
            [Defaults to None.]

        RETURNS
        -------
        qnm_frequencies : complex jax array, shape=(num_qnms,)
            The complex QNM frequencies. 
        """
        if chif is None:
            chif = self.sim.chif_mag

        qnm_frequencies = np.zeros(self.num_qnms, dtype=complex)

        num_linear_qnms = len(self.linear_qnms)
        num_quadratic_qnms = len(self.quadratic_qnms)
        num_cubic_qnms = len(self.cubic_qnms)
        num_constant_offsets = len(self.constant_offsets)

        if num_linear_qnms > 0:
            a = 0
            b = num_linear_qnms
            qnm_frequencies[a:b] = qnmfits.qnm.omega_list(
                                                        self.linear_qnms, 
                                                        chif, self.sim.Mf)
        if num_quadratic_qnms > 0:
            a = num_linear_qnms
            b = num_linear_qnms + num_quadratic_qnms
            qnm_frequencies[a:b] = qnmfits.qnm.omega_list(
                                                        self.quadratic_qnms, 
                                                        chif, self.sim.Mf)
            
        if num_cubic_qnms > 0:
            a = num_linear_qnms + num_quadratic_qnms
            b = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_frequencies[a:b] = qnmfits.qnm.omega_list(
                                                        self.cubic_qnms, 
                                                        chif, self.sim.Mf)
            
        if num_constant_offsets > 0:
            a = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_frequencies[a:] = 0.*(1j)

        return jnp.array(qnm_frequencies)
    
    def get_qnm_frequncy_derivatives(self, chif=None, delta=1.0e-3):
        """
        Calculate the derivatives of the frequencies of all the QNMs with 
        respect to the BH spin. Uses second-order central finite difference.

        INPUTS
        ------
        chif: float, optional
            The BH spin. If None, the value in self.sim is used.
            [Defaults to None.]
        delta: float, optional
            The step size used for the finite difference. 
            [Defaults to 1.0e-3]

        RETURNS
        -------
        qnm_frequency_derivs : complex jax array, shape=(num_qnms,)
            The derivatives of the QNM frequencies with respect to the BH spin.
        """
        if chif is None:
            chif = self.sim.chif_mag

        assert chif>2*delta and chif<1.-2*delta, \
                                "chif must be safely between 0 and 1."
        
        omega_minus = self.get_qnm_frequncies(chif=chif-delta)
        omega_plus = self.get_qnm_frequncies(chif=chif+delta)

        omega_derivs = (omega_plus-omega_minus)/(2*delta)

        return omega_derivs
    
    def get_qnm_mode_mixing_coefficients(self, chif=None):
        """
        Calculate the mode mixing coefficients (i.e. the mu's) for all the QNMs
        in the model. 
        
        For the linear QNMs, the Kerr mixing coefficients are used, calculated 
        using the QNM package. For the quadratic, cubic and constant offset 
        QNMs, no mode mixing is used; i.e. the coefficients are set to 1 or 0.

        INPUTS
        ------
        chif: float, optional
            The BH spin. If None, the value in self.sim is used.
            [Defaults to None.]

        RETURNS
        -------
        mus : complex jax array, shape = (num_spherical_modes, num_qnms)
            The complex mode mixing coefficients of all the QNMs in the model. 
        """
        if chif is None:
            chif = self.sim.chif_mag

        qnm_mus = np.zeros((len(self.spherical_modes), self.num_qnms), 
                           dtype=complex)
        
        num_linear_qnms = len(self.linear_qnms)
        num_quadratic_qnms = len(self.quadratic_qnms)
        num_cubic_qnms = len(self.cubic_qnms)
        num_constant_offsets = len(self.constant_offsets)
        
        if num_linear_qnms > 0:
            a = 0
            b = num_linear_qnms
            qnm_mus[:,a:b] =  np.array([ qnmfits.qnm.mu_list([ mode+qnm 
                                            for qnm in self.linear_qnms], 
                                            chif)
                                        for mode in self.spherical_modes ])
            
        if num_quadratic_qnms > 0:
            a = num_linear_qnms
            b = num_linear_qnms + num_quadratic_qnms
            qnm_mus[:,a:b] = np.array([ [ 1.0+0.0*(1j) if 
                                          ( qnm[0]+qnm[4]==mode[0] and \
                                            qnm[1]+qnm[5]==mode[1] )
                                          else 0.0*(1j) 
                                        for qnm in self.quadratic_qnms]
                                    for mode in self.spherical_modes ])  
            
        if num_cubic_qnms > 0:
            a = num_linear_qnms + num_quadratic_qnms
            b = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_mus[:,a:b] = np.array([ [ 1.0+0.0*(1j) if 
                                          ( qnm[0]+qnm[4]+qnm[8]==mode[0] and \
                                            qnm[1]+qnm[5]+qnm[9]==mode[1] )
                                          else 0.0*(1j) 
                                        for qnm in self.cubic_qnms]
                                    for mode in self.spherical_modes ])  
            
        if num_constant_offsets > 0:
            a = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_mus[:,a:] = np.array([ [ 1.0+0.0*(1j) if 
                                          ( offset[0]==mode[0] and \
                                            offset[1]==mode[1] )
                                          else 0.0*(1j) 
                                        for offset in self.constant_offsets]
                                    for mode in self.spherical_modes ]) 

        return jnp.array(qnm_mus) 

    def get_qnm_mode_mixing_coefficient_derivatives(self, 
                                                    chif=None, 
                                                    delta=1.0e-3):
        """
        Calculate the derivatives of the mus of all the QNMs with respect to the 
        BH spin. Uses second-order central finite difference.

        INPUTS
        ------
        chif: float, optional
            The BH spin. If None, the value in self.sim is used.
            [Defaults to None.]
        delta: float, optional
            The step size used for the finite difference. 
            [Defaults to 1.0e-3]

        RETURNS
        -------
        mu_derivs : complex jax array, shape = (num_spherical_modes, num_qnms)
            The derivatives of the QNM frequencies with respect to the BH spin.
        """
        if chif is None:
            chif = self.sim.chif_mag

        assert self.sim.chif_mag>2*delta and self.sim.chif_mag<1.-2*delta, \
                                        "chif must be safely between 0 and 1."
        
        mu_minus = self.get_qnm_mode_mixing_coefficients(chif=chif-delta)
        mu_plus = self.get_qnm_mode_mixing_coefficients(chif=chif+delta)

        mu_derivs = (mu_plus-mu_minus)/(2*delta)

        return mu_derivs
    
    def get_plt_mode_mixing_coefficients(self):
        """
        Calculate the mode mixing coefficients (i.e. the mu's) for all the PLTs
        in the model. No mode mixing is used here, meaning the coefficients are 
        set to 1 if the l and m indices match and 0 otherwise.

        RETURNS
        -------
        mus : complex jax array, shape=(num_spherical_modes, num_plts)
            The PLT mode mixing coefficients. 
        """
        plt_mus = np.array([ [ 1.0+0.0*(1j) if 
                                ( plt_[0]==mode[0] and \
                                  plt_[1]==mode[1] )
                                else 0.0*(1j) 
                            for plt_ in self.power_law_tails]
                            for mode in self.spherical_modes ])  
        
        return jnp.array(plt_mus)
    
    def get_qnm_model_matrix(self):
        """
        Create the model matrix for the QNM part of the model. This matrix is a 
        constant (doesn't depend on theta) and therefore can be pre-computed 
        once and stored as an attrribute.

        RETURNS
        -------
        qnm_model_matrix : complex jax array, shape = (num_qnm_params,
                                                       num_spherical_modes,
                                                       num_times)
        """
        qnm_model_matrix = np.zeros((len(self.qnm_param_names), 
                                     len(self.spherical_modes), 
                                     len(self.times)), dtype=complex)
        
        all_modes = self.linear_qnms + self.quadratic_qnms + \
                    self.cubic_qnms + self.constant_offsets

        for i, qnm in enumerate(all_modes):
            omega = self.qnm_frequencies[i]
            mu = self.qnm_mus[:, i]
            exp_term = np.exp(-1j * omega * (self.times[np.newaxis,:]-self.t0))
            qnm_model_matrix[2*i] = mu[:,np.newaxis] * exp_term
            qnm_model_matrix[2*i+1] = (1j) * mu[:,np.newaxis] * exp_term

        if self.include_Mf:
            idx = self.qnm_param_names.index('Mf')
            for i, qnm in enumerate(all_modes):
                omega = self.qnm_frequencies[i]
                mu = self.qnm_mus[:, i]
                Cref = self.qnm_ref_params[2*i] + \
                            1j*self.qnm_ref_params[2*i+1]
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-self.t0))
                qnm_model_matrix[idx] += ((1j) * 
                                            mu[:,np.newaxis] * 
                                            omega * 
                                            (self.times[np.newaxis,:]-self.t0) * 
                                            Cref * 
                                            exp_term / 
                                            self.qnm_ref_params[idx] )
                
        if self.include_chif:
            idx = self.qnm_param_names.index('chif')
            for i, qnm in enumerate(all_modes):
                omega = self.qnm_frequencies[i]
                mu = self.qnm_mus[:, i]
                Cref = self.qnm_ref_params[2*i] + \
                            1j*self.qnm_ref_params[2*i+1]
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-self.t0))
                diff_term = ( self.qnm_mu_derivs[:,i,np.newaxis] - 
                        (1j)*mu[:,np.newaxis] * self.qnm_frequency_derivs[i] * \
                              (self.times[np.newaxis,:]-self.t0))
                qnm_model_matrix[idx] += ( Cref * 
                                                exp_term * 
                                                diff_term )
                
        return jnp.array(qnm_model_matrix)
    
    def plt_model_matrix(self, plt_psi_params):
        """
        Create the model matrix for the nonlinear PLT part of the model. Unlike
        the qnm_model_matrix, this matrix is a function on the PLT parameters 
        psi; therefore this function returns the model matrix rather than 
        storing it as an attribute.

        INPUTS
        ------
        plt_psi_params: jax array, shape=(num_plt_psi_params)
            The PLT psi parameters. 

        RETURNS
        -------
        plt_model_matrix: complex jax array, shape=(num_plt_phi_params, 
                                                    num_spherical_modes,
                                                    num_times)
            The model matrix for the PLT part of the model.
        """
        plt_model_matrix = jnp.zeros((len(self.plt_phi_params), 
                                      len(self.spherical_modes), 
                                      len(self.times)), dtype=complex)
        
        for i, plt_ in enumerate(self.power_law_tails):

            tau = plt_psi_params[self.skip*i]

            if self.include_lambda:
                lambda_ = plt_psi_params[self.skip*i+1]
            else:
                l, m = plt_
                lambda_ = self.default_power_law_index(l)

            plt_model_matrix = plt_model_matrix.at[2*i].set(
                    self.plt_mus[:,i,jnp.newaxis] * \
                    ((self.times[jnp.newaxis,:]-tau)/(self.t0-tau))**(-lambda_)
                )
            plt_model_matrix = plt_model_matrix.at[2*i+1].set(
                (1j) * self.plt_mus[:,i,jnp.newaxis] * \
                    ((self.times[jnp.newaxis,:]-tau)/(self.t0-tau))**(-lambda_)
                )
            
        return plt_model_matrix
    
    #@partial(jax.jit, static_argnums=(0,))
    def plt_model_matrices(self, psi_samples):
        """
        Like self.plt_model_matrix, but vectorised over many psi values. 

        INPUTS
        ------
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params)
            The PLT psi parameters. 

        RETURNS
        -------
        plt_model_matrices: complex jax array, shape=(num_samples,
                                                      num_plt_phi_params, 
                                                      num_spherical_modes,
                                                      num_times)
            The model matrix for the PLT part of the model.
        """
        plt_model_matrices = jnp.zeros((self.num_samples,
                                        len(self.plt_phi_params), 
                                        len(self.spherical_modes), 
                                        len(self.times)), dtype=complex)
        
        for i, plt_ in enumerate(self.power_law_tails):

            tau = psi_samples[:, self.skip*i]

            if self.include_lambda:
                lambda_ = psi_samples[:, self.skip*i+1]
            else:
                l, m = plt_
                lambda_ = self.default_power_law_index(l) * jnp.ones_like(tau)

            plt_model_matrices = plt_model_matrices.at[:, 2*i].set(
                jnp.einsum('s,tn->nst',
                    self.plt_mus[:,i],
                    ((self.times[:,jnp.newaxis]-tau[jnp.newaxis,:]) / \
                    (self.t0-tau[jnp.newaxis,:]))**(-lambda_[jnp.newaxis,:]))  
                )

            plt_model_matrices = plt_model_matrices.at[:, 2*i+1].set(
                jnp.einsum('s,tn->nst',
                    (1j) * self.plt_mus[:,i],
                    ((self.times[:,jnp.newaxis]-tau[jnp.newaxis,:]) / \
                    (self.t0-tau[jnp.newaxis,:]))**(-lambda_[jnp.newaxis,:]))  
                )
        
        return plt_model_matrices
    
    def get_qnm_covariance_matrix(self):
        """
        Calculate the covariance matrix of the QNM part of the model and its 
        inverse. This matrix is a constant (doesn't depend on theta) and 
        therefore can be pre-computed once and is stored as an attrribute of the
        class.
        
        RETURNS
        -------
        qnm_covariance_matrix: jax array, shape=(num_qnms, num_qnms)
            The matrix Sigma_(theta) with upstairs indices mu and nu.
        qnm_inv_covariance_matrix: jax array, shape=(num_qnms, num_qnms)
            The inverse matrix.
        """
        qnm_inv_covariance_matrix = jnp.einsum('iax,axy,jay->ij', 
                                            self.qnm_model_matrix,
                                            self.inv_sigma_array, 
                                            self.qnm_model_matrix.conj()).real
        
        qnm_covariance_matrix = get_cov_inverse(qnm_inv_covariance_matrix,
                                                tol=self.tol)
        
        return qnm_covariance_matrix, qnm_inv_covariance_matrix
    
    @partial(jax.jit, static_argnums=(0,))
    def log_likelihood_plt_nonlinearparams_marginal(self, plt_psi_params):
        """
        Calculate the log-likelihood of the PLT parameters (psi), marginalised 
        over all of the linear QNM parameters (theta) and the linear amplitude
        PLT parameters (phi).

        INPUTS
        ------
        plt_psi_params: jax array, shape=(num_plt_psi_params,)
            The nonlinear PLT psi parameters. 

        RETURNS
        -------
        log_posterior: float
            The marginal log-likelihood for the nonlinear PLT parameters.
        """
        PLT_MODEL_MATRIX = self.plt_model_matrix(plt_psi_params)

        pltmod_qnmmod = jnp.einsum('ast,stu,msu->am',
                                PLT_MODEL_MATRIX,
                                self.inv_sigma_array,
                                self.qnm_model_matrix.conj()).real
        
        res_qnmmod = jnp.einsum('st,stu,msu->m',
                                self.ref_residuals,
                                self.inv_sigma_array,
                                self.qnm_model_matrix.conj()).real

        plt_V = jnp.einsum('st,stu,asu->a',
                        self.ref_residuals,
                        self.inv_sigma_array,
                        PLT_MODEL_MATRIX.conj()).real - \
                            jnp.einsum('am,n,mn->a',
                                    pltmod_qnmmod, 
                                    res_qnmmod, 
                                    self.qnm_covariance_matrix)

        plt_M = jnp.einsum('ast,stu,bsu->ab',
                        PLT_MODEL_MATRIX,
                        self.inv_sigma_array,
                        PLT_MODEL_MATRIX.conj()).real -\
                            jnp.einsum('am,bn,mn->ab', 
                                    pltmod_qnmmod, 
                                    pltmod_qnmmod,
                                    self.qnm_covariance_matrix)

        Minv = get_cov_inverse(plt_M)

        log_like = 0.5 * jnp.einsum('ab,a,b->', Minv, plt_V, plt_V) 
        
        log_like -= 0.5 * jnp.linalg.slogdet(Minv)[1]
        
        return log_like 
    
    @partial(jax.jit, static_argnums=(0,))
    def log_prior_plt_nonlinearparams_marginal(self, plt_psi_params):
        """
        Calculate the log-prior of the PLT parameters (psi).

        INPUTS
        ------
        plt_psi_params: jax array, shape=(num_plt_psi_params,)
            The nonlinear PLT psi parameters. 

        RETURNS
        -------
        log_prior: float
            The log prior PDF.
        """
        x = ( plt_psi_params - self.psi_bounds_array[:,0] ) / \
                     jnp.ptp(self.psi_bounds_array, axis=1)
        
        log_prior = jnp.sum(self.safe_log_prior(x))

        return log_prior
    
    def safe_log_prior(self, x):
        """
        This wraps the nearly_uniform.log_prob method in such a way as to ensure 
        it return -inf (instead of nan) when the input is outside of bounds.

        INPUTS
        ------
        x: jax array, shape=(N,)
            Passed to self.nearly_uniform.log_prob.
        """
        in_support = (x > 0) & (x < 1)
        logp = self.nearly_uniform.log_prob(x)
        return jnp.where(in_support, logp, -jnp.inf)
    
    @partial(jax.jit, static_argnums=(0,))
    def log_posterior_plt_nonlinearparams_marginal(self, plt_psi_params):
        """
        Calculate the log-posterior of the PLT parameters (psi), marginalised 
        over all of the linear QNM parameters (theta) and the linear amplitude
        PLT parameters (phi).

        INPUTS
        ------
        plt_psi_params: jax array, shape=(num_plt_psi_params,)
            The nonlinear PLT psi parameters. 

        RETURNS
        -------
        logP: float
            The marginal log-posterior for the nonlinear PLT parameters.
        """
        logP = self.log_likelihood_plt_nonlinearparams_marginal(plt_psi_params)
        logP += self.log_prior_plt_nonlinearparams_marginal(plt_psi_params)
        return logP
    
    @partial(jax.jit, static_argnums=(0,))
    def potential_fn(self, x):
        """ 
        This wraps the log_posterior function for use by numpyro.
        """
        return -self.log_posterior_plt_nonlinearparams_marginal(x['params'])
    
    def sample_posterior(self):
        """
        Sample from the posterior distribution for the full ringdown model.
        """
        if len(self.power_law_tails)==0:
            self.key, subkey = jax.random.split(self.key)
            theta_samples, self.key = self.sample_qnm_params_simple(subkey)

            self.samples = pd.DataFrame(theta_samples, 
                                        columns=self.qnm_param_names)

        else:
            # If there are tails, then it's a 3 stage process:
            # 1. use numpyro NUTS to sample the PLT psi parameters from P(psi).
            # --------
            t0_ = time.time()
            print(f"Running MCMC method {self.mcmc_method}")
            if self.mcmc_method=='numpyro_nuts':
                psi_samples = self.sample_numpyro_nuts_plt_psi_params_marginal(
                                        diagnostic_plots=self.diagnostic_plots)
            elif self.mcmc_method=='numpyro_ess':
                psi_samples = self.sample_numpyro_ess_plt_psi_params_marginal(
                                        diagnostic_plots=self.diagnostic_plots)
            else:
                raise ValueError(f"Invalid mcmc_method {self.mcmc_method}.")
            t1 = time.time()
            print(f"Sampling stage 1: {t1-t0_:.2f} seconds")
            
            # 2. sample the plt_phi parameters from P(phi|psi).
            # --------
            t0_ = time.time()
            self.key, subkey = jax.random.split(self.key)
            phi_samples, self.key = self.sample_plt_phi_params_conditional(
                                                                psi_samples,
                                                                subkey)
            t1 = time.time()
            print(f"Sampling stage 2: {t1-t0_:.2f} seconds")

            # 3. sample the QNM theta parameters from P(theta|phi,psi).
            # --------
            t0_ = time.time()
            self.key, subkey = jax.random.split(self.key)
            theta_samples, self.key = self.sample_qnm_params_conditional(
                                                                phi_samples,
                                                                psi_samples,
                                                                subkey)
            t1 = time.time()
            print(f"Sampling stage 3: {t1-t0_:.2f} seconds")

            self.samples = np.hstack((theta_samples, phi_samples, psi_samples))

            self.samples = pd.DataFrame(self.samples, 
                                        columns=self.qnm_param_names +
                                                self.plt_phi_param_names +
                                                self.plt_psi_param_names)

    def sample_numpyro_nuts_plt_psi_params_marginal(self, 
                                                    diagnostic_plots=True):
        """
        Use the numpyro NUTS sampler to sample the PLT psi parameters from their
        posterior distribution marginalised over all theta and phi parameters. 
        
        INPUTS
        ------
        diagnostic_plots: bool, optional
            If True, diagnostic plots are shown. [Defaults to True.]

        RETURNS
        -------
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params)
            The psi samples.
        """
        nuts_kernel = NUTS(potential_fn=self.potential_fn)

        a = self.num_samples*self.thin
        b = self.num_chains
        num_iter = int(a/b) + int(a%b>0)

        mcmc = MCMC(nuts_kernel, 
                    num_warmup=self.num_warmup, 
                    num_samples=num_iter,
                    num_chains=self.num_chains, 
                    chain_method="parallel")
        
        start = self.initial_psi_mcmc_positions()
        
        self.key, subkey = jax.random.split(self.key)
        mcmc.run(subkey, init_params={"params": start})
        
        mcmc.print_summary()

        psi_samples = mcmc.get_samples()['params']
        psi_samples = psi_samples[::self.thin][:,:self.num_samples]

        print(f"Drawn {len(psi_samples)} psi samples from P(psi|d).")

        if diagnostic_plots:
            self.diagnostic_psi_mcmc_plots(mcmc)

        return psi_samples
    
    def sample_numpyro_ess_plt_psi_params_marginal(self, diagnostic_plots=True):
        """
        Use the numpyro ESS sampler to sample the PLT psi parameters from their
        posterior distribution marginalised over all theta and phi parameters. 
        
        INPUTS
        ------
        diagnostic_plots: bool, optional
            If True, diagnostic plots are shown. [Defaults to True.]

        RETURNS
        -------
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params)
            The psi samples.
        """
        ess_kernel = ESS(potential_fn=self.potential_fn)

        a = self.num_samples*self.thin
        b = self.num_chains
        num_iter = int(a/b) + int(a%b>0)

        mcmc = MCMC(ess_kernel, 
                    num_warmup=self.num_warmup, 
                    num_samples=num_iter,
                    num_chains=self.num_chains, 
                    chain_method="vectorized")
        
        start = self.initial_psi_mcmc_positions()
        
        self.key, subkey = jax.random.split(self.key)
        mcmc.run(subkey, init_params={"params": start})
        
        mcmc.print_summary()

        psi_samples = mcmc.get_samples()['params']
        psi_samples = psi_samples[::self.thin][:,:self.num_samples]
    
        print(f"Drawn {len(psi_samples)} psi samples from P(psi|d).")

        if diagnostic_plots:
            self.diagnostic_psi_mcmc_plots(mcmc)

        return psi_samples
    
    def initial_psi_mcmc_positions(self):
        """
        Initial positions for the mcmc walkers in the psi coordinates.

        RETURNS
        -------
        start: jax array
            The initial positions for the mcmc walkers.
            If num_chains==1, then shape=(num_plt_psi_params).
            If num_chains>1, then shape=(num_chains, num_plt_psi_params).
        """
        start = jnp.array([[ bound[0]+np.random.beta(2,2)*(bound[1]-bound[0])
                                for bound in self.psi_bounds_array ] 
                                for i in range(self.num_chains) ])
        
        if self.num_chains==1:
            start = start[0]

        return start
    
    def diagnostic_psi_mcmc_plots(self, mcmc):
        """
        Plot a trace- and corner- plot for the mcmc object.

        INPUTS
        ------
        mcmc: numpyro MCMC object
            Contains the MCMC chains.
        """
        samples = mcmc.get_samples()['params']
        psi_samples = np.array(samples[::self.thin])[:,:self.num_samples]
        samples_by_chain = mcmc.get_samples(group_by_chain=True)['params']

        labels = self.plt_psi_param_names

        fig, ax = plt.subplots(nrows=len(self.plt_psi_param_names), ncols=1)
        if len(self.plt_psi_param_names)>1:
            for i in range(len(self.plt_psi_param_names)):
                for w in range(self.num_chains):
                    ax[i].plot(samples_by_chain[w, :, i], alpha=0.5)
                ax[i].set_ylabel(labels[i])
                if i == len(self.plt_psi_param_names)-1:
                    ax[i].set_xlabel('Iteration')
        else:
            i=0
            for w in range(self.num_chains):
                ax.plot(samples[w, :], alpha=0.5)
            ax.set_ylabel(labels[i])
            ax.set_xlabel('Iteration')
        fig.suptitle('Traceplots')
        plt.tight_layout()
        plt.show()

        corner.corner(psi_samples, labels=labels)
        plt.show()
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_plt_phi_params_conditional(self, psi_samples, key):
        """
        Given a large number of psi samples, sample corresponding values for the
        PLT phi parameters from the conditional distribution phi|psi.

        INPUTS
        ------
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params)
            Samples from the marginal posterior on the PLT psi parameters.
        key: jax rng key
            The jax random number generator key.

        RETURNS
        -------
        phi_samples: jax array, shape=(num_samples, num_plt_phi_params)
            Samples from the conditional posterior on the PLT phi parameters.
        key: jax rng key
            The jax random number generator key.
        """
        nsamples, _ = psi_samples.shape

        phi_samples = jnp.zeros((nsamples, len(self.plt_phi_param_names))) 

        PLT_MODEL_MATRIX = self.plt_model_matrices(psi_samples)

        pltmod_qnmmod = jnp.einsum('Nast,stu,msu->Nam',
                                    PLT_MODEL_MATRIX,
                                    self.inv_sigma_array,
                                    self.qnm_model_matrix.conj()).real
            
        res_qnmmod = jnp.einsum('st,stu,msu->m',
                                self.ref_residuals,
                                self.inv_sigma_array,
                                self.qnm_model_matrix.conj()).real

        plt_V = jnp.einsum('st,stu,Nasu->Na',
                            self.ref_residuals, 
                            self.inv_sigma_array,
                            PLT_MODEL_MATRIX.conj()).real - \
                                jnp.einsum('Nam,n,mn->Na',
                                            pltmod_qnmmod, 
                                            res_qnmmod, 
                                            self.qnm_covariance_matrix)
        
        plt_M = jnp.einsum('Nast,stu,Nbsu->Nab',
                            PLT_MODEL_MATRIX,
                            self.inv_sigma_array,
                            PLT_MODEL_MATRIX.conj()).real - \
                                jnp.einsum('Nam,Nbn,mn->Nab', 
                                            pltmod_qnmmod, 
                                            pltmod_qnmmod,
                                            self.qnm_covariance_matrix)
        
        inv_M = get_cov_inverse(plt_M)
        
        phi_mean = jnp.einsum('Nab,Na->Nb', inv_M, plt_V)

        key, subkey = jax.random.split(key)
        phi_samples = jax.random.multivariate_normal(subkey, phi_mean, inv_M)

        return phi_samples, key
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_qnm_params_simple(self, key):
        """ 
        INPUTS
        ------
        key: jax rng key
            The jax random number generator key.

        RETURNS
        -------
        theta_samples: jax array, shape=(num_samples, num_qnm_params)
            The QNM theta parameters.
        key: jax rng key
            The jax random number generator key.
        """
        shift = jnp.einsum('ist,stu,su->i',
                          self.qnm_model_matrix,
                          self.inv_sigma_array,
                          self.ref_residuals.conj()).real
    
        mean = self.qnm_ref_params + \
                             jnp.einsum('ij,j->i', 
                                        self.qnm_covariance_matrix,
                                        shift)

        key, subkey = jax.random.split(self.key)
        theta = jax.random.multivariate_normal(subkey, 
                                               mean, 
                                               self.qnm_covariance_matrix,
                                               shape=(self.num_samples))
        
        return theta, key
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_qnm_params_conditional(self, phi_samples, psi_samples, key):
        """
        Sample the conditional distribution theta|phi,psi.

        INPUTS
        ------
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params)
            The PLT psi parameters.
        phi_samples: jax array, shape=(num_samples, num_plt_phi_params)
            The PLT phi parameters.
        key: jax rng key
            The jax random number generator key.

        RETURNS
        -------
        theta_samples: jax array, shape=(num_samples, num_qnm_params)
            The QNM theta parameters.
        key: jax rng key
            The jax random number generator key.
        """
        num_samples, _ = phi_samples.shape

        residuals = self.ref_residuals[jnp.newaxis,:,:] - \
                    self.plt_models(phi_samples, psi_samples)
        
        shift = jnp.einsum('ist,stu,Nsu->Ni',
                          self.qnm_model_matrix,
                          self.inv_sigma_array,
                          residuals.conj()).real
    
        mean = self.qnm_ref_params[jnp.newaxis,:] + \
                             jnp.einsum('ij,Nj->Ni', 
                                        self.qnm_covariance_matrix,
                                        shift)
    
        new_cov = jnp.broadcast_to(self.qnm_covariance_matrix, 
                            (num_samples, *self.qnm_covariance_matrix.shape))

        key, subkey = jax.random.split(self.key)
        theta = jax.random.multivariate_normal(subkey, 
                                               mean, 
                                               new_cov)
        
        print(f"Drawn {num_samples} theta samples from P(theta|phi,psi,d).")

        return theta, key
    
    @partial(jax.jit, static_argnums=(0,))
    def plt_model(self, plt_phi_params, plt_psi_params):
        """
        Return the PLT part of the waveform model.

        INPUTS
        ------
        plt_phi_params: jax array, shape=(num_plt_phi_params,)
            The PLT phi parameters. 
        plt_psi_params: jax array, shape=(num_plt_psi_params,)
            The PLT psi parameters. 

        RETURNS
        -------
        hPLT: jax array, shape=(num_spherical_modes, num_times)
            The PLT part of the waveform model.
        """
        PLT_MODEL_MATRIX = self.plt_model_matrix(plt_psi_params)

        return jnp.einsum('a,ast->st', plt_phi_params, PLT_MODEL_MATRIX)
    
    def qnm_model(self, qnm_params):
        """
        Return the QNM part of the waveform model.

        INPUTS
        ------
        qnm_params: ndarray, shape=(num_qnm_params,)
            The QNM theta parameters. 

        RETURNS
        -------
        hQNM: ndarray, shape=(num_spherical_modes, num_times)
            The QNM part of the waveform model.
        """
        hQNM = self.ref_model
        hQNM += jnp.einsum('a,ast->st', 
                           qnm_params-self.qnm_ref_params, 
                           np.array(self.qnm_model_matrix))
        return hQNM
    
    def map_model(self):
        """
        """
        shift = jnp.einsum('ist,stu,su->i',
                          self.qnm_model_matrix,
                          self.inv_sigma_array,
                          self.ref_residuals.conj()).real
    
        qnm_map_params = self.qnm_ref_params + \
                        jnp.einsum('ij,j->i', 
                                    self.qnm_covariance_matrix,
                                    shift)

        return self.qnm_model(qnm_map_params)
    
    @partial(jax.jit, static_argnums=(0,))
    def plt_models(self, phi_samples, psi_samples):
        """
        Same as self.plt_model, but vectorised over many phi and psi values.

        INPUTS
        ------
        phi_samples: jax array, shape=(num_samples, num_plt_phi_params,)
            The PLT phi parameters. 
        psi_samples: jax array, shape=(num_samples, num_plt_psi_params,)
            The PLT psi parameters. 

        RETURNS
        -------
        hPLT: jax array, shape=(num_samples, num_spherical_modes, num_times)
            The PLT part of the waveform model.
        """
        PLT_MODEL_MATRICES = self.plt_model_matrices(psi_samples)

        return jnp.einsum('Na,Nast->Nst', phi_samples, PLT_MODEL_MATRICES)
    
    def bsquared(self, mode):
        """
        Calculate the squared modulus of the b vector for a given mode. This
        is related to the log-significance (see below).

        INPUTS
        ------
        mode: tuple
            The mode to calculate the significance of. Can be any of the 
            linear/quadratic/cubic QNMs, constant offset, or PLT in the fit.
            E.g. (2,2,0,1), (2,2,0,1,2,2,0,1), or (2,2) ot (2,2,'T').

        RETURNS
        -------
        bsq: float
            The square of the modulus of b.
        """
        if ((len(mode) == 4) or (len(mode) == 8) or (len(mode) == 12)): 
            x = 'Re C_{}'.format(mode)
            y = 'Im C_{}'.format(mode)
        elif len(mode) == 2:
            x = 'Re B_{}'.format(mode)
            y = 'Im B_{}'.format(mode)
        elif len(mode) == 3:
            assert mode[2] == "T", f"Invalid mode, don't recognise {mode}."
            plt_ = (mode[0], mode[1])
            x = 'Re A_{}'.format(plt_)
            y = 'Im A_{}'.format(plt_)
        else:
            raise ValueError(f"Unrecognised option, {mode=}")
        
        samples = np.vstack((self.samples[x], self.samples[y])).T

        marginal_mean = np.mean(samples, axis=0)
        marginal_covariance = np.cov(samples, rowvar=False)

        L = np.linalg.cholesky(marginal_covariance)
        b_a = np.linalg.solve(L, marginal_mean) 

        b_squared = np.dot(b_a, b_a)

        return b_squared
    
    def log_significance(self, mode):
        """
        Estimate the significance of given mode in the fit with complex 
        amplitude C=x+iy. The significance is defined as 

        .. math::
            S = int_{x,y|P(x,y)>P(0,0)} dx dy P(x, y).
        
        This integral is estimated by approximating the integrand (the 2D 
        marginalised posterior on x and y) as a 2D Gaussian using the sample 
        mean and covariance calculated from the posterior samples.

        INPUTS
        ------
        mode: tuple
            The mode to calculate the significance of. Can be any of the 
            linear/quadratic/cubic QNMs, constant offset, or PLT in the fit.
            E.g. (2,2,0,1), (2,2,0,1,2,2,0,1), or (2,2) ot (2,2,'T').

        RETURNS
        -------
        logS: float
            The log significance of the given mode.
        """
        if ((len(mode) == 4) or (len(mode) == 8) or (len(mode) == 12)): 
            x = 'Re C_{}'.format(mode)
            y = 'Im C_{}'.format(mode)
        elif len(mode) == 2:
            x = 'Re B_{}'.format(mode)
            y = 'Im B_{}'.format(mode)
        elif len(mode) == 3:
            assert mode[2] == "T", f"Invalid mode, don't recognise {mode}."
            plt_ = (mode[0], mode[1])
            x = 'Re A_{}'.format(plt_)
            y = 'Im A_{}'.format(plt_)
        else:
            raise ValueError(f"Unrecognised option, {mode=}")
        
        samples = np.vstack((self.samples[x], self.samples[y])).T

        marginal_mean = np.mean(samples, axis=0)
        marginal_covariance = np.cov(samples, rowvar=False)

        L = np.linalg.cholesky(marginal_covariance)
        b_a = np.linalg.solve(L, marginal_mean) 

        b_squared = np.dot(b_a, b_a)

        if b_squared<25:
            logS = np.log( 1 - np.exp(-0.5*b_squared) )
        else:
            logS = -np.exp(-0.5*b_squared)

        return logS
    
    def reweight_samples(self):
        """
        By default, the posterior samples are drawn using priors that are flat
        on the real and imaginary parts of every amplitude parameter. If instead 
        flat priors on the amplitude and phase are desired, then use the 
        importance sampling weights computed here.

        self.log_weights : ndarray, shape=(num_samples,)
        """
        log_weights = np.zeros(len(self.samples))

        all_modes = self.linear_qnms + self.quadratic_qnms + \
                    self.cubic_qnms + self.constant_offsets
        
        for i, qnm in enumerate(all_modes):
            if len(qnm)==2:
                re = self.samples['Re B_{}'.format(qnm)]
                im = self.samples['Im B_{}'.format(qnm)] 
            else:
                re = self.samples['Re C_{}'.format(qnm)]
                im = self.samples['Im C_{}'.format(qnm)]
            amp = np.sqrt(re**2 + im**2)
            log_weights -= np.log(amp)

        for i, plt_ in enumerate(self.power_law_tails):
            re = self.samples['Re A_{}'.format(plt_)]
            im = self.samples['Im A_{}'.format(plt_)]
            amp = np.sqrt(re**2 + im**2)
            log_weights -= np.log(amp)

        self.samples['log_weights'] = log_weights

        self.neff = np.sum(np.exp(log_weights))**2 / \
                        np.sum(np.exp(log_weights)**2)

        if self.neff<10:
            print("WARNING: effective number of reweighted samples", 
                  "is very low, neff={}.".format(self.neff))
            
    def default_power_law_index(self, l):
        """
        If include_lambda is False, then by default the power law indeces are 
        set to the Price values.

        INPUTS
        ------
        l: int
            The spherical harmonic l index of the PLT.
        
        RETURNS
        -------
        lambda_: float
            The default power law index for the PLT.
        """
        lambda_ = l + 3
        return lambda_
    

def test_load(id="0001", t0=20.0, T=80.0, spherical_modes=[(2,2)]):
    """
    UNIT TESTING
    Load the data.
    """
    sim = SXS_CCE(id, lev="Lev5", radius="R2")

    times, _ = get_analysis_times(sim, t0, T)

    with open('tuned_params.pkl', 'rb') as f:
        params = pickle.load(f)
    tuned_param_dict_main = params[id]

    inv_noise_covariances = get_inv_GP_covariance_matrix(
                                                times, 
                                                kernel_main,
                                                tuned_param_dict_main, 
                                                spherical_modes=spherical_modes
                                                ).real
    
    return sim, inv_noise_covariances


def test1(id="0001", t0=30.0, T=70.0):
    """
    UNIT TESTING
    The simplest type of fit - fitting a couple of QNMs to a single harmonic
    """
    print("RUNNING TEST 1...")

    n_max = 2
    mode_list = [(2, 2, n, 1) for n in range(n_max+1)] + [(3, 2, 0, 1)]

    spherical_modes = [(2, 2)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances,
                            num_samples=10000)
    
    for mode in mode_list:
        print(f"Significance S_{mode}={np.exp(fit.log_significance(mode))}.")
    
    x, y = 'Re C_(2, 2, 0, 1)', 'Im C_(2, 2, 0, 1)'
    least_squares_value = np.array([fit.ls_linear_qnm_fit['C'][0].real,
                                    fit.ls_linear_qnm_fit['C'][0].imag])
    fig = corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                labels=[x, y], truths=least_squares_value, color='red')
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                        labels=[x, y], fig=fig, 
                        weights=np.exp(fit.samples['log_weights']), 
                        color='blue')
    plt.show()

    print("... PASSED TEST 1\n")

def test2(id="0001", t0=10.0, T=90.0):
    """
    UNIT TESTING
    A complicated multimode QNM fit involving linear QNMs, nonlinear QNMs, 
    constant offsets and with the mass and spin included as free parameters.
    """
    print("RUNNING TEST 2...")
    n_max = 3
    mode_list = [(2, 2, n, 1) for n in range(n_max+1)] + [(3, 2, 0, 1)]
    n_max = 3
    mode_list += [(4, 4, n, 1) for n in range(n_max+1)] 
    n_max = 1
    mode_list += [(6, 6, n, 1) for n in range(n_max+1)]
    mode_list += [(2, 2, 0, 1, 2, 2, 0, 1), (3, 3, 0, 1, 3, 3, 0, 1)]
    mode_list += [(2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1)]
    mode_list += [(2, 2), (4, 4), (6, 6)]
    
    spherical_modes = [(2, 2), (4, 4), (6, 6)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances,
                            include_Mf=True,
                            include_chif=True)
    
    for mode in mode_list:
        print(f"Significance S_{mode}={np.exp(fit.log_significance(mode))}.")

    x, y = 'Mf', 'chif'
    fig = corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                        labels=[x, y],truths=[fit.sim.Mf, fit.sim.chif_mag],
                        color='red')
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                        labels=[x, y],truths=[fit.sim.Mf, fit.sim.chif_mag],
                        fig=fig, weights=np.exp(fit.samples['log_weights']),
                        color='blue')
    plt.show()

    print("... PASSED TEST 2\n")


def test3(id="0001", t0=20.0, T=80.0):
    """
    UNIT TESTING
    A fit involving fitting for a large power law tail that we inject.
    """
    print("RUNNING TEST 3...")

    mode_list = [(2, 2, 0, 1), (2, 2, 1, 1), (3, 2, 0, 1), (4, 4, 0, 1)]
    mode_list += [(4, 4, 'T')]

    spherical_modes = [(2, 2), (4, 4)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)
    
    A_inj, tau_inj, lambda_inj = 0.1, -10.0, 7.0

    sim.h[(4,4)] = inject_tail(sim.times, A_inj, t0, tau_inj, lambda_inj)

    plt.plot(sim.times, sim.h[(4,4)].real)
    plt.show()

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances, 
                            tau_range=(tau_inj-1, tau_inj+1))
    
    x, y, z = 'Re A_(4, 4)', 'Im A_(4, 4)', 'tau_(4, 4)'
    corner.corner(np.vstack((fit.samples[x], 
                             fit.samples[y], 
                             fit.samples[z])).T, 
                  labels=[x, y, z], 
                  truths=[A_inj.real, A_inj.imag, tau_inj])
    plt.show()

    print("... PASSED TEST 3\n")


def test4(id="0001", t0=50.0, T=100.0):
    """ 
    UNIT TESTING
    Fit for lots of things at once, including a power law tail with lambda 
    included as a free parameter.
    """
    print("RUNNING TEST 4...")

    mode_list = [(2, 2, 0, 1), (2, 2, 'T')]

    spherical_modes = [(2, 2)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances, 
                            include_lambda=True,
                            tau_range=(-10., -1.),
                            lambda_range=(1., 5.))
    
    w, x, y, z = 'Re A_(2, 2)', 'Im A_(2, 2)', 'tau_(2, 2)', 'lambda_(2, 2)'
    corner.corner(np.vstack((fit.samples[w], fit.samples[x], 
                             fit.samples[y], fit.samples[z])).T, 
                  labels=[w, x, y, z])
    plt.show()

    print("... PASSED TEST 4\n")


if __name__ == "__main__":

    test1()
    test2()
    test3()
    test4()

    print("#######################")
    print("## PASSED ALL TESTS! ##")
    print("#######################")
