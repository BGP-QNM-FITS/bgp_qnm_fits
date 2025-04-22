import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import scipy
import pickle
import time

# Eliot's qnmfits package and Richard's CCE and Bayesian fitting code
import qnmfits
from funcs.CCE import SXS_CCE
from funcs.GP_funcs import get_inv_GP_covariance_matrix, kernel_main

# zeus_mcmc is used for the MCMC sampling
import zeus
from multiprocessing import Pool


def logP(x, fit):
    """
    This wraps the target PDF that is passed into zeus.

    INPUTS
    ------
    x: ndarray, shape=(num_params,) 
        the nonlinear psi parameters of the PLT part of the model
    fit: class
        The QNM_PLT_BAYES_FIT object
    
    RETURNS
    -------
    logP: float
        the logarithm of the posterior PDF
    """
    return fit.log_posterior_plt_nonlinearparams_marginal(plt_psi_params=x)


def inject_tail(times, amp, t0, tau, lambda_):
    """
    A power-law tail function. This is used to artificially inject power law 
    tails into waveforms for testing.

    For t<t0, the function returns a constant A, for t>t0 the function is
    .. math::
        h(t) = A \\left( \\frac{\\tau-t}/{t0-\\tau} \\right)^{-\\lambda}.

    INPUTS
    ------
    times: ndarray, shape=(num_times,)
        The times at which to evaluate the function.
    amp: complex
        The amplitude of the tail. Our convention is this is the amplitude at 
        time t0.
    t0: float
        The start time of the ringdown tail.
    tau: float
        How long before t0 does the power law diverge? Must be negative, tau<0.
    lambda_: float
        The exponent of the power law tail. Must be positive, lambda_>0.

    RETURNS
    -------
    h: ndarray, shape=(num_times,)
        The power-law tail function evaluated at the sample times.
    """
    assert tau<0, "tau must be negative."
    assert lambda_>0, "lambda must be positive."
    h = amp * np.ones_like(times, dtype=complex)
    mask = times>t0
    h[mask] = amp * ((times[mask]-tau)/(t0-tau))**(-lambda_)
    return h


def get_analysis_times(sim, t0, T, t0_method='geq', epsilon=1.0e-9):
    """
    Given a NR simulation, find the ringdown analysis times..

    INPUTS
    ------
    sim: sim object
        The SXS simulation, containing numerical relativity simulation data.
    t0: float
        The start time of the ringdown model.
    T: float
        The duration of the analysis data. The end time is t0 + T.
    t0_method: str, optional
        A requested ringdown start time will in general lie between times on
        the time array (the same is true for the end time of the analysis). 
        There are different approaches to deal with this, which can be 
        specified here. [Defaults to 'geq']
            - 'geq'
                Take data at times greater than or equal to t0. Note that
                we still treat the ringdown start time as occuring at t0,
                so the best fit coefficients are defined with respect to 
                t0.
            - 'closest'
                Identify the data point occuring at a time closest to t0, 
                and take times from there.
    epsilon: float, optional
        A small tolerance value used for the 'geq' method. 
        [Defaults to 1e-9.]

    RETURNS
    -------
    times: ndarray, shape=(num_times,)
        The analysis times.
    data_mask: boolean ndarray, shape=(num_sim_times,)
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

    return times, data_mask


def get_cov_inverse(cov, tol=1.0e-10):
    """
    Get the inverse of a covariance matrix using its eigenvalue decomposition.

    INPUTS
    ------
    cov: ndarray, shape=(N,N)
        The covariance matrix to invert. Must be symmetric positive-definite.
    tol: float
        A small tolerance value. This is the minimum allowed eigenvalue.
        [Defaults to 1.0e-10.]
    
    RETURNS
    -------
    inv_cov: ndarray, shape=(N,N)
        The inverse of the covariance matrix.
    """ 
    vals, vecs = scipy.linalg.eigh(cov)
    vals = np.maximum(vals, tol)
    return np.einsum('ik, k, jk -> ij', vecs, 1/vals, vecs)


class QNM_PLT_BAYES_FIT:
    r"""
    This class is used to perform bayesian fits of combinations of linear quasi
    normal modes (QNMs), quadratic QNMS, cubic QNMS, constant offsets and 
    power-law tails (PLTs) to multiple spherical harmonics of waveforms from 
    numerical relativity simulations.

    The data is the (complex) numerical relativity waveforms in multiple 
    spherical harmonic modes, h^\\beta(t), where the index \\beta=(l,m).

    The model is split into two parts:

    .. math::
        h^\\beta(t) = h^\\beta_{QNM}(t) + h^\\beta_{PLT}(t) .

    Here, the QNM part contains all the QNMs (linear, quadratic, cubic etc) 
    and any constant offset terms which can be regarded as zero-frequency QNMs.
    part and h^\\beta_{PLT}(t) is the power-law tail part. The PLT part contains
    only a sum of power-law tails.

    The QNM part of the model is linear in all of its parameters. (If the mass
    and spin are included, then the model is linearised in these parameters.)
    The model is expanded about some reference parameters which are determined
    by a least-squares fit using just the linear QNMs. The model is then

    .. math::
        h^\\beta_{QNM}(t) = H_*^\\beta + 
                        \\sum_\mu (\\theta^\mu-\\theta_*^\mu) dh^\\beta_\\mu(t)

    where H_*^\\beta is the reference model, \\theta^\\mu are QNM parameters, 
    \\theta_*^\\mu are the reference QNM parameters and dh^\beta_\\mu(t) is the 
    model matrix (i.e. the matrix of first derivatives).

    The PLT part of the model consists of a sum of power-law tails in each 
    spherical harmonic mode. (I.e. no mode mixing.) Each term looks like, 

    .. math::
        h^\\beta_{PLT}(t) = A^\\beta \\left( \\frac{\\tau-t}/{t0-\\tau} 
                                            \\right)^(-\\lambda).

    The PLT parameters are split into two groups: the A^\\beta amplitude 
    parameters collectively denoted \\phi, and the peak time and exponent 
    parameters which are collectively denoted \\psi. The model is linear in the 
    \\phi, but nonlinear in \\psi. The \\lambda indices can either be included 
    as free parameters or fixed to the Price values of l+3. 

    The fits are performed using a Gaussian likelihood function, which treats
    the noise (a.k.a. numerical error) in each \\beta harmonic as an independent
    zero-mean Gaussian with noise covariance matrix \\Sigma^\\beta_{ij}. Flat 
    priors are used for the real/imag parts of all amplitude parameters, the
    BH mass and spin, the peak times (tau) and the power law indices (lambda).

    The strategy for sampling the full posterior on all of the model parameters
    (\\theta,\\phi,\\psi) exploits the fact that the model is linear in most
    of the parameters. We use the following factorisation of the posterior: 

    .. math::
        P(\\theta,\\phi,\\psi) = P(\\psi) P(\\phi|\\psi) P(\\theta|\\phi,\\psi).

    An MCMC is used to sample the (analytically marginalised) posterior on the
    nonlinear PLT parameters P(\\psi). Then the conditionals P(A|\\psi) and 
    P(\\theta|A,\\psi) are both Gaussian and can be sampled easily.
    """

    def __init__(self, sim, modes, spherical_modes, t0, T, 
                 inv_noise_covariances, t0_method="geq", 
                 include_Mf=False, include_chif=False, include_lambda=False, 
                 tau_range=(-100.0, -1.0), lambda_range=(1.0, 10.0),
                 epsilon=1.0e-9, tol=1.0e-10,
                 num_samples=1000, MCMCnwalkers=20, MCMCnburnin=10, MCMCthin=5):
        """
        INPUTS
        ------
        sim: SXS object
            The SXS simulation, containing the numerical relativity simulation 
            data as well as the mass and spin of the remnant BH.
        modes: list ot tuples
            This determines what terms to include in the model.
            A 4-tuple (l,m,n,p) indicates a linear QNM, an 8-tuple 
            (l,m,n,p,l',m',n',p') indicates a quadratic QNM and a 12-tuple 
            (l,m,n,p,l',m',n',p',l'',m'',n'',p'') indicates a cubic QNM.
            A 2-tuple (l, m) indicates a constant offset (i.e. a zero-frequency 
            QNM). And a 3-tuple of the form (l, m, 'T') indicates a power-law 
            tail in the corresponding spherical harmonic.
            E.g. [(2,2,0,1), (2,2,0,2,2,2,0,1), (2,2), (2,2,'T')].
        spherical_modes: list, length = num_spherical_modes
            The spherical harmonic modes \beta=(l,m) to be used in the fit.
            E.g. [(2,2), (4,4)].
        t0: float
            The start time of the ringdown model.
        T: float
            The duration of the analysis data. The end time is t0 + T. 
        inv_noise_covariances: ndarray, shape = (num_spherical_modes, 
                                                     num_times,
                                                     num_times)
            The inverse noise covariance matrices for each spherical mode.
        t0_method: str, optional
            Passes to get_analysis_times(). [Defaults to 'geq'.]
        include_Mf: bool, optional
            Whether to include the BH mass as a free parameter in the model. 
            [Defaults to False.]
        include_chif: bool, optional
            Whether to include the BH spin as a free parameter in the model. 
            [Defaults to False.]
        include_lambda: bool, optional
            Whether to include the power-law index as a free parameter in the
            model. If False, then the lambda values will be fixed to l+3 in 
            each mode. [Defaults to False.]
        tau_range: tuple, optional
            The prior range of the power-law tail time constants tau. These are
            defined as offsets BEFORE t0; so a value of -10 means 10 M before 
            the start of the ringdown at time t0. [Defaults to (-100.0, -1.0).]
        lambda_range: tuple, optional
            The prior range of the power-law tail indices lambda. Only used if
            include_lambda is True. [Defaults to (-10.0, -1.0).]
        epsilon: float, optional
            A small tolerance value used for the t0_method='geq' method of 
            determining the analysis times. [Defaults to 1.0e-9.]
        tol: float, optional
            A small tolerance value used for inverting the covariance matrices.
            [Defaults to 1.0e-10.]
        num_samples: int, optional
            The number of samples to take from the posterior distribution.
            This is used to set the length of the MCMC chains.
            [Defaults to 1000.]
        MCMCnwalkers: int, optional
            The number of walkers to use in the MCMC. [Defaults to 20.]
        MCMCnburnin: int, optional
            The number of burn-in steps to take in the MCMC. [Defaults to 10.]
        MCMCthin: int, optional
            The thinning factor to use for the MCMC chain. [Defaults to 5.]
        """
        self.sim = sim
        self.modes = modes
        self.spherical_modes = spherical_modes
        self.t0 = t0
        self.T = T
        self.inv_sigma_array = inv_noise_covariances
        self.t0_method = t0_method
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.include_lambda = include_lambda
        self.tau_range = tau_range
        self.lambda_range = lambda_range
        self.tol = tol

        # this is used to help index the self.plt_psi_params array later
        self.skip = (2 if include_lambda else 1) 

        assert len(modes)==len(set(modes)), "Error: duplicate modes."

        self.times, self.data_mask = get_analysis_times(sim, t0, T, 
                                                        t0_method=t0_method, 
                                                        epsilon=epsilon)

        self.get_data_array()
        self.get_model_content()
        self.get_model_parameters()
        self.get_reference_model_and_params()
        self.qnm_frequencies = self.get_qnm_frequncies()
        self.qnm_frequency_derivs = self.get_qnm_frequncy_derivatives()
        self.qnm_mus = self.get_qnm_mode_mixing_coefficients()
        self.qnm_mu_derivs = self.get_qnm_mode_mixing_coefficient_derivatives()
        self.plt_mus = self.get_plt_mode_mixing_coefficients()
        self.get_qnm_model_matrix()
        self.get_qnm_covariance_matrix()

        self.sample_posterior(num_samples=num_samples, nwalkers=MCMCnwalkers, 
                              burnin=MCMCnburnin, thin=MCMCthin)

        #self.reweight_samples()

    def get_data_array(self):
        """
        Arrange all the analysis data into a single array.

        self.times : ndarray shape = (num_times,)
        self.data : complex ndarray shape = (num_spherical_modes, num_times)
        """
        self.data = np.zeros((len(self.spherical_modes), 
                              len(self.times)), dtype=complex)
        
        for i, mode in enumerate(self.spherical_modes):
            self.data[i] = self.sim.h[mode][self.data_mask]

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
        model and create ndarrays to store these parameters.

        self.qnm_param_names : list of strings, length = num_qnm_params
        self.qnm_params : ndarray, shape = (num_qnm_params,)
        self.plt_phi_param_names : list of strings, length = num_plt_phi_params
        self.plt_phi_params : ndarray, shape = (num_plt_phi_params,)
        self.plt_psi_param_names : list of strings, length = num_plt_psi_params
        self.plt_psi_params : ndarray, shape = (num_plt_psi_params,)
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

        self.qnm_params = np.zeros(len(self.qnm_param_names))

        self.plt_phi_param_names = []
        for mode in self.power_law_tails:
            self.plt_phi_param_names.append(f"Re A_{mode}")
            self.plt_phi_param_names.append(f"Im A_{mode}")

        self.plt_psi_param_names = []
        for mode in self.power_law_tails:
            self.plt_psi_param_names.append(f"tau_{mode}")
            if self.include_lambda:
                self.plt_psi_param_names.append(f"lambda_{mode}")

        self.plt_phi_params = np.zeros(len(self.plt_phi_param_names))
        self.plt_psi_params = np.zeros(len(self.plt_psi_param_names))
        for i, name in enumerate(self.plt_psi_param_names):
            if name.startswith("tau_"):
                self.plt_psi_params[i] = np.mean(self.tau_range)
            elif name.startswith("lambda_"):
                self.plt_psi_params[i] = np.mean(self.lambda_range)   

    def get_reference_model_and_params(self):
        """
        A least-squares using just the linear QNMs (with the mass and spin 
        fixed to the values in self.sim) is used to define the reference model.

        self.reference_model : complex ndarray, shape = (num_spherical_modes, 
                                                         num_times)
        self.qnm_reference_params : ndarray, shape = (num_qnm_params,)
        """
        self.ls_linear_qnm_fit = qnmfits.multimode_ringdown_fit(
                        self.sim.times, self.sim.h, modes=self.linear_qnms,
                        Mf=self.sim.Mf, chif=self.sim.chif_mag, t0=self.t0, 
                        T=self.T, t0_method='closest', # THIS SHOULD BE FIXED!
                        spherical_modes=self.spherical_modes)

        self.reference_model = self.zero_model()

        for i, mode in enumerate(self.spherical_modes):
            self.reference_model[i] = self.ls_linear_qnm_fit['model'][mode]

        self.reference_residuals = self.data - self.reference_model

        self.qnm_reference_params = np.zeros(len(self.qnm_param_names))

        for i, qnm in enumerate(self.linear_qnms):
            C = self.ls_linear_qnm_fit['C'][i]
            self.qnm_reference_params[2*i] = C.real
            self.qnm_reference_params[2*i+1] = C.imag

        if self.include_Mf:
            idx = self.qnm_param_names.index("Mf")
            self.qnm_reference_params[idx] = self.sim.Mf

        if self.include_chif:
            idx = self.qnm_param_names.index("chif")
            self.qnm_reference_params[idx] = self.sim.chif_mag

        self.qnm_params = self.qnm_reference_params.copy()

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
        qnm_frequencies : complex ndarray, shape = (num_qnms,)
            The complex frequencies of all the QNMs in the model. 
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

        return qnm_frequencies

    def get_qnm_frequncy_derivatives(self, delta=1.0e-3):
        """
        Calculate the derivatives of the frequencies of all the QNMs with 
        respect to the BH spin. Using second-order central finite difference.

        INPUTS
        ------
        delta: float, optional
            The step size used for the finite difference. 
            [Defaults to 1.0e-3]

        RETURNS
        -------
        qnm_frequency_derivs : complex ndarray, shape = (num_qnms,)
            The derivatives of the QNM frequencies with respect to the BH spin.
        """
        assert self.sim.chif_mag>2*delta and self.sim.chif_mag<1.-2*delta, \
                                        "chif must be safely between 0 and 1."
        
        omega_minus = self.get_qnm_frequncies(chif=self.sim.chif_mag-delta)
        omega_plus = self.get_qnm_frequncies(chif=self.sim.chif_mag+delta)

        omega_derivs = (omega_plus-omega_minus)/(2*delta)

        return omega_derivs
    
    def get_qnm_mode_mixing_coefficients(self, chif=None):
        """
        Calculate the mode mixing coefficients (i.e. the mu's) for all the QNMs
        in the model. 
        
        For the linear QNMs, the Kerr mixing coefficients are used, calculated 
        using the QNM package. 
        
        For the quadratic, cubic and constant offset QNMs, no mode mixing is 
        used, meaning the coefficients are set to 1 if the l and m indices match
        and 0 otherwise.

        INPUTS
        ------
        chif: float, optional
            The BH spin. If None, the value in self.sim is used.
            [Defaults to None.]

        RETURNS
        -------
        mus : complex ndarray, shape = (num_spherical_modes, num_qnms)
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
            
        return qnm_mus

    def get_qnm_mode_mixing_coefficient_derivatives(self, delta=1.0e-3):
        """
        Calculate the derivatives of the mus of all the QNMs with 
        respect to the BH spin. Using second-order central finite difference.

        INPUTS
        ------
        delta: float, optional
            The step size used for the finite difference. 
            [Defaults to 1.0e-3]

        RETURNS
        -------
        mu_derivs : complex ndarray, shape = (num_spherical_modes, num_qnms)
            The derivatives of the QNM frequencies with respect to the BH spin.
        """
        assert self.sim.chif_mag>2*delta and self.sim.chif_mag<1.-2*delta, \
                                        "chif must be safely between 0 and 1."
        
        mu_minus = self.get_qnm_mode_mixing_coefficients(chif=self.sim.chif_mag
                                                        - delta)
        mu_plus = self.get_qnm_mode_mixing_coefficients(chif=self.sim.chif_mag
                                                        + delta)

        mu_derivs = (mu_plus-mu_minus) / (2*delta)

        return mu_derivs
    
    def get_plt_mode_mixing_coefficients(self):
        """
        Calculate the mode mixing coefficients (i.e. the mu's) for all the PLTs
        in the model. No mode mixing is used hear, meaning the coefficients are 
        set to 1 if the l and m indices match and 0 otherwise.

        RETURNS
        -------
        mus : complex ndarray, shape = (num_spherical_modes, num_plts)
            The complex mode mixing coefficients of all the QNMs in the model. 
        """
        plt_mus = np.array([ [ 1.0+0.0*(1j) if 
                                ( plt_[0]==mode[0] and \
                                  plt_[1]==mode[1] )
                                else 0.0*(1j) 
                            for plt_ in self.power_law_tails]
                            for mode in self.spherical_modes ])  
        
        return plt_mus
    
    def get_qnm_model_matrix(self):
        """
        Create the model matrix for the QNM part of the model. This matrix is a 
        constant (doesn't depend on theta) and therefore can be pre-computed 
        once and is stored as an attrribute of the class.

        self.qnm_model_matrix : complex ndarray, shape = (num_qnm_params, 
                                                          num_spherical_modes,
                                                          num_times)
        """
        self.qnm_model_matrix = np.zeros((len(self.qnm_param_names), 
                                          len(self.spherical_modes), 
                                          len(self.times)), dtype=complex)
        
        all_modes = self.linear_qnms + self.quadratic_qnms + \
                    self.cubic_qnms + self.constant_offsets

        for i, qnm in enumerate(all_modes):
            omega = self.qnm_frequencies[i]
            mu = self.qnm_mus[:, i]
            exp_term = np.exp(-1j * omega * (self.times[np.newaxis,:]-self.t0))
            self.qnm_model_matrix[2*i] = mu[:,np.newaxis] * exp_term
            self.qnm_model_matrix[2*i+1] = (1j) * mu[:,np.newaxis] * exp_term

        if self.include_Mf:
            idx = self.qnm_param_names.index('Mf')
            for i, qnm in enumerate(all_modes):
                omega = self.qnm_frequencies[i]
                mu = self.qnm_mus[:, i]
                Cref = self.qnm_reference_params[2*i] + \
                            1j*self.qnm_reference_params[2*i+1]
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-self.t0))
                self.qnm_model_matrix[idx] += ((1j) * 
                                            mu[:,np.newaxis] * 
                                            omega * 
                                            (self.times[np.newaxis,:]-self.t0) * 
                                            Cref * 
                                            exp_term / 
                                            self.qnm_reference_params[idx] )
                
        if self.include_chif:
            idx = self.qnm_param_names.index('chif')
            for i, qnm in enumerate(all_modes):
                omega = self.qnm_frequencies[i]
                mu = self.qnm_mus[:, i]
                Cref = self.qnm_reference_params[2*i] + \
                            1j*self.qnm_reference_params[2*i+1]
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-self.t0))
                diff_term = ( self.qnm_mu_derivs[:,i,np.newaxis] - 
                        (1j)*mu[:,np.newaxis] * self.qnm_frequency_derivs[i] * \
                              (self.times[np.newaxis,:]-self.t0))
                self.qnm_model_matrix[idx] += ( Cref * 
                                                exp_term * 
                                                diff_term )

    def plt_model_matrix(self, plt_psi_params):
        """
        Create the model matrix for the nonlinear PLT part of the model. Unlike
        the qnm_model_matrix, this matrix is a function on the PLT parameters 
        psi; therefore this function returns the model matrix rather than 
        storing it as an attribute.

        INPUTS
        ------
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. 

        RETURNS
        -------
        plt_model_matrix: complex ndarray, shape = (num_plt_phi_params, 
                                                    num_spherical_modes,
                                                    num_times)
            The model matrix for the PLT part of the model.
        """
        plt_model_matrix = np.zeros((len(self.plt_phi_params), 
                                    len(self.spherical_modes), 
                                    len(self.times)), dtype=complex)
        
        for i, plt_ in enumerate(self.power_law_tails):

            tau = plt_psi_params[self.skip*i]

            if self.include_lambda:
                lambda_ = plt_psi_params[self.skip*i+1]
            else:
                l, m = plt_
                lambda_ = self.default_power_law_index(l)

            tail = self.plt_mus[:,i,np.newaxis] * \
                    ((self.times[np.newaxis,:]-tau)/(self.t0-tau))**(-lambda_)

            plt_model_matrix[2*i] = tail
            plt_model_matrix[2*i+1] = (1j)*tail
            
        return plt_model_matrix

    def get_qnm_covariance_matrix(self):
        """
        Calculate the covariance matrix of the QNM part of the model and its 
        inverse. This matrix is a constant (doesn't depend on theta) and 
        therefore can be pre-computed once and is stored as an attrribute of the
        class.
        
        self.qnm_covariance_matrix : complex ndarray, shape=(num_qnms, num_qnms)
        """
        self.qnm_inv_covariance_matrix = np.einsum('iax,axy,jay->ij', 
                                            self.qnm_model_matrix,
                                            self.inv_sigma_array, 
                                            self.qnm_model_matrix.conj()).real
        
        self.qnm_covariance_matrix = get_cov_inverse(
                                            self.qnm_inv_covariance_matrix,
                                            tol=self.tol)
        
    def get_plt_vectors_and_matrices(self, plt_psi_params):
        """
        This computes some matrices and vectors for the PLT part of the model.
        These are functions of the PLT parameters psi. These are needed for 
        sampling from the PLT parts of the posterior distribution.

        INPUTS
        ------
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. 

        RETURNS
        -------
        plt_V: ndarray, shape=(num_plt_psi_params,)
            The vector V_a as defined in my handwritten notes.
        plt_M: ndarray, shape=(num_plt_psi_params, num_plt_psi_params)
            The vector M_ab as defined in my handwritten notes.
        """
        PLT_MODEL_MATRIX = self.plt_model_matrix(plt_psi_params)

        plt_V = np.einsum('st,stu,asu->a',
                          self.reference_residuals,
                          self.inv_sigma_array,
                          PLT_MODEL_MATRIX.conj()).real
        
        pltmod_qnmmod = np.einsum('ast,stu,msu->am',
                                    PLT_MODEL_MATRIX,
                                    self.inv_sigma_array,
                                    self.qnm_model_matrix.conj()).real
        
        res_qnmmod = np.einsum('st,stu,msu->m',
                      self.reference_residuals,
                      self.inv_sigma_array,
                      self.qnm_model_matrix.conj()).real
        
        plt_V -= np.einsum('am,n,mn->a',
                           pltmod_qnmmod, 
                           res_qnmmod, 
                           self.qnm_covariance_matrix)

        plt_M = np.einsum('ast,stu,bsu->ab',
                          PLT_MODEL_MATRIX,
                          self.inv_sigma_array,
                          PLT_MODEL_MATRIX.conj()).real
        
        plt_M -= np.einsum('am,bn,mn->ab', 
                           pltmod_qnmmod, 
                           pltmod_qnmmod, 
                           self.qnm_covariance_matrix)
        
        return plt_V, plt_M

    def zero_model(self, dtype=complex):
        """
        Return an array of zeros for the model waveform.

        RETURNS
        -------
        model: ndarray, shape=(num_spherical_modes, num_times)
            2D array of zeros.
        """
        return np.zeros((len(self.spherical_modes), len(self.times)), 
                        dtype=dtype)
    
    def qnm_model(self, qnm_params=None):
        """
        Return the QNM part of the waveform model.

        INPUTS
        ------
        qnm_params: ndarray, shape=(num_qnm_params,)
            The QNM parameters. If None, the self.qnm_params is used.
            [Defaults to None.]

        RETURNS
        -------
        hQNM: ndarray, shape=(num_spherical_modes, num_times)
            The QNM part of the waveform model.
        """
        if qnm_params is None:
            qnm_params = self.qnm_params

        return self.reference_model + np.einsum('i,iab->ab', 
                                        qnm_params-self.qnm_reference_params, 
                                        self.qnm_model_matrix)
    
    def plt_model(self, plt_phi_params=None, plt_psi_params=None):
        """
        Return the PLT part of the waveform model.

        INPUTS
        ------
        plt_phi_params: ndarray, shape=(num_plt_phi_params,)
            The PLT phi parameters. If None, the self.plt_phi_params is used.
            [Defaults to None.]
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. If None, the self.plt_psi_params is used.
            [Defaults to None.]

        RETURNS
        -------
        hPLT: ndarray, shape=(num_spherical_modes, num_times)
            The PLT part of the waveform model.
        """
        if plt_phi_params is None:
            plt_phi_params = self.plt_phi_params

        if plt_psi_params is None:
            plt_psi_params = self.plt_psi_params

        hPLT = self.zero_model()

        for i, mode in enumerate(self.power_law_tails):

            A = plt_phi_params[2*i] + (1j)*plt_phi_params[2*i+1]
            tau = plt_psi_params[self.skip*i]

            if self.include_lambda:
                lambda_ = plt_psi_params[self.skip*i+1]
            else:
                l, m = mode
                lambda_ = self.default_power_law_index(l)

            hPLT += A * self.plt_mus[:,i,np.newaxis] * \
                    ((self.times[np.newaxis,:]-tau)/(self.t0-tau))**(-lambda_)

        return hPLT
    
    def full_model(self, 
                   qnm_params=None, 
                   plt_phi_params=None, 
                   plt_psi_params=None):
        """
        Return the full waveform model, h = hQNM + hPLT.

        INPUTS
        ------
        qnm_params: ndarray, shape=(num_qnm_params,)
            The QNM parameters. If None, the self.qnm_params is used.
            [Defaults to None.]
        plt_phi_params: ndarray, shape=(num_plt_phi_params,)
            The PLT phi parameters. If None, the self.plt_phi_params is used.
            [Defaults to None.]
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. If None, the self.plt_psi_params is used.
            [Defaults to None.]

        RETURNS
        -------
        h: ndarray, shape=(num_spherical_modes, num_times)
            The full waveform model.
        """
        if qnm_params is None:
            qnm_params = self.qnm_params

        if plt_phi_params is None:
            plt_phi_params = self.plt_phi_params

        if plt_psi_params is None:
            plt_psi_params = self.plt_psi_params
        
        hQNM = self.qnm_model(qnm_params=qnm_params)
        hPLT = self.plt_model(plt_phi_params=plt_phi_params,
                              plt_psi_params=plt_psi_params)
        
        h = hQNM + hPLT

        return h

    def sample_qnm_params_conditional(self, 
                                      num_samples=None, 
                                      plt_phi_params=None,
                                      plt_psi_params=None):
        """
        Draw samples from the posterior distribution on the QNM parameters,
        conditioned on fixed values of the PLT model parameters.

        INPUTS
        ------
        num_samples: int, optional
            The number of samples to draw from the conditional posterior. 
            [Defaults to None.]
        plt_phi_params: ndarray, shape=(num_plt_phi_params,)
            The PLT phi parameters. If None, the self.plt_phi_params is used.
            [Defaults to None.]
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. If None, the self.plt_psi_params is used.
            [Defaults to None.]

        RETURNS
        -------
        samples: ndarray
            The samples from the conditional posterior.
            shape=(num_qnm_params,) if num_samples is None, else
            shape=(num_samples, num_qnm_params).
        """
        if plt_phi_params is None:
            plt_phi_params = self.plt_phi_params

        if plt_psi_params is None:
            plt_psi_params = self.plt_psi_params

        residuals = self.reference_residuals - \
                    self.plt_model(plt_phi_params=plt_phi_params,
                                   plt_psi_params=plt_psi_params)
        
        shift = np.einsum('ist,stu,su->i',
                          self.qnm_model_matrix,
                          self.inv_sigma_array,
                          residuals.conj()).real
    
        mean = self.qnm_reference_params + np.einsum('ij,j->i', 
                                                     self.qnm_covariance_matrix,
                                                     shift)

        samples = mean + np.random.multivariate_normal(
                                                np.zeros(len(self.qnm_params)),
                                                self.qnm_covariance_matrix,
                                                size=num_samples)

        return samples
    
    def plt_psi_params_in_bounds(self, plt_psi_params):
        """
        Check if the PLT parameters are within the prior bounds.

        INPUTS
        ------
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. 
        
        RETURNS
        -------
        in_bounds: bool
            True if PLT psi parameters are within prior bounds, False otherwise.
        """
        in_bounds = True

        for i, name in enumerate(self.plt_psi_param_names):
            if name.startswith("tau_"):
                tau = plt_psi_params[i]
                if (not (tau>self.tau_range[0] and 
                                        tau<self.tau_range[1])):
                    return False
            elif name.startswith("lambda_"):
                lambda_ = plt_psi_params[i]
                if (not (lambda_>self.lambda_range[0] and 
                                        lambda_<self.lambda_range[1])):
                    return False

        return in_bounds
    
    def log_posterior_plt_params_marginal(self, 
                                          plt_phi_params=None,
                                          plt_psi_params=None):
        """
        Calculate the log-posterior for the PLT parameters (phi, psi), 
        marginalised over the QNM parameters.

        THIS FUNCTION IS NO LONGER USED, INCLUDED ONLY FOR TESTING PURPOSES.

        INPUTS
        ------
        plt_phi_params: ndarray, shape=(num_plt_phi_params,)
            The PLT phi parameters. If None, the self.plt_phi_params is used.
            [Defaults to None.]
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The PLT psi parameters. If None, the self.plt_psi_params is used.
            [Defaults to None.]

        RETURNS
        -------
        log_posterior: float
            The marginal log-posterior for the PLT parameters.
        """
        if plt_phi_params is None:
            plt_phi_params = self.plt_phi_params
        
        if plt_psi_params is None:
            plt_psi_params = self.plt_psi_params

        if not self.plt_psi_params_in_bounds(plt_psi_params=plt_psi_params):
            return -np.inf

        residuals = self.reference_residuals - \
                    self.plt_model(plt_phi_params=plt_phi_params,
                                   plt_psi_params=plt_psi_params)
        
        log_posterior = -0.5 * np.einsum('st,stu,su->',
                                        residuals,
                                        self.inv_sigma_array,
                                        residuals.conj()).real
        
        proj_residuals = np.einsum('st,stu,msu->m',
                                   residuals,
                                   self.inv_sigma_array,
                                   self.qnm_model_matrix.conj()).real
        
        log_posterior += 0.5 * np.einsum('mn,m,n->',
                                         self.qnm_covariance_matrix,
                                         proj_residuals,
                                         proj_residuals)

        return log_posterior
    
    def log_posterior_plt_nonlinearparams_marginal(self, plt_psi_params):
        """
        Calculate the log-posterior of the PLT parameters (psi), marginalised 
        over all of the linear QNM parameters (theta) and the linear amplitude
        PLT parameters (phi).

        INPUTS
        ------
        plt_psi_params: ndarray, shape=(num_plt_psi_params,)
            The nonlinear PLT psi parameters. 

        RETURNS
        -------
        log_posterior: float
            The marginal log-posterior for the nonlinear PLT parameters.
        """
        if not self.plt_psi_params_in_bounds(plt_psi_params):
            return -np.inf
        
        V, M = self.get_plt_vectors_and_matrices(plt_psi_params)

        Minv = get_cov_inverse(M)

        log_posterior = 0.5 * np.einsum('ab,a,b->', Minv, V, V)

        log_posterior -= 0.5 * np.linalg.slogdet(Minv)[1]

        return log_posterior
    
    def sample_posterior(self, num_samples=1000, 
                         nwalkers=20, burnin=10, thin=2):
        """
        Sample from the posterior distribution for the full ringdown model.

        self.samples : pd.DataFrame
            The samples from the posterior distribution. The columns are the
            parameter names

        INPUTS
        ------
        num_samples: int, optional
            The number of samples to draw from the posterior. 
            [Defaults to 1000.]
        nwalkers: int, optional
            The number of walkers to use in the MCMC ensemble.
            [Defaults to 20.]
        burnin: int, optional
            The number of steps to discard as part of the burn-in phase.
            [Defaults to 10.]
        thin: int, optional
            The thinning factor to use in the MCMC chain.
            [Defaults to 2.]
        """
        if len(self.power_law_tails)==0:
            
            # If there are no power law tails, then just sample the QNM 
            # parameters analytically from their Gaussian posterior.
            # --------
            s = self.sample_qnm_params_conditional(num_samples=num_samples)
            self.samples = pd.DataFrame(s, columns=self.qnm_param_names)

        else:
        
            # If there tails, then it's a 3 stage process:
            # 1. use zeus_mcmc to sample the PLT psi parameters from P(psi).
            # --------
            t0_ = time.time()
            psi_samples = self.sample_plt_psi_params_marginal(
                                                num_samples=num_samples,
                                                nwalkers=nwalkers, 
                                                burnin=burnin, 
                                                thin=thin)
            t1 = time.time()
            print(f"Stage 1: {t1-t0_:.2f} seconds")

            # 2. sample the plt_phi parameters from P(phi|psi).
            # --------
            t0_ = time.time()
            phi_samples = self.sample_plt_phi_params_conditional(psi_samples)
            t1 = time.time()
            print(f"Stage 2: {t1-t0_:.2f} seconds")

            # 3. sample the QNM theta parameters from P(theta|phi,psi).
            # --------
            t0_ = time.time()
            theta_samples = np.zeros((num_samples, len(self.qnm_param_names)))
            for i in range(num_samples):
                theta_samples[i] = self.sample_qnm_params_conditional(
                                                plt_phi_params=phi_samples[i],
                                                plt_psi_params=psi_samples[i])
            t1 = time.time()
            print(f"Stage 3: {t1-t0_:.2f} seconds")

            self.samples = np.hstack((theta_samples, phi_samples, psi_samples))

            self.samples = pd.DataFrame(self.samples, 
                                        columns=self.qnm_param_names +
                                                self.plt_phi_param_names +
                                                self.plt_psi_param_names)
    
    def sample_plt_psi_params_marginal(self, num_samples=1000, nwalkers=20, 
                                       burnin=10, thin=2):
        """
        Sample the PLT psi parameters from their marginal posterior distribution 
        using the zeus ensemble sampler.

        INPUTS
        ------
        num_samples: int, optional
            The number of samples to draw from the marginal posterior.
            [Defaults to 1000.]
        nwalkers: int, optional
            The number of walkers to use in the MCMC ensemble.
            [Defaults to 20.]
        burnin: int, optional
            The number of steps to discard as part of the burn-in phase.
            [Defaults to 10.]
        thin: int, optional
            The thinning factor to use in the MCMC chain.
            [Defaults to 2.]

        RETURNS
        -------
        psi_samples: ndarray, shape=(num_samples, num_plt_psi_params)
            The samples from the marginal posterior.
        """
        ndim = len(self.plt_psi_param_names)

        labels = self.plt_psi_param_names
        
        start = np.zeros((nwalkers, ndim))
        for i, name in enumerate(self.plt_psi_param_names):
            if name.startswith("tau_"):
                start[:,i] = np.random.uniform(self.tau_range[0], 
                                                self.tau_range[1],
                                                    nwalkers)
            elif name.startswith("lambda_"):
                start[:,i] = np.random.uniform(self.lambda_range[0], 
                                                self.lambda_range[1],
                                                nwalkers)
                
        nsteps = int( (num_samples * thin) / nwalkers + burnin ) + 1
        
        with Pool() as pool:
            sampler = zeus.EnsembleSampler(nwalkers, ndim, logP, 
                                           pool=pool, args=(self,))
            sampler.run_mcmc(start, nsteps, progress=True)
        
        sampler.summary
        
        full_chain = sampler.get_chain()
        
        fig, ax = plt.subplots(nrows=ndim, ncols=1)
        if ndim>1:
            for i in range(ndim):
                for w in range(nwalkers):
                    ax[i].plot(full_chain[:, w, i], alpha=0.5)
                ax[i].set_ylabel(labels[i])
                if i == ndim-1:
                    ax[i].set_xlabel('Iteration')
        else:
            i=0
            for w in range(nwalkers):
                ax.plot(full_chain[:, w, i], alpha=0.5)
            ax.set_ylabel(labels[i])
            ax.set_xlabel('Iteration')
        fig.suptitle('Traceplots')
        plt.tight_layout()
        plt.show()
        
        psi_samples = sampler.get_chain(flat=True, discard=burnin, thin=thin)
        
        corner.corner(psi_samples, labels=labels)
        plt.show()

        idx = np.random.choice(np.arange(len(psi_samples)), 
                               size=num_samples, replace=False)
        psi_samples = psi_samples[idx,:]

        return psi_samples
    
    def sample_plt_phi_params_conditional(self, psi_samples):
        """
        Given psi samples, sample from the distribution of phi|psi.

        INPUTS
        ------
        psi_samples: ndarray, shape=(num_samples, num_plt_psi_params)
            Samples from the marginal posterior on the PLT psi parameters.

        RETURNS
        -------
        phi_samples: ndarray, shape=(num_samples, num_plt_phi_params)
            Samples from the conditional posterior on the PLT phi parameters.
        """
        nsamples, _ = psi_samples.shape

        phi_samples = np.zeros((nsamples, len(self.plt_phi_param_names)))

        for i in range(nsamples):
            plt_V, plt_M = self.get_plt_vectors_and_matrices(psi_samples[i])
            
            inv_M = get_cov_inverse(plt_M)
            
            phi_mean = np.einsum('ab,a->b', inv_M, plt_V)

            phi_samples[i] = np.random.multivariate_normal(phi_mean, inv_M)

        return phi_samples
    
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
            
    def log_significance(self, mode):
        """
        Estimate the significance of given mode in the fit with complex 
        amplitude C=x+iy. The significance is defined as 

        .. math::
            S = int_{x,y|P(x,y)>P(0,0)} dx dy P(x, y)
        
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

    n_max = 6
    mode_list = [(2, 2, n, 1) for n in range(n_max+1)]

    spherical_modes = [(2, 2)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances,
                            num_samples=100000)
    
    for mode in mode_list:
        print(f"Significance S_{mode}={np.exp(fit.log_significance(mode))}.")
    
    x, y = 'Re C_(2, 2, 0, 1)', 'Im C_(2, 2, 0, 1)'
    least_squares_value = np.array([fit.ls_linear_qnm_fit['C'][0].real,
                                    fit.ls_linear_qnm_fit['C'][0].imag])
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                labels=[x, y], truths=least_squares_value)
    plt.show()

    print("... PASSED TEST 1\n")


def test2(id="0001", t0=10.0, T=90.0):
    """
    UNIT TESTING
    A complicated multimode QNM fit involving linear QNMs, nonlinear QNMs, 
    constant offsets and with the mass and spin included as free parameters.
    """
    print("RUNNING TEST 2...")
    n_max = 6
    mode_list = [(2, 2, n, 1) for n in range(n_max+1)] + [(3, 2, 0, 1)]
    mode_list += [(4, 4, n, 1) for n in range(n_max+1)] 
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
                            include_chif=True,
                            num_samples=100000)
    
    for mode in mode_list:
        print(f"Significance S_{mode}={np.exp(fit.log_significance(mode))}.")

    x, y = 'Mf', 'chif'
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, labels=[x, y],
                  truths=[fit.sim.Mf, fit.sim.chif_mag])
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
                            tau_range=(tau_inj-1, tau_inj+1),
                            lambda_range=(lambda_inj-1, lambda_inj+1),
                            include_lambda=True,
                            MCMCnwalkers=10,
                            MCMCnburnin=2, 
                            MCMCthin=1,
                            num_samples=500)
    
    w, x, y, z = 'Re A_(4, 4)', 'Im A_(4, 4)', 'tau_(4, 4)', 'lambda_(4, 4)'
    corner.corner(np.vstack((fit.samples[w], fit.samples[x], 
                             fit.samples[y], fit.samples[z])).T, 
                  labels=[w, x, y, z], 
                  truths=[A_inj.real, A_inj.imag, tau_inj, lambda_inj])
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
                            MCMCnwalkers=10,
                            MCMCnburnin=5, 
                            MCMCthin=2,
                            num_samples=500)
    
    w, x, y, z = 'Re A_(2, 2)', 'Im A_(2, 2)', 'tau_(2, 2)', 'lambda_(2, 2)'
    corner.corner(np.vstack((fit.samples[w], fit.samples[x], 
                             fit.samples[y], fit.samples[z])).T, 
                  labels=[w, x, y, z])
    plt.show()

    print("... PASSED TEST 4\n")

def test5(id="0001", t0=20.0, T=100.0):
    """
    UNIT TESTING
    Benchmark the key function.
    """
    n_max = 7
    mode_list = []
    mode_list += [(2, 2, n, 1) for n in range(n_max+1)]
    mode_list += [(4, 4, n, 1) for n in range(n_max+1)]
    mode_list += [(3, 2, 0, 1), (2, 2), (4, 4)]
    mode_list += [(4, 4, 'T')]

    spherical_modes = [(2, 2), (4, 4)]

    sim, inv_noise_covariances = test_load(id=id, t0=t0, T=T, 
                                           spherical_modes=spherical_modes)
    
    A_inj, tau_inj, lambda_inj = 0.1, -10.0, 7.0

    sim.h[(4,4)] = inject_tail(sim.times, A_inj, t0, tau_inj, lambda_inj)

    fit = QNM_PLT_BAYES_FIT(sim, 
                            mode_list, 
                            spherical_modes, 
                            t0, T, 
                            inv_noise_covariances, 
                            tau_range=(tau_inj-3, tau_inj+3),
                            lambda_range=(lambda_inj-3, lambda_inj+3),
                            include_lambda=True,
                            MCMCnwalkers=10,
                            MCMCnburnin=5, 
                            MCMCthin=1,
                            num_samples=1000)
    
    x, y= 'Re A_(4, 4)', 'Im A_(4, 4)'
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                  labels=[x, y])
    plt.show()

    x, y= 'Re C_(2, 2, 2, 1)', 'Im C_(2, 2, 2, 1)'
    corner.corner(np.vstack((fit.samples[x], fit.samples[y])).T, 
                  labels=[x, y])
    plt.show()
    


if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    #test4()
    test5()

    print("#######################")
    print("## PASSED ALL TESTS! ##")
    print("#######################")
