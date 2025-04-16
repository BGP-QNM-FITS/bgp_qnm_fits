import numpy as np
import scipy
import qnmfits
from CCE import SXS_CCE


class QNM_PLT_BAYES_FIT:
    """
    This class is used to perform bayesian fits of combinations of linear QNMs, 
    quadratic QNMS, cubic QNMS, power-law tails and constant offsets to multiple
    modes of waveforms from numerical relativity simulations.

    The data is the (complex) numerical relativity waveforms in multiple 
    spherical harmonic modes, h^\beta(t), where the index \beta=(l,m).

    The model is split into two parts:
    .. math::
        h^\beta(t) = h^\beta_{QNM}(t) + h^\beta_{PLT}(t) .
    Here QNM part contains all the QNMs (linear, quadratic, cubic etc) 
    and any constant offset terms which can be regarded as zero-frequency QNMs.
    part and h^\beta_{PLT}(t) is the power-law tail part. The PLT part contains
    the power-law tails.

    .. math::
        h^\beta_{QNM}(t)

    .. math::
        h^\beta_{PLT}(t)

    The fits are performed using a Gaussian likelihood function, which treats
    the noise (a.k.a. numerical error) in each \beta harmonic as an independent
    zero-mean Gaussian with noise covariance matrix Sigma^\beta_{ij}. Flat 
    priors are used for the real/imag parts of all amplitude parameters, the
    BH mass and spin, the peak times (tau) and the power law indices (lambda).
    """

    def __init__(self, sim, modes, spherical_modes, t0, T, noise_covariances,
                 t0_method="geq", include_Mf=False, include_chif=False,
                 epsilon=1.0e-9):
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
        noise_covariances: dict
            A dictionary of noise covariance matrices for each spherical mode.
            The keys are the spherical harmonic modes (l,m) and the values 
            are the corresponding noise covariance matrices which should be
            real num_times by num_times positive definite matrices.
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
        include_Mf: bool, optional
            Whether to include the BH mass as a free parameter in the model. 
            [Defaults to False.]
        include_chif: bool, optional
            Whether to include the BH spin as a free parameter in the model. 
            [Defaults to False.]
        epsilon: float, optional
            A small tolerance value used for the t0_method='geq' method of 
            determining the analysis times. [Defaults to 1.0e-9.]
        """
        self.sim = sim
        self.modes = modes
        self.spherical_modes = spherical_modes
        self.t0 = t0
        self.T = T
        self.noise_covariances = noise_covariances
        self.t0_method = t0_method
        self.include_Mf = include_Mf
        self.include_chif = include_chif

        self.get_analysis_times(epsilon=epsilon)
        self.get_inv_noise_cov_array()
        self.get_data_array()
        self.get_model_content()
        self.get_model_parameters()
        self.get_reference_model_and_params()
        self.qnm_frequencies = self.get_qnm_frequncies()
        self.qnm_frequency_derivs = self.get_qnm_frequncy_derivatives()
        self.qnm_mus = self.get_qnm_mode_mixing_coefficients()
        self.qnm_mu_derivs = self.get_qnm_mode_mixing_coefficient_derivatives()
        self.get_qnm_model_matrix()
        self.get_qnm_covariance_matrix()

    def get_analysis_times(self, epsilon=1.0e-9):
        """
        Define the analysis times used for the fit.

        self.data_mask : boolean ndarray shape=(num_times, )
        self.times.shape : ndarray shape=(num_times, )

        INPUTS
        ------
        epsilon: float, optional
            A small tolerance value used for the 'geq' method. 
            [Defaults to 1e-9.]
        """
        if self.t0_method == "geq":
            self.data_mask = ( (self.sim.times >= self.t0 - epsilon) & \
                            (self.sim.times < self.t0 + self.T - epsilon) )
            self.times = self.sim.times[self.data_mask]

        elif self.t0_method == "closest":
            start_index = np.argmin((self.sim.times - self.t0) ** 2)
            end_index = np.argmin((self.sim.times - self.t0 - self.T) ** 2)
            self.data_mask = ( (start_index<=np.arange(len(self.sim.times))) & \
                                (np.arange(len(self.sim.times))<end_index) )
            self.times = self.sim.times[self.data_mask]

        else:
            ValueError("Invalid t0_method, choose between geq and closest.")

    def get_inv_noise_cov_array(self):
        """
        Compute the inverse of all the noise covariance matrices for the 
        spherical harmonic modes and arrange them into a single array.

        inv_sigma_array.shape : ndarray shape = (num_spherical_modes, 
                                                 num_times, 
                                                 num_times)
        """
        self.inv_sigma_array = np.zeros((len(self.spherical_modes), 
                                        len(self.times), 
                                        len(self.times)))
        
        for i, lm in enumerate(self.spherical_modes):
            inv_cov = self.get_cov_inverse(self.noise_covariances[lm])
            self.inv_sigma_array[i] = inv_cov

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
            
    def get_model_parameters(self):
        """
        Create a list of the names of all the parameters in both parts of the 
        model and create ndarrays to store these parameters.

        self.qnm_param_names : list of strings, length = num_qnm_params
        self.qnm_params : ndarray, shape = (num_qnm_params,)
        self.plt_param_names : list of strings, length = num_plt_params
        self.plt_params : ndarray, shape = (num_plt_params,)
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

        self.plt_param_names = []
        for mode in self.power_law_tails:
            self.plt_param_names.append(f"Re A_{mode}")
            self.plt_param_names.append(f"Im A_{mode}")

        self.plt_params = np.zeros(len(self.plt_param_names))

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
                        T=self.T, spherical_modes=self.spherical_modes)

        self.reference_model = self.zero_model()

        for i, mode in enumerate(self.spherical_modes):
            self.reference_model[i] = self.ls_linear_qnm_fit['model'][mode]

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
        omega_central = self.get_qnm_frequncies(chif=self.sim.chif_mag)
        omega_plus = self.get_qnm_frequncies(chif=self.sim.chif_mag+delta)

        omega_derivs = (omega_plus-2.0*omega_central+omega_minus)/(delta**2)

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
                                            self.sim.chif_mag)
                                        for mode in self.spherical_modes ])
            
        if num_quadratic_qnms > 0:
            a = num_linear_qnms
            b = num_linear_qnms + num_quadratic_qnms
            qnm_mus[:,a:b] = np.array([ [ 1.0j if 
                                          ( qnm[0]+qnm[4]==mode[0] and \
                                            qnm[1]+qnm[5]==mode[1] )
                                          else 0.0*(1j) 
                                        for qnm in self.quadratic_qnms]
                                    for mode in self.spherical_modes ])  
            
        if num_cubic_qnms > 0:
            a = num_linear_qnms + num_quadratic_qnms
            b = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_mus[:,a:b] = np.array([ [ 1.0j if 
                                          ( qnm[0]+qnm[4]+qnm[8]==mode[0] and \
                                            qnm[1]+qnm[5]+qnm[9]==mode[1] )
                                          else 0.0*(1j) 
                                        for qnm in self.cubic_qnms]
                                    for mode in self.spherical_modes ])  
            
        if num_constant_offsets > 0:
            a = num_linear_qnms + num_quadratic_qnms + num_cubic_qnms
            qnm_mus[:,a:] = np.array([ [ 1.0j if 
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
        mu = self.get_qnm_mode_mixing_coefficients(chif=self.sim.chif_mag)
        mu_plus = self.get_qnm_mode_mixing_coefficients(chif=self.sim.chif_mag
                                                        + delta)

        mu_derivs = (mu_plus-2.0*mu+mu_minus) / (delta**2)

        return mu_derivs
    
    def get_qnm_model_matrix(self):
        """
        Create the model matrix for the QNM part of the model.

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
            exp_term = np.exp(-1j * omega * (self.times[np.newaxis,:]-t0))
            self.qnm_model_matrix[2*i] = mu[:,np.newaxis] * exp_term
            self.qnm_model_matrix[2*i+1] = (1j) * mu[:,np.newaxis] * exp_term

        if self.include_Mf:
            idx = self.qnm_param_names.index('Mf')
            for i, qnm in enumerate(all_modes):
                omega = self.qnm_frequencies[i]
                mu = self.qnm_mus[:, i]
                Cref = self.qnm_reference_params[2*i] + \
                            1j*self.qnm_reference_params[2*i+1]
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-t0))
                self.qnm_model_matrix[idx] += ( (1j) * 
                                                mu[:,np.newaxis] * 
                                                omega * 
                                                (self.times[np.newaxis,:]-t0) * 
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
                exp_term = np.exp(-1j*omega*(self.times[np.newaxis,:]-t0))
                diff_term = ( self.qnm_mu_derivs[:,i,np.newaxis] - 
                        (1j)*mu[:,np.newaxis] * self.qnm_frequency_derivs[i] * \
                              (self.times[np.newaxis,:]-t0))
                self.qnm_model_matrix[idx] += ( Cref * 
                                                exp_term * 
                                                diff_term)

            self.qnm_mu_derivs

    def get_qnm_covariance_matrix(self):
        """
        Calculate the covariance matrix of the QNM part of the model.
        Also initialise a zero-mean multivariate normal distribution with this
        covariance matrix.
        
        self.qnm_covariance_matrix : complex ndarray, shape=(num_qnms, num_qnms)
        """
        self.qnm_inv_covariance_matrix = np.einsum('iax,axy,jay->ij', 
                                            self.qnm_model_matrix,
                                            self.inv_sigma_array, 
                                            self.qnm_model_matrix.conj()).real
        
        self.qnm_covariance_matrix = self.get_cov_inverse(
                                            self.qnm_inv_covariance_matrix)
        
        self.qnm_dist = scipy.stats.multivariate_normal(cov=
                                                    self.qnm_covariance_matrix)

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
                                                qnm_params, 
                                                self.qnm_model_matrix)
    
    def plt_model(self, plt_params=None):
        """
        Return the PLT part of the waveform model.

        INPUTS
        ------
        plt_params: ndarray, shape=(num_plt_params,)
            The QNM parameters. If None, the self.qnm_params is used.
            [Defaults to None.]

        RETURNS
        -------
        hPLT: ndarray, shape=(num_spherical_modes, num_times)
            The PLT part of the waveform model.
        """
        return self.zero_model()

    def sample_qnm_params(self, num_samples=None):
        """
        Draw samples from the posterior distribution on the QNM parameters,
        conditioned on fixed values of the PLT model parameters.

        INPUTS
        ------
        num_samples: int, optional
            The number of samples to draw from the conditional posterior. 
            [Defaults to None.]

        RETURNS
        -------
        samples: ndarray
            The samples from the conditional posterior.
            shape=(num_qnm_params,) if num_samples is None, else
            shape=(num_samples, num_qnm_params).
        """
        residuals = self.data - \
                    self.reference_model - \
                    self.plt_model(plt_params=self.plt_params)
        
        shift = np.einsum('ist,stu,su->i',
                          self.qnm_model_matrix,
                          self.inv_sigma_array,
                          residuals).real

        mean = self.qnm_reference_params + np.einsum('ij,j->i', 
                                                     self.qnm_covariance_matrix,
                                                     shift)

        samples = mean + self.qnm_dist.rvs(size=num_samples)

        return samples
    
    def get_cov_inverse(self, cov, epsilon=1e-10):
        """
        Get the inverse of a covariance matrix using eigenvalue decomposition.

        INPUTS
        ------
        cov: ndarray, shape=(N,N)
            The covariance matrix to invert. Must be sym positive-definite.
        epsilon: float
            A small tolerance value.
        
        RETURNS
        -------
        inv_cov: ndarray, shape=(N,N)
            The inverse of the covariance matrix.
        """ 
        vals, vecs = scipy.linalg.eigh(cov)
        vals = np.maximum(vals, epsilon)
        return np.einsum('ik, k, jk -> ij', vecs, 1/vals, vecs)



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import corner
    import pickle
    from GP_funcs import get_cov_inverse, kernel_main

    with open('../tuned_params.pkl', 'rb') as f:
        params = pickle.load(f)

    tuned_param_dict_main = params[id]

    
    print(get_cov_inverse(np.arange(50, 0.1), kernel_main, tuned_param_dict_main, spherical_modes=[(2,2)]))
    exit()

    id = "0001"
    sim = SXS_CCE(id, lev="Lev5", radius="R2")

    n_max = 5
    mode_list = [(2, 2, n, 1) for n in range(n_max+1)]
    mode_list += [(2, 2)]

    spherical_modes = [(2, 2)]

    INCLUDE_MF = INCLUDE_CHIF = True

    t0, T = 17.0, 100.0

    noise_covariances = {mode: 1.0e-5*np.eye(int((T)/0.1)) 
                            for mode in spherical_modes}

    fit = QNM_PLT_BAYES_FIT(sim, mode_list, spherical_modes, t0, T, 
                            noise_covariances, include_Mf=INCLUDE_MF, 
                            include_chif=INCLUDE_CHIF)

    def plot_fit_mode(fit, mode=(2,2)):
        idx = fit.spherical_modes.index(mode)

        fig, axes = plt.subplots(nrows=2, figsize=(10, 5), sharex=True)
        axes[0].plot(fit.sim.times, fit.sim.h[mode].real, c='C0', ls='-')
        axes[0].plot(fit.sim.times, fit.sim.h[mode].imag, c='C0', ls=':')
        axes[0].plot(fit.times, fit.qnm_model()[idx].real, c='C1', ls='-')
        axes[0].plot(fit.times, fit.qnm_model()[idx].imag, c='C1', ls='-')

        axes[1].plot(fit.sim.times, np.abs(fit.sim.h[mode].real),c='C0',ls='-')
        axes[1].plot(fit.times, np.abs(fit.qnm_model()[idx].real),c='C1',ls='-')
        axes[1].set_yscale('log')

        axes[0].set_ylabel(r"$h_{22}$")
        axes[1].set_ylabel(r"$h_{22}$")
        axes[1].set_xlabel(r"$t$ [$M$]")
        plt.show()

    plot_fit_mode(fit, mode=(2,2))

    samples = fit.sample_qnm_params(num_samples=1000)

    def plot_amp_posteriors(fit, samples, mode=(2,2,0,1)):
        if len(mode) == 4 or len(mode) == 8 or len(mode) == 12:
            symbol = 'C'
        elif len(mode) == 2:
            symbol = 'B'
        elif len(mode) == 3:
            symbol = 'A'
        x = "Re "+symbol+f"_{mode}"
        y = "Im "+symbol+f"_{mode}"
        i = fit.qnm_param_names.index(x)
        j = fit.qnm_param_names.index(y)
        corner.corner(samples[:,[i,j]], labels=[x,y], truths=[0,0])
        plt.show()

    def plot_mass_spin_posteriors(fit, samples):
        x = "Mf"
        y = "chif"
        i = fit.qnm_param_names.index(x)
        j = fit.qnm_param_names.index(y)
        corner.corner(samples[:,[i,j]], labels=[x,y], truths=[fit.sim.Mf, 
                                                            fit.sim.chif_mag])
        plt.show()

    for n in range(n_max+1):
        plot_amp_posteriors(fit, samples, mode=(2,2,n,1))
    plot_amp_posteriors(fit, samples, mode=(2,2))
    plot_mass_spin_posteriors(fit, samples)
