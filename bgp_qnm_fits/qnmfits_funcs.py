"""

This are modified versions of functions from the qnmfits package. 

"""

import h5py
import quaternionic
import spherical
import numpy as np
import qnm as qnm_loader

from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from urllib.request import urlretrieve
from scipy.integrate import dblquad as dbl_integrate
from spherical import Wigner3j as w3j
from bgp_qnm_fits.utils import mismatch
import pickle
from scipy.interpolate import interp1d

class qnm:
    """
    Class for loading quasinormal mode (QNM) frequencies and spherical-
    spheroidal mixing coefficients. This makes use of the qnm package,
    https://arxiv.org/abs/1908.10377
    """

    def __init__(self):
        """
        Initialise the class.
        """

        # Dictionary to store the qnm functions
        self._qnm_funcs = {}

        # Dictionary to store interpolated qnm functions for quicker
        # evaluation
        self._interpolated_qnm_funcs = {}

        #Load the Cmus.json file
        data_dir = Path(__file__).parent
        cmus_file = data_dir / 'Cmus.pkl'
        with open(cmus_file, 'rb') as file:
            Cmus = pickle.load(file)

        indices_list = [
            (6, 6, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1),
            (7, 6, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1),
            (6, -6, 2, -2, 0, -1, 2, -2, 0, -1, 2, -2, 0, -1),
            (7, -6, 2, -2, 0, -1, 2, -2, 0, -1, 2, -2, 0, -1)
        ]

        self._Cmus_interp = {}

        for i, indices in enumerate(indices_list):
            Cmus_values = Cmus[indices]
            spins = np.array([float(key) for key in Cmus_values.keys()])
            cmu_values = np.array([complex(value) for value in Cmus_values.values()])
            cmu_real_interp = interp1d(spins, cmu_values.real, kind="cubic", fill_value="extrapolate")
            cmu_imag_interp = interp1d(spins, cmu_values.imag, kind="cubic", fill_value="extrapolate")
            self._Cmus_interp[indices] = lambda spin, real_interp=cmu_real_interp, imag_interp=cmu_imag_interp: (
            real_interp(spin) + 1j * imag_interp(spin)
            )

        # The method used by the qnm package breaks down for certain modes that
        # approach the imaginary axis (perhaps most notably, the (2,2,8) mode).
        # We load data for these modes separately, computed by Cook &
        # Zalutskiy.

        data_dir = Path(__file__).parent / 'Data'

        # Dictionary to store the mode data, using our preferred labelling
        # convention
        multiplet_data = {}

        # A list of known multiplets
        self.multiplet_list = [(2, 0, 8), (2, 1, 8), (2, 2, 8)]

        # Keep track of what data has been downloaded (useful for warnings)
        self.download_check = {}

        for ell, m, n in self.multiplet_list:

            file_path = data_dir / f'KerrQNM_{n:02}.h5'
            self.download_check[n] = file_path.exists()
            if self.download_check[n]:

                # Open file
                with h5py.File(file_path, 'r') as f:

                    # Read data for each multiplet, and store in the
                    # multiplet_data dictionary with the preferred labelling
                    # convention
                    for i in [0, 1]:
                        multiplet_data[(ell, m, n+i)] = np.array(
                            f[f'n{n:02}/m{m:+03}/{{{ell},{m},{{{n},{i}}}}}']
                            )

        for key, data in multiplet_data.items():

            # Extract relevant quantities
            spins = data[:, 0]
            real_omega = data[:, 1]
            imag_omega = data[:, 2]
            # real_A = data[:, 3]
            # imag_A = data[:, 4]
            all_real_mu = data[:, 5::2]
            all_imag_mu = data[:, 6::2]

            # Interpolate omegas
            real_omega_interp = UnivariateSpline(spins, real_omega, s=0)
            imag_omega_interp = UnivariateSpline(spins, imag_omega, s=0)

            # Interpolate angular separation constants
            # real_A_interp = UnivariateSpline(spins, real_A, s=0)
            # imag_A_interp = UnivariateSpline(spins, imag_A, s=0)

            # Interpolate mus
            mu_interp = []

            for real_mu, imag_mu in zip(all_real_mu.T, all_imag_mu.T):

                real_mu_interp = UnivariateSpline(spins, real_mu, s=0)
                imag_mu_interp = UnivariateSpline(spins, imag_mu, s=0)

                mu_interp.append((real_mu_interp, imag_mu_interp))

            # Add these interpolated functions to the frequency_funcs
            # dictionary
            self._interpolated_qnm_funcs[key] = [
                (real_omega_interp, imag_omega_interp), mu_interp
                ]

    def _interpolate(self, ell, m, n, s=-2):

        # If there is a known multiplet with the same l and m, we need to
        # be careful with the n index
        n_load = n
        for ellp, mp, nprime in self.multiplet_list:
            if (ell == ellp) & (m == mp):
                if n > nprime+1:
                    n_load -= 1

        qnm_func = qnm_loader.modes_cache(s, ell, m, n_load)

        # Extract relevant quantities
        spins = qnm_func.a
        real_omega = np.real(qnm_func.omega)
        imag_omega = np.imag(qnm_func.omega)
        all_real_mu = np.real(qnm_func.C)
        all_imag_mu = np.imag(qnm_func.C)

        # Interpolate omegas
        real_omega_interp = UnivariateSpline(spins, real_omega, s=0)
        imag_omega_interp = UnivariateSpline(spins, imag_omega, s=0)

        # Interpolate mus
        mu_interp = []

        for real_mu, imag_mu in zip(all_real_mu.T, all_imag_mu.T):

            real_mu_interp = UnivariateSpline(spins, real_mu, s=0)
            imag_mu_interp = UnivariateSpline(spins, imag_mu, s=0)

            mu_interp.append((real_mu_interp, imag_mu_interp))

        # Add these interpolated functions to the frequency_funcs dictionary
        self._interpolated_qnm_funcs[ell, m, n, s] = [
            (real_omega_interp, imag_omega_interp), mu_interp
            ]

    def omega(self, ell, m, n, sign, chif, Mf=1, s=-2):
        r"""
        Return a complex frequency, :math:`\omega_{\ell m n}(M_f, \chi_f)`,
        for a particular mass, spin, and mode. One or both of chif and Mf can
        be array_like, in which case an ndarray of complex frequencies is
        returned.

        Parameters
        ----------
        ell : int
            The angular number of the mode.

        m : int
            The azimuthal number of the mode.

        n : int
            The overtone number of the mode.

        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the frequency. This way any regular (+1) or mirror (-1)
            mode can be requested. Alternatively, this can be thought of as
            prograde (sign = sgn(m)) or retrograde (sign = -sgn(m)) modes.

        chif : float or array_like
            The dimensionless spin magnitude of the black hole.

        Mf : float or array_like, optional
            The mass of the final black hole. This is the factor which the QNM
            frequencies are divided through by, and so determines the units of
            the returned quantity.

            If Mf is in units of seconds, then the returned frequency has units
            :math:`\mathrm{s}^{-1}`.

            When working with SXS simulations and GW surrogates, we work in
            units scaled by the total mass of the binary system, M. In this
            case, providing the dimensionless Mf value (the final mass scaled
            by the total binary mass) will ensure the QNM frequencies are in
            the correct units (scaled by the total binary mass). This is
            because the frequencies loaded from file are scaled by the remnant
            black hole mass (Mf*omega). So, by dividing by the remnant black
            hole mass scaled by the total binary mass (Mf/M), we are left with
            Mf*omega/(Mf/M) = M*omega.

            The default is 1, in which case the frequencies are returned in
            units of the remnant black hole mass.

        s : int, optional
            The spin weight of the mode. The default is -2.

        Returns
        -------
        complex or ndarray
            The complex QNM frequency or frequencies of length
            max(len(chif), len(Mf)).
        """
        # Load the correct qnm based on the type we want
        m *= sign

        # Test if the interpolated qnm function has been created (we create
        # our own interpolant so that we can evaluate the frequencies for
        # all spins simultaneously)
        if (ell, m, n, s) not in self._interpolated_qnm_funcs:
            self._interpolate(ell, m, n, s)

        omega_interp = self._interpolated_qnm_funcs[ell, m, n, s][0]
        omega = omega_interp[0](chif) + 1j*omega_interp[1](chif)

        # Use symmetry properties to get the mirror mode, if requested
        if sign == -1:
            omega = -np.conjugate(omega)

        return omega/Mf

    def omega_list(self, modes, chif, Mf=1, s=-2):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin).

        Parameters
        ----------
        modes : array_like
            A sequence of (l,m,n,sign) tuples to specify which QNMs to load
            frequencies for. For nonlinear modes, the tuple has the form
            (l1,m1,n1,sign1,l2,m2,n2,sign2,...).

        chif : float or array_like
            The dimensionless spin magnitude of the final black hole.

        Mf : float or array_like, optional
            The mass of the final black hole. See the qnm.omega docstring for
            details on units. The default is 1.

        s : int, optional
            The spin weight of the mode. The default is -2.

        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the qnm function and append the result to the
        # list

        # Code for linear QNMs:
        # return [self.omega(l, m, n, sign, chif, Mf) for l, m, n, sign in
        # modes]

        if any(len(mode) == 2 for mode in modes):
            return_list = []
            for mode in modes:
                if len(mode) == 2:
                    return_list.append(0.0 + 0.0j)
                else:
                    sum_list = []
                    for i in range(0, len(mode), 4):
                        l, m, n, sign = mode[i:i+4]
                        sum_list.append(self.omega(l, m, n, sign, chif, Mf))
                    return_list.append(sum(sum_list))
            return return_list
        
        else:
            return [
                sum([
                    self.omega(ell, m, n, sign, chif, Mf, s)
                    for ell, m, n, sign in [
                        mode[i:i+4] for i in range(0, len(mode), 4)
                    ]
                ])
                for mode in modes
            ]

        # Writen out, the above is doing the following:

        # return_list = []
        # for mode in modes:
        #     sum_list = []
        #     for i in range(0, len(mode), 4):
        #         l, m, n, sign = mode[i:i+4]
        #         sum_list.append(self.omega(l, m, n, sign, chif, Mf))
        #     return_list.append(sum(sum_list))
        # return return_list

    def mu(self, ell, m, ellp, mp, nprime, sign, chif, s=-2):
        r"""
        Return a spherical-spheroidal mixing coefficient,
        :math:`\mu_{\ell m \ell' m' n'}(\chi_f)`, for a particular spin and
        mode combination. The indices (l,m) refer to the spherical harmonic.
        The indices (l',m',n') refer to the spheroidal harmonic. The spin chif
        can be a float or array_like.

        Parameters
        ----------
        ell : int
            The angular number of the spherical-harmonic mode.

        m : int
            The azimuthal number of the spherical-harmonic mode.

        ellp : int
            The angular number of the spheroidal-harmonic mode.

        mp : int
            The azimuthal number of the spheroidal-harmonic mode.

        nprime : int
            The overtone number of the spheroidal-harmonic mode.

        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the QNM frequency. If the mixing coefficient associated
            with a -1 QNM (i.e. a mirror mode) is requested, then symmetry
            properties are used for the calculation.

        chif : float or array_like
            The dimensionless spin magnitude of the final black hole.

        s : int, optional
            The spin weight of the mode. The default is -2.

        Returns
        -------
        complex or ndarray
            The spherical-spheroidal mixing coefficient.
        """
        # There is no overlap between different values of m
        if mp != m:
            return 0

        # Load the correct qnm based on the type we want
        m *= sign
        mp *= sign

        # Our functions return all mixing coefficients with the given
        # (l',m',n'), so we need to index it to get the requested l
        if abs(m) > abs(s):
            index = ell - abs(m)
        else:
            index = ell - abs(s)

        if (ellp, mp, nprime, s) not in self._interpolated_qnm_funcs:
            self._interpolate(ellp, mp, nprime, s)

        mu_interp = self._interpolated_qnm_funcs[ellp, mp, nprime, s][1][index]
        mu = mu_interp[0](chif) + 1j*mu_interp[1](chif)

        # Use symmetry properties to get the mirror mixing coefficient, if
        # requested
        if sign == -1:
            mu = (-1)**(ell+ellp)*np.conjugate(mu)

        return mu

    def mu_list(self, indices, chif, s=-2):
        """
        Return a list of mixing coefficients, for all requested indices. See
        the qnm.mu() docstring for more details.

        Parameters
        ----------
        indices : array_like
            A sequence of (ell,m,ell',m',n',sign) tuples specifying which
            mixing coefficients to return.

        chif : float
            The dimensionless spin magnitude of the final black hole.

        s : int, optional
            The spin weight of the mode. The default is -2.

        Returns
        -------
        mus : list
            The list of spherical-spheroidal mixing coefficients.
        """
        # List to store the mixing coeffs
        mus = []

        # For each mode, call the qnm function and append the result to the
        # list

        for k in indices:
            if len(k) == 4:
                l, m, lp, mp = k 
                if (l, m) == (lp, mp):
                    mus.append(1.0 + 0.0j)
                else:
                    mus.append(0.0 + 0.0j)
            elif len(k) == 6:
                ell, m, ellp, mp, nprime, sign = k
                mus.append(self.mu(ell, m, ellp, mp, nprime, sign, chif, s))
            elif len(k) == 10:
                mus.append(Qmu_D([k], chif, 8)[0]) 
            elif len(k) == 14:
                if k in self._Cmus_interp.keys():
                    mus.append(self._Cmus_interp[k](chif))
                else:
                    mus.append(0.0 + 0.0j)

        return mus

qnm = qnm()

def sYlm(l, m, theta, phi, s=-2, l_max=8):
    """
    A function to calculate the spin-weighted spherical harmonics.

    Parameters
    ----------
    l : int
        The l-mode of the spherical harmonic.
    m : int
        The m-mode of the spherical harmonic.
    theta : array_like
        The polar angles at which to evaluate the spherical harmonic.
    phi : array_like
        The azimuthal angles at which to evaluate the spherical harmonic.
    s : int, optional
        The spin weight of the spherical harmonic. The default is -2.
    l_max : int, optional
        The maximum l-mode to consider in the reconstruction. The default is 8.

    Returns
    -------
    Y : array_like
        The spin-weighted spherical harmonic evaluated at the given angles.

    """
    wigner = spherical.Wigner(l_max)
    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    Y = wigner.sYlm(s, R)
    return Y[wigner.Yindex(l, m)]


def kappa(i, j, d, h, b, f, s1, s2):
    """
    A function to determine the kappa coefficient in the quadratic mode mixing.

    Parameters
    ----------
    i : int
        The l-mode of the third spherical mode.
    j : int
        The m-mode of the third spherical mode.
    d : int
        The l-mode of the first spherical mode.
    h : int
        The l-mode of the second spherical mode.
    b : int
        The m-mode of the first spherical mode.
    f : int
        The m-mode of the second spherical mode.
    s1 : int
        The spin weight of the first spheroidal harmonic.
    s2 : int
        The spin weight of the second spheroidal harmonic.

    Returns
    -------
    kappa : float
        The kappa coefficient.

    """

    return (
        (((2 * d + 1) * (2 * h + 1) * (2 * i + 1)) / (4 * np.pi)) ** (1 / 2)
        * w3j(d, h, i, -s1, -s2, s1 + s2)
        * w3j(d, h, i, b, f, -j)
        * (-1) ** (j + s1 + s2)
    )


def triple_kappa_numerical(l4, m4, l1, l2, l3, m1, m2, m3, s1, s2, s3, l_max=8):
    """
    Numerically compute a generalized kappa coefficient by direct integration over the sphere.
    
    This computes the integral: ∫ Y_{s1,m1}^{l1} Y_{s2,m2}^{l2} Y_{s3,m3}^{l3} ̅Y_{s1+s2+s3,m4}^{l4} dΩ
    where Y are spin-weighted spherical harmonics and the bar denotes complex conjugate.
    
    Parameters
    ----------
    l4 : int
        The l-mode of the fourth (conjugated) spherical harmonic.
    m4 : int
        The m-mode of the fourth (conjugated) spherical harmonic.
    l1 : int
        The l-mode of the first spherical harmonic.
    l2 : int
        The l-mode of the second spherical harmonic.
    l3 : int
        The l-mode of the third spherical harmonic.
    m1 : int
        The m-mode of the first spherical harmonic.
    m2 : int
        The m-mode of the second spherical harmonic.
    m3 : int
        The m-mode of the third spherical harmonic.
    s1 : int
        The spin weight of the first spherical harmonic.
    s2 : int
        The spin weight of the second spherical harmonic.
    s3 : int
        The spin weight of the third spherical harmonic.
    l_max : int, optional
        The maximum l-mode for spherical harmonic computation. Default is 8.
        
    Returns
    -------
    kappa : complex
        The numerically computed kappa coefficient.
    """

    #if m1 + m2 + m3 != m4:
    #    return 0.0 + 0.0j
    
    def integrand_real(theta, phi):
        # Real part of the integrand
        return np.real(
            np.sin(theta) *  # Jacobian factor for spherical coordinates
            sYlm(l1, m1, theta, phi, s=s1, l_max=l_max) * 
            sYlm(l2, m2, theta, phi, s=s2, l_max=l_max) * 
            sYlm(l3, m3, theta, phi, s=s3, l_max=l_max) *
            np.conj(sYlm(l4, m4, theta, phi, s=s1+s2+s3, l_max=l_max))
        )
    
    def integrand_imag(theta, phi):
        # Imaginary part of the integrand
        return np.imag(
            np.sin(theta) *  # Jacobian factor for spherical coordinates
            sYlm(l1, m1, theta, phi, s=s1, l_max=l_max) * 
            sYlm(l2, m2, theta, phi, s=s2, l_max=l_max) * 
            sYlm(l3, m3, theta, phi, s=s3, l_max=l_max) *
            np.conj(sYlm(l4, m4, theta, phi, s=s1+s2+s3, l_max=l_max))
        )
    
    # Integrate real and imaginary parts separately
    real_part = dbl_integrate(integrand_real, 0, 2*np.pi, 0, np.pi)[0]
    imag_part = dbl_integrate(integrand_imag, 0, 2*np.pi, 0, np.pi)[0]
    
    return real_part + 1j * imag_part


def Qmu_D(indices, chif, l_max, **kwargs):
    """
    A function to calculate the D prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the D prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Here for consistency when passed into other functions.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 


    """

    return [
        sum(
            qnm.mu(d, b, a, b, c, sign1, chif, -2)
            * qnm.mu(h, f, e, f, g, sign2, chif, -2)
            * kappa(i, j, d, h, b, f, -2, -2)
            * np.sqrt((i + 4) * (i - 3) * (i + 3) * (i - 2))
            for d in range(2, l_max + 1)
            for h in range(2, l_max + 1)
        )
        for i, j, a, b, c, sign1, e, f, g, sign2 in indices
    ]


def Cmu_D(indices, chif, l_max, **kwargs):
    """
    A function to calculate the D prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the D prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Here for consistency when passed into other functions.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 

    """

    return [
        sum(
            qnm.mu(l1, m, l, m, n, p, chif, -2)
            * qnm.mu(l2, mp, lp, mp, nprime, pp, chif, -2)
            * qnm.mu(l3, mpp, lpp, mpp, npp, ppp, chif, -2)
            * triple_kappa_numerical(l4, m4, l1, l2, l3, m, mp, mpp, -2, -2, -2)
            * np.sqrt((l4 + 6) * (l4 - 5) * (l4 + 5) * (l4 - 4) * (l4 + 4) * (l4 - 3) * (l4 + 3) * (l4 - 2))
            for l1 in range(2, l_max + 1)
            for l2 in range(2, l_max + 1)
            for l3 in range(2, l_max + 1)
        )
        for l4, m4, l, m, n, p, lp, mp, nprime, pp, lpp, mpp, npp, ppp in indices
    ]


def multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0, 
                           t0_method='geq', T=100, spherical_modes=None):

    # Use the requested spherical modes
    if spherical_modes is None:
        spherical_modes = list(data_dict.keys())
    
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0 - 1e-9) & (times<t0+T - 1e-9)

        times = times[data_mask]
        data = np.concatenate(
            [data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {lm: data_dict[lm][data_mask] for lm in spherical_modes}
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        
        times = times[start_index:end_index]
        data = np.concatenate(
            [data_dict[lm][start_index:end_index] for lm in spherical_modes])
        data_dict_mask = {
            lm: data_dict[lm][start_index:end_index] for lm in spherical_modes}
        
    else:
        print("""Requested t0_method is not valid. Please choose between
              'geq' and 'closest'.""")
    
    data_dict = data_dict_mask
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(modes, chif, Mf))
    
    # Construct the coefficient matrix for use with NumPy's lstsq function. 
    
    # Mixing coefficients
    # -------------------
    
    # A list of lists for the mixing coefficient indices. The first list is
    # associated with the first lm mode. The second list is associated with
    # the second lm mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
    indices_lists = [
        [lm_mode+mode for mode in modes] for lm_mode in spherical_modes]
    
    # Convert each tuple of indices in indices_lists to a mu value
    mu_lists = [qnm.mu_list(indices, chif) for indices in indices_lists]
        
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.concatenate([np.array([
        mu_lists[i][j]*np.exp(-1j*frequencies[j]*(times-t0)) 
        for j in range(len(frequencies))]).T 
        for i in range(len(spherical_modes))])

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum('ij,j->i', a, C)
    
    # Split up the result into the separate spherical harmonic modes, and
    # store to a dictionary. We also store the "weighted" complex amplitudes 
    # to a dictionary.
    model_dict = {}
    weighted_C = {}
    
    for i, lm in enumerate(spherical_modes):
        model_dict[lm] = model[i*len(times):(i+1)*len(times)]
        weighted_C[lm] = np.array(mu_lists[i])*C

    # Convert model_dict and data_dict into arrays of shape (len(spherical_modes), len(times))
    model_array = np.array([model_dict[lm] for lm in spherical_modes])
    data_array = np.array([data_dict[lm] for lm in spherical_modes])
    
    # Calculate the (sky-averaged) mismatch for the fit
    mm = mismatch(model_array, data_array)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'weighted_C': weighted_C,
        'data': data_dict,
        'model': model_dict,
        'model_times': times,
        't0': t0,
        'modes': modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit