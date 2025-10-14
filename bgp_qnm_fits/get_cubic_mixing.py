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

from bgp_qnm_fits.qnmfits_funcs import qnm
import json
import pickle


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

    # if m1 + m2 + m3 != m4:
    #    return 0.0 + 0.0j

    def integrand_real(theta, phi):
        # Real part of the integrand
        return np.real(
            np.sin(theta)  # Jacobian factor for spherical coordinates
            * sYlm(l1, m1, theta, phi, s=s1, l_max=l_max)
            * sYlm(l2, m2, theta, phi, s=s2, l_max=l_max)
            * sYlm(l3, m3, theta, phi, s=s3, l_max=l_max)
            * np.conj(sYlm(l4, m4, theta, phi, s=s1 + s2 + s3, l_max=l_max))
        )

    def integrand_imag(theta, phi):
        # Imaginary part of the integrand
        return np.imag(
            np.sin(theta)  # Jacobian factor for spherical coordinates
            * sYlm(l1, m1, theta, phi, s=s1, l_max=l_max)
            * sYlm(l2, m2, theta, phi, s=s2, l_max=l_max)
            * sYlm(l3, m3, theta, phi, s=s3, l_max=l_max)
            * np.conj(sYlm(l4, m4, theta, phi, s=s1 + s2 + s3, l_max=l_max))
        )

    # Integrate real and imaginary parts separately
    real_part = dbl_integrate(integrand_real, 0, 2 * np.pi, 0, np.pi)[0]
    imag_part = dbl_integrate(integrand_imag, 0, 2 * np.pi, 0, np.pi)[0]

    return real_part + 1j * imag_part


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


def __main__():
    chifs = np.linspace(0, 0.99, 10)
    l_max = 6
    indices1 = (6, 6, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1)
    indices2 = (7, 6, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1)

    indices3 = (6, -6, 2, -2, 0, -1, 2, -2, 0, -1, 2, -2, 0, -1)
    indices4 = (7, -6, 2, -2, 0, -1, 2, -2, 0, -1, 2, -2, 0, -1)

    Cmu_1 = {}
    Cmu_2 = {}
    Cmu_3 = {}
    Cmu_4 = {}

    # for chif in chifs:

    # print(f"Computing Cmu_D for chif = {chif:.2f}")

    # Cmu_1[chif] = Cmu_D([indices1], chif, l_max)
    # Cmu_2[chif] = Cmu_D([indices2], chif, l_max)
    # Cmu_3[chif] = Cmu_D([indices3], chif, l_max)[0]
    # Cmu_4[chif] = Cmu_D([indices4], chif, l_max)[0]

    # print(f"Cmu_1 = {Cmu_1[chif]}")
    # print(f"Cmu_2 = {Cmu_2[chif]}")
    # print(f"Cmu_3 = {Cmu_3[chif]}")
    # print(f"Cmu_4 = {Cmu_4[chif]}")

    # Cmus = {
    #    "(6,6)": Cmu_1,
    #    "(7,6)": Cmu_2,
    # }

    Cmus = {
        indices1: {
            0.00: 131.76112901859358 + 2.477174854073867e-15j,
            0.11: 133.91799758041174 - 0.5019862465515939j,
            0.22: 136.2910628889915 - 1.0256744476226212j,
            0.33: 138.93681905930904 - 1.5705254478953303j,
            0.44: 141.93848044609575 - 2.1340433381093113j,
            0.55: 145.42672773114637 - 2.709074763200386j,
            0.66: 149.62628690137032 - 3.2764331475638504j,
            0.77: 154.98211679174213 - 3.7808551806075656j,
            0.88: 162.6377844657466 - 4.022743312876532j,
            0.99: 179.73457387479093 - 2.203303238663466j,
        },
        indices2: {
            0.00: 2.3471357963881303e-13 + 5.252566568326395e-15j,
            0.11: 4.078879600053165 - 0.945100711880101j,
            0.22: 8.63835644935798 - 1.93792540958653j,
            0.33: 13.80725233895815 - 2.978625434301076j,
            0.44: 19.776279011333013 - 4.0640182835001175j,
            0.55: 26.84663829050606 - 5.182621073823807j,
            0.66: 35.53969022481968 - 6.300823937121592j,
            0.77: 46.89576836366866 - 7.317065689505932j,
            0.88: 63.61455108385072 - 7.852278951908485j,
            0.99: 102.81144176470261 - 4.378494059773748j,
        },
        indices3: {
            0.00: 131.76112901859358 + 2.477174854073867e-15j,
            0.11: 133.91799758041174 + 0.5019862465515939j,
            0.22: 136.2910628889915 + 1.0256744476226212j,
            0.33: 138.93681905930904 + 1.5705254478953303j,
            0.44: 141.93848044609575 + 2.1340433381093113j,
            0.55: 145.42672773114637 + 2.709074763200386j,
            0.66: 149.62628690137032 + 3.2764331475638504j,
            0.77: 154.98211679174213 + 3.7808551806075656j,
            0.88: 162.6377844657466 + 4.022743312876532j,
            0.99: 179.73457387479093 + 2.203303238663466j,
        },
        indices4: {
            0.00: 2.3471357963881303e-13 + 5.252566568326395e-15j,
            0.11: -4.078879600053165 - 0.945100711880101j,
            0.22: -8.63835644935798 - 1.93792540958653j,
            0.33: -13.80725233895815 - 2.978625434301076j,
            0.44: -19.776279011333013 - 4.0640182835001175j,
            0.55: -26.84663829050606 - 5.182621073823807j,
            0.66: -35.53969022481968 - 6.300823937121592j,
            0.77: -46.89576836366866 - 7.317065689505932j,
            0.88: -63.61455108385072 - 7.852278951908485j,
            0.99: -102.81144176470261 - 4.378494059773748j,
        },
    }

    output_file = Path("Cmus.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(Cmus, f)


if __name__ == "__main__":
    __main__()
