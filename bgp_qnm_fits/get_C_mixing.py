import h5py
import quaternionic
import spherical
import numpy as np
import qnm as qnm_loader
import qnmfits_funcs as qnmfits
import spheroidal  

from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from urllib.request import urlretrieve
from scipy.integrate import dblquad as dbl_integrate
from spherical import Wigner3j as w3j
from bgp_qnm_fits.utils import mismatch
import bgp_qnm_fits as bgp

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


def Qmu_C(indices, chif, l_max, **kwargs):
    """
    A function to calculate the C prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the C prediction for.
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

    alphas = []

    for i, j, a, b, c, sign1, e, f, g, sign2 in indices:
        L = a + e
        M = b + f
        omega = qnmfits.qnm.omega_list([(a, b, c, sign1, e, f, g, sign2)], chif, 1)
        gamma = chif * omega[0]
        S = spheroidal.harmonic(-2, L, M, gamma)

        def f_real(theta, phi):
            return np.real(
                np.sin(theta) * S(theta, phi) * np.conj(sYlm(i, j, theta, phi))
            )

        def f_imag(theta, phi):
            return np.imag(
                np.sin(theta) * S(theta, phi) * np.conj(sYlm(i, j, theta, phi))
            )

        alpha_real = dbl_integrate(f_real, 0, 2 * np.pi, 0, np.pi)[0]
        alpha_imag = dbl_integrate(f_imag, 0, 2 * np.pi, 0, np.pi)[0]

        alphas.append(alpha_real + 1j * alpha_imag)

    return alphas


def __main__():
    sim = bgp.SXS_CCE("0010", type="strain", lev="Lev5", radius="R2")
    chif = sim.chif_mag
    l_max = 8

    Qmu44 = Qmu_C([(4, 4, 2, 2, 0, 1, 2, 2, 0, 1)], chif, l_max)
    Qmu54 = Qmu_C([(5, 4, 2, 2, 0, 1, 2, 2, 0, 1)], chif, l_max)

    print(Qmu44)
    print(Qmu54)

    Qmu4m4 = Qmu_C([(4,-4, 2, -2, 0, -1, 2, -2, 0, -1)], chif, l_max)
    Qmu5m4 = Qmu_C([(5,-4, 2, -2, 0, -1, 2, -2, 0, -1)], chif, l_max)

    print(Qmu4m4)
    print(Qmu5m4)

    Qmu55 = Qmu_C([(5, 5, 3, 3, 0, 1, 2, 2, 0, 1)], chif, l_max)
    Qmu65 = Qmu_C([(6, 5, 3, 3, 0, 1, 2, 2, 0, 1)], chif, l_max)

    print(Qmu55)
    print(Qmu65)

    Qmu5m5 = Qmu_C([(5,-5, 3, -3, 0, -1, 2, -2, 0, -1)], chif, l_max)
    Qmu6m5 = Qmu_C([(6,-5, 3, -3, 0, -1, 2, -2, 0, -1)], chif, l_max)

    print(Qmu5m5)
    print(Qmu6m5)

    Qmu66_1 = Qmu_C([(6, 6, 4, 4, 0, 1, 2, 2, 0, 1)], chif, l_max)
    Qmu76_1 = Qmu_C([(7, 6, 4, 4, 0, 1, 2, 2, 0, 1)], chif, l_max)

    print(Qmu66_1)
    print(Qmu76_1)

    Qmu6m6_1 = Qmu_C([(6,-6, 4, -4, 0, -1, 2, -2, 0, -1)], chif, l_max)
    Qmu7m6_1 = Qmu_C([(7,-6, 4, -4, 0, -1, 2, -2, 0, -1)], chif, l_max)

    print(Qmu6m6_1)
    print(Qmu7m6_1)

    Qmu66_2 = Qmu_C([(6, 6, 3, 3, 0, 1, 3, 3, 0, 1)], chif, l_max)
    Qmu76_2 = Qmu_C([(7, 6, 3, 3, 0, 1, 3, 3, 0, 1)], chif, l_max)

    print(Qmu66_2)
    print(Qmu76_2)

    Qmu6m6_2 = Qmu_C([(6,-6, 3, -3, 0, -1, 3, -3, 0, -1)], chif, l_max)
    Qmu7m6_2 = Qmu_C([(7,-6, 3, -3, 0, -1, 3, -3, 0, -1)], chif, l_max)

    print(Qmu6m6_2)
    print(Qmu7m6_2)

    mixing_0010_dict = {
        (4, 4, 2, 2, 0, 1, 2, 2, 0, 1): Qmu44[0],
        (5, 4, 2, 2, 0, 1, 2, 2, 0, 1): Qmu54[0],
        (4,-4, 2, -2, 0, -1, 2, -2, 0, -1): Qmu4m4[0],
        (5,-4, 2, -2, 0, -1, 2, -2, 0, -1): Qmu5m4[0],
        (5, 5, 3, 3, 0, 1, 2, 2, 0, 1): Qmu55[0],
        (6, 5, 3, 3, 0, 1, 2, 2, 0, 1): Qmu65[0],
        (5,-5, 3, -3, 0, -1, 2, -2, 0, -1): Qmu5m5[0],
        (6,-5, 3, -3, 0, -1, 2, -2, 0, -1): Qmu6m5[0],
        (6, 6, 4, 4, 0, 1, 2, 2, 0, 1): Qmu66_1[0],
        (7, 6, 4, 4, 0, 1, 2, 2, 0, 1): Qmu76_1[0],
        (6,-6, 4,-4, 0,-1, 2,-2, 0,-1): Qmu6m6_1[0],
        (7,-6, 4,-4, 0,-1, 2,-2, 0,-1): Qmu7m6_1[0],
        (6, 6, 3, 3, 0 ,1 ,3 ,3 ,0 ,1 ): Qmu66_2[0],
        (7 ,6 ,3 ,3 ,0 ,1 ,3 ,3 ,0 ,1 ): Qmu76_2[0],
        (6 ,-6 ,3 ,-3 ,0 ,-1 ,3 ,-3 ,0 ,-1 ): Qmu6m6_2[0],
        (7 ,-6 ,3 ,-3 ,0 ,-1 ,3 ,-3 ,0 ,-1 ): Qmu7m6_2[0],
    } 

    output_file = Path("Qmus_0010_C.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(mixing_0010_dict, f)

if __name__ == "__main__":
    __main__()