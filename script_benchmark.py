import numpy as np
import os
import json
import jax
import jax.numpy as jnp
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import corner

import qnmfits
from funcs.CCE import SXS_CCE
from funcs.GP_funcs import get_inv_GP_covariance_matrix, compute_kernel_matrix, kernel_main

from power_law_tail_fits_jax import QNM_PLT_BAYES_FIT, get_analysis_times, get_cov_inverse

import argparse
import time


if __name__=='__main__':

    id = '0001'
    t0_vals = np.arange(10.0, 50.1, 1)

    with open('tuned_params.pkl', 'rb') as f:
        params = pickle.load(f)
    tuned_param_dict_main = params[id]

    def get_GP_covariance_matrix(analysis_times, 
                                 kernel, 
                                 tuned_param_dict, 
                                 spherical_modes=None):
        if spherical_modes == None:
            spherical_modes = tuned_param_dict.keys()
        kernel_dict = {
            mode: compute_kernel_matrix(analysis_times, 
                                        tuned_param_dict[mode], 
                                        kernel)
            for mode in spherical_modes
            }

        return np.array([kernel_dict[mode] for mode in spherical_modes])
    
    sim = SXS_CCE(id, lev="Lev5", radius="R2")

    spherical_modes = [(2,2), (3,2), (4,4)]

    n_max = 7
    modes = []
    modes += [(2,2,n,1) for n in np.arange(0, n_max+1)]
    modes += [(3,2,n,1) for n in np.arange(0, n_max+1)]
    modes += [(4,4,n,1) for n in np.arange(0, n_max+1)]
    modes += [(2,2,0,1,2,2,0,1)]

    t_start = time.time()

    fits = {}
    for t0 in t0_vals:

        T = 150.0 - t0

        times, _ = get_analysis_times(sim, t0, T)

        inv_noise_covs = get_cov_inverse(get_GP_covariance_matrix(
                                        times, 
                                        kernel_main, 
                                        tuned_param_dict_main, 
                                        spherical_modes=spherical_modes))
        
        fits[int(t0)] = QNM_PLT_BAYES_FIT(sim, modes, spherical_modes, t0, T, 
                                          inv_noise_covs, include_chif=True, 
                                          include_Mf=True)
        
    t_end = time.time()

    print("Benchmarking: time taken = {:.3f} seconds".format(t_end - t_start))

tref = 17

x, y = 'Re C_(2, 2, 0, 1)', 'Im C_(2, 2, 0, 1)'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()

x, y = 'Re C_(2, 2, 1, 1)', 'Im C_(2, 2, 1, 1)'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()

x, y = 'Re C_(3, 2, 1, 1)', 'Im C_(3, 2, 1, 1)'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()

x, y = 'Re C_(4, 4, 1, 1)', 'Im C_(4, 4, 1, 1)'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()

x, y = 'Re C_(2, 2, 0, 1, 2, 2, 0, 1)', 'Im C_(2, 2, 0, 1, 2, 2, 0, 1)'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()

x, y = 'Mf', 'chif'
corner.corner(np.vstack((fits[tref].samples[x], fits[tref].samples[y])).T, labels=[x, y])
plt.show()
