import numpy as np
import os
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import corner
import qnmfits

from bayes_qnm_GP_likelihood import *
from BGP_fits import BGP_fit

import argparse
import time


if __name__=='__main__':

    id = '0001'
    t0_vals = np.arange(10.0, 50.1, 1)

    with open('data/tuned_params.pkl', 'rb') as f:
        params = pickle.load(f)
    tuned_param_dict_main = params[id]
    
    sim = SXS_CCE(id, lev="Lev5", radius="R2")

    spherical_modes = [(2,2), (3,2), (4,4)]

    n_max = 7
    modes = []
    modes += [(2,2,n,1) for n in np.arange(0, n_max+1)]
    modes += [(3,2,n,1) for n in np.arange(0, n_max+1)]
    modes += [(4,4,n,1) for n in np.arange(0, n_max+1)]
    #modes += [(2,2,0,1,2,2,0,1)]

    chif= sim.chif_mag
    Mf = sim.Mf

    samples = {}

    t_start = time.time()

    for t0 in t0_vals:

        T = 150.0 - t0

        fit = BGP_fit(sim.times, 
                        sim.h, 
                        modes, 
                        Mf, 
                        chif, 
                        t0, 
                        tuned_param_dict_main, 
                        kernel_main, 
                        t0_method='closest', 
                        T=T, 
                        spherical_modes=spherical_modes,
                        include_chif=True,
                        include_Mf=True)
        
        #samples[t0] = scipy.stats.multivariate_normal(
        #    fit['mean'], fit['covariance'], allow_singular=True
        #).rvs(size=1000)
        
    t_end = time.time()

    print("Benchmarking: time taken = {:.3f} seconds".format(t_end - t_start))

tref = 17

param_list = [qnm for qnm in modes for _ in range(2)] + ["chif"] + ["Mf"]  

param_choice = [(2,2,0,1)]
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()

param_choice = [(2,2,1,1)]
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()

param_choice = [(3,2,1,1)]
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()

param_choice = [(4,4,1,1)]
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()

param_choice = [(2,2,0,1,2,2,0,1)]
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()

param_choice = ['Mf', 'chif']
param_indices = [i for i, param in enumerate(param_list) if param in param_choice]

corner.corner(
    samples[tref][:, param_indices],
    title=param_choice
)
plt.show()