import numpy as np
import matplotlib.pyplot as plt
import qnmfits
import CCE 
from likelihood_funcs import * 
import corner
import scipy
from GP_funcs import * 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import pickle

start_times = [
    17.0,
    21.0,
    23.0,
    26.0,
    17.0,
    17.0,
    17.0,
    11.0,
    29.0,
    16.0,
    12.0,
    17.0,
    6.0,
]
simnums = [
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
    "0013",
]

residual_sim_dict = {}

for sim_id in simnums: 
    sim_main = CCE.SXS_CCE(sim_id, lev="Lev5", radius="R2")
    sim_options = [("Lev4", "R2"), ("Lev4", "R3"), ("Lev5", "R3")]
    sims = []

    for lev, rad in sim_options:
        try:
            sim = CCE.SXS_CCE(sim_id, lev=lev, radius=rad)
            sims.append(sim)
        except:
            pass
        
    # Perform a time shift 

    for i, sim in enumerate(sims):
        shifts = np.arange(-0.1, 0.1, 0.0001)
        # TODO: Review whether angle averaged mismatch is the best way to find the shift; it takes too long at the moment 
        shift_idx = np.argmin([data_mismatch(sim_main, sim, modes=None, shift=s) for s in shifts])
        sims[i].zero_time = -shifts[shift_idx]
        sims[i].time_shift()

    sims_interp = []

    # Interpolate all simulations onto the same time grid 

    for i, sim in enumerate(sims):
        sim_interp = {}
        for ell, m in sim.h.keys():
            sim_interp[ell,m] = spline(sim.times, np.real(sim.h[ell,m]), ext=1)(sim_main.times) + \
            1j*spline(sim.times, np.imag(sim.h[ell,m]), ext=1)(sim_main.times)
        sims_interp.append(sim_interp)

    
    # Get the residual dictionary 

    R = get_residual(sim_main, sims_interp)
    residual_sim_dict[sim_id] = R

with open('residual_sim_dict.pkl', 'wb') as f:
    pickle.dump(residual_sim_dict, f)