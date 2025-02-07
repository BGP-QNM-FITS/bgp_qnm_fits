import numpy as np
import CCE
from likelihood_funcs import *
from GP_funcs import *
import pickle

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
    sim_lower = CCE.SXS_CCE(sim_id, lev="Lev4", radius="R2")  # This has the smallest residual relative to sim_main

    # Perform a time shift

    #TODO shift sensitive to range, maybe because of cutting the waveform and doing fft (but it's not a huge effect; effects things on the order of 0.01) 
    time_shift = get_time_shift(sim_main, sim_lower, dt=0.0001, range=35) 
    sim_lower.zero_time = -time_shift
    sim_lower.time_shift()

    # Interpolate onto the same grids

    DT = 0.5

    new_times = np.arange(sim_main.times[0], sim_main.times[-1], DT)
    sim_main_interp = sim_interpolator(sim_main, new_times)
    sim_lower_interp = sim_interpolator(sim_lower, new_times)

    # Get the residual dictionary

    residual_sim_dict[sim_id] = {
        key: sim_main_interp.h[key] - sim_lower_interp.h[key]
        for key in sim_main_interp.h.keys()
    }

with open("residual_sim_dict.pkl", "wb") as f:
    pickle.dump(residual_sim_dict, f)
