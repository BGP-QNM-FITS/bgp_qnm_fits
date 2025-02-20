import numpy as np
import qnmfits
import CCE
from likelihood_funcs import *
from GP_funcs import *
from utils import * 
import pickle
import pdb

SIMNUMS = [
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

START_TIMES = [
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

SMOOTHNESS = 16

T0 = 0
T = 50
DT = 0.5  # Interpolation time step

with open("residual_sim_dict.pkl", "rb") as f:
    residual_sim_dict = pickle.load(f)

param_dict_sim_lm = {}
f_dict_sim_lm = {}

# Get 'core' parameters for each simulation and mode

for i, sim_id in enumerate(SIMNUMS):

    start_time = START_TIMES[i]

    sim_main = CCE.SXS_CCE(sim_id, lev="Lev5", radius="R2")
    new_times = np.arange(sim_main.times[0], sim_main.times[-1], DT)
    sim_main = sim_interpolator(sim_main, new_times)
    analysis_mask = (sim_main.times >= T0 - 1e-9) & (sim_main.times < T0 + T)

    param_dict_lm = {}
    f_dict_lm = {}

    for mode in sim_main.h.keys():

        ell, m = mode

        R = residual_sim_dict[sim_id][mode]
        f = R[analysis_mask]
        R_amplitude = np.max(np.abs(f))

        if m < 0:
            p = -1
        else:
            p = 1

        omega = qnmfits.qnm.omega(ell, m, 0, p, sim_main.chif_mag, sim_main.Mf)

        param_dict = {
            "sigma_max": R_amplitude,
            "sigma_min": R_amplitude / 10,  # This is actually recomputed later anyway
            "t_s": start_time,
            "sharpness": SMOOTHNESS,
            "length_scale": -1 / omega.imag,
            "period": (2 * np.pi) / omega.real,
            "length_scale_2": -1 / omega.imag, # For the complicated kernel only 
            "period_2": (2 * np.pi) / omega.real, # For the complicated kernel only 
            "a": 0.5 # For the complicated kernel only 
        }

        param_dict_lm[mode] = param_dict
        f_dict_lm[mode] = f

    param_dict_sim_lm[sim_id] = param_dict_lm
    f_dict_sim_lm[sim_id] = f_dict_lm

with open("param_dict_sim_lm_full.pkl", "wb") as f:
    pickle.dump(param_dict_sim_lm, f)

with open("f_dict_sim_lm_full.pkl", "wb") as f:
    pickle.dump(f_dict_sim_lm, f)
