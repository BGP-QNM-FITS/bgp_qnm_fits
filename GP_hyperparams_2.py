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

with open('residual_sim_dict.pkl', 'rb') as f:
    residual_sim_dict = pickle.load(f)

spherical_modes = [(2,2),(2,1),(3,3),(3,2),(4,4)]
params = ['sigma_max', 't_s', 'length_scale', 'period']

t0 = 0 # TODO: Should this be the variable start time? 
T = 100

param_log_evidence_total = {
    'sigma_max': np.zeros(100),
    't_s': np.zeros(100),
    'length_scale': np.zeros(100),
    'period': np.zeros(100)
}

for sim_id in simnums:

    print(sim_id)

    sim_main = CCE.SXS_CCE(sim_id, lev="Lev5", radius="R2")
    analysis_mask = (sim_main.times >= t0 - 1e-9) & (sim_main.times < t0+T - 1e-9)
    analysis_times = sim_main.times[analysis_mask]

    for mode in spherical_modes:

        ell, m = mode

        R = residual_sim_dict[sim_id][mode]
        R_amplitude = np.max(np.abs(R[analysis_mask]))

        omega = qnmfits.qnm.omega(ell, m, 0, 1, sim_main.chif_mag, sim_main.Mf)
        hyperparam_dict = {
            'sigma_max': R_amplitude,
            'sigma_min': R_amplitude / 10,
            't_s': 17,
            'sharpness': 16,
            'length_scale': -1/omega.imag,
            'period': (2*np.pi)/omega.real
        }

        for i, param in enumerate(params):

            values_ts = np.linspace(hyperparam_dict['t_s'] - 30, hyperparam_dict['t_s'] + 30, 100)
            values_other = hyperparam_dict[param] * np.linspace(0.1, 3, 100)

            if param == 't_s':
                values = values_ts
            else:
                values = values_other

            log_evidence_values = log_evidence_range(R, 
                                                    param, 
                                                    analysis_times, 
                                                    analysis_mask, 
                                                    hyperparam_dict,
                                                    values)
            
            param_log_evidence_total[param] += log_evidence_values

fig, axs = plt.subplots(len(params), 1, figsize=(10, 20))

for i, param in enumerate(params):
    if param == 't_s':
        axs[i].axvline(hyperparam_dict[param], color='r', linestyle='--')
        values = values_ts
    else:
        values = np.linspace(0.1, 10, 100)
        axs[i].axvline(1, color='r', linestyle='--')

    axs[i].plot(values, param_log_evidence_total[param])
    axs[i].set_title(f'Log Evidence vs {param}')
    axs[i].set_xlabel(param)
    axs[i].set_ylabel('Log Evidence')

plt.tight_layout()
plt.show()