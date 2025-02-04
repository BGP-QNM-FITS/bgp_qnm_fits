import numpy as np
import matplotlib.pyplot as plt
import qnmfits
import CCE
from likelihood_funcs import *
from GP_funcs import *
from scipy.optimize import minimize
import pdb
import pickle

T0 = 0 
T = 50
DT = 1 # Interpolation time step

analysis_times = np.arange(T0, T0 + T, DT) # this should be fine but worth checking 

SIGMA_MAX_LOWER = 0.1
SIGMA_MAX_UPPER = 10

T_S_LOWER = -20
T_S_UPPER = 30

LENGTH_SCALE_LOWER = 0.1
LENGTH_SCALE_UPPER = 10

PERIOD_LOWER = 0.1
PERIOD_UPPER = 10

hyperparam_rule_dict = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}

with open('param_dict_sim_lm.pkl', 'rb') as f:
    param_dict_sim_lm = pickle.load(f)

with open('f_dict_sim_lm.pkl', 'rb') as f:
    f_dict_sim_lm = pickle.load(f)

initial_params = [1., 1., 1., 1.] 
bounds = [(SIGMA_MAX_LOWER, SIGMA_MAX_UPPER), (T_S_LOWER, T_S_UPPER), (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER), (PERIOD_LOWER, PERIOD_UPPER)]
result = minimize(get_total_log_evidence, initial_params, args = (param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict, analysis_times), method='Nelder-Mead', bounds=bounds)
optimal_params = result.x

print("Optimal parameters:", dict(zip(hyperparam_rule_dict.keys(), optimal_params)))