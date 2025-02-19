import numpy as np
import matplotlib.pyplot as plt
import qnmfits
import CCE
from likelihood_funcs import *
from GP_funcs import *
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
import pdb
import pickle

T0 = 0 
T = 50
DT = 0.5 # Interpolation time step

analysis_times = np.arange(T0, T0 + T, DT) # this should be fine but might be better to use the same times as the data for each sim

SIGMA_MAX_LOWER = 0.1
SIGMA_MAX_UPPER = 5

T_S_LOWER = -20
T_S_UPPER = 30

SHARPNESS_LOWER = -15
SHARPNESS_UPPER = 15

LENGTH_SCALE_LOWER = 0.1
LENGTH_SCALE_UPPER = 5

PERIOD_LOWER = 0.1
PERIOD_UPPER = 5

A_LOWER = -0.5
A_UPPER = 0.5 

hyperparam_rule_dict_s = {
    "sigma_max": "multiply",
}

hyperparam_rule_dict = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}

hyperparam_rule_dict_c = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "sharpness": "sum",
    "length_scale": "multiply",
    "period": "multiply",
    "length_scale_2": "multiply",
    "period_2": "multiply",
    "a": "sum",
}

with open('param_dict_sim_lm.pkl', 'rb') as f:
    param_dict_sim_lm = pickle.load(f)

with open('f_dict_sim_lm.pkl', 'rb') as f:
    f_dict_sim_lm = pickle.load(f)

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

initial_params_s = [1.] 
bounds_s = [(SIGMA_MAX_LOWER, SIGMA_MAX_UPPER)]
    
initial_params = [1., 0., 1., 1.]
#initial_params = [1.2991995385483859, -8.946004823431707, 3.6959875930550123, 0.5004499079132851] 
#initial_params = [1.62701874418161, -8.72864974245108, 3.8094845651320215, 0.5203116863214627]
bounds = [(SIGMA_MAX_LOWER, SIGMA_MAX_UPPER), (T_S_LOWER, T_S_UPPER), (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER), (PERIOD_LOWER, PERIOD_UPPER)]

#initial_params_c = [1.3617638865940755, -1.7508402081427596, 4.218088388785169, 0.5131512024371444, 1., 1., 0] 
initial_params_c = [1., 0., 0., 1., 1., 1., 1., 0.] 
bounds_c = [(SIGMA_MAX_LOWER, SIGMA_MAX_UPPER), 
            (T_S_LOWER, T_S_UPPER), 
            (SHARPNESS_LOWER, SHARPNESS_UPPER),
            (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER), 
            (PERIOD_LOWER, PERIOD_UPPER), 
            (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER), 
            (PERIOD_LOWER, PERIOD_UPPER), 
            (A_LOWER, A_UPPER)]

args_s = (param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict_s, analysis_times, kernel_s)
args = (param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict, analysis_times, kernel)
args_c = (param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict_c, analysis_times, kernel_c)

#result = minimize(get_total_log_evidence, initial_params_s, args = args_s, method='Nelder-Mead', bounds=bounds_s)
#result = minimize(get_total_log_evidence, initial_params, args = args, method='Nelder-Mead', bounds=bounds)
result = minimize(get_total_log_evidence, initial_params_c, args = args_c, method='Nelder-Mead', bounds=bounds_c)

# Experimenting with other options 

#result = differential_evolution(get_total_log_evidence, bounds, args=args)

#minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "args": args}
#result = basinhopping(get_total_log_evidence, initial_params, minimizer_kwargs=minimizer_kwargs)

#result = dual_annealing(get_total_log_evidence, bounds, args=args)

optimal_params = result.x

print("Optimal parameters:", dict(zip(hyperparam_rule_dict_c.keys(), optimal_params)), "Log evidence:", result.fun)