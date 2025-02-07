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
DT = 0.5 # Interpolation time step

analysis_times = np.arange(T0, T0 + T, DT) # this should be fine but might be better to use the same times as the data for each sim

SIGMA_MAX_LOWER = 0.1
SIGMA_MAX_UPPER = 5

T_S_LOWER = -20
T_S_UPPER = 30

LENGTH_SCALE_LOWER = 0.1
LENGTH_SCALE_UPPER = 5

PERIOD_LOWER = 0.1
PERIOD_UPPER = 5

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
    
initial_params = [1., 1., 1., 1.] 
bounds = [(SIGMA_MAX_LOWER, SIGMA_MAX_UPPER), (T_S_LOWER, T_S_UPPER), (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER), (PERIOD_LOWER, PERIOD_UPPER)]
args = (param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict, analysis_times)
result = minimize(get_total_log_evidence, initial_params, args = args, method='Nelder-Mead', bounds=bounds)
optimal_params = result.x
print("Optimal parameters:", dict(zip(hyperparam_rule_dict.keys(), optimal_params)))

optimal_param_list = list(optimal_params)

mu_list = np.linspace(0.1, 5, 100)
#mu_list = np.linspace(-20, 30, 100)

results = np.zeros(len(mu_list))

for i, mu in enumerate(mu_list):
    optimal_param_list[2] = mu
    results[i] = get_total_log_evidence(optimal_param_list, param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict, analysis_times)

plt.plot(mu_list, results, label='0001, (2,2)')
plt.ylim(min(results), max(results))
plt.show() 

                
