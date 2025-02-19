import numpy as np
import matplotlib.pyplot as plt
from likelihood_funcs import * 
from GP_funcs import *
import pickle 
import pdb 
import sys

sys.setrecursionlimit(10000)

T0 = 0 
T = 50
DT = 0.5 # Interpolation time step

analysis_times = np.arange(T0, T0 + T, DT) 

with open('param_dict_sim_lm_full.pkl', 'rb') as f:
    param_dict_sim_lm = pickle.load(f)

with open('f_dict_sim_lm_full.pkl', 'rb') as f:
    f_dict_sim_lm = pickle.load(f)

tuning_hyperparams_s = [0.4011230468749995] 
tuning_hyperparams = [1.321358102430008, -0.01345115218261082, 4.093379916592142, 0.5113668268176057] 
tuning_hyperparams_c = [1.316402500108596, 0.4326335562654318, -0.13408450413327633, 4.039978781750003, 0.1971672415404262, 4.933667981055878, 4.969654116813137, -0.35133507763297867] 

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

SPH_MODES_FULL = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
]

SPH_MODES_NO_ODD = [
        (2, 2),
        (3, 2),
        (4, 4),
        (2, -2),
        (3, -2),
        (4, -4)
]

SPH_MODES_NO_NEGATIVE = [
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
]

SPH_MODES_NO_ODD_NO_NEGATIVE = [
        (2, 2),
        (3, 2),
        (4, 4)
]

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

SPH_MODE = (2,2)

#print(get_total_log_evidence(tuning_hyperparams_s, param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict_s, analysis_times, kernel_s))
#print(get_total_log_evidence(tuning_hyperparams, param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict, analysis_times, kernel))
#print(get_total_log_evidence(tuning_hyperparams_c, param_dict_sim_lm, f_dict_sim_lm, hyperparam_rule_dict_c, analysis_times, kernel_c))

sn_list_full = []
cn_list_full = []
sc_list_full = []

for sim_id in simnums:

    sn_list = []
    cn_list = []
    sc_list = []

    spherical_modes = param_dict_sim_lm[sim_id].keys()

    if sim_id == "0001" or sim_id == "0002" or sim_id == "0003" or sim_id == "0004":
        spherical_modes = [mode for mode in spherical_modes if mode[1] >= 0 and mode[1] % 2 == 0]
    elif sim_id == "0005" or sim_id == "0006" or sim_id == "0007" or sim_id == "0010" or sim_id == "0011" or sim_id == "0012":
        spherical_modes = [mode for mode in spherical_modes if mode[1] >= 0]
    elif sim_id == "0009":
        spherical_modes = [mode for mode in spherical_modes if mode[1] % 2 == 0]
    elif sim_id == "0008" or sim_id == "0013":
        spherical_modes = spherical_modes

    for sph_mode in spherical_modes:

        tuned_param_dict_s = get_new_params(param_dict_sim_lm[sim_id][sph_mode], tuning_hyperparams_s, hyperparam_rule_dict_s)
        tuned_param_dict = get_new_params(param_dict_sim_lm[sim_id][sph_mode], tuning_hyperparams, hyperparam_rule_dict)
        tuned_param_dict_c = get_new_params(param_dict_sim_lm[sim_id][sph_mode], tuning_hyperparams_c, hyperparam_rule_dict_c)

        n_times = len(analysis_times)

        kernel_matrix_s = kernel_s(np.asarray(analysis_times), **tuned_param_dict_s) + np.eye(len(analysis_times)) * 1e-12
        kernel_matrix = kernel(np.asarray(analysis_times), **tuned_param_dict) + np.eye(len(analysis_times)) * 1e-12
        kernel_matrix_c = kernel_c(np.asarray(analysis_times), **tuned_param_dict_c) + np.eye(len(analysis_times)) * 1e-12

        #kl_div_sn = kl_divergence(kernel_matrix_s, kernel_matrix)
        #kl_div_cn = kl_divergence(kernel_matrix, kernel_matrix_c)
        #kl_div_sc = kl_divergence(kernel_matrix_s, kernel_matrix_c)

        kl_div_sn = js_divergence(kernel_matrix_s, kernel_matrix)
        kl_div_cn = js_divergence(kernel_matrix, kernel_matrix_c)
        kl_div_sc = js_divergence(kernel_matrix_s, kernel_matrix_c)

        #kl_div_sn = ws_distance(kernel_matrix_s, kernel_matrix)
        #kl_div_cn = ws_distance(kernel_matrix, kernel_matrix_c)
        #kl_div_sc = ws_distance(kernel_matrix_s, kernel_matrix_c)

        #kl_div_sn_normalised = kl_div_sn / kl_div_sn 
        #kl_div_cn_normalised = kl_div_cn / kl_div_sn
        #kl_div_sc_normalised = kl_div_sc / kl_div_sn

        #kl_div_sn = hellinger_distance(kernel_matrix_s, kernel_matrix)
        #kl_div_cn = hellinger_distance(kernel_matrix, kernel_matrix_c)
        #kl_div_sc = hellinger_distance(kernel_matrix_s, kernel_matrix_c)

        sn_list.append(kl_div_sn)
        cn_list.append(kl_div_cn)
        sc_list.append(kl_div_sc)

    sn_list_full.extend(sn_list)
    cn_list_full.extend(cn_list)
    sc_list_full.extend(sc_list)

    fig, ax = plt.subplots(figsize=(20, 6))

    spherical_mode_labels = [f"{mode[0]},{mode[1]}" for mode in spherical_modes]
    ax.scatter(spherical_mode_labels, np.log(cn_list), label='Normal,Complicated', marker='s')
    ax.scatter(spherical_mode_labels, np.log(sc_list), label='Simple,Complicated', marker='^')
    ax.scatter(spherical_mode_labels, np.log(sn_list), label='Simple,Normal', marker='o')

    ax.set_xlabel('Spherical mode', fontsize=16)
    ax.set_ylabel('Log of JS Divergence', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)

    fig.savefig(f'figs/JS_{sim_id}.pdf')
    plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 6))

cn_log = np.log(cn_list_full)
sc_log = np.log(sc_list_full)
sn_log = np.log(sn_list_full)

cn_log[cn_log == -np.inf] = np.nan
sc_log[sc_log == -np.inf] = np.nan
sn_log[sn_log == -np.inf] = np.nan

ax.hist(cn_log, bins=20, alpha=0.5, label='Normal,Complicated')
ax.hist(sc_log, bins=20, alpha=0.5, label='Simple,Complicated')
ax.hist(sn_log, bins=20, alpha=0.5, label='Simple,Normal')

ax.set_xlabel('Log of JS Divergence', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.legend(frameon=False, fontsize=14)
ax.grid(False)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

fig.savefig(f'figs/JS_histogram.pdf')