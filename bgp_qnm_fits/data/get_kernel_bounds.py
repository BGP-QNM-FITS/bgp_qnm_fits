import numpy as np
import pickle
import sys
from scipy.optimize import differential_evolution
from pathlib import Path

notebook_dir = Path().resolve()
sys.path.append(str(notebook_dir.parent))

from bgp_qnm_fits import *

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
RINGDOWN_START_TIMES = [
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
TRAINING_SPH_MODES = [
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

SIM_TRAINING_MODE_RULES = {
    "0001": "PE",
    "0002": "PE",
    "0003": "PE",
    "0004": "PE",
    "0005": "P",
    "0006": "P",
    "0007": "P",
    "0008": "ALL",
    "0009": "E",
    "0010": "P",
    "0011": "P",
    "0012": "P",
    "0013": "ALL",
}

SMOOTHNESS = 16
EPSILON = 1 / 10

# These determine the parameter and training range but do not have to match `analysis times' used later.

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 1

# Define training bounds

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.1, 5
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 5

SMOOTHNESS_LOWER, SMOOTHNESS_UPPER = 0, 30
LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER = 0.1, 5
PERIOD_2_LOWER, PERIOD_2_UPPER = 0.1, 5
A_LOWER, A_UPPER = 0, 0.9

BOUNDS_WN = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
]

BOUNDS_GP = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (T_S_LOWER, T_S_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
]

BOUNDS_GPC = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (T_S_LOWER, T_S_UPPER),
    (SMOOTHNESS_LOWER, SMOOTHNESS_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
    (LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER),
    (PERIOD_2_LOWER, PERIOD_2_UPPER),
    (A_LOWER, A_UPPER),
]

# Set initial params

INITIAL_PARAMS_WN = [0.2427084073890334]
INITIAL_PARAMS_GP = [
    0.3980389941570934,
    8.809357469573628,
    0.9402560980229044,
    0.2534488252986766,
]
INITIAL_PARAMS_GPC = [
    0.5678699426741673,
    3.3680141572797027,
    7.841502124072786,
    1.241209026430354,
    0.9894982312667636,
    0.1064862157208278,
    0.139811581920352,
    0.5917377132835934,
]

# Define rules for updating params

HYPERPARAM_RULE_DICT_WN = {
    "sigma_max": "multiply",
}

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}

HYPERPARAM_RULE_DICT_GPC = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "sharpness": "replace",
    "length_scale": "multiply",
    "period": "multiply",
    "length_scale_2": "multiply",
    "period_2": "multiply",
    "a": "replace",
}


def get_parameters():
    R_dict = {}
    param_dict = {}

    for i, sim_id in enumerate(SIMNUMS):
        print(sim_id)

        sim_main = CCE.SXS_CCE(sim_id, lev="Lev5", radius="R2")
        sim_lower = CCE.SXS_CCE(sim_id, lev="Lev4", radius="R2")  # This has the smallest residual relative to sim_main

        R_lm = get_residuals(sim_main, sim_lower, TRAINING_START_TIME, TRAINING_END_TIME, dt=TIME_STEP)

        params_lm = get_params(
            R_lm,
            sim_main.Mf,
            sim_main.chif_mag,
            RINGDOWN_START_TIMES[i],
            SMOOTHNESS,
            EPSILON,
        )

        R_dict[sim_id] = R_lm
        param_dict[sim_id] = params_lm

    with open("R_dict.pkl", "wb") as f:
        pickle.dump(R_dict, f)

    with open("param_dict.pkl", "wb") as f:
        pickle.dump(param_dict, f)

    return R_dict, param_dict


def get_hyperparams_WN(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME,
        TRAINING_END_TIME,
        TIME_STEP,
        INITIAL_PARAMS_WN,
        BOUNDS_WN,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_WN,
        kernel_s,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


def get_hyperparams_GP(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME,
        TRAINING_END_TIME,
        TIME_STEP,
        INITIAL_PARAMS_GP,
        BOUNDS_GP,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GP,
        kernel_main,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


def get_hyperparams_GPC(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME,
        TRAINING_END_TIME,
        TIME_STEP,
        INITIAL_PARAMS_GPC,
        BOUNDS_GPC,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GPC,
        kernel_c,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


def get_hyperparams_WN_global(R_dict, param_dict):
    analysis_times = np.arange(
        TRAINING_START_TIME,
        TRAINING_START_TIME + TRAINING_END_TIME,
        TIME_STEP,
    )

    args = (
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_WN,
        analysis_times,
        kernel_s,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    result = differential_evolution(
        get_total_log_evidence,
        BOUNDS_WN,
        args,
    )

    print("Optimisation success?", result.success)
    print("Optimisation message:", result.message)

    print(
        "Optimal parameters:",
        dict(zip(HYPERPARAM_RULE_DICT_WN.keys(), result.x)),
        "Log evidence:",
        result.fun,
    )

    return result.x, result.fun


def get_hyperparams_GP_global(R_dict, param_dict):
    analysis_times = np.arange(
        TRAINING_START_TIME,
        TRAINING_START_TIME + TRAINING_END_TIME,
        TIME_STEP,
    )

    args = (
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GP,
        analysis_times,
        kernel_main,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    result = differential_evolution(
        get_total_log_evidence,
        BOUNDS_GP,
        args,
    )

    print("Optimisation success?", result.success)
    print("Optimisation message:", result.message)

    print(
        "Optimal parameters:",
        dict(zip(HYPERPARAM_RULE_DICT_GP.keys(), result.x)),
        "Log evidence:",
        result.fun,
    )

    return result.x, result.fun


def get_hyperparams_GPC_global(R_dict, param_dict):
    analysis_times = np.arange(
        TRAINING_START_TIME,
        TRAINING_START_TIME + TRAINING_END_TIME,
        TIME_STEP,
    )

    args = (
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GPC,
        analysis_times,
        kernel_c,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    result = differential_evolution(
        get_total_log_evidence,
        BOUNDS_GPC,
        args,
    )

    print("Optimisation success?", result.success)
    print("Optimisation message:", result.message)

    print(
        "Optimal parameters:",
        dict(zip(HYPERPARAM_RULE_DICT_GPC.keys(), result.x)),
        "Log evidence:",
        result.fun,
    )

    return result.x, result.fun


if __name__ == "__main__":
    # R_dict, param_dict = get_parameters()

    # with open("R_dict_mini.pkl", "wb") as f:
    #    pickle.dump(R_dict, f)

    # with open("param_dict_mini.pkl", "wb") as f:
    #    pickle.dump(param_dict, f)

    with open("param_dict_mini.pkl", "rb") as f:
        param_dict = pickle.load(f)
    with open("R_dict_mini.pkl", "rb") as f:
        R_dict = pickle.load(f)

    intital_params_list = []
    log_evidence_list = []
    hyperparams_list = []

    # hyperparams_list_WN_global, le_WN_global = get_hyperparams_WN_global(R_dict, param_dict)
    # hyperparams_list_GP_global, le_GP_global = get_hyperparams_GP_global(R_dict, param_dict)
    # hyperparam_list_GPC_global, le_GPC_global = get_hyperparams_GPC_global(R_dict, param_dict)

    hyperparams_list_WN, le_WN, tuned_params_WN = get_hyperparams_WN(R_dict, param_dict)
    hyperparams_list_GP, le_GP, tuned_params_GP = get_hyperparams_GP(R_dict, param_dict)
    hyperparam_list_GPC, le_GPC, tuned_params_GPC = get_hyperparams_GPC(R_dict, param_dict)

    print("############ Global hyperparameters ############")
    print("Hyperparameters for WN:", hyperparams_list_WN)
    print("Log evidence for WN:", le_WN)
    print("Hyperparameters for GP:", hyperparams_list_GP)
    print("Log evidence for GP:", le_GP)
    print("Hyperparameters for GPC:", hyperparam_list_GPC)
    print("Log evidence for GPC:", le_GPC)

    print("############ Scipy minimize hyperparameters ############")
    print("Hyperparameters for WN:", hyperparams_list_WN_global)
    print("Log evidence for WN:", le_WN_global)
    print("Hyperparameters for GP:", hyperparams_list_GP_global)
    print("Log evidence for GP:", le_GP_global)
    print("Hyperparameters for GPC:", hyperparam_list_GPC_global)
    print("Log evidence for GPC:", le_GPC_global)

    # for i, initial_params in enumerate(initial_params_list):
    #    print("Initial params:", initial_params)
    #    hyperparam_list_GPC, le_GPC, tuned_params_GPC  = get_hyperparams_GPC(R_dict, param_dict, initial_params)
    #    print("Hyperparams GPC:", hyperparam_list_GPC)
    #    print("Log evidence GPC:", le_GPC)
    #    intital_params_list.append(initial_params)
    #    log_evidence_list.append(le_GPC)
    #    hyperparams_list.append(hyperparam_list_GPC)
