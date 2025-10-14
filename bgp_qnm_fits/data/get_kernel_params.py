import pickle
import sys
import CCE as CCE
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bgp_qnm_fits import (
    get_residuals,
    get_params,
    train_hyper_params,
    kernel_WN,
    kernel_GP,
    kernel_GPC,
    get_tuned_params,
)

notebook_dir = Path().resolve()
sys.path.append(str(notebook_dir.parent))

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

SMOOTHNESS = 1e-3
TIME_SHIFT = 0

# These determine the parameter and training range but do not have to match `analysis times' used later.

RESIDUAL_BIG_START = -10
RESIDUAL_BIG_END = 310
TIME_STEP = 0.1

TRAINING_START_TIME_WN = 0
TRAINING_RANGE_WN = 200

TRAINING_START_TIME_GP = 20
TRAINING_RANGE_GP = 60

# Define training bounds

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.01, 50
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 10

# SMOOTHNESS_LOWER, SMOOTHNESS_UPPER = 1e-4, 1e-2
LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER = 0.1, 10
PERIOD_2_LOWER, PERIOD_2_UPPER = 0.1, 10
A_LOWER, A_UPPER = 0, 1

BOUNDS_WN = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
]

BOUNDS_GP = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
]

BOUNDS_GPC = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
    (LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER),
    (PERIOD_2_LOWER, PERIOD_2_UPPER),
    (A_LOWER, A_UPPER),
]

# Set initial params

INITIAL_PARAMS_WN = [0.291450707195285]
INITIAL_PARAMS_GP = [5.51806949954791, 1.608138148779154]
INITIAL_PARAMS_GPC = [5.51806949954791, 1.608138148779154, 1, 1.608138148779154, 0.5]

# Define rules for updating params

HYPERPARAM_RULE_DICT_WN = {
    "sigma_max": "multiply",
}

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "period": "multiply",
}

HYPERPARAM_RULE_DICT_GPC = {
    "sigma_max": "multiply",
    "period": "multiply",
    "length_scale_2": "multiply",
    "period_2": "multiply",
    "a": "replace",
}


def get_parameters(data_type="strain", R_lm_id_big=None):
    R_dict_WN = {}
    R_dict_GP = {}
    R_dict_big = {}
    param_dict = {}

    times = np.arange(RESIDUAL_BIG_START, RESIDUAL_BIG_END + RESIDUAL_BIG_START, TIME_STEP)

    mask_WN = (times >= TRAINING_START_TIME_WN) & (times < TRAINING_START_TIME_WN + TRAINING_RANGE_WN)
    mask_GP = (times >= TRAINING_START_TIME_GP) & (times < TRAINING_START_TIME_GP + TRAINING_RANGE_GP)

    for sim_id in SIMNUMS:
        print(sim_id)

        sim_main = CCE.SXS_CCE(sim_id, type=data_type, lev="Lev5", radius="R2")
        sim_lower = CCE.SXS_CCE(
            sim_id, type=data_type, lev="Lev4", radius="R2"
        )  # This has the smallest residual relative to sim_main

        if R_lm_id_big is None:
            R_lm_big = get_residuals(
                sim_main,
                sim_lower,
                RESIDUAL_BIG_START,
                RESIDUAL_BIG_END,
                dt=TIME_STEP,
            )
        else:
            R_lm_big = R_lm_id_big[sim_id]

        params_lm = get_params(
            R_lm_big,
            np.arange(RESIDUAL_BIG_START, RESIDUAL_BIG_START + RESIDUAL_BIG_END, TIME_STEP),
            sim_main.Mf,
            sim_main.chif_mag,
            SMOOTHNESS,
            TIME_SHIFT,
            data_type=data_type,
        )

        R_dict_WN[sim_id] = {key: value[mask_WN] for key, value in R_lm_big.items()}
        R_dict_GP[sim_id] = {key: value[mask_GP] for key, value in R_lm_big.items()}
        R_dict_big[sim_id] = R_lm_big
        param_dict[sim_id] = params_lm

    with open(f"R_dict_{data_type}_big.pkl", "wb") as f:
        pickle.dump(R_dict_big, f)

    with open(f"param_dict_{data_type}.pkl", "wb") as f:
        pickle.dump(param_dict, f)

    return R_dict_WN, R_dict_GP, param_dict


def get_hyperparams_WN(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME_WN,
        TRAINING_RANGE_WN,
        TIME_STEP,
        INITIAL_PARAMS_WN,
        BOUNDS_WN,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_WN,
        kernel_WN,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


def get_hyperparams_GP(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME_GP,
        TRAINING_RANGE_GP,
        TIME_STEP,
        INITIAL_PARAMS_GP,
        BOUNDS_GP,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GP,
        kernel_GP,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


def get_hyperparams_GPC(R_dict, param_dict):
    hyperparam_list, le, tuned_params = train_hyper_params(
        TRAINING_START_TIME_GP,
        TRAINING_RANGE_GP,
        TIME_STEP,
        INITIAL_PARAMS_GPC,
        BOUNDS_GPC,
        param_dict,
        R_dict,
        HYPERPARAM_RULE_DICT_GPC,
        kernel_GPC,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


if __name__ == "__main__":

    # for data_type in ["strain", "news", "psi4"]:
    for data_type in ["news"]:

        print(f"Training on {data_type}...")

        with open(f"R_dict_{data_type}_big.pkl", "rb") as f:
            R_lm_id_big = pickle.load(f)

        R_dict_WN, R_dict_GP, param_dict = get_parameters(data_type=data_type, R_lm_id_big=R_lm_id_big)

        hyperparam_list_WN, le_WN, tuned_params_WN = get_hyperparams_WN(R_dict_WN, param_dict)
        hyperparam_list_GP, le_GP, tuned_params_GP = get_hyperparams_GP(R_dict_GP, param_dict)
        hyperparam_list_GPC, le_GPC, tuned_params_GPC = get_hyperparams_GPC(R_dict_GP, param_dict)

        with open(f"tuned_params_WN_{data_type}.pkl", "wb") as f:
            pickle.dump(tuned_params_WN, f)
        with open(f"tuned_params_GP_{data_type}.pkl", "wb") as f:
            pickle.dump(tuned_params_GP, f)
        with open(f"tuned_params_GPC_{data_type}.pkl", "wb") as f:
            pickle.dump(tuned_params_GPC, f)


# Strain:
# Optimal parameters: {'sigma_max': 0.20294461292163857} Log evidence: -3765876.728482591
# Optimal parameters: {'sigma_max': 5.517601498993699, 'period': 1.608100776477014} Log evidence: -1740056.1988961035

# News:
# Optimal parameters: {'sigma_max': 0.20202315621798156} Log evidence: -4313245.295242246
# Optimal parameters: {'sigma_max': 6.916479343926145, 'period': 1.6788343847045542} Log evidence: -2169834.025735345
# Optimal parameters: {'sigma_max': 8.28904248745442, 'period': 1.959288624090839, 'length_scale_2': 0.4204419582784812, 'period_2': 1.0019473516595827, 'a': 0.11533919911875894}

# Psi4:
# Optimal parameters: {'sigma_max': 0.19031816565801468} Log evidence: -4965094.051965869
# Optimal parameters: {'sigma_max': 6.4913508308296635, 'period': 1.7270960691077046} Log evidence: -2503035.56179217
