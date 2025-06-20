import pickle
import sys
import CCE as CCE
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
EPSILON = 1 / 10

# These determine the parameter and training range but do not have to match `analysis times' used later.

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

# Define training bounds

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.01, 5
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 5

# SMOOTHNESS_LOWER, SMOOTHNESS_UPPER = 1e-4, 1e-2
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
    # (SMOOTHNESS_LOWER, SMOOTHNESS_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
    (LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER),
    (PERIOD_2_LOWER, PERIOD_2_UPPER),
    (A_LOWER, A_UPPER),
]

# Set initial params

# INITIAL_PARAMS_WN = [1.]
# INITIAL_PARAMS_GP = [1.0, 0.0, 1.0, 1.0]
# INITIAL_PARAMS_GPC = [0.5715534011443748, 0.0032311845355438894, SMOOTHNESS, 1.7176362780942858, 0.31558556618927797, 1.7176362780942858, 0.31558556618927797, 0.5]

INITIAL_PARAMS_WN = [0.7431875096008772]
INITIAL_PARAMS_GP = [
    0.7431875096008772,
    13.708466653509452,
    1.1105379786215344,
    0.414650392357301,
]
# INITIAL_PARAMS_GPC = [0.5678699426741673, 3.3680141572797027, 7.841502124072786, 1.241209026430354, 0.9894982312667636, 0.1064862157208278, 0.139811581920352, 0.5917377132835934]
INITIAL_PARAMS_GPC = [
    0.7431875096008772,
    13.708466653509452,
    # 1e-3,
    1.1105379786215344,
    0.414650392357301,
    1.1105379786215344,
    0.414650392357301,
    0.5,
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
        kernel_WN,
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
        kernel_GP,
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
        kernel_GPC,
        TRAINING_SPH_MODES,
        SIM_TRAINING_MODE_RULES,
    )

    return hyperparam_list, le, tuned_params


if __name__ == "__main__":

    R_dict, param_dict = get_parameters()

    # print("Getting hyperparameters...")
    with open("param_dict.pkl", "rb") as f:
        param_dict = pickle.load(f)
    with open("R_dict.pkl", "rb") as f:
        R_dict = pickle.load(f)

    hyperparam_list_WN, le_WN, tuned_params_WN = get_hyperparams_WN(R_dict, param_dict)
    print("Hyperparameters for WN:", hyperparam_list_WN)
    hyperparam_list_GP, le_GP, tuned_params_GP = get_hyperparams_GP(R_dict, param_dict)
    print("Hyperparameters for GP:", hyperparam_list_GP)
    hyperparam_list_GPC, le_GPC, tuned_params_GPC = get_hyperparams_GPC(R_dict, param_dict)
    print("Hyperparameters for GPC:", hyperparam_list_GPC)

    # hyperparam_list_WN = [0.29247586]
    # hyperparam_list_GP = [0.74318751, 13.70846665, 1.11053798, 0.41465039]
    # hyperparam_list_GPC = [1.69423002, 6.37401748, 0.00994697, 1.05935143, 0.37516458, 4.3820775, 4.36964301, 0.67815892]

    tuned_params_WN = {}
    tuned_params_GP = {}
    tuned_params_GPC = {}
    for sim in SIMNUMS:
        tuned_params_WN[sim] = get_tuned_params(param_dict[sim], hyperparam_list_WN, HYPERPARAM_RULE_DICT_WN)
        tuned_params_GP[sim] = get_tuned_params(param_dict[sim], hyperparam_list_GP, HYPERPARAM_RULE_DICT_GP)
        tuned_params_GPC[sim] = get_tuned_params(param_dict[sim], hyperparam_list_GPC, HYPERPARAM_RULE_DICT_GPC)

    with open("tuned_params_WN.pkl", "wb") as f:
        pickle.dump(tuned_params_WN, f)
    with open("tuned_params_GP.pkl", "wb") as f:
        pickle.dump(tuned_params_GP, f)
    with open("tuned_params_GPC.pkl", "wb") as f:
        pickle.dump(tuned_params_GPC, f)
