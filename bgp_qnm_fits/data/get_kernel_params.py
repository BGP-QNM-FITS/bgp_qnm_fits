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
TRAINING_RANGE_WN = 80

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

    times = np.arange(
        RESIDUAL_BIG_START, RESIDUAL_BIG_END + RESIDUAL_BIG_START, TIME_STEP
    )

    mask_WN = (times >= TRAINING_START_TIME_WN) & (times < TRAINING_START_TIME_WN + TRAINING_RANGE_WN)
    mask_GP = (times >= TRAINING_START_TIME_GP) & (times < TRAINING_START_TIME_GP + TRAINING_RANGE_GP) 

    for sim_id in SIMNUMS:
        print(sim_id)

        sim_main = CCE.SXS_CCE(sim_id, type=data_type, lev="Lev5", radius="R2")
        sim_lower = CCE.SXS_CCE(sim_id, type=data_type, lev="Lev4", radius="R2")  # This has the smallest residual relative to sim_main

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

    for data_type in ["strain", "news", "psi4"]:

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


### Version with fixed everything except mu and sigma max (trained from t0 = 20, T= 60)

hyperparam_list_WN_strain = [0.1884539057633573]
hyperparam_list_GP_strain = [6.1930717137040645, 0.38265931880613024]

tuned_params_WN = {}
tuned_params_GP = {}
for sim in SIMNUMS:
    tuned_params_WN[sim] = get_tuned_params(param_dict[sim], hyperparam_list_WN_strain, HYPERPARAM_RULE_DICT_WN, spherical_modes=None)
    tuned_params_GP[sim] = get_tuned_params(param_dict[sim], hyperparam_list_GP_strain, HYPERPARAM_RULE_DICT_GP, spherical_modes=None)

### Version with fixed nu and fixed time shift (3-4 July 2025) 

hyperparam_list_WN_strain = [0.291450707195285]
hyperparam_list_GP_strain = [1.7355105821067085, 0.3334169889885236] 

hyperparam_list_WN_news = [0.3135866632917954]
hyperparam_list_GP_news = [1.7648275353291094, 0.29969283952112363] 

hyperparam_list_WN_psi4 = [0.31201718771610026]
hyperparam_list_GP_psi4 = [1.0627300656346779, 0.13823248846415154] 

#### Version up to 3 July (inc. time shift, nu). 

hyperparam_list_WN_strain = [0.291450707195285]
hyperparam_list_GP_strain = [0.6433438749720057, 12.071825519382735, 1.2443842751955567, 0.3942422358459211]
hyperparam_list_GPC_strain = [1.2510383685002138, -5.23517776049999, 1.6396780693858353, 0.39328926598661273, 4.990821887052544, 1.1414985586111737, 0.8600678005516489] 

hyperparam_list_WN_news = [0.2961591339223706]
hyperparam_list_GP_news = [0.6163200395338854, 8.075565751518583, 1.3250133617942454, 0.37178491850208534]

hyperparam_list_WN_psi4 = [0.2675502250902616]
hyperparam_list_GP_psi4 = [1.3935121549152314, -19.968782952159998, 1.0667804726346821, 0.1389420816783665]
hyperparam_list_GPC_psi4 = [3.0473501991837004, -18.41993769051296, 1.034593909413467, 0.34017403823176584, 4.824659045058732, 0.2680546231597701, 0.8961742418631116] 

# print("Getting hyperparameters...")
    #with open("param_dict_psi4.pkl", "rb") as f:
    #    param_dict = pickle.load(f)
    #with open("R_dict_psi4.pkl", "rb") as f:
    #    R_dict = pickle.load(f)

    #hyperparam_list_WN, le_WN, tuned_params_WN = get_hyperparams_WN(R_dict, param_dict)
    #print("Hyperparameters for WN:", hyperparam_list_WN)
    #hyperparam_list_GP, le_GP, tuned_params_GP = get_hyperparams_GP(R_dict, param_dict)
    #print("Hyperparameters for GP:", hyperparam_list_GP)
    #hyperparam_list_GPC, le_GPC, tuned_params_GPC = get_hyperparams_GPC(R_dict, param_dict)
    #print("Hyperparameters for GPC:", hyperparam_list_GPC)

    #hyperparam_list_WN = [0.29247586]
    #hyperparam_list_GP = [0.74318751, 13.70846665, 1.11053798, 0.41465039]
    #hyperparam_list_GPC = [1.69423002, 6.37401748, 0.00994697, 1.05935143, 0.37516458, 4.3820775, 4.36964301, 0.67815892]

    #hyperparam_list_WN_news = [0.29524285]
    #hyperparam_list_GP_news = [0.86340414, 7.80535758, 1.04620105, 0.3850662] 
    #hyperparam_list_GPC_news = [1.17017152, -1.28313241, 1.21961471, 0.10000007, 1.42072186, 2.71785424, 0.06359145]