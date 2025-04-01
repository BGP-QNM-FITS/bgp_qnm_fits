import numpy as np
import funcs.CCE as CCE
from funcs.likelihood_funcs import *
from funcs.GP_funcs import *
from funcs.utils import *
from funcs.kernel_param_funcs import *
import pickle

SIMNUMS = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013"]
RINGDOWN_START_TIMES = [17.0, 21.0, 23.0, 26.0, 17.0, 17.0, 17.0, 11.0, 29.0, 16.0, 12.0, 17.0, 6.0]
TRAINING_SPH_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]

SMOOTHNESS = 16
EPSILON = 1 / 10

# These determine the parameter and training range but do not have to match `analysis times' used later.

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

# These are the bounds of the minimisation for the kernel hyperparameters

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.1, 5
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 5

BOUNDS = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (T_S_LOWER, T_S_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
]

INITIAL_PARAMS = [1.0, 0.0, 1.0, 1.0]

HYPERPARAM_RULE_DICT = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}

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

R_sim_lm = {}
param_dict_sim_lm = {}

print("Getting parameters...")

# This first section is extremely generic and can be used for any subset of modes or simulations. In principle if 
# the timesteps in R are downsampled this might slightly alter the values of sigma_max and sigma_min
# but this shouldn't affect things too much. 

for i, sim_id in enumerate(SIMNUMS):

    print(sim_id)

    sim_main = CCE.SXS_CCE(sim_id, lev="Lev5", radius="R2")
    sim_lower = CCE.SXS_CCE(
        sim_id, lev="Lev4", radius="R2"
    )  # This has the smallest residual relative to sim_main

    R_lm = get_residuals(
        sim_main, sim_lower, TRAINING_START_TIME, TRAINING_END_TIME, dt=TIME_STEP
    )

    param_dict_full = get_params(
        R_lm,
        sim_main.Mf,
        sim_main.chif_mag,
        RINGDOWN_START_TIMES[i],
        SMOOTHNESS,
        EPSILON,
    )

    R_sim_lm[sim_id] = R_lm
    param_dict_sim_lm[sim_id] = param_dict_full

with open("param_dict_sim_lm_full.pkl", "wb") as f:
    pickle.dump(param_dict_sim_lm, f)

with open("R_dict_sim_lm_full.pkl", "wb") as f:
    pickle.dump(R_sim_lm, f)

print("Getting hyperparameters...")

hyperparam_list, le, tuned_params_sim_lm = train_hyper_params(
    TRAINING_START_TIME,
    TRAINING_END_TIME,
    TIME_STEP,
    INITIAL_PARAMS,
    BOUNDS,
    param_dict_sim_lm,
    R_sim_lm,
    HYPERPARAM_RULE_DICT,
    kernel_main,
    TRAINING_SPH_MODES,
    SIM_TRAINING_MODE_RULES,
)

with open("tuned_params.pkl", "wb") as f:
    pickle.dump(tuned_params_sim_lm, f)