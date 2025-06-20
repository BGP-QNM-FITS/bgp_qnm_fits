import os
import pickle
from .CCE import SXS_CCE

DATA_DIR = os.path.dirname(__file__)
WN_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_WN.pkl")
GP_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GP.pkl")
GPc_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GPC.pkl")

RESIDUAL_PATH = os.path.join(DATA_DIR, "R_dict.pkl")
RESIDUAL_BIG_PATH = os.path.join(DATA_DIR, "R_dict_big.pkl")
PARAM_DICT_PATH = os.path.join(DATA_DIR, "param_dict.pkl")


# Load the data files
def get_param_data(kernel_type):
    if kernel_type == "WN":
        with open(WN_PARAMS_PATH, "rb") as f:
            return pickle.load(f)
    elif kernel_type == "GP":
        with open(GP_PARAMS_PATH, "rb") as f:
            return pickle.load(f)
    elif kernel_type == "GPc":
        with open(GPc_PARAMS_PATH, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Invalid kernel type. Choose 'WN', 'GP', or 'GPc'.")


def get_residual_data(big=False):
    if big:
        with open(RESIDUAL_BIG_PATH, "rb") as f:
            return pickle.load(f)
    else:
        with open(RESIDUAL_PATH, "rb") as f:
            return pickle.load(f)


def get_param_dict():
    with open(PARAM_DICT_PATH, "rb") as f:
        return pickle.load(f)


__all__ = ["get_param_data", "get_residual_data", "get_param_dict", "SXS_CCE"]
