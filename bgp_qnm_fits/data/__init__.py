import os
import pickle
from .CCE import SXS_CCE

DATA_DIR = os.path.dirname(__file__)

WN_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_WN_strain.pkl")
GP_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GP_strain.pkl")
GPc_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GPC_strain.pkl")

WN_NEWS_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_WN_news.pkl")
GP_NEWS_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GP_news.pkl")
GPc_NEWS_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GPC_news.pkl")

WN_PSI4_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_WN_psi4.pkl")
GP_PSI4_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GP_psi4.pkl")
GPc_PSI4_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GPC_psi4.pkl")

RESIDUAL_PATH = os.path.join(DATA_DIR, "R_dict_strain.pkl")
RESIDUAL_BIG_PATH = os.path.join(DATA_DIR, "R_dict_strain_big.pkl")

RESIDUAL_PATH_NEWS = os.path.join(DATA_DIR, "R_dict_news.pkl")
RESIDUAL_BIG_PATH_NEWS = os.path.join(DATA_DIR, "R_dict_news_big.pkl")

RESIDUAL_PATH_PSI4 = os.path.join(DATA_DIR, "R_dict_psi4.pkl")
RESIDUAL_BIG_PATH_PSI4 = os.path.join(DATA_DIR, "R_dict_psi4_big.pkl")

PARAM_DICT_PATH_STRAIN = os.path.join(DATA_DIR, "param_dict_strain.pkl")
PARAM_DICT_PATH_NEWS = os.path.join(DATA_DIR, "param_dict_news.pkl")
PARAM_DICT_PATH_PSI4 = os.path.join(DATA_DIR, "param_dict_psi4.pkl")


# Load the data files
def get_param_data(kernel_type, data_type="strain"):
    if data_type == "strain":
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
    elif data_type == "news":
        if kernel_type == "WN":
            with open(WN_NEWS_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        elif kernel_type == "GP":
            with open(GP_NEWS_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        elif kernel_type == "GPc":
            with open(GPc_NEWS_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Invalid kernel type. Choose 'WN', 'GP', or 'GPc'.")
    elif data_type == "psi4":
        if kernel_type == "WN":
            with open(WN_PSI4_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        elif kernel_type == "GP":
            with open(GP_PSI4_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        elif kernel_type == "GPc":
            with open(GPc_PSI4_PARAMS_PATH, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Invalid kernel type. Choose 'WN', 'GP', or 'GPc'.")


def get_residual_data(big=False, data_type="strain"):
    if data_type == "strain":
        if big:
            with open(RESIDUAL_BIG_PATH, "rb") as f:
                return pickle.load(f)
        else:
            with open(RESIDUAL_PATH, "rb") as f:
                return pickle.load(f)
    elif data_type == "news":
        if big:
            with open(RESIDUAL_BIG_PATH_NEWS, "rb") as f:
                return pickle.load(f)
        else:
            with open(RESIDUAL_PATH_NEWS, "rb") as f:
                return pickle.load(f)
    elif data_type == "psi4":
        if big:
            with open(RESIDUAL_BIG_PATH_PSI4, "rb") as f:
                return pickle.load(f)
        else:
            with open(RESIDUAL_PATH_PSI4, "rb") as f:
                return pickle.load(f)


def get_param_dict(data_type = "strain"):
    if data_type == "strain":
        with open(PARAM_DICT_PATH_STRAIN, "rb") as f:
            return pickle.load(f)
    elif data_type == "news":
        with open(PARAM_DICT_PATH_NEWS, "rb") as f:
            return pickle.load(f)
    elif data_type == "psi4":
        with open(PARAM_DICT_PATH_PSI4, "rb") as f:
            return pickle.load(f)



__all__ = ["get_param_data", "get_residual_data", "get_param_dict", "SXS_CCE"]
