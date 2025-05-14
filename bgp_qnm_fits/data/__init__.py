import os
import pickle

# Define the path to the data files
DATA_DIR = os.path.dirname(__file__)
WN_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_WN.pkl")
GP_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GP.pkl")
GPc_PARAMS_PATH = os.path.join(DATA_DIR, "tuned_params_GPc.pkl")


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


__all__ = ["get_param_data"]
