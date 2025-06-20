import jax
from .main_fit import BGP_fit
from .gp_kernels import (
    kernel_test,
    kernel_WN,
    kernel_GP,
    kernel_GPC,
    compute_kernel_matrix,
)
from .qnm_funcs import get_significance, get_significance_list

from .gp_training import (
    get_residuals,
    train_hyper_params,
    get_params,
    get_total_log_likelihood,
    get_tuned_params,
    js_divergence,
)

from .utils import (
    sim_interpolator,
    sim_interpolator_data,
    get_time_shift,
    get_inverse,
    mismatch,
)
from .data import get_param_data, get_residual_data, get_param_dict, SXS_CCE

__all__ = [
    "BGP_fit",
    "kernel_test",
    "kernel_WN",
    "kernel_GP",
    "kernel_GPC",
    "compute_kernel_matrix",
    "get_significance",
    "get_significance_list",
    "get_residuals",
    "train_hyper_params",
    "get_params",
    "get_total_log_likelihood",
    "get_tuned_params",
    "js_divergence",
    "sim_interpolator",
    "sim_interpolator_data",
    "get_time_shift",
    "get_inverse",
    "mismatch",
    "get_param_data",
    "get_residual_data",
    "get_param_dict",
    "SXS_CCE",
]

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
