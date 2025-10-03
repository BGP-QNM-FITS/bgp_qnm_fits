#import jax
from .main_fit import BGP_fit
from .select_fit import BGP_select
from .main_fit_lite import BGP_fit_lite
from .gp_kernels import (
    kernel_test,
    kernel_WN,
    kernel_GP,
    kernel_GP_test,
    kernel_GPC,
    compute_kernel_matrix,
    compute_kernel_matrix_test
)
from .qnm_funcs import get_significance, get_log_significance_mode, get_significance_list

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
from .data import get_tuned_param_dict, get_residual_data, get_param_dict, SXS_CCE

__all__ = [
    "BGP_fit",
    "BGP_select",
    "BGP_fit_lite", 
    "kernel_test",
    "kernel_WN",
    "kernel_GP",
    "kernel_GP_test",
    "kernel_GPC",
    "compute_kernel_matrix",
    "compute_kernel_matrix_test",
    "get_significance",
    "get_log_significance_mode",
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
    "get_tuned_param_dict",
    "get_residual_data",
    "get_param_dict",
    "SXS_CCE",
]

#jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
