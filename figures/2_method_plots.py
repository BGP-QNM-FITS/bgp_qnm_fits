import qnmfits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from pathlib import Path
from plot_config import PlotConfig

notebook_dir = Path().resolve()
sys.path.append(str(notebook_dir.parent))

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from bayes_qnm_GP_likelihood import *
from bayes_qnm_GP_likelihood.BGP_fits import BGP_fit

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

custom_colormap = LinearSegmentedColormap.from_list(
            "custom_colormap", config.colors
        )
colors = custom_colormap(np.linspace(0, 1, 3))

data_dir = notebook_dir.parent / "data"

SIMNUMS = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013"]
TRAINING_SPH_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]

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

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

mode_filters = {
        "PE": lambda mode: mode[1] >= 0 and mode[1] % 2 == 0,
        "P": lambda mode: mode[1] >= 0,
        "E": lambda mode: mode[1] % 2 == 0,
        "ALL": lambda mode: True,
    }

analysis_times = np.arange(TRAINING_START_TIME, TRAINING_END_TIME, TIME_STEP)

with open(data_dir / "tuned_params_GPC.pkl", "rb") as f:
    params_GPC = pickle.load(f)

with open(data_dir / "tuned_params_GP.pkl", "rb") as f:
    params_GP = pickle.load(f)

with open(data_dir / "tuned_params_WN.pkl", "rb") as f:
    params_WN = pickle.load(f)


def js_divergence_figs():

    sn_list_full = np.array([])
    cn_list_full = np.array([])
    sc_list_full = np.array([])

    for sim_id, mode_rule in SIM_TRAINING_MODE_RULES.items():  # Only want one figure

        sn_list = np.array([])
        cn_list = np.array([])
        sc_list = np.array([])

        spherical_modes = params_WN[sim_id].keys()
        spherical_modes = [mode for mode in spherical_modes if mode[1] != 0]

        spherical_mode_choice = [
            mode for mode in spherical_modes if mode_filters[mode_rule](mode)
        ]

        for sph_mode in spherical_mode_choice:

            kernel_matrix_WN = compute_kernel_matrix(analysis_times, params_WN[sim_id][sph_mode], kernel_s)
            kernel_matrix_GP = compute_kernel_matrix(analysis_times, params_GP[sim_id][sph_mode], kernel_main)
            kernel_matrix_GPC = compute_kernel_matrix(analysis_times, params_GPC[sim_id][sph_mode], kernel_c)

            kl_div_sn = js_divergence(kernel_matrix_WN, kernel_matrix_GP)
            kl_div_cn = js_divergence(kernel_matrix_GP, kernel_matrix_GPC)
            kl_div_sc = js_divergence(kernel_matrix_WN, kernel_matrix_GPC)

            sn_list = np.append(sn_list, np.log(kl_div_sn))
            cn_list = np.append(cn_list, np.log(kl_div_cn))
            sc_list = np.append(sc_list, np.log(kl_div_sc))

        sn_list[sn_list == -np.inf] = np.nan
        cn_list[cn_list == -np.inf] = np.nan
        sc_list[sc_list == -np.inf] = np.nan

        sn_list_full = np.append(sn_list_full, sn_list)
        cn_list_full = np.append(cn_list_full, cn_list)
        sc_list_full = np.append(sc_list_full, sc_list)

        if sim_id != "0001":
            continue 

        fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
        spherical_mode_labels = [f"{mode}" for mode, sn, cn, sc in zip(spherical_mode_choice, sn_list, cn_list, sc_list) 
                 if not (np.isnan(sn) or np.isnan(cn) or np.isnan(sc))]
        filtered_sn_list = [sn for sn in sn_list if not np.isnan(sn)]
        filtered_cn_list = [cn for cn in cn_list if not np.isnan(cn)]
        filtered_sc_list = [sc for sc in sc_list if not np.isnan(sc)]

        ax.scatter(spherical_mode_labels, filtered_cn_list, label='GP, GPc', color=colors[0], alpha=0.8)
        ax.scatter(spherical_mode_labels, filtered_sc_list, label='WN, GPc', color=colors[1], alpha=0.8)
        ax.scatter(spherical_mode_labels, filtered_sn_list, label='WN, GP', color=colors[2], alpha=0.8)

        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel('Log(JSD)')

        ax.tick_params(axis='x', labelsize=8, rotation=90)

        ax.legend()

        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle='-', alpha=0.8)

        fig.savefig(f'outputs/JS_{sim_id}.pdf')
        plt.close(fig)

    return sn_list_full, cn_list_full, sc_list_full

def js_divergence_histogram(sn_list_full, cn_list_full, sc_list_full):
    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))

    breakpoint() 

    ax.hist(cn_list_full, bins=30, alpha=0.8, label='GP, GPc', color=colors[0])
    ax.hist(sc_list_full, bins=30, alpha=0.8, label='WN, GPc', color=colors[1])
    ax.hist(sn_list_full, bins=30, alpha=0.8, label='WN, GP', color=colors[2])

    ax.set_xlabel('Log(JSD)')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=False)
    ax.grid(False)

    plt.xticks()
    plt.yticks()

    fig.savefig(f'outputs/JS_histogram.pdf')


if __name__ == "__main__":
    sn_list_full, cn_list_full, sc_list_full = js_divergence_figs()
    js_divergence_histogram(sn_list_full, cn_list_full, sc_list_full)