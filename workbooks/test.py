import qnmfits
import numpy as np
import scipy
import corner
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from pathlib import Path

notebook_dir = Path().resolve()
sys.path.append(str(notebook_dir.parent))

notebook_dir = Path().resolve()
data_dir = notebook_dir.parent

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from qnmfits.spatial_mapping_functions import * 
from bayes_qnm_GP_likelihood import *
from bayes_qnm_GP_likelihood.bayes_qnm_GP_likelihood.BGP_fits import BGP_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time

id = "0001"
sim_main = SXS_CCE(id, lev="Lev5", radius="R2")

n_max = 2

T0 = 0
T = 100

qnm_list = [(2,2,n,1) for n in np.arange(0, n_max+1)] #+ [(3,2,0,1)] + [(4,4,n,1) for n in np.arange(0, n_max+1)] + [(5,4,0,1)] 
spherical_modes = [(2, 2)] 

chif_mag_0 = sim_main.chif_mag
Mf_0 = sim_main.Mf

with open(data_dir / 'bayes_qnm_GP_likelihood/data/tuned_params.pkl', 'rb') as f:
    params = pickle.load(f)

tuned_param_dict_main = params[id]
# Benchmarking the speed of BGP_fit

t0_vals = np.arange(10.0, 100.1, 1)

start_time = time.time()
fit = BGP_fit(sim_main.times, 
            sim_main.h, 
            qnm_list, 
            Mf_0, 
            chif_mag_0, 
            t0_vals, 
            tuned_param_dict_main, 
            kernel_main, 
            t0_method='closest', 
            T=T, 
            spherical_modes=spherical_modes,
            include_chif=True,
            include_Mf=True)


end_time = time.time()
qnm_bgp_fit_time = end_time - start_time
print(f"BGP_fit execution time: {qnm_bgp_fit_time:.4f} seconds")

start_time = time.time()

for t0 in t0_vals:

    T = 150.0 - t0

    # Benchmarking the speed of qnm_BGP_fit
    fit_main = qnm_BGP_fit(
            sim_main.times,
            sim_main.h,
            qnm_list,
            Mf_0,
            chif_mag_0,
            t0,
            tuned_param_dict_main,
            kernel_main,
            t0_method="closest",
            T=T,
            spherical_modes=spherical_modes,
            include_chif=True,
            include_Mf=True,
        )
    
end_time = time.time()
qnm_bgp_fit_time = end_time - start_time
print(f"qnm_BGP_fit execution time: {qnm_bgp_fit_time:.4f} seconds")
