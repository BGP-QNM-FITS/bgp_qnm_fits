import numpy as np
from matplotlib import pyplot as plt
import bgp_qnm_fits as bgp
from matplotlib import cm
import bgp_qnm_fits.qnmfits_funcs as qnmfits

sph_mode = (6,-6)
sph_mode2 = (7,-6)
mode = (2,-2,0,-1,2,-2,0,-1,2,-2,0,-1) 
chif = 0.7 

chif_range = np.linspace(0, 1, 100)  # Range of chif values from 0 to 1
mu_values_sph_mode = [qnmfits.qnm.mu_list([sph_mode + mode], chif)[0] for chif in chif_range]
mu_values_sph_mode2 = [qnmfits.qnm.mu_list([sph_mode2 + mode], chif)[0] for chif in chif_range]

plt.plot(chif_range.real, mu_values_sph_mode, label="mu vs chif (sph_mode)")
plt.plot(chif_range.real, mu_values_sph_mode2, label="mu vs chif (sph_mode2)")

plt.xlabel("chif")
plt.ylabel("mu")
plt.title("Variation of mu with chif")
plt.legend()
plt.grid()
plt.show()
