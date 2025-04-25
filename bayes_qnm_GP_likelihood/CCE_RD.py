import pickle 
import json
import numpy as np
import qnmfits 

filepath  = '/data/rvnd2-2/CCE_data/superrest_data_test'

def SXS_CCE_RD(ID, zero_time=0, lev = 'Lev5', radius = 'R2', type = 'h'):

    if type == 'h':
        suffix = '_h'
        factor = 1.0
    elif type == 'news':
        suffix = '_news'
        factor = 1.0
    elif type == 'psi4':
        suffix = '_psi4'
        factor = -2.0 # This accounts for the MB convention 

    with open(f'{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}{suffix}.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    with open(f'{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_metadata.json', 'r') as f:
        metadata = json.load(f)

    times = data_dict.pop('times')
    data_dict_corrected = {key: value * factor for key, value in data_dict.items()}

    sim = qnmfits.Custom(
            times, 
            data_dict_corrected, 
            metadata={'remnant_mass':metadata['M_f'], 'remnant_dimensionless_spin':metadata['chi_f']}, 
            zero_time=zero_time
            )

    return sim 