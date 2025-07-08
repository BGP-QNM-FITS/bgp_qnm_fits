import pickle
import json
import qnmfits as qnmfits

filepath = "/data/rvnd2-2/CCE_data/superrest_data"


def get_strain_zero_time(ID, lev, radius, zero_time):

    data_filepath = f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_h.pickle"
    with open(data_filepath, "rb") as f:
        h_prime_dict = pickle.load(f)

    times = h_prime_dict.pop("times")

    sim = qnmfits.Custom(
            times,
            h_prime_dict,
            metadata={
                "remnant_mass": 1,
                "remnant_dimensionless_spin": [1,0,0],
            },
            zero_time=zero_time,
        )
    
    return sim.zero_time 



def SXS_CCE(ID, type="strain", lev="Lev5", radius="R2", zero_time=(2,2)):

    if ID == "0305":

        print("Note that 0305 only has one level and radius. These arguments will be ignored.")

        with open(
            f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}.pickle",
            "rb",
        ) as f:
            h_prime_dict = pickle.load(f)
        with open(
            f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_metadata.json",
            "r",
        ) as f:
            metadata = json.load(f)

        times = h_prime_dict.pop("times")

        sim = qnmfits.Custom(
            times,
            h_prime_dict,
            metadata={
                "remnant_mass": metadata["remnant_mass"],
                "remnant_dimensionless_spin": metadata["remnant_dimensionless_spin"],
            },
            zero_time=zero_time,
        )

    else:
        
        if type == "strain":
            data_filepath = f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_h.pickle"
            strain_zero_time = zero_time 
        elif type == "news":
            data_filepath = f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_news.pickle"
            strain_zero_time = get_strain_zero_time(ID, lev, radius, zero_time)
        elif type == "psi4":
            data_filepath = f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_psi4.pickle"
            strain_zero_time = get_strain_zero_time(ID, lev, radius, zero_time)

        with open(
            data_filepath,
            "rb",
        ) as f:
            h_prime_dict = pickle.load(f)

        with open(
            f"{filepath}/SXS:BBH_ExtCCE_superrest:{ID}/SXS:BBH_ExtCCE_superrest:{ID}_{lev}_{radius}_metadata.json",
            "r",
        ) as f:
            metadata = json.load(f)

        times = h_prime_dict.pop("times")

        sim = qnmfits.Custom(
            times,
            h_prime_dict,
            metadata={
                "remnant_mass": metadata["M_f"],
                "remnant_dimensionless_spin": metadata["chi_f"],
            },
            zero_time=strain_zero_time,
        )

    return sim
