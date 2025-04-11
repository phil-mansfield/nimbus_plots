import numpy as np
import symlib
import pickle
import sys
from symlib import *

cache_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/star_cache"
base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/"

def read_stars(method_name, suite, i_host):
    """ Returns a tuple of (stars, gal_hist)
    """
    fname = "%s/%s_%s_%d.dat" % (cache_dir, method_name, suite, i_host)

    with open(fname, "rb") as fp:
        return pickle.load(fp)

def get_gal_halo_model(name):
    if name == "fid_dwarf":
        return DWARF_GALAXY_HALO_MODEL
    if name == "fid_dwarf_no_um":
        return DWARF_GALAXY_HALO_MODEL_NO_UM
    elif len(name) > 8 and name[:2] == "r=" and name[-6:] == "_no_um":
        return GalaxyHaloModel(
            StellarMassModel(
                UniverseMachineMStarFit(),
                DarkMatterSFH()
            ),
            ProfileModel(
                FixedRHalf(float(name[2:-6])),
                DeprojectedSersicProfile()
            ),
            MetalModel(
                Kirby2013Metallicity(),
                Kirby2013MDF(model_type="gaussian"),
                FlatFeHProfile(),
                GaussianCoupalaCorrelation()
            )
        )
    elif len(name) > 2 and name[:2] == "r=":
        return GalaxyHaloModel(
            StellarMassModel(
                UniverseMachineMStar(),
                UniverseMachineSFH()
            ),
            ProfileModel(
                FixedRHalf(float(name[2:])),
                DeprojectedSersicProfile()
            ),
            MetalModel(
                Kirby2013Metallicity(),
                Kirby2013MDF(model_type="gaussian"),
                FlatFeHProfile(),
                GaussianCoupalaCorrelation()
            )
        )
    
def main():
    suites = ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup",
              "SymphonyLCluster", "SymphonyCluster", "SymphonyMilkyWayHR",
              "MWest"]

    no_um = ["MWest", "SymphonyMilkyWayHR", "SymphonyCluster"]

    method_names = ["fid_dwarf", "r=0.0038", "r=0.0060", "r=0.0094",
                    "r=0.015",
                    "r=0.024", "r=0.038", "r=0.060", "r=0.15", "r=1"]
    method_names_no_um = ["fid_dwarf_no_um", "r=0.0038", "r=0.0060_no_um",
                          "r=0.0094_no_um", "r=0.015_no_um", "r=0.024_no_um",
                          "r=0.038_no_um", "r=0.060_no_um", "r=1_no_um"]
    
    gal_halos = [get_gal_halo_model(name) for name in method_names]
    gal_halos_no_um = [get_gal_halo_model(name) for name in method_names_no_um]

    for suite in suites:
        n_host = symlib.n_hosts(suite)
        for i_host in range(n_host):
            if i_host != 0: continue
            print(suite, i_host+1, "/", n_host)
            sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

            retag_state = None
            
            for i_model in range(len(gal_halos)):
                if suite in no_um:
                    gal_halo = gal_halos_no_um[i_model]
                    method_name = method_names_no_um[i_model]
                else:
                    gal_halo = gal_halos[i_model]
                    method_name = method_names[i_model]
                    
                print("   ", method_name)
                
                if i_model == 0:
                    stars, gal_hist, ranks = tag_stars(sim_dir, gal_halo)
                else:
                    stars, gal_hist, retag_state = retag_stars(
                        sim_dir, gal_halo, ranks, retag_state
                    )
                    
                fname = "%s/%s_%s_%d.dat" % (cache_dir, method_name, suite, i_host)

                with open(fname, "wb+") as fp:
                    pickle.dump((stars, gal_hist), fp)

if __name__ == "__main__": main()
