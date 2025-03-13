import numpy as np
import matplotlib.pyplot as plt
import palette
import tidal_track as tt
from palette import pc
import symlilb
import errani_model

def mass_time_plot():
    suite = "SymphonyMilkyWay"
    for i_host in range(symlib.n_hosts(suite)):
        if i_host >= 1: continue
        sim_dir = symlib.get_Host_directory(lib.base_dir, suite, i_host)

        sf, hist = symlib.read_symfind(sim_dir)
        
        param = symlib.simulation_parameters(suite)
        mp = param["mp"]/param["h100"]

        scale = symlib.scale_factors(sim_dir)
        eps = param["eps"]/param["h100"]*scale
        npeak = hist["mpeak_pre"] / mp
        
        _, _, ranks = symlib.tag_stars(
            sim_dir, symlib.DWARF_GALAXY_HALO_MODEL)
        t_relax = ranks[i].relaxation_time(mp, eps[snap])

        m_mpeak = 
        
        
def main():
    mass_time_plot()

if __name__ == "__main__": main()
