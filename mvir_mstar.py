import symlib
import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import lib
import cache_stars

def main():
    palette.configure(True)

    suites = ["SymphonyLMC", "SymphonyMilkyWay",
             "SymphonyGroup", "SymphonyLCluster"]

    p = [0.5-0.68/2, 0.5, 0.5+0.68/2]
    shape = (4,3)
    mvir = np.zeros(shape)
    ms_cen = np.zeros(shape)
    ms_sat = np.zeros(shape)
    ms_halo = np.zeros(shape)
    
    for i_suite, suite in enumerate(suites):
        n_hosts = symlib.n_hosts(suite)
        if suite == "SymphonyLCluster": n_hosts = 32
        
        mvir_i = np.zeros(n_hosts)
        ms_cen_i = np.zeros(n_hosts)
        ms_sat_i = np.zeros(n_hosts)
        ms_halo_i = np.zeros(n_hosts)
        
        for i_host in range(n_hosts):
            print(suite, i_host)
            sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)

            rs, hist = symlib.read_rockstar(sim_dir)
            sf, _ = symlib.read_symfind(sim_dir)
            um = symlib.read_um(sim_dir)

            stars, gal_hist = cache_stars.read_stars(
                "fid_dwarf", suite, i_host)

            ms_tot = 0.0
            for i in range(1, len(stars)):
                ms_tot += np.sum(stars[i]["mp"])
                
            mvir_i[i_host] = rs["m"][0,-1]
            ms_cen_i[i_host] = um["m_star"][0,-1]

            try:
                gal, _ = symlib.read_galaxies(sim_dir, "fid_dwarf")
                ms_sat_i[i_host] = np.sum(gal["m_star"][1:,-1])
            except:
                ms_sat_i[i_host] = np.sum(gal_hist["m_star_i"][sf[:,-1]["ok"]])
                
            ms_halo_i[i_host] = ms_tot - ms_sat_i[i_host]
            
            for i in range(3):
                mvir[i_suite,i] = np.quantile(mvir_i, p[i])
                ms_cen[i_suite,i] = np.quantile(ms_cen_i, p[i])
                ms_sat[i_suite,i] = np.quantile(ms_sat_i, p[i])
                ms_halo[i_suite,i] = np.quantile(ms_halo_i, p[i])
                
    fig, ax = plt.subplots()
        
    ax.plot(mvir[:,1], ms_cen[:,1], c=pc("r"),
            label=r"$M_X = M_{\rm\star, cen}$")
    ax.plot(mvir[:,1], ms_cen[:,1], "o", c=pc("r"))
    ax.fill_between(mvir[:,1], ms_cen[:,0], ms_cen[:,2],
                        color=pc("r"), alpha=0.2)

    ax.plot(mvir[:,1], ms_sat[:,1], c=pc("o"),
            label=r"$M_X = M_{\rm\star, sat}$")
    ax.plot(mvir[:,1], ms_sat[:,1], "o", c=pc("o"))
    ax.fill_between(mvir[:,1], ms_sat[:,0], ms_sat[:,2],
                    color=pc("o"), alpha=0.2)

    ax.plot(mvir[:,1], ms_halo[:,1], c=pc("b"),
            label=r"$M_X = M_{\rm\star, halo}$")
    ax.plot(mvir[:,1], ms_halo[:,1], "o", c=pc("b"))
    ax.fill_between(mvir[:,1], ms_halo[:,0], ms_halo[:,2],
                    color=pc("b"), alpha=0.2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylabel(r"$M_X\ (M_\odot)$")
    ax.set_xlabel(r"$M_{\rm vir}\ (M_\odot)$")
    ax.legend(loc="upper left")
    fig.savefig("plots/mvir_mstar.pdf")
        
            
if __name__ == "__main__": main()
