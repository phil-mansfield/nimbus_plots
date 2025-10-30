import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import numpy.random as random
import lib
import numpy.linalg as linalg
import symlib
import scipy.stats as stats
import gravitree
import cache_stars

palette.configure(True)

"""
def bootstrap_mean(x, q=(0.5-0.68/2, 0.5, 0.50+0.68/2), n_step=1000):
    means = np.zeros((n_step, x.shape[1]))
    idx = np.arange(len(x), dtype=int)
    for i in range(n_step):
        idx_i = random.choice(idx, size=len(x), replace=True)
        sample = x[idx_i]
        means[i] = np.mean(sample, axis=0)

    out = np.zeros((len(q), x.shape[1]))
    for i in range(len(q)):
        out[i] = np.quantile(means, q[i], axis=0)

    return out
"""

def axis_ratio(x, mp):
    M = np.zeros((3, 3))
    for ii in range(3):
        for jj in range(3):
            M[ii,jj] = np.sum(x[:,ii]*x[:,jj]*mp)
    lamb, _ = linalg.eig(M)
    return np.sqrt(np.min(lamb)/np.max(lamb))

def get_profiles(r_bins, suite):
    n_bin = len(r_bins) - 1
    n_halo = symlib.n_hosts(suite)
    
    if suite == "SymphonyLCluster": n_halo = 32 # Only the first 32 work

    #n_halo = 1
    
    rho, Fe_H = np.zeros((n_halo, n_bin)), np.zeros((n_halo, n_bin))
    c_a, f_merger = np.zeros((n_halo, n_bin)), np.zeros((n_halo, n_bin))
    
    for i_halo in range(n_halo):
        print("    %s, Halo %d" % (suite, i_halo))
        
        sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_halo)
        param = symlib.simulation_parameters(sim_dir)
        mp, eps = param["mp"]/param["h100"], param["eps"]/param["h100"]

        part = symlib.Particles(sim_dir, include=["E"])

        sf, hist = symlib.read_symfind(sim_dir)
        rs, _ = symlib.read_rockstar(sim_dir)
        last_snap = len(symlib.scale_factors(sim_dir)) - 1

        #p_star = part.read(last_snap, mode="stars")
        p = part.read(last_snap, mode="all")
        stars, _ = cache_stars.read_stars("fid_dwarf", suite, i_halo)

        print(stars[1]["mp"])
        print(stars[2]["mp"])
        print(stars[3]["mp"])
        print(stars[4]["mp"])
        
        x_ub, mp_ub, is_merger_ub, Fe_H_ub = [], [], [], []
        for i in range(1, len(p)):
            dx, dv = p[i]["x"], p[i]["v"]
            if sf["ok"][i,-1]:
                dx -= sf["x"][i,-1]
                dv -= sf["x"][i,-1]
                ke = np.sum(dv**2, axis=1)/2
                E = p[i]["E"]
                ok_dm = E < 0
                ok = ok_dm[p[i]["smooth"]]
            else:
                ok = np.zeros(np.sum(p[i]["smooth"]), dtype=bool)

                
            #print(len(dx), len(stars[i]), np.sum(p[i]["smooth"], len(ok)))
            x_ub.append(dx[p[i]["smooth"]][~ok])
            mp_ub.append(stars[i]["mp"][~ok])
            Fe_H_ub.append(stars[i]["Fe_H"][~ok])
            
            is_merger = np.zeros(len(ok), dtype=bool)
            if hist[i]["merger_ratio"] > 0.15:
                is_merger[:] = True
            
            is_merger_ub.append(is_merger[~ok])
            
        x_ub, mp_ub = np.concatenate(x_ub, axis=0), np.hstack(mp_ub)
        is_merger_ub, Fe_H_ub = np.hstack(is_merger_ub), np.hstack(Fe_H_ub)

        mstar_tot = np.sum(mp_ub)
        rho_norm = mstar_tot / (4*np.pi/4 * rs["rvir"][0,-1]**3)
        
        r_ub = np.sqrt(np.sum(x_ub**2, axis=1)) / rs[0,-1]["rvir"]
        mass, _, _ = stats.binned_statistic(
            r_ub, mp_ub, bins=r_bins, statistic="sum")
        merger_mass, _, _ = stats.binned_statistic(
            r_ub[is_merger_ub], mp_ub[is_merger_ub],
            bins=r_bins, statistic="sum")
        Fe_H_weight, _, _ = stats.binned_statistic(
            r_ub, mp_ub*Fe_H_ub, statistic="sum", bins=r_bins)
                    
        vol = 4*np.pi/3 * (r_bins[1:]**3 - r_bins[:-1]**3)

        rho[i_halo] = mass/vol / mstar_tot
        Fe_H[i_halo] = Fe_H_weight / mass
        f_merger[i_halo] = merger_mass / mass

        p_idx = np.arange(len(x_ub), dtype=int)
        c_a[i_halo], _, _ = stats.binned_statistic(
            r_ub, p_idx, bins=r_bins, statistic=
            lambda idx: axis_ratio(x_ub[idx], mp_ub[idx]))
        
    return rho, Fe_H, c_a, f_merger
        
def main():
    suites = ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup",
              "SymphonyLCluster"]
    colors = [pc("r"), pc("o"), pc("b"), pc("p")]

    r_bins = 10**np.linspace(-2, 0, 30)
    r = np.sqrt(r_bins[1:]*r_bins[:-1])
    
    for i_suite in range(len(suites)):

        if i_suite != 2: continue
        
        c = colors[i_suite]
        rho, Fe_H, c_a, f_merger = get_profiles(r_bins, suites[i_suite])
        log_rho = np.log10(rho)
        
        q = [0.5-0.68/2, 0.5, 0.5+0.66/2]
        plt.figure(0)
        rho_l, rho_m, rho_h = np.quantile(rho*r**3, q, axis=0)
        plt.plot(r, rho_m, c=c)
        if i_suite == 1:
            plt.fill_between(r, rho_l, rho_h, alpha=0.2, color=c)
        plt.figure(1)
        Fe_H_l, Fe_H_m, Fe_H_h = np.quantile(Fe_H, q, axis=0)
        plt.plot(r, Fe_H_m, c=c)
        if i_suite == 1:
            plt.fill_between(r, Fe_H_l, Fe_H_h, alpha=0.2, color=c)
        plt.figure(2)
        c_a_l, c_a_m, c_a_h = np.quantile(c_a, q, axis=0)
        plt.plot(r, c_a_m, c=c)
        if i_suite == 1:
            plt.fill_between(r, c_a_l, c_a_h, alpha=0.2, color=c)
        plt.figure(3)
        f_merger_l, f_merger_m, f_merger_h = np.quantile(f_merger, q, axis=0)
        plt.plot(r, f_merger_m, c=c)
        if i_suite == 1:
            plt.fill_between(r, f_merger_l, f_merger_h, alpha=0.2, color=c)
        

    plt.figure(0)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"$\rho_\star \cdot (r/R_{\rm vir})^3 / M_\star$")
    plt.plot([], [], pc("r"), label=r"${\rm SymphonyLMC}$")
    plt.plot([], [], pc("o"), label=r"${\rm SymphonyMilkyWay}$")
    plt.plot([], [], pc("g"), label=r"${\rm SymphonyGroup}$")
    plt.plot([], [], pc("b"), label=r"${\rm SymphonyLCluster}$")
    plt.legend(loc="lower left")
    #plt.xlabel(r"$r/R_{\rm vir}$")
    plt.savefig("plots/fiducial_rho.pdf")
    
    plt.figure(1)
    plt.xscale("log")
    plt.ylabel(r"${\rm \langle[ Fe/H]\rangle}$")
    #plt.xlabel(r"$r/R_{\rm vir}$")
    plt.savefig("plots/fiducial_Fe_H.pdf")

    plt.figure(2)
    plt.xscale("log")
    plt.ylabel(r"$(c/a)_\star$")
    plt.xlabel(r"$r/R_{\rm vir}$")
    plt.savefig("plots/fiducial_c_a.pdf")

    plt.figure(3)
    plt.xscale("log")
    plt.ylabel(r"$f_{\rm merger}$")
    plt.xlabel(r"$r/R_{\rm vir}$")
    plt.savefig("plots/fiducial_f_merger.pdf")
    
if __name__ == "__main__": main()
