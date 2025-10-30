import numpy as np
import matplotlib.pyplot as plt
import palette
import symlib
from palette import pc
import lib
import scipy.stats as stats
import errani_model as em
import cache_stars

palette.configure(True)

res_bins = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
colors = [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]
bin_names = [
    r"$3\times 10^2 < n < 1\times10^3$",
    r"$1\times 10^2 < n < 3\times10^3$",
    r"$3\times 10^3 < n < 1\times10^4$",
    r"$1\times 10^2 < n < 3\times10^4$",
    r"$3\times 10^4 < n < 1\times10^5$"
]

def main():
    model_names = ["fid_dwarf", "r=0.0038", "r=0.0060", "r=0.0095",
                   "r=0.015", "r=0.024", "r=0.038", "r=0.060", "r=0.095"
                   "r=0.15", "r=1"]
    model_names_no_um = [name + "_no_um" for name in model_names]

    model_names = ["r=0.0095", "r=0.095"]

    for name in model_names:
        plot_model(name, name == model_names[0], False)

def errani_tidal_track(r_half_rvir, model_name):
    z = 0.0
    a = 1/(z + 1)
    mvir = 1e11
    cvir = em.c_model(mvir, z)
    rvir = em.mvir_to_rvir(mvir*0.7, a, em.param["Om0"])/0.7
    rvir_rmx = cvir/em.rmx_rs
    
    r_half_rmx = r_half_rvir * rvir_rmx

    eps_dm, dN_deps_dm = em.eddington_inversion(
        em.rho_rho0, em.phi_vmx2,
        r_apo_max=rvir_rmx*em.r_apo_max_rvir)
    
    n_t = 50
    
    if model_name == "plummer":
        rho_gal = em.create_plummer_rho(r_half_rmx)
    else:
        assert(0)

    eps_gal, dN_deps_gal = em.eddington_inversion(
        rho_gal, em.phi_vmx2,
        r_apo_max=rvir_rmx*em.r_apo_max_rvir)

    n_t = 50
    eps_ti = 10**np.linspace(-2, 0, n_t)
    mf_dm, mf_gal = np.zeros(n_t), np.zeros(n_t)
    
    for i in range(len(eps_ti)):
        dN_deps_gal_t = em.truncate_eps(eps_gal, dN_deps_gal, eps_ti[i])
        dN_deps_dm_t = em.truncate_eps(eps_dm, dN_deps_dm, eps_ti[i])

        mf_gal[i] = em.eps_mass(eps_gal, dN_deps_gal_t)
        mf_dm[i] = em.eps_mass(eps_dm, dN_deps_dm_t)
        #mf_dm[i] /= em.mvir_mtot(cvir)

    print("mvir/mtot =", em.mvir_mtot(cvir))
        
    return mf_dm, mf_gal

def plot_model(model_name, include_labels, include_relax):
    print(model_name, include_labels, include_relax)
    #suite = "SymphonyMilkyWayHR"
    suite = "SymphonyMilkyWay"
    counts = np.zeros(len(res_bins) - 1, dtype=int)
    
    fig, ax = plt.subplots()

    f_dm = [[] for _ in range(len(res_bins))]
    f_star = [[] for _ in range(len(res_bins))]
    f_star_relax = [[] for _ in range(len(res_bins))]
    f_star_relax2 = [[] for _ in range(len(res_bins))]
    f_star_relax3 = [[] for _ in range(len(res_bins))]
    
    ninfalls, npeaks = [], []
    
    for i_host in range(symlib.n_hosts(suite)):

        print(suite, i_host, symlib.n_hosts(suite))
        sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)
            
        rs, hist = symlib.read_rockstar(sim_dir)
        sf, hist = symlib.read_symfind(sim_dir)
        gal, gal_hist = symlib.read_galaxies(sim_dir, model_name)

        param = symlib.simulation_parameters(suite)
        mp = param["mp"]/param["h100"]

        scale = symlib.scale_factors(sim_dir)
        eps = param["eps"]/param["h100"]*scale
        npeak = hist["mpeak_pre"] / mp
        
        ok = hist["merger_ratio"] < 0.15
        counts += np.histogram(npeak[ok], bins=res_bins)[0]

        # We just need the ranks, we don't need anything else from the tags
        if include_relax:
            _, _, ranks = symlib.tag_stars(
                sim_dir, symlib.DWARF_GALAXY_HALO_MODEL_NO_UM)
            stars, _ = cache_stars.read_stars(model_name, suite, i_host)

        snaps = np.arange(len(scale), dtype=int)

        part = symlib.Particles(sim_dir)
        for i in range(1,len(gal)):
            if include_relax and ranks[i].x is None: continue
            
            snap = hist["first_infall_snap"][i]
            fm = sf["m"][i] / sf["m"][i,snap]
            if gal_hist["m_star_i"][i] != 0:
                fs = gal["m_star"][i] / gal_hist["m_star_i"][i]
            else:
                fs = np.zeros(len(fm))
                        
            if include_relax:
                mrelax = np.zeros(len(fs))
                mrelax2 = np.zeros(len(fs))
                mrelax3 = np.zeros(len(fs))
                
                dt = lib.t - lib.t[snap]
                t_relax = ranks[i].relaxation_time(mp, eps[snap])
                #p = part.read(snap, mode="smooth", halo=i)
                #ok = p["ok"]
                idx = ranks[i].idx
                for s in range(snap, len(scale)):
                    mrelax[s] = np.sum(stars[i]["mp"][idx][t_relax < dt[s]])
                    mrelax2[s] = np.sum(stars[i]["mp"][idx][t_relax/3 < dt[s]])
                    mrelax3[s] = np.sum(stars[i]["mp"][idx][t_relax/10 < dt[s]])

                mstar_tot = np.sum(stars[i]["mp"])
                if mstar_tot != 0: 
                    mrelax /= mstar_tot
                    mrelax2 /= mstar_tot
                    mrelax3 /= mstar_tot
                    
            ok = snap <= snaps
            for j in range(len(res_bins)-1):
                if res_bins[j] < npeak[i] < res_bins[j+1]:
                    f_dm[j].append(fm[ok])
                    f_star[j].append(fs[ok])

                    if include_relax:
                        f_star_relax[j].append(mrelax[ok])
                        f_star_relax2[j].append(mrelax2[ok])
                        f_star_relax3[j].append(mrelax3[ok])
                    
    print("Bin counts:")
    print(counts)

    bin_edges = 10**np.linspace(-3, 0, 21)
    bin_centers = np.sqrt(bin_edges[1:]*bin_edges[:-1])

    meds = np.zeros((len(bin_centers), len(res_bins)-1))
    
    for j in range(len(res_bins) - 1):
        med, _, _ = stats.binned_statistic(
            np.hstack(f_dm[j]), np.hstack(f_star[j]),
            "median", bins=bin_edges)

        meds[:,j] = med

        if include_relax:
            med_relax, _, _ = stats.binned_statistic(
                np.hstack(f_dm[j]), np.hstack(f_star_relax[j]),
                "median", bins=bin_edges)
            med_relax2, _, _ = stats.binned_statistic(
                np.hstack(f_dm[j]), np.hstack(f_star_relax2[j]),
                "median", bins=bin_edges)
            med_relax3, _, _ = stats.binned_statistic(
                np.hstack(f_dm[j]), np.hstack(f_star_relax3[j]),
                "median", bins=bin_edges)
            ok = med > med_relax1
            
            ax.plot(bin_centers, med, "--", c=colors[j], lw=1.5)
            ax.plot(bin_centers[ok], med[ok], colors[j],
                    label=bin_names[j])
        else:
            ax.plot(bin_centers, med, c=colors[j],
                    label=bin_names[j])
        
    ax.plot([], [], "--", c="k", label=r"${\rm Errani+22}$")
        
    if include_labels:
        ax.legend(loc="lower right", fontsize=17)
        
    if "=" in model_name:
        i0 = model_name.index("=")
        r_half_rvir_2d = float(model_name[i0+1:])
        r_half_rvir = r_half_rvir_2d#/1.3**2 #* 1/np.sqrt(2**(2/3) - 1)
        fm_dm, fm_gal = errani_tidal_track(r_half_rvir, "plummer")
        ax.plot(fm_dm, fm_gal, "--", c="k")
        
        with open("data/errani_%s.txt" % model_name, "w+") as f:
            print("# 0 - f_dm", file=f)
            print("# 1 - f_*", file=f)
            for i in range(len(fm_dm)):
                print("%.5g %.5g" % (fm_dm[i], fm_gal[i]), file=f)

        
    with open("data/sim_%s.txt" % model_name, "w+") as f:
        print("# 0 - f_dm", file=f)
        for i in range(len(res_bins) - 1):
            print("# %d - f_* (%.2g < n_peak < %.2g)" %
                  (i, res_bins[i], res_bins[i+1]), file=f)

        for i in range(len(meds)):
            print("%8.5f" % bin_centers[i], file=f, end=" ")
            for j in range(len(res_bins) - 1):
                print("%8.5f" % meds[i,j], file=f, end=" ")
                
            print("", file=f)
                                
    ax.set_xlim(1e-3, 1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_{\rm dm}/m_{\rm dm,infall}$")
    ax.set_ylabel(r"$m_{\star}/m_{\rm \star,infall}$")

    if model_name == "r=0.015":
        ax.legend(fontsize=16, loc="lower right")
    
    fig.savefig("plots/basic_tidal_tracks_%s.png" % model_name)
    
        
if __name__ == "__main__": main()
