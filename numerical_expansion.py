import symlib
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, mass_so
import palette
from palette import pc
import matplotlib.pyplot as plt
import scipy.stats as stats

palette.configure(True)

suite = "SymphonyMilkyWay"

param = symlib.simulation_parameters(suite)
mp = param["mp"]/param["h100"]
cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))
base_dir = "/sdf/home/p/phil1/ZoomIns"

def running_mpeak(m):
    out = np.zeros(m.shape)
    for i in range(1, len(m)):
        out[i] = max(out[i-1], m[i])
    return out

def pseudo_evolve_limit(scale, halo, snap_f):
    z = 1/scale - 1    
    ratio_limit = np.zeros(halo.shape)

    m = running_mpeak(halo["m"])

    for snap in range(snap_f):
        if not halo["ok"][snap]: continue
        Mf_lim, _, _ = mass_defs.pseudoEvolve(
            m[snap], halo["cvir"][snap], "vir", z[snap], z[snap_f])
        ratio_limit[snap] = m[snap]/Mf_lim

    #print(ratio_limit)
    return ratio_limit

def compare_pseudo_evolve():
    out_dir = "plots/"
    sim_dir = symlib.get_host_directory(base_dir, suite, 0)
    h, hist = symlib.read_rockstar(sim_dir)
    scale = symlib.scale_factors(sim_dir)
    
    colors = [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]
    for i in range(1, 6):
        snap_f = np.argmax(h["m"][i,:hist["first_infall_snap"][i]])
        snap_f2 = hist["first_infall_snap"][i]
        Mf = h["m"][i,snap_f]
        plt.plot(scale[:snap_f2+1], h["m"][i,:snap_f2+1]/Mf,
                 c=colors[(i-1) % len(colors)])
        lim = pseudo_evolve_limit(scale, h[i], snap_f)
        ok = lim > 0
        plt.plot(scale[ok], lim[ok], "--", c=colors[(i-1)%len(colors)], lw=2)
    
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel(r"$a(t)$")
    plt.ylabel(r"$M/M_{\rm final}$")
    
    plt.savefig("%s/pseudo_evolve_commpare.png" % out_dir)

def pre_snap_max(h, hist):
    return np.argmax(h["m"][:hist["first_infall_snap"]])

def peak_infall_gap():
    out_dir = "plots/"
    base_dir = "/sdf/home/p/phil1/ZoomIns"

    m_ratios = []

    plt.figure()
    for suite in ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup",
                  "SymphonyLCluster"]:

        param = symlib.simulation_parameters(suite)
        mp = param["mp"]/param["h100"]
        cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))
        scale = symlib.scale_factors(suite)
        z = 1/scale - 1
        t = cosmo.age(z)
        t_dyn = mass_so.dynamicalTime(z, "vir", "crossing")    

        n_ok = 0
        for i_host in range(symlib.n_hosts(suite)):
            sim_dir = symlib.get_host_directory(base_dir, suite, i_host)
            h, hist = symlib.read_rockstar(sim_dir)

            snap_max = np.zeros(hist.shape, dtype=int)
            for i in range(len(snap_max)):
                if hist["first_infall_snap"][i] == 0:
                    snap_max[i] = 0
                    continue
                snap_max[i] = pre_snap_max(h[i], hist[i])

            dt = t[hist["first_infall_snap"]] - t[snap_max]
            dt_T = dt/t_dyn[snap_max]

            mf_mi = h["m"][:,hist["first_infall_snap"][i]]/hist["mpeak_pre"]
            ok1 = (hist["mpeak"]/mp > 1e4) & (dt_T > 1) & (hist["mpeak"]/mp < 1e6)
            ok2 = mf_mi > 0.5
            plt.plot(dt_T[ok1 & ok2], hist["mpeak"][ok1 & ok2]/mp,
                 ".", c=pc("r"))
            plt.plot(dt_T[ok1 & (~ok2)], hist["mpeak"][ok1 & (~ok2)]/mp,
                     "x", c=pc("r"), alpha=0.2)
            plt.plot(dt_T[(~ok1) & ok2], hist["mpeak"][(~ok1) & ok2]/mp,
                     ".", c=pc("k"), alpha=0.2)
            plt.plot(dt_T[(~ok1) & (~ok2)], hist["mpeak"][(~ok1) & (~ok2)]/mp,
                     "x", c=pc("k"), alpha=0.2)
            n_ok += np.sum(ok1 & ok2)
            print(i_host, np.where(ok1&ok2)[0])
        print(suite, n_ok)

    #m_ratios = np.hstack(m_ratios)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$n_{\rm peak}$")
    plt.xlabel(r"$\Delta t / T_{\rm dyn}$")

    plt.savefig("%s/peak_infall_gap.png" % out_dir)

    #plt.figure()

    #plt.hist(m_ratios, histtype="step", color=pc("r"), lw=2)
    #plt.xlabel(r"$m_{\rm peak}/m_{\rm infall}$")
    #plt.savefig("%s/peak_infall_mass_gap.png" % out_dir)

def is_slow_grower(h, hist, t, t_dyn, mp):
    snap_max = np.zeros(hist.shape, dtype=int)
    for i in range(len(snap_max)):
        if hist["first_infall_snap"][i] == 0:
            snap_max[i] = 0
            continue
        snap_max[i] = pre_snap_max(h[i], hist[i])

    dt = t[hist["first_infall_snap"]] - t[snap_max]
    dt_T = dt/t_dyn[snap_max]

    mf_mi = h["m"][:,hist["first_infall_snap"][i]]/hist["mpeak_pre"]
    ok1 = (hist["mpeak"]/mp > 1e4) & (dt_T > 1) & (hist["mpeak"]/mp < 1e6)
    ok2 = mf_mi > 0.5
    ok1[0] = False

    return ok1 & ok2

def expansion_examples():
    out_dir = "plots/"
    suite = "SymphonyGroup"
    i_host = 2

    param = symlib.simulation_parameters(suite)
    cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))
    scale = symlib.scale_factors(suite)
    z = 1/scale - 1
    T = cosmo.age(z)

    mp = param["mp"]/param["h100"]
    eps = param["eps"]*scale

    z = 1/scale - 1
    t = cosmo.age(z)
    t_dyn = mass_so.dynamicalTime(z, "vir", "crossing")    

    colors = [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]
    
    ts, ratios = [[] for _ in range(5)], [[] for _ in range(5)]

    plt.figure()

    for i_host in range(symlib.n_hosts(suite)):
        if i_host > 0: continue

        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)
        h, hist = symlib.read_rockstar(sim_dir)
        is_slow = is_slow_grower(h, hist, t, t_dyn, mp)
        i_subs = np.where(is_slow)[0]
    
        part = symlib.Particles(sim_dir)

        for i_sub in i_subs:
            plt.cla()

            snap_i = pre_snap_max(h[i_sub], hist[i_sub])
            snap_f = hist["first_infall_snap"][i_sub]
            print(i_host, i_sub, snap_i, snap_f)

            mode = "all"
            #mode = "smooth"
            p = part.read(snap_i, i_sub, mode=mode)
        
            ok = p["ok"]
            x = p["x"] - h[i_sub,snap_i]["x"]
            v = p["v"] - h[i_sub,snap_i]["v"]
            idx = np.where(ok)[0]
            rvir = h["rvir"][i_sub, snap_i]

            print(len(ok), np.sum(ok))
            
            ranks = symlib.RadialEnergyRanking(
                param, x[ok], v[ok], idx, len(x), rvir)
            ranks.load_particles(x[ok], None, idx)
            
            n_snap = snap_f-snap_i + 1
            r50 = np.zeros(((np.max(ranks.ranks)+1), n_snap))

            T_orbit = mass_so.dynamicalTime(z, "vir", "orbit")
            dt = T - T[snap_i]
            dt_T_orbit = (T - T[snap_i]) / T_orbit

            t_relax = ranks.ranked_relaxation_time(mp, eps[snap_i])
        
            r50[:,0] = ranks.ranked_halfmass_radius()
        
            for snap in range(snap_i+1, snap_f+1):
                p = part.read(snap, i_sub, mode=mode)
                x_core = np.median(p["x"][ranks.core_idx], axis=0)
                
                ok = p["ok"]
                x = p["x"][ok] - x_core
                idx = np.arange(len(ok), dtype=int)[ok]
            
                ranks.load_particles(x, None, idx)
                #print(ranks.ranked_halfmass_radius())
                r50[:,snap - snap_i] = ranks.ranked_halfmass_radius()

            snap_0 = snap_i+1
        
            for i in range(len(colors)):
                _dt = (dt[snap_0:snap_f+1] - dt[snap_0])/t_relax[i]
                _r50 = r50[i, snap_0-snap_i:]
                plt.plot(_dt[1:], _r50[1:]/_r50[0], "o", c=colors[i])
                plt.plot(_dt[1:], _r50[1:]/_r50[0], c=colors[i])

                ts[i].append(_dt[1:])
                ratios[i].append(_r50[1:]/_r50[0])


    ts = [np.hstack(ts[i]) for i in range(5)]
    ratios = [np.hstack(ratios[i]) for i in range(5)]

    #for i in range(5):
    #    print(i)
    #    print("ts    ", ts[i])
    #    print("ratios", ratios[i])
    #    print()
    
    plt.figure()
    for i in range(5):
        plt.plot(ts[i], ratios[i], ".", c=colors[i], alpha=0.2)
        
    t, ratio = np.hstack(ts), np.hstack(ratios)

    t_bins = 10**np.linspace(np.log10(3e-3), np.log10(10), 20)
    t_mid = np.sqrt(t_bins[1:]*t_bins[:-1])

    ratio_mid, _, _ = stats.binned_statistic(t, ratio, "median", bins=t_bins)
    plt.plot(t_mid, ratio_mid, c="k")
    print("t_mid", t_mid)
    print("ratio_mid", ratio_mid)

    plt.xscale("log")
    xlo, xhi = plt.xlim()
    plt.ylim(0.5, 3.5)
    ylo, yhi = plt.ylim()
    plt.xlim(xlo, xhi)
    plt.plot([xlo, xhi], [1, 1], "--", c="k", lw=1.5)
    plt.fill_between([0.3, xhi], [ylo, ylo], [yhi, yhi], alpha=0.2, color="k")

    plt.xlabel(r"$\Delta t/T_{relax}$")
    plt.ylabel(r"$r_{50}/r_{\rm 50}(t_{\rm tag})$")
    
    plt.savefig("%s/expansion.png" % (out_dir,))

def main():
    #compare_pseudo_evolve()
    #peak_infall_gap()
    expansion_examples()
    
if __name__ == "__main__": main()
