import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import symlib
import lib
import scipy.stats as stats
import scipy.optimize as optimize
import numpy.random as random

palette.configure(False)

def radial_distr(mp, Fe_H, dx, r_half):
    r = np.sqrt(np.sum(dx**2, axis=1))
    
    fig, ax = plt.subplots()
    ax.plot(r/r_half, Fe_H, ".", alpha=0.1, c=pc("k"))

    idx = np.arange(len(r), dtype=int)
    def weighted_mean(idx):
        return np.sum(Fe_H[idx]*mp[idx])/np.sum(mp[idx])
    
    bins = np.linspace(0, 10, 21)
    means, _, _ = stats.binned_statistic(r, idx, weighted_mean, bins=bins)
    mids = (bins[1:] + bins[:-1]) / 2

    ax.plot(mids, means, c=pc("r"))
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"$r/r_{50}$")
    ax.set_ylabel(r"$[{\rm Fe/H}]$")

    return fig, ax

def weighted_std(x, w):
    avg = np.average(x, weights=w)
    var = np.average((x - avg)**2, weights=w)
    return np.sqrt(var)

def fit_Fe_H_2(f_Fe_H, Fe_H, dx, ok, mp, ranks, r_half):
    r_bins = np.linspace(0, 10, 21)
    r_mid = (r_bins[1:] + r_bins[:-1]) / 2
    Fe_H_target = f_Fe_H(r_mid)
        
    r = np.sqrt(np.sum(dx**2, axis=1))/r_half
    
    m_star, _ = np.histogram(r, bins=r_bins, weights=mp)

    M = np.zeros((len(r_bins)-1, ranks.n_max+1))
    
    for i in range(ranks.n_max+1):
        oki = (ranks.ranks == i) & ok
        ri = r[oki]
        mpi = mp[oki]
        N, _ = np.histogram(ri, bins=r_bins, weights=mpi)
        M[:,i] = N

    res = optimize.lsq_linear(M, Fe_H_target*m_star,
                              bounds=(-5, 2))

    Fe_H_2 = np.zeros(len(r))
    for i in range(ranks.n_max+1):
        Fe_H_2[ranks.ranks == i] = res.x[i]

    delta = np.mean(Fe_H) - np.sum(mp*Fe_H_2)/np.sum(mp)
    sigma = 0.1

    ok = mp > 1e1
    n_ok = np.sum(ok)
        
    fig, ax = plt.subplots()
    ax.plot(r, Fe_H_2, ".", c=pc("r"), alpha=0.1)
    ax.plot(r, Fe_H_2 + random.randn(len(mp))*sigma, ".", c=pc("b"), alpha=0.1)
    rr = np.linspace(0, 10, 100)
    ax.plot(rr, f_Fe_H(rr), c=pc("k"))
    ax.set_xlim(0, 10)
    fig.savefig("plots/radial_Fe_H_1.png")
    
    Fe_H_2 += np.mean(Fe_H) - np.sum(mp*Fe_H_2)/np.sum(mp)
    Fe_H_3 = Fe_H_2 + random.randn(len(mp))*sigma
    
    fig, ax = plt.subplots()
    ax.hist(Fe_H, color="k", lw=2, histtype="step",
                 weights=mp, density=True, bins=40)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.hist(Fe_H_2, color=pc("r"), lw=2, histtype="step",
            weights=mp, density=True, bins=40)
    ax.hist(Fe_H_3, color=pc("b"), lw=2, histtype="step",
            weights=mp, density=True, bins=40)
    fig.savefig("plots/Fe_H_hist.png")
    
    
    """
    res = optimize.lsq_linear(
        M, dm_star_enc_target, bounds=(np.zeros(n), np.inf*np.ones(n))
    )
    self.mp_star_table = res.x

    self.mp_star[self.idx] = self.mp_star_table[self.ranks[self.idx]]
    is_nil = self.ranks[self.idx] == NIL_RANK
    self.mp_star[self.idx[is_nil]] = 0
    
    mp_star_tot = np.sum(self.mp_star)
    if mp_star_tot == 0:
        correction_frac = 1.0
    else:
        correction_frac = m_star/np.sum(self.mp_star)
        self.mp_star *= correction_frac
        self.mp_star_table *= correction_frac
        
    return self.mp_star
    """

def fit_Fe_H_3(f_Fe_H, Fe_H, dx, dv, ok, mp, r_half):
    r = np.sqrt(np.sum(dx**2, axis=1))/r_half
    Fe_H_3 = f_Fe_H(r)
    noise = random.randn(len(Fe_H))*np.std(Fe_H)/5
    Fe_H_4 = f_Fe_H(r) + noise
    
    r_bins = np.linspace(0, 5, 21)[1:]
    r_lim = r_bins[0]
    target_sigma = np.std(Fe_H)
    for i in range(len(r_bins)):
        oki = (r < r_bins[i]) & ok
        Fe_H_5 = np.copy(Fe_H_4)
        Fe_H_5[~oki] = f_Fe_H(r_bins[i]) + noise[~oki]
        if weighted_std(Fe_H_4[oki], mp[oki]) > target_sigma: break
        r_lim = r_bins[i]

        
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    ax.hist(Fe_H[ok], bins=40, histtype="step", weights=mp[ok],
            color=pc("k"), lw=2, density=True,
            label=r"${\rm target}$")
    
    Fe_H_3[r > r_lim] = f_Fe_H(r_lim)
    ax2.plot(r[ok], Fe_H_3[ok], ".", alpha=0.2, c=pc("r"))
    ax.hist(Fe_H_3[ok], bins=40, histtype="step", color=pc("r"), lw=2,
            weights=mp[ok], density=True, label=r"${\rm step\ 1}$")
    Fe_H_3 += np.mean(Fe_H[ok]) - np.average(Fe_H_3[ok], weights=mp[ok])
    ax.hist(Fe_H_3[ok], bins=40, histtype="step", color=pc("o"), lw=2,
            weights=mp[ok], density=True, label=r"${\rm step\ 2}$")
    
    sigma = np.sqrt(target_sigma**2 - weighted_std(Fe_H_3[ok], mp[ok])**2)
    Fe_H_3 += random.randn(len(Fe_H_3))*sigma
    
    ax.hist(Fe_H_3[ok], bins=40, histtype="step", color=pc("b"), lw=2,
            weights=mp[ok], density=True, label=r"${\rm step\ 3}$")
    Fe_H_3[ok] = weighted_abundance_match(Fe_H[ok], Fe_H_3[ok], mp[ok])
    
    ax.hist(Fe_H_3[ok], bins=40, histtype="step", color=pc("p"), lw=2,
            weights=mp[ok], density=True, ls="--", label=r"${\rm step\ 1}$")
    ax2.plot(r[ok], Fe_H_3[ok], ".", alpha=0.2, c=pc("p"))
    ax.legend(loc="upper left", fontsize=17)

    ax.set_xlabel(r"${\rm [Fe/H]}$")
    ax.set_ylabel(r"${\rm P([Fe/H])}$")
    
    fig.savefig("plots/Fe_H_hist.png")

    ax2.set_xlim(0, 7)
    ax2.set_xlabel(r"$r/R_e$")
    ax2.set_ylabel(r"${\rm [Fe/H]}$")
    fig2.savefig("plots/Fe_H_prof.png")
    
def weighted_abundance_match(x, y, wy):
    x = np.sort(x)

    order = np.argsort(y)
    orig_idx = np.arange(len(y))[order]
    sy, swy = y[order], wy[order]
    
    sy = np.asarray(sy, dtype=np.float64)
    swy = np.asarray(swy, dtype=np.float64)

    P = np.cumsum(swy) / np.sum(swy)
    f_idx = P*(len(x) - 1)
    
    prev_idx = np.array(np.floor(f_idx), dtype=int)
    frac = f_idx - prev_idx

    # Deal with annoying floating point squishiness
    prev_idx[prev_idx >= len(x)] = len(x) - 1
    prev_idx[prev_idx < 0] = 0
    frac[frac > 1] = 1
    next_idx = prev_idx + 1
    next_idx[next_idx >= len(x)] = len(x) - 1

    dx = x[next_idx] - x[prev_idx]
    zero_dx = dx <= 0
    dx[zero_dx], frac[zero_dx] = 1, 0
    out = np.zeros(len(y))
    out[orig_idx] = (x[prev_idx] + dx*frac)
    return out

def energy_fit(f_Fe_H, Fe_H, mp, param, dx, dv, ok, r_half):
    r = np.sqrt(np.sum(dx**2, axis=1))
    rmax, vmax, PE, order = symlib.profile_info(param, dx, ok=ok)
    
    KE = 0.5 * np.sum(dv**2 / vmax**2, axis=1)
    E = KE + PE

    order = np.argsort(E)
    r_sort = r[order]
    mp_sort = mp[order]
    ok_sort = ok[order]

    """
    fig, ax = plt.subplots()
    sample = random.random(len(r)) < 0.1
    ax.plot(r[sample], E[sample], ".", c="k", alpha=0.2)
    ax.set_xscale("log")

    fig.savefig("plots/radial_energy.png")
    """

    fig, ax = plt.subplots()
    window = 10
    #min_idx = np.maximum(order - window, 0)
    #max_idx = np.minimum(order + window, len(r) - 1)
    idx = np.arange(len(r_sort), dtype=int)
    min_idx = np.maximum(idx - window, 0)
    max_idx = np.minimum(idx + window, len(r_sort)-1)
    idx = random.randint(min_idx, max_idx)
    r_eval = r_sort[idx]

    ax.set_xlim(0, 7)

    Fe_H_0 = f_Fe_H(r_sort)
    
    Fe_H_i = f_Fe_H(r_eval)
    std = np.std(Fe_H[ok])
    ok2 = (~np.isnan(mp_sort)) & (~np.isnan(Fe_H_i))
    w_std = weighted_std(Fe_H_i[ok2], mp_sort[ok2])
    # std and not w_std????
    Fe_H_i += random.randn(len(r))*std/5

    print(std)
    print(w_std)
    
    bins = np.linspace(0, 7, 20)
    mids = (bins[1:]+bins[:-1])/2

    avg = np.average(Fe_H_i[ok2], weights=mp_sort[ok2])
    Fe_H_i = Fe_H_i - avg + np.mean(Fe_H)

    Fe_H_i[ok2] = weighted_abundance_match(
        Fe_H[ok2], Fe_H_i[ok2], mp_sort[ok2])

    # You need a thing here to correct the scatter if it's wrong (it isn't
    # in this case).

    plt.plot(r_sort/r_half, Fe_H_i, ".", c="k", alpha=0.2)
    
    def weighted_mean(idx):
        return np.average(Fe_H_i[idx], weights=mp_sort[idx])
    avg, _, _ = stats.binned_statistic(r_sort[ok2]/r_half,
                                       np.arange(len(r_sort), dtype=int)[ok2],
                                       statistic=weighted_mean, bins=bins)
    plt.plot(mids, avg, c=pc("b"), label=r"${\rm Nimbus}$")
    
    rr = np.linspace(0, 7, 100)
    ok = rr < 5
    plt.plot(rr[ok], f_Fe_H(rr[ok])-0.4, c=pc("r"), ls="--",
             label=r"${\rm target}$")
    plt.legend(loc="upper right", fontsize=17)
    plt.ylabel(r"${\rm [Fe/H]}$")
    plt.xlabel(r"$r/r_{50}$")
    fig.savefig("plots/radial_Fe_H_2.png")
    
def mp_distr(mp):
    lim = 1e-3
    lo, hi = np.min(mp[mp > lim]), np.max(mp)*1.01
    bins = 10**np.linspace(np.log10(lo), np.log10(hi), 100)
    n, _ = np.histogram(mp, bins=bins)
    mids = np.sqrt(bins[1:]*bins[:-1])
    dln_m = np.log10(mids[1]) - np.log10(mids[0])
    w = n*mids
    
    w = (np.cumsum(w) + np.sum(mp[mp < lim]))/np.sum(mp)
    n = (np.cumsum(n) + np.sum(mp < lim))/len(mp)
    
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(bins[1:], w, c=pc("b"), label=r"$m_\star-{\rm weighted}$")
    ax.plot(bins[1:], n, c=pc("r"), label=r"$m_{\rm dm}-{\rm weighted}$")
    ax.legend(loc="lower right", fontsize=17)
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_{p\star}$")
    ax.set_ylabel(r"$f(<m_{p\star})$")
    ax.set_ylim(1e-4, 1.1)
    
    return fig, ax
    
    
def old_main():
    palette.configure(False)
    random.seed(1)
    
    sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, 0)
    param = symlib.simulation_parameters(sim_dir)
    rs, hist = symlib.read_rockstar(sim_dir)

    i_sub = 1

    stars, gals, ranks = symlib.tag_stars(
        sim_dir,
        symlib.DWARF_GALAXY_HALO_MODEL,
        target_subs=(i_sub,)
    )
    part = symlib.Particles(sim_dir)
    snap = hist["first_infall_snap"][i_sub]
    p = part.read(snap, mode="stars")
    
    r_half = gals["r_half_2d_i"][i_sub]
    mp, Fe_H = stars[i_sub]["mp"], stars[i_sub]["Fe_H"]
    mp[mp < 1e-1] = 0
    a_form = stars[i_sub]["a_form"]
    dx, ok = p[i_sub]["x"] - rs["x"][i_sub,snap], p[i_sub]["ok"]
    dv = p[i_sub]["v"] - rs["v"][i_sub,snap]

    fig, ax = mp_distr(mp)
    fig.savefig("plots/mp_distr_%02d_%03d.png" % (0, i_sub))

    fig, ax = radial_distr(mp, Fe_H, dx, r_half)
    fig.savefig("plots/radial_Fe_H/prof_%02d_%03d.png" % (0, i_sub))

    f_Fe_H = lambda x: -0.25*x
    def f_Fe_H(r):
        large = r > 6
        out = - 0.25*r
        out[large] = -1.5
        return out 
    
    #Fe_H_2 = fit_Fe_H_2(f_Fe_H, Fe_H, dx, ok, mp, ranks[i_sub], r_half)
    #Fe_H_3 = fit_Fe_H_3(f_Fe_H, Fe_H, dx, dv, ok, mp, r_half)
    energy_fit(f_Fe_H, Fe_H, mp, param, dx, dv, ok, r_half)
    #energy_fit_2(f_Fe_H, mp, param, dx, dv, ok, r_half)

def integration_test():
    gal_halo = symlib.DWARF_GALAXY_HALO_MODEL
    sim_dir = symlib.get_host_directory(
        "/sdf/home/p/phil1/ZoomIns", "SymphonyMilkyWay", 0)

    sf, hist = symlib.read_symfind(sim_dir)
    rs, hist = symlib.read_rockstar(sim_dir)
    sub = 1

    snap = np.arange(sf.shape[1], dtype=int)[sf["ok"][sub]]
    snap = snap[snap >= hist["first_infall_snap"][sub]]
    
    i1, i4 = np.min(snap), np.max(snap)
    di = i4 - i1
    i2, i3 = int((1/3)*di + i1), int((2/3)*di + i1)
    
    snaps = [i1, i2, i3, i4]
    fig, ax = plt.subplots(1, 4, figsize=(28,7))
    
    stars, gal_hists, ranks = symlib.tag_stars(
        sim_dir, gal_halo, target_subs=(sub,), )
    stars, gal_hist = stars[sub], gal_hists[sub]

    part = symlib.Particles(sim_dir)

    rr = np.linspace(0, 8, 100)
    def f_Fe_H(rr):
        Fe_H_lim = 6*np.abs(gal_hist["delta_Fe_H_i"])
        return np.maximum(np.minimum(
            gal_hist["delta_Fe_H_i"]*rr, Fe_H_lim), -Fe_H_lim)

    bins = np.linspace(0, 8, 24)
    for i, snap in enumerate(snaps):
        if i != 0: continue
        p = part.read(snap, mode="smooth", halo=sub)
        p2 = part.read(snap, mode="smooth", halo=sub)
        p3 = part.read(snap, mode="smooth", halo=sub)

        print(p["x"][:4] - rs["x"][sub,snap])
        print(p2["x"][:4] - rs["x"][sub,snap])
        print(p3["x"][:4] - rs["x"][sub,snap])
        
        r = (p["x"] - rs["x"][sub,snap])**2
        ok = p["ok"] & (stars["mp"] > 0)
        
        r_scale = r/gal_hist["r_half_2d_i"]
        
        idx = np.arange(np.sum(ok))
        ax[i].plot(r_scale[ok], stars["Fe_H"][ok],
                   ".", alpha=0.1, c="k")
        ax[i].plot(rr, f_Fe_H(rr), c=pc("r"))

        #stats.binned_statistic(
        #    r
        #)
        
        ax[i].set_xlim(0, 8)
        if i == 0:
            fig2, ax2 = plt.subplots()
            ax2.hist(stars["Fe_H"][ok], cumulative=True,
                     weights=stars["mp"][ok], density=True,
                     histtype="step", color="k", bins=300)
            fig2.savefig("plots/nimbus_radial_Fe_H_hist.png")


        
    fig.savefig("plots/nimbus_radial_Fe_H.png")    
    #for i in range(1, 20):
    #    print(i, "%.4g" % hist[i]["mpeak"],  len(snap[sf[i]["ok"]]))
    
def main():
    #old_main()
    integration_test()
    
if __name__ == "__main__": main()
