import symlib
import lib
import matplotlib.pyplot as plt
import palette
from palette import pc
import numpy as np
import scipy.interpolate as interpolate
palette.configure(True)

def start_end(t, t_dyn, h, hist, mult=1.0):
    if np.argmax(h["m"]) >= hist["first_infall_snap"]: return -1, -1, False
    start = lib.pre_snap_max(h, hist)
    dt_T = (t - t[start])/t_dyn[start]
    start = np.searchsorted(dt_T, 0)
    end = np.searchsorted(dt_T, mult)
    if end >= len(t) or end >= hist["first_infall_snap"] or dt_T[end] < mult or h["m"][end]/h["m"][start] < 0.5 or hist["mpeak"]/lib.mp < 1e4 or hist["mpeak"]/lib.mp > 1e6:
        return -1, -1, False
    return start, end, True

R_HALF_RVIR = 0.1

def make_gal_halo_model(profile_shape_model):
    return symlib.GalaxyHaloModel(
        symlib.StellarMassModel(
            symlib.UniverseMachineMStarFit(),
            symlib.DarkMatterSFH()
        ),
        symlib.ProfileModel(
            symlib.FixedRHalf(R_HALF_RVIR),
            symlib.EinastoProfile(profile_shape_model)
        ),
        symlib.MetalModel(
            symlib.Kirby2013Metallicity(),
            symlib.Kirby2013MDF(model_type="leaky box"),
            symlib.FlatFeHProfile(),
            symlib.GaussianCoupalaCorrelation()
        )
    )

def t_relax_at_r_half(mp, eps, ranks, r_half):
    t_relax = ranks.ranked_relaxation_time(mp, eps)
    r50 = ranks.ranked_halfmass_radius()
    f = interpolate.interp1d(r50, t_relax,
                             fill_value=(t_relax[0], t_relax[-1]),
                             bounds_error=False)
    return f(r_half)

def cumulative_profile(mp, p, core_x, rvir0):
    ok = p["ok"]
    m_star = np.sum(mp[ok])
    dx = p["x"] - core_x
    r = np.sqrt(np.sum(dx**2, axis=1))/rvir0
    
    r_bins = np.linspace(0, 0.2, 200)
    mass, _ = np.histogram(r[ok], bins=r_bins, weights=mp[ok])
    outer_mass = np.sum(mp[ok & (r > r_bins[-1])])
    mass = (np.cumsum(mass[::-1])[::-1] + outer_mass)/m_star

    return r_bins[1:], mass

def main():
    alphas = [0.125, 0.25, 0.5, 1, 2]
    n_alpha = len(alphas)
    profile_shape_models = [symlib.EinastoProfile(alpha) for alpha in alphas]
    
    fig, ax = plt.subplots()
    colors = [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]
    profs = [[] for _ in range(n_alpha)]
    
    for i_host in range(symlib.n_hosts(lib.suite)):
        if i_host > 5: continue
        sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, i_host)
        h, hist = symlib.read_rockstar(sim_dir)

        start = -1*np.ones(len(h), dtype=int)
        target = []
            
        for i in range(1, len(h)):
            start_i, _, ok = start_end(
                lib.t, lib.t_dyn, h[i], hist[i], 1.0)
            if ok:
                start[i] = start_i
                target.append(i)

        target = np.array(target, dtype=int)

        if len(target) == 0: continue
        
        for i_alpha in range(n_alpha):
            gal_halo = make_gal_halo_model(alphas[i_alpha])
            stars, gal, ranks = symlib.tag_stars(
                sim_dir, gal_halo, target_subs=target, star_snap=start)
        
            part = symlib.Particles(sim_dir)
            for j, i in enumerate(target):
                p = part.read(start[i], halo=i, mode="stars")
                core_x = h["x"][i,start[i]]
                core_v = h["v"][i,start[i]]
                rvir = h["rvir"][i,start[i]]
            
                rvir0 = h["rvir"][i,start[i]]
                r, mass = cumulative_profile(stars[i]["mp"], p, core_x, rvir0)
                profs[i_alpha].append(mass)
    
    ax.set_xlabel(r"$r/R_{\rm vir}$")
    ax.set_ylabel(r"$M_\star(>r)/M_\star$")

    for i in range(n_alpha):
        prof = symlib.EinastoProfile(alphas[i])
        m_target = prof.m_enc(1.0, 1.0, r)
        ax.plot(r/R_HALF_RVIR, np.median(profs[i], axis=0), c=colors[i])
        ax.fill_between(r/R_HALF_RVIR, np.quantile(profs[i], 0.5+0.68/2, axis=0),
                        np.quantile(profs[i], 0.5-0.68/2, axis=0),
                        color=colors[i], alpha=0.2)
        
        ax.plot(r, 1-m_target, "--", c="k")
    ax.legend(loc="lower left", fontsize=17)
    ax.set_ylim(0, 1)
    
    ax.set_xlabel(r"$r/R_{50}$")
    ax.set_ylabel(r"$M_\star(>r)/M_\star$")

    fig.savefig("plots/fit_accuracy.png")
    
if __name__ == "__main__": main()
