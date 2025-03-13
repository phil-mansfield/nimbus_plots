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

r_half_3d = 0.05
gal_halo = symlib.GalaxyHaloModel(
    symlib.StellarMassModel(
        symlib.UniverseMachineMStarFit(),
        symlib.DarkMatterSFH()
    ),
    symlib.ProfileModel(
        symlib.FixedRHalf(r_half_3d*1.3),
        symlib.PlummerProfile()
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

def position_based_mp(p, x_core, r_half):
    prof = symlib.PlummerProfile()
    dx = p["x"] - x_core
    r = np.sqrt(np.sum(dx**2, axis=1))
    idx = np.arange(len(r), dtype=int)
    order = np.argsort(r)
    orig_order = idx[order]

    ok = p["ok"][order]
    r = r[order][ok]
    orig_order = orig_order[ok]

    m_enc = prof.m_enc(1.0, r_half, r)
    dm = np.copy(m_enc)
    dm[1:] = m_enc[1:] - m_enc[:-1]

    mp = np.zeros(len(p))
    mp[orig_order] = dm

    return mp

def halfmass_energy_cutoff(p, x_core, v_core, r_half):
    rmax, vmax, pe_vmax2, _ = symlib.profile_info(
        lib.param, p["x"]-x_core, ok=p["ok"])

    dv = p["v"] - v_core
    ke_vmax2 = 0.5*np.sum(dv**2, axis=1) / vmax**2
    E_vmax2 = ke_vmax2 + pe_vmax2

    dx = p["x"] - x_core
    r = np.sqrt(np.sum(dx**2, axis=1))
    
    order = np.argsort(E_vmax2)
    ok = p["ok"][order]
    r = r[order][ok]

    bins = 10**np.linspace(-4, 0, 41)
    r_half_i = np.zeros(len(bins))
    
    for j, f in enumerate(bins):
        i = int(round(f*len(r)))
        if i >= len(p): i = len(p) - 1
        r_half_i[j] = np.median(r[:i+1])

    #idx = np.where(r_half_i < r_half)[0]
    idx = np.where(r_half_i > r_half)[0]
    if len(idx) == 0: idx = [-1]
    
    return E_vmax2, bins[idx[0]]

def main():
    fig, ax = plt.subplots(1, 2, figsize=(16, 16/2), sharey=True)
    colors = [pc("r"), pc("o"), pc("b"), pc("p")]# pc("g"), pc("b"), pc("p")]
    n_method = 3
    s_profs = [[] for _ in range(n_method)]
    e_profs = [[] for _ in range(n_method)]
    t_relax_t_dyn = []
    
    fs = []
    
    for i_host in range(symlib.n_hosts(lib.suite)):
        #if i_host >= 5: continue
        sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, i_host)
        h, hist = symlib.read_rockstar(sim_dir)

        start = -1*np.ones(len(h), dtype=int)
        end = -1*np.ones(len(h), dtype=int)
        target = []
            
        for i in range(1, len(h)):
            start_i, end_i, ok = start_end(lib.t, lib.t_dyn,
                                           h[i], hist[i], 1.0)
            if ok:
                start[i] = start_i
                end[i] = end_i
                target.append(i)

        target = np.array(target, dtype=int)

        if len(target) == 0: continue
        
        print("tagging", target)
        stars, gal, ranks = symlib.tag_stars(sim_dir, gal_halo,
                                             target_subs=target,
                                             star_snap=start,
                                             )#E_snap=start)
        
        part = symlib.Particles(sim_dir)
        for j, i in enumerate(target):
            ps = part.read(start[i], halo=i, mode="stars")
            pe = part.read(end[i], halo=i, mode="stars")

            core_xs = h["x"][i,start[i]]
            core_vs = h["v"][i,start[i]]
            core_xe = h["x"][i,end[i]]
            rvir = h["rvir"][i,start[i]]
            mp_pos = position_based_mp(ps, core_xs, rvir*r_half_3d*1.3)

            E, f = halfmass_energy_cutoff(ps, core_xs, core_vs, rvir*r_half_3d*1.3)
            E_cut = np.quantile(E[ps["ok"]], 0.1)
            E_cut_2 = np.quantile(E[ps["ok"]], f)
            mp_E = np.zeros(len(E))
            mp_E[E < E_cut] = 1.0
            mp_E_2 = np.zeros(len(E))
            mp_E_2[E < E_cut_2] = 1.0
            fs.append(f)

            r_half = gal["r_half_3d_i"][i]
            t_relax = t_relax_at_r_half(lib.mp, lib.eps[start[i]],
                                        ranks[i], rvir*r_half_3d)
            t_relax_t_dyn.append(t_relax / lib.t_dyn[start[i]])
            
            for i_fig, p, core_x, profs in [(1, ps, core_xs, s_profs), (2, pe, core_xe, e_profs)]:
                rvir0 = h["rvir"][i,start[i]]
                r, mass = cumulative_profile(stars[i]["mp"], p, core_x, rvir0)
                profs[0].append(mass)
                r, mass = cumulative_profile(mp_pos, p, core_x, rvir0)
                profs[1].append(mass)
                r, mass = cumulative_profile(mp_E, p, core_x, rvir0)
                profs[2].append(mass)
                #r, mass = cumulative_profile(mp_E_2, p, core_x, rvir0)
                #profs[3].append(mass)

                
    print(np.median(fs))
    print()
    print(np.quantile(t_relax_t_dyn, 0.5 + 0.68/2))
    print(np.median(t_relax_t_dyn))
    print(np.quantile(t_relax_t_dyn, 0.5 - 0.68/2))
    
    ax[0].set_xlabel(r"$r/R_{\rm vir}$")
    ax[0].set_ylabel(r"$M_\star(>r)/M_\star$")

    prof = symlib.PlummerProfile()
    m_target = prof.m_enc(1.0, r_half_3d*1.3, r)

    names = [r"${\rm fiducial}$", r"${\rm profile-matching}$",
             r"${\rm global\ energy\ cut}$",
             r"${\rm individual\ energy\ cut}$"]
    for i in range(n_method):
        ax[0].plot(r, np.median(s_profs[i], axis=0), c=colors[i],
                   label=names[i])
        ax[0].fill_between(r, np.quantile(s_profs[i], 0.5+0.68/2, axis=0),
                           np.quantile(s_profs[i], 0.5-0.68/2, axis=0),
                           color=colors[i], alpha=0.2)
    ax[0].plot(r, 1-m_target, "--", c="k")
    ax[0].legend(loc="lower left", fontsize=17)
    ax[0].set_ylim(0, 1)
    ax[0].set_title(r"$\Delta t = 0$")
    
    ax[1].set_xlabel(r"$r/R_{\rm vir}$")
    ax[1].set_ylabel(r"$M_\star(>r)/M_\star$")

    for i in range(n_method):
        ax[1].plot(r, np.median(e_profs[i], axis=0), c=colors[i])
        ax[1].fill_between(r, np.quantile(e_profs[i], 0.5+0.68/2, axis=0),
                           np.quantile(e_profs[i], 0.5-0.68/2, axis=0),
                           color=colors[i], alpha=0.2)
    ax[1].plot(r, 1-m_target, "--", c="k")
    ax[1].set_ylim(0, 1)
    ax[1].set_title(r"$\Delta t = t_{\rm cross}$")
    
    fig.savefig("plots/profile_comparison.pdf")
    
if __name__ == "__main__": main()
