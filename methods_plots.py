import lib
import matplotlib.pyplot as plt
import palette
from palette import pc
import numpy as np
import symlib
import numpy.random as random
import matplotlib as mpl
import scipy.interpolate as interpolate
import scipy.integrate as integrate

palette.configure(True)

def fixed_r_gal_halo(r_ratio):
    return symlib.GalaxyHaloModel(
        symlib.StellarMassModel(
            symlib.UniverseMachineMStar(),
            symlib.UniverseMachineSFH()
        ),
        symlib.ProfileModel(
            symlib.FixedRHalf(r_ratio),
            symlib.PlummerProfile()
        ),
        symlib.MetalModel(
            symlib.Kirby2013Metallicity(),
            symlib.Kirby2013MDF(model_type="gaussian"),
            symlib.FlatFeHProfile(),
            symlib.GaussianCoupalaCorrelation()
        )
    )

def fig_1():
    random.seed(1)
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    i_host = 0
    sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, i_host)

    rs, hist = symlib.read_rockstar(sim_dir)
    sf, hist = symlib.read_symfind(sim_dir)

    part = symlib.Particles(sim_dir)
    p = part.read(hist["first_infall_snap"][1], mode="stars", halo=1)

    gal_halo = fixed_r_gal_halo(0.05)
    stars, gals, ranks = symlib.tag_stars(sim_dir, gal_halo, target_subs=(1,),
                                          energy_method="E_sph")
    M_enc = np.cumsum(ranks[1].M, axis=0)*lib.mp

    c_lo, c_hi = 0.5, 0.9
    n_prof = M_enc.shape[1]
    r_bins = ranks[1].r_bins
    for i in range(n_prof):
        c = pc("k", c_lo + (c_hi - c_lo)*i/n_prof)
        ax[0].plot(r_bins[1:], M_enc[:,i], lw=2, c=c)

    
    models = [
        (fixed_r_gal_halo(0.025), pc("r"), 0.025),
        (fixed_r_gal_halo(0.05), pc("o"), 0.05),
        (fixed_r_gal_halo(0.10), pc("b"), 0.10),
        (fixed_r_gal_halo(0.20), pc("p"), 0.20)
    ]

    state = None
    for i in range(len(models)):
        model, c, r_half2d = models[i]

        m_mult = 10**(random.randn())
        
        stars, gals, state = symlib.retag_stars(
            sim_dir, model, ranks, target_subs=(1,), state=state)
        m_enc = model.profile_shape_model.m_enc(
            gals["m_star_i"][1], r_half2d, r_bins[1:]) * m_mult
        ax[0].plot(r_bins[1:], m_enc, c=c)
        
        snap = hist["first_infall_snap"][1]
        rvir = rs["rvir"][1, snap]
        x0 = rs["x"][1, snap]
        r = np.sqrt(np.sum((p["x"] - x0)**2, axis=1)) / rvir
        mass, _ = np.histogram(r, bins=r_bins, weights=stars[1]["mp"])
        mass = np.cumsum(mass) * m_mult

        edges = ranks[1].E_edges
        E = (edges[1:] + edges[:-1])/2
        ok = ranks[1].mp_star_table > 1e-2
        ax[1].plot(E[ok], ranks[1].mp_star_table[ok], c=c)
        
        ax[2].plot(r_bins[1:], mass, "--", c=c, lw=2)
        ax[2].plot(r_bins[1:], m_enc, c=c)


    ax[0].plot([], [], c="k", lw=2, label=r"${\rm DM\ profiles}$")
    ax[0].plot([], [], c=pc("r"), lw=2, label=r"${\rm target\ profiles}$")
        
    ax[2].plot([], [], "-", c=pc("a"), lw=3,
               label=r"${\rm target\ profile}$")
    ax[2].plot([], [], "--", c=pc("a"), lw=2,
               label=r"${\rm recovered\ profile}$")
        
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e6, None)
    ax[0].set_xlabel(r"$r/r_{\rm vir}$")
    ax[0].set_ylabel(r"$m(<r)\ (M_\odot)$")
    ax[0].legend(loc="upper left", fontsize=17)
    
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$E/V^2_{\rm max}$")
    ax[1].set_ylabel(r"$m_p\ (M_\odot)$")

    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$r/r_{\rm vir}$")
    ax[2].set_ylabel(r"$m(<r)\ (M_\odot)$")
    ax[2].legend(loc="lower right", fontsize=17)
    ax[2].set_ylim(1e3, 1e9)
    
    fig.savefig("plots/methods_1.pdf")

def fig_2():
    random.seed(1)
    
    fig, ax = plt.subplots(2, 2)

    i_host = 0
    sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, i_host)
    
    rs, hist = symlib.read_rockstar(sim_dir)
    um = symlib.read_um(sim_dir)
    a = symlib.scale_factors(sim_dir)

    j = 3
    stars, gals, ranks = symlib.tag_stars(
        sim_dir, symlib.DWARF_GALAXY_HALO_MODEL, target_subs=(j,),
        energy_method="E_sph"
    )
    stars = stars[j]

    age_infall = lib.cosmo.age(1/a[hist["first_infall_snap"][j]] - 1)
    age = lib.cosmo.age(1/stars["a_form"] - 1)
    
    norm = mpl.colors.LogNorm(vmin=1, vmax=5000)
    ax[1,0].hexbin(age, stars["Fe_H"], norm=norm, cmap="inferno",
                   gridsize=50)

    #xlo, xhi = ax[1,0].get_xlim()
    xlo, xhi = 0, lib.cosmo.age(0)
    ylo, yhi = ax[1,0].get_ylim()
    ax[1,0].set_xlim(xlo, xhi)
    ax[1,0].set_ylim(ylo, yhi)
    ax[0,0].set_xlim(xlo, xhi)
    ax[1,1].set_ylim(ylo, yhi)

    cumu = False
    bins = np.linspace(xlo, xhi, 50)
    #bins = lib.cosmo.age(1/a - 1)
    ax[0,0].hist(age, weights=stars["mp"],
                 density=True,
                 color=pc("k"), cumulative=cumu, histtype="step", lw=3,
                 bins=bins)
    ax[0,0].plot([], [], c="k", label=r"${\rm Nimbus}$")

    age = lib.cosmo.age(1/a - 1)
    norm = np.trapz(um["sfr"][j], age)
    
    ax[0,0].plot(age, um["sfr"][j]/norm, "--", color=pc("r"), lw=2,
                 label=r"${\rm target}$")
    ax[0,0].legend(loc="upper left", fontsize=17)
    bins = np.linspace(ylo, yhi, 50)
    ax[1,1].hist(stars["Fe_H"], weights=stars["mp"], density=True,
                 color=pc("k"), cumulative=cumu, histtype="step", lw=3,
                 bins=bins, orientation="horizontal")
    ax[1,1].plot([], [], c="k", label=r"${\rm Nimbus}$")

    mu = gals[j]["Fe_H_i"]
    sigma = gals[j]["sigma_Fe_H_i"]
    P = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * (bins - mu)**2 / sigma**2)
    ax[1,1].plot(P, bins, "--", lw=2, c=pc("r"),
                 label=r"${\rm target}$")
    
    ax[1,0].plot([age_infall, age_infall], [ylo, yhi], "--", c=pc("a"))
    ylo, yhi = ax[0,0].get_ylim()
    ax[0,0].set_ylim(ylo, yhi)
    ax[0,0].plot([age_infall, age_infall], [ylo, yhi], "--", c=pc("a"))
    
    ax[0,0].set_ylabel(r"${\rm Pr}(t_{\rm form})$")
    ax[1,1].set_xlabel(r"${\rm Pr}([{\rm Fe/H}])$")
    ax[1,0].set_xlabel(r"$t_{\rm form}\ ({\rm Gyr})$")
    ax[1,0].set_ylabel(r"$[{\rm Fe/H}]$")
    ax[1,1].tick_params(labelleft=False)
    ax[0,0].tick_params(labelbottom=False)
    ax[0,1].set_visible(False)
    
    fig.savefig("plots/methods_2.pdf")
    
def main():
    fig_1()
    #fig_2()

if __name__ == "__main__": main()
