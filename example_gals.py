import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import lib
import symlib

palette.configure(False)

def main():
    i_host = 0
    suite = "SymphonyMilkyWay"
    sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)

    sf, hist = symlib.read_symfind(sim_dir)
    gal, gal_hist = symlib.read_galaxies(sim_dir)
    colors = [pc("p"), pc("b"), pc("g"), pc("o"), pc("r")]
    modes = ["r=0.005", "r=0.008", "r=0.015", "r=0.025", "r=0.05"]
    gals = [symlib.read_galaxies(sim_dir, mode)[0] for mode in modes]
    a = symlib.scale_factors(sim_dir)
    
    n_plots = 0
    fig, ax = plt.subplots(2, 2, figsize=(16,16), sharex=True)
    for i in range(1, len(gal)):
        if hist["merger_ratio"][i] > 0.15: continue

        ok = sf["ok"][i]

        ax[0,0].cla()
        ax[0,0].plot(a[ok], sf["m"][i,ok], c=pc("k"),
                     label=r"$m_{\rm dm}$")
        for im in range(len(modes)):
            ax[0,0].plot(a[ok], gals[im]["m_star"][i,ok], c=colors[im],
                         label=r"$m_\star(%s)$" % modes[im])
        ax[0,0].set_yscale("log")
        ax[0,0].set_ylabel(r"$m\ (M_\odot)$")
        ax[0,0].legend(loc="lower left", fontsize=16)
        
        ax[0,1].cla()
        ax[0,1].plot(a[ok], np.sqrt(np.sum(sf["x"][i,ok], axis=1)**2),
                     c=pc("r"))
        ax[0,1].set_yscale("log")
        ax[0,1].set_ylabel(r"$r\ ({\rm kpc})$")
        ax[0,1].legend(loc="lower left", fontsize=17)

        ax[1,0].cla()
        ax[1,0].plot(a[ok], sf["r_half"][i,ok], c=pc("k"),
                     label=r"$r_{50,{\rm dm}}$")

        for im in range(len(modes)):
            ax[1,0].plot(a[ok], gals[im]["r_half"][i,ok], c=colors[im],
                         label=r"$r_{50,{\star}}(%s)$" % modes[im])
        ax[1,0].set_yscale("log")
        ax[1,0].set_ylabel(r"$r_{50}\ ({\rm kpc})$")
        ax[1,0].legend(loc="lower left", fontsize=16)
        
        ax[1,1].cla()
        ax[1,1].set_yscale("log")
        ax[1,1].plot(a[ok], gal["v_disp_3d_dm"][i,ok], c=pc("r"),
                     label=r"$\sigma_{\rm dm}$")
        ax[1,1].plot(a[ok], gal["vmax_dm"][i,ok], c=pc("o"),
                     label=r"$v_{\rm max}$")
        ax[1,1].plot(a[ok], gal["vmax_dm_debias"][i,ok], "--", c=pc("o"),
                     lw=2, label=r"$f_{\rm MA20}\cdot v_{\rm max}$")
        ax[1,1].plot(a[ok], gal["v_disp_3d_star"][i,ok], c=pc("b"),
                     label=r"$\sigma_\star$")
        ax[1,1].set_ylabel(r"$v\ ({\rm km\,s^{-1}})$")
        ax[1,1].legend(loc="lower left", fontsize=17)
        
        ax[0,0].set_xlim(0, 1)
        ax[1,0].set_xlabel(r"$a(t)$")
        ax[1,1].set_xlabel(r"$a(t)$")

        fig.savefig("plots/example_gals/gal_%03d.png" % i)
        
        n_plots += 1
        if n_plots >= 100: break
        
if __name__ == "__main__": main()
