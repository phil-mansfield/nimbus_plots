import numpy as np
import symlib
import matplotlib.pyplot as plt
import palette
from palette import pc
import astropy.table as table

palette.configure(False)

lvd_commit_name = "1e1fed0769b25981fd6993d7a6118b0774a9c970"

def read_host(name, cache={}):
    if name in cache:
        return cache[name][0], cache[name][1]

    if name in ["mw", "m31"]:
        if name == "mw":
            dwarfs = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_mw.csv' % lvd_commit_name)
        elif name == "m31":
            dwarfs = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_m31.csv' % lvd_commit_name)

        ok = (dwarfs["mass_stellar"] > 0) & (dwarfs["rhalf"] > 0)
        dm = np.array(dwarfs["distance_modulus"])
        dist = 10**(1 + dm/5)
        r50 = dist * np.array(dwarfs["rhalf"])/(60*360)*(2*np.pi)

        cache[name] = (10**dwarfs["mass_stellar"][ok], r50[ok])
        return 10**dwarfs["mass_stellar"][ok], r50[ok]
    else:
        assert(0)

def main():
    model = symlib.FIDUCIAL_MODEL

    sim_dir = symlib.get_host_directory("/sdf/home/p/phil1/ZoomIns", "SymphonyMilkyWay", 0)
    gal = model.galaxy_properties(sim_dir)

    stars, gals, ranks = symlib.tag_stars(sim_dir, model)
    symlib.retag_stars(sim_dir, model, ranks)
    
    #plot_1(gal)

def plot_1(gal):
    save_dir = "../plots/garbage_plots/"

    ms_mw, r50_mw = read_host("mw")
    ms_m31, r50_m31 = read_host("m31")
    
    plt.figure()
    plt.plot(gal["m_star"], gal["r_half_2d"], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$r_{\rm 50,2d}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.plot(ms_mw, r50_mw*1e-3, "o", c=pc("r"))
    plt.plot(ms_m31, r50_m31*1e-3, "o", c=pc("b"))
    
    plt.savefig("%s/mstar_r2d.png" % save_dir)

    plt.figure()
    plt.plot(gal["m_star"], gal["r_half_3d"], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$r_{\rm 50,3d}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_r3d.png" % save_dir)
    
    plt.figure()
    plt.plot(gal["m_star"], gal["Fe_H"], ".", c="k")
    plt.ylabel(r"$[{\rm Fe/H}]$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_Fe_H.png" % save_dir)

    plt.figure()
    plt.plot(gal["m_star"], gal["sigma_Fe_H"], ".", c="k")
    plt.ylabel(r"$\sigma_{\rm Fe/H}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_sigma_Fe_H.png" % save_dir)
    
    plt.figure()
    plt.plot(gal["m_star"], gal["a50"], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$a_{50}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_a50.png" % save_dir)
    
    plt.figure()
    plt.plot(gal["m_star"], gal["a90"], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$a_{90}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_a90.png" % save_dir)
    
    plt.figure()
    plt.plot(gal["m_star"], gal["profile_params"][:,0], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$n_{\rm sersic}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mstar_n.png" % save_dir)
    
    rs, hist = symlib.read_rockstar(sim_dir)
    mpeak = hist["mpeak_pre"]
    
    plt.figure()
    plt.plot(mpeak, gal["m_star"], ".", c="k")
    plt.yscale("log")
    plt.ylabel(r"$m_{\rm peak}$")
    plt.xscale("log")
    plt.xlabel(r"$m_\star$")

    plt.savefig("%s/mpeak_mstar.png" % save_dir)
    
if __name__ == "__main__": main()
