import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import symlib
import lib

def main():
    palette.configure(False)
    
    base_dir = lib.base_dir
    sim_dir = symlib.get_host_directory(base_dir, "SymphonyMilkyWay", 0)
    scale = symlib.scale_factors(sim_dir)

    gal, gal_hist = symlib.read_galaxies(sim_dir, "fid_dwarf")
    
    i = 2

    rs, _    = symlib.read_rockstar(sim_dir)
    sf, hist = symlib.read_symfind(sim_dir)

    plt.figure(1)
    ok = rs["ok"][i]
    plt.figure()
    plt.plot(scale[ok], rs["m"][i,ok], c=pc("b"), label=r"$m_{\rm rockstar}$")
    ok = sf["ok"][i]
    plt.plot(scale[ok], sf["m"][i,ok], c=pc("r"),
             label=r"$m_{\rm symfind}\ ({\tt 'm'})$")
    
    plt.plot(scale[ok], sf["m_raw"][i,ok], lw=1.5, c=pc("r"),
             label=r"$m_{\rm symfind}\ ({\tt' m\_raw'})$")
    ok = gal["ok"][i]
    plt.plot(scale[ok], gal["m_star"][i,ok], c=pc("o"), label=r"$m_\star$")
    plt.yscale("log")
    plt.legend(loc="lower left", frameon=True, fontsize=17)
    plt.savefig("plots/example_mass.png")
    
    plt.cla()
    ok = gal["ok"][i] & sf["ok"][i]
    snap = hist["first_infall_snap"][i]
    plt.plot(sf["m"][i,ok] / sf["m"][i,snap],
             gal["m_star"][i,ok] / gal["m_star"][i,snap], c="k")
    plt.plot(sf["m"][i,ok] / sf["m"][i,snap],
             gal["m_star"][i,ok] / gal["m_star"][i,snap], "o", c="k")

    
    plt.xscale("log")
    plt.ylabel(r"$f_\star$")
    plt.xlabel(r"$f_{\rm dm}$")
    plt.savefig("plots/example_track.png")
    

    
if __name__ == "__main__": main()
