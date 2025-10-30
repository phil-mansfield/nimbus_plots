import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
import palette
from palette import pc
import scipy.stats as stats
import symlib
from colossus.cosmology import cosmology
import scipy.stats as stats

commit_name = "1e1fed0769b25981fd6993d7a6118b0774a9c970"

def parse_fornax():
    with open("data/fornax_dwarf.dat", "r") as f: text = f.read()
    lines = [line.strip() for line in text.split("\n")
             if len(line.strip()) > 0]

    r, g, rh = zip(*[(line[73: 80], line[90: 97], line[137: 146])
                     for line in lines])
    r = np.array([float(rr) for rr in r])
    g = np.array([float(gg) for gg in g])
    rh = np.array([float(rr) for rr in rh])

    return r, g, rh

class JennyModel(object):
    def __init__(self):
        self.b, self.eta = 0.07, 3.88

    def gamma(self, tau, Mi_Mf):
        e = np.log10(Mi_Mf)/np.log10(tau)
        return e*0.07 + 0.984

    def f_dm(self, mi_Mf, Mi_Mf, tau_i, tau_f):
        g = self.gamma(tau_i, Mi_Mf)
        G = 1 - g
        b, eta = self.b, self.eta
        return (b * mi_Mf**b * eta * (tau_f**G - tau_i**G)/G + 1)**(-1/b)

    def f_star(self, f_dm, f_50):
        return 1 - np.exp(np.log(0.5)*f_dm/f_50)
    
def main():
    palette.configure(False)

    plt.figure(11)
    
    r50_all = []
    ms_all = []
    
    dwarf_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_mw.csv' % commit_name)
    dwarf_m31 = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_m31.csv' % commit_name)
    #dwarf_lf = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_local_field.csv' % commit_name)
    #dwarf_lfd = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/%s/data/dwarf_local_field_distant.csv' % commit_name)

    dwarfs = [dwarf_mw, dwarf_m31]
    colors = [pc("r"), pc("o")]
    
    for i in range(len(dwarfs)):
        dwarf = dwarfs[i]
        c = colors[i]
       
        ok = (dwarf["mass_stellar"] > 0) & (dwarf["rhalf"] > 0)
        dm = np.array(dwarf["distance_modulus"])
        dist = 10**(1 + dm/5)
        r50 = dist * np.array(dwarf["rhalf"])/(60*360)*(2*np.pi)
        
        plt.plot(10**dwarf["mass_stellar"][ok], r50[ok], "o", c=colors[i])

        r50_all.append(r50[ok])
        ms_all.append(10**dwarf["mass_stellar"][ok])
        
    ####

    # Ferrarese et al. (2020)
    # (Survey definition is Ferrarese et al. 2012)
    membership = np.loadtxt("data/virgo_sats.txt", skiprows=56,
                            usecols=(1,), dtype=int)
    
    g_sersic, re = np.loadtxt("data/virgo_sats.txt", skiprows=56,
                              usecols=(8, 10)).T
    gi = np.loadtxt("data/virgo_sat_colors.txt", skiprows=28,
                    usecols=(5,))
    
    # Convert re from arcsec to rad
    re = re * (2*np.pi) / (360 * 60 * 60)
    # Convert to distance
    # Distance from Simona Mei et al. (2007)
    D_virgo = 16.5 # Mpc (+/- 0.1; stat) (+/- 1.1; sys)
    re = re * D_virgo*1e6
    
    DM_virgo = 5*np.log10(D_virgo/1e-5)
    
    Mg = g_sersic - DM_virgo
    Mg_sun = 5.03 # Wilmer et al. (2018), CFHT g-band
    Lg = 10**(0.4*(Mg_sun - Mg))    
    
    # From Bell et al. (2003)
    a, b = -0.379, 0.914
    M_to_Lg = 10**(a + b*gi)

    Mstar = Lg * M_to_Lg

    ok = membership <= 1
    plt.plot(Mstar[ok], re[ok],  ".", pc("b"))

    r50_all.append(re[ok])
    ms_all.append(Mstar[ok])
    
    plt.plot([], [], "o", c=pc("r"), label=r"${\rm MW}$")
    plt.plot([], [], "o", c=pc("o"), label=r"${\rm M31}$")
    plt.plot([], [], ".", c=pc("b"), label=r"${\rm Virgo\ Cluster}$")

    plt.legend(loc="lower right", fontsize=17, frameon=True)
    plt.ylabel(r"$\log_{10}(r_{50})\ ({\rm pc})$")
    plt.xlabel(r"$\log_{10}(M_\star)\ (M_\odot)$")
    plt.xscale("log")
    plt.yscale("log")

    bins = 10**np.linspace(2, 11, 12)
    mids = np.sqrt(bins[1:]*bins[:-1])
    
    r50 = np.hstack(r50_all)
    ms = np.hstack(ms_all)
    ok = ms < 2e11
    r50, ms = r50[ok], ms[ok]
    
    md = stats.binned_statistic(ms, r50, "median", bins=bins)[0]
    lo = stats.binned_statistic(ms, r50, lambda x: np.quantile(x, 0.5 - 0.68/2), bins=bins)[0]
    hi = stats.binned_statistic(ms, r50, lambda x: np.quantile(x, 0.5 + 0.68/2), bins=bins)[0]

    print("mean 68% scatter (binned; log)")
    print(np.mean(np.log10(hi) - np.log10(lo))/2)
    
    plt.plot(mids, md, c="k")
    plt.fill_between(mids, lo, hi, color="k", alpha=0.2)

    p1 = np.polyfit(np.log10(ms), np.log10(r50), 3)
    p2 = np.polyfit(np.log10(mids), np.log10(md), 3)

    mass = 10**np.linspace(1, 12, 200)
    #plt.plot(mass, 10**np.polyval(p1, np.log10(mass)), lw=1.5, c="k")
    plt.plot(mass, 10**np.polyval(p2, np.log10(mass)), "--", lw=1.5, c="k")

    colors = [pc("r"), pc("o"), pc("b"), pc("p")]
    #suites = ["SymphonyGroup", "SymphonyMilkyWay", "SymphonyLMC", "SymphonyLCluster"]
    suites = ["SymphonyMilkyWay", "SymphonyLCluster"]

    plt.savefig("plots/size_mass.png")

    ms_all = []
    rvir_all = []
    f_dm_all = []
    f_star_all_1 = []
    f_star_all_2 = []
    f_star_all_3 = []
    
    for i_suite in range(len(suites)):
        suite = suites[i_suite]
        color = colors[i_suite]

        scale = symlib.scale_factors(suite)
        z = 1/scale - 1        
        
        param = symlib.simulation_parameters(suite)
        mp, eps = param["mp"]/param["h100"], param["eps"]*scale/param["h100"]
        cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))

        t = cosmo.age(z)
                
        model = JennyModel()
        
        for i_host in range(symlib.n_hosts(suite)):
            print(i_host+1,symlib.n_hosts(suite))
            if suite == "SymphonyLCluster" and i_host == 32: continue
            base_dir = "/sdf/home/p/phil1/ZoomIns"
            sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

            rs, hist = symlib.read_rockstar(sim_dir)
            um = symlib.read_um(sim_dir)

            snap = hist["first_infall_snap"]
            idx = np.arange(len(um), dtype=int)
            ms_infall = um["m_star"][idx,snap]
            rvir_infall = rs["rvir"][idx,snap]

            is_merged = (hist["merger_ratio"] > 0.15) & (~rs["ok"][:,-1])
            is_host = idx == 0
            is_err = ms_infall <= 1
            ok = ~(is_merged | is_host | is_err)
            
            ms_all.append(ms_infall[ok])
            rvir_all.append(rvir_infall[ok])

            mi_Mf = hist["mpeak_pre"][ok]/rs[0,-1]["m"]
            Mi_Mf = rs[0,snap[ok]]["m"]/rs[0,-1]["m"]
            tau_i = t[snap[ok]]/t[-1]
            tau_f = 1.0

            f_dm = model.f_dm(mi_Mf, Mi_Mf, tau_i, tau_f)
            f_star_1 = model.f_star(f_dm, 1/3)
            f_star_2 = model.f_star(f_dm, 1/30)
            f_star_3 = model.f_star(f_dm, 1/300)

        
            f_dm_all.append(f_dm)
            f_star_all_1.append(f_star_1)
            f_star_all_2.append(f_star_2)
            f_star_all_3.append(f_star_3)
            
    ms = np.hstack(ms_all)
    rvir = np.hstack(rvir_all)
    f_dm = np.hstack(f_dm_all)
    f_star_1 = np.hstack(f_star_all_1)
    f_star_2 = np.hstack(f_star_all_2)
    f_star_3 = np.hstack(f_star_all_3)
        
    plt.figure(10)
    plt.plot(ms, f_star_1, ".", c=pc("r"), alpha=0.2)
    plt.plot(ms, f_star_2, ".", c=pc("o"), alpha=0.2)
    plt.plot(ms, f_star_3, ".", c=pc("b"), alpha=0.2)

    plt.figure(11)
    #plt.plot(f_star_3*ms, 0.015*rvir*1e3, ".", c=pc("b"), alpha=0.2)
    #plt.plot(f_star_2*ms, 0.015*rvir*1e3, ".", c=pc("o"), alpha=0.2)
    #plt.plot(f_star_1*ms, 0.015*rvir*1e3, ".", c=pc("r"), alpha=0.2)
    bins = 10**np.linspace(3.5, 11, 20)
    mids = np.sqrt(bins[1:]*bins[:-1])

    ms_ok = (~np.isnan(ms)) & (~np.isinf(ms))
    ok_1 = (~np.isnan(f_star_1)) & (~np.isinf(f_star_1))
    ok_2 = (~np.isnan(f_star_2)) & (~np.isinf(f_star_2))
    ok_3 = (~np.isnan(f_star_3)) & (~np.isinf(f_star_3))

    ok = ms_ok & ok_1
    md_1 = stats.binned_statistic((f_star_1*ms)[ok], 0.015*rvir[ok]*1e3,
                                  "median", bins=bins)[0]
    ok = ms_ok & ok_1
    md_2 = stats.binned_statistic((f_star_2*ms)[ok], 0.015*rvir[ok]*1e3,
                                  "median", bins=bins)[0]
    ok = ms_ok & ok_1
    md_3 = stats.binned_statistic((f_star_3*ms)[ok], 0.015*rvir[ok]*1e3,
                                  "median", bins=bins)[0]

    plt.plot(mids, md_1, ":", c=pc("p"))
    plt.plot(mids, md_2, "--", c=pc("p"))
    plt.plot(mids, md_3, "-", c=pc("p"))
        
    plt.figure(10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$m_{\rm \star,infall}$")
    plt.ylabel(r"$f_\star = m_\star(z=0)/m_{\rm \star,infall}$")

    plt.savefig("plots/ms_fstar_%s.png" % suite)
    
    plt.figure(11)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$m_\star(z=0)$")
    plt.ylabel(r"$0.015\times r_{\rm vir,infall}\ ({\rm pc})$")
    plt.xlim(1e2, 1e11)
    
    plt.savefig("plots/ms_rvir_%s.png" % suite)
    
if __name__ == "__main__": main()
