import numpy as np
import symlib
import matplotlib.pyplot as plt
import palette
from palette import pc
import astropy.table as table
from colossus.cosmology import cosmology
import lib
import scipy.stats as stats
import numpy.random as random

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
    
    elif name == "virgo":
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

        return Mstar[ok], re[ok]
        
    
def main():
    palette.configure(True)
    
    modes = ["ms_r50"]
    #modes = ["ethan_comp"]
    #modes = ["rmax_comp"]

    galaxy_data = SimData("SymphonyMilkyWay")
    
    if "ms_r50" in modes:
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                               figsize=(24, 16))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)

        cluster_data = SimData("SymphonyLCluster")
    
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                plot_panel(ax[i,j], (i,j), galaxy_data, cluster_data)

        fig.savefig("plots/ms_rvir_sats.png")

        fig, ax = plt.subplots()
        
        plot_panel(ax, (-1, 0), galaxy_data, cluster_data)
        
        fig.savefig("plots/ms_rvir_sats_model_comp.png")

    if "ethan_comp" in modes:

        fig, ax = plt.subplots()
        plot_ethan_ms_r50(fig, ax, galaxy_data)
        fig.savefig("plots/ethan_comp_ms_r50.png")

        fig, ax = plt.subplots()
        plot_ethan_comp_mpeak_ms(fig, ax, galaxy_data)
        fig.savefig("plots/ethan_comp_mpeak_ms.png")
        

    if "rmax_comp" in modes:
        fig, ax = plt.subplots()
        plot_rmax_comp(fig, ax, galaxy_data)
        fig.savefig("plots/rmax_comp.png")
        
    
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

def combine(xs):
    shape = (sum([len(x) for x in xs]),)
    if len(xs[0].shape) > 1:
        shape += xs[0].shape[1:]
    out = np.zeros(shape, dtype=xs[0].dtype)
        
    i_low, i_high = 0, 0
    for i in range(len(xs)):
        i_low = i_high
        i_high = i_low + len(xs[i])
        out[i_low:i_high] = xs[i]

    return out
    
class SimData(object):
    def __init__(self, suite):
        scale = symlib.scale_factors(suite)
        z = 1/scale - 1        
        self.z = z

        self.model = JennyModel()
        
        param = symlib.simulation_parameters(suite)
        mp, eps = param["mp"]/param["h100"], param["eps"]*scale/param["h100"]
        cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))

        t = cosmo.age(z)

        n_hosts = symlib.n_hosts(suite)
        sf, hist, um, rs = [], [], [], []
        host_masses, host_idx = np.zeros((n_hosts, len(self.z))), []
        for i_host in range(n_hosts):            
            sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)
            if "Halo_050" in sim_dir: continue
            sf_i, hist_i = symlib.read_symfind(sim_dir)
            rs_i, _ = symlib.read_rockstar(sim_dir)
            um_i = symlib.read_um(sim_dir)

            rs.append(rs_i[1:])
            sf.append(sf_i[1:])
            um.append(um_i[1:])
            hist.append(hist_i[1:])

            host_masses[i_host] = rs_i["m"][0]
            host_idx.append(np.ones(len(rs_i)-1, dtype=int)*i_host)
            
            if i_host > 1:
                print("ENDING EARLY")
                break
            
        sf = combine(sf)
        rs = combine(rs)
        um = combine(um)

        hist = combine(hist)
        host_idx = combine(host_idx)

        snap_i = hist["first_infall_snap"]
        idx = np.arange(len(um), dtype=int)
        
        self.ms_infall = um["m_star"][idx, snap_i]
        self.ms_max = np.max(um["m_star"], axis=1)
        self.mvir_infall = rs["m"][idx, snap_i]
        self.rvir_infall = rs["rvir"][idx, snap_i]
        self.vmax_infall = rs["vmax"][idx, snap_i]
        self.rvmax_infall = rs["rvmax"][idx, snap_i]
        self.z_infall = self.z[hist["first_infall_snap"]]
        self.c_infall =rs["cvir"][idx, snap_i]
        
        is_merged = (hist["merger_ratio"] > 0.15) & (~sf["ok"][:,-1])
        is_err = self.ms_infall <= 0
        ok = ~(is_merged | is_err)
        
        sf, hist, um, rs = sf[ok], hist[ok], um[ok], rs[ok]
        host_idx = host_idx[ok]

        # This is an extrapolation for larger halos (npeak ~1e6 is outside
        # the fit range). There's also a typo in the published paper. If you
        # read the text, that 1/8 should be there, but I didn't put it in.
        x = np.log10(hist["mpeak_pre"]*8/mp)
        m_lim = 10**(-0.01853*x**2 + 0.3861*x + 1.6597)*mp/8
        
        full_snap = np.arange(len(z))
        snap_f = np.zeros(len(sf), dtype=int)
        for i_sub in range(len(hist)):
            ok_i = sf["ok"][i_sub] & (sf["m"][i_sub] > m_lim[i_sub])
            snap = full_snap[ok_i]

            snap_f[i_sub] = snap_i[i_sub] if len(snap) == 0 else np.max(snap)
            
        # This is unfortunate notation, but the final snapshot of the subhalo
        # (snap_f) is the initial snapshot of the orphan model (mi and Mi).
        host_mass_i = host_masses[host_idx,snap_f]
        host_mass_f = host_masses[host_idx,-1]

        idx = np.arange(len(snap_f), dtype=int)
        mi_Mf = sf["m"][idx,snap_f]/host_mass_f
        Mi_Mf = host_mass_i/host_mass_f
        tau_i = np.minimum(t[snap_f]/t[-1], 1 - 1e-9)
        tau_f = 1.0

        self.f_dm = np.zeros(len(ok))
        self.f_dm[ok] = self.model.f_dm(mi_Mf, Mi_Mf, tau_i, tau_f)
        self.f_dm[ok] *= sf["m"][idx,snap_f]/hist["mpeak_pre"]
        
def select_colors(n_curves):

    if n_curves == 2:
        return [pc("r"), pc("b")]
    elif n_curves <= 4:
        return [pc("r"), pc("o"), pc("b"), pc("p")][:n_curves]
    elif n_curves == 5:
        return [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]
    else:
        assert(0)

param_set = {panel_id: [] for panel_id in
          [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (-1, 0)]}
label_set = {panel_id: [] for panel_id in
          [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (-1, 0)]}
        
param_set[(0, 0)] = [
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/3},
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/30},
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/3000}
]

label_set[(0,0)] = [
    r"$f_{\rm 50} = 1/3$",
    r"$f_{\rm 50} = 1/30$",
    r"$f_{\rm 50} = 1/300$",
    r"$f_{\rm 50} = 1/3000$",
]

param_set[(0, 1)] = [
    {"AK": 0.015, "ac": 0.0, "az": 0.25, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": 0.0, "az": -0.25, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": 0.0, "az": -0.5, "sig": 0.0, "f50": 1/300}
]

label_set[(0,1)] = [
    r"$a_z = 0.25$",
    r"$a_z = 0$",
    r"$a_z = -0.25$",
    r"$a_z = -0.5$",
]

param_set[(0, 2)] = [
    {"AK": 0.015, "ac": 0.5, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": -0.5, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.015, "ac": -1.0, "az": 0.0, "sig": 0.0, "f50": 1/300}
]

label_set[(0,2)] = [
    r"$a_c = 0.5$",
    r"$a_c = 0$",
    r"$a_c = -0.5$",
    r"$a_c = -1.0$",
]

param_set[(-1, 0)] = [
    {"AK": 0.015, "ac": 0.0, "az": 0.0, "sig": 0.0, "f50": 1/300},
    {"AK": 0.02, "ac": -0.7, "az": -0.2, "sig": 0.0, "f50": 1/300},
    {"AN": 37, "n": 1.07, "sig": 0.0, "f50": 1/300},
    #{"AN": 80, "n": 1.75, "sig": 0.0, "f50": 1/300}
]

label_set[(-1, 0)] = [
    r"${\rm Kravtsov+2013}$",
    r"${\rm Jiang+2019}$",
    r"${\rm Nadler+2020}$",
    #r"$A=80\ {\rm pc},\ n=1.75$"
]

param_set[(1, 0)] = param_set[(0, 0)]
label_set[(1, 0)] = label_set[(0, 0)]

param_set[(1, 1)] = param_set[(0, 1)]
label_set[(1, 1)] = label_set[(0, 1)]

param_set[(1, 2)] = param_set[(0, 2)]
label_set[(1, 2)] = label_set[(0, 2)]

def plot_panel(ax, panel_id, galaxy_data, cluster_data):
    (row, col) = panel_id
    if row == 0: mode = "cluster"
    elif row == 1: mode = "galaxy"
    elif row == -1: mode = "both"
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    if row == 1: ax.set_xlabel(r"$m_\star\ (M_\odot)$")
    if col == 0: ax.set_ylabel(r"$r_{50}$")

    if row == -1:
        ax.set_xlabel(r"$m_\star\ (M_\odot)$")
        ax.set_ylabel(r"$r_{50}$")

    plot_data(ax, mode, panel_id)
    
    labels, params = label_set[panel_id], param_set[panel_id]
    n_param = len(params)
    colors = select_colors(n_param)

    assert(len(param_set[panel_id]) == len(label_set[panel_id]))
    for i in range(n_param):
        param, label = params[i], labels[i]

        galaxy_bins = 10**np.linspace(3.5, 9, 11)
        cluster_bins = 10**np.linspace(6, 11, 11)
        
        if row == 0:
            datasets = [cluster_data]
            bin_sets = [cluster_bins]
            ls = ["-"]
        elif row == 1:
            datasets = [galaxy_data]
            bin_sets = [galaxy_bins]
            ls = ["-"]
        else:
            datasets = [galaxy_data, cluster_data]
            bin_sets = [galaxy_bins, cluster_bins]
            ls = ["-", "--"]
            
            
        for j in range(len(datasets)):
            data, bins = datasets[j], bin_sets[j]

            if "AK" in param:
                zp1 = data.z_infall + 1
                r50 = (data.rvir_infall * param["AK"] * zp1**param["az"] *
                       (data.c_infall/10)**param["ac"])
                r50 *= 1e3 # Convert to pc
            elif "AN" in param:
                r50 = param["AN"]*(data.rvir_infall*(data.z_infall + 1)/10)**param["n"]

            r50 = 10**(np.log10(r50) + param["sig"]*random.randn(len(r50)))
        
            ms = data.ms_max*data.model.f_star(data.f_dm, param["f50"])
            
            mids = np.sqrt(bins[1:] * bins[:-1])
            meds, _, _ = stats.binned_statistic(ms, r50, "median", bins=bins)

            if j == 0:
                ax.plot(mids, meds, c=colors[i], ls=ls[j], label=label)
            else:
                ax.plot(mids, meds, c=colors[i], ls=ls[j])

    if n_param > 0:
        ax.legend(loc="lower right", frameon=True, fontsize=17)
    
    
def plot_data(ax, mode, panel_id):
    (row, col) = panel_id
    
    if mode == "galaxy":
        galaxy_ms, cluster_ms = "o", "x"
        galaxy_c, cluster_c = pc("k"), pc("a")
        if col == 0: ax.plot([], [], "o", c=pc("k"),
                             label=r"${\rm MW+M31}$")
    if mode == "cluster":
        galaxy_ms, cluster_ms = "x", "o"
        galaxy_c, cluster_c = pc("a"), pc("k")
        if col == 0: ax.plot([], [], "o", c=pc("k"),
                             label=r"${\rm Virgo}$")
        
    if mode == "both":
        galaxy_ms, cluster_ms = "o", "o"
        galaxy_c, cluster_c = pc("a"), pc("k")
        ax.plot([], [], "o", c=pc("k"), label=r"${\rm Virgo}$")
        ax.plot([], [], "o", c=pc("a"), label=r"${\rm MW+M31}$")
        
        
    ms, re = read_host("virgo")
    ax.plot(ms, re, cluster_ms, c=cluster_c)

    ms, re = read_host("mw")
    ax.plot(ms, re, galaxy_ms, c=galaxy_c)
    ms, re = read_host("m31")
    ax.plot(ms, re, galaxy_ms, c=galaxy_c)


def plot_ethan_ms_r50(fig, ax, sim):
    ms, re = read_host("mw")
    ax.plot(re, ms, "o", c=pc("k"), label=r"${\rm MW}$")
    ms, re = read_host("m31")
    ax.plot(re, ms, "o", c=pc("a"), label=r"${\rm M31}$")

    ax.set_xlabel(r"$r_{50}\ {\rm (pc)}$")
    ax.set_ylabel(r"$M_\star\ {\rm (M_\odot)}$")

    ax.set_xlim(10, 1000)
    ax.set_ylim(10**1.5, 10**10)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True)

    ax.grid()
    
def plot_ethan_comp_mpeak_ms(fig, ax, sim):
    #ax.plot(sim.mvir_infall, sim.ms_infall, ".",
    #        c=pc("a"), alpha=0.2)

    bins = 10**np.linspace(8, 10.5, 10)
    mids = stats.binned_statistic(
        sim.mvir_infall, sim.mvir_infall, "median", bins=bins)[0]

    med_sim = stats.binned_statistic(
        sim.mvir_infall, sim.ms_infall, "median", bins=bins)[0]

    ax.plot(mids, med_sim, pc("a"))
    
    um = UniverseMachineMStarFit(mode="all")
    ms_fit = um.m_star(sim.mvir_infall, sim.z_infall)
    med_um = stats.binned_statistic(
        sim.mvir_infall, ms_fit, "median", bins=bins)[0]
    ax.plot(mids, med_um, pc("r"))

    um = UniverseMachineMStarFit(mode="cen")
    ms_fit = um.m_star(sim.mvir_infall, sim.z_infall)
    med_um = stats.binned_statistic(
        sim.mvir_infall, ms_fit, "median", bins=bins)[0]
    ax.plot(mids, med_um, pc("o"))

    um = UniverseMachineMStarFit(mode="sat")
    ms_fit = um.m_star(sim.mvir_infall, sim.z_infall)
    med_um = stats.binned_statistic(
        sim.mvir_infall, ms_fit, "median", bins=bins)[0]
    ax.plot(mids, med_um, pc("b"))
    
    mpeak, ms = np.loadtxt("data/ethan_mpeak_ms.txt").T
    
    ax.plot(10**mpeak, 10**ms, "o", c=pc("k"))
    
    ax.set_xlabel(r"$m_{\rm peak}\ (M_\odot)$")
    ax.set_ylabel(r"$m_\star\ (M_\odot)$")

    ax.set_xlim(1e7, 1e11)
    ax.set_ylim(2e1, 4e8)
    
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid()


class UniverseMachineMStarFit(object):
    def __init__(self, scatter=0.2, alpha_z0=None, mode="all"):
        self.scatter = scatter
        self.alpha_z0 = alpha_z0
        self.mode = mode
    
    def m_star(self, mpeak, z):        
        mpeak = mpeak
        
        a = 1/(1 + z)

        if self.mode == "all":
            e0 = -1.435
            al_lna = -1.732

            ea = 1.831
            alz = 0.178

            e_lna = 1.368
            b0 = 0.482

            ez = -0.217
            ba = -0.841

            m0 = 12.035
            bz = -0.471

            ma = 4.556
            d0 = 0.411

            m_lna = 4.417
            g0 = -1.034

            mz = -0.731
            ga = -3.100

            if self.alpha_z0 is None:
                al0 = 1.963
            else:
                al0 = self.alpha_z0
            gz = -1.055

            ala = -2.316
            
        elif self.mode == "sat":
            e0    = -1.449
            ea    = -1.256
            e_lna = -1.031
            ez    =  0.108
            
            m0    = 11.896
            ma    =  3.284
            m_lna =  3.413
            mz    = -0.580
            
            if self.alpha_z0 is None:
                al0 = 1.949
            else:
                al0 = self.alpha_z0
                
            ala    = -4.096
            al_lna = -3.226
            alz    =  0.401
            
            b0    =  0.477
            ba   =  0.046
            bz = -0.214
            
            d0    =  0.357
            
            g0 = -0.755
            ga =  0.461
            gz =  0.025

        elif self.mode == "cen":
            e0 = -1.435
            ea =  1.813
            e_lna =  1.353
            ez = -0.214
            
            m0 = 12.081
            ma = 4.696
            m_lna = 4.485
            mz = -0.740

            if self.alpha_z0 is None:
                al0 = 1.957
            else:
                al0 = self.alpha_z0            
            ala = -2.650
            al_lna = -1.953
            alz =  0.204
 
            b0 = 0.474
            ba = -0.903
            bz = -0.492
            
            d0 = 0.386
            
            g0 = -1.065
            ga = -3.243
            gz = -1.107
            
        log10_M1_Msun = m0 + ma*(a-1) - m_lna*np.log(a) + mz*z
        e = e0 + ea*(a - 1) - e_lna*np.log(a) + ez*z
        al = al0 + ala*(a - 1) - al_lna*np.log(a) + alz*z
        b = b0 + ba*(a - 1) + bz*z
        d = d0
        g = 10**(g0 + ga*(a - 1) + gz*z)

        x = np.log10(mpeak/10**log10_M1_Msun)
      
        log10_Ms_M1 = (e - np.log10(10**(-al*x) + 10**(-b*x)) +
                       g*np.exp(-0.5*(x/d)**2))
                       
        log10_Ms_Msun = log10_Ms_M1 + log10_M1_Msun

        if self.scatter > 0.0:
            log_scatter = self.scatter*random.normal(
                0, 1, size=np.shape(mpeak))
            log10_Ms_Msun += log_scatter
        
        Ms = 10**log10_Ms_Msun
        
        return Ms

def plot_rmax_comp(fig, ax, sim):
    ax.plot(sim.vmax_infall, sim.rvmax_infall/sim.rvir_infall,
            ".", c=pc("r"), label=r"$r_X = r_{\rm max}$")

    ratio = sim.rvmax_infall/sim.rvir_infall
    #ratio_lo = np.quantile(ratio, 0.5-0.68/2)
    #ratio_md = np.quantile(ratio, 0.5)
    #ratio_hi = np.quantile(ratio, 0.5+0.68/2)
    print(np.quantile(ratio, 0.5))
    
    lo, hi = 5, 200
    ax.set_xlim(lo, hi)
    
    #ax.plot([lo, hi], 2*[ratio_md], "--", c=pc("r"))
    #ax.fill_between([lo, hi], 2*[ratio_lo], 2*[ratio_hi],
    #                color=pc("r"), alpha=0.2)

    ax.plot([lo, hi], [0.015, 0.015], c=pc("o"),
            label=r"$r_X = r_{\rm Kravtsov}$")

    print(0.015)
    
    zp1 = sim.z_infall + 1
    ax.plot(sim.vmax_infall, 0.02 * zp1**-0.2 * (sim.c_infall/10)**-0.7,
            ".", c=pc("b"), label=r"$r_X = r_{\rm Jiang}$")

    print(np.quantile(0.02 * zp1**-0.2 * (sim.c_infall/10)**-0.7, 0.5))
    
    ax.plot(sim.vmax_infall, 37e-3/10 * (sim.rvir_infall/10)**0.07,
            ".", c=pc("p"), label=r"$r_X = r_{\rm Nadler}$")

    print(np.quantile(37e-3/10 * (sim.rvir_infall/10)**0.07, 0.5))
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=17)
    
    ax.set_xlabel(r"$v_{\rm max}\ ({\rm km\,s^{-1}})$")
    ax.set_ylabel(r"$(r_X/r_{\rm vir})_0$")
        
if __name__ == "__main__": main()
