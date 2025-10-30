import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import scipy.interpolate as interpolate
import astropy.table as table
import numpy.random as random
import scipy.stats as stats
import scipy.optimize as optimize
import scipy.interpolate as interpolate

palette.configure(True)

lvd_commit_name = "1e1fed0769b25981fd6993d7a6118b0774a9c970"
rng = np.random.default_rng(1)

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

def mvir_to_rvir(mvir, a, omega_M=0.286):
    # This uses little-h units
    omega_L = 1 - omega_M
    Ez = np.sqrt(omega_M/a**3 + omega_L)
    rho_crit = 2.77519737e11*Ez**2
    omega_Mz = (omega_M/a**3)/Ez**2

    x = omega_Mz - 1
    delta_vir = 18*np.pi**2 + 82*x - 39.0*x**2
    rho_vir = rho_crit*delta_vir

    r_phys = (mvir/(rho_vir * (4*np.pi / 3)))**(1.0/3)
    return r_phys

q = np.array([0.025, 0.160, 0.500, 0.860, 0.975])
log_a = np.array([-0.897, -0.681, -0.448, -0.205, -0.039])
a = 10**log_a
rvir = mvir_to_rvir(1, a)
print("""# 0 - quantile of the a_infall distribution
# 1 - a_infall
# 2 - rvir(m, a_infall)/rvir(m, <a_infall>)
# 2 - rvir(m, a_infall)/rvir(m, <a_infall>) * (1 + z)^0.26
# 2 - rvir(m, a_infall)/rvir(m, <a_infall>) * (1 + z)""")
for i in range(len(q)):
    print("%.3f %.3f %.3f %.3f %.3f" %
          (q[i], a[i], rvir[i]/rvir[2],
           rvir[i]/rvir[2]*(a[i]/a[2])**0.26,
           rvir[i]/rvir[2]*(a[i]/a[2])))

def main1():
    # Data for fiducial alpha
    fig2, ax2 = plt.subplots()
    # Alpha dependence
    fig3, ax3 = plt.subplots()
    
    mh0, ms0 = 1e11, 3e8
    colors = ["", pc("r", 0.7), pc("r", 0.3),
              pc("k"), pc("b", 0.3), pc("b", 0.7)]

    # log-slope of ms(mh)
    alpha_z0s = np.array([1.957, 3.0, 2.5, 2.0, 1.5, 1.0])
    # Scatter in ms(mh)
    sigma_ms = 0.2
    
    ms_mw,  r50_mw  = read_host("mw")
    ms_m31, r50_m31 = read_host("m31")

    n_mw = len(ms_mw)
    ms_obs  = np.hstack([ms_mw,  ms_m31])
    r50_obs = np.hstack([r50_mw, r50_m31])
    #ms_obs  = np.hstack([ms_m31])
    #r50_obs = np.hstack([r50_m31])
    ok = ms_obs < 1e8
    ms_obs, r50_obs = ms_obs[ok], r50_obs[ok]
    
    for i in range(len(alpha_z0s)):

        # 68% and median quantiles in z_infall
        z_infall = 1/10**np.array([-0.681, -0.448, -0.205]) - 1

        um = UniverseMachineMStarFit(scatter=0.2, alpha_z0=alpha_z0s[i],
                                     mode="cen")

        mh_mid, sigma_mh = um.m_halo(ms_obs, z_infall[1])        
        rh_mid   = mvir_to_rvir(mh_mid*0.7,   1/(z_infall[1] + 1))*1e6/0.7

        def f(x, a, b): return -1.58 + a*np.exp(b*x)
        mu, cov = optimize.curve_fit(f, np.log10(ms_obs),
                                     np.log10(r50_obs/rh_mid))

        
        if i == 0:
            print("best-fitting exponential fit to log10(r_50,2d/Rvir) with")
            print("UM parameters:")
            print("%.4f * np.exp(%.4f * log10(m_*)) - 1.58" % (mu[0], mu[1]))
            
            mh_early, _      = um.m_halo(ms_obs, z_infall[0])
            mh_late, _       = um.m_halo(ms_obs, z_infall[2])
            rh_early = mvir_to_rvir(mh_early*0.7, 1/(z_infall[0] + 1))*1e6/0.7
            rh_late  = mvir_to_rvir(mh_late*0.7,  1/(z_infall[2] + 1))*1e6/0.7

            rh_low, rh_high = rh_early, rh_late
            switch = rh_low > rh_high
            tmp_low, tmp_high = rh_low[switch], rh_high[switch]
            rh_low[switch], rh_high[switch] = tmp_high, tmp_low
            yerr = np.zeros((2,len(rh_mid)))
            yerr[0,:] = np.sqrt((np.log10(rh_mid) - np.log10(rh_low))**2 +
                                (sigma_mh/3)**2)
            yerr[1,:] = np.sqrt((np.log10(rh_high) - np.log10(rh_mid))**2 +
                                (sigma_mh/3)**2)
            xx = np.log10(ms_obs)
            yy = np.log10(r50_obs/rh_mid)
            
            ax2.errorbar(xx[:n_mw], yy[:n_mw], elinewidth=1.5,
                         fmt="o", color="k", yerr=yerr[:,:n_mw], ms=5)
            ax2.errorbar(xx[n_mw:], yy[n_mw:],
                         elinewidth=1.5,
                         fmt="o", color="k", yerr=yerr[:,n_mw:], ms=5,
                         linewidth=1.5)
            ax2.errorbar(np.log10(ms_obs)[n_mw:],
                         np.log10(r50_obs/rh_mid)[n_mw:],
                         fmt="o", color="w", ms=1.5)

            delta = yy - f(xx, *mu)
            err = yerr[0,:]
            err[delta > 0] = yerr[1,delta > 0]
            print("|delta|/err for fit", np.mean(np.abs(delta)/err))
            
            x = np.linspace(2, 8, 100)
            ax2.plot(x, f(x, *mu), c=pc("r"))
            low, high = curve_contours(f, x, mu, cov, rng)
            ax2.fill_between(x, low, high, color=pc("r"), alpha=0.2)
        else:
            bins = np.linspace(2.0, 8.0, 6)
        
            mean_x, _, _ = stats.binned_statistic(
                np.log10(ms_obs), np.log10(ms_obs), "median", bins=bins
            )
            mean_y, _, _ = stats.binned_statistic(
                np.log10(ms_obs), np.log10(r50_obs/rh_mid), "median", bins=bins
            )

            ax3.plot(mean_x, mean_y, lw=2.5, c=colors[i],
                     label=r"$\alpha_{\rm SMHM}(z=0)=%.1f$" % alpha_z0s[i])

        ax2.set_xlim(2, 8)
        ax2.set_ylim(-3.25, -0.25)

        ax3.set_xlim(2, 8)
        ax3.set_ylim(-3.25, -0.25)

    # This is the mean r_50/rvir value for z = median z_infall for the
    # lower-mass sample in Somerville et al. (2019).
    ax2.plot([2, 8], 2*[-1.58], "--", c=pc("r"), lw=2)
    
    ax2.set_xlabel(r"$\log_{10}(m_\star)\ (M_\odot)$")
    ax2.set_ylabel(r"$\log_{10}(r_{50,\star,2d}/r_{\rm vir,infall})\ (M_\odot)$")

    ax3.set_xlabel(r"$\log_{10}(m_\star)\ (M_\odot)$")
    ax3.set_ylabel(r"$\log_{10}(r_{50,\star,2d}/r_{\rm vir,infall})\ (M_\odot)$")

    ax3.legend(loc="lower right", fontsize=17)

    fig2.savefig("plots/size_mass_panel_2.pdf")
    fig3.savefig("plots/size_mass_panel_3.pdf")
    
def main2():
    fig, ax = plt.subplots()
    s = np.loadtxt("data/somerville.txt").T

    mass = s[0,1:]
    z = s[1:,0]
    pivot_zp1 = np.mean(np.log10(z+1))
    ratio = s[1:,1:]

    ratio_hm = ratio[:,6:]
    ratio_lm = ratio[:,:6]

    mean_hm, mean_lm = np.mean(ratio_hm, axis=1), np.mean(ratio_lm, axis=1)
    
    ax.plot(np.log10(1+z), np.log10(mean_hm), "o", c=pc("b"),
            label=r"$m_\star > 10^{10.5}M_\odot$")
    ax.plot(np.log10(1+z), np.log10(mean_lm), "o", c=pc("r"),
            label=r"$10^9 M_\odot<m_\star < 10^{10.5} M_\odot$")
            

    p_hm, err_hm, cov_hm = line_fit(np.log10(1+z) - pivot_zp1, np.log10(mean_hm))
    
    p_lm, err_lm, cov_lm = line_fit(np.log10(1+z) - pivot_zp1, np.log10(mean_lm))

    print("pivot_zp1:", pivot_zp1)
    print("p_hm:", p_hm)
    print("err_hm:", err_hm)
    print("p_lm:", p_lm)
    print("err_lm:", err_lm)
    
    x1 = np.linspace(0, 0.8, 100)
    x2 = x1[(x1 > np.log10(z[0]+1)) & (x1 < np.log10(z[-1] + 1))]
    ax.plot(x1, np.polyval(p_hm, x1-pivot_zp1), "--", c=pc("b"), lw=1.5)
    ax.plot(x1, np.polyval(p_lm, x1-pivot_zp1), "--", c=pc("r"), lw=1.5)
    ax.plot(x2, np.polyval(p_hm, x2-pivot_zp1), "-", c=pc("b"), lw=2)
    ax.plot(x2, np.polyval(p_lm, x2-pivot_zp1), "-", c=pc("r"), lw=2)

    f = lambda x, a, b: a*x + b
    low_hm, high_hm = curve_contours(f, x1 - pivot_zp1, p_hm, cov_hm, rng)
    low_lm, high_lm = curve_contours(f, x1 - pivot_zp1, p_lm, cov_hm, rng)

    ax.fill_between(x1, low_hm, high_hm, color=pc("b"), alpha=0.2)
    ax.fill_between(x1, low_lm, high_lm, color=pc("r"), alpha=0.2)
    
    q, log_a = np.loadtxt("data/a_infall.txt").T
    
    log_zp1 = -log_a

    #lo, hi = ax.get_ylim()
    #hi = -1.45
    #ax.set_ylim(lo, hi)
    ax.set_ylim(-3.25, -0.25)
    lo, hi = ax.get_ylim()
    
    ax.set_xlim(0, 0.8)
    ax.plot(2*[log_zp1[2]], [lo, hi], "--", c="k", lw=2)
    ax.fill_between([log_zp1[1], log_zp1[3]], [lo, lo], [hi, hi],
                    color="k", alpha=0.2)
    
    # Converged mass range in Mansfield & Avestruz
    print("r200c/rvir scatter (dex)", 0.0084/3)
    print("r200c/rvir(mvir = 10^8.5): %.4f" % ma_r200c_rvir(10**8.5))
    print("r200c/rvir(mvir = 10^15): %.4f" % ma_r200c_rvir(10**15))
    
    ax.set_xlabel(r"$\log_{10}(1+z)$")
    ax.set_ylabel(r"$\log_{10}(\langle r_{\star,50,\star,2d}/r_{\rm vir}\rangle)$")
    ax.legend(loc="upper left", fontsize=17, frameon=True)    

    fig.savefig("plots/size_mass_panel_1.pdf")
    
    plt.show()
    
def line_fit(x, y):
    def f(x, a, b): return a*x + b
    opt, cov = optimize.curve_fit(f, x, y)
    return opt, (np.sqrt(cov[0,0]), np.sqrt(cov[1,1])), cov

def curve_contours(f, x, mu, cov, rng, iters=10000):
    params = rng.multivariate_normal(mu, cov, size=iters)

    y = np.zeros((iters, len(x)))
    for i in range(iters):
        y[i,:] = f(x, *params[i])

    return np.quantile(y,0.5-0.68/2, axis=0), np.quantile(y,0.5+0.68/2, axis=0)
    

def ma_r200c_rvir(mvir):
    x = np.log10(mvir) - 12.5
    p = [0.8381219, -0.0229426, -0.0001334, 0.0011298, 0.0001582][::-1]
    return np.polyval(p, x)**(1/3)

class UniverseMachineMStarFit(object):
    def __init__(self, scatter=0.2, alpha_z0=None, mode="all",
                 rng=random.default_rng()):
        self.scatter = scatter
        self.alpha_z0 = alpha_z0
        self.mode = mode
        self.rng = rng
    
    def m_star(self, mpeak, z, scatter=None):
        if scatter is None: scatter = self.scatter
        
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
            log_scatter = scatter*self.rng.normal(
                0, 1, size=np.shape(mpeak))
            log10_Ms_Msun += log_scatter
        
        Ms = 10**log10_Ms_Msun
        
        return Ms

    def m_halo(self, m_star, z):
        mh_0 = 10**np.linspace(4, 17, 200)
        ms_0 = self.m_star(mh_0, z, scatter=0)

        dln_ms_dln_mh_0 = np.zeros(len(mh_0))
        dln_ms_dln_mh_0[1:-1] = ((np.log(ms_0[2:]) - np.log(ms_0[:-2])) /
                                 (np.log(mh_0[2:]) - np.log(mh_0[:-2])))
        dln_ms_dln_mh_0[0]  = dln_ms_dln_mh_0[1]
        dln_ms_dln_mh_0[-1] = dln_ms_dln_mh_0[-2]
        
        f_beta = interpolate.interp1d(np.log10(ms_0), dln_ms_dln_mh_0)
        f_ms = interpolate.interp1d(np.log10(ms_0), np.log10(mh_0))
        beta = f_beta(np.log10(m_star))
        ms_to_mh = lambda ms: 10**f_ms(np.log10(ms))

        # From Symphony
        alpha = -1.92
        return invert_smhm(m_star, alpha, beta, self.scatter, ms_to_mh)

def invert_smhm(m_star, alpha, beta, sigma_ms, ms_to_mh):
    # alpha is the log-slope of HMF, beta is the log-slope of the SMHM relation
    # sigma_ms is in dex, m_star is in Msun, and ms_to_mh is a function
    # with units Msun -> Msun
    inv_sigma = sigma_ms/beta
    inv_mu = 10**((sigma_ms/beta)**2 * (1 + alpha) +
                  np.log10(ms_to_mh(m_star)))
    return inv_mu, inv_sigma
    
if __name__ == "__main__":
    main1()
    main2()
