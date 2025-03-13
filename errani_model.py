import numpy as np
import palette
import matplotlib.pyplot as plt
from palette import pc
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from colossus.cosmology import cosmology
from colossus.halo import concentration
import symlib
import scipy.special as special
import scipy.optimize as optimize

#profile_type = "NFW"
profile_type = "Einasto"

param = symlib.simulation_parameters("SymphonyMilkyWay")
c_param = symlib.colossus_parameters(param)
cosmo = cosmology.setCosmology("", c_param)

palette.configure(False)
figsize=(7,7)

A_nfw = 2.163 # Rvmax/Rs

# c_model(mvir, z)
c_model = lambda mvir, z: concentration.concentration(mvir, "vir", z, model="diemer19")
# m_star_model.m_star(mpeak, z)
m_star_model = symlib.UniverseMachineMStarFit(scatter=0.0)
# r_half_model.r_half(rvir, cvir, z)
r_half_model = symlib.Jiang2019RHalf(scatter=0)

def mvir_to_rvir(mvir, a, omega_M):
    """ mvir_to_rvir converts a Bryan & Norman virial mass in Msun/h to a virial
    radius in comoving Mpc/h at a given scale factor a, and omega_M.
    """
    
    if type(mvir) == np.ndarray:
        mvir = np.maximum(np.zeros(len(mvir)), mvir)
    else:
        mvir = max(0, mvir)

    omega_L = 1 - omega_M
    Ez = np.sqrt(omega_M/a**3 + omega_L)
    rho_crit = 2.77519737e11*Ez**2
    omega_Mz = (omega_M/a**3)/Ez**2

    x = omega_Mz - 1
    delta_vir = 18*np.pi**2 + 82*x - 39.0*x**2
    rho_vir = rho_crit*delta_vir

    r_phys = (mvir/(rho_vir * (4*np.pi / 3)))**(1.0/3)
    r_cmov = r_phys/a

    return r_cmov

def phi_vmx2_nfw(r_rmx):
    return -2.14 / r_rmx * np.log(1 + A_nfw*r_rmx)

def rho_rho0_nfw(r_rmx):
    r_rs = r_rmx*A_nfw
    return 1/(r_rs * (1 + r_rs)**2)

def A_ein(a):
    return 1.715*a**-0.00183 * (a + 0.0817)**-0.179488

def B_ein(a):
    return 9.529*a**-0.00635 * (a + 0.3036)**-0.206

def phi_vmx2_ein(r_rmx, alpha):
    # I dare you to double check my math on this. Do it, coward, you'll be the
    # one who suffers, not me.
    a = alpha
    A = A_ein(a)
    B = B_ein(a)

    def gamma(a): return special.gamma(a)
    def gammaincc(a, x): return special.gamma(a)*special.gammaincc(a, x)

    x = A*r_rmx # x = r/rs

    y = 2*x**a / a
    F = (special.gammainc(3/a, y)/x +
        (2/a)**(1/a) * (gamma(2/a) + gammaincc(2/a, y))/gamma(3/a))
    xi = 4*np.pi/a * (2/a)**(-3/a) * special.gamma(3/a) * np.exp(2/a)  

    # F_lim is lim x -> \infty F(x)
    F_lim = (2/a)**(1/a) * gamma(2/a)/gamma(3/a)

    return -xi/B * (F - F_lim)

def rho_rho0_ein(r_rmx, alpha=0.16):
    A = A_ein(alpha)
    return np.exp(-2/alpha * ((A*r_rmx)**alpha - 1))

if profile_type == "NFW":
    phi_vmx2, rho_rho0 = phi_vmx2_nfw, rho_rho0_nfw
    r_apo_max_rvir, rmx_rs = 1, A_nfw
    mvir_mtot = lambda c: 1.0 # uhhhh, I don't know how to do this for NFW
else:
    alpha = 0.16
    phi_vmx2 = lambda r_rmx: phi_vmx2_ein(r_rmx, alpha)
    rho_rho0 = lambda r_rmx: rho_rho0_ein(r_rmx, alpha)
    r_apo_max_rvir = np.inf
    rmx_rs = A_ein(alpha)
    mvir_mtot = lambda c: special.gammainc(3/alpha, 2*c**alpha/alpha)

def create_plummer_rho(r_half):
    """ r50 is 3D """
    #a = np.sqrt(2**(2/3) - 1) * r_half
    a = r_half
    return lambda r: (1 + (r/a)**2)**-2.5

def create_ein_rho(r_half, alpha):
    x_half =  optimize.root_scalar(
        lambda x: special.gammainc(3/alpha, 2*x**alpha/alpha) - 0.5,
        x0 = 1
    ).root

    rho0 = np.exp(-2/alpha * (x_half**alpha - 1))

    return lambda x: np.exp(-2/alpha * ((x_half*x/r_half)**alpha - 1))/rho0

def eps_mass(eps, dN_deps):
    f_dN = interpolate.interp1d(eps, dN_deps, kind="linear")
    return integrate.quad(f_dN, eps[0], eps[-1])[0]

def truncate_eps(eps, dN_deps, eps_t):
    a = 0.85
    b = 12.0
    return dN_deps/(1 + (a*eps/eps_t)**b)
    
def eddington_inversion(rho_prof, phi_vmx2, r_apo_max=np.inf):
    r = 10**np.linspace(-2, 2, 200)
    rho = rho_prof(r)
    
    phi = phi_vmx2(r)
    phi_max = phi[-1]
    phi_min = phi_vmx2(1e-6)
    eps = 1 - phi/phi_min

    log_phi_to_r = interpolate.interp1d(phi, np.log10(r), kind="linear")
    phi_to_r = lambda phi: 10**log_phi_to_r(phi)
    
    deriv = (np.gradient(rho, edge_order=2) /
             np.gradient(phi, edge_order=2))
    
    deriv_func = interpolate.interp1d(phi, deriv, kind="linear")
    integral = np.zeros(phi.shape)
    for i in range(len(integral)):
        E = phi[i]
        integral[i] = integrate.quad(
            lambda phi_: deriv_func(phi_)/np.sqrt(phi_ - E), E, phi_max)[0]

    fE = (np.gradient(integral, edge_order=2) /
          np.gradient(phi, edge_order=2))

    pE = np.zeros(phi.shape)
    for i in range(len(pE)):
        E = phi[i]
        r_apo = phi_to_r(E)
        if r_apo > r_apo_max: r_apo = r_apo_max
        pE[i] = integrate.quad(
            lambda r_: r_**2 * np.sqrt(2*(E - phi_vmx2(r_))),
            0, r_apo)[0]

    dN_deps = pE*fE

    ok = ~np.isnan(dN_deps)
    eps, dN_deps = eps[ok], dN_deps[ok]

    return eps, dN_deps/eps_mass(eps, dN_deps)
    
def energy_distribution():
    z = 0.0
    a = 1/(z + 1)
    mvir = 10**np.array([10, 11, 12, 13])[::-1]
    cvir = c_model(mvir, z)
    rvir = mvir_to_rvir(mvir*0.7, a, param["Om0"])/0.7

    mass_idx = ["10", "11", "12", "13"][::-1]
    colors = [pc("r"), pc("o"), pc("b"), pc("p")][::-1]
    
    m_star = m_star_model.m_star(mvir, z)
    r_half = r_half_model.r_half(rvir, cvir, z)

    rvir_scale = 10/rmx_rs
    eps_nfw, dN_deps_nfw = eddington_inversion(
        rho_rho0, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)

    plt.figure(0, figsize=figsize)
    plt.plot(eps_nfw, dN_deps_nfw, "--", c=pc("k"))

    eps_t = 0.15 # This is picked arbitrarily for visualization
    dN_deps_t = truncate_eps(eps_nfw, dN_deps_nfw, eps_t)
    plt.plot(eps_nfw, dN_deps_t, pc("k"), label=r"${\rm DM}$")
    
    for i in range(len(mvir)):
        rvir_scale = cvir[i]/rmx_rs
        r = 10**np.linspace(-2, 2, 200)
        rhalf_scale = rvir_scale * r_half[i]/rvir[i]
        
        eps_nfw, dN_deps_nfw = eddington_inversion(
            rho_rho0, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)
        
        plummer_rho = create_plummer_rho(rhalf_scale)
        eps_pl, dN_deps_pl = eddington_inversion(
            plummer_rho, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)

        plt.figure(0, figsize=figsize)
        plt.plot(eps_pl, dN_deps_pl*m_star[i]/mvir[i], "--", c=colors[i])

        dN_deps_pl_t = truncate_eps(eps_pl, dN_deps_pl, eps_t)
        plt.plot(eps_pl, dN_deps_pl_t*m_star[i]/mvir[i], c=colors[i],
                 label=r"${\rm Stars}\ M_{\rm sub}=10^{%s}$" % mass_idx[i])

        if i == 1:
            eps_pl2, dN_deps_pl2 = eps_pl, dN_deps_pl
            plummer_rho = create_plummer_rho(rhalf_scale/10**0.2)
            eps_pl1, dN_deps_pl1 = eddington_inversion(
                plummer_rho, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)
            plummer_rho = create_plummer_rho(rhalf_scale*10**0.2)
            eps_pl3, dN_deps_pl3 = eddington_inversion(
                plummer_rho, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)
            
            eps_ti = 10**np.linspace(-2, 0, 50)
            mf_dm, mf_gal1 = np.zeros(eps_ti.shape), np.zeros(eps_ti.shape)
            mf_gal2, mf_gal3 = np.zeros(eps_ti.shape), np.zeros(eps_ti.shape)
            for j in range(len(eps_ti)):
                dN_deps_pl_t1 = truncate_eps(eps_pl1, dN_deps_pl1, eps_ti[j])
                dN_deps_pl_t2 = truncate_eps(eps_pl2, dN_deps_pl2, eps_ti[j])
                dN_deps_pl_t3 = truncate_eps(eps_pl3, dN_deps_pl3, eps_ti[j])
                dN_deps_nfw_t = truncate_eps(eps_nfw, dN_deps_nfw, eps_ti[j])

                mf_gal1[j] = eps_mass(eps_pl1, dN_deps_pl_t1)
                mf_gal2[j] = eps_mass(eps_pl2, dN_deps_pl_t2)
                mf_gal3[j] = eps_mass(eps_pl3, dN_deps_pl_t3)
                mf_dm[j] = eps_mass(eps_nfw, dN_deps_nfw_t)
                mf_dm[j] /= mvir_mtot(cvir[i])

            fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
            fig.suptitle(r"$m_{\rm dm,infall} = 10^{12}\,M_\odot$")
            ax[0].plot(mf_dm, mf_gal2, c=colors[i])
            ax[0].fill_between(mf_dm, mf_gal1, mf_gal3,
                               color=colors[i], alpha=0.3)
            ax[1].plot(mf_dm, mf_gal2/mf_dm * m_star[i]/mvir[i],
                       color=colors[i])
            ax[1].fill_between(mf_dm, mf_gal1/mf_dm * m_star[i]/mvir[i],
                               mf_gal3/mf_dm * m_star[i]/mvir[i],
                               color=colors[i], alpha=0.3)
    
    plt.figure(0)    
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", fontsize=17)
    
    plt.xlim(0.01, 1)
    plt.ylim(0.0005, 20)

    plt.xlabel(r"$\varepsilon \equiv 1 - \phi/\phi_{\rm min}$")
    plt.ylabel(r"$dN/d\varepsilon$")

    m_low = 1e-4
    ax[0].set_xlim(m_low, 1)
    ax[0].set_ylim(1e-4, 1)
    ax[0].plot([m_low, 1], [1, 1], "--", lw=2, c="k")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$m_{\rm \star}/m_{\rm \star,infall}$")

    ax[1].set_xlim(m_low, 1)
    ax[1].set_ylim(1e-3, 10)
    ax[1].plot([m_low, 1], [1, 1], "--", lw=2, c="k")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$m_\star/m_{\rm dm}$")
    ax[1].set_xlabel(r"$m_{\rm dm}/m_{\rm dm,infall}$")

def m_vs_star_frac():
    z = 0.0
    a = 1/(z + 1)
    mvir = 10**np.linspace(10, 13, 50)
    cvir = c_model(mvir, z)
    rvir = mvir_to_rvir(mvir*0.7, a, param["Om0"])/0.7
    
    m_star = m_star_model.m_star(mvir, z)
    r_half = r_half_model.r_half(rvir, cvir, z)

    plt.figure(3, figsize=figsize)

    ms_frac = [np.zeros(mvir.shape) for _ in range(3)]
    r_half_mult = [1/10**0.2, 1, 10**0.2]
    
    for i in range(len(mvir)):
        rvir_scale = cvir[i]/A
        rhalf_scale = rvir_scale * r_half[i]/rvir[i]
        eps_nfw, dN_deps_nfw = eddington_inversion(
            rho_rho0, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)

        for j in range(3):
            plummer_rho = create_plummer_rho(rhalf_scale*r_half_mult[j])
            eps_pl, dN_deps_pl = eddington_inversion(
                plummer_rho, phi_vmx2, r_apo_max=rvir_scale*r_apo_max_rvir)

            eps_ti = 10**np.linspace(-2, 0, 50)
            mf_dm, mf_gal = np.zeros(eps_ti.shape), np.zeros(eps_ti.shape)
            for k in range(len(eps_ti)):
                dN_deps_pl_t = truncate_eps(eps_pl, dN_deps_pl, eps_ti[k])
                dN_deps_nfw_t = truncate_eps(eps_nfw, dN_deps_nfw, eps_ti[k])

                mf_gal[k] = eps_mass(eps_pl, dN_deps_pl_t)
                mf_dm[k] = eps_mass(eps_nfw, dN_deps_nfw_t)

            k = np.searchsorted(mf_gal, 0.5)
            print(i, j, k)
            ms_frac[j][i] = mf_gal[k]/mf_dm[k] * m_star[i]/mvir[i]

    plt.plot(mvir, ms_frac[1], color=pc("r"))
    plt.fill_between(mvir, ms_frac[0], ms_frac[2], color=pc("r"), alpha=0.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$m_{\rm dm,infall}$")
    plt.ylabel(r"$(m_\star/m_{\rm dm})(m_\star/m_{\rm \star,infall} = 0.5)$")

    plt.xlim(1e10, 1e13)
    plt.plot([1e10, 1e13], [1, 1], c="k")
    plt.plot([1e10, 1e13], [10**0.3, 10**0.3], "--", lw=2, c="k")
    plt.plot([1e10, 1e13], [10**-0.3, 10**-0.3], "--", lw=2, c="k")
    
#def tidal_track_table():
#

def einasto_tidal_track_table():
    cvir = 10
    #r50_rvirs = [1/300, 1/100, 1/30, 1/10]
    r50_rvirs = [0.015]
    alphas = [1/8, 1/4, 1/2, 1, 1/0.5] # /roughly/ 1/n for sersisc galaxies

    shades = [0.7, 0.55, 0.45, 0.3]
    shades = [0.5]
    color_strs = ["r", "o", "g", "b", "p"]
    colors = [[pc(color_str, shade) for color_str in color_strs] for shade in shades]

    z = 0.0
    a = 1/(z + 1)
    mvir = 1e11
    cvir = c_model(mvir, z)
    rvir = mvir_to_rvir(mvir*0.7, a, param["Om0"])/0.7
    
    m_star = m_star_model.m_star(mvir, z)
    r_half = r_half_model.r_half(rvir, cvir, z)
    
    # rvir_rmx = Rvir/Rmax. All other variables similarly named are in
    # the same units. It actually doens't matter what you normalize these things
    # by.
    rvir_rmx = cvir/rmx_rs 
    eps_dm, dN_deps_dm = eddington_inversion(
        rho_rho0, phi_vmx2, r_apo_max=rvir_rmx*r_apo_max_rvir)

    r_rmx = 10**np.linspace(-3, 2, 200)
    n_t = 50
    mf_gal = np.zeros((len(r50_rvirs), len(alphas), n_t))
    mf_dm = np.zeros((len(r50_rvirs), len(alphas), n_t))

    for i in range(len(r50_rvirs)):
        for j in range(len(alphas)):
            rhalf_rmx = rvir_rmx * r50_rvirs[i]
            rho_gal = create_ein_rho(rhalf_rmx, alphas[j])
            eps_gal, dN_deps_gal = eddington_inversion(
                rho_gal, phi_vmx2, r_apo_max=rvir_rmx*r_apo_max_rvir)
            
            eps_ti = 10**np.linspace(-2, 0, n_t)
            for k in range(len(eps_ti)):
                dN_deps_gal_t = truncate_eps(eps_gal, dN_deps_gal, eps_ti[k])
                dN_deps_dm_t = truncate_eps(eps_dm, dN_deps_dm, eps_ti[k])

                mf_gal[i,j,k] = eps_mass(eps_gal, dN_deps_gal_t)
                mf_dm[i,j,k] = eps_mass(eps_dm, dN_deps_dm_t)
                mf_dm[i,j,k] /= mvir_mtot(cvir)


    plt.figure()
    for i in range(len(r50_rvirs)):
        for j in range(len(alphas)):
            plt.plot(mf_dm[i,j], mf_gal[i,j], lw=2, c=colors[i][j],
                label=r"$\alpha=%.3f$" % alphas[j])

    #plt.plot(mf_dm[0,0], 1 - np.exp(-14.2*mf_dm[i,j]), c="k", lw=3,
    #    label=r"${\rm Smith+2016}$")

    cs = np.array([2, 2, 2])
    bs = np.array([0.65, 1.2, 3])/cs
    fs0 = 0.45
    for k in range(len(bs)):
        b = bs[k]
        c = cs[k]
        D = -np.log(1 - fs0**(1/c))
        plt.plot(mf_dm[0,0], (1 - np.exp(-D*(mf_dm[i,j]/0.0075)**b))**c,
                c=pc("k"), ls="--", lw=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-2, 2)
    plt.xlabel(r"$m_{\rm dm}/m_{\rm vir,0}$")
    plt.ylabel(r"$m_\star/m_{\star,0}$")
    plt.xlim(1e-4, 1)
    plt.legend(loc="lower right")

    #for k in range(len(mf_dm[0,0])):
    #    print("%.3f" % mf_dm[0,0], end=" ")
    #for i in range(i):


def einasto_f0_size():
    cvir = 10
    r50_rvirs = 10**np.linspace(-3, -0.5, 10)
    alphas = [1/8, 1/4, 1/2, 1, 1/0.5] # /roughly/ 1/n for sersisc galaxies
    colors = [pc("r"), pc("o"), pc("g"), pc("b"), pc("p")]

    z = 0.0
    a = 1/(z + 1)
    mvir = 1e11
    cvir = c_model(mvir, z)
    rvir = mvir_to_rvir(mvir*0.7, a, param["Om0"])/0.7
    
    m_star = m_star_model.m_star(mvir, z)
    r_half = r_half_model.r_half(rvir, cvir, z)
    
    # rvir_rmx = Rvir/Rmax. All other variables similarly named are in
    # the same units. It actually doens't matter what you normalize these things
    # by.
    rvir_rmx = cvir/rmx_rs
    eps_dm, dN_deps_dm = eddington_inversion(
        rho_rho0, phi_vmx2, r_apo_max=rvir_rmx*r_apo_max_rvir)

    r_rmx = 10**np.linspace(-3, 2, 200)
    f50 = np.zeros((len(r50_rvirs), len(alphas)))
    n_t = 100

    for i in range(len(r50_rvirs)):
        for j in range(len(alphas)):
            rhalf_rmx = rvir_rmx * r50_rvirs[i]
            rho_gal = create_ein_rho(rhalf_rmx, alphas[j])
            eps_gal, dN_deps_gal = eddington_inversion(
                rho_gal, phi_vmx2, r_apo_max=rvir_rmx*r_apo_max_rvir)
            
            eps_ti = 10**np.linspace(-2, 0, n_t)
            mf_dm, mf_gal = np.zeros(n_t), np.zeros(n_t)
            for k in range(len(eps_ti)):
                dN_deps_gal_t = truncate_eps(eps_gal, dN_deps_gal, eps_ti[k])
                dN_deps_dm_t = truncate_eps(eps_dm, dN_deps_dm, eps_ti[k])

                mf_gal[k] = eps_mass(eps_gal, dN_deps_gal_t)
                mf_dm[k] = eps_mass(eps_dm, dN_deps_dm_t)
                mf_dm[k] /= mvir_mtot(cvir)

            k0 = np.searchsorted(mf_gal, 0.5)
            f50[i, j] = mf_dm[k0]


    plt.figure()
    for j in range(len(alphas)):
        plt.plot(r50_rvirs, f50[:,j], c=colors[j],
                label=r"$\alpha=%0.3f$" % alphas[j])        
    
    plt.legend(loc="upper left", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$r_{50,\star}/R_{\rm vir,peak}$")
    plt.ylabel(r"$f_{\rm dm,50}$")

def main():
    #energy_distribution()
    #m_vs_star_frac()    
    einasto_tidal_track_table()
    #einasto_f0_size()
    plt.show()
    
if __name__ == "__main__": main()
