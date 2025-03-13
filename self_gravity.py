import numpy as np
import palette
import matplotlib.pyplot as plt
from palette import pc
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from colossus.cosmology import cosmology
from colossus.halo import concentration
import symlib

param = symlib.simulation_parameters("SymphonyMilkyWay")
c_param = symlib.colossus_parameters(param)
cosmo = cosmology.setCosmology("", c_param)

palette.configure(True)
figsize=(7,7)

A = 2.163
B = 1.64

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

def phi_vmx2(r_rmx):
    """ returns E in units of vmx^2 """
    #return -4*np.pi/(B*A**2) / r_rmx * np.log(1 + B*r_rmx)
    return -4.63/B / r_rmx * np.log(1 + B*r_rmx)

def rho_rho0(r_rmx):
    r_rs = r_rmx*B
    return 1/(r_rs * (1 + r_rs)**2)

def create_plummer_rho(r_half):
    """ r50 is 3D """
    a = 1/np.sqrt(2**(2/3) - 1) * r_half
    return lambda r: (1 + (r/a)**2)**-2.5

def eps_mass(eps, dN_deps):
    f_dN = interpolate.interp1d(eps, dN_deps, kind="linear")
    return integrate.quad(f_dN, eps[0], eps[-1])[0]

def truncate_eps(eps, dN_deps, eps_t):
    a = 0.85
    b = 12.0
    return dN_deps/(1 + (a*eps/eps_t)**b)
    
def nfw_eddington_inversion(rho_prof, r_apo_max=np.inf):
    r = 10**np.linspace(-2, 2, 200)
    rho = rho_prof(r)
    
    phi = phi_vmx2(r)
    phi_min, phi_max = phi[0], phi[-1]
    phi0 = -4.63
    eps = 1 - phi/phi0
    
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

    rvir_scale = 10/A
    eps_nfw, dN_deps_nfw = nfw_eddington_inversion(rho_rho0,
                                                   r_apo_max=rvir_scale)

    plt.figure(0, figsize=figsize)
    plt.plot(eps_nfw, dN_deps_nfw, "--", c=pc("k"))

    eps_t = 0.117 # 1:100 mass loss for a cvir=10 subhalo
    dN_deps_t = truncate_eps(eps_nfw, dN_deps_nfw, eps_t)
    plt.plot(eps_nfw, dN_deps_t, pc("k"), label=r"${\rm DM}$")
    
    for i in range(len(mvir)):
        print("E distr", i)
        rvir_scale = cvir[i]/A
        r = 10**np.linspace(-2, 2, 200)
        rhalf_scale = rvir_scale * r_half[i]/rvir[i]
        
        eps_nfw, dN_deps_nfw = nfw_eddington_inversion(
            rho_rho0, r_apo_max=rvir_scale)
        
        plummer_rho = create_plummer_rho(rhalf_scale)
        eps_pl, dN_deps_pl = nfw_eddington_inversion(
            plummer_rho, r_apo_max=rvir_scale)

        plt.figure(0, figsize=figsize)
        plt.plot(eps_pl, dN_deps_pl*m_star[i]/mvir[i], "--", c=colors[i])

        dN_deps_pl_t = truncate_eps(eps_pl, dN_deps_pl, eps_t)
        plt.plot(eps_pl, dN_deps_pl_t*m_star[i]/mvir[i], c=colors[i],
                 label=r"${\rm Stars,}\ m_{\rm dm,infall}=10^{%s}$" % mass_idx[i])

        if i == 1:
            eps_pl2, dN_deps_pl2 = eps_pl, dN_deps_pl
            plummer_rho = create_plummer_rho(rhalf_scale/10**0.2)
            eps_pl1, dN_deps_pl1 = nfw_eddington_inversion(
                plummer_rho, r_apo_max=rvir_scale)
            plummer_rho = create_plummer_rho(rhalf_scale*10**0.2)
            eps_pl3, dN_deps_pl3 = nfw_eddington_inversion(
                plummer_rho, r_apo_max=rvir_scale)
            
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

    plt.xlabel(r"$\varepsilon \equiv 1 - E/\phi_{\rm min}$")
    plt.ylabel(r"$dm/d\varepsilon/m_{\rm dm,infall}$")
    plt.savefig("plots/energy_distr.pdf")
    
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

    fig.savefig("plots/stellar_tidal_track.pdf")
    
def m_vs_star_frac():
    mvir = 10**np.linspace(10, 13, 50)
    
    cvir_z0 = c_model(mvir, 0.0)
    rvir_z0 = mvir_to_rvir(mvir*0.7, 1, param["Om0"])/0.7
    m_star_z0 = m_star_model.m_star(mvir, 0.0)
    r_half_z0 = r_half_model.r_half(rvir_z0, cvir_z0, 0.0)

    cvir_z2 = c_model(mvir, 2.0)
    rvir_z2 = mvir_to_rvir(mvir*0.7, 1/3, param["Om0"])/0.7
    m_star_z2 = m_star_model.m_star(mvir, 2.0)
    r_half_z2 = r_half_model.r_half(rvir_z2, cvir_z2, 2.0)
    
    ms_frac_z0 = [np.zeros(mvir.shape) for _ in range(3)]
    ms_frac_z2 = [np.zeros(mvir.shape) for _ in range(3)]
    dm_frac_z0 = [np.zeros(mvir.shape) for _ in range(3)]
    dm_frac_z2 = [np.zeros(mvir.shape) for _ in range(3)]
    r_half_mult = [1/10**0.2, 1, 10**0.2]

    z_arrays = [
        (cvir_z0, rvir_z0, m_star_z0, r_half_z0, ms_frac_z0, dm_frac_z0),
        (cvir_z2, rvir_z2, m_star_z2, r_half_z2, ms_frac_z2, dm_frac_z2)
    ]

    for iz in range(2):
        cvir, rvir, m_star, r_half, ms_frac, dm_frac = z_arrays[iz]
        for i in range(len(mvir)):
            rvir_scale = cvir[i]/A
            rhalf_scale = rvir_scale * r_half[i]/rvir[i]
            eps_nfw, dN_deps_nfw = nfw_eddington_inversion(
                rho_rho0, r_apo_max=rvir_scale)

            for j in range(3):
                print(iz, i, j)
                plummer_rho = create_plummer_rho(rhalf_scale*r_half_mult[j])
                eps_pl, dN_deps_pl = nfw_eddington_inversion(
                    plummer_rho, r_apo_max=rvir_scale)

                eps_ti = 10**np.linspace(-2, 0, 50)
                mf_dm, mf_gal = np.zeros(eps_ti.shape), np.zeros(eps_ti.shape)
                for k in range(len(eps_ti)):
                    dN_deps_pl_t = truncate_eps(eps_pl, dN_deps_pl, eps_ti[k])
                    dN_deps_nfw_t = truncate_eps(eps_nfw, dN_deps_nfw, eps_ti[k])

                    mf_gal[k] = eps_mass(eps_pl, dN_deps_pl_t)
                    mf_dm[k] = eps_mass(eps_nfw, dN_deps_nfw_t)
                
                f_intr = interpolate.interp1d(mf_gal, mf_dm)
                ms_frac[j][i] = 0.5/f_intr(0.5) * m_star[i]/mvir[i]
                dm_frac[j][i] = f_intr(0.5)
                
    plt.figure(3, figsize=figsize)
    plt.plot(mvir, ms_frac_z0[1], color=pc("r"), label=r"$z=0$")
    plt.fill_between(mvir, ms_frac_z0[0], ms_frac_z0[2],
                     color=pc("r"), alpha=0.2)
    plt.plot(mvir, ms_frac_z2[1], color=pc("b"), label=r"$z=2$")
    plt.fill_between(mvir, ms_frac_z2[0], ms_frac_z2[2],
                     color=pc("b"), alpha=0.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$m_{\rm dm,infall}\ (M_\odot)$")
    plt.ylabel(r"$(m_\star/m_{\rm dm})(m_\star/m_{\rm \star,infall} = 0.5)$")

    plt.xlim(1e10, 1e13)
    plt.legend(fontsize=17, loc="upper left")
    plt.plot([1e10, 1e13], [1, 1], c="k")
    plt.plot([1e10, 1e13], [10**0.3, 10**0.3], "--", lw=2, c="k")
    plt.plot([1e10, 1e13], [10**-0.3, 10**-0.3], "--", lw=2, c="k")

    
    plt.savefig("plots/m_vs_star_frac.pdf")

    plt.figure(4, figsize=figsize)

    plt.plot(mvir, dm_frac_z0[1], color=pc("r"), label=r"$z=0$")
    plt.fill_between(mvir, dm_frac_z0[0], dm_frac_z0[2],
                     color=pc("r"), alpha=0.2)
    plt.plot(mvir, dm_frac_z2[1], color=pc("b"), label=r"$z=2$")
    plt.fill_between(mvir, dm_frac_z2[0], dm_frac_z2[2],
                     color=pc("b"), alpha=0.2)

    plt.ylim(1e-3, 1e-1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$m_{\rm dm,infall}\ (M_\odot)$")
    plt.ylabel(r"$m_{\rm dm}/m_{\rm dm, infall})(m_\star/m_{\rm \star,infall} = 0.5)$")

    plt.xlim(1e10, 1e13)
    plt.legend(fontsize=17)
    
    plt.savefig("plots/m_vs_dm_frac.pdf")
    
def main():
    #energy_distribution()
    m_vs_star_frac()    
    #plt.show()


    
if __name__ == "__main__": main()
