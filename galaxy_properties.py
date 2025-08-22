import numpy as np
import symlib
import numpy.random as ranomd
import gravitree
import os
import sys
import scipy.signal as signal
import scipy.special as special
import lib
import cache_stars

def v_disp(v, mp):
    mp = mp * np.ones(len(v))
    sigma_sq_3d = 0.0
    for dim in range(3):
        sigma_sq_3d += np.sum(v[:,dim]**2*mp)
    m_tot = np.sum(mp)
    if np.sum(mp) <= 0: m_tot = 0 # Doesn't matter, it'll be nonsense anyway
    sigma_sq_3d /= 3*m_tot
    
    return np.sqrt(sigma_sq_3d)

def capped_rel_max(x, x_min, debug=False):
    i = signal.argrelextrema(x, np.greater)[0]
    if len(i) <= 1:
        return x_min
    else:
        return np.max(x[i[1:]])

def vmax(x, mp, eps):
    if len(x) == 0: return 0, 0
    r = np.sqrt(np.sum(x**2, axis=1))
    order = np.argsort(r)
    r = r[order]

    m = np.cumsum(mp*np.ones(len(r)))
    # Some shenanigans are needed to prevent putting Vmax at the
    # first or second particles.
    v = 2.074e-3 * m**0.5 * (r**2 + eps**2)**-0.25 
    vmax_min = 2.074e-3 * mp**0.5 * eps**-0.5
    vmax = capped_rel_max(v, vmax_min)

    h = eps / 0.357
    A, beta = 0.172, -0.522
    v_debias = v / (1 - np.exp(-(A*h/np.sqrt(r**2 + eps**2))**beta))

    vmax_debias = capped_rel_max(v_debias, vmax_min)
    return vmax, vmax_debias

def m23_S_moments(n_peak):
    z90 = 1.2816
    log_n = np.log10(n_peak)
    p9 = -0.3473 -0.3756*log_n
    p5 = -0.5054 -0.5034*log_n
    p1 = 0.0526 - 0.8121*log_n

    return p1, p5, p9

def m23_S(n_peak, mu):
    p1, p5, p9 = m23_S_moments(n_peak)

    log_mu = np.log10(mu)
    S = np.zeros(len(mu))
    low = log_mu < p5

    d_high = (log_mu - p5)/(p9 - p5)
    d_low = (log_mu - p5)/(p5 - p1)

    S[low] = (1+special.erf(d_low[low]*1.2816/np.sqrt(2)))/2
    S[~low] = (1+special.erf(d_high[~low]*1.2816/np.sqrt(2)))/2

    return S

def get_m23_weight(n_peak, mu):
    return 1/m23_S(n_peak, mu)

def get_m23_v_conv_lim(npeak):
    x = np.log10(npeak)
    b2, b1, b0 = -0.01853, 0.3861, 1.6597
    return 10**(x*x*b2 + x*b1 + b0)
    
def galaxy_catalog(suite, i_host, model_names):
    sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)
    n_model = len(model_names)

    gal_dir = os.path.join(sim_dir, "galaxies")
    if not os.path.exists(gal_dir):
        os.makedirs(gal_dir)
        
    scale = symlib.scale_factors(sim_dir)
    param = symlib.simulation_parameters(sim_dir)
    eps = param["eps"]/param["h100"] * scale
    mp = param["mp"]/param["h100"]

    print("Starting I/O")
    part = symlib.Particles(sim_dir, include="E")
    sf, hist = symlib.read_symfind(sim_dir)

    stars = [None]*n_model
    gal_hist = [None]*n_model
    state = None
    
    for im in range(n_model):
        print("Reading model %d, %s" % (im, model_names[im]))
        stars[im], gal_hist[im] = cache_stars.read_stars(
            model_names[im], suite, i_host)

            
    print("Tagging done")

    r_half = np.zeros((n_model,) + sf.shape, dtype=np.float32)
    m_star = np.zeros((n_model,) + sf.shape, dtype=np.float32)
    x0_star = np.zeros((n_model,) + sf.shape + (3,), dtype=np.float32)
    v0_star = np.zeros((n_model,) + sf.shape + (3,), dtype=np.float32)
    m_dyn = np.zeros((n_model,) + sf.shape, dtype=np.float32)
    v_disp_3d_star = np.zeros((n_model,) + sf.shape, dtype=np.float32)

    # DM properties, same for every star model.
    v_disp_3d_dm = np.zeros(sf.shape, dtype=np.float32)
    vmax_dm = np.zeros(sf.shape, dtype=np.float32)
    vmax_dm_debias = np.zeros(sf.shape, dtype=np.float32)
    m23_weight = np.zeros(sf.shape, dtype=np.float32)
    m23_m_conv = np.zeros(sf.shape, dtype=bool)
    m23_v_conv = np.zeros(sf.shape, dtype=bool)

    # infall properties
    m_star_i = np.zeros((n_model, len(sf)))
    r_half_i = np.zeros((n_model, len(sf)))

    for im in range(n_model):
        m_star_i[im] = np.asarray(gal_hist[im]["m_star_i"],
                                  dtype=np.float32)
        r_half_i[im] = np.asarray(gal_hist[im]["r_half_3d_i"],
                                  dtype=np.float32)    
        
    for snap in range(len(scale)):
        if np.sum(sf["ok"][1:,snap]) == 0:
            continue
        if snap % 10 == 0:
            print("   ", snap)

        p = part.read(snap, mode="all")

        ok = sf["ok"][:,snap]
        m23_weight[ok,snap] = get_m23_weight(
            hist["mpeak"][ok]/mp,
            sf["m"][ok,snap]/hist["mpeak"][ok]
        )

        m, npeak = sf["m"][:,snap], hist["mpeak"]/mp
        m23_m_conv[:,snap] = m > get_m23_v_conv_lim(8*npeak)*mp
        m23_v_conv[:,snap] = m > get_m23_v_conv_lim(npeak)*mp

        for i in range(1, len(sf)):
            if not sf["ok"][i,snap]: continue

            x, v, ok = p[i]["x"], p[i]["v"], p[i]["ok"]
            x_all = x - sf["x"][i,snap]
            x, v = x - sf["x"][i,snap], v - sf["v"][i, snap]
            smooth = p[i]["smooth"]
            is_bound = np.zeros(len(x), dtype=bool)
            idx = np.arange(len(x))[ok]

            E = p[i]["E"]
            is_bound = E < 0

            for im in range(n_model):
                mp_star_i = stars[im][i]["mp"][is_bound[smooth]]
                x_star = x[is_bound & smooth]
                v_star = v[is_bound & smooth]
                v_star_w, x_star_w = np.copy(v_star), np.copy(x_star)
                for dim in range(3):
                    x_star_w[:,dim] *= mp_star_i
                    v_star_w[:,dim] *= mp_star_i

                if np.sum(mp_star_i) > 0:
                    x0_star[im,i,snap] = np.sum(x_star_w, axis=0)
                    x0_star[im,i,snap] /= np.sum(mp_star_i)
                    v0_star[im,i,snap] = np.sum(v_star_w, axis=0)
                    v0_star[im,i,snap] /= np.sum(mp_star_i)

                r = np.sqrt(np.sum((x - x0_star[im,i,snap])**2, axis=1))
                r = r[is_bound & smooth]
                order = np.argsort(r)
                r = r[order]
                mp_star_i = mp_star_i[order]

                x_all = x_all - x0_star[im,i,snap]
                r_all = np.sum(x_all**2, axis=1)
            
                m_star[im,i,snap] = np.sum(mp_star_i)
                c_mass = np.cumsum(mp_star_i)
                if len(c_mass) != 0:
                    r_half[im,i,snap] = r[np.searchsorted(c_mass,c_mass[-1]/2)]
                    m_dyn[im,i,snap] = np.sum(r_all < r_half[im,i,snap])*mp

                x0_star[im,i,snap] += sf["x"][i,snap]
                v0_star[im,i,snap] += sf["v"][i,snap]
                v_disp_3d_star[im,i,snap] = v_disp(v_star, mp_star_i)
                
            # dm property calculations
            v_disp_3d_dm[i,snap] = v_disp(v[is_bound], mp)
            vmax_dm[i,snap], vmax_dm_debias[i,snap] = vmax(
                    x[is_bound], mp, eps[snap])

    for im in range(n_model):
        file_name = os.path.join(gal_dir, "gal_cat_%s.dat" % model_names[im])
        print("file_name", file_name)
        n_snap, n_halo = sf.shape
        with open(file_name, "w+") as fp:
            m_star_i[im].tofile(fp)
            r_half_i[im].tofile(fp)
            m_star[im].tofile(fp)
            r_half[im].tofile(fp)
            m_dyn[im].tofile(fp)
            x0_star[im].tofile(fp)
            v0_star[im].tofile(fp)
            v_disp_3d_dm.tofile(fp) # dm property
            v_disp_3d_star[im].tofile(fp)
            vmax_dm.tofile(fp) # dm property
            vmax_dm_debias.tofile(fp) # dm property
            m23_weight.tofile(fp) # dm property
            m23_m_conv.tofile(fp) # dm property
            m23_v_conv.tofile(fp) # dm property
            sf["ok"].tofile(fp) # dm property

def main():
    suites = ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup",
              "SymphonyLCluster", #"SymphonyCluster",
              "SymphonyMilkyWayHR",
              "MWest"]
    
    no_um = ["MWest", "SymphonyMilkyWayHR", "SymphonyCluster"]

    method_names = ["fid_dwarf", "r=0.0038", "r=0.0060", "r=0.0094",
                    "r=0.015",
                    "r=0.024", "r=0.038", "r=0.060", "r=0.15", "r=1"]
    method_names_no_um = ["fid_dwarf_no_um", "r=0.0038", "r=0.0060_no_um",
                          "r=0.0094_no_um", "r=0.015_no_um", "r=0.024_no_um",
                          "r=0.038_no_um", "r=0.060_no_um", "r=1_no_um"]
    
    for suite in suites:
        if suite in no_um:
            model_names = method_names_no_um
        else:
            model_names = method_names            
        
        n_hosts = symlib.n_hosts(suite)
        for i_host in range(n_hosts):            
            galaxy_catalog(suite, i_host, model_names)
        
if __name__ == "__main__": main()
