import numpy as np
from symlib import *

def main():
    base_dir = "/sdf/home/p/phil1/ZoomIns"
    suite = "SymphonyMilkyWay"
    i_host = 0
    sim_dir = get_host_directory(base_dir, suite, i_host)

    gal_halo = GalaxyHaloModel(
        StellarMassModel(
            UniverseMachineMStar(),
            UniverseMachineSFH()
        ),
        ProfileModel(
            FixedRHalf(0.1, scatter=0.2),
            DeprojectedSersicProfile(n_sersic=2.0)
        ),
        MetalModel(
            Kirby2013Metallicity(),
            Kirby2013MDF(model_type="gaussian"),
            FlatFeHProfile(),
            GaussianCoupalaCorrelation()
        )
    )


    i = 11
    h, hist = read_subhalos(sim_dir)
    snap = hist["first_infall_snap"][i]
    print(h[i,snap]["rvir"])
    stars, gal_hist, _ = tag_stars(sim_dir, gal_halo, target_subs=(i,), seed=1234)
    stars = stars[i]
    print(gal_hist["r_half_2d_i"][i])
    print(gal_hist["r_half_3d_i"][i])

    part = Particles(sim_dir)
    p = part.read(snap, mode="stars", halo=i)
    dx = p["x"] - h[i,snap]["x"]
    dr = np.sqrt(np.sum(dx**2, axis=1))

    xp = np.copy(p["x"])
    for dim in range(3): xp[:,dim] *= stars["mp"]
    ok = ~np.isnan(xp[:,0])
    print(np.sum(xp[ok],axis=0)/np.sum(stars["mp"][ok]))
    print(h["x"][i,snap])

    print("%g" % np.sum(stars["mp"][dr < gal_hist["r_half_3d_i"][i]]))
    print("%g" % np.sum(stars["mp"][dr > gal_hist["r_half_3d_i"][i]]))
    print("%g" % np.sum(stars["mp"]))
    
if __name__ == "__main__":
    main()
    
