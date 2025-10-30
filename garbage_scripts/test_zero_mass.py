import symlib
import numpy as np

def main():
    base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns"
    gal_halo = symlib.GalaxyHaloModel(
        symlib.StellarMassModel(
            symlib.UniverseMachineMStarFit(),
            symlib.DarkMatterSFH()
        ),
        symlib.ProfileModel(
            symlib.Jiang2019RHalf(scatter=0.),
            symlib.PlummerProfile()
        ),
        symlib.MetalModel(
            symlib.Kirby2013Metallicity(),
            symlib.Kirby2013MDF(model_type="gaussian"),
            symlib.FlatFeHProfile(),
            symlib.GaussianCoupalaCorrelation()
        )
    )

    suites = ["MWest", "SymphonyMilkyWay"]

    for suite in suites:
        print("Testing suite %s" % suite)
        
        sim_dir = symlib.get_host_directory(base_dir, suite, 0)

        sf, hist = symlib.read_symfind(sim_dir)
        stars, _, _ = symlib.tag_stars(sim_dir, gal_halo)
        
        for i in range(1, 11):
            print("sub %2d: mpeak = %.4g mstar = %.4g" %
                  (i, hist["mpeak_pre"][i], np.sum(stars[i]["mp"])))
            
if __name__ == "__main__": main()
