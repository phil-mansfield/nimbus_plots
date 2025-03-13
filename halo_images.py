import numpy as np
import symlib
import matplotlib.pyplot as plt
import palette
from palette import pc
import lib
import matplotlib as mpl
import cache_stars

SUITES = ["SymphonyMilkyWay"]
METHODS = ["fid_dwarf", "r=0.005", "r=0.008", "r=0.015", "r=0.025", "r=0.05"]
METHODS_NO_UM = [m + "_no_um" for m in METHODS]

def image_grid(suite, i_host):
    if suite == "SymphonyMilkyWayHR":
        methods = METHODS_NO_UM
    else:
        methods = METHODS

    fig, ax = plt.subplots(3, len(methods), sharex=True, sharey=True,
                           figsize=(48, 24))

    sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)
    rs, hist = symlib.read_rockstar(sim_dir)
    host = rs[0]
    lim = 1.5*host["rvir"][-1]

    part = symlib.Particles(sim_dir)
    p = np.hstack(part.read(235, mode="stars")[1:])
    norm = mpl.colors.LogNorm(vmin=1, vmax=1e6)
    kwargs = {"range": [[-lim, lim], [-lim, lim]],
              "norm": norm, "cmap": "inferno", "bins": 200}

    dims = ((0, 1), (1, 2), (2, 0))
    
    for icol in range(len(methods)):
        stars, gal_hist = cache_stars.read_stars(
            methods[icol], suite, i_host
        )
        stars = np.hstack(stars[1:])

        for irow in range(3):            
            ax[irow,icol].set_xlim(-lim, lim)
            ax[irow,icol].set_ylim(-lim, lim)
            
            ax[irow, icol].hist2d(
                p["x"][:,dims[irow][0]], p["x"][:,dims[irow][1]],
                weights=stars["mp"], **kwargs
            )
            ax[irow,icol].set_facecolor("k")
            
    plt.savefig("plots/image_grids/grid_%s_%02d.png" % (suite, i_host))

def make_all_images():
    palette.configure(False)
    for suite in SUITES:
        n_hosts = symlib.n_hosts(suite)
        for i_host in range(n_hosts):
            print(suite, i_host)
            image_grid(suite, i_host)
            
if __name__ == "__main__":
    #main()
    make_all_images()
