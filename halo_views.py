import kde_sampler
import cache_stars
import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import time
import symlib
import matplotlib.colors as colors
import matplotlib as mpl

def main():
    mpl.rc("font", size=15)
    
    base_dir = "/sdf/home/p/phil1/ZoomIns"
    #suite = "SymphonyMilkyWayHR"
    suite = "SymphonyMilkyWay"
    model = "fid_dwarf"

    k = 8
    
    batch_size = int(1e7)
    mp_star_mean = 1.0
    mp_batch = np.ones(int(1.5*batch_size))*mp_star_mean
    
    fig, ax = plt.subplots(2, 4, figsize=(32, 16))
    
    xlim = 150
    vlim = 250
    
    bins_1 = 200
    bins_2 = 200
    
    n_hosts = symlib.n_hosts(suite)
    for i_host in range(n_hosts):
        #fig.clf()
        
        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

        print("i_host = %d/%d" % (i_host+1, n_hosts))

        print("  I/O")
        
        t0 = time.time()
        part = symlib.Particles(sim_dir, include=["E"])
        p = part.read(235, mode="stars")
        sf, hist = symlib.read_symfind(sim_dir)
        stars, gal_hist = cache_stars.read_stars(model, suite, i_host)
        
        t1 = time.time()
        
        print("    %.2f s" % (t1 - t0))
        print("  sampling")

        kernel = kde_sampler.M4Kernel(dim=6)

        t2 = time.time()

        xx_1 = np.zeros((bins_1, bins_1))
        xv_1 = np.zeros((bins_1, bins_1))
        vv_1 = np.zeros((bins_1, bins_1))
        rv_1 = np.zeros((bins_1, bins_1))

        xx_2 = np.zeros((bins_2, bins_2))
        xv_2 = np.zeros((bins_2, bins_2))
        vv_2 = np.zeros((bins_2, bins_2))
        rv_2 = np.zeros((bins_2, bins_2))

        for i in range(1, len(sf)):

            # This halo is a Rockstar error
            if len(p[i]) < 50: continue

            coord = np.zeros((len(p[i]), 6))
            coord[:,:3], coord[:,3:] = p[i]["x"], p[i]["v"]
            mp = stars[i]["mp"]

            if np.sum(mp) == 0: continue
            
            update_grids(xx_1, xv_1, vv_1, rv_1, xlim, vlim, coord, mp)

            mean, norm = sigma_clip_norm(coord, mp, 2.0)
            
            coord -= mean
            coord /= norm

            n_star = int(np.sum(mp))/mp_star_mean
            batches = int(np.ceil(n_star/batch_size))
            
            sampler = kde_sampler.KDESampler(kernel, k, coord, mp)
            print("       %2d: %.2g stars %d batches" % (i, n_star, batches))
            
            for j in range(batches):
                print(" "*12, j+1, "/", batches)
                if (j+1)*batch_size > n_star:
                    n_sample = n_star - batch_size*j
                else:
                    n_sample = batch_size
                    
                sample = sampler.sample(n_sample)

                sample *= norm
                sample += mean

                update_grids(xx_2, xv_2, vv_2, rv_2, xlim, vlim, sample,
                             mp_batch[:len(sample)])

        t3 = time.time()
        print("    %.2f s" % (t3 - t2))

        cmin, cmax = 1, 1e6
        norm = colors.LogNorm(cmin, cmax)
        xx_1 = np.maximum(xx_1, cmin)
        ax[0,0].imshow(xx_1.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-xlim, xlim, -xlim, xlim])
        ax[0,0].set_xlabel(r"$x\ ({\rm kpc})$")
        ax[0,0].set_ylabel(r"$y\ ({\rm kpc})$")

        xx_2 = np.maximum(xx_2, cmin)
        ax[1,0].imshow(xx_2.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-xlim, xlim, -xlim, xlim])
        ax[1,0].set_xlabel(r"$x\ ({\rm kpc})$")
        ax[1,0].set_ylabel(r"$y\ ({\rm kpc})$")
        
        ###

        xv_1 = np.maximum(xv_1, cmin)
        ax[0,1].imshow(xv_1.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-xlim, xlim, -vlim, vlim])
        ax[0,1].set_xlabel(r"$x\ ({\rm km\,s^{-1}})$")
        ax[0,1].set_ylabel(r"$v_x\ ({\rm km\,s^{-1}})$")

        xv_2 = np.maximum(xv_2, cmin)
        ax[1,1].imshow(xv_2.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-xlim, xlim, -vlim, vlim])
        ax[1,1].set_xlabel(r"$x\ ({\rm km\,s^{-1}})$")
        ax[1,1].set_ylabel(r"$v_x\ ({\rm km\,s^{-1}})$")

        ###

        vv_1 = np.maximum(vv_1, cmin)
        ax[0,2].imshow(vv_1.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-vlim, vlim, -vlim, vlim])
        ax[0,2].set_xlabel(r"$v_x\ ({\rm km\,s^{-1}})$")
        ax[0,2].set_ylabel(r"$v_y\ ({\rm km\,s^{-1}})$")

        vv_2 = np.maximum(vv_2, cmin)
        ax[1,2].imshow(vv_2.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-vlim, vlim, -vlim, vlim])
        ax[1,2].set_xlabel(r"$v_x\ ({\rm km\,s^{-1}})$")
        ax[1,2].set_ylabel(r"$v_y\ ({\rm km\,s^{-1}})$")

        ###

        rv_1 = np.maximum(rv_1, cmin)
        ax[0,3].imshow(rv_1.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[-xlim, xlim, -vlim, vlim])
        ax[0,3].set_xlabel(r"$r\ ({\rm km\,s^{-1}})$")
        ax[0,3].set_ylabel(r"$v_r\ ({\rm km\,s^{-1}})$")

        rv_2 = np.maximum(rv_2, cmin)
        ax[1,3].imshow(rv_2.T, cmap="viridis", norm=norm, aspect="auto",
                       origin="lower", extent=[0, xlim, -vlim, vlim])
        ax[1,3].set_xlabel(r"$r\ ({\rm km\,s^{-1}})$")
        ax[1,3].set_ylabel(r"$v_r\ ({\rm km\,s^{-1}})$")

        
        fig.savefig("plots/halo_views/%02d_%s.png" % (i_host, model))

        break
        

def update_grids(xx, xv, vv, rv, xlim, vlim, coord, mp):
    # xx
    H, _, _ = np.histogram2d(coord[:,0], coord[:,1], bins=xx.shape,
                             range=[[-xlim, xlim], [-xlim, xlim]],
                             weights=mp)
    xx += H

    #xv
    H, _, _ = np.histogram2d(coord[:,0], coord[:,3], bins=xv.shape,
                             range=[[-xlim, xlim], [-vlim, vlim]],
                             weights=mp)
    xv += H

    # vv
    H, _, _ = np.histogram2d(coord[:,3], coord[:,4], bins=xv.shape,
                             range=[[-xlim, xlim], [-vlim, vlim]],
                             weights=mp)
    vv += H

    # rv
    r = np.sqrt(np.sum(coord[:,:3]**2, axis=1))
    vr = ((np.sum(coord[:,:3]*coord[:,3:], axis=1)).T/r).T
    H, _, _ = np.histogram2d(r, vr, bins=rv.shape,
                             range=[[0, xlim], [-vlim, vlim]],
                             weights=mp)
    rv += H

def sigma_clip_norm(w, mass, exclude_sigma):
    mean = np.average(w, axis=0, weights=mass)
    norm = np.sqrt(np.average(
        (w - mean)**2, weights=mass, axis=0
    ))
    
    lim_low  = mean - exclude_sigma*norm
    lim_high = mean + exclude_sigma*norm

    dim = w.shape[1]
    mean, norm = np.zeros(dim), np.zeros(dim)
    
    for i in range(dim):
        ok = (w[:,i] > lim_low[i]) & (w[:,i] < lim_high[i])
        mean[i] = np.average(w[ok,i], axis=0, weights=mass[ok])
        norm[i] = np.sqrt(np.average(
            (w[ok,i] - mean[i])**2, weights=mass[ok], axis=0
        ))

    return mean, norm
        
    
if __name__ == "__main__": main()
