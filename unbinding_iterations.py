import lib
import symlib
import palette
from palette import pc
import time
import gravitree
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import scipy.stats as stats

out_name = "data/unbinding_iters.txt"

def compute():
    base_dir = lib.base_dir
    suite = "SymphonyMilkyWay"

    iters = 20

    t0 = time.time()

    fp = open(out_name, "w+")
    
    for i_host in range(symlib.n_hosts(suite)):
        print(i_host)
        t1 = time.time()
        if t1 - t0 < 60:
            print("%.2f sec" % (t1-t0))
        elif t1 - t0 < 3600:
            print("%.2f min" % ((t1-t0)/60))
        else:
            print("%.2f hr" % ((t1-t0)/3600))
            
        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)
        param = symlib.simulation_parameters(sim_dir)
        mp, eps = param["mp"]/param["h100"], param["eps"]/param["h100"]
        sf, _ = symlib.read_symfind(sim_dir)
        i_sub = np.where(sf["ok"][:,-1])[0]

        pp = symlib.Particles(sim_dir).read(235)
        
        for i in i_sub:
            t2 = time.time()
            
            p = pp[i]
            dx = p["x"] - sf["x"][i,-1]
            dv = p["v"] - sf["v"][i,-1]
            
            ke = np.sum(dv**2/2, axis=1)

            n_bound = np.zeros(iters)
            n_bound[0] = len(p)
            ok = np.ones(len(p), dtype=bool)
            
            for j in range(1, iters):
                E = gravitree.binding_energy(
                    dx, dv, mp, eps, n_iter=1, ok=ok)
                _ok = E < 0

                n_bound[j] = np.sum(_ok)

                if n_bound[j] == n_bound[j-1]: break
                ok = _ok
                
            t3 = time.time()
                
            for _j in range(j, iters):
                n_bound[_j] = n_bound[j]
                
            print("%2d %3d %.4f" % (i_host, i, t3 - t2), end=" ", file=fp)
            for j in range(iters):
                print("%5d" % n_bound[j], end=" ", file=fp)
            print(file=fp)

    fp.close()
            
def plot():    
    fig1, ax1 = plt.subplots(1, 3, figsize=(21,7))

    i_host, i_sub, dt = np.loadtxt(out_name, usecols=(0, 1, 2)).T
    n_bound = np.loadtxt(out_name, usecols=np.arange(3, 23, dtype=int))

    delta = n_bound[:,:-1] - n_bound[:,1:]
    frac = delta/n_bound[:,:-1]

    frac_nan = np.copy(frac)
    ok = frac_nan == 0
    frac_nan[ok] = np.nan
    
    f_ub = np.zeros(n_bound.shape)
    for i in range(len(f_ub)):
        f_ub[i] = 1 - n_bound[i,-1]/n_bound[i]
    ok = f_ub == 0
    f_ub_nan = np.copy(f_ub)
    f_ub_nan[ok] = np.nan
    
    n_iter = np.zeros(len(n_bound))
    
    for i in range(len(n_bound)):
        _n_iter = np.where(delta[i] == 0)[0]
        n_iter[i] = 20 if len(_n_iter) == 0 else _n_iter[0] + 1

    n_iter = np.sort(n_iter)
    print(n_iter)
    idx = (np.arange(len(n_iter)) + 1)[::-1]

    #bins = np.linspace(-0.5, 20.5, 21)
    
    ax1[0].plot(n_iter, idx/idx[0], c=pc("k"))
    ax1[0].set_yscale("log")
    ax1[0].set_xlabel(r"$n_{\rm iter}$")
    ax1[0].set_ylabel(r"${\rm Pr}(\leq n_{\rm iter})$")

    i_iter = np.linspace(0, 19, 20)
    ii_iter = np.linspace(1, 19, 19)
    
    for iter, x, x_nan, i_ax in [(i_iter, f_ub, f_ub_nan, 1), (ii_iter, frac, frac_nan, 2)]:
        print(iter.shape, x.shape)
        ax1[i_ax].fill_between(iter, np.quantile(x, 0.5-0.68/2, axis=0),
                               np.quantile(x, 0.5+0.68/2, axis=0),
                               color=pc("b"), alpha=0.2)
        ax1[i_ax].plot(iter, np.quantile(x, 0.5, axis=0), c=pc("b"),
                       label=r"${\rm all\ subhalos}$")
        ax1[i_ax].fill_between(iter, np.nanquantile(x_nan, 0.5-0.68/2, axis=0),
                               np.nanquantile(x_nan, 0.5+0.68/2, axis=0),
                               color=pc("r"), alpha=0.2)
        ax1[i_ax].plot(iter, np.nanquantile(x_nan, 0.5, axis=0), c=pc("r"),
                       label=r"${\rm iterating\ subhalos}$")
        
    ax1[1].legend(loc="upper right", fontsize=17)
    ax1[1].set_yscale("log")
    ax1[1].set_xlabel(r"$n_{\rm iter}$")
    ax1[1].set_ylabel(r"$f_{\rm unbound}$")

    ax1[2].set_yscale("log")
    ax1[2].set_xlabel(r"$n_{\rm iter}$")
    ax1[2].set_ylabel(r"$n_{i+1}/n_i$")
    
    fig1.savefig("plots/unbinding/n_iter_hist.png")

def scatter():
    i_host, i_sub, dt = np.loadtxt(out_name, usecols=(0, 1, 2)).T
    n_bound = np.loadtxt(out_name, usecols=np.arange(3, 23, dtype=int))
    
    base_dir = lib.base_dir
    suite = "SymphonyMilkyWay"

    n_iter = []
    npeak = []
    
    for j_host in range(symlib.n_hosts(suite)):
        sim_dir = symlib.get_host_directory(base_dir, suite, j_host)
        sf, hist = symlib.read_symfind(sim_dir)

        param = symlib.simulation_parameters(sim_dir)
        mp = param["mp"]/param["h100"]
        
        ok = i_host == j_host
        _n_bound = n_bound[ok,:]
        
        delta = _n_bound[:,:-1] - _n_bound[:,1:]
        
        _n_iter = np.zeros(len(_n_bound))
        for i in range(len(_n_bound)):
            __n_iter = np.where(delta[i] == 0)[0]
            _n_iter[i] = 20 if len(__n_iter) == 0 else __n_iter[0] + 1

        n_iter.append(_n_iter)
        ok = sf["ok"][:,-1]
        npeak.append(hist["mpeak_pre"][ok]/mp)

    n_iter = np.hstack(n_iter)
    npeak = np.hstack(npeak)
    #dt = dt[:len(npeak)]
    
    n_iter += random.random(len(n_iter)) - 0.5
    
    fig, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].plot(npeak, n_iter, ".", c="k", alpha=0.2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$n_{\rm peak}$")
    ax[0].set_ylabel(r"$n_{\rm iter}$")
    ax[0].set_xlim((1e3, None))
    
    ax[1].plot(npeak, dt/npeak * 1e6, ".", c="k", alpha=0.2)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$n_{\rm peak}$")
    ax[1].set_ylabel(r"$\delta t/n_{\rm peak}\ ({\rm \mu s})$")
    ax[1].set_xlim((1e3, None))
    
    fig.savefig("plots/unbinding/scatter.png")
        
if __name__ == "__main__":
    palette.configure(True)
    
    #compute()
    plot()
    scatter()
