import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import lib
import symlib
import numpy.random as random

def main():
    palette.configure(False)
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    t, t_dyn = lib.t, lib.t_dyn
    base_dir = lib.base_dir
    
    ax[0].plot(t/t[-1], t_dyn/t[-1], c=pc("p"))
    ax[0].set_xlabel(r"$t/t(z=0)$")
    ax[0].set_ylabel(r"$t_{\rm dyn}(t)/t(z=0)$")

    p = np.polyfit(np.log10(t/t[-1]), np.log10(t_dyn/t[-1]), 1)
    print("t_dyn fit")
    print(p)
    ax[0].plot(t/t[-1], 10**(np.polyval(p, np.log10(t/t[-1]))), "--", c="k")
    
    suite = "SymphonyMilkyWay"
    n_hosts = symlib.n_hosts(suite)
    for i_host in range(n_hosts):
        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

        rs, hist = symlib.read_rockstar(sim_dir)

        m = rs["m"][0]/rs["m"][0,-1]
        ok = m > 0

        r = random.random()
        
        ax[1].plot(t[ok]/t[-1], m[ok], pc("r", 0.2 + r*0.6), lw=1.5)
        ax[2].plot(t[ok]/t[-1], m[ok]**0.06*t_dyn[ok], pc("b", 0.2 + r*0.6), lw=1.5)
        
    ax[1].set_xlabel(r"$t/t(z=0)$")
    ax[1].set_ylabel(r"$m/m(z=0)$")
    ax[2].set_xlabel(r"$t/t(z=0)$")
    ax[2].set_ylabel(r"$f(t)$")

    for i in range(3):
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        
    fig.savefig("plots/mass_loss_approx.png")
    
if __name__ == "__main__": main()
