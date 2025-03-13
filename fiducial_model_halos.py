import cache_stars
import palette
from palette import pc
import symfind
import numpy as np

def main():
    suites = ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup",
              "SymphonyLCluster"]

    r_rvir = 10**np.linspace(-2, 0, 40)
    
    fig, ax = plt.subplots(2, 2, figsize=(16, 16), sharex=True)
    ax[0,1].set_xlim(0.01, )1
    ax[1,1].set_xlim(0.01, 1)
    ax[0,1].set_xlabel(r"$r/R_{\rm vir}$")
    ax[1,1].set_xlabel(r"$r/R_{\rm vir}$")
    ax[0,1].set_ylabel(r"$\rho / (M_\star R_{\rm vir}^{-3})$")
    ax[1,1].set_ylabel(r"$\rho_{\rm major}/\rho_{\rm tot}$")
    ax[0,0].set_ylabel(r"$c/a$")
    ax[1,0].set_ylabel(r"$[Fe/H]$")
    
    for suite in suites:
        n_hosts = symlib.n_hosts(suite)
        rho, rho_big, rho_small = 
        for i_host in range(n_hosts):
            pass

    fig.savefig("plots/fiducial_halos.png")

if __name__ == "__main__": main()
