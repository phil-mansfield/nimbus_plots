import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc
import symlib
import lib

def main():
    palette.configure(False)
    
    suites = ["SymphonyLMC", "SymphonyMilkyWay",
              "SymphonyGroup", "SymphonyLCluster"]

    n0 = []
    n = []
    
    for suite in suites:
        n_host = symlib.n_hosts(suite)
        for i_host in range(n_host):
            sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)

            part = symlib.Particles(sim_dir)
            n0.append(part.part_info.part_hd.n0)
            n.append(part.part_info.part_hd.sizes)
            
    n, n0 = np.sort(np.hstack(n)), np.sort(np.hstack(n0))
    count = np.arange(len(n))+1

    plt.plot(n0, count, c=pc("r"), label=r"$n_{\rm smooth}$")
    plt.plot(n, count, c=pc("b"), label=r"$n_{\rm all}$")
    plt.legend(loc="lower right")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$n_p$")
    plt.ylabel(r"$N(< n_p)$")
    ylo, yhi = plt.ylim()
    plt.ylim(ylo, yhi)
    plt.plot([300, 300], [ylo, yhi], "--", c=pc("a"), lw=2)

    plt.savefig("plots/n0_histogram.pdf")
    
if __name__ == "__main__": main()
