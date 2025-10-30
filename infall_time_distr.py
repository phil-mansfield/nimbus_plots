import numpy as np
import symlib
import lib

def main():
    suite = "SymphonyLCluster"
    n_host = symlib.n_hosts(suite)

    a_in = []
    
    for i_host in range(n_host):
        sim_dir = symlib.get_host_directory(lib.base_dir, suite, i_host)
        scale = symlib.scale_factors(sim_dir)
        rs, hist = symlib.read_rockstar(sim_dir)

        a_in.append(scale[hist["first_infall_snap"]])

    a_in = np.log10(np.hstack(a_in))


    print("# First-infall redshift distribution in %s" % suite)
    print("# 0 - quantile")
    print("# 1 - log10()")
    q = [0.5 - 0.95/2, 0.5 - 0.68/2, 0.5, 0.5 + 0.68/2, 0.5 + 0.95/2]
    for i in range(len(q)):
        print("%3f %.3f" % (q[i], np.quantile(a_in, q[i])))

        

if __name__ == "__main__": main()
