import symlib
import numpy as np

def main():
    base_dir = "/sdf/home/p/phil1/ZoomIns"
    suite = "SymphonyMilkyWay"

    print("""# 0 - host index
# 1 - mpeak,pre
# 2 - m*,infall
# 3 - m*,z=0
# 4 - merger ratio
# 5 - cvir
# 6 - rvir,infall
# 7 - z,infall """)

    for i_host in range(symlib.n_hosts(suite)):
        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

        gal, hist = symlib.read_galaxies(sim_dir)
        rs, rs_hist = symlib.read_rockstar(sim_dir)

        scale = symlib.scale_factors(sim_dir)
        z = 1/scale - 1
        
        for i in range(len(gal)):

            snap = rs_hist["first_infall_snap"][i]
            print("%2d %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g" %
                  (i_host, rs_hist["mpeak_pre"][i],
                   hist["m_star_i"][i], gal["m_star"][i,-1],
                   rs_hist["merger_ratio"][i],
                   rs["cvir"][i,snap], rs[i,snap]["rvir"], z[snap]))
            
            

if __name__ == "__main__": main()
