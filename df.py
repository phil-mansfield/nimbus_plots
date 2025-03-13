import lib
import symlib
import numpy as np
import matplotlib.pyplot as plt
from palette import pc
import palette

class EnergyProfile(object):
    def __init__(self, params, p):
        rmax, vmax, pe_vmax, order = symlib.profile_info(params, p["x"])
        v_vmax = p["x"] / vmax
        ke_vmax = np.sum(v_vmax**2, axis=1) / 2
        E = ke_vmax - pe_vmax

        r = np.sqrt(np.sum(p["x"]**2, axis=1))
        self.E = E[order]
        self.r = r[order]

        self.E_snap = np.zeros(np.max(p["snap"]) + 1)
        for snap in range(len(self.E_snap)):
            ok = (p["snap"] == snap) & (p["owner"] == 0)
            self.E_snap = np.median(E[ok])

    # This technically works.
    def E(self, r):
        return self.E[np.searchsorted(self.r, r)]
    
def z0_disrupted():
    fig, ax = plt.subplots()

    m, r, v = [], [], []
    for i_host in range(symlib.n_hosts(lib.suite)):
        print("host", i_host)
        sim_dir = symlib.get_host_directory(lib.base_dir, lib.suite, i_host)
        params = symlib.simulation_parameters(sim_dir)
        rs, hist = symlib.read_rockstar(sim_dir)
        sf, hist = symlib.read_symfind(sim_dir)
        
        part = symlib.Particles(sim_dir)
        snap = 235
        #p_host = part.read(snap, mode="all", halo=0)
        p = part.read(snap, mode="smooth")
        core_idx = part.core_indices(snap, mode="smooth")
    
        for i in range(1, len(sf)):
            if sf[i,snap]["ok"] or snap < hist["merger_snap"][i] or hist["preprocess"][i] != -1 or hist["mpeak"][i]/lib.mp < 3e3 or hist["merger_ratio"][i] > 1: continue
            cx = p[i]["x"][core_idx[i]]
            rr = np.sqrt(np.sum(cx**2, axis=1))
            r.append(np.median(rr) / rs[0,hist["merger_snap"][i]]["rvir"])
            cv = p[i]["v"][core_idx[i]]
            vv = np.sqrt(np.sum(cv**2, axis=1))
            #v_dot_r = np.sum(cx*cv, axis=1)
            #vr = v_dot_r / rr
            #v.append(np.abs(np.median(vr)))
            v.append(np.median(vv))
            m.append(hist["merger_ratio"][i])
    
    m, r, v = np.array(m), np.array(r), np.array(v)

    ok20 = m > 0.20
    ok15 = (m > 0.15) & (m < 0.2)
    ok10 = (m > 0.10) & (m < 0.15)
    ok = (~ok20) & (~ok15) & (~ok10)
    ax.plot(r[ok], v[ok], "o", c="k")
    ax.plot(r[ok10], v[ok10], "o", c=pc("b"), label=r"$m/M > 0.10$")
    ax.plot(r[ok15], v[ok15], "o", c=pc("o"), label=r"$m/M > 0.15$")
    ax.plot(r[ok20], v[ok20], "o", c=pc("r"), label=r"$m/M > 0.20$")
    ax.legend(loc="upper left", fontsize=17)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\delta v$")
        
    fig.savefig("plots/sf_z0_disrupted.png")

    ax.cla()
    ax.plot(m, r, "o", c=pc("r"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ylo, yhi = ax.get_ylim()
    plt.plot([0.15, 0.15], [ylo, yhi], "--", c="k")
    ax.set_ylim(ylo, yhi)

    ax.set_xlabel(r"$(m/M_{\rm host})(z_{\rm infall})$")
    ax.set_ylabel(r"$r_{\rm med}(z=0)/R_{\rm vir}(z_{\rm infall})$")
    fig.savefig("plots/sf_z0_disrupted_trend.png")
    
def main():
    palette.configure(False)
    z0_disrupted()


if __name__ == "__main__": main()
