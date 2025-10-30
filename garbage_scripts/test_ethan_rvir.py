import symlib
import numpy as np

def mvir_to_rvir(mvir, a, omega_M=0.286):
    # Units are in little-h.
    omega_L = 1 - omega_M
    Ez = np.sqrt(omega_M/a**3 + omega_L)
    rho_crit = 2.77519737e11*Ez**2
    omega_Mz = (omega_M/a**3)/Ez**2

    x = omega_Mz - 1
    delta_vir = 18*np.pi**2 + 82*x - 39.0*x**2
    rho_vir = rho_crit*delta_vir

    r_phys = (mvir/(rho_vir * (4*np.pi / 3)))**(1.0/3)
    return r_phys * 1e3

print(mvir_to_rvir(520000000.0/0.7, 1))
print()
print(mvir_to_rvir(10**8.75*0.7, 1)/0.7)
print(mvir_to_rvir(10**8.75*0.7, 0.8)/0.7)
print(mvir_to_rvir(10**8.75*0.7, 0.6)/0.7)
print(mvir_to_rvir(10**8.75*0.7, 0.4)/0.7)
print(mvir_to_rvir(10**8.75*0.7, 0.2)/0.7)

def main():
    scale = symlib.scale_factors("SymphonyMilkyWay")
    a = scale[172]
    print("a(snap = 172) = %.4f" % a)
    print("Test halo A, mvir = %.4g Msun, at snap 172 from Census catalog:" %
          11945715000.0)
    print("rvir = 32.196 pkpc")
    print("Test halo A, rvir(mvir, a):")
    print("%.4f pkpc" % (mvir_to_rvir(11945715000.0*0.7, a)/0.7))
    print()
    print("Test halo B, Mvir = %.4g Msun, at snap 172 from rockstar catlog:" %
          (9.7190e+09/0.7))
    print("rvir = 52.897 ckpc/h")
    print("rvir = %.3f" % (52.897/0.7*a))
    print("Test halo B, rvir(mvir, a):")
    print("%.4f pkpc" % (mvir_to_rvir(9.7190e+09, a)/0.7))
    
if __name__ == "__main__": main()
