import symlib
import numpy as np
import scipy.special as special

def main():
    #n_sersics = [0.5, 1, 2, 4, 8]
    n_sersics = [0.5, 1, 2, 4, 8]
    for i in range(len(n_sersics)):
        prof = symlib.DeprojectedSersicProfile(n_sersics[i])
        
        r = 10**np.linspace(-2, 2, 21)
        m = prof.m_enc(2, 3, r)        
        
if __name__ == "__main__": main()
