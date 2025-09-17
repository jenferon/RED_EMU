import numpy as np
from RED_EMU.make_data.make_lightcones import run_lightcone, make_power_spectra
import random
import pandas as pd
import matplotlib.pyplot as plt

def simulator(r_bubble, eta, Tvir, zmin, zmax, box_dim, nruns, kbins, seed=np.random.seed(), SAVE=True, PLOT=False):
    save_labels = pd.DataFrame()
    save_data = np.zeros([nruns,kbins])

    for ii in range(0,nruns):
        
        astro_params = pd.DataFrame({'R_bubble':[r_bubble[ii]], 'Ionising_effciency':[eta[ii]], 'T_vir(min)':[Tvir[ii]]})
        save_labels =  save_labels._append(astro_params)
        
        #make lightcone
        lightcone = run_lightcone(r_bubble[ii], eta[ii], Tvir[ii], 
                                seed=seed, zmin=zmin, zmax=zmax, box_dim=box_dim)
        delta_Tb = lightcone.brightness_temp 
        print(np.mean(delta_Tb))
        if PLOT == True:
            plt.imshow(delta_Tb[:,:,0])
            plt.savefig("/home/ppxjf3/repos/RED_EMU/test_Tb.pdf")
        #run power spectra
        ps, k = make_power_spectra(delta_Tb, box_dim, zmin, zmax, kbins=kbins)
        save_data[ii,:] = ps
        
    if SAVE == True:
        np.save('dataset/training_data', save_data)
        save_labels.to_csv('dataset/training_labels.csv', sep=',', index=False, encoding='utf-8')
    
    return ps, k

if __name__ == "__main__":
    #vary the input parameters within a given range
    zmin = 6.0
    zmax = 6.2
    box_dim = 250
    nruns = 1
    kbins = 10

    #range to sample betweem for each astro param
    r_bubble = []
    eta = []
    Tvir = []
    
    ps, k = simulator(r_bubble, eta, Tvir, zmin, zmax, box_dim, 100, kbins, SAVE=True)
    



    
    