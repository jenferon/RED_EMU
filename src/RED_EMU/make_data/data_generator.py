import numpy as np
from RED_EMU.make_data.make_lightcones import run_lightcone, make_power_spectra
import random
import pandas as pd
import matplotlib.pyplot as plt

def simulator(fstar10_bounds, alpha_star_bounds, zmin, zmax, box_dim, nruns, kbins, seed=np.random.seed(), SAVE=True):
    save_labels = pd.DataFrame()
    save_data = np.zeros([nruns,kbins])

    for ii in range(0,nruns):
        #get astro params
        fstar_10 = random.uniform(fstar10_bounds[0],fstar10_bounds[1])
        alpha_star = random.uniform(alpha_star_bounds[0],alpha_star_bounds[1])
        
        astro_params = pd.DataFrame({'f* 10':[fstar_10], 'alpha*':[alpha_star]})
        save_labels =  save_labels._append(astro_params)
        
        #make lightcone
        lightcone = run_lightcone(fstar_10=np.log10(fstar_10), alpha_star=alpha_star, fesc_10=-1.0, 
                                alpha_esc=-0.5, t_star=0.5, Mturn=8.7, L_X=40.5, 
                                seed=seed, zmin=zmin, zmax=zmax, box_dim=box_dim)
        delta_Tb = lightcone.brightness_temp 
        print(np.mean(delta_Tb))
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
    fstar10_bounds = [0.5,0.0001]
    alpha_star_bounds = [0.0,1.5]
    
    ps, k = simulator(fstar10_bounds, alpha_star_bounds, zmin, zmax, box_dim, 100, kbins, SAVE=True)
    



    
    