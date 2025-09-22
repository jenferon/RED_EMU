import numpy as np
from RED_EMU.make_data.make_lightcones import run_coeval, make_power_spectra
from RED_EMU.make_data.sampler import LHS
import random
import pandas as pd
import matplotlib.pyplot as plt

def simulator(sampler, z, box_dim, nruns, kbins=7, seed=np.random.seed(), SAVE=True, PLOT=False, dir='/home/ppxjf3/repos/RED_EMU/src/RED_EMU/make_data/'):
    save_labels = pd.DataFrame()
    save_data = np.zeros([nruns,2,kbins])
    idx=0
    for ii in range(0,nruns-1):
        is_zero = True
        while is_zero == True:
            #make lightcone
            lightcone = run_coeval(sampler[idx,0], sampler[idx,1], sampler[idx,2], 
                                    seed=seed, z=z, box_dim=box_dim)
            delta_Tb = lightcone.brightness_temp 
            if PLOT == True:
                plt.imshow(delta_Tb[:,:,0])
                plt.savefig("/home/ppxjf3/repos/RED_EMU/test_Tb.pdf")
            #run power spectra
            ps, k = make_power_spectra(delta_Tb, box_dim, kbins=kbins)
            print(ps)
            is_zero = not np.any(ps)
            print(is_zero)
            idx+=1
        
        print('saving non zero data in index {}'.format(ii))
        if SAVE == True:
            save_data[ii,:,:] = [ps,k]        
            astro_params = pd.DataFrame({'R_bubble':[sampler[idx,0]], 'Ionising_effciency':[sampler[idx,1]], 'T_vir(min)':[sampler[idx,2]]})
            save_labels =  save_labels._append(astro_params)
        
    if SAVE == True:
        np.save(dir+'dataset/training_data', save_data)
        save_labels.to_csv(dir+'dataset/training_labels.csv', sep=',', index=False, encoding='utf-8')
    
    return ps, k

if __name__ == "__main__":
    #vary the input parameters within a given range
    z = 7.0
    box_dim = 250
    nruns = 500
    kbins = 10

    #range to sample betweem for each astro param
    r_bubble = [1.12,40.32] #Mahida+25
    eta = [5,100] #Schmitt+18
    Tvir = [4,5] #Schmitt+18
    sampler = LHS([r_bubble[0], eta[0], Tvir[0]], [r_bubble[1], eta[1], Tvir[1]], 3, nruns+50)
    
    ps, k = simulator(sampler, z, box_dim, nruns, kbins, SAVE=True, PLOT=False)
    



    
    