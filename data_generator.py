import numpy as np
from make_data.make_lightcones import run_lightcone, make_power_spectra
import random
import pandas as pd

#vary the input parameters within a given range
zmin = 6.0
zmax = 6.2
box_dim = 250
nruns = 1

#range to sample betweem for each astro param
fstar10_bounds = [0.1,0.001]
alpha_star_bounds = [0.0,1.5]
Fesc10_bounds = [0.01,0.5]
alpha_esc_bounds = [-1.0,1.0]

save_labels = pd.DataFrame()

for ii in range(0,nruns):
    #get astro params
    fstar_10 = random.uniform(fstar10_bounds[0],fstar10_bounds[1])
    alpha_star = random.uniform(alpha_star_bounds[0],alpha_star_bounds[1])
    
    astro_params = pd.DataFrame({'f* 10':[fstar_10], 'alpha*':[alpha_star]})
    save_labels =  save_labels._append(astro_params)
    
    #make lightcone
    lightcone = run_lightcone(fstar_10=fstar_10, alpha_star=alpha_star, fesc_10=-1.0, 
                             alpha_esc=-0.5, t_star=0.5, Mturn=8.7, L_X=40.5, 
                             seed=np.random.seed(), zmin=zmin, zmax=zmax, box_dim=box_dim)
    delta_Tb = lightcone.brightness_temp 
    print(delta_Tb.shape)
    
    #run power spectra
    ps, k = make_power_spectra(delta_Tb, box_dim, zmin, zmax)
    
    
    