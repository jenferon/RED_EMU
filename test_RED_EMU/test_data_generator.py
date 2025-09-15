import numpy as np
from RED_EMU.make_data.make_lightcones import run_lightcone, make_power_spectra
from RED_EMU.make_data.data_generator import simulator 

def test_simulator():
    fstar_10 = 0.01
    fstar10_bounds = [0.01,0.01]
    zmin=6.0
    zmax=6.5
    alpha_star = 1.0
    alpha_star_bounds = [1.0,1.0]
    box_dim = 128
    kbins = 10
    

    lightcone = run_lightcone(fstar_10=np.log10(fstar_10), alpha_star=alpha_star, fesc_10=-1.0, 
                                alpha_esc=-0.5, t_star=0.5, Mturn=8.7, L_X=40.5, 
                                seed=4, zmin=zmin, zmax=zmax, box_dim=box_dim)
    delta_Tb = lightcone.brightness_temp 
    print(delta_Tb.shape)
        
    #run power spectra
    ps, k = make_power_spectra(delta_Tb, box_dim, zmin, zmax, kbins=kbins)
    
    assert(np.allclose(simulator(fstar10_bounds, alpha_star_bounds, zmin, zmax, box_dim, 1, kbins, seed=4), ps, rtol=1e-6, atol=1e-8))
