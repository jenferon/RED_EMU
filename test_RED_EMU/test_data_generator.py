import numpy as np
import py21cmfast as p21c
from RED_EMU.make_data.make_lightcones import make_power_spectra
from RED_EMU.make_data.data_generator import simulator 

def test_simulator():
    r_bubble = 1.12
    eta = 5
    Tvir = 4
    box_dim = 200
    z=7.0
    

    delta_Tb = p21c.run_coeval(
    redshift=z,
    user_params=p21c.UserParams(
        BOX_LEN=box_dim,
        HII_DIM=128,
        DIM=int(128 * 2),
    ),
    astro_params={"R_BUBBLE_MAX":r_bubble, "HII_EFF_FACTOR":eta,"ION_Tvir_MIN":Tvir},
    cosmo_params=p21c.CosmoParams(),
    random_seed=42,
    ).brightness_temp 
        
    #run power spectra
    ps, k = make_power_spectra(delta_Tb)
    samples = np.empty([1,3])
    samples[0,0] = r_bubble
    samples[0,1] = eta
    samples[0,2] = Tvir
    ps_test, k = simulator(samples, 7, box_dim, 1, seed=42, SAVE=False, PLOT=False)
    print(ps_test)
    print(ps)
    assert(np.allclose(ps_test, ps, rtol=1e-6, atol=1e-8))
