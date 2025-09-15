import py21cmfast as p21c

print(f"Using 21cmFAST version {p21c.__version__}")

def run_lightcone(fstar_10, alpha_star, fesc_10, alpha_esc, t_star, Mturn, L_X, seed, zmin, zmax, box_dim=250,HII_DIM=128):
    """
    Function to create a 21cmFast lightcone for given astrophysical parameters
    
    In:
    fstar_10 ()
    """
    
    lightcone = p21c.run_lightcone(
    redshift=zmin,
    max_redshift=zmax,
    user_params=p21c.UserParams(
        BOX_LEN=box_dim,
        HII_DIM=HII_DIM,
        DIM=int(HII_DIM * 2),
    ),
    global_quantities=("brightness_temp", "density", "velocity", "xH_box"),
    astro_params={"F_STAR10":fstar_10, "ALPHA_STAR":alpha_star,"F_ESC10":fesc_10, 
                  "t_STAR":t_star, "M_TURN":Mturn, "L_X":L_X},
    cosmo_params=p21c.CosmoParams(),
    random_seed=seed,
    )
    
    return lightcone
        
def make_power_spectra(brigthness_temp, kbins=10, box_len, z_min, z_max):
    """
    function to make a cylindrical power spectrum from a bightness temperature lightcone
    """
    z_mid = (z_min+z_max)/2
    L_para = cosmo.comoving_distance(z_min) - cosmo.comoving_distance(z_max)
    theta = box_len/cosmo.comoving_distance(z_min)
    L_perp = cosmo.comoving_distance(z_mid) * theta_FOV 
    print(L_perp)
    
    p, k = t2c.power_spectrum_1d(brigthness_temp, kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=False)
    
    return p, k
