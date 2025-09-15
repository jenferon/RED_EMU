import py21cmfast as p21c
import tools21cm as t2c

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)

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
        
def make_power_spectra(brigthness_temp, box_len, z_min, z_max, kbins=10):
    """
    function to make a cylindrical power spectrum from a bightness temperature lightcone
    """
    z_mid = (z_min+z_max)/2
    L_para = (cosmo.comoving_distance(z_min) - cosmo.comoving_distance(z_max)).value
    theta = box_len/cosmo.comoving_distance(z_min)
    L_perp = (cosmo.comoving_distance(z_mid) * theta).value
    print(L_perp)
    
    box_dims = [L_para,L_para,L_perp]
    
    p, k = t2c.power_spectrum_1d(brigthness_temp, kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=False)
    
    return p, k
