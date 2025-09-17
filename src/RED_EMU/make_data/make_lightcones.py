import py21cmfast as p21c
import tools21cm as t2c

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)

print(f"Using 21cmFAST version {p21c.__version__}")

def run_coeval(r_bubble, eta, Tvir, seed, z, box_dim=250,HII_DIM=128):
    """
    Function to create a 21cmFast lightcone for given astrophysical parameters
    
    In:
    fstar_10 ()
    """
    
    lightcone = p21c.run_coeval(
    redshift=z,
    user_params=p21c.UserParams(
        BOX_LEN=box_dim,
        HII_DIM=HII_DIM,
        DIM=int(HII_DIM * 2),
    ),
    #global_quantities=("brightness_temp", "density", "velocity", "xH_box"),
    astro_params={"R_BUBBLE_MAX":r_bubble, "HII_EFF_FACTOR":eta,"ION_Tvir_MIN":Tvir},
    cosmo_params=p21c.CosmoParams(),
    random_seed=seed,
    )
    
    return lightcone
        
def make_power_spectra(brigthness_temp, box_len, z, kbins=10):
    """
    function to make a cylindrical power spectrum from a bightness temperature lightcone
    """
    """z_mid = (z_min+z_max)/2
    L_para = (cosmo.comoving_distance(z_min) - cosmo.comoving_distance(z_max)).value
    
    theta = box_len/cosmo.comoving_distance(z_min)
    L_perp = (cosmo.comoving_distance(z_mid) * theta).value
    
    box_dims = [L_perp,L_perp,L_para]"""
    
    p, k = t2c.power_spectrum_1d(brigthness_temp, kbins=kbins, binning = 'log')# quick fix but need to calculate the box dims prioperly, box_dims=box_dims, return_n_modes=False)
    
    return p, k
