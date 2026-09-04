
# Computing Patch Quantities

import numpy as np
import healpy as hp
from astropy.io import fits
import pynkowski as mf   # For Minkowski Functionals
from astropy.utils.data import clear_download_cache
clear_download_cache()

nside = 8
disk_radii = [8,10,12]

# SELECT DESIDERD CMB MASK
mask = hp.read_map("/hydrarepo/mmartin/MFs/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits").astype(np.bool_)
mask = hp.ud_grade(mask, nside_out = 512) # Mask has 0 in points that should not be considered

map8 =  hp.pix2vec(nside, range(12*nside**2))


for k in range(1000):
    if k==970:
        continue
    # Load Planck simulations
    simulation_noise = (hp.read_map(f'/hydrarepo/mmartin/PLANCK/Noise/dx12_v3_smica_noise_mc_{k%300:05}_raw.fits'))*1e6
    simulation_cmb = (hp.read_map(f'/hydrarepo/mmartin/PLANCK/Simulations/dx12_v3_smica_cmb_mc_{k:05}_raw.fits'))*1e6

    # Apply smoothing before calculating MFs
    simulation  = simulation_noise + simulation_cmb
    simulation = hp.smoothing(simulation, fwhm=np.radians(1.))
    simulation = hp.ud_grade(simulation, nside_out=512)
    data  = mf.Healpix(simulation, normalise = False)
    data.mask = mask
    # We get the derivatives here to avoid doing it in the loop
    data.get_first_der()

    
    for j,R in enumerate(disk_radii):

        patch_sigma2 = []
        patch_tau2 = []

        for i in range(12*nside**2):  # Over the map
            # Select point in the 16 map
            vec = (map8[0][i], map8[1][i], map8[2][i]) 
        
            # Select patch disk
            ipix_disc = hp.query_disc(nside=512, vec= vec, radius = np.radians(R)) 
            disk_mask = np.zeros(12*512**2)
            disk_mask[ipix_disc] = 1 # This again sets 0s where it should not be considered

            full_mask = np.array(disk_mask * np.array(mask), dtype=bool)

            # Apply disk mask
            
            disk_sigma2 = np.nanvar(data.field[full_mask])
            disk_tau2 = 0.5*np.nanmean(np.sum(data.first_der[:,full_mask]**2, axis=0))

            if np.isnan(disk_sigma2) or np.isnan(disk_tau2):
                patch_sigma2.append(np.nan) 
                patch_tau2.append(np.nan)    
                print('NaN found in simulation cmb+noise number', k,'radius', R, 'patch', i)
                continue
            
            patch_sigma2.append(disk_sigma2)
            patch_tau2.append(disk_tau2)
            
            
        print('\n Saving simulation cmb+noise number', k,'radius', R)
        
        #Saving Results to .fits. 
        hdu = fits.PrimaryHDU(patch_sigma2)
        # Create header entries to describe the axes
        header = hdu.header
        header['RADIUS'] = str(R)   
        header['LENGTH'] = '12x'+str(nside)+'^2'   
        header['MCInp'] = f'{k:05}'           # length of the 1D array (3072)
        header['MASK'] = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00'           # length of the 1D array (3072)
        hdu.writeto(f'/hydrarepo/mmartin/MFs/patch/patch_sigma2_{k:05}_r_{int(R)}.fits', overwrite=True)

        hdu = fits.PrimaryHDU(patch_tau2)
        header = hdu.header
        header['RADIUS'] = str(R)  
        header['LENGTH'] = '12x'+str(nside)+'^2'   
        header['MCInp'] = f'{k:05}'           # length of the 1D array (3072)
        header['MASK'] = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00'           # length of the 1D array (3072)
        hdu.writeto(f'/hydrarepo/mmartin/MFs/patch/patch_tau2_{k:05}_r_{int(R)}.fits', overwrite=True)
            


