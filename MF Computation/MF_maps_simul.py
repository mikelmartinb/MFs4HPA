
# Replicating Joseba's MF maps, threshold shift

import numpy as np
import healpy as hp
from astropy.io import fits
import pynkowski as mf   # For Minkowski Functionals
import matplotlib.pyplot as plt
from astropy.utils.data import clear_download_cache
clear_download_cache()


sigma0 =  68.86260066414593
us = np.arange(-sigma0*3.5, sigma0*3.5, sigma0*0.15)
step = us[1] - us[0]
len_us = len(us)

nside = 8
disk_radii = [12]

# SELECT DESIDERD CMB MASK
mask = hp.read_map("/hydrarepo/mmartin/MFs/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits").astype(np.bool_)
mask = hp.ud_grade(mask, nside_out = 512) # Mask has 0 in points that should not be considered

map16 =  hp.pix2vec(nside, range(12*nside**2))

kmin = 0
kmax = 1000
ks = np.arange(kmin,kmax)
for k in ks:
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
    data.get_second_der()

    # Calculate local MFs and mean temp
    
    MF1 = []
    MF2 = []
    MF3 = []
    
    for i in range(12*nside**2):  # Over the map
        # Select point in the 16 map
        vec = (map16[0][i], map16[1][i], map16[2][i]) 
        # Select patch disk
        ipix_disc = hp.query_disc(nside=512, vec= vec, radius = np.radians(disk_radii[0])) 
        disk_mask = np.zeros(12*512**2)
        disk_mask[ipix_disc] = 1 # This again sets 0s where it should not be considered

        # Apply disk mask
        data_copy = data
        data_copy.mask =  np.array(disk_mask * np.array(mask.data), dtype=bool) # To consider the mask & patch. Pynkowski does not follow same criteria as healpy for the mask.
        
        disk_mean = np.mean(data_copy.field[np.array(disk_mask * np.array(mask.data), dtype=bool)])
        if np.isnan(disk_mean):
            MF1.append(np.zeros(len_us)+np.nan)
            MF2.append(np.zeros(len_us)+np.nan)
            MF3.append(np.zeros(len_us)+np.nan)
            continue
        new_us = us + disk_mean
        MF1.append(mf.V0(data_copy, new_us, verbose = False))
        MF2.append(mf.V1(data_copy, new_us, verbose = False)) 
        MF3.append(mf.V2(data_copy, new_us, verbose = False))
        
    print('\n Saving simulation cmb+noise number', k,'radius', disk_radii[0])
    
    #Saving Results to .fits. # Proposals for smarter use of fits are welcome.
    hdu = fits.PrimaryHDU(MF1)
    # Create header entries to describe the axes
    header = hdu.header
    header['RADIUS'] = str(disk_radii[0])   # starting value of u
    header['LENGTH'] = '12x'+str(nside)+'^2'   
    header['MCInp'] = f'{k:05}'           # length of the 1D array (3072)
    header['MASK'] = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00'           # length of the 1D array (3072)
    hdu.writeto(f'/hydrarepo/mmartin/MFs/new_MF1/MF1_new_mc_{k:05}_r_{int(disk_radii[0])}_nside_{nside}.fits', overwrite=True)

    hdu = fits.PrimaryHDU(MF2)
    header = hdu.header
    header['RADIUS'] = str(disk_radii[0])   # starting value of u
    header['LENGTH'] = '12x'+str(nside)+'^2'   
    header['MCInp'] = f'{k:05}'           # length of the 1D array (3072)
    header['MASK'] = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00'           # length of the 1D array (3072)
    hdu.writeto(f'/hydrarepo/mmartin/MFs/new_MF2/MF2_new_mc_{k:05}_r_{int(disk_radii[0])}_nside_{nside}.fits', overwrite=True)
    
    hdu = fits.PrimaryHDU(MF3)
    header = hdu.header
    header['RADIUS'] = str(disk_radii[0])   # starting value of u
    header['LENGTH'] = '12x'+str(nside)+'^2'   
    header['MCInp'] = f'{k:05}'           # length of the 1D array (3072)
    header['MASK'] = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00'           # length of the 1D array (3072)
    hdu.writeto(f'/hydrarepo/mmartin/MFs/new_MF3/MF3_new_mc_{k:05}_r_{int(disk_radii[0])}_nside_{nside}.fits', overwrite=True)



