import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# 1. Load the Planck CMB map
filename = 'COM_CMB_IQU-commander_1024_R2.02_full.fits'
print("Loading map...")
t_map = hp.read_map(filename, field=0, dtype=np.float64)
print(f"Map loaded! NSIDE: {hp.npix2nside(len(t_map))}")

# 2. Apply a galactic mask
nside = int(hp.npix2nside(len(t_map)))
print(f"NSIDE: {nside}")

# Create a mask that excludes the galactic plane
galactic_mask = np.ones(len(t_map), dtype=bool)
gl, gb = hp.pix2ang(nside, np.arange(len(t_map)), lonlat=True)
galactic_mask[np.abs(gb) < 20.0] = False
print(f"Masking {np.sum(~galactic_mask)} pixels")

# 3. Create a masked array
t_map_masked = hp.ma(t_map)
t_map_masked.mask = ~galactic_mask

# 4. Compute the Angular Power Spectrum
lmax = 1024
result = hp.anafast(t_map_masked, lmax=lmax, pol=False)
ell = result[0]
cl = result[1]
print(f"Power spectrum computed up to l = {lmax}")

# 5. Plot the Power Spectrum
l_cl = ell * (ell + 1) * cl / (2.0 * np.pi)

plt.figure(figsize=(12, 6))
plt.plot(ell, l_cl, 'b-', linewidth=1, label='Planck Data (T)')

# Load the theoretical spectrum from the FITS file
theoretical_file = 'COM_PowerSpect_CMB_R2.02.fits'
try:
    with fits.open(theoretical_file) as hdul:
        data = hdul[1].data
        ell_th = data['ELL']
        cl_th = data['D_ELL']
    
    # Calculate l * Cl / (2pi) for the theoretical curve
    l_cl_th = ell_th * (ell_th + 1) * cl_th / (2.0 * np.pi)
    plt.plot(ell_th, l_cl_th, 'r--', linewidth=2, label='Theoretical ΛCDM')
    
except FileNotFoundError:
    print(f"Theoretical file '{theoretical_file}' not found. Skipping theoretical plot.")
except Exception as e:
    print(f"Error loading theoretical file: {e}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Multipole Moment ($\\ell$)', fontsize=12)
plt.ylabel('$\\ell(\\ell+1)C_\\ell / (2\\pi)$ ($\\mu K^2$)', fontsize=12)
plt.title('Planck CMB Temperature Power Spectrum', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('cmb_power_spectrum_final.png', dpi=150)
plt.show()
print("Final plot saved as 'cmb_power_spectrum_final.png'")
