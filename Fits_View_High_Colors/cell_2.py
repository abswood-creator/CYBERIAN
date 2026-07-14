# --- Cell 2: load the data ---
filename = "j8q901081_drz.fits"

with fits.open(filename) as hdul:
    hdul.info()  # peek at extensions — some instruments split SCI,1 / SCI,2
    data = hdul['SCI'].data if 'SCI' in hdul else hdul[0].data

data = np.nan_to_num(data, nan=0.0)
