import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wavelength values
wvl = np.arange(280,1205,5)
wvl_5 = np.arange(280,400,5)
wvl_10 = np.arange(400,1205,10)
wvl_conc = np.concatenate([wvl_5, wvl_10])

# Get spectral response
s_r_ = pvlib.spectrum.get_example_spectral_response(wavelength=None)
am15 = pvlib.spectrum.get_am15g(wavelength=None)

# Interpolate
x = np.arange(280, 1201, 1)
yinterp = np.interp(x, xp=wvl, fp=s_r_)
s_r = pd.Series(data=yinterp, index=x)

# Import spectrum data
specific_rows = np.arange(2)  # cut out crap at the top 
df = pd.read_csv(r'C:\Users\mark\Documents\L4-Project-Code\Data\Desert Rock\Spectrum_2021_DRA_0deg.csv',skiprows = specific_rows)
spec = df.iloc[:,32:1080:1]/1000
spec.columns = [wvl_conc]

# Loop time!
yeet = np.zeros(len(spec))
for i in range(len(spec)):
    
    spec_i = spec.iloc[i]
    
    if np.sum(spec_i) > 0:
    
        # Unpack and repack the data as mismatch does not like it straight from data
        spec_numpy = spec_i.to_numpy()
        spec_pd = pd.Series(data=spec_numpy, index=wvl_conc)
        
        # Calculate spectral mismatch
        yeet[i] = pvlib.spectrum.calc_spectral_mismatch_field(sr=s_r, e_sun=spec_pd, e_ref=None)

    else:
        # Set to 0 as doing the mismatch would give div by 0 errors
        yeet[i]=0


fig, ax1 = plt.subplots(figsize=(20,10))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Global Horizontal Irradiation [W/m^2 nm]")
plt.title("Desert Rock, Nevada 2022-05-02")

ax1.plot(wvl_conc,spec.iloc[20], label = 'NSRDB Winter')
ax1.plot(wvl_conc,spec.iloc[2588], label = 'NSRDB Summer')
ax1.plot(am15[280:1200], label = 'AM1.5 Irradiance')

ax2 = ax1.twinx()
ax2.set_ylabel('Spectral Response (A/W)')
ax2.plot(wvl,s_r, color='red', label='Spectral Response ')

ax1.grid()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0, 2*np.pi, 10)
# y = np.sin(x)
# xvals = np.linspace(0, 2*np.pi, 50)
# yinterp = np.interp(xvals, x, y)

# plt.plot(x, y, 'o')
# plt.plot(xvals, yinterp, '-x')
# plt.show()