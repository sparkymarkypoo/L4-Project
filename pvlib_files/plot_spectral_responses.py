import pvlib
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# Trim spectrum data
farms_metadata = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', nrows=1)
farms_data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', skiprows=2)
time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
time = pd.to_datetime(time_pd) + tz_offset
farms_data.index = time
spectrum = farms_data.iloc[:,32:2034:1]/1000


# Wavelengths
wvl_str = spectrum.columns
wvl = 1000*np.array([float(re.sub("[^0-9.\-]","",x)) for x in wvl_str])
wvl = wvl.round(1)
spectrum.columns = wvl 


# Get POA
poa = np.trapz(spectrum, x=wvl)
poa = pd.Series(data=poa, index=spectrum.index)


# Find and filter nagtive (bad) values
neg_values = (spectrum < 0).any(1)
neg_index = neg_values.index[neg_values].tolist()
poa.loc[neg_index] = np.nan


# Get AM1.5
am15 = pvlib.spectrum.get_am15g(wavelength=None)


# Get spectral response
folder = 'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/Spectral_Response'
arr = os.listdir(folder)
types = ['a-Si', 'CdTe', 'CIGS', 'mono-Si', 'poly-Si', 'PVSK', 'PVSK-Si', 'Triple Junction']
i=0
arr.remove('Images')
#plt.figure(dpi=250)
for a in arr:
    spec_resp_full = pd.read_csv(rf'{folder}/{a}', index_col=0, header=None)
            
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(spec_resp_full.index, spec_resp_full, k=3)  # type: BSpline
    spec_resp_interp = spl(wvl)
    spec_resp_interp[spec_resp_interp < 0] = 0
    s_r = pd.Series(data=np.squeeze(spec_resp_interp), index=wvl)
    
    plt.plot(s_r.loc[:2000.0], label=types[i])
    i=i+1
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Spectral Response')
plt.ylim(bottom=0)
plt.xlim(left=230, right=2500)
plt.legend()
plt.show()




spec_resp_full = pd.read_csv(rf'{folder}/monoSi.csv', index_col=0, header=None)
       
from scipy.interpolate import make_interp_spline
spl = make_interp_spline(spec_resp_full.index, spec_resp_full, k=3)  # type: BSpline
spec_resp_interp = spl(wvl)
spec_resp_interp[spec_resp_interp < 0] = 0
s_r = pd.Series(data=np.squeeze(spec_resp_interp), index=wvl)
spec_mod = pvlib.spectrum.calc_spectral_mismatch_field(sr=s_r, e_sun=spectrum, e_ref=am15)
spec_mod.loc[neg_index] = np.nan
spec_mod = spec_mod.dropna()
spec_mod = round(spec_mod,3)


# Plot spectrum at different times of the day
date = '2021-07-20'
bob = spectrum.loc[[f'{date} 06:00:00', f'{date} 07:00:00', f'{date} 08:00:00',
                    f'{date} 09:00:00', f'{date} 12:00:00']]
plt.figure(dpi=250)
fig, ax1 = plt.subplots()
ax1.plot(am15.loc[:2500], label='AM1.5', linewidth=1)
for b in bob.T:
    ax1.plot(bob.loc[b].loc[:2500], linewidth=1, label=f'{str(b)[11:16]} - {spec_mod.loc[b]}')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Irradiance (W/m^2 nm)')
ax1.set_ylim(bottom=0)
ax1.set_xlim(left=230, right=2500)
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('Relative Spectral Response')
ax2.plot(s_r.loc[:2500], color='black')
ax2.set_ylim(bottom=0)

