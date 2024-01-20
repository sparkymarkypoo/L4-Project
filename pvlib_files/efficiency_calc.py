import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib import iotools, location
from pvlib.irradiance import get_total_irradiance
from pvlib.pvarray import pvefficiency_adr
from pvlib_files.spec_response_function import calc_spectral_modifier

# Get data
data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Grid/2021-204887-fixed_tilt.csv', skiprows=2)


# Reformat for modelling
df = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'],
                   'dhi':data['DHI'], 'temp_air':data['Temperature'],
                   'wind_speed':data['Wind Speed']})
time_pd = data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
time = pd.to_datetime(time_pd)
df.index = time


# Select location
latitude = 34.321
longitude = -118.225
loc = location.Location(latitude=latitude,longitude=longitude,
                    altitude=pvlib.location.lookup_altitude(latitude=latitude, longitude=longitude))
solpos = loc.get_solarposition(df.index)


# Get POA irradiance
tilt = 30
azi = 180
total_irrad = get_total_irradiance(tilt, azi,
                                   solpos.apparent_zenith, solpos.azimuth,
                                   df.dni, df.ghi, df.dhi)
spec_mismatch = calc_spectral_modifier()
df['poa_global'] = total_irrad.poa_global * spec_mismatch


# Module parameters
df['temp_pv'] = pvlib.temperature.faiman(df.poa_global, df.temp_air,
                                         df.wind_speed)
adr_params = {'k_a': 0.99924,
              'k_d': -5.49097,
              'tc_d': 0.01918,
              'k_rs': 0.06999,
              'k_rsh': 0.26144
              }
df['eta_rel'] = pvefficiency_adr(df['poa_global'], df['temp_pv'], **adr_params)


# Set the desired array size:
P_STC = 5000.   # (W)
# and the irradiance level needed to achieve this output:
G_STC = 1000.   # (W/m2)
df['p_mp'] = P_STC * df['eta_rel'] * (df['poa_global'] / G_STC)



# Plots
plt.figure()
pc = plt.scatter(df['poa_global'], df['eta_rel'], c=df['temp_pv'], cmap='jet')
plt.colorbar(label='Temperature [C]', ax=plt.gca())
pc.set_alpha(0.25)
plt.grid(alpha=0.5)
plt.ylim(0.48)
plt.xlabel('Irradiance [W/m²]')
plt.ylabel('Relative efficiency [-]')
plt.show()

plt.figure()
pc = plt.scatter(df['poa_global'], df['p_mp'], c=df['temp_pv'], cmap='jet')
plt.colorbar(label='Temperature [C]', ax=plt.gca())
pc.set_alpha(0.25)
plt.grid(alpha=0.5)
plt.xlabel('Irradiance [W/m²]')
plt.ylabel('Array power [W]')
plt.show()