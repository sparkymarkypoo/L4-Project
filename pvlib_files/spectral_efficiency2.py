# Import pvlib stuff
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Import general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
from Data.open_data import NRELdata

# Import coordinates/plotting stuff
from US_map import county_map
coords = county_map()
from pvlib_files.spec_response_function import calc_spectral_modifier



# Define solar panel parameters
SAM_URL = 'https://github.com/NREL/SAM/raw/develop/deploy/libraries/CEC%20Modules.csv'
CEC_mods = pvlib.pvsystem.retrieve_sam(path=SAM_URL)
modules = CEC_mods[['First_Solar_Inc__FS_7530A_TR1','Miasole_FLEX_03_480W','Jinko_Solar_Co__Ltd_JKM340PP_72','Jinko_Solar_Co__Ltd_JKMS350M_72']]
CEC_mods = pvlib.pvsystem.retrieve_sam(name='CECMod')
modules['Xunlight_XR36_300'] = CEC_mods['Xunlight_XR36_300']
types = ['CdTe','CIGS','multi-Si','mono-Si','a-Si']
modules.loc['Technology'] = types


# Open NSRDB data
farms_metadata = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', nrows=1)
farms_data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', skiprows=2)
time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
time = pd.to_datetime(time_pd) + tz_offset
farms_data.index = time


latitude = 34.33
longitude = -118.22
from Data.open_data import open_copernicus
cop = open_copernicus(latitude, longitude, tz_offset)


# Trim first few days (assuming -ve time zone means overspill into previous year)
month0 = time[0].month
year0 = time[0].year
farms_data = farms_data[farms_data.index > f'{year0}-{month0}-31 23:59']
cop = cop[cop.index > f'{year0}-{month0}-31 23:59']
cop.loc['2021-12-31 14:00:00'] = 0
cop.loc['2021-12-31 15:00:00'] = 0


# Separate stuff into farms_df as farms_data is big with long labels
farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                         'cloud':farms_data['Cloud Type'], 'albedo':farms_data['Surface Albedo'], 'precip':farms_data['Precipitable Water'],
                         'ghi':farms_data['GHI'], 'dhi':farms_data['DHI'], 'dni':farms_data['DNI'],
                         'zenith':farms_data['Solar Zenith Angle'], 'azimuth':farms_data['Solar Azimuth Angle']})


# POA Irradiance
tilt = farms_data['Panel Tilt'].iloc[0]
farms_df['poa_isotropic'] = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=180,
                                                               solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                                                               dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],       
                                                               albedo=farms_df['albedo'], model='isotropic').poa_global
farms_df['poa_klucher'] = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=180,
                                                               solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                                                               dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],       
                                                               albedo=farms_df['albedo'], model='klucher').poa_global
farms_df['poa_global'] = 0
farms_df['poa_global'][farms_df['cloud']<3] = farms_df['poa_isotropic']
farms_df['poa_global'][farms_df['cloud']>2] = farms_df['poa_klucher']

farms_clear = farms_df[farms_df.cloud < 3]
farms_cloud = farms_df[farms_df.cloud > 2]


# Get effective irradiance
mpp = pd.DataFrame(columns = types)
eta_rel = pd.DataFrame(columns = types)
i = 0
spec_mismatch = pd.DataFrame(columns=types)
for material in modules.loc['Technology']:
    spec_mismatch[material], farms_df['poa_farms'], mat = calc_spectral_modifier(material, farms_data)
    farms_df['effective_irradiance'] = farms_df['poa_global'] * spec_mismatch[material]
    farms_df = farms_df.fillna(0)
    
    
    # Calculate cell parameters
    temp_cell = pvlib.temperature.faiman(farms_df['poa_global'], farms_df['temp_air'], farms_df['wind_speed'])
    cec_params = pvlib.pvsystem.calcparams_cec(farms_df['effective_irradiance'], temp_cell, modules.loc['alpha_sc'].iloc[i],
                                               modules.loc['a_ref'].iloc[i], modules.loc['I_L_ref'].iloc[i],
                                               modules.loc['I_o_ref'].iloc[i], modules.loc['R_sh_ref'].iloc[i],
                                               modules.loc['R_s'].iloc[i], modules.loc['Adjust'].iloc[i])


    # Max power point
    mpp[material] = pvlib.pvsystem.max_power_point(*cec_params, method='newton').p_mp
    eta_rel[material] = (mpp[material] / modules.loc['STC'].iloc[i]) / (farms_df['poa_global'] / 1000)
    i=i+1
    
    eta_rel_clear = eta_rel[material][farms_df.cloud < 3]
    eta_rel_cloud = eta_rel[material][farms_df.cloud > 2]
    
    # Plots
    # plt.figure()
    # pc = plt.scatter(farms_clear['poa_global'], eta_rel_clear, c=farms_clear.cloud, cmap='jet')
    # plt.colorbar(label='Temperature [C]', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.ylim(0.48)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Relative efficiency [-]')
    # plt.show()
    
    # # plt.figure()
    # pc = plt.scatter(farms_cloud['poa_global'], eta_rel_cloud, c=farms_cloud.cloud, cmap='jet')
    # plt.colorbar(label='Temperature [C]', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.ylim(0.48)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Relative efficiency [-]')
    # plt.show()
    
    # plt.figure()
    # pc = plt.scatter(farms_df['poa_global'], eta_rel[material], c=farms_df.cloud, cmap='jet')
    # plt.colorbar(label='Temperature [C]', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.ylim(0.48)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Relative efficiency [-]')
    # plt.show()

    # plt.figure()
    # pc = plt.scatter(farms_df['poa_global'], eta_rel[material], c=temp_cell, cmap='jet')
    # plt.colorbar(label='Temperature [C]', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.ylim(0.48)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Relative efficiency [-]')
    # plt.show()
    
    # plt.figure()
    # pc = plt.scatter(farms_df['poa_global'], spec_mismatch[material], c=farms_df['cloud'], cmap='jet')
    # plt.colorbar(label='Cloud Type', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Spectral Mismatch (-)')
    # plt.show()
    


mpp_monthly = mpp.resample('M').apply(np.trapz)
poa_monthly = farms_df['poa_global'].resample('M').apply(np.trapz)
spec_mismatch_monthly = spec_mismatch[spec_mismatch>0].resample('M').apply(np.mean)

farms_weekly = farms_df.resample('W').apply(np.mean)
farms_weekly['poa_global'] = farms_df['poa_global'].resample('W').apply(np.trapz)
mpp_weekly = mpp.resample('W').apply(np.trapz)
poa_weekly = farms_df['poa_global'].resample('W').apply(np.trapz)
spec_mismatch_weekly = spec_mismatch[spec_mismatch>0].resample('W').apply(np.mean)

farms_daily = farms_df.resample('D').apply(np.mean)
farms_daily['poa_global'] = farms_df['poa_global'].resample('D').apply(np.trapz)
mpp_daily = mpp.resample('D').apply(np.trapz)
poa_daily = farms_df['poa_global'].resample('D').apply(np.trapz)
spec_mismatch_daily = spec_mismatch[spec_mismatch>0].resample('D').apply(np.mean)

plt.scatter(farms_daily.precip, spec_mismatch_daily[material])
plt.xlabel('Precipitable Water (mm)')
plt.ylabel('Spectral Mismatch for a-Si')


# Mean spectral mismatch by month
ax = plt.figure(figsize=(10,6))
ax = spec_mismatch_monthly.plot()
ax.set_ylabel("Mean Spectral Mismatch")

# Mean spectral mismatch by hour
hour = farms_df.index.hour
spec_mismatch_hourly = spec_mismatch.groupby(hour).mean()
ax = plt.figure(figsize=(10,6))
ax = spec_mismatch_hourly.plot()
ax.set_ylabel("Mean Spectral Mismatch")
ax.set_xlabel('Hour')
ax.set_xticks(np.arange(0,28,4))

# Mean eta_rel by hour
eta_rel_hourly = eta_rel.groupby(hour).mean()
ax = plt.figure(figsize=(10,6))
ax = eta_rel_hourly.plot()
ax.set_ylabel("Mean Relative Efficiency")
ax.set_xlabel('Hour')
ax.set_xticks(np.arange(0,28,4))

# Mean relative efficiency by averaging DC Power and POA by month
plt.figure()
i=0
for t in types:
    eta_power_monthly = (mpp_monthly[t] / modules.loc['STC'].iloc[i]) / (poa_monthly / 1000)
    eta_power_monthly.name = t
    i=i+1
    ax = eta_power_monthly.plot()
    ax.set_ylabel("Mean Relative Efficiency")
ax.legend()

