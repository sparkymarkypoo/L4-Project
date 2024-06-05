
# Import general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

# Import data
from pvlib_files.Functions.spec_response_function import calc_spectral_modifier

# Folder location
import os
cwd = os.path.dirname(os.getcwd())

year=2018

# Define solar panel parameters
sheet = pd.read_csv(os.path.join(cwd, 'Data\spec_sheets.csv'), index_col=0)#, dtype=np.float64)#dtype={'cell_type':str})
modules = pd.DataFrame(columns=sheet.columns, index=['pstc', 'area', 'alpha_sc', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'a_ref', 'Adjust'])
for s in sheet:

    pstc = sheet[s].loc['pstc']
    area = sheet[s].loc['area']
    alpha_sc = sheet[s].loc['alpha_sc'] * sheet[s].loc['isc']/100
    
    params = pvlib.ivtools.sdm.fit_cec_sam(
        celltype = s,
        v_mp = sheet[s].loc['vmp'],
        i_mp = sheet[s].loc['imp'],
        v_oc = sheet[s].loc['voc'],
        i_sc = sheet[s].loc['isc'],
        alpha_sc = alpha_sc,
        beta_voc = sheet[s].loc['beta_voc'] * sheet[s].loc['voc']/100,
        gamma_pmp = sheet[s].loc['gamma_pmp'],
        cells_in_series = sheet[s].loc['cells_in_series'])

    modules[s] = (pstc, area, alpha_sc) + params
types1 = ['CdTe','CIGS','multi-Si','mono-Si','a-Si']   # THESE ONES FOR POWER SIM
modules.loc['Technology'] = types1
types2 = ['perovskite','perovskite-si','triple']       # THESE ONES ONLY FOR POA SIM
types = types1 + types2


# Open NSRDB data
farms_metadata = pd.read_csv(os.path.join(cwd, 'Data\Camp_Fire_2018.csv'), nrows=1)
farms_data = pd.read_csv(os.path.join(cwd, 'Data\Camp_Fire_2018.csv'), skiprows=2)
time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
time = pd.to_datetime(time_pd) + tz_offset
farms_data.index = time


# Trim first few days (assuming -ve time zone means overspill into previous year)
month0 = time[0].month
year0 = time[0].year
farms_data = farms_data[farms_data.index > f'{year0}-{month0}-31 23:59']


# Separate stuff into farms_df as farms_data is big with long labels
farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                         'cloud':farms_data['Cloud Type'], 'albedo':farms_data['Surface Albedo'], 'precip':farms_data['Precipitable Water'],
                         'ghi':farms_data['GHI'], 'dhi':farms_data['DHI'], 'dni':farms_data['DNI'],
                         'zenith':farms_data['Solar Zenith Angle'], 'azimuth':farms_data['Solar Azimuth Angle']})
latitude = float(farms_metadata['Latitude'])
longitude = float(farms_metadata['Longitude'])   


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
farms_df['poa_global'][farms_df['cloud']<2] = farms_df['poa_isotropic']
farms_df['poa_global'][farms_df['cloud']>1] = farms_df['poa_klucher']

farms_clear = farms_df[farms_df.cloud < 2]
farms_cloud = farms_df[farms_df.cloud > 1]


# Get effective irradiance
mpp = pd.DataFrame(columns = types1)
eta_rel = pd.DataFrame(columns = types1)
i = 0
spec_mismatch = pd.DataFrame(columns=types)
for material in types:
    spec_mismatch[material], farms_df['poa_farms'], mat = calc_spectral_modifier(material, farms_data)
    farms_df['effective_irradiance'] = farms_df['poa_global'] * spec_mismatch[material]
    farms_df = farms_df.fillna(0)
    
    if material in types2:  # perovskite and triple cannot use single diode model
        0
    else:
    
        # Calculate cell parameters
        temp_cell = pvlib.temperature.faiman(farms_df['poa_global'], farms_df['temp_air'], farms_df['wind_speed'])
        cec_params = pvlib.pvsystem.calcparams_cec(farms_df['effective_irradiance'], temp_cell, modules.loc['alpha_sc'].iloc[i],
                                                   modules.loc['a_ref'].iloc[i], modules.loc['I_L_ref'].iloc[i],
                                                   modules.loc['I_o_ref'].iloc[i], modules.loc['R_sh_ref'].iloc[i],
                                                   modules.loc['R_s'].iloc[i], modules.loc['Adjust'].iloc[i])
    
    
        # Max power point
        mpp[material] = pvlib.pvsystem.max_power_point(*cec_params, method='newton').p_mp
        eta_rel[material] = (mpp[material] / modules.loc['pstc'].iloc[i]) / (farms_df['poa_global'] / 1000)
    i=i+1
    
    
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
for t in types1:
    eta_power_monthly = (mpp_monthly[t] / modules.loc['pstc'].iloc[i]) / (poa_monthly / 1000)
    eta_power_monthly.name = t
    i=i+1
    ax = eta_power_monthly.plot()
    ax.set_ylabel("Mean Relative Efficiency")
ax.legend()

