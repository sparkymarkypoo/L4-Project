
import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time as stopwatch

# Import coordinates/plotting stuff
from US_map import state_map
coords = state_map()
from pvlib_files.spec_response_function import calc_spectral_modifier



# Define solar panel parameters
SAM_URL = 'https://github.com/NREL/SAM/raw/develop/deploy/libraries/CEC%20Modules.csv'
CEC_mods = pvlib.pvsystem.retrieve_sam(path=SAM_URL)
modules = CEC_mods[['First_Solar_Inc__FS_7530A_TR1','Miasole_FLEX_03_480W','Jinko_Solar_Co__Ltd_JKM340PP_72','Jinko_Solar_Co__Ltd_JKMS350M_72']]
CEC_mods = pvlib.pvsystem.retrieve_sam(name='CECMod')
modules['Xunlight_XR36_300'] = CEC_mods['Xunlight_XR36_300']
types = ['CdTe','CIGS','multi-Si','mono-Si','a-Si']
modules.loc['Technology'] = types


# Find NSRDB data
folder = 'D:/NSRDB_Data'
arr = os.listdir(folder)
year = '2021'
#year = '8_year2021_nsrdb.csv'
arr2 = [k for k in arr if year in k]
states = [sub[5:7] for sub in arr2]

# Prepare dataframes for analysis
locs = pd.DataFrame(columns=['lat','long'], index=states)
eta_rel_list, mpp_list, spec_list = [], [], []
year_eta_rel_list, year_mpp_list, year_spec_list = [], [], []
j=0
for a in arr2:

    # Open NSRDB data
    farms_metadata = pd.read_csv(rf'{folder}/{a}', nrows=1)
    farms_data = pd.read_csv(rf'{folder}/{a}', skiprows=2)
    time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
    tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
    time = pd.to_datetime(time_pd) + tz_offset
    farms_data.index = time
    
    
    locs['lat'].loc[states[j]] = float(farms_metadata['Latitude'])
    locs['long'].loc[states[j]] = float(farms_metadata['Longitude'])
    # from Data.open_data import open_copernicus
    # cop = open_copernicus(latitude, longitude, tz_offset)
    
    
    # Trim first few days (assuming -ve time zone means overspill into previous year)
    month0 = time[0].month
    year0 = time[0].year
    farms_data = farms_data[farms_data.index > f'{year0}-{month0}-31 23:59']
    # cop = cop[cop.index > f'{year0}-{month0}-31 23:59']
    # cop.loc['2021-12-31 14:00:00'] = 0
    # cop.loc['2021-12-31 15:00:00'] = 0
    
    
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
    farms_df.loc[farms_df['cloud'] < 3, 'poa_global'] = farms_df['poa_isotropic']
    farms_df.loc[farms_df['cloud'] >2, 'poa_global'] = farms_df['poa_klucher']
            
    # Get effective irradiance
    mpp = pd.DataFrame(columns = types)
    eta_rel = pd.DataFrame(columns = types)
    i = 0
    spec_mismatch = pd.DataFrame(columns=types)
    spec_mon = pd.DataFrame(columns=types)
    spec_year = pd.Series(index=types)
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
        spec_mon[material] = farms_df['effective_irradiance'].resample('M').apply(np.trapz) / farms_df['poa_global'].resample('M').apply(np.trapz)
        spec_year[material] = np.trapz(farms_df['effective_irradiance']) / np.trapz(farms_df['poa_global'])
        i=i+1
    
    # Append yearly variables
    year_spec_list.append(spec_year)
    year_mpp_list.append(np.trapz(mpp, axis=0))
    top = np.trapz(mpp, axis=0) / modules.loc['STC'].to_numpy()   
    bot = np.trapz(farms_df['poa_global'] / 1000)
    year_eta_rel_list.append(top/bot)
    
    # Append monthly variables
    spec_list.append(spec_mon)
    
    poa_mon = farms_df['poa_global'].resample('M').apply(np.trapz)
    mpp_mon = mpp.resample('M').apply(np.trapz)
    mpp_list.append(round(mpp_mon, 1))
 
    top = mpp_mon / modules.loc['STC'].to_numpy()   
    bot = (poa_mon / 1000).to_numpy().reshape(12,1)
    eta_rel_mon = top / bot
    eta_rel_list.append(round(eta_rel_mon,3))
    j=j+1

def make_2d_pandas(np_input, types, states):
    df = pd.DataFrame(data=np_input, columns=types)
    df.index = states
    return df
mpp_yearly = make_2d_pandas(year_mpp_list, types, states)
eta_rel_yearly = make_2d_pandas(year_eta_rel_list, types, states)
spec_yearly = make_2d_pandas(year_spec_list, types, states)

    
def make_3d_pandas(np_input):
    index = pd.MultiIndex.from_product([range(s)for s in np_input.shape])
    df = pd.DataFrame({'A': np_input.flatten()}, index=index)['A']
    df = df.unstack()
    df.columns = types
    df.index.names = ['Location', 'Month']
    return df
mpp_monthly = make_3d_pandas(np.array(mpp_list))
eta_rel_monthly = make_3d_pandas(np.array(eta_rel_list))
spec_monthly = make_3d_pandas(np.array(spec_list))

eta_rel_var = pd.DataFrame(columns=types, index=states, dtype=float)
k=0
for s in states:
    eta_rel_var.loc[s] = eta_rel_monthly.loc[k].max() - eta_rel_monthly.loc[k].min()
    k=k+1


for t in types:
    coords['CUSTOM'] = spec_yearly[t]
    plt.figure()
    figmap = coords.plot(column='CUSTOM',figsize=(25, 13), legend=True, cmap='coolwarm')
    plt.xlabel('Longitude',fontsize=20)
    plt.ylabel('Latitude',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cb_ax = figmap.figure.axes[1]
    cb_ax.tick_params(labelsize=18)
    for i in range(len(locs)):
        plt.text(locs.long[i],locs.lat[i],locs.index[i],size=10)
    plt.show()