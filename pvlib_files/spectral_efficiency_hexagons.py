# General imports
import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# Location
import os
cwd = os.path.dirname(os.getcwd())

# Import data and tools
from pvlib_files.Functions.spec_response_function import calc_spectral_modifier

# Import maps
from pvlib_files.Functions.US_map import state_map
usmap = state_map()
from pvlib_files.Functions.US_map import hexagons_map
hexagons = hexagons_map()

# Plotting tools
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


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


# Find NSRDB data
folder = 'D:/Hexagon_downloads'    # WHERE YOUR NSRDB_FARMS DATA IS LOCATED
arr = os.listdir(folder)
code = [sub[0:15] for sub in arr]

# Prepare dataframes for analysis
locs = pd.DataFrame(columns=['lat','long','alt'], index=code)
wet = ['cloud', 'temp', 'precip', 'poa']
weather_year = pd.Series(index=wet)
weather_mon = pd.DataFrame(columns=wet)
hour_eta_rel_list, hour_spec_list = [], []
mon_weather_list, mon_eta_list, mon_eta_rel_list, mon_mpp_list, mon_spec_list = [], [], [], [], []
year_weather_list, year_eta_list, year_eta_rel_list, year_mpp_list, year_spec_list = [], [], [], [], []
j=0
for a in arr:

    # Open NSRDB data
    farms_metadata = pd.read_csv(rf'{folder}/{a}', nrows=1)
    farms_data = pd.read_csv(rf'{folder}/{a}', skiprows=2)
    time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
    tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
    time = pd.to_datetime(time_pd) + tz_offset
    farms_data.index = time
    
    
    locs['lat'].loc[code[j]] = float(farms_metadata['Latitude'])
    locs['long'].loc[code[j]] = float(farms_metadata['Longitude'])
    locs['alt'].loc[code[j]] = pvlib.location.lookup_altitude(locs['lat'].loc[code[j]], locs['long'].loc[code[j]])
    
    # Trim first few days (assuming -ve time zone means overspill into previous year)
    month0 = time[0].month
    year0 = time[0].year
    farms_data = farms_data[farms_data.index > f'{year0}-{month0}-31 23:59']
    
    
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
    farms_df.loc[farms_df['cloud'] < 2, 'poa_global'] = farms_df['poa_isotropic']
    farms_df.loc[farms_df['cloud'] >1, 'poa_global'] = farms_df['poa_klucher']
    
    
    # Weather statistics
    def sort_weather(weather, poa): # Getmonthly and yearly averages
        temp = weather[(poa>0)]
        mon = temp.resample('M').mean()
        year = temp.mean()
        return mon, year
    
    weather_mon['temp'], weather_year['temp'] = sort_weather(farms_df['temp_air'], farms_df['poa_global'])
    weather_mon['precip'], weather_year['precip'] = sort_weather(farms_df['precip'], farms_df['poa_global'])
    
    cloudy_times = farms_df['cloud'][(farms_df['poa_global']>0) & (farms_df['cloud']>1)]
    sunny_times = farms_df['cloud'][(farms_df['poa_global']>0)]
    weather_mon['cloud'] = 100* cloudy_times.resample('M').count() / sunny_times.resample('M').count()
    weather_year['cloud'] = 100* cloudy_times.count() / sunny_times.count()
    
    weather_mon['poa'] = farms_df['poa_global'].resample('M').apply(np.trapz)
    weather_year['poa'] = np.trapz(farms_df['poa_global'])
    
    
    # Prepare looped variables
    mpp = pd.DataFrame(columns = types1)
    eta = pd.DataFrame(columns = types1)
    eta_rel = pd.DataFrame(columns = types1)
    spec_mismatch = pd.DataFrame(columns=types)
    spec_mon = pd.DataFrame(columns=types)
    spec_hour = pd.DataFrame(columns=types)
    spec_year = pd.Series(index=types)
    hour = farms_df.index.hour
    
    i = 0
    # Loop for material types
    for material in types:
        # Calculate spectral mismatch and effective irrad
        spec_mismatch[material], farms_df['poa_farms'], mat = calc_spectral_modifier(material, farms_data)
        farms_df['effective_irradiance'] = farms_df['poa_global'] * spec_mismatch[material]
        # Whenever effective irrad is nan, also force this on global irrad
        farms_df['poa_global'][farms_df['effective_irradiance'].isna()] = np.nan
        # Set to 0 to avoid later errors
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
            
        spec_hour[material] = farms_df['effective_irradiance'].groupby(hour).apply(np.trapz) / farms_df['poa_global'].groupby(hour).apply(np.trapz)
        spec_mon[material] = farms_df['effective_irradiance'].resample('M').apply(np.trapz) / farms_df['poa_global'].resample('M').apply(np.trapz)
        spec_year[material] = np.trapz(farms_df['effective_irradiance']) / np.trapz(farms_df['poa_global'])
        i=i+1
    
    # Append hourly variables
    hour_spec_list.append(spec_hour)
    top = mpp.groupby(hour).apply(np.mean) / modules.loc['pstc'].to_numpy()   
    bot = farms_df['poa_global'].groupby(hour).apply(np.mean) / 1000
    hour_eta_rel_list.append(top.divide(bot.replace(0,np.nan), axis=0))
    
    # Append yearly variables
    year_spec_list.append(spec_year)
    year_mpp_list.append(np.trapz(mpp, axis=0))
    year_weather_list.append(weather_year.copy())
    
    top = 100 * np.trapz(mpp, axis=0)
    bot = np.trapz(farms_df['poa_global']) * modules.loc['area'].to_numpy()   
    year_eta_list.append(top/bot)
    
    top = np.trapz(mpp, axis=0) / modules.loc['pstc'].to_numpy()   
    bot = np.trapz(farms_df['poa_global'] / 1000)
    year_eta_rel_list.append(top/bot)
    
    
    # Append monthly variables
    mon_spec_list.append(spec_mon)   
    mpp_mon = mpp.resample('M').apply(np.trapz)
    mon_mpp_list.append(round(mpp_mon, 1)) 
    mon_weather_list.append(weather_mon.copy())
    
    top = 100 *mpp_mon / modules.loc['area'].to_numpy()
    bot = (farms_df['poa_global'].resample('M').apply(np.trapz)).to_numpy().reshape(12,1)
    mon_eta_list.append(round(top / bot,3))
    
    top = mpp_mon / modules.loc['pstc'].to_numpy()   
    bot = (farms_df['poa_global'].resample('M').apply(np.trapz) / 1000).to_numpy().reshape(12,1)
    mon_eta_rel_list.append(round(top / bot,3))
    j=j+1


# Yearly variables into pandas
def make_2d_pandas(list_input, types, code):
    df = pd.DataFrame(data=list_input, columns=types, dtype=float)
    df.index = code
    return df
mpp_yearly = make_2d_pandas(year_mpp_list, types1, code)
eta_yearly = make_2d_pandas(year_eta_list, types1, code)
eta_rel_yearly = make_2d_pandas(year_eta_rel_list, types1, code)
spec_yearly = make_2d_pandas(year_spec_list, types, code)
weather_yearly = make_2d_pandas(year_weather_list, wet, code)

# Monthly variables into pandas
def make_3d_pandas_monthly(list_input, types, states):
    np_input = np.array(list_input)
    index = pd.MultiIndex.from_product([range(s)for s in np_input.shape])
    df = pd.DataFrame({'A': np_input.flatten()}, index=index, dtype=float)['A']
    df = df.unstack()
    df.columns = types
    df.index.names = ['State', 'Month']
    df.index = df.index.set_levels(states, level=0)
    df.index = df.index.set_levels(np.arange(1,13), level=1)
    return df
mpp_monthly = make_3d_pandas_monthly(mon_mpp_list, types1, code)
eta_monthly = make_3d_pandas_monthly(mon_eta_list, types1, code)
eta_rel_monthly = make_3d_pandas_monthly(mon_eta_rel_list, types1, code)
spec_monthly = make_3d_pandas_monthly(mon_spec_list, types, code)
weather_monthly = make_3d_pandas_monthly(mon_weather_list, wet, code)

# Hourly variables into pandas
def make_3d_pandas_hourly(list_input, types, states):
    np_input = np.array(list_input)
    index = pd.MultiIndex.from_product([range(s)for s in np_input.shape])
    df = pd.DataFrame({'A': np_input.flatten()}, index=index, dtype=float)['A']
    df = df.unstack()
    df.columns = types
    df.index.names = ['State', 'Hour']
    df.index = df.index.set_levels(states, level=0)
    df.index = df.index.set_levels(np.arange(0,24), level=1)
    return df
eta_rel_hourly = make_3d_pandas_hourly(hour_eta_rel_list, types1, code)
spec_hourly = make_3d_pandas_hourly(hour_spec_list, types, code)

eta_rel_var = pd.DataFrame(columns=types, index=code, dtype=float)
for s in code:
    eta_rel_var.loc[s] = eta_rel_monthly.loc[s].max() - eta_rel_monthly.loc[s].min()



# USA MAP
def US_multiplot(hexagons, usmap, yearly_stat, four_types, labels, vmin, vmax, clabel):

    result = pd.concat([hexagons, yearly_stat], axis=1)

    cmap = 'coolwarm_r'
    
    f, axes = plt.subplots(figsize=(12.5, 6.25), ncols=2, nrows=2, sharex=True, sharey=True)
    
    result.plot(ax=axes[0][0], column=four_types[0], vmin=vmin, vmax=vmax, cmap=cmap)
    result.plot(ax=axes[0][1], column=four_types[1], vmin=vmin, vmax=vmax, cmap=cmap)
    result.plot(ax=axes[1][0], column=four_types[2], vmin=vmin, vmax=vmax, cmap=cmap)
    result.plot(ax=axes[1][1], column=four_types[3], vmin=vmin, vmax=vmax, cmap=cmap)
    
    usmap.plot(ax=axes[0][0], facecolor="none", edgecolor='black', linewidth=0.1)
    usmap.plot(ax=axes[0][1], facecolor="none", edgecolor='black', linewidth=0.1)
    usmap.plot(ax=axes[1][0], facecolor="none", edgecolor='black', linewidth=0.1)
    usmap.plot(ax=axes[1][1], facecolor="none", edgecolor='black', linewidth=0.1)
    
    axes[0][0].text(0.15, 0.1, labels[0], horizontalalignment='center', verticalalignment='center',
                    transform=axes[0][0].transAxes, fontsize=11)
    axes[0][1].text(0.15, 0.1, labels[1], horizontalalignment='center', verticalalignment='center',
                    transform=axes[0][1].transAxes, fontsize=11)
    axes[1][0].text(0.15, 0.1, labels[2], horizontalalignment='center', verticalalignment='center',
                    transform=axes[1][0].transAxes, fontsize=11)
    axes[1][1].text(0.15, 0.1, labels[3], horizontalalignment='center', verticalalignment='center',
                    transform=axes[1][1].transAxes, fontsize=11)
    
    mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
    # [xpos, ypos, length, width]
    cb_ax = f.add_axes([0.9, 0.1, 0.015, 0.8])
    cbar = f.colorbar(mappable, cax=cb_ax, orientation='vertical')
    plt.subplots_adjust(wspace=-0.05, hspace=0.05, 
                        top=0.9, bottom=0.1, 
                        left=0.1, right=0.9) 
    plt.ylabel(clabel) # Color bar label
    f.supxlabel('Longitude')
    f.supylabel('Latitude', x=0.07)
    
    return f

f = US_multiplot(hexagons=hexagons, usmap=usmap, yearly_stat=spec_yearly,
                 four_types=['mono-Si','CdTe','perovskite','triple'],
                 labels=['mono-Si','CdTe','Perovskite','Triple-Junction'],
                 vmin=0.94, vmax=1.06, clabel='Spectral Mismatch')

f = US_multiplot(hexagons=hexagons, usmap=usmap, yearly_stat=spec_yearly,
                 four_types=['multi-Si','CIGS','perovskite-si','a-Si'],
                 labels=['multi-Si','CIGS','Perovskite-Si','a-Si'],
                 vmin=0.94, vmax=1.06, clabel='Spectral Mismatch')

f = US_multiplot(hexagons=hexagons, usmap=usmap, yearly_stat=eta_rel_yearly,
                 four_types=['mono-Si','CdTe','CIGS','a-Si'],
                 labels=['mono-Si','CdTe','CIGS','a-Si'],
                 vmin=0.90, vmax=1.10, clabel='Relative Efficiency')


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', # Color blind cycle
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# Spectral Modifier
plt.figure(figsize=(6,4))
labels=types1+['PVSK','PVSK-Si','Triple Junct.']
spec_monthly.columns = labels
spec_monthly.groupby(level=1).mean().plot(color=CB_color_cycle)
plt.ylabel('Spectral Modifier', fontsize=11)
plt.xlabel('Month', fontsize=11)
plt.xticks(np.arange(1,13,1), fontsize=11)
plt.yticks(np.arange(0.96,1.20,0.04), fontsize=11)
plt.ylim(bottom=0.94, top=1.17)
plt.figure(figsize=(6,4))
spec_hourly.groupby(level=1).mean().plot(color=CB_color_cycle, legend=None)
plt.ylabel('Spectral Modifier', fontsize=11)
plt.xlabel('Hour', fontsize=11)
plt.xticks(np.arange(0,22,2), fontsize=11)
plt.yticks(np.arange(0.96,1.20,0.04), fontsize=11)
plt.ylim(bottom=0.94, top=1.17)
plt.xlim(left=4, right=21)


# Relative Efficiency  
plt.figure(figsize=(6,4))
eta_rel_monthly.groupby(level=1).mean().plot(color=CB_color_cycle)
plt.ylabel('Relative Efficiency', fontsize=11)
plt.xlabel('Month', fontsize=11)
plt.xticks(np.arange(1,13,1), fontsize=11)
plt.yticks(np.arange(0.88,1.10,0.04), fontsize=11)
plt.ylim(top=1.08, bottom=0.88)
plt.figure(figsize=(6,4))
eta_rel_hourly[['CdTe','CIGS','multi-Si','mono-Si']].groupby(level=1).mean().plot(color=CB_color_cycle)
plt.ylabel('Relative Efficiency', fontsize=11)
plt.xlabel('Hour', fontsize=11)
plt.xticks(np.arange(0,24,2), fontsize=11)
plt.yticks(np.arange(0.88,1.10,0.04), fontsize=11)
plt.xlim(left=4, right=21)
plt.ylim(top=1.08, bottom=0.88)


# TRENDS WITH WEATHER
def weather_plot(weather, effect):
   fig = plt.figure(figsize=(6,4))
   for t in effect.columns:
       y = effect[t]
       r2 = scipy.stats.linregress(x, y).rvalue
       plt.scatter(x, y, alpha=0.1)
       z = np.polyfit(x, y, 1)
       p = np.poly1d(z)
       plt.plot(x,p(x),"-", label=f'{t}: {round(r2,2)}')
   return fig, r2


# Precip
x = weather_yearly['precip']
fig, r2 = weather_plot(x, spec_yearly)
plt.xlabel('Yearly Mean Precipitable Water [mm] (Daylight Hours)')
plt.ylabel('Spectral Modifier')
#plt.ylim(top=1.1)
plt.legend()

# Altitude
x = locs.alt.astype(np.float64)
fig, r2 = weather_plot(x, spec_yearly)
plt.xlabel('Altitude [m]')
plt.ylabel('Spectral Modifier')
plt.legend()

# POA
x = weather_yearly['poa']/1000
fig, r2 = weather_plot(x, spec_yearly)
plt.xlabel('Yearly Global POA [kWh/m^2]')
plt.ylabel('Spectral Modifier')
plt.legend()

# Cloud
x = weather_yearly['cloud']
fig, r2 = weather_plot(x, mpp_yearly/1000)
plt.xlabel('Yearly % Daylight Hours with Cloud Cover (cloud type > 1)')
plt.ylabel('Energy Generated [kWh]')
plt.legend()


