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
metadata = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', nrows=1)
data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', skiprows=2)


# Define location and system
#tilt = 30 #round(latitude,1)
time_pd = data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
time = pd.to_datetime(time_pd)


# Get effective irradiance
df = pd.DataFrame({'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed'],
                    'albedo':data['Surface Albedo'], 'cloud':data['Cloud Type']})
mpp = pd.DataFrame(columns = types)
eta_rel = pd.DataFrame(columns = types)
i = 0
spec_mismatch = pd.DataFrame(columns=types)
for material in modules.loc['Technology']:
    spec_mismatch[material], df['poa_global'] = calc_spectral_modifier(material)
    df['effective_irradiance'] = df['poa_global'] * spec_mismatch[material]
    df = df.fillna(0)
    
    
    # Calculate cell parameters
    temp_cell = pvlib.temperature.faiman(data['poa_global'], data['Temperature'], data['Wind Speed'])
    cec_params = pvlib.pvsystem.calcparams_cec(df['effective_irradiance'], temp_cell, modules.loc['alpha_sc'].iloc[i],
                                               modules.loc['a_ref'].iloc[i], modules.loc['I_L_ref'].iloc[i],
                                               modules.loc['I_o_ref'].iloc[i], modules.loc['R_sh_ref'].iloc[i],
                                               modules.loc['R_s'].iloc[i], modules.loc['Adjust'].iloc[i])


    # Max power point
    mpp[material] = pvlib.pvsystem.max_power_point(*cec_params, method='newton').p_mp
    eta_rel[material] = (mpp[material] / modules.loc['STC'].iloc[i]) / (df['poa_global'] / 1000)
    i=i+1

    # # Plots
    # plt.figure()
    # pc = plt.scatter(df['poa_global'], eta_rel[material], c=temp_cell, cmap='jet')
    # plt.colorbar(label='Temperature [C]', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.ylim(0.48)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Relative efficiency [-]')
    # plt.show()
    
    # plt.figure()
    # pc = plt.scatter(df['poa_global'], spec_mismatch[material], c=df['cloud'], cmap='jet')
    # plt.colorbar(label='Cloud Type', ax=plt.gca())
    # pc.set_alpha(0.25)
    # plt.grid(alpha=0.5)
    # plt.xlabel('POA Irradiance [W/m²]')
    # plt.ylabel('Spectral Mismatch (-)')
    # plt.show()
    

mpp.index = time
eta_rel.index = time
df.index = time
spec_mismatch.index = time

mpp_monthly = mpp.resample('M').apply(np.trapz)
eta_rel_monthly = eta_rel.resample('M').apply(np.mean)
poa_monthly = df['poa_global'].resample('M').apply(np.trapz)
spec_mismatch_monthly = spec_mismatch[spec_mismatch>0].resample('M').apply(np.mean)

ax = plt.figure(figsize=(10,6))
ax = eta_rel_monthly.plot()
ax.set_ylabel("Mean Relative Efficiency")

ax = plt.figure(figsize=(10,6))
ax = spec_mismatch_monthly.plot()
ax.set_ylabel("Mean Spectral Mismatch")


plt.figure()
i=0
for t in types:
    #eta_power_monthly = (mpp_monthly['mono-Si'] / modules.loc['STC'].iloc[3]) / (poa_monthly / 1000)
    eta_power_monthly = (mpp_monthly[t] / modules.loc['STC'].iloc[i]) / (poa_monthly / 1000)
    eta_power_monthly.name = t
    i=i+1
    ax = eta_power_monthly.plot()
    ax.set_ylabel("Mean Relative Efficiency")
ax.legend()


