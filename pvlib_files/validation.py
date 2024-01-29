# Import pvlib stuff
import pvlib

# Import general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Get PV model coefficints
cec_coeffs = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/CEC_coeffs.csv')
sandia_coeffs = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/Sandia_coeffs.csv')

#for location.....:

# Open FARMS data
farms_metadata = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/FARMS_Data/Golden_2012_FARMS.csv', nrows=1)
farms_data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/FARMS_Data/Golden_2012_FARMS.csv', skiprows=2)
time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
farms_time = pd.to_datetime(time_pd)

farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                    'albedo':farms_data['Surface Albedo'], 'cloud':farms_data['Cloud Type']})


# Get validation data
arr = os.listdir('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/Golden')
for a in arr:
    vali_metadata = pd.read_csv(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/Golden/{a}', nrows=1)
    vali_data = pd.read_csv(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/Golden/{a}', skiprows=2, index_col=0)
    vali_time = pd.to_datetime(vali_data.index).floor('Min')
    
    vali_df = pd.DataFrame({'p_mp':vali_data['Pmp (W)'], 'i_mp':vali_data['Imp (A)'], 'v_mp':vali_data['Vmp (V)'], 'temp_air':vali_data['Dry bulb temperature (degC)'],
                       'poa':vali_data['POA irradiance CMP22 pyranometer (W/m2)'], 'ghi':vali_data['Global horizontal irradiance (W/m2)'],
                       'dni':vali_data['Direct normal irradiance (W/m2)'], 'dhi':vali_data['Diffuse horizontal irradiance (W/m2)']})
    vali_df.index = vali_time


mpp = pd.DataFrame(columns = types)
eta_rel = pd.DataFrame(columns = types)
i = 0
spec_mismatch = pd.DataFrame(columns=types)
for material in modules.loc['Technology']:
    spec_mismatch[material], df['poa_global'] = calc_spectral_modifier(material)
    df['effective_irradiance'] = df['poa_global'] * spec_mismatch[material]
    df = df.fillna(0)
    
    
    # Calculate cell parameters
    temp_cell = pvlib.temperature.faiman(data['GHI'], data['Temperature'], data['Wind Speed'])
    cec_params = pvlib.pvsystem.calcparams_cec(df['effective_irradiance'], temp_cell, modules.loc['alpha_sc'].iloc[i],
                                               modules.loc['a_ref'].iloc[i], modules.loc['I_L_ref'].iloc[i],
                                               modules.loc['I_o_ref'].iloc[i], modules.loc['R_sh_ref'].iloc[i],
                                               modules.loc['R_s'].iloc[i], modules.loc['Adjust'].iloc[i])


    # Max power point
    mpp[material] = pvlib.pvsystem.max_power_point(*cec_params, method='newton').p_mp
    eta_rel[material] = (mpp[material] / modules.loc['STC'].iloc[i]) / (df['poa_global'] / 1000)
    i=i+1

