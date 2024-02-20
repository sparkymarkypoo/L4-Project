def open_farms_and_pv_params(folder, location):

    from Data.open_data import open_aod550  
    import pandas as pd
        
    # Get PV model coefficints
    cec_params = pd.read_csv(rf'{folder}/CEC_coeffs.csv', index_col=0)
    cec_params = cec_params.T
    sandia_params = pd.read_csv(rf'{folder}/Sandia_coeffs.csv', index_col=0)
        
    # Open FARMS data
    farms_metadata = pd.read_csv(rf'{folder}/FARMS_Data/{location}_FARMS.csv', nrows=1)
    farms_data = pd.read_csv(rf'{folder}/FARMS_Data/{location}_FARMS.csv', skiprows=2)
    time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
    tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
    farms_time = pd.to_datetime(time_pd) + tz_offset
    farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                        'albedo':farms_data['Surface Albedo'], 'cloud':farms_data['Cloud Type'],
                        'ghi':farms_data['GHI'], 'dhi':farms_data['DHI'], 'dni':farms_data['DNI'],
                        'zenith':farms_data['Solar Zenith Angle'], 'azimuth':farms_data['Solar Azimuth Angle'],
                        'precip':farms_data['Precipitable Water']})
    farms_df.index = farms_data.index = farms_time
    farms_df['aod550'] = open_aod550(location, tz_offset)
       
    return farms_df, farms_data, cec_params, sandia_params



def open_validation_data(folder, location, module):
    
    import pandas as pd
    
    vali_data = pd.read_csv(rf'{folder}/{location}/{module}', skiprows=2, index_col=0)
    vali_metadata = pd.read_csv(rf'{folder}/{location}/{module}', nrows=1)
  
    vali_time = pd.to_datetime(vali_data.index).floor('Min')
    vali_data.index = vali_time
    vali_data = vali_data[~vali_data.index.duplicated(keep='first')]
    
    # 30 min offset
    vali_data = vali_data.resample('H').mean()
    vali_data = vali_data.fillna(0)
    vali_time = vali_data.index + pd.DateOffset(minutes=30)
    vali_data.index = vali_time
    vali_offset_time = vali_time - pd.DateOffset(minutes=30)
    vali_data = vali_data.append(pd.DataFrame(index=vali_offset_time))
    vali_data = vali_data.sort_index()
    vali_data = vali_data.interpolate(method='linear', axis=0)
    vali_data = vali_data.drop(index=vali_time)
      
    real_mpp = vali_data['Pmp (W)']
    real_poa = vali_data['POA irradiance CMP22 pyranometer (W/m2)']
    real_temp = vali_data['PV module back surface temperature (degC)']
    
    pv_name = vali_metadata['PV module identifier']
    
    return real_mpp, real_poa, real_temp, pv_name



def get_cec_ceoffs(cec_params, material, pv_name):
    import pvlib
    
    v_mp = float(cec_params[pv_name].loc['Vmp (V)'])
    i_mp = float(cec_params[pv_name].loc['Imp (A)'])
    v_oc = float(cec_params[pv_name].loc['Voc (V)'])
    i_sc = float(cec_params[pv_name].loc['Isc (A)'])
    alpha_sc = float((cec_params[pv_name].loc['Isc (%/C)']/100) * i_sc)
    beta_voc = float((cec_params[pv_name].loc['Voc (%/C)']/100) * v_oc)
    gamma_pmp = float(cec_params[pv_name].loc['Pm (%/C)'])
    cells_in_series = float(cec_params[pv_name].loc['Series Cells'])
    
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
                celltype=material, v_mp = v_mp, i_mp = i_mp, v_oc = v_oc, i_sc = i_sc,
                alpha_sc = alpha_sc, beta_voc = beta_voc, gamma_pmp = gamma_pmp,
                cells_in_series = cells_in_series)
    
    return I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust, alpha_sc



def cut_times(real_data, sim_data):  # for cutting data to when the real data has been measured
    sim_data = sim_data[(sim_data.index >= real_data.index[0]) & (sim_data.index <= real_data.index[-1])]
    return sim_data



def stats_calcs(real_power, sim_power):
      
    real_power = real_power.replace(0,np.nan)
    
    nmbe = 100* (sim_power - real_power).sum()/(real_power).sum()
    #nbe = 100* (sim_power - real_power)/(real_power)
    mbe = (sim_power - real_power).mean()
    rmse = np.sqrt((((real_power.dropna() - sim_power.dropna())**2).sum())/(len((sim_power-real_power).dropna())))
    nrmse = 100* rmse/(real_power.mean())   

    return nmbe, nrmse



def alternative_spectral_mismatch(farms_df, material, module, location, trans_model):
                
    from pvlib.spectrum import spectral_factor_sapm, spectral_factor_firstsolar, spectral_factor_caballero
    
    if material == 'amorphous':
        material = 'asi'
    else:
        material = material.lower()
    
    if location == 'Golden':
        tilt = 40
    elif location == 'Eugene':
        tilt = 44
    elif location == 'Cocoa':
        tilt = 28.5
    
    # Transposition
    farms_df['dni_extra'] = pvlib.irradiance.get_extra_radiation(farms_df.index, solar_constant=1366.1, method='spencer')
    farms_df['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=farms_df['zenith'], model= 'gueymard2003')

    farms_df['poa_trans'] = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=180,
                                solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                                dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],
                                dni_extra=farms_df['dni_extra'], airmass=farms_df['airmass'], albedo=farms_df['albedo'],
                                model=trans_model, model_perez='sandiacomposite1988').poa_global
    # model = 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    
    alt_mismatch = pd.DataFrame(columns=['sapm', 'firstsolar', 'caballero'])
    
    # Mismatch
    alt_mismatch['sapm'] = spectral_factor_sapm(airmass_absolute=farms_df['airmass'], module=module)

    alt_mismatch['firstsolar'] = spectral_factor_firstsolar(precipitable_water=farms_df['precip'], airmass_absolute=farms_df['airmass'],
                                            module_type=material, min_precipitable_water=0.1, max_precipitable_water=8)

    alt_mismatch['caballero'] = spectral_factor_caballero(precipitable_water=farms_df['precip'], airmass_absolute=farms_df['airmass'],
                                          aod500=farms_df['aod550'], module_type=material)

    return alt_mismatch, farms_df



def temperature_models(farms_df, sp_cec_params, sp_sandia_params, model):

    if model == 'faiman':    
        temp_cell = pvlib.temperature.faiman(farms_df['poa_global'], farms_df['temp_air'], farms_df['wind_speed'])
    
    elif model == 'pvsyst':
        temp_cell = pvlib.temperature.pvsyst_cell(farms_df['poa_global'], farms_df['temp_air'], farms_df['wind_speed'],
                                                  module_efficiency=float(sp_cec_params.loc['Eff (%)'])/100)
    elif model == 'sapm':
        temp_cell = pvlib.temperature.sapm_cell(farms_df['poa_global'], farms_df['temp_air'], farms_df['wind_speed'],
                                                        a=float(sp_sandia_params['a']), b=float(sp_sandia_params['b']),
                                                        deltaT=float(sp_sandia_params['d(Tc)']))
      
    elif model == 'noct':
        temp_cell = pvlib.temperature.noct_sam(poa_global=farms_df['poa_global'], temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                                               noct=float(sp_cec_params.loc['NOCT']), module_efficiency=float(sp_cec_params.loc['Eff (%)'])/100,
                                               effective_irradiance=farms_df['effective_irradiance'], array_height=1, mount_standoff=4)
    
    else: print('Not a valid model choice. Pick "faiman", "pvsyst", "sapm" or "noct"')
    
    return temp_cell




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pvlib_files.spec_response_function import calc_spectral_modifier
import pvlib
import os

folder = r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models'
locations = ['Golden', 'Eugene', 'Cocoa']
sum_stats = pd.DataFrame()
sum_temp_stats = pd.DataFrame()
for location in locations:

    # Get FARMS data
    full_farms_df, full_farms_data, cec_params, sandia_params = open_farms_and_pv_params(folder, location)
    
    # Loop for different types (note - loops are used to allow this script to be entirely modular. There may be faster options)
    arr = os.listdir(f'{folder}/{location}')
    stats = pd.DataFrame(columns=arr, index = ['nmbe', 'nrmse'])
    temp_stats = pd.DataFrame(columns=arr, index = ['nmbe', 'nrmse'])
    for a in arr:
    
        # Get measured data
        real_mpp, real_poa, real_temp, pv_name = open_validation_data(folder, location, module=a)
        
        # Cut data to match (helps analysis later)
        farms_df = cut_times(real_mpp, full_farms_df).copy()
        farms_data = cut_times(real_mpp, full_farms_data).copy()
       
        # Get effective irradiance
        spec_mismatch, farms_df['poa_global'], material = calc_spectral_modifier(str(pv_name.values), farms_data)
        farms_df['effective_irradiance'] = farms_df['poa_global'] * spec_mismatch
        #farms_df['effective_irradiance'] = real_poa * spec_mismatch
      
        # Calculate alternative spectral mismatches
        alt_mismatch, farms_df = alternative_spectral_mismatch(farms_df, material=material, module=sandia_params.loc[pv_name],
                                                               location=location, trans_model='isotropic')
        
        farms_df['poa_global'] = farms_df['poa_trans']
        farms_df['effective_irradiance'] = farms_df['poa_trans'] * spec_mismatch
               
        # Estimate CEC params from specsheet
        I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust, alpha_sc = get_cec_ceoffs(cec_params, material, pv_name)
                                       
        # PV temperature model
        cec_temp_cell = temperature_models(farms_df, sp_cec_params=cec_params[pv_name],
                                           sp_sandia_params=sandia_params.loc[pv_name], model='pvsyst')
        
        # Run CEC model
        cec_coeffs = pvlib.pvsystem.calcparams_cec(farms_df['effective_irradiance'], cec_temp_cell, float(alpha_sc),
                                                   a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)
        cec_mpp = pvlib.pvsystem.max_power_point(*cec_coeffs, method='newton').p_mp
        
        # Run Sandia model
        sandia_temp_cell = cec_temp_cell
        
        # Have to repeat params as must be same length as temp/irrad
        sandia_params_repeated = pd.DataFrame(data=np.repeat(sandia_params.loc[pv_name].values, len(sandia_temp_cell), axis=0),
                                              columns=sandia_params.columns, index=sandia_temp_cell.index)
        sandia_mpp = pvlib.pvsystem.sapm(farms_df['effective_irradiance'], sandia_temp_cell, sandia_params_repeated).p_mp
        
        # Calc power (overall) errors
        stats[a] = stats_calcs(real_mpp, cec_mpp*0.93)
        #stats[a] = stats_calcs(real_mpp, sandia_mpp)
        stats = stats.rename(columns={a: material})
        
        temp_stats[a] = stats_calcs(real_temp, cec_temp_cell)
        temp_stats = temp_stats.rename(columns={a: material})
    
    # Merge columns with same PV type together
    def same_merge(x): return ','.join(x[x.notnull()].astype(str))
    stats = stats.groupby(level=0, axis=1).apply(lambda x: x.apply(np.mean, axis=1))
    temp_stats = temp_stats.groupby(level=0, axis=1).apply(lambda x: x.apply(np.mean, axis=1))
    
    if sum_stats.empty == True:
        sum_stats = abs(stats)
        sum_temp_stats = abs(temp_stats)
    
    else:
        sum_stats = abs(stats) + sum_stats
        sum_temp_stats = abs(temp_stats) + sum_temp_stats
        
        
        
    # # Calc poa errors
    # trans_models = ['isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez']
    # poa_stats = pd.DataFrame(columns = trans_models, index = ['nmbe', 'nrmse'])
    # for t in trans_models:
        
    #     alt_mismatch, farms_df = alternative_spectral_mismatch(farms_df, material=material, module=sandia_params.loc[pv_name], location=location,
    #                                                             trans_model=t)
    #     poa_stats[t] = stats_calcs(real_poa, farms_df['poa_trans'])
        
    # poa_stats['FARMS'] = stats_calcs(real_poa, farms_df['poa_global'])   
    # print(poa_stats)
        
    

fin_stats = sum_stats/3
fin_temp_stats = sum_temp_stats/3


#print(fin_stats.T)
print(fin_temp_stats.T)






# # PLOT
# import matplotlib.pyplot as plt
# plt.figure()
# # pc = plt.scatter(real_mpp, cec_mpp, c=cec_temp_cell, cmap='jet')
# # pc = plt.scatter(real_mpp, cec_mpp, c=farms_df['cloud'], cmap='jet')
# pc = plt.scatter(real_mpp, cec_mpp, c=spec_mismatch, cmap='jet')
# # pc = plt.scatter(real_mpp, sandia_mpp, c=spec_mismatch, cmap='jet')
# plt.colorbar(label='spectral modifier', ax=plt.gca())
# pc.set_alpha(0.5)
# plt.grid(alpha=0.5)
# plt.xlabel('Measured Power [W]')
# plt.ylabel('Simulated Power [W]')
# plt.title(a)
# plt.plot([0,real_mpp.max()], [0,real_mpp.max()])
# plt.show()
