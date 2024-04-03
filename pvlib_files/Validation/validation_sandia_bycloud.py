# Import pvlib stuff
import pvlib
from pvlib_files.spec_response_function import calc_spectral_modifier
from pvlib.spectrum import spectral_factor_sapm, spectral_factor_firstsolar, spectral_factor_caballero


# Import general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data.open_data import open_aod550

folder = r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/Sandia'


# Location
tilt = 35
azimuth = 180
latitude = 35.05
longitude = -106.54


# CEC params
noct = 45
area = 1.650 * 0.992
v_mp = 31.3
i_mp = 8.80
eff = (v_mp*i_mp)/(1000*area)
v_oc = 38.3
i_sc = 9.31
alpha_sc = (0.053/100) * i_sc
beta_voc = (-0.31/100) * v_oc
gamma_pmp = -0.41
cells_in_series = 60
material = 'monoSi'
I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
    celltype=material, v_mp = v_mp, i_mp = i_mp, v_oc = v_oc, i_sc = i_sc,
    alpha_sc = alpha_sc, beta_voc = beta_voc, gamma_pmp = gamma_pmp,
    cells_in_series = cells_in_series)



# Open FARMS data
farms_metadata = pd.read_csv(rf'{folder}/FARMS.csv', nrows=1)
farms_data = pd.read_csv(rf'{folder}/FARMS.csv', skiprows=2)
time_pd = farms_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
tz_offset = pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))
farms_time = pd.to_datetime(time_pd) + tz_offset

# separate stuff into farms_df as farms_data is big with long labels
farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                    'albedo':farms_data['Surface Albedo'], 'cloud':farms_data['Cloud Type'],
                    'ghi':farms_data['GHI'], 'dhi':farms_data['DHI'], 'dni':farms_data['DNI'],
                    'zenith':farms_data['Solar Zenith Angle'], 'azimuth':farms_data['Solar Azimuth Angle'],
                    'precip':farms_data['Precipitable Water']})
farms_df.index = farms_data.index = farms_time
farms_df['aod550'] = open_aod550(location='Sandia', tz_offset=tz_offset)
aod_percentiles, precip_percentiles = np.zeros(9), np.zeros(9)
for i in range(0, 9):
    aod_percentiles[i] = np.percentile(farms_df['aod550'], 10*(i+1), axis=0)
    precip_percentiles[i] = np.percentile(farms_df['precip'], 10*(i+1), axis=0)




# Transposition
farms_df['dni_extra'] = pvlib.irradiance.get_extra_radiation(farms_df.index, solar_constant=1366.1, method='spencer', epoch_year=2014)
farms_df['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=farms_df['zenith'], model='kastenyoung1989')

cloud = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth,
                            solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                            dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],
                            dni_extra=farms_df['dni_extra'], airmass=farms_df['airmass'], albedo=farms_df['albedo'],
                            model='klucher').poa_global
clear = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth,
                            solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                            dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],
                            dni_extra=farms_df['dni_extra'], airmass=farms_df['airmass'], albedo=farms_df['albedo'],
                            model='isotropic').poa_global
farms_df['poa_trans'] = cloud
farms_df['poa_trans'][farms_df['cloud']<2] = clear[farms_df['cloud']<2]
# 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'



# Calculate alternative spectral mismatch factors
alt_mismatch = pd.DataFrame(columns=['sapm', 'firstsolar', 'caballero'])

sandia_mods = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
module = sandia_mods['Canadian_Solar_CS6X_300M__2013_']
alt_mismatch['sapm'] = spectral_factor_sapm(airmass_absolute=farms_df['airmass'], module=module)

alt_mismatch['firstsolar'] = spectral_factor_firstsolar(precipitable_water=farms_df['precip'], airmass_absolute=farms_df['airmass'],
                                        module_type='monosi', min_precipitable_water=0.1, max_precipitable_water=8)

alt_mismatch['caballero'] = spectral_factor_caballero(precipitable_water=farms_df['precip'], airmass_absolute=farms_df['airmass'],
                                      aod500=farms_df['aod550'], module_type='monosi')


# Get validation data
vali_data = pd.read_csv(rf'{folder}/Scenario2.csv', index_col=0)
time_pd = vali_data[['Year', 'Month', 'Day', 'Hour']]
vali_time = pd.to_datetime(time_pd) - pd.DateOffset(minutes=30)
vali_data.index = vali_time

# 30 min offset
bill = vali_data.copy()
vali_offset_time = vali_time + pd.DateOffset(minutes=30)
vali_data = vali_data.append(pd.DataFrame(index=vali_offset_time))
vali_data = vali_data.sort_index()
vali_data = vali_data.interpolate(method='linear', axis=0)
vali_data = vali_data.drop(index=vali_time)

# Trim the times to match
vali_data = vali_data[(vali_data.index <= farms_df.index[-1])]
def trim_data(simulated, measured):
    simulated = simulated[(simulated.index >= measured.index[0])]                        # trim to same dates
    simulated = simulated[~((simulated.index.month == 2) & (simulated.index.day == 29))] # remove leap day
    return simulated
farms_df = trim_data(farms_df, vali_data)
farms_data = trim_data(farms_data, vali_data)
alt_mismatch = trim_data(alt_mismatch, vali_data)

real_mpp = vali_data['Measured DC power']
real_poa = vali_data['Measured POA']
real_temp = vali_data['Measured module temperature']



# Get effective irradiance
spec_mismatch, farms_df['poa_global'], material = calc_spectral_modifier('monoSi', farms_data)


def stats_calcs(real_power, sim_power):
    
    real_power = real_power.replace(0,np.nan)
    # Have to set this as any unrecorded points are set as 0 in the data
    # so now only considering daylight recorded points
       
    nmbe = 100* (sim_power - real_power).sum()/(real_power).sum()
    #nbe = 100* (sim_power - real_power)/(real_power)
    mbe = (sim_power - real_power).mean()
    rmse = np.sqrt((((real_power.dropna() - sim_power.dropna())**2).sum())/(len((sim_power-real_power).dropna())))
    nrmse = 100* rmse/(real_power.mean())   

    return nmbe, nrmse


spec_models = ['farms-a', 'farms-b', 'sapm', 'firstsolar', 'caballero', 'none']
pow_nmbe = pd.DataFrame(index=range(0,10), columns = spec_models)
pow_nrmse = pd.DataFrame(index=range(0,10), columns = spec_models)
cloud_pow = pd.DataFrame(index = ['nmbe','nrmse'], columns = spec_models)
clear_pow = pd.DataFrame(index = ['nmbe','nrmse'], columns = spec_models)
# Run model for all spectral models
for t in spec_models:
    
    if t=='farms-a':
        poa_glo = farms_df['poa_global']
        poa_eff = farms_df['poa_global'] * spec_mismatch
        
    elif t=='farms-b':
        poa_glo = farms_df['poa_trans']
        poa_eff = farms_df['poa_trans'] * spec_mismatch
        
    elif t=='none':
        poa_glo = farms_df['poa_trans']
        poa_eff = farms_df['poa_trans']
    
    else:
        poa_glo = farms_df['poa_trans']
        poa_eff = farms_df['poa_trans'] * alt_mismatch[t]


    # Calculate cec output
    cec_temp_cell = pvlib.temperature.pvsyst_cell(poa_global=poa_glo, temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                                  u_c=29.0, u_v=0.0, module_efficiency=eff, alpha_absorption=0.9)
    cec_coeffs = pvlib.pvsystem.calcparams_cec(poa_eff, cec_temp_cell, alpha_sc,
                                               a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)
       
    # CEC max power point
    cec_mpp = pvlib.pvsystem.max_power_point(*cec_coeffs, method='newton').p_mp
    cec_mpp = cec_mpp * 12 * 0.96 # Scaling + losses


    # KOWALSKI, ANALYSIS!
    cec_mpp = cec_mpp.fillna(0)
    
    for cloud in range(0,10):   
        cloud_cec = cec_mpp[farms_df['cloud']==cloud].copy()
        cloud_real = real_mpp[farms_df['cloud']==cloud].copy()    
        pow_nmbe[t].loc[cloud], pow_nrmse[t].loc[cloud] = stats_calcs(cloud_real, cloud_cec)
        
    # for aod in range(0,9):    
    #     cloud_cec = cec_mpp[farms_df['aod550']>=aod_percentiles[aod]].copy()
    #     cloud_real = real_mpp[farms_df['aod550']>=aod_percentiles[aod]].copy()    
    #     pow_nmbe[t].loc[aod], pow_nrmse[t].loc[aod] = stats_calcs(cloud_real, cloud_cec)
        
    # for precip in range(0,9):    
    #     cloud_cec = cec_mpp[farms_df['precip']>=precip_percentiles[precip]].copy()
    #     cloud_real = real_mpp[farms_df['precip']>=precip_percentiles[precip]].copy()    
    #     pow_nmbe[t].loc[precip], pow_nrmse[t].loc[precip] = stats_calcs(cloud_real, cloud_cec)  

    cloud_cec = cec_mpp[farms_df['cloud']>1].copy()
    cloud_real = real_mpp[farms_df['cloud']>1].copy()
    clear_cec = cec_mpp[farms_df['cloud']<2].copy()
    clear_real = real_mpp[farms_df['cloud']<2].copy()

    cloud_pow[t].loc['nmbe'], cloud_pow[t].loc['nrmse'] = stats_calcs(cloud_real, cloud_cec)
    clear_pow[t].loc['nmbe'], clear_pow[t].loc['nrmse'] = stats_calcs(clear_real, clear_cec)

print(clear_pow)
print(cloud_pow)

def transposition_poa(farms_df, module, trans_model):    
        
    # Transposition
    farms_df['dni_extra'] = pvlib.irradiance.get_extra_radiation(farms_df.index, solar_constant=1366.1, method='spencer', epoch_year=2014)
    farms_df['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=farms_df['zenith'], model= 'gueymard2003')

    farms_df['poa_trans'] = pvlib.irradiance.get_total_irradiance(surface_tilt=35, surface_azimuth=180,
                                solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                                dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],
                                dni_extra=farms_df['dni_extra'], airmass=farms_df['airmass'], albedo=farms_df['albedo'],
                                model=trans_model, model_perez='sandiacomposite1988').poa_global
    farms_df['poa_trans'] = farms_df['poa_trans'].fillna(0)  # some models give nan instead of 0 when no light present
    # model = 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    
    
    return farms_df


# Cloud types
trans_models = ['isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez']
poa_nmbe = pd.DataFrame(index=range(0,10), columns = trans_models)
poa_nrmse = pd.DataFrame(index=range(0,10), columns = trans_models)
pd.options.mode.chained_assignment = None  # default='warn'
for cloud in poa_nmbe.index:
    cloud_farms = farms_df[farms_df['cloud']==cloud].copy()
    cloud_real = real_poa[farms_df['cloud']==cloud].copy()
    for t in trans_models:       
        cloud_farms = transposition_poa(farms_df=cloud_farms, module=module, trans_model=t)
        poa_nmbe[t].loc[cloud], poa_nrmse[t].loc[cloud] = stats_calcs(cloud_real, cloud_farms['poa_trans'])        
    if cloud == 0:
        poa_nmbe['FARMS'], poa_nrmse['FARMS'] = stats_calcs(cloud_real, cloud_farms['poa_global'])
    else: 
        poa_nmbe['FARMS'].loc[cloud], poa_nrmse['FARMS'].loc[cloud] = stats_calcs(cloud_real, cloud_farms['poa_global'])

poa_nmbe = poa_nmbe.dropna()
poa_nrmse = poa_nrmse.dropna()


# # Cloud types
# all_poa_nmbe = pd.Series(index = trans_models)
# all_poa_nrmse = pd.Series(index = trans_models)
# for cloud in poa_nmbe.index:
#     cloud_farms = farms_df[farms_df['cloud']>0].copy()
#     cloud_real = real_poa[farms_df['cloud']>0].copy()
#     for t in trans_models:       
#         cloud_farms = transposition_poa(farms_df=cloud_farms, module=module, trans_model=t)
#         all_poa_nmbe[t], all_poa_nrmse[t] = stats_calcs(cloud_real, cloud_farms['poa_trans'])           
#     if cloud == 0:
#         all_poa_nmbe['FARMS'], all_poa_nrmse['FARMS'] = stats_calcs(cloud_real, cloud_farms['poa_global'])
#     else: 
#         all_poa_nmbe['FARMS'], all_poa_nrmse['FARMS'] = stats_calcs(cloud_real, cloud_farms['poa_global'])




# # PLOT
# plt.figure()
# # pc = plt.scatter(real_mpp, cec_mpp, c=cec_temp_cell, cmap='jet')
# # pc = plt.scatter(real_mpp, cec_mpp, c=farms_df['cloud'], cmap='jet')
# pc = plt.scatter(real_mpp, cec_mpp, c=spec_mismatch, cmap='jet')
# plt.colorbar(label='spectral modifier', ax=plt.gca())
# pc.set_alpha(0.5)
# plt.grid(alpha=0.5)
# plt.xlabel('Measured Power [W]')
# plt.ylabel('Simulated Power [W]')
# plt.plot([0,real_mpp.max()], [0,real_mpp.max()])
# plt.show()
