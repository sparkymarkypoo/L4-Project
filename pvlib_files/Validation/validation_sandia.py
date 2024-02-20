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
farms_time = pd.to_datetime(time_pd) + pd.DateOffset(hours=float(farms_metadata['Local Time Zone']))

# separate stuff into farms_df as farms_data is big with long labels
farms_df = pd.DataFrame({'temp_air':farms_data['Temperature'], 'wind_speed':farms_data['Wind Speed'],
                    'albedo':farms_data['Surface Albedo'], 'cloud':farms_data['Cloud Type'],
                    'ghi':farms_data['GHI'], 'dhi':farms_data['DHI'], 'dni':farms_data['DNI'],
                    'zenith':farms_data['Solar Zenith Angle'], 'azimuth':farms_data['Solar Azimuth Angle'],
                    'precip':farms_data['Precipitable Water']})
farms_df.index = farms_data.index = farms_time
farms_df['aod550'] = open_aod550(location='Sandia', tz_offset=-7)



# Transposition
farms_df['dni_extra'] = pvlib.irradiance.get_extra_radiation(farms_df.index, solar_constant=1366.1, method='spencer', epoch_year=2014)
farms_df['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith=farms_df['zenith'], model='kastenyoung1989')

farms_df['poa_trans'] = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth,
                            solar_zenith=farms_df['zenith'], solar_azimuth=farms_df['azimuth'],
                            dni=farms_df['dni'], ghi=farms_df['ghi'], dhi=farms_df['dhi'],
                            dni_extra=farms_df['dni_extra'], airmass=farms_df['airmass'], albedo=farms_df['albedo'],
                            model='haydavies').poa_global
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
#farms_df['effective_irradiance'] = real_poa# * spec_mismatch
farms_df['effective_irradiance'] = farms_df['poa_global'] * spec_mismatch


def stats_calcs(real_power, sim_power):
    
    real_power = real_power.replace(0,np.nan)
    # Have to set this as any unrecorded points are set as 0 in the data
    # so now only considering daylight recorded points
       
    nmbe = 100* (sim_power - real_power).sum()/(real_power).sum()
    #nbe = 100* (sim_power - real_power)/(real_power)
    mbe = (sim_power - real_power).mean()
    rmse = np.sqrt((((real_power.dropna() - sim_power.dropna())**2).sum())/(len((sim_power-real_power).dropna())))
    nrmse = 100* rmse/(real_power.mean())   

    return mbe, nmbe, rmse, nrmse


# Run model for all spectral models
stats = pd.DataFrame(columns=['farms-a', 'farms-b', 'sapm', 'firstsolar', 'caballero'], index = ['mbe', 'nmbe', 'rmse', 'nrmse'])

for c in stats:
    
    if c=='farms-a':
        poa_glo = farms_df['poa_global']
        poa_eff = farms_df['effective_irradiance']
        
    elif c=='farms-b':
        poa_glo = farms_df['poa_trans']
        poa_eff = farms_df['poa_trans'] * spec_mismatch
    
    else:
        poa_glo = farms_df['poa_trans']
        poa_eff = farms_df['poa_trans'] * alt_mismatch[c]


    # Calculate cec output
    cec_temp_cell = pvlib.temperature.faiman(poa_glo, farms_df['temp_air'], farms_df['wind_speed'])
    cec_coeffs = pvlib.pvsystem.calcparams_cec(poa_eff, cec_temp_cell, alpha_sc,
                                               a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)
          
    # CEC max power point
    cec_mpp = pvlib.pvsystem.max_power_point(*cec_coeffs, method='newton').p_mp
    cec_mpp = cec_mpp * 12 * 0.93 # Scaling + losses


    # KOWALSKI, ANALYSIS!
    cec_mpp = cec_mpp.fillna(0)
    cec_temp_cell = cec_temp_cell.fillna(0)
    
    stats[c] = stats_calcs(real_mpp, cec_mpp)   
    



# Calc temp errors
temp_models = ['faiman', 'pvsyst', 'sapm', 'noct']
temp_stats = pd.DataFrame(columns = temp_models, index = ['mbe', 'nmbe', 'rmse', 'nrmse'])
temp = pd.DataFrame(columns = temp_models)

# Using default values where none available
temp['faiman'] = pvlib.temperature.faiman(poa_global=farms_df['poa_global'], temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                         u0=25.0, u1=6.84)

temp['pvsyst'] = pvlib.temperature.pvsyst_cell(poa_global=farms_df['poa_global'], temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                              u_c=29.0, u_v=0.0, module_efficiency=eff, alpha_absorption=0.9)

temp['sapm'] = pvlib.temperature.sapm_cell(poa_global=farms_df['poa_global'], temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                            a=module.loc['A'], b=module.loc['B'], deltaT=module.loc['DTC'], irrad_ref=1000.0)

temp['noct'] = pvlib.temperature.noct_sam(poa_global=farms_df['poa_global'], temp_air=farms_df['temp_air'], wind_speed=farms_df['wind_speed'],
                           noct=noct, module_efficiency=eff, effective_irradiance=farms_df['effective_irradiance'],
                           transmittance_absorptance=0.9, array_height=1, mount_standoff=4)

for t in temp_models:
    
    temp_stats[t] = stats_calcs(real_temp, temp[t])

    # # PLOT
    # plt.figure()
    # pc = plt.scatter(real_temp, temp[t], c=real_poa, cmap='jet')
    # plt.colorbar(label='POA [W/m^2]', ax=plt.gca())
    # pc.set_alpha(0.5)
    # plt.grid(alpha=0.5)
    # plt.xlabel('Measured Temp')
    # plt.ylabel('Simulated Temp')
    # plt.plot([0,real_temp.max()], [0,real_temp.max()])
    # plt.title(t)
    # plt.show()



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

# Calc poa errors
trans_models = ['isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez']
poa_stats = pd.DataFrame(columns = trans_models, index = ['mbe', 'nmbe', 'rmse', 'nrmse'])
for t in trans_models:
    
    farms_df = transposition_poa(farms_df, module=module, trans_model=t)
    poa_stats[t] = stats_calcs(real_poa, farms_df['poa_trans'])
poa_stats['FARMS'] = stats_calcs(real_poa, farms_df['poa_global'])




# PLOT
plt.figure()
# pc = plt.scatter(real_mpp, cec_mpp, c=cec_temp_cell, cmap='jet')
# pc = plt.scatter(real_mpp, cec_mpp, c=farms_df['cloud'], cmap='jet')
pc = plt.scatter(real_mpp, cec_mpp, c=spec_mismatch, cmap='jet')
plt.colorbar(label='spectral modifier', ax=plt.gca())
pc.set_alpha(0.5)
plt.grid(alpha=0.5)
plt.xlabel('Measured Power [W]')
plt.ylabel('Simulated Power [W]')
plt.plot([0,real_mpp.max()], [0,real_mpp.max()])
plt.show()
