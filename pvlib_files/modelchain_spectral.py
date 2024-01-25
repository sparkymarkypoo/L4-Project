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


# Choose year:
year = 2021

# Get the module and inverter specifications from SAM
#sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
#module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
module = cec_modules['Zytech_Solar_ZT320P']
inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# TEMPORARY ADJUSTMENT TO SAVE TIME
coords = coords[coords['NAME']=='Los Angeles']
# Loop time!
for index, row in coords.iterrows():
    
    # Get NSRDB data
    latitude = round(float(row['LAT']),3)
    longitude = round(float(row['LONG']),3)
    name = row['NAME']
    interval = 60 # in minutes  
    
    metadata = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', nrows=1)
    data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', skiprows=2)


    # Define location and system
    location = Location(latitude=latitude,longitude=longitude,
                        altitude=pvlib.location.lookup_altitude(latitude=latitude, longitude=longitude))
    tilt = 30 #round(latitude,1)
    system = PVSystem(surface_azimuth=180, surface_tilt=tilt, module_parameters=module, inverter_parameters=inverter,
                      temperature_model_parameters=temperature)


    ## SAPM MODEL
    # Reformat for modelling
    df = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'],
                       'dhi':data['DHI'], 'temp_air':data['Temperature'],
                       'wind_speed':data['Wind Speed'], 'albedo':data['Surface Albedo'],
                       'cloud':data['Cloud Type']})
    time_pd = data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
    time = pd.to_datetime(time_pd)
    df.index = time
    
    # Define and run model
    modelchain_sapm = ModelChain(system, location, aoi_model='no_loss', spectral_model='sapm')#, dc_model='sapm') 
    sapm_run = modelchain_sapm.run_model(df)
    ##


    ## SPECTRAL MODEL
    # Get POA irradiance
    df2 = pd.DataFrame({'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed'],
                        'albedo':data['Surface Albedo'], 'cloud':data['Cloud Type']})
    df2.index = time
    spec_mismatch, spec_poa = calc_spectral_modifier()
    spec_results = []
    for material in spec_mismatch:
        df2['effective_irradiance'] = spec_mismatch[material] * spec_poa
        df2 = df2.fillna(0)
        
        # Define and run model
        modelchain_spect = ModelChain(system, location, dc_model='sapm', spectral_model='no_loss', aoi_model='no_loss') 
        spect_run = modelchain_spect.run_model_from_effective_irradiance(df2)
        spec_results.append(spect_run.results)
    
    
    
    
    # Examine results
    spect_power_monthly = spect_run.results.dc.p_mp.resample('M').apply(np.trapz)
    sapm_power_monthly = sapm_run.results.dc.p_mp.resample('M').apply(np.trapz)
    
    # m = {'spect':spec_mismatch, 'sapm':sapm_run.results.spectral_modifier}
    # mod = pd.DataFrame(data=m)   
    # mod = mod[mod['spect']>0]
    
    
    # from scipy import stats
    # lingress_result = stats.linregress(x=mod['spect'], y=mod['sapm'], alternative='two-sided')
    # R_2 = lingress_result.rvalue
    
    # rel_error = (100*(mod['spect']-mod['sapm'])/mod['spect'])
    # rel_error_monthly = rel_error.resample('M').mean()
    
    # plt.figure()
    # plt.plot(rel_error_monthly)
    
   
    # plt.figure()
    # plt.plot(spect_power_monthly)
    # plt.plot(sapm_power_monthly)
   
    # plt.figure()
    # plt.plot(spect_run.results.dc.p_mp.loc['2021-05-31':'2021-06-03'])
    # plt.plot(sapm_run.results.dc.p_mp.loc['2021-05-31':'2021-06-03'])
    
    # plt.figure()
    # plt.plot(spec_mismatch.loc['2021-05-31':'2021-06-03'])
    # plt.plot(sapm_run.results.spectral_modifier.loc['2021-05-31':'2021-06-03'])
    # plt.ylim((0.95,1.05))
    