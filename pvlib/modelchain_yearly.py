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
import matplotlib.dates as mdates

# Import data
from Data.open_data import NRELdata , format_looped_data

# Import coordinates/plotting stuff
from US_map import county_map
coords = county_map()


j=0
# Choose years:
years = range(1998,2023)
perc_loss_yearly = av_loss = np.zeros(len(years))
for year in years:

    # Initialise looped variables
    clear_copy = []
    cloud_copy = []
    type_copy = []
    
    # Get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    tilt = 30
    system = PVSystem(surface_azimuth=180, surface_tilt=tilt, module_parameters=module, inverter_parameters=inverter,
                      temperature_model_parameters=temperature)#, albedo=data['Surface Albedo'])
    
    coords = coords[coords['NAME']=='Pinal']
    # Loop time!
    for index, row in coords.iterrows():
        
        # Get NSRDB data
        latitude = round(float(row['LAT']),3)
        longitude = round(float(row['LONG']),3)
        name = row['NAME']
        interval = 60 # in minutes
        
        while True:
            try:
                data = NRELdata(year, latitude, longitude, name, interval)
            except:
                continue
            break
    
        # Define location
        location = Location(latitude=latitude,longitude=0,#set to 0 as using UTC. Set to longitude if using local time.
                            altitude=pvlib.location.lookup_altitude(latitude=latitude, longitude=longitude)
                            #,tz='US/Pacific'  #default is UTC
                            )
        
        str_time = pd.to_datetime(data.index)
        time = str_time.tz_localize(None)
      
        #losses = pvlib.pvsystem.pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2, connections=0.5, lid=1.5, nameplate_rating=1, age=0, availability=3)
        
        # Define and run model
        modelchain = ModelChain(system, location, spectral_model='sapm') # location and system have no default so must be specified
         
        clear_data = pd.DataFrame({'ghi':data['Clearsky GHI'], 'dni':data['Clearsky DNI'], 'dhi':data['Clearsky DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
        clear_copy = format_looped_data(clear_copy, clear_data, time, modelchain)
        
        cloud_data = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'], 'dhi':data['DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
        cloud_copy = format_looped_data(cloud_copy, cloud_data, time, modelchain)
        
        type_data = pd.Series(data=data['Cloud Type'])
        type_data.index = time
        type_copy.append(type_data)
        
        
    # Reformat saved list values
    def reformat_list(copy, coords):
        power = pd.DataFrame(copy, index=coords['NAME'] + coords['STATEFP'] + coords['COUNTYFP'])
        power = power.T   # transpose
        return power
    clear_power = reformat_list(clear_copy, coords)
    cloud_power = reformat_list(cloud_copy, coords)
    cloud_type = reformat_list(type_copy, coords)
    
    
    # Total errors
    diff = cloud_power - clear_power
    loss = np.trapz(-diff, dx=interval*60, axis=0) # dx is time in seconds
    perc_loss = loss/np.trapz(clear_power, dx=interval*60, axis=0)*100
    perc_loss = perc_loss.round(1)
    
    perc_loss_yearly[j] = perc_loss
    #av_loss[j] = perc_loss_yearly[perc_loss_yearly!=0].mean()
    j = j+1
    
plt.plot(years, perc_loss_yearly)
#plt.plot(years, av_loss)


