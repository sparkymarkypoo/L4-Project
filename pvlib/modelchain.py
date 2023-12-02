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
from Data.open_data import NRELdata

# Choose time range:
year = 2022
start = np.datetime64(f'{year}-01-01T00:00') # 01-01T00:00 for whole year
end = np.datetime64(f'{year}-12-30T23:55')   # 12-30T23:55 for whole year

# # BSRN optional extra for 2022-05
# from Data.open_data import BSRNdata
# BSRN_time, BSRN_dir, BSRN_dif, BSRN_ghi = BSRNdata(start,end)

# Coordinates for simulation (latitude, longitude)
coords = np.array([[36.626, -116.018], [33, -120]])

# Initialise looped variables
clear_copy = []
cloud_copy = []

# Loop time!
for i in coords:
    
    # Get NSRDB data
    latitude = i[0]
    longitude = i[1]
    interval = 5 # in minutes
    tilt = 30
    data = NRELdata(start, end, year, latitude, longitude, interval)
    # TODO return actual interval and use that
    time = np.arange(start, end+5, dtype=f'datetime64[{interval}m]')

    # Define location
    location = Location(latitude=latitude,longitude=0,#set to 0 as using UTC. Set to longitude if using local time.
                        altitude=900
                        #,tz='US/Pacific'
                        )

    # Get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = PVSystem(surface_azimuth=180, surface_tilt=tilt, module_parameters=module, inverter_parameters=inverter,
                      temperature_model_parameters=temperature)#, albedo=data['Surface Albedo'])
    
    #losses = pvlib.pvsystem.pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2, connections=0.5, lid=1.5, nameplate_rating=1, age=0, availability=3)
    
    # Define and run model
    modelchain = ModelChain(system, location, spectral_model='sapm') # location and system have no default so must be specified
    
    clear_data = pd.DataFrame({'ghi':data['Clearsky GHI'], 'dni':data['Clearsky DNI'], 'dhi':data['Clearsky DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
    clear_data.index = pd.date_range(start,end,freq=f'{interval}min')
    clear_run = modelchain.run_model(clear_data)
    clear_copy.append(clear_run.results.ac) # Model chain gets overwritten so save results
    
    cloud_data = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'], 'dhi':data['DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
    cloud_data.index = pd.date_range(start,end,freq=f'{interval}min')
    cloud_run = modelchain.run_model(cloud_data)
    cloud_copy.append(cloud_run.results.ac)
  
# Reformat saved list values
clear_power = pd.DataFrame(clear_copy)
clear_power = clear_power.T   # transpose
cloud_power = pd.DataFrame(cloud_copy)
cloud_power = cloud_power.T   # transpose

# Errors
diff = cloud_power - clear_power
loss = np.trapz(-diff, dx=interval*60, axis=0) # dx is time in seconds
perc_loss = loss/np.trapz(clear_power, dx=interval*60, axis=0)*100
print(perc_loss)




# PLOT
fig, ax1 = plt.subplots(figsize=(20,10))
plt.ylim(0,300)
plt.xlabel("Day")
plt.ylabel("AC Power W")
plt.title(f"Lat({latitude})-Long({longitude}) {start} to {end}")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.DayLocator()) # Make ticks one per day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax1.plot(time, cloud_power[0], label = 'Power (Cloud)')
ax1.plot(time, clear_power[0], label = 'Power (Clear)')
ax2 = ax1.twinx()

# Error
# ax2.plot(time, perc, label='% Difference',color='red')
ax2.plot(time, diff, label='Difference',color='red')

# # Irradiance
# ax2.set_ylim(0, 1200)
# ax2.set_ylabel('Global Horizontal Irradiation [W/m^2]')
# ax2.plot(time,data['GHI'], color='green', label='Irradiance')

# # Cloud Type
# ax2.set_ylim(0, 8)
# ax2.set_ylabel('Cloud Type')
# # longth = len(data)
# # from scipy.interpolate import make_interp_spline
# # Spline = make_interp_spline(np.arange(longth), cloud_type)
# # X_ = np.linspace(np.arange(longth).min(), np.arange(longth).max(), 173)
# # Y_ = Spline(X_)
# # ax2.plot(time[0::5], Y_, color='red')
# ax2.plot(time,data['Cloud Type'], color='red', label='Cloud Type')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

# plt.scatter(data['GHI'],BSRN_ghi, c=data['Cloud Type'])
# plt.colorbar()
# plt.ylabel('Measured Irradiance')
# plt.xlabel('NSRDB Irradiance')
# plt.title('Simulated vs Actual Irradiance, colored by Cloud Type')

# plt.scatter(FARMS_ghi, run1.results.ac, c=air_temp)
# plt.colorbar()
# plt.ylabel('Array Power [W]')
# plt.xlabel('POA Irradiance [W/m^2]')
# plt.title('Power vs POA, colored by amb. temp.')

# plt.scatter(data['Cloud Type'], diff)#, c=air_temp)
# plt.colorbar()
# plt.ylabel('Difference in Power [W]')
# plt.xlabel('Cloud Type')
# plt.title('Power vs POA, colored by amb. temp.')