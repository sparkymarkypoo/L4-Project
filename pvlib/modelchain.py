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
start = np.datetime64(f'{year}-05-26T00:00')
end = np.datetime64(f'{year}-05-30T00:00')

# Get NREL (FARMS) data
latitude = 36.626
longitude = -116.018 
interval = 5 # in minutes
data = NRELdata(start, end, year, latitude, longitude, interval)
FARMS_time = np.arange(start, end+5, dtype=f'datetime64[{interval}m]')

# Define location
location = Location(latitude=latitude,longitude=0,#longitude
                    altitude=900
                    #,tz='US/Pacific'
                    )

# Get the module and inverter specifications from SAM
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = PVSystem(surface_azimuth=180, surface_tilt=37, module_parameters=module, inverter_parameters=inverter,
                  temperature_model_parameters=temperature)

#losses = pvlib.pvsystem.pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2, connections=0.5, lid=1.5, nameplate_rating=1, age=0, availability=3)

# Define and run model
modelchain = ModelChain(system, location, spectral_model='sapm') # location and system have no default so must be specified

clear_data = pd.DataFrame({'ghi':data['Clearsky GHI'], 'dni':data['Clearsky DNI'], 'dhi':data['Clearsky DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
clear_data.index = pd.date_range(start,end,freq=f'{interval}min')
clear_run = modelchain.run_model(clear_data)
clear_copy = clear_run.results.ac # Model chain gets overwritten so save results

cloud_data = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'], 'dhi':data['DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
cloud_data.index = pd.date_range(start,end,freq=f'{interval}min')
cloud_run = modelchain.run_model(cloud_data)

# Errors
diff = cloud_run.results.ac - clear_copy
perc = (diff/clear_copy) *100
perc.loc[perc > 100] = 1




# PLOT
fig, ax1 = plt.subplots(figsize=(20,10))
plt.ylim(0,300)
plt.xlabel("Day")
plt.ylabel("AC Power MW")
plt.title("Solar Radiation in Desert Rock, Nevada 2022-05-26 to 30")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.DayLocator()) # Make ticks one per day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax1.plot(FARMS_time, cloud_run.results.ac, label = 'Power')
ax1.plot(FARMS_time, clear_copy, label = 'Power')
ax2 = ax1.twinx()

# Error
# ax2.plot(FARMS_time, perc, label='% Difference',color='red')
ax2.plot(FARMS_time, diff, label='Difference',color='red')

# # Irradiance
# ax2.set_ylim(0, 1200)
# ax2.set_ylabel('Global Horizontal Irradiation [W/m^2]')
# ax2.plot(FARMS_time,data['GHI'], color='red', label='Irradiance')

# # Cloud Type
# ax2.set_ylim(0, 8)
# ax2.set_ylabel('Cloud Type')
# # longth = len(data)
# # from scipy.interpolate import make_interp_spline
# # Spline = make_interp_spline(np.arange(longth), cloud_type)
# # X_ = np.linspace(np.arange(longth).min(), np.arange(longth).max(), 173)
# # Y_ = Spline(X_)
# # ax2.plot(FARMS_time[0::5], Y_, color='red')
# ax2.plot(FARMS_time,data['Cloud Type'], color='red', label='Cloud Type')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

# plt.scatter(FARMS_ghi, run1.results.ac, c=air_temp)
# plt.colorbar()
# plt.ylabel('Array Power [W]')
# plt.xlabel('POA Irradiance [W/m^2]')
# plt.title('Power vs POA, colored by amb. temp.')

