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

# Import coordinates/plotting stuff
from US_map import get_capitals
from US_map import draw_map

# Choose time range:
year = 2022
start = np.datetime64(f'{year}-01-01T00:00') # 01-01T00:00 for whole year
end = np.datetime64(f'{year}-12-30T23:55')   # 12-30T23:55 for whole year

# # BSRN optional extra for 2022-05
# from Data.open_data import BSRNdata
# BSRN_time, BSRN_dir, BSRN_dif, BSRN_ghi = BSRNdata(start,end)

# Coordinates for simulation (latitude, longitude, description)
# coords = np.array([[36.626, -116.018, 'Desert_Rock']])
coords, capitals = get_capitals()

# Initialise looped variables
clear_copy = []
cloud_copy = []
type_copy = []

# Loop time!
for i in coords:
    
    # Get NSRDB data
    latitude = round(float(i[0]),3)
    longitude = round(float(i[1]),3)
    name = i[2]
    interval = 60 # in minutes
    tilt = 30
    data = NRELdata(start, end, year, latitude, longitude, name, interval)
    # TODO return actual interval and use that - maybe not needed as I can't get API working outside of USA
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
    
    type_copy.append(data['Cloud Type'])
   
    
# Reformat saved list values
clear_power = pd.DataFrame(clear_copy, index=capitals['CAPITAL'])
clear_power = clear_power.T   # transpose
cloud_power = pd.DataFrame(cloud_copy, index=capitals['CAPITAL'])
cloud_power = cloud_power.T   # transpose
cloud_type = pd.DataFrame(type_copy, index=capitals['CAPITAL'])
cloud_type = cloud_type.T   # transpose

# Errors
diff = cloud_power - clear_power
loss = np.trapz(-diff, dx=interval*60, axis=0) # dx is time in seconds
perc_loss = loss/np.trapz(clear_power, dx=interval*60, axis=0)*100
perc_loss = perc_loss.round(1)

# Clouds
cloud_average = cloud_type.mean()
cloud_average = cloud_average.round(1)
cloud_average = cloud_average.to_numpy()
cloud_std = cloud_type.std()
cloud_std = cloud_std.round(1)
cloud_std = cloud_std.to_numpy()



# Plot US map
geo_merge = draw_map(perc_loss) 
geo_merge.plot(column='PERC_LOSS',figsize=(25, 13),legend=True, cmap='Blues')
plt.xlim(-127,-66)
plt.ylim(24,50)
for i in range(len(geo_merge)):
    plt.text(geo_merge.LONG[i],geo_merge.LAT[i],"{}\n{}".format(geo_merge.CAPITAL[i],geo_merge.PERC_LOSS[i]),size=11)
plt.title('% Power loss due to cloud cover',fontsize=25)
plt.show()

# geo_merge = draw_map(cloud_average) 
# geo_merge.plot(column='PERC_LOSS',figsize=(25, 13),legend=True, cmap='Blues')
# plt.xlim(-127,-66)
# plt.ylim(24,50)
# for i in range(len(geo_merge)):
#     plt.text(geo_merge.LONG[i],geo_merge.LAT[i],"{}\n{}".format(geo_merge.CAPITAL[i],geo_merge.PERC_LOSS[i]),size=11)
# plt.title('Cloud type average',fontsize=25)
# plt.show()



# # Plot box plots
# location = 'Olympia'
# box = pd.DataFrame(data=cloud_type[location])
# box = box.rename(columns={location: "Type"})
# box['Diff'] = diff[location].to_numpy()
# box['Clear'] = clear_power[location].to_numpy()
# box['Cloud'] = cloud_power[location].to_numpy()
# box['Perc_Diff'] = 100* box['Diff']/box['Clear']

# type_label = ['Water','Super-Cooled Water','Mixed','Opaque Ice','Cirrus','Overlapping','Overshooting']
# plt.figure(figsize=(10,10)) 
# for i in range(3,10):
#     cloud = box['Diff'][(box['Type'] == i) & (box['Diff'] != 0)]
#     longth = len(cloud)
#     plt.boxplot(cloud, positions = [i])
# plt.ylabel('Difference in Power [W]')
# plt.xlabel('Cloud Type')
# plt.title('Power Loss by Cloud Type')

# plt.figure(figsize=(10,10))   
# for i in range(3,10):
#     cloud = box['Perc_Diff'][(box['Type'] == i) & (box['Diff'] < 0) & (box['Cloud'] > 0) & (box['Clear'] > 0)]
#     longth = len(cloud)
#     plt.boxplot(cloud, positions = [i])
# plt.ylabel('Difference in Power %')
# plt.xlabel('Cloud Type')
# plt.title('% Power Loss by Cloud Type')



# # Time Series Plot
# fig, ax1 = plt.subplots(figsize=(20,10))
# plt.ylim(0,300)
# plt.xlabel("Day")
# plt.ylabel("AC Power W")
# plt.title(f"Lat({latitude})-Long({longitude}) {start} to {end}")
# ax1.grid()
# ax1.xaxis.set_major_locator(mdates.DayLocator()) # Make ticks one per day
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
# ax1.plot(time, cloud_power['Desert Rock'], label = 'Power (Cloud)')
# ax1.plot(time, clear_power['Desert Rock'], label = 'Power (Clear)')
# ax2 = ax1.twinx()
# # Error
# ax2.plot(time, diff['Desert Rock'], label='Difference',color='red')
# # Legends
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.show()


# # Measured vs Simulated Irradiance
# plt.scatter(data['GHI'],BSRN_ghi, c=data['Cloud Type'])
# plt.colorbar()
# plt.ylabel('Measured Irradiance')
# plt.xlabel('NSRDB Irradiance')
# plt.title('Simulated vs Actual Irradiance, colored by Cloud Type')

# # Power loss by cloud type
# plt.scatter(cloud_power.iloc[:,-1], diff.iloc[:,-1], c=data['Cloud Type'])
# plt.colorbar()
# plt.ylabel('Difference in Power [W]')
# plt.xlabel('Cloud Type')
