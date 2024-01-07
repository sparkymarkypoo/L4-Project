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

# Choose year:
year = 2021

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
    clear_data.index = time
    clear_run = modelchain.run_model(clear_data)
    clear_results = clear_run.results.total_irrad
    clear_copy.append(clear_run.results.ac)
    
    cloud_data = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'], 'dhi':data['DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
    cloud_data.index = time
    cloud_run = modelchain.run_model(cloud_data)
    cloud_results = cloud_run.results.total_irrad
    cloud_copy.append(cloud_run.results.ac)
    
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


# Errors by month
perc_loss_monthly_mean =  np.zeros(12)
cloud_monthly_mean =  np.zeros(12)
cloud_monthly_no = np.zeros(12)
ghi_monthly_mean =  np.zeros(12)
energy_monthly_mean = np.zeros(12)
clearghi_monthly_mean =  np.zeros(12)
clear_energy_monthly_mean = np.zeros(12)

wind_monthly_mean = np.zeros(12)
temp_monthly_mean = np.zeros(12)
for i in range(1,13):
    
    power_monthly = cloud_power[(time.month == i)]
    energy_monthly = np.trapz(power_monthly, dx=interval*60, axis=0)
    energy_monthly_mean[i-1] = np.mean(energy_monthly)
       
    clear_power_monthly = clear_power[(time.month == i)]
    clear_energy_monthly = np.trapz(clear_power_monthly, dx=interval*60, axis=0) 
    clear_energy_monthly_mean[i-1] = np.mean(clear_energy_monthly)
    
    diff_monthly = diff[(time.month == i)]
    loss_monthly = np.trapz(-diff_monthly, dx=interval*60, axis=0) # dx is time in seconds 
    perc_loss_monthly = (loss_monthly/clear_energy_monthly)*100
    perc_loss_monthly = perc_loss_monthly.round(1)
    perc_loss_monthly_mean[i-1] = np.mean(perc_loss_monthly)
    
    cloud_monthly = cloud_type[(time.month == i)]
    cloud_monthly_mean[i-1] = np.nanmean(cloud_monthly)
    day_total = cloud_monthly.count().sum()
    cloud_monthly_no[i-1] = 100*(cloud_monthly[(cloud_monthly > 2)].count().sum())/day_total
    
    ghi_monthly = data['GHI'][(time.month == i)]
    clearghi_monthly = data['Clearsky GHI'][(time.month == i)]
    clearghi_monthly_mean[i-1] = np.mean(clearghi_monthly)
    ghi_monthly_mean[i-1] = np.mean(ghi_monthly)
    
    wind_monthly = data['Wind Speed'][(time.month == i)]
    temp_monthly = data['Temperature'][(time.month == i)]
    wind_monthly_mean[i-1] = np.mean(wind_monthly)
    temp_monthly_mean[i-1] = np.mean(temp_monthly)
    
 
# # Errors by week
# perc_loss_weekly_mean = np.zeros(52)
# for i in range(1,53):
    
#     if i < 52:
#         diff_weekly = diff[(time.isocalendar().week == i)]
#         clear_power_weekly = clear_power[(time.isocalendar().week == i)]
    
#     else:
#         diff_weekly = diff[(time > f'{year}-12-26')]
#         clear_power_weekly = clear_power[(time > f'{year}-12-26')]
        
#     loss_weekly = np.trapz(-diff_weekly, dx=interval*60, axis=0) # dx is time in seconds
#     perc_loss_weekly = loss_weekly/np.trapz(clear_power_weekly, dx=interval*60, axis=0)*100
#     perc_loss_weekly = perc_loss_weekly.round(1)
#     perc_loss_weekly_mean[i-1] = np.mean(perc_loss_weekly)    




# Plot US map
coords['PERC_LOSS'] = perc_loss
figmap = coords.plot(column='PERC_LOSS',figsize=(25, 13),legend=True, cmap='Blues')
# plt.xlim(-127,-66)
# plt.ylim(24,50)
# for index, row in coords.iterrows():
#     plt.text(row.LONG,row.LAT,row.PERC_LOSS,size=11)
plt.xlabel('Longitude',fontsize=20)
plt.ylabel('Latitude',fontsize=20)
plt.title('% Energy loss due to cloud cover',fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb_ax = figmap.figure.axes[1]
cb_ax.tick_params(labelsize=18)
plt.show()


# Plot errors by time
from scipy.interpolate import make_interp_spline
month_smooth = np.linspace(1, 12, 300) 
spl1 = make_interp_spline(np.arange(1,13), perc_loss_monthly_mean, k=3)  # type: BSpline
loss_smooth = spl1(month_smooth)
spl2 = make_interp_spline(np.arange(1,13), cloud_monthly_no, k=3)  # type: BSpline
cloud_smooth = spl2(month_smooth)

fig, ax1 = plt.subplots(figsize=(10,10))
plt.xticks(ticks=np.arange(1,13), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Month', fontsize=18)
plt.ylabel('% Loss', fontsize=18)
plt.title('% Average energy loss by month', fontsize=20)
ax1.plot(month_smooth, loss_smooth, label='% Average energy loss')
ax1.legend(loc='upper left', fontsize=14)
#ax1.grid()
ax2 = ax1.twinx()
ax2.plot(month_smooth, cloud_smooth, label='% Of time with cloud cover', color='red')
ax2.set_ylabel('% Of time with cloud cover', fontsize=18) 
ax2.legend(loc='upper right', fontsize=14)
ax2.tick_params(labelsize=14)
plt.show()


# fig, ax1 = plt.subplots(figsize=(10,10))
# plt.xticks(ticks=np.arange(1,13))
# plt.xlabel('Month')
# plt.ylabel('% Loss')
# plt.title('% Average energy loss')
# ax1.plot(np.arange(1,13), perc_loss_monthly_mean, label='% Average energy loss')
# ax1.legend(loc='upper left')
# #ax1.grid()
# ax2 = ax1.twinx()
# ax2.plot(np.arange(1,13), cloud_monthly_no, label='% Of time with cloud cover', color='red')
# ax2.set_ylabel('% Of time with cloud cover') 
# ax2.legend(loc='upper right')
# plt.show()


# Plot box plots
cloud = []
for location in diff:
    box = pd.DataFrame(data=cloud_type[location])
    box = box.rename(columns={location: "Type"})
    box['Diff'] = diff[location].to_numpy()
    box['Clear'] = clear_power[location].to_numpy()
    box['Cloud'] = cloud_power[location].to_numpy()
    box['Perc_Diff'] = 100* box['Diff']/box['Clear']
    for i in range(3,10):
        pd_cloud = box['Perc_Diff'][(box['Type'] == i) & (box['Diff'] < 0) & (box['Cloud'] > 0) & (box['Clear'] > 30)]
        cloud.append(-pd_cloud.values)
plt.figure(figsize=(10,10))   
for i in range(3,10):
    flat_list = []
    for row in cloud[i - 3::7]:
        flat_list.extend(row)
    plt.boxplot(flat_list, positions = [i])
plt.ylabel('% Power loss', fontsize=18)
plt.xlabel('Cloud Type', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('% Power Loss by Cloud Type', fontsize=20)



# plt.figure(figsize=(10,10))   
# for i in range(3,10):
#     cloud = box['Perc_Diff'][(box['Type'] == i) & (box['Diff'] < 0) & (box['Cloud'] > 0) & (box['Clear'] > 30)]
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
# plt.scatter(clear_power['Alameda'], cloud_power['Alameda'], c=cloud_type['Alameda'])
# plt.colorbar()
# plt.ylabel('All sky')
# plt.xlabel('Clearsky')
# plt.title('Simulated vs Actual Irradiance, colored by Cloud Type')

# # Power loss by cloud type
# plt.scatter(cloud_power.iloc[:,-1], diff.iloc[:,-1], c=data['Cloud Type'])
# plt.colorbar()
# plt.ylabel('Difference in Power [W]')
# plt.xlabel('Cloud Type')


