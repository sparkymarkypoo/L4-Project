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
irrad_copy = []
type_copy = []

# Get the module and inverter specifications from SAM
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# TEMPORARY ADJUSTMENT TO SAVE TIME
coords = coords[coords['STATEFP']=='06']
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

    # Define location and system
    location = Location(latitude=latitude,longitude=longitude,#set to 0 as using UTC. Set to longitude if using local time.
                        altitude=pvlib.location.lookup_altitude(latitude=latitude, longitude=longitude)
                        #,tz='US/Pacific'  #default is UTC
                        )
    tilt = round(latitude,1)
    system = PVSystem(surface_azimuth=180, surface_tilt=tilt, module_parameters=module, inverter_parameters=inverter,
                      temperature_model_parameters=temperature)#, albedo=data['Surface Albedo'])
    
  
    # Define and run model
    time = pd.to_datetime(data.index)
    modelchain = ModelChain(system, location, dc_model='sapm', spectral_model='sapm', aoi_model='no_loss')
     
    clear_data = pd.DataFrame({'ghi':data['Clearsky GHI'], 'dni':data['Clearsky DNI'], 'dhi':data['Clearsky DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
    clear_data.index = time
    clear_run = modelchain.run_model(clear_data) # Run the ModelChain
    clear_copy.append(clear_run.results.dc.p_mp)
    
    cloud_data = pd.DataFrame({'ghi':data['GHI'], 'dni':data['DNI'], 'dhi':data['DHI'], 'temp_air':data['Temperature'], 'wind_speed':data['Wind Speed']})
    cloud_data.index = time
    cloud_run = modelchain.run_model(cloud_data) # Run the ModelChain
    cloud_copy.append(cloud_run.results.dc.p_mp)
    
    irrad_copy.append(cloud_run.results.total_irrad['poa_global']) 
    
    type_data = pd.Series(data=data['Cloud Type'])
    type_data.index = time
    type_copy.append(type_data)
    
    
# Reformat saved list values
def reformat_list(copy, coords, time):
    tz_power = pd.DataFrame(copy, index=coords['NAME'] + coords['STATEFP'] + coords['COUNTYFP'])
    tz_power = tz_power.T   # transpose
    notz_power = tz_power.apply(lambda x: pd.Series(x.dropna().values))
    notz_power.index = time.tz_localize(None)
    notz_power = notz_power.replace(-0.075,0)
    return notz_power
clear_power = reformat_list(clear_copy, coords, time)
cloud_power = reformat_list(cloud_copy, coords, time)
total_irrad = reformat_list(irrad_copy, coords, time)
cloud_type = reformat_list(type_copy, coords, time)


# Total errors
diff = cloud_power - clear_power
loss = np.trapz(-diff, dx=interval*60, axis=0) # dx is time in seconds
perc_loss = loss/np.trapz(clear_power, dx=interval*60, axis=0)*100
perc_loss = perc_loss.round(1)

efficiency = 100*cloud_power/(total_irrad * 1.7)
#efficiency = efficiency.fillna(0)
eff_mean = efficiency.mean(axis=0)
energy_tot = np.trapz(cloud_power, dx=interval*60, axis=0)


# Errors by month
cloud_power_monthly = cloud_power.resample('M').apply(np.trapz) # Monthly means for each county
cloud_power_monthly_mean = cloud_power_monthly.mean(axis=1) # Monthly means for all counties averaged

clear_power_monthly = clear_power.resample('M').apply(np.trapz)
clear_power_monthly_mean = clear_power_monthly.mean(axis=1)

perc_loss_monthly = 100*(clear_power_monthly - cloud_power_monthly)/clear_power_monthly
perc_loss_monthly_mean = perc_loss_monthly.mean(axis=1)


# Cloud by month
time = cloud_power.index
perc_times_cloudy_monthly = []
#cloud_monthly_no = pd.DataFrame()#(columns = cloud_power.columns)
for i in range(1,13):    
    cloud_monthly = cloud_type[(time.month == i)]
    
    pow_over_0 = cloud_monthly[(clear_power > 0)] # Contains the cloud type for when the clear sky power > 0
    cloud_over_2 = pow_over_0[(pow_over_0 > 2)] # The above with only cloud types > 2
     
    p_t = 100*cloud_over_2.count()/pow_over_0.count() # Gives the % of times in daylight hours with cloud
    perc_times_cloudy_monthly.append(p_t)

cloud_monthly_no = pd.DataFrame(perc_times_cloudy_monthly) # for each county
cloud_monthly_no_mean = cloud_monthly_no.mean(axis=1) # averaged over all counties
cloud_county_no = cloud_monthly_no.mean(axis=0) # months averaged for each county


# Find Rsquared for cloud/ploss for each county
from scipy import stats
Rsquared = np.zeros(cloud_monthly_no.shape[1])
i=0
for (columnName, columnData) in cloud_monthly_no.iteritems(): 
    lingress_result = stats.linregress(x=cloud_monthly_no[columnName], y=perc_loss_monthly[columnName], alternative='two-sided')
    Rsquared[i] = lingress_result.rvalue
    i=i+1




# Plot US map
#coords['CUSTOM'] = cloud_county_no.to_numpy()
#coords['CUSTOM'] = eff_mean.to_numpy()
#coords['CUSTOM'] = energy_tot
#coords['CUSTOM'] = Rsquared
coords['CUSTOM'] = perc_loss
figmap = coords.plot(column='CUSTOM',figsize=(25, 13),legend=True, cmap='coolwarm')
plt.xlabel('Longitude',fontsize=20)
plt.ylabel('Latitude',fontsize=20)
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
spl2 = make_interp_spline(np.arange(1,13), cloud_monthly_no_mean, k=3)  # type: BSpline
cloud_smooth = spl2(month_smooth)

fig, ax1 = plt.subplots(figsize=(10,10))
plt.xticks(ticks=np.arange(1,13), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Month', fontsize=18)
plt.ylabel('% Average energy loss due to cloud', fontsize=18)
ax1.plot(month_smooth, loss_smooth, label='% Loss')
ax1.legend(loc='upper left', fontsize=14)
#ax1.grid()
ax2 = ax1.twinx()
ax2.plot(month_smooth, cloud_smooth, label='% Time', color='red')
ax2.set_ylabel('% Of time with cloud cover', fontsize=18) 
ax2.legend(loc='upper right', fontsize=14)
ax2.tick_params(labelsize=14)
plt.show()


# # Plot box plots
# cloud = []
# for location in diff:
#     box = pd.DataFrame(data=cloud_type[location])
#     box = box.rename(columns={location: "Type"})
#     box['Diff'] = diff[location].to_numpy()
#     box['Clear'] = clear_power[location].to_numpy()
#     box['Cloud'] = cloud_power[location].to_numpy()
#     box['Perc_Diff'] = 100* box['Diff']/box['Clear']
#     for i in range(3,10):
#         pd_cloud = box['Perc_Diff'][(box['Type'] == i) & (box['Diff'] < 0) & (box['Cloud'] > 0) & (box['Clear'] > 30)]
#         cloud.append(-pd_cloud.values)
# plt.figure(figsize=(10,10))   
# for i in range(3,10):
#     flat_list = []
#     for row in cloud[i - 3::7]:
#         flat_list.extend(row)
#     plt.boxplot(flat_list, positions = [i])
# plt.ylabel('% Power loss due to cloud', fontsize=18)
# plt.xlabel('Cloud Type', fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

