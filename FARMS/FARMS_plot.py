import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from SMARTS.open_copernicus import BSRNdata
from SMARTS.open_copernicus import NRELdata
# %edit C:\Users\mark\anaconda3\Lib\site-packages\pySMARTS\main.py

# Choose time range:
start = np.datetime64('2022-05-27T12:00')
end = np.datetime64('2022-05-29T12:00')

# Get NREL (FARMS) stuff
year, FARMS_ghi, FARMS_dhi, FARMS_dni, cloud_type, wind, temperature = NRELdata(start,end)
FARMS_time = np.arange(start, end+5, dtype='datetime64[5m]')

# Get BSRN stuff
BSRN_time, BSRN_dir, BSRN_dif, BSRN_ghi = BSRNdata(start,end)

# Find error
temp_farms = FARMS_ghi.to_numpy()
perc_error = np.zeros(len(temp_farms))
for i in range(len(temp_farms)):
    if BSRN_ghi[i] > 10 and temp_farms[i] > 10:
        perc_error[i] = (BSRN_ghi[i] - temp_farms[i])*100/BSRN_ghi[i]



# PLOTTING TIME!
fig, ax1 = plt.subplots(figsize=(20,10))
plt.ylim(0,1200)
plt.xlabel("Day")
plt.ylabel("Global Horizontal Irradiation [W/m^2]")
plt.title("Solar Radiation in Desert Rock, Nevada 2022-05-05 to 08")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.DayLocator()) # Make ticks one per day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

# BSRN data
ax1.plot(BSRN_time-7*60, BSRN_ghi, label = 'BSRN')
# plt.plot(BSRN_time-7*60, BSRN_dir, label = 'DIR')
# plt.plot(BSRN_time-7*60, BSRN_dif, label = 'DIF')

# FARMS data
ax1.plot(FARMS_time-7*12, FARMS_ghi, label = 'FARMS All Sky')

## Copernicus data
#ax1.plot(rad_time, GHI_all, label = 'COP All Sky')
#ax1.plot(rad_time, GHI_clear, label = 'COP Clear Sky')

# # SMARTS data
# ax1.plot(time, area, label = 'SMARTS')

# # Percentage Error
# ax2 = ax1.twinx()
# ax2.set_ylabel('Percentage Error')
# ax2.plot(BSRN_time-7*60,perc_error, color='red')

# # Total Cloud Cover (Copernicus)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Total Cloud Cover')
# ax2.plot(time,total_cloud, color='red')

# # Cloud Type (FARMS)
# reee = farms_time
# from scipy.interpolate import make_interp_spline
# ax2 = ax1.twinx()
# ax2.set_ylabel('Cloud Type')
# Spline = make_interp_spline(np.arange(2017), cloud_type)
# X_ = np.linspace(np.arange(2017).min(), np.arange(2017).max(), 202)
# Y_ = Spline(X_)
# ax2.plot(reee[0::10], Y_, color='red')

ax1.legend(loc='upper right')
plt.show()