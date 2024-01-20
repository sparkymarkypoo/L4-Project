import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Data.open_data import NRELdata

# Choose time range:
start = np.datetime64('2021-01-01T00:00')
end = np.datetime64('2021-01-07T00:00') #bad data on 18th pm to 19th 00

# Get NREL (FARMS) stuff
file = r'C:\Users\mark\Documents\L4-Project-Code\Data\Desert Rock\NREL_2021_DRA_5min.csv'
f_year, f_FARMS_ghi, f_FARMS_dhi, f_FARMS_dni, f_cloud_type, f_wind, f_temperature = NRELdata(start,end,file,interval=5)
f_FARMS_time = np.arange(start, end+5, dtype='datetime64[5m]')

file = r'C:\Users\mark\Documents\L4-Project-Code\Data\Desert Rock\NREL_2021_DRA_60min.csv'
s_year, s_FARMS_ghi, s_FARMS_dhi, s_FARMS_dni, s_cloud_type, s_wind, s_temperature = NRELdata(start,end,file,interval=60)
s_FARMS_time = np.arange(start, end+60, dtype='datetime64[h]')

# Get NREL (FARMS) stuff
specific_rows = np.arange(2)  # cut out crap at the top 
df = pd.read_csv(r'C:\Users\mark\Documents\L4-Project-Code\Data\Desert Rock\Spectrum_2021_DRA_0deg.csv',skiprows = specific_rows)
GHI = df['GHI']


# # Find error
# temp_farms = FARMS_ghi.to_numpy()
# perc_error = np.zeros(len(temp_farms))
# for i in range(len(temp_farms)):
#     if BSRN_ghi[i] > 10 and temp_farms[i] > 10:
#         perc_error[i] = (BSRN_ghi[i] - temp_farms[i])*100/BSRN_ghi[i]



# PLOTTING TIME!
fig, ax1 = plt.subplots(figsize=(16,9))
plt.ylim(0,600)
plt.xlabel("Day")
plt.ylabel("Global Horizontal Irradiation [W/m^2]")
plt.title(f"Solar Radiation in Desert Rock, Nevada {start} to {end}")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.DayLocator()) # Make ticks one per day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

# FARMS data
#ax1.plot(f_FARMS_time-7*13, f_FARMS_ghi, label = '5min')
ax1.plot(s_FARMS_time-7, s_FARMS_ghi, label = '60min')
adjust = 9
ax1.plot(s_FARMS_time-7, GHI.iloc[adjust:145+adjust], label = 'Spectral 60min')


# # Percentage Error
# ax2 = ax1.twinx()
# ax2.set_ylabel('Percentage Error')
# ax2.plot(BSRN_time-7*60,perc_error, color='red')

ax1.legend(loc='upper right')
plt.show()