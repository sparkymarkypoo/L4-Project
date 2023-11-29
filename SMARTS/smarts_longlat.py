import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pySMARTS
from scipy.integrate import simpson
from numpy import trapz

# Specify Desired Output
IOUT = '1, 4, 8' # Diffuse, Global, Direct horizontal irradiance W m-2

# Define Date, Time, and Location (ex: Worcs, UK)
YEAR = '2023'
MONTH = '10'
DAY = '09'
HOUR = '13'
LATIT = '52.2'
LONGIT = '-2.2'
ALTIT = '0.03' # km above sea level
ZONE = '+1' # Timezone
MAT = 'Seawater'

# Albedo at 8am
demo = pySMARTS.SMARTSTimeLocation(IOUT=IOUT, YEAR=YEAR, MONTH=MONTH, DAY=DAY, HOUR=HOUR,
                                    LATIT=LATIT, LONGIT=LONGIT, ALTIT=ALTIT, ZONE=ZONE, material=MAT)

# Loop for months of the year
lat_start = 80 #first latitude
lat_end = -100 #last latitude
lat_gap = -20 #latitude interval
lat = np.arange(lat_start,lat_end,lat_gap)
month = np.arange(1,13,1)
area = np.zeros((len(lat), 12), dtype=int)
j=0
for l in lat:
    # Loop albedo for different hours in the day
    ghi = pd.DataFrame(index=demo.index)
    ghi['Wvlgth'] = demo.Wvlgth
    i=0
    for m in month:
        try:
            tmp = pySMARTS.SMARTSTimeLocation(IOUT=IOUT, YEAR=YEAR, MONTH=m, DAY=DAY, HOUR=HOUR,
                                              LATIT=l, LONGIT=LONGIT, ALTIT=ALTIT, ZONE=ZONE, material=MAT)
            ghi[m] = tmp.Global_horizn_irradiance
            area[j,i] = trapz(ghi.iloc[:, i+1])  
        except:
            ghi[m] = 0
            area[j,i] = 0  
        i=i+1
    j=j+1

# Plot irradiance for different hours in the day    
fig1 = plt.figure(figsize=(10,10))
plt.ylim(0,1200)
plt.xlim(1,12)
plt.tick_params(direction='in')
j=0
for l in lat:
    plt.plot(month,area[j,:],label=l) 
    j=j+1
plt.legend(title='Latitude')
plt.xlabel('Month')
plt.ylabel('Irradiance [W m-2]')
plt.title('Global Horizontal Irradiance - Monthly by Latitude')
plt.grid()
plt.show()
