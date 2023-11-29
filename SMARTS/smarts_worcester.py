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

# # Albedo for AM1.5
# am15 = pySMARTS.SMARTSAirMass(IOUT=IOUT, material=MAT, AMASS='1.5', min_wvl='20', max_wvl='5000')

# Albedo at 8am
demo = pySMARTS.SMARTSTimeLocation(IOUT, YEAR, MONTH, DAY, HOUR,
                                    LATIT, LONGIT, ALTIT, ZONE, material=MAT)


# # Compute the area using the composite trapezoidal rule.
# area = trapz(am15.Global_tilted_irradiance)
# print("area =", area)
# # Compute the area using the composite Simpson's rule.
# area = simpson(am15.Extraterrestrial_spectrm)
# print("area =", area)

# # Plot albedo for AM1.5 vs Worcester   
# fig0 = plt.figure(figsize=(10,10))
# plt.ylim(0,1.5)
# plt.xlim(250,2500)
# plt.tick_params(direction='in')
# plt.plot(demo.Wvlgth,demo.Global_horizn_irradiance,label='Worcs 13:00')
# plt.plot(am15.Wvlgth,am15.Global_horizn_irradiance,label='AM 1.5')
# plt.plot(am15.Wvlgth,am15.Extraterrestrial_spectrm,label='AM 0')
# plt.legend(title='Hour')
# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Irradiance [W m-2 nm-1]')
# plt.title('Global Horizontal Irradiance')
# plt.show()

# Loop for months of the year
start = 0 #first hour
end = 25 #last hour
gap = 1 #time interval
hour = np.arange(start,end,gap)
month = np.arange(1,13,2)
area = np.zeros((len(month), int((end-start)/gap)), dtype=int)
j=0
for m in month:
    # Loop albedo for different hours in the day
    ghi = pd.DataFrame(index=demo.index)
    ghi['Wvlgth'] = demo.Wvlgth
    #gti = ghi.copy()
    i=0
    for h in hour:
        try:
            tmp = pySMARTS.SMARTSTimeLocation(IOUT=IOUT, YEAR=YEAR, MONTH=m, DAY=DAY, HOUR=h,
                                              LATIT=LATIT, LONGIT=LONGIT, ALTIT=ALTIT, ZONE=ZONE, material=MAT)
            ghi[h] = tmp.Global_horizn_irradiance
            #gti[h] = tmp.Global_tilted_irradiance
            area[j,i] = trapz(ghi.iloc[:, i+1])  
        except:
            ghi[h] = 0
            #gti[h] = 0
            area[j,i] = 0  
        i=i+1
    j=j+1

# Plot irradiance for different hours in the day    
fig1 = plt.figure(figsize=(10,10))
plt.ylim(0,1000)
plt.xlim(0,24)
plt.xticks(np.arange(0,28,4))
plt.tick_params(direction='in')
j=0
for m in month:
    plt.plot(hour,area[j,:],label=m) 
    j=j+1
plt.legend(title='Month')
plt.xlabel('Time (Hour)')
plt.ylabel('Irradiance [W m-2]')
plt.title('Worcester UK, Global Horizontal Irradiance - Hourly')
plt.grid()
plt.show()
