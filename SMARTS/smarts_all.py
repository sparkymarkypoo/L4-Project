import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pySMARTS
from numpy import trapz
from open_copernicus import CAMSdata
from open_copernicus import radiationdata
from open_copernicus import NRELdata
from open_copernicus import BSRNdata

# %edit C:\Users\mark\anaconda3\Lib\site-packages\pySMARTS\main.py

# Choose time range:
start = np.datetime64('2022-05-01T12:00')
end = np.datetime64('2022-05-08T08:00')

# Get Copernicus shit
longit, latit, pressure, ad550, tc_o3, tc_wv, total_cloud, time, temp, ch2o, ch4, co, o3, hno3, no2, no, so2 = CAMSdata(start,end)
area = np.zeros(len(time))

## Get radiation shit
#GHI_all, GHI_clear, rad_time = radiationdata()

# Get NREL (FARMS) shit
year, FARMS_all, FARMS_clear, cloud_type, z_angle_deg = NRELdata(start,end)
farms_time = np.arange(start, end+5, dtype='datetime64[5m]')

# Get BSRN shit
BSRN_time, BSRN_dir, BSRN_dif = BSRNdata(start,end)
temp_z = z_angle_deg.to_numpy()
temp_dif = BSRN_dif.to_numpy()
temp_dir = BSRN_dir.to_numpy()
z_angle_rad = np.pi * temp_z / 180
BSRN_ghi = temp_dir * np.cos(z_angle_rad) + temp_dif

# Find error
temp_farms = FARMS_all.to_numpy()
perc_error = np.zeros(len(temp_farms))
for i in range(len(temp_farms)):
    if BSRN_ghi[i] > 10 and temp_farms[i] > 10:
        perc_error[i] = (BSRN_ghi[i] - temp_farms[i])*100/BSRN_ghi[i]

# Specify Desired Output
IOUT = '4' # Global horizontal irradiance W m-2

for i in range(len(time)):
    
    # Card 2a: Small Effect
    SPR = pressure[i] # Surface pressure (mbar).
    
    # Card 2a: Small Effect
    ALTIT = '0.05' # Elevation above sea level (km)
    HEIGHT = '0' # Height above the ground surface (km)
    
    # Card 3a: Tiny Effect
    RH = '50' # Relative humidity at site level (%)
    TAIR = temp[i] # Atmospheric temperature at site level (°C)
    SEASON = 'WINTER' # Can be either `WINTER` or `SUMMER`
    TDAY = '20' # Average daily temperature at site level (°C)
    
    # Card 4a: Large Effect
    W = tc_wv[i] # Precipitable water above the site (cm or g/cm2)
    
    # Card 5a: Large Effect
    AbO3 = tc_o3[i] # Ozone total-column abundance
    
    # Card 6b: Large Effect
    # Volumetric concentration in the assumed 1-km deep tropospheric pollution layer (ppmv)
    ApCH2O = ch2o[i] # Formaldehyde
    ApCH4 = ch4[i] # Methane
    ApCO = co[i] # Carbon monoxide
    ApHNO2 = 0 # Nitrous acid
    ApHNO3 = hno3[i] # Nitric acid
    ApNO = no[i] # Nitrogen monoxide
    ApNO2 = no2[i] # Nitrogen dioxide
    ApNO3 = 0 # Nitrogen trioxide
    ApO3 = 0
    #ApO3 = o3 # Ozone        NOT WORKING!?! REEEE
    ApSO2 = so2[i] # Sulfur dioxide
    
    # Card 7: Small Effect
    qCO2 = '0' # CO2 columnar volumetric concentration (ppmv)
    
    # Card 8a: Tiny Effect
    ALPHA1 = '0.7' # Average Ångström's wavelength exponent for wavelengths < 500 nm (usually 0.0 to 2.6)
    ALPHA2 = '0.7' # Average Ångström's wavelength exponent for wavelengths > 500 nm (usually 0.0 to 2.6)
    OMEGL = '0.7' # Aerosol single scattering albedo (generally between 0.6 and 1.0)
    GG = '0.7' # Aerosol asymmetry parameter (generally between 0.5 and 0.9)
    
    # Card 9: ITURB is an option to select the correct turbidity data input
    ITURB = '5'
    # Card 9a: Large Effect
    TAU5 = '' # 0 - Aerosol optical depth at 500 nm (τ5)
    BETA = '' # 1 - Ångström’s turbidity coefficient (aerosol optical depth at 1000 nm)
    BCHUEP = '' # 2 - Schüepp’s turbidity coefficient (decadic aerosol optical depth at 500 nm)
    TAU550 = ad550[i] # 5 -Aerosol optical depth at 500 nm (τ550)
    
    # Card 10: # Small Effect
    material = 'Grass' # Far Field Albedo for backscattering
    
    # Card 10a:
    RHOX = '0' # Zonal broadband Lambertian ground albedo (for backscattering, between 0 and 1)
                            
    # Card 10c:
    TILT = '37'   # Tilt angle of the receiving surface (0 to 90 deg.)
    WAZIM = '180' # Surface azimuth (0 to 360 deg.) clockwise from North; e.g., 270 for West
                  # Use -999 for a sun-tracking surface for both TILT and WAZIM
                  
    # Card 10d:
    RHOG = '0' # Local broadband Lambertian foreground albedo (for tilted plane), between 0 and 1
    
    # Card 11: Spectral range for all Calculations
    min_wvl = '280' #Min wavelength
    max_wvl = '4000' #Max wavelength
    
    # Card 17a: IMASS = 3 Input date, time and coordinates
    str_time = str(time[i])
    YEAR = str_time[:4]
    MONTH = str_time[5:7]
    DAY = str_time[8:10]
    HOUR = str_time[11:13]
    LATIT = latit
    LONGIT = longit
    ZONE = '-7'
    
    try:
        data = pySMARTS.SMARTSSRRL(IOUT, YEAR, MONTH, DAY, HOUR, LATIT, LONGIT, ALTIT, ZONE, 
                                   W, RH, TAIR, SEASON, TDAY, SPR, TILT, WAZIM,
                                   RHOG, ALPHA1, ALPHA2, OMEGL, GG, TAU550, AbO3, qCO2,
                                   ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, ApO3, ApSO2, 
                                   HEIGHT, material, min_wvl , max_wvl)
    
        # Compute the area using the composite trapezoidal rule.
        area[i] = trapz(data.Global_horizn_irradiance)
        #print("Irradiance =", area, "W/m^2")
        
    except:
        area[i] = 0


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
ax1.plot(farms_time-7*12, FARMS_all, label = 'FARMS All Sky')
ax1.plot(farms_time-7*12, FARMS_clear, label = 'FARMS Clear Sky')

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
