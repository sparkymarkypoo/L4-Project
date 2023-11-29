#import cdsapi as cds
import xarray as xr
#import netCDF4 as nc
import numpy as np
import pandas as pd
#from scipy.io import netcdf
import matplotlib.pyplot as plt

#file2read = netcdf.NetCDFFile('download.nc','r')

#file2read = nc.Dataset(r'C:\Users\mark\Documents\L4 Project\download.nc')

df = xr.open_dataset(r'C:\Users\mark\Documents\L4 Project\ERA5.nc', engine='netcdf4')

## What's inside?
#print(df)

## extract the variables from the data
temp = df['t2m']
cloud = df['tcc']
#print(temp)
#print(cloud)

## extract the temperature from the data
temp_ = df.t2m.values
## print to see temp_'s values
#print(temp_)
## <xarray.DataArray 't2m' (time: 5, latitude: 5, longitude: 5)>
## print to the shape of the temperature array
#print(temp_.shape)

mygraph = plt.figure()
## pick a temperature value at all time steps, one longitude point
for lat in range(5):
    new_temp = temp_[:,lat,0] - 273 #convert to celsius
    #print(new_temp)
    plt.plot(new_temp, label = lat)
    
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.title("T")