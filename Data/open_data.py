
def CAMSdata(start,end): # REDUNDANT - was used for extracting variables from Copernicus CAMS data
                                     # for use in the SMARTS model

    import xarray as xr
    
    sdf = xr.open_dataset(r'C:\Users\mark\Documents\L4-Project-Code\Copernicus\Desert Rock\Cop_dra_single.nc', engine='netcdf4')
    mdf = xr.open_dataset(r'C:\Users\mark\Documents\L4-Project-Code\Copernicus\Desert Rock\Cop_dra_multi.nc', engine='netcdf4')
    
    longit = sdf.longitude.values[0]
    latit = sdf.latitude.values[0]
    
    # Get range of time we want
    complete_time = sdf.time.values
    i = j = 0   
    for t in complete_time:
        if start > t:
            i=i+1      
        elif start <= t <= end:
            j=j+1
        
    # extract the variables from the data           
    pressure = sdf.sp.values[i:i+j,0,0]/100 # convert to mbar
    ad550 = sdf.aod550.values[i:i+j,0,0] # aerosol depth
    tc_o3 = sdf.gtco3.values[i:i+j,0,0]/10 # total column ozone kg/m2 to g/cm2
    tc_wv = sdf.tcwv.values[i:i+j,0,0]/10 # total column water vapour kg/m2 to g/cm2
    total_cloud = sdf.tcc.values[i:i+j,0,0]
    time = sdf.time.values[i:i+j]
    
    temp = mdf.t.values[i:i+j,0,0]-273.15 # temperature K to C
    #humid = mdf.r.values[i:i+j,0,0] # relative humidity THIS IS MISSING REEEEE
    
    m_air = 28.96 # molar mass of dry air
    # ppmv = kg/kg * (m_air/m_x) * 1e6
    ch2o = mdf.hcho.values[i:i+j,0,0] * (m_air/30.03) * 1e6
    ch4 = mdf.ch4_c.values[i:i+j,0,0] * (m_air/16.04) * 1e6
    co = mdf.co.values[i:i+j,0,0] * (m_air/28.01) * 1e6
    o3 = mdf.go3.values[i:i+j,0,0] * (m_air/48.00) * 1e6
    hno3 = mdf.hno3.values[i:i+j,0,0] * (m_air/63.01) * 1e6
    no2 = mdf.no2.values[i:i+j,0,0] * (m_air/46.01) * 1e6
    no = mdf.no.values[i:i+j,0,0] * (m_air/30.01) * 1e6
    so2 = mdf.so2.values[i:i+j,0,0]  * (m_air/64.07) * 1e6

    return longit, latit, pressure, ad550, tc_o3, tc_wv, total_cloud, time, temp, ch2o, ch4, co, o3, hno3, no2, no, so2

# pressure, ad550, time, temp, sp_humid, ch2o, ch4, co, o3, hno3, no2, no, so2 = CAMSdata()


def radiationdata(): # REDUNDANT - was used for opening CAMS radiation datasets

    import xarray as xr

    df = xr.open_dataset(r'C:\Users\mark\Documents\L4-Project-Code\Copernicus\radiation.nc', engine='netcdf4')

    # extract the temperature from the data
    GHI_all = df.GHI.values[:,0,0,0]
    GHI_clear = df.CLEAR_SKY_GHI.values[:,0,0,0]
    rad_time = df.time.values
    
    return GHI_all, GHI_clear, rad_time


def NRELdata(start, end, year, latitude, longitude, interval): # For opening NREL psmv3 data
    
    import numpy as np
    import pandas as pd
    from NSRDB_API import PSMv3_API

    try:
        df = pd.read_csv(rf'C:\Users\mark\Documents\L4-Project-Data\NSRDB Data\lat{latitude}-long{longitude}_{year}_{interval}min.csv') # Try to open this
    except:
        df = PSMv3_API(latitude, longitude, year, interval) # If it doesn't exist then run api request
                                                            # df = PSMv3_API(36.626,-116.018,2021,5)
        df.to_csv(rf'C:\Users\mark\Documents\L4-Project-Data\NSRDB Data\lat{latitude}-long{longitude}_{year}_{interval}min.csv')

    # Get range of time we want
    complete_time = np.arange('%s-01-01' %(year), '%s-12-31' %(year), dtype=f'datetime64[{interval}m]')
    i = j = 0   
    for t in complete_time:
        if start > t:
            i=i+1      
        elif start <= t <= end:
            j=j+1
    
    NSRDB_data = df[i:i+j]  # select the times we want

    return NSRDB_data

 
def BSRNdata(start, end): # For opening Baseline Surface Radiation Network data
   
    import numpy as np
    import pandas as pd
   
    specific_rows = np.arange(22)  # cut out crap at the top     
    df = pd.read_csv(r'C:\Users\mark\Documents\L4-Project-Data\DRA_radiation_2022-05.tab', delimiter ='\t', skiprows = specific_rows)
    
    ts = pd.to_datetime(df['Date/Time']) - pd.Timedelta(hours=8)   # Timestamp
    complete_time = np.zeros(len(ts), dtype='datetime64[1m]')
    for k in range(len(ts)):
        complete_time[k] = ts[k].to_datetime64()+8*6000   # Datetime64
    
    i = j = 0   
    for t in complete_time:
        if start > t:
            i=i+1      
        elif start <= t <= end:
            j=j+1
                
    BSRN_time= complete_time[i:i+j:5]
    temp_ = df["DIR [W/m**2]"]      # extract the data
    BSRN_dir = temp_[i:i+j:5]       # select the times we want
    temp_ = df["DIF [W/m**2]"]
    BSRN_dif = temp_[i:i+j:5]
    temp_ = df["SWD [W/m**2]"]
    BSRN_ghi = temp_[i:i+j:5]
    
    return BSRN_time, BSRN_dir, BSRN_dif, BSRN_ghi
