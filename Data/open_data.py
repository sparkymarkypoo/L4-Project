
def NRELdata(year, latitude, longitude, name, interval): # For opening NREL psmv3 data

    import pandas as pd
    from NSRDB_API import PSMv3_API

    try:
        NSRDB_data = pd.read_csv(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data\{name}_lat{latitude}-long{longitude}_{year}_{interval}min.csv', index_col=0) # Try to open this
    except:
        NSRDB_data = PSMv3_API(latitude, longitude, year, interval) # If it doesn't exist then run api request
                                                            # df, tz = PSMv3_API(36.626,-116.018,2021,5)
        NSRDB_data.to_csv(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data\{name}_lat{latitude}-long{longitude}_{year}_{interval}min.csv')

    return NSRDB_data



def open_aod550(location, tz_offset): # for opening aod550 data in netcdf format (from copernicus ads)
                   # needed for validation_sandia
    import xarray as xr
    import numpy as np
    import pandas as pd

    # open data
    df = xr.open_dataset(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Data For Validating Models/aod550/{location}_aod550.nc', engine='netcdf4')

    # extract the variables from the data
    aod550_ = np.squeeze(df['aod550'].values)
    time = df.time.values

    # interpolate
    time_smooth = pd.date_range(time[0], time[-1], freq='1h')
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(time, aod550_, k=3)  # type: BSpline
    aod550_smooth = spl(time_smooth)
    
    # time zone
    time_smooth = time_smooth + tz_offset
    
    # reformat
    aod550 = pd.Series(index=time_smooth, data=aod550_smooth)
    
    return aod550

 

def open_copernicus(lat, long, tz_offset): # for opening aod550 data in netcdf format (from copernicus ads)
                   # needed for validation_sandia
    import xarray as xr
    import numpy as np
    import pandas as pd

    # open data
    df = xr.open_dataset(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Copernicus/{lat}_{long}_cop.nc', engine='netcdf4')

    # extract the variables from the data
    cop_ = df.to_dataframe()
    time = df.time.values
    cop_.index = time
    cols = cop_.columns

    # interpolate
    time_smooth = pd.date_range(time[0], time[-1], freq='1h')
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(time, cop_, k=3)  # type: BSpline
    cop_smooth = spl(time_smooth)
    
    # time zone
    time_smooth = time_smooth + tz_offset
    
    # reformat
    cop = pd.DataFrame(index=time_smooth, data=cop_smooth, columns = cols)
    
    return cop



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



def format_looped_data(copy, dataframe, time, modelchain):
    dataframe.index = time
    run = modelchain.run_model(dataframe)
    copy.append(run.results.ac)
    return copy