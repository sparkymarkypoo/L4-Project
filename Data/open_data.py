
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


def open_aod550_final(lat, long, year, tz_offset): # for opening aod550 data in netcdf format (from copernicus ads)
                   # needed for validation files. see open_aod550_final for one used in spectal_efficiency_final
    import xarray as xr
    import numpy as np
    import pandas as pd

    # open data
    try:
        df = xr.open_dataset(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/aod550/{lat}_{long}_{year}_aod550.nc', engine='netcdf4')
        
    except:
        import cdsapi

        c = cdsapi.Client()

        c.retrieve(
            'cams-global-reanalysis-eac4',
            {
                'variable': ['total_aerosol_optical_depth_550nm'],
                'date': f'{year}-01-01/{year}-12-31',
                'time': [
                    '00:00', '03:00', '06:00',
                    '09:00', '12:00', '15:00',
                    '18:00', '21:00',
                ],
                'area': [
                    lat+0.374, long-0.374, lat-0.374,
                    long+0.374,
                ],
                'format': 'netcdf',
            },
            rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/aod550/{lat}_{long}_{year}_aod550.nc')
        
        df = xr.open_dataset(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/aod550/{lat}_{long}_{year}_aod550.nc', engine='netcdf4')

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




def open_aod550(location, tz_offset): # for opening aod550 data in netcdf format (from copernicus ads)
                   # needed for validation files. see open_aod550_final for one used in spectal_efficiency_final
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


def format_looped_data(copy, dataframe, time, modelchain):
    dataframe.index = time
    run = modelchain.run_model(dataframe)
    copy.append(run.results.ac)
    return copy