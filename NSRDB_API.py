def PSMv3_API(latitude, longitude, year, interval):
    
    import pvlib
    
    attributes=('air_temperature', 'dhi', 'dni', 'ghi',
    'surface_albedo', 'wind_speed', 'cloud_type',
    'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'surface_albedo')
    #,'dew_point', 'surface_pressure'
    
    data, metadata = pvlib.iotools.get_psm3(latitude=latitude, longitude=longitude, names=f'{year}', 
                                            attributes=attributes, interval=interval,
                                            api_key='hQ8W1ZA3mLtMmrP7ee45addpSTwyh1Kid4rurDRc',
                                            email='mark.r.salkeld@gmail.com', leap_day=False,
                                            full_name='Mark Salkeld', affiliation='pvlib python', timeout=30)
    
    return data

# data = PSMv3_API(36.626,-116.018,2021,5)
#'air_temperature', 'dhi', 'dni', 'ghi',