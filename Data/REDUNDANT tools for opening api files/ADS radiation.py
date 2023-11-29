import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-solar-radiation-timeseries',
    {
        'sky_type': 'observed_cloud',
        'location': {
            'latitude': 36.75,
            'longitude': -4,
        },
        'altitude': '15',
        'date': '2019-01-01/2019-01-07',
        'time_step': '1hour',
        'time_reference': 'universal_time',
        'format': 'netcdf',
    },
    'radiation.nc')