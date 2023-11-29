import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'date': '2022-01-01/2022-01-01',
        'format': 'grib',
        'variable': 'temperature',
        'pressure_level': '900',
        'area': [
            1, 0, 0,
            1,
        ],
        'time': '12:00',
        'model_level': '60',
    },
    'download.nc')
