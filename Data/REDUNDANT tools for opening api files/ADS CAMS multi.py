import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'variable': [
            'carbon_monoxide', 'formaldehyde', 'methane_chemistry',
            'nitric_acid', 'nitrogen_dioxide', 'nitrogen_monoxide',
            'ozone', 'specific_humidity', 'sulphur_dioxide',
            'temperature',
        ],
        'model_level': '60',
        'date': '2022-06-01/2022-06-07',
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'area': [
            51.75, 0, 51.0,
            0.75,
        ],
        'format': 'netcdf',
        'pressure_level': '1000',
    }, 
    'CAMS_multi.nc')