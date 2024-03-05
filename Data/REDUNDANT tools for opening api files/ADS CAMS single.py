
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'variable': ['total_aerosol_optical_depth_550nm',],
        'model_level': '60',
        'date': '2022-06-01/2022-06-07',
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'area': [
            35.75, -4, 36.75,
            -3.75,
        ],
        'format': 'netcdf',
        'pressure_level': '1000',
    },
    'CAMS_single.nc')