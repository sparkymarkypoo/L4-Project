import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'area': [
            1, 0, 0,
            1,
        ],
        'time': '12:00',
        'day': [
            '01', '02', '03',
            '04', '05',
        ],
        'month': '01',
        'year': '2020',
        'variable': [
            '2m_temperature', 'total_cloud_cover',
        ],
    },
    'ERA5.nc')